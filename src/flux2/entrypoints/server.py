"""FastAPI inference server for FLUX.2 container deployment.

Endpoints: /health, /ready, /generate, /generate/profile
"""

import asyncio
import base64
import io
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import torch
import uvicorn
from einops import rearrange
from fastapi import FastAPI, HTTPException
from PIL import ExifTags, Image
from pydantic import BaseModel, Field
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from ..remote_text_encoder import RemoteTextEncoder, RemoteTextEncoderError
from ..sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cfg,
    get_schedule,
    scatter_ids,
)
from ..settings import HFHomeNotSetError, get_settings
from ..util import (
    FLUX2_MODEL_INFO,
    load_ae,
    load_flow_model,
    load_text_encoder,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    width: int = Field(default=1360, ge=64, le=4096, description="Output width in pixels")
    height: int = Field(default=768, ge=64, le=4096, description="Output height in pixels")
    num_steps: int = Field(default=50, ge=1, le=100, description="Number of denoising steps")
    guidance: float = Field(default=4.0, ge=0.0, le=20.0, description="Guidance scale")
    seed: int | None = Field(default=None, description="Random seed (None for random)")
    upsample: bool = Field(default=False, description="Upsample prompt using text encoder LLM")


class GenerateResponse(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded PNG image")
    seed: int = Field(..., description="Seed used for generation")
    prompt: str = Field(..., description="Prompt used for generation")
    width: int = Field(..., description="Output width")
    height: int = Field(..., description="Output height")
    flagged: bool = Field(default=False, description="Whether output was flagged by moderation")


class HealthResponse(BaseModel):
    status: str
    model_name: str
    moderation_enabled: bool


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    text_encoder_loaded: bool
    ae_loaded: bool
    moderation_loaded: bool
    warmup_complete: bool


@dataclass
class ModelState:
    """Typed container for loaded models."""

    text_encoder: Any = None
    text_encoder_remote: bool = False
    mod_model: Any = None
    flow_model: Any = None
    ae: Any = None
    model_name: str = ""
    model_info: dict = field(default_factory=dict)
    device: Any = None
    warmup_complete: bool = False

    @property
    def is_ready(self) -> bool:
        return all([self.flow_model, self.text_encoder, self.ae, self.warmup_complete])


_state = ModelState()
_settings: Any = None
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


def create_profiler(seed: int, num_steps: int):
    """Create a PyTorch profiler for tracing inference.

    Args:
        seed: Random seed for unique trace filename
        num_steps: Total denoising steps for schedule calculation

    Returns:
        Configured profiler instance, or None if profiling is disabled
    """
    if not _settings or not _settings.profiler_enabled:
        return None

    output_dir = Path(_settings.profiler_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = output_dir / f"trace_{seed}"

    active = max(1, num_steps - 2)
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=0, active=active, repeat=1),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    )
    prof._trace_dir = trace_dir  # Store for later export
    return prof


def load_models() -> None:
    """Load all required models into memory."""
    global _state, _settings

    _settings = get_settings()
    model_name = _settings.model_name
    enable_moderation = _settings.enable_moderation

    if _settings.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  TF32 enabled for matmul/cudnn operations")

    print(f"Loading models for {model_name}...")
    torch_device = torch.device("cuda")

    if _settings.text_encoder_url:
        print(f"  Using remote text encoder at {_settings.text_encoder_url}")
        _state.text_encoder = RemoteTextEncoder(_settings.text_encoder_url)
        _state.text_encoder_remote = True
    else:
        print("  Loading text encoder...")
        _state.text_encoder = load_text_encoder(model_name, device=torch_device)
        _state.text_encoder.eval()
        _state.text_encoder_remote = False

    _state.mod_model = None
    if enable_moderation:
        if "klein" in model_name:
            print("  Loading moderation model (Mistral-24B)...")
            _state.mod_model = load_text_encoder("flux.2-dev", device=torch_device)
            _state.mod_model.eval()
        elif not _state.text_encoder_remote:
            _state.mod_model = _state.text_encoder
        else:
            print("  WARNING: Moderation disabled with remote text encoder for non-klein models")
    elif "klein" not in model_name and not _state.text_encoder_remote:
        _state.mod_model = _state.text_encoder

    print("  Loading flow model...")
    _state.flow_model = load_flow_model(model_name, device=torch_device)

    print("  Loading autoencoder...")
    _state.ae = load_ae(model_name, device=torch_device)
    _state.ae.eval()

    if _settings.torch_compile:
        if _settings.torch_logs:
            import os

            os.environ["TORCH_LOGS"] = _settings.torch_logs
            print(f"  TORCH_LOGS={_settings.torch_logs}")
        mode = _settings.torch_compile_mode
        print(f"  Compiling flow model with torch.compile(mode='{mode}')...")
        _state.flow_model = torch.compile(_state.flow_model, mode=mode)
        if _settings.torch_compile_ae:
            print(f"  Compiling autoencoder with torch.compile(mode='{mode}')...")
            _state.ae = torch.compile(_state.ae, mode=mode)

    _state.model_name = model_name
    _state.model_info = FLUX2_MODEL_INFO[model_name.lower()]
    _state.device = torch_device

    print("All models loaded successfully.")


async def run_warmup() -> None:
    """Run a warmup generation to trigger torch.compile/Triton autotuning."""
    global _state

    print("Running warmup generation...")
    t0 = time.perf_counter()

    model_info = _state.model_info
    warmup_steps = 4 if model_info["guidance_distilled"] else 10
    default_request = GenerateRequest(prompt="warmup")
    warmup_width = default_request.width
    warmup_height = default_request.height

    if _state.text_encoder_remote:
        print("Waiting for remote text encoder...")
        timeout_s = 120.0
        deadline = time.perf_counter() + timeout_s
        while True:
            try:
                if await _state.text_encoder.ready():
                    break
            except Exception:
                pass
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    f"Remote text encoder not ready after {timeout_s:.0f}s: {_settings.text_encoder_url}"
                )
            await asyncio.sleep(2.0)

    async def run_warmup_request(profile_seed: int | None = None) -> None:
        req = GenerateRequest(
            prompt="warmup",
            width=warmup_width,
            height=warmup_height,
            num_steps=warmup_steps,
            guidance=float(model_info["defaults"]["guidance"]),
            seed=profile_seed,
        )
        profiler = create_profiler(profile_seed, warmup_steps) if profile_seed is not None else None
        await _run_generate(req, skip_ready_check=True, profiler=profiler)

    for idx in range(2):
        print(f"  Warmup request {idx + 1}/2")
        await run_warmup_request(profile_seed=None)

    if _settings and _settings.profiler_enabled:
        print("  Warmup request 3/3 (profiling starts at step 3)")
        await run_warmup_request(profile_seed=0)

    _state.warmup_complete = True
    print(f"Warmup complete in {time.perf_counter() - t0:.1f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    try:
        load_models()
        await run_warmup()
    except HFHomeNotSetError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load models: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    yield

    global _state
    _state = ModelState()


app = FastAPI(
    title="FLUX.2 Inference API",
    description="Image generation API for FLUX.2 models",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe - always returns OK if server is running."""
    return HealthResponse(
        status="ok",
        model_name=_state.model_name or "loading",
        moderation_enabled=_state.mod_model is not None,
    )


@app.get("/ready", response_model=ReadyResponse)
def ready():
    """Readiness probe - returns OK only if all models are loaded and warmup is complete."""
    return ReadyResponse(
        ready=_state.is_ready,
        model_loaded=_state.flow_model is not None,
        text_encoder_loaded=_state.text_encoder is not None,
        ae_loaded=_state.ae is not None,
        moderation_loaded=_state.mod_model is not None,
        warmup_complete=_state.warmup_complete,
    )


def _run_gpu_inference(
    req: GenerateRequest,
    seed: int,
    prompt: str,
    ctx: torch.Tensor,
    ctx_ids: torch.Tensor,
    model_info: dict,
    profiler=None,
) -> tuple[str, bool, dict[str, float]]:
    """Run GPU-bound inference in a separate thread.

    Returns:
        Tuple of (base64_image, flagged, timing_dict)
    """
    timings: dict[str, float] = {}

    with torch.no_grad():
        if profiler is not None:
            profiler.start()

        shape = (1, 128, req.height // 16, req.width // 16)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
        x, x_ids = batched_prc_img(randn)

        t0 = time.perf_counter()
        timesteps = get_schedule(req.num_steps, x.shape[1])

        with record_function("denoise"):
            torch.cuda.nvtx.range_push("denoise")
            if model_info["guidance_distilled"]:
                x = denoise(
                    _state.flow_model,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=req.guidance,
                    profiler=profiler,
                )
            else:
                x = denoise_cfg(
                    _state.flow_model,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=req.guidance,
                    profiler=profiler,
                )
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        timings["denoise"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with record_function("decode"):
            torch.cuda.nvtx.range_push("decode")
            out_hw = (req.height // 16, req.width // 16)
            with record_function("scatter_ids"):
                x = torch.cat(scatter_ids(x, x_ids, out_hw)).squeeze(2)
            timings["scatter"] = time.perf_counter() - t0
            with record_function("ae_decode"):
                x = _state.ae.decode(x).float()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        timings["decode"] = time.perf_counter() - t0

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    flagged = False
    if _state.mod_model is not None and _state.mod_model.test_image(img):
        flagged = True

    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;flux2"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"

    buffered = io.BytesIO()
    img.save(buffered, format="PNG", exif=exif_data, quality=95, subsampling=0)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    if profiler is not None:
        profiler.stop()
        trace_dir = profiler._trace_dir
        trace_dir.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_dir / "trace.json"))
        print(f"[profiler] Trace saved to {trace_dir}/trace.json")

    return img_base64, flagged, timings


async def _run_generate(
    req: GenerateRequest,
    *,
    skip_ready_check: bool = False,
    profiler=None,
) -> GenerateResponse:
    t_start = time.perf_counter()

    if not _state.is_ready and not skip_ready_check:
        raise HTTPException(status_code=503, detail="Models not loaded")

    model_info = _state.model_info

    defaults = model_info["defaults"]
    fixed_params = model_info["fixed_params"]

    if "num_steps" in fixed_params and req.num_steps != defaults["num_steps"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model requires num_steps={defaults['num_steps']}, got {req.num_steps}",
        )

    if "guidance" in fixed_params and req.guidance != defaults["guidance"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model requires guidance={defaults['guidance']}, got {req.guidance}",
        )

    if _state.mod_model is not None:
        if _state.mod_model.test_txt(req.prompt):
            raise HTTPException(
                status_code=400,
                detail="Prompt flagged for potential copyright or public persona concerns",
            )

    prompt = req.prompt
    t_upsample = 0.0
    if req.upsample:
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push("upsample")
        try:
            if _state.text_encoder_remote:
                prompt = (await _state.text_encoder.upsample([req.prompt]))[0]
            elif hasattr(_state.text_encoder, "upsample_prompt"):
                prompt = _state.text_encoder.upsample_prompt([req.prompt])[0]
        except RemoteTextEncoderError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        finally:
            torch.cuda.nvtx.range_pop()
        t_upsample = time.perf_counter() - t0

    seed = req.seed if req.seed is not None else random.randrange(2**31)

    try:
        t0 = time.perf_counter()
        with record_function("encode"):
            torch.cuda.nvtx.range_push("encode")
            if _state.text_encoder_remote:
                include_empty = not model_info["guidance_distilled"]
                ctx, ctx_ids = await _state.text_encoder.encode(
                    [prompt], include_empty=include_empty, device=_state.device
                )
            else:
                with torch.no_grad():
                    if model_info["guidance_distilled"]:
                        ctx = _state.text_encoder([prompt]).to(torch.bfloat16)
                    else:
                        ctx_empty = _state.text_encoder([""]).to(torch.bfloat16)
                        ctx_prompt = _state.text_encoder([prompt]).to(torch.bfloat16)
                        ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
                    ctx, ctx_ids = batched_prc_txt(ctx)
            torch.cuda.nvtx.range_pop()
        t_encode = time.perf_counter() - t0

        loop = asyncio.get_event_loop()
        img_base64, flagged, timings = await loop.run_in_executor(
            _gpu_executor,
            _run_gpu_inference,
            req,
            seed,
            prompt,
            ctx,
            ctx_ids,
            model_info,
            profiler,
        )

        t_total = time.perf_counter() - t_start
        t_ae = timings["decode"] - timings["scatter"]
        print(
            f"[generate] {req.width}x{req.height} steps={req.num_steps} seed={seed} "
            f"total={t_total:.2f}s (upsample={t_upsample:.2f}s encode={t_encode:.2f}s "
            f"denoise={timings['denoise']:.2f}s decode={timings['decode']:.2f}s "
            f"[scatter={timings['scatter']:.2f}s ae={t_ae:.2f}s])"
        )

        return GenerateResponse(
            image_base64=img_base64,
            seed=seed,
            prompt=prompt,
            width=req.width,
            height=req.height,
            flagged=flagged,
        )

    except Exception as e:
        print(f"[generate] ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(
                f"[generate] CUDA memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved",
                file=sys.stderr,
            )
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate an image from a text prompt."""
    return await _run_generate(req)


@click.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev only)")
def main(host: str, port: int, reload: bool) -> None:
    """Start the inference server."""
    try:
        uvicorn.run("flux2.entrypoints.server:app", host=host, port=port, reload=reload)
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 0)


if __name__ == "__main__":
    main()
