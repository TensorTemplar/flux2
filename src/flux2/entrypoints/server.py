"""FastAPI inference server for FLUX.2 container deployment.

Endpoints: /health, /ready, /generate
"""

import base64
import io
import random
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import click
import torch
import uvicorn
from einops import rearrange
from fastapi import FastAPI, HTTPException
from PIL import ExifTags, Image
from pydantic import BaseModel, Field

from ..remote_text_encoder import RemoteTextEncoder
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

    @property
    def is_ready(self) -> bool:
        return all([self.flow_model, self.text_encoder, self.ae])


_state = ModelState()
_settings: Any = None


def load_models() -> None:
    """Load all required models into memory."""
    global _state, _settings

    _settings = get_settings()
    model_name = _settings.model_name
    enable_moderation = _settings.enable_moderation

    print(f"Loading models for {model_name}...")
    torch_device = torch.device("cuda")

    # Load text encoder (local or remote)
    if _settings.text_encoder_url:
        print(f"  Using remote text encoder at {_settings.text_encoder_url}")
        _state.text_encoder = RemoteTextEncoder(_settings.text_encoder_url)
        _state.text_encoder_remote = True
    else:
        print("  Loading text encoder...")
        _state.text_encoder = load_text_encoder(model_name, device=torch_device)
        _state.text_encoder.eval()
        _state.text_encoder_remote = False

    # Load moderation model if enabled
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

    # Load flow model
    print("  Loading flow model...")
    _state.flow_model = load_flow_model(model_name, device=torch_device)

    # Load autoencoder
    print("  Loading autoencoder...")
    _state.ae = load_ae(model_name, device=torch_device)
    _state.ae.eval()

    _state.model_name = model_name
    _state.model_info = FLUX2_MODEL_INFO[model_name.lower()]
    _state.device = torch_device

    print("All models loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    try:
        load_models()
    except HFHomeNotSetError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load models: {e}", file=sys.stderr)
        sys.exit(1)

    yield

    # Cleanup on shutdown (reset state)
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
    """Readiness probe - returns OK only if all models are loaded."""
    return ReadyResponse(
        ready=_state.is_ready,
        model_loaded=_state.flow_model is not None,
        text_encoder_loaded=_state.text_encoder is not None,
        ae_loaded=_state.ae is not None,
        moderation_loaded=_state.mod_model is not None,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate an image from a text prompt."""
    if not _state.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    model_info = _state.model_info

    # Validate params against model requirements (these keys always exist in FLUX2_MODEL_INFO)
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

    # Input moderation
    if _state.mod_model is not None:
        if _state.mod_model.test_txt(req.prompt):
            raise HTTPException(
                status_code=400,
                detail="Prompt flagged for potential copyright or public persona concerns",
            )

    seed = req.seed if req.seed is not None else random.randrange(2**31)

    try:
        with torch.no_grad():
            # Encode prompt (remote or local)
            if _state.text_encoder_remote:
                include_empty = not model_info["guidance_distilled"]
                ctx, ctx_ids = await _state.text_encoder.encode(
                    [req.prompt], include_empty=include_empty, device=_state.device
                )
            else:
                if model_info["guidance_distilled"]:
                    ctx = _state.text_encoder([req.prompt]).to(torch.bfloat16)
                else:
                    ctx_empty = _state.text_encoder([""]).to(torch.bfloat16)
                    ctx_prompt = _state.text_encoder([req.prompt]).to(torch.bfloat16)
                    ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
                ctx, ctx_ids = batched_prc_txt(ctx)

            # Create noise
            shape = (1, 128, req.height // 16, req.width // 16)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
            x, x_ids = batched_prc_img(randn)

            # Denoise
            timesteps = get_schedule(req.num_steps, x.shape[1])

            if model_info["guidance_distilled"]:
                x = denoise(
                    _state.flow_model,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=req.guidance,
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
                )

            # Decode
            x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
            x = _state.ae.decode(x).float()

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        # Output moderation
        flagged = False
        if _state.mod_model is not None and _state.mod_model.test_image(img):
            flagged = True

        # Add EXIF data
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;flux2"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", exif=exif_data, quality=95, subsampling=0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return GenerateResponse(
            image_base64=img_base64,
            seed=seed,
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            flagged=flagged,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


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
