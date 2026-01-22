import base64
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional

import click
import httpx
from PIL import ExifTags, Image
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Lazy imports for local inference - these require CUDA
# Import at module level only for type checking
if TYPE_CHECKING:
    pass

_DEFAULT_PROMPT = "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture"


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_name: str = Field(default="flux.2-klein-base-4b", validation_alias="MODEL_NAME")
    enable_moderation: bool = Field(default=False, validation_alias="ENABLE_MODERATION")
    output_dir: Path = Field(default=Path("output"), validation_alias="OUTPUT_DIR")
    prompt: str = _DEFAULT_PROMPT
    seed: Optional[int] = None
    width: int = 1360
    height: int = 768
    num_steps: int = 50
    guidance: float = 4.0
    input_images: List[Path] = Field(default_factory=list)
    match_image_size: Optional[int] = None
    upsample_prompt_mode: Literal["none", "local", "api"] = Field(
        default="none", validation_alias="UPSAMPLE_PROMPT_MODE"
    )


def print_config(cfg: Config):
    print("Current config:")
    print(f"  prompt: {cfg.prompt}")
    print(f"  seed: {cfg.seed}")
    print(f"  width: {cfg.width}")
    print(f"  height: {cfg.height}")
    print(f"  num_steps: {cfg.num_steps}")
    print(f"  guidance: {cfg.guidance}")
    print(f"  input_images: {[str(p) for p in cfg.input_images]}")
    print(f"  match_image_size: {cfg.match_image_size}")
    print(f"  upsample_prompt_mode: {cfg.upsample_prompt_mode}")
    print(f"  output_dir: {cfg.output_dir}")
    print()


def validate_model_params(model_name: str, cfg: Config, model_info: dict) -> bool:
    """Validate model parameters against fixed constraints."""
    defaults = model_info["defaults"]
    fixed_params = model_info["fixed_params"]

    if "num_steps" in fixed_params and cfg.num_steps != defaults["num_steps"]:
        click.echo(f"Error: {model_name} requires num_steps={defaults['num_steps']}", err=True)
        return False

    if "guidance" in fixed_params and cfg.guidance != defaults["guidance"]:
        click.echo(f"Error: {model_name} requires guidance={defaults['guidance']}", err=True)
        return False

    return True


def generate_image(
    cfg: Config,
    model_name: str,
    model,
    ae,
    text_encoder,
    mod_and_upsampling_model,
    api_upsampling_client,
    model_info: dict,
    torch_device,
    cpu_offloading: bool,
    enable_profiler: bool = False,
) -> bool:
    """Run a single generation. Returns True if successful."""
    import torch
    from einops import rearrange
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    from flux2.sampling import (
        batched_prc_img,
        batched_prc_txt,
        denoise,
        denoise_cfg,
        encode_image_refs,
        get_schedule,
        scatter_ids,
    )

    img_ctx = [Image.open(p) for p in cfg.input_images]

    width, height = cfg.width, cfg.height
    if cfg.match_image_size is not None:
        if 0 <= cfg.match_image_size < len(img_ctx):
            ref_img = img_ctx[cfg.match_image_size]
            width, height = ref_img.size
            click.echo(f"  Matched dimensions from image {cfg.match_image_size}: {width}x{height}")
        else:
            click.echo(f"  ! match_image_size={cfg.match_image_size} out of range", err=True)

    seed = cfg.seed if cfg.seed is not None else random.randrange(2**31)
    cfg.output_dir.mkdir(exist_ok=True)
    output_name = cfg.output_dir / f"sample_{len(list(cfg.output_dir.glob('*')))}.png"

    with torch.no_grad():
        ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

        if cfg.upsample_prompt_mode == "api" and api_upsampling_client is not None:
            click.echo(f"  Upsampling prompt via {api_upsampling_client.base_url}...")
            upsampled = api_upsampling_client.upsample_prompt(
                [cfg.prompt], img=[img_ctx] if img_ctx else None
            )
            prompt = upsampled[0] if upsampled else cfg.prompt
        elif cfg.upsample_prompt_mode == "local" and mod_and_upsampling_model is not None:
            upsampled = mod_and_upsampling_model.upsample_prompt(
                [cfg.prompt], img=[img_ctx] if img_ctx else None
            )
            prompt = upsampled[0] if upsampled else cfg.prompt
        else:
            prompt = cfg.prompt

        click.echo(f"Generating with prompt: {prompt}")

        if model_info["guidance_distilled"]:
            ctx = text_encoder([prompt]).to(torch.bfloat16)
        else:
            ctx_empty = text_encoder([""]).to(torch.bfloat16)
            ctx_prompt = text_encoder([prompt]).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
        ctx, ctx_ids = batched_prc_txt(ctx)

        if cpu_offloading:
            text_encoder.cpu()
            torch.cuda.empty_cache()
            model.to(torch_device)
            if mod_and_upsampling_model is not None:
                mod_and_upsampling_model.cpu()

        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(cfg.num_steps, x.shape[1])

        prof = None
        if enable_profiler:
            profiler_dir = Path("./profiler") / f"trace_{seed}"
            profiler_dir.mkdir(parents=True, exist_ok=True)
            wait = min(1, cfg.num_steps - 1)
            active = min(3, cfg.num_steps - wait - 1)
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=wait, warmup=1, active=active, repeat=1),
                on_trace_ready=tensorboard_trace_handler(str(profiler_dir)),
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            )
            prof.start()
            click.echo(f"  Profiler started (traces to {profiler_dir})")

        if model_info["guidance_distilled"]:
            x = denoise(
                model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=cfg.guidance,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
                profiler=prof,
            )
        else:
            x = denoise_cfg(
                model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=cfg.guidance,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
                profiler=prof,
            )

        if prof is not None:
            prof.stop()
            click.echo(f"  Profiler trace saved to {profiler_dir}")

        out_hw = (height // 16, width // 16)
        x = torch.cat(scatter_ids(x, x_ids, out_hw)).squeeze(2)
        x = ae.decode(x).float()

        if cpu_offloading:
            model.cpu()
            torch.cuda.empty_cache()
            text_encoder.to(torch_device)
            if mod_and_upsampling_model is not None:
                mod_and_upsampling_model.to(torch_device)

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    if mod_and_upsampling_model is not None and mod_and_upsampling_model.test_image(img):
        click.echo("Output flagged by moderation. Please try a different prompt.", err=True)
        return False

    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;flux2"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    img.save(output_name, exif=exif_data, quality=95, subsampling=0)
    click.echo(f"Saved {output_name}")
    return True


def generate_image_remote(cfg: Config, server_url: str) -> bool:
    """Generate image via remote inference server. Returns True if successful."""
    cfg.output_dir.mkdir(exist_ok=True)
    output_name = cfg.output_dir / f"sample_{len(list(cfg.output_dir.glob('*')))}.png"

    payload = {
        "prompt": cfg.prompt,
        "width": cfg.width,
        "height": cfg.height,
        "num_steps": cfg.num_steps,
        "guidance": cfg.guidance,
        "upsample": cfg.upsample_prompt_mode == "local",
    }
    if cfg.seed is not None:
        payload["seed"] = cfg.seed

    click.echo(f"Generating via {server_url}...")
    click.echo(f"  prompt: {cfg.prompt}")
    if payload["upsample"]:
        click.echo("  upsampling: enabled (server-side)")

    def _post_request(http2: bool, http1: bool = True) -> dict:
        with httpx.Client(timeout=300.0, http2=http2, http1=http1) as client:
            resp = client.post(f"{server_url}/generate", json=payload)
            resp.raise_for_status()
            return resp.json()

    try:
        try:
            data = _post_request(http2=False)
        except httpx.RemoteProtocolError as exc:
            if server_url.startswith("https://"):
                click.echo("  Remote protocol error, retrying with HTTP/2...", err=True)
                data = _post_request(http2=True, http1=False)
            else:
                click.echo(
                    "Error: Remote protocol error. Check the port-forward and ensure it targets the HTTP server.",
                    err=True,
                )
                return False
    except httpx.ConnectError:
        click.echo(f"Error: Cannot connect to {server_url}. Is the server running?", err=True)
        click.echo("Start with: docker compose up", err=True)
        return False
    except httpx.HTTPStatusError as e:
        if e.response.content:
            error_body = e.response.json()
            click.echo(f"Error: {error_body['detail']}", err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        return False
    except httpx.RemoteProtocolError as e:
        click.echo(
            f"Error: {e}. Check that {server_url} is an HTTP/1.1 or HTTP/2 endpoint.",
            err=True,
        )
        return False
    except httpx.WriteError as e:
        click.echo(
            f"Error: {e}. Connection dropped before the request completed.",
            err=True,
        )
        return False

    if data["flagged"]:
        click.echo("Warning: Output was flagged by moderation", err=True)

    if data["prompt"] != cfg.prompt:
        click.echo(f"  upsampled: {data['prompt']}")

    img_bytes = base64.b64decode(data["image_base64"])
    with open(output_name, "wb") as f:
        f.write(img_bytes)

    click.echo(f"Saved {output_name} (seed={data['seed']})")
    return True


def _resolve_remote_url(remote_arg: str | None) -> str | None:
    """Resolve remote URL from CLI arg or environment. Returns None if local mode."""
    if remote_arg is not None and remote_arg != "__USE_ENV__":
        return remote_arg

    from flux2.settings import get_settings

    try:
        settings = get_settings()
        return settings.inference_url
    except Exception:
        return None


@click.command()
@click.option("--model-name", "-m", default=None, help="Model name (default: from MODEL_NAME env)")
@click.option("--prompt", "-p", default=None, help="Generation prompt")
@click.option("--seed", "-s", type=int, default=None, help="Random seed")
@click.option("--width", "-W", type=int, default=None, help="Output width")
@click.option("--height", "-H", type=int, default=None, help="Output height")
@click.option("--num-steps", "-n", type=int, default=None, help="Denoising steps")
@click.option("--guidance", "-g", type=float, default=None, help="Guidance scale")
@click.option(
    "--input-images", "-i", multiple=True, type=click.Path(exists=True, path_type=Path), help="Input images"
)
@click.option(
    "--upsample",
    "-u",
    type=click.Choice(["none", "local", "api"]),
    default=None,
    help="Prompt upsampling mode",
)
@click.option("--single-eval", is_flag=True, help="Single generation then exit")
@click.option("--enable-moderation", is_flag=True, default=None, help="Enable content moderation")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.option("--cpu-offload", is_flag=True, help="CPU offloading for low VRAM")
@click.option(
    "--remote",
    "-r",
    is_flag=False,
    flag_value="__USE_ENV__",
    default=None,
    help="Use remote inference. Optionally provide URL, otherwise uses INFERENCE_URL env",
)
@click.option("--profile", is_flag=True, help="Enable PyTorch profiler (saves trace to ./profiler/)")
def main(
    model_name: str | None,
    prompt: str | None,
    seed: int | None,
    width: int | None,
    height: int | None,
    num_steps: int | None,
    guidance: float | None,
    input_images: tuple[Path, ...],
    upsample: str | None,
    single_eval: bool,
    enable_moderation: bool | None,
    debug: bool,
    cpu_offload: bool,
    remote: str | None,
    profile: bool,
):
    """FLUX.2 image generation CLI."""
    # Check for remote mode first - avoids heavy imports
    remote_url = _resolve_remote_url(remote)

    if remote_url is not None:
        # Remote mode - minimal imports, no CUDA needed
        cfg = Config()
        if model_name is not None:
            cfg.model_name = model_name
        if prompt is not None:
            cfg.prompt = prompt
        if seed is not None:
            cfg.seed = seed
        if width is not None:
            cfg.width = width
        if height is not None:
            cfg.height = height
        if num_steps is not None:
            cfg.num_steps = num_steps
        if guidance is not None:
            cfg.guidance = guidance
        if input_images:
            cfg.input_images = list(input_images)
        if upsample is not None:
            cfg.upsample_prompt_mode = upsample  # type: ignore[assignment]

        if single_eval:
            success = generate_image_remote(cfg, remote_url)
            sys.exit(0 if success else 1)
        else:
            while True:
                try:
                    line = click.prompt(">", default="", show_default=False).strip()
                except (EOFError, KeyboardInterrupt, click.Abort):
                    click.echo("\nbye!")
                    break
                match line:
                    case "quit" | "q" | "exit":
                        click.echo("bye!")
                        break
                    case "show":
                        print_config(cfg)
                    case "" | "run":
                        generate_image_remote(cfg, remote_url)
                    case _:
                        cfg.prompt = line
                        generate_image_remote(cfg, remote_url)
        return

    # Local mode - import heavy modules now (requires CUDA)
    import torch

    from flux2.entrypoints.startup import check_model_availability
    from flux2.openai_client import get_openai_client
    from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

    cfg = Config()

    if model_name is not None:
        cfg.model_name = model_name
    if enable_moderation is not None:
        cfg.enable_moderation = enable_moderation

    model_name = cfg.model_name
    if model_name.lower() not in FLUX2_MODEL_INFO:
        available = ", ".join(FLUX2_MODEL_INFO.keys())
        click.echo(f"Unknown model: {model_name}. Available: {available}", err=True)
        sys.exit(1)

    model_info = FLUX2_MODEL_INFO[model_name]
    defaults = model_info["defaults"]

    if num_steps is not None:
        cfg.num_steps = num_steps
    else:
        cfg.num_steps = int(defaults["num_steps"])

    if guidance is not None:
        cfg.guidance = guidance
    else:
        cfg.guidance = defaults["guidance"]

    if prompt is not None:
        cfg.prompt = prompt
    if seed is not None:
        cfg.seed = seed
    if width is not None:
        cfg.width = width
    if height is not None:
        cfg.height = height
    if input_images:
        cfg.input_images = list(input_images)
    if upsample is not None:
        cfg.upsample_prompt_mode = upsample  # type: ignore[assignment]

    if not validate_model_params(model_name, cfg, model_info):
        sys.exit(1)

    status = check_model_availability(model_name, check_moderation=cfg.enable_moderation)
    if not status.ready_for_inference:
        if not status.flow_model:
            click.echo(f"Error: Flow model not cached. Run: hf download {model_info['repo_id']}", err=True)
        if not status.text_encoder:
            variant = "4B" if "4b" in model_name.lower() else "8B"
            click.echo(f"Error: Text encoder not cached. Run: hf download Qwen/Qwen3-{variant}-FP8", err=True)
        if not status.autoencoder:
            click.echo(
                "Error: Autoencoder not cached. Run: hf download black-forest-labs/FLUX.2-dev", err=True
            )
        sys.exit(1)
    if cfg.enable_moderation and not status.moderation_model:
        click.echo(
            "Error: Moderation model not cached. Run: hf download mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            err=True,
        )
        sys.exit(1)

    torch_device = torch.device("cuda")
    click.echo(f"Loading {model_name}...")

    text_encoder = load_text_encoder(model_name, device=torch_device)
    text_encoder.eval()

    is_klein = "klein" in model_name
    if is_klein and cfg.enable_moderation:
        click.echo("Loading moderation model (Mistral-24B)...")
        mod_and_upsampling_model = load_text_encoder("flux.2-dev")
        mod_and_upsampling_model.eval()
    else:
        mod_and_upsampling_model = None if is_klein else text_encoder

    model = load_flow_model(model_name, debug_mode=debug, device="cpu" if cpu_offload else torch_device)
    ae = load_ae(model_name)
    ae.eval()

    api_upsampling_client = None
    if cfg.upsample_prompt_mode == "api":
        api_upsampling_client = get_openai_client()
        click.echo(f"API upsampling: {api_upsampling_client.base_url} ({api_upsampling_client.model})")

    print_config(cfg)

    if single_eval:
        generate_image(
            cfg,
            model_name,
            model,
            ae,
            text_encoder,
            mod_and_upsampling_model,
            api_upsampling_client,
            model_info,
            torch_device,
            cpu_offload,
            enable_profiler=profile,
        )
        return

    while True:
        try:
            line = click.prompt(">", default="", show_default=False).strip()
        except (EOFError, KeyboardInterrupt, click.Abort):
            click.echo("\nbye!")
            break

        match line:
            case "quit" | "q" | "exit":
                click.echo("bye!")
                break
            case "show":
                print_config(cfg)
            case "help" | "h" | "?":
                click.echo("Commands: run, show, reset, quit")
                click.echo("Set params: prompt=... width=... seed=...")
            case "reset":
                cfg = Config()
                cfg.num_steps = int(defaults["num_steps"])
                cfg.guidance = float(defaults["guidance"])
                print_config(cfg)
            case "" | "run":
                try:
                    generate_image(
                        cfg,
                        model_name,
                        model,
                        ae,
                        text_encoder,
                        mod_and_upsampling_model,
                        api_upsampling_client,
                        model_info,
                        torch_device,
                        cpu_offload,
                        enable_profiler=profile,
                    )
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
            case _ if "=" in line:
                for part in line.split():
                    if "=" not in part:
                        continue
                    key, val = part.split("=", 1)
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    try:
                        match key:
                            case "prompt":
                                cfg.prompt = val
                            case "seed":
                                cfg.seed = int(val) if val.lower() != "none" else None
                            case "width":
                                cfg.width = int(val)
                            case "height":
                                cfg.height = int(val)
                            case "num_steps":
                                cfg.num_steps = int(val)
                            case "guidance":
                                cfg.guidance = float(val)
                            case "input_images":
                                cfg.input_images = [Path(p.strip()) for p in val.split(",") if p.strip()]
                            case "match_image_size":
                                cfg.match_image_size = int(val) if val.lower() != "none" else None
                            case "upsample_prompt_mode":
                                cfg.upsample_prompt_mode = val
                            case _:
                                click.echo(f"Unknown key: {key}", err=True)
                    except ValueError as e:
                        click.echo(f"Invalid value for {key}: {e}", err=True)
                if validate_model_params(model_name, cfg, model_info):
                    print_config(cfg)
            case _:
                cfg.prompt = line
                print_config(cfg)


if __name__ == "__main__":
    main()
