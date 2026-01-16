import base64
import io
import sys
from pathlib import Path
from typing import Callable, TypedDict

import huggingface_hub
import huggingface_hub.errors
import torch
from PIL import Image
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import Flux2, Flux2Params, Klein4BParams, Klein9BParams
from .settings import get_settings
from .text_encoder import (
    Mistral3SmallEmbedder,
    Qwen3Embedder,
    load_mistral_small_embedder,
    load_qwen3_embedder,
)
from .vae_loader import VAEFormatError, load_vae_state_dict


class ModelInfo(TypedDict):
    repo_id: str
    filename: str
    filename_ae: str
    params: Flux2Params | Klein4BParams | Klein9BParams
    text_encoder_load_fn: Callable[..., Mistral3SmallEmbedder | Qwen3Embedder]
    model_path: str
    defaults: dict[str, float | int]
    fixed_params: set[str]
    guidance_distilled: bool


FLUX2_MODEL_INFO: dict[str, ModelInfo] = {
    "flux.2-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "filename": "flux-2-klein-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
    },
    "flux.2-klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "filename": "flux-2-klein-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
    },
    "flux.2-klein-base-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "filename": "flux-2-klein-base-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": set(),
        "guidance_distilled": False,
    },
    "flux.2-klein-base-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "filename": "flux-2-klein-base-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": set(),
        "guidance_distilled": False,
    },
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Flux2Params(),
        "text_encoder_load_fn": load_mistral_small_embedder,
        "model_path": "FLUX2_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": set(),
        "guidance_distilled": True,
    },
}


class ModelNotAvailableError(Exception):
    """Raised when a required model is not available in the HF cache."""


def check_flow_model_available(model_name: str) -> bool:
    """Check if flow model weights are available in HF cache."""
    settings = get_settings()
    return settings.get_model_path(model_name) is not None


def check_text_encoder_available(model_name: str) -> bool:
    """Check if text encoder model is available in HF cache.

    For klein models, checks Qwen3. For flux.2-dev, checks Mistral.
    """
    # Validate model name exists
    _ = FLUX2_MODEL_INFO[model_name.lower()]

    if "klein" in model_name.lower():
        variant = "4B" if "4b" in model_name.lower() else "8B"
        encoder_repo = f"Qwen/Qwen3-{variant}-FP8"
    else:
        encoder_repo = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

    try:
        huggingface_hub.snapshot_download(
            repo_id=encoder_repo,
            local_files_only=True,
            repo_type="model",
        )
        return True
    except huggingface_hub.errors.LocalEntryNotFoundError:
        return False
    except Exception:
        return False


def check_ae_available() -> bool:
    """Check if VAE/autoencoder weights are available in HF cache."""
    settings = get_settings()
    return settings.ae_model_path is not None


def check_moderation_model_available() -> bool:
    """Check if Mistral moderation/upsampling model is available in HF cache."""
    try:
        huggingface_hub.snapshot_download(
            repo_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            local_files_only=True,
            repo_type="model",
        )
        return True
    except huggingface_hub.errors.LocalEntryNotFoundError:
        return False
    except Exception:
        return False


def load_flow_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]
    settings = get_settings()

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        # Try to get path from settings (auto-discovered or env var)
        weight_path = settings.get_model_path(model_name)

        if weight_path is None:
            # Fallback to HuggingFace download
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    repo_type="model",
                )
            except huggingface_hub.errors.RepositoryNotFoundError:
                print(
                    f"Failed to access the model repository. Please check your internet "
                    f"connection and make sure you've access to {config['repo_id']}."
                    "Stopping."
                )
                sys.exit(1)
        else:
            weight_path = str(weight_path)
            print(f"Using model path from settings: {weight_path}")

    if not debug_mode:
        with torch.device("meta"):
            model = Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)
        print(f"Loading {weight_path} for the FLUX.2 weights")
        sd = load_sft(weight_path, device=str(device))
        model.load_state_dict(sd, strict=True, assign=True)
        return model.to(device)
    else:
        with torch.device(device):
            return Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)


def load_text_encoder(model_name: str, device: str | torch.device = "cuda"):
    config = FLUX2_MODEL_INFO[model_name.lower()]
    return config["text_encoder_load_fn"](device=device)


def load_ae(model_name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    config = FLUX2_MODEL_INFO[model_name.lower()]
    settings = get_settings()

    weight_path = settings.ae_model_path

    if weight_path is None:
        try:
            weight_path = huggingface_hub.hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename_ae"],
                repo_type="model",
            )
        except huggingface_hub.errors.RepositoryNotFoundError:
            print(
                f"Failed to access the model repository. Please check your internet "
                f"connection and make sure you've access to {config['repo_id']}."
                "Stopping."
            )
            sys.exit(1)
    else:
        weight_path = str(weight_path)
        print(f"Using AE path from settings: {weight_path}")

    if isinstance(device, str):
        device = torch.device(device)
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    print(f"Loading {weight_path} for the AutoEncoder weights")
    try:
        sd = load_vae_state_dict(Path(weight_path), device=str(device))
    except VAEFormatError as e:
        print(f"VAE format error: {e}")
        sys.exit(1)

    ae.load_state_dict(sd, strict=True, assign=True)
    return ae.to(device)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
