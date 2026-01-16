"""VAE loading utilities with format detection and selection."""

import os
from pathlib import Path

import torch
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams

_DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface"


class VAEFormatError(Exception):
    """Raised when VAE format is incompatible."""

    pass


def get_expected_vae_keys() -> set[str]:
    """Get expected state dict keys from AutoEncoder model."""
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())
    return set(ae.state_dict().keys())


def check_vae_compatibility(state_dict: dict[str, torch.Tensor]) -> tuple[bool, str]:
    """Check if a state dict is compatible with the BFL AutoEncoder.

    Returns:
        Tuple of (is_compatible, message)
    """
    expected_keys = get_expected_vae_keys()
    actual_keys = set(state_dict.keys())

    missing_keys = expected_keys - actual_keys
    extra_keys = actual_keys - expected_keys

    if missing_keys or extra_keys:
        msg_parts = []
        if missing_keys:
            sample = sorted(missing_keys)[:5]
            msg_parts.append(f"missing {len(missing_keys)} keys (e.g., {sample})")
        if extra_keys:
            sample = sorted(extra_keys)[:5]
            msg_parts.append(f"extra {len(extra_keys)} keys (e.g., {sample})")
        return False, "; ".join(msg_parts)

    return True, "all keys match"


def detect_vae_format(weight_path: Path) -> str:
    """Detect VAE format from file path.

    Returns:
        'bfl_native' for ae.safetensors format
        'diffusers' for vae/diffusion_pytorch_model.safetensors format
        'unknown' for other formats
    """
    path_str = str(weight_path)
    filename = weight_path.name

    if filename == "ae.safetensors":
        return "bfl_native"
    elif "diffusion_pytorch_model" in filename or "/vae/" in path_str:
        return "diffusers"
    return "unknown"


def find_vae_path(hf_home: Path | None = None) -> tuple[Path | None, str]:
    """Find VAE weights, preferring BFL native format.

    Args:
        hf_home: HuggingFace home directory (defaults to ~/.cache/huggingface)

    Returns:
        Tuple of (path, format) where format is 'bfl_native', 'diffusers', or None
    """
    if hf_home is None:
        hf_home = Path(os.environ.get("HF_HOME", _DEFAULT_HF_HOME))

    hub_dir = hf_home / "hub" if not str(hf_home).endswith("hub") else hf_home

    # Model repos to search (order matters - BFL native repos first)
    repo_ids = [
        "black-forest-labs/FLUX.2-dev",  # Most likely to have ae.safetensors
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-base-9B",
        "black-forest-labs/FLUX.2-klein-9B",
    ]

    # First pass: look for BFL native format (ae.safetensors)
    for repo_id in repo_ids:
        cache_name = f"models--{repo_id.replace('/', '--')}"
        model_cache = hub_dir / cache_name
        if not model_cache.exists():
            continue

        snapshots_dir = model_cache / "snapshots"
        if not snapshots_dir.exists():
            continue

        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            continue

        latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
        ae_file = latest_snapshot / "ae.safetensors"
        if ae_file.exists():
            return ae_file.resolve(), "bfl_native"

    # Second pass: look for diffusers format (fallback)
    for repo_id in repo_ids:
        cache_name = f"models--{repo_id.replace('/', '--')}"
        model_cache = hub_dir / cache_name
        if not model_cache.exists():
            continue

        snapshots_dir = model_cache / "snapshots"
        if not snapshots_dir.exists():
            continue

        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            continue

        latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
        vae_file = latest_snapshot / "vae" / "diffusion_pytorch_model.safetensors"
        if vae_file.exists():
            return vae_file.resolve(), "diffusers"

    return None, "none"


def load_vae_state_dict(weight_path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load VAE state dict with format validation.

    Args:
        weight_path: Path to VAE weights
        device: Device to load tensors to

    Returns:
        State dict compatible with BFL AutoEncoder

    Raises:
        VAEFormatError: If format is incompatible
    """
    vae_format = detect_vae_format(weight_path)
    sd = load_sft(str(weight_path), device=device)

    is_compatible, msg = check_vae_compatibility(sd)

    if not is_compatible:
        if vae_format == "diffusers":
            raise VAEFormatError(
                f"Diffusers VAE format at '{weight_path}' is incompatible with BFL AutoEncoder: {msg}. "
                "The diffusers format uses different tensor shapes (Linear vs Conv2d for attention). "
                "Please download the BFL native 'ae.safetensors' from the FLUX.2-dev model repository."
            )
        else:
            raise VAEFormatError(f"VAE at '{weight_path}' has incompatible format: {msg}")

    return sd
