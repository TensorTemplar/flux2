"""Setup/download models for FLUX.2 container deployment.

Exit codes:
- 0: All models downloaded successfully
- 1: HF_HOME not set or invalid
- 2: Download failed
"""

import sys

import click
import huggingface_hub

from ..settings import HFHomeNotSetError, get_settings
from ..util import (
    FLUX2_MODEL_INFO,
    check_ae_available,
    check_flow_model_available,
    check_moderation_model_available,
    check_text_encoder_available,
)


def download_flow_model(model_name: str) -> bool:
    """Download flow model weights if missing."""
    if check_flow_model_available(model_name):
        print("  Flow model already cached")
        return True

    config = FLUX2_MODEL_INFO[model_name.lower()]
    print(f"  Downloading flow model from {config['repo_id']}...")

    try:
        huggingface_hub.hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
            repo_type="model",
        )
        print("  Flow model downloaded successfully")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to download flow model: {e}", file=sys.stderr)
        return False


def download_text_encoder(model_name: str) -> bool:
    """Download text encoder model if missing."""
    if check_text_encoder_available(model_name):
        print("  Text encoder already cached")
        return True

    if "klein" in model_name.lower():
        variant = "4B" if "4b" in model_name.lower() else "8B"
        encoder_repo = f"Qwen/Qwen3-{variant}-FP8"
    else:
        encoder_repo = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

    print(f"  Downloading text encoder from {encoder_repo}...")

    try:
        huggingface_hub.snapshot_download(
            repo_id=encoder_repo,
            repo_type="model",
        )
        print("  Text encoder downloaded successfully")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to download text encoder: {e}", file=sys.stderr)
        return False


def download_autoencoder(model_name: str) -> bool:
    """Download autoencoder (VAE) if missing."""
    if check_ae_available():
        print("  Autoencoder already cached")
        return True

    config = FLUX2_MODEL_INFO[model_name.lower()]
    print(f"  Downloading autoencoder from {config['repo_id']}...")

    try:
        huggingface_hub.hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename_ae"],
            repo_type="model",
        )
        print("  Autoencoder downloaded successfully")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to download autoencoder: {e}", file=sys.stderr)
        return False


def download_moderation_model() -> bool:
    """Download Mistral moderation/upsampling model if missing."""
    if check_moderation_model_available():
        print("  Moderation model already cached")
        return True

    encoder_repo = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    print(f"  Downloading moderation model from {encoder_repo}...")

    try:
        huggingface_hub.snapshot_download(
            repo_id=encoder_repo,
            repo_type="model",
        )
        print("  Moderation model downloaded successfully")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to download moderation model: {e}", file=sys.stderr)
        return False


@click.command()
@click.option("--model-name", "-m", default=None, help="Model to download (default: MODEL_NAME env)")
@click.option("--enable-moderation", is_flag=True, default=None, help="Download moderation model")
@click.option("--skip-flow", is_flag=True, help="Skip flow model download")
@click.option("--skip-encoder", is_flag=True, help="Skip text encoder download")
@click.option("--skip-ae", is_flag=True, help="Skip autoencoder download")
def main(
    model_name: str | None,
    enable_moderation: bool | None,
    skip_flow: bool,
    skip_encoder: bool,
    skip_ae: bool,
) -> None:
    """Setup - download missing models."""
    # Load settings
    try:
        settings = get_settings()
    except HFHomeNotSetError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Use settings defaults if not provided
    model_name = model_name or settings.model_name
    if enable_moderation is None:
        enable_moderation = settings.enable_moderation

    # Validate model name
    if model_name.lower() not in FLUX2_MODEL_INFO:
        print(f"ERROR: Unknown model '{model_name}'", file=sys.stderr)
        print(f"Available models: {', '.join(FLUX2_MODEL_INFO.keys())}", file=sys.stderr)
        sys.exit(1)

    # Check HF_HOME
    hf_home = settings.get_hf_home()
    print(f"HF_HOME: {hf_home}")
    print(f"Model: {model_name}")
    print(f"Moderation: {'enabled' if enable_moderation else 'disabled'}")
    print()

    success = True

    # Download required models
    print("Checking and downloading models...")

    if not skip_flow:
        if not download_flow_model(model_name):
            success = False

    if not skip_encoder:
        if not download_text_encoder(model_name):
            success = False

    if not skip_ae:
        if not download_autoencoder(model_name):
            success = False

    if enable_moderation:
        if not download_moderation_model():
            success = False

    print()
    if success:
        print("Setup complete. All required models are available.")
        sys.exit(0)
    else:
        print("ERROR: Some downloads failed.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
