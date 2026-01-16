"""Startup validation for FLUX.2 container deployment.

Exit codes:
- 0: All checks passed
- 1: HF_HOME not set or invalid
- 2: Required models missing
"""

import sys
from dataclasses import dataclass

import click

from ..settings import HFHomeNotSetError, get_settings
from ..util import (
    FLUX2_MODEL_INFO,
    check_ae_available,
    check_flow_model_available,
    check_moderation_model_available,
    check_text_encoder_available,
)


@dataclass
class ModelStatus:
    flow_model: bool
    text_encoder: bool
    autoencoder: bool
    moderation_model: bool

    @property
    def ready_for_inference(self) -> bool:
        return self.flow_model and self.text_encoder and self.autoencoder

    @property
    def ready_for_moderation(self) -> bool:
        return self.ready_for_inference and self.moderation_model


def check_model_availability(model_name: str, check_moderation: bool = False) -> ModelStatus:
    """Check if required models are available in HF cache."""
    return ModelStatus(
        flow_model=check_flow_model_available(model_name),
        text_encoder=check_text_encoder_available(model_name),
        autoencoder=check_ae_available(),
        moderation_model=check_moderation_model_available() if check_moderation else False,
    )


def print_status(model_name: str, status: ModelStatus, enable_moderation: bool) -> None:
    """Print model availability status."""
    print(f"Model: {model_name}")
    print(f"  Flow model:      {'OK' if status.flow_model else 'MISSING'}")
    print(f"  Text encoder:    {'OK' if status.text_encoder else 'MISSING'}")
    print(f"  Autoencoder:     {'OK' if status.autoencoder else 'MISSING'}")
    if enable_moderation:
        print(f"  Moderation:      {'OK' if status.moderation_model else 'MISSING'}")
    print()


@click.command()
@click.option("--model-name", "-m", default=None, help="Model to check (default: MODEL_NAME env)")
@click.option("--enable-moderation", is_flag=True, default=None, help="Check moderation model")
def main(model_name: str | None, enable_moderation: bool | None) -> None:
    """Startup check - fast fail if models missing."""
    # Load settings (validates HF_HOME in strict mode)
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
    if not hf_home.exists():
        print(f"ERROR: HF_HOME directory does not exist: {hf_home}", file=sys.stderr)
        sys.exit(1)

    print(f"HF_HOME: {hf_home}")
    print(f"Strict mode: {settings.strict_mode}")
    print()

    # Check model availability
    status = check_model_availability(model_name, check_moderation=enable_moderation)
    print_status(model_name, status, enable_moderation)

    # Determine if ready
    if enable_moderation:
        ready = status.ready_for_moderation
    else:
        ready = status.ready_for_inference

    if ready:
        print("All required models are available.")
        sys.exit(0)
    else:
        print("ERROR: Some required models are missing.", file=sys.stderr)
        print("Run 'flux2-setup' to download missing models.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
