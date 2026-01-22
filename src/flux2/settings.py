import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _find_hf_model_path(repo_id: str, filename: str) -> Path | None:
    """Find a model file in HuggingFace cache directory."""
    hf_home = Path(os.environ.get("HF_HOME", _DEFAULT_HF_HOME))
    hub_dir = hf_home / "hub" if not str(hf_home).endswith("hub") else hf_home

    # Convert repo_id to HF cache format (e.g., "black-forest-labs/FLUX.2-klein-base-4B" -> "models--black-forest-labs--FLUX.2-klein-base-4B")
    cache_name = f"models--{repo_id.replace('/', '--')}"
    model_cache = hub_dir / cache_name

    if not model_cache.exists():
        return None

    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        return None

    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)

    model_file = latest_snapshot / filename
    if model_file.exists():
        return model_file.resolve()

    return None


def _find_hf_vae_path_for_repo(repo_id: str) -> Path | None:
    """Find VAE/autoencoder in HuggingFace cache for a specific repo (BFL native format only)."""
    hf_home = Path(os.environ.get("HF_HOME", _DEFAULT_HF_HOME))
    hub_dir = hf_home / "hub" if not str(hf_home).endswith("hub") else hf_home

    cache_name = f"models--{repo_id.replace('/', '--')}"
    model_cache = hub_dir / cache_name

    if not model_cache.exists():
        return None

    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        return None

    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)

    # Only check for ae.safetensors (BFL native format)
    # Diffusers format (vae/diffusion_pytorch_model.safetensors) has incompatible keys
    ae_file = latest_snapshot / "ae.safetensors"
    if ae_file.exists():
        return ae_file.resolve()

    return None


class HFHomeNotSetError(Exception):
    """Raised when HF_HOME is required but not set."""


class Flux2Settings(BaseSettings):
    """Settings for FLUX.2 model paths with automatic HuggingFace cache discovery."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    strict_mode: bool = Field(
        default=True,
        validation_alias="STRICT_MODE",
        description="Fail fast if HF_HOME is not set or directory doesn't exist",
    )
    enable_moderation: bool = Field(
        default=False,
        validation_alias="ENABLE_MODERATION",
        description="Enable content moderation (requires 50GB Mistral-24B model)",
    )
    model_name: str = Field(
        default="flux.2-klein-base-4b",
        validation_alias="MODEL_NAME",
        description="FLUX.2 model variant to use",
    )
    hf_home: Path | None = Field(
        default=None,
        validation_alias="HF_HOME",
        description="HuggingFace cache directory path",
    )
    flux2_model_path: Path | None = Field(
        default=None,
        validation_alias="FLUX2_MODEL_PATH",
        description="Path to FLUX.2-dev model (auto-discovered from HF cache if not set)",
    )
    ae_model_path: Path | None = Field(
        default=None,
        validation_alias="AE_MODEL_PATH",
        description="Path to autoencoder model (auto-discovered from HF cache if not set)",
    )
    klein_4b_model_path: Path | None = Field(
        default=None,
        validation_alias="KLEIN_4B_MODEL_PATH",
        description="Path to FLUX.2-klein-4B distilled model (auto-discovered from HF cache if not set)",
    )
    klein_4b_base_model_path: Path | None = Field(
        default=None,
        validation_alias="KLEIN_4B_BASE_MODEL_PATH",
        description="Path to FLUX.2-klein-base-4B model (auto-discovered from HF cache if not set)",
    )
    klein_9b_model_path: Path | None = Field(
        default=None,
        validation_alias="KLEIN_9B_MODEL_PATH",
        description="Path to FLUX.2-klein-9B distilled model (auto-discovered from HF cache if not set)",
    )
    klein_9b_base_model_path: Path | None = Field(
        default=None,
        validation_alias="KLEIN_9B_BASE_MODEL_PATH",
        description="Path to FLUX.2-klein-base-9B model (auto-discovered from HF cache if not set)",
    )
    flux2_nvfp4_model_path: Path | None = Field(
        default=None,
        validation_alias="FLUX2_NVFP4_MODEL_PATH",
        description="Path to FLUX.2-dev-NVFP4 model (auto-discovered from HF cache if not set)",
    )
    upsample_prompt_mode: Literal["none", "local", "api"] = Field(
        default="none",
        validation_alias="UPSAMPLE_PROMPT_MODE",
        description="Prompt upsampling mode: none, local (use text encoder), or api (OpenAI-compatible)",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="API key for OpenAI-compatible prompt upsampling service",
    )
    openai_base_url: str = Field(
        default=_DEFAULT_OPENAI_BASE_URL,
        validation_alias="OPENAI_BASE_URL",
        description="Base URL for OpenAI-compatible API",
    )
    openai_model: str = Field(
        default="gpt-4o",
        validation_alias="OPENAI_MODEL",
        description="Model name for OpenAI-compatible prompt upsampling",
    )
    text_encoder_url: str | None = Field(
        default=None,
        validation_alias="TEXT_ENCODER_URL",
        description="URL of remote text encoder service for distributed inference",
    )
    inference_url: str | None = Field(
        default=None,
        validation_alias="INFERENCE_URL",
        description="URL of remote inference service (for CLI --remote default)",
    )
    output_dir: Path = Field(
        default=Path("output"),
        validation_alias="OUTPUT_DIR",
        description="Directory for saving generated images",
    )
    hf_token: SecretStr | None = Field(
        default=None,
        validation_alias="HF_TOKEN",
        description="HuggingFace API token for model downloads",
    )
    hf_hub_offline: bool = Field(
        default=False,
        validation_alias="HF_HUB_OFFLINE",
        description="Run in offline mode - no network requests to HuggingFace Hub",
    )
    text_encoder_fp8: bool = Field(
        default=True,
        validation_alias="TEXT_ENCODER_FP8",
        description="Use FP8 quantized Mistral text encoder for flux.2-dev (reduces VRAM ~50%)",
    )
    torch_compile: bool = Field(
        default=False,
        validation_alias="TORCH_COMPILE",
        description="Use torch.compile() on flow model and autoencoder for faster inference",
    )
    torch_compile_mode: Literal["reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = Field(
        default="reduce-overhead",
        validation_alias="TORCH_COMPILE_MODE",
        description="torch.compile mode: reduce-overhead (CUDA graphs), max-autotune (Triton+CUDA graphs), max-autotune-no-cudagraphs",
    )
    torch_compile_ae: bool = Field(
        default=False,
        validation_alias="TORCH_COMPILE_AE",
        description="Also compile autoencoder (can hurt decode performance with max-autotune)",
    )
    torch_logs: str = Field(
        default="",
        validation_alias="TORCH_LOGS",
        description="TORCH_LOGS categories: perf_hints, graph_breaks, recompiles (comma-separated)",
    )
    profiler_enabled: bool = Field(
        default=False,
        validation_alias="PROFILER_ENABLED",
        description="Enable PyTorch profiler for inference tracing",
    )
    profiler_output_dir: Path = Field(
        default=Path("/profiler"),
        validation_alias="PROFILER_OUTPUT_DIR",
        description="Directory for profiler trace output",
    )
    allow_tf32: bool = Field(
        default=True,
        validation_alias="ALLOW_TF32",
        description="Allow TF32 for matmul/cudnn ops (faster on Ampere+ GPUs with minimal precision loss)",
    )
    text_encoder_quantization: Literal["torchao_fp8", "compressed_fp8", "none"] = Field(
        default="torchao_fp8",
        validation_alias="TEXT_ENCODER_QUANTIZATION",
        description="Text encoder quantization: torchao_fp8 (native FP8 compute), compressed_fp8 (storage-only), none",
    )
    hf_download_workers: int = Field(
        default=4,
        validation_alias="HF_DOWNLOAD_WORKERS",
        description="Number of parallel workers for HuggingFace downloads (default 4)",
    )

    @model_validator(mode="after")
    def validate_strict_mode(self) -> "Flux2Settings":
        """Validate HF_HOME is set and directory exists when strict_mode is enabled."""
        if self.strict_mode:
            if self.hf_home is None:
                raise HFHomeNotSetError(
                    "HF_HOME must be set when running in strict mode. "
                    "Mount your HuggingFace cache directory and set HF_HOME environment variable."
                )
            if not self.hf_home.is_dir():
                raise HFHomeNotSetError(
                    f"HF_HOME directory not found at {self.hf_home}. "
                    "Mount your HuggingFace cache directory to /hf_cache."
                )
        return self

    @model_validator(mode="after")
    def discover_hf_paths(self) -> "Flux2Settings":
        """Auto-discover model paths from HuggingFace cache if not explicitly set."""
        model_mappings = {
            "klein_4b_model_path": ("black-forest-labs/FLUX.2-klein-4B", "flux-2-klein-4b.safetensors"),
            "klein_4b_base_model_path": (
                "black-forest-labs/FLUX.2-klein-base-4B",
                "flux-2-klein-base-4b.safetensors",
            ),
            "klein_9b_model_path": ("black-forest-labs/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors"),
            "klein_9b_base_model_path": (
                "black-forest-labs/FLUX.2-klein-base-9B",
                "flux-2-klein-base-9b.safetensors",
            ),
            "flux2_model_path": ("black-forest-labs/FLUX.2-dev", "flux2-dev.safetensors"),
            "flux2_nvfp4_model_path": ("black-forest-labs/FLUX.2-dev-NVFP4", "flux2-dev-nvfp4.safetensors"),
        }

        for attr, (repo_id, filename) in model_mappings.items():
            if getattr(self, attr) is None:
                discovered = _find_hf_model_path(repo_id, filename)
                if discovered:
                    object.__setattr__(self, attr, discovered)

        # Auto-discover AE path from any available model repo (prefer FLUX.2-dev which has native format)
        if self.ae_model_path is None:
            for repo_id in [
                "black-forest-labs/FLUX.2-dev",  # Most likely to have BFL native ae.safetensors
                "black-forest-labs/FLUX.2-klein-base-4B",
                "black-forest-labs/FLUX.2-klein-4B",
                "black-forest-labs/FLUX.2-klein-base-9B",
                "black-forest-labs/FLUX.2-klein-9B",
            ]:
                vae_path = _find_hf_vae_path_for_repo(repo_id)
                if vae_path:
                    object.__setattr__(self, "ae_model_path", vae_path)
                    break

        return self

    def get_model_path(self, model_name: str) -> Path | None:
        """Get model path by model name."""
        match model_name.lower():
            case "flux.2-klein-4b":
                return self.klein_4b_model_path
            case "flux.2-klein-9b":
                return self.klein_9b_model_path
            case "flux.2-klein-base-4b":
                return self.klein_4b_base_model_path
            case "flux.2-klein-base-9b":
                return self.klein_9b_base_model_path
            case "flux.2-dev":
                return self.flux2_model_path
            case "flux.2-dev-nvfp4":
                return self.flux2_nvfp4_model_path
            case _:
                raise ValueError(f"Unknown model: {model_name}")

    def get_hf_home(self) -> Path:
        """Get effective HF_HOME path.

        Returns:
            Path to HuggingFace cache directory

        Raises:
            HFHomeNotSetError: If HF_HOME is not set in strict mode
        """
        if self.hf_home is not None:
            return self.hf_home
        if self.strict_mode:
            raise HFHomeNotSetError(
                "HF_HOME must be set when running in strict mode. "
                "Mount your HuggingFace cache directory and set HF_HOME environment variable."
            )
        return _DEFAULT_HF_HOME


@lru_cache
def get_settings() -> Flux2Settings:
    """Get cached settings instance."""
    return Flux2Settings()
