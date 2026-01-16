"""Tests for VAE format compatibility."""

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

import flux2.settings
from flux2.autoencoder import AutoEncoder, AutoEncoderParams
from flux2.settings import get_settings
from flux2.vae_loader import (
    VAEFormatError,
    check_vae_compatibility,
    detect_vae_format,
    load_vae_state_dict,
)


@pytest.fixture
def expected_vae_keys() -> set[str]:
    """Get expected state dict keys from AutoEncoder model."""
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())
    return set(ae.state_dict().keys())


@pytest.fixture
def settings():
    """Get settings with cache cleared for fresh discovery."""
    flux2.settings.get_settings.cache_clear()
    return get_settings()


class TestVAEFormatCompatibility:
    """Test VAE state dict format compatibility."""

    def test_vae_key_count_matches(self, expected_vae_keys: set[str], settings):
        """Verify loaded VAE has same number of keys as expected."""
        if settings.ae_model_path is None:
            pytest.skip("No ae.safetensors found in HF cache")

        actual_sd = load_file(str(settings.ae_model_path))
        actual_keys = set(actual_sd.keys())

        assert len(actual_keys) == len(
            expected_vae_keys
        ), f"Key count mismatch: expected {len(expected_vae_keys)}, got {len(actual_keys)}"

    def test_vae_keys_match_exactly(self, expected_vae_keys: set[str], settings):
        """Verify loaded VAE has exactly the same keys as expected."""
        if settings.ae_model_path is None:
            pytest.skip("No ae.safetensors found in HF cache")

        actual_sd = load_file(str(settings.ae_model_path))
        actual_keys = set(actual_sd.keys())

        missing_keys = expected_vae_keys - actual_keys
        extra_keys = actual_keys - expected_vae_keys

        assert not missing_keys, f"Missing keys in loaded VAE: {sorted(missing_keys)[:10]}..."
        assert not extra_keys, f"Extra keys in loaded VAE: {sorted(extra_keys)[:10]}..."

    def test_vae_keys_have_expected_prefixes(self, expected_vae_keys: set[str]):
        """Verify expected VAE keys have correct module prefixes."""
        prefixes = {"encoder.", "decoder.", "bn."}

        for key in expected_vae_keys:
            has_valid_prefix = any(key.startswith(p) for p in prefixes)
            assert has_valid_prefix, f"Unexpected key prefix: {key}"

    def test_diffusers_vae_format_incompatible(self, expected_vae_keys: set[str], settings):
        """Verify diffusers VAE format is detected as incompatible.

        Diffusers VAE uses different key naming conventions that don't match
        the BFL native AutoEncoder state dict format.
        """
        # Diffusers format uses these patterns (not compatible):
        diffusers_patterns = [
            "decoder.conv_norm_out",  # vs decoder.norm_out
            "decoder.mid_block.attentions",  # vs decoder.mid.attn_1
            "encoder.down_blocks",  # vs encoder.down
        ]

        # Our expected format should NOT have diffusers patterns
        for pattern in diffusers_patterns:
            matching = [k for k in expected_vae_keys if pattern in k]
            assert not matching, f"Unexpected diffusers pattern '{pattern}' found in expected keys"


class TestVAELoader:
    """Tests for VAE loader functions."""

    def test_detect_bfl_native_format(self):
        """Verify BFL native format detection."""
        assert detect_vae_format(Path("/some/path/ae.safetensors")) == "bfl_native"

    def test_detect_diffusers_format(self):
        """Verify diffusers format detection."""
        assert detect_vae_format(Path("/vae/diffusion_pytorch_model.safetensors")) == "diffusers"
        assert detect_vae_format(Path("/some/model/vae/model.safetensors")) == "diffusers"

    def test_detect_unknown_format(self):
        """Verify unknown format detection."""
        assert detect_vae_format(Path("/some/random.safetensors")) == "unknown"

    def test_check_compatibility_passes_for_valid_sd(self, expected_vae_keys: set[str]):
        """Verify compatibility check passes for matching state dict."""
        # Create a fake state dict with expected keys
        fake_sd = {k: torch.zeros(1) for k in expected_vae_keys}
        is_compatible, msg = check_vae_compatibility(fake_sd)
        assert is_compatible
        assert msg == "all keys match"

    def test_check_compatibility_fails_for_missing_keys(self, expected_vae_keys: set[str]):
        """Verify compatibility check fails for missing keys."""
        # Create a state dict missing some keys
        partial_keys = list(expected_vae_keys)[:10]
        fake_sd = {k: torch.zeros(1) for k in partial_keys}
        is_compatible, msg = check_vae_compatibility(fake_sd)
        assert not is_compatible
        assert "missing" in msg

    def test_check_compatibility_fails_for_extra_keys(self, expected_vae_keys: set[str]):
        """Verify compatibility check fails for extra keys."""
        fake_sd = {k: torch.zeros(1) for k in expected_vae_keys}
        fake_sd["extra.unexpected.key"] = torch.zeros(1)
        is_compatible, msg = check_vae_compatibility(fake_sd)
        assert not is_compatible
        assert "extra" in msg

    def test_load_vae_state_dict_with_compatible_format(self, settings):
        """Verify loading works with BFL native format."""
        if settings.ae_model_path is None:
            pytest.skip("No ae.safetensors found in HF cache")

        sd = load_vae_state_dict(settings.ae_model_path)
        assert len(sd) == 251  # Expected number of keys

    def test_load_vae_state_dict_raises_for_diffusers_format(self):
        """Verify loading raises error for incompatible diffusers format."""
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        diffusers_path = hf_home / "hub" / "models--black-forest-labs--FLUX.2-klein-base-4B" / "snapshots"
        if not diffusers_path.exists():
            pytest.skip("Klein-base-4B not in HF cache")

        # Find any snapshot
        snapshots = list(diffusers_path.iterdir())
        if not snapshots:
            pytest.skip("No snapshots found")

        vae_path = snapshots[0] / "vae" / "diffusion_pytorch_model.safetensors"
        if not vae_path.exists():
            pytest.skip("Diffusers VAE not found")

        with pytest.raises(VAEFormatError) as exc_info:
            load_vae_state_dict(vae_path)

        assert "incompatible" in str(exc_info.value).lower()
