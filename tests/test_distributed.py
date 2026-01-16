"""Tests for distributed text encoder components."""

import torch

from flux2.entrypoints.text_encoder_server import EncodeRequest, HealthResponse, ReadyResponse
from flux2.remote_text_encoder import RemoteTextEncoder, RemoteTextEncoderError
from flux2.tensor_transport import deserialize_tensors, serialize_tensors
from flux2.util import FLUX2_MODEL_INFO


class TestTensorTransport:
    """Tests for safetensors serialization/deserialization."""

    def test_serialize_deserialize_roundtrip_float32(self):
        """Test roundtrip with float32 tensors."""
        original = {
            "a": torch.randn(2, 3, dtype=torch.float32),
            "b": torch.randn(4, 5, dtype=torch.float32),
        }

        data = serialize_tensors(original)
        assert isinstance(data, bytes)
        assert len(data) > 0

        restored = deserialize_tensors(data, device="cpu")
        assert set(restored.keys()) == set(original.keys())
        for key in original:
            torch.testing.assert_close(restored[key], original[key])

    def test_serialize_deserialize_roundtrip_bfloat16(self):
        """Test roundtrip preserves bfloat16 dtype."""
        original = {
            "ctx": torch.randn(1, 512, 7680, dtype=torch.bfloat16),
            "ctx_ids": torch.randint(0, 100, (1, 512, 4), dtype=torch.int64),
        }

        data = serialize_tensors(original)
        restored = deserialize_tensors(data, device="cpu")

        assert restored["ctx"].dtype == torch.bfloat16
        assert restored["ctx_ids"].dtype == torch.int64
        torch.testing.assert_close(restored["ctx"], original["ctx"])
        torch.testing.assert_close(restored["ctx_ids"], original["ctx_ids"])

    def test_serialize_empty_dict(self):
        """Test serializing empty dict."""
        data = serialize_tensors({})
        assert isinstance(data, bytes)
        restored = deserialize_tensors(data, device="cpu")
        assert restored == {}

    def test_serialize_large_tensor(self):
        """Test serializing larger tensor (simulating real text encoder output)."""
        # Typical text encoder output: (batch, seq_len, context_dim)
        ctx = torch.randn(1, 512, 12288, dtype=torch.bfloat16)
        ctx_ids = torch.zeros(1, 512, 4, dtype=torch.int64)

        data = serialize_tensors({"ctx": ctx, "ctx_ids": ctx_ids})

        # ~12MB for ctx + ~16KB for ctx_ids
        assert len(data) > 10_000_000  # At least 10MB

        restored = deserialize_tensors(data, device="cpu")
        torch.testing.assert_close(restored["ctx"], ctx)

    def test_deserialize_to_device(self):
        """Test deserializing to specified device."""
        original = {"x": torch.randn(2, 3)}
        data = serialize_tensors(original)

        # Deserialize to CPU explicitly
        restored = deserialize_tensors(data, device="cpu")
        assert restored["x"].device == torch.device("cpu")

    def test_multiple_dtypes(self):
        """Test tensors with multiple dtypes."""
        original = {
            "float32": torch.randn(2, 3, dtype=torch.float32),
            "float16": torch.randn(2, 3, dtype=torch.float16),
            "bfloat16": torch.randn(2, 3, dtype=torch.bfloat16),
            "int64": torch.randint(0, 100, (2, 3), dtype=torch.int64),
            "int32": torch.randint(0, 100, (2, 3), dtype=torch.int32),
        }

        data = serialize_tensors(original)
        restored = deserialize_tensors(data, device="cpu")

        for key in original:
            assert restored[key].dtype == original[key].dtype
            torch.testing.assert_close(restored[key], original[key])


class TestTextEncoderServerEndpoints:
    """Tests for text encoder server request/response models."""

    def test_encode_request_model(self):
        """Test EncodeRequest pydantic model."""
        # Basic request
        req = EncodeRequest(prompts=["a cat"])
        assert req.prompts == ["a cat"]
        assert req.include_empty is False

        # With include_empty
        req = EncodeRequest(prompts=["a cat", "a dog"], include_empty=True)
        assert len(req.prompts) == 2
        assert req.include_empty is True

    def test_health_response_model(self):
        """Test HealthResponse pydantic model."""
        resp = HealthResponse(status="ok", model_name="flux.2-klein-base-4b")
        assert resp.status == "ok"
        assert resp.model_name == "flux.2-klein-base-4b"

    def test_ready_response_model(self):
        """Test ReadyResponse pydantic model."""
        resp = ReadyResponse(ready=True, encoder_loaded=True)
        assert resp.ready is True
        assert resp.encoder_loaded is True


class TestRemoteTextEncoder:
    """Tests for remote text encoder client."""

    def test_client_initialization(self):
        """Test RemoteTextEncoder client initialization."""
        client = RemoteTextEncoder("http://localhost:8001")
        assert client.base_url == "http://localhost:8001"
        assert client.timeout == 60.0

        client = RemoteTextEncoder("http://localhost:8001/", timeout=30.0)
        assert client.base_url == "http://localhost:8001"  # Trailing slash stripped
        assert client.timeout == 30.0

    def test_remote_encoder_error(self):
        """Test RemoteTextEncoderError exception."""
        err = RemoteTextEncoderError("Connection failed")
        assert str(err) == "Connection failed"


class TestModelInfoTyping:
    """Tests for FLUX2_MODEL_INFO type annotations."""

    def test_model_info_structure(self):
        """Test that all model info entries have required keys."""
        required_keys = {
            "repo_id",
            "filename",
            "filename_ae",
            "params",
            "text_encoder_load_fn",
            "model_path",
            "defaults",
            "fixed_params",
            "guidance_distilled",
        }

        for model_name, info in FLUX2_MODEL_INFO.items():
            assert set(info.keys()) == required_keys, f"Missing keys in {model_name}"
            assert isinstance(info["repo_id"], str)
            assert isinstance(info["filename"], str)
            assert isinstance(info["defaults"], dict)
            assert isinstance(info["fixed_params"], set)
            assert isinstance(info["guidance_distilled"], bool)
            assert callable(info["text_encoder_load_fn"])

    def test_model_defaults_have_required_params(self):
        """Test that model defaults contain guidance and num_steps."""
        for model_name, info in FLUX2_MODEL_INFO.items():
            defaults = info["defaults"]
            assert "guidance" in defaults, f"Missing 'guidance' in {model_name}"
            assert "num_steps" in defaults, f"Missing 'num_steps' in {model_name}"
