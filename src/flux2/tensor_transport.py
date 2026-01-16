"""Tensor serialization for distributed inference via HTTP.

Uses safetensors format for efficient, safe tensor transport.
"""

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save


def serialize_tensors(tensors: dict[str, torch.Tensor]) -> bytes:
    """Serialize tensor dict to safetensors bytes."""
    return st_save(tensors)


def deserialize_tensors(data: bytes, device: str | torch.device = "cuda") -> dict[str, torch.Tensor]:
    """Deserialize safetensors bytes to tensor dict on device."""
    tensors = st_load(data)
    return {k: v.to(device) for k, v in tensors.items()}
