"""Async HTTP client for remote text encoder service.

Drop-in replacement for local text encoder when TEXT_ENCODER_URL is configured.
"""

import httpx
import torch
from torch import Tensor

from .tensor_transport import deserialize_tensors


class RemoteTextEncoderError(Exception):
    """Raised when remote text encoder request fails."""


class RemoteTextEncoder:
    """Async client for remote text encoder service."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        """Initialize client.

        Args:
            base_url: Base URL of text encoder service (e.g., http://text-encoder:8001)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def encode(
        self,
        prompts: list[str],
        include_empty: bool = False,
        device: str | torch.device = "cuda",
    ) -> tuple[Tensor, Tensor]:
        """Encode prompts via remote service.

        Args:
            prompts: List of text prompts to encode
            include_empty: Include empty string for CFG (prepended to batch)
            device: Device to move tensors to after deserialization

        Returns:
            Tuple of (ctx, ctx_ids) tensors on specified device
        """
        client = await self._get_client()

        response = await client.post(
            "/encode",
            json={"prompts": prompts, "include_empty": include_empty},
        )

        if response.status_code != 200:
            raise RemoteTextEncoderError(
                f"Text encoder request failed: {response.status_code} {response.text}"
            )

        tensors = deserialize_tensors(response.content, device=device)
        return tensors["ctx"], tensors["ctx_ids"]

    async def upsample(
        self,
        prompts: list[str],
        temperature: float = 0.15,
    ) -> list[str]:
        """Upsample prompts via remote text encoder.

        Args:
            prompts: List of text prompts to upsample
            temperature: Sampling temperature (default 0.15)

        Returns:
            List of upsampled prompts
        """
        client = await self._get_client()

        response = await client.post(
            "/upsample",
            json={"prompts": prompts, "temperature": temperature},
        )

        if response.status_code == 501:
            raise RemoteTextEncoderError("Upsampling not supported by remote encoder (Qwen models)")

        if response.status_code != 200:
            raise RemoteTextEncoderError(f"Upsample request failed: {response.status_code} {response.text}")

        return response.json()["prompts"]

    async def health(self) -> dict:
        """Check service health."""
        client = await self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()

    async def ready(self) -> bool:
        """Check if service is ready."""
        client = await self._get_client()
        response = await client.get("/ready")
        if response.status_code != 200:
            return False
        return response.json()["ready"]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "RemoteTextEncoder":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
