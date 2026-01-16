"""OpenAI-compatible API client for prompt upsampling.

Supports any OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, etc.)
"""

from functools import lru_cache

from openai import OpenAI
from PIL import Image

from .settings import get_settings
from .system_messages import SYSTEM_MESSAGE_UPSAMPLING_I2I, SYSTEM_MESSAGE_UPSAMPLING_T2I
from .util import image_to_base64


class OpenAIClientNotConfiguredError(Exception):
    """Raised when OpenAI client is requested but OPENAI_API_KEY is not set."""


class OpenAIClient:
    """Client for OpenAI-compatible API prompt upsampling."""

    def __init__(self):
        """Initialize from settings. Fails if OPENAI_API_KEY not set."""
        settings = get_settings()
        if settings.openai_api_key is None:
            raise OpenAIClientNotConfiguredError(
                "OPENAI_API_KEY not set. Set environment variable or use upsample_prompt_mode='none'"
            )
        self.model = settings.openai_model
        self.base_url = settings.openai_base_url
        self.max_tokens = 768
        self.temperature = 0.2
        self.client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
        )

    def _format_messages(
        self,
        prompt: str,
        system_message: str,
        images: list[Image.Image] | None = None,
    ) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": system_message},
        ]

        if images:
            content = []
            for img in images:
                img_base64 = image_to_base64(img)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    }
                )
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

    def upsample_prompt(
        self,
        txt: list[str],
        img: list[Image.Image] | list[list[Image.Image]] | None = None,
    ) -> list[str]:
        """Upsample prompts using OpenAI-compatible API.

        Args:
            txt: List of input prompts to upsample
            img: Optional list of images or list of lists of images.
                 If None or empty, uses t2i mode, otherwise i2i mode.

        Returns:
            List of upsampled prompts
        """
        has_images = img is not None and len(img) > 0
        if has_images and isinstance(img[0], list):
            has_images = len(img[0]) > 0

        system_message = SYSTEM_MESSAGE_UPSAMPLING_I2I if has_images else SYSTEM_MESSAGE_UPSAMPLING_T2I

        upsampled_prompts = []

        for i, prompt in enumerate(txt):
            prompt_images: list[Image.Image] | None = None
            if img is not None and len(img) > i:
                if isinstance(img[i], list):
                    prompt_images = img[i] if len(img[i]) > 0 else None
                elif isinstance(img[i], Image.Image):
                    prompt_images = [img[i]]

            messages = self._format_messages(
                prompt=prompt,
                system_message=system_message,
                images=prompt_images,
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            upsampled = response.choices[0].message.content.strip()
            upsampled_prompts.append(upsampled)

        return upsampled_prompts


@lru_cache
def get_openai_client() -> OpenAIClient:
    """Get cached OpenAI client singleton. Fails if not configured."""
    return OpenAIClient()
