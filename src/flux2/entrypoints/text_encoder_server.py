"""FastAPI server for distributed text encoding.

Runs the text encoder (Mistral/Qwen) as a separate service, returning
encoded embeddings via HTTP using safetensors serialization.
"""

import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import click
import torch
import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from ..sampling import batched_prc_txt
from ..settings import HFHomeNotSetError, get_settings
from ..tensor_transport import serialize_tensors
from ..util import load_text_encoder


class EncodeRequest(BaseModel):
    prompts: list[str] = Field(..., description="Text prompts to encode")
    include_empty: bool = Field(
        default=False,
        description="Include empty string encoding for CFG (prepended to batch)",
    )


class UpsampleRequest(BaseModel):
    prompts: list[str] = Field(..., description="Text prompts to upsample")
    temperature: float = Field(default=0.15, description="Sampling temperature")


class UpsampleResponse(BaseModel):
    prompts: list[str] = Field(..., description="Upsampled prompts")


class HealthResponse(BaseModel):
    status: str
    model_name: str


class ReadyResponse(BaseModel):
    ready: bool
    encoder_loaded: bool


@dataclass
class EncoderState:
    """State container for loaded text encoder."""

    text_encoder: Any = None
    model_name: str = ""
    device: Any = None

    @property
    def is_ready(self) -> bool:
        return self.text_encoder is not None


_state = EncoderState()


def load_encoder() -> None:
    """Load text encoder into memory."""
    global _state

    settings = get_settings()
    model_name = settings.model_name

    print(f"Loading text encoder for {model_name}...")
    torch_device = torch.device("cuda")

    _state.text_encoder = load_text_encoder(model_name, device=torch_device)
    _state.text_encoder.eval()
    _state.model_name = model_name
    _state.device = torch_device

    print("Text encoder loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load encoder on startup."""
    try:
        load_encoder()
    except HFHomeNotSetError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load text encoder: {e}", file=sys.stderr)
        sys.exit(1)

    yield

    global _state
    _state = EncoderState()


app = FastAPI(
    title="FLUX.2 Text Encoder API",
    description="Distributed text encoding service for FLUX.2",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        model_name=_state.model_name or "loading",
    )


@app.get("/ready", response_model=ReadyResponse)
def ready():
    """Readiness probe."""
    return ReadyResponse(
        ready=_state.is_ready,
        encoder_loaded=_state.text_encoder is not None,
    )


@app.post("/encode")
def encode(req: EncodeRequest):
    """Encode text prompts to embeddings.

    Returns safetensors binary with:
    - ctx: (B, seq_len, context_dim) bfloat16
    - ctx_ids: (B, seq_len, 4) int64
    """
    t_start = time.perf_counter()

    if not _state.is_ready:
        return Response(status_code=503, content="Encoder not loaded")

    prompts = req.prompts

    with torch.no_grad():
        if req.include_empty:
            ctx_empty = _state.text_encoder([""]).to(torch.bfloat16)
            ctx_prompt = _state.text_encoder(prompts).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
        else:
            ctx = _state.text_encoder(prompts).to(torch.bfloat16)

        ctx, ctx_ids = batched_prc_txt(ctx)

    data = serialize_tensors({"ctx": ctx.cpu(), "ctx_ids": ctx_ids.cpu()})

    t_total = time.perf_counter() - t_start
    print(f"[encode] prompts={len(prompts)} include_empty={req.include_empty} total={t_total:.2f}s")

    return Response(
        content=data,
        media_type="application/octet-stream",
    )


@app.post("/upsample", response_model=UpsampleResponse)
def upsample(req: UpsampleRequest):
    """Upsample text prompts using the text encoder's language model.

    Uses Mistral's generation capabilities to expand terse prompts into
    detailed image descriptions suitable for FLUX.2.
    """
    t_start = time.perf_counter()

    if not _state.is_ready:
        return Response(status_code=503, content="Encoder not loaded")

    if not hasattr(_state.text_encoder, "upsample_prompt"):
        return Response(status_code=501, content="Upsampling not supported by this encoder")

    with torch.no_grad():
        upsampled = _state.text_encoder.upsample_prompt(
            req.prompts,
            temperature=req.temperature,
        )

    t_total = time.perf_counter() - t_start
    print(f"[upsample] prompts={len(req.prompts)} total={t_total:.2f}s")

    return UpsampleResponse(prompts=upsampled)


@click.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8001, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev only)")
def main(host: str, port: int, reload: bool) -> None:
    """Start the text encoder server."""
    try:
        uvicorn.run("flux2.entrypoints.text_encoder_server:app", host=host, port=port, reload=reload)
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 0)


if __name__ == "__main__":
    main()
