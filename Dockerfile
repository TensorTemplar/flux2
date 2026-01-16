# FLUX.2 Inference Container
#
# Build:
#   docker build -t flux2-inference .
#
# Run (mount your HF cache to /hf_cache):
#   docker run -v $HF_HOME:/hf_cache -p 8000:8000 flux2-inference

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up Python environment
ENV UV_PYTHON=python3.12
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Create app directory
WORKDIR /app

# Copy project files
COPY ./pyproject.toml ./uv.lock* README.md ./
COPY src/ src/
COPY scripts/ scripts/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV HF_HOME=/hf_cache
ENV MODEL_NAME=flux.2-klein-base-4b
ENV ENABLE_MODERATION=false

EXPOSE 8000

ENTRYPOINT ["uv", "run", "flux2-serve"]
