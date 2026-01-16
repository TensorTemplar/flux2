# FLUX.2 Deployment Architecture

This document describes the containerized deployment architecture for FLUX.2, designed for multi-node K8s scaling with decoupled startup, setup, and inference phases.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Container Lifecycle                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐                  │
│   │ startup  │ ───► │  setup   │ ───► │  serve   │                  │
│   └──────────┘      └──────────┘      └──────────┘                  │
│        │                 │                 │                         │
│        ▼                 ▼                 ▼                         │
│   Validate env      Download         FastAPI server                  │
│   Check models      missing          /health /ready                  │
│   Exit 0/1/2        models           /generate                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    HF_HOME      │
                    │  (mounted vol)  │
                    └─────────────────┘
```

## Components

### Entrypoints

| Command | Purpose | Exit Codes |
|---------|---------|------------|
| `flux2-startup` | Validate environment, check model availability | 0=ready, 1=HF_HOME error, 2=models missing |
| `flux2-setup` | Download missing models to HF cache | 0=success, 1=HF_HOME error, 2=download failed |
| `flux2-serve` | Start FastAPI inference server | 0=clean shutdown, 1=startup failed |
| `flux2-text-encoder` | Start distributed text encoder service | 0=clean shutdown, 1=startup failed |

### Model Loading Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         Model Components                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   │
│  │  Text Encoder   │   │   Flow Model    │   │   Autoencoder   │   │
│  │                 │   │                 │   │                 │   │
│  │  Klein: Qwen3   │   │  Klein: 4B/9B   │   │  ae.safetensors │   │
│  │  (~8GB FP8)     │   │  Dev: 32B       │   │  (~250MB)       │   │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘   │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                               │
│  │ Moderation Model│  ◄── LAZY LOADED (--enable-moderation)        │
│  │                 │                                               │
│  │ Mistral-24B     │      Only for Klein models                    │
│  │ (~50GB)         │      Flux.2-dev reuses text encoder           │
│  └─────────────────┘                                               │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Moderation Model Loading

The moderation model (Mistral-24B) is **lazy-loaded** to avoid unnecessary 50GB downloads:

| Model | Text Encoder | Moderation Model |
|-------|--------------|------------------|
| Klein 4B/9B | Qwen3-4B/8B-FP8 | Mistral-24B (optional, 50GB) |
| Flux.2-dev | Mistral-24B | Same as text encoder |

**Without `--enable-moderation`**: Klein models skip the 50GB Mistral download entirely.

**With `--enable-moderation`**: Full content moderation for prompts and outputs.

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_HOME` | Yes (strict mode) | `~/.cache/huggingface` | HuggingFace cache directory |
| `MODEL_NAME` | No | `flux.2-klein-base-4b` | Model to load |
| `ENABLE_MODERATION` | No | `false` | Load Mistral-24B for content moderation |
| `STRICT_MODE` | No | `false` | Fail if models missing (no auto-download) |
| `TEXT_ENCODER_URL` | No | None | URL of remote text encoder service (distributed mode) |
| `OPENAI_API_KEY` | No | None | API key for prompt upsampling (SecretStr) |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `OPENAI_MODEL` | No | `gpt-4o` | Model for prompt upsampling |

### Settings Discovery

The `Flux2Settings` class (Pydantic BaseSettings) auto-discovers model paths:

1. Check explicit environment variables (e.g., `KLEIN_4B_MODEL_PATH`)
2. Auto-discover from HuggingFace cache structure
3. Fallback to HuggingFace Hub download (if not in strict mode)

```python
from flux2.settings import get_settings

settings = get_settings()
print(settings.klein_4b_base_model_path)  # Auto-discovered or None
print(settings.ae_model_path)              # VAE path (prefers BFL native format)
```

## Docker Usage

### Build

```bash
docker build -t flux2-inference .

# Or with registry
docker build -t registry.local/flux2-inference:latest .
docker push registry.local/flux2-inference:latest
```

### Run Phases

```bash
# 1. Check if models are available (fast-fail)
docker run --rm \
  -v $HF_HOME:/hf_cache:ro \
  flux2-inference startup

# 2. Download missing models (if needed)
docker run --rm \
  -v $HF_HOME:/hf_cache:rw \
  -e STRICT_MODE=false \
  flux2-inference setup

# 3. Start inference server
docker run -d \
  --gpus all \
  -v $HF_HOME:/hf_cache:ro \
  -p 8000:8000 \
  flux2-inference serve
```

### With Moderation

```bash
docker run -d \
  --gpus all \
  -v $HF_HOME:/hf_cache:ro \
  -e ENABLE_MODERATION=true \
  -p 8000:8000 \
  flux2-inference serve
```

### Docker Compose

```bash
# Check models
HF_HOME=/path/to/hf_cache docker compose run --rm flux2-startup

# Download models
HF_HOME=/path/to/hf_cache docker compose --profile setup run --rm flux2-setup

# Start server
HF_HOME=/path/to/hf_cache docker compose up flux2-inference
```

## API Endpoints

### Health Check (Liveness)

```
GET /health
```

```json
{
  "status": "ok",
  "model_name": "flux.2-klein-base-4b",
  "moderation_enabled": false
}
```

### Ready Check (Readiness)

```
GET /ready
```

```json
{
  "ready": true,
  "model_loaded": true,
  "text_encoder_loaded": true,
  "ae_loaded": true,
  "moderation_loaded": false
}
```

### Generate Image

```
POST /generate
Content-Type: application/json

{
  "prompt": "a photo of a cat",
  "width": 1360,
  "height": 768,
  "num_steps": 50,
  "guidance": 4.0,
  "seed": 42
}
```

Response:
```json
{
  "image_base64": "iVBORw0KGgo...",
  "seed": 42,
  "prompt": "a photo of a cat",
  "width": 1360,
  "height": 768,
  "flagged": false
}
```

## K8s Deployment

### Init Container Pattern

```yaml
apiVersion: v1
kind: Pod
spec:
  initContainers:
    - name: model-check
      image: registry.local/flux2-inference:latest
      command: ["startup"]
      env:
        - name: HF_HOME
          value: /hf_cache
      volumeMounts:
        - name: hf-cache
          mountPath: /hf_cache
          readOnly: true

  containers:
    - name: inference
      image: registry.local/flux2-inference:latest
      command: ["serve"]
      ports:
        - containerPort: 8000
      env:
        - name: HF_HOME
          value: /hf_cache
        - name: MODEL_NAME
          value: flux.2-klein-base-4b
      volumeMounts:
        - name: hf-cache
          mountPath: /hf_cache
          readOnly: true
      resources:
        limits:
          nvidia.com/gpu: 1
      livenessProbe:
        httpGet:
          path: /health
          port: 8000
        initialDelaySeconds: 30
      readinessProbe:
        httpGet:
          path: /ready
          port: 8000
        initialDelaySeconds: 120
        periodSeconds: 10

  volumes:
    - name: hf-cache
      persistentVolumeClaim:
        claimName: hf-cache-pvc
```

### Scaling Considerations

1. **Shared HF Cache**: Mount the same PVC across nodes (ReadOnlyMany)
2. **Model Preloading**: Run `flux2-setup` as a Job before deploying inference pods
3. **GPU Scheduling**: Use node selectors or affinity for GPU nodes
4. **Memory**: Klein 4B needs ~12GB VRAM, Klein 9B needs ~20GB VRAM

## Distributed Text Encoder

For multi-GPU setups, the text encoder can run as a separate service, freeing VRAM on the inference server for larger batch sizes or higher resolution.

### Architecture

```
┌──────────────────┐                    ┌──────────────────┐
│ Text Encoder Svc │  POST /encode      │   Inference Svc  │
│ (GPU 0: Qwen)    │───────────────────▶│   (GPU 1: Flow)  │
│ :8001            │  safetensors       │   :8000          │
└──────────────────┘                    └──────────────────┘
```

### Endpoints (Text Encoder Service)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe |
| `/encode` | POST | Encode prompts → safetensors binary |

### Docker Compose (Distributed)

```yaml
services:
  text-encoder:
    image: flux2-inference:latest
    command: flux2-text-encoder --port 8001
    environment:
      - MODEL_NAME=flux.2-klein-base-4b
    volumes:
      - ${HF_HOME}:/hf_cache:ro
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              device_ids: ['0']

  inference:
    image: flux2-inference:latest
    command: flux2-serve --port 8000
    environment:
      - MODEL_NAME=flux.2-klein-base-4b
      - TEXT_ENCODER_URL=http://text-encoder:8001
    volumes:
      - ${HF_HOME}:/hf_cache:ro
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              device_ids: ['1']
    depends_on:
      - text-encoder
```

### Local Testing

```bash
# Terminal 1: Start text encoder
HF_HOME=/path/to/cache MODEL_NAME=flux.2-klein-base-4b flux2-text-encoder --port 8001

# Terminal 2: Start inference with remote encoder
HF_HOME=/path/to/cache TEXT_ENCODER_URL=http://localhost:8001 MODEL_NAME=flux.2-klein-base-4b flux2-serve --port 8000

# Test
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt": "a cat", "width": 1024, "height": 768}'
```

### VRAM Savings

| Configuration | Text Encoder | Inference Server | Total |
|--------------|--------------|------------------|-------|
| Local (Klein 4B) | - | ~12GB | 12GB |
| Distributed | ~7GB (GPU 0) | ~5GB (GPU 1) | 12GB split |

Distributed mode allows running on two smaller GPUs instead of one large GPU.

## File Structure

```
src/flux2/
├── entrypoints/
│   ├── __init__.py
│   ├── startup.py            # Model availability check
│   ├── setup.py              # Model download
│   ├── server.py             # FastAPI inference
│   └── text_encoder_server.py # Distributed text encoder service
├── settings.py               # Pydantic BaseSettings with auto-discovery
├── tensor_transport.py       # Safetensors serialization for distributed mode
├── remote_text_encoder.py    # Async httpx client for remote encoder
├── util.py                   # Model loading functions + availability checks
├── vae_loader.py             # VAE format detection and loading
└── ...

Dockerfile              # CUDA 12.9 + uv
docker-compose.yml      # Local development/testing
pyproject.toml          # Dependencies + PyTorch cu129 index
```

## Troubleshooting

### "HF_HOME directory not found"

Mount your HuggingFace cache to `/hf_cache`:
```bash
docker run -v /path/to/cache:/hf_cache ...
```

### "Models not loaded" (503 on /generate)

Check `/ready` endpoint. Models take 1-2 minutes to load. Increase `readinessProbe.initialDelaySeconds`.

### VAE Format Error

The BFL native `ae.safetensors` format is required. Diffusers VAE format is incompatible. Download from `black-forest-labs/FLUX.2-dev`.

### Out of Memory

- Klein 4B: ~12GB VRAM
- Klein 9B: ~20GB VRAM
- With moderation: +24GB for Mistral-24B

Consider using `--enable-moderation=false` or OpenRouter for prompt upsampling.
