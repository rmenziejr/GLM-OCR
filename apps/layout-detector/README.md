# GLM-OCR Layout Detector Service

A lightweight FastAPI service that wraps the **PP-DocLayoutV3** model and
exposes it as a scalable HTTP API.  It implements the contract expected by
[`glmocr.layout.api_layout_detector.ApiLayoutDetector`](../../glmocr/layout/api_layout_detector.py),
enabling the main `glmocr` pipeline to delegate layout detection to this
container without requiring a local GPU or HuggingFace installation.

## API

### `POST /layout`

Detect layout regions in one or more images.

**Request body** (`application/json`):

```json
{
  "images": ["<base64-encoded-PNG>", "..."]
}
```

**Response body** (`application/json`):

```json
{
  "results": [
    [
      {
        "index": 0,
        "label": "text",
        "score": 0.91,
        "bbox_2d": [x1, y1, x2, y2],
        "polygon": [[px, py], "..."],
        "task_type": "text"
      }
    ]
  ]
}
```

Coordinates in `bbox_2d` and `polygon` are normalised to 0–1000.

### `GET /health`

Returns `{"status": "healthy"}` once the model is loaded.

---

## Running with Docker

### Build

Build from the **repository root**:

```bash
docker build -f apps/layout-detector/Dockerfile -t glmocr-layout-detector .
```

### Run (CPU)

```bash
docker run -p 8010:8010 glmocr-layout-detector
```

### Run (GPU — device 0)

```bash
docker run --gpus '"device=0"' -p 8010:8010 \
    -e GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES=0 \
    glmocr-layout-detector
```

### Mount a pre-downloaded model

The container downloads the model from the HuggingFace Hub on first start.
To avoid re-downloading, mount a local copy:

```bash
docker run -p 8010:8010 \
    -v /path/to/PP-DocLayoutV3_safetensors:/model \
    -e LAYOUT_MODEL_DIR=/model \
    glmocr-layout-detector
```

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `LAYOUT_MODEL_DIR` | `PaddlePaddle/PP-DocLayoutV3_safetensors` | Model directory (local path or HuggingFace hub id) |
| `GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES` | `"0"` | GPU device index; CPU is used automatically when CUDA is unavailable |
| `GLMOCR_LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LAYOUT_CORS_ORIGINS` | `*` | Comma-separated list of allowed CORS origins; use specific domains in production (e.g. `https://app.example.com`) |

All `GLMOCR_*` variables from the main SDK are also honoured — see
[`glmocr/config.py`](../../glmocr/config.py) for the full list.

---

## Connecting the glmocr pipeline to this service

Set the following env vars (or update `config.yaml`) in the `glmocr` client:

```bash
GLMOCR_LAYOUT_BACKEND=api
GLMOCR_LAYOUT_API_URL=http://<container-host>:8010/layout
# optional bearer token:
# GLMOCR_LAYOUT_API_KEY=<secret>
```

Or in Python:

```python
from glmocr.config import load_config

cfg = load_config(
    layout_backend="api",
    layout_api_url="http://localhost:8010/layout",
)
```

---

## Scaling

To serve multiple concurrent requests, run several replicas behind a load
balancer (e.g. nginx, Traefik, or a Kubernetes Service):

```bash
docker run -d -p 8011:8010 --name layout-1 glmocr-layout-detector
docker run -d -p 8012:8010 --name layout-2 glmocr-layout-detector
```

> **Note:** each replica loads the full model into VRAM/RAM.  Horizontal
> scaling therefore requires proportionally more GPU memory.
