"""Layout detector FastAPI service.

Exposes ``POST /layout`` backed by the PP-DocLayoutV3 model so that the main
``glmocr`` pipeline can delegate layout detection to this container instead of
running the heavy torch/transformers stack in-process.

This service implements the API contract expected by
``glmocr.layout.api_layout_detector.ApiLayoutDetector``:

Request::

    POST /layout
    Content-Type: application/json
    Authorization: Bearer <token>   # required only when LAYOUT_API_KEY is set

    {"images": ["<base64-encoded-PNG>", ...]}

Response::

    {"results": [[{"index": 0, "label": "text", "score": 0.91,
                   "bbox_2d": [x1, y1, x2, y2],
                   "polygon": [[px, py], ...],
                   "task_type": "text"}, ...], ...]}

Coordinates in ``bbox_2d`` and ``polygon`` are normalised to 0–1000.

Environment variables
---------------------
LAYOUT_MODEL_DIR
    Path to the PP-DocLayoutV3 model (local directory or HuggingFace hub id).
    Defaults to ``PaddlePaddle/PP-DocLayoutV3_safetensors``.
GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES
    GPU device index to use (e.g. ``"0"``).  Falls back to CPU automatically.
GLMOCR_LOG_LEVEL
    Logging verbosity: ``DEBUG``, ``INFO`` (default), ``WARNING``, ``ERROR``.
LAYOUT_CORS_ORIGINS
    Comma-separated list of allowed CORS origins.  Defaults to ``*`` (any
    origin), which is safe for container-to-container communication but should
    be tightened when the service is exposed to the public internet.
LAYOUT_API_KEY
    Optional shared secret.  When set, every request to ``POST /layout`` must
    supply ``Authorization: Bearer <token>`` with a matching value.  Leave
    unset to disable authentication (suitable for private networks).
"""

from __future__ import annotations

import base64
import os
import threading
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from PIL import Image

from glmocr.config import GlmOcrConfig
from glmocr.layout.layout_detector import PPDocLayoutDetector
from glmocr.utils.logging import ensure_logging_configured, get_logger

logger = get_logger(__name__)

# Module-level detector instance (set during lifespan startup).
_detector: PPDocLayoutDetector | None = None

# Serialises concurrent calls to _detector.process() so the single model
# instance is never invoked from multiple threads simultaneously.
_detect_lock = threading.Lock()

# Optional API key read once at startup.
_api_key: Optional[str] = os.environ.get("LAYOUT_API_KEY") or None

_http_bearer = HTTPBearer(auto_error=False)


def _verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_http_bearer),
) -> None:
    """Raise HTTP 401 when bearer-token authentication is enabled and fails."""
    if _api_key is None:
        return  # auth disabled
    token = credentials.credentials if credentials else None
    if token != _api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


def _build_full_config() -> GlmOcrConfig:
    """Return a fully-loaded :class:`~glmocr.config.GlmOcrConfig`.

    Honours all ``GLMOCR_*`` environment variables plus the convenience
    variable ``LAYOUT_MODEL_DIR`` for overriding the model directory without
    touching the full config hierarchy.
    """
    cfg = GlmOcrConfig.from_env()

    model_dir_env = os.environ.get("LAYOUT_MODEL_DIR")
    if model_dir_env:
        cfg.pipeline.layout.model_dir = model_dir_env

    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup and release it on shutdown."""
    global _detector

    cfg = _build_full_config()

    # Apply logging settings from config/env before anything else logs.
    ensure_logging_configured(
        level=cfg.logging.level,
        format_string=cfg.logging.format,
    )

    layout_cfg = cfg.pipeline.layout
    logger.info("Loading PP-DocLayoutV3 from '%s' …", layout_cfg.model_dir)
    _detector = PPDocLayoutDetector(layout_cfg)
    _detector.start()
    logger.info("Layout detector ready.")

    yield

    logger.info("Shutting down layout detector …")
    _detector.stop()
    _detector = None


app = FastAPI(
    title="GLM-OCR Layout Detector",
    description=(
        "PP-DocLayoutV3 document layout detection service.\n\n"
        "Implements the HTTP API contract expected by "
        "`glmocr.layout.api_layout_detector.ApiLayoutDetector`."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the set of trusted origins to be tightened in production via the
# LAYOUT_CORS_ORIGINS env var (comma-separated list, e.g.
# "https://my-app.example.com,https://other.example.com").
# Defaults to "*" which is acceptable for an internal container-to-container
# service but should be restricted when exposed to the public internet.
_cors_origins_env = os.environ.get("LAYOUT_CORS_ORIGINS", "*")
_cors_origins = (
    ["*"]
    if _cors_origins_env.strip() == "*"
    else [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class LayoutRequest(BaseModel):
    """Request body for the layout detection endpoint."""

    images: List[str]
    """Base64-encoded PNG strings, one per page/image."""


class LayoutResponse(BaseModel):
    """Response body for the layout detection endpoint."""

    results: List[List[Dict[str, Any]]]
    """Per-image list of detected regions (same format as PPDocLayoutDetector)."""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
def health():
    """Return service health status."""
    return {"status": "healthy" if _detector is not None else "starting"}


@app.post(
    "/layout",
    response_model=LayoutResponse,
    summary="Detect layout regions in one or more images",
    dependencies=[Security(_verify_token)],
)
def detect_layout(request: LayoutRequest):
    """Decode base64 images and run PP-DocLayoutV3 layout detection.

    Returns one result list per input image, each containing detected
    regions with normalised coordinates (0–1000).
    """
    if _detector is None:
        raise HTTPException(status_code=503, detail="Layout detector not ready.")

    pil_images: List[Image.Image] = []
    for idx, b64_str in enumerate(request.images):
        try:
            img_bytes = base64.b64decode(b64_str)
            pil_images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode image at index {idx}: {exc}",
            ) from exc

    with _detect_lock:
        results = _detector.process(pil_images)
    return LayoutResponse(results=results)
