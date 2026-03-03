"""Layout detector FastAPI service.

Exposes ``POST /layout`` backed by the PP-DocLayoutV3 model so that the main
``glmocr`` pipeline can delegate layout detection to this container instead of
running the heavy torch/transformers stack in-process.

This service implements the API contract expected by
``glmocr.layout.api_layout_detector.ApiLayoutDetector``:

Request::

    POST /layout
    Content-Type: application/json

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
"""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from glmocr.config import GlmOcrConfig
from glmocr.layout.layout_detector import PPDocLayoutDetector
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level detector instance (set during lifespan startup).
_detector: PPDocLayoutDetector | None = None


def _build_layout_config():
    """Return a :class:`~glmocr.config.LayoutConfig` ready for this service.

    Honours all ``GLMOCR_*`` environment variables plus the convenience
    variable ``LAYOUT_MODEL_DIR`` for overriding the model directory without
    touching the full config hierarchy.
    """
    cfg = GlmOcrConfig.from_env()
    layout_cfg = cfg.pipeline.layout

    model_dir_env = os.environ.get("LAYOUT_MODEL_DIR")
    if model_dir_env:
        layout_cfg.model_dir = model_dir_env

    return layout_cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup and release it on shutdown."""
    global _detector

    layout_cfg = _build_layout_config()
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

    results = _detector.process(pil_images)
    return LayoutResponse(results=results)
