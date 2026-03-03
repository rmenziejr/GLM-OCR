# ---------------------------------------------------------------------------
# GLM-OCR Layout Detector — Docker image
#
# Build from the repository root:
#
#   docker build -f apps/layout-detector/Dockerfile -t glmocr-layout-detector .
#
# Run (CPU):
#   docker run -p 8010:8010 glmocr-layout-detector
#
# Run (GPU, device 0):
#   docker run --gpus '"device=0"' -p 8010:8010 \
#       -e GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES=0 \
#       glmocr-layout-detector
#
# Mount a pre-downloaded model to avoid re-downloading on every start:
#   docker run -p 8010:8010 \
#       -v /path/to/PP-DocLayoutV3_safetensors:/model \
#       -e LAYOUT_MODEL_DIR=/model \
#       glmocr-layout-detector
# ---------------------------------------------------------------------------

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libraries required by torch (libgomp1) and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install glmocr with local layout extras (torch + transformers + opencv)
# ---------------------------------------------------------------------------
COPY glmocr ./glmocr
COPY pyproject.toml setup.py README.md ./
RUN pip install -e ".[local]"

# ---------------------------------------------------------------------------
# Install FastAPI runtime
# ---------------------------------------------------------------------------
RUN pip install "fastapi>=0.104.0" "uvicorn[standard]>=0.24.0"

# ---------------------------------------------------------------------------
# Copy the service application
# ---------------------------------------------------------------------------
COPY apps/layout-detector/app ./app

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# Model directory: can be a local path or a HuggingFace hub model id.
# Override with -e LAYOUT_MODEL_DIR=/model (and -v /host/model:/model).
ENV LAYOUT_MODEL_DIR=PaddlePaddle/PP-DocLayoutV3_safetensors

EXPOSE 8010

# Allow extra time on first start for HuggingFace model download / GPU init.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8010/health')" \
        || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
