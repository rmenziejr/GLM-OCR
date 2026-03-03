"""API-based layout detector.

Delegates layout detection to a remote HTTP service instead of running the
HuggingFace model in-process.  This keeps ``torch``/``transformers`` as
optional dependencies (``pip install 'glmocr[local]'``) and allows the layout
model to run in a separate Docker container, e.g. a FastAPI deployment.

Expected service contract
--------------------------
POST  <api_url>
Content-Type: application/json

Request body::

    {
        "images": ["<base64-encoded-PNG>", ...]
    }

Response body (HTTP 200)::

    {
        "results": [
            [
                {
                    "index": 0,
                    "label": "text",
                    "score": 0.91,
                    "bbox_2d": [x1, y1, x2, y2],
                    "polygon": [[px, py], ...],
                    "task_type": "text"
                },
                ...
            ],
            ...
        ]
    }

Where coordinates in ``bbox_2d`` and ``polygon`` are normalized to 0–1000.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional

import requests
from PIL import Image

from glmocr.layout.base import BaseLayoutDetector
from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.config import LayoutConfig

logger = get_logger(__name__)


class ApiLayoutDetector(BaseLayoutDetector):
    """Layout detector backed by a remote HTTP API.

    No local GPU or HuggingFace model is required.  The detector simply
    serialises the input images as base64-encoded PNGs, posts them to the
    configured endpoint, and returns the JSON results.

    Args:
        config: :class:`~glmocr.config.LayoutConfig` instance.
            ``config.api.api_url``   – endpoint URL.
            ``config.api.api_key``   – optional Bearer token.
            ``config.api.connect_timeout`` / ``request_timeout`` – timeouts.
    """

    def __init__(self, config: "LayoutConfig"):
        super().__init__(config)

        api_cfg = config.api
        self._api_url: str = api_cfg.api_url
        self._api_key: Optional[str] = api_cfg.api_key
        self._connect_timeout: int = api_cfg.connect_timeout
        self._request_timeout: int = api_cfg.request_timeout
        self._verify_ssl: bool = api_cfg.verify_ssl

        self._session: Optional[requests.Session] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create the HTTP session."""
        self._session = requests.Session()
        if self._api_key:
            self._session.headers["Authorization"] = f"Bearer {self._api_key}"
        self._session.headers["Content-Type"] = "application/json"
        logger.debug("ApiLayoutDetector started (endpoint: %s)", self._api_url)

    def stop(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None
        logger.debug("ApiLayoutDetector stopped.")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Return a base64-encoded PNG string for *image*."""
        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        visualization_output_dir: Optional[str] = None,
        global_start_idx: int = 0,
    ) -> List[List[Dict]]:
        """Send *images* to the layout API and return per-image detections.

        Args:
            images: Input PIL images.
            save_visualization: Ignored (visualization is handled server-side).
            visualization_output_dir: Ignored.
            global_start_idx: Ignored.

        Returns:
            ``List[List[Dict]]`` – same format as
            :class:`~glmocr.layout.layout_detector.PPDocLayoutDetector`.
        """
        if self._session is None:
            raise RuntimeError("ApiLayoutDetector not started. Call start() first.")

        payload = {"images": [self._encode_image(img) for img in images]}

        try:
            resp = self._session.post(
                self._api_url,
                json=payload,
                timeout=(self._connect_timeout, self._request_timeout),
                verify=self._verify_ssl,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Layout API request to '{self._api_url}' failed: {exc}"
            ) from exc

        data = resp.json()
        if "results" not in data:
            raise ValueError(
                f"Layout API response missing 'results' key. Got: {list(data.keys())}"
            )

        results: List[List[Dict]] = data["results"]
        if len(results) != len(images):
            raise ValueError(
                f"Layout API returned {len(results)} result(s) for {len(images)} image(s). "
                "The service must return exactly one result list per input image."
            )
        return results
