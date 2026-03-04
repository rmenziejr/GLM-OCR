"""Example: using the original Pipeline directly.

This script demonstrates how to use the low-level ``Pipeline`` class to parse
images and PDFs.  Use this approach when you need fine-grained control over the
pipeline configuration or want to integrate the pipeline into a larger system.

For a simpler, high-level API see ``glmocr.GlmOcr``.

Prerequisites
-------------
* A running GLM-OCR model server (vLLM / SGLang / Ollama) accessible at
  ``OCR_API_HOST:OCR_API_PORT`` (defaults: localhost:5002).
* (Optional) For layout-aware parsing install the local layout extra::

      pip install 'glmocr[local]'

Environment variables (or ``.env`` file)
-----------------------------------------
* ``GLMOCR_OCR_API_HOST`` – OCR API hostname (default: localhost)
* ``GLMOCR_OCR_API_PORT`` – OCR API port      (default: 5002)
* ``GLMOCR_ENABLE_LAYOUT`` – set to ``true`` to enable layout detection

Usage
-----
    python examples/pipeline_example.py [file1 file2 ...]

If no files are given the script falls back to ``examples/source/``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from glmocr.config import load_config
from glmocr.pipeline import Pipeline


def build_request(image_paths: list[str]) -> dict:
    """Build an OpenAI-compatible request payload from a list of file paths."""
    content = []
    for path in image_paths:
        p = Path(path)
        if not p.is_absolute():
            p = p.resolve()
        url = f"file://{p}"
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"messages": [{"role": "user", "content": content}]}


def main() -> int:
    here = Path(__file__).resolve().parent

    # Collect input files from CLI args or fall back to examples/source/
    if len(sys.argv) > 1:
        inputs = [Path(a).resolve() for a in sys.argv[1:]]
    else:
        source_dir = here / "source"
        if not source_dir.exists():
            print(f"No inputs supplied and source dir not found: {source_dir}")
            return 1
        inputs = sorted(
            [
                *source_dir.glob("*.png"),
                *source_dir.glob("*.jpg"),
                *source_dir.glob("*.jpeg"),
                *source_dir.glob("*.pdf"),
            ]
        )
        if not inputs:
            print(f"No input files found under: {source_dir}")
            return 1

    output_dir = here / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Warn if poppler is missing and PDFs were requested
    poppler_ok = any(
        shutil.which(cmd) is not None for cmd in ("pdfinfo", "pdftoppm", "pdftocairo")
    )
    if not poppler_ok and any(p.suffix.lower() == ".pdf" for p in inputs):
        print(
            "WARNING: Poppler not found (pdfinfo/pdftoppm/pdftocairo). "
            "PDF inputs will be skipped. On macOS: brew install poppler"
        )

    print(f"Processing {len(inputs)} file(s)…")
    print(f"Results will be written to: {output_dir}")

    # --- Configuration ---------------------------------------------------
    # load_config() reads config.yaml + GLMOCR_* env vars.  Keyword
    # arguments take the highest priority and override everything else.
    cfg = load_config(
        # Uncomment and adjust to point to your vLLM / SGLang server:
        # ocr_api_host="localhost",
        # ocr_api_port=5002,
        # enable_layout=True,
    )

    # --- Pipeline lifecycle ----------------------------------------------
    # The pipeline must be started before use (checks OCR API availability)
    # and stopped afterwards to release resources.  Using it as a context
    # manager handles this automatically.
    with Pipeline(cfg.pipeline) as pipeline:
        for p in inputs:
            if p.suffix.lower() == ".pdf" and not poppler_ok:
                print(f"  Skipping PDF (missing poppler): {p.name}")
                continue

            print(f"\n=== Parsing: {p.name} ===")

            # Build the OpenAI-compatible request payload.
            # Each image_url entry is one input unit; Pipeline.process()
            # yields one PipelineResult per input unit.
            request_data = build_request([str(p)])

            try:
                for result in pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                ):
                    print(f"  Markdown preview (first 200 chars):")
                    preview = (result.markdown_result or "")[:200]
                    print(f"  {preview!r}")

                    # Save JSON + Markdown to output_dir/<filename>/
                    result.save(output_dir=output_dir / p.stem)
                    print(f"  Saved to: {output_dir / p.stem}")

            except Exception as exc:
                print(f"  ERROR processing {p.name}: {exc}")
                continue

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
