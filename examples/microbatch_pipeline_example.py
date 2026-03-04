"""Example: using MicrobatchPipeline for concurrent multi-document processing.

This script demonstrates how to use the low-level ``MicrobatchPipeline`` class
to process *multiple* documents simultaneously.  All documents share a single
set of async pipeline stages (page loading → layout detection → OCR), which
keeps GPU utilisation high by batching pages from different documents together.

Use this approach when you need to OCR many documents at once and want to
maximise throughput.  For simpler single-document use-cases see
``examples/pipeline_example.py`` or the high-level ``glmocr.GlmOcr`` API.

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
    python examples/microbatch_pipeline_example.py [file1 file2 ...]

If no files are given the script falls back to ``examples/source/``.

Key differences from the standard Pipeline
-------------------------------------------
* ``process_batch()`` is an *async generator* — it must be consumed inside an
  ``async`` function and driven by ``asyncio.run()``.
* It accepts a *list* of request payloads (one per document) rather than a
  single payload.
* Results are yielded as ``(doc_index, PipelineResult)`` tuples and may arrive
  out of order (shorter documents finish first).
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from glmocr.config import load_config
from glmocr.pipeline import MicrobatchPipeline


def build_request(image_path: str) -> dict:
    """Build a single-document OpenAI-compatible request payload."""
    p = Path(image_path)
    if not p.is_absolute():
        p = p.resolve()
    url = f"file://{p}"
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            }
        ]
    }


async def process_documents(
    pipeline: MicrobatchPipeline,
    inputs: list[Path],
    output_dir: Path,
) -> None:
    """Process all documents through the microbatch pipeline."""
    # Build one request dict per document
    request_list = [build_request(str(p)) for p in inputs]

    print(f"Submitting {len(request_list)} document(s) to MicrobatchPipeline…")

    # process_batch() is an async generator.
    # Results arrive as (doc_index, PipelineResult) tuples, potentially
    # out of order — a document with fewer pages will finish first.
    async for doc_idx, result in pipeline.process_batch(
        request_list,
        save_layout_visualization=False,
    ):
        p = inputs[doc_idx]
        print(f"\n=== Completed doc [{doc_idx}]: {p.name} ===")

        preview = (result.markdown_result or "")[:200]
        print(f"  Markdown preview (first 200 chars):")
        print(f"  {preview!r}")

        # Save JSON + Markdown to output_dir/<filename>/
        dest = output_dir / p.stem
        result.save(output_dir=dest)
        print(f"  Saved to: {dest}")


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
    pdf_inputs = [p for p in inputs if p.suffix.lower() == ".pdf"]
    if not poppler_ok and pdf_inputs:
        print(
            "WARNING: Poppler not found (pdfinfo/pdftoppm/pdftocairo). "
            "PDF inputs will be skipped. On macOS: brew install poppler"
        )
        inputs = [p for p in inputs if p.suffix.lower() != ".pdf"]
        if not inputs:
            print("No non-PDF inputs remaining — nothing to do.")
            return 1

    print(f"Processing {len(inputs)} file(s) concurrently…")
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
    # The pipeline must be started before use and stopped afterwards.
    # Using it as a context manager handles this automatically.
    with MicrobatchPipeline(cfg.pipeline) as pipeline:
        # process_batch() is an async generator, so we need asyncio.run()
        # to drive the event loop.
        asyncio.run(process_documents(pipeline, inputs, output_dir))

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
