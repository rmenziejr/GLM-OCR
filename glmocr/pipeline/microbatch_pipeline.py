"""GLM-OCR Microbatch Pipeline

Async multi-document pipeline with decoupled stages and shared microbatch queues.

Unlike the standard Pipeline (which processes one document request at a time),
MicrobatchPipeline accepts a *list* of request payloads and routes pages from
ALL documents through three shared, decoupled stage queues:

    Stage 1 – Page loading (one coroutine per document):
        Each document's pages are loaded concurrently via the default thread-pool
        executor so that blocking I/O does not stall the event loop.  Every page is
        tagged with its ``(doc_index, page_index)`` and placed on a shared async
        ``page_queue``.

    Stage 2 – Layout detection (single-thread executor, microbatch-aware):
        Consumes pages from all documents.  Accumulates up to
        ``layout_detector.batch_size`` pages before running one GPU forward pass,
        which is offloaded to a dedicated single-thread ``ThreadPoolExecutor`` via
        ``loop.run_in_executor`` — keeping GPU access serialised without blocking
        the event loop.  Because pages from *different* documents share the same
        batch, GPU utilisation stays high even when individual documents have few
        pages.  Detected regions are pushed onto a shared async ``region_queue``.

    Stage 3 – OCR recognition (httpx.AsyncClient + asyncio.Semaphore):
        Consumes layout regions and fires one ``httpx`` async HTTP call per region.
        Concurrency is bounded by ``max_workers`` via a semaphore.  All calls for a
        batch run concurrently within the event loop — no thread pool required.

All three stages run as asyncio tasks.  Results are yielded per-document as soon
as ALL of that document's regions have been recognised, so short documents do not
wait for long ones.

Example::

    import asyncio
    from glmocr.config import load_config
    from glmocr.pipeline import MicrobatchPipeline

    cfg = load_config()
    pipeline = MicrobatchPipeline(cfg.pipeline)
    pipeline.start()

    request_list = [
        {"messages": [{"role": "user", "content": [{"type": "image_url",
            "image_url": {"url": "file:///path/a.pdf"}}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url",
            "image_url": {"url": "file:///path/b.png"}}]}]},
    ]

    async def main():
        async for doc_idx, result in pipeline.process_batch(request_list):
            result.save(output_dir=f"./output/{doc_idx}")

    asyncio.run(main())
    pipeline.stop()
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
)

import httpx

from glmocr.parser_result import PipelineResult
from glmocr.pipeline.pipeline import Pipeline
from glmocr.utils.image_utils import crop_image_region
from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.config import PipelineConfig
    from glmocr.layout.base import BaseLayoutDetector
    from glmocr.postprocess import ResultFormatter

logger = get_logger(__name__)


@dataclass
class _AsyncMicrobatchState:
    """Shared mutable state wired through all three async pipeline stages.

    No threading locks are needed because all mutations occur in asyncio
    coroutines that run on the single-threaded event loop.  The layout
    executor offloads the *blocking* GPU call to a thread, but state is
    only mutated **after** ``await run_in_executor`` returns, which means
    we are back on the event loop before touching any shared data.
    """

    num_docs: int

    # Set to ``page_count`` once a doc's loader coroutine finishes.
    pages_per_doc: List[Optional[int]]

    # layout_results[doc_idx][page_idx] = list of region dicts
    layout_results: List[Dict[int, List]]

    # Set to total region count once all pages for a doc have been laid out.
    regions_per_doc: List[Optional[int]]

    # recognition_results[doc_idx] = [(global_page_idx, region_dict), ...]
    recognition_results: List[List[Tuple[int, Dict]]]

    # Tracks which docs have already been put on the ready_queue.
    docs_notified: set

    # global_page_idx → PIL Image (kept for layout visualisation)
    images_dict: Dict[int, Any]

    # doc_idx → ordered list of global page indices
    doc_global_pages: List[List[int]]

    # Exceptions raised in background tasks.
    exceptions: List[Tuple[str, Exception]]


def _make_state(num_docs: int) -> _AsyncMicrobatchState:
    return _AsyncMicrobatchState(
        num_docs=num_docs,
        pages_per_doc=[None] * num_docs,
        layout_results=[{} for _ in range(num_docs)],
        regions_per_doc=[None] * num_docs,
        recognition_results=[[] for _ in range(num_docs)],
        docs_notified=set(),
        images_dict={},
        doc_global_pages=[[] for _ in range(num_docs)],
        exceptions=[],
    )


class MicrobatchPipeline(Pipeline):
    """Multi-document microbatch pipeline with fully async stages.

    Extends :class:`~glmocr.pipeline.Pipeline` with an async
    :meth:`process_batch` method that accepts a *list* of request payloads
    and routes pages from ALL documents through shared async stage queues.

    The three stages run as concurrent asyncio tasks:

    * **Page loading** — one coroutine per document; blocking I/O offloaded
      to the default thread-pool executor via ``loop.run_in_executor``.
    * **Layout detection** — a single coroutine that accumulates pages into
      microbatches and offloads each GPU forward pass to a dedicated
      single-thread ``ThreadPoolExecutor``, keeping GPU access serialised
      without blocking the event loop.
    * **OCR recognition** — each region fires an async ``httpx`` HTTP call;
      concurrency bounded by ``max_workers`` via ``asyncio.Semaphore``.

    Args:
        config: PipelineConfig instance.
        layout_detector: Custom layout detector (optional).
        result_formatter: Custom result formatter (optional).
        page_qsize: Maximum pages held in the inter-stage page queue.
        region_qsize: Maximum regions held in the inter-stage region queue.
    """

    def __init__(
        self,
        config: "PipelineConfig",
        layout_detector: Optional["BaseLayoutDetector"] = None,
        result_formatter: Optional["ResultFormatter"] = None,
        page_qsize: int = 200,
        region_qsize: int = 1600,
    ):
        super().__init__(
            config=config,
            layout_detector=layout_detector,
            result_formatter=result_formatter,
        )
        self._page_qsize = page_qsize
        self._region_qsize = region_qsize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_batch(
        self,
        request_data_list: List[Dict[str, Any]],
        save_layout_visualization: bool = False,
        layout_vis_output_dir: Optional[str] = None,
    ) -> AsyncGenerator[Tuple[int, PipelineResult], None]:
        """Process a microbatch of documents through shared async pipeline stages.

        Yields ``(doc_index, PipelineResult)`` tuples as each document
        completes.  Documents may complete out of order — shorter documents
        (fewer pages / regions) will typically finish first.

        When :attr:`enable_layout` is ``False`` each document is delegated to
        the synchronous :meth:`~glmocr.pipeline.Pipeline.process` via the
        default thread-pool executor, and the tuple ``(doc_index, result)``
        is yielded in submission order.

        Args:
            request_data_list: List of request payloads, one per document.
            save_layout_visualization: Whether to save layout visualisation.
            layout_vis_output_dir: Output directory for layout visualisations.

        Yields:
            ``(doc_index, PipelineResult)`` for every input document.
        """
        if not request_data_list:
            return

        if not self.enable_layout:
            loop = asyncio.get_event_loop()
            for doc_idx, req in enumerate(request_data_list):
                results = await loop.run_in_executor(
                    None,
                    lambda r=req: list(
                        self.process(
                            r,
                            save_layout_visualization=save_layout_visualization,
                            layout_vis_output_dir=layout_vis_output_dir,
                        )
                    ),
                )
                for result in results:
                    yield doc_idx, result
            return

        num_docs = len(request_data_list)
        state = _make_state(num_docs)

        page_queue: asyncio.Queue = asyncio.Queue(maxsize=self._page_qsize)
        region_queue: asyncio.Queue = asyncio.Queue(maxsize=self._region_qsize)
        ready_queue: asyncio.Queue = asyncio.Queue()

        image_urls_per_doc: List[List[str]] = [
            self._extract_image_urls(req) for req in request_data_list
        ]
        original_inputs: List[List[str]] = [
            [(u[7:] if u.startswith("file://") else u) for u in urls]
            for urls in image_urls_per_doc
        ]

        # Monotonically increasing global page index — only mutated inside
        # loader coroutines, which yield at `await` points between docs but
        # never concurrently within a single doc's page loop.
        global_counter = [0]
        loop = asyncio.get_event_loop()

        # ----------------------------------------------------------------
        # Helper: mark doc ready when all its regions are recognised
        # ----------------------------------------------------------------

        def _maybe_notify_ready(doc_idx: int) -> None:
            expected = state.regions_per_doc[doc_idx]
            if expected is None:
                return
            done = len(state.recognition_results[doc_idx])
            if done >= expected and doc_idx not in state.docs_notified:
                state.docs_notified.add(doc_idx)
                ready_queue.put_nowait(doc_idx)

        # ----------------------------------------------------------------
        # Stage 1: page loading (one coroutine per doc)
        # ----------------------------------------------------------------

        async def _load_doc(doc_idx: int) -> None:
            urls = image_urls_per_doc[doc_idx]
            page_count = 0
            try:
                pages = await loop.run_in_executor(
                    None,
                    lambda: list(
                        self.page_loader.iter_pages_with_unit_indices(urls)
                    ),
                )
                for page, _unit_idx in pages:
                    g_idx = global_counter[0]
                    global_counter[0] += 1
                    state.images_dict[g_idx] = page
                    state.doc_global_pages[doc_idx].append(g_idx)
                    await page_queue.put(("page", doc_idx, page_count, g_idx, page))
                    page_count += 1
            except Exception as exc:
                logger.exception(
                    "MicrobatchPipeline: loader error for doc %d: %s", doc_idx, exc
                )
                state.exceptions.append((f"LoaderDoc{doc_idx}", exc))
                await page_queue.put(("error", doc_idx, exc))

            state.pages_per_doc[doc_idx] = page_count
            await page_queue.put(("done_doc", doc_idx))

        async def _all_loaders() -> None:
            await asyncio.gather(*[_load_doc(i) for i in range(num_docs)])
            await page_queue.put(("done_all",))

        # ----------------------------------------------------------------
        # Stage 2: layout detection
        #   – blocking GPU call offloaded to a dedicated single-thread
        #     executor so the event loop stays responsive, and so that
        #     concurrent GPU access is prevented.
        # ----------------------------------------------------------------

        layout_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mb-layout"
        )

        async def _run_layout() -> None:
            batch_imgs: List[Any] = []
            batch_meta: List[Tuple[int, int, int]] = []  # (doc_idx, page_idx, g_idx)
            global_start_idx = 0
            docs_pages_received: Dict[int, int] = {i: 0 for i in range(num_docs)}

            async def _flush_batch() -> None:
                nonlocal global_start_idx
                if not batch_imgs:
                    return
                imgs = list(batch_imgs)
                meta = list(batch_meta)
                batch_imgs.clear()
                batch_meta.clear()

                try:
                    layout_results = await loop.run_in_executor(
                        layout_executor,
                        functools.partial(
                            self.layout_detector.process,
                            imgs,
                            save_visualization=(
                                save_layout_visualization
                                and layout_vis_output_dir is not None
                            ),
                            visualization_output_dir=layout_vis_output_dir,
                            global_start_idx=global_start_idx,
                        ),
                    )
                except Exception as exc:
                    logger.exception(
                        "MicrobatchPipeline: layout detection error: %s", exc
                    )
                    layout_results = [[] for _ in imgs]
                    state.exceptions.append(("LayoutStage", exc))

                global_start_idx += len(imgs)

                for (doc_idx, page_idx, g_idx), image, regions in zip(
                    meta, imgs, layout_results
                ):
                    state.layout_results[doc_idx][page_idx] = regions
                    for region in regions:
                        cropped = crop_image_region(
                            image, region["bbox_2d"], region["polygon"]
                        )
                        await region_queue.put(
                            (
                                "region",
                                doc_idx,
                                g_idx,
                                cropped,
                                region,
                                region["task_type"],
                            )
                        )

            def _maybe_mark_layout_done(doc_idx: int) -> None:
                expected = state.pages_per_doc[doc_idx]
                if expected is None:
                    return
                if docs_pages_received[doc_idx] < expected:
                    return
                total_regions = sum(
                    len(v) for v in state.layout_results[doc_idx].values()
                )
                if state.regions_per_doc[doc_idx] is None:
                    state.regions_per_doc[doc_idx] = total_regions
                    _maybe_notify_ready(doc_idx)

            try:
                while True:
                    item = await page_queue.get()

                    if item[0] == "page":
                        _, doc_idx, page_idx, g_idx, page = item
                        batch_imgs.append(page)
                        batch_meta.append((doc_idx, page_idx, g_idx))
                        docs_pages_received[doc_idx] += 1
                        if len(batch_imgs) >= self.layout_detector.batch_size:
                            docs_in_batch = {m[0] for m in batch_meta}
                            await _flush_batch()
                            for d in docs_in_batch:
                                _maybe_mark_layout_done(d)

                    elif item[0] == "done_doc":
                        _, doc_idx = item
                        if batch_imgs:
                            await _flush_batch()
                        _maybe_mark_layout_done(doc_idx)

                    elif item[0] == "done_all":
                        if batch_imgs:
                            await _flush_batch()
                        for d in range(num_docs):
                            _maybe_mark_layout_done(d)
                        break

                    elif item[0] == "error":
                        _, doc_idx, _exc = item
                        if state.regions_per_doc[doc_idx] is None:
                            state.regions_per_doc[doc_idx] = 0
                        _maybe_notify_ready(doc_idx)

            except Exception as exc:
                logger.exception(
                    "MicrobatchPipeline: layout stage error: %s", exc
                )
                state.exceptions.append(("LayoutStage", exc))
            finally:
                layout_executor.shutdown(wait=False)
                await region_queue.put(("done",))

        # ----------------------------------------------------------------
        # Stage 3: OCR recognition
        #   – each region fires a single httpx async HTTP request;
        #     concurrency is bounded by max_workers via asyncio.Semaphore.
        # ----------------------------------------------------------------

        async def _run_ocr() -> None:
            semaphore = asyncio.Semaphore(min(self.max_workers, 128))
            ocr_tasks: List[asyncio.Task] = []

            async def _process_region(
                doc_idx: int,
                g_idx: int,
                cropped: Any,
                region: Dict,
                task_type: str,
                client: httpx.AsyncClient,
            ) -> None:
                async with semaphore:
                    req = self.page_loader.build_request_from_image(
                        cropped, task_type
                    )
                    try:
                        resp, status = await self.ocr_client.process_async(
                            req, client
                        )
                        if status == 200:
                            region["content"] = (
                                resp["choices"][0]["message"]["content"].strip()
                            )
                        else:
                            region["content"] = ""
                    except Exception as exc:
                        logger.warning(
                            "MicrobatchPipeline: recognition failed "
                            "(doc=%d g_page=%d): %s",
                            doc_idx,
                            g_idx,
                            exc,
                        )
                        region["content"] = ""
                state.recognition_results[doc_idx].append((g_idx, region))
                _maybe_notify_ready(doc_idx)

            try:
                async with httpx.AsyncClient(
                    verify=self.ocr_client.verify_ssl,
                    timeout=self.ocr_client.request_timeout,
                ) as client:
                    while True:
                        item = await region_queue.get()

                        if item[0] == "region":
                            _, doc_idx, g_idx, cropped, region, task_type = item
                            if task_type == "skip":
                                region["content"] = None
                                state.recognition_results[doc_idx].append(
                                    (g_idx, region)
                                )
                                _maybe_notify_ready(doc_idx)
                            else:
                                task = asyncio.create_task(
                                    _process_region(
                                        doc_idx,
                                        g_idx,
                                        cropped,
                                        region,
                                        task_type,
                                        client,
                                    )
                                )
                                ocr_tasks.append(task)

                        elif item[0] in ("done", "error"):
                            break

                    if ocr_tasks:
                        task_results = await asyncio.gather(
                            *ocr_tasks, return_exceptions=True
                        )
                        for exc in task_results:
                            if isinstance(exc, Exception):
                                logger.exception(
                                    "MicrobatchPipeline: unexpected OCR task error: %s",
                                    exc,
                                )
                                state.exceptions.append(("OCRStage", exc))

            except Exception as exc:
                logger.exception(
                    "MicrobatchPipeline: OCR stage error: %s", exc
                )
                state.exceptions.append(("OCRStage", exc))
            finally:
                # Ensure every doc gets notified even on error.
                for d in range(num_docs):
                    if d not in state.docs_notified:
                        state.docs_notified.add(d)
                        ready_queue.put_nowait(d)

        # ----------------------------------------------------------------
        # Run all stages concurrently
        # ----------------------------------------------------------------

        pipeline_done = asyncio.Event()

        async def _run_pipeline() -> None:
            await asyncio.gather(
                _all_loaders(),
                _run_layout(),
                _run_ocr(),
                return_exceptions=True,
            )
            pipeline_done.set()

        pipeline_task = asyncio.create_task(_run_pipeline())

        # ----------------------------------------------------------------
        # Main: yield results as each doc becomes ready
        # ----------------------------------------------------------------

        def _assemble_result(doc_idx: int) -> PipelineResult:
            g_pages = state.doc_global_pages[doc_idx]
            g_page_set = set(g_pages)
            doc_rec = [
                (g, r)
                for g, r in state.recognition_results[doc_idx]
                if g in g_page_set
            ]
            g_to_pos = {g: pos for pos, g in enumerate(g_pages)}
            grouped: List[List[Dict]] = [[] for _ in g_pages]
            for g_idx, region in doc_rec:
                pos = g_to_pos.get(g_idx)
                if pos is not None:
                    grouped[pos].append(region)
            json_result, md_result = self.result_formatter.process(grouped)
            return PipelineResult(
                json_result=json_result,
                markdown_result=md_result,
                original_images=original_inputs[doc_idx],
                layout_vis_dir=layout_vis_output_dir,
                layout_image_indices=list(g_pages),
            )

        emitted: set = set()
        while len(emitted) < num_docs:
            try:
                doc_idx = await asyncio.wait_for(ready_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if pipeline_done.is_set():
                    # Force any remaining docs ready to avoid a deadlock.
                    for d in range(num_docs):
                        if d not in emitted:
                            state.docs_notified.add(d)
                            ready_queue.put_nowait(d)
                continue

            if doc_idx in emitted:
                continue

            # Guard: verify all regions are truly available before yielding.
            expected = state.regions_per_doc[doc_idx]
            if expected is not None:
                done = len(state.recognition_results[doc_idx])
                if done < expected:
                    state.docs_notified.discard(doc_idx)
                    ready_queue.put_nowait(doc_idx)
                    continue

            emitted.add(doc_idx)
            yield doc_idx, _assemble_result(doc_idx)

        await pipeline_task

        if state.exceptions:
            raise RuntimeError(
                "; ".join(f"{name}: {exc}" for name, exc in state.exceptions)
            )
