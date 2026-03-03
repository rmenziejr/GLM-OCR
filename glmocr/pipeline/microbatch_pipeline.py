"""GLM-OCR Microbatch Pipeline

Multi-document pipeline with decoupled stages and shared microbatch queues.

Unlike the standard Pipeline (which processes one document request at a time),
MicrobatchPipeline accepts a *list* of request payloads and routes pages from
ALL documents through three shared, decoupled stage queues:

    Stage 1 – Page loading (one loader thread per document):
        Each document's pages are loaded concurrently.  Every page is tagged
        with its (doc_index, page_index) and placed on a shared page_queue.

    Stage 2 – Layout detection (single thread, microbatch-aware):
        Consumes pages from all documents.  Accumulates up to
        ``layout_detector.batch_size`` pages before running one GPU forward
        pass.  Because pages from *different* documents share the same batch,
        GPU utilisation stays high even when individual documents have few
        pages.  Detected regions are pushed onto a shared region_queue.

    Stage 3 – OCR recognition (thread pool):
        Consumes layout regions concurrently with up to ``max_workers``
        parallel API calls.  Results are assembled per-document.

Results are yielded per-document as soon as ALL of that document's regions
have been recognised, so short documents do not wait for long ones.

Example::

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

    for doc_idx, result in pipeline.process_batch(request_list):
        result.save(output_dir=f"./output/{doc_idx}")

    pipeline.stop()
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple

from glmocr.parser_result import PipelineResult
from glmocr.pipeline.pipeline import Pipeline
from glmocr.utils.image_utils import crop_image_region
from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.config import PipelineConfig
    from glmocr.layout.base import BaseLayoutDetector
    from glmocr.postprocess import ResultFormatter

logger = get_logger(__name__)

# Sentinel placed on queues to signal end-of-stream for a single producer.
_DONE = object()


@dataclass
class _MicrobatchState:
    """Shared mutable state wired through all three pipeline stages."""

    # ---- queues --------------------------------------------------------
    # Each item: ("page", doc_idx, page_idx, global_idx, pil_image)
    #         or ("done_doc", doc_idx)          -- all pages of doc loaded
    #         or ("done_all",)                  -- every loader finished
    #         or ("error", doc_idx, exc)
    page_queue: queue.Queue

    # Each item: ("region", doc_idx, global_page_idx, cropped_img, region_dict, task_type)
    #         or ("done",)
    #         or ("error", exc)
    region_queue: queue.Queue

    # ---- per-document tracking -----------------------------------------
    num_docs: int

    # Total pages loaded per doc (set when loader for that doc finishes).
    pages_per_doc: List[Optional[int]]  # index = doc_idx
    pages_per_doc_lock: threading.Lock

    # layout_results[doc_idx][page_idx] = list of region dicts
    layout_results: List[Dict[int, List]]
    layout_lock: threading.Lock

    # Number of layout-detected regions per doc (known once all pages are laid out).
    regions_per_doc: List[Optional[int]]
    regions_per_doc_lock: threading.Lock

    # Recognition results: doc_idx → list of (global_page_idx, region_dict)
    recognition_results: List[List[Tuple[int, Dict]]]
    recognition_lock: threading.Lock

    # Ready queue – doc indices whose all regions have been recognised.
    ready_queue: queue.Queue

    # Track which docs have already been put on ready_queue.
    docs_notified: set
    docs_notified_lock: threading.Lock

    # Exceptions from background threads.
    exceptions: List[Tuple[str, Exception]]
    exception_lock: threading.Lock

    # images_dict: global_page_idx → PIL image (kept for layout visualisation)
    images_dict: Dict[int, Any]
    images_dict_lock: threading.Lock

    # doc_idx → list of global_page_indices (in order)
    doc_global_pages: List[List[int]]


def _make_state(num_docs: int, page_qsize: int, region_qsize: int) -> _MicrobatchState:
    return _MicrobatchState(
        page_queue=queue.Queue(maxsize=page_qsize),
        region_queue=queue.Queue(maxsize=region_qsize),
        num_docs=num_docs,
        pages_per_doc=[None] * num_docs,
        pages_per_doc_lock=threading.Lock(),
        layout_results=[{} for _ in range(num_docs)],
        layout_lock=threading.Lock(),
        regions_per_doc=[None] * num_docs,
        regions_per_doc_lock=threading.Lock(),
        recognition_results=[[] for _ in range(num_docs)],
        recognition_lock=threading.Lock(),
        ready_queue=queue.Queue(),
        docs_notified=set(),
        docs_notified_lock=threading.Lock(),
        exceptions=[],
        exception_lock=threading.Lock(),
        images_dict={},
        images_dict_lock=threading.Lock(),
        doc_global_pages=[[] for _ in range(num_docs)],
    )


class MicrobatchPipeline(Pipeline):
    """Multi-document microbatch pipeline with decoupled stages.

    Extends :class:`~glmocr.pipeline.Pipeline` with a
    :meth:`process_batch` method that accepts a *list* of request payloads
    and routes pages from ALL documents through shared stage queues.

    The layout-detection stage accumulates pages from different documents into
    a single GPU microbatch (up to ``layout_detector.batch_size`` pages), so
    throughput scales with the total number of pages rather than the per-
    document page count.

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

    def process_batch(
        self,
        request_data_list: List[Dict[str, Any]],
        save_layout_visualization: bool = False,
        layout_vis_output_dir: Optional[str] = None,
    ) -> Generator[Tuple[int, PipelineResult], None, None]:
        """Process a microbatch of documents through shared pipeline stages.

        Yields ``(doc_index, PipelineResult)`` tuples as each document
        completes.  Documents may complete out of order — shorter documents
        (fewer pages / regions) will typically finish first.

        When :attr:`enable_layout` is ``False`` each document is delegated to
        :meth:`~glmocr.pipeline.Pipeline.process` serially and the tuple
        ``(doc_index, result)`` is yielded in submission order.

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
            # Fallback: delegate each doc to the standard single-doc pipeline.
            for doc_idx, req in enumerate(request_data_list):
                for result in self.process(
                    req,
                    save_layout_visualization=save_layout_visualization,
                    layout_vis_output_dir=layout_vis_output_dir,
                ):
                    yield doc_idx, result
            return

        num_docs = len(request_data_list)
        state = _make_state(
            num_docs=num_docs,
            page_qsize=self._page_qsize,
            region_qsize=self._region_qsize,
        )

        # Extract image URLs per doc.
        image_urls_per_doc: List[List[str]] = [
            self._extract_image_urls(req) for req in request_data_list
        ]
        original_inputs: List[List[str]] = [
            [(u[7:] if u.startswith("file://") else u) for u in urls]
            for urls in image_urls_per_doc
        ]

        # Global page index counter (shared across all loader threads).
        global_counter = [0]
        global_counter_lock = threading.Lock()

        # ----------------------------------------------------------------
        # Stage 1: Page loader threads (one per document)
        # ----------------------------------------------------------------

        loader_done_count = [0]
        loader_done_lock = threading.Lock()

        def _loader_for_doc(doc_idx: int) -> None:
            urls = image_urls_per_doc[doc_idx]
            page_count = 0
            try:
                for page, _unit_idx in self.page_loader.iter_pages_with_unit_indices(
                    urls
                ):
                    with global_counter_lock:
                        g_idx = global_counter[0]
                        global_counter[0] += 1
                    with state.images_dict_lock:
                        state.images_dict[g_idx] = page
                    state.doc_global_pages[doc_idx].append(g_idx)
                    state.page_queue.put(("page", doc_idx, page_count, g_idx, page))
                    page_count += 1
            except Exception as exc:
                logger.exception(
                    "MicrobatchPipeline: loader error for doc %d: %s", doc_idx, exc
                )
                with state.exception_lock:
                    state.exceptions.append((f"LoaderDoc{doc_idx}", exc))
                state.page_queue.put(("error", doc_idx, exc))

            with state.pages_per_doc_lock:
                state.pages_per_doc[doc_idx] = page_count
            state.page_queue.put(("done_doc", doc_idx))

            with loader_done_lock:
                loader_done_count[0] += 1
                if loader_done_count[0] == num_docs:
                    state.page_queue.put(("done_all",))

        loader_threads = [
            threading.Thread(
                target=_loader_for_doc, args=(i,), daemon=True, name=f"mb-loader-{i}"
            )
            for i in range(num_docs)
        ]

        # ----------------------------------------------------------------
        # Stage 2: Layout detection thread (microbatch across all docs)
        # ----------------------------------------------------------------

        def _layout_thread() -> None:
            batch_imgs: List[Any] = []
            batch_meta: List[Tuple[int, int, int]] = []  # (doc_idx, page_idx, g_idx)
            all_loaders_done = False
            global_start_idx = 0
            docs_pages_received: Dict[int, int] = {i: 0 for i in range(num_docs)}

            def _flush_batch() -> None:
                nonlocal global_start_idx
                if not batch_imgs:
                    return
                try:
                    layout_results = self.layout_detector.process(
                        batch_imgs,
                        save_visualization=save_layout_visualization
                        and layout_vis_output_dir is not None,
                        visualization_output_dir=layout_vis_output_dir,
                        global_start_idx=global_start_idx,
                    )
                except Exception as exc:
                    logger.exception(
                        "MicrobatchPipeline: layout detection error: %s", exc
                    )
                    layout_results = [[] for _ in batch_imgs]
                    with state.exception_lock:
                        state.exceptions.append(("LayoutThread", exc))

                global_start_idx += len(batch_imgs)

                for (doc_idx, page_idx, g_idx), image, regions in zip(
                    batch_meta, batch_imgs, layout_results
                ):
                    with state.layout_lock:
                        state.layout_results[doc_idx][page_idx] = regions
                    for region in regions:
                        cropped = crop_image_region(
                            image, region["bbox_2d"], region["polygon"]
                        )
                        state.region_queue.put(
                            (
                                "region",
                                doc_idx,
                                g_idx,
                                cropped,
                                region,
                                region["task_type"],
                            )
                        )

                batch_imgs.clear()
                batch_meta.clear()

            def _maybe_mark_doc_layout_done(doc_idx: int) -> None:
                """Once all pages for a doc have been laid out, record total regions."""
                with state.pages_per_doc_lock:
                    expected = state.pages_per_doc[doc_idx]
                if expected is None:
                    return
                if docs_pages_received[doc_idx] < expected:
                    return
                with state.layout_lock:
                    total_regions = sum(
                        len(v)
                        for v in state.layout_results[doc_idx].values()
                    )
                with state.regions_per_doc_lock:
                    if state.regions_per_doc[doc_idx] is None:
                        state.regions_per_doc[doc_idx] = total_regions
                _maybe_notify_ready(doc_idx)

            try:
                while True:
                    try:
                        item = state.page_queue.get(timeout=0.5)
                    except queue.Empty:
                        if all_loaders_done and batch_imgs:
                            _flush_batch()
                            for d in range(num_docs):
                                _maybe_mark_doc_layout_done(d)
                        continue

                    if item[0] == "page":
                        _, doc_idx, page_idx, g_idx, page = item
                        batch_imgs.append(page)
                        batch_meta.append((doc_idx, page_idx, g_idx))
                        docs_pages_received[doc_idx] = (
                            docs_pages_received[doc_idx] + 1
                        )
                        if len(batch_imgs) >= self.layout_detector.batch_size:
                            docs_in_batch = set(m[0] for m in batch_meta)
                            _flush_batch()
                            for d in docs_in_batch:
                                _maybe_mark_doc_layout_done(d)

                    elif item[0] == "done_doc":
                        _, doc_idx = item
                        # Flush immediately to unblock recognition for this doc.
                        if batch_imgs:
                            _flush_batch()
                        _maybe_mark_doc_layout_done(doc_idx)

                    elif item[0] == "done_all":
                        all_loaders_done = True
                        if batch_imgs:
                            _flush_batch()
                        for d in range(num_docs):
                            _maybe_mark_doc_layout_done(d)
                        break

                    elif item[0] == "error":
                        # Loader failed for a doc; mark it as having 0 regions.
                        _, doc_idx, _exc = item
                        with state.regions_per_doc_lock:
                            if state.regions_per_doc[doc_idx] is None:
                                state.regions_per_doc[doc_idx] = 0
                        _maybe_notify_ready(doc_idx)

            except Exception as exc:
                logger.exception("MicrobatchPipeline: layout thread error: %s", exc)
                with state.exception_lock:
                    state.exceptions.append(("LayoutThread", exc))
            finally:
                state.region_queue.put(("done",))

        # ----------------------------------------------------------------
        # Helpers shared between layout and recognition threads
        # ----------------------------------------------------------------

        def _maybe_notify_ready(doc_idx: int) -> None:
            """Put doc_idx on ready_queue when all its regions are recognised."""
            with state.regions_per_doc_lock:
                expected = state.regions_per_doc[doc_idx]
            if expected is None:
                return
            with state.recognition_lock:
                done = len(state.recognition_results[doc_idx])
            if done >= expected:
                with state.docs_notified_lock:
                    if doc_idx not in state.docs_notified:
                        state.docs_notified.add(doc_idx)
                        state.ready_queue.put(doc_idx)

        # ----------------------------------------------------------------
        # Stage 3: OCR recognition thread (thread pool)
        # ----------------------------------------------------------------

        def _recognition_thread() -> None:
            executor = ThreadPoolExecutor(
                max_workers=min(self.max_workers, 128),
                thread_name_prefix="mb-ocr",
            )
            futures: Dict[Any, Tuple[int, int, Dict, str]] = {}
            processing_complete = False
            pending_skip: List[Tuple[int, int, Dict]] = []

            def _harvest_done_futures() -> None:
                for fut in list(futures.keys()):
                    if not fut.done():
                        continue
                    doc_idx, g_idx, region, _task = futures.pop(fut)
                    try:
                        resp, status = fut.result()
                        if status == 200:
                            region["content"] = (
                                resp["choices"][0]["message"]["content"].strip()
                            )
                        else:
                            region["content"] = ""
                    except Exception as exc:
                        logger.warning(
                            "MicrobatchPipeline: recognition failed (doc=%d g_page=%d): %s",
                            doc_idx,
                            g_idx,
                            exc,
                        )
                        region["content"] = ""
                    with state.recognition_lock:
                        state.recognition_results[doc_idx].append((g_idx, region))
                    _maybe_notify_ready(doc_idx)

            try:
                while True:
                    _harvest_done_futures()

                    try:
                        item = state.region_queue.get(timeout=0.02)
                    except queue.Empty:
                        if processing_complete and not futures:
                            # Flush skipped regions.
                            for doc_idx, g_idx, region in pending_skip:
                                region["content"] = None
                                with state.recognition_lock:
                                    state.recognition_results[doc_idx].append(
                                        (g_idx, region)
                                    )
                                _maybe_notify_ready(doc_idx)
                            pending_skip.clear()
                            break
                        if futures and not processing_complete:
                            try:
                                next(as_completed(list(futures.keys()), timeout=0.05))
                            except (StopIteration, TimeoutError):
                                pass
                        continue

                    if item[0] == "region":
                        _, doc_idx, g_idx, cropped, region, task_type = item
                        if task_type == "skip":
                            pending_skip.append((doc_idx, g_idx, region))
                        else:
                            req = self.page_loader.build_request_from_image(
                                cropped, task_type
                            )
                            fut = executor.submit(self.ocr_client.process, req)
                            futures[fut] = (doc_idx, g_idx, region, task_type)

                    elif item[0] == "done":
                        processing_complete = True

                    elif item[0] == "error":
                        break

                # Drain any remaining futures.
                if futures:
                    for fut in as_completed(list(futures.keys())):
                        doc_idx, g_idx, region, _task = futures.pop(fut)
                        try:
                            resp, status = fut.result()
                            if status == 200:
                                region["content"] = (
                                    resp["choices"][0]["message"]["content"].strip()
                                )
                            else:
                                region["content"] = ""
                        except Exception as exc:
                            logger.warning(
                                "MicrobatchPipeline: recognition failed (doc=%d): %s",
                                doc_idx,
                                exc,
                            )
                            region["content"] = ""
                        with state.recognition_lock:
                            state.recognition_results[doc_idx].append(
                                (g_idx, region)
                            )
                        _maybe_notify_ready(doc_idx)

            except Exception as exc:
                logger.exception(
                    "MicrobatchPipeline: recognition thread error: %s", exc
                )
                with state.exception_lock:
                    state.exceptions.append(("RecognitionThread", exc))
            finally:
                executor.shutdown(wait=False)
                # Ensure every doc gets notified even on error.
                for d in range(num_docs):
                    with state.docs_notified_lock:
                        if d not in state.docs_notified:
                            state.docs_notified.add(d)
                            state.ready_queue.put(d)

        # ----------------------------------------------------------------
        # Start threads
        # ----------------------------------------------------------------

        for t in loader_threads:
            t.start()

        t_layout = threading.Thread(
            target=_layout_thread, daemon=True, name="mb-layout"
        )
        t_recog = threading.Thread(
            target=_recognition_thread, daemon=True, name="mb-recognition"
        )
        t_layout.start()
        t_recog.start()

        # ----------------------------------------------------------------
        # Main thread: yield results as each doc becomes ready
        # ----------------------------------------------------------------

        emitted: set = set()
        while len(emitted) < num_docs:
            try:
                doc_idx = state.ready_queue.get(timeout=1.0)
            except queue.Empty:
                # Check whether all threads are done (deadlock guard).
                if (
                    not t_layout.is_alive()
                    and not t_recog.is_alive()
                    and all(not t.is_alive() for t in loader_threads)
                ):
                    # Force remaining docs ready.
                    for d in range(num_docs):
                        if d not in emitted:
                            with state.docs_notified_lock:
                                if d not in state.docs_notified:
                                    state.docs_notified.add(d)
                            state.ready_queue.put(d)
                continue

            if doc_idx in emitted:
                continue

            # Verify all regions are truly available (guard against spurious wakeup).
            with state.regions_per_doc_lock:
                expected = state.regions_per_doc[doc_idx]
            if expected is not None:
                with state.recognition_lock:
                    done = len(state.recognition_results[doc_idx])
                if done < expected:
                    # Re-enqueue and wait for more results.
                    with state.docs_notified_lock:
                        state.docs_notified.discard(doc_idx)
                    state.ready_queue.put(doc_idx)
                    continue

            # Assemble per-doc result.
            g_pages = state.doc_global_pages[doc_idx]
            g_page_set = set(g_pages)
            with state.recognition_lock:
                doc_rec = [
                    (g, r)
                    for g, r in state.recognition_results[doc_idx]
                    if g in g_page_set
                ]

            # Group regions by global page index, in page order.
            g_to_page_pos = {g: pos for pos, g in enumerate(g_pages)}
            grouped: List[List[Dict]] = [[] for _ in g_pages]
            for g_idx, region in doc_rec:
                pos = g_to_page_pos.get(g_idx)
                if pos is not None:
                    grouped[pos].append(region)

            json_result, md_result = self.result_formatter.process(grouped)
            yield doc_idx, PipelineResult(
                json_result=json_result,
                markdown_result=md_result,
                original_images=original_inputs[doc_idx],
                layout_vis_dir=layout_vis_output_dir,
                layout_image_indices=list(g_pages),
            )
            emitted.add(doc_idx)

        # Wait for all threads to finish, then surface any errors.
        for t in loader_threads:
            t.join()
        t_layout.join()
        t_recog.join()

        with state.exception_lock:
            if state.exceptions:
                raise RuntimeError(
                    "; ".join(f"{name}: {exc}" for name, exc in state.exceptions)
                )
