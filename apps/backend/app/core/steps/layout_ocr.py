"""
版面分析和OCR处理步骤
"""

import json
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
from app.core.ocr_client import LayoutAndOCRClient
import httpx
import base64
import os
from app.core.flows.base import ProcessingContext
from app.utils.config import settings
from app.utils.image_processer import crop_image_by_bbox_to_path, vlm_bbox_convert
from app.utils.logger import logger


class LayoutOcrStepInput:
    image_files_path: List[str]  # 图片文件路径列表
    page_count: Optional[int]  # 页数
    images_dir: Optional[str]  # 图片目录

    def __init__(
        self,
        image_files_path: List[str],
        page_count: Optional[int] = None,
        images_dir: Optional[str] = None,
    ) -> None:
        self.image_files_path = image_files_path
        self.page_count = page_count
        self.images_dir = images_dir


async def layout_and_ocr(
    context: ProcessingContext,
    input: LayoutOcrStepInput,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    执行版面分析和OCR处理

    Args:
        context: 处理上下文
        pdf_result: PDF转图片的结果，包含:
            - output_files: list[str] - 图片文件路径列表
            - page_count: int - 页数
            - images_dir: str - 图片目录
        progress_callback: 进度回调函数

    Returns:
        Dict[str, Any]: OCR结果
    """
    task_id = context.task_id
    ocr_config = context.ocr_config

    # 获取图片文件列表
    image_files = input.image_files_path
    images_dir = input.images_dir
    page_count = input.page_count
    page_size = context.metadata.get("page_size")
    logger.info(f"[{task_id}] Starting layout and OCR processing")
    logger.info(f"[{task_id}] Processing {page_count} pages from {images_dir}")

    try:
        # 调用实际的版面分析和OCR服务
        # 这里需要根据实际的OCR API来实现

        # 示例：假设我们有一个OCR客户端
        result = await _call_ocr_service(
            page_size=page_size,
            image_files=image_files,
            images_dir=images_dir,
            page_count=page_count,
            config=ocr_config,
            output_dir=context.get_output_dir(),
            progress_callback=progress_callback,
        )

        logger.info(f"[{task_id}] Layout and OCR processing completed")

        return result

    except Exception as e:
        logger.error(f"[{task_id}] Layout and OCR processing failed: {e}")
        raise


async def _call_ocr_service(
    image_files: list[str],
    images_dir: str,
    page_count: int,
    config: Dict[str, Any],
    output_dir: str,
    page_size: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    microbatch_size: int = 4,
) -> Dict[str, Any]:
    """
    调用OCR服务（支持微批次处理）

    使用微批次（microbatch）模式将多张页面图片合并为一次请求，
    减少与OCR服务的往返次数，提升整体吞吐量。

    Args:
        image_files: 图片文件路径列表
        images_dir: 图片目录
        page_count: 页数
        config: OCR配置
        output_dir: 输出目录
        page_size: 页面尺寸信息（含 width / height）
        progress_callback: 进度回调
        microbatch_size: 每次发送给OCR服务的图片数量（默认4）
    """

    if progress_callback:
        await progress_callback(0.0, f"Initializing OCR service for {page_count} pages")
    custom_url = config.get("custom_url", None)
    cli = LayoutAndOCRClient()
    pages_result = []
    page_width = page_size.get("width")
    page_height = page_size.get("height")
    block_idx = 1
    ref_image_paths = []

    # --- microbatch loop: send up to microbatch_size pages per request ---
    processed = 0
    for batch_start in range(0, len(image_files), microbatch_size):
        batch_files = image_files[batch_start : batch_start + microbatch_size]

        # Send the whole microbatch in a single HTTP round-trip.
        batch_results: List[List[Dict[str, Any]]] = await cli.process_images(
            batch_files, custom_url=custom_url
        )

        for j, (image_file, result) in enumerate(zip(batch_files, batch_results)):
            page_num = batch_start + j + 1
            processed += 1

            if progress_callback:
                progress = (processed / page_count) * 100
                await progress_callback(
                    progress, f"Processing page {page_num}/{page_count}"
                )

            page_blocks = []
            for block in result:
                block_label = block.get("label", "text")
                block_bbox = block.get("bbox_2d", [0, 0, 0, 0])
                block_content = block.get("content", None)
                block_index = block_idx
                normalized_box = vlm_bbox_convert(block_bbox, page_width, page_height)

                # 如果 label 为 image，则裁剪图片并添加到 image_path 字段
                image_path_field = None
                if block_label == "image":
                    try:
                        split_filename = f"split_{page_num}_{block_idx:04d}.png"
                        split_path = os.path.join(output_dir, split_filename)
                        crop_image_by_bbox_to_path(image_file, normalized_box, split_path)
                        image_path_field = split_path
                        ref_image_paths.append(image_path_field)
                        logger.info(f"裁剪图片块 {block_idx}: {split_filename}")
                    except Exception as e:
                        logger.warning(f"裁剪图片块 {block_idx} 失败: {str(e)}")

                block_info = {
                    "layout_type": block_label,
                    "layout_box": normalized_box,
                    "content": block_content,
                    "index": block_index,
                    "image_path": image_path_field,
                    "page_index": page_num,
                }
                page_blocks.append(block_info)
                block_idx += 1

            pages_result.append(
                {
                    "page_index": page_num,
                    "image_file": image_file,
                    "layout": {"blocks": page_blocks},
                }
            )

    if progress_callback:
        await progress_callback(100.0, "OCR processing completed")

    # 保存OCR结果到JSON
    ocr_result_file = Path(output_dir) / "ocr_result.json"
    ocr_result_data = {
        "success": True,
        "pages": pages_result,
        "total_pages": page_count,
        "images_dir": images_dir,
        "ocr_result_file": f"{ocr_result_file}",
        "ref_image_paths": ref_image_paths,
    }

    try:
        with open(ocr_result_file, "w", encoding="utf-8") as f:
            json.dump(ocr_result_data, f, ensure_ascii=False, indent=2)
        logger.info(f"OCR results saved to: {ocr_result_file}")
    except Exception as e:
        logger.error(f"Failed to save OCR results: {e}")

    return ocr_result_data

