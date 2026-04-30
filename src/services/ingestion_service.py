import asyncio
import structlog
import torch
from typing import List
from PIL import Image
import fitz  # pymupdf
from ..core.milvus_client import get_client, get_collection_name, get_pages_collection_name, delete_report_data
from ..core.exception import IngestionError, RollbackError
from ..utils.image import to_base64_bounded
from ..utils.lock import DistributedLock
from . import retrieval_service
from ..config import settings

log = structlog.get_logger()


def _repair_pdf_with_pymupdf(path: str) -> str:
    """尝试用 PyMuPDF 重新保存 PDF，重建 xref/page tree。失败时回退到原路径。"""
    repaired = path + ".repaired.pdf"
    doc = None
    try:
        doc = fitz.open(path)
        doc.save(repaired, garbage=4, deflate=True, clean=True)
        return repaired
    except Exception as exc:
        log.warning("pdf.repair_failed", path=path, error=str(exc))
        return path
    finally:
        if doc is not None:
            doc.close()


def _iter_pdf_pages(path: str, dpi: int = 150):
    """按页渲染 PDF，保留原始 page_num（1-based）。坏页跳过并记录日志。"""
    doc = None
    try:
        doc = fitz.open(path)
        if doc.page_count == 0:
            raise IngestionError(f"PDF 文件不含任何页面：{path!r}")
        ok_pages = 0
        bad_pages = []
        for i in range(doc.page_count):
            page_num = i + 1
            try:
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ok_pages += 1
                yield page_num, img
            except Exception as exc:
                log.warning("pdf.bad_page", path=path, page_num=page_num, error=str(exc))
                bad_pages.append(page_num)
        if ok_pages == 0:
            raise IngestionError(f"PDF 所有页面均无法渲染：{path!r}")
        if bad_pages:
            log.warning("pdf.partial_ingest", path=path, ok_pages=ok_pages, bad_pages=bad_pages)
    finally:
        if doc is not None:
            doc.close()


def _load_images(paths: List[str]) -> List[Image.Image]:
    """仅用于非 PDF 图片文件。"""
    images = []
    for p in paths:
        ext = p.lower().rsplit(".", 1)[-1] if "." in p else ""
        try:
            img = Image.open(p)
            img.verify()
        except Exception as exc:
            raise IngestionError(f"图片文件损坏或格式不支持（{ext}）：{p!r} — {exc}") from exc
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as exc:
            raise IngestionError(f"图片转换 RGB 失败：{p!r} — {exc}") from exc
    return images


def _process_batch(
    report_id: str,
    page_items: List[tuple],
) -> tuple[list, list[str]]:
    """编码一批 (page_num, Image) 并写入 Milvus。"""
    images = [img for _, img in page_items]
    model, processor = retrieval_service._load_model()
    batch_tensor = processor.process_images(images).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch_tensor)
    del batch_tensor

    client = get_client()
    name = get_collection_name()
    pages_name = get_pages_collection_name()
    inserted_patch_pks: list = []
    inserted_page_ids: list[str] = []

    for i, emb in enumerate(embeddings):
        page_num, image = page_items[i]
        page_id = f"{report_id}_{page_num}"

        rows = [
            {"report_id": report_id, "page_num": page_num, "colpali_embeddings": vec.tolist()}
            for vec in emb.cpu().float()
        ]
        result = client.insert(collection_name=name, data=rows)
        inserted_patch_pks.extend(result["ids"])

        b64 = to_base64_bounded(image)
        client.upsert(collection_name=pages_name, data=[{
            "page_id": page_id,
            "report_id": report_id,
            "page_num": page_num,
            "image_base64": b64,
            "_vec": [0.0, 0.0],
        }])
        inserted_page_ids.append(page_id)

    del embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return inserted_patch_pks, inserted_page_ids


async def ingest_report(report_id: str, image_paths: List[str]) -> None:
    lock = DistributedLock(f"ingest:{report_id}")
    if not await lock.acquire():
        raise IngestionError(f"Report {report_id} is already being ingested")

    inserted_patch_pks: list = []
    inserted_page_ids: list[str] = []
    try:
        await asyncio.to_thread(delete_report_data, report_id)

        for path in image_paths:
            ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
            if ext == "pdf":
                def _ingest_single_pdf(p=path):
                    repaired = _repair_pdf_with_pymupdf(p)
                    batch: list[tuple] = []
                    all_pks: list = []
                    all_pids: list[str] = []
                    for page_num, image in _iter_pdf_pages(repaired, dpi=150):
                        batch.append((page_num, image))
                        if len(batch) >= settings.MAX_BATCH_SIZE:
                            pks, pids = _process_batch(report_id, batch)
                            all_pks.extend(pks)
                            all_pids.extend(pids)
                            batch.clear()
                    if batch:
                        pks, pids = _process_batch(report_id, batch)
                        all_pks.extend(pks)
                        all_pids.extend(pids)
                    return all_pks, all_pids

                pks, pids = await asyncio.to_thread(_ingest_single_pdf)
                inserted_patch_pks.extend(pks)
                inserted_page_ids.extend(pids)
            else:
                images = await asyncio.to_thread(_load_images, [path])
                page_items = [(1, images[0])]
                pks, pids = await asyncio.to_thread(_process_batch, report_id, page_items)
                inserted_patch_pks.extend(pks)
                inserted_page_ids.extend(pids)

    except Exception as exc:
        rb_exc_val = None
        if inserted_patch_pks:
            try:
                await asyncio.to_thread(
                    get_client().delete,
                    collection_name=get_collection_name(),
                    ids=inserted_patch_pks,
                )
            except Exception as rb_exc:
                rb_exc_val = rb_exc
        if inserted_page_ids:
            try:
                import json as _json
                await asyncio.to_thread(
                    get_client().delete,
                    collection_name=get_pages_collection_name(),
                    filter=f"page_id in {_json.dumps(inserted_page_ids)}",
                )
            except Exception as rb_exc:
                rb_exc_val = rb_exc
        if rb_exc_val is not None:
            raise RollbackError(str(rb_exc_val)) from exc
        raise IngestionError(str(exc)) from exc
    finally:
        await lock.release()
