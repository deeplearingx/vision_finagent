import asyncio
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


def _load_images(paths: List[str]) -> List[Image.Image]:
    """Open image files directly; convert PDF pages to PIL Images via pymupdf."""
    images = []
    for p in paths:
        ext = p.lower().rsplit(".", 1)[-1] if "." in p else ""
        try:
            if ext == "pdf":
                try:
                    doc = fitz.open(p)
                except Exception as exc:
                    raise IngestionError(
                        f"PDF 文件无法打开（可能已损坏或加密）：{p!r} — {exc}"
                    ) from exc
                if doc.page_count == 0:
                    doc.close()
                    raise IngestionError(f"PDF 文件不含任何页面：{p!r}")
                bad_pages = []
                for i in range(doc.page_count):
                    try:
                        page = doc.load_page(i)
                        pix = page.get_pixmap(dpi=150)
                        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
                    except Exception:
                        bad_pages.append(i + 1)
                if bad_pages and len(bad_pages) == doc.page_count:
                    doc.close()
                    raise IngestionError(f"PDF 所有页面均无法渲染：{p!r}")
                if bad_pages:
                    import structlog
                    structlog.get_logger().warning("pdf.bad_pages", path=p, bad_pages=bad_pages)
                doc.close()
            else:
                try:
                    img = Image.open(p)
                    img.verify()          # detect truncated / corrupt headers
                except Exception as exc:
                    raise IngestionError(
                        f"图片文件损坏或格式不支持（{ext}）：{p!r} — {exc}"
                    ) from exc
                # Re-open after verify() (verify() closes the file internally)
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception as exc:
                    raise IngestionError(
                        f"图片转换 RGB 失败（子格式异常，ext={ext}）：{p!r} — {exc}"
                    ) from exc
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"无法解析文件 {p!r}（ext={ext}）：{exc}") from exc
    return images


def _process_batch(
    report_id: str,
    images: List[Image.Image],
    page_offset: int,  # 0-based index of first image in this batch within the full document
) -> tuple[list, list[str]]:
    """Compute embeddings for one batch and write to Milvus immediately.

    Returns (inserted_patch_pks, inserted_page_ids) for rollback tracking.
    Frees GPU tensors after writing to keep peak VRAM proportional to batch size.
    """
    model, processor = retrieval_service._load_model()
    batch_tensor = processor.process_images(images).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch_tensor)  # (B, num_patches, 128)
    del batch_tensor

    client = get_client()
    name = get_collection_name()
    pages_name = get_pages_collection_name()
    inserted_patch_pks: list = []
    inserted_page_ids: list[str] = []

    for i, emb in enumerate(embeddings):
        page_num = page_offset + i + 1
        page_id = f"{report_id}_{page_num}"

        rows = [
            {"report_id": report_id, "page_num": page_num, "colpali_embeddings": vec.tolist()}
            for vec in emb.cpu().float()
        ]
        result = client.insert(collection_name=name, data=rows)
        inserted_patch_pks.extend(result["ids"])

        b64 = to_base64_bounded(images[i])
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
        images = await asyncio.to_thread(_load_images, image_paths)
        batch_size = settings.MAX_BATCH_SIZE
        for offset in range(0, len(images), batch_size):
            batch = images[offset: offset + batch_size]
            pks, pids = await asyncio.to_thread(_process_batch, report_id, batch, offset)
            inserted_patch_pks.extend(pks)
            inserted_page_ids.extend(pids)
    except Exception as exc:
        rb_exc_val = None
        # Rollback patches by exact auto-generated PKs
        if inserted_patch_pks:
            try:
                await asyncio.to_thread(
                    get_client().delete,
                    collection_name=get_collection_name(),
                    ids=inserted_patch_pks,
                )
            except Exception as rb_exc:
                rb_exc_val = rb_exc
        # Rollback pages by exact page_ids actually written (not estimated range)
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
