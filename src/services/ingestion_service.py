import asyncio
import torch
from typing import List
from PIL import Image
import fitz  # pymupdf
from colpali_engine.models.paligemma.colpali.modeling_colpali import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from ..config import settings
from ..core.milvus_client import get_client, get_collection_name, get_pages_collection_name, delete_report_data
from ..core.exception import IngestionError, RollbackError
from ..utils.image import to_base64_jpeg
from ..utils.lock import DistributedLock

_model: ColPali | None = None
_processor: ColPaliProcessor | None = None


def _load_model():
    global _model, _processor
    if _model is None:
        _model = ColPali.from_pretrained(
            settings.MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        _processor = ColPaliProcessor.from_pretrained(settings.MODEL_PATH)
    return _model, _processor


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
                for page in doc:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
                    except Exception as exc:
                        doc.close()
                        raise IngestionError(
                            f"PDF 第 {page.number + 1} 页渲染失败：{p!r} — {exc}"
                        ) from exc
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


def _compute_embeddings(images: List[Image.Image]):
    model, processor = _load_model()
    batch = processor.process_images(images).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch)  # (N, num_patches, 128)
    return embeddings


def _insert_pages(report_id: str, images: List[Image.Image], embeddings):
    # Write order: patch vectors FIRST, page metadata SECOND.
    #
    # Rationale for this ordering:
    #   - patch vectors are the primary search target; a query hit on a patch
    #     that has no corresponding page row produces a degraded (but not
    #     corrupt) result — the caller simply gets no image thumbnail.
    #   - The reverse (page written, patch missing) is worse: the page row
    #     implies the report is fully indexed, yet searches return nothing,
    #     which looks like a silent data loss to the user.
    #   - Milvus offers no cross-collection transactions, so a partial failure
    #     window is unavoidable.  We choose the window whose failure mode is
    #     more observable and less misleading.
    #
    # Rollback contract (enforced by ingest_report):
    #   inserted_patch_pks  → deleted by exact auto-generated PKs
    #   inserted_page_ids   → deleted by filter on page_id list
    #   Both lists are populated only AFTER the respective write succeeds,
    #   so a mid-loop crash leaves only already-tracked rows to roll back.
    client = get_client()
    name = get_collection_name()
    pages_name = get_pages_collection_name()
    inserted_patch_pks: list = []
    inserted_page_ids: list[str] = []
    for idx, emb in enumerate(embeddings):
        page_num = idx + 1
        page_id = f"{report_id}_{page_num}"

        # 1. Write patch vectors first (search-critical path).
        rows = [
            {
                "report_id": report_id,
                "page_num": page_num,
                "colpali_embeddings": vec.tolist(),
            }
            for vec in emb.cpu().float()
        ]
        result = client.insert(collection_name=name, data=rows)
        inserted_patch_pks.extend(result["ids"])  # track only after success

        # 2. Write page metadata second (display/thumbnail path).
        #    If this fails, the patch rows above are rolled back by ingest_report.
        b64 = to_base64_jpeg(images[idx])
        client.upsert(collection_name=pages_name, data=[{
            "page_id": page_id,
            "report_id": report_id,
            "page_num": page_num,
            "image_base64": b64,
        }])
        inserted_page_ids.append(page_id)  # track only after success

    return inserted_patch_pks, inserted_page_ids


async def ingest_report(report_id: str, image_paths: List[str]) -> None:
    lock = DistributedLock(f"ingest:{report_id}")
    if not await lock.acquire():
        raise IngestionError(f"Report {report_id} is already being ingested")

    inserted_patch_pks: list = []
    inserted_page_ids: list[str] = []
    try:
        # Clean up any stale data from a previous failed ingest for this report_id.
        # This prevents patch vector accumulation on retries.
        # Safe because: the idempotency token ensures no concurrent ingest for the
        # same report_id can reach this point simultaneously.
        await asyncio.to_thread(delete_report_data, report_id)
        images = await asyncio.to_thread(_load_images, image_paths)
        embeddings = await asyncio.to_thread(_compute_embeddings, images)
        inserted_patch_pks, inserted_page_ids = await asyncio.to_thread(
            _insert_pages, report_id, images, embeddings
        )
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
