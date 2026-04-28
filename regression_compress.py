"""Regression: compare VLM latency/quality with image compression enabled."""
import asyncio
import time
import sys
import os
import structlog

# Stub colpali before src imports
import types
for _n in [
    "colpali_engine",
    "colpali_engine.models",
    "colpali_engine.models.paligemma",
    "colpali_engine.models.paligemma.colpali",
    "colpali_engine.models.paligemma.colpali.modeling_colpali",
    "colpali_engine.models.paligemma.colpali.processing_colpali",
]:
    if _n not in sys.modules:
        sys.modules[_n] = types.ModuleType(_n)

sys.path.insert(0, os.path.dirname(__file__))

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()


async def run_vlm_only(query: str, pages, timeout: float = 90.0):
    """Run VLM on pre-fetched pages, return (answer, degrade, elapsed)."""
    from src.services.vlm_service import generate_answer
    t0 = time.time()
    answer, degrade = await generate_answer(query, pages, timeout=timeout)
    return answer, degrade, time.time() - t0


def fetch_pages_from_milvus(report_id: str, top_k: int = 5):
    """Directly fetch pages from Milvus pages collection (no model needed)."""
    import json
    from src.config import settings
    from src.core.milvus_client import get_client, get_pages_collection_name
    from src.models.schema import PageResult

    client = get_client()
    pages_col = get_pages_collection_name()

    rows = client.query(
        collection_name=pages_col,
        filter=f'report_id == {json.dumps(report_id)}',
        output_fields=["report_id", "page_num", "image_base64"],
        limit=top_k,
    )
    return [
        PageResult(
            report_id=r["report_id"],
            page_num=r["page_num"],
            image_base64=r.get("image_base64", ""),
            maxsim_score=1.0,
        )
        for r in rows
    ]


def sample_report_id() -> str | None:
    """Get one report_id from Milvus pages collection."""
    from src.config import settings
    from src.core.milvus_client import get_client, get_pages_collection_name

    client = get_client()
    rows = client.query(
        collection_name=get_pages_collection_name(),
        filter="",
        output_fields=["report_id"],
        limit=1,
    )
    return rows[0]["report_id"] if rows else None


async def main():
    from src.config import settings

    report_id = sample_report_id()
    if not report_id:
        log.error("No data found in Milvus pages collection")
        return

    log.info("regression.report", report_id=report_id)
    pages = fetch_pages_from_milvus(report_id, top_k=3)
    log.info("regression.pages_fetched", count=len(pages),
             has_images=sum(1 for p in pages if p.image_base64))

    if not any(p.image_base64 for p in pages):
        log.error("No image_base64 in fetched pages, cannot test compression")
        return

    queries = [
        "公司主要业务是什么？",
        "2023年营业收入是多少？",
    ]

    print(f"\n=== Compression config: max_side={settings.VLM_IMG_MAX_SIDE}, "
          f"quality={settings.VLM_IMG_JPEG_QUALITY}, max_bytes={settings.VLM_IMG_MAX_BYTES} ===\n")
    print(f"{'Query':<30} {'VLM(s)':<10} {'Degrade':<18} {'Ans Len'}")
    print("-" * 70)

    for q in queries:
        answer, degrade, elapsed = await run_vlm_only(q, pages)
        ans_len = len(answer) if answer else 0
        print(f"{q:<30} {elapsed:<10.2f} {degrade.value:<18} {ans_len}")
        log.info("query.result", query=q, vlm_time=elapsed,
                 degrade=degrade.value, answer_len=ans_len,
                 answer_preview=(answer or "")[:100])
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
