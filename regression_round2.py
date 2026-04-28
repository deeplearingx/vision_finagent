"""Round-2 regression: real financial table pages, single-image + Top-5 multi-image.

Observability per query:
  - image_compress.summary  → total_orig_bytes / total_comp_bytes / ratio
  - vlm.start / vlm.done / vlm.timeout / vlm.empty_response
  - degrade_reason, elapsed, answer_len, answer_preview
"""
import asyncio
import time
import sys
import os
import types
import structlog

# ── stub colpali so src imports work without GPU model ──────────────────────
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

# ── financial table keywords for page filtering ─────────────────────────────
_TABLE_KEYWORDS = [
    "营业收入", "净利润", "资产负债", "现金流", "利润表",
    "revenue", "profit", "balance sheet", "cash flow", "income",
    "总资产", "负债", "权益", "每股", "毛利",
]


def _is_table_page(page) -> bool:
    """Heuristic: page_num >= 5 avoids cover/TOC; no keyword filter needed
    since we cannot decode stored images here — rely on page_num offset."""
    return page.page_num >= 5


def fetch_pages(report_id: str, top_k: int = 5):
    import json
    from src.core.milvus_client import get_client, get_pages_collection_name
    from src.models.schema import PageResult

    client = get_client()
    rows = client.query(
        collection_name=get_pages_collection_name(),
        filter=f'report_id == {json.dumps(report_id)}',
        output_fields=["report_id", "page_num", "image_base64"],
        limit=50,  # fetch more, then filter
    )
    pages = [
        PageResult(
            report_id=r["report_id"],
            page_num=r["page_num"],
            image_base64=r.get("image_base64", ""),
            maxsim_score=1.0,
        )
        for r in rows if r.get("image_base64")
    ]
    # prefer pages that look like table/body pages (page_num >= 5)
    table_pages = [p for p in pages if _is_table_page(p)]
    pool = table_pages if table_pages else pages
    pool.sort(key=lambda p: p.page_num)
    return pool[:top_k]


def sample_report_ids(n: int = 3) -> list[str]:
    from src.core.milvus_client import get_client, get_pages_collection_name

    client = get_client()
    rows = client.query(
        collection_name=get_pages_collection_name(),
        filter="",
        output_fields=["report_id"],
        limit=100,
    )
    seen, ids = set(), []
    for r in rows:
        rid = r["report_id"]
        if rid not in seen:
            seen.add(rid)
            ids.append(rid)
        if len(ids) >= n:
            break
    return ids


async def run_vlm(query: str, pages, timeout: float = 90.0):
    from src.services.vlm_service import generate_answer
    t0 = time.time()
    answer, degrade = await generate_answer(query, pages, timeout=timeout)
    return answer, degrade, time.time() - t0


def _bytes_info(pages) -> str:
    """Estimate raw base64 payload size before compression."""
    total = sum(len(p.image_base64) * 3 // 4 for p in pages if p.image_base64)
    return f"{total // 1024} KB"


async def run_round(label: str, query: str, pages, timeout: float = 90.0):
    raw_kb = _bytes_info(pages)
    log.info("round.start", label=label, query=query,
             pages=len(pages), raw_payload_kb=raw_kb)
    answer, degrade, elapsed = await run_vlm(query, pages, timeout=timeout)
    ans_len = len(answer) if answer else 0
    timed_out = degrade.value == "vlm_timeout"
    empty = (answer is None or ans_len < 5)
    log.info("round.done", label=label, elapsed_s=round(elapsed, 2),
             degrade=degrade.value, ans_len=ans_len,
             timed_out=timed_out, empty_response=empty,
             answer_preview=(answer or "")[:120])
    return {
        "label": label, "query": query, "pages": len(pages),
        "raw_kb": raw_kb, "elapsed_s": round(elapsed, 2),
        "degrade": degrade.value, "ans_len": ans_len,
        "timed_out": timed_out, "empty": empty,
        "answer_preview": (answer or "")[:120],
    }


async def main():
    from src.config import settings

    print(f"\n=== Compression config: MAX_SIDE={settings.VLM_IMG_MAX_SIDE}, "
          f"QUALITY={settings.VLM_IMG_JPEG_QUALITY}, MAX_BYTES={settings.VLM_IMG_MAX_BYTES}, "
          f"MAX_VLM_IMAGES={settings.MAX_VLM_IMAGES} ===\n")

    report_ids = sample_report_ids(3)
    if not report_ids:
        log.error("No data in Milvus pages collection")
        return

    log.info("sampled_reports", count=len(report_ids), ids=report_ids)

    results = []

    # ── Round A: single-image real table query ───────────────────────────────
    for rid in report_ids[:2]:
        pages1 = fetch_pages(rid, top_k=1)
        if not pages1:
            log.warning("no_table_pages", report_id=rid)
            continue
        r = await run_round(
            label=f"single_img/{rid[:12]}",
            query="请列出本页中所有财务数据（营业收入、净利润、总资产等），并说明数据来源页码。",
            pages=pages1,
        )
        results.append(r)
        await asyncio.sleep(2)

    # ── Round B: 3-image real table query ───────────────────────────────────
    for rid in report_ids[:2]:
        pages3 = fetch_pages(rid, top_k=3)
        if not pages3:
            continue
        r = await run_round(
            label=f"3img/{rid[:12]}",
            query="2023年营业收入和净利润分别是多少？请引用具体页码和数字。",
            pages=pages3,
        )
        results.append(r)
        await asyncio.sleep(2)

    # ── Round C: Top-5 multi-image ───────────────────────────────────────────
    for rid in report_ids[:2]:
        pages5 = fetch_pages(rid, top_k=5)
        if not pages5:
            continue
        r = await run_round(
            label=f"top5/{rid[:12]}",
            query="请综合所有提供页面，列出公司最近一年的核心财务指标（收入、利润、资产、负债），并注明数据所在页码。",
            pages=pages5,
        )
        results.append(r)
        await asyncio.sleep(2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Label':<28} {'Pages':>5} {'RawKB':>8} {'VLM(s)':>7} "
          f"{'Degrade':<18} {'AnsLen':>6} {'Timeout':>7} {'Empty':>5}")
    print("-" * 90)
    passed, failed = [], []
    for r in results:
        ok = not r["timed_out"] and not r["empty"] and r["degrade"] == "none"
        flag = "✓" if ok else "✗"
        print(f"{flag} {r['label']:<26} {r['pages']:>5} {r['raw_kb']:>8} "
              f"{r['elapsed_s']:>7.2f} {r['degrade']:<18} {r['ans_len']:>6} "
              f"{str(r['timed_out']):>7} {str(r['empty']):>5}")
        (passed if ok else failed).append(r["label"])

    print("=" * 90)
    print(f"\n通过: {len(passed)} 项  →  {passed}")
    print(f"未通过: {len(failed)} 项  →  {failed}")

    # ── Recommendation ───────────────────────────────────────────────────────
    all_pass = len(failed) == 0
    any_timeout = any(r["timed_out"] for r in results)
    any_empty = any(r["empty"] for r in results)
    avg_elapsed = sum(r["elapsed_s"] for r in results) / len(results) if results else 0

    print(f"\n平均 VLM 耗时: {avg_elapsed:.2f}s")
    print(f"是否出现超时: {any_timeout}")
    print(f"是否出现空响应: {any_empty}")

    if all_pass and not any_timeout and avg_elapsed < 60:
        print("\n✅ 建议：压缩回归全部通过，延迟可控，建议继续推进两阶段 VLM 证据扩展。")
    elif any_timeout or avg_elapsed >= 60:
        print("\n⚠️  建议：存在超时或高延迟，建议先调低 MAX_VLM_IMAGES 或提高压缩强度，再评估两阶段扩展。")
    else:
        print("\n⚠️  建议：部分用例未通过，需排查 degrade_reason 后再决定是否进入两阶段扩展。")


if __name__ == "__main__":
    asyncio.run(main())
