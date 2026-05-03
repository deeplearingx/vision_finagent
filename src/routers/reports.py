import asyncio
import json
import os
import structlog
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, List

from ..config import settings
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

log = structlog.get_logger()
from ..models.schema import PageResult, EvidenceSummary
from ..models.enums import ReportStatus, DegradeReason
from ..utils.idempotency import check_and_set, release as _release_idem
from ..services.retrieval_service import retrieve
from ..services.vlm_service import (
    generate_answer, is_insufficient_evidence,
    is_market_like_question, MARKET_BOUNDARY_HINT,
    stream_generate_answer,
)
from ..services.task_service import submit_ingest_task, get_task
from ..core.redis_client import get_redis
from ..core.milvus_client import clear_report_collections, list_reports_inventory, delete_report_data, delete_report_data_by_prefix

# Redis key: session:{session_id}  TTL: 2 days
_SESSION_TTL = 2 * 86400
_SESSION_MAX_HISTORY = 40
_SESSION_INDEX_KEY = "session_index"
_SESSION_LIST_LIMIT = 50

router = APIRouter(prefix="/reports")

# ---------------------------------------------------------------------------
# Maintenance lock: prevents concurrent upload/query during clear-collections.
# ---------------------------------------------------------------------------
_maintenance_lock = asyncio.Lock()
_maintenance_active = False


_ALLOWED_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}


def _is_valid_pdf(data: bytes) -> bool:
    tail = data[-1024:]
    return data[:4] == b"%PDF" and (b"%%EOF" in tail or b"%EOF" in tail)


def _is_valid_image(data: bytes, ext: str) -> bool:
    if ext in (".jpg", ".jpeg"):
        return data[:3] == b"\xff\xd8\xff"
    if ext == ".png":
        return data[:4] == b"\x89PNG"
    if ext == ".webp":
        return data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


class QueryRequest(BaseModel):
    question: str
    target_companies: list[str] = []
    report_ids: list[str] = []
    top_k: int = 5
    candidate_k: int = 50
    session_id: str | None = None
    use_retrieval: bool = True
    refresh_retrieval: bool = False


@router.post("/upload", status_code=202)
async def upload_report(
    file: UploadFile = File(...),
    report_id: str = Form(default=""),
):
    if _maintenance_active:
        raise HTTPException(503, "系统维护中，暂停上传，请稍后重试")
    if not report_id:
        report_id = f"report_{uuid.uuid4().hex[:8]}"
    token = f"upload:{report_id}"
    if not await check_and_set(token):
        raise HTTPException(409, "Duplicate upload")

    data = await file.read()

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".pdf"
    if suffix not in _ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {', '.join(_ALLOWED_EXTS)}")

    if suffix == ".pdf":
        if not _is_valid_pdf(data):
            raise HTTPException(400, "Invalid PDF: missing %PDF header or %%EOF marker")
    else:
        if not _is_valid_image(data, suffix):
            raise HTTPException(400, f"Invalid image: magic bytes do not match {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    task_id = await submit_ingest_task(report_id, tmp_path)
    return {
        "report_id": report_id,
        "task_id": task_id,
        "status": ReportStatus.PENDING,
    }


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    info = await get_task(task_id)
    if info is None:
        raise HTTPException(404, "Task not found")
    return {
        "task_id": info.task_id,
        "report_id": info.report_id,
        "status": info.status,
        "detail": info.detail,
        "created_at": info.created_at,
        "updated_at": info.updated_at,
    }


async def _load_history(session_id: str) -> list[dict]:
    r = await get_redis()
    raw = await r.get(f"session:{session_id}")
    if raw is None:
        return []
    return json.loads(raw)


async def _save_history(session_id: str, hist: list[dict]) -> None:
    r = await get_redis()
    await r.set(f"session:{session_id}", json.dumps(hist), ex=_SESSION_TTL)


async def _load_session_meta(session_id: str) -> dict[str, Any]:
    r = await get_redis()
    raw = await r.get(f"session:{session_id}:meta")
    if raw is None:
        return {}
    return json.loads(raw)


async def _save_session_meta(session_id: str, meta: dict[str, Any]) -> None:
    r = await get_redis()
    await r.set(f"session:{session_id}:meta", json.dumps(meta), ex=_SESSION_TTL)


async def _touch_session_index(session_id: str, updated_ts: float) -> None:
    r = await get_redis()
    await r.zadd(_SESSION_INDEX_KEY, {session_id: updated_ts})
    await r.expire(_SESSION_INDEX_KEY, _SESSION_TTL)


def _iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


async def _persist_session_state(
    session_id: str,
    hist: list[dict],
    *,
    last_question: str,
    has_evidence: bool,
) -> None:
    updated_ts = time.time()
    await _save_history(session_id, hist)
    await _save_session_meta(
        session_id,
        {
            "session_id": session_id,
            "updated_at": _iso_from_ts(updated_ts),
            "updated_ts": updated_ts,
            "last_question": last_question[:120],
            "turn_count": sum(1 for item in hist if item.get("role") == "user"),
            "has_evidence": has_evidence,
        },
    )
    await _touch_session_index(session_id, updated_ts)


async def _load_evidence_cache(session_id: str) -> dict[str, Any] | None:
    r = await get_redis()
    raw = await r.get(f"session:{session_id}:evidence")
    if raw is None:
        return None
    return json.loads(raw)


async def _save_evidence_cache(session_id: str, pages: list[PageResult]) -> None:
    """Store only lightweight meta (no image_base64) to keep Redis payload small."""
    r = await get_redis()
    payload = {
        "cached_at": _iso_from_ts(time.time()),
        "evidence": [
            {
                "report_id": page.report_id,
                "page_num": page.page_num,
                "maxsim_score": page.maxsim_score,
            }
            for page in pages
        ],
    }
    await r.set(f"session:{session_id}:evidence", json.dumps(payload), ex=_SESSION_TTL)


async def _clear_evidence_cache(session_id: str) -> None:
    r = await get_redis()
    await r.delete(f"session:{session_id}:evidence")


async def _build_pages_from_cache(
    session_id: str,
) -> tuple[list["PageResult"], bool, bool]:
    """Load cached evidence and fetch images from Milvus.

    Returns:
        (pages, found_cache, image_fetch_incomplete)
        - found_cache=False means no cache exists at all.
    """
    from ..core.milvus_client import get_client as _get_mc, get_pages_collection_name as _pages_col
    cached = await _load_evidence_cache(session_id)
    cached_evidence = cached.get("evidence", []) if cached else []
    if not cached_evidence:
        return [], False, False

    page_ids = [f"{e['report_id']}_{e['page_num']}" for e in cached_evidence]
    img_map: dict = {}
    image_fetch_incomplete = False
    try:
        meta_rows = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_mc().query(
                collection_name=_pages_col(),
                filter=f"page_id in {json.dumps(page_ids)}",
                output_fields=["report_id", "page_num", "image_base64"],
                limit=len(page_ids) + 1,
            ),
        )
        img_map = {(r["report_id"], r["page_num"]): r["image_base64"] for r in meta_rows}
        missing = [pid for pid in page_ids if not img_map.get(tuple(pid.rsplit("_", 1)))]
        # rebuild missing check properly
        missing = [
            f"{e['report_id']}_{e['page_num']}"
            for e in cached_evidence
            if not img_map.get((e["report_id"], e["page_num"]))
        ]
        if missing:
            image_fetch_incomplete = True
            log.warning(
                "cache_reuse_image_missing",
                session_id=session_id,
                missing_page_ids=missing,
                detail="Some pages have no image_base64 in Milvus; VLM quality may degrade",
            )
    except Exception:
        image_fetch_incomplete = True
        log.warning(
            "cache_reuse_image_fetch_failed",
            session_id=session_id,
            page_ids=page_ids,
            detail="Milvus image query failed; image_base64 will be empty, VLM quality may degrade",
        )

    pages = [
        PageResult(
            report_id=e["report_id"],
            page_num=e["page_num"],
            image_base64=img_map.get((e["report_id"], e["page_num"]), ""),
            maxsim_score=e["maxsim_score"],
        )
        for e in cached_evidence
    ]
    return pages, True, image_fetch_incomplete


async def _backfill_meta(session_id: str, score: float) -> dict:
    """Minimal meta backfill for legacy sessions that have history but no meta key."""
    hist = await _load_history(session_id)
    if not hist:
        return {}
    user_turns = [m for m in hist if m.get("role") == "user"]
    last_q = user_turns[-1]["content"] if user_turns else ""
    meta = {
        "session_id": session_id,
        "updated_at": _iso_from_ts(score),
        "updated_ts": score,
        "last_question": last_q[:120],
        "turn_count": len(user_turns),
        "has_evidence": False,
        "_backfilled": True,
    }
    await _save_session_meta(session_id, meta)
    log.info("session_meta_backfilled", session_id=session_id)
    return meta


@router.get("/sessions")
async def list_sessions():
    r = await get_redis()
    entries = await r.zrevrange(_SESSION_INDEX_KEY, 0, _SESSION_LIST_LIMIT - 1, withscores=True)
    result = []
    for sid, score in entries:
        meta = await _load_session_meta(sid)
        if not meta:
            # Issue 4: legacy session has no meta — attempt minimal backfill
            meta = await _backfill_meta(sid, score)
        if not meta:
            continue
        has_evidence = bool(meta.get("has_evidence"))
        result.append({
            "session_id": sid,
            "updated_at": meta.get("updated_at") or _iso_from_ts(score),
            "last_question": meta.get("last_question", "")[:80],
            "turn_count": meta.get("turn_count", 0),
            "has_evidence": has_evidence,
        })
    return {"sessions": result}


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    hist = await _load_history(session_id)
    if not hist:
        raise HTTPException(404, "Session not found")
    return {"session_id": session_id, "history": hist}


@router.get("/sessions/{session_id}/evidence-status")
async def get_session_evidence_status(session_id: str):
    """Return real-time evidence cache status for a session."""
    evidence_cache = await _load_evidence_cache(session_id)
    has_evidence = bool(evidence_cache and evidence_cache.get("evidence"))
    return {"session_id": session_id, "has_evidence": has_evidence}


@router.delete("/sessions/{session_id}", status_code=200)
async def delete_session(session_id: str):
    """Delete a session and all its associated Redis keys."""
    r = await get_redis()
    await r.delete(
        f"session:{session_id}",
        f"session:{session_id}:meta",
        f"session:{session_id}:evidence",
    )
    await r.zrem(_SESSION_INDEX_KEY, session_id)
    return {"session_id": session_id, "deleted": True}


_MAINTENANCE_REDIS_KEY = "maintenance:clear_collections"
_MAINTENANCE_REDIS_TTL = 300  # 5 min safety expiry


async def _acquire_redis_maintenance(r) -> bool:
    """SET NX EX — returns True if this caller acquired the lock."""
    return bool(await r.set(_MAINTENANCE_REDIS_KEY, "1", nx=True, ex=_MAINTENANCE_REDIS_TTL))


async def _release_redis_maintenance(r) -> None:
    await r.delete(_MAINTENANCE_REDIS_KEY)


@router.get("/admin/inventory")
async def get_reports_inventory():
    """只读：枚举 pages 集合中所有 report_id 及其页数统计。"""
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, list_reports_inventory)
        return result
    except Exception as exc:
        log.error("inventory_failed", error=str(exc))
        raise HTTPException(500, f"查询报告清单失败: {exc}")


@router.get("/admin/inventory/{report_id}")
async def check_report_in_inventory(report_id: str):
    """只读：检查指定 report_id 是否已入库，返回页数统计。"""
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, list_reports_inventory)
    except Exception as exc:
        log.error("inventory_check_failed", report_id=report_id, error=str(exc))
        raise HTTPException(500, f"查询报告清单失败: {exc}")

    matched = next((r for r in result["reports"] if r["report_id"] == report_id), None)
    return {
        "report_id": report_id,
        "found": matched is not None,
        "page_count": matched["page_count"] if matched else 0,
        "page_nums": matched["page_nums"] if matched else [],
        "collection": result["collection"],
        "truncated": result["truncated"],
    }


@router.delete("/admin/report/{report_id}", status_code=200)
async def delete_report(report_id: str):
    """删除指定 report_id 的所有 patches 和 pages 数据，并释放幂等 token。"""
    try:
        await asyncio.get_event_loop().run_in_executor(None, delete_report_data, report_id)
        await _release_idem(f"upload:{report_id}")
        return {"report_id": report_id, "deleted": True}
    except Exception as exc:
        raise HTTPException(500, f"删除失败：{exc}")


@router.delete("/admin/report-prefix/{prefix}", status_code=200)
async def delete_reports_by_prefix(prefix: str):
    """删除所有 report_id 以 prefix 开头的报告数据，并释放对应幂等 token。"""
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, delete_report_data_by_prefix, prefix)
        for rid in result.get("matched_ids", []):
            await _release_idem(f"upload:{rid}")
        return result
    except Exception as exc:
        raise HTTPException(500, f"批量删除失败：{exc}")


@router.post("/admin/clear-collections", status_code=200)
async def clear_collections():
    global _maintenance_active
    r = await get_redis()

    # Fast-path: in-process flag (single-process guard, zero-latency)
    if _maintenance_active:
        raise HTTPException(409, "清库操作已在进行中")

    # Cross-process guard via Redis SET NX
    if not await _acquire_redis_maintenance(r):
        raise HTTPException(409, "清库操作已在进行中（其他实例）")

    _maintenance_active = True
    try:
        async with _maintenance_lock:
            # 1. Clear all evidence caches
            cursor = 0
            while True:
                cursor, keys = await r.scan(cursor, match="session:*:evidence", count=200)
                if keys:
                    await r.delete(*keys)
                if cursor == 0:
                    break

            # 2. Patch every session meta: set has_evidence=False
            cursor = 0
            while True:
                cursor, meta_keys = await r.scan(cursor, match="session:*:meta", count=200)
                for mk in meta_keys:
                    raw = await r.get(mk)
                    if raw is None:
                        continue
                    try:
                        meta = json.loads(raw)
                    except Exception:
                        continue
                    if meta.get("has_evidence"):
                        meta["has_evidence"] = False
                        ttl = await r.ttl(mk)
                        await r.set(mk, json.dumps(meta), ex=max(ttl, 1) if ttl > 0 else _SESSION_TTL)
                if cursor == 0:
                    break

            result = await asyncio.get_event_loop().run_in_executor(None, clear_report_collections)
            result["schema_rebuilt"] = True
            result["notice"] = "集合已重建（schema 已更新），所有历史数据已清除，请重新上传文件以应用新 schema。"
            if result.get("schema_drift_detected"):
                result["schema_warning"] = "重建后仍检测到 schema 漂移，请检查 schema_health 字段"
            log.info("clear_collections_done", result=result)
            return result
    finally:
        _maintenance_active = False
        await _release_redis_maintenance(r)


# ---------- query ----------

@router.post("/query")
async def query_report(req: QueryRequest):
    if _maintenance_active:
        raise HTTPException(503, "系统维护中，暂停查询，请稍后重试")
    session_id = req.session_id or str(uuid.uuid4())
    degraded = False
    degrade_reason = DegradeReason.NONE
    answer: str | None = None
    pages: List[PageResult] = []
    evidence_source = "new_retrieval"
    image_fetch_incomplete = False

    run_retrieval = req.use_retrieval or req.refresh_retrieval
    vlm_passes = 0
    second_pass_triggered = False

    if run_retrieval:
        try:
            _loop = asyncio.get_event_loop()
            _pool = getattr(_loop, "_default_executor", None)
            _pool_info = f"threads={getattr(_pool, '_max_workers', '?')} queue={getattr(getattr(_pool, '_work_queue', None), 'qsize', lambda: '?')()}" if _pool else "default(not-init)"
            log.info("retrieval.submit", session_id=session_id, executor_info=_pool_info, timeout=settings.QUERY_TIMEOUT)
            pages = await asyncio.wait_for(
                _loop.run_in_executor(
                    None,
                    lambda: retrieve(req.question, req.target_companies, req.top_k, req.candidate_k, "first_pass", req.report_ids or None),
                ),
                timeout=settings.QUERY_TIMEOUT,
            )
            log.info("retrieval.submit_done", session_id=session_id, pages=len(pages))
        except asyncio.TimeoutError:
            log.error("retrieval.timeout", session_id=session_id, question=req.question,
                      evidence_source="live")
            degrade_reason = DegradeReason.RETRIEVAL_TIMEOUT
            pages = []
        except Exception:
            log.exception("retrieval.error", session_id=session_id, question=req.question,
                          evidence_source="live")
            degrade_reason = DegradeReason.RETRIEVAL_ERROR
            pages = []

        if pages:
            await _save_evidence_cache(session_id, pages)
            log.info("retrieval.evidence_source", source="live", session_id=session_id, pages=len(pages))
        elif degrade_reason != DegradeReason.NONE:
            # Retrieval failed — try fallback to cached evidence
            fallback_pages, found_cache, fetch_incomplete = await _build_pages_from_cache(session_id)
            if found_cache:
                pages = fallback_pages
                image_fetch_incomplete = fetch_incomplete
                evidence_source = "cached_fallback"
                log.warning(
                    "retrieval.cached_fallback",
                    session_id=session_id,
                    degrade_reason=degrade_reason.value,
                    cached_pages=len(pages),
                )
            else:
                degraded = True
                log.warning(
                    "retrieval.no_cache_fallback",
                    session_id=session_id,
                    degrade_reason=degrade_reason.value,
                    evidence_source="none",
                )
        else:
            # Retrieval succeeded but returned no results — clear stale cache
            await _clear_evidence_cache(session_id)
            log.info("retrieval.evidence_source", source="live_empty", session_id=session_id)
    else:
        cached_pages, found_cache, fetch_incomplete = await _build_pages_from_cache(session_id)
        if found_cache:
            pages = cached_pages
            image_fetch_incomplete = fetch_incomplete
            evidence_source = "cached"
            log.info("retrieval.evidence_source", source="cached", session_id=session_id, pages=len(pages))
        else:
            degraded = True
            degrade_reason = DegradeReason.NO_EVIDENCE
            evidence_source = "none"
            answer = "当前会话尚无可复用的检索证据，请开启「检索数据」后再提问，或先进行一次检索。"
            log.warning("retrieval.evidence_source", source="none", session_id=session_id)

    if not degraded and pages:
        t0 = time.monotonic()
        answer, vlm_reason = await generate_answer(req.question, pages)
        vlm_passes = 1
        first_pass_elapsed = round(time.monotonic() - t0, 3)
        log.info(
            "vlm.first_pass_done",
            session_id=session_id,
            pages=len(pages),
            elapsed=first_pass_elapsed,
            vlm_reason=vlm_reason.value,
            answer_preview=(answer or "")[:80],
        )

        # Second-pass: only when VLM explicitly signals insufficient evidence
        if (
            vlm_reason == DegradeReason.NONE
            and settings.VLM_SECOND_PASS_ENABLED
            and is_insufficient_evidence(answer)
        ):
            second_pass_triggered = True
            log.info(
                "vlm.second_pass_triggered",
                session_id=session_id,
                first_pass_pages=len(pages),
                expanded_top_k=settings.VLM_SECOND_PASS_TOP_K,
                expanded_candidate_k=settings.VLM_SECOND_PASS_CANDIDATE_K,
            )
            t1 = time.monotonic()
            try:
                pages2 = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: retrieve(
                            req.question,
                            req.target_companies,
                            settings.VLM_SECOND_PASS_TOP_K,
                            settings.VLM_SECOND_PASS_CANDIDATE_K,
                            "second_pass",
                            req.report_ids or None,
                        ),
                    ),
                    timeout=settings.QUERY_TIMEOUT,
                )
            except Exception:
                log.exception("vlm.second_pass_retrieval_failed", session_id=session_id)
                pages2 = []

            if pages2:
                # Temporarily cap images for second pass
                import copy
                pages2_capped = pages2[: settings.VLM_SECOND_PASS_MAX_IMAGES]
                answer2, vlm_reason2 = await generate_answer(req.question, pages2_capped)
                vlm_passes = 2
                second_pass_elapsed = round(time.monotonic() - t1, 3)
                log.info(
                    "vlm.second_pass_done",
                    session_id=session_id,
                    pages=len(pages2_capped),
                    elapsed=second_pass_elapsed,
                    vlm_reason=vlm_reason2.value,
                    answer_preview=(answer2 or "")[:80],
                )
                if vlm_reason2 == DegradeReason.NONE and answer2:
                    # Second pass succeeded — use its answer and pages
                    answer = answer2
                    pages = pages2
                    evidence_source = "second_pass"
                else:
                    # Second pass failed — keep first-pass answer (degraded semantics preserved)
                    log.warning(
                        "vlm.second_pass_no_improvement",
                        session_id=session_id,
                        vlm_reason2=vlm_reason2.value,
                    )
            else:
                log.warning("vlm.second_pass_no_pages", session_id=session_id)

        if vlm_reason != DegradeReason.NONE and not second_pass_triggered:
            degraded = True
            degrade_reason = vlm_reason
            answer = f"VLM 生成失败（{vlm_reason.value}），已找到 {len(pages)} 个相关页面，最高相关度 {pages[0].maxsim_score:.2f}（{pages[0].report_id} 第 {pages[0].page_num} 页）。"

        # cached_fallback: prepend notice so user knows retrieval failed
        if evidence_source == "cached_fallback" and answer:
            answer = "（本次检索失败，已自动复用上次证据继续回答）\n\n" + answer

        # Soft market-boundary hint: append only when evidence is still insufficient
        # after all passes AND the question looks market/price-trend oriented.
        # Never blocks retrieval or VLM — purely additive to the answer text.
        if (
            answer
            and is_insufficient_evidence(answer)
            and is_market_like_question(req.question)
        ):
            answer = answer + MARKET_BOUNDARY_HINT
            log.info("vlm.market_boundary_hint_appended", session_id=session_id, question=req.question)

    elif not degraded and not pages:
        # No retrieval results — call VLM text-only with no-evidence note
        answer, vlm_reason = await generate_answer(req.question, [])
        if not answer or vlm_reason != DegradeReason.NONE:
            answer = "未找到相关页面。"
        evidence_source = "none"

    evidence: list[EvidenceSummary] = [
        EvidenceSummary(report_id=p.report_id, page_num=p.page_num, maxsim_score=p.maxsim_score)
        for p in pages
    ]

    hist = await _load_history(session_id)
    hist.append({"role": "user", "content": req.question})
    hist.append({
        "role": "assistant",
        "content": answer or (f"找到 {len(pages)} 个相关页面" if pages else "未找到相关页面"),
        "degraded": degraded,
        "degrade_reason": degrade_reason.value,
        "evidence": [e.model_dump() for e in evidence],
        "evidence_source": evidence_source,
    })
    if len(hist) > _SESSION_MAX_HISTORY:
        hist = hist[-_SESSION_MAX_HISTORY:]

    has_evidence = bool((await _load_evidence_cache(session_id) or {}).get("evidence", []))
    await _persist_session_state(
        session_id,
        hist,
        last_question=req.question,
        has_evidence=has_evidence,
    )

    return {
        "session_id": session_id,
        "answer": answer,
        "degraded": degraded,
        "degrade_reason": degrade_reason.value,
        "evidence": [e.model_dump() for e in evidence],
        "evidence_source": evidence_source,
        "image_fetch_incomplete": image_fetch_incomplete,
        "retrieved_pages": [
            {"report_id": p.report_id, "page_num": p.page_num, "maxsim_score": p.maxsim_score, "image_base64": p.image_base64}
            for p in pages
        ],
        # Observability fields (additive — no existing field removed)
        "vlm_passes": vlm_passes,
        "second_pass_triggered": second_pass_triggered,
        "evidence_source_detail": evidence_source,
    }


@router.post("/query/stream")
async def query_report_stream(req: QueryRequest):
    async def sse_generator():
        session_id = req.session_id or str(uuid.uuid4())
        degraded = False
        degrade_reason = DegradeReason.NONE
        answer: str | None = None
        pages: List[PageResult] = []
        evidence_source = "new_retrieval"
        image_fetch_incomplete = False

        run_retrieval = req.use_retrieval or req.refresh_retrieval
        vlm_passes = 0
        second_pass_triggered = False

        try:
            if _maintenance_active:
                yield f"event: error\ndata: {json.dumps({'code': 'maintenance', 'message': '系统维护中，暂停查询，请稍后重试'})}\n\n"
                return

            # --- meta event ---
            yield f"event: meta\ndata: {json.dumps({'session_id': session_id, 'degraded': degraded, 'vlm_passes': vlm_passes, 'second_pass_triggered': second_pass_triggered})}\n\n"

            # --- retrieval logic (mirrors /query exactly) ---
            if run_retrieval:
                yield f"event: status\ndata: {json.dumps({'message': '正在检索相关页面...'})}\n\n"
                try:
                    pages = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: retrieve(req.question, req.target_companies, req.top_k, req.candidate_k, "first_pass", req.report_ids or None),
                        ),
                        timeout=settings.QUERY_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    log.error("retrieval.timeout", session_id=session_id, question=req.question,
                              evidence_source="live")
                    degrade_reason = DegradeReason.RETRIEVAL_TIMEOUT
                    pages = []
                except Exception:
                    log.exception("retrieval.error", session_id=session_id, question=req.question,
                                  evidence_source="live")
                    degrade_reason = DegradeReason.RETRIEVAL_ERROR
                    pages = []

                if pages:
                    await _save_evidence_cache(session_id, pages)
                    log.info("retrieval.evidence_source", source="live", session_id=session_id, pages=len(pages))
                    yield f"event: status\ndata: {json.dumps({'message': f'已找到 {len(pages)} 个相关页面，正在生成回答...'})}\n\n"
                elif degrade_reason != DegradeReason.NONE:
                    # Retrieval failed — try fallback to cached evidence
                    fallback_pages, found_cache, fetch_incomplete = await _build_pages_from_cache(session_id)
                    if found_cache:
                        pages = fallback_pages
                        image_fetch_incomplete = fetch_incomplete
                        evidence_source = "cached_fallback"
                        log.warning(
                            "retrieval.cached_fallback",
                            session_id=session_id,
                            degrade_reason=degrade_reason.value,
                            cached_pages=len(pages),
                        )
                        yield f"event: status\ndata: {json.dumps({'message': f'本次检索失败，已自动复用上次证据继续回答（{len(pages)} 个页面）...'})}\n\n"
                    else:
                        degraded = True
                        evidence_source = "none"
                        log.warning(
                            "retrieval.no_cache_fallback",
                            session_id=session_id,
                            degrade_reason=degrade_reason.value,
                            evidence_source="none",
                        )
                        _msg = "检索超时，无法获取相关证据，且当前会话无缓存证据，请稍后重试。" \
                            if degrade_reason == DegradeReason.RETRIEVAL_TIMEOUT \
                            else "检索异常，无法获取相关证据，且当前会话无缓存证据，请稍后重试。"
                        yield f"event: content\ndata: {json.dumps({'chunk': _msg})}\n\n"
                else:
                    await _clear_evidence_cache(session_id)
                    log.info("retrieval.evidence_source", source="live_empty", session_id=session_id)
            else:
                cached_pages, found_cache, fetch_incomplete = await _build_pages_from_cache(session_id)
                if found_cache:
                    pages = cached_pages
                    image_fetch_incomplete = fetch_incomplete
                    evidence_source = "cached"
                    log.info("retrieval.evidence_source", source="cached", session_id=session_id, pages=len(pages))
                else:
                    degraded = True
                    degrade_reason = DegradeReason.NO_EVIDENCE
                    evidence_source = "none"
                    answer = "当前会话尚无可复用的检索证据，请开启「检索数据」后再提问，或先进行一次检索。"
                    log.warning("retrieval.evidence_source", source="none", session_id=session_id)
                    yield f"event: content\ndata: {json.dumps({'chunk': answer})}\n\n"

            evidence: list[EvidenceSummary] = [
                EvidenceSummary(report_id=p.report_id, page_num=p.page_num, maxsim_score=p.maxsim_score)
                for p in pages
            ]

            # --- evidence event (include image_base64 so frontend can render pages) ---
            yield f"event: evidence\ndata: {json.dumps({'evidence': [e.model_dump() for e in evidence], 'retrieved_pages': [{'report_id': p.report_id, 'page_num': p.page_num, 'maxsim_score': p.maxsim_score, 'image_base64': p.image_base64} for p in pages], 'image_fetch_incomplete': image_fetch_incomplete})}\n\n"

            if not degraded and pages:
                # cached_fallback: notify user before streaming answer
                if evidence_source == "cached_fallback":
                    _notice = "（本次检索失败，已自动复用上次证据继续回答）\n\n"
                    yield f"event: content\ndata: {json.dumps({'chunk': _notice})}\n\n"

                # First pass streaming
                t0 = time.monotonic()
                first_pass_answer = ""
                try:
                    async for chunk in stream_generate_answer(req.question, pages):
                        first_pass_answer += chunk
                        yield f"event: content\ndata: {json.dumps({'chunk': chunk})}\n\n"
                except asyncio.TimeoutError:
                    log.warning("vlm.stream_timeout", session_id=session_id, query=req.question)
                    yield f"event: error\ndata: {json.dumps({'code': 'vlm_timeout', 'message': 'VLM 生成超时'})}\n\n"
                    return
                except Exception as exc:
                    log.error("vlm.stream_error", session_id=session_id, error=str(exc))
                    yield f"event: error\ndata: {json.dumps({'code': 'vlm_error', 'message': str(exc)})}\n\n"
                    return

                vlm_passes = 1
                answer = first_pass_answer
                first_pass_elapsed = round(time.monotonic() - t0, 3)
                log.info(
                    "vlm.first_pass_stream_done",
                    session_id=session_id,
                    pages=len(pages),
                    elapsed=first_pass_elapsed,
                    answer_preview=(answer or "")[:80],
                )

                if not answer or len(answer) < 5:
                    degraded = True
                    degrade_reason = DegradeReason.VLM_ERROR
                    answer = f"VLM 生成失败（{DegradeReason.VLM_ERROR.value}），已找到 {len(pages)} 个相关页面，最高相关度 {pages[0].maxsim_score:.2f}（{pages[0].report_id} 第 {pages[0].page_num} 页）。"
                    yield f"event: content\ndata: {json.dumps({'chunk': answer})}\n\n"

                # Second pass
                if (
                    settings.VLM_SECOND_PASS_ENABLED
                    and is_insufficient_evidence(answer)
                ):
                    second_pass_triggered = True
                    yield f"event: second_pass\ndata: {json.dumps({})}\n\n"
                    log.info(
                        "vlm.second_pass_triggered",
                        session_id=session_id,
                        first_pass_pages=len(pages),
                        expanded_top_k=settings.VLM_SECOND_PASS_TOP_K,
                        expanded_candidate_k=settings.VLM_SECOND_PASS_CANDIDATE_K,
                    )
                    t1 = time.monotonic()
                    try:
                        pages2 = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: retrieve(
                                    req.question,
                                    req.target_companies,
                                    settings.VLM_SECOND_PASS_TOP_K,
                                    settings.VLM_SECOND_PASS_CANDIDATE_K,
                                    "second_pass",
                                    req.report_ids or None,
                                ),
                            ),
                            timeout=settings.QUERY_TIMEOUT,
                        )
                    except Exception:
                        log.exception("vlm.second_pass_retrieval_failed", session_id=session_id)
                        pages2 = []

                    if pages2:
                        pages2_capped = pages2[: settings.VLM_SECOND_PASS_MAX_IMAGES]
                        # New evidence (second pass — include images)
                        evidence = [
                            EvidenceSummary(report_id=p.report_id, page_num=p.page_num, maxsim_score=p.maxsim_score)
                            for p in pages2
                        ]
                        yield f"event: evidence\ndata: {json.dumps({'evidence': [e.model_dump() for e in evidence], 'retrieved_pages': [{'report_id': p.report_id, 'page_num': p.page_num, 'maxsim_score': p.maxsim_score, 'image_base64': p.image_base64} for p in pages2], 'image_fetch_incomplete': False})}\n\n"

                        second_pass_answer = ""
                        try:
                            async for chunk in stream_generate_answer(req.question, pages2_capped):
                                second_pass_answer += chunk
                                yield f"event: content\ndata: {json.dumps({'chunk': chunk})}\n\n"
                        except asyncio.TimeoutError:
                            log.warning("vlm.second_stream_timeout", session_id=session_id)
                            yield f"event: error\ndata: {json.dumps({'code': 'vlm_timeout', 'message': 'VLM 二轮生成超时'})}\n\n"
                            return
                        except Exception as exc:
                            log.error("vlm.second_stream_error", session_id=session_id, error=str(exc))
                            yield f"event: error\ndata: {json.dumps({'code': 'vlm_error', 'message': str(exc)})}\n\n"
                            return

                        vlm_passes = 2
                        answer = second_pass_answer
                        evidence_source = "second_pass"
                        pages = pages2
                        second_pass_elapsed = round(time.monotonic() - t1, 3)
                        log.info(
                            "vlm.second_pass_stream_done",
                            session_id=session_id,
                            pages=len(pages2_capped),
                            elapsed=second_pass_elapsed,
                            answer_preview=(answer or "")[:80],
                        )
                    else:
                        log.warning("vlm.second_pass_no_pages", session_id=session_id)

                if (
                    answer
                    and is_insufficient_evidence(answer)
                    and is_market_like_question(req.question)
                ):
                    answer = answer + MARKET_BOUNDARY_HINT
                    yield f"event: content\ndata: {json.dumps({'chunk': MARKET_BOUNDARY_HINT})}\n\n"
                    log.info("vlm.market_boundary_hint_appended", session_id=session_id, question=req.question)

            elif not degraded and not pages:
                # No retrieval results — stream VLM text-only with no-evidence note
                evidence_source = "none"
                text_only_answer = ""
                try:
                    async for chunk in stream_generate_answer(req.question, []):
                        text_only_answer += chunk
                        yield f"event: content\ndata: {json.dumps({'chunk': chunk})}\n\n"
                except Exception as exc:
                    log.error("vlm.text_only_stream_error", session_id=session_id, error=str(exc))
                if not text_only_answer:
                    text_only_answer = "未找到相关页面。"
                    yield f"event: content\ndata: {json.dumps({'chunk': text_only_answer})}\n\n"
                answer = text_only_answer

            if degraded and answer and not vlm_passes:
                evidence_source = "none"
                yield f"event: content\ndata: {json.dumps({'chunk': answer})}\n\n"

            # Rebuild evidence with final pages
            evidence = [
                EvidenceSummary(report_id=p.report_id, page_num=p.page_num, maxsim_score=p.maxsim_score)
                for p in pages
            ]

            # --- done event ---
            done_payload = {
                "session_id": session_id,
                "degraded": degraded,
                "degrade_reason": degrade_reason.value,
                "evidence_source": evidence_source,
                "vlm_passes": vlm_passes,
                "second_pass_triggered": second_pass_triggered,
                "image_fetch_incomplete": image_fetch_incomplete,
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

            # --- persist to Redis (after stream ends) ---
            hist = await _load_history(session_id)
            hist.append({"role": "user", "content": req.question})
            hist.append({
                "role": "assistant",
                "content": answer or (f"找到 {len(pages)} 个相关页面" if pages else "未找到相关页面"),
                "degraded": degraded,
                "degrade_reason": degrade_reason.value,
                "evidence": [e.model_dump() for e in evidence],
                "evidence_source": evidence_source,
            })
            if len(hist) > _SESSION_MAX_HISTORY:
                hist = hist[-_SESSION_MAX_HISTORY:]

            has_evidence = bool((await _load_evidence_cache(session_id) or {}).get("evidence", []))
            await _persist_session_state(
                session_id,
                hist,
                last_question=req.question,
                has_evidence=has_evidence,
            )

        except Exception as exc:
            log.error("stream_fatal_error", session_id=session_id, error=str(exc))
            yield f"event: error\ndata: {json.dumps({'code': 'internal_error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{report_id}")
async def get_report(report_id: str):
    return {"report_id": report_id, "status": ReportStatus.PENDING}
