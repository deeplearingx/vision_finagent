"""Regression tests: retrieval timeout/error fallback to cached evidence.

Covers:
- sync query: successful live retrieval (evidence_source=new_retrieval)
- sync query: timeout with cache → cached_fallback, answer prefixed with notice
- sync query: error with cache → cached_fallback
- sync query: timeout without cache → degraded, no fallback
- stream query: successful live retrieval
- stream query: timeout with cache → cached_fallback status event + notice chunk
- stream query: timeout without cache → degraded content chunk
- use_retrieval=False with cache → cached
- use_retrieval=False without cache → no_evidence degraded
- _build_pages_from_cache: missing images observability
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.schema import PageResult
from src.models.enums import DegradeReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query(client, question="test?", use_retrieval=True, session_id=None):
    body = {"question": question, "use_retrieval": use_retrieval}
    if session_id:
        body["session_id"] = session_id
    return client.post("/reports/query", json=body)


def _stream_events(client, question="test?", use_retrieval=True, session_id=None):
    """Collect all SSE events from /query/stream as list of (event, data) tuples."""
    body = {"question": question, "use_retrieval": use_retrieval}
    if session_id:
        body["session_id"] = session_id
    events = []
    with client.stream("POST", "/reports/query/stream", json=body) as resp:
        event_name = "message"
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
                events.append((event_name, data))
                event_name = "message"
    return events


def _seed_cache(fake_redis, session_id, evidence=None):
    if evidence is None:
        evidence = [{"report_id": "rpt1", "page_num": 1, "maxsim_score": 0.9}]
    payload = json.dumps({"cached_at": "2026-01-01T00:00:00+00:00", "evidence": evidence})
    asyncio.get_event_loop().run_until_complete(
        fake_redis.set(f"session:{session_id}:evidence", payload)
    )


_PAGES = [PageResult(report_id="rpt1", page_num=1, image_base64="img", maxsim_score=0.9)]


# ---------------------------------------------------------------------------
# Sync: live retrieval success
# ---------------------------------------------------------------------------
def test_sync_live_retrieval_success(client):
    with patch("src.routers.reports.retrieve", return_value=_PAGES), \
         patch("src.routers.reports.generate_answer", new_callable=AsyncMock,
               return_value=("答案", DegradeReason.NONE)):
        r = _query(client, session_id="s_live")
    body = r.json()
    assert r.status_code == 200
    assert body["evidence_source"] == "new_retrieval"
    assert body["degraded"] is False


# ---------------------------------------------------------------------------
# Sync: timeout WITH cache → cached_fallback
# ---------------------------------------------------------------------------
def test_sync_timeout_with_cache_fallback(client, fake_redis):
    sid = "s_timeout_cache"
    _seed_cache(fake_redis, sid)

    with patch("src.routers.reports.retrieve", side_effect=asyncio.TimeoutError()), \
         patch("src.routers.reports.generate_answer", new_callable=AsyncMock,
               return_value=("VLM答案", DegradeReason.NONE)):
        r = _query(client, session_id=sid)

    body = r.json()
    assert r.status_code == 200
    assert body["evidence_source"] == "cached_fallback"
    assert body["degraded"] is False
    assert "本次检索失败" in body["answer"]
    assert "自动复用上次证据" in body["answer"]


# ---------------------------------------------------------------------------
# Sync: error WITH cache → cached_fallback
# ---------------------------------------------------------------------------
def test_sync_error_with_cache_fallback(client, fake_redis):
    sid = "s_error_cache"
    _seed_cache(fake_redis, sid)

    with patch("src.routers.reports.retrieve", side_effect=RuntimeError("conn refused")), \
         patch("src.routers.reports.generate_answer", new_callable=AsyncMock,
               return_value=("VLM答案", DegradeReason.NONE)):
        r = _query(client, session_id=sid)

    body = r.json()
    assert body["evidence_source"] == "cached_fallback"
    assert body["degraded"] is False


# ---------------------------------------------------------------------------
# Sync: timeout WITHOUT cache → degraded, no fallback
# ---------------------------------------------------------------------------
def test_sync_timeout_no_cache_degraded(client):
    with patch("src.routers.reports.retrieve", side_effect=asyncio.TimeoutError()):
        r = _query(client, session_id="s_timeout_nocache")

    body = r.json()
    assert body["degraded"] is True
    assert body["degrade_reason"] == "retrieval_timeout"
    assert body["evidence_source"] == "new_retrieval"  # unchanged default


# ---------------------------------------------------------------------------
# Sync: use_retrieval=False WITH cache → cached
# ---------------------------------------------------------------------------
def test_sync_cache_reuse(client, fake_redis):
    sid = "s_cache_reuse"
    _seed_cache(fake_redis, sid)

    with patch("src.routers.reports.generate_answer", new_callable=AsyncMock,
               return_value=("答案", DegradeReason.NONE)):
        r = _query(client, use_retrieval=False, session_id=sid)

    body = r.json()
    assert body["evidence_source"] == "cached"
    assert body["degraded"] is False


# ---------------------------------------------------------------------------
# Sync: use_retrieval=False WITHOUT cache → no_evidence
# ---------------------------------------------------------------------------
def test_sync_no_cache_no_retrieval_degraded(client):
    r = _query(client, use_retrieval=False, session_id="s_empty")
    body = r.json()
    assert body["degraded"] is True
    assert body["degrade_reason"] == "no_evidence"


# ---------------------------------------------------------------------------
# Stream: live retrieval success
# ---------------------------------------------------------------------------
def test_stream_live_retrieval_success(client):
    async def _fake_stream(q, pages):
        yield "答"
        yield "案"

    with patch("src.routers.reports.retrieve", return_value=_PAGES), \
         patch("src.routers.reports.stream_generate_answer", side_effect=_fake_stream):
        events = _stream_events(client, session_id="ss_live")

    done = next((d for e, d in events if e == "done"), None)
    assert done is not None
    assert done["evidence_source"] == "new_retrieval"
    assert done["degraded"] is False


# ---------------------------------------------------------------------------
# Stream: timeout WITH cache → cached_fallback status + notice chunk
# ---------------------------------------------------------------------------
def test_stream_timeout_with_cache_fallback(client, fake_redis):
    sid = "ss_timeout_cache"
    _seed_cache(fake_redis, sid)

    async def _fake_stream(q, pages):
        yield "VLM内容"

    with patch("src.routers.reports.retrieve", side_effect=asyncio.TimeoutError()), \
         patch("src.routers.reports.stream_generate_answer", side_effect=_fake_stream):
        events = _stream_events(client, session_id=sid)

    done = next((d for e, d in events if e == "done"), None)
    assert done["evidence_source"] == "cached_fallback"
    assert done["degraded"] is False

    # status event must mention fallback
    status_msgs = [d["message"] for e, d in events if e == "status"]
    assert any("本次检索失败" in m for m in status_msgs)

    # content must include notice chunk
    content_chunks = "".join(d.get("chunk", "") for e, d in events if e == "content")
    assert "本次检索失败" in content_chunks


# ---------------------------------------------------------------------------
# Stream: timeout WITHOUT cache → degraded content chunk
# ---------------------------------------------------------------------------
def test_stream_timeout_no_cache_degraded(client):
    with patch("src.routers.reports.retrieve", side_effect=asyncio.TimeoutError()):
        events = _stream_events(client, session_id="ss_timeout_nocache")

    done = next((d for e, d in events if e == "done"), None)
    assert done["degraded"] is True
    assert done["degrade_reason"] == "retrieval_timeout"

    content = "".join(d.get("chunk", "") for e, d in events if e == "content")
    assert "无缓存证据" in content or "请稍后重试" in content


# ---------------------------------------------------------------------------
# _build_pages_from_cache: Milvus image fetch failure → image_fetch_incomplete
# ---------------------------------------------------------------------------
def test_sync_cache_fallback_image_fetch_incomplete(client, fake_redis, patch_milvus):
    sid = "s_img_fail"
    _seed_cache(fake_redis, sid)

    patch_milvus.query.side_effect = RuntimeError("milvus down")

    with patch("src.routers.reports.retrieve", side_effect=asyncio.TimeoutError()), \
         patch("src.routers.reports.generate_answer", new_callable=AsyncMock,
               return_value=("答案", DegradeReason.NONE)):
        r = _query(client, session_id=sid)

    body = r.json()
    assert body["evidence_source"] == "cached_fallback"
    assert body["image_fetch_incomplete"] is True


# ---------------------------------------------------------------------------
# _build_pages_from_cache: some images missing in Milvus → image_fetch_incomplete
# ---------------------------------------------------------------------------
def test_sync_cache_fallback_image_missing(client, fake_redis, patch_milvus):
    sid = "s_img_missing"
    _seed_cache(fake_redis, sid, evidence=[
        {"report_id": "rpt1", "page_num": 1, "maxsim_score": 0.9},
        {"report_id": "rpt1", "page_num": 2, "maxsim_score": 0.8},
    ])
    # Only return image for page 1, page 2 missing
    patch_milvus.query.return_value = [
        {"report_id": "rpt1", "page_num": 1, "image_base64": "img1"},
    ]

    with patch("src.routers.reports.retrieve", side_effect=asyncio.TimeoutError()), \
         patch("src.routers.reports.generate_answer", new_callable=AsyncMock,
               return_value=("答案", DegradeReason.NONE)):
        r = _query(client, session_id=sid)

    body = r.json()
    assert body["evidence_source"] == "cached_fallback"
    assert body["image_fetch_incomplete"] is True
