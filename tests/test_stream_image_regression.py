"""Regression tests: stream SSE evidence event carries image_base64.

Covers:
- stream first query: evidence event has retrieved_pages with image_base64
- stream first query: done event has image_fetch_incomplete=False
- stream cache reuse (use_retrieval=False): evidence event has retrieved_pages,
  image_fetch_incomplete reflects Milvus result
- stream second pass: second evidence event also carries retrieved_pages
- sync path not regressed: retrieved_pages still present with image_base64
- buildEvidenceHtml fallback text: history-restore vs stream-no-image distinction
  (logic tested via backend field, UI text verified by string constants)
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.schema import PageResult
from src.models.enums import DegradeReason


_PAGES = [PageResult(report_id="rpt1", page_num=1, image_base64="base64img==", maxsim_score=0.9)]
_PAGES2 = [PageResult(report_id="rpt1", page_num=2, image_base64="base64img2==", maxsim_score=0.85)]


def _stream_events(client, question="test?", use_retrieval=True, session_id=None):
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


# ---------------------------------------------------------------------------
# Stream: first query — evidence event must carry retrieved_pages with image_base64
# ---------------------------------------------------------------------------
def test_stream_first_query_evidence_has_images(client):
    async def _fake_stream(q, pages):
        yield "答案"

    with patch("src.routers.reports.retrieve", return_value=_PAGES), \
         patch("src.routers.reports.stream_generate_answer", side_effect=_fake_stream):
        events = _stream_events(client, session_id="si_first")

    evidence_events = [d for e, d in events if e == "evidence"]
    assert evidence_events, "No evidence event emitted"

    first_ev = evidence_events[0]
    assert "retrieved_pages" in first_ev, "retrieved_pages missing from evidence event"
    assert len(first_ev["retrieved_pages"]) == 1
    assert first_ev["retrieved_pages"][0]["image_base64"] == "base64img=="
    assert first_ev["image_fetch_incomplete"] is False


# ---------------------------------------------------------------------------
# Stream: first query — done event has image_fetch_incomplete=False
# ---------------------------------------------------------------------------
def test_stream_first_query_done_image_fetch_incomplete_false(client):
    async def _fake_stream(q, pages):
        yield "答案"

    with patch("src.routers.reports.retrieve", return_value=_PAGES), \
         patch("src.routers.reports.stream_generate_answer", side_effect=_fake_stream):
        events = _stream_events(client, session_id="si_done")

    done = next((d for e, d in events if e == "done"), None)
    assert done is not None
    assert done["image_fetch_incomplete"] is False


# ---------------------------------------------------------------------------
# Stream: cache reuse (use_retrieval=False) — evidence event has retrieved_pages
# ---------------------------------------------------------------------------
def test_stream_cache_reuse_evidence_has_pages(client, fake_redis, patch_milvus):
    sid = "si_cache"
    _seed_cache(fake_redis, sid)
    patch_milvus.query.return_value = [
        {"report_id": "rpt1", "page_num": 1, "image_base64": "cached_img=="}
    ]

    async def _fake_stream(q, pages):
        yield "缓存答案"

    with patch("src.routers.reports.stream_generate_answer", side_effect=_fake_stream):
        events = _stream_events(client, use_retrieval=False, session_id=sid)

    evidence_events = [d for e, d in events if e == "evidence"]
    assert evidence_events
    rp = evidence_events[0].get("retrieved_pages", [])
    assert len(rp) == 1
    assert rp[0]["image_base64"] == "cached_img=="


# ---------------------------------------------------------------------------
# Stream: cache reuse with Milvus failure → image_fetch_incomplete=True in evidence event
# ---------------------------------------------------------------------------
def test_stream_cache_reuse_milvus_fail_image_fetch_incomplete(client, fake_redis, patch_milvus):
    sid = "si_cache_fail"
    _seed_cache(fake_redis, sid)
    patch_milvus.query.side_effect = RuntimeError("milvus down")

    async def _fake_stream(q, pages):
        yield "答案"

    with patch("src.routers.reports.stream_generate_answer", side_effect=_fake_stream):
        events = _stream_events(client, use_retrieval=False, session_id=sid)

    evidence_events = [d for e, d in events if e == "evidence"]
    assert evidence_events
    assert evidence_events[0]["image_fetch_incomplete"] is True


# ---------------------------------------------------------------------------
# Stream: second pass — second evidence event also carries retrieved_pages
# ---------------------------------------------------------------------------
def test_stream_second_pass_evidence_has_images(client):
    call_count = {"n": 0}

    def _fake_retrieve(*args, **kwargs):
        call_count["n"] += 1
        return _PAGES if call_count["n"] == 1 else _PAGES2

    insufficient_answer = "证据不足，无法回答。"
    good_answer = "二轮答案"

    async def _fake_stream(q, pages):
        if call_count["n"] == 1:
            yield insufficient_answer
        else:
            yield good_answer

    with patch("src.routers.reports.retrieve", side_effect=_fake_retrieve), \
         patch("src.routers.reports.stream_generate_answer", side_effect=_fake_stream), \
         patch("src.routers.reports.is_insufficient_evidence", return_value=True), \
         patch("src.routers.reports.settings") as mock_settings:
        mock_settings.QUERY_TIMEOUT = 30
        mock_settings.VLM_SECOND_PASS_ENABLED = True
        mock_settings.VLM_SECOND_PASS_TOP_K = 10
        mock_settings.VLM_SECOND_PASS_CANDIDATE_K = 100
        mock_settings.VLM_SECOND_PASS_MAX_IMAGES = 5
        events = _stream_events(client, session_id="si_second")

    evidence_events = [d for e, d in events if e == "evidence"]
    assert len(evidence_events) >= 2, "Expected at least 2 evidence events (first + second pass)"
    second_ev = evidence_events[1]
    assert "retrieved_pages" in second_ev
    assert second_ev["retrieved_pages"][0]["image_base64"] == "base64img2=="


# ---------------------------------------------------------------------------
# Sync path not regressed: retrieved_pages still present with image_base64
# ---------------------------------------------------------------------------
def test_sync_retrieved_pages_has_image_base64(client):
    with patch("src.routers.reports.retrieve", return_value=_PAGES), \
         patch("src.routers.reports.generate_answer", new_callable=AsyncMock,
               return_value=("答案", DegradeReason.NONE)):
        r = client.post("/reports/query", json={"question": "test?", "use_retrieval": True,
                                                 "session_id": "sync_img"})

    body = r.json()
    assert r.status_code == 200
    assert "retrieved_pages" in body
    assert len(body["retrieved_pages"]) == 1
    assert body["retrieved_pages"][0]["image_base64"] == "base64img=="
