"""Regression tests: /reports/query — degradation, cache reuse, maintenance gate."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _query(client, question="test?", use_retrieval=True, session_id=None, refresh=False):
    body = {
        "question": question,
        "use_retrieval": use_retrieval,
        "refresh_retrieval": refresh,
    }
    if session_id:
        body["session_id"] = session_id
    return client.post("/reports/query", json=body)


# ---------------------------------------------------------------------------
# Maintenance gate
# ---------------------------------------------------------------------------
def test_query_blocked_during_maintenance(client):
    with patch("src.routers.reports._maintenance_active", True):
        r = _query(client)
    assert r.status_code == 503
    assert "维护" in r.json()["detail"]


# ---------------------------------------------------------------------------
# No evidence + retrieval disabled → explicit degradation
# ---------------------------------------------------------------------------
def test_query_no_evidence_retrieval_off_degrades(client):
    """use_retrieval=False with empty cache must return NO_EVIDENCE degradation."""
    r = _query(client, use_retrieval=False, session_id="sess_empty")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["degrade_reason"] == "no_evidence"
    assert body["evidence_source"] == "none"
    assert body["answer"] is not None  # must contain a human-readable message


# ---------------------------------------------------------------------------
# Cache reuse path: structure stability
# ---------------------------------------------------------------------------
def test_query_cache_reuse_structure_stable(client, fake_redis):
    """When evidence cache exists and use_retrieval=False, response must contain
    evidence_source='cached' and image_fetch_incomplete field."""
    sid = "sess_cached"
    cache = {
        "cached_at": "2026-01-01T00:00:00+00:00",
        "evidence": [
            {"report_id": "rpt1", "page_num": 1, "maxsim_score": 0.9},
        ],
    }
    import asyncio
    asyncio.get_event_loop().run_until_complete(
        fake_redis.set(f"session:{sid}:evidence", json.dumps(cache))
    )

    # Milvus image query returns empty (simulates missing images)
    with patch("src.routers.reports.generate_answer", new_callable=AsyncMock) as mock_vlm:
        from src.models.enums import DegradeReason
        mock_vlm.return_value = ("答案文本", DegradeReason.NONE)

        r = _query(client, use_retrieval=False, session_id=sid)

    assert r.status_code == 200
    body = r.json()
    assert body["evidence_source"] == "cached"
    assert "image_fetch_incomplete" in body
    assert isinstance(body["image_fetch_incomplete"], bool)
    assert len(body["evidence"]) == 1
    assert body["evidence"][0]["report_id"] == "rpt1"


# ---------------------------------------------------------------------------
# Retrieval timeout → degraded with reason
# ---------------------------------------------------------------------------
def test_query_retrieval_timeout_degrades(client):
    import asyncio

    async def _slow(*a, **kw):
        raise asyncio.TimeoutError()

    with patch("src.routers.reports.retrieve", side_effect=Exception("timeout sim")):
        r = _query(client, use_retrieval=True)

    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["degrade_reason"] in ("retrieval_timeout", "retrieval_error")


# ---------------------------------------------------------------------------
# Successful retrieval → evidence_source = new_retrieval
# ---------------------------------------------------------------------------
def test_query_new_retrieval_evidence_source(client):
    from src.models.schema import PageResult
    from src.models.enums import DegradeReason

    pages = [PageResult(report_id="rpt1", page_num=1, image_base64="abc", maxsim_score=0.85)]

    with patch("src.routers.reports.retrieve", return_value=pages), \
         patch("src.routers.reports.generate_answer", new_callable=AsyncMock,
               return_value=("答案", DegradeReason.NONE)):
        r = _query(client, use_retrieval=True, session_id="sess_new")

    assert r.status_code == 200
    body = r.json()
    assert body["evidence_source"] == "new_retrieval"
    assert body["degraded"] is False
    assert len(body["evidence"]) == 1
