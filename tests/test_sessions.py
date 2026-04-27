"""Regression tests: /sessions list & evidence-status — backfill & has_evidence semantics."""
import json
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _seed_session(fake_redis, sid, meta=None, history=None, evidence=None):
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(fake_redis.zadd("session_index", {sid: 1700000000.0}))
    if history is not None:
        loop.run_until_complete(
            fake_redis.set(f"session:{sid}", json.dumps(history))
        )
    if meta is not None:
        loop.run_until_complete(
            fake_redis.set(f"session:{sid}:meta", json.dumps(meta))
        )
    if evidence is not None:
        loop.run_until_complete(
            fake_redis.set(f"session:{sid}:evidence", json.dumps(evidence))
        )


# ---------------------------------------------------------------------------
# Legacy session (no meta) → backfill
# ---------------------------------------------------------------------------
def test_list_sessions_backfills_legacy_session(client, fake_redis):
    sid = "legacy_sess"
    _seed_session(
        fake_redis, sid,
        history=[
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧答案"},
        ],
    )
    r = client.get("/reports/sessions")
    assert r.status_code == 200
    sessions = r.json()["sessions"]
    found = next((s for s in sessions if s["session_id"] == sid), None)
    assert found is not None, "legacy session should appear after backfill"
    assert found["last_question"] == "旧问题"
    assert found["turn_count"] == 1


# ---------------------------------------------------------------------------
# Session with meta → has_evidence reflects stored value
# ---------------------------------------------------------------------------
def test_list_sessions_has_evidence_true(client, fake_redis):
    sid = "sess_with_ev"
    _seed_session(
        fake_redis, sid,
        meta={
            "session_id": sid,
            "updated_at": "2026-01-01T00:00:00+00:00",
            "updated_ts": 1700000000.0,
            "last_question": "有证据的问题",
            "turn_count": 2,
            "has_evidence": True,
        },
    )
    r = client.get("/reports/sessions")
    assert r.status_code == 200
    found = next(s for s in r.json()["sessions"] if s["session_id"] == sid)
    assert found["has_evidence"] is True


def test_list_sessions_has_evidence_false(client, fake_redis):
    sid = "sess_no_ev"
    _seed_session(
        fake_redis, sid,
        meta={
            "session_id": sid,
            "updated_at": "2026-01-01T00:00:00+00:00",
            "updated_ts": 1700000000.0,
            "last_question": "无证据问题",
            "turn_count": 1,
            "has_evidence": False,
        },
    )
    r = client.get("/reports/sessions")
    assert r.status_code == 200
    found = next(s for s in r.json()["sessions"] if s["session_id"] == sid)
    assert found["has_evidence"] is False


# ---------------------------------------------------------------------------
# evidence-status endpoint
# ---------------------------------------------------------------------------
def test_evidence_status_with_cache(client, fake_redis):
    sid = "sess_ev_status"
    _seed_session(
        fake_redis, sid,
        evidence={
            "cached_at": "2026-01-01T00:00:00+00:00",
            "evidence": [{"report_id": "r1", "page_num": 1, "maxsim_score": 0.8}],
        },
    )
    r = client.get(f"/reports/sessions/{sid}/evidence-status")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["has_evidence"] is True


def test_evidence_status_without_cache(client):
    r = client.get("/reports/sessions/nonexistent_sess/evidence-status")
    assert r.status_code == 200
    assert r.json()["has_evidence"] is False
