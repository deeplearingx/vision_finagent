"""Standalone test runner using unittest + asyncio (no pytest needed)."""
import sys, os, json, asyncio, unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fake Redis
# ---------------------------------------------------------------------------
class FakeRedis:
    def __init__(self):
        self._store: dict = {}
        self._zsets: dict = {}

    async def set(self, key, value, nx=False, ex=None, **kw):
        if nx and key in self._store:
            return None
        self._store[key] = value
        return True

    async def get(self, key):
        return self._store.get(key)

    async def delete(self, *keys):
        for k in keys:
            self._store.pop(k, None)

    async def expire(self, key, ttl):
        pass

    async def zadd(self, name, mapping):
        self._zsets.setdefault(name, {}).update(mapping)

    async def zrevrange(self, name, start, stop, withscores=False):
        items = sorted(self._zsets.get(name, {}).items(), key=lambda x: -x[1])
        sliced = items[start: stop + 1 if stop >= 0 else None]
        return [(k, v) for k, v in sliced] if withscores else [k for k, _ in sliced]

    async def ttl(self, key):
        return 3600

    async def scan(self, cursor, match="*", count=100):
        import fnmatch
        return 0, [k for k in self._store if fnmatch.fnmatch(k, match)]


def _make_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from src.routers import reports, health
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(health.router)
    app.include_router(reports.router)
    return app


def _client(app):
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


def _pdf(valid=True):
    return b"%PDF-1.4 " + b"x" * 100 + b"%%EOF" if valid else b"not a pdf"

def _png(valid=True):
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 20 if valid else b"FAKEPNG"

def _jpg(valid=True):
    return b"\xff\xd8\xff\xe0" + b"\x00" * 20 if valid else b"FAKEJPG"


def _upload(client, data, filename, report_id="rpt_test"):
    import io
    return client.post(
        "/reports/upload",
        data={"report_id": report_id},
        files={"file": (filename, io.BytesIO(data), "application/octet-stream")},
    )


# ---------------------------------------------------------------------------
# Base test case with common patches
# ---------------------------------------------------------------------------
class BaseCase(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()

        async def _get_redis():
            return self.redis

        self.mock_milvus = MagicMock()
        self.mock_milvus.has_collection.return_value = False
        self.mock_milvus.describe_collection.return_value = {"fields": []}
        self.mock_milvus.list_indexes.return_value = []
        self.mock_milvus.query.return_value = []

        self._patches = [
            patch("src.core.redis_client.get_redis", _get_redis),
            patch("src.routers.reports.get_redis", _get_redis),
            patch("src.utils.idempotency.get_redis", _get_redis),
            patch("src.services.task_service.get_redis", _get_redis),
            patch("src.core.milvus_client.get_client", return_value=self.mock_milvus),
            patch("src.core.milvus_client._client", self.mock_milvus),
            patch("src.routers.reports.submit_ingest_task",
                  new_callable=AsyncMock, return_value="task_test123"),
        ]
        for p in self._patches:
            p.start()

        self.app = _make_app()
        self.client = _client(self.app)

    def tearDown(self):
        for p in self._patches:
            p.stop()


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------
class TestUpload(BaseCase):
    def test_valid_pdf_accepted(self):
        r = _upload(self.client, _pdf(True), "report.pdf")
        self.assertEqual(r.status_code, 202)
        self.assertEqual(r.json()["task_id"], "task_test123")

    def test_valid_png_accepted(self):
        r = _upload(self.client, _png(True), "chart.png", "rpt_png")
        self.assertEqual(r.status_code, 202)

    def test_valid_jpg_accepted(self):
        r = _upload(self.client, _jpg(True), "chart.jpg", "rpt_jpg")
        self.assertEqual(r.status_code, 202)

    def test_invalid_pdf_rejected(self):
        r = _upload(self.client, _pdf(False), "bad.pdf")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Invalid PDF", r.json()["detail"])

    def test_invalid_png_rejected(self):
        r = _upload(self.client, _png(False), "bad.png", "rpt_badpng")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Invalid image", r.json()["detail"])

    def test_invalid_jpg_rejected(self):
        r = _upload(self.client, _jpg(False), "bad.jpg", "rpt_badjpg")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Invalid image", r.json()["detail"])

    def test_unsupported_extension_rejected(self):
        r = _upload(self.client, b"data", "report.docx", "rpt_docx")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Unsupported file type", r.json()["detail"])

    def test_upload_blocked_during_maintenance(self):
        with patch("src.routers.reports._maintenance_active", True):
            r = _upload(self.client, _pdf(True), "report.pdf")
        self.assertEqual(r.status_code, 503)
        self.assertIn("维护", r.json()["detail"])

    def test_duplicate_upload_rejected(self):
        asyncio.get_event_loop().run_until_complete(
            self.redis.set("upload:rpt_dup", "1", nx=True)
        )
        r = _upload(self.client, _pdf(True), "report.pdf", "rpt_dup")
        self.assertEqual(r.status_code, 409)


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------
class TestQuery(BaseCase):
    def _query(self, use_retrieval=True, session_id=None, question="test?"):
        body = {"question": question, "use_retrieval": use_retrieval}
        if session_id:
            body["session_id"] = session_id
        return self.client.post("/reports/query", json=body)

    def test_query_blocked_during_maintenance(self):
        with patch("src.routers.reports._maintenance_active", True):
            r = self._query()
        self.assertEqual(r.status_code, 503)
        self.assertIn("维护", r.json()["detail"])

    def test_no_evidence_retrieval_off_degrades(self):
        r = self._query(use_retrieval=False, session_id="sess_empty")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["degraded"])
        self.assertEqual(body["degrade_reason"], "no_evidence")
        self.assertEqual(body["evidence_source"], "none")
        self.assertIsNotNone(body["answer"])

    def test_cache_reuse_structure_stable(self):
        sid = "sess_cached"
        cache = {
            "cached_at": "2026-01-01T00:00:00+00:00",
            "evidence": [{"report_id": "rpt1", "page_num": 1, "maxsim_score": 0.9}],
        }
        asyncio.get_event_loop().run_until_complete(
            self.redis.set(f"session:{sid}:evidence", json.dumps(cache))
        )
        from src.models.enums import DegradeReason
        with patch("src.routers.reports.generate_answer",
                   new_callable=AsyncMock, return_value=("答案", DegradeReason.NONE)):
            r = self._query(use_retrieval=False, session_id=sid)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["evidence_source"], "cached")
        self.assertIn("image_fetch_incomplete", body)
        self.assertIsInstance(body["image_fetch_incomplete"], bool)
        self.assertEqual(len(body["evidence"]), 1)

    def test_retrieval_error_degrades(self):
        with patch("src.routers.reports.retrieve", side_effect=Exception("boom")):
            r = self._query(use_retrieval=True)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["degraded"])
        self.assertIn(body["degrade_reason"], ("retrieval_timeout", "retrieval_error"))

    def test_new_retrieval_evidence_source(self):
        from src.models.schema import PageResult
        from src.models.enums import DegradeReason
        pages = [PageResult(report_id="rpt1", page_num=1, image_base64="abc", maxsim_score=0.85)]
        with patch("src.routers.reports.retrieve", return_value=pages), \
             patch("src.routers.reports.generate_answer",
                   new_callable=AsyncMock, return_value=("答案", DegradeReason.NONE)):
            r = self._query(use_retrieval=True, session_id="sess_new")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["evidence_source"], "new_retrieval")
        self.assertFalse(body["degraded"])
        self.assertEqual(len(body["evidence"]), 1)

    # --- fallback tests ---

    def _seed_cache(self, sid, evidence=None):
        if evidence is None:
            evidence = [{"report_id": "rpt1", "page_num": 1, "maxsim_score": 0.9}]
        payload = json.dumps({"cached_at": "2026-01-01T00:00:00+00:00", "evidence": evidence})
        asyncio.get_event_loop().run_until_complete(
            self.redis.set(f"session:{sid}:evidence", payload)
        )

    def test_timeout_with_cache_fallback(self):
        """Timeout + cache → cached_fallback, not degraded, answer has notice."""
        from src.models.enums import DegradeReason
        sid = "s_timeout_cache"
        self._seed_cache(sid)
        with patch("src.routers.reports.retrieve", side_effect=asyncio.TimeoutError()), \
             patch("src.routers.reports.generate_answer",
                   new_callable=AsyncMock, return_value=("VLM答案", DegradeReason.NONE)):
            r = self._query(session_id=sid)
        body = r.json()
        self.assertEqual(body["evidence_source"], "cached_fallback")
        self.assertFalse(body["degraded"])
        self.assertIn("本次检索失败", body["answer"])
        self.assertIn("自动复用上次证据", body["answer"])

    def test_error_with_cache_fallback(self):
        """Retrieval error + cache → cached_fallback, not degraded."""
        from src.models.enums import DegradeReason
        sid = "s_error_cache"
        self._seed_cache(sid)
        with patch("src.routers.reports.retrieve", side_effect=RuntimeError("conn refused")), \
             patch("src.routers.reports.generate_answer",
                   new_callable=AsyncMock, return_value=("VLM答案", DegradeReason.NONE)):
            r = self._query(session_id=sid)
        body = r.json()
        self.assertEqual(body["evidence_source"], "cached_fallback")
        self.assertFalse(body["degraded"])

    def test_timeout_no_cache_degraded(self):
        """Timeout + no cache → degraded=True, degrade_reason=retrieval_timeout."""
        with patch("src.routers.reports.retrieve", side_effect=asyncio.TimeoutError()):
            r = self._query(session_id="s_timeout_nocache")
        body = r.json()
        self.assertTrue(body["degraded"])
        self.assertEqual(body["degrade_reason"], "retrieval_timeout")

    def test_cache_fallback_image_fetch_incomplete(self):
        """Fallback + Milvus image fetch failure → image_fetch_incomplete=True."""
        from src.models.enums import DegradeReason
        sid = "s_img_fail"
        self._seed_cache(sid)
        self.mock_milvus.query.side_effect = RuntimeError("milvus down")
        with patch("src.routers.reports.retrieve", side_effect=asyncio.TimeoutError()), \
             patch("src.routers.reports.generate_answer",
                   new_callable=AsyncMock, return_value=("答案", DegradeReason.NONE)):
            r = self._query(session_id=sid)
        body = r.json()
        self.assertEqual(body["evidence_source"], "cached_fallback")
        self.assertTrue(body["image_fetch_incomplete"])
        self.mock_milvus.query.side_effect = None  # reset

    def test_cache_fallback_image_missing(self):
        """Fallback + some images missing in Milvus → image_fetch_incomplete=True."""
        from src.models.enums import DegradeReason
        sid = "s_img_missing"
        self._seed_cache(sid, evidence=[
            {"report_id": "rpt1", "page_num": 1, "maxsim_score": 0.9},
            {"report_id": "rpt1", "page_num": 2, "maxsim_score": 0.8},
        ])
        self.mock_milvus.query.return_value = [
            {"report_id": "rpt1", "page_num": 1, "image_base64": "img1"},
        ]
        with patch("src.routers.reports.retrieve", side_effect=asyncio.TimeoutError()), \
             patch("src.routers.reports.generate_answer",
                   new_callable=AsyncMock, return_value=("答案", DegradeReason.NONE)):
            r = self._query(session_id=sid)
        body = r.json()
        self.assertEqual(body["evidence_source"], "cached_fallback")
        self.assertTrue(body["image_fetch_incomplete"])
        self.mock_milvus.query.return_value = []  # reset


# ---------------------------------------------------------------------------
# Sessions tests
# ---------------------------------------------------------------------------
class TestSessions(BaseCase):
    def _seed(self, sid, meta=None, history=None, evidence=None):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.redis.zadd("session_index", {sid: 1700000000.0}))
        if history:
            loop.run_until_complete(self.redis.set(f"session:{sid}", json.dumps(history)))
        if meta:
            loop.run_until_complete(self.redis.set(f"session:{sid}:meta", json.dumps(meta)))
        if evidence:
            loop.run_until_complete(self.redis.set(f"session:{sid}:evidence", json.dumps(evidence)))

    def test_backfills_legacy_session(self):
        sid = "legacy_sess"
        self._seed(sid, history=[
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧答案"},
        ])
        r = self.client.get("/reports/sessions")
        self.assertEqual(r.status_code, 200)
        found = next((s for s in r.json()["sessions"] if s["session_id"] == sid), None)
        self.assertIsNotNone(found)
        self.assertEqual(found["last_question"], "旧问题")
        self.assertEqual(found["turn_count"], 1)

    def test_has_evidence_true(self):
        sid = "sess_ev"
        self._seed(sid, meta={
            "session_id": sid, "updated_at": "2026-01-01T00:00:00+00:00",
            "updated_ts": 1700000000.0, "last_question": "q", "turn_count": 1,
            "has_evidence": True,
        })
        r = self.client.get("/reports/sessions")
        found = next(s for s in r.json()["sessions"] if s["session_id"] == sid)
        self.assertTrue(found["has_evidence"])

    def test_has_evidence_false(self):
        sid = "sess_no_ev"
        self._seed(sid, meta={
            "session_id": sid, "updated_at": "2026-01-01T00:00:00+00:00",
            "updated_ts": 1700000000.0, "last_question": "q", "turn_count": 1,
            "has_evidence": False,
        })
        r = self.client.get("/reports/sessions")
        found = next(s for s in r.json()["sessions"] if s["session_id"] == sid)
        self.assertFalse(found["has_evidence"])

    def test_evidence_status_with_cache(self):
        sid = "sess_ev_status"
        self._seed(sid, evidence={
            "cached_at": "2026-01-01T00:00:00+00:00",
            "evidence": [{"report_id": "r1", "page_num": 1, "maxsim_score": 0.8}],
        })
        r = self.client.get(f"/reports/sessions/{sid}/evidence-status")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["has_evidence"])

    def test_evidence_status_without_cache(self):
        r = self.client.get("/reports/sessions/nonexistent/evidence-status")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["has_evidence"])


# ---------------------------------------------------------------------------
# Milvus schema health tests
# ---------------------------------------------------------------------------
class TestMilvusSchema(unittest.TestCase):
    def _make_client(self, patches_fields, pages_fields):
        mc = MagicMock()
        mc.has_collection.return_value = True
        def _desc(collection_name):
            fields = pages_fields if collection_name.endswith("_pages") else patches_fields
            return {"fields": [{"name": f} for f in fields]}
        mc.describe_collection.side_effect = _desc
        return mc

    def test_no_drift(self):
        from src.core.milvus_client import check_schema_health
        mc = self._make_client(
            ["pk", "report_id", "page_num", "colpali_embeddings"],
            ["page_id", "report_id", "page_num", "image_base64"],
        )
        with patch("src.core.milvus_client.get_client", return_value=mc):
            results = check_schema_health()
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.get("missing", []), [])
            self.assertEqual(r.get("extra", []), [])

    def test_missing_field_detected(self):
        from src.core.milvus_client import check_schema_health
        mc = self._make_client(
            ["pk", "report_id", "page_num"],  # missing colpali_embeddings
            ["page_id", "report_id", "page_num", "image_base64"],
        )
        with patch("src.core.milvus_client.get_client", return_value=mc):
            results = check_schema_health()
        patches_r = next(r for r in results if not r["collection"].endswith("_pages"))
        self.assertIn("colpali_embeddings", patches_r["missing"])

    def test_extra_field_detected(self):
        from src.core.milvus_client import check_schema_health
        mc = self._make_client(
            ["pk", "report_id", "page_num", "colpali_embeddings", "unexpected"],
            ["page_id", "report_id", "page_num", "image_base64"],
        )
        with patch("src.core.milvus_client.get_client", return_value=mc):
            results = check_schema_health()
        patches_r = next(r for r in results if not r["collection"].endswith("_pages"))
        self.assertIn("unexpected", patches_r["extra"])

    def test_collection_absent(self):
        from src.core.milvus_client import check_schema_health
        mc = MagicMock()
        mc.has_collection.return_value = False
        with patch("src.core.milvus_client.get_client", return_value=mc):
            results = check_schema_health()
        for r in results:
            self.assertEqual(r.get("note"), "collection_absent")
            self.assertTrue(len(r["missing"]) > 0)

    def test_return_structure(self):
        from src.core.milvus_client import check_schema_health
        mc = self._make_client(
            ["pk", "report_id", "page_num", "colpali_embeddings"],
            ["page_id", "report_id", "page_num", "image_base64"],
        )
        with patch("src.core.milvus_client.get_client", return_value=mc):
            results = check_schema_health()
        for r in results:
            self.assertIn("collection", r)
            self.assertIn("missing", r)
            self.assertIn("extra", r)


# ---------------------------------------------------------------------------
# Idempotency tests
# ---------------------------------------------------------------------------
class TestIdempotency(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()
        async def _get():
            return self.redis
        self._patches = [
            patch("src.core.redis_client.get_redis", _get),
            patch("src.utils.idempotency.get_redis", _get),
            patch("src.services.task_service.get_redis", _get),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_first_call_true(self):
        from src.utils.idempotency import check_and_set
        self.assertTrue(self._run(check_and_set("upload:new")))

    def test_duplicate_false(self):
        from src.utils.idempotency import check_and_set
        self._run(check_and_set("upload:dup"))
        self.assertFalse(self._run(check_and_set("upload:dup")))

    def test_release_allows_retry(self):
        from src.utils.idempotency import check_and_set, release
        self._run(check_and_set("upload:retry"))
        self.assertFalse(self._run(check_and_set("upload:retry")))
        self._run(release("upload:retry"))
        self.assertTrue(self._run(check_and_set("upload:retry")))

    def test_failed_task_releases_token(self):
        from src.utils.idempotency import check_and_set
        from src.services import task_service

        report_id = "rpt_fail"
        self._run(self.redis.set(f"upload:{report_id}", "1", nx=True))

        async def _run_test():
            with patch("src.services.task_service.ingest_report",
                       new_callable=AsyncMock, side_effect=RuntimeError("boom")):
                await task_service.submit_ingest_task(report_id, "/tmp/fake.pdf")
                await asyncio.sleep(0.15)

        self._run(_run_test())
        can_retry = self._run(check_and_set(f"upload:{report_id}"))
        self.assertTrue(can_retry, "token must be released after task failure")

    def test_success_task_keeps_token(self):
        from src.utils.idempotency import check_and_set
        from src.services import task_service

        report_id = "rpt_ok"
        self._run(self.redis.set(f"upload:{report_id}", "1", nx=True))

        async def _run_test():
            with patch("src.services.task_service.ingest_report",
                       new_callable=AsyncMock, return_value=None):
                await task_service.submit_ingest_task(report_id, "/tmp/fake.pdf")
                await asyncio.sleep(0.15)

        self._run(_run_test())
        can_resubmit = self._run(check_and_set(f"upload:{report_id}"))
        self.assertFalse(can_resubmit, "token must be kept after task success")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestUpload, TestQuery, TestSessions, TestMilvusSchema, TestIdempotency]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
