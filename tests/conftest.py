"""Shared fixtures: fake Redis, patched model loading, FastAPI test client."""
import sys
import types
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def _inject_fake_colpali():
    """Inject stub colpali_engine modules before any src.* import triggers the real one."""
    if "colpali_engine" in sys.modules:
        return
    pkg = types.ModuleType("colpali_engine")
    models = types.ModuleType("colpali_engine.models")
    pali = types.ModuleType("colpali_engine.models.paligemma")
    cp = types.ModuleType("colpali_engine.models.paligemma.colpali")
    mcp = types.ModuleType("colpali_engine.models.paligemma.colpali.modeling_colpali")
    pcp = types.ModuleType("colpali_engine.models.paligemma.colpali.processing_colpali")

    class _FakeColPali:
        pass

    class _FakeProcessor:
        pass

    mcp.ColPali = _FakeColPali
    pcp.ColPaliProcessor = _FakeProcessor

    for name, mod in [
        ("colpali_engine", pkg),
        ("colpali_engine.models", models),
        ("colpali_engine.models.paligemma", pali),
        ("colpali_engine.models.paligemma.colpali", cp),
        ("colpali_engine.models.paligemma.colpali.modeling_colpali", mcp),
        ("colpali_engine.models.paligemma.colpali.processing_colpali", pcp),
    ]:
        sys.modules[name] = mod


_inject_fake_colpali()


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
        if withscores:
            return [(k, v) for k, v in sliced]
        return [k for k, _ in sliced]

    async def ttl(self, key):
        return 3600

    async def scan(self, cursor, match="*", count=100):
        import fnmatch
        matched = [k for k in self._store if fnmatch.fnmatch(k, match)]
        return 0, matched


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture(autouse=True)
def patch_redis(fake_redis):
    async def _get():
        return fake_redis

    with patch("src.core.redis_client.get_redis", _get), \
         patch("src.routers.reports.get_redis", _get), \
         patch("src.utils.idempotency.get_redis", _get), \
         patch("src.services.task_service.get_redis", _get):
        yield fake_redis


@pytest.fixture(autouse=True)
def patch_milvus():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = False
    mock_client.describe_collection.return_value = {"fields": []}
    mock_client.list_indexes.return_value = []
    mock_client.query.return_value = []

    with patch("src.core.milvus_client.get_client", return_value=mock_client), \
         patch("src.core.milvus_client._client", mock_client):
        yield mock_client


@pytest.fixture(autouse=True)
def patch_retrieval_model():
    """Prevent actual ColPali model loading during tests."""
    with patch("src.services.retrieval_service._load_model"), \
         patch("src.services.retrieval_service.warmup_retrieval_model"):
        yield


@pytest.fixture(autouse=True)
def patch_submit_task():
    with patch(
        "src.routers.reports.submit_ingest_task",
        new_callable=AsyncMock,
        return_value="task_test123",
    ):
        yield


@pytest.fixture(scope="session")
def app():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from src.routers import reports, health

    _app = FastAPI()
    _app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    _app.include_router(health.router)
    _app.include_router(reports.router)
    return _app


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
