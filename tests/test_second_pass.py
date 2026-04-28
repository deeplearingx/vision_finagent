"""Tests for conservative two-pass VLM evidence expansion."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.vlm_service import is_insufficient_evidence, INSUFFICIENT_EVIDENCE_PHRASE
from src.models.enums import DegradeReason


# ---------------------------------------------------------------------------
# 1. is_insufficient_evidence() unit tests
# ---------------------------------------------------------------------------

def test_insufficient_evidence_exact():
    assert is_insufficient_evidence(INSUFFICIENT_EVIDENCE_PHRASE) is True


def test_insufficient_evidence_case_insensitive():
    assert is_insufficient_evidence("insufficient evidence in provided pages") is True


def test_insufficient_evidence_embedded():
    assert is_insufficient_evidence(
        f"Based on the documents: {INSUFFICIENT_EVIDENCE_PHRASE}."
    ) is True


def test_insufficient_evidence_normal_answer():
    assert is_insufficient_evidence("净利润为 12.3 亿元。") is False


def test_insufficient_evidence_none():
    assert is_insufficient_evidence(None) is False


def test_insufficient_evidence_empty():
    assert is_insufficient_evidence("") is False


def test_insufficient_evidence_vlm_error_text():
    # Errors / timeouts must NOT trigger second pass
    assert is_insufficient_evidence("VLM 生成失败（vlm_timeout）") is False


# ---------------------------------------------------------------------------
# 2. Config defaults
# ---------------------------------------------------------------------------

def test_second_pass_config_defaults():
    from src.config import settings
    assert settings.VLM_SECOND_PASS_ENABLED is True
    assert settings.VLM_SECOND_PASS_TOP_K > 0
    assert settings.VLM_SECOND_PASS_CANDIDATE_K > 0
    assert settings.VLM_SECOND_PASS_MAX_IMAGES > 0
    # Second pass should expand beyond first pass defaults
    assert settings.VLM_SECOND_PASS_TOP_K >= settings.MAX_VLM_IMAGES


# ---------------------------------------------------------------------------
# 3. /reports/query control-flow: second pass triggered only on sentinel phrase
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_mocks(client):
    return client


def _page_payload(use_retrieval=True):
    return {
        "question": "净利润是多少？",
        "session_id": "test-sess-001",
        "use_retrieval": use_retrieval,
        "refresh_retrieval": False,
        "target_companies": [],
        "top_k": 3,
        "candidate_k": 20,
    }


def _fake_pages():
    from src.models.schema import PageResult
    return [PageResult(report_id="r1", page_num=1, image_base64="", maxsim_score=0.9)]


@pytest.mark.asyncio
async def test_second_pass_not_triggered_on_normal_answer(client):
    """When first-pass returns a normal answer, second pass must NOT run."""
    pages = _fake_pages()
    with patch("src.routers.reports.retrieve", return_value=pages) as mock_retrieve, \
         patch("src.routers.reports.generate_answer",
               new=AsyncMock(return_value=("净利润 12 亿", DegradeReason.NONE))):
        resp = client.post("/reports/query", json=_page_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["vlm_passes"] == 1
    assert data["second_pass_triggered"] is False
    # retrieve called exactly once
    assert mock_retrieve.call_count == 1


@pytest.mark.asyncio
async def test_second_pass_triggered_on_insufficient_evidence(client):
    """When first-pass returns the sentinel phrase, second pass must run."""
    pages = _fake_pages()
    pages2 = [p.model_copy(update={"maxsim_score": 0.95}) for p in pages]

    call_count = {"n": 0}

    async def _fake_generate(question, p, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (INSUFFICIENT_EVIDENCE_PHRASE, DegradeReason.NONE)
        return ("净利润 15 亿（第二轮）", DegradeReason.NONE)

    with patch("src.routers.reports.retrieve", side_effect=[pages, pages2]) as mock_retrieve, \
         patch("src.routers.reports.generate_answer", side_effect=_fake_generate):
        resp = client.post("/reports/query", json=_page_payload())

    assert resp.status_code == 200
    data = resp.json()
    assert data["second_pass_triggered"] is True
    assert data["vlm_passes"] == 2
    assert "第二轮" in data["answer"]
    assert data["evidence_source"] == "second_pass"
    assert mock_retrieve.call_count == 2


@pytest.mark.asyncio
async def test_second_pass_not_triggered_on_vlm_error(client):
    """VLM timeout/error must NOT trigger second pass."""
    pages = _fake_pages()
    with patch("src.routers.reports.retrieve", return_value=pages), \
         patch("src.routers.reports.generate_answer",
               new=AsyncMock(return_value=(None, DegradeReason.VLM_TIMEOUT))):
        resp = client.post("/reports/query", json=_page_payload())

    assert resp.status_code == 200
    data = resp.json()
    assert data["second_pass_triggered"] is False
    assert data["degraded"] is True


@pytest.mark.asyncio
async def test_second_pass_fallback_to_first_on_second_failure(client):
    """If second pass also fails, first-pass answer is preserved."""
    pages = _fake_pages()
    pages2 = _fake_pages()

    call_count = {"n": 0}

    async def _fake_generate(question, p, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (INSUFFICIENT_EVIDENCE_PHRASE, DegradeReason.NONE)
        return (None, DegradeReason.VLM_ERROR)

    with patch("src.routers.reports.retrieve", side_effect=[pages, pages2]), \
         patch("src.routers.reports.generate_answer", side_effect=_fake_generate):
        resp = client.post("/reports/query", json=_page_payload())

    assert resp.status_code == 200
    data = resp.json()
    assert data["second_pass_triggered"] is True
    # First-pass answer (the sentinel phrase) is preserved
    assert INSUFFICIENT_EVIDENCE_PHRASE in data["answer"]


@pytest.mark.asyncio
async def test_second_pass_disabled_by_config(client):
    """When VLM_SECOND_PASS_ENABLED=False, second pass must never run."""
    pages = _fake_pages()
    with patch("src.routers.reports.retrieve", return_value=pages) as mock_retrieve, \
         patch("src.routers.reports.generate_answer",
               new=AsyncMock(return_value=(INSUFFICIENT_EVIDENCE_PHRASE, DegradeReason.NONE))), \
         patch("src.routers.reports.settings") as mock_settings:
        # Copy real settings, override only the flag
        from src.config import settings as real_settings
        mock_settings.VLM_SECOND_PASS_ENABLED = False
        mock_settings.QUERY_TIMEOUT = real_settings.QUERY_TIMEOUT
        mock_settings.VLM_SECOND_PASS_TOP_K = real_settings.VLM_SECOND_PASS_TOP_K
        mock_settings.VLM_SECOND_PASS_CANDIDATE_K = real_settings.VLM_SECOND_PASS_CANDIDATE_K
        mock_settings.VLM_SECOND_PASS_MAX_IMAGES = real_settings.VLM_SECOND_PASS_MAX_IMAGES

        resp = client.post("/reports/query", json=_page_payload())

    assert resp.status_code == 200
    data = resp.json()
    assert data["second_pass_triggered"] is False
    assert mock_retrieve.call_count == 1
