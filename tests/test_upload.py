"""Regression tests: /reports/upload — file validation & maintenance gate."""
import io
import pytest
from unittest.mock import patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pdf_bytes(valid=True):
    if valid:
        return b"%PDF-1.4 fake content" + b" " * 100 + b"%%EOF"
    return b"not a pdf at all"


def _png_bytes(valid=True):
    if valid:
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    return b"FAKEPNG garbage"


def _jpg_bytes(valid=True):
    if valid:
        return b"\xff\xd8\xff\xe0" + b"\x00" * 20
    return b"FAKEJPG garbage"


def _upload(client, data: bytes, filename: str, report_id: str = "rpt_test"):
    return client.post(
        "/reports/upload",
        data={"report_id": report_id},
        files={"file": (filename, io.BytesIO(data), "application/octet-stream")},
    )


# ---------------------------------------------------------------------------
# Valid uploads → 202
# ---------------------------------------------------------------------------
def test_valid_pdf_accepted(client):
    r = _upload(client, _pdf_bytes(True), "report.pdf")
    assert r.status_code == 202
    body = r.json()
    assert body["report_id"] == "rpt_test"
    assert body["task_id"] == "task_test123"


def test_valid_png_accepted(client):
    r = _upload(client, _png_bytes(True), "chart.png", report_id="rpt_png")
    assert r.status_code == 202


def test_valid_jpg_accepted(client):
    r = _upload(client, _jpg_bytes(True), "chart.jpg", report_id="rpt_jpg")
    assert r.status_code == 202


# ---------------------------------------------------------------------------
# Invalid file content → 400
# ---------------------------------------------------------------------------
def test_invalid_pdf_rejected(client):
    r = _upload(client, _pdf_bytes(False), "bad.pdf")
    assert r.status_code == 400
    assert "Invalid PDF" in r.json()["detail"]


def test_invalid_png_rejected(client):
    r = _upload(client, _png_bytes(False), "bad.png", report_id="rpt_badpng")
    assert r.status_code == 400
    assert "Invalid image" in r.json()["detail"]


def test_invalid_jpg_rejected(client):
    r = _upload(client, _jpg_bytes(False), "bad.jpg", report_id="rpt_badjpg")
    assert r.status_code == 400
    assert "Invalid image" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Unsupported extension → 400
# ---------------------------------------------------------------------------
def test_unsupported_extension_rejected(client):
    r = _upload(client, b"some data", "report.docx", report_id="rpt_docx")
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Maintenance mode → 503
# ---------------------------------------------------------------------------
def test_upload_blocked_during_maintenance(client):
    with patch("src.routers.reports._maintenance_active", True):
        r = _upload(client, _pdf_bytes(True), "report.pdf")
    assert r.status_code == 503
    assert "维护" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Duplicate upload (idempotency) → 409
# ---------------------------------------------------------------------------
def test_duplicate_upload_rejected(client, fake_redis):
    # Pre-seed the idempotency token as already taken
    import asyncio
    asyncio.get_event_loop().run_until_complete(
        fake_redis.set("upload:rpt_dup", "1", nx=True)
    )
    r = _upload(client, _pdf_bytes(True), "report.pdf", report_id="rpt_dup")
    assert r.status_code == 409
