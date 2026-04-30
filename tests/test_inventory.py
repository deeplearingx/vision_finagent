"""Tests for GET /reports/admin/inventory and /reports/admin/inventory/{report_id}."""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(*report_pages: tuple[str, list[int]]) -> list[dict]:
    """Build fake Milvus query rows: [(report_id, [page_nums]), ...]"""
    rows = []
    for rid, pages in report_pages:
        for pn in pages:
            rows.append({"report_id": rid, "page_num": pn})
    return rows


# ---------------------------------------------------------------------------
# list_reports_inventory unit tests (pure function, no HTTP)
# ---------------------------------------------------------------------------

class TestListReportsInventory:
    def test_collection_absent(self):
        from src.core.milvus_client import list_reports_inventory
        with patch("src.core.milvus_client.get_client") as mc:
            mc.return_value.has_collection.return_value = False
            result = list_reports_inventory()

        assert result["total_reports"] == 0
        assert result["reports"] == []
        assert result["truncated"] is False

    def test_empty_collection(self):
        from src.core.milvus_client import list_reports_inventory
        with patch("src.core.milvus_client.get_client") as mc:
            mc.return_value.has_collection.return_value = True
            mc.return_value.query.return_value = []
            result = list_reports_inventory()

        assert result["total_reports"] == 0
        assert result["truncated"] is False

    def test_single_report(self):
        from src.core.milvus_client import list_reports_inventory
        rows = _make_rows(("rpt_001", [1, 2, 3]))
        with patch("src.core.milvus_client.get_client") as mc:
            mc.return_value.has_collection.return_value = True
            mc.return_value.query.return_value = rows
            result = list_reports_inventory()

        assert result["total_reports"] == 1
        assert result["reports"][0]["report_id"] == "rpt_001"
        assert result["reports"][0]["page_count"] == 3
        assert result["reports"][0]["page_nums"] == [1, 2, 3]
        assert result["truncated"] is False

    def test_multiple_reports(self):
        from src.core.milvus_client import list_reports_inventory
        rows = _make_rows(("rpt_a", [1, 2]), ("rpt_b", [1, 2, 3, 4]))
        with patch("src.core.milvus_client.get_client") as mc:
            mc.return_value.has_collection.return_value = True
            mc.return_value.query.return_value = rows
            result = list_reports_inventory()

        assert result["total_reports"] == 2
        assert result["total_rows_fetched"] == 6

    def test_truncated_flag(self):
        from src.core.milvus_client import list_reports_inventory
        rows = _make_rows(("rpt_x", list(range(5))))
        with patch("src.core.milvus_client.get_client") as mc:
            mc.return_value.has_collection.return_value = True
            mc.return_value.query.return_value = rows
            # max_rows == len(rows) → truncated
            result = list_reports_inventory(max_rows=5)

        assert result["truncated"] is True


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------

class TestInventoryEndpoints:
    def test_inventory_empty(self, client: TestClient, patch_milvus):
        patch_milvus.has_collection.return_value = True
        patch_milvus.query.return_value = []
        resp = client.get("/reports/admin/inventory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_reports"] == 0
        assert data["reports"] == []

    def test_inventory_with_reports(self, client: TestClient, patch_milvus):
        patch_milvus.has_collection.return_value = True
        patch_milvus.query.return_value = _make_rows(("rpt_001", [1, 2, 3]))
        resp = client.get("/reports/admin/inventory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_reports"] == 1
        assert data["reports"][0]["page_count"] == 3

    def test_check_report_found(self, client: TestClient, patch_milvus):
        patch_milvus.has_collection.return_value = True
        patch_milvus.query.return_value = _make_rows(("rpt_001", [1, 2]))
        resp = client.get("/reports/admin/inventory/rpt_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["page_count"] == 2

    def test_check_report_not_found(self, client: TestClient, patch_milvus):
        patch_milvus.has_collection.return_value = True
        patch_milvus.query.return_value = _make_rows(("rpt_other", [1]))
        resp = client.get("/reports/admin/inventory/rpt_missing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False
        assert data["page_count"] == 0
