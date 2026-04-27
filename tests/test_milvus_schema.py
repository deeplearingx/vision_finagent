"""Regression tests: milvus_client.check_schema_health() drift detection."""
import pytest
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_client(patches_fields, pages_fields):
    """Return a mock MilvusClient whose describe_collection returns given fields."""
    mc = MagicMock()
    mc.has_collection.return_value = True

    def _describe(collection_name):
        if collection_name.endswith("_pages"):
            return {"fields": [{"name": f} for f in pages_fields]}
        return {"fields": [{"name": f} for f in patches_fields]}

    mc.describe_collection.side_effect = _describe
    return mc


# ---------------------------------------------------------------------------
# Healthy schema → no drift
# ---------------------------------------------------------------------------
def test_schema_health_no_drift():
    from src.core.milvus_client import check_schema_health
    mc = _make_client(
        patches_fields=["pk", "report_id", "page_num", "colpali_embeddings"],
        pages_fields=["page_id", "report_id", "page_num", "image_base64"],
    )
    with patch("src.core.milvus_client.get_client", return_value=mc):
        results = check_schema_health()

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert r.get("missing", []) == []
        assert r.get("extra", []) == []


# ---------------------------------------------------------------------------
# Missing field → drift detected
# ---------------------------------------------------------------------------
def test_schema_health_missing_field_detected():
    from src.core.milvus_client import check_schema_health
    # patches collection is missing 'colpali_embeddings'
    mc = _make_client(
        patches_fields=["pk", "report_id", "page_num"],
        pages_fields=["page_id", "report_id", "page_num", "image_base64"],
    )
    with patch("src.core.milvus_client.get_client", return_value=mc):
        results = check_schema_health()

    patches_result = next(r for r in results if not r["collection"].endswith("_pages"))
    assert "colpali_embeddings" in patches_result["missing"]


# ---------------------------------------------------------------------------
# Extra field → drift detected
# ---------------------------------------------------------------------------
def test_schema_health_extra_field_detected():
    from src.core.milvus_client import check_schema_health
    mc = _make_client(
        patches_fields=["pk", "report_id", "page_num", "colpali_embeddings", "unexpected_col"],
        pages_fields=["page_id", "report_id", "page_num", "image_base64"],
    )
    with patch("src.core.milvus_client.get_client", return_value=mc):
        results = check_schema_health()

    patches_result = next(r for r in results if not r["collection"].endswith("_pages"))
    assert "unexpected_col" in patches_result["extra"]


# ---------------------------------------------------------------------------
# Collection absent → note field present
# ---------------------------------------------------------------------------
def test_schema_health_collection_absent():
    from src.core.milvus_client import check_schema_health
    mc = MagicMock()
    mc.has_collection.return_value = False

    with patch("src.core.milvus_client.get_client", return_value=mc):
        results = check_schema_health()

    for r in results:
        assert r.get("note") == "collection_absent"
        assert set(r["missing"]) != set()  # all expected fields reported missing


# ---------------------------------------------------------------------------
# Return structure is always a list of dicts with 'collection' key
# ---------------------------------------------------------------------------
def test_schema_health_return_structure():
    from src.core.milvus_client import check_schema_health
    mc = _make_client(
        patches_fields=["pk", "report_id", "page_num", "colpali_embeddings"],
        pages_fields=["page_id", "report_id", "page_num", "image_base64"],
    )
    with patch("src.core.milvus_client.get_client", return_value=mc):
        results = check_schema_health()

    for r in results:
        assert "collection" in r
        assert "missing" in r
        assert "extra" in r
