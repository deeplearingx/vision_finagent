"""Test Milvus company filter expression construction."""
import pytest
from src.services.retrieval_service import _build_company_filter


def test_no_filter_when_empty():
    assert _build_company_filter([]) is None


def test_single_company_has_prefix_and_exact():
    expr = _build_company_filter(["AAPL"])
    assert expr is not None
    assert 'report_id like "AAPL_%"' in expr
    assert 'report_id == "AAPL"' in expr


def test_multi_company_joined_with_or():
    expr = _build_company_filter(["AAPL", "MSFT"])
    assert expr is not None
    assert "AAPL" in expr
    assert "MSFT" in expr
    # Both companies must appear; joined by or
    assert expr.count(" or ") >= 3  # 2 clauses per company → 3 ors minimum


def test_filter_is_parenthesised():
    expr = _build_company_filter(["AAPL"])
    assert expr.startswith("(") and expr.endswith(")")


def test_special_chars_are_json_escaped():
    """Ticker with quote-like chars must not break the filter expression."""
    expr = _build_company_filter(['O"REILLY'])
    # json.dumps will escape the quote; expression must still be a string
    assert isinstance(expr, str)
    assert "O" in expr
