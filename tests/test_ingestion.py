"""Tests for ingestion_service: PDF repair, page iteration."""
import sys
import types
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image


# ---------------------------------------------------------------------------
# Stub heavy dependencies before any src.* import
# ---------------------------------------------------------------------------

def _stub_modules():
    stubs = [
        "colpali_engine",
        "colpali_engine.models",
        "colpali_engine.models.paligemma",
        "colpali_engine.models.paligemma.colpali",
        "colpali_engine.models.paligemma.colpali.modeling_colpali",
        "colpali_engine.models.paligemma.colpali.processing_colpali",
    ]
    for name in stubs:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    # Provide required attributes
    sys.modules["colpali_engine.models.paligemma.colpali.modeling_colpali"].ColPali = object
    sys.modules["colpali_engine.models.paligemma.colpali.processing_colpali"].ColPaliProcessor = object

    # Stub torch to avoid GPU/segfault
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.no_grad = lambda: MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None)
        torch_mod.cuda = MagicMock()
        torch_mod.cuda.is_available = lambda: False
        sys.modules["torch"] = torch_mod


_stub_modules()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_pixmap(w=4, h=4):
    pix = MagicMock()
    pix.width = w
    pix.height = h
    pix.samples = bytes(w * h * 3)
    return pix


def _make_fake_doc(page_count: int, bad_pages: set = None):
    bad_pages = bad_pages or set()
    doc = MagicMock()
    doc.page_count = page_count

    def _load_page(i):
        if i + 1 in bad_pages:
            raise RuntimeError(f"MuPDF error on page {i + 1}")
        page = MagicMock()
        page.get_pixmap.return_value = _fake_pixmap()
        return page

    doc.load_page.side_effect = _load_page
    return doc


def _run_iter(doc, path="test.pdf"):
    with patch("src.services.ingestion_service._repair_pdf_with_pymupdf", return_value=path), \
         patch("src.services.ingestion_service.fitz.open", return_value=doc):
        from src.services.ingestion_service import _iter_pdf_pages
        return list(_iter_pdf_pages(path))


# ---------------------------------------------------------------------------
# _repair_pdf_with_pymupdf
# ---------------------------------------------------------------------------

class TestRepairPdf:
    def test_success_returns_repaired_path(self, tmp_path):
        src = str(tmp_path / "test.pdf")
        repaired = src + ".repaired.pdf"
        fake_doc = MagicMock()
        with patch("src.services.ingestion_service.fitz.open", return_value=fake_doc):
            from src.services.ingestion_service import _repair_pdf_with_pymupdf
            result = _repair_pdf_with_pymupdf(src)
        fake_doc.save.assert_called_once_with(repaired, garbage=4, deflate=True, clean=True)
        fake_doc.close.assert_called_once()
        assert result == repaired

    def test_open_failure_returns_original(self, tmp_path):
        src = str(tmp_path / "bad.pdf")
        with patch("src.services.ingestion_service.fitz.open", side_effect=Exception("corrupt")):
            from src.services.ingestion_service import _repair_pdf_with_pymupdf
            assert _repair_pdf_with_pymupdf(src) == src

    def test_save_failure_returns_original_and_closes(self, tmp_path):
        src = str(tmp_path / "bad.pdf")
        fake_doc = MagicMock()
        fake_doc.save.side_effect = Exception("write error")
        with patch("src.services.ingestion_service.fitz.open", return_value=fake_doc):
            from src.services.ingestion_service import _repair_pdf_with_pymupdf
            result = _repair_pdf_with_pymupdf(src)
        fake_doc.close.assert_called_once()
        assert result == src


# ---------------------------------------------------------------------------
# _iter_pdf_pages
# ---------------------------------------------------------------------------

class TestIterPdfPages:
    def test_normal_two_pages(self):
        doc = _make_fake_doc(2)
        results = _run_iter(doc)
        assert [pn for pn, _ in results] == [1, 2]
        assert all(isinstance(img, Image.Image) for _, img in results)

    def test_bad_middle_page_skipped_nums_preserved(self):
        doc = _make_fake_doc(3, bad_pages={2})
        results = _run_iter(doc)
        assert [pn for pn, _ in results] == [1, 3]

    def test_all_bad_raises_ingestion_error(self):
        from src.core.exception import IngestionError
        doc = _make_fake_doc(2, bad_pages={1, 2})
        with pytest.raises(IngestionError, match="所有页面"):
            _run_iter(doc)

    def test_empty_pdf_raises_ingestion_error(self):
        from src.core.exception import IngestionError
        doc = _make_fake_doc(0)
        with pytest.raises(IngestionError, match="不含任何页面"):
            _run_iter(doc)

    def test_doc_always_closed_on_success(self):
        doc = _make_fake_doc(2)
        _run_iter(doc)
        doc.close.assert_called_once()

    def test_doc_closed_on_all_bad(self):
        from src.core.exception import IngestionError
        doc = _make_fake_doc(1, bad_pages={1})
        with pytest.raises(IngestionError):
            _run_iter(doc)
        doc.close.assert_called_once()
