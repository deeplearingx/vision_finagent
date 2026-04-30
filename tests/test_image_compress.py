"""Tests for image compression utilities."""
import base64
import io
import pytest
from PIL import Image
from unittest.mock import patch


def _make_b64_image(width: int, height: int, quality: int = 95) -> str:
    """Create a synthetic JPEG base64 string of given dimensions."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_compress_large_image_reduces_size():
    """Large image (2048x2048) should be resized and produce smaller output."""
    from src.services.vlm_service import _compress_image_base64

    b64 = _make_b64_image(2048, 2048)
    compressed, orig_b, comp_b = _compress_image_base64(b64)

    assert comp_b < orig_b, "Compressed bytes should be less than original"
    # Verify output is valid base64 JPEG
    decoded = base64.b64decode(compressed)
    img = Image.open(io.BytesIO(decoded))
    assert max(img.size) <= 1024


def test_compress_small_image_no_resize():
    """Small image (256x256) should not be upscaled, only re-encoded."""
    from src.services.vlm_service import _compress_image_base64

    b64 = _make_b64_image(256, 256)
    compressed, orig_b, comp_b = _compress_image_base64(b64)

    decoded = base64.b64decode(compressed)
    img = Image.open(io.BytesIO(decoded))
    # Dimensions must not exceed original
    assert img.width <= 256 and img.height <= 256


def test_compress_empty_string_passthrough():
    """Empty base64 string should be returned unchanged."""
    from src.services.vlm_service import _compress_image_base64

    result, orig_b, comp_b = _compress_image_base64("")
    assert result == ""
    assert orig_b == 0 and comp_b == 0


def test_compress_disabled_when_both_zero():
    """When MAX_SIDE=0 and QUALITY=0, image is returned as-is."""
    from src.services.vlm_service import _compress_image_base64

    b64 = _make_b64_image(2048, 2048)
    with patch("src.services.vlm_service.settings") as mock_settings:
        mock_settings.VLM_IMG_MAX_SIDE = 0
        mock_settings.VLM_IMG_JPEG_QUALITY = 0
        mock_settings.VLM_IMG_MAX_BYTES = 0
        result, orig_b, comp_b = _compress_image_base64(b64)

    assert result == b64
    assert orig_b == comp_b


# ---------------------------------------------------------------------------
# to_base64_bounded – DB field length guard
# ---------------------------------------------------------------------------

def test_bounded_large_image_within_limit():
    """A large synthetic image must encode to < 65536 chars."""
    from src.utils.image import to_base64_bounded, _DB_B64_LIMIT
    img = Image.new("RGB", (3000, 2000), color=(200, 100, 50))
    b64 = to_base64_bounded(img)
    assert len(b64) < _DB_B64_LIMIT


def test_bounded_small_image_within_limit():
    """A small image must also satisfy the limit (sanity check)."""
    from src.utils.image import to_base64_bounded, _DB_B64_LIMIT
    img = Image.new("RGB", (100, 100), color=(10, 20, 30))
    b64 = to_base64_bounded(img)
    assert len(b64) < _DB_B64_LIMIT
    assert len(b64) > 0
