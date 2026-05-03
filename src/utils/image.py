import base64
import io
import re
from pathlib import Path
from PIL import Image
from ..config import settings

MAX_SIZE = (2048, 2048)
# Milvus/Zilliz JSON field hard limit (bytes of the base64 string).
# Use 64000 (< 65536) to leave headroom for Milvus internal encoding overhead.
_DB_B64_LIMIT = 64000


def to_base64(img: Image.Image) -> str:
    img.thumbnail(MAX_SIZE)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def load_and_encode(path: str) -> str:
    return to_base64(Image.open(path))

to_base64_jpeg = to_base64


def to_base64_bounded(img: Image.Image, limit: int = _DB_B64_LIMIT) -> str:
    """Encode *img* to a JPEG base64 string whose length is strictly < *limit*.

    Strategy: iteratively halve the longer dimension and drop quality until the
    encoded length fits.  Starts at quality=75 / max-side=512 to stay well
    inside the 65 536-char budget from the first attempt for typical PDF pages.
    """
    img = img.convert("RGB")
    max_side = 512
    quality = 75
    while True:
        clone = img.copy()
        clone.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        clone.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        if len(b64) < limit:
            return b64
        # Still too large — reduce further
        if quality > 40:
            quality -= 15
        else:
            max_side = max(64, max_side // 2)
            quality = 75


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def save_page_image(img: Image.Image, report_id: str, page_num: int) -> str:
    """保存高清页面图到磁盘，返回文件路径。"""
    out_dir = Path(settings.PAGE_IMAGE_DIR) / _safe_name(report_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{page_num:04d}.jpg"
    img.convert("RGB").save(out_path, format="JPEG", quality=95)
    return str(out_path)


def encode_image_file_for_vlm(path: str) -> str:
    """从磁盘读取高清图，按 VLM_IMG_MAX_SIDE 缩放后编码给 VLM。"""
    img = Image.open(path).convert("RGB")
    if settings.VLM_IMG_MAX_SIDE > 0:
        img.thumbnail((settings.VLM_IMG_MAX_SIDE, settings.VLM_IMG_MAX_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=settings.VLM_IMG_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
