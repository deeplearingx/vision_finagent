import base64
import io
from PIL import Image

MAX_SIZE = (2048, 2048)
# Milvus/Zilliz JSON field hard limit (bytes of the base64 string)
_DB_B64_LIMIT = 65536


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
