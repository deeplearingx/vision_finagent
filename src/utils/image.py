import base64
import io
from PIL import Image

MAX_SIZE = (2048, 2048)


def to_base64(img: Image.Image) -> str:
    img.thumbnail(MAX_SIZE)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def load_and_encode(path: str) -> str:
    return to_base64(Image.open(path))

to_base64_jpeg = to_base64
