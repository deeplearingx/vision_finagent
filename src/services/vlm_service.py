import asyncio
import base64
import io
import time
import re
import structlog
from typing import List

from PIL import Image
from openai import OpenAI
from ..config import settings
from ..models.schema import PageResult
from ..models.enums import DegradeReason

log = structlog.get_logger()
_client: OpenAI | None = None


# Sentinel value indicating no real key is configured
_EMPTY_KEY = "EMPTY"


def _check_vlm_config() -> str | None:
    """Return a human-readable reason string if VLM config is missing, else None."""
    if not settings.VLM_API_KEY or settings.VLM_API_KEY == _EMPTY_KEY:
        return "VLM_API_KEY is not configured (value is EMPTY or blank)"
    if not settings.VLM_API_BASE:
        return "VLM_API_BASE is not configured"
    if not settings.VLM_MODEL:
        return "VLM_MODEL is not configured"
    return None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=settings.VLM_API_BASE,
            api_key=settings.VLM_API_KEY,
            timeout=settings.VLM_TIMEOUT,
        )
    return _client


# Financial document analysis instruction prepended to every VLM query.
# Kept short to avoid token waste; covers the three known failure modes:
#   1. Cover/TOC pages misleading the model → explicit skip instruction
#   2. Evidence insufficiency → require explicit statement rather than hallucination
#   3. Priority guidance → tables and MD&A sections first
# Canonical sentinel phrase emitted by VLM when evidence is insufficient.
# Used both in the system prompt and in is_insufficient_evidence() — keep in sync.
INSUFFICIENT_EVIDENCE_PHRASE = "Insufficient evidence in provided pages"

_FIN_DOC_SYSTEM = (
    "You are a financial document analyst. "
    "Rules: "
    "(1) IGNORE cover pages, table-of-contents pages, and disclaimer pages — do not extract data from them. "
    "(2) Prioritize data tables, financial statements, and Management Discussion & Analysis sections. "
    f"(3) If the provided pages do not contain sufficient evidence to answer the question, "
    f"explicitly state '{INSUFFICIENT_EVIDENCE_PHRASE}' rather than guessing."
)


def is_insufficient_evidence(answer: str | None) -> bool:
    """Return True only when VLM explicitly signals evidence insufficiency.

    Conservative: only triggers on the canonical phrase, not on errors/timeouts.
    """
    if not answer:
        return False
    return INSUFFICIENT_EVIDENCE_PHRASE.lower() in answer.lower()


def _compress_image_base64(b64_str: str) -> tuple[str, int, int]:
    """Compress base64 image according to VLM_IMG_* settings.
    
    Returns: (compressed_b64, original_bytes, compressed_bytes)
    """
    if not b64_str:
        return b64_str, 0, 0
    
    original_bytes = len(b64_str) * 3 // 4  # approximate decoded size
    
    # Skip if all compression disabled
    if settings.VLM_IMG_MAX_SIDE == 0 and settings.VLM_IMG_JPEG_QUALITY == 0:
        return b64_str, original_bytes, original_bytes
    
    try:
        img_data = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # Resize if needed
        if settings.VLM_IMG_MAX_SIDE > 0:
            w, h = img.size
            max_side = max(w, h)
            if max_side > settings.VLM_IMG_MAX_SIDE:
                scale = settings.VLM_IMG_MAX_SIDE / max_side
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Re-encode as JPEG if quality specified
        buf = io.BytesIO()
        if settings.VLM_IMG_JPEG_QUALITY > 0:
            img.convert("RGB").save(buf, format="JPEG", quality=settings.VLM_IMG_JPEG_QUALITY)
        else:
            img.save(buf, format="JPEG")
        
        compressed_data = buf.getvalue()
        compressed_bytes = len(compressed_data)
        
        # Check byte budget if set
        if settings.VLM_IMG_MAX_BYTES > 0 and compressed_bytes > settings.VLM_IMG_MAX_BYTES:
            # Retry with lower quality
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=50)
            compressed_data = buf.getvalue()
            compressed_bytes = len(compressed_data)
        
        compressed_b64 = base64.b64encode(compressed_data).decode("utf-8")
        return compressed_b64, original_bytes, compressed_bytes
    
    except Exception as e:
        log.warning("image_compress.failed", error=str(e))
        return b64_str, original_bytes, original_bytes


def _build_messages(query: str, pages: List[PageResult]) -> list[dict]:
    """Build VLM message list with financial-document analysis instructions.

    Structure: system instruction (text) → user query (text) → page images.
    Applies outbound image compression per VLM_IMG_* settings before sending.
    """
    content: list[dict] = [
        {"type": "text", "text": _FIN_DOC_SYSTEM},
        {"type": "text", "text": query},
    ]
    total_orig, total_comp, img_count = 0, 0, 0
    for p in pages[: settings.MAX_VLM_IMAGES]:
        if p.image_base64:
            compressed_b64, orig_b, comp_b = _compress_image_base64(p.image_base64)
            total_orig += orig_b
            total_comp += comp_b
            img_count += 1
            log.debug(
                "image_compress.single",
                page=p.page_num,
                orig_bytes=orig_b,
                comp_bytes=comp_b,
                ratio=round(comp_b / orig_b, 3) if orig_b else 1.0,
            )
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{compressed_b64}"},
            })
    log.info(
        "image_compress.summary",
        images=img_count,
        total_orig_bytes=total_orig,
        total_comp_bytes=total_comp,
        saved_bytes=total_orig - total_comp,
        ratio=round(total_comp / total_orig, 3) if total_orig else 1.0,
    )
    return [{"role": "user", "content": content}]


# Keywords indicating a market/price-trend question.
# Used only for soft labelling — never for hard rejection.
_MARKET_KEYWORDS = re.compile(
    r"走势|涨跌|K线|k线|股价|行情|成交量|均线|技术指标|price trend|stock chart|candlestick",
    re.IGNORECASE,
)

# Hint appended when evidence is still insufficient after second pass AND the
# question looks market-oriented.  Keeps the main answer intact.
MARKET_BOUNDARY_HINT = (
    "\n\n[提示] 当前上传证据中未找到足够的行情走势数据。"
    "若需要实时或区间股价走势，请接入市场行情数据源。"
)


def is_market_like_question(question: str) -> bool:
    """Soft label: True if question looks market/price-trend oriented."""
    return bool(_MARKET_KEYWORDS.search(question))


def _is_connection_error(exc: Exception) -> bool:
    """Return True for transient network/connection errors worth retrying."""
    msg = str(exc).lower()
    return any(k in msg for k in ("connection error", "connection reset", "connect timeout",
                                   "remotedisconnected", "connectionerror", "network"))


def _call_vlm_sync(query: str, pages: List[PageResult]) -> str:
    """Synchronous VLM call with connection-level retry; run inside thread/executor."""
    client = _get_client()
    messages = _build_messages(query, pages)
    max_attempts = settings.VLM_RETRY_MAX_ATTEMPTS if settings.VLM_RETRY_ENABLED else 1
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.VLM_MODEL,
                messages=messages,
                max_tokens=1024,
            )
            if attempt > 1:
                log.info("vlm.retry_succeeded", attempt=attempt, query=query)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts and _is_connection_error(exc):
                log.warning("vlm.retry_scheduled", attempt=attempt, error=str(exc), query=query)
                time.sleep(settings.VLM_RETRY_BACKOFF_SECONDS)
            else:
                raise
    raise last_exc  # unreachable but satisfies type checker


async def generate_answer(
    query: str,
    pages: List[PageResult],
    timeout: float | None = None,
) -> tuple[str | None, DegradeReason]:
    """Call VLM with retrieved pages and return (answer, degrade_reason).

    If timeout or any exception occurs, returns (None, reason).
    """
    if not pages:
        return None, DegradeReason.NONE

    # --- Config guard: fail fast with observable reason ---
    config_err = _check_vlm_config()
    if config_err:
        log.warning("vlm.config_missing", reason=config_err, query=query)
        return None, DegradeReason.VLM_NO_CONFIG

    timeout = timeout or settings.VLM_QUERY_TIMEOUT
    log.info("vlm.start", query=query, images=min(len(pages), settings.MAX_VLM_IMAGES), timeout=timeout)

    try:
        answer = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _call_vlm_sync, query, pages),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        log.warning("vlm.timeout", query=query, timeout=timeout)
        return None, DegradeReason.VLM_TIMEOUT
    except Exception as exc:
        log.error("vlm.error", query=query, error=str(exc))
        return None, DegradeReason.VLM_ERROR

    if not answer or len(answer) < 5:
        log.warning("vlm.empty_response", query=query)
        return None, DegradeReason.VLM_ERROR

    log.info("vlm.done", query=query, answer_preview=answer[:120])
    return answer, DegradeReason.NONE
