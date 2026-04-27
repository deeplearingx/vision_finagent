import asyncio
import structlog
from typing import List

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


def _build_messages(query: str, pages: List[PageResult]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": query}]
    for p in pages[: settings.MAX_VLM_IMAGES]:
        if p.image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{p.image_base64}"},
            })
    return [{"role": "user", "content": content}]


def _call_vlm_sync(query: str, pages: List[PageResult]) -> str:
    """Synchronous VLM call; run inside thread/executor."""
    client = _get_client()
    messages = _build_messages(query, pages)
    resp = client.chat.completions.create(
        model=settings.VLM_MODEL,
        messages=messages,
        max_tokens=1024,
    )
    text = resp.choices[0].message.content or ""
    return text.strip()


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
