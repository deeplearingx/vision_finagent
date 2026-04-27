from openai import OpenAI
from ...config import settings
from ...graph.state import AgentState

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=settings.VLM_API_BASE,
            api_key=settings.VLM_API_KEY,
            timeout=settings.VLM_TIMEOUT,
        )
    return _client


def vlm_node(state: AgentState) -> dict[str, str | None]:
    images = [p.image_base64 for p in state.get("retrieved_pages", [])[:4]]
    content = [{"type": "text", "text": state["query"]}]
    for b64 in images[:4]:  # cap at 4 images per request
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    resp = _get_client().chat.completions.create(
        model=settings.VLM_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
    )
    return {"vlm_response": resp.choices[0].message.content}
