from typing import Literal

from ...graph.state import AgentState

MAX_RETRIES = 3


def validator_node(state: AgentState) -> dict[str, bool | int]:
    response = state.get("vlm_response", "")
    # Basic sanity: response must be non-empty and not contain error markers
    passed = bool(response and len(response.strip()) > 10)
    if passed:
        return {"validation_passed": True}
    return {
        "validation_passed": False,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def should_retry(state: AgentState) -> Literal["end", "retry", "rollback"]:
    if state.get("validation_passed"):
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "rollback"
    return "retry"
