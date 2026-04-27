import structlog
from ...graph.state import AgentState

log = structlog.get_logger()


def rollback_node(state: AgentState) -> dict[str, object]:
    log.warning("rollback", query=state.get("query"), retries=state.get("retry_count"))
    return {
        "vlm_response": None,
        "retrieved_pages": [],
        "validation_passed": False,
        "error_msg": "Max retries exceeded; state rolled back.",
    }
