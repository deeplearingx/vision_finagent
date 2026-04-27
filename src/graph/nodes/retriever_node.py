from ...graph.state import AgentState
from ...services.retrieval_service import retrieve


def retriever_node(state: AgentState) -> dict:
    pages = retrieve(
        state["query"],
        state.get("target_companies", []),
        top_k=state.get("top_k", 3),
        candidate_k=state.get("candidate_k", 50),
    )
    return {"retrieved_pages": pages}
