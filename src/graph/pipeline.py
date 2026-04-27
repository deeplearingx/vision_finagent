from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes.retriever_node import retriever_node
from .nodes.vlm_node import vlm_node
from .nodes.validator_node import validator_node, should_retry
from .nodes.rollback_node import rollback_node


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retriever", retriever_node)
    g.add_node("vlm", vlm_node)
    g.add_node("validator", validator_node)
    g.add_node("rollback", rollback_node)

    g.add_edge(START, "retriever")
    g.add_edge("retriever", "vlm")
    g.add_edge("vlm", "validator")
    g.add_conditional_edges(
        "validator",
        should_retry,
        {"end": END, "retry": "retriever", "rollback": "rollback"},
    )
    g.add_edge("rollback", END)
    return g.compile(checkpointer=MemorySaver())


graph = build_graph()
