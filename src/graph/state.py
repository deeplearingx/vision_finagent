from typing import Annotated
from typing_extensions import NotRequired, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from ..models.schema import PageResult


class AgentState(TypedDict):
    query: str
    target_companies: list[str]
    top_k: NotRequired[int]
    candidate_k: NotRequired[int]
    retrieved_pages: NotRequired[list[PageResult]]
    vlm_response: NotRequired[str | None]
    validation_passed: NotRequired[bool]
    retry_count: NotRequired[int]
    error_msg: NotRequired[str | None]
    messages: Annotated[list[BaseMessage], add_messages]
