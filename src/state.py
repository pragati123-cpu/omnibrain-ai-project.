from typing import TypedDict


class AgentState(TypedDict):
    messages: list
    next: str
    current_agent: str
    task: str
    result: str
    step_count: int
    