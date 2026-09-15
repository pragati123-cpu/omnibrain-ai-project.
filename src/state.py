from typing import TypedDict
from typing import TypedDict, List, Optional


class RetrievedChunk(TypedDict):
    chunk_id: str
    text: str
    page_number: int
    source_file: str
    score: float


class AgentState(TypedDict):
    messages: list
    next: str
    current_agent: str
    task: str
    result: str
    step_count: int
    
    task: str            # NOTE: currently used as the user's raw query text.
                          # Confirm this with the team -- if "task" is meant
                          # for something else (e.g. an agent-routing label),
                          # the fields below should read from a different key.
    result: str
    step_count: int

    # --- Added for Week 3 / Task 1: Self-RAG Retrieval Validation ---
    retrieved_docs: List[RetrievedChunk]  # raw results from the Search Agent
    is_relevant: Optional[bool]            # relevance verdict for the current retrieval
    relevance_reason: Optional[str]        # human-readable explanation (for Task 5's UI)
    relevance_score: Optional[float]       # numeric signal (top Qdrant score), for tuning/debugging

    visual_sources: list
