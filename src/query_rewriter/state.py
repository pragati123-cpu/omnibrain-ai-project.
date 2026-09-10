from typing import TypedDict


class QueryRewriteState(TypedDict):
    original_query: str
    current_query: str

    results: list
    retrieval_score: float

    rewritten_query: str

    retry_count: int
    max_retries: int

    status: str