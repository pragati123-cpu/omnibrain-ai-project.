from .state import QueryRewriteState


SCORE_THRESHOLD = 0.70


def search(state: QueryRewriteState):
    query = state["current_query"]

    print(f"\nSearching: {query}")

    # Temporary mock retrieval
    # We will replace this with a real retriever later.

    if "state graph" in query.lower():
        score = 0.85
    else:
        score = 0.35

    results = [
        f"Document retrieved for: {query}"
    ]

    print(f"Retrieval score: {score}")

    return {
        "results": results,
        "retrieval_score": score,
    }


def evaluate_retrieval(state: QueryRewriteState):

    score = state["retrieval_score"]

    print(f"\nEvaluating score: {score}")

    if score >= SCORE_THRESHOLD:
        return "success"

    if state["retry_count"] >= state["max_retries"]:
        return "failure"

    return "rewrite"


def rewrite_query(state: QueryRewriteState):

    current_query = state["current_query"]

    print("\nLow retrieval score.")
    print(f"Original query: {current_query}")

    rewritten_query = (
        f"{current_query} state graph workflow"
    )

    print(f"Rewritten query: {rewritten_query}")

    return {
        "current_query": rewritten_query,
        "rewritten_query": rewritten_query,
        "retry_count": state["retry_count"] + 1,
    }


def success(state: QueryRewriteState):

    print("\nRelevant results found!")

    return {
        "status": "success"
    }


def failure(state: QueryRewriteState):

    print("\nMaximum retries reached.")

    return {
        "status": "failed"
    }