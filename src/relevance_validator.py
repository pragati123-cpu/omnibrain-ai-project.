"""
relevance_validator.py
------------------------
Week 3, Task 1: Self-RAG Retrieval Validation Logic.

Evaluates whether the chunks the Search Agent retrieved from Qdrant
are actually relevant to the user's query, and writes that verdict
into shared graph state for downstream nodes to react to:
  - Task 2's rewriter/fallback agent reads `is_relevant` to decide
    whether to rewrite the query and retry.
  - Task 5's Streamlit UI reads `relevance_reason` to show something
    like "Retrieved data irrelevant, rewriting query..."

This module deliberately does NOT decide what to do about a bad
verdict (retry, rewrite, give up) -- that's Task 2's job. This keeps
the two tasks cleanly separated: this file detects, Task 2 acts.

Two validation strategies are provided:

1. THRESHOLD CHECK (default, fast, no extra LLM call) -- looks at
   the top retrieved chunk's Qdrant cosine similarity score. Cheap,
   zero added latency/cost. Limitation: a high similarity score means
   the chunk's *wording* is close to the query's wording, which isn't
   always the same as the chunk actually *answering* the question.

2. LLM-JUDGE CHECK (optional, more robust, adds one LLM call) -- asks
   an LLM to look at the query and the top chunk's actual text and
   judge relevance directly. Catches cases where the similarity score
   is misleading, at the cost of latency and (with a paid model) cost.

RELEVANCE_SCORE_THRESHOLD below is a STARTING POINT based on typical
all-MiniLM-L6-v2 cosine-similarity behavior, not calibrated against
the real 500-page document. Recalibrate once real retrieval scores
are available -- see the "Known limitations" section in this repo's
Week 3 notes before trusting this threshold blindly in a demo.
"""

from typing import Optional, Tuple

# Cosine similarity from all-MiniLM-L6-v2: genuinely relevant
# query/chunk pairs commonly score ~0.5-0.8+; unrelated pairs commonly
# fall below ~0.3. 0.35 is a conservative starting point -- tune this
# once you can see real retrieval scores on the actual document.
RELEVANCE_SCORE_THRESHOLD = 0.35


def validate_relevance_threshold(retrieved_docs: list) -> Tuple[bool, str, Optional[float]]:
    """
    Fast path: relevance based purely on the top result's Qdrant score.
    Returns (is_relevant, reason, top_score).
    """
    if not retrieved_docs:
        return False, "No documents were retrieved for this query.", None

    top_score = retrieved_docs[0].get("score")

    if top_score is None:
        return False, "Top retrieved chunk has no similarity score to evaluate.", None

    if top_score >= RELEVANCE_SCORE_THRESHOLD:
        return (
            True,
            f"Top match score {top_score:.3f} meets the relevance threshold ({RELEVANCE_SCORE_THRESHOLD}).",
            top_score,
        )

    return (
        False,
        f"Top match score {top_score:.3f} is below the relevance threshold ({RELEVANCE_SCORE_THRESHOLD}).",
        top_score,
    )


def validate_relevance_llm(query: str, retrieved_docs: list) -> Tuple[bool, str, Optional[float]]:
    """
    Slower, more robust path: asks an LLM to judge relevance directly
    from the query + the top chunk's actual text, rather than relying
    on the embedding similarity score alone.

    Requires OPENAI_API_KEY (same client pattern as sql_agent.py).
    """
    if not retrieved_docs:
        return False, "No documents were retrieved for this query.", None

    top_chunk_text = retrieved_docs[0].get("text", "")

    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # a yes/no judgment call doesn't need the full model
        messages=[
            {
                "role": "system",
                "content": (
                    "You judge whether a retrieved passage actually answers or is "
                    "relevant to a user's question. Respond with exactly 'RELEVANT' "
                    "or 'NOT_RELEVANT' on the first line, then a one-sentence reason "
                    "on the second line."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nRetrieved passage: {top_chunk_text}",
            },
        ],
        max_tokens=60,
        temperature=0,
    )

    content = response.choices[0].message.content.strip()
    lines = content.split("\n", 1)
    verdict = lines[0].strip().upper()
    reason = lines[1].strip() if len(lines) > 1 else content

    is_relevant = verdict.startswith("RELEVANT") and not verdict.startswith("NOT_RELEVANT")
    return is_relevant, reason, retrieved_docs[0].get("score")


def validate_relevance_node(state):
    """
    LangGraph node: runs after retrieval (the "worker" node), before
    control returns to the Supervisor. Writes is_relevant,
    relevance_reason, and relevance_score into state.

    Uses the threshold check by default -- fast and needs no API key,
    which matters for anyone on the team running this without OpenAI
    access configured. To use the LLM-judge instead, call
    validate_relevance_llm(query, retrieved_docs) below instead.
    Kept as two separate functions (rather than one with a flag) so
    either can be unit-tested and reasoned about independently.
    """
    retrieved_docs = state.get("retrieved_docs", [])

    is_relevant, reason, score = validate_relevance_threshold(retrieved_docs)

    print(f"Relevance check: {is_relevant} -- {reason}")

    return {
        "current_agent": "relevance_validator",
        "is_relevant": is_relevant,
        "relevance_reason": reason,
        "relevance_score": score,
    }


if __name__ == "__main__":
    # Quick manual smoke test with representative fake data
    good_docs = [{"chunk_id": "c1", "text": "Revenue grew 12%...", "score": 0.71}]
    bad_docs = [{"chunk_id": "c2", "text": "Unrelated boilerplate...", "score": 0.12}]
    empty_docs = []

    for label, docs in [("relevant", good_docs), ("irrelevant", bad_docs), ("empty", empty_docs)]:
        result = validate_relevance_node({"retrieved_docs": docs})
        print(f"{label}: {result}")
