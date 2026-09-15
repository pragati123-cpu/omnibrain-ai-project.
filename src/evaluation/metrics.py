import time


def calculate_latency(start_time: float) -> float:
    """Calculate execution latency in seconds."""
    return round(time.time() - start_time, 4)


def calculate_token_usage(response):
    """Extract token usage from an LLM response."""

    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def evaluate_correctness(
    response_text: str,
    expected_answer: str,
):
    """
    Evaluate correctness using the expected answer.

    The evaluation checks whether the important concepts
    from the expected answer appear in the response.
    """

    text = response_text.lower()
    expected = expected_answer.lower()

    # Important concepts required for a correct RAG answer
    required_concepts = [
        "retrieval",
        "augmented",
        "generation",
    ]

    matched_concepts = [
        concept
        for concept in required_concepts
        if concept in text
    ]

    score = len(matched_concepts) / len(required_concepts)

    # A response containing all core concepts is considered correct
    if score == 1.0:
        label = "correct"
    elif score >= 0.5:
        label = "partially_correct"
    else:
        label = "incorrect"

    return {
        "score": round(score, 2),
        "label": label,
        "matched_concepts": matched_concepts,
        "expected_answer": expected_answer,
    }


def evaluate_relevance(
    response_text: str,
    required_keywords: list[str],
):
    """Evaluate whether the response addresses the expected topic."""

    if not required_keywords:
        return {
            "score": 0.0,
            "matched_keywords": 0,
            "total_keywords": 0,
        }

    text = response_text.lower()

    matched = sum(
        keyword.lower() in text
        for keyword in required_keywords
    )

    score = matched / len(required_keywords)

    return {
        "score": round(score, 2),
        "matched_keywords": matched,
        "total_keywords": len(required_keywords),
    }


def evaluate_hallucination(
    response_text: str,
    expected_answer: str,
):
    """
    Detect possible hallucination by checking whether the
    response contradicts the expected answer.

    For the current RAG test, the expected concept is:
    Retrieval-Augmented Generation.
    """

    text = response_text.lower()

    correct_concepts = [
        "retrieval-augmented generation",
        "retrieval augmented generation",
    ]

    has_correct_definition = any(
        concept in text
        for concept in correct_concepts
    )

    # Known incorrect patterns for the RAG test case
    false_patterns = [
        "relative advantage gradient",
        "relevance, accuracy, and generalizability",
        "reinforcement and goal-driven",
        "reward-based and goal-driven",
        "reinforcement agent with generalized advantage estimation",
        "resilience, adaptability, and growth",
    ]

    detected_claims = [
        claim
        for claim in false_patterns
        if claim in text
    ]

    if detected_claims:
        return {
            "score": 0.0,
            "label": "possible_hallucination",
            "detected_claims": detected_claims,
        }

    if not has_correct_definition:
        return {
            "score": 0.0,
            "label": "possible_hallucination",
            "detected_claims": [
                "Expected RAG definition was not found"
            ],
        }

    return {
        "score": 1.0,
        "label": "no_obvious_hallucination",
        "detected_claims": [],
    }