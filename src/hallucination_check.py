import re
from typing import Tuple


SAFE_FALLBACK = (
    "I can only answer questions based on the information "
    "available in the provided documents."
)


def normalize_text(text: str) -> str:
    """Convert text into a comparable lowercase format."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_answer(answer: str, context: str) -> Tuple[bool, str]:
    """
    Check whether the generated answer is supported by the
    retrieved document context.

    Returns:
        (True, answer) if the answer is sufficiently supported.
        (False, SAFE_FALLBACK) otherwise.
    """

    if not answer or not context:
        return False, SAFE_FALLBACK

    normalized_answer = normalize_text(answer)
    normalized_context = normalize_text(context)

    answer_words = set(normalized_answer.split())
    context_words = set(normalized_context.split())

    if not answer_words:
        return False, SAFE_FALLBACK

    # Common words provide little evidence of factual support.
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "of", "to", "in", "on", "and", "or", "for",
        "this", "that", "it", "with", "from", "as"
    }

    meaningful_words = answer_words - stop_words

    if not meaningful_words:
        return False, SAFE_FALLBACK

    matched_words = meaningful_words.intersection(context_words)

    support_ratio = len(matched_words) / len(meaningful_words)

    # At least 50% of meaningful answer words should
    # appear in the retrieved context.
    if support_ratio >= 0.5:
        return True, answer

    return False, SAFE_FALLBACK