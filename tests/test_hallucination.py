import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hallucination_check import (
    SAFE_FALLBACK,
    validate_answer,
)


def test_supported_answer():
    context = (
        "Qdrant is a vector database used for storing embeddings."
    )

    answer = (
        "Qdrant is a vector database used for storing embeddings."
    )

    is_valid, result = validate_answer(answer, context)

    assert is_valid is True
    assert result == answer


def test_hallucinated_answer():
    context = (
        "Qdrant is a vector database used for storing embeddings."
    )

    answer = (
        "Qdrant was founded in 2019 and has 500 employees."
    )

    is_valid, result = validate_answer(answer, context)

    assert is_valid is False
    assert result == SAFE_FALLBACK


def test_empty_context():
    context = ""
    answer = "Qdrant is a vector database."

    is_valid, result = validate_answer(answer, context)

    assert is_valid is False
    assert result == SAFE_FALLBACK