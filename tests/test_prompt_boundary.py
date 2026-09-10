import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_boundary import (
    SAFE_FALLBACK,
    build_grounded_prompt,
)


def test_prompt_contains_context_and_question():
    context = "Qdrant is a vector database."
    question = "What is Qdrant?"

    prompt = build_grounded_prompt(context, question)

    assert context in prompt
    assert question in prompt


def test_prompt_contains_boundary_rules():
    context = "Qdrant is a vector database."
    question = "What is Qdrant?"

    prompt = build_grounded_prompt(context, question)

    assert "ONLY using the information provided in the context" in prompt
    assert "Do NOT use outside knowledge" in prompt
    assert "Do NOT guess or invent facts" in prompt


def test_prompt_contains_safe_fallback():
    context = "Qdrant is a vector database."
    question = "Who founded Qdrant?"

    prompt = build_grounded_prompt(context, question)

    assert SAFE_FALLBACK in prompt