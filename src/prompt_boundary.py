SAFE_FALLBACK = (
    "I can only answer questions based on the information "
    "available in the provided documents."
)


def build_grounded_prompt(context: str, question: str) -> str:
    """
    Build a prompt that restricts the LLM to the retrieved
    document context.
    """

    return f"""
You are a document-based question answering assistant.

IMPORTANT RULES:
1. Answer ONLY using the information provided in the context.
2. Do NOT use outside knowledge.
3. Do NOT guess or invent facts.
4. If the context does not contain enough information to answer
   the question, return exactly this message:

"{SAFE_FALLBACK}"

Retrieved Context:
{context}

User Question:
{question}

Answer:
""".strip()