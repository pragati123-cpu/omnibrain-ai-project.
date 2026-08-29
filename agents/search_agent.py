"""
search_agent.py
----------------
Search Agent: handles queries answerable from the document's prose
text. Embeds the query with the same model used at ingestion time
(all-MiniLM-L6-v2), searches the `omnibrain_text` Qdrant collection,
and returns the top matching chunks with full citation metadata
(page number, source file).

This agent does NOT call an LLM itself -- its job is retrieval only.
Whoever wires the Supervisor together decides how to pass these
chunks into a final synthesis step.

Requires: the `omnibrain_text` Qdrant collection already populated
(via the teammate's ingest_to_qdrant.py).
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Works both when run standalone (`cd agents && python search_agent.py`,
# where `config` is a top-level module) and when imported as a package
# from elsewhere (`from agents.search_agent import search_agent`, e.g.
# from src/graph.py, where it must be `agents.config`).
try:
    from config import QDRANT_HOST, QDRANT_PORT, TEXT_COLLECTION, TEXT_EMBED_MODEL
except ImportError:
    from agents.config import QDRANT_HOST, QDRANT_PORT, TEXT_COLLECTION, TEXT_EMBED_MODEL

# Lazy-loaded singletons -- avoids reloading the model / reconnecting
# on every call if this module is imported once and reused (e.g. by
# the Supervisor graph), while still working fine when run standalone.
_text_model = None
_client = None


def _get_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(TEXT_EMBED_MODEL)
    return _text_model


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def search_agent(query: str, top_k: int = 3) -> dict:
    model = _get_model()
    client = _get_client()

    query_vector = model.encode(query).tolist()

    hits = client.query_points(
        collection_name=TEXT_COLLECTION,
        query=query_vector,
        limit=top_k,
    ).points

    results = [
        {
            "chunk_id": hit.payload.get("chunk_id"),
            "text": hit.payload.get("text"),
            "page_number": hit.payload.get("page_number"),
            "source_file": hit.payload.get("source_file"),
            "score": hit.score,
        }
        for hit in hits
    ]

    return {
        "agent": "search",
        "query": query,
        "results": results,
        "citations": [
            {"page_number": r["page_number"], "source_file": r["source_file"]}
            for r in results
        ],
    }


if __name__ == "__main__":
    import json
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What was the revenue in 2025?"
    print(json.dumps(search_agent(query), indent=2))
