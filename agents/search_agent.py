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

--- Week 3, Task 2: instrumented for latency tracking ---
Split into two traced sub-steps (embedding vs. the actual Qdrant
query) rather than tracing the whole function as one block, so
latency data actually tells you WHICH part is slow -- e.g. a slow
embedding model load on first call vs. a slow/overloaded Qdrant
server are very different problems with very different fixes.
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Ensures the repo root is on sys.path so `from src.observability import
# traced` resolves correctly regardless of HOW this file is run --
# standalone (`cd agents && python search_agent.py`), imported as a
# package (`from agents.search_agent import search_agent`), or via
# pytest. Without this, `src` and `agents` are just sibling folders
# with no reliable path relationship to each other.
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Works both when run standalone (`cd agents && python search_agent.py`,
# where `config` is a top-level module) and when imported as a package
# from elsewhere (`from agents.search_agent import search_agent`, e.g.
# from src/graph.py, where it must be `agents.config`).
try:
    from config import QDRANT_HOST, QDRANT_PORT, TEXT_COLLECTION, TEXT_EMBED_MODEL
except ImportError:
    from agents.config import QDRANT_HOST, QDRANT_PORT, TEXT_COLLECTION, TEXT_EMBED_MODEL

from src.observability import traced

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


@traced("search_agent.embed_query", as_type="embedding")
def _embed_query(query: str) -> list:
    model = _get_model()
    return model.encode(query).tolist()


@traced("search_agent.qdrant_query", as_type="retriever")
def _query_qdrant(query_vector: list, top_k: int):
    client = _get_client()
    return client.query_points(
        collection_name=TEXT_COLLECTION,
        query=query_vector,
        limit=top_k,
    ).points


@traced("search_agent", as_type="retriever")
def search_agent(query: str, top_k: int = 3) -> dict:
    query_vector = _embed_query(query)
    hits = _query_qdrant(query_vector, top_k)

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
