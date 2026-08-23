"""
vision_agent.py
----------------
Vision Agent: handles queries about charts, graphs, and tables
embedded in the document. Embeds the query with CLIP's text encoder,
searches the `omnibrain_images` Qdrant collection for the most
relevant image, then sends that image + the user's question to a
LOCAL LLaVA model (via Ollama) to actually read and reason over the
numbers in it.

This is the piece the project spec is explicit about: tables and
charts are read by a VLM at query time, not parsed into structured
data ahead of time -- see ingestion/README.md's "Design decisions"
section for the reasoning behind that choice.

Why LLaVA over GPT-4o: open-source, runs locally, no per-call API
cost or external dependency -- worth it for a portfolio project where
you want the whole pipeline runnable without anyone else's API key.
Trade-off: LLaVA's numeric/OCR-style reading of charts and tables is
generally less accurate than GPT-4o's, so if you see the agent
misreading numbers, that's a known limitation of the model, not
necessarily a bug in this code -- worth testing against a few known
values before trusting it for a demo.

Requires:
  - Ollama installed and running locally (https://ollama.com)
  - The LLaVA model pulled once: `ollama pull llava`
  - the `omnibrain_images` Qdrant collection already populated
    (via the teammate's embed_images.py)

Note on the CLIP package: uses OpenAI's official `clip` (installed
via `pip install git+https://github.com/openai/CLIP.git`), matching
what embed_images.py uses -- NOT the unrelated `clip` package on
PyPI. See ingestion review notes for why that distinction matters.
"""

import clip
import torch
import ollama
from qdrant_client import QdrantClient

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    IMAGE_COLLECTION,
    CLIP_MODEL_NAME,
    LLAVA_MODEL_NAME,
    PROJECT_ROOT,
)

_clip_model = None
_clip_preprocess = None
_qdrant_client = None
_ollama_client = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load(CLIP_MODEL_NAME, device=_device)
    return _clip_model, _clip_preprocess


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


def _get_ollama():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(host=OLLAMA_HOST)
    return _ollama_client


def vision_agent(query: str, top_k: int = 1) -> dict:
    model, _ = _get_clip()
    qdrant = _get_qdrant()

    text_tokens = clip.tokenize([query]).to(_device)
    with torch.no_grad():
        query_vector = model.encode_text(text_tokens)
    query_vector = query_vector.cpu().numpy()[0].tolist()

    hits = qdrant.query_points(
        collection_name=IMAGE_COLLECTION,
        query=query_vector,
        limit=top_k,
    ).points

    if not hits:
        return {
            "agent": "vision",
            "query": query,
            "answer": "No relevant image found for this query.",
            "citations": [],
        }

    top_hit = hits[0]
    file_path = top_hit.payload.get("file_path")
    image_path = str(PROJECT_ROOT / file_path)

    response = _get_ollama().chat(
        model=LLAVA_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer this question using ONLY what's visible in the "
                    f"attached image. Be precise with numbers. Question: {query}"
                ),
                "images": [image_path],
            }
        ],
    )

    return {
        "agent": "vision",
        "query": query,
        "answer": response["message"]["content"],
        "image_used": top_hit.payload.get("image_id"),
        "citations": [
            {
                "page_number": top_hit.payload.get("page_number"),
                "source_file": top_hit.payload.get("source_file"),
                "kind": top_hit.payload.get("kind"),
            }
        ],
    }


if __name__ == "__main__":
    import json
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What does the revenue chart show?"
    print(json.dumps(vision_agent(query), indent=2))
