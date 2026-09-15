import os
from PIL import Image
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from config import LLAVA_MODEL_NAME, OLLAMA_HOST

def process_image(image: Image.Image, prompt: str) -> str:
"""
vision_agent.py
----------------
Two entry points, kept intentionally separate because they serve
different use cases:

1. process_image(image, prompt) -- analyzes a PIL Image the user
   uploads DIRECTLY through the Streamlit sidebar (e.g. a receipt),
   with no Qdrant retrieval step. Used by app.py's upload panel.

2. vision_agent(query, top_k) -- embeds the query with CLIP, searches
   the `omnibrain_images` Qdrant collection for the most relevant
   PAGE IMAGE FROM THE INGESTED DOCUMENT, then sends that retrieved
   image (not a user upload) to LLaVA. This is the "Vision Agent" in
   the documented multi-agent architecture.

Requires:
  - Ollama installed and running locally (https://ollama.com),
    with the LLaVA model pulled once: `ollama pull llava`
  - pip install langchain-ollama langchain-core ollama pillow
  - For vision_agent(): the `omnibrain_images` Qdrant collection
    already populated, plus
    `pip install git+https://github.com/openai/CLIP.git`
    (NOT `pip install clip` -- see ingestion review notes for why).

--- Week 3, Task 2: instrumented for latency tracking ---
vision_agent() is split into a retrieval sub-step (CLIP embed + Qdrant
search) and a generation sub-step (the LLaVA call itself) -- these
have very different latency profiles (LLaVA reading an image is
typically much slower than a vector search), so lumping them into one
number would hide which part actually needs optimizing.
"""

import base64
import io
import sys
from pathlib import Path

import clip
import ollama
import torch
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient

# Ensures the repo root is on sys.path so `from src.observability import
# traced` resolves regardless of how this file is invoked -- see the
# matching comment in search_agent.py for the full explanation.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from config import (
        QDRANT_HOST,
        QDRANT_PORT,
        IMAGE_COLLECTION,
        CLIP_MODEL_NAME,
        LLAVA_MODEL_NAME,
        OLLAMA_HOST,
        PROJECT_ROOT,
    )
except ImportError:
    from agents.config import (
        QDRANT_HOST,
        QDRANT_PORT,
        IMAGE_COLLECTION,
        CLIP_MODEL_NAME,
        LLAVA_MODEL_NAME,
        OLLAMA_HOST,
        PROJECT_ROOT,
    )

from src.observability import traced

_clip_model = None
_clip_preprocess = None
_qdrant_client = None
_ollama_client = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _pil_to_data_uri(image: Image.Image) -> str:
    """Encodes a PIL Image as a base64 data URI LangChain/Ollama can accept."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@traced("vision_agent.process_uploaded_image", as_type="generation")
def process_image(image: Image.Image, prompt: str) -> str:
    """
    Directly analyzes a user-uploaded image (e.g. a receipt) -- no
    Qdrant retrieval involved. Used by app.py's sidebar upload panel.
    """
    try:
        if image is None:
            return "Please upload an image first for vision analysis."

        # Initialize local LLaVA model via Ollama (No API key needed)
        llm = ChatOllama(model=LLAVA_MODEL_NAME, base_url=OLLAMA_HOST, temperature=0.2)

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt if prompt else "Analyze this image and extract all relevant numerical data, text, or chart details."},
                {"type": "image_url", "image_url": image}
                {
                    "type": "text",
                    "text": prompt if prompt else "Analyze this image and extract all relevant numerical data, text, or chart details.",
                },
                {"type": "image_url", "image_url": _pil_to_data_uri(image)},
            ]
        )

        response = llm.invoke([message])
        return response.content

    except Exception as e:
        return f"Error in Vision Agent: {str(e)}"
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
@traced("vision_agent.retrieve_image", as_type="retriever")
def _retrieve_image(query: str, top_k: int):
    model, _ = _get_clip()
    qdrant = _get_qdrant()

    text_tokens = clip.tokenize([query]).to(_device)
    with torch.no_grad():
        query_vector = model.encode_text(text_tokens)
    query_vector = query_vector.cpu().numpy()[0].tolist()

    hits = qdrant.query_points(
    return qdrant.query_points(
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


@traced("vision_agent.read_with_llava", as_type="generation")
def _read_image_with_llava(image_path: str, query: str) -> str:
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
    return response["message"]["content"]


@traced("vision_agent", as_type="retriever")
def vision_agent(query: str, top_k: int = 1) -> dict:
    """
    Retrieves the most relevant image from the ingested document (via
    CLIP + Qdrant) for a natural-language query, then reads it with
    LLaVA. Distinct from process_image() above, which skips retrieval
    entirely and analyzes a directly-uploaded image instead.
    """
    hits = _retrieve_image(query, top_k)

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

    answer = _read_image_with_llava(image_path, query)
    
    visual_source = {
    "image_path": file_path,
    "page_number": top_hit.payload.get("page_number"),
    "source_file": top_hit.payload.get("source_file"),
    "kind": top_hit.payload.get("kind"),
}

    return {
    "agent": "vision",
    "query": query,
    "answer": answer,
    "image_used": top_hit.payload.get("image_id"),
    "visual_source": {
        "image_path": file_path,
        "page_number": top_hit.payload.get("page_number"),
        "source_file": top_hit.payload.get("source_file"),
        "kind": top_hit.payload.get("kind"),
    },
    "citations": [
        {
            "page_number": top_hit.payload.get("page_number"),
            "source_file": top_hit.payload.get("source_file"),
            "kind": top_hit.payload.get("kind"),
            "image_path": file_path,
        }
    ],
}


if __name__ == "__main__":
    import json
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What does the revenue chart show?"
    print(json.dumps(vision_agent(query), indent=2))
