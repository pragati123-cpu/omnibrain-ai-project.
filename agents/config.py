"""
config.py
---------
Shared configuration for all sub-agents: Qdrant connection details,
collection names, and project-root path resolution.

Path resolution uses Path(__file__).resolve() rather than a plain
relative string -- this is the exact fix that was needed earlier in
the ingestion/vector_store scripts, applied here from the start so
these agents work regardless of which directory they're run from.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Qdrant connection -- matches the teammate's vector_store setup
# (a standalone Qdrant server via `docker run -p 6333:6333 qdrant/qdrant`)
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

TEXT_COLLECTION = "omnibrain_text"
IMAGE_COLLECTION = "omnibrain_images"

TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"   # must match the model used at ingestion time
CLIP_MODEL_NAME = "ViT-B/32"             # must match embed_images.py

# Vision Agent: local LLaVA via Ollama (open-source, no API key needed)
LLAVA_MODEL_NAME = os.environ.get("LLAVA_MODEL_NAME", "llava")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")  # ollama's default

# SQL Agent: still uses GPT-4o for Text-to-SQL translation
OPENAI_MODEL_SQL = "gpt-4o"

STOCK_DB_PATH = PROJECT_ROOT / "data" / "stock_data" / "historical_stock_prices.db"
