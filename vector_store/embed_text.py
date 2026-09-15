from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import json

# Qdrant connection
client = QdrantClient(host="localhost", port=6333)

# Text embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# manifest load
with open("../data/manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

points = []

for idx, chunk in enumerate(manifest["text_chunks"]):
    vector = model.encode(chunk["text"]).tolist()

    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "page_number": chunk["page_number"],
                "source_file": chunk["source_file"],
            },
        )
    )

client.upsert(collection_name="text_embeddings", points=points)

print(f"Stored {len(points)} text embeddings successfully!")