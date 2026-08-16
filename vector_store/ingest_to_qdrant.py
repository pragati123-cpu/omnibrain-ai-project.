from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# -----------------------------------
# Step 1: Load embedding model
# -----------------------------------
text_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

# -----------------------------------
# Step 2: Connect Qdrant
# -----------------------------------
client = QdrantClient(host="localhost", port=6333)
print("Connected to Qdrant!")

# -----------------------------------
# Step 3: Create collection (only once)
# -----------------------------------
if not client.collection_exists("omnibrain_text"):
    client.create_collection(
        collection_name="omnibrain_text",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
    print("Collection created!")
else:
    print("Collection already exists!")

# -----------------------------------
# Step 4: Sample text
# -----------------------------------
text = "Total revenue for fiscal year 2025 was $4.2 billion."

# Generate embedding
vector = text_model.encode(text)

print("Vector dimension:", len(vector))
print("First 5 values:", vector[:5])


    
# -----------------------------------
# Step 8: Load manifest.json
# -----------------------------------
import json

manifest_path = "../data/manifest.json"

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print("\nManifest loaded successfully!")
print("Keys:", manifest.keys())
print("Number of text chunks:", len(manifest.get("text_chunks", [])))
print("Number of images:", len(manifest.get("images", [])))

# -----------------------------------
# Step 9: Bulk text ingestion
# -----------------------------------

text_chunks = manifest.get("text_chunks", [])

print(f"\nStarting bulk ingestion of {len(text_chunks)} text chunks...")

points = []

for idx, chunk in enumerate(text_chunks, start=1):

    text = chunk.get("text", "").strip()

    if not text:
        continue

    # Generate embedding
    vector = text_model.encode(text)

    # Create point
    point = PointStruct(
        id=idx,
        vector=vector.tolist(),
        payload={
            "chunk_id": chunk.get("chunk_id"),
            "text": text,
            "page_number": chunk.get("page_number"),
            "source_file": chunk.get("source_file"),
            "chunk_index_on_page": chunk.get("chunk_index_on_page")
        }
    )

    points.append(point)

# Insert all points
client.upsert(
    collection_name="omnibrain_text",
    points=points
)

print(f"Successfully inserted {len(points)} text chunks into Qdrant!")

# -----------------------------------
# Step 10: Test semantic search
# -----------------------------------

query = "What was the revenue in 2025?"

query_vector = text_model.encode(query)

results = client.query_points(
    collection_name="omnibrain_text",
    query=query_vector.tolist(),
    limit=3
)

print("\nSearch Results:")

for point in results.points:
    print("-" * 50)
    print("Score:", point.score)
    print("Chunk ID:", point.payload.get("chunk_id"))
    print("Page:", point.payload.get("page_number"))
    print("Text:", point.payload.get("text")[:200])
    
