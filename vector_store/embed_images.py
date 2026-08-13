import json
from pathlib import Path
import clip
import torch
from PIL import Image

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# -----------------------------------
# Step 1: Load manifest
# -----------------------------------

manifest_path = "../data/manifest.json"

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

images = manifest.get("images", [])

print(f"Found {len(images)} images")

# -----------------------------------
# Step 2: Load CLIP model
# -----------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

print("CLIP model loaded successfully!")

# -----------------------------------
# Step 3: Connect Qdrant
# -----------------------------------

client = QdrantClient(host="localhost", port=6333)

print("Connected to Qdrant!")

# -----------------------------------
# Step 4: Create image collection
# -----------------------------------

if not client.collection_exists("omnibrain_images"):
    client.create_collection(
        collection_name="omnibrain_images",
        vectors_config=VectorParams(
            size=512,
            distance=Distance.COSINE
        )
    )
    print("Image collection created!")
else:
    print("Image collection already exists!")

# -----------------------------------
# Step 5: Prepare paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

points = []

# -----------------------------------
# Step 6: Embed all images
# -----------------------------------

for idx, img in enumerate(images, start=1):

    image_path = PROJECT_ROOT / img["file_path"]

    print(f"Embedding image {idx}: {image_path.name}")

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_vector = model.encode_image(image)

    image_vector = image_vector.cpu().numpy()[0]

    point = PointStruct(
        id=idx,
        vector=image_vector.tolist(),
        payload={
            "image_id": img.get("image_id"),
            "page_number": img.get("page_number"),
            "source_file": manifest.get("source_file"),
            "file_path": img.get("file_path"),
            "kind": img.get("kind")
        }
    )

    points.append(point)

# -----------------------------------
# Step 7: Insert into Qdrant
# -----------------------------------

client.upsert(
    collection_name="omnibrain_images",
    points=points
)

print(f"\nSuccessfully inserted {len(points)} image vectors into Qdrant!")

# -----------------------------------
# Step 8: Simple image retrieval test
# -----------------------------------

query = "revenue chart"

text_tokens = clip.tokenize([query]).to(device)

with torch.no_grad():
    query_vector = model.encode_text(text_tokens)

query_vector = query_vector.cpu().numpy()[0]

results = client.query_points(
    collection_name="omnibrain_images",
    query=query_vector.tolist(),
    limit=3
)

print("\nImage Search Results:")

for point in results.points:
    print("-" * 50)
    print("Score:", point.score)
    print("Image ID:", point.payload.get("image_id"))
    print("Kind:", point.payload.get("kind"))
    print("File:", point.payload.get("file_path"))