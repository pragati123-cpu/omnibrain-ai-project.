from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="localhost", port=6333)

# Text collection
client.recreate_collection(
    collection_name="text_embeddings",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Image collection
client.recreate_collection(
    collection_name="image_embeddings",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

print("Collections created successfully!")