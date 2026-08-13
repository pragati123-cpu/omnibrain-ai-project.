from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

text_count = client.count(collection_name="text_embeddings")
image_count = client.count(collection_name="omnibrain_images")

print("Text vectors:", text_count.count)
print("Image vectors:", image_count.count)