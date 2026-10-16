from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document
from dotenv import load_dotenv

load_dotenv()

# connect to Qdrant Cloud
client = QdrantClient(
    url="https://xyz-example.eu-central.aws.cloud.qdrant.io",
    cloud_inference=True
)

client.create_collection(
    collection_name="items",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)