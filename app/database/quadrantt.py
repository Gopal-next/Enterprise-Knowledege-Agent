import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = "enterprise_knowledge"
EMBEDDING_DIM = 3072  # Matches gemini-embedding-2

def get_qdrant_client() -> QdrantClient:
    """Returns an authenticated QdrantClient instance."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY missing from environment.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def init_collection(client: QdrantClient = None, collection_name: str = COLLECTION_NAME):
    """Ensures the collection exists with correct vector dimensions (3072)."""
    client = client or get_qdrant_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created successfully with size {EMBEDDING_DIM}.")
    return client

# Backward-compatible client instance
try:
    if QDRANT_URL and QDRANT_API_KEY:
        client = get_qdrant_client()
        init_collection(client)
    else:
        client = None
except Exception as e:
    client = None