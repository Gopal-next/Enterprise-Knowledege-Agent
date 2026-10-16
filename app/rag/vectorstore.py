import os
import time
import math
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

try:
    from app.rag.embeddings import get_embedding_model
except ImportError:
    from rag.embeddings import get_embedding_model

QDRANT_COLLECTION = "enterprise_knowledge"
EMBEDDING_DIM = 3072  # Matches gemini-embedding-2
VECTOR_STORE_DIR = PROJECT_ROOT / "Data" / "faiss_index"


def is_qdrant_configured() -> bool:
    """Checks whether valid Qdrant Cloud credentials exist in .env."""
    url = os.getenv("QDRANT_URL", "").strip()
    key = os.getenv("QDRANT_API_KEY", "").strip()
    return bool(url and key and "xyz-example" not in url)


def get_qdrant_client():
    """Initializes and returns an authenticated QdrantClient."""
    if not is_qdrant_configured():
        return None
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(
            url=os.getenv("QDRANT_URL", "").strip(),
            api_key=os.getenv("QDRANT_API_KEY", "").strip(),
        )
    except Exception as e:
        print(f"Warning: Failed to initialize QdrantClient: {e}")
        return None


def get_qdrant_vectorstore(embedding_model=None):
    """
    Returns a LangChain QdrantVectorStore connected to Qdrant Cloud.
    Ensures the target collection exists with 3072 dimensions.
    """
    client = get_qdrant_client()
    if client is None:
        return None

    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client.models import Distance, VectorParams

        if not client.collection_exists(QDRANT_COLLECTION):
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            print(f"Created Qdrant Cloud collection: {QDRANT_COLLECTION} (dim={EMBEDDING_DIM})")

        embeddings = embedding_model or get_embedding_model()
        return QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
        )
    except Exception as e:
        print(f"Warning: Failed to initialize QdrantVectorStore: {e}")
        return None


def get_existing_vectorstore(embedding_model=None):
    """
    Connects to the persistent vector store.
    Prioritizes Qdrant Cloud if configured, falling back to local FAISS.
    """
    # 1. Try Qdrant Cloud first
    if is_qdrant_configured():
        qstore = get_qdrant_vectorstore(embedding_model)
        if qstore is not None:
            try:
                client = get_qdrant_client()
                info = client.get_collection(QDRANT_COLLECTION)
                # If Qdrant is connected and ready (has points or is collection-ready)
                if info.points_count > 0:
                    return qstore
            except Exception as e:
                print(f"Notice: Qdrant check returned: {e}")
            # If Qdrant is configured but empty yet, still return it so caller can use it
            return qstore

    # 2. Fallback to local FAISS
    index_file = VECTOR_STORE_DIR / "index.faiss"
    if index_file.exists():
        try:
            from langchain_community.vectorstores import FAISS
            embeddings = embedding_model or get_embedding_model()
            return FAISS.load_local(
                folder_path=str(VECTOR_STORE_DIR),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"Notice: Failed to load local FAISS index: {e}")

    return None


def add_documents_to_vectorstore(chunks, embedding_model=None, batch_size=20):
    """
    Adds document chunks directly to Qdrant Cloud (or FAISS if Qdrant not available).
    Includes rate-limit retry and batching.
    """
    import time
    import math

    if not chunks:
        return get_existing_vectorstore(embedding_model)

    embeddings = embedding_model or get_embedding_model()
    total_chunks = len(chunks)
    total_batches = math.ceil(total_chunks / batch_size)

    # Branch A: Qdrant Cloud Ingestion
    if is_qdrant_configured():
        qstore = get_qdrant_vectorstore(embeddings)
        if qstore is not None:
            print(f"--> Ingesting {total_chunks} chunks directly into Qdrant Cloud ({QDRANT_COLLECTION})...")
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_chunks)
                batch = chunks[start_idx:end_idx]

                max_retries = 4
                retry_delay = 15.0

                for attempt in range(max_retries):
                    try:
                        qstore.add_documents(batch)
                        print(f"  [Qdrant] Batch {batch_idx + 1}/{total_batches} indexed ({len(batch)} chunks).")
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                            print(f"Rate limit on batch {batch_idx + 1}. Waiting {retry_delay:.0f}s before retry (attempt {attempt + 1}/{max_retries})...")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5
                        else:
                            raise e
                else:
                    raise RuntimeError(f"Failed to ingest batch {batch_idx + 1} to Qdrant after {max_retries} retries.")

                if batch_idx < total_batches - 1:
                    time.sleep(1.0)

            return qstore

    # Branch B: Local FAISS Ingestion (Fallback)
    from langchain_community.vectorstores import FAISS
    existing_store = None
    index_file = VECTOR_STORE_DIR / "index.faiss"
    if index_file.exists():
        try:
            existing_store = FAISS.load_local(
                folder_path=str(VECTOR_STORE_DIR),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            existing_store = None

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    store = existing_store

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_chunks)
        batch = chunks[start_idx:end_idx]

        max_retries = 4
        retry_delay = 15.0

        for attempt in range(max_retries):
            try:
                if store is None:
                    store = FAISS.from_documents(documents=batch, embedding=embeddings)
                else:
                    store.add_documents(batch)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    print(f"Rate limit on batch {batch_idx + 1}. Waiting {retry_delay:.0f}s before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    raise e
        else:
            raise RuntimeError(f"Failed to ingest batch {batch_idx + 1} after {max_retries} retries.")

        if batch_idx < total_batches - 1:
            time.sleep(1.0)

    if store is not None:
        store.save_local(str(VECTOR_STORE_DIR))

    return store


def create_vectorstore(chunks, embedding_model=None):
    """
    Legacy wrapper for backward-compatibility.
    Saves chunks persistently into Qdrant Cloud / FAISS and returns the vector store.
    """
    return add_documents_to_vectorstore(chunks, embedding_model)

