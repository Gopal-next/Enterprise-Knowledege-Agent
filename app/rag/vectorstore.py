import os
from pathlib import Path
from langchain_community.vectorstores import FAISS

try:
    from app.rag.embeddings import get_embedding_model
except ImportError:
    from rag.embeddings import get_embedding_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = PROJECT_ROOT / "Data" / "faiss_index"


def get_existing_vectorstore(embedding_model=None):
    """
    Connects to the persistent vector store on disk.
    Returns None if the index has not been created yet.
    """
    index_file = VECTOR_STORE_DIR / "index.faiss"
    if not index_file.exists():
        return None

    embeddings = embedding_model or get_embedding_model()
    return FAISS.load_local(
        folder_path=str(VECTOR_STORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def add_documents_to_vectorstore(chunks, embedding_model=None, batch_size=20):
    """
    Adds new document chunks to the persistent FAISS vector store on disk.
    Processes in batches with automatic rate-limit backoff and retry.
    """
    import time
    import math

    if not chunks:
        return get_existing_vectorstore(embedding_model)

    embeddings = embedding_model or get_embedding_model()
    existing_store = get_existing_vectorstore(embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = len(chunks)
    total_batches = math.ceil(total_chunks / batch_size)

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
                    print(f"Rate limit hit on batch {batch_idx + 1}/{total_batches}. Waiting {retry_delay:.0f}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    raise e
        else:
            raise RuntimeError(f"Failed to ingest batch {batch_idx + 1} after {max_retries} retries.")

        # Brief pause between batches to smooth out API request pacing
        if batch_idx < total_batches - 1:
            time.sleep(1.0)

    if store is not None:
        store.save_local(str(VECTOR_STORE_DIR))

    return store


def create_vectorstore(chunks, embedding_model=None):
    """
    Legacy wrapper for backward-compatibility.
    Saves chunks persistently and returns the vector store.
    """
    return add_documents_to_vectorstore(chunks, embedding_model)
