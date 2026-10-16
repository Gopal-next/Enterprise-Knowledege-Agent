"""
Standalone test: confirms your Qdrant Cloud credentials work and the
connection is alive — no embeddings, no LangChain, just the raw client.

Usage:
    python test_qdrant_connection.py
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def main():
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("QDRANT_URL or QDRANT_API_KEY missing from .env — set both first.")
        return

    print(f"Connecting to: {QDRANT_URL}")

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # This actually hits the server — if credentials or the URL are
        # wrong, this is where it'll fail, not at client construction.
        collections = client.get_collections()

        print("Connected successfully!")
        print(f"Existing collections: {[c.name for c in collections.collections]}")

        if not collections.collections:
            print("No collections yet — that's expected if you haven't run ingestion yet.")

    except Exception as e:
        print("Connection FAILED.")
        print(f"Error: {e}")
        print(
            "\nCommon causes:\n"
            "- QDRANT_URL missing 'https://' or has a typo\n"
            "- QDRANT_API_KEY is wrong or was regenerated since you copied it\n"
            "- Cluster is still spinning up (wait a minute after creating it)\n"
        )


if __name__ == "__main__":
    main()