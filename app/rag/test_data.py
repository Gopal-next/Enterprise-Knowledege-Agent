"""
Standalone test: put a PDF in data/pdfs/, run this, and it will
ingest it into Qdrant and then let you ask it a question — the full
pipeline end to end, without needing the Streamlit UI running.

Usage:
    python test_ingest_and_query.py
"""

import os
from retriever import ingest_pdfs, retriever_qa

PDF_FOLDER = "data/pdfs"


def main():
    os.makedirs(PDF_FOLDER, exist_ok=True)

    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs found in {PDF_FOLDER}/ — put a .pdf file there first.")
        return

    print(f"Found {len(pdfs)} PDF(s): {pdfs}")
    print("Ingesting (loading, chunking, embedding, storing in Qdrant)...")

    ingest_pdfs(PDF_FOLDER)

    print("\nIngestion done. Now ask it something about the document.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer = retriever_qa(question)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()