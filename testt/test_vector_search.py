import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.vectorstore import create_vectorstore
from app.rag.search import search_documents

docs = load_pdf(
    "D:\Enterprise knowledge agent\data\pdfs\leave_policy.pdf"
)

chunks = split_documents(docs)

vectorstore = create_vectorstore(chunks)

results = search_documents(
    vectorstore,
    "How many sick leaves are allowed?"
)

for r in results:
    print("\n")
    print(r.page_content)