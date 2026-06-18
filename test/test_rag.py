import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.vectorstore import create_vectorstore
from app.service.rag_service import answer_question

docs = load_pdf(
    "D:\Enterprise knowledge agent\Data\pdfs\leave_policy.pdf"
)

chunks = split_documents(docs)

vectorstore = create_vectorstore(chunks)

retriever = vectorstore.as_retriever()

question = "How many leaves are per annum?"

answer = answer_question(
    question,
    retriever
)

print(answer)