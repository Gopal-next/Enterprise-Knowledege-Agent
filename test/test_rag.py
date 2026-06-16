from app.rag.loader import load_pdf
from app.rag.splitter import split_documents
from app.rag.vectorstore import create_vectorstore
from app.services.rag_service import answer_question

docs = load_pdf(
    "data/pdfs/company_leave_policy.pdf"
)

chunks = split_documents(docs)

vectorstore = create_vectorstore(chunks)

retriever = vectorstore.as_retriever()

question = "How many sick leaves are allowed?"

answer = answer_question(
    question,
    retriever
)

print(answer)