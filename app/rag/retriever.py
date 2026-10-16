import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

try:
    from app.rag.loader import load_pdf
    from app.rag.splitter import split_documents
    from app.rag.vectorstore import (
        get_existing_vectorstore,
        add_documents_to_vectorstore,
    )
except ImportError:
    from rag.loader import load_pdf
    from rag.splitter import split_documents
    from rag.vectorstore import (
        get_existing_vectorstore,
        add_documents_to_vectorstore,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Explicitly load .env with absolute path so environment is always present
load_dotenv(PROJECT_ROOT / ".env", override=True)
os.environ["LANGSMITH_DISABLE_RUN_COMPRESSION"] = "true"

PDF_FOLDER = PROJECT_ROOT / "Data" / "pdfs"


def ingest_pdf(file_path: str) -> int:
    """
    Ingests a single PDF file into the persistent FAISS vector store.
    Returns the number of chunks indexed.
    """
    docs = load_pdf(str(file_path))
    if not docs:
        return 0

    chunks = split_documents(docs)
    add_documents_to_vectorstore(chunks)
    return len(chunks)


def ingest_all_pdfs(folder_path: str = None) -> int:
    """
    Scans the given folder (default: Data/pdfs) and ingests all PDFs
    into the persistent vector store. Returns total chunk count.
    """
    target_folder = Path(folder_path) if folder_path else PDF_FOLDER
    if not target_folder.exists():
        return 0

    all_docs = []
    for file in os.listdir(target_folder):
        if file.lower().endswith(".pdf"):
            pdf_path = target_folder / file
            docs = load_pdf(str(pdf_path))
            all_docs.extend(docs)

    if not all_docs:
        return 0

    chunks = split_documents(all_docs)
    add_documents_to_vectorstore(chunks)
    return len(chunks)


def retriever_qa(query: str) -> str:
    """
    Queries the persistent vector store and generates an answer via Gemini.
    Uses modern LCEL for native, end-to-end LangSmith observability.
    """
    vectorstore = get_existing_vectorstore()

    # Graceful fallback: If vector store doesn't exist yet, auto-ingest existing PDFs once
    if vectorstore is None:
        count = ingest_all_pdfs()
        if count == 0:
            return "No documents have been indexed yet. Please upload a PDF file first."
        vectorstore = get_existing_vectorstore()

    retriever_obj = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash"
    )

    system_prompt = """
        You are an intelligent Enterprise Knowledge AI Assistant.

        Your primary responsibility is to answer user questions accurately using the provided context from company documents.

        Guidelines:
        1. Base your answer primarily on the provided context.
        2. If the context does not fully answer the question, provide helpful general guidance while clearly distinguishing it from company-specific facts.
        3. Never invent company policies, pricing, dates, or numbers not present in the context.
        4. Keep answers clear, conversational, and professional.
        5. If the user asks in another language (e.g. Hindi), answer naturally in that respective language.
        6. Do not explicitly say "according to the retrieved context". Answer naturally as a knowledgeable assistant.

        Company Context:
        {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL pipeline: natively traces retriever, prompt, model, and output to LangSmith
    rag_chain = (
        {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            return "Your Gemini quota has finished, please check back later."
        raise e
