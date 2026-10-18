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

from app.rag.hybrid_retriever import global_hybrid_retriever

PDF_FOLDER = PROJECT_ROOT / "Data" / "pdfs"


def ingest_pdf(file_path: str) -> int:
    """
    Ingests a single PDF file into the persistent vector store (Qdrant/FAISS)
    and synchronizes in-memory BM25 index for hybrid retrieval.
    """
    docs = load_pdf(str(file_path))
    if not docs:
        return 0

    chunks = split_documents(docs)
    add_documents_to_vectorstore(chunks)
    global_hybrid_retriever.sync_corpus(chunks)
    return len(chunks)


def ingest_all_pdfs(folder_path: str = None) -> int:
    """
    Scans the given folder (default: Data/pdfs) and ingests all PDFs
    into persistent vector store and BM25 index.
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
    global_hybrid_retriever.sync_corpus(chunks)
    return len(chunks)


def _get_relevant_documents(query: str):
    """Retrieve top-k documents using Hybrid Search (Qdrant Dense + BM25 Sparse RRF)."""
    vectorstore = get_existing_vectorstore()
    if vectorstore is None:
        count = ingest_all_pdfs()
        if count == 0:
            return []
        vectorstore = get_existing_vectorstore()

    return global_hybrid_retriever.retrieve(query, vectorstore=vectorstore, top_k=4)


def get_retriever_context(query: str) -> str:
    """Retrieve and format context documents for auditing and evaluation."""
    docs = _get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in docs)


SYSTEM_RAG_PROMPT = """
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


def retriever_qa(query: str) -> str:
    """
    Queries the persistent vector store using Hybrid Search and generates
    an answer via Gemini with end-to-end LangSmith observability.
    """
    context = get_retriever_context(query)
    if not context:
        return "No relevant company documents found to answer this question. Please upload a PDF first."

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_RAG_PROMPT),
        ("human", "{question}")
    ])

    rag_chain = prompt | llm | StrOutputParser()

    try:
        response = rag_chain.invoke({"context": context, "question": query})
        return response
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            return "Your Gemini quota has finished, please check back later."
        raise e


def stream_retriever_qa(query: str):
    """
    Generator that streams LLM tokens in real-time chunk-by-chunk
    using Gemini streaming API (Server-Sent Events pattern).
    """
    context = get_retriever_context(query)
    if not context:
        yield "No relevant company documents found to answer this question. Please upload a PDF first."
        return

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_RAG_PROMPT),
        ("human", "{question}")
    ])

    formatted = prompt.format_messages(context=context, question=query)

    try:
        for chunk in llm.stream(formatted):
            token = chunk.content
            if token:
                yield token
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            yield "Your Gemini quota has finished, please check back later."
        else:
            yield f"\n\n⚠️ Error during streaming: {err_msg}"
