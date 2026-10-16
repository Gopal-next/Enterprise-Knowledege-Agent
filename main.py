import os
import time
from dotenv import load_dotenv

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
os.environ["LANGSMITH_DISABLE_RUN_COMPRESSION"] = "true"

import streamlit as st
from app.rag.retriever import retriever_qa, ingest_pdf
from app.service.sql_service import ask_database
from app.database.supabase_client import save_chat
from app.utils.logger import logger

st.set_page_config(
    page_title="Enterprise Knowledge Agent",
    layout="wide"
)

st.title("Knowledge Agent")

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Chat Assistant",
        "Upload File",
        "Analytics"
    ]
)

st.sidebar.divider()
ls_key = os.getenv("LANGCHAIN_API_KEY", "")
if os.getenv("LANGCHAIN_TRACING_V2") == "true" and ls_key and ls_key != "your_langsmith_api_key_here":
    st.sidebar.caption("🟢 **Observability:** LangSmith Active")
else:
    st.sidebar.caption("⚪ **Observability:** LangSmith (Add key in .env)")

qdrant_url = os.getenv("QDRANT_URL", "").strip()
if qdrant_url and "qdrant.io" in qdrant_url:
    st.sidebar.caption("🟢 **Vector DB:** Qdrant Cloud Active")
else:
    st.sidebar.caption("📁 **Vector DB:** Local FAISS")


if "questions_count" not in st.session_state:
    st.session_state.questions_count = 0

if "pdf_queries" not in st.session_state:
    st.session_state.pdf_queries = 0

if "db_queries" not in st.session_state:
    st.session_state.db_queries = 0

if "total_response_time" not in st.session_state:
    st.session_state.total_response_time = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []

st.sidebar.caption(f"📄 **Active Documents:** {len(st.session_state.uploaded_pdfs)}/2 PDFs")

pdf_dir = "Data/pdfs" if os.path.exists("Data/pdfs") else "data/pdfs"
os.makedirs(pdf_dir, exist_ok=True)
pdf_count = len(
    [f for f in os.listdir(pdf_dir)
     if f.endswith(".pdf")]
)

if st.session_state.questions_count > 0:
    avg_time = (
        st.session_state.total_response_time /
        st.session_state.questions_count
    )
else:
    avg_time = 0


if menu == "Chat Assistant":

    st.header("Chat Assistant")

    source = st.radio(
        "Choose Source",
        ["PDF", "Database"],
        horizontal=True
    )

    can_chat = True

    if source == "PDF":
        pdf_mode = st.radio(
            "Document Option",
            [
                "🏢 Talk with Pre-Existing Enterprise Data (Demo / Privacy Mode — No Upload Needed)",
                "📤 Upload Your Own Custom Document (Max 2 PDFs)"
            ],
            index=0 if len(st.session_state.uploaded_pdfs) == 0 else 1
        )

        if pdf_mode.startswith("🏢"):
            st.info("🛡️ **Privacy Assured:** You are chatting with pre-existing sample company documents (Leave Policy & Employee Handbook). No personal or confidential files are requested or uploaded.")
            st.caption("💡 *Try asking: 'What is the leave policy?' • 'How many annual leaves do employees get?' • 'What are the office guidelines?'*")
            can_chat = True

        else:
            if len(st.session_state.uploaded_pdfs) == 0:
                can_chat = False
                st.warning("🔒 **Upload required:** Please upload a PDF below to start chatting with your own document. (Maximum 2 PDFs allowed).")

                quick_pdf = st.file_uploader(
                    "Upload PDF to unlock chat (Allowed: up to 2 PDFs)",
                    type=["pdf"],
                    key="chat_pdf_uploader"
                )
                if quick_pdf:
                    save_folder = "Data/pdfs" if os.path.exists("Data/pdfs") else "data/pdfs"
                    os.makedirs(save_folder, exist_ok=True)
                    filepath = os.path.join(save_folder, quick_pdf.name)
                    with open(filepath, "wb") as f:
                        f.write(quick_pdf.getbuffer())

                    with st.spinner("Indexing document into Qdrant Vector Database..."):
                        num_chunks = ingest_pdf(filepath)

                    if quick_pdf.name not in st.session_state.uploaded_pdfs:
                        st.session_state.uploaded_pdfs.append(quick_pdf.name)

                    st.success(f"✅ {quick_pdf.name} uploaded & indexed ({num_chunks} chunks). Chat unlocked!")
                    st.rerun()

            else:
                col_docs, col_reset = st.columns([3, 1])
                with col_docs:
                    docs_badge = " • ".join([f"📄 `{p}`" for p in st.session_state.uploaded_pdfs])
                    st.caption(f"**Your Active Documents ({len(st.session_state.uploaded_pdfs)}/2):** {docs_badge}")
                with col_reset:
                    if st.button("🔄 Reset PDFs"):
                        st.session_state.uploaded_pdfs = []
                        st.session_state.messages = []
                        st.rerun()

                if len(st.session_state.uploaded_pdfs) < 2:
                    with st.expander("➕ Add second PDF (Optional, max 2 total)"):
                        second_pdf = st.file_uploader("Upload 2nd PDF", type=["pdf"], key="second_pdf_uploader")
                        if second_pdf and second_pdf.name not in st.session_state.uploaded_pdfs:
                            save_folder = "Data/pdfs" if os.path.exists("Data/pdfs") else "data/pdfs"
                            os.makedirs(save_folder, exist_ok=True)
                            filepath = os.path.join(save_folder, second_pdf.name)
                            with open(filepath, "wb") as f:
                                f.write(second_pdf.getbuffer())
                            with st.spinner("Indexing second PDF into Qdrant..."):
                                num_chunks = ingest_pdf(filepath)
                            st.session_state.uploaded_pdfs.append(second_pdf.name)
                            st.success(f"✅ Added {second_pdf.name} ({num_chunks} chunks)!")
                            st.rerun()

    if can_chat:
        # Display old messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask a Question")

        if question:
            st.session_state.questions_count += 1

            if source == "PDF":
                st.session_state.pdf_queries += 1
            else:
                st.session_state.db_queries += 1

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )
            logger.info(f"Question Received: {question}")
            logger.info(f"Tool Used: {source}")
            with st.chat_message("user"):
                st.write(question)

            with st.spinner("Thinking..."):
                start_time = time.time()
                answer = "Sorry, I could not generate an answer at this moment."
                try:
                    if source == "PDF":
                        answer = retriever_qa(question)
                    else:
                        answer = ask_database(question)
                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Error: {err_msg}")
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                        answer = "Your Gemini quota has finished, please check back later."
                    else:
                        answer = f"⚠️ Something went wrong: {err_msg}"
                        st.error(answer)

                end_time = time.time()

                response_time = end_time - start_time
                logger.info(f"Response Time: {response_time:.2f} sec")
                logger.info(f"Answer of question: {answer}")

                st.session_state.total_response_time += response_time

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            save_chat(
                question=question,
                answer=answer,
                tool_used=source,
                response_time=response_time
            )       

            with st.chat_message("assistant"):
                st.write(answer)

            st.caption(f"Tool Used: {source}")
            st.write(f"Tool Used: {source} retriever")

elif menu == "Upload File":

    st.header("Upload PDF / Excel / CSV")

    current_pdf_count = len(st.session_state.uploaded_pdfs)
    st.info(f"📄 **PDF Upload Status:** {current_pdf_count}/2 allowed PDFs active in current session.")

    if current_pdf_count >= 2:
        st.warning("⚠️ **Maximum 2 PDFs reached.** You can now chat with your documents, or reset below to upload different ones.")
        if st.button("🔄 Reset Uploaded PDFs", key="reset_from_upload_page"):
            st.session_state.uploaded_pdfs = []
            st.session_state.messages = []
            st.rerun()

    uploaded_file = st.file_uploader(
        "Choose File",
        type=["pdf", "xlsx", "xls", "csv"]
    )

    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "pdf" and current_pdf_count >= 2 and uploaded_file.name not in st.session_state.uploaded_pdfs:
            st.error("❌ Cannot upload more than 2 PDFs. Please reset existing PDFs first.")
        else:
            if file_extension == "pdf":
                save_folder = "Data/pdfs" if os.path.exists("Data/pdfs") else "data/pdfs"
            else:
                save_folder = "Data/excelfile" if os.path.exists("Data/excelfile") else "data/excelfile"

            os.makedirs(save_folder, exist_ok=True)
            filepath = os.path.join(save_folder, uploaded_file.name)

            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"✅ {uploaded_file.name} uploaded successfully!")

            if file_extension == "pdf":
                with st.spinner("Indexing document into Qdrant Vector Database..."):
                    num_chunks = ingest_pdf(filepath)

                if uploaded_file.name not in st.session_state.uploaded_pdfs:
                    st.session_state.uploaded_pdfs.append(uploaded_file.name)

                st.info(f"⚡ Ingestion Complete: {num_chunks} chunks indexed. Active PDFs: {len(st.session_state.uploaded_pdfs)}/2")

            st.write(f"Saved to: `{filepath}`")

elif menu == "Analytics":

    st.header("Analytics Dashboard")

    col1, col2, col3 , col4= st.columns(4)

    with col1:
        st.metric(
            "Documents",
            pdf_count
        )

    with col2:
        st.metric(
            "Questions",
            st.session_state.questions_count
        )

    with col3:
        st.metric(
            "PDF Queries",
            st.session_state.pdf_queries
        )
    with col4:
        st.metric(
            "Database Queries",
            st.session_state.db_queries
        )
    
    st.metric(
        "Average Search Time",
        f"{avg_time:.2f} sec"
    )
    if st.session_state.pdf_queries > st.session_state.db_queries:
        most_used = "PDF Retriever"
    else:
        most_used = "SQL Database"

    st.write(f"Most Used Tool: {most_used}")
