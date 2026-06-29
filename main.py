import os
import time
import streamlit as st
from app.rag.retriever import retriever_qa
from app.service.sql_service import ask_database
from app.database.supabase_client import save_chat
from app.utils.logger import logger

st.set_page_config(
    page_title="Enterprise Knowledge Agent",
    layout="wide"
)

st.title("Enterprise Knowledge Agent")

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Chat Assistant",
        "Upload PDF",
        "Analytics"
    ]
)

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


pdf_count = len(
    [f for f in os.listdir("data/pdfs")
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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    source = st.radio(
        "Choose Source",
        ["PDF", "Database"]
    )

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
            try:
                if source == "PDF":
                    answer = retriever_qa(question)
                else:
                    answer = ask_database(question)
            except Exception as e:
                    
                logger.error(
                    f"Error: {str(e)}"
                )

                st.error("Something went wrong")

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

    uploaded_file = st.file_uploader(
        "Choose File",
        type=["pdf", "xlsx", "xls", "csv"]
    )

    if uploaded_file:

        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "pdf":
            save_folder = "data/pdfs"
        else:
            save_folder = "data/excelfile"

        os.makedirs(
            save_folder,
            exist_ok=True
        )

        filepath = os.path.join(
            save_folder,
            uploaded_file.name
        )

        with open(filepath, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        st.success(
            f"{uploaded_file.name} uploaded successfully"
        )

        st.write(
            f"Saved to: {filepath}"
        )

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
