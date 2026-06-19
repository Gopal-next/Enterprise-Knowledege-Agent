# from fastapi import FastAPI

# from api.upload import router as upload_router

# app = FastAPI(title="Enterprise Knowledge Agent")

# app.include_router(upload_router)

# @app.get("/")
# def home():
#     return {
#         "message": "Enterprise Knowledge Agent Running"
#     }


# from fastapi import FastAPI

# app = FastAPI(
#     title="Enterprise Knowledge Agent"
# )

# @app.get("/")
# def home():
#     return {
#         "message": "API Running"
#     }


# from app.api.routes.health import router as health_router

# app.include_router(
#     health_router
# )



import streamlit as st
import os
from rag.retriever import retriever_qa
# from service.rag_service import answer_question
# from rag.vectorstore import create_vectorstore
# from service.sql_service import ask_database
import time

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

if "questions_count" not in st.session_state:
    st.session_state.questions_count = 0


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

    question = st.text_input(
        "Ask a Question"
    )
    source = st.radio(
        "Choose Source",[
            "PDF",
            "Database"
        ]
    )

    
    if st.button("Ask"):
        st.session_state.questions_count += 1

        if source == "PDF":
            st.session_state.pdf_queries += 1
        else:
            st.session_state.db_queries += 1

        with st.spinner("Thinking..."):
            # if source == "pdf":
            start_time = time.time()
            answer = retriever_qa(question)
            # else:
            #     answer= ask_database(question)
            

            # if source == "PDF":
                # answer = retriever_qa(question)
            # else:
                # answer = ask_database(question)

            end_time = time.time()

            response_time = end_time - start_time

            st.session_state.total_response_time += response_time
            st.session_state.questions_count += 1
        st.success(answer)

        st.write(f"Tool Used: {source} retriever")

elif menu == "Upload PDF":

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    if uploaded_file:

        os.makedirs(
            "data/pdfs",
            exist_ok=True
        )

        filepath = os.path.join(
            "data/pdfs",
            uploaded_file.name
        )

        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(
            f"{uploaded_file.name} uploaded successfully"
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

    
    # check core folder
