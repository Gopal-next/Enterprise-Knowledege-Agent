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


if menu == "Chat Assistant":

    st.header("Chat Assistant")

    question = st.text_input(
        "Ask a Question"
    )

    if st.button("Ask"):

        with st.spinner("Searching documents and generating answer..."):

            answer = retriever_qa(question)

        st.success(answer)

        st.write("Tool Used: PDF Retriever")

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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Documents",
            2
        )

    with col2:
        st.metric(
            "Questions",
            45
        )

    with col3:
        st.metric(
            "Response Time",
            "1.5 sec"
        )

    st.write("Most Used Tool: PDF Retriever")