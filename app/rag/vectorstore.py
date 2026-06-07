from langchain_community.vectorstores import Chroma
from app.rag.embeddings import get_embedding_model

def create_vectorstore(chunks):

    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="data/chroma"
    )

    return vectorstore