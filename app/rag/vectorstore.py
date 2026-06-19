from langchain_community.vectorstores import FAISS
from rag.embeddings import get_embedding_model

def create_vectorstore(chunks):
    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore

