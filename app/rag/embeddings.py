from langchain_community.embeddings import SentenceTransformerEmbeddings

def get_embedding_model():

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return embeddings


# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# load_dotenv()

# def get_embedding_model():
#     return GoogleGenerativeAIEmbeddings(
#         model="gemini-embedding-2"
#     )
# from langchain_huggingface import HuggingFaceEmbeddings
# def get_embedding_model():
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )