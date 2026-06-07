from langchain_community.embeddings import SentenceTransformerEmbeddings

def get_embedding_model():

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return embeddings