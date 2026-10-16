import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from google.genai import types
load_dotenv()

def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        config = types.EmbedContentConfig(task_type='SEMANTIC_SIMILARITY')

    )

