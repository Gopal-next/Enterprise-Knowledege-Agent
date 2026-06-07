def search_documents(vectorstore, query):

    results = vectorstore.similarity_search(
        query,
        k=3
    )

    return results