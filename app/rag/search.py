def search_documents(vectorstore, query):

    results = vectorstore.similarity_search(
        query,
        k=3
    )

    return results



# def search_documents(vectorstore, query):

#     results = vectorstore.max_marginal_relevance_search(
#         query,
#         k=3,
#         fetch_k= 20
#     )

#     return results


# # I use Max Marginal Relevance (MMR) retrieval instead of basic 
# # similarity search. MMR first retrieves a larger set of candidate 
# # chunks and then selects the most relevant and diverse chunks, 
# # reducing duplicate context and improving answer quality.