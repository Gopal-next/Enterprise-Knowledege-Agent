import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from rag.loader import load_pdf
from rag.splitter import split_documents
from rag.vectorstore import create_vectorstore


def retriever_qa(query):

    pdf_folder = "data/pdfs"

    all_docs = []

    for file in os.listdir(pdf_folder):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(
                pdf_folder,
                file
            )

            docs = load_pdf(pdf_path)

            all_docs.extend(docs)

    chunks = split_documents(all_docs)

    vectordb = create_vectorstore(chunks)

    retriever_obj = vectordb.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    system_prompt = (
        "Use the given pieces of context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever_obj,
        question_answer_chain
    )

    response = rag_chain.invoke({
        "input": query
    })

    return response["answer"]