from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

RAG_PROMPT = """
You are an enterprise assistant.

Use only the provided context.

If answer is not present,
say:
'I could not find that information in the documents.'

Context:
{context}

Question:
{question}

Answer:
"""

def answer_question(question, retriever):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    response = llm.invoke(prompt)

    return response.content

# docs = retriever.invoke(question)

# for doc in docs:
#     print(doc.metadata)

# for doc in docs:
#     print(
#         f"Page: {doc.metadata.get('page')}"
#     )