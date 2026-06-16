from app.core.llm import get_llm
from app.core.prompts import RAG_PROMPT

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

    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content

# docs = retriever.invoke(question)

# for doc in docs:
#     print(doc.metadata)

# for doc in docs:
#     print(
#         f"Page: {doc.metadata.get('page')}"
#     )