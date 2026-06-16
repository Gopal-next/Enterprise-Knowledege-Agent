from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain

from app.database.sql_tool import get_database

def ask_database(question):

    db = get_database()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    chain = create_sql_query_chain(
        llm,
        db
    )

    sql_query = chain.invoke(
        {"question": question}
    )

    return sql_query