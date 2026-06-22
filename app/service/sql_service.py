from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_sql_query_chain
from app.database.sql_tool import get_database
from dotenv import load_dotenv
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate
load_dotenv()


# prompt = PromptTemplate(
#     input_variables=["input", "table_info", "top_k"],
#     template="""
#         You are a business data assistant.

#         Users may ask questions in plain English.
#         Convert the user's question into a valid SQLite query.

#         Rules:
#         - Use only the tables and columns provided.
#         - Never invent columns.
#         - Return only SQL.
#         - Understand non-technical business questions.

#         Database Schema:
#         {table_info}

#         Question:
#         {input}
#     """
# )


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

    executor = QuerySQLDatabaseTool(db=db)

    sql_query = sql_query

    if "SQLQuery:" in sql_query:
        sql_query = sql_query.split("SQLQuery:")[-1].strip()

    result = executor.invoke(sql_query)

    return result