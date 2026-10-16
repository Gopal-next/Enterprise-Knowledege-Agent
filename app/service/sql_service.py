import os
import re
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

try:
    from app.database.sql_tool import get_database
except ImportError:
    from database.sql_tool import get_database

load_dotenv()


def ask_database(question: str) -> str:
    """
    Converts a natural language question to SQL, executes it against the
    database, and returns an executive conversational summary with the SQL query.
    """
    try:
        db = get_database()
        if isinstance(db, tuple):
            db = db[0]

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash"
        )

        chain = create_sql_query_chain(
            llm,
            db
        )

        sql_query = chain.invoke({
            "question": question
        })

        # Clean markdown formatting from LLM SQL output
        sql_query = re.sub(
            r"```(?:sql|sqlite)?",
            "",
            sql_query
        ).strip()

        if "SQLQuery:" in sql_query:
            sql_query = sql_query.split("SQLQuery:")[-1].strip()

        executor = QuerySQLDatabaseTool(
            db=db
        )

        result = executor.invoke(
            sql_query
        )

        # Synthesize business-friendly answer from raw query results
        synthesis_prompt = f"""
        You are a helpful business data analyst.
        A user asked a question, an SQL query was generated, and the database returned a result.
        Synthesize a clean, natural, customer-friendly answer.

        User Question: {question}
        SQL Query: {sql_query}
        Database Result: {result}

        Answer:
        """
        synthesis = llm.invoke(synthesis_prompt)
        natural_answer = synthesis.content.strip()

        return f"{natural_answer}\n\n---\n*Executed SQL:* `{sql_query}`"
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            return "Your Gemini quota has finished, please check back later."
        raise e