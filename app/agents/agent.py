from app.agents.tools import pdf_tool
from app.agents.tools import sql_tool


def route_question(question):

    question = question.lower()

    if "policy" in question:
        return pdf_tool(question)

    if "employee" in question:
        return sql_tool(question)

    return "Unable to determine tool."


def route_question(question):

    question = question.lower()

    if "according to" in question:

        return "HYBRID"

    if "policy" in question:

        return "PDF"

    if "employee" in question:

        return "SQL"

    return "UNKNOWN"