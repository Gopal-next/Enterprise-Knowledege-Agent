def answer_hybrid_question(question):

    pdf_result = get_policy_data()

    sql_result = get_employee_data()

    return {
        "pdf": pdf_result,
        "sql": sql_result
    }