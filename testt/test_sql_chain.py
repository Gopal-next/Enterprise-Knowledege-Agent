from app.services.sql_service import ask_database

question = "How many employees work in Engineering?"

result = ask_database(question)

print(result)