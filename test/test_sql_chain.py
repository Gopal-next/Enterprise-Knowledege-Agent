import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.service.sql_service import ask_database

question = "How many employees work in Engineering?"

result = ask_database(question)

print(result)