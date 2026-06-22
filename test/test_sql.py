import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database.connection import get_connection

conn = get_connection()

cursor = conn.cursor()

cursor.execute(
    "SELECT * FROM employees"
)

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()