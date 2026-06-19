from app.database.connection import get_connection

conn = get_connection()

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS employees(
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    joining_date TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS leave_records(
    employee_id INTEGER,
    leave_days INTEGER
)
""")

cursor.execute("""
INSERT INTO employees
(name, department, joining_date)
VALUES
('John','Engineering','2024-01-10'),
('Alice','HR','2023-05-20'),
('Bob','Engineering','2022-08-15')
""")

cursor.execute("""
INSERT INTO leave_records
(employee_id, leave_days)
VALUES
(1,12),
(2,5),
(3,20)
""")

conn.commit()
# Read data
cursor.execute("SELECT * FROM employees")

# Print rows
rows = cursor.fetchall()

for row in rows:
    print(row)


print("Tables Created")

conn.close()

