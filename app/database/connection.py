import sqlite3

def get_connection():
    return sqlite3.connect("D:/Enterprise knowledge agent/app/data/company.db")