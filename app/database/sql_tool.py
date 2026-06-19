from langchain_community.utilities import SQLDatabase

def get_database():

    db = SQLDatabase.from_uri(
        "sqlite:///D:/Enterprise knowledge agent/app/data/company.db"
    )

    return db