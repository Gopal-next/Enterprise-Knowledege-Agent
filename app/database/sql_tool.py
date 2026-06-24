from langchain_community.utilities import SQLDatabase
from utils.file_to_db import create_temp_db


def get_database():

    db_path = create_temp_db()

    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path}"
    )

    return db, db_path