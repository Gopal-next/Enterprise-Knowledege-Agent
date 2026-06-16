from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(
    "sqlite:///data/company.db"
)

print(db.get_table_info())


# from langchain_community.utilities import SQLDatabase

# def get_database():

#     db = SQLDatabase.from_uri(
#         "sqlite:///data/company.db"
#     )

#     return db