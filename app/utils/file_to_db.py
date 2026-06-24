import sqlite3
import pandas as pd
from pathlib import Path


def create_temp_db():

    folder = Path(
        r"D:\\Enterprise knowledge agent\\app\data\\excelfile"
    )

    files = (
        list(folder.glob("*.csv"))
        + list(folder.glob("*.xlsx"))
        + list(folder.glob("*.xls"))
    )

    if not files:
        raise Exception(
            "No CSV or XLSX file found in excelfile folder."
        )

    file = files[0]

    if file.suffix.lower() == ".csv":
        df = pd.read_csv(file)

    else:
        df = pd.read_excel(file)

    db_path = folder / "temp.db"

    conn = sqlite3.connect(db_path)

    table_name = file.stem.lower()

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    return str(db_path)