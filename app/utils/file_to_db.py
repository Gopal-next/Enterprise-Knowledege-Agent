import os
import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_temp_db():
    candidate_folders = [
        PROJECT_ROOT / "Data" / "excel file",
        PROJECT_ROOT / "Data" / "excelfile",
        PROJECT_ROOT / "app" / "data" / "excelfile",
        PROJECT_ROOT / "data" / "excelfile",
    ]

    files = []
    chosen_folder = None
    for folder in candidate_folders:
        if folder.exists():
            f_list = (
                list(folder.glob("*.csv"))
                + list(folder.glob("*.xlsx"))
                + list(folder.glob("*.xls"))
            )
            if f_list:
                files = f_list
                chosen_folder = folder
                break

    if not files:
        # Fallback to create folder if none exists
        default_folder = PROJECT_ROOT / "Data" / "excelfile"
        default_folder.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            "No CSV or XLSX file found in excelfile folder. Please upload an Excel or CSV file in the 'Upload File' tab."
        )

    file = files[0]

    if file.suffix.lower() == ".csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    db_path = chosen_folder / "temp.db"
    conn = sqlite3.connect(str(db_path))

    table_name = file.stem.lower().replace(" ", "_").replace("-", "_")

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    return str(db_path)