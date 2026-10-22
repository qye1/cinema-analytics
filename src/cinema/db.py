# работа с базой sqlite: создание, загрузка таблиц, запуск sql файлов

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = ROOT / "sql"
DB_PATH = ROOT / "data" / "cinema.db"

DATE_COLUMNS = {"users": ["signup_date"], "subscriptions": ["start_date", "end_date"],
                "payments": ["pay_date"], "engagement": ["week_start"], "tickets": ["created_date"],
                "marketing_spend": ["date"], "experiment_assignment": ["assigned_date"],
                "price_list": []}


def connect(path=DB_PATH):
    return sqlite3.connect(path)


def run_script(conn, path):
    conn.executescript(Path(path).read_text(encoding="utf-8"))
    conn.commit()


def create_db(tables, path=DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    run_script(conn, SQL_DIR / "00_schema.sql")
    for name, df in tables.items():
        df = df.copy()
        for c in DATE_COLUMNS[name]:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
            df[c] = df[c].where(df[c].notna(), None)
        df = df.astype(object).where(df.notna(), None)
        cols = ",".join(df.columns)
        q = ",".join("?" * len(df.columns))
        conn.executemany(f"INSERT INTO {name} ({cols}) VALUES ({q})", df.itertuples(index=False, name=None))
    conn.commit()
    return conn


def query(conn, sql, params=None):
    return pd.read_sql_query(sql, conn, params=params)


def query_file(conn, path, params=None):
    return query(conn, Path(path).read_text(encoding="utf-8"), params)
