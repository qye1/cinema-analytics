# маленький мир для тестов: генерируется один раз на весь запуск и лежит во временной папке

import json
import tempfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

from cinema import db, gen, marts


@lru_cache(maxsize=1)
def small():
    d = Path(tempfile.mkdtemp(prefix="cinema_test_"))
    tables, truth = gen.generate(d, 6000, 3)
    conn = db.connect(d / "cinema.db")
    marts.build(conn)
    users = pd.read_csv(d / "truth_users.csv")
    return {"dir": d, "conn": conn, "truth": truth, "truth_users": users, "tables": tables}
