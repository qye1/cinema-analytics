# генерация мира и запись его в базу, одна функция, чтобы ее звали и скрипт, и тесты, и общий запуск

import json
from pathlib import Path

from . import db, world


def generate(out_dir, users=60000, seed=42, with_truth=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables, truth, truth_users, _ = world.build_world(seed, users, with_truth=with_truth)
    conn = db.create_db(tables, out / "cinema.db")
    conn.close()
    (out / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    truth_users.to_csv(out / "truth_users.csv", index=False)
    return tables, truth
