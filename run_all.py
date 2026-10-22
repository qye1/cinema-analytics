# запускает все: данные, sql, анализ, графики, отчет и Excel
# python run_all.py              берет готовую базу из data, если ее нет, генерирует
# python run_all.py --regen      заново сгенерировать данные
# python run_all.py --users 20000 --seed 7 --regen --out results_small    быстрый прогон на меньшем мире

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from cinema import pipeline, scoreboard      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    res = pipeline.run_all(args.data, args.out, args.users, args.seed, args.regen)
    hit, total = scoreboard.summary_counts(res["scoreboard"])
    print(f"оценок внутри интервала заложенной правды: {hit} из {total}")
    print("отчет:", Path(args.out) / "report.html")
    print("Excel:", Path(args.out) / "cinema_report.xlsx")


if __name__ == "__main__":
    main()
