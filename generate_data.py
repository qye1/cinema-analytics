# создает базу data/cinema.db и файлы с настоящими ответами: python generate_data.py

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from cinema import gen       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()
    tables, _ = gen.generate(args.out, args.users, args.seed)
    print("готово:", Path(args.out) / "cinema.db")
    for name, df in tables.items():
        print(f"  {name}: {len(df)} строк")


if __name__ == "__main__":
    main()
