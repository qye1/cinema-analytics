# один сквозной прогон на маленьком мире: все файлы должны появиться, а отчет не содержать пустых подстановок

import re
import tempfile
from pathlib import Path

from cinema import pipeline, scoreboard


def test_run_all_small_world_end_to_end():
    root = Path(tempfile.mkdtemp(prefix="cinema_smoke_"))
    res = pipeline.run_all(root / "data", root / "results", users=6000, seed=3, regen=True, log=lambda *a: None)
    out = root / "results"
    assert (out / "report.html").stat().st_size > 1_000_000
    assert (out / "cinema_report.xlsx").exists()
    assert len(list((out / "charts").glob("*.png"))) >= 30
    html = (out / "report.html").read_text(encoding="utf-8")
    # смотрим только на текст, без встроенных картинок и тегов
    text = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "", html)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    for bad in ["{{", "}}", "{%", "None", " nan ", "inf%"]:
        assert bad not in text, bad
    hit, total = scoreboard.summary_counts(res["scoreboard"])
    assert total >= 18 and hit / total > 0.7
