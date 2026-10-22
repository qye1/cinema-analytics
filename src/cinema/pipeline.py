# общий запуск: данные, sql, все блоки анализа, графики, таблицы, отчет и Excel

import json
import time
from pathlib import Path

import pandas as pd

from . import (anomaly, churn, db, excel_export, experiments, forecast, gen, marts, metrics, plots_churn,
               plots_experiments, plots_forecast, plots_metrics, plots_segments, plots_survival, report, scoreboard,
               segments, survival)
from .config import CAMPAIGN_DAYS

ROOT = Path(__file__).resolve().parents[2]


class Steps:
    # маленький помощник: печатает, сколько заняла каждая часть
    def __init__(self, log=print):
        self.log = log
        self.t = time.time()

    def done(self, name):
        now = time.time()
        self.log(f"  {name}: {now - self.t:.1f} с")
        self.t = now


def top_risk_list(conn, res_churn, n=200):
    # кого стоит обработать в первую очередь: свежий снимок активных подписчиков и риск по обученной модели
    from .config import NW
    snap = churn.build_snapshots(conn, [NW - 1], drop_offered=False)
    model = res_churn["models"]["gbm"]
    snap["risk_4w"] = model.predict_proba(snap[churn.ALL_FEATURES])[:, 1]
    cols = ["user_id", "risk_4w", "hours_ratio", "hours_l2", "tenure_weeks", "plan_basic", "plan_premium"]
    return snap.sort_values("risk_4w", ascending=False)[cols].head(n).reset_index(drop=True)


def run_all(data_dir=None, out_dir=None, users=60000, seed=42, regen=False, log=print):
    data_dir = Path(data_dir or ROOT / "data")
    out_dir = Path(out_dir or ROOT / "results")
    charts = out_dir / "charts"
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    st = Steps(log)
    if regen or not (data_dir / "cinema.db").exists():
        log("генерирую данные")
        gen.generate(data_dir, users, seed)
        st.done("данные")
    truth = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    truth_users = pd.read_csv(data_dir / "truth_users.csv")
    conn = db.connect(data_dir / "cinema.db")
    marts.build(conn)
    res = {"truth": truth, "quality": marts.quality(conn)}
    st.done("sql слой")

    res["metrics"] = metrics.build_all(conn)
    res["kpi"] = metrics.kpi_summary(res["metrics"]["waterfall"], res["metrics"]["cohorts"], res["metrics"]["funnel"])
    plots_metrics.make_all(res["metrics"], charts)
    st.done("метрики")

    res["survival"] = survival.run(conn)
    res["survival"]["diag_lost_cause"], _ = survival.hazard_by_utype(
        res["survival"]["episodes"], res["survival"]["spells"], truth_users)
    plots_survival.make_all(res["survival"], truth, charts, res["survival"]["diag_lost_cause"])
    st.done("выживаемость")

    seg = segments.run(conn, truth_users)
    seg_km, seg_tab, seg_lr = segments.segment_survival(res["survival"]["episodes"], seg["features"])
    res["segments"] = seg
    res["segments"].update({"km": seg_km, "surv_table": seg_tab, "logrank": seg_lr})
    plots_segments.make_all(seg, seg_km, charts)
    st.done("сегменты")

    res["churn"] = churn.run(conn)
    plots_churn.make_all(res["churn"], charts)
    res["top_risk"] = top_risk_list(conn, res["churn"])
    st.done("модель оттока")

    res["exp"] = experiments.run(conn, truth_users, truth)
    plots_experiments.make_all(res["exp"], truth, charts)
    st.done("эксперименты")

    res["forecast"] = forecast.run(conn, CAMPAIGN_DAYS)
    res["anomaly"] = anomaly.run(conn, CAMPAIGN_DAYS)
    plots_forecast.make_all(res["forecast"], res["anomaly"], charts)
    st.done("прогноз и аномалии")

    res["scoreboard"] = scoreboard.build(res, truth)
    save_tables(res, tables_dir)
    report.render(res, charts, out_dir / "report.html")
    excel_export.build(res, out_dir / "cinema_report.xlsx")
    st.done("таблицы, отчет и Excel")
    conn.close()
    return res


def save_tables(res, d):
    m = res["metrics"]
    items = {
        "quality_checks": res["quality"], "mrr_waterfall": m["waterfall"], "cohorts": m["cohorts"],
        "channel_economics": m["funnel"], "km_summary_by_channel": res["survival"]["km_summary"],
        "hazard_model": res["survival"]["hazard_naive"], "hazard_model_lag4": res["survival"]["hazard_lag"],
        "segment_profiles": res["segments"]["profile"].reset_index(), "churn_model_metrics": res["churn"]["results"],
        "churn_importance": res["churn"]["importance"], "churn_gains": res["churn"]["gains"],
        "top_risk_users": res["top_risk"], "experiment_subgroups_hours": res["exp"]["by_ratio"],
        "forecast_scores": res["forecast"]["score"].reset_index(), "forecast_28d": res["forecast"]["final"],
        "payment_failure_methods": res["anomaly"]["eval"], "scoreboard": res["scoreboard"],
    }
    for name, df in items.items():
        df.to_csv(d / f"{name}.csv", index=False, encoding="utf-8-sig")
