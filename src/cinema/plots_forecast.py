# графики блока прогноза и аномалий

import numpy as np
import pandas as pd

from . import style
from .config import CAMPAIGN_DAYS, INCIDENT_DAYS
from .style import AQUA, BLUE, INK2, MUTED, ORANGE, YELLOW, plt

MODEL_NAMES = {"seasonal_naive": "прошлая неделя", "seasonal_mean4": "среднее за 4 недели",
               "holt_winters": "Хольт Винтерс", "ridge_calendar": "регрессия на календаре"}
MODEL_COLORS = {"seasonal_naive": MUTED, "seasonal_mean4": YELLOW, "holt_winters": BLUE, "ridge_calendar": ORANGE}


def forecast_chart(res, out, history=120):
    s = res["series"].copy()
    s["day"] = pd.to_datetime(s["day"])
    h = s.tail(history)
    f = res["final"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(h["day"], h["signups"], color=INK2, linewidth=1.4, label="факт")
    ax.fill_between(f["date"], f["lo"], f["hi"], color=ORANGE, alpha=0.2, linewidth=0, label="интервал 90%")
    ax.plot(f["date"], f["forecast"], color=ORANGE, label=f"прогноз: {MODEL_NAMES[res['best']]}")
    ax.axvline(h["day"].iloc[-1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_ylabel("регистраций в день")
    ax.set_title("Прогноз регистраций на 28 дней: по выходным выше, интервал шире к концу горизонта")
    ax.legend(loc="upper left", ncol=3)
    fig.autofmt_xdate()
    return style.save(fig, out)


def backtest_chart(res, out):
    bh = res["by_horizon"]
    mean_level = res["series"]["signups"].tail(365).mean()
    floor = 0.8 / np.sqrt(mean_level)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    models = list(res["score"].index)
    w = 0.2
    x = np.arange(bh.shape[0])
    for i, m in enumerate(models):
        ax.bar(x + (i - 1.5) * w, bh[m] * 100, width=w, color=MODEL_COLORS[m], label=MODEL_NAMES[m])
    ax.axhline(floor * 100, color=INK2, linewidth=1, linestyle=(0, (4, 3)), label="предел из шума Пуассона")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} неделя" for k in bh.index])
    ax.set_ylabel("WAPE, % (меньше лучше)")
    ax.set_title("Ошибка по неделям горизонта")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 21)
    ax = axes[1]
    bt = res["backtest"]
    o = res["origins"][len(res["origins"]) * 3 // 4]
    d = bt[(bt["origin"] == o) & (bt["model"] == "ridge_calendar")]
    ax.plot(d["date"], d["actual"], color=INK2, label="факт", linewidth=1.6)
    for m in ["seasonal_naive", "holt_winters", "ridge_calendar"]:
        dm = bt[(bt["origin"] == o) & (bt["model"] == m)]
        ax.plot(dm["date"], dm["forecast"], color=MODEL_COLORS[m], label=MODEL_NAMES[m], linewidth=1.4)
    ax.set_title("Пример одного окна проверки")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.tick_params(axis="x", rotation=30)
    return style.save(fig, out)


def anomaly_chart(an, out):
    d = an["failures"].copy()
    d["day"] = pd.to_datetime(d["day"])
    sg = an["signups"].copy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4))
    ax = axes[0]
    ax.plot(d["day"], d["fail_rate"] * 100, color=MUTED, linewidth=1, label="доля неудачных оплат")
    ax.plot(d["day"], d["base"] * 100, color=BLUE, linewidth=1.6, label="обычный уровень (медиана 28 дней)")
    b = d[d["flag_binom"]]
    ax.scatter(b["day"], b["fail_rate"] * 100, color=ORANGE, s=34, zorder=3, label="тревога: биномиальный тест")
    ax.axvspan(pd.Timestamp(INCIDENT_DAYS[0]) - pd.Timedelta(hours=12), pd.Timestamp(INCIDENT_DAYS[-1]) + pd.Timedelta(hours=12),
               color=ORANGE, alpha=0.12, linewidth=0)
    ax.set_ylabel("%")
    ax.set_title("Платежи: сбой шлюза 10-12 марта виден сразу и без ложных тревог")
    ax.legend(loc="upper right", fontsize=8.5)
    ax = axes[1]
    ax.plot(sg["day"], sg["z"], color=MUTED, linewidth=1)
    ax.axhline(4, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axhline(-4, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    f = sg[sg["flag"]]
    ax.scatter(f["day"], f["z"], color=ORANGE, s=30, zorder=3, label="тревога")
    cd = pd.to_datetime(pd.Series(CAMPAIGN_DAYS))
    for i, (a, b2) in enumerate([(cd.iloc[0], cd.iloc[3]), (cd.iloc[4], cd.iloc[-1])]):
        ax.axvspan(a - pd.Timedelta(hours=12), b2 + pd.Timedelta(hours=12), color=YELLOW, alpha=0.2, linewidth=0,
                   label="рекламная кампания" if i == 0 else None)
    ax.set_ylabel("отклонение от обычного, в сигмах")
    ax.set_title("Регистрации: резкий скачок ловится, растянутая новогодняя кампания остается под порогом")
    ax.legend(loc="upper left", fontsize=8.5)
    return style.save(fig, out)


def make_all(fc, an, outdir):
    from pathlib import Path
    out = Path(outdir)
    return [
        forecast_chart(fc, out / "29_forecast.png"),
        backtest_chart(fc, out / "30_forecast_backtest.png"),
        anomaly_chart(an, out / "31_anomalies.png"),
    ]
