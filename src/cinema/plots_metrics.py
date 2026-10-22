# графики блока метрик: водопад MRR, NRR и GRR, когорты, LTV по каналам, окупаемость каналов

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import style
from .style import (AQUA, BLUE, CHANNEL_COLORS, INK, INK2, MUTED, ORANGE, YELLOW, plt)

CHANNEL_ORDER = ["organic", "search", "social", "referral", "partner"]


def _month_labels(w):
    return [d[:7] for d in w["month_end"]]


def _mln(x, _pos=None):
    return f"{x / 1e6:.1f}"


def mrr_waterfall(w, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(w))
    pos = [("new_mrr", "новые", BLUE), ("reactivation_mrr", "вернувшиеся", AQUA),
           ("expansion_mrr", "апгрейд тарифа", YELLOW), ("price_uplift_mrr", "рост цены", ORANGE)]
    neg = [("contraction_mrr", "даунгрейд", MUTED), ("churn_mrr", "уход", INK2)]
    bottom = np.zeros(len(w))
    for col, name, color in pos:
        ax.bar(x, w[col], bottom=bottom, color=color, label=name, width=0.7)
        bottom += w[col].values
    bottom = np.zeros(len(w))
    for col, name, color in neg:
        ax.bar(x, w[col], bottom=bottom, color=color, label=name, width=0.7)
        bottom += w[col].values
    ax.axhline(0, color=INK2, linewidth=0.8)
    hike = int(w.index[w["price_uplift_mrr"].abs().idxmax()]) if w["price_uplift_mrr"].abs().max() > 0 else None
    if hike is not None:
        ax.annotate("повышение цены\nв трех регионах", xy=(hike, w["price_uplift_mrr"].iloc[hike] + w["new_mrr"].iloc[hike]),
                    xytext=(hike - 5.5, w["new_mrr"].max() * 1.12), fontsize=8.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_xticks(x[::2])
    ax.set_xticklabels(_month_labels(w)[::2], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(_mln))
    ax.set_ylabel("изменение MRR за месяц, млн руб")
    ax.set_title("Приток и отток MRR: растем за счет новых, платим за это уходами")
    ax.legend(ncol=3, loc="lower left")
    style.only_y_grid(ax)
    return style.save(fig, out)


def retention_ratios(w, out):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    d = w.dropna(subset=["nrr"])
    x = np.arange(len(d))
    ax.plot(x, d["nrr"] * 100, color=BLUE, label="NRR, чистое удержание денег")
    ax.plot(x, d["grr"] * 100, color=ORANGE, label="GRR, без учета роста")
    ax.axhline(100, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xticks(x[::2])
    ax.set_xticklabels([m[:7] for m in d["month_end"]][::2], rotation=45, ha="right")
    ax.set_ylabel("% от MRR на начало месяца")
    ax.set_ylim(90, 101)
    ax.set_title("Месячные NRR и GRR: до 100% его поднимает только повышение цены")
    ax.legend(loc="lower right")
    return style.save(fig, out)


def _heatmap(ax, mat, title, vmax=100, label="%"):
    cmap = style.blue_cmap()
    m = np.ma.masked_invalid(mat.values.astype(float))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(0, mat.shape[1], 2))
    ax.set_xticklabels(mat.columns[::2])
    ax.set_yticks(range(0, mat.shape[0], 2))
    ax.set_yticklabels([f"{i}" for i in mat.index[::2]])
    ax.set_xlabel("месяцев с первой оплаты")
    ax.grid(False)
    ax.set_title(title)
    return im


def cohort_heatmaps(ret, rev, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    _heatmap(axes[0], ret, "Сколько людей остается")
    im = _heatmap(axes[1], rev.clip(upper=100), "Сколько денег остается")
    axes[0].set_ylabel("месяц первой оплаты (номер когорты)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="% от стартового значения")
    return style.save(fig, out)


def retention_curve(rc, out):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(rc["age"], rc["retention_pct"], color=BLUE, marker="o", markersize=4, label="люди")
    ax.plot(rc["age"], rc["revenue_retention_pct"], color=ORANGE, marker="o", markersize=4, label="деньги")
    for age in (3, 6, 12):
        row = rc[rc["age"] == age].iloc[0]
        ax.annotate(f"{row['retention_pct']:.0f}%", xy=(age, row["retention_pct"]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("месяцев с первой оплаты")
    ax.set_ylabel("%")
    ax.set_ylim(40, 102)
    ax.set_title("Средняя кривая удержания по когортам")
    ax.legend()
    return style.save(fig, out)


def ltv_curves(curves, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for ch in CHANNEL_ORDER:
        ax.plot(curves.index, curves[ch], color=CHANNEL_COLORS[ch], label=ch)
        ax.annotate(f"{curves[ch].iloc[-1]:.0f}", xy=(curves.index[-1], curves[ch].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("дней с регистрации")
    ax.set_ylabel("накопленная выручка на одну регистрацию, руб")
    ax.set_title("Накопленная выручка на регистрацию: referral приносит почти вдвое больше social")
    ax.legend(loc="upper left")
    ax.set_xlim(0, curves.index[-1] + 25)
    return style.save(fig, out)


def channel_funnel(f, out):
    f = f.set_index("channel").loc[CHANNEL_ORDER]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(f))
    h = 0.27
    ax.barh(y - h, f["activation_pct"], height=h, color=BLUE, label="активировались за 3 дня")
    ax.barh(y, f["conversion_pct"], height=h, color=ORANGE, label="стали платящими")
    ax.set_yticks(y)
    ax.set_yticklabels(f.index)
    ax.invert_yaxis()
    ax.set_xlabel("% от регистраций")
    ax.set_title("Воронка по каналам: в social платящих заметно меньше")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    style.only_y_grid(ax)
    ax.grid(axis="y", visible=False)
    return style.save(fig, out)


def cac_vs_revenue(f, out):
    f = f[f["channel"] != "organic"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for _, r in f.iterrows():
        ax.scatter(r["cac_per_signup"], r["revenue_180_per_signup"], s=r["signups"] / 20,
                   color=CHANNEL_COLORS[r["channel"]], alpha=0.85, edgecolor=style.SURFACE)
        ax.annotate(r["channel"], xy=(r["cac_per_signup"], r["revenue_180_per_signup"]),
                    xytext=(10, 8), textcoords="offset points", fontsize=9, color=INK)
    lim = max(f["cac_per_signup"].max(), f["revenue_180_per_signup"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("окупаемость 1 к 1", xy=(380, 380), xytext=(0, 8), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, lim)
    ax.set_xlabel("стоимость регистрации, руб")
    ax.set_ylabel("выручка за 180 дней на регистрацию, руб")
    ax.set_title("Цена против отдачи по каналам (размер = число регистраций)")
    return style.save(fig, out)


def channel_mix(mix, out):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(mix))
    ax.stackplot(x, [mix[c] * 100 for c in CHANNEL_ORDER], labels=CHANNEL_ORDER,
                 colors=[CHANNEL_COLORS[c] for c in CHANNEL_ORDER], alpha=0.95)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(mix.index[::2], rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(mix) - 1)
    ax.set_ylabel("% регистраций")
    ax.set_title("Структура регистраций: доля social растет, а это самый дорогой по итогу канал")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    return style.save(fig, out)


def make_all(r, outdir):
    from pathlib import Path
    from .metrics import cohort_matrices
    out = Path(outdir)
    ret, rev, _ = cohort_matrices(r["cohorts"])
    return [
        mrr_waterfall(r["waterfall"], out / "01_mrr_waterfall.png"),
        retention_ratios(r["waterfall"], out / "02_nrr_grr.png"),
        cohort_heatmaps(ret, rev, out / "03_cohorts.png"),
        retention_curve(r["retention_curve"], out / "04_retention_curve.png"),
        ltv_curves(r["curves"], out / "05_ltv_by_channel.png"),
        channel_funnel(r["funnel"], out / "06_channel_funnel.png"),
        cac_vs_revenue(r["funnel"], out / "07_cac_vs_revenue.png"),
        channel_mix(r["mix"], out / "08_channel_mix.png"),
    ]
