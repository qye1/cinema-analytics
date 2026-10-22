# графики блока выживаемости: кривые Каплана Майера, риск по неделям, модель против заложенных эффектов

import numpy as np
from matplotlib.ticker import PercentFormatter

from . import style
from .style import BLUE, CHANNEL_COLORS, INK2, MUTED, ORANGE, plt

CHANNEL_ORDER = ["organic", "search", "social", "referral", "partner"]
PLAN_COLORS = {"basic": style.ORANGE, "standard": style.BLUE, "premium": style.AQUA}


def km_channels(km, summary, out, horizon=78):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for ch in CHANNEL_ORDER:
        k = km[ch]
        k = k[k["time"] <= horizon]
        ax.step(k["time"], k["surv"], where="post", color=CHANNEL_COLORS[ch], label=ch)
        ax.fill_between(k["time"], k["lo"], k["hi"], step="post", color=CHANNEL_COLORS[ch], alpha=0.12, linewidth=0)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("недель с первой оплаты")
    ax.set_ylabel("доля подписок, которые еще не отменили")
    s = summary.set_index("channel")
    ax.set_title(f"Кривые дожития по каналам: через полгода остается {s['surv_26w'].max():.0%} "
                 f"({s['surv_26w'].idxmax()}) против {s['surv_26w'].min():.0%} ({s['surv_26w'].idxmin()})")
    ax.legend(loc="upper right", ncol=1)
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1.02)
    return style.save(fig, out)


def km_plans(km, out, horizon=78):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for pl in ["basic", "standard", "premium"]:
        k = km[pl]
        k = k[k["time"] <= horizon]
        ax.step(k["time"], k["surv"], where="post", color=PLAN_COLORS[pl], label=pl)
        ax.fill_between(k["time"], k["lo"], k["hi"], step="post", color=PLAN_COLORS[pl], alpha=0.12, linewidth=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("недель с первой оплаты")
    ax.set_ylabel("доля не отменивших")
    ax.set_title("Дожитие по стартовому тарифу: чем дороже тариф, тем дольше живут")
    ax.legend()
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1.02)
    return style.save(fig, out)


def hazard_by_week(eh, out):
    # первые недели без отмен: спад интереса идет около месяца, поэтому берем уже недели с уходами
    d = eh[(eh["at_risk"] >= 300) & (eh["events"] > 0)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(d["week"], d["hazard"] * 100, color=BLUE, width=0.8)
    smooth = d["hazard"].rolling(5, center=True, min_periods=1).mean() * 100
    ax.plot(d["week"], smooth, color=ORANGE, linewidth=2, label="сглаживание, окно 5 недель")
    ax.set_xlabel("неделя жизни подписки")
    ax.set_ylabel("риск отмены на неделе, %")
    ax.set_title("Риск отмены по неделям: максимум в первые полтора месяца, потом втрое ниже")
    ax.legend()
    ax.grid(axis="x", visible=False)
    return style.save(fig, out)


def forest(naive, lag, truth, out, diag=None):
    terms = ["social", "partner", "search", "referral", "basic", "premium", "tv", "web",
             "engagement_per_log_hour", "hike", "summer"]
    labels = {"social": "канал social", "partner": "канал partner", "search": "канал search",
              "referral": "канал referral", "basic": "тариф basic", "premium": "тариф premium",
              "tv": "устройство tv", "web": "устройство web",
              "engagement_per_log_hour": "вовлеченность (на 1 лог. час)",
              "hike": "повышение цены", "summer": "лето"}
    nv = naive.set_index("term")
    lg = lag.set_index("term")
    dg = diag.set_index("term") if diag is not None else None
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = np.arange(len(terms))[::-1]
    for yi, t in zip(y, terms):
        row = lg.loc[t] if t in ("hike", "summer") else nv.loc[t]
        ax.plot([row["lo"], row["hi"]], [yi, yi], color=BLUE, linewidth=2.2, solid_capstyle="round")
        ax.scatter(row["coef"], yi, color=BLUE, s=36, zorder=3)
        if dg is not None:
            d = dg.loc[t]
            ax.plot([d["lo"], d["hi"]], [yi - 0.22, yi - 0.22], color=INK2, linewidth=1.4, solid_capstyle="round")
            ax.scatter(d["coef"], yi - 0.22, facecolor="white", edgecolor=INK2, s=26, zorder=3, linewidth=1.3)
        ax.scatter(truth[t], yi, marker="D", color=ORANGE, s=44, zorder=4)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[t] for t in terms])
    ax.set_xlabel("лог отношения рисков (больше нуля: уходят чаще)")
    ax.set_title("Модель риска против заложенных эффектов\nсиний: все подписчики, серый: только тип lost_cause, ромб: правда")
    ax.grid(axis="y", visible=False)
    return style.save(fig, out)


def make_all(res, truth, outdir, diag=None):
    from pathlib import Path
    out = Path(outdir)
    planted = truth["hazard_ratios"]["planted_log_hazard"]
    return [
        km_channels(res["km_channel"], res["km_summary"], out / "09_km_channel.png"),
        km_plans(res["km_plan"], out / "10_km_plan.png"),
        hazard_by_week(res["empirical_hazard"], out / "11_hazard_by_week.png"),
        forest(res["hazard_naive"], res["hazard_lag"], planted, out / "12_hazard_forest.png", diag),
    ]
