# графики блока экспериментов: A/B оффера, подгруппы, uplift, difference in differences

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from . import style
from .config import HIKE_WEEK
from .style import AQUA, BLUE, INK, INK2, MUTED, ORANGE, YELLOW, plt


def ab_result(itt, truth_effect, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    rates = [itt["p_control"], itt["p_offer"]]
    ns = [itt["n_control"], itt["n_offer"]]
    errs = [1.96 * np.sqrt(p * (1 - p) / n) for p, n in zip(rates, ns)]
    ax.bar(["без оффера", "с оффером"], [r * 100 for r in rates], yerr=[e * 100 for e in errs],
           color=[MUTED, BLUE], width=0.55, error_kw=dict(ecolor=INK2, capsize=4, linewidth=1.2))
    for i, r in enumerate(rates):
        ax.text(i, r * 100 + errs[i] * 100 + 0.8, f"{r:.1%}", ha="center", fontsize=10, color=INK)
    ax.set_ylabel("ушли в течение 4 недель, %")
    ax.set_title("Отток через 4 недели")
    ax.set_ylim(0, max(rates) * 100 + 9)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.hist(itt["boot"] * 100, bins=40, color=BLUE, alpha=0.75)
    lo, hi = itt["ci_boot"]
    ax.axvline(lo * 100, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.axvline(hi * 100, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.axvline(0, color=INK, linewidth=1)
    ax.axvline(truth_effect * 100, color=ORANGE, linewidth=2)
    ymax = ax.get_ylim()[1]
    ax.text(truth_effect * 100, ymax * 0.97, " заложенный\n эффект", color=ORANGE, fontsize=9, va="top", ha="right")
    ax.set_xlabel("разница в уходах, п.п. (бутстреп 3000 раз)")
    ax.set_ylabel("число повторов")
    ax.set_title(f"Эффект {itt['diff'] * 100:.1f} п.п., интервал от {lo * 100:.1f} до {hi * 100:.1f}")
    return style.save(fig, out)


def subgroups(by_ratio, by_plan, by_type, truth_by_type, out):
    rows = []
    for name, df, prefix in [("падение часов", by_ratio, ""), ("тариф", by_plan, ""), ("тип зрителя (знает только симулятор)", by_type, "")]:
        for _, r in df.iterrows():
            rows.append((name, r["group"], r["diff"], r["lo"], r["hi"], r["n"]))
    labels = {"1/3": "самое сильное падение", "2/3": "среднее падение", "3/3": "слабое падение"}
    types = {"persuadable": "уговариваемые", "sure_thing": "и так останутся", "lost_cause": "все равно уйдут",
             "sleeping_dog": "спящие собаки"}
    fig, ax = plt.subplots(figsize=(9.5, 6))
    y = 0
    ticks, tl = [], []
    last = None
    for name, grp, diff, lo, hi, n in rows:
        if name != last:
            y += 0.7
            ax.text(-0.5, y - 0.1, name, fontsize=9.5, color=INK2, fontweight="bold", ha="left", va="center",
                    transform=ax.get_yaxis_transform())
            y += 0.7
            last = name
        lab = labels.get(grp, types.get(grp, grp))
        ax.plot([lo * 100, hi * 100], [y, y], color=BLUE, linewidth=2.2, solid_capstyle="round")
        ax.scatter(diff * 100, y, color=BLUE, s=36, zorder=3)
        if name.startswith("тип") and grp in truth_by_type and truth_by_type[grp] is not None:
            ax.scatter(truth_by_type[grp] * 100, y, marker="D", color=ORANGE, s=44, zorder=4)
        ticks.append(y)
        tl.append(f"{lab} (n={n})")
        y += 1
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tl)
    ax.invert_yaxis()
    ax.set_xlabel("разница в уходах за 4 недели, п.п. (меньше нуля: оффер помогает)")
    ax.set_title("Кому оффер помогает: синий это оценка, ромб это заложенное значение")
    ax.grid(axis="y", visible=False)
    return style.save(fig, out)


def qini(qini_res, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    names = {"uplift_logreg": ("uplift, логистическая регрессия", BLUE), "uplift_gbm": ("uplift, бустинг", AQUA),
             "risk": ("по риску ухода", ORANGE)}
    for k, (lab, c) in names.items():
        x, q, a = qini_res[k]
        ax.plot(x, q, color=c, label=f"{lab}")
    x, q, _ = qini_res["risk"]
    ax.plot([0, 1], [0, q[-1]], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label="случайный выбор")
    ax.set_xlabel("доля людей, которым предложили оффер")
    ax.set_ylabel("прирост удержанных людей")
    ax.set_title("Кривые Qini: в пределах шума модели не лучше случайного выбора")
    ax.legend(loc="upper left")
    return style.save(fig, out)


def policy_value(policy, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    names = {"oracle": ("идеальный выбор (знает типы)", YELLOW), "uplift_logreg": ("uplift, логистическая регрессия", BLUE),
             "risk": ("по риску ухода", ORANGE), "random": ("случайный выбор", MUTED)}
    for k, (lab, c) in names.items():
        x, s = policy[k]
        ax.plot(x, s, color=c, label=lab)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("доля людей, которым предложили оффер")
    ax.set_ylabel("сколько людей оффер удержал (по настоящим эффектам)")
    ax.set_title("Если бы мы знали настоящие эффекты: потолок намного выше, чем дают признаки")
    ax.legend(loc="lower right")
    return style.save(fig, out)


def did_chart(h, out):
    s = h["series"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    sm = s[["treated", "control"]].rolling(4, min_periods=1).mean() * 100
    ax.plot(sm.index, sm["control"], color=MUTED, label="регионы без повышения")
    ax.plot(sm.index, sm["treated"], color=ORANGE, label="Volga, South, Siberia")
    ax.axvline(HIKE_WEEK, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(HIKE_WEEK + 0.5, ax.get_ylim()[1] * 0.98, "повышение цены", fontsize=8.5, color=INK2, va="top")
    ax.set_xlabel("номер недели")
    ax.set_ylabel("недельный отток, % (среднее за 4 недели)")
    ax.set_title("Отток в двух группах регионов")
    ax.legend(loc="lower left")
    ax = axes[1]
    es = h["event_study"]
    ax.axhspan(-1.96 * h["base_sd"] * 100, 1.96 * h["base_sd"] * 100, color=MUTED, alpha=0.18, linewidth=0)
    ax.scatter(es["rel_week"], es["effect"] * 100, color=BLUE, s=16, alpha=0.7)
    roll = es["effect"].rolling(4, min_periods=1).mean() * 100
    ax.plot(es["rel_week"], roll, color=BLUE, linewidth=2)
    ax.axhline(h["truth_pp"], color=ORANGE, linewidth=1.6)
    ax.text(es["rel_week"].min() + 1, h["truth_pp"] + 0.04, "заложенный эффект", color=ORANGE, fontsize=9)
    ax.axvline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("недель до и после повышения цены")
    ax.set_ylabel("разница между группами, п.п. в неделю")
    ax.set_title(f"Разница между группами: до нуля, после около {h['did'] * 100:.2f} п.п.")
    return style.save(fig, out)


def did_summary(h, out):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    vals = [(h["naive"] * 100, None, "до и после в одной группе"), (h["did"] * 100, h["ci"], "difference in differences"),
            (h["placebo"] * 100, None, "плацебо: фальшивая дата")]
    for i, (v, ci, lab) in enumerate(vals[::-1]):
        color = BLUE if "differences" in lab else MUTED
        if ci is not None:
            ax.plot([ci[0] * 100, ci[1] * 100], [i, i], color=color, linewidth=2.4, solid_capstyle="round")
        ax.scatter(v, i, color=color, s=50, zorder=3)
    ax.axvline(h["truth_pp"], color=ORANGE, linewidth=2)
    ax.text(h["truth_pp"], 2.55, " заложенный эффект", color=ORANGE, fontsize=9, va="center")
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(3))
    ax.set_yticklabels([v[2] for v in vals[::-1]])
    ax.set_ylim(-0.5, 2.9)
    ax.set_xlabel("эффект повышения цены на недельный отток, п.п.")
    ax.set_title("Три способа оценить эффект: наивное сравнение его почти не видит")
    ax.grid(axis="y", visible=False)
    return style.save(fig, out)


def make_all(res, truth, outdir):
    from pathlib import Path
    out = Path(outdir)
    tb = {k: v["effect"] for k, v in truth["offer"]["by_type"].items()}
    return [
        ab_result(res["itt"], truth["offer"]["true_effect_on_treated_4w_churn"], out / "23_ab_result.png"),
        subgroups(res["by_ratio"], res["by_plan"], res["by_type"], tb, out / "24_ab_subgroups.png"),
        qini(res["uplift"]["qini"], out / "25_qini.png"),
        policy_value(res["uplift"]["policy"], out / "26_policy_value.png"),
        did_chart(res["hike"], out / "27_did_event_study.png"),
        did_summary(res["hike"], out / "28_did_summary.png"),
    ]
