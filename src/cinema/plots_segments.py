# графики сегментации: выбор числа кластеров, карта PCA, профили сегментов, дожитие

import numpy as np
from matplotlib.ticker import PercentFormatter

from . import style
from .style import AQUA, BLUE, INK, INK2, MUTED, ORANGE, YELLOW, plt

SEG_COLORS = [BLUE, ORANGE, AQUA, YELLOW]
PERSONA_LABELS = {"binge": "binge", "weekend": "weekend", "casual": "casual", "light": "light"}


def choose_k(scan, k, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(scan["k"], scan["inertia"] / 1000, color=BLUE, marker="o")
    axes[0].set_title("Инерция: локоть около четырех")
    axes[0].set_ylabel("сумма квадратов, тыс")
    axes[1].plot(scan["k"], scan["silhouette"], color=ORANGE, marker="o")
    axes[1].set_title("Силуэт: лучший у трех, четыре чуть хуже")
    axes[1].set_ylabel("средний силуэт")
    for ax in axes:
        ax.axvline(k, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlabel("число сегментов")
        ax.grid(axis="x", visible=False)
    return style.save(fig, out)


def pca_map(pts, evr, names, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for s in sorted(pts["segment"].unique()):
        m = pts["segment"] == s
        axes[0].scatter(pts.loc[m, "pc1"], pts.loc[m, "pc2"], s=5, alpha=0.45, color=SEG_COLORS[s], label=names[s])
    for i, p in enumerate(["binge", "weekend", "casual", "light"]):
        m = pts["persona"] == p
        axes[1].scatter(pts.loc[m, "pc1"], pts.loc[m, "pc2"], s=5, alpha=0.45, color=SEG_COLORS[i], label=p)
    axes[0].set_title("Что нашел kmeans")
    axes[1].set_title("Что заложено в данных")
    for ax in axes:
        ax.set_xlabel(f"главная компонента 1 ({evr[0]:.0%} разброса)")
        leg = ax.legend(markerscale=4, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), columnspacing=1.0)
        for h in leg.legend_handles:
            h.set_alpha(1)
    axes[0].set_ylabel(f"главная компонента 2 ({evr[1]:.0%})")
    return style.save(fig, out)


def profile_heatmap(prof, names, out):
    cols = ["mean_hours", "hours_per_session", "titles_per_session", "weekend_share", "hours_cv"]
    labels = ["часов\nв неделю", "часов\nза сессию", "названий\nза сессию", "доля\nвыходных", "разброс\nпо неделям"]
    vals = prof[cols].values
    z = (vals - vals.mean(axis=0)) / vals.std(axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.imshow(z, cmap=style.blue_cmap(), vmin=-1.8, vmax=1.8, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            txt = f"{vals[i, j]:.0%}" if cols[j] == "weekend_share" else f"{vals[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="white" if abs(z[i, j]) > 1.0 and z[i, j] > 0 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels([f"{names[i]} ({prof.loc[i, 'share']:.0%})" for i in prof.index])
    ax.grid(False)
    ax.set_title("Портрет сегментов (цвет: выше или ниже среднего по признаку)")
    return style.save(fig, out)


def persona_table(tab, names, out):
    share = tab.div(tab.sum(axis=1), axis=0)
    cols = ["binge", "weekend", "casual", "light"]
    share = share[cols]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.imshow(share.values, cmap=style.blue_cmap(), vmin=0, vmax=1, aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(share.shape[0]))
    ax.set_yticklabels([names[i] for i in share.index])
    ax.set_xlabel("заложенный тип зрителя")
    ax.grid(False)
    ax.set_title("Кто попал в какой сегмент: casual и light смешались")
    return style.save(fig, out)


def km_segments(km, names, out, horizon=78):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for s in sorted(km):
        k = km[s]
        k = k[k["time"] <= horizon]
        ax.step(k["time"], k["surv"], where="post", color=SEG_COLORS[s], label=names[s])
        ax.fill_between(k["time"], k["lo"], k["hi"], step="post", color=SEG_COLORS[s], alpha=0.12, linewidth=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("недель с первой оплаты")
    ax.set_ylabel("доля не отменивших")
    ax.set_title("Дожитие по сегментам: редкие зрители уходят заметно раньше остальных")
    ax.legend()
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1.02)
    return style.save(fig, out)


def make_all(seg, km, outdir):
    from pathlib import Path
    out = Path(outdir)
    return [
        choose_k(seg["scan"], seg["k"], out / "13_choose_k.png"),
        pca_map(seg["pca"], seg["pca_var"], seg["names"], out / "14_segments_pca.png"),
        profile_heatmap(seg["profile"], seg["names"], out / "15_segment_profiles.png"),
        persona_table(seg["agreement"]["table"], seg["names"], out / "16_segment_vs_persona.png"),
        km_segments(km, seg["names"], out / "17_km_segments.png"),
    ]
