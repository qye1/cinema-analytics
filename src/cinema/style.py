# общий стиль графиков, цвета те же, что в соседних проектах

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
# ступени одного синего цвета от светлого к темному, для тепловой карты
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]


# цвета каналов одни и те же на всех графиках, organic серый как нейтральный
CHANNEL_COLORS = {"organic": MUTED, "search": BLUE, "social": ORANGE, "referral": AQUA, "partner": YELLOW}


def blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


def save(fig, path):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def only_y_grid(ax):
    ax.grid(axis="x", visible=False)


def reference_line(ax, y, text, color=MUTED):
    ax.axhline(y, color=color, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(text, xy=(1, y), xycoords=("axes fraction", "data"),
                xytext=(-2, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.5, color=INK2)


setup_style()
