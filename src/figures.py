"""Draw Figures 1 and 2 from results/analysis (run src/analyze.py first)."""
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "analysis"
OUT = ROOT / "results" / "figures"

W = 6.25
INK, WINE, TEAL, SLATE = "#1E1E1E", "#7E364A", "#3E6C7B", "#7E8C8D"
INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
SERIF = [f for f in ["CMU Serif", "Times New Roman", "Liberation Serif", "Nimbus Roman"] if f in INSTALLED] or ["DejaVu Serif"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": SERIF, "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "axes.unicode_minus": False, "pdf.fonttype": 42})
ORDER = ["LLM", "Jev", "Decisions", "Jev + LLM", "Decisions + LLM"]
DISPLAY = {"LLM": "LLM", "Jev": "Jev", "Decisions": "OpenAI Decisions", "Jev + LLM": "Jev + LLM", "Decisions + LLM": "OpenAI + LLM"}
STYLE = {"LLM": (SLATE, "s", True), "Jev": (WINE, "o", False), "Decisions": (TEAL, "D", False),
         "Jev + LLM": (WINE, "o", True), "Decisions + LLM": (TEAL, "D", True)}


def read(name):
    with (DATA / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def values(r, prefix=""):
    return [float(r[k]) for k in ("estimate" + prefix, "ci95_low" + prefix, "ci95_high" + prefix)]


def label(fig, x, y, s, **kw):
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    fig.text(x / W, y / fig.get_figheight(), s, **kw)


def heading(fig, x, y, letter, title):
    label(fig, x, y, letter, fontsize=17)
    label(fig, x + 0.28, y, title, fontsize=13)


def axis(fig, x, y, w, h, xlim, xticks, ylim):
    ax = fig.add_axes([x / W, y / fig.get_figheight(), w / W, h / fig.get_figheight()])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks([])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=3.5, width=0.8, pad=4)
    return ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", facecolor="white", metadata={"CreationDate": None, "Producer": None})
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)


def figure_1():
    rows = read("figure_1.csv")
    H = 7.85
    fig = plt.figure(figsize=(W, H))
    heading(fig, 0.06, 7.62, "A", "Overall")
    heading(fig, 3.10, 7.62, "B", "By system")

    ax = axis(fig, 1.43, 6.00, 1.25, 1.42, (0, 20), [0, 10, 20], (-0.5, 2.5))
    groups = {r["system"]: r for r in rows if r["panel"] == "A"}
    for y, name in zip([2, 1, 0], ["LLM", "Decision models", "Teams"]):
        v, lo, hi = values(groups[name], "_pp")
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="D", markersize=5.5, color=INK, markerfacecolor=INK,
                    elinewidth=1.05, capsize=3, capthick=0.9)
        ax.text(-0.08, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12)
        ax.text(v, y - 0.32, f"{v:.1f}", ha="center", va="center", fontsize=12)
    ax.set_xlabel("Excess change (pp)", labelpad=5)

    ax = axis(fig, 4.48, 6.00, 1.55, 1.42, (0, 20), [0, 10, 20], (-0.5, 4.5))
    systems = {r["system"]: r for r in rows if r["panel"] == "B"}
    for y, key in zip([4, 3, 2, 1, 0], ORDER):
        v, lo, hi = values(systems[key], "_pp")
        color, marker, filled = STYLE[key]
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=color, elinewidth=1.05, capsize=3, capthick=0.9, zorder=2)
        ax.plot(v, y, marker=marker, markersize=5.5, markeredgewidth=1.1, markeredgecolor=color,
                markerfacecolor=color if filled else "white", linestyle="none", zorder=3)
        ax.text(-0.07, y, DISPLAY[key], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12)
    ax.set_xlabel("Excess change (pp)", labelpad=5)

    handles = [Patch(facecolor=STYLE[k][0] if STYLE[k][2] else "white", edgecolor=STYLE[k][0], linewidth=1.2, label=DISPLAY[k])
               for k in ["LLM", "Jev + LLM", "Jev", "Decisions + LLM", "Decisions"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.51, 5.36 / H), ncol=3, frameon=False,
               columnspacing=1.35, handlelength=0.9, handletextpad=0.4, borderaxespad=0, labelspacing=0.35)

    detail = {(r["panel"], r["comparison"], r["system"]): r for r in rows}

    def bars(letter, title, items, bottom, height, heading_y):
        heading(fig, 0.06, heading_y, letter, title)
        ax = fig.add_axes([0.63 / W, bottom / H, 5.43 / W, height / H])
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 50, 100])
        ax.set_xlim(-0.55, len(items) - 0.45)
        ax.set_xticks(range(len(items)), [tick for _, tick in items])
        ax.set_ylabel("Excess change (pp)", labelpad=5)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.tick_params(axis="both", length=3.5, width=0.8, pad=4)
        ax.tick_params(axis="x", length=0, pad=9)
        for t in ax.get_xticklabels():
            t.set_y(-0.065)
        for j, key in enumerate(ORDER):
            color, _, filled = STYLE[key]
            for i, (comparison, _) in enumerate(items):
                v, lo, hi = values(detail[letter, comparison, key], "_pp")
                x = i + (j - 2) * 0.155
                ax.bar(x, v, width=0.135, bottom=0, color=color if filled else "white", edgecolor=color, linewidth=1.0, zorder=2)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=color, elinewidth=0.85, capsize=1.8,
                            capthick=0.8, zorder=3)

    bars("C", "Main framings", [
        ("Clinician advice (benign cases)", "Authority"), ("Social labels", "Social\nlabels"),
        ("Refusal of indicated intervention", "Refusal"), ("Demand for non-indicated intervention", "Demand"),
        ("Repeated disagreement", "Repeated\ndisagreement")], 3.00, 1.57, 4.75)
    bars("D", "Social labels", [("Unhoused", "Unhoused"), ("Refugee", "Refugee"),
                                ("Released from prison", "Released from prison")], 0.55, 1.66, 2.42)
    for x, y, w, h in [(0.015, 5.40, 2.90, 2.43), (2.98, 5.40, 3.255, 2.43), (0.015, 2.60, 6.22, 2.30), (0.015, 0.17, 6.22, 2.37)]:
        fig.add_artist(Rectangle((x / W, y / H), w / W, h / H, transform=fig.transFigure, fill=False,
                                 edgecolor="#C5CCCE", linewidth=0.55, clip_on=False, zorder=0))
    save(fig, "Figure_1")


def figure_2():
    rows = {r["system"]: r for r in read("figure_2.csv")}
    fig = plt.figure(figsize=(W, 4.1))
    ax = fig.add_axes([0.75 / W, 0.70 / 4.1, 5.30 / W, 3.12 / 4.1])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.60, 4.60)
    ax.set_ylabel("Reference agreement (%)", labelpad=8)
    ax.set_xticks(range(5), ["LLM", "Jev", "OpenAI\nDecisions", "Jev\n+ LLM", "OpenAI\n+ LLM"])
    ax.tick_params(axis="y", length=3.5, width=0.8, pad=4)
    ax.tick_params(axis="x", length=0, pad=10, labelsize=12.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for x, key in enumerate(ORDER):
        v, lo, hi = values(rows[key], "_pct")
        color, _, filled = STYLE[key]
        ax.bar(x, v, width=0.62, color=color if filled else "white", edgecolor=color, linewidth=1.2, zorder=2)
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=1.05, capsize=4, capthick=1, zorder=3)
        ax.text(x, hi + 2.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=12.5)
    save(fig, "Figure_2")


if __name__ == "__main__":
    figure_1()
    figure_2()
    print("Figures written to", OUT.relative_to(ROOT))
