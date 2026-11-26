"""Chain - direct against how many times larger each chain dictionary is (break-even grid).

See README.md. Reads experiments/break-even-grid/results/cells.csv (committed, 7c08d0a): MRR of
the chain and the direct map per direction and size combination (both chain dictionaries cut to
H word pairs, the direct one to N). Averages each pair's two directions, then the pairs present
per combination; per ratio H/N, the mean over the combinations and their range (a band). The
same for six sample pairs. Writes break_even_ratio(_pairs).pdf and .png next to this file and
prints the numbers behind them.
Deterministic.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clwe_chains.llm import NAMES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CELLS = REPO / "experiments" / "break-even-grid" / "results" / "cells.csv"
CROSSINGS = CELLS.with_name("crossings.csv")
PAIRS = REPO / "experiments" / "graph-all-pairs" / "pairs.csv"     # stratum (a) / (b)
SAMPLE_HUB, PERCENTILES = 16000, [0, 50, 100]
# dataviz reference palette: categorical slots 1-2; text and grid tokens (light)
CHAIN, DIRECT = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chain - direct per size combination (each pair's two directions averaged, then the pairs
    present in that combination), and per ratio H/N: the mean over its combinations and their
    range."""
    c = pd.read_csv(CELLS, dtype={"hub": str})
    c = c[c["hub"] != "full"].astype({"hub": int})
    pc = c.groupby(["s", "t", "hub", "direct"])["delta"].mean().reset_index()
    combo = pc.groupby(["hub", "direct"])["delta"].agg(["mean", "size"]).reset_index()
    combo = combo.rename(columns={"mean": "delta", "size": "n_pairs"})
    combo["lr"] = np.log2(combo["hub"] / combo["direct"]).round().astype(int)
    by_ratio = combo.groupby("lr")["delta"].agg(["mean", "min", "max", "size"]).reset_index()
    return combo, by_ratio


def crossing(g: pd.DataFrame) -> float:
    """Where the mean line crosses zero, by linear interpolation on log2 of the ratio."""
    x, y = g["lr"].to_numpy(float), g["mean"].to_numpy()
    i = int(np.flatnonzero((y[:-1] <= 0) & (y[1:] > 0))[0])
    return x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])


def sample_pairs() -> pd.DataFrame:
    """The sample (decisions.md 2026-11-26; replaces the 2026-11-25 rule of six percentiles over
    all pairs): within each stratum, (a) two high-resource languages and (b) at least one weaker,
    the pairs present at hub 16k, ranked by their break-even direct size at hub 16k (pairs still
    above zero at their largest direct size rank highest, ties by the gain there), and the pairs
    at the 0th, 50th and 100th percentile (nearest rank). (a) first, then (b)."""
    x = pd.read_csv(CROSSINGS, dtype={"hub": str})
    d = x[x["hub"] == str(SAMPLE_HUB)].copy()
    strata = pd.read_csv(PAIRS)[["s", "t", "stratum"]]
    d = d.merge(strata, on=["s", "t"], how="left", validate="one_to_one")
    assert d["stratum"].notna().all(), "a pair at hub 16k is not among the 40 pairs"
    d["key"] = np.where(d["status"] == "crossed", d["log2"].astype(float),
                        np.where(d["status"] == "below", -np.inf, np.inf))
    d["last_gain"] = d["gains"].map(lambda s: float(s.split(";")[-1].split(":")[1]))
    out = []
    for stratum in ("a", "b"):
        g = d[d["stratum"] == stratum].sort_values(["key", "last_gain", "s", "t"])
        g = g.reset_index(drop=True)
        ranks = [max(1, math.ceil(p / 100 * len(g))) for p in PERCENTILES]
        pick = g.iloc[[r - 1 for r in ranks]].copy()
        pick["percentile"], pick["n_stratum"] = PERCENTILES, len(g)
        out.append(pick)
    return pd.concat(out, ignore_index=True)


def frame(ax, xt: list[int], small: bool, cross: float = 2) -> None:
    """Axes, grid and a dotted line at the crossing (log2 of the ratio; 4x unless given)."""
    ax.axvline(cross, color=INK2, lw=1, ls=(0, (2, 2)), zorder=1)
    ax.set_xticks(xt, [f"1/{2 ** -k}" if k < 0 else f"{2 ** k}" for k in xt])
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if small:
        ax.tick_params(labelsize=7)


def pair_panels() -> None:
    c = pd.read_csv(CELLS, dtype={"hub": str})
    c = c[c["hub"] != "full"].astype({"hub": int})
    picks = sample_pairs()
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.4), sharey=True)
    crosses = []
    for ax, (_, p) in zip(axes.flat, picks.iterrows(), strict=True):
        d = c[(c["s"] == p["s"]) & (c["t"] == p["t"])]
        d = d.groupby(["hub", "direct"])["delta"].mean().reset_index()    # the pair's 2 directions
        d["lr"] = np.log2(d["hub"] / d["direct"]).round().astype(int)
        g = d.groupby("lr")["delta"].agg(["mean", "min", "max", "size"]).reset_index()
        x0 = crossing(g)               # where this pair's mean line crosses zero
        crosses.append(2 ** x0)
        frame(ax, [-4, -2, 0, 2, 4, 6], small=True, cross=x0)
        ax.text(x0 + 0.15, -0.6, f"{2 ** x0:.1f}×", fontsize=7, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        b = g[g["size"] >= 2]          # a range needs two or more size combinations
        ax.fill_between(b["lr"], b["min"], b["max"], color=CHAIN, alpha=0.15, lw=0, zorder=2,
                        label="range over dictionary sizes")
        ax.plot(g["lr"], g["mean"], color=CHAIN, lw=1.8, zorder=3,
                label="mean over dictionary sizes")
        ax.set_xlim(-4.5, 7.5)
        name = f"{NAMES[p['s']].split(' (')[0]}–{NAMES[p['t']].split(' (')[0]} ({p['stratum']})"
        ax.set_title(name, fontsize=7.5, color=INK, loc="left")
    fig.supylabel("difference between chain and direct (MRR)", fontsize=8, color=INK2, x=0.005)
    fig.supxlabel("how many times larger each chain dictionary is than the direct dictionary",
                  fontsize=8, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=7.5,
               labelcolor=INK2)
    fig.tight_layout(pad=0.3, rect=(0.025, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"break_even_ratio_pairs.{ext}", dpi=300,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    picks["mean_line_crosses"] = np.round(crosses, 2)
    print(picks[["stratum", "n_stratum", "percentile", "s", "t", "status", "n_star",
                 "mean_line_crosses"]]
          .to_string(index=False))


def main() -> None:
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "font.family": "DejaVu Sans"})
    pair_panels()
    combo, g = load()
    print(g.round(3).to_string(index=False))
    print(f"mean line crosses zero at {2 ** crossing(g):.2f}x")
    at4 = combo[combo["lr"] == 2]
    print("gain at 4x per combination:\n", at4.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    b = g[g["size"] >= 2]              # a range needs two or more size combinations
    ax.fill_between(b["lr"], b["min"], b["max"], color=CHAIN, alpha=0.15, lw=0, zorder=2,
                    label="range over dictionary sizes")
    ax.plot(g["lr"], g["mean"], color=CHAIN, lw=2.2, zorder=3, label="mean over dictionary sizes")
    ax.axvline(2, color=INK2, lw=1, ls=(0, (2, 2)), zorder=1)
    ax.text(2.12, -0.6, "4×", fontsize=8, color=INK2)
    arrow = {"arrowstyle": "-|>", "color": INK2, "lw": 1}
    ax.annotate("", xy=(0.5, 0.47), xytext=(1.85, 0.47), arrowprops=arrow)
    ax.annotate("", xy=(3.5, 0.47), xytext=(2.15, 0.47), arrowprops=arrow)
    ax.text(0.4, 0.47, "direct is better", ha="right", va="center", fontsize=7.5, color=INK)
    ax.text(3.6, 0.47, "chain is better", ha="left", va="center", fontsize=7.5, color=INK)
    frame(ax, [-4, -2, 0, 2, 4, 6], small=False)
    ax.set_ylim(-0.65, 0.55)
    ax.set_xlabel("how many times larger each chain dictionary is than the direct dictionary")
    ax.set_ylabel("difference between chain and direct (MRR)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"break_even_ratio.{ext}", dpi=300,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
