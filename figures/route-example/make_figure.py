"""One language pair's direct dictionary and its three widest hubs (the methods figure).

See README.md. Reads <phase6>/routes/dict_sizes.csv (distinct word pairs per dictionary, the
graph of 30 nodes and stage 1's six low-resource languages) and experiments/scarce-natural-s1/
pairs.csv (the under-500 pairs). The example pair is chosen by rule: the under-500 pair whose
ratio (the widest hub's smaller dictionary / direct) is closest to the group's median ratio (in
log). Hubs are ranked as prep/19_strong_hubs.py ranks them (the largest smaller dictionary
first). Writes route_example.pdf and .png next to this file. Deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from clwe_chains import load_config
from clwe_chains.llm import NAMES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAIRS = REPO / "experiments" / "scarce-natural-s1" / "pairs.csv"
LOW = {"yo", "or", "mg", "mhr", "as", "myv"}           # stage 1's low-resource languages
N_HUBS, N_SHOWN = 3, 5          # several hubs uses three; two more shown, greyed (user, 2026-11-25)
# dataviz reference palette: blue (the chain), its lighter step (the other hubs used), grey (the
# next hubs), yellow marks a low-resource language; text tokens (light)
CHAIN, OTHER, UNUSED, YELLOW = "#2a78d6", "#86b6ef", "#c9c8c2", "#f2c94c"
INK, INK2 = "#0b0b0b", "#52514e"
LW = 2.4


def name(code: str) -> str:
    return NAMES[code].split(" (")[0]


def load_sizes(phase6: Path) -> tuple[dict[frozenset, int], list[str]]:
    r = pd.read_csv(phase6 / "routes" / "dict_sizes.csv", dtype={"pairs": int})
    size = {frozenset((a, b)): n for a, b, n in zip(r["a"], r["b"], r["pairs"], strict=True)}
    nodes = sorted((set(r["a"]) | set(r["b"])) - LOW)
    return size, nodes


def ranked(size, nodes, s, t):
    via = [(h, size[frozenset((s, h))], size[frozenset((h, t))]) for h in nodes
           if h not in (s, t) and frozenset((s, h)) in size and frozenset((h, t)) in size]
    return sorted(via, key=lambda v: -min(v[1], v[2]))


def pick_pair(size, nodes) -> tuple[str, str, float, float]:
    p = pd.read_csv(PAIRS)
    rows = []
    for s, t in zip(p["s"], p["t"], strict=True):
        _, a, b = ranked(size, nodes, s, t)[0]
        rows.append((s, t, min(a, b) / size[frozenset((s, t))]))
    q = pd.DataFrame(rows, columns=["s", "t", "ratio"])
    med = q["ratio"].median()
    q["dist"] = (np.log(q["ratio"]) - np.log(med)).abs()
    best = q.sort_values(["dist", "s", "t"]).iloc[0]
    return best["s"], best["t"], best["ratio"], med


def label(ax, p, q, text, above: bool) -> None:
    """A number at the middle of segment p-q, set off the line on one side."""
    mid = (p + q) / 2
    d = q - p
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    if (nrm[1] < 0) == above:
        nrm = -nrm
    x, y = mid + 0.07 * nrm
    ax.text(x, y, text, ha="center", va="center", fontsize=7, color=INK2, zorder=4)


def main() -> None:
    cfg = load_config()
    phase6 = cfg.path("phase6")
    size, nodes = load_sizes(phase6)
    s, t, ratio, med = pick_pair(size, nodes)
    hubs = ranked(size, nodes, s, t)[:N_SHOWN]
    direct = size[frozenset((s, t))]
    print(f"pair {s}-{t}: ratio {ratio:.1f} (group median {med:.1f}); direct {direct}; "
          f"hubs {hubs}")

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    S, T = np.array([-1.0, 0.8]), np.array([1.0, 0.8])
    ys = np.linspace(1.5, 0.1, N_SHOWN)
    for i, ((h, a, b), y) in enumerate(zip(hubs, ys, strict=True)):
        col = CHAIN if i == 0 else (OTHER if i < N_HUBS else UNUSED)
        H = np.array([0.0, y])
        left, right = H - np.array([0.13, 0]), H + np.array([0.13, 0])
        ax.plot(*zip(S, left, strict=True), color=col, lw=LW, solid_capstyle="round", zorder=1)
        ax.plot(*zip(right, T, strict=True), color=col, lw=LW, solid_capstyle="round", zorder=1)
        above = y >= S[1]
        label(ax, S, left, f"{a:,}", above)
        label(ax, right, T, f"{b:,}", above)
        ax.text(*H, name(h), ha="center", va="center", fontsize=7.5, color=INK, zorder=3,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": col, "lw": 1.4})
    # the direct dictionary: an arc below the hubs
    u = np.linspace(0, 1, 60)
    ctrl = np.array([0.0, -1.15])
    arc = ((1 - u) ** 2)[:, None] * S + (2 * (1 - u) * u)[:, None] * ctrl + (u ** 2)[:, None] * T
    ax.plot(arc[:, 0], arc[:, 1], color=INK2, lw=LW, ls=(0, (3, 2)), zorder=1)
    ax.text(0, arc[:, 1].min() - 0.05, f"{direct:,}", ha="center", va="top", fontsize=7,
            color=INK2)
    for P, c in ((S, s), (T, t)):
        ax.scatter(*P, s=1300, color=YELLOW if c in LOW else "white", edgecolor=INK2, lw=1,
                   zorder=2)
        ax.text(*P, name(c), ha="center", va="center", fontsize=8, color=INK, zorder=3)
    handles = [Line2D([], [], color=CHAIN, lw=LW), Line2D([], [], color=OTHER, lw=LW),
               Line2D([], [], color=UNUSED, lw=LW),
               Line2D([], [], color=INK2, lw=LW, ls=(0, (3, 2)))]
    ax.legend(handles, ["chain (widest hub)", "also in several hubs", "next hubs (not used)",
                        "direct dictionary"], loc="lower center", ncol=4, frameon=False,
              fontsize=6.5, bbox_to_anchor=(0.5, 0.0), labelcolor=INK2, handlelength=1.6)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.55, 1.68)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"route_example.{ext}", dpi=300,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
