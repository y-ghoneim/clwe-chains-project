"""The language graph with natural dictionaries (review C3). See README.md (approved 2026-11-21).

Reads stage 1E's edge sizes (v1/phase6/routes/dict_sizes.csv) and stage 1's hubs
(experiments/scarce-natural-s1/pairs.csv); writes language_graph.pdf and .png next to this file.
Panel (a): the 30 nodes on a circle by CC-100 size, the six low-resource languages inside, every
edge grey by log10(pairs), each low-resource language's edge to its hub in colour. Panel (b):
Yoruba-Arabic, direct against the route through English. Deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clwe_chains import load_config

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLE = ("yo", "ar", "en")          # panel (b): source, target, hub (chosen before plotting)
HUB_COLOURS = {"en": "#1f6fb4", "hi": "#d1495b", "fr": "#2a9d4b"}


def size_of(sizes: pd.DataFrame, a: str, b: str) -> int:
    hit = sizes[((sizes["a"] == a) & (sizes["b"] == b)) | ((sizes["a"] == b) & (sizes["b"] == a))]
    return int(hit["pairs"].iloc[0])


def layout(nodes: list[str], low: list[str], hub_of: dict[str, str]) -> dict[str, np.ndarray]:
    """Nodes on the unit circle from the top, clockwise, in the given order; each low-resource
    language at radius 0.55, at its hub's angle, fanned out when several share a hub."""
    pos = {}
    for i, n in enumerate(nodes):
        a = np.pi / 2 - 2 * np.pi * i / len(nodes)
        pos[n] = np.array([np.cos(a), np.sin(a)])
    for hub in sorted(set(hub_of.values())):
        mine = [x for x in low if hub_of[x] == hub]
        base = np.arctan2(*pos[hub][::-1])
        for j, x in enumerate(mine):
            a = base + 0.4 * (j - (len(mine) - 1) / 2)
            pos[x] = 0.55 * np.array([np.cos(a), np.sin(a)])
    return pos


def main() -> None:
    cfg = load_config()
    sizes = pd.read_csv(cfg.path("phase6") / "routes" / "dict_sizes.csv")
    pairs = pd.read_csv(REPO / "experiments" / "scarce-natural-s1" / "pairs.csv")
    low = list(cfg["phase6"]["low"])
    hub_of = pairs.groupby("s")["hub"].agg(lambda v: v.value_counts().idxmax()).to_dict()
    gib = cfg["pairs"]["sizes_gib"]
    nodes = sorted((set(sizes["a"]) | set(sizes["b"])) - set(low),
                   key=lambda x: (gib.get(x) is None, -(gib.get(x) or 0.0), x))
    pos = layout(nodes, low, hub_of)
    lw = np.log10(sizes["pairs"].to_numpy())
    lo, hi = lw.min(), lw.max()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for (a, b), w in zip(zip(sizes["a"], sizes["b"], strict=True), lw, strict=True):
        f = (w - lo) / (hi - lo)
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.plot([x0, x1], [y0, y1], color="0.55", lw=0.2 + 2.2 * f, alpha=0.08 + 0.45 * f,
                zorder=1, solid_capstyle="round")
    for x in low:
        h = hub_of[x]
        (x0, y0), (x1, y1) = pos[x], pos[h]
        ax.plot([x0, x1], [y0, y1], color=HUB_COLOURS.get(h, "k"), lw=3.2, zorder=3)
    for n in nodes:
        ax.scatter(*pos[n], s=260, color="white", edgecolor="0.2", zorder=4)
        ax.text(*pos[n], n, ha="center", va="center", fontsize=8, zorder=5)
    for x in low:
        ax.scatter(*pos[x], s=360, color="#f4d35e", edgecolor="0.2", zorder=4)
        ax.text(*pos[x], x, ha="center", va="center", fontsize=7, zorder=5)
    ax.set_title("(a) Natural dictionaries between languages\n"
                 "nodes by CC-100 size (clockwise from top; mt, not in CC-100, last); "
                 "low-resource languages inside,\njoined to the hub the widest-path rule picks "
                 "for them; edge width: log dictionary size (word pairs)", fontsize=9)
    for h, c in HUB_COLOURS.items():
        ax.plot([], [], color=c, lw=3.2, label=f"hub {h}")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_aspect("equal")
    ax.axis("off")

    s, t, h = EXAMPLE
    direct, sh, ht = size_of(sizes, s, t), size_of(sizes, s, h), size_of(sizes, h, t)
    p = {s: np.array([0.0, 0.0]), t: np.array([1.0, 0.0]), h: np.array([0.5, 0.7])}
    for (a, b), n, style in (((s, t), direct, "--"), ((s, h), sh, "-"), ((h, t), ht, "-")):
        (x0, y0), (x1, y1) = p[a], p[b]
        width = 0.6 + 2.5 * (np.log10(n) - lo) / (hi - lo)
        bx.plot([x0, x1], [y0, y1], style, color="0.3" if style == "--" else HUB_COLOURS[h],
                lw=width)
        bx.text((x0 + x1) / 2, (y0 + y1) / 2 + (0.05 if style == "--" else 0.0),
                f"{n:,} pairs", ha="center", va="bottom", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    for n, xy in p.items():
        bx.scatter(*xy, s=500, color="#f4d35e" if n in low else "white", edgecolor="0.2",
                   zorder=4)
        bx.text(*xy, n, ha="center", va="center", fontsize=10, zorder=5)
    bx.set_title(f"(b) One route: {s} → {t}\ndirect (dashed) against the chain through "
                 f"{h}: its narrower step has {min(sh, ht) / direct:.0f}× the direct "
                 "dictionary", fontsize=9)
    bx.set_xlim(-0.25, 1.25)
    bx.set_ylim(-0.3, 0.95)
    bx.set_aspect("equal")
    bx.axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"language_graph.{ext}", dpi=200, bbox_inches="tight",
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    print(f"{len(nodes)} nodes, {len(low)} low-resource languages, {len(sizes)} edges; "
          f"{s}-{t} direct {direct}, {s}-{h} {sh}, {h}-{t} {ht}")


if __name__ == "__main__":
    main()
