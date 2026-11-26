"""The break-even direct size N*(H) against the hub-step size H (break-even grid, figure 2).

See README.md. Reads experiments/break-even-grid/results/nstar.csv (committed, 7c08d0a) and the
grid's config; writes break_even.pdf and .png next to this file. Medians over pairs with their
90% pair-bootstrap intervals, the pooled-mean crossing as a check, censored values as arrows, and
a slope-1 line (growth in proportion to H). Deterministic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "experiments" / "break-even-grid" / "results"
# dataviz reference palette: categorical slots 1 and 2, text and grid tokens (light mode)
MEDIAN, POOLED = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
RATIO = 4      # the slope-1 line is drawn at N* = H / RATIO (README: a reference, not a fit)


def main() -> None:
    xcfg = yaml.safe_load((REPO / "experiments" / "break-even-grid" / "config.yaml")
                          .read_text(encoding="utf-8"))
    lo_n, hi_n = min(xcfg["direct_sizes"]), max(xcfg["direct_sizes"])
    t = pd.read_csv(RESULTS / "nstar.csv")
    t = t[t["hub"] != "full"].assign(hub=lambda d: d["hub"].astype(int)).reset_index(drop=True)
    h = t["hub"].to_numpy(float)
    med, lo, hi = (2.0 ** t[c].to_numpy(float) for c in ("median_log2", "lo_log2", "hi_log2"))
    pooled = 2.0 ** t["pooled_log2"].to_numpy(float)

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    xmin, xmax, ymin, ymax = 700, 92000, 250, 32000
    # outside the direct sizes measured: N* there can only be "below" or "above"
    for y0, y1 in ((ymin, lo_n), (hi_n, ymax)):
        ax.axhspan(y0, y1, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(xmax * 0.95, lo_n * 0.93, f"below {lo_n:,}: not measured", va="top", ha="right",
            fontsize=6.5, color=INK2)
    ax.text(xmin * 1.08, hi_n * 1.08, f"above {hi_n:,}: not measured", va="bottom",
            fontsize=6.5, color=INK2)
    xs = np.array([xmin, xmax])
    ax.plot(xs, xs / RATIO, ls=(0, (4, 3)), lw=1, color=MUTED, zorder=1,
            label=f"N* = H / {RATIO}: growth in proportion to H")

    ok = np.isfinite(med) & (med > 0)          # "below 500" is stored as -inf (2^-inf = 0)
    # 90% intervals; an open upper end becomes an arrow from the grid's top
    for x, a, b in zip(h[ok], lo[ok], hi[ok], strict=True):
        ax.plot([x, x], [a, min(b, hi_n)], color=MEDIAN, lw=1.5, solid_capstyle="round",
                zorder=2)
        if not np.isfinite(b):
            ax.annotate("", xy=(x, hi_n * 1.9), xytext=(x, hi_n),
                        arrowprops={"arrowstyle": "-|>", "color": MEDIAN, "lw": 1.5}, zorder=2)
    ax.plot(h[ok], med[ok], color=MEDIAN, lw=2, marker="o", ms=5, mec="white", mew=1,
            zorder=4, label="median over pairs (90% interval)")
    okp = np.isfinite(pooled) & (pooled > 0)
    ax.plot(h[okp], pooled[okp], ls="none", marker="o", ms=5, mfc="none", mec=POOLED, mew=1.3,
            zorder=3, label="pooled-mean crossing (check)")
    # "below 500": the chain never beats direct at the smallest direct size
    for x in h[~ok]:
        ax.annotate("", xy=(x, lo_n / 1.9), xytext=(x, lo_n),
                    arrowprops={"arrowstyle": "-|>", "color": MEDIAN, "lw": 1.5}, zorder=2)
        ax.plot(x, lo_n, marker="v", ms=5, color=MEDIAN, mec="white", mew=1, zorder=4)
    for x, m in zip(h[ok], med[ok], strict=True):
        ax.annotate(f"{m:,.0f}", (x, m), xytext=(-6, 3), textcoords="offset points",
                    ha="right", fontsize=6.5, color=INK)

    ax.set_xticks(h, [f"{int(x / 1000)}k\n({n})" for x, n in zip(h, t["n_pairs"], strict=True)])
    yt = [500, 1000, 2000, 4000, 8000, 16000]
    ax.set_yticks(yt, [f"{y / 1000:g}k" if y >= 1000 else f"{y}" for y in yt])
    ax.minorticks_off()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("hub-step size H, both steps (language pairs)")
    ax.set_ylabel("break-even direct size N*(H)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.87), frameon=False, fontsize=6.5,
              labelcolor=INK2, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"break_even.{ext}", dpi=300,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
