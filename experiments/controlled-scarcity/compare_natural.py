"""Controlled scarcity against the natural gain curve (README outputs; prediction 2).

Reads results/cells.csv and the natural analysis's pairs table after the English list-word fix
(experiments/natural-highres-d/gain-curve/out/pairs.csv, distant pairs; the analysis the
report uses) and writes, with the natural
analysis's units (a pair: mean over its two directions) and bins (one unit of r, floor):
- results/vs_natural_by_r.csv: per method and bin of r, mean gain and pair count, controlled
  (every budget's pairs) and natural (distant pairs);
- results/vs_natural_by_direct.csv (post hoc, after the results): the same by direct dictionary
  size, the bins of the natural analysis's post-hoc table;
- results/fixed_pairs.csv (post hoc): per budget, mean gain over the pairs present at 4,000, 8,000
  or 16,000 (n_pairs: how many of them that budget has; only the 16,000 set is complete at every
  budget), so the curve is not a change of pairs;
- results/crossing.csv (post hoc): per direction, the first budget at which the chain's change in
  MRR is 0 or below ("never" if it stays above at the full dictionary).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ORDER = ["250", "500", "1000", "2000", "4000", "8000", "16000", "full"]
DIRECT_BINS = [0, 500, 1000, 3000, 10000, np.inf]
DIRECT_LABELS = ["<500", "500-1k", "1k-3k", "3k-10k", ">=10k"]
KW = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}


def main() -> None:
    cells = pd.read_csv(HERE / "results" / "cells.csv", dtype={"budget": str})
    cells["pair"] = [" ".join(sorted(p)) for p in zip(cells["src"], cells["tgt"], strict=True)]
    pc = (cells.groupby(["budget", "pair", "method"], sort=False)
          .agg(gain=("delta", "mean"), r=("r", "mean"), direct=("direct_pairs", "first"))
          .reset_index())
    nat = pd.read_csv(REPO / "experiments" / "natural-highres-d" / "gain-curve" / "out"
                      / "pairs.csv")
    nat = nat[nat["stage"] != "B"]                     # the natural fit's distant pairs
    by_r = []
    for name, df in (("controlled", pc), ("natural", nat)):
        g = (df.assign(r_bin=np.floor(df["r"]).astype(int))
             .groupby(["method", "r_bin"])["gain"].agg(n="size", mean_gain="mean").reset_index())
        by_r.append(g.assign(source=name))
    pd.concat(by_r).to_csv(HERE / "results" / "vs_natural_by_r.csv", **KW)
    by_d = []
    for name, df, col in (("controlled", pc, "direct"), ("natural", nat, "direct_pairs")):
        g = (df.assign(direct_bin=pd.cut(df[col], DIRECT_BINS, right=False, labels=DIRECT_LABELS))
             .groupby(["method", "direct_bin"], observed=True)["gain"]
             .agg(n="size", mean_gain="mean", share_above=lambda v: (v > 0).mean())
             .reset_index())
        by_d.append(g.assign(source=name))
    pd.concat(by_d).to_csv(HERE / "results" / "vs_natural_by_direct.csv", **KW)
    fixed = []
    for b in ("4000", "8000", "16000"):
        keep = set(pc.loc[(pc["budget"] == b) & (pc["method"] == "chain"), "pair"])
        g = (pc[pc["pair"].isin(keep)].groupby(["budget", "method"], sort=False)["gain"]
             .agg(gain="mean", n_pairs="size").reset_index())
        fixed.append(g.assign(pairs_of=b))
    pd.concat(fixed).to_csv(HERE / "results" / "fixed_pairs.csv", **KW)
    ch = cells[cells["method"] == "chain"].assign(
        bi=lambda d: d["budget"].map(ORDER.index)).sort_values("bi")
    cross = []
    for (s, t), g in ch.groupby(["src", "tgt"], sort=True):
        neg = g[g["delta"] <= 0]
        cross.append({"src": s, "tgt": t, "hub": g["hub"].iloc[0],
                      "full_direct": int(g.loc[g["budget"] == "full", "direct_pairs"].iloc[0]),
                      "first_nonpositive": neg["budget"].iloc[0] if len(neg) else "never"})
    pd.DataFrame(cross).to_csv(HERE / "results" / "crossing.csv", **KW)


if __name__ == "__main__":
    main()
