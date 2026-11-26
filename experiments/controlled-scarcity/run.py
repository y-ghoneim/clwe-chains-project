"""Controlled scarcity (BLI): stage D's runner on each budget's data. See README.md (approved
before running).

For every budget folder <phase6>/dc/b<budget>/ (prep/18; "full" is the reference), stage 1's runner
(experiments/scarce-natural-s1/run.py, unchanged) runs this experiment's methods on that folder's
pairs: direct on the budget (thin), the chain on the full hub dictionaries, their combination and
joined direct (mutual best). Each budget has its own cache (<results>/controlled-scarcity/
b<budget>/) and its own results/b<budget>/; curve.csv and cells.csv then collect them.
`--data-files` lists every input for the remote copy; `--results-only` rebuilds the tables from
the caches.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
NAME = HERE.name
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    f"{NAME}_s1_run", HERE.parent / "scarce-natural-s1" / "run.py")
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)
BUDGETS = [*s1.CFG["controlled_scarcity"]["budgets"], "full"]


def fold_dicts(ctx, s: str, t: str, x: str, fold: int) -> tuple[dict, dict]:
    """Stage 1's fold_dicts with the direct dictionary taken from the budget folder's direct/
    (the shrunk one) instead of dicts/ (where every edge is full, hub edges included)."""
    test = ctx.folds[s, t][fold]
    ts, tt = ctx.test_rows(s, test), ctx.test_rows(t, test)
    direct = s1.remove_test(pd.read_parquet(ctx.p6 / "direct" / f"{s}__{t}.parquet"), ts, tt)
    d = {"direct": direct,
         "sx": s1.remove_test(ctx.dictionary(f"{s}__{x}"), ts, set()),
         "xt": s1.remove_test(ctx.dictionary(f"{x}__{t}"), set(), tt)}
    if any(m[0] == "joined" for m in ctx.methods(s, x)):
        counts = ctx.dictionary(f"{s}__{x}__{t}")
        d["joined_mb"] = s1.joined_dictionary(direct, counts, ts, tt, filtered=True)
    return d, {name: s1.dict_sizes(v) for name, v in d.items()}


class Ctx(s1.Ctx):
    """Stage 1's context on a budget folder: that folder's pairs, this experiment's methods."""

    def __init__(self, limit: int | None):
        super().__init__(limit)
        self.all_pairs = pd.read_csv(self.p6 / "pairs.csv")
        self.pairs = self.all_pairs if limit is None else self.all_pairs.head(limit)

    def methods(self, s: str, hub: str) -> list[tuple[str, ...]]:
        return [s1.method_of(m, hub) for m in XCFG["methods"]]


def configure(budget) -> None:
    """Point stage 1's runner at one budget's data, cache and results."""
    s1.NAME, s1.DATA = f"{NAME}/b{budget}", f"{XCFG['data']}/b{budget}"
    s1.HERE, s1.RESULTS = HERE, HERE / "results" / f"b{budget}"
    (HERE / "results").mkdir(exist_ok=True)
    s1.RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/scarce-natural-s1/run.py",
                       "experiments/scarce-natural-s1/config.yaml", f"experiments/{NAME}/")
    s1.OWN_RESULTS = f"experiments/{NAME}/results/"
    s1.Ctx = Ctx
    s1.fold_dicts = fold_dicts
    s1.XCFG = {**s1.XCFG, "stage": XCFG, "budget": budget}


def chain_steps(sizes: pd.Series, s: str, x: str, t: str) -> int:
    return int(min(sizes.get(f"{s}__{x}", sizes.get(f"{x}__{s}")),
                   sizes.get(f"{x}__{t}", sizes.get(f"{t}__{x}"))))


R_BINS = [-np.inf, 0, 1, 2, 3, 4, 5, np.inf]
R_LABELS = ["r<=0", "0-1", "1-2", "2-3", "3-4", "4-5", "r>5"]


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    """cells.csv: every (budget, direction, method) row of the budgets' pooled.csv with the
    budget's direct size and r = log2(narrowest chain step / direct pairs); curve.csv: per budget
    and method, the mean change in MRR over directions and how many are above / below 0."""
    p6 = s1.CFG.path("phase6") / XCFG["data"]
    cells = []
    for b in BUDGETS:
        f = HERE / "results" / f"b{b}" / "pooled.csv"
        if not f.exists():
            continue
        sizes = pd.read_csv(p6 / f"b{b}" / "dict_sizes.csv").set_index("edge")["pairs"]
        pairs = pd.read_csv(p6 / f"b{b}" / "pairs.csv")
        hub = {frozenset((s, t)): x for s, t, x in zip(pairs["s"], pairs["t"], pairs["hub"],
                                                        strict=True)}
        for r in pd.read_csv(f).itertuples():
            x = hub[frozenset((r.src, r.tgt))]
            s, t = next((s, t) for s, t in zip(pairs["s"], pairs["t"], strict=True)
                        if {s, t} == {r.src, r.tgt})
            direct = int(sizes[f"direct:{s}__{t}"])
            cells.append({"budget": b, "src": r.src, "tgt": r.tgt, "hub": x, "method": r.method,
                          "direct_pairs": direct, "mrr": r.mrr, "mrr_thin": r.mrr_thin,
                          "delta": r.delta, "lo": r.lo, "hi": r.hi,
                          "r": float(np.log2(chain_steps(sizes, s, x, t) / direct))})
    cells = pd.DataFrame(cells)
    if cells.empty:
        return cells, cells
    curve = (cells.groupby(["budget", "method"], sort=False)
             .agg(n_dirs=("delta", "size"), mean_delta=("delta", "mean"),
                  mean_mrr=("mrr", "mean"), mean_mrr_thin=("mrr_thin", "mean"),
                  n_above=("lo", lambda v: int((v > 0).sum())),
                  n_below=("hi", lambda v: int((v < 0).sum())))
             .reset_index())
    return cells, curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs per budget")
    ap.add_argument("--data-files", action="store_true", help="list inputs and exit")
    ap.add_argument("--results-only", action="store_true", help="rebuild results from the caches")
    ap.add_argument("--budgets", default=None, help="comma-separated subset (default: all)")
    args = ap.parse_args()
    budgets = BUDGETS if args.budgets is None else [
        b if b == "full" else int(b) for b in args.budgets.split(",")]
    if args.data_files:
        files = set()
        for b in budgets:
            configure(b)
            files |= set(s1.data_files(pd.read_csv(s1.p6_dir() / "pairs.csv")))
        for f in sorted(files):
            print(f.as_posix())
        return
    for b in budgets:
        configure(b)
        s1.check_clean(s1.git("status", "--porcelain").splitlines())
        print(f"budget {b}", flush=True)
        ctx = Ctx(args.limit)
        if not args.results_only:
            s1.run_all(ctx)
        s1.write_results(ctx)
    cells, curve = collect()
    kw = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    cells.to_csv(HERE / "results" / "cells.csv", **kw)
    if not cells.empty:          # the controlled curve against r, for the natural one next to it
        (cells.assign(r_bin=pd.cut(cells["r"], R_BINS, labels=R_LABELS))
         .groupby(["method", "r_bin"], observed=True)["delta"]
         .agg(n="size", mean_delta="mean").reset_index()
         .to_csv(HERE / "results" / "curve_by_r.csv", **kw))
    curve.to_csv(HERE / "results" / "curve.csv", **kw)
    print(f"collected {len(cells)} cells into results/cells.csv and results/curve.csv")


if __name__ == "__main__":
    main()
