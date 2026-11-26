"""Break-even grid (BLI): the chain at fixed hub sizes against controlled scarcity's direct maps.
See README.md (approved 2026-11-23).

For every hub size H (prep/20: <phase6>/dh/h<H>/, both hub steps cut to H), stage 1's runner loop
(experiments/scarce-natural-s1/run.py, unchanged) scores the chain alone through each pair's stage D
hub, one CSLS pass per direction and fold, cached per hub size (<results>/break-even-grid/h<H>/).
The direct maps are not refitted: their ranks come from controlled scarcity's caches
(<results>/controlled-scarcity/b<N>/, the same folds and test words). The results then join the two
per test word: cells.csv (ΔMRR chain − direct per hub size, direct size and direction), grid.csv,
crossings.csv (each pair's break-even direct size per hub size), nstar.csv (N*(H): the median
crossing over pairs, and the pooled-mean crossing) and growth.json (how much N* grows from hub 16k
to 64k, and the verdict). The full routes are a reference line, from controlled scarcity's own
chain ranks. `--check` rescores the full-route chain for a few pairs and compares it with those.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from clwe_chains.dictionaries import dict_sizes, remove_test
from clwe_chains.evaluate import bli_multi
from clwe_chains.runs import Cache, file_fingerprints
from clwe_chains.stats import paired_bootstrap

HERE = Path(__file__).resolve().parent
NAME = HERE.name
RESULTS = HERE / "results"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    f"{NAME}_s1_run", HERE.parent / "scarce-natural-s1" / "run.py")
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)
BG = s1.CFG["break_even_grid"]
HUBS = list(BG["hub_sizes"])
KW = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}


# ---------------------------------------------------------------------------------------------
# Scoring the chain

def fold_dicts(ctx, s: str, t: str, x: str, fold: int) -> tuple[dict, dict]:
    """The fold's two hub steps (cut to the hub size by prep/20), test words removed."""
    test = ctx.folds[s, t][fold]
    ts, tt = ctx.test_rows(s, test), ctx.test_rows(t, test)
    d = {"sx": remove_test(ctx.dictionary(f"{s}__{x}"), ts, set()),
         "xt": remove_test(ctx.dictionary(f"{x}__{t}"), set(), tt)}
    return d, {name: dict_sizes(v) for name, v in d.items()}


def evaluate_chain(ctx, s: str, t: str, x: str, fold: int, forward: bool, d: dict,
                   sizes: dict) -> int:
    """The chain's CSLS ranks for one direction of one fold, as stage 1 composes it; returns
    evaluations run (stage 1's evaluate_direction without direct, combination and the rest)."""
    src, tgt = (s, t) if forward else (t, s)
    method = ("chain", x)
    if s1.key(src, tgt, method, fold) in ctx.cache:
        return 0
    es, et, ex = ctx.store[s], ctx.store[t], ctx.store[x]
    e_src, e_tgt = (es, et) if forward else (et, es)
    concepts = ctx.folds[s, t][fold]
    queries = np.array([ctx.query[src, c] for c in concepts])
    golds = [[ctx.gold[src, tgt, kind, c] for c in concepts] for kind in ("main", "list")]
    if forward:
        w = s1.compose([s1.edge_map(d["sx"], es, ex, True), s1.edge_map(d["xt"], ex, et, True)])
    else:
        w = s1.compose([s1.edge_map(d["xt"], ex, et, False),
                        s1.edge_map(d["sx"], es, ex, False)])
    t0 = time.perf_counter()
    ranks = bli_multi(e_src, e_tgt, w, queries, golds, retrieval="csls", k=s1.EVAL["csls_k"],
                      block=s1.CFG["eval"]["block_rows"], half=s1.EVAL["half"])
    ctx.cache.put(s1.key(src, tgt, method, fold),
                  {"concepts": concepts, "rank": ranks[0].tolist(),
                   "rank_list": ranks[1].tolist(), "n_train": sizes,
                   "seconds": time.perf_counter() - t0})
    return 1


class Ctx(s1.Ctx):
    """Stage 1's context on a hub-size folder: that folder's pairs, the chain alone."""

    def __init__(self, limit: int | None):
        super().__init__(limit)
        self.all_pairs = pd.read_csv(self.p6 / "pairs.csv")
        self.pairs = self.all_pairs if limit is None else self.all_pairs.head(limit)

    def methods(self, s: str, hub: str) -> list[tuple[str, ...]]:
        return [("chain", hub)]


def configure(hub) -> None:
    """Point stage 1's runner at one hub size's data and cache ("full": the base stage, uncut;
    used by --check only)."""
    s1.NAME = f"{NAME}/h{hub}"
    s1.DATA = BG["base"] if hub == "full" else f"{BG['data']}/h{hub}"
    s1.RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/scarce-natural-s1/run.py",
                       "experiments/scarce-natural-s1/config.yaml", f"experiments/{NAME}/")
    s1.OWN_RESULTS = f"experiments/{NAME}/results/"
    s1.Ctx = Ctx
    s1.fold_dicts = fold_dicts
    s1.evaluate_direction = evaluate_chain


# ---------------------------------------------------------------------------------------------
# Crossings, N* and its growth (README "Measures")

def crossing(log_sizes: np.ndarray, gains: np.ndarray) -> dict:
    """A pair's break-even direct size on log2 scale: the first size where its gain goes from
    positive to <= 0, linearly interpolated on log2(size); -inf if <= 0 already at the first
    size, +inf if still positive at its largest (`bound`); `rebound` if it rises above 0 again
    after crossing."""
    log_sizes, gains = np.asarray(log_sizes, float), np.asarray(gains, float)
    if gains[0] <= 0:
        return {"log2": -np.inf, "status": "below", "bound": float(log_sizes[0]),
                "rebound": bool((gains[1:] > 0).any())}
    for i in range(1, len(gains)):
        if gains[i] <= 0:
            g0, g1 = gains[i - 1], gains[i]
            at = log_sizes[i - 1] + g0 / (g0 - g1) * (log_sizes[i] - log_sizes[i - 1])
            return {"log2": float(at), "status": "crossed", "bound": float(at),
                    "rebound": bool((gains[i + 1:] > 0).any())}
    return {"log2": np.inf, "status": "above", "bound": float(log_sizes[-1]), "rebound": False}


def median_log(values: np.ndarray) -> float:
    """The median of log2 crossings with censored ones at -inf / +inf (README: placed at the low
    and high ends): the middle value, or the mean of the two middle ones (nan if they are -inf and
    +inf)."""
    v = np.sort(np.asarray(values, float))
    n = len(v)
    if n == 0:
        return np.nan
    if n % 2:
        return float(v[n // 2])
    a, b = v[n // 2 - 1], v[n // 2]
    if np.isinf(a) and np.isinf(b) and a != b:
        return np.nan
    return float(a) if np.isinf(a) else float(b) if np.isinf(b) else float((a + b) / 2)


def growth(log_nstar: list[float]) -> float:
    """How much N* grows from the first to the last growth size (16k -> 64k, 4x): (64k / 16k) **
    slope of log2 N* on log2 H (least squares over the growth sizes). With censoring the slope
    cannot be computed (README); a bootstrap draw then counts as unbounded growth (+inf) only
    where growth is plain: N* above the grid at the largest size or below it at the smallest, the
    other sizes finite. Every other censored draw is nan (left out, its share reported)."""
    y = np.asarray(log_nstar, float)
    sizes = np.asarray(XCFG["growth_sizes"], float)
    if np.isnan(y).any():
        return np.nan
    if np.isfinite(y).all():
        return float((sizes[-1] / sizes[0]) ** np.polyfit(np.log2(sizes), y, 1)[0])
    if y[-1] == np.inf and np.isfinite(y[:-1]).all():
        return np.inf
    if y[0] == -np.inf and np.isfinite(y[1:]).all():
        return np.inf
    return np.nan


def growth_verdict(log_nstar: list[float], lo: float, hi: float) -> str:
    """The verdict on the medians themselves: the README's rule when all three are numbers;
    otherwise what the censoring says."""
    y = np.asarray(log_nstar, float)
    sizes = XCFG["growth_sizes"]
    if np.isfinite(y).all():
        return verdict(lo, hi)
    above = [h for h, v in zip(sizes, y, strict=True) if v == np.inf]
    below = [h for h, v in zip(sizes, y, strict=True) if v == -np.inf]
    parts = []
    if above:
        parts.append(f"no break-even within direct <= {XCFG['direct_sizes'][-1]} at hub "
                     + ", ".join(str(h) for h in above))
    if below:
        parts.append(f"N* below {XCFG['direct_sizes'][0]} at hub "
                     + ", ".join(str(h) for h in below))
    if np.isnan(y).any():
        parts.append("median undefined at hub "
                     + ", ".join(str(h) for h, v in zip(sizes, y, strict=True) if np.isnan(v)))
    return "; ".join(parts) + " (growth not computed)"


def verdict(lo: float, hi: float) -> str:
    """README's rule: "levels off" if the interval lies below the cutoff (2x); "grows in
    proportion" if it contains 4x and excludes 1x; otherwise "undecided"."""
    if hi < XCFG["growth_cutoff"]:
        return "levels off"
    if lo <= XCFG["proportional"] <= hi and lo > 1:
        return "grows in proportion"
    return "undecided"


def pair_crossings(cells: pd.DataFrame) -> pd.DataFrame:
    """Per hub size and pair: the gain at each direct size (mean ΔMRR of its two directions;
    sizes where both are present) and its crossing."""
    rows = []
    both = cells.groupby(["hub", "s", "t", "direct"]).filter(lambda g: len(g) == 2)
    for (hub, s, t), g in both.groupby(["hub", "s", "t"], sort=False):
        gain = g.groupby("direct")["delta"].mean().sort_index()
        c = crossing(np.log2(gain.index.to_numpy(float)), gain.to_numpy())
        rows.append({"hub": hub, "s": s, "t": t, **c,
                     "n_star": 2 ** c["log2"] if c["status"] == "crossed" else np.nan,
                     "gains": ";".join(f"{n}:{v:.4f}" for n, v in gain.items())})
    return pd.DataFrame(rows)


def describe(log2: float, bound: float | None = None) -> str:
    """A log2 crossing as text; "above X" uses `bound` (log2) when given, else the grid's top."""
    if np.isnan(log2):
        return "undefined"
    if log2 == -np.inf:
        return f"below {XCFG['direct_sizes'][0]}"
    if log2 == np.inf:
        return f"above {round(2 ** bound) if bound is not None else XCFG['direct_sizes'][-1]}"
    return f"{2 ** log2:.0f}"


def median_bound(log2: np.ndarray, bound: np.ndarray) -> float:
    """A lower bound for a median that is "above": the median with each pair still positive at
    its largest direct size placed at that size (its crossing is somewhere above it)."""
    v = np.where(np.asarray(log2) == np.inf, np.asarray(bound, float), np.asarray(log2))
    return median_log(v)


def finite_or_text(v):
    """growth.json stays strict JSON: non-finite numbers as the strings "inf", "-inf", "nan"."""
    if isinstance(v, float) and not np.isfinite(v):
        return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


def nstar_table(cells: pd.DataFrame, cross: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """nstar.csv (per hub size: the median crossing, its bootstrap interval over pairs, the
    pooled-mean crossing, counts) and growth.json (the growth of N* over the growth sizes, its
    interval and the verdict). The bootstrap resamples stage D's pairs; each hub size uses the
    resampled pairs present in it, with their multiplicity."""
    q = (100 - XCFG["interval"]) / 2
    bs = XCFG["pair_bootstrap"]
    pairs = sorted({(s, t) for s, t in zip(cross["s"], cross["t"], strict=True)})
    rng = np.random.default_rng(bs["seed"])
    draws = rng.integers(0, len(pairs), size=(bs["n"], len(pairs)))
    by_hub = {h: g.set_index(["s", "t"])["log2"] for h, g in cross.groupby("hub", sort=False)}
    rows, boot = [], {}
    for hub, v in by_hub.items():
        sample = [[v[pairs[i]] for i in d if pairs[i] in v.index] for d in draws]
        boot[hub] = np.array([median_log(np.array(s)) for s in sample])
        g = cross[cross["hub"] == hub]
        med = median_log(v.to_numpy())
        mean = cells[cells["hub"] == hub].groupby("direct")["delta"].mean().sort_index()
        pooled = crossing(np.log2(mean.index.to_numpy(float)), mean.to_numpy())
        b = boot[hub][~np.isnan(boot[hub])]
        lo, hi = (np.quantile(b, [q / 100, 1 - q / 100], method="inverted_cdf") if len(b)
                  else (np.nan, np.nan))
        rows.append({"hub": hub, "n_pairs": len(g), "n_crossed": int((g["status"] == "crossed")
                                                                     .sum()),
                     "n_below": int((g["status"] == "below").sum()),
                     "n_above": int((g["status"] == "above").sum()),
                     "n_rebound": int(g["rebound"].sum()),
                     "median_log2": med,
                     "n_star": describe(med, median_bound(g["log2"].to_numpy(),
                                                          g["bound"].to_numpy())),
                     "lo_log2": lo, "hi_log2": hi, "lo": describe(lo), "hi": describe(hi),
                     "undefined_share": float(np.isnan(boot[hub]).mean()),
                     "pooled_log2": pooled["log2"],
                     "pooled_n_star": describe(pooled["log2"], pooled["bound"]),
                     "pooled_rebound": pooled["rebound"]})
    table = pd.DataFrame(rows)
    sizes = [h for h in XCFG["growth_sizes"] if h in boot]
    out = {"sizes": sizes}
    if len(sizes) == len(XCFG["growth_sizes"]):
        meds = [table.set_index("hub").loc[h, "median_log2"] for h in sizes]
        point = growth(meds)
        gs = np.array([growth([boot[h][i] for h in sizes]) for i in range(bs["n"])])
        kept = gs[~np.isnan(gs)]
        lo, hi = (np.quantile(kept, [q / 100, 1 - q / 100], method="inverted_cdf")
                  if len(kept) else (np.nan, np.nan))
        out |= {"medians_log2": meds, "growth": point, "lo": float(lo), "hi": float(hi),
                "unbounded_share": float(np.isinf(gs).mean()),
                "undefined_share": float(np.isnan(gs).mean()),
                "verdict": growth_verdict(meds, lo, hi),
                "same_pairs": same_pairs_growth(cross)}
    return table, out


def same_pairs_growth(cross: pd.DataFrame) -> dict:
    """Sensitivity check (user, 2026-11-23): the growth on the pairs present at every growth size
    only, so that a change of pairs between hub sizes cannot move N*. Same medians, bootstrap
    (over these pairs) and verdict as the main growth."""
    q = (100 - XCFG["interval"]) / 2
    bs, sizes = XCFG["pair_bootstrap"], XCFG["growth_sizes"]
    sub = cross[cross["hub"].isin(sizes)]
    present = sub.groupby(["s", "t"])["hub"].nunique()
    pairs = sorted(present[present == len(sizes)].index)
    if not pairs:
        return {"n_pairs": 0}
    by_hub = {h: sub[sub["hub"] == h].set_index(["s", "t"])["log2"].loc[pairs].to_numpy()
              for h in sizes}
    meds = [median_log(by_hub[h]) for h in sizes]
    draws = np.random.default_rng(bs["seed"]).integers(0, len(pairs), size=(bs["n"], len(pairs)))
    gs = np.array([growth([median_log(by_hub[h][d]) for h in sizes]) for d in draws])
    kept = gs[~np.isnan(gs)]
    lo, hi = (np.quantile(kept, [q / 100, 1 - q / 100], method="inverted_cdf")
              if len(kept) else (np.nan, np.nan))
    return {"n_pairs": len(pairs), "medians_log2": meds, "growth": growth(meds),
            "lo": float(lo), "hi": float(hi), "unbounded_share": float(np.isinf(gs).mean()),
            "undefined_share": float(np.isnan(gs).mean()),
            "verdict": growth_verdict(meds, lo, hi)}


# ---------------------------------------------------------------------------------------------
# Joining the chain with controlled scarcity's direct ranks

def direct_cache(n: int) -> Cache:
    return Cache(s1.CFG.path("results") / "controlled-scarcity" / f"b{n}" / "evals.jsonl")


def direct_pairs(n: int) -> set[tuple[str, str]]:
    p = pd.read_csv(s1.CFG.path("phase6") / XCFG["direct"] / f"b{n}" / "pairs.csv")
    return set(zip(p["s"], p["t"], strict=True))


def build_cells(folds: dict) -> pd.DataFrame:
    """Per hub size (and "full"), direct size and direction: MRR of the chain and of direct,
    ΔMRR with the paired bootstrap over test words, and r = log2(hub size / direct size)."""
    bs, rows = s1.XCFG["bootstrap"], []
    direct = {n: (direct_cache(n), direct_pairs(n)) for n in XCFG["direct_sizes"]}
    p6 = s1.CFG.path("phase6")
    for hub in [*HUBS, "full"]:
        if hub == "full":
            chains = None
            hub_pairs = pd.read_csv(p6 / BG["base"] / "pairs.csv")
        else:
            path = s1.CFG.path("results") / NAME / f"h{hub}" / "evals.jsonl"
            if not path.exists():
                continue
            chains = Cache(path)
            hub_pairs = pd.read_csv(p6 / BG["data"] / f"h{hub}" / "pairs.csv")
        for s, t, x in zip(hub_pairs["s"], hub_pairs["t"], hub_pairs["hub"], strict=True):
            f = sorted(folds[s, t])
            for n, (dcache, dpairs) in direct.items():
                if (s, t) not in dpairs:
                    continue
                for src, tgt in ((s, t), (t, s)):
                    a = s1.pooled(dcache if chains is None else chains, src, tgt, ("chain", x), f)
                    b = s1.pooled(dcache, src, tgt, ("thin",), f)
                    if a is None or b is None:
                        continue
                    assert (a["concept"].to_numpy() == b["concept"].to_numpy()).all()
                    mean, lo, hi = paired_bootstrap(a["rr"].to_numpy(), b["rr"].to_numpy(), **bs)
                    rows.append({"hub": hub, "direct": n, "s": s, "t": t, "src": src, "tgt": tgt,
                                 "via": x, "mrr_chain": a["rr"].mean(),
                                 "mrr_direct": b["rr"].mean(),
                                 "delta": mean, "lo": lo, "hi": hi, "n": len(a),
                                 "r": np.nan if hub == "full" else float(np.log2(hub / n))})
    return pd.DataFrame(rows)


def grid_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Per hub size and direct size: pairs, directions, mean ΔMRR and MRRs, counts clearly above
    and below 0."""
    return (cells.assign(pair=cells["s"] + "-" + cells["t"]).groupby(["hub", "direct"], sort=False)
            .agg(n_pairs=("pair", "nunique"), n_dirs=("delta", "size"),
                 mean_delta=("delta", "mean"), mean_mrr_chain=("mrr_chain", "mean"),
                 mean_mrr_direct=("mrr_direct", "mean"),
                 n_above=("lo", lambda v: int((v > 0).sum())),
                 n_below=("hi", lambda v: int((v < 0).sum())))
            .reset_index())


def write_results(folds: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    cells = build_cells(folds)
    if cells.empty:
        print("no hub size scored yet")
        return
    cells.to_csv(RESULTS / "cells.csv", **KW)
    grid_table(cells).to_csv(RESULTS / "grid.csv", **KW)
    cross = pair_crossings(cells[cells["hub"] != "full"])
    cross_full = pair_crossings(cells[cells["hub"] == "full"])
    pd.concat([cross, cross_full]).to_csv(RESULTS / "crossings.csv", **KW)
    table, grow = nstar_table(cells, pd.concat([cross, cross_full]))
    table.to_csv(RESULTS / "nstar.csv", **KW)
    def clean(d: dict) -> dict:
        return {k: clean(v) if isinstance(v, dict) else
                [finite_or_text(x) for x in v] if isinstance(v, list) else finite_or_text(v)
                for k, v in d.items()}

    grow = clean(grow)
    (RESULTS / "growth.json").write_text(json.dumps(grow, indent=1) + "\n", encoding="utf-8")
    res, p6 = s1.CFG.path("results"), s1.CFG.path("phase6")
    own = [res / NAME / f"h{h}" for h in HUBS]
    reused = [res / "controlled-scarcity" / f"b{n}" for n in [*XCFG["direct_sizes"], "full"]]

    def commits(dirs):
        return list(dict.fromkeys(c for d in dirs if (d / "run.log").exists()
                                  for c in s1.eval_commits((d / "run.log").read_text("utf-8"))))

    paths = [f for h in HUBS for f in sorted((p6 / BG["data"] / f"h{h}").rglob("*"))
             if f.is_file()]
    paths += [d / "evals.jsonl" for d in own + reused if (d / "evals.jsonl").exists()]
    files = {str(f.relative_to(s1.CFG.data_root)): f for f in paths}
    info = {"commit": s1.git("rev-parse", "HEAD").strip(), "eval_commits": commits(own),
            "direct_eval_commits": commits(reused), "config": XCFG, "grid": BG, "eval": s1.EVAL,
            "word_bootstrap": s1.XCFG["bootstrap"],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "inputs": file_fingerprints(dict(sorted(files.items())))}
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(table[["hub", "n_pairs", "n_star", "lo", "hi", "pooled_n_star"]].to_string(index=False))
    print(json.dumps(grow, indent=1))


def reproduction_check(ctx: Ctx) -> pd.DataFrame:
    """The full-route chain rescored here against controlled scarcity's (bfull): records
    compared, share of queries with the same rank, largest rank and MRR differences."""
    theirs, rows = direct_cache("full"), []
    for s, t, x in zip(ctx.pairs["s"], ctx.pairs["t"], ctx.pairs["hub"], strict=True):
        for src, tgt in ((s, t), (t, s)):
            for f in sorted(ctx.folds[s, t]):
                a = ctx.cache.get(s1.key(src, tgt, ("chain", x), f))
                b = theirs.get(s1.key(src, tgt, ("chain", x), f))
                if a is None or b is None:
                    continue
                assert a["concepts"] == b["concepts"], (src, tgt, f)
                ra, rb = np.asarray(a["rank"]), np.asarray(b["rank"])
                rows.append({"src": src, "tgt": tgt, "fold": f, "n": len(ra),
                             "same_rank_share": float((ra == rb).mean()),
                             "max_rank_diff": int(np.abs(ra - rb).max()),
                             "mrr_diff": float((1 / ra).mean() - (1 / rb).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs per size")
    ap.add_argument("--hubs", default=None, help="comma-separated hub sizes (default: all)")
    ap.add_argument("--results-only", action="store_true", help="rebuild results from the caches")
    ap.add_argument("--check", action="store_true",
                    help="rescore the full-route chain (--limit pairs, default 2) and compare")
    args = ap.parse_args()
    if args.check:
        configure("full")
        s1.check_clean(s1.git("status", "--porcelain").splitlines())
        ctx = Ctx(args.limit or 2)
        s1.run_all(ctx)
        out = reproduction_check(ctx)
        RESULTS.mkdir(exist_ok=True)
        out.to_csv(RESULTS / "check.csv", **KW)
        print(out.to_string(index=False))
        tol = XCFG["check_tolerance"]
        ok = len(out) > 0 and float(out["mrr_diff"].abs().max()) <= tol
        print(f"check {'PASS' if ok else 'FAIL'}: {len(out)} records, largest |MRR difference| "
              f"{out['mrr_diff'].abs().max() if len(out) else float('nan'):.2g} (tolerance {tol})")
        if not ok:
            raise SystemExit(1)
        return
    hubs = HUBS if args.hubs is None else [int(h) for h in args.hubs.split(",")]
    folds = None
    for hub in hubs:
        configure(hub)
        s1.check_clean(s1.git("status", "--porcelain").splitlines())
        print(f"hub {hub}", flush=True)
        ctx = Ctx(args.limit)
        folds = ctx.folds
        if not args.results_only:
            s1.run_all(ctx)
    if folds is None:
        configure(HUBS[0])
        folds = Ctx(None).folds
    write_results(folds)


if __name__ == "__main__":
    main()
