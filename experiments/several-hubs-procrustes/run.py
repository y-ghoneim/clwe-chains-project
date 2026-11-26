"""Several hubs with plain Procrustes maps: where does strong maps' best result come from? See
README.md (approved before running).

Strong maps' runner (../strong-maps/run.py, imported unchanged) supplies the units, pairs, hubs
1-3, folds, test words, dictionaries (test words removed from every edge), gold and evaluation;
this runner fits plain Procrustes rotations instead of self-learned ones and scores, per direction
and fold in one pass: direct (thin), the chains through hubs 1-3, the combination (direct + chain)
and several hubs (direct + the 3 chains), CSLS scores averaged with equal weights. Own cache per
unit (<results>/several-hubs-procrustes/<unit>/). Results compare with strong maps' committed cache.
`--check` compares direct, chain and combination with the stages' own caches.
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

from clwe_chains.align import compose
from clwe_chains.evaluate import bli_with_combinations
from clwe_chains.runs import Cache, file_fingerprints
from clwe_chains.stats import paired_bootstrap

HERE = Path(__file__).resolve().parent
NAME = HERE.name
RESULTS = HERE / "results"
_spec = importlib.util.spec_from_file_location(
    f"{NAME}_sm_run", HERE.parent / "strong-maps" / "run.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)
s1 = sm.s1
UNITS = ["s1", "a", "a2", "d"]                       # strong maps' full run (570 directions)
MAPS = ["thin", "chain", "chain2", "chain3"]         # evaluated; the combinations are free
COMBOS = {"combo": ("thin", "chain"), "hubs3": ("thin", "chain", "chain2", "chain3")}
METHODS = [("proc", n) for n in MAPS + list(COMBOS)]
# (a, b): a minus b, per direction and pooled per unit; vm_* from strong maps' cache
COMPARE = [("proc_hubs3", "proc_thin"), ("proc_hubs3", "proc_chain"), ("proc_hubs3", "proc_combo"),
           ("proc_chain", "proc_thin"), ("proc_combo", "proc_thin"),
           ("vm_hubs3", "proc_hubs3"), ("vm_thin", "proc_thin"), ("vm_chain", "proc_chain"),
           ("vm_hubs3", "proc_thin")]
OURS_STRONG = [("vm", "thin"), ("vm", "chain"), ("vm", "hubs3")]
KW = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}


def configure(unit: str) -> None:
    """Strong maps' configuration for the unit's data, with this experiment's cache and log."""
    sm.configure(unit)
    s1.NAME = f"{NAME}/{unit}"
    s1.RESULT_PATHS = (*s1.RESULT_PATHS, f"experiments/{NAME}/")
    s1.OWN_RESULTS = f"experiments/{NAME}/results/"


def strong_cache(unit: str) -> Cache:
    return Cache(s1.CFG.path("results") / "strong-maps" / sm.VERSION / unit / "evals.jsonl")


def unit_log(unit: str) -> Path:
    return s1.CFG.path("results") / NAME / unit / "run.log"


# ---------------------------------------------------------------------------------------------
# Evaluation

def chain_map(d: dict, i: str, es, ex, et, forward: bool) -> torch.Tensor:
    """The chain through hub i ("" = hub 1): two Procrustes rotations, composed."""
    sx, xt = d[f"sx{i}"], d[f"xt{i}"]
    if forward:
        return compose([s1.edge_map(sx, es, ex, True), s1.edge_map(xt, ex, et, True)])
    return compose([s1.edge_map(xt, ex, et, False), s1.edge_map(sx, es, ex, False)])


def evaluate_fold(ctx, s: str, t: str, hubs: list[str], fold: int) -> int:
    """Every method of one pair and fold, both directions; returns the maps evaluated."""
    todo = [fw for fw in (True, False)
            if any(s1.key(*((s, t) if fw else (t, s)), m, fold) not in ctx.cache for m in METHODS)]
    if not todo:
        return 0
    d, _, _, sizes = sm.fold_dicts(ctx, s, t, hubs[0], hubs[1:], fold)
    es, et = ctx.store[s], ctx.store[t]
    ev = {"k": s1.EVAL["csls_k"], "block": s1.CFG["eval"]["block_rows"], "half": s1.EVAL["half"]}
    concepts = ctx.folds[s, t][fold]
    for fw in todo:
        src, tgt = (s, t) if fw else (t, s)
        queries = np.array([ctx.query[src, c] for c in concepts])
        golds = [[ctx.gold[src, tgt, kind, c] for c in concepts] for kind in ("main", "list")]
        t0 = time.perf_counter()
        maps = {"thin": s1.edge_map(d["direct"], es, et, fw)}
        for i, h in zip(("", "2", "3"), hubs, strict=True):
            maps[f"chain{i}"] = chain_map(d, i, es, ctx.store[h], et, fw)
        sm.finite(maps, src, tgt, fold)
        out = bli_with_combinations(es if fw else et, et if fw else es, maps, COMBOS, queries,
                                    golds, **ev)
        dt = (time.perf_counter() - t0) / len(maps)
        for _, n in METHODS:
            ctx.cache.put(s1.key(src, tgt, ("proc", n), fold),
                          {"concepts": concepts, "rank": out[n][0].tolist(),
                           "rank_list": out[n][1].tolist(), "n_train": sizes,
                           "seconds": dt if n in maps else 0.0})
    return len(MAPS) * len(todo)


def run_unit(ctx, pairs: pd.DataFrame) -> None:
    log = unit_log(ctx.unit)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"start commit {s1.git('rev-parse', 'HEAD').strip()}\n")
    start, total = time.time(), 0
    for i, r in enumerate(pairs.itertuples()):
        for fold in sorted(ctx.folds[r.s, r.t]):
            total += evaluate_fold(ctx, r.s, r.t, [r.hub, r.hub2, r.hub3], fold)
        el = time.time() - start
        msg = (f"{time.strftime('%H:%M:%S')} {ctx.unit} pair {i + 1}/{len(pairs)} {r.s}-{r.t}: "
               f"{total} maps, {el / 60:.1f} min, {el / max(total, 1):.2f} s each")
        print(msg, flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


# ---------------------------------------------------------------------------------------------
# Results

def direction_scores(ctx, pairs: pd.DataFrame) -> dict:
    """(src, tgt) -> method -> reciprocal ranks over every fold (sorted by concept); proc from
    this cache, vm from strong maps'."""
    strong, out = strong_cache(ctx.unit), {}
    for r in pairs.itertuples():
        folds = sorted(ctx.folds[r.s, r.t])
        for src, tgt in ((r.s, r.t), (r.t, r.s)):
            got = {}
            for cache, methods in ((ctx.cache, METHODS), (strong, OURS_STRONG)):
                for m in methods:
                    p = s1.pooled(cache, src, tgt, m, folds)
                    if p is not None:
                        got[f"{m[0]}_{m[1]}"] = p
            out[src, tgt] = got
    return out


def compare(unit: str, scores: dict, bs: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Per direction and method; per direction and comparison; per unit and comparison (pooled
    over the unit's test words, paired; directions above / below 0 by their own bootstrap)."""
    per, rows = [], []
    for (src, tgt), got in scores.items():
        for m, p in got.items():
            per.append({"unit": unit, "src": src, "tgt": tgt, "method": m, "mrr": p["rr"].mean(),
                        "p1": (p["rr"] == 1).mean(), "mrr_list": p["rr_list"].mean(),
                        "n": len(p)})
        for a, b in COMPARE:
            if a in got and b in got:
                if not (got[a]["concept"].to_numpy() == got[b]["concept"].to_numpy()).all():
                    raise ValueError(f"{src}->{tgt}: {a} and {b} differ in test words")
                mean, lo, hi = paired_bootstrap(got[a]["rr"].to_numpy(), got[b]["rr"].to_numpy(),
                                                **bs)
                rows.append({"unit": unit, "src": src, "tgt": tgt, "a": a, "b": b,
                             "mrr_a": got[a]["rr"].mean(), "mrr_b": got[b]["rr"].mean(),
                             "delta": mean, "lo": lo, "hi": hi})
    summary = []
    if rows:
        for (a, b), d in pd.DataFrame(rows).groupby(["a", "b"], sort=False):
            ra = np.concatenate([scores[s, t][a]["rr"].to_numpy()
                                 for s, t in zip(d["src"], d["tgt"], strict=True)])
            rb = np.concatenate([scores[s, t][b]["rr"].to_numpy()
                                 for s, t in zip(d["src"], d["tgt"], strict=True)])
            mean, lo, hi = paired_bootstrap(ra, rb, **bs)
            summary.append({"unit": unit, "a": a, "b": b, "n_dirs": len(d), "n_words": len(ra),
                            "mrr_a": ra.mean(), "mrr_b": rb.mean(), "delta": mean, "lo": lo,
                            "hi": hi, "mean_delta_dirs": d["delta"].mean(),
                            "n_above": int((d["lo"] > 0).sum()),
                            "n_below": int((d["hi"] < 0).sum())})
    return per, rows, summary


def write_results(ctxs: dict, chosen: dict) -> None:
    per, rows, summary = [], [], []
    for unit, ctx in ctxs.items():
        configure(unit)
        p, r, s = compare(unit, direction_scores(ctx, chosen[unit]), s1.XCFG["bootstrap"])
        per, rows, summary = per + p, rows + r, summary + s
    if not per:
        print("no complete directions yet")
        return
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(per).to_csv(RESULTS / "pooled.csv", **KW)
    pd.DataFrame(rows).to_csv(RESULTS / "comparisons.csv", **KW)
    out = pd.DataFrame(summary)
    out.to_csv(RESULTS / "summary.csv", **KW)
    files = {}
    for ctx in ctxs.values():
        files.update({str(f.relative_to(s1.CFG.data_root)): f for f in sm.needed_files(ctx)
                      if s1.CFG.path("v0_embeddings") not in f.parents})
    commits = [c for u in ctxs if unit_log(u).exists()
               for c in s1.eval_commits(unit_log(u).read_text(encoding="utf-8"))]
    info = {"commit": s1.git("rev-parse", "HEAD").strip(),
            "eval_commits": list(dict.fromkeys(commits)), "units": list(ctxs),
            "strong_maps_cache": sm.VERSION, "eval": s1.EVAL,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "n_records": {u: sum(1 for _ in c.cache) for u, c in ctxs.items()},
            "inputs": file_fingerprints(dict(sorted(files.items())))}
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(out[["unit", "a", "b", "n_dirs", "mrr_a", "mrr_b", "delta", "lo", "hi", "n_above",
               "n_below"]].to_string(index=False))
    print(f"results written to {RESULTS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs per unit")
    ap.add_argument("--results-only", action="store_true", help="rebuild results from caches")
    ap.add_argument("--data-files", action="store_true", help="list inputs and exit")
    ap.add_argument("--check", action="store_true", help="against the stages' own caches")
    args = ap.parse_args()
    todo = UNITS if args.units is None else args.units.split(",")
    ctxs, chosen, paths = {}, {}, set()
    for unit in todo:
        configure(unit)
        ctx = sm.Ctx(unit)
        pairs = ctx.all_pairs if args.limit is None else ctx.all_pairs.head(args.limit)
        ctxs[unit], chosen[unit] = ctx, pairs
        paths |= set(s1.RESULT_PATHS)
    if args.data_files:
        for f in sorted({f for ctx in ctxs.values() for f in sm.needed_files(ctx)}):
            print(f.relative_to(s1.CFG.data_root).as_posix())
        return
    if args.check:
        out = pd.concat([sm.reproduction_check(ctxs[u], chosen[u]) for u in ctxs],
                        ignore_index=True)
        RESULTS.mkdir(exist_ok=True)
        out.to_csv(RESULTS / "check.csv", **KW)
        print(out.to_string(index=False))
        return
    s1.RESULT_PATHS = tuple(sorted(paths))
    s1.check_clean(s1.git("status", "--porcelain").splitlines())
    if not args.results_only:
        for unit, ctx in ctxs.items():
            configure(unit)
            print(f"unit {unit}: {len(chosen[unit])} pairs", flush=True)
            run_unit(ctx, chosen[unit])
    write_results(ctxs, chosen)


if __name__ == "__main__":
    main()
