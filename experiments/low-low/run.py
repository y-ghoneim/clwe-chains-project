"""Both ends low-resource: word translation between low-resource languages of different families.
See README.md (approved before running).

The new phase 6 stage `ll` (configs/default.yaml; prep/07, 08 and 19 with --stage ll) run through
strong maps' runner (../strong-maps/run.py, as a scenario added here: no file of it changes) with
plain Procrustes maps, as ../several-hubs-procrustes/ (its chain map and combinations imported
unchanged). Per direction and fold, in one pass: direct (thin), the chains through hubs 1-3, pooled
direct (joined, mutual best), the combination (direct + chain) and several hubs (direct + the 3
chains), CSLS scores averaged. Results per group (all; direct under 100 pairs; 100 and above), pairs
with under 50 test words listed and left out of the summaries.
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

from clwe_chains.evaluate import bli_with_combinations
from clwe_chains.runs import file_fingerprints
from clwe_chains.stats import paired_bootstrap

HERE = Path(__file__).resolve().parent
NAME = HERE.name
RESULTS = HERE / "results"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    f"{NAME}_shp_run", HERE.parent / "several-hubs-procrustes" / "run.py")
shp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shp)
sm, s1 = shp.sm, shp.s1
UNIT = "ll"
# the stage as a strong-maps scenario (in memory only; strong maps' config is not touched)
sm.XCFG["scenarios"][UNIT] = {"experiment": NAME, "data": "ll", "hubs": "ll", "joined": True,
                              "rcsls": False}
MAPS = ["thin", "chain", "chain2", "chain3", "joined"]
METHODS = [("proc", n) for n in MAPS + list(shp.COMBOS)]
COMPARE = [("chain", "thin"), ("combo", "thin"), ("hubs3", "thin"), ("joined", "thin"),
           ("hubs3", "chain")]
SMALL, MIN_TEST = XCFG["small"], XCFG["min_test"]      # README's rules, fixed before results
KW = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
# the stage's inputs whose change would make cached ranks stale (the cache key is the evaluation
# settings only): fingerprinted at the first run, checked on every later one
STAGE_FILES = ("pairs.csv", "dict_sizes.csv", "folds.parquet", "gold.parquet", "words.parquet")


def configure() -> None:
    """Strong maps' configuration for the stage, with this experiment's cache, log and code."""
    sm.configure(UNIT)
    s1.NAME = f"{NAME}/{UNIT}"
    s1.RESULT_PATHS = (*s1.RESULT_PATHS, "experiments/several-hubs-procrustes/",
                       f"experiments/{NAME}/")
    s1.OWN_RESULTS = f"experiments/{NAME}/results/"


def unit_log() -> Path:
    return s1.CFG.path("results") / NAME / UNIT / "run.log"


def stage_inputs(ctx) -> dict:
    """Fingerprints of the stage's files and its hubs file (the dictionaries are fixed by
    dict_sizes.csv and the hubs' own dict_sizes.csv)."""
    files = {f"ll/{f}": ctx.p6 / f for f in STAGE_FILES}
    files |= {f"hubs/ll/{f}": ctx.hub_dir.parent / f for f in ("hubs.csv", "dict_sizes.csv")}
    return file_fingerprints(files)


def guard_inputs(ctx) -> None:
    """Record the stage's inputs beside the cache at the first run; refuse to go on if they
    changed since (the cached ranks would come from other data)."""
    path = s1.out_dir() / "inputs.json"
    now = stage_inputs(ctx)
    if path.exists():
        was = json.loads(path.read_text(encoding="utf-8"))
        if was != now:
            changed = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
            raise SystemExit(f"stage inputs changed since the cache was made: {changed}; "
                             f"delete {s1.out_dir()} to start afresh")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(now, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Groups

def direct_sizes(p6: Path) -> dict[tuple[str, str], int]:
    """Each pair's direct dictionary (distinct word pairs, before test removal)."""
    sizes = pd.read_csv(p6 / "dict_sizes.csv").set_index("edge")["pairs"]
    return {tuple(e.split("__")): int(n) for e, n in sizes.items() if e.count("__") == 1}


def pair_groups(pairs: pd.DataFrame, sizes: dict, n_test: dict) -> pd.DataFrame:
    """Per pair: its direct size, test words, size group, and whether it is in the summaries."""
    out = pairs[["s", "t", "hub"]].copy()
    out["direct"] = [sizes[s, t] for s, t in zip(out["s"], out["t"], strict=True)]
    out["test_words"] = [n_test[s, t] for s, t in zip(out["s"], out["t"], strict=True)]
    out["group"] = np.where(out["direct"] < SMALL, f"under {SMALL}", f"{SMALL} and above")
    out["included"] = out["test_words"] >= MIN_TEST
    return out


# ---------------------------------------------------------------------------------------------
# Evaluation

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
        maps = {"thin": s1.edge_map(d["direct"], es, et, fw),
                "joined": s1.edge_map(d["joined"], es, et, fw)}
        for i, h in zip(("", "2", "3"), hubs, strict=True):
            maps[f"chain{i}"] = shp.chain_map(d, i, es, ctx.store[h], et, fw)
        sm.finite(maps, src, tgt, fold)
        out = bli_with_combinations(es if fw else et, et if fw else es, maps, shp.COMBOS, queries,
                                    golds, **ev)
        dt = (time.perf_counter() - t0) / len(maps)
        for _, n in METHODS:
            ctx.cache.put(s1.key(src, tgt, ("proc", n), fold),
                          {"concepts": concepts, "rank": out[n][0].tolist(),
                           "rank_list": out[n][1].tolist(), "n_train": sizes,
                           "seconds": dt if n in maps else 0.0})
    return len(MAPS) * len(todo)


def run_pairs(ctx, pairs: pd.DataFrame) -> None:
    log = unit_log()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"start commit {s1.git('rev-parse', 'HEAD').strip()}\n")
    start, total = time.time(), 0
    for i, r in enumerate(pairs.itertuples()):
        for fold in sorted(ctx.folds.get((r.s, r.t), {})):      # no folds: no eligible concept
            total += evaluate_fold(ctx, r.s, r.t, [r.hub, r.hub2, r.hub3], fold)
        el = time.time() - start
        msg = (f"{time.strftime('%H:%M:%S')} pair {i + 1}/{len(pairs)} {r.s}-{r.t} via {r.hub}: "
               f"{total} maps, {el / 60:.1f} min, {el / max(total, 1):.2f} s each")
        print(msg, flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


# ---------------------------------------------------------------------------------------------
# Results

def scores(ctx, pairs: pd.DataFrame) -> dict:
    """(src, tgt) -> method -> reciprocal ranks over every fold (sorted by concept)."""
    out = {}
    for r in pairs.itertuples():
        folds = sorted(ctx.folds.get((r.s, r.t), {}))
        if not folds:
            continue
        for src, tgt in ((r.s, r.t), (r.t, r.s)):
            out[src, tgt] = {n: p for _, n in METHODS
                             if (p := s1.pooled(ctx.cache, src, tgt, ("proc", n), folds))
                             is not None}
    return out


def tables(sc: dict, groups: pd.DataFrame, bs: dict
           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per direction and method; per direction and comparison; per group and comparison (pooled
    over the group's test words, paired; directions above / below 0 by their own bootstrap). The
    group summaries use only included pairs (at least MIN_TEST test words)."""
    g = groups.set_index(["s", "t"])
    per, rows = [], []
    for (src, tgt), got in sc.items():
        key = (src, tgt) if (src, tgt) in g.index else (tgt, src)
        info = {"src": src, "tgt": tgt, "group": g.at[key, "group"],
                "direct_pairs": int(g.at[key, "direct"]), "included": bool(g.at[key, "included"])}
        for m, p in got.items():
            per.append({**info, "method": m, "mrr": p["rr"].mean(), "p1": (p["rr"] == 1).mean(),
                        "mrr_list": p["rr_list"].mean(), "n": len(p)})
        for a, b in COMPARE:
            if a in got and b in got:
                if not (got[a]["concept"].to_numpy() == got[b]["concept"].to_numpy()).all():
                    raise ValueError(f"{src}->{tgt}: {a} and {b} differ in test words")
                mean, lo, hi = paired_bootstrap(got[a]["rr"].to_numpy(), got[b]["rr"].to_numpy(),
                                                **bs)
                rows.append({**info, "a": a, "b": b, "mrr_a": got[a]["rr"].mean(),
                             "mrr_b": got[b]["rr"].mean(), "delta": mean, "lo": lo, "hi": hi})
    cmp_dir = pd.DataFrame(rows)
    summary = []
    if not cmp_dir.empty:
        inc = cmp_dir[cmp_dir["included"]]
        for name, part in [("all", inc), *inc.groupby("group")]:
            for (a, b), d in part.groupby(["a", "b"], sort=False):
                ra = np.concatenate([sc[s, t][a]["rr"].to_numpy()
                                     for s, t in zip(d["src"], d["tgt"], strict=True)])
                rb = np.concatenate([sc[s, t][b]["rr"].to_numpy()
                                     for s, t in zip(d["src"], d["tgt"], strict=True)])
                mean, lo, hi = paired_bootstrap(ra, rb, **bs)
                summary.append({"group": name, "a": a, "b": b, "n_dirs": len(d),
                                "n_words": len(ra), "mrr_a": ra.mean(), "mrr_b": rb.mean(),
                                "delta": mean, "lo": lo, "hi": hi,
                                "mean_delta_dirs": d["delta"].mean(),
                                "n_above": int((d["lo"] > 0).sum()),
                                "n_below": int((d["hi"] < 0).sum())})
    return pd.DataFrame(per), cmp_dir, pd.DataFrame(summary)


def write_results(ctx, pairs: pd.DataFrame, groups: pd.DataFrame, smoke: bool) -> None:
    """results/ (results/smoke/ for a --limit run, so a partial run never replaces the real
    tables)."""
    per, cmp_dir, summary = tables(scores(ctx, pairs), groups, s1.XCFG["bootstrap"])
    if per.empty:
        print("no complete directions yet")
        return
    out = RESULTS / "smoke" if smoke else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    groups.to_csv(out / "pairs_groups.csv", **KW)
    per.to_csv(out / "pooled.csv", **KW)
    cmp_dir.to_csv(out / "comparisons.csv", **KW)
    summary.to_csv(out / "summary.csv", **KW)
    files = {str(f.relative_to(s1.CFG.data_root)): f for f in sm.needed_files(ctx)
             if s1.CFG.path("v0_embeddings") not in f.parents}
    log = unit_log()
    info = {"commit": s1.git("rev-parse", "HEAD").strip(), "smoke": smoke,
            "pairs": len(pairs),
            "eval_commits": s1.eval_commits(log.read_text(encoding="utf-8"))
            if log.exists() else [],
            "config": XCFG, "eval": s1.EVAL, "bootstrap": s1.XCFG["bootstrap"],
            "scenario": sm.XCFG["scenarios"][UNIT],
            "left_out": groups[~groups["included"]][["s", "t", "test_words"]].to_dict("records"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "n_records": sum(1 for _ in ctx.cache),
            "inputs": file_fingerprints(dict(sorted(files.items())))}
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(summary[["group", "a", "b", "n_dirs", "mrr_a", "mrr_b", "delta", "lo", "hi", "n_above",
                   "n_below"]].to_string(index=False))
    print(f"results written to {out}")


def positive(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=positive, default=None,
                    help="only the first N pairs; results go to results/smoke/")
    ap.add_argument("--results-only", action="store_true", help="rebuild results from the cache")
    args = ap.parse_args()
    configure()
    ctx = sm.Ctx(UNIT)
    pairs = ctx.all_pairs if args.limit is None else ctx.all_pairs.head(args.limit)
    n_test = {(s, t): sum(len(c) for c in ctx.folds.get((s, t), {}).values())
              for s, t in zip(ctx.all_pairs["s"], ctx.all_pairs["t"], strict=True)}
    groups = pair_groups(ctx.all_pairs, direct_sizes(ctx.p6), n_test)
    s1.check_clean(s1.git("status", "--porcelain").splitlines())
    guard_inputs(ctx)
    if not args.results_only:
        print(f"{len(pairs)} pairs", flush=True)
        run_pairs(ctx, pairs)
    write_results(ctx, pairs, groups, smoke=args.limit is not None)


if __name__ == "__main__":
    main()
