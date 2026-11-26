"""Ensemble control (review A3): is the combination's gain just averaging two maps? See README.md
(approved before running).

For each stage in config.yaml (stage D, stage 1), stage 1's runner (experiments/scarce-natural-s1/
run.py, unchanged) runs on that stage's pairs, folds, test words and dictionaries with this
experiment's maps: direct (thin), the chain through the hub and, on the same direct dictionary,
a Procrustes map on a bootstrap resample (boot) and, on stage D, unconstrained least squares
(lstsq). Ranked in one pass per direction and fold: each map alone and the equal-weight
combinations direct + chain (combo, as the stages), direct + boot (ensboot), direct + lstsq
(enslstsq). Each stage has its own cache (<results>/ensemble-control/<stage>/) and results/<stage>/;
summary.csv then sets each combination against direct and combo against each ensemble.
`--data-files` lists every input for the remote copy; `--results-only` rebuilds the tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from clwe_chains.align import compose, fit
from clwe_chains.dictionaries import dict_sizes
from clwe_chains.evaluate import bli_with_combinations
from clwe_chains.runs import file_fingerprints
from clwe_chains.stats import paired_bootstrap

HERE = Path(__file__).resolve().parent
NAME = HERE.name
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    f"{NAME}_s1_run", HERE.parent / "scarce-natural-s1" / "run.py")
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)
_FOLD_DICTS = s1.fold_dicts
COMBOS = {"combo": "chain", "ensboot": "boot", "enslstsq": "lstsq"}   # direct + this map
STAGE = None                # the stage being run (configure)


def bootstrap(d: pd.DataFrame, seed: int, name: str, fold: int) -> pd.DataFrame:
    """A resample of a dictionary's rows with replacement, as many rows as it has; fixed by the
    seed, the pair's name and the fold."""
    rng = np.random.default_rng([seed, zlib.crc32(name.encode("utf-8")), fold])
    return d.iloc[np.sort(rng.integers(0, len(d), len(d)))].reset_index(drop=True)


def partners() -> list[str]:
    return XCFG["stages"][STAGE]["partners"]


def methods_for(hub: str, parts: list[str]) -> list[tuple[str, ...]]:
    """Cache methods: thin, chain, combo as stage 1 names them; each partner alone and with
    direct."""
    out = [("thin",), ("chain", hub), ("combo", hub)]
    for p in parts:
        out += [(p,), (f"ens{p}",)]
    return out


class Ctx(s1.Ctx):
    """Stage 1's context on one stage's data, with this experiment's methods."""

    def methods(self, s: str, hub: str) -> list[tuple[str, ...]]:
        return methods_for(hub, partners())


def fold_dicts(ctx, s: str, t: str, x: str, fold: int) -> tuple[dict, dict]:
    """Stage 1's fold dictionaries plus the bootstrap resample of the fold's direct dictionary
    (after test removal; both directions use it, as they use the direct dictionary)."""
    d, sizes = _FOLD_DICTS(ctx, s, t, x, fold)
    d["boot"] = bootstrap(d["direct"], XCFG["seed"], f"{s}__{t}", fold)
    sizes["boot"] = dict_sizes(d["boot"])
    return d, sizes


def lstsq_map(d: pd.DataFrame, emb_a, emb_b, forward: bool):
    """Unconstrained least squares on an edge dictionary (a, b): a -> b if forward, else b -> a
    (stage 1's edge_map with the other solver)."""
    xa, xb = emb_a[s1._idx(d["a"], emb_a.device)], emb_b[s1._idx(d["b"], emb_b.device)]
    return fit(xa, xb, "lstsq") if forward else fit(xb, xa, "lstsq")


def evaluate_direction(ctx, s: str, t: str, x: str, fold: int, forward: bool, d: dict,
                       sizes: dict) -> int:
    """Every map and combination for one direction of one fold, in one pass; returns the number
    of maps evaluated."""
    src, tgt = (s, t) if forward else (t, s)
    todo = [m for m in ctx.methods(s, x) if s1.key(src, tgt, m, fold) not in ctx.cache]
    if not todo:
        return 0
    es, et, ex = ctx.store[s], ctx.store[t], ctx.store[x]
    e_src, e_tgt = (es, et) if forward else (et, es)
    concepts = ctx.folds[s, t][fold]
    queries = np.array([ctx.query[src, c] for c in concepts])
    golds = [[ctx.gold[src, tgt, kind, c] for c in concepts] for kind in ("main", "list")]
    ev = {"k": s1.EVAL["csls_k"], "block": s1.CFG["eval"]["block_rows"], "half": s1.EVAL["half"]}
    if forward:
        w_chain = compose([s1.edge_map(d["sx"], es, ex, True), s1.edge_map(d["xt"], ex, et, True)])
    else:
        w_chain = compose([s1.edge_map(d["xt"], ex, et, False),
                           s1.edge_map(d["sx"], es, ex, False)])
    maps = {"thin": s1.edge_map(d["direct"], es, et, forward), "chain": w_chain}
    for p in partners():
        maps[p] = (s1.edge_map(d["boot"], es, et, forward) if p == "boot"
                   else lstsq_map(d["direct"], es, et, forward))
    combos = {c: ("thin", m) for c, m in COMBOS.items() if m in maps}
    t0 = time.perf_counter()
    out = bli_with_combinations(e_src, e_tgt, maps, combos, queries, golds, **ev)
    dt = (time.perf_counter() - t0) / len(maps)
    for m in ctx.methods(s, x):
        name = m[0]                  # thin, chain, combo, boot, ensboot, lstsq, enslstsq
        ctx.cache.put(s1.key(src, tgt, m, fold),
                      {"concepts": concepts, "rank": out[name][0].tolist(),
                       "rank_list": out[name][1].tolist(), "n_train": sizes,
                       "seconds": dt if name in maps else 0.0})
    return len(maps)


def needed_files(pairs: pd.DataFrame) -> list[Path]:
    """Every input the run reads, absolute: the route languages' vectors, the stage's folds, gold
    and test words, and each pair's direct and hub dictionaries."""
    p6 = s1.p6_dir()
    files = [s1.CFG.path("v0_embeddings") / f"{lang}.npy" for lang in s1.route_langs(pairs)]
    files += [p6 / f for f in ("folds.parquet", "gold.parquet", "words.parquet")]
    for s, t, x in zip(pairs["s"], pairs["t"], pairs["hub"], strict=True):
        files += [p6 / "dicts" / f"{n}.parquet" for n in (f"{s}__{t}", f"{s}__{x}", f"{x}__{t}")]
    return sorted(set(files))


def configure(stage: str) -> None:
    """Point stage 1's runner at one stage's pairs and data, this experiment's cache, results and
    maps."""
    global STAGE
    STAGE = stage
    sc = XCFG["stages"][stage]
    s1.NAME, s1.DATA = f"{NAME}/{stage}", sc["data"]
    s1.HERE = HERE.parent / sc["experiment"]                  # its pairs.csv
    s1.RESULTS = HERE / "results" / stage
    (HERE / "results").mkdir(exist_ok=True)
    s1.RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/scarce-natural-s1/run.py",
                       "experiments/scarce-natural-s1/config.yaml",
                       f"experiments/{sc['experiment']}/pairs.csv", f"experiments/{NAME}/")
    s1.OWN_RESULTS = f"experiments/{NAME}/results/"
    s1.Ctx = Ctx
    s1.fold_dicts = fold_dicts
    s1.evaluate_direction = evaluate_direction
    s1.XCFG = {**s1.XCFG, "stage": XCFG, "which": stage}
    p6 = s1.p6_dir()
    keep = {str(f.relative_to(p6)) for f in needed_files(pd.read_csv(s1.HERE / "pairs.csv"))
            if p6 in f.parents}
    # run_info.json fingerprints only the files this run reads (stage 1's data folder is the whole
    # v1/phase6 root, every other stage's data included)
    s1.file_fingerprints = lambda files: file_fingerprints(
        {k: v for k, v in files.items() if k in keep})


def summary(stage: str) -> pd.DataFrame:
    """Per combination: mean ΔMRR against direct over directions (and over languages, as the
    stages' overall.csv), directions above / below 0; and combo − each ensemble, paired per test
    word, with directions above / below 0."""
    res = HERE / "results" / stage
    pooled = pd.read_csv(res / "pooled.csv")
    rows = []
    for name in ("combo", *[f"ens{p}" for p in XCFG["stages"][stage]["partners"]],
                 *XCFG["stages"][stage]["partners"], "chain"):
        g = pooled[pooled["method"] == name]
        lang = g.groupby("s")["delta"].mean()
        rows.append({"stage": stage, "what": f"{name} - thin", "n_dirs": len(g),
                     "mean_over_dirs": g["delta"].mean(), "mean_over_languages": lang.mean(),
                     "n_above": int((g["lo"] > 0).sum()), "n_below": int((g["hi"] < 0).sum())})
    return pd.DataFrame(rows)


def combo_gaps(stage: str, ctx) -> pd.DataFrame:
    """combo − ens<partner> per direction, paired per test word over all folds (bootstrap as the
    stages), from the cache."""
    rows = []
    for s, t, x in zip(ctx.all_pairs["s"], ctx.all_pairs["t"], ctx.all_pairs["hub"], strict=True):
        folds = sorted(ctx.folds[s, t])
        for src, tgt in ((s, t), (t, s)):
            combo = s1.pooled(ctx.cache, src, tgt, ("combo", x), folds)
            for p in XCFG["stages"][stage]["partners"]:
                ens = s1.pooled(ctx.cache, src, tgt, (f"ens{p}",), folds)
                if combo is None or ens is None:
                    continue
                mean, lo, hi = paired_bootstrap(combo["rr"].to_numpy(), ens["rr"].to_numpy(),
                                                **s1.XCFG["bootstrap"])
                rows.append({"stage": stage, "src": src, "tgt": tgt, "s": s,
                             "what": f"combo - ens{p}", "delta": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def gap_summary(gap: pd.DataFrame) -> pd.DataFrame:
    """Per stage and gap: the mean over directions with an interval from a bootstrap over pairs
    (of each pair's mean over its two directions), the mean over languages, directions above /
    below 0."""
    rows = []
    for (stage, what), g in gap.groupby(["stage", "what"], sort=False):
        pair = g.assign(pair=[" ".join(sorted(p)) for p in zip(g["src"], g["tgt"], strict=True)])
        v = pair.groupby("pair")["delta"].mean().to_numpy()
        _, lo, hi = paired_bootstrap(v, np.zeros_like(v), **s1.XCFG["bootstrap"])
        rows.append({"stage": stage, "what": what, "n_dirs": len(g),
                     "mean_over_dirs": g["delta"].mean(),
                     "mean_over_languages": g.groupby("s")["delta"].mean().mean(),
                     "lo_pairs": lo, "hi_pairs": hi,
                     "n_above": int((g["lo"] > 0).sum()), "n_below": int((g["hi"] < 0).sum())})
    return pd.DataFrame(rows)


def reproduction_check(stage: str) -> pd.DataFrame:
    """thin, chain and combo ranks of this run against the stage's own cache (README check 4),
    per method: records compared, share of queries with the same rank, largest rank difference.
    Empty when the stage's cache is not on this machine (the remote); run --check on the laptop."""
    sc = XCFG["stages"][stage]
    theirs_path = s1.CFG.path("results") / sc["experiment"] / "evals.jsonl"
    if not theirs_path.exists():
        return pd.DataFrame()
    mine, theirs = s1.Cache(s1.out_dir() / "evals.jsonl"), s1.Cache(theirs_path)
    pairs = pd.read_csv(s1.HERE / "pairs.csv")
    folds = pd.read_parquet(s1.p6_dir() / "folds.parquet")
    folds = {(s, t): sorted(g["fold"].unique().tolist()) for (s, t), g in folds.groupby(["s", "t"])}
    stats: dict[str, list] = {}
    for s, t, x in zip(pairs["s"], pairs["t"], pairs["hub"], strict=True):
        for src, tgt in ((s, t), (t, s)):
            for m in (("thin",), ("chain", x), ("combo", x)):
                for f in folds[s, t]:
                    rec, old = mine.get(s1.key(src, tgt, m, f)), theirs.get(s1.key(src, tgt, m, f))
                    if rec is None or old is None:
                        continue
                    a, b = np.asarray(rec["rank"]), np.asarray(old["rank"])
                    stats.setdefault(m[0], []).append(
                        (len(a), int((a == b).sum()), int(np.abs(a - b).max())))
    return pd.DataFrame([{"stage": stage, "method": m, "records": len(v),
                          "same_rank_share": sum(x[1] for x in v) / sum(x[0] for x in v),
                          "max_rank_diff": max(x[2] for x in v)} for m, v in stats.items()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs per stage")
    ap.add_argument("--data-files", action="store_true", help="list inputs and exit")
    ap.add_argument("--results-only", action="store_true", help="rebuild results from the caches")
    ap.add_argument("--check", action="store_true",
                    help="only compare thin / chain / combo with the stages' caches (laptop)")
    ap.add_argument("--stages", default=None, help="comma-separated subset (default: all)")
    args = ap.parse_args()
    stages = list(XCFG["stages"]) if args.stages is None else args.stages.split(",")
    kw = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    if args.data_files:
        files = set()
        for st in stages:
            configure(st)
            files |= set(needed_files(pd.read_csv(s1.HERE / "pairs.csv")))
        for f in sorted(files):
            print(f.relative_to(s1.CFG.data_root).as_posix())
        return
    if args.check:
        checks = []
        for st in stages:
            configure(st)
            checks.append(reproduction_check(st))
        out = pd.concat(checks, ignore_index=True)
        out.to_csv(HERE / "results" / "check.csv", **kw)
        print(out.to_string(index=False))
        return
    for st in stages:
        configure(st)
        s1.check_clean(s1.git("status", "--porcelain").splitlines())
        print(f"stage {st}", flush=True)
        ctx = Ctx(args.limit)
        if not args.results_only:
            s1.run_all(ctx)
        s1.write_results(ctx)
        if not (s1.RESULTS / "pooled.csv").exists():
            continue
        combo_gaps(st, ctx).to_csv(s1.RESULTS / "gaps.csv", **kw)
    # the tables cover every stage with results, not only the ones run now
    done = [st for st in XCFG["stages"] if (HERE / "results" / st / "gaps.csv").exists()]
    gap = pd.concat([pd.read_csv(HERE / "results" / st / "gaps.csv") for st in done],
                    ignore_index=True)
    gap.to_csv(HERE / "results" / "gaps.csv", **kw)
    pd.concat([*[summary(st) for st in done], gap_summary(gap)], ignore_index=True).to_csv(
        HERE / "results" / "summary.csv", **kw)
    print("summary written to results/summary.csv")


if __name__ == "__main__":
    main()
