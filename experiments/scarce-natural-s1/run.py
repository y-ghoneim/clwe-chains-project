"""Phase 6 stage 1: chains on naturally scarce pairs. See README.md (approved before running).

77 pairs of a low-resource language with a node language (pairs.csv); each edge trains on its own
natural dictionary (prep/08), with each fold's test words removed from every edge. Per pair,
direction and fold: thin direct, the chain through the pair's hub, the chain through en (mg and
or pairs), joined direct (mutual best and unfiltered), chain-as-teacher for every alpha, and the
direct + chain combination. Every evaluation is exact CSLS over the target's full vocabulary,
cached on disk, so the run resumes after an interruption. results/ is rebuilt from the cache at
the end (or with --results-only).

Inputs: the vectors of every language in the routes and the prep outputs under v1/phase6/
(`python run.py --data-files` lists them, relative to the data root, for the remote copy).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from clwe_chains import load_config
from clwe_chains.align import compose, fit, fit_teacher
from clwe_chains.dictionaries import dict_sizes, joined_dictionary, remove_test
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.evaluate import bli_multi, bli_with_combination
from clwe_chains.runs import (
    Cache,
    cache_key,
    file_fingerprints,
    forget,
    result_affecting,
    settings_hash,
)
from clwe_chains.stats import paired_bootstrap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/scarce-natural-s1/")
OWN_RESULTS = "experiments/scarce-natural-s1/results/"
# Overridden by later stages that reuse this runner (experiments/scarce-moderate-a/,
# experiments/natural-highres-d/): the experiment's name (cache and log folder) and its data
# subfolder under v1/phase6 ("" = stage 1's own data).
NAME = "scarce-natural-s1"
DATA = ""


def p6_dir() -> Path:
    return CFG.path("phase6") / DATA if DATA else CFG.path("phase6")


def out_dir() -> Path:
    return CFG.path("results") / NAME
CFG = load_config()
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
EVAL = {"csls_k": CFG["eval"]["csls_k"], "half": CFG["eval"]["half"], "retrieval": "csls"}
SETTINGS = settings_hash(EVAL)
ALPHAS = sorted(XCFG["alphas"])
# chain-as-teacher families: lambda = alpha x n_direct / scale(d). "teacher" is the pre-registered
# lambda = alpha x n_direct, which outweighs the data term by about d (final review I2);
# "teacherd" divides by the dimension d (user, 2026-11-16).
TEACHERS = ("teacher", "teacherd")


def teacher_lambda(family: str, alpha: float, n: int, d: int) -> float:
    """lambda for chain-as-teacher: alpha x n_direct, divided by d for the "teacherd" family."""
    return alpha * n / (d if family == "teacherd" else 1)


# ---------------------------------------------------------------------------------------------
# Small helpers (tests/test_scarce_s1_experiment.py)

def key(src: str, tgt: str, method: tuple[str, ...], fold: int) -> str:
    return cache_key("csls", SETTINGS, (src, tgt), method, fold, "full")


def label(method: tuple[str, ...], hub: str) -> str:
    """Column name of a method: thin, chain, chain_en, joined_mb, joined_all, teacher_<a>, combo."""
    kind = method[0]
    if kind == "chain":
        return "chain" if method[1] == hub else f"chain_{method[1]}"
    if kind == "joined":
        return f"joined_{method[1]}"
    if kind in TEACHERS:
        return f"{kind}_{method[1]}"
    return kind


def teacher_label(alpha, family: str = "teacher") -> str:
    """teacher_<alpha> as stored, for an alpha that may have come back as a float (pandas turns
    the YAML's mixed 0.1 / 1 keys into floats: 1 -> 1.0)."""
    return f"{family}_{ {float(a): a for a in ALPHAS}[float(alpha)] }"


def eval_commits(log_text: str) -> list[str]:
    """The commits the run's evaluations were made at, from run.log's start lines, in order."""
    found = [line.split()[2] for line in log_text.splitlines() if line.startswith("start commit")]
    return list(dict.fromkeys(found))


def check_clean(status_lines: list[str]) -> None:
    """Refuse to evaluate or rebuild results with result-affecting uncommitted changes."""
    dirty = result_affecting(status_lines, RESULT_PATHS, OWN_RESULTS)
    if dirty:
        sys.exit("uncommitted changes that could affect results:\n" + "\n".join(dirty))


def method_of(name: str, hub: str) -> tuple[str, ...]:
    if name in ("thin",):
        return ("thin",)
    if name in ("chain", "combo"):
        return (name, hub)
    kind, arg = name.split("_", 1)
    return (kind, arg)


def pooled(cache: Cache, src: str, tgt: str, method: tuple[str, ...],
           folds: list[int]) -> pd.DataFrame | None:
    """concept, rr, rr_list over all folds (each concept is tested in one fold), sorted by
    concept; None if any fold is missing."""
    parts = []
    for f in folds:
        rec = cache.get(key(src, tgt, method, f))
        if rec is None:
            return None
        parts.append(pd.DataFrame({"concept": rec["concepts"],
                                   "rr": 1.0 / np.asarray(rec["rank"], dtype=float),
                                   "rr_list": 1.0 / np.asarray(rec["rank_list"], dtype=float)}))
    return pd.concat(parts).sort_values("concept", ignore_index=True)


def alpha_cv(score: pd.DataFrame, language: pd.Series) -> tuple[pd.Series, dict]:
    """Chain-as-teacher's alpha, cross-validated leaving one low-resource language out: for each
    language, the alpha with the best mean over the other languages' pair-directions; ties go to
    the smaller alpha. score: pair-directions x alphas (pooled MRR)."""
    score = score[sorted(score.columns)]
    langs = np.asarray(language.loc[score.index])
    chosen, per = {}, pd.Series(np.nan, index=score.index)
    for lang in sorted(set(langs.tolist())):
        best = score[langs != lang].mean(axis=0).idxmax()     # first maximum: smaller alpha
        chosen[lang] = best
        per[langs == lang] = score.loc[langs == lang, best]
    return per, chosen


def language_table(diffs: dict[str, list[np.ndarray]], pooled: pd.DataFrame) -> pd.DataFrame:
    """Per language, descriptive only (no pass/fail; user, 2026-11-16): the mean over its pairs
    (each pair weighted equally; a pair is a directions x concepts array of reciprocal-rank
    differences) and how many of its pair-directions have their paired bootstrap interval
    (pooled: columns s, lo, hi for one method) above or below 0."""
    rows = []
    for lang, arrays in diffs.items():
        p = pooled[pooled["s"] == lang]
        rows.append({"lang": lang, "mean": float(np.mean([a.mean() for a in arrays])),
                     "n_pairs": len(arrays), "n_dirs": len(p), "n_above": int((p["lo"] > 0).sum()),
                     "n_below": int((p["hi"] < 0).sum())})
    return pd.DataFrame(rows, columns=["lang", "mean", "n_pairs", "n_dirs", "n_above", "n_below"])


# ---------------------------------------------------------------------------------------------
# Inputs

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def route_langs(pairs: pd.DataFrame) -> list[str]:
    langs = set(pairs["s"]) | set(pairs["t"]) | set(pairs["hub"])
    if set(pairs["s"]) & set(CFG["phase6"]["en_chain"]):
        langs.add("en")
    return sorted(langs)


def data_files(pairs: pd.DataFrame) -> list[Path]:
    """Every input the run reads, relative to the data root."""
    root = CFG.data_root
    files = [CFG.path("v0_embeddings") / f"{lang}.npy" for lang in route_langs(pairs)]
    p6 = p6_dir()
    files += sorted(f for f in p6.rglob("*") if f.is_file() and f.parent.name != "list")
    return [f.relative_to(root) for f in files]


class Ctx:
    """Pairs, folds, test words, gold, dictionaries (loaded lazily), vectors, cache."""

    def __init__(self, limit: int | None):
        p6 = p6_dir()
        self.all_pairs = pd.read_csv(HERE / "pairs.csv")
        self.pairs = self.all_pairs if limit is None else self.all_pairs.head(limit)
        f = pd.read_parquet(p6 / "folds.parquet")
        self.folds = {(s, t): {int(k): g["concept_id"].tolist() for k, g in d.groupby("fold")}
                      for (s, t), d in f.groupby(["s", "t"])}
        w = pd.read_parquet(p6 / "words.parquet")
        self.rows = {(lang, int(c)): g["row"].tolist() for (lang, c), g in
                     w.groupby(["lang", "concept_id"])}
        q = w[w["is_word"]]
        self.query = {(lang, int(c)): int(r) for lang, c, r in
                      zip(q["lang"], q["concept_id"], q["row"], strict=True)}
        g = pd.read_parquet(p6 / "gold.parquet")
        self.gold = {k: v["row"].tolist() for k, v in
                     g.groupby(["src", "tgt", "kind", "concept_id"])}
        self.p6 = p6
        self._dicts: dict[str, pd.DataFrame] = {}
        self.store = EmbeddingStore(CFG)
        self.cache = Cache(out_dir() / "evals.jsonl")
        self.en_chain = set(CFG["phase6"]["en_chain"])

    def dictionary(self, name: str) -> pd.DataFrame:
        if name not in self._dicts:
            sub = "joined" if name.count("__") == 2 else "dicts"
            self._dicts[name] = pd.read_parquet(self.p6 / sub / f"{name}.parquet")
        return self._dicts[name]

    def test_rows(self, lang: str, concepts: list[int]) -> set[int]:
        return {r for c in concepts for r in self.rows[lang, c]}

    def methods(self, s: str, hub: str) -> list[tuple[str, ...]]:
        out = [("thin",), ("chain", hub), ("combo", hub), ("joined", "mb"), ("joined", "all")]
        if s in self.en_chain and hub != "en":
            out.append(("chain", "en"))
        return out + [(fam, str(a)) for fam in TEACHERS for a in ALPHAS]


# ---------------------------------------------------------------------------------------------
# Evaluation

def _idx(rows, dev) -> torch.Tensor:
    return torch.as_tensor(np.array(rows, dtype=np.int64), device=dev)


def edge_map(d: pd.DataFrame, emb_a: torch.Tensor, emb_b: torch.Tensor,
             forward: bool) -> torch.Tensor:
    """Procrustes on an edge dictionary (a, b): a -> b if forward, else b -> a."""
    xa, xb = emb_a[_idx(d["a"], emb_a.device)], emb_b[_idx(d["b"], emb_b.device)]
    return fit(xa, xb, "procrustes") if forward else fit(xb, xa, "procrustes")


def fold_dicts(ctx: Ctx, s: str, t: str, x: str, fold: int) -> tuple[dict, dict]:
    """The fold's test-removed dictionaries (s-t, s-x, x-t, joined; s-en, en-t if needed)."""
    test = ctx.folds[s, t][fold]
    ts, tt = ctx.test_rows(s, test), ctx.test_rows(t, test)
    direct = remove_test(ctx.dictionary(f"{s}__{t}"), ts, tt)
    d = {"direct": direct,
         "sx": remove_test(ctx.dictionary(f"{s}__{x}"), ts, set()),
         "xt": remove_test(ctx.dictionary(f"{x}__{t}"), set(), tt)}
    if any(m[0] == "joined" for m in ctx.methods(s, x)):     # later stages may drop joined
        counts = ctx.dictionary(f"{s}__{x}__{t}")
        d["joined_mb"] = joined_dictionary(direct, counts, ts, tt, filtered=True)
        d["joined_all"] = joined_dictionary(direct, counts, ts, tt, filtered=False)
    if ("chain", "en") in ctx.methods(s, x):            # only stages that score the en chain
        d["sen"] = remove_test(ctx.dictionary(f"{s}__en"), ts, set())
        d["ent"] = remove_test(ctx.dictionary(f"en__{t}"), set(), tt)
    return d, {name: dict_sizes(v) for name, v in d.items()}


def evaluate_direction(ctx: Ctx, s: str, t: str, x: str, fold: int, forward: bool,
                       d: dict, sizes: dict) -> int:
    """Every method's CSLS ranks for one direction of one fold; returns evaluations run."""
    src, tgt = (s, t) if forward else (t, s)
    todo = [m for m in ctx.methods(s, x) if key(src, tgt, m, fold) not in ctx.cache]
    if not todo:
        return 0
    es, et, ex = ctx.store[s], ctx.store[t], ctx.store[x]
    e_src, e_tgt = (es, et) if forward else (et, es)
    concepts = ctx.folds[s, t][fold]
    queries = np.array([ctx.query[src, c] for c in concepts])
    golds = [[ctx.gold[src, tgt, kind, c] for c in concepts] for kind in ("main", "list")]
    ev = {"k": EVAL["csls_k"], "block": CFG["eval"]["block_rows"], "half": EVAL["half"]}

    def chain(first: pd.DataFrame, second: pd.DataFrame, emb_mid: torch.Tensor) -> torch.Tensor:
        if forward:
            return compose([edge_map(first, es, emb_mid, True),
                            edge_map(second, emb_mid, et, True)])
        return compose([edge_map(second, emb_mid, et, False), edge_map(first, es, emb_mid, False)])

    w_direct = edge_map(d["direct"], es, et, forward)
    w_chain = chain(d["sx"], d["xt"], ex)
    done = 0

    def put(method, ranks, seconds):
        ctx.cache.put(key(src, tgt, method, fold),
                      {"concepts": concepts, "rank": ranks[0].tolist(),
                       "rank_list": ranks[1].tolist(), "n_train": sizes, "seconds": seconds})

    if {("thin",), ("chain", x), ("combo", x)} & set(todo):
        t0 = time.perf_counter()
        out = bli_with_combination(e_src, e_tgt, w_direct, w_chain, queries, golds, **ev)
        dt = (time.perf_counter() - t0) / 2
        put(("thin",), out["a"], dt)
        put(("chain", x), out["b"], dt)
        put(("combo", x), out["combo"], 0.0)
        done += 2
    xy = (es[_idx(d["direct"]["a"], es.device)], et[_idx(d["direct"]["b"], et.device)])
    x_dir, y_dir = xy if forward else xy[::-1]
    for m in todo:
        if m[0] in ("thin", "combo") or m == ("chain", x):
            continue
        if m == ("chain", "en"):
            w = chain(d["sen"], d["ent"], ctx.store["en"])
        elif m[0] == "joined":
            w = edge_map(d[f"joined_{m[1]}"], es, et, forward)
        else:
            lam = teacher_lambda(m[0], float(m[1]), n=len(d["direct"]), d=es.shape[1])
            w = fit_teacher(x_dir, y_dir, w_chain, lam)
        t0 = time.perf_counter()
        ranks = bli_multi(e_src, e_tgt, w, queries, golds, retrieval="csls", **ev)
        put(m, ranks, time.perf_counter() - t0)
        done += 1
    return done


def run_all(ctx: Ctx) -> None:
    log = out_dir() / "run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"start commit {git('rev-parse', 'HEAD').strip()}\n")
    start, total = time.time(), 0
    for i, (s, t, x) in enumerate(zip(ctx.pairs["s"], ctx.pairs["t"], ctx.pairs["hub"],
                                      strict=True)):
        for fold in sorted(ctx.folds[s, t]):
            d, sizes = fold_dicts(ctx, s, t, x, fold)
            for forward in (True, False):
                total += evaluate_direction(ctx, s, t, x, fold, forward, d, sizes)
        el = time.time() - start
        msg = (f"{time.strftime('%H:%M:%S')} pair {i + 1}/{len(ctx.pairs)} {s}-{t} via {x}: "
               f"{total} evaluations, {el / 60:.1f} min, {el / max(total, 1):.2f} s each")
        print(msg, flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


# ---------------------------------------------------------------------------------------------
# Results

def write_results(ctx: Ctx) -> None:
    RESULTS.mkdir(exist_ok=True)
    bs = XCFG["bootstrap"]
    metrics, pooled_rows, sizes_rows, score, lang_of, rr = [], [], [], {}, {}, {}
    for s, t, x in zip(ctx.all_pairs["s"], ctx.all_pairs["t"], ctx.all_pairs["hub"], strict=True):
        folds = sorted(ctx.folds[s, t])
        for src, tgt in ((s, t), (t, s)):
            pd_key = f"{src}>{tgt}"
            for m in ctx.methods(s, x):
                p = pooled(ctx.cache, src, tgt, m, folds)
                if p is None:
                    continue
                rr[pd_key, label(m, x)] = p
                for f in folds:
                    rec = ctx.cache.get(key(src, tgt, m, f))
                    r, rl = np.asarray(rec["rank"]), np.asarray(rec["rank_list"])
                    metrics.append({"src": src, "tgt": tgt, "s": s, "hub": x,
                                    "method": label(m, x), "fold": f, "mrr": (1 / r).mean(),
                                    "p1": (r == 1).mean(), "mrr_list": (1 / rl).mean(),
                                    "n_queries": len(r),
                                    "n_train": rec["n_train"].get("direct", {}).get("pairs")})
                    if m == ("thin",):
                        sizes_rows += [{"src": src, "tgt": tgt, "fold": f, "dictionary": name,
                                        **v} for name, v in rec["n_train"].items()]
            lang_of[pd_key] = s
            for fam in TEACHERS:
                if all((pd_key, f"{fam}_{a}") in rr for a in ALPHAS):
                    score.setdefault(fam, {})[pd_key] = {a: rr[pd_key, f"{fam}_{a}"]["rr"].mean()
                                                         for a in ALPHAS}
    if not rr:
        print("no complete pair-directions yet")
        return
    pd.DataFrame(metrics).to_csv(RESULTS / "metrics.csv", index=False, lineterminator="\n",
                                 float_format="%.6g")
    pd.DataFrame(sizes_rows).to_csv(RESULTS / "dict_sizes.csv", index=False, lineterminator="\n")

    # alpha by leave-one-language-out cross-validation -> <family>_cv
    alpha_rows = []
    for fam, sc in score.items():
        score_df = pd.DataFrame(sc).T
        if len({lang_of[k] for k in score_df.index}) < 2:
            continue
        _, chosen = alpha_cv(score_df, pd.Series(lang_of))
        for k in score_df.index:
            rr[k, f"{fam}_cv"] = rr[k, teacher_label(chosen[lang_of[k]], fam)]
        alpha_rows += [{"family": fam, "lang": lang,
                        "alpha": teacher_label(a, fam).removeprefix(f"{fam}_")}
                       for lang, a in chosen.items()]
    pd.DataFrame(alpha_rows, columns=["family", "lang", "alpha"]).to_csv(
        RESULTS / "alpha_cv.csv", index=False, lineterminator="\n")

    names = sorted({name for _, name in rr if name != "thin"})
    diffs: dict[str, dict[str, dict[tuple, list]]] = {n: {} for n in names}
    for (pd_key, name), p in rr.items():
        if name == "thin":
            continue
        thin = rr[pd_key, "thin"]
        d = p["rr"].to_numpy() - thin["rr"].to_numpy()
        mean, lo, hi = paired_bootstrap(p["rr"].to_numpy(), thin["rr"].to_numpy(), **bs)
        src, tgt = pd_key.split(">")
        pooled_rows.append({"src": src, "tgt": tgt, "s": lang_of[pd_key], "method": name,
                            "mrr": p["rr"].mean(), "mrr_thin": thin["rr"].mean(),
                            "delta": mean, "lo": lo, "hi": hi,
                            "mrr_list": p["rr_list"].mean(), "n": len(p)})
        pair = tuple(sorted((src, tgt)))
        diffs[name].setdefault(lang_of[pd_key], {}).setdefault(pair, []).append(d)
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(RESULTS / "pooled.csv", index=False, lineterminator="\n", float_format="%.6g")

    lang_rows, overall_rows = [], []
    for name in names:
        per_lang = {lang: [np.vstack(ds) for ds in pairs.values()]
                    for lang, pairs in sorted(diffs[name].items())}
        t = language_table(per_lang, pooled_df[pooled_df["method"] == name])
        t.insert(0, "method", name)
        lang_rows.append(t)
        overall_rows.append({"method": name, "mean_over_languages": t["mean"].mean(),
                             "n_languages": len(t), "n_dirs": int(t["n_dirs"].sum()),
                             "n_above": int(t["n_above"].sum()),
                             "n_below": int(t["n_below"].sum())})
    languages = pd.concat(lang_rows, ignore_index=True)
    languages.to_csv(RESULTS / "languages.csv", index=False, lineterminator="\n",
                     float_format="%.6g")
    pd.DataFrame(overall_rows).to_csv(RESULTS / "overall.csv", index=False, lineterminator="\n",
                                      float_format="%.6g")

    # share of the gap to joined direct recovered, where joined (mutual best) beats thin
    gap = pooled_df.merge(pooled_df[pooled_df["method"] == "joined_mb"][["src", "tgt", "delta"]]
                          .rename(columns={"delta": "joined_gain"}), on=["src", "tgt"])
    gap = gap[gap["joined_gain"] > 0].assign(share=lambda g: g["delta"] / g["joined_gain"])
    gap.groupby(["method", "s"])["share"].agg(["mean", "median", "size"]).reset_index().to_csv(
        RESULTS / "gap.csv", index=False, lineterminator="\n", float_format="%.6g")

    secs = [rec["seconds"] for k in ctx.cache if (rec := ctx.cache.get(k))["seconds"] > 0]
    p6 = p6_dir()
    log = out_dir() / "run.log"
    info = {"commit": git("rev-parse", "HEAD").strip(),
            "eval_commits": eval_commits(log.read_text(encoding="utf-8")) if log.exists() else [],
            "config": XCFG, "eval": EVAL,
            "phase6_config": CFG["phase6"],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "n_records": sum(1 for _ in ctx.cache),
            "seconds_per_evaluation": float(np.mean(secs)) if secs else None,
            "inputs": file_fingerprints({str(f.relative_to(p6)): f for f in sorted(p6.rglob("*"))
                                         if f.is_file() and f.parent.name != "list"})}
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(f"results written to {RESULTS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs")
    ap.add_argument("--data-files", action="store_true", help="list inputs and exit")
    ap.add_argument("--results-only", action="store_true", help="rebuild results/ from cache")
    ap.add_argument("--forget-source", default=None,
                    help="drop cached evaluations with this source language first (their query "
                         "words changed; decisions.md 2026-11-19), then run as usual")
    args = ap.parse_args()
    if args.data_files:
        for f in data_files(pd.read_csv(HERE / "pairs.csv")):
            print(f.as_posix())
        return
    check_clean(git("status", "--porcelain").splitlines())
    if args.forget_source:
        n = forget(out_dir() / "evals.jsonl", lambda k: k["s"] == args.forget_source)
        print(f"dropped {n} cached evaluations with source {args.forget_source}", flush=True)
    ctx = Ctx(args.limit)
    if not args.results_only:
        run_all(ctx)
    write_results(ctx)


if __name__ == "__main__":
    main()
