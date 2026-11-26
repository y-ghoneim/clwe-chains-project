"""Our method on a standard benchmark: chains and several hubs on PanLex-BLI (Vulić et al. 2019).
See README.md (approved before running).

Per directed pair s-t of the benchmark's 15 languages: plain Procrustes (our vectors and
normalization) on the benchmark's direct training file (`--direct`, as published); hub steps on
its 5,000-pair files s-x and x-t with the test words removed (s-x without the test source words,
x-t without the test target words, every casing); the 3 hubs among the other 13 languages with
the widest steps (largest min(|s-x|, |x-t|) after the removal, ties by code). Methods: direct,
the chain through each hub, the combination (direct + chain), several hubs (direct + the 3
chains, CSLS scores averaged), and the same-data control (the direct file plus the s-t pairs
joined through the 3 hubs' steps on the hub word, mutual best, one map). CSLS (k as configured)
over the target's full vocabulary, both conventions: per test pair, its one translation (Vulić
et al.: "MAP is equal to MRR"), and per source word, any of its translations.

Resumable: one record per direction in <results>/panlex-bli/d<direct>/evals.jsonl. `--limit N`
(first N directions; no clean-tree check) keeps its own cache (d<direct>/smoke/) and writes
results/smoke/; `--results-only` rebuilds results/d<direct>/ from the cache.
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
from clwe_chains.align import compose, fit
from clwe_chains.dictionaries import COLUMNS, join_counts, mutual_best
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.evaluate import bli_with_combinations
from clwe_chains.muse import Vocab, train_rows
from clwe_chains.runs import Cache, file_fingerprints, result_affecting
from clwe_chains.stats import paired_bootstrap
from clwe_chains.text import load_vocab

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NAME = HERE.name
RESULT_PATHS = ("src/", "configs/", f"experiments/{NAME}/")
OWN_RESULTS = f"experiments/{NAME}/results/"
CFG = load_config()
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
MAPS = ["thin", "chain", "chain2", "chain3", "joined"]
COMBOS = {"combo": ("thin", "chain"), "hubs3": ("thin", "chain", "chain2", "chain3")}
METHODS = [*MAPS, *COMBOS]
# comparisons (a minus b), our method first
COMPARE = [("hubs3", "thin"), ("chain", "thin"), ("combo", "thin"), ("joined", "thin"),
           ("hubs3", "combo"), ("hubs3", "joined"), ("hubs3", "chain")]
CONVENTIONS = {"pair": "ranks", "word": "ranks_word"}
KW = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


# ---------------------------------------------------------------------------------------------
# The benchmark's files

def trans_file(s: str, t: str, split: str) -> Path:
    return CFG.data_root / XCFG["data"] / f"{s}-{t}" / f"{s}-{t}.{split}.cc.trans"


def read_trans(path: Path) -> pd.DataFrame:
    """src, tgt columns: one row per tab-separated line with exactly two fields, in file order."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[0] and parts[1]:
            rows.append(parts)
    return pd.DataFrame(rows, columns=["src", "tgt"])


def directions(langs: list[str]) -> list[tuple[str, str]]:
    return [(s, t) for s in langs for t in langs if s != t]


def inputs(langs: list[str], direct: int) -> dict[str, Path]:
    files = {}
    for s, t in directions(langs):
        for split in (f"train.{direct}", f"train.{XCFG['hub_file']}", "test.2000"):
            files[f"{s}-{t}.{split}"] = trans_file(s, t, split)
    return files


# ---------------------------------------------------------------------------------------------
# Test words, hub steps, the same-data control

def test_queries(test: pd.DataFrame, vs: Vocab, vt: Vocab) -> dict:
    """Per test pair: its query row and its one gold row; per source word: the index of its first
    pair and every gold row of the word (pairs with either word outside the vocabularies are
    dropped)."""
    q, g = [], []
    for w, x in zip(test["src"], test["tgt"], strict=True):
        a, b = vs.row(w), vt.row(x)
        if a is not None and b is not None:
            q.append(a)
            g.append(b)
    first: dict[int, int] = {}
    golds: dict[int, set[int]] = {}
    for i, (a, b) in enumerate(zip(q, g, strict=True)):
        first.setdefault(a, i)
        golds.setdefault(a, set()).add(b)
    return {"queries": q, "gold_pair": [[b] for b in g],
            "gold_word": [sorted(golds[a]) for a in q], "word_first": sorted(first.values()),
            "test_pairs": len(test), "test_pairs_used": len(q),
            "test_words": int(test["src"].nunique()), "test_words_used": len(first)}


def casing_rows(words, vocab: Vocab) -> set[int]:
    """Every row of every casing of the words."""
    return {r for w in set(words) for r in vocab.casings(w)} | \
        {r for w in set(words) if (r := vocab.row(w)) is not None}


def strip_test(d: pd.DataFrame, rows: set[int], side: str) -> pd.DataFrame:
    """d without the pairs whose `side` word ("a" or "b") is a test word."""
    return d[~d[side].isin(list(rows))].reset_index(drop=True)


def rank_hubs(sizes: dict[str, tuple[int, int]], k: int) -> list[str]:
    """The k hubs with the largest min(|s-x|, |x-t|); ties by language code."""
    return sorted(sizes, key=lambda x: (-min(sizes[x]), x))[:k]


def joined(direct: pd.DataFrame, steps: list[tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """The direct pairs plus the s-t pairs joined through the hubs' (test-removed) steps on the
    hub word, mutual best; links counted as distinct hub words over all hubs."""
    counts = pd.concat([join_counts(sx, xt) for sx, xt in steps], ignore_index=True)
    counts = counts.groupby(COLUMNS)["n"].sum().reset_index()
    extra = mutual_best(counts)
    out = pd.concat([direct[COLUMNS], extra[COLUMNS]], ignore_index=True).drop_duplicates()
    return out.sort_values(COLUMNS, ignore_index=True)


# ---------------------------------------------------------------------------------------------
# One direction

class Data:
    """Vocabularies and parsed files, loaded on demand."""

    def __init__(self, direct: int):
        self.direct = direct
        self._vocab: dict[str, Vocab] = {}
        self._rows: dict[tuple, pd.DataFrame] = {}

    def vocab(self, lang: str) -> Vocab:
        if lang not in self._vocab:
            self._vocab[lang] = Vocab(load_vocab(CFG.path("v0_embeddings") /
                                                 f"{lang}.vocab.txt"), lang)
        return self._vocab[lang]

    def rows(self, s: str, t: str, split: str) -> pd.DataFrame:
        if (s, t, split) not in self._rows:
            self._rows[s, t, split] = train_rows(read_trans(trans_file(s, t, split)),
                                                 self.vocab(s), self.vocab(t))
        return self._rows[s, t, split]


def plan(data: Data, s: str, t: str) -> dict:
    """The direction's test queries, direct pairs, ranked hubs and their test-removed steps."""
    test = read_trans(trans_file(s, t, "test.2000"))
    tq = test_queries(test, data.vocab(s), data.vocab(t))
    test_s = casing_rows(test["src"], data.vocab(s))
    test_t = casing_rows(test["tgt"], data.vocab(t))
    hub = f"train.{XCFG['hub_file']}"
    steps = {}
    for x in XCFG["languages"]:
        if x in (s, t):
            continue
        steps[x] = (strip_test(data.rows(s, x, hub), test_s, "a"),
                    strip_test(data.rows(x, t, hub), test_t, "b"))
    sizes = {x: (len(a), len(b)) for x, (a, b) in steps.items()}
    hubs = rank_hubs(sizes, XCFG["n_hubs"])
    return {"tq": tq, "direct": data.rows(s, t, f"train.{data.direct}"), "hubs": hubs,
            "steps": [steps[x] for x in hubs], "sizes": {x: sizes[x] for x in hubs}}


def evaluate(store: EmbeddingStore, p: dict, s: str, t: str, ev: dict) -> dict:
    """Every method's ranks for the direction, both conventions."""
    es, et = store[s], store[t]
    dev = es.device

    def w(a: torch.Tensor, b: torch.Tensor, d: pd.DataFrame) -> torch.Tensor:
        return fit(a[torch.as_tensor(d["a"].to_numpy(), device=dev)],
                   b[torch.as_tensor(d["b"].to_numpy(), device=dev)], "procrustes")

    maps = {"thin": w(es, et, p["direct"])}
    for i, (x, (sx, xt)) in enumerate(zip(p["hubs"], p["steps"], strict=True)):
        ex = store[x]
        maps["chain" if i == 0 else f"chain{i + 1}"] = compose([w(es, ex, sx), w(ex, et, xt)])
    control = joined(p["direct"], p["steps"])
    maps["joined"] = w(es, et, control)
    tq = p["tq"]
    out = bli_with_combinations(es, et, maps, COMBOS, np.array(tq["queries"]),
                                [tq["gold_pair"], tq["gold_word"]], **ev)   # ev: incl. retrieval
    first = np.array(tq["word_first"])
    return {"hubs": p["hubs"], "hub_sizes": p["sizes"], "direct_pairs": len(p["direct"]),
            "joined_pairs": len(control),
            **{k: v for k, v in tq.items() if k.startswith("test_")},
            "ranks": {m: out[m][0].tolist() for m in METHODS},
            "ranks_word": {m: out[m][1][first].tolist() for m in METHODS}}


# ---------------------------------------------------------------------------------------------
# Results

def pooled_bootstrap(d: np.ndarray, n: int, seed: int, chunk: int = 50
                     ) -> tuple[float, float, float]:
    """paired_bootstrap's statistic for a long vector of differences, resampled in chunks (one
    1,000 x 350k index matrix would not fit in memory)."""
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(0, len(d), size=(min(chunk, n - i), len(d)))]
                            .mean(axis=1) for i in range(0, n, chunk)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def tables(cache: Cache, dirs: list[tuple[str, str]], key) -> tuple[pd.DataFrame, ...]:
    per, cmp_rows = [], []
    rr = {conv: {m: [] for m in METHODS} for conv in CONVENTIONS}
    bs = XCFG["bootstrap"]
    for s, t in dirs:
        rec = cache.get(key(s, t))
        if rec is None:
            continue
        for conv, field in CONVENTIONS.items():
            r = {m: 1.0 / np.asarray(rec[field][m], dtype=float) for m in METHODS}
            for m in METHODS:
                rr[conv][m].append(r[m])
                per.append({"s": s, "t": t, "convention": conv, "method": m,
                            "mrr": r[m].mean(), "p1": float((r[m] == 1).mean()), "n": len(r[m]),
                            "hub1": rec["hubs"][0], "direct_pairs": rec["direct_pairs"],
                            "joined_pairs": rec["joined_pairs"]})
            for a, b in COMPARE:
                mean, lo, hi = paired_bootstrap(r[a], r[b], n=bs["n"], seed=bs["seed"])
                cmp_rows.append({"s": s, "t": t, "convention": conv, "a": a, "b": b,
                                 "delta": mean, "lo": lo, "hi": hi})
    per, cmp_dir = pd.DataFrame(per), pd.DataFrame(cmp_rows)
    if per.empty:
        return per, cmp_dir, pd.DataFrame(), pd.DataFrame()
    avg = (per.groupby(["convention", "method"], sort=False)[["mrr", "p1"]].mean()
           .reset_index().assign(n_directions=len(per) // (len(METHODS) * len(CONVENTIONS))))
    summ = []
    for conv in CONVENTIONS:
        pooled = {m: np.concatenate(rr[conv][m]) for m in METHODS}
        for a, b in COMPARE:
            mean, lo, hi = pooled_bootstrap(pooled[a] - pooled[b], bs["n"], bs["seed"])
            c = cmp_dir[(cmp_dir["convention"] == conv) & (cmp_dir["a"] == a)
                        & (cmp_dir["b"] == b)]
            summ.append({"convention": conv, "a": a, "b": b, "n_dirs": len(c),
                         "n_items": len(pooled[a]), "mrr_a": pooled[a].mean(),
                         "mrr_b": pooled[b].mean(), "delta": mean, "lo": lo, "hi": hi,
                         "mean_delta_dirs": c["delta"].mean(),
                         "n_above": int((c["lo"] > 0).sum()), "n_below": int((c["hi"] < 0).sum())})
    return per, cmp_dir, avg, pd.DataFrame(summ)


def reproduction(avg: pd.DataFrame, direct: int) -> pd.DataFrame:
    """Our direct map's average MRR (both conventions) next to Vulić et al.'s orthogonal map."""
    if direct != 1000 or avg.empty:
        return pd.DataFrame()
    thin = avg[avg["method"] == "thin"].set_index("convention")["mrr"]
    pub = XCFG["published_mrr_1k"]["orthogonal"]
    return pd.DataFrame([{"convention": c, "ours_direct_mrr": thin[c], "published_orthogonal": pub,
                          "diff": thin[c] - pub} for c in thin.index])


# ---------------------------------------------------------------------------------------------

def positive(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", type=int, choices=XCFG["direct"], default=XCFG["direct"][0])
    ap.add_argument("--basis", choices=list(XCFG["bases"]), default="ours",
                    help="ours: our normalization and CSLS; paper: length only, nearest neighbour")
    ap.add_argument("--limit", type=positive, default=None,
                    help="smoke run on the first N directions; writes results/smoke/")
    ap.add_argument("--results-only", action="store_true", help="build results/ from the cache")
    args = ap.parse_args()
    smoke = args.limit is not None
    dirty = result_affecting(git("status", "--porcelain").splitlines(), RESULT_PATHS, OWN_RESULTS)
    if dirty and not smoke:
        sys.exit("uncommitted changes that could affect results:\n" + "\n".join(dirty))
    langs = XCFG["languages"]
    dirs = directions(langs)[:args.limit] if smoke else directions(langs)
    basis = XCFG["bases"][args.basis]
    tag = f"d{args.direct}" + ("" if args.basis == "ours" else f"-{args.basis}")
    root = CFG.path("results") / NAME / tag / ("smoke" if smoke else "")  # smoke: own cache
    cache = Cache(root / "evals.jsonl")
    ev = {"k": CFG["eval"]["csls_k"], "block": min(CFG["eval"]["block_rows"], XCFG["block_rows"]),
          "half": CFG["eval"]["half"],       # the block size doesn't change ranks (config.yaml)
          "retrieval": basis["retrieval"]}

    def key(s: str, t: str) -> str:
        return (f"{s}-{t}|direct{args.direct}|hub{XCFG['hub_file']}|k{ev['k']}"
                + ("" if args.basis == "ours" else f"|{args.basis}"))

    if not args.results_only:
        store = EmbeddingStore(CFG, max_cached=6, steps=basis["normalize"])
        data = Data(args.direct)
        log = root / "run.log"
        root.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"start commit {git('rev-parse', 'HEAD').strip()}\n")
        start = time.time()
        for i, (s, t) in enumerate(dirs):
            if key(s, t) not in cache:
                cache.put(key(s, t), evaluate(store, plan(data, s, t), s, t, ev))
            rec = cache.get(key(s, t))
            line = (f"{time.strftime('%H:%M:%S')} {i + 1}/{len(dirs)} {s}-{t} via "
                    f"{','.join(rec['hubs'])}: {(time.time() - start) / 60:.1f} min")
            print(line, flush=True)
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    per, cmp_dir, avg, summ = tables(cache, dirs, key)
    if per.empty:
        print("no results yet")
        return
    out = HERE / "results" / ("smoke" if smoke else tag)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_direction.csv", **KW)
    cmp_dir.to_csv(out / "comparisons.csv", **KW)
    avg.to_csv(out / "averages.csv", **KW)
    summ.to_csv(out / "summary.csv", **KW)
    reproduction(avg, args.direct).to_csv(out / "reproduction.csv", **KW)
    log_text = (root / "run.log").read_text(encoding="utf-8") if (root / "run.log").exists() else ""
    info = {"commit": git("rev-parse", "HEAD").strip(), "smoke": smoke, "direct": args.direct,
            "basis": args.basis,
            "normalize_used": basis["normalize"] or CFG["embeddings"]["normalize"],
            "eval_commits": sorted({ln.split()[-1] for ln in log_text.splitlines()
                                    if ln.startswith("start commit")}),
            "config": XCFG, "eval": ev, "normalize": CFG["embeddings"]["normalize"],
            "n_directions": int(avg["n_directions"].iloc[0]),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "inputs": file_fingerprints(inputs(langs, args.direct))}
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(avg.to_string(index=False))
    print(summ.to_string(index=False))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
