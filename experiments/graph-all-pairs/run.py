"""Phase 4: the language graph on 40 distant pairs. See README.md.

Three stages, run in order; every evaluation is cached on disk, so a stage can be rerun after an
interruption and skips what's done:
  A : edge-measure cost matrices; each measure's cheapest chain (direct excluded), plus direct,
      via hub and the AUS rule; exact CSLS on the test concepts, full list and robustness subset;
  B : oracle: every one- and two-hop route scored by nearest neighbour on split 0's test
      concepts, the shortlist rescored with exact CSLS on all splits; an exhaustive exact check
      of the shortlist on seeded pairs;
  C : learned combination (cross-validated over pairs), d^p graphs, threshold rule.
Each stage ends by rebuilding results/ from the cache: routes.csv, edges.csv, metrics.csv,
bootstrap.csv, summary_pairs.csv, summary_measures.csv, run_info.json; oracle.csv,
oracle_check.csv and best_hub.csv once B has run; cv.csv once C has run.

The cache (evaluation records, cost matrices) lives under the data root, not in the repo.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import kendalltau, wilcoxon

from clwe_chains import learned, load_config
from clwe_chains.align import compose, fit
from clwe_chains.aus_rule import cosine_distance, feature_vector, path_cost
from clwe_chains.cv import cv_choose, pair_folds, threshold_scores
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.evaluate import bli
from clwe_chains.gold import gold_sets
from clwe_chains.graph import all_routes, path_costs, ranked, transform
from clwe_chains.measures import (
    econd_hm,
    edge_quality,
    pairwise,
    resource_cost,
    rsim,
    scale_max,
    spectrum,
    svg,
    uriel_cost,
    validation_concepts,
)
from clwe_chains.shared_list import list_languages
from clwe_chains.sources import wiktionary_candidates
from clwe_chains.stats import paired_bootstrap
from clwe_chains.text import load_vocab, vocab_index

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
# Paths whose changes can alter results; anything else (docs, the AUS PDF) can't.
RESULT_PATHS = ("src/", "prep/", "configs/", "tests/", "experiments/aus-replication/",
                "experiments/graph-all-pairs/")
OWN_RESULTS = "experiments/graph-all-pairs/results/"

URIEL = ("featural", "genetic", "geographic", "inventory", "morphological", "phonological",
         "syntactic")
MEASURES = ("quality", "quality_neglog", "resource", "svg", "econd_hm", "rsim",
            *(f"uriel_{k}" for k in URIEL))
SPLIT_MEASURES = ("quality", "quality_neglog", "rsim")   # one matrix (and route) per split
FEATURES = tuple(m for m in MEASURES if m != "quality_neglog")   # learned combination inputs
BASELINES = ("direct", "hub", "aus_syntax", "aus_geo")
SUBSETS = ("full", "robust")


# ---------------------------------------------------------------------------------------------
# Small helpers (tests/test_graph_experiment.py)

def assert_disjoint(train_ids, test_ids) -> None:
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError(f"{len(overlap)} test concepts among the training rows")


def test_concepts(test_ids, identical: pd.Series) -> tuple[list[int], int]:
    """Test concepts minus those whose target word equals the English string (phase 3)."""
    kept = [c for c in test_ids if not bool(identical.get(c, False))]
    return kept, len(test_ids) - len(kept)


def subset_split(split: dict, subset_ids: set) -> dict:
    return {part: [c for c in split[part] if c in subset_ids] for part in ("train", "test")}


def result_affecting(status_lines: list[str]) -> list[str]:
    """`git status --porcelain` lines that could change results (not this run's results/)."""
    out = []
    for line in status_lines:
        path = line[3:].strip().strip('"')
        if path.startswith(OWN_RESULTS):
            continue
        if path.startswith(RESULT_PATHS):
            out.append(line)
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def cache_key(kind: str, pair: tuple[str, str], route: tuple[str, ...], split: int,
              subset: str) -> str:
    """kind: csls (exact, test concepts), nn-test (oracle shortlist), nn-val (learned targets).
    The kind, not the stage, is in the key, so stages share evaluations (e.g. the oracle's
    rescoring reuses stage A's exact scores of the same route)."""
    return f"{kind}|{pair[0]}>{pair[1]}|{'-'.join(route)}|{split}|{subset}"


def parse_key(key: str) -> dict:
    kind, pair, route, split, subset = key.split("|")
    s, t = pair.split(">")
    return {"kind": kind, "s": s, "t": t, "route": route, "split": int(split), "subset": subset}


class Cache:
    """Evaluation records in an append-only JSONL file, loaded at start. A line cut short by an
    interruption is ignored (its evaluation is simply redone)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self._newline = False
        if self.path.exists():
            text = self.path.read_text(encoding="utf-8")
            for line in text.splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._data[rec["key"]] = rec["value"]
            self._newline = bool(text) and not text.endswith("\n")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            if self._newline:
                fh.write("\n")
                self._newline = False
            fh.write(json.dumps({"key": key, "value": record}) + "\n")
        self._data[key] = record

    def get_or_compute(self, key: str, fn) -> dict:
        if key not in self._data:
            self.put(key, fn())
        return self._data[key]


def hub_route(s: str, t: str, hubs: list[str]) -> tuple[str, str, str]:
    """s -> hub -> t with the first hub that isn't an endpoint (en, else ru)."""
    hub = next(h for h in hubs if h not in (s, t))
    return (s, hub, t)


def lang2vec_vectors(feats: dict[str, list], codes: dict[str, str],
                     langs: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Feature vectors per language, and the languages lang2vec has no values for (absent, or
    every value missing): those are reported and left out of the AUS rule."""
    vecs, missing = {}, []
    for lang in langs:
        values = feats.get(codes[lang])
        if values is None or all(v == "--" for v in values):
            missing.append(lang)
        else:
            vecs[lang] = feature_vector(values)
    return vecs, missing


def best_named(D: np.ndarray, names: list[str], s: str, t: str,
               max_mid: int) -> tuple[tuple[str, ...], float]:
    """Cheapest chain (direct excluded) as language names, with its cost."""
    routes, costs = ranked(D, names.index(s), names.index(t), max_mid)
    return tuple(names[i] for i in routes[0] if i >= 0), float(costs[0])


def aus_rule_routes(vecs: dict[str, np.ndarray], langs: list[str], s: str, t: str,
                    max_mid: int = 3) -> tuple[str, ...] | None:
    """The AUS rule: cheapest chain under the lang2vec cosine path cost, over the languages that
    have vectors; None if an endpoint has none."""
    if s not in vecs or t not in vecs:
        return None
    have = [x for x in langs if x in vecs]
    D = pairwise(have, lambda a, b: cosine_distance(vecs[a], vecs[b]))
    return best_named(D, have, s, t, max_mid)[0]


def threshold_cv(direct_cost: np.ndarray, mrr_chain: np.ndarray, mrr_direct: np.ndarray,
                 folds: np.ndarray) -> pd.DataFrame:
    """Threshold rule (ii) cross-validated over pairs. Arrays are pairs x splits. Per fold, the
    candidate τ are the training-fold pairs' direct costs; the τ with the best mean MRR over the
    training-fold pairs (mean over splits; ties: smallest τ) is applied to the held-out pairs.
    Returns per pair: fold, tau, mrr (mean over splits), n_chained (splits where it chains)."""
    n_splits = direct_cost.shape[1]
    out = pd.DataFrame({"fold": folds, "tau": np.nan, "mrr": np.nan, "n_chained": 0})
    for f in sorted(set(folds.tolist())):
        train, held = folds != f, folds == f
        taus = np.unique(direct_cost[train])
        score = np.mean([threshold_scores(direct_cost[:, j], mrr_chain[:, j], mrr_direct[:, j],
                                          taus).to_numpy() for j in range(n_splits)], axis=0)
        best = int(np.argmax(score[train].mean(axis=0)))
        out.loc[held, "tau"] = taus[best]
        out.loc[held, "mrr"] = score[held, best]
        out.loc[held, "n_chained"] = (direct_cost[held] > taus[best]).sum(axis=1)
    return out


def route_str(route) -> str:
    return "-".join(route)


# ---------------------------------------------------------------------------------------------
# Inputs, vectors and maps

class Ctx:
    """Everything a stage needs: config, list, splits, pairs, vectors, maps, cache."""

    def __init__(self, limit: int | None):
        self.cfg = load_config()
        self.xcfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
        cfg, lst = self.cfg, self.cfg.path("list")
        shared = pd.read_parquet(lst / "shared_list.parquet")
        self.langs = list_languages(shared)
        self.ids = shared["concept_id"].astype(int).tolist()
        self.id_set = set(self.ids)
        cand = pd.read_parquet(lst / "candidates.parquet")
        self.concepts = cand[cand["concept_id"].isin(self.id_set)]
        raw = json.loads((lst / "splits.json").read_text(encoding="utf-8"))["splits"]
        self.splits = {int(k): v for k, v in raw.items()}
        sense = pd.read_parquet(lst / "sense.parquet")
        rule = cfg["list"]["sense_subset"]
        self.robust = set(sense[(sense["coherence"] >= rule["min_coherence"])
                                & (sense["n_evaluable"] >= rule["min_evaluable"])]["concept_id"])
        pairs = pd.read_csv(HERE / "pairs.csv")
        self.all_pairs = pairs
        self.pairs = pairs if limit is None else pairs.head(limit)
        # word[lang]: concept_id -> vocab row, over every concept the language covers (round
        # trip passed; English covers every candidate). The list concepts are covered by all.
        self.word = {"en": cand.set_index("concept_id")["en_rank"].astype(int)}
        self.identical = {}
        for lang in self.langs:
            if lang == "en":
                continue
            p = pd.read_parquet(lst / "per_lang" / f"{lang}.parquet")
            p = p[p["round_trip"].astype(bool)].set_index("concept_id")
            self.word[lang] = p["word_idx"].astype(int)
            self.identical[lang] = p["identical"].astype(bool)
        self.device = torch.device(cfg["device"])
        self.store = EmbeddingStore(cfg)
        self.cache_dir = cfg.path("results") / "graph-all-pairs"
        self.cache = Cache(self.cache_dir / "evals.jsonl")
        self.ev = cfg["eval"]
        self.rows: dict[str, tuple[pd.Index, torch.Tensor]] = {}
        self.maps: dict[tuple, torch.Tensor] = {}
        self._wk = None
        self.n_identical = 0

    def gather(self) -> None:
        """Normalized vectors of every covered concept's word, per language, kept on the CPU."""
        for lang in self.langs:
            emb = self.store[lang]
            idx = torch.as_tensor(self.word[lang].to_numpy(), device=emb.device)
            self.rows[lang] = (pd.Index(self.word[lang].index), emb[idx].cpu())
        print("vectors gathered", flush=True)

    def vec(self, lang: str, concept_ids) -> torch.Tensor:
        index, rows = self.rows[lang]
        pos = index.get_indexer(list(concept_ids))
        if (pos < 0).any():
            raise KeyError(f"{lang} doesn't cover {int((pos < 0).sum())} requested concepts")
        return rows[torch.as_tensor(pos)].to(self.device)

    def split(self, split: int, subset: str) -> dict:
        sp = self.splits[split]
        return sp if subset == "full" else subset_split(sp, self.robust)

    def step_map(self, split: int, subset: str, a: str, b: str) -> torch.Tensor:
        """Procrustes a -> b on the split's (subset's) training concepts; memoized on the CPU."""
        key = (split, subset, a, b)
        if key not in self.maps:
            train = self.split(split, subset)["train"]
            self.maps[key] = fit(self.vec(a, train), self.vec(b, train), "procrustes").cpu()
        return self.maps[key]

    def route_map(self, route, split: int, subset: str) -> torch.Tensor:
        return compose([self.step_map(split, subset, a, b).to(self.device)
                        for a, b in itertools.pairwise(route)])

    def queries(self, s: str, t: str, split: int, subset: str) -> list[int]:
        sp = self.split(split, subset)
        q_ids, n_excl = test_concepts(sp["test"], self.identical[t])
        self.n_identical = max(self.n_identical, n_excl)
        assert_disjoint(sp["train"], q_ids)
        return q_ids

    def gold(self, s: str, t: str) -> dict[int, list[int]]:
        """List word + PanLex translations + (English source) Wiktionary, as in phase 3."""
        if t == "en":
            raise ValueError("gold for an English target isn't built here (no pair has one)")
        cfg = self.cfg
        if s == "en":
            src_key = self.concepts.set_index("concept_id").loc[self.ids, "en_key"]
            if self._wk is None:
                self._wk = pd.read_parquet(cfg.path("list") / "wiktionary_en.parquet",
                                           columns=["en", "pos", "sense", "code", "word"])
            vocab_t = load_vocab(cfg.path("v0_embeddings") / f"{t}.vocab.txt")
            wk_pairs, _ = wiktionary_candidates(self._wk, cfg["list"]["wikt_codes"].get(t, [t]),
                                                self.concepts, vocab_index(vocab_t, t), t)
        else:
            src_key, wk_pairs = self.word[s].loc[self.ids], None
        pl = cfg.path("v0_panlex")
        return gold_sets(self.word[t].loc[self.ids], src_key, pd.read_parquet(pl / f"{s}.parquet"),
                         pd.read_parquet(pl / f"{t}.parquet"), wk_pairs)

    def validation(self, a: str, b: str) -> list[int]:
        n = self.xcfg["validation"]["n"]
        val = validation_concepts(set(self.word[a].index), set(self.word[b].index), self.id_set,
                                  n)
        if len(val) < n:
            raise ValueError(f"{a}-{b}: only {len(val)} validation concepts")
        return val

    # evaluations (each returns a cache record)
    def eval_test(self, s, t, route, split, subset, emb_s, emb_t, gold, retrieval) -> dict:
        q_ids = self.queries(s, t, split, subset)
        ranks = bli(emb_s, emb_t, self.route_map(route, split, subset),
                    self.word[s].loc[q_ids].to_numpy(), [gold[c] for c in q_ids],
                    k=self.ev["csls_k"], block=self.ev["block_rows"], retrieval=retrieval,
                    half=bool(self.ev["half"]) and retrieval == "csls")
        return {"ranks": ranks.tolist()}

    def eval_val(self, s, t, route, split, emb_t) -> dict:
        val = self.validation(s, t)
        mrr = edge_quality(self.vec(s, val), emb_t, self.route_map(route, split, "full"),
                           np.arange(len(val)), [[int(self.word[t][c])] for c in val],
                           block=self.ev["block_rows"])
        return {"mrr": mrr, "n": len(val)}

    def csls(self, s, t, route, split, subset, emb_s, emb_t, gold) -> dict:
        return self.cache.get_or_compute(
            cache_key("csls", (s, t), route, split, subset),
            lambda: self.eval_test(s, t, route, split, subset, emb_s, emb_t, gold, "csls"))


def lang2vec_all(ctx: Ctx) -> dict[str, tuple[dict, list]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lang2vec imports the deprecated pkg_resources
        import lang2vec.lang2vec as l2v
    la = pd.read_csv(ctx.cfg.path("v0_languages")).set_index("ft_code")
    codes = {lang: ctx.xcfg["lang2vec_codes"].get(lang, la.loc[lang, "iso3"])
             for lang in ctx.langs}
    out = {}
    for fs in ("syntax_wals", "geo"):
        feats = l2v.get_features(sorted(set(codes.values())), fs)
        out[fs] = lang2vec_vectors(feats, codes, ctx.langs)
    return out


# ---------------------------------------------------------------------------------------------
# Cost matrices

def matrices(ctx: Ctx) -> dict[str, np.ndarray]:
    """Language-level matrices and, per split, the quality (directed) and RSIM matrices. Cached
    in matrices.npz; computing them needs ctx.gather() first."""
    path = ctx.cache_dir / "matrices.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    langs, cfg = ctx.langs, ctx.cfg
    t0 = time.time()
    mats = {"resource": resource_cost(cfg["pairs"]["sizes_gib"], langs)}
    spec = {}
    for lang in langs:   # raw vectors: spectrum() length-normalizes and centres them itself
        emb = torch.from_numpy(np.load(cfg.path("v0_embeddings") / f"{lang}.npy"))
        spec[lang] = spectrum(emb.to(ctx.device))
        del emb
    mats["svg"] = pairwise(langs, lambda a, b: svg(spec[a], spec[b]))
    mats["econd_hm"] = scale_max(pairwise(langs, lambda a, b: econd_hm(spec[a], spec[b])))
    dist = pd.read_parquet(cfg.path("v0_distances"))
    for kind in URIEL:
        mats[f"uriel_{kind}"] = uriel_cost(dist, langs, kind)
    print(f"language-level matrices: {time.time() - t0:.0f} s", flush=True)

    n = len(langs)
    val = {frozenset((a, b)): ctx.validation(a, b) for a, b in itertools.combinations(langs, 2)}
    for split in ctx.splits:
        train = ctx.splits[split]["train"]
        mats[f"rsim_{split}"] = pairwise(
            langs, lambda a, b, tr=train: 1.0 - rsim(ctx.vec(a, tr), ctx.vec(b, tr)))
        mats[f"quality_{split}"] = np.zeros((n, n))
    for j, b in enumerate(langs):   # target-major: each full target vocabulary loaded once
        t1 = time.time()
        emb_b = ctx.store[b]
        for split in ctx.splits:
            for i, a in enumerate(langs):
                if a == b:
                    continue
                v = val[frozenset((a, b))]
                w = ctx.step_map(split, "full", a, b).to(ctx.device)
                mrr = edge_quality(ctx.vec(a, v), emb_b, w,
                                   np.arange(len(v)), [[int(ctx.word[b][c])] for c in v],
                                   block=ctx.ev["block_rows"])
                mats[f"quality_{split}"][i, j] = 1.0 - mrr
        print(f"quality edges into {b}: {time.time() - t1:.0f} s", flush=True)
    ctx.maps.clear()   # the pair loop refits only the edges it uses
    ctx.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, **mats)
    print(f"matrices: {time.time() - t0:.0f} s", flush=True)
    return mats


def matrix(mats: dict, name: str, split: int) -> np.ndarray:
    if name == "quality_neglog":
        return transform(mats[f"quality_{split}"], "neglog")
    if name in SPLIT_MEASURES:
        return mats[f"{name}_{split}"]
    return mats[name]


def edges_table(ctx: Ctx, mats: dict) -> pd.DataFrame:
    rows = []
    for name in MEASURES:
        for split in (ctx.splits if name in SPLIT_MEASURES else ["all"]):
            D = matrix(mats, name, 0 if split == "all" else split)
            for i, a in enumerate(ctx.langs):
                for j, b in enumerate(ctx.langs):
                    if i != j:
                        rows.append({"measure": name, "split": split, "a": a, "b": b,
                                     "cost": D[i, j]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Routes table (results/routes.csv: s, t, measure, split, route, hops, cost)

def route_row(s, t, measure, split, route, cost) -> dict:
    return {"s": s, "t": t, "measure": measure, "split": split, "route": route_str(route),
            "hops": len(route) - 2, "cost": cost}


def headline_routes(ctx: Ctx, mats: dict) -> pd.DataFrame:
    max_mid = ctx.xcfg["max_mid"]
    l2v = lang2vec_all(ctx)
    rows = []
    for s, t in zip(ctx.all_pairs["s"], ctx.all_pairs["t"], strict=True):
        for split in ctx.splits:
            for name in MEASURES:
                route, cost = best_named(matrix(mats, name, split), ctx.langs, s, t, max_mid)
                rows.append(route_row(s, t, name, split, route, cost))
            rows.append(route_row(s, t, "direct", split, (s, t), np.nan))
            rows.append(route_row(s, t, "hub", split, hub_route(s, t, ctx.xcfg["hub"]), np.nan))
            for fs, name in (("syntax_wals", "aus_syntax"), ("geo", "aus_geo")):
                vecs = l2v[fs][0]
                route = aus_rule_routes(vecs, ctx.langs, s, t, max_mid)
                if route is not None:
                    rows.append(route_row(s, t, name, split, route, path_cost(vecs, list(route))))
    return pd.DataFrame(rows)


def update_routes(part: pd.DataFrame) -> pd.DataFrame:
    """Replace part's measures in results/routes.csv, keeping the others."""
    path = RESULTS / "routes.csv"
    if path.exists():
        old = pd.read_csv(path)
        part = pd.concat([old[~old["measure"].isin(set(part["measure"]))], part],
                         ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    part.to_csv(path, index=False, lineterminator="\n", float_format="%.6g")
    return part


# ---------------------------------------------------------------------------------------------
# Stages

class Progress:
    def __init__(self, stage: str, todo: int, n_pairs: int):
        self.stage, self.todo, self.n_pairs = stage, todo, n_pairs
        self.done, self.t0 = 0, time.time()

    def pair(self, i: int, s: str, t: str, new: int) -> None:
        self.done += new
        el = time.time() - self.t0
        eta = el / self.done * (self.todo - self.done) if self.done else float("nan")
        print(f"{self.stage} {i + 1}/{self.n_pairs} {s}>{t}: {new} new, {self.done}/{self.todo} "
              f"done, {el / 60:.0f} min elapsed, ~{eta / 60:.0f} min left", flush=True)


def evaluate_routes(ctx: Ctx, stage: str, jobs: dict) -> int:
    """jobs: (s, t) -> list of (route, split, subset), evaluated with exact CSLS, pair by pair."""
    todo = sum(cache_key("csls", p, r, sp, sub) not in ctx.cache
               for p, js in jobs.items() for r, sp, sub in js)
    print(f"{stage}: {todo} exact evaluations to run (~{todo * 4.1 / 3600:.1f} h at 4.1 s)",
          flush=True)
    prog = Progress(stage, todo, len(jobs))
    for i, ((s, t), js) in enumerate(jobs.items()):
        new = sum(cache_key("csls", (s, t), r, sp, sub) not in ctx.cache for r, sp, sub in js)
        if new:
            emb_s, emb_t, gold = ctx.store[s], ctx.store[t], ctx.gold(s, t)
            for route, split, subset in js:
                ctx.csls(s, t, route, split, subset, emb_s, emb_t, gold)
        prog.pair(i, s, t, new)
    return todo


def stage_a(ctx: Ctx) -> int:
    mats = matrices(ctx)
    edges_table(ctx, mats).to_csv(RESULTS / "edges.csv", index=False, lineterminator="\n",
                                  float_format="%.6g")
    routes = update_routes(headline_routes(ctx, mats))
    routes = routes[routes["measure"].isin([*MEASURES, *BASELINES])]
    jobs = {}
    for s, t in zip(ctx.pairs["s"], ctx.pairs["t"], strict=True):
        r = routes[(routes["s"] == s) & (routes["t"] == t)]
        uniq = sorted({(tuple(x.split("-")), int(sp)) for x, sp in zip(r["route"], r["split"],
                                                                        strict=True)},
                      key=lambda rs: (rs[1], len(rs[0]), rs[0]))
        jobs[(s, t)] = [(route, sp, sub) for sub in SUBSETS for route, sp in uniq]
    return evaluate_routes(ctx, "A", jobs)


def stage_b(ctx: Ctx) -> int:
    ocfg = ctx.xcfg["oracle"]
    n = len(ctx.langs)
    new, t0 = 0, time.time()
    short = {}
    for i, (s, t) in enumerate(zip(ctx.pairs["s"], ctx.pairs["t"], strict=True)):
        chains = [tuple(ctx.langs[v] for v in r if v >= 0) for r in
                  all_routes(n, ctx.langs.index(s), ctx.langs.index(t), ocfg["max_mid"])[1:]]
        emb_s, emb_t, gold = ctx.store[s], ctx.store[t], ctx.gold(s, t)
        nn = []
        for route in chains:
            key = cache_key("nn-test", (s, t), route, 0, "full")
            if key not in ctx.cache:
                ctx.cache.put(key, ctx.eval_test(s, t, route, 0, "full", emb_s, emb_t, gold, "nn"))
                new += 1
            rec = ctx.cache.get(key)
            nn.append(float(np.mean(1.0 / np.asarray(rec["ranks"]))))
        order = np.argsort(-np.asarray(nn), kind="stable")
        short[(s, t)] = [chains[k] for k in order[:ocfg["shortlist"]]]
        print(f"B shortlist {i + 1}/{len(ctx.pairs)} {s}>{t}: {time.time() - t0:.0f} s",
              flush=True)
    jobs = {p: [(r, sp, "full") for r in rs for sp in ctx.splits] for p, rs in short.items()}
    new += evaluate_routes(ctx, "B rescoring", jobs)
    check = check_pairs(ctx)
    jobs = {}
    for s, t in check:
        if not ((ctx.pairs["s"] == s) & (ctx.pairs["t"] == t)).any():
            continue   # a --limit run
        chains = [tuple(ctx.langs[v] for v in r if v >= 0) for r in
                  all_routes(n, ctx.langs.index(s), ctx.langs.index(t), ocfg["max_mid"])[1:]]
        jobs[(s, t)] = [(r, 0, "full") for r in chains]
    new += evaluate_routes(ctx, "B check", jobs)
    oracle_tables(ctx)
    return new


def check_pairs(ctx: Ctx) -> list[tuple[str, str]]:
    ocfg = ctx.xcfg["oracle"]
    pick = np.random.default_rng(ocfg["seed"]).choice(len(ctx.all_pairs), ocfg["check_pairs"],
                                                      replace=False)
    return [(ctx.all_pairs["s"][k], ctx.all_pairs["t"][k]) for k in sorted(pick.tolist())]


def mrr_of(rec: dict | None) -> float:
    return float(np.mean(1.0 / np.asarray(rec["ranks"]))) if rec else float("nan")


def oracle_tables(ctx: Ctx) -> None:
    """oracle.csv (each pair's shortlist, rescored), oracle_check.csv, best_hub.csv, and the
    oracle's rows in routes.csv. Pairs whose shortlist isn't fully scored yet are left out."""
    ocfg, n = ctx.xcfg["oracle"], len(ctx.langs)
    rows, check_rows, hub_rows, route_rows = [], [], [], []
    check = set(check_pairs(ctx))
    for s, t, stratum in zip(ctx.all_pairs["s"], ctx.all_pairs["t"], ctx.all_pairs["stratum"],
                             strict=True):
        chains = [tuple(ctx.langs[v] for v in r if v >= 0) for r in
                  all_routes(n, ctx.langs.index(s), ctx.langs.index(t), ocfg["max_mid"])[1:]]
        nn = [mrr_of(ctx.cache.get(cache_key("nn-test", (s, t), r, 0, "full"))) for r in chains]
        if np.isnan(nn).any():
            continue
        order = np.argsort(-np.asarray(nn), kind="stable")
        nn_rank = {chains[k]: rank + 1 for rank, k in enumerate(order)}
        short = [chains[k] for k in order[:ocfg["shortlist"]]]
        per = {r: [mrr_of(ctx.cache.get(cache_key("csls", (s, t), r, sp, "full")))
                   for sp in ctx.splits] for r in short}
        if any(np.isnan(v).any() for v in per.values()):
            continue
        means = {r: float(np.mean(v)) for r, v in per.items()}
        best = max(short, key=lambda r: means[r])   # ties: shortlist order
        for r in short:
            rows.append({"s": s, "t": t, "route": route_str(r), "hops": len(r) - 2,
                         "nn_rank": nn_rank[r], "nn_mrr": nn[chains.index(r)],
                         "mrr_mean": means[r], **{f"mrr_{sp}": v for sp, v in
                                                  zip(ctx.splits, per[r], strict=True)},
                         "oracle": r == best})
        for sp in ctx.splits:
            route_rows.append(route_row(s, t, "oracle", sp, best, np.nan))
        mids = best[1:-1]
        hub_rows.append({"s": s, "t": t, "stratum": stratum, "oracle_route": route_str(best),
                         "intermediates": "-".join(mids), "through_en": "en" in mids,
                         "hops": len(best) - 2})
        if (s, t) in check:
            exact = [mrr_of(ctx.cache.get(cache_key("csls", (s, t), r, 0, "full")))
                     for r in chains]
            if np.isnan(exact).any():
                continue
            ex_order = np.argsort(-np.asarray(exact), kind="stable")
            ex_rank = {chains[k]: rank + 1 for rank, k in enumerate(ex_order)}
            for r, e in zip(chains, exact, strict=True):
                check_rows.append({"s": s, "t": t, "route": route_str(r), "nn_rank": nn_rank[r],
                                   "nn_mrr": nn[chains.index(r)], "exact_mrr_split0": e,
                                   "exact_rank": ex_rank[r],
                                   "in_shortlist": nn_rank[r] <= ocfg["shortlist"]})
    out = {"oracle.csv": rows, "oracle_check.csv": check_rows, "best_hub.csv": hub_rows}
    for name, data in out.items():
        pd.DataFrame(data).to_csv(RESULTS / name, index=False, lineterminator="\n",
                                  float_format="%.6g")
    if route_rows:
        update_routes(pd.DataFrame(route_rows))


def stage_c(ctx: Ctx) -> int:
    mats = matrices(ctx)
    lcfg, max_mid, n = ctx.xcfg["learned"], ctx.xcfg["max_mid"], len(ctx.langs)
    pairs = list(zip(ctx.all_pairs["s"], ctx.all_pairs["t"], strict=True))
    run_pairs = set(zip(ctx.pairs["s"], ctx.pairs["t"], strict=True))
    folds = pair_folds(ctx.all_pairs["stratum"].tolist(), **ctx.xcfg["cv"])
    costs = {sp: {name: matrix(mats, name, sp) for name in FEATURES} for sp in ctx.splits}

    # 1. learned combination: training targets (NN MRR on the pair's validation concepts)
    def train_routes(i, s, t, sp):
        rs = learned.training_routes(costs[sp], ctx.langs.index(s), ctx.langs.index(t),
                                     lcfg["per_measure"], lcfg["n_random"], lcfg["seed"] + i)
        return rs, [tuple(ctx.langs[v] for v in r if v >= 0) for r in rs]

    new, t0 = 0, time.time()
    for i, (s, t) in enumerate(pairs):
        if (s, t) not in run_pairs:
            continue
        emb_t = ctx.store[t]
        for sp in ctx.splits:
            for route in train_routes(i, s, t, sp)[1]:
                key = cache_key("nn-val", (s, t), route, sp, "full")
                if key not in ctx.cache:
                    ctx.cache.put(key, ctx.eval_val(s, t, route, sp, emb_t))
                    new += 1
        print(f"C targets {i + 1}/{len(pairs)} {s}>{t}: {time.time() - t0:.0f} s", flush=True)
    if len(run_pairs) < len(pairs):
        print("C: a --limit run stops after the targets (the fits need every pair)")
        return new

    route_rows = []
    for sp in ctx.splits:
        X, y, fold_of = [], [], []
        for i, (s, t) in enumerate(pairs):
            rs, named = train_routes(i, s, t, sp)
            X.append(learned.route_features(costs[sp], rs))
            y.append([ctx.cache.get(cache_key("nn-val", (s, t), r, sp, "full"))["mrr"]
                      for r in named])
            fold_of.append(np.full(len(rs), folds[i]))
        X, y, fold_of = np.vstack(X), np.concatenate(y), np.concatenate(fold_of)
        for f in sorted(set(folds.tolist())):
            model = learned.fit(X[fold_of != f], y[fold_of != f], seed=lcfg["seed"])
            for i, (s, t) in enumerate(pairs):
                if folds[i] != f:
                    continue
                rs = all_routes(n, ctx.langs.index(s), ctx.langs.index(t), max_mid)[1:]
                feats = learned.route_features(costs[sp], rs)
                pick = learned.pick(model, feats, rs)
                pred = float(model.predict(feats).max())   # the "cost" column: predicted MRR
                route_rows.append(route_row(s, t, "learned", sp,
                                            tuple(ctx.langs[v] for v in pick), pred))

    # 2. d^p graphs (full list only)
    for name in MEASURES:
        for p in ctx.xcfg["alt_p"]:
            for s, t in pairs:
                for sp in ctx.splits:
                    D = transform(matrix(mats, name, sp), "power", p)
                    route, cost = best_named(D, ctx.langs, s, t, max_mid)
                    route_rows.append(route_row(s, t, f"{name}^{p}", sp, route, cost))
    routes = update_routes(pd.DataFrame(route_rows))
    jobs = {}
    for s, t in pairs:
        r = routes[(routes["s"] == s) & (routes["t"] == t)]
        is_learned = r["measure"] == "learned"
        is_power = r["measure"].str.contains("^", regex=False)
        js = [(tuple(x.split("-")), int(sp), sub) for x, sp in zip(r["route"][is_learned],
                                                                   r["split"][is_learned],
                                                                   strict=True)
              for sub in SUBSETS]
        js += [(tuple(x.split("-")), int(sp), "full") for x, sp in zip(r["route"][is_power],
                                                                       r["split"][is_power],
                                                                       strict=True)]
        jobs[(s, t)] = sorted(set(js), key=lambda j: (j[1], j[2], len(j[0]), j[0]))
    new += evaluate_routes(ctx, "C", jobs)
    cv_table(ctx, mats, routes, folds)
    return new


def pair_split_mrr(ctx: Ctx, routes: pd.DataFrame, measure: str, pairs, subset="full"):
    """pairs x splits matrix of the measure's route MRR (exact CSLS)."""
    r = routes[routes["measure"] == measure].set_index(["s", "t", "split"])["route"]
    return np.array([[mrr_of(ctx.cache.get(cache_key("csls", (s, t),
                                                     tuple(r[(s, t, sp)].split("-")), sp,
                                                     subset)))
                      for sp in ctx.splits] for s, t in pairs])


def cv_table(ctx: Ctx, mats: dict, routes: pd.DataFrame, folds: np.ndarray) -> None:
    """cv.csv: the exponent p and the threshold rule, each cross-validated over pairs."""
    pairs = list(zip(ctx.all_pairs["s"], ctx.all_pairs["t"], strict=True))
    direct = pair_split_mrr(ctx, routes, "direct", pairs)
    rows = []
    for name in MEASURES:
        ps = [1, *ctx.xcfg["alt_p"]]
        score = pd.DataFrame({p: pair_split_mrr(ctx, routes, name if p == 1 else f"{name}^{p}",
                                                pairs).mean(axis=1) for p in ps})
        per_pair, chosen = cv_choose(score, folds)
        for i, (s, t) in enumerate(pairs):
            rows.append({"kind": "power", "measure": name, "s": s, "t": t,
                         "fold": int(folds[i]), "choice": chosen[int(folds[i])],
                         "mrr": per_pair.iloc[i], "mrr_ref": score[1].iloc[i],
                         "mrr_direct": direct[i].mean(), "n_chained": np.nan})
        chain = pair_split_mrr(ctx, routes, name, pairs)
        cost = np.array([[matrix(mats, name, sp)[ctx.langs.index(s), ctx.langs.index(t)]
                          for sp in ctx.splits] for s, t in pairs])
        th = threshold_cv(cost, chain, direct, folds)
        for i, (s, t) in enumerate(pairs):
            rows.append({"kind": "threshold", "measure": name, "s": s, "t": t,
                         "fold": int(folds[i]), "choice": th["tau"].iloc[i],
                         "mrr": th["mrr"].iloc[i], "mrr_ref": direct[i].mean(),
                         "mrr_direct": direct[i].mean(), "n_chained": th["n_chained"].iloc[i]})
    pd.DataFrame(rows).to_csv(RESULTS / "cv.csv", index=False, lineterminator="\n",
                              float_format="%.6g")


# ---------------------------------------------------------------------------------------------
# Results (rebuilt from the cache after every stage)

def write_results(ctx: Ctx) -> None:
    bcfg = ctx.xcfg["bootstrap"]
    recs = [{**parse_key(k), "ranks": ctx.cache.get(k)["ranks"]} for k in ctx.cache
            if k.startswith("csls|")]
    if not recs:
        return
    met = pd.DataFrame([{k: v for k, v in r.items() if k not in ("kind", "ranks")}
                        | {"n": len(r["ranks"]),
                           "p1": float((np.asarray(r["ranks"]) <= 1).mean()),
                           "p5": float((np.asarray(r["ranks"]) <= 5).mean()),
                           "mrr": mrr_of(r)} for r in recs])
    met.to_csv(RESULTS / "metrics.csv", index=False, lineterminator="\n", float_format="%.6g")

    ranks = {(r["s"], r["t"], r["route"], r["split"], r["subset"]): np.asarray(r["ranks"])
             for r in recs}
    boot = []
    for (s, t, route, sp, sub), rk in ranks.items():
        d = ranks.get((s, t, f"{s}-{t}", sp, sub))
        if route == f"{s}-{t}" or d is None:
            continue
        dm, lo, hi = paired_bootstrap(1.0 / rk, 1.0 / d, **bcfg)
        boot.append({"s": s, "t": t, "route": route, "split": sp, "subset": sub, "d_mrr": dm,
                     "lo": lo, "hi": hi})
    boot = pd.DataFrame(boot, columns=["s", "t", "route", "split", "subset", "d_mrr", "lo", "hi"])
    boot.to_csv(RESULTS / "bootstrap.csv", index=False, lineterminator="\n",
                float_format="%.6g")

    routes = pd.read_csv(RESULTS / "routes.csv")
    keys = ["s", "t", "route", "split"]
    rows = []
    for sub in SUBSETS:
        m = met[met["subset"] == sub][[*keys, "mrr"]]
        b = boot[boot["subset"] == sub][[*keys, "d_mrr", "lo", "hi"]]
        j = routes.merge(m, on=keys, how="inner").merge(b, on=keys, how="left")
        j.loc[j["route"] == j["s"] + "-" + j["t"], ["d_mrr", "lo", "hi"]] = 0.0
        j["subset"] = sub
        rows.append(j)
    j = pd.concat(rows, ignore_index=True)
    pair_sum = j.groupby(["measure", "subset", "s", "t"], sort=False).agg(
        routes=("route", lambda x: ";".join(dict.fromkeys(x))), n_splits=("mrr", "size"),
        mrr=("mrr", "mean"), d_mrr=("d_mrr", "mean"),
        splits_above_0=("lo", lambda x: int((x > 0).sum())),
        splits_below_0=("hi", lambda x: int((x < 0).sum()))).reset_index()
    pair_sum = pair_sum[pair_sum["n_splits"] == len(ctx.splits)]
    pair_sum = add_oracle_columns(ctx, pair_sum, routes)
    strata = ctx.all_pairs.set_index(["s", "t"])["stratum"]
    pair_sum["stratum"] = [strata[(s, t)] for s, t in zip(pair_sum["s"], pair_sum["t"],
                                                          strict=True)]
    pair_sum.to_csv(RESULTS / "summary_pairs.csv", index=False, lineterminator="\n",
                    float_format="%.6g")
    summary_measures(pair_sum).to_csv(RESULTS / "summary_measures.csv", index=False,
                                      lineterminator="\n", float_format="%.6g")


def add_oracle_columns(ctx: Ctx, pair_sum: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """regret (oracle MRR - the measure's), top_pick (share of splits whose route is the
    oracle's) and kendall_tau (between -path cost and MRR over the pair's shortlist; base
    measures only), full list only."""
    path = RESULTS / "oracle.csv"
    pair_sum = pair_sum.assign(regret=np.nan, top_pick=np.nan, kendall_tau=np.nan)
    if not path.exists() or path.stat().st_size < 2:
        return pair_sum
    try:
        orc = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pair_sum
    mats = matrices(ctx)
    full = pair_sum["subset"] == "full"
    for (s, t), g in orc.groupby(["s", "t"], sort=False):
        best = g[g["oracle"]].iloc[0]
        sel = full & (pair_sum["s"] == s) & (pair_sum["t"] == t)
        pair_sum.loc[sel, "regret"] = best["mrr_mean"] - pair_sum.loc[sel, "mrr"]
        r = routes[(routes["s"] == s) & (routes["t"] == t)]
        hit = (r["route"] == best["route"]).groupby(r["measure"]).mean()
        pair_sum.loc[sel, "top_pick"] = pair_sum.loc[sel, "measure"].map(hit)
        short = [tuple(x.split("-")) for x in g["route"]]
        width = ctx.xcfg["oracle"]["max_mid"] + 2
        idx = np.array([[ctx.langs.index(x) for x in rt] + [-1] * (width - len(rt))
                        for rt in short])
        for name in MEASURES:
            c = np.mean([path_costs(matrix(mats, name, sp), idx) for sp in ctx.splits], axis=0)
            tau = kendalltau(-c, g["mrr_mean"].to_numpy()).statistic
            pair_sum.loc[sel & (pair_sum["measure"] == name), "kendall_tau"] = tau
    return pair_sum


def summary_measures(pair_sum: pd.DataFrame) -> pd.DataFrame:
    groups = {"all": lambda d: d, "stratum_a": lambda d: d[d["stratum"] == "a"],
              "stratum_b": lambda d: d[d["stratum"] == "b"],
              "en_source": lambda d: d[d["s"] == "en"],
              "non_en_source": lambda d: d[d["s"] != "en"]}
    out = []
    for (measure, sub), d in pair_sum.groupby(["measure", "subset"], sort=False):
        for gname, sel in groups.items():
            g = sel(d)
            if not len(g):
                continue
            dm = g["d_mrr"].to_numpy()
            try:
                p = float(wilcoxon(dm).pvalue) if np.any(dm != 0) else np.nan
            except ValueError:
                p = np.nan
            out.append({"measure": measure, "subset": sub, "group": gname, "n_pairs": len(g),
                        "mrr": g["mrr"].mean(), "d_mrr_mean": dm.mean(),
                        "d_mrr_median": float(np.median(dm)), "wilcoxon_p": p,
                        "pairs_won": int((dm > 0).sum()),
                        "pairs_above_3of5": int((g["splits_above_0"] >= 3).sum()),
                        "pairs_below_3of5": int((g["splits_below_0"] >= 3).sum()),
                        "regret_mean": g["regret"].mean(), "top_pick_rate": g["top_pick"].mean(),
                        "kendall_tau_mean": g["kendall_tau"].mean()})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["A", "B", "C", "results"], required=True)
    ap.add_argument("--limit", type=int, help="only the first N pairs (a smoke run; the cache "
                    "is shared, so the full run reuses it)")
    args = ap.parse_args()
    dirty = result_affecting(git("status", "--porcelain").splitlines())
    if dirty:
        sys.exit("uncommitted changes that can affect results:\n" + "\n".join(dirty))
    started, t0 = time.strftime("%Y-%m-%dT%H:%M:%S"), time.time()
    RESULTS.mkdir(exist_ok=True)
    ctx = Ctx(args.limit)
    new = 0
    if args.stage != "results":
        ctx.gather()
        new = {"A": stage_a, "B": stage_b, "C": stage_c}[args.stage](ctx)
    write_results(ctx)

    info_path = RESULTS / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    l2v = lang2vec_all(ctx)
    info.update({"commit": git("rev-parse", "HEAD").strip(), "config": ctx.xcfg,
                 "list_config": ctx.cfg["list"], "eval_config": ctx.ev,
                 "n_concepts": len(ctx.ids), "n_robust": len(ctx.robust),
                 "n_pairs": len(ctx.all_pairs), "cache_dir": str(ctx.cache_dir),
                 "lang2vec_missing": {fs: v[1] for fs, v in l2v.items()},
                 "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                            else "cpu")})
    info.setdefault("stages", {})
    info["stages"].setdefault(args.stage, []).append(
        {"commit": info["commit"], "started": started,
         "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0),
         "limit": args.limit, "new_evaluations": new,
         "max_identical_excluded_per_split": ctx.n_identical})
    info_path.write_text(json.dumps(info, indent=1, default=str) + "\n", encoding="utf-8")
    print("done", args.stage, info["commit"], f"{(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
