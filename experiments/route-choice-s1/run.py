"""Phase 6 stage 1E: route choice under natural dictionaries. See README.md (approved first).

Stage 1's 77 pairs, folds, test words and gold (experiments/scarce-natural-s1/run.py is reused for
them); routes with one or two intermediates among the 30 nodes, each step Procrustes on its own
natural dictionary with the fold's test words removed (prep/09 builds the edges). Steps:
  1. edge measures on validation pairs from each edge's own dictionary (never a list word of the
     pair's endpoints): quality edge (1 - NN MRR) and RSIM; size and URIEL come from prep;
  2. each measure's graph picks one route per pair (s -> t costs; ties: fewer hops);
  3. oracle screening: every route by NN MRR on fold 0, s -> t;
  4. exact CSLS on every distinct picked route, the stage 1 hub route, the oracle's top routes and
     thin direct, all folds, both directions (the route reversed for t -> s).
Every evaluation and measure is cached, so the run resumes; results/ is rebuilt from the cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import kendalltau

from clwe_chains.align import compose
from clwe_chains.dictionaries import remove_test
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.evaluate import bli, bli_multi
from clwe_chains.graph import ranked, widest_ranked
from clwe_chains.measures import rsim
from clwe_chains.runs import (
    Cache,
    cache_key,
    file_fingerprints,
    parse_key,
    result_affecting,
    settings_hash,
)
from clwe_chains.stats import paired_bootstrap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
_spec = importlib.util.spec_from_file_location("scarce_s1_run",
                                               HERE.parent / "scarce-natural-s1" / "run.py")
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)
RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/route-choice-s1/",
                "experiments/scarce-natural-s1/run.py", "experiments/scarce-natural-s1/config.yaml",
                "experiments/scarce-natural-s1/pairs.csv")
OWN_RESULTS = "experiments/route-choice-s1/results/"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
MEASURES = ("size_widest", "size_sum", "quality", "rsim", "uriel")


# ---------------------------------------------------------------------------------------------
# Small helpers (tests/test_route_choice_experiment.py)

def key(kind: str, src: str, tgt: str, route: tuple[str, ...], fold: int) -> str:
    return cache_key(kind, s1.SETTINGS, (src, tgt), route, fold, "full")


def route_str(route) -> str:
    return "-".join(route)


def validation_split(d: pd.DataFrame, exclude_a: set[int], exclude_b: set[int], frac: float,
                     cap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, validation) for an edge measure: pairs touching an excluded row are dropped; the
    validation words are the most frequent (smallest rows) frac of the remaining a words, at most
    cap, with all their translations; train is every remaining pair of the other a words."""
    d = remove_test(d, exclude_a, exclude_b)
    words = np.sort(d["a"].unique())
    val = set(words[:min(cap, round(frac * len(words)))].tolist())
    is_val = d["a"].isin(list(val))
    return d[~is_val].reset_index(drop=True), d[is_val].reset_index(drop=True)


def step_test(a: str, b: str, s: str, t: str, ts: set[int], tt: set[int]) -> tuple[set, set]:
    """Rows to remove from the a and b sides of step a -> b of a route between s and t."""
    side = {s: ts, t: tt}
    return side.get(a, set()), side.get(b, set())


def pick(costs: dict[str, np.ndarray], sizes: np.ndarray, s: int, t: int,
         max_mid: int) -> dict[str, tuple[int, ...]]:
    """Each measure's route (node indices, s first): the cheapest summed cost, or for
    size_widest the widest narrowest step; ties go to fewer hops, then route order."""
    out = {"size_widest": widest_ranked(sizes, s, t, max_mid)[0][0]}
    for name, D in costs.items():
        out[name] = ranked(D, s, t, max_mid)[0][0]
    return {k: tuple(int(x) for x in r if x >= 0) for k, r in out.items()}


def kendall(cost: list[float], mrr: list[float]) -> float:
    """Kendall tau between route cost and MRR, sign flipped so that 1 = cheaper is better."""
    if len(cost) < 3 or len(set(cost)) < 2:
        return np.nan
    return float(-kendalltau(cost, mrr).statistic)


# ---------------------------------------------------------------------------------------------
# Inputs

def check_clean(status_lines: list[str]) -> None:
    dirty = result_affecting(status_lines, RESULT_PATHS, OWN_RESULTS)
    if dirty:
        sys.exit("uncommitted changes that could affect results:\n" + "\n".join(dirty))


class Ctx(s1.Ctx):
    """Stage 1's pairs, folds, test words and gold, plus every route edge."""

    def __init__(self, limit: int | None):
        super().__init__(limit)
        r = s1.CFG.path("phase6") / "routes"
        self.rdir = r
        self.cache = Cache(s1.CFG.path("results") / "route-choice-s1" / "evals.jsonl")
        self.store = EmbeddingStore(s1.CFG, max_cached=XCFG["max_cached_vectors"])
        sz = pd.read_csv(r / "dict_sizes.csv")
        self.size = {frozenset((a, b)): min(x, y) for a, b, x, y in
                     zip(sz["a"], sz["b"], sz["a_words"], sz["b_words"], strict=True)}
        self.biggest = float(max(self.size.values()))       # M in log(M / size)
        ur = pd.read_csv(r / "uriel.csv")
        self.uriel = {frozenset((a, b)): v for a, b, v in zip(ur["a"], ur["b"], ur["uriel"],
                                                              strict=True)}
        low = set(s1.CFG["phase6"]["low"])
        self.nodes = sorted({x for e in self.uriel for x in e} - low)
        self.list_rows: dict[str, set[int]] = {}
        for (lang, _), rows in self.rows.items():
            self.list_rows.setdefault(lang, set()).update(rows)
        self._edges: dict[tuple[str, str], pd.DataFrame] = {}
        self._maps: dict[tuple[str, str], torch.Tensor] = {}
        self._pair_maps: dict[tuple[int, str, str], torch.Tensor] = {}
        self._pair_key: tuple[str, str] | None = None

    def edge(self, a: str, b: str) -> pd.DataFrame:
        """The natural dictionary of a -> b (columns a, b in that orientation)."""
        if (a, b) not in self._edges:
            f = self.rdir / "dicts" / f"{a}__{b}.parquet"
            if f.exists():
                d = pd.read_parquet(f)
            else:
                d = pd.read_parquet(self.rdir / "dicts" / f"{b}__{a}.parquet").rename(
                    columns={"a": "b", "b": "a"})[["a", "b"]]
            self._edges[a, b] = d
        return self._edges[a, b]


# ---------------------------------------------------------------------------------------------
# Edge measures

def edge_measure(ctx: Ctx, a: str, b: str, s: str, t: str) -> dict:
    """Quality edge and RSIM of step a -> b in a route from s to t, on validation pairs from the
    edge's own dictionary, cached. Pairs touching a list word of s (on the a side of s -> x) or of
    t (on the b side of x -> t) are dropped first, so no fold's test word is used; a step between
    two intermediates drops nothing (README; final review C2). Too few validation words: the worst
    cost (quality 0, RSIM -1)."""
    excluded = (["a"] if a == s else []) + (["b"] if b == t else [])
    v = XCFG["validation"]
    k = key("measure", a, b, ("edge", *excluded, settings_hash(v)), -1)
    if k in ctx.cache:
        return ctx.cache.get(k)
    ex_a = ctx.list_rows.get(a, set()) if "a" in excluded else set()
    ex_b = ctx.list_rows.get(b, set()) if "b" in excluded else set()
    train, val = validation_split(ctx.edge(a, b), ex_a, ex_b, v["frac"], v["max"])
    n_val = int(val["a"].nunique())
    rec = {"excluded": excluded, "n_train": len(train), "n_val": n_val,
           "val_min_row": int(val["a"].min()) if n_val else None, "quality": 0.0, "rsim": -1.0}
    if n_val >= v["min"] and len(train) > 0:
        ea, eb = ctx.store[a], ctx.store[b]
        w = s1.edge_map(train, ea, eb, True)
        gold = val.groupby("a")["b"].apply(list)
        q = np.array(gold.index, dtype=np.int64)
        ranks = bli(ea, eb, w, q, gold.tolist(), block=s1.CFG["eval"]["block_rows"],
                    retrieval="nn")
        first = val.drop_duplicates("a")
        rec["quality"] = float((1.0 / ranks).mean())
        rec["rsim"] = rsim(ea[s1._idx(first["a"], ea.device)], eb[s1._idx(first["b"], eb.device)])
    ctx.cache.put(k, rec)
    return rec


def size_cost(sizes: np.ndarray, biggest: float) -> np.ndarray:
    """log(biggest / size): 0 for the largest edge, larger for smaller ones, never negative, so a
    summed route cost can't be lowered by adding hops (final review C1; -log(size) could)."""
    with np.errstate(divide="ignore"):
        return np.where(sizes > 0, np.log(biggest / np.where(sizes > 0, sizes, 1)), np.inf)


def cost_matrices(ctx: Ctx, s: str, t: str, langs: list[str]) -> tuple[dict, np.ndarray]:
    """Per measure, the step-cost matrix over langs (nodes + s) for routes s -> t, and the size
    matrix. Steps into s or out of t are never used (infinite cost)."""
    n = len(langs)
    sizes = np.zeros((n, n))
    costs = {m: np.zeros((n, n)) for m in ("size_sum", "quality", "rsim", "uriel")}
    for i, a in enumerate(langs):
        for j, b in enumerate(langs):
            if i == j:
                continue
            e = frozenset((a, b))
            sz = ctx.size.get(e, 0)
            sizes[i, j] = sz
            costs["uriel"][i, j] = ctx.uriel[e]
            if b == s or a == t or sz == 0:      # unusable steps
                costs["quality"][i, j] = costs["rsim"][i, j] = np.inf
                continue
            m = edge_measure(ctx, a, b, s, t)
            costs["quality"][i, j] = 1.0 - m["quality"]
            costs["rsim"][i, j] = 1.0 - m["rsim"]
    costs["size_sum"] = size_cost(sizes, ctx.biggest)
    np.fill_diagonal(costs["size_sum"], 0.0)
    return costs, sizes


# ---------------------------------------------------------------------------------------------
# Maps and evaluation

def route_map(ctx: Ctx, route: tuple[str, ...], s: str, t: str, fold: int) -> torch.Tensor:
    """Composed map of a route (src first), each step Procrustes on its fold dictionary; steps
    between two nodes other than s and t carry no test words and are cached across pairs."""
    test = ctx.folds[s, t][fold]
    ts, tt = ctx.test_rows(s, test), ctx.test_rows(t, test)
    if ctx._pair_key != (s, t):            # steps touching s or t: reused within a pair
        ctx._pair_maps, ctx._pair_key = {}, (s, t)
    maps = []
    for a, b in itertools.pairwise(route):
        clean = a not in (s, t) and b not in (s, t)
        cache, k = (ctx._maps, (a, b)) if clean else (ctx._pair_maps, (fold, a, b))
        if k not in cache:
            ra, rb = step_test(a, b, s, t, ts, tt)
            d = remove_test(ctx.edge(a, b), ra, rb)
            cache[k] = s1.edge_map(d, ctx.store[a], ctx.store[b], True)
        maps.append(cache[k])
    return compose(maps)


def thin_map(ctx: Ctx, src: str, tgt: str, s: str, t: str, fold: int) -> torch.Tensor:
    test = ctx.folds[s, t][fold]
    d = remove_test(ctx.dictionary(f"{s}__{t}"), ctx.test_rows(s, test), ctx.test_rows(t, test))
    return s1.edge_map(d, ctx.store[s], ctx.store[t], src == s)


def queries_gold(ctx: Ctx, src: str, tgt: str, s: str, t: str, fold: int):
    concepts = ctx.folds[s, t][fold]
    q = np.array([ctx.query[src, c] for c in concepts])
    golds = [[ctx.gold[src, tgt, kind, c] for c in concepts] for kind in ("main", "list")]
    return concepts, q, golds


def screen(ctx: Ctx, s: str, t: str, routes: list[tuple[str, ...]]) -> dict[str, float]:
    """NN MRR of every route on fold 0, s -> t (the oracle's screening), cached per route."""
    _, q, golds = queries_gold(ctx, s, t, s, t, 0)
    es, et = ctx.store[s], ctx.store[t]
    src_q = es[s1._idx(q, es.device)]
    out = {}
    for r in routes:
        k = key("nn-screen", s, t, r, 0)
        if k not in ctx.cache:
            w = route_map(ctx, r, s, t, 0)
            ranks = bli(src_q, et, w, np.arange(len(q)), golds[0],
                        block=s1.CFG["eval"]["block_rows"], retrieval="nn")
            ctx.cache.put(k, {"mrr": float((1.0 / ranks).mean())})
        out[route_str(r)] = ctx.cache.get(k)["mrr"]
    return out


def rescore(ctx: Ctx, s: str, t: str, routes: list[tuple[str, ...]]) -> int:
    """Exact CSLS for thin direct and each route, every fold, both directions."""
    done, ev = 0, {"k": s1.EVAL["csls_k"], "block": s1.CFG["eval"]["block_rows"],
                   "half": s1.EVAL["half"]}
    for fold in sorted(ctx.folds[s, t]):
        for src, tgt in ((s, t), (t, s)):
            jobs = [("direct",)] + [r if src == s else tuple(reversed(r)) for r in routes]
            jobs = [j for j in dict.fromkeys(jobs)
                    if key("csls", src, tgt, j, fold) not in ctx.cache]
            if not jobs:
                continue
            concepts, q, golds = queries_gold(ctx, src, tgt, s, t, fold)
            for j in jobs:
                w = (thin_map(ctx, src, tgt, s, t, fold) if j == ("direct",)
                     else route_map(ctx, j, s, t, fold))
                t0 = time.perf_counter()
                ranks = bli_multi(ctx.store[src], ctx.store[tgt], w, q, golds, retrieval="csls",
                                  **ev)
                ctx.cache.put(key("csls", src, tgt, j, fold),
                              {"concepts": concepts, "rank": ranks[0].tolist(),
                               "rank_list": ranks[1].tolist(),
                               "seconds": time.perf_counter() - t0})
                done += 1
    return done


def pair_routes(ctx: Ctx, s: str, t: str, hub: str) -> dict:
    """Every measure's pick, the stage 1 hub route and the oracle's top routes for one pair."""
    langs = [*ctx.nodes, s]
    costs, sizes = cost_matrices(ctx, s, t, langs)
    si, ti = langs.index(s), langs.index(t)
    picks = {m: tuple(langs[i] for i in r) for m, r in
             pick(costs, sizes, si, ti, XCFG["max_mid"]).items()}
    cands = ranked(costs["uriel"], si, ti, XCFG["max_mid"])[0]       # every route, any order
    all_r = [tuple(langs[int(i)] for i in r if i >= 0) for r in cands]
    nn = screen(ctx, s, t, all_r)
    top = sorted(nn, key=lambda r: (-nn[r], r.count("-"), r))[:XCFG["oracle_top"]]
    route_cost = {}
    for m in MEASURES:
        D = sizes if m == "size_widest" else costs[m]
        for r in [*picks.values(), (s, hub, t), *(tuple(x.split("-")) for x in top)]:
            idx = [langs.index(x) for x in r]
            steps = [D[a, b] for a, b in itertools.pairwise(idx)]
            route_cost[m, route_str(r)] = (-min(steps) if m == "size_widest" else sum(steps))
    return {"picks": picks, "hub": (s, hub, t), "top": [tuple(x.split("-")) for x in top],
            "nn": nn, "cost": route_cost}


def run_all(ctx: Ctx) -> dict:
    log = s1.CFG.path("results") / "route-choice-s1" / "run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"start commit {s1.git('rev-parse', 'HEAD').strip()}\n")
    start, total, plans = time.time(), 0, {}
    for i, (s, t, hub) in enumerate(zip(ctx.pairs["s"], ctx.pairs["t"], ctx.pairs["hub"],
                                        strict=True)):
        plan = pair_routes(ctx, s, t, hub)
        plans[s, t] = plan
        routes = list(dict.fromkeys([*plan["picks"].values(), plan["hub"], *plan["top"]]))
        total += rescore(ctx, s, t, routes)
        el = time.time() - start
        msg = (f"{time.strftime('%H:%M:%S')} pair {i + 1}/{len(ctx.pairs)} {s}-{t}: "
               f"{len(routes)} routes, {total} CSLS evaluations, {el / 60:.1f} min")
        print(msg, flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    return plans


# ---------------------------------------------------------------------------------------------
# Results

def pooled_rr(ctx: Ctx, src: str, tgt: str, route: tuple[str, ...], folds) -> pd.DataFrame | None:
    parts = []
    for f in folds:
        rec = ctx.cache.get(key("csls", src, tgt, route, f))
        if rec is None:
            return None
        parts.append(pd.DataFrame({"concept": rec["concepts"],
                                   "rr": 1.0 / np.asarray(rec["rank"], dtype=float),
                                   "rr_list": 1.0 / np.asarray(rec["rank_list"], dtype=float)}))
    return pd.concat(parts).sort_values("concept", ignore_index=True)


def write_results(ctx: Ctx, plans: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    bs = XCFG["bootstrap"]
    route_rows, pooled_rows, summary_rows, kend_rows = [], [], [], []
    for (s, t), plan in plans.items():
        folds = sorted(ctx.folds[s, t])
        choices = {**{m: r for m, r in plan["picks"].items()}, "hub": plan["hub"]}
        for m, r in choices.items():
            route_rows.append({"s": s, "t": t, "chooser": m, "route": route_str(r),
                               "hops": len(r) - 2, "nn_fold0": plan["nn"][route_str(r)]})
        for rank, r in enumerate(plan["top"], 1):
            route_rows.append({"s": s, "t": t, "chooser": f"oracle_{rank}", "route": route_str(r),
                               "hops": len(r) - 2, "nn_fold0": plan["nn"][route_str(r)]})
        scored = list(dict.fromkeys([*choices.values(), *plan["top"]]))
        for src, tgt in ((s, t), (t, s)):
            thin = pooled_rr(ctx, src, tgt, ("direct",), folds)
            mrr = {}
            for r in scored:
                rr = r if src == s else tuple(reversed(r))
                p = pooled_rr(ctx, src, tgt, rr, folds)
                if p is None or thin is None:
                    continue
                mrr[route_str(r)] = p["rr"].mean()
                delta, lo, hi = paired_bootstrap(p["rr"].to_numpy(), thin["rr"].to_numpy(), **bs)
                pooled_rows.append({"src": src, "tgt": tgt, "s": s, "route": route_str(r),
                                    "mrr": p["rr"].mean(), "mrr_list": p["rr_list"].mean(),
                                    "mrr_thin": thin["rr"].mean(), "delta": delta, "lo": lo,
                                    "hi": hi, "n": len(p)})
            if not mrr:
                continue
            best = max(mrr, key=lambda r: (mrr[r], -r.count("-")))
            hub_mrr = mrr[route_str(plan["hub"])]
            for m, r in [*choices.items(), ("oracle", tuple(best.split("-")))]:
                rs = route_str(r)
                p = next(x for x in pooled_rows[::-1]
                         if (x["src"], x["tgt"], x["route"]) == (src, tgt, rs))
                summary_rows.append({"chooser": m, "src": src, "tgt": tgt, "s": s, "route": rs,
                                     "hops": len(r) - 2, "mrr": mrr[rs], "delta_thin": p["delta"],
                                     "above": p["lo"] > 0, "below": p["hi"] < 0,
                                     "delta_hub": mrr[rs] - hub_mrr,
                                     "regret": mrr[best] - mrr[rs], "hit": rs == best})
            for m in MEASURES:
                rs = list(mrr)
                kend_rows.append({"measure": m, "src": src, "tgt": tgt, "n_routes": len(rs),
                                  "tau": kendall([plan["cost"][m, r] for r in rs],
                                                 [mrr[r] for r in rs])})
    pd.DataFrame(route_rows).to_csv(RESULTS / "routes.csv", index=False, lineterminator="\n",
                                    float_format="%.6g")
    pd.DataFrame(pooled_rows).to_csv(RESULTS / "pooled.csv", index=False, lineterminator="\n",
                                     float_format="%.6g")
    per = pd.DataFrame(summary_rows)
    per.to_csv(RESULTS / "choosers.csv", index=False, lineterminator="\n", float_format="%.6g")
    kend = pd.DataFrame(kend_rows)
    if len(per):
        summ = per.groupby("chooser").agg(
            pair_dirs=("mrr", "size"), mrr=("mrr", "mean"), delta_thin=("delta_thin", "mean"),
            n_above=("above", "sum"), n_below=("below", "sum"), delta_hub=("delta_hub", "mean"),
            regret=("regret", "mean"), hit_rate=("hit", "mean"),
            two_hop_share=("hops", lambda h: (h == 2).mean()))
        summ = summ.join(kend.groupby("measure")["tau"].mean().rename("kendall_tau"))
        summ.reset_index().to_csv(RESULTS / "summary.csv", index=False, lineterminator="\n",
                                  float_format="%.6g")
        per.groupby(["chooser", "s"])[["mrr", "delta_thin", "delta_hub", "regret"]].mean() \
            .reset_index().to_csv(RESULTS / "by_language.csv", index=False, lineterminator="\n",
                                  float_format="%.6g")
    edges = [{"a": parse_key(k)["s"], "b": parse_key(k)["t"], **ctx.cache.get(k)}
             for k in ctx.cache if k.startswith("measure|")]
    pd.DataFrame(edges).to_csv(RESULTS / "edges.csv", index=False, lineterminator="\n",
                               float_format="%.6g")
    log = s1.CFG.path("results") / "route-choice-s1" / "run.log"
    secs = [rec["seconds"] for k in ctx.cache if k.startswith("csls|")
            for rec in [ctx.cache.get(k)]]
    info = {"commit": s1.git("rev-parse", "HEAD").strip(),
            "eval_commits": s1.eval_commits(log.read_text(encoding="utf-8")) if log.exists()
            else [], "config": XCFG, "eval": s1.EVAL,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "n_records": sum(1 for _ in ctx.cache),
            "seconds_per_csls": float(np.mean(secs)) if secs else None,
            "inputs": file_fingerprints({str(f.relative_to(ctx.rdir.parent)): f for f in
                                         [*sorted(ctx.rdir.rglob("*.*")),
                                          *sorted((ctx.rdir.parent / "dicts").glob("*.parquet")),
                                          *(ctx.rdir.parent / n for n in
                                            ("folds.parquet", "words.parquet", "gold.parquet"))]})}
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(f"results written to {RESULTS}")


def data_files(ctx_pairs: pd.DataFrame) -> list[Path]:
    """Inputs beyond stage 1's (run.py --data-files there): every node's vectors, the route
    dictionaries and URIEL, relative to the data root."""
    p6, root = s1.CFG.path("phase6"), s1.CFG.data_root
    langs = pd.read_csv(p6 / "routes" / "uriel.csv")
    langs = sorted(set(langs["a"]) | set(langs["b"]))
    files = [s1.CFG.path("v0_embeddings") / f"{lang}.npy" for lang in langs]
    files += sorted(f for f in (p6 / "routes").rglob("*") if f.is_file())
    return [f.relative_to(root) for f in files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs")
    ap.add_argument("--data-files", action="store_true", help="list inputs and exit")
    args = ap.parse_args()
    if args.data_files:
        for f in data_files(None):
            print(f.as_posix())
        return
    check_clean(s1.git("status", "--porcelain").splitlines())
    ctx = Ctx(args.limit)
    plans = run_all(ctx)
    write_results(ctx, plans)


if __name__ == "__main__":
    main()
