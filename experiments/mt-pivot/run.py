"""MT pivot experiment: translate test words directly or through chains of languages with NLLB-200.

For each pair direction (src -> tgt) of a phase 6 stage (D = phase 4's 40 pairs by default) and a
sample of its test words, routes are translated hop by hop: each hop translates the previous hop's
top output, cleaned of phrase padding ("The fox ." -> "fox"); the final hop keeps the top
`n_return` candidates. The first `select_words` words (selection) get every route: direct, every
one-hop route through another graph node, and the named routes; the rest (evaluation) get only
the named routes (direct, via en, via the stage's hub, phase 4's split-0 routes of every measure,
reversed for the reverse direction) and the selection's best one-hop route ("heldout_best").
Every hop is cached by (model, from, to, input), so a hop shared by several routes is translated
once and a rerun resumes. Scored against the stage's gold strings (prep/10): lenient P@1
(headline; the cleaned output or any of its words is a gold answer), P@5, MRR, and strict P@1
(whole output or first word). Results are written after every pair.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from clwe_chains import load_config
from clwe_chains.llm import MAX_FAILED, BatchTranslator, cache_tag
from clwe_chains.mt import (  # noqa: F401  (re-exported for tests/test_mt_pivot.py)
    _PUNCT,
    CODES,
    NLLB,
    NODES,
    HopCache,
    clean_hop,
    distinct,
    norm_answer,
    rank,
    rank_lenient,
    run_routes,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# Routes (tests/test_mt_pivot.py)

def routes_for(src: str, tgt: str, nodes: list[str], fixed: list[tuple[str, ...]]
               ) -> list[tuple[str, ...]]:
    """Direct, every one-hop route through another node, and the given routes (src first),
    without duplicates, in that order."""
    out = [(src, tgt)] + [(src, x, tgt) for x in nodes if x not in (src, tgt)]
    return list(dict.fromkeys(out + [tuple(r) for r in fixed]))


def covered(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The pairs whose two languages the MT system has (NLLB lacks mhr, myv, nah)."""
    return [(s, t) for s, t in pairs if s in CODES and t in CODES]


def phase4_routes(path: Path, split: int = 0) -> dict[tuple[str, str], dict[str, tuple]]:
    """(s, t) -> {measure: route} from phase 4's routes.csv (one split)."""
    r = pd.read_csv(path)
    r = r[r["split"] == split]
    out: dict[tuple[str, str], dict[str, tuple]] = {}
    for s, t, m, route in zip(r["s"], r["t"], r["measure"], r["route"], strict=True):
        out.setdefault((s, t), {})[m] = tuple(route.split("-"))
    return out


# ---------------------------------------------------------------------------------------------
# Run

def score(final: dict, gold: dict, split: str) -> pd.DataFrame:
    """Per job: strict and lenient rank. final: (src, tgt, route, concept) -> candidates."""
    rows = []
    for (src, tgt, route, c), cands in final.items():
        rows.append({"src": src, "tgt": tgt, "split": split, "route": route,
                     "hops": route.count("-") - 1, "concept_id": c,
                     "rank": rank_lenient(cands, gold[src, tgt, c], tgt),
                     "rank_strict": rank(cands, gold[src, tgt, c], tgt)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (src, tgt, split, route): lenient P@1 (headline), P@5, MRR, and strict P@1."""
    df = df.assign(p1=df["rank"] == 1, p5=df["rank"] > 0, p1_strict=df["rank_strict"] == 1,
                   mrr=np.where(df["rank"] > 0, 1.0 / df["rank"].clip(lower=1), 0.0))
    return df.groupby(["src", "tgt", "split", "route", "hops"])[
        ["p1", "p5", "mrr", "p1_strict"]].mean().reset_index()


def best_route(sel: pd.DataFrame) -> str | None:
    """The one-hop route with the highest lenient P@1 on the selection words (first on ties)."""
    one = sel[sel["hops"] == 1]
    return one.loc[one["p1"].idxmax(), "route"] if len(one) else None


def named_routes(src: str, tgt: str, hub: str | None, fixed: dict, basic: bool) -> dict:
    """The routes every evaluation word gets: direct, via en, via the stage's hub, and (unless
    basic) phase 4's routes for the pair (already oriented src -> tgt)."""
    named = {"direct": (src, tgt)}
    if "en" not in (src, tgt):
        named["via_en"] = (src, "en", tgt)
    if hub and hub not in (src, tgt):
        named["via_hub"] = (src, hub, tgt)
    if not basic:
        for meas, r in fixed.items():
            named[f"p4_{meas}"] = tuple(r)
    return named


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=XCFG["model"])
    ap.add_argument("--run", default="pilot2", help="a key of config.yaml runs")
    ap.add_argument("--system", default="nllb", choices=["nllb", "llm"],
                    help="llm: Claude through the batch API (configs machine_dicts.llm)")
    ap.add_argument("--dry-run", action="store_true", help="llm: count requests, call nothing")
    args = ap.parse_args()
    cfg, rc = load_config(), XCFG["runs"][args.run]
    basic = rc.get("routes", "all") == "basic"
    p6 = cfg.path("phase6") / rc["data"]
    items = pd.read_parquet(p6 / "mt_items.parquet")
    pairs = [tuple(p.split("-")) for p in rc["pairs"]] if rc.get("pairs") else \
        sorted(set(zip(items["s"], items["t"], strict=True)))
    pairs = covered(pairs)
    hubs = {(s, t): h for s, t, h in pd.read_csv(p6 / "pairs.csv")[["s", "t", "hub"]].itertuples(
        index=False)}
    p4 = phase4_routes(REPO / "experiments" / "graph-all-pairs" / "results" / "routes.csv")
    gold = {(a, b, int(c)): {norm_answer(g, b) for g in gs} for a, b, c, gs in
            zip(items["src"], items["tgt"], items["concept_id"], items["gold"], strict=True)}
    if args.system == "llm":
        if not basic:
            raise SystemExit("the llm runs use basic routes (config.yaml routes: basic)")
        lc = cfg["machine_dicts"]["llm"]
        n = XCFG["llm_candidates"]
        model_tag, label = cache_tag(lc["model"], f"top{n}-v1"), f"{lc['model']} (top {n})"
        cache = HopCache(cfg.path("results") / "mt-pivot" / "hops_llm.jsonl")
        out_dir = cfg.path("results") / "mt-pivot" / f"{args.run}-llm"
    else:
        model_tag, label = args.model.split("/")[-1], args.model
        cache = HopCache(cfg.path("results") / "mt-pivot" / "hops.jsonl")
        out_dir = cfg.path("results") / "mt-pivot" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    translate = None
    missing: dict[str, int] = {}

    def lazy(words, a, b):
        nonlocal translate
        if args.system == "llm":    # translated up front; what failed there counts as no answer
            missing[f"{a}>{b}"] = missing.get(f"{a}>{b}", 0) + len(words)
            return [None] * len(words)
        if translate is None:
            translate = NLLB(args.model, XCFG["beams"], XCFG["n_return"], XCFG["max_new_tokens"],
                             XCFG["batch"])
        return translate(words, a, b)

    def jobs_for(src, tgt, routes, sub):
        return [((src, tgt, "-".join(r), int(c)), r, w) for r in routes
                for c, w in zip(sub["concept_id"], sub["src_word"], strict=True)]

    def plan(s: str, t: str, src: str, tgt: str):
        sub = items[(items["src"] == src) & (items["tgt"] == tgt)]
        sub = sub.sample(n=min(rc["words"], len(sub)), random_state=rc["seed"])
        fixed = {m: r if src == s else r[::-1] for m, r in p4.get((s, t), {}).items()}
        named = named_routes(src, tgt, hubs.get((s, t)), fixed, basic)
        return sub.iloc[:rc["select_words"]], sub.iloc[rc["select_words"]:], named

    llm = None
    if args.system == "llm":                           # one batch per depth for every pair
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        llm = BatchTranslator(cfg["machine_dicts"]["llm"], n=XCFG["llm_candidates"],
                              dry_run=args.dry_run, state_dir=cache.path.parent,
                              words_per_request=XCFG["llm_words_per_request"])
        jobs = []
        for s, t in pairs:
            for src, tgt in ((s, t), (t, s)):
                sel_w, ev_w, named = plan(s, t, src, tgt)
                routes = list(dict.fromkeys(named.values()))
                jobs += jobs_for(src, tgt, routes, sel_w) + jobs_for(src, tgt, routes, ev_w)
        print(f"{len(jobs)} route jobs for {model_tag}", flush=True)
        run_routes(jobs, lazy, cache, model_tag, translate_many=llm)
        failed, total = llm.failed_share()
        if failed > max(MAX_FAILED * total, 50):
            raise SystemExit(f"{failed} of {total} translations failed (not cached): rerun to "
                             "retry them before scoring")

    # Per pair: the selection words get every route (basic: the named routes only); the
    # evaluation words get the named routes and the selection's best one-hop route. Results are
    # written after every pair; the cache makes a rerun resume.
    t0, scored, labels, ranks = time.time(), [], [], []
    for k, (s, t) in enumerate(pairs, 1):
        for src, tgt in ((s, t), (t, s)):
            sel_w, ev_w, named = plan(s, t, src, tgt)
            every = list(named.values()) if basic else \
                routes_for(src, tgt, NODES, list(named.values()))
            sel_r = score(run_routes(jobs_for(src, tgt, every, sel_w), lazy, cache, model_tag),
                          gold, "select")
            sel = summarize(sel_r)
            best = None if basic else best_route(sel)
            if best:
                named["heldout_best"] = tuple(best.split("-"))
            ev_routes = list(dict.fromkeys(named.values()))
            ev_r = score(run_routes(jobs_for(src, tgt, ev_routes, ev_w), lazy, cache, model_tag),
                         gold, "eval") if len(ev_w) else sel_r[:0]
            ev = summarize(ev_r) if len(ev_w) else sel[:0]
            ranks += [sel_r, ev_r]
            scored += [sel, ev]
            for name, r in named.items():
                for split, m in (("eval", ev), ("select", sel)):
                    row = m[m["route"] == "-".join(r)]
                    if len(row):
                        labels.append({"src": src, "tgt": tgt, "chooser": name, "split": split,
                                       **row.iloc[0][["route", "p1", "p5", "mrr",
                                                      "p1_strict"]].to_dict()})
        pd.concat(scored).to_csv(out_dir / "metrics.csv", index=False, lineterminator="\n",
                                 float_format="%.4f")
        pd.DataFrame(labels).to_csv(out_dir / "choosers.csv", index=False, lineterminator="\n",
                                    float_format="%.4f")
        pd.concat(ranks).to_csv(out_dir / "ranks.csv", index=False, lineterminator="\n")
        done = translate.done if translate else 0
        print(f"{time.strftime('%H:%M:%S')} pair {k}/{len(pairs)} {s}-{t} done; {done} new "
              f"translations, {(time.time() - t0) / 60:.1f} min", flush=True)

        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                check=False, text=True).stdout.strip()
        seconds = time.time() - t0
        info = {"commit": commit, "model": label, "system": args.system, "run": args.run,
                "config": XCFG, "pairs_done": k, "pairs": len(pairs), "new_translations": done,
                "seconds": round(seconds, 1),
                "words_per_second": round(done / max(seconds, 1e-9), 1)}
        if llm is not None:            # usage of this process only; every batch is in usage.jsonl
            info |= {"llm_config": cfg["machine_dicts"]["llm"], "llm_missing": missing,
                     "llm_requests": llm.requests, "llm_tokens": llm.tokens,
                     "llm_cost_usd": round(llm.cost(cfg["machine_dicts"]["llm"]
                                                    ["price_per_mtok"]), 2)}
        (out_dir / "run_info.json").write_text(json.dumps(info, indent=1) + "\n",
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
