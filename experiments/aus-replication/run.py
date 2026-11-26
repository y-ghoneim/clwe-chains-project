"""Phase 3: AUS replication on the shared list (en→ar via es/ga/mt). See README.md.

Three parts, all over direct + AUS's 15 chains:
  aus_exact : AUS's pipeline as published (scheme, raw vectors, train = test, their metrics);
  faithful  : per-step least squares, raw vectors, NN among the list words, 5 splits;
  proper    : per-step Procrustes, normalized vectors, CSLS over the full vocabulary, 5 splits,
              plus the robustness rerun on the high-agreement subset.
Writes results/: aus_exact.csv, metrics.csv, bootstrap.csv, summary.csv, aus_rule.csv,
run_info.json.
"""

from __future__ import annotations

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

from clwe_chains import load_config
from clwe_chains.align import compose, fit
from clwe_chains.aus_metrics import (
    aus_scheme_map,
    frobenius_error,
    hausdorff,
    neighbourhood_overlap,
    precision_at_k,
)
from clwe_chains.aus_rule import feature_vector, path_cost
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.evaluate import bli, metrics
from clwe_chains.gold import gold_sets
from clwe_chains.routes import aus_chains
from clwe_chains.sources import wiktionary_candidates
from clwe_chains.stats import paired_bootstrap
from clwe_chains.text import load_vocab, vocab_index

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
# Paths whose changes can alter results; anything else (docs, the AUS PDF) can't.
RESULT_PATHS = ("src/", "prep/", "configs/", "tests/", "experiments/aus-replication/")


def assert_disjoint(train_ids, test_ids) -> None:
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError(f"{len(overlap)} test concepts among the training rows")


def test_concepts(test_ids, identical: pd.Series) -> tuple[list[int], int]:
    """Test concepts minus those whose target word equals the English string."""
    kept = [c for c in test_ids if not bool(identical.get(c, False))]
    return kept, len(test_ids) - len(kept)


def subset_split(split: dict, subset_ids: set) -> dict:
    return {part: [c for c in split[part] if c in subset_ids] for part in ("train", "test")}


def result_affecting(status_lines: list[str]) -> list[str]:
    """`git status --porcelain` lines that could change results (not this run's results/)."""
    out = []
    for line in status_lines:
        path = line[3:].strip().strip('"')
        if path.startswith("experiments/aus-replication/results/"):
            continue
        if path.startswith(RESULT_PATHS):
            out.append(line)
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def rows(lang: str, ids, lst: Path, concepts: pd.DataFrame) -> np.ndarray:
    """Vocab rows of each concept's word in lang (English: en_rank)."""
    if lang == "en":
        table = concepts.set_index("concept_id")["en_rank"]
    else:
        table = pd.read_parquet(lst / "per_lang" / f"{lang}.parquet").set_index(
            "concept_id")["word_idx"]
    return table.loc[list(ids)].to_numpy(dtype=np.int64)


def aus_exact(raw, langs, ids, lst, concepts, routes, ecfg) -> pd.DataFrame:
    xs = {lang: raw[lang][torch.as_tensor(rows(lang, ids, lst, concepts),
                                          device=raw[lang].device)].double() for lang in langs}
    direct = aus_scheme_map([xs["en"], xs["ar"]])
    out = []
    for route in routes:
        w = aus_scheme_map([xs[lang] for lang in route])
        u, v = xs["en"] @ w, xs["ar"]
        rec = {"route": "-".join(route), "error": frobenius_error(u, v),
               "hausdorff": hausdorff(u, v),
               "overlap": neighbourhood_overlap(u, v, ecfg["overlap_k"]),
               "max_abs_diff_vs_direct": float((w - direct).abs().max())}
        for metric in ("cosine", "euclidean"):
            for k, p in precision_at_k(u, v, tuple(ecfg["ks"]), metric).items():
                rec[f"p{k}_{metric}"] = p
        out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    dirty = result_affecting(git("status", "--porcelain").splitlines())
    if dirty:
        sys.exit("uncommitted changes that can affect results:\n" + "\n".join(dirty))
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    xcfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    cfg = load_config()
    lst = cfg.path("list")
    src, tgt = xcfg["pair"]
    routes = [[src, tgt], *aus_chains(src, tgt, xcfg["mids"])]
    langs = sorted({lang for r in routes for lang in r})

    concepts = pd.read_parquet(lst / "candidates.parquet")
    ids = pd.read_parquet(lst / "shared_list.parquet", columns=["concept_id"])["concept_id"]
    ids = ids.tolist()
    concepts = concepts[concepts["concept_id"].isin(ids)]
    splits = json.loads((lst / "splits.json").read_text(encoding="utf-8"))["splits"]
    sense = pd.read_parquet(lst / "sense.parquet")
    rule = cfg["list"]["sense_subset"]
    robust = set(sense[(sense["coherence"] >= rule["min_coherence"])
                       & (sense["n_evaluable"] >= rule["min_evaluable"])]["concept_id"])
    tgt_list = pd.read_parquet(lst / "per_lang" / f"{tgt}.parquet").set_index("concept_id")
    identical = tgt_list["identical"]

    # Gold answers for the proper setting: list word + PanLex + Wiktionary translations.
    vocab_t = load_vocab(cfg.path("v0_embeddings") / f"{tgt}.vocab.txt")
    wk = pd.read_parquet(lst / "wiktionary_en.parquet", columns=["en", "pos", "sense", "code",
                                                                 "word"])
    wk_pairs, _ = wiktionary_candidates(wk, cfg["list"]["wikt_codes"].get(tgt, [tgt]), concepts,
                                        vocab_index(vocab_t, tgt), tgt)
    gold_full = gold_sets(tgt_list.loc[ids, "word_idx"],
                          concepts.set_index("concept_id").loc[ids, "en_key"],
                          pd.read_parquet(cfg.path("v0_panlex") / "en.parquet"),
                          pd.read_parquet(cfg.path("v0_panlex") / f"{tgt}.parquet"), wk_pairs)

    raw = EmbeddingStore(cfg, steps=[])
    norm = EmbeddingStore(cfg)
    RESULTS.mkdir(exist_ok=True)

    print("aus_exact ...", flush=True)
    aus_exact(raw, langs, ids, lst, concepts, routes, xcfg["aus_exact"]).to_csv(
        RESULTS / "aus_exact.csv", index=False, lineterminator="\n")

    ev = cfg["eval"]
    cand_t = rows(tgt, ids, lst, concepts)
    metric_rows, boot_rows, n_identical = [], [], 0
    jobs = [("faithful", "full"), ("proper", "full"), ("proper", "robust")]
    for setting, subset in jobs:
        s = xcfg["settings"][setting]
        store = raw if s["vectors"] == "raw" else norm
        for seed, split in sorted(splits.items(), key=lambda kv: int(kv[0])):
            if subset == "robust":
                split = subset_split(split, robust)
            test_ids, n_excl = test_concepts(split["test"], identical)
            n_identical = max(n_identical, n_excl)
            assert_disjoint(split["train"], test_ids)
            step_maps = {}
            for route in routes:
                for a, b in itertools.pairwise(route):
                    if (a, b) not in step_maps:
                        xa = store[a][torch.as_tensor(rows(a, split["train"], lst, concepts),
                                                      device=store[a].device)]
                        xb = store[b][torch.as_tensor(rows(b, split["train"], lst, concepts),
                                                      device=store[b].device)]
                        step_maps[(a, b)] = fit(xa, xb, s["solver"])
            evals = [("test", test_ids)]
            if s.get("train_eq_test"):
                evals.append(("train", test_concepts(split["train"], identical)[0]))
            direct_rr = {}
            for route in routes:
                w = compose([step_maps[(a, b)] for a, b in itertools.pairwise(route)])
                for part, q_ids in evals:
                    queries = rows(src, q_ids, lst, concepts)
                    if s["gold"] == "list":
                        gold = [[int(tgt_list.loc[c, "word_idx"])] for c in q_ids]
                    else:
                        gold = [gold_full[c] for c in q_ids]
                    ranks = bli(store[src], store[tgt], w, queries, gold, k=ev["csls_k"],
                                block=ev["block_rows"], retrieval=s["retrieval"],
                                candidates=cand_t if s["retrieval"] == "nn" else None,
                                half=bool(ev["half"]) and s["retrieval"] == "csls")
                    m = metrics(ranks)
                    metric_rows.append({"setting": setting, "subset": subset, "split": int(seed),
                                        "route": "-".join(route), "eval": part, "n": m["n"],
                                        "p1": m["p1"], "p5": m["p5"],
                                        "p10": float((ranks <= 10).mean()), "mrr": m["mrr"]})
                    if part != "test":
                        continue
                    rr = 1.0 / ranks
                    if len(route) == 2:
                        direct_rr[seed] = rr
                    else:
                        d, lo, hi = paired_bootstrap(rr, direct_rr[seed], **xcfg["bootstrap"])
                        boot_rows.append({"setting": setting, "subset": subset,
                                          "split": int(seed), "route": "-".join(route),
                                          "d_mrr": d, "lo": lo, "hi": hi})
            print(setting, subset, "split", seed, "done", flush=True)

    met = pd.DataFrame(metric_rows)
    boot = pd.DataFrame(boot_rows)
    met.to_csv(RESULTS / "metrics.csv", index=False, lineterminator="\n")
    boot.to_csv(RESULTS / "bootstrap.csv", index=False, lineterminator="\n")
    test = met[met["eval"] == "test"]
    summ = test.groupby(["setting", "subset", "route"])[["p1", "p5", "p10", "mrr"]].agg(
        ["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    b = boot.groupby(["setting", "subset", "route"]).agg(
        d_mrr_mean=("d_mrr", "mean"), splits_above_0=("lo", lambda x: int((x > 0).sum())),
        splits_below_0=("hi", lambda x: int((x < 0).sum())))
    summ.join(b).reset_index().to_csv(RESULTS / "summary.csv", index=False,
                                      lineterminator="\n", float_format="%.4f")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lang2vec imports the deprecated pkg_resources
        import lang2vec.lang2vec as l2v
    codes = xcfg["lang2vec_codes"]
    rule_rows = [{"route": "-".join(r)} for r in routes]
    for fs in ("syntax_wals", "geo"):
        feats = l2v.get_features([codes[lang] for lang in langs], fs)
        vecs = {lang: feature_vector(feats[codes[lang]]) for lang in langs}
        for rec, r in zip(rule_rows, routes, strict=True):
            rec[f"{fs}_cost"] = path_cost(vecs, r)
    rule_df = pd.DataFrame(rule_rows)
    for fs in ("syntax_wals", "geo"):
        rule_df[f"{fs}_rank"] = rule_df[f"{fs}_cost"].rank(method="min").astype(int)
    rule_df.to_csv(RESULTS / "aus_rule.csv", index=False, lineterminator="\n",
                   float_format="%.4f")

    info = {"commit": git("rev-parse", "HEAD").strip(), "started": started,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": xcfg,
            "list_config": cfg["list"], "eval_config": ev,
            "n_concepts": len(ids), "n_robust": len(robust),
            "max_identical_excluded_per_split": n_identical,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print("done", info["commit"], flush=True)


if __name__ == "__main__":
    main()
