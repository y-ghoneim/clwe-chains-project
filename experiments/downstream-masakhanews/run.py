"""News topic transfer into yo, sw, am, so (MasakhaNEWS). See README.md (approved before building).

Main set: for every pair (X, Y) of stage mn, X in {yo, sw, am, so} and Y in {en, fr}, X's test
articles are moved into Y's space by the direct map or the chain through the pair's hub (Procrustes
on the full natural dictionaries of stage mn) and classified by Y's classifiers. Reverse set
(secondary, reported apart): Y's test articles into X's space, X's classifiers. The label sets
differ across languages, so every direction keeps only the topics both languages have
(decisions.md 2026-11-20): the classifier is trained on them, the test language is scored on its
articles of them, and the ceiling (the test language's own classifier) and the floor (its majority
topic) use the same topics. Needs prep/16 and prep/07, prep/08 `--stage mn`; the BLI companion's
results (experiments/downstream-masakhanews-bli) are joined if present.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from clwe_chains import load_config
from clwe_chains.align import compose, fit
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.runs import file_fingerprints, result_affecting
from clwe_chains.stats import paired_bootstrap
from clwe_chains.transfer import (
    LR_GRID,
    MLP_BATCH,
    MLP_HIDDEN,
    MLP_MAX_EPOCHS,
    MLP_PATIENCE,
    chain_ratio,
    edge_indices,
    fit_classifiers,
    load_centred,
    macro_f1,
    majority_floor,
    map_features,
    paired_f1_bootstrap,
    predict_kinds,
    spearman_by_group,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CFG = load_config()
MC = CFG["downstream"]["masakhanews"]
X_LANGS, Y_LANGS = MC["languages"], MC["labelled"]
P6 = CFG.path("phase6") / "mn"
FEAT = CFG.path("downstream") / "masakhanews"
BLI_RESULTS = HERE.parent / "downstream-masakhanews-bli" / "results"
RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/downstream-masakhanews/",
                "experiments/downstream-masakhanews-bli/pairs_spec.csv",
                "experiments/downstream-masakhanews-bli/results/")   # the joined BLI scores
OWN_RESULTS = "experiments/downstream-masakhanews/results/"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
N_MLP_SEEDS, N_BOOT = XCFG["n_mlp_seeds"], XCFG["n_boot"]
KINDS = ("lr", "mlp")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def edge_map(store: EmbeddingStore, a: str, b: str) -> torch.Tensor:
    ia, ib = edge_indices(P6 / "dicts", a, b)
    dev = store[a].device
    return fit(store[a][torch.as_tensor(ia, device=dev)], store[b][torch.as_tensor(ib, device=dev)],
               "procrustes")


def directions(pair) -> list[tuple[str, str, str]]:
    """(set, source, target) of a pair: main = the African language into en or fr."""
    return [("main", pair.s, pair.t), ("reverse", pair.t, pair.s)]


def shared_topics(data: dict, a: str, b: str) -> tuple[str, ...]:
    """The topics both languages have in their training splits, sorted."""
    return tuple(sorted(set(data[a]["labels_train"]) & set(data[b]["labels_train"])))


def restrict(d: dict, topics: tuple[str, ...]) -> dict:
    """A language's splits with only the articles of these topics, labels as 0..k-1 in their
    order (features keep the centring on the language's full training mean)."""
    ids = {t: i for i, t in enumerate(topics)}
    out = {}
    for split in ("train", "validation", "test"):
        keep = np.isin(d[f"labels_{split}"], topics)
        out[f"X_{split}"] = d[f"X_{split}"][keep]
        out[f"y_{split}"] = np.array([ids[t] for t in d[f"labels_{split}"][keep]], dtype=np.int64)
    return out


class Models:
    """Classifiers per (language, topics), trained once."""

    def __init__(self, data: dict):
        self.data, self._cache, self.chosen_c = data, {}, {}

    def __call__(self, lang: str, topics: tuple[str, ...]) -> dict:
        if (lang, topics) not in self._cache:
            m = fit_classifiers(restrict(self.data[lang], topics), N_MLP_SEEDS, len(topics))
            self._cache[lang, topics] = m
            self.chosen_c[f"{lang} {'/'.join(topics)}"] = m["C"]
            print(f"trained {lang} on {len(topics)} topics: C = {m['C']}", flush=True)
        return self._cache[lang, topics]


def run_directions(pairs: pd.DataFrame, data: dict, models: Models) -> tuple[pd.DataFrame, dict,
                                                                             pd.DataFrame]:
    store = EmbeddingStore(CFG, max_cached=4)
    rows, preds, ref = [], {}, []
    for pair in pairs.itertuples():
        for side, src, tgt in directions(pair):
            topics = shared_topics(data, src, tgt)
            k = len(topics)
            test = restrict(data[src], topics)
            x_test, y_test = test["X_test"], test["y_test"]
            for kind, p in predict_kinds(models(src, topics), x_test, KINDS).items():
                ref.append({"set": side, "src": src, "tgt": tgt, "model": kind, "n_topics": k,
                            "topics": "/".join(topics), "n_test": len(y_test),
                            "ceiling_macro_f1": float(np.mean([macro_f1(y_test, q, k)
                                                               for q in p])),
                            "floor_macro_f1": majority_floor(y_test, k)})
            maps = {"direct": edge_map(store, src, tgt),
                    "chain": compose([edge_map(store, src, pair.hub),
                                      edge_map(store, pair.hub, tgt)]),
                    "nomap": None}
            for method, w in maps.items():
                feats = x_test if w is None else map_features(x_test, w.cpu().numpy())
                for kind, p in predict_kinds(models(tgt, topics), feats, KINDS).items():
                    preds[src, tgt, method, kind] = p
                    rows.append({"set": side, "src": src, "tgt": tgt, "hub": pair.hub,
                                 "method": method, "model": kind, "n_topics": k,
                                 "macro_f1": float(np.mean([macro_f1(y_test, q, k) for q in p])),
                                 "accuracy": float(np.mean([(q == y_test).mean() for q in p]))})
            preds[src, tgt, "y"], preds[src, tgt, "k"] = y_test, np.array(k)
            print(f"{side}: {src}->{tgt} via {pair.hub} on {k} topics done", flush=True)
    return pd.DataFrame(rows), preds, pd.DataFrame(ref)


def delta_table(pairs: pd.DataFrame, preds: dict, sizes: pd.Series) -> pd.DataFrame:
    """Per direction and model: chain minus direct macro-F1 with a paired bootstrap over test
    articles, the direction's r and, if the BLI companion has run, its BLI scores."""
    bli = None
    if (BLI_RESULTS / "pooled.csv").exists():
        b = pd.read_csv(BLI_RESULTS / "pooled.csv")
        bli = b[b["method"] == "chain"].set_index(["src", "tgt"])
    rows = []
    for i, pair in enumerate(pairs.itertuples()):
        for side, src, tgt in directions(pair):
            y = preds[src, tgt, "y"]
            k = int(preds[src, tgt, "k"])
            for kind in KINDS:
                d, lo, hi = paired_f1_bootstrap(y, preds[src, tgt, "chain", kind],
                                                preds[src, tgt, "direct", kind], N_BOOT, seed=0,
                                                n_classes=k)
                row = {"set": side, "pair": i, "african": pair.s, "src": src, "tgt": tgt,
                       "hub": pair.hub, "model": kind, "delta_f1": d, "lo": lo, "hi": hi,
                       "r": chain_ratio(sizes, src, pair.hub, tgt)}
                if bli is not None and (src, tgt) in bli.index:
                    rec = bli.loc[src, tgt]
                    row.update(mrr_direct=float(rec["mrr_thin"]), mrr_chain=float(rec["mrr"]),
                               delta_mrr=float(rec["delta"]), mrr_lo=float(rec["lo"]),
                               mrr_hi=float(rec["hi"]))
                rows.append(row)
    return pd.DataFrame(rows)


def summary_table(delta: pd.DataFrame) -> pd.DataFrame:
    """Per set and model: each African language's directions and all 8, mean chain advantage with
    a bootstrap over directions, and the directions above / below 0."""
    rows = []
    for (side, kind), d in delta.groupby(["set", "model"], sort=False):
        for name, g in [("all", d)] + list(d.groupby("african", sort=False)):
            v = g["delta_f1"].to_numpy()
            mean, lo, hi = paired_bootstrap(v, np.zeros_like(v), n=N_BOOT, seed=0)
            rows.append({"set": side, "model": kind, "group": name, "n_directions": len(g),
                         "mean_delta_f1": mean, "lo": lo, "hi": hi,
                         "above": int((g["lo"] > 0).sum()), "below": int((g["hi"] < 0).sum())})
    return pd.DataFrame(rows)


def correlation_table(delta: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    """Main set only, descriptive (8 directions): Spearman of the advantage against r and against
    the MRR advantage, and F1 against MRR over directions x {direct, chain}."""
    rows = []
    for kind in KINDS:
        d = delta[(delta["set"] == "main") & (delta["model"] == kind)].reset_index(drop=True)
        rho, lo, hi = spearman_by_group(d["r"], d["delta_f1"], d["pair"], N_BOOT, 0)
        rows.append({"model": kind, "what": "advantage in F1 vs r", "n": len(d), "rho": rho,
                     "lo": lo, "hi": hi})
        if "delta_mrr" not in d:
            continue
        d = d.dropna(subset=["delta_mrr"]).reset_index(drop=True)
        rho, lo, hi = spearman_by_group(d["delta_mrr"], d["delta_f1"], d["pair"], N_BOOT, 0)
        rows.append({"model": kind, "what": "advantage in F1 vs advantage in MRR", "n": len(d),
                     "rho": rho, "lo": lo, "hi": hi})
        t = transfer[(transfer["set"] == "main") & (transfer["model"] == kind)
                     & transfer["method"].isin(["direct", "chain"])]
        m = t.merge(d[["src", "tgt", "pair", "mrr_direct", "mrr_chain"]], on=["src", "tgt"])
        mrr = np.where(m["method"] == "direct", m["mrr_direct"], m["mrr_chain"])
        rho, lo, hi = spearman_by_group(mrr, m["macro_f1"], m["pair"], N_BOOT, 0)
        rows.append({"model": kind, "what": "F1 vs MRR, directions x {direct, chain}",
                     "n": len(m), "rho": rho, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke run on the first N pairs; writes results/smoke/ and skips the "
                         "clean-tree check")
    args = ap.parse_args()
    status = git("status", "--porcelain").splitlines()
    if args.limit is None and (dirty := result_affecting(status, RESULT_PATHS, OWN_RESULTS)):
        sys.exit("uncommitted changes that could affect results:\n" + "\n".join(dirty))
    out = HERE / "results" / ("smoke" if args.limit else "")
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(P6 / "pairs.csv")
    if args.limit:
        pairs = pairs.head(args.limit)
    sizes = pd.read_csv(P6 / "dict_sizes.csv").set_index("edge")["pairs"]
    data = {lang: load_centred(FEAT / f"{lang}.npz") for lang in X_LANGS + Y_LANGS}
    models = Models(data)
    transfer, preds, reference = run_directions(pairs, data, models)
    delta = delta_table(pairs, preds, sizes)
    kw = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    transfer.to_csv(out / "transfer.csv", **kw)
    reference.to_csv(out / "reference.csv", **kw)
    delta.to_csv(out / "delta.csv", **kw)
    summary_table(delta).to_csv(out / "summary.csv", **kw)
    correlation_table(delta, transfer).to_csv(out / "correlations.csv", **kw)
    np.savez_compressed(out / "predictions.npz",
                        **{"|".join(k): v for k, v in preds.items()})
    info = {"commit": git("rev-parse", "HEAD").strip(), "smoke": bool(args.limit), "config": XCFG,
            "african": X_LANGS, "labelled": Y_LANGS, "pairs": len(pairs), "n_boot": N_BOOT,
            "classifiers": {"lr_grid": LR_GRID, "lr_C_chosen": models.chosen_c,
                            "mlp": {"hidden": MLP_HIDDEN, "seeds": N_MLP_SEEDS,
                                    "max_epochs": MLP_MAX_EPOCHS, "patience": MLP_PATIENCE,
                                    "batch": MLP_BATCH}},
            "texts_without_a_matched_token": pd.read_csv(FEAT / "stats.csv").query(
                "no_token > 0")[["lang", "split", "no_token"]].to_dict("records"),
            "bli_companion_joined": (BLI_RESULTS / "pooled.csv").exists(),
            "inputs": file_fingerprints(
                {**{f"features/{lang}": FEAT / f"{lang}.npz" for lang in X_LANGS + Y_LANGS},
                 "masakhanews/stats.csv": FEAT / "stats.csv",
                 "pairs.csv": P6 / "pairs.csv", "dict_sizes.csv": P6 / "dict_sizes.csv",
                 "bli_companion/pooled.csv": BLI_RESULTS / "pooled.csv",
                 "bli_companion/run_info.json": BLI_RESULTS / "run_info.json",
                 **{f"dicts/{f.name}": f for f in sorted((P6 / "dicts").glob("*.parquet"))}})}
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
