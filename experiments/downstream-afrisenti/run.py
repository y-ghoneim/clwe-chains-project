"""Cross-lingual sentiment transfer into Yoruba, Swahili and Amharic (AfriSenti). See README.md
(approved before building).

Main set: for every pair (X, Y) of stage af, X in {yo, sw, am} and Y a UMSAB language other than hi,
X's AfriSenti test tweets are moved into Y's space by the direct map or the chain through the pair's
hub (Procrustes on the full natural dictionaries of stage af) and classified by Y's UMSAB
classifiers. Reverse set (secondary, reported apart): Y's UMSAB test tweets moved into X's space and
classified by X's AfriSenti classifiers. Reference points: no map, the test language's own
classifier (ceiling) and its majority class (floor). Needs prep/13, prep/14 and prep/07, prep/08
`--stage af`; the BLI companion's results (experiments/downstream-afrisenti-bli) are joined if
present.
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
DC = CFG["downstream"]["afrisenti"]
X_LANGS, Y_LANGS = DC["languages"], DC["labelled"]
P6 = CFG.path("phase6") / "af"
FEAT = {**{x: CFG.path("downstream") / "afrisenti" / f"{x}.npz" for x in X_LANGS},
        **{y: CFG.path("downstream") / "umsab" / f"{y}.npz" for y in Y_LANGS}}
BLI_RESULTS = HERE.parent / "downstream-afrisenti-bli" / "results"
RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/downstream-afrisenti/",
                "experiments/downstream-afrisenti-bli/pairs_spec.csv",
                "experiments/downstream-afrisenti-bli/results/")   # the joined BLI scores
OWN_RESULTS = "experiments/downstream-afrisenti/results/"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
N_MLP_SEEDS, N_BOOT = XCFG["n_mlp_seeds"], XCFG["n_boot"]
KINDS = ("lr", "mlp")
SETS = ("main", "reverse")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def edge_map(store: EmbeddingStore, a: str, b: str) -> torch.Tensor:
    ia, ib = edge_indices(P6 / "dicts", a, b)
    dev = store[a].device
    return fit(store[a][torch.as_tensor(ia, device=dev)], store[b][torch.as_tensor(ib, device=dev)],
               "procrustes")


def directions(pair) -> list[tuple[str, str, str]]:
    """(set, source, target) of a pair: main = the AfriSenti language into the UMSAB language."""
    return [("main", pair.s, pair.t), ("reverse", pair.t, pair.s)]


def run_directions(pairs: pd.DataFrame, data: dict, models: dict) -> tuple[pd.DataFrame, dict]:
    store = EmbeddingStore(CFG, max_cached=4)
    rows, preds = [], {}
    for pair in pairs.itertuples():
        for side, src, tgt in directions(pair):
            x_test, y_test = data[src]["X_test"], data[src]["y_test"]
            maps = {"direct": edge_map(store, src, tgt),
                    "chain": compose([edge_map(store, src, pair.hub),
                                      edge_map(store, pair.hub, tgt)]),
                    "nomap": None}
            for method, w in maps.items():
                feats = x_test if w is None else map_features(x_test, w.cpu().numpy())
                for kind, p in predict_kinds(models[tgt], feats, KINDS).items():
                    preds[src, tgt, method, kind] = p
                    rows.append({"set": side, "src": src, "tgt": tgt, "hub": pair.hub,
                                 "method": method, "model": kind,
                                 "macro_f1": float(np.mean([macro_f1(y_test, q) for q in p])),
                                 "accuracy": float(np.mean([(q == y_test).mean() for q in p]))})
            print(f"{side}: {src}->{tgt} via {pair.hub} done", flush=True)
    return pd.DataFrame(rows), preds


def reference_table(data: dict, models: dict) -> pd.DataFrame:
    """Per language: its own classifier on its own test tweets (ceiling) and the majority class of
    its test labels (floor; not 1/6, AfriSenti's classes are unbalanced)."""
    rows = []
    for lang, d in data.items():
        for kind, p in predict_kinds(models[lang], d["X_test"], KINDS).items():
            rows.append({"lang": lang, "model": kind, "ceiling_macro_f1":
                         float(np.mean([macro_f1(d["y_test"], q) for q in p])),
                         "floor_macro_f1": majority_floor(d["y_test"]),
                         "C": models[lang]["C"] if kind == "lr" else None})
    return pd.DataFrame(rows)


def delta_table(pairs: pd.DataFrame, data: dict, preds: dict, sizes: pd.Series) -> pd.DataFrame:
    """Per direction and model: chain minus direct macro-F1 with a paired bootstrap over tweets,
    the direction's r and, if the BLI companion has run, its BLI scores."""
    bli = None
    if (BLI_RESULTS / "pooled.csv").exists():
        b = pd.read_csv(BLI_RESULTS / "pooled.csv")
        bli = b[b["method"] == "chain"].set_index(["src", "tgt"])
    rows = []
    for i, pair in enumerate(pairs.itertuples()):
        for side, src, tgt in directions(pair):
            for kind in KINDS:
                d, lo, hi = paired_f1_bootstrap(data[src]["y_test"], preds[src, tgt, "chain", kind],
                                                preds[src, tgt, "direct", kind], N_BOOT, seed=0)
                row = {"set": side, "pair": i, "afrisenti": pair.s, "src": src, "tgt": tgt,
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
    """Per set, model and AfriSenti language (never pooled over languages: README, 'Evaluation'):
    mean chain advantage over its directions with a bootstrap over directions, and the directions
    above / below 0. The tweet intervals cover test sampling only."""
    rows = []
    for (side, kind, lang), d in delta.groupby(["set", "model", "afrisenti"], sort=False):
        v = d["delta_f1"].to_numpy()
        mean, lo, hi = paired_bootstrap(v, np.zeros_like(v), n=N_BOOT, seed=0)
        rows.append({"set": side, "model": kind, "afrisenti": lang, "n_directions": len(d),
                     "mean_delta_f1": mean, "lo": lo, "hi": hi,
                     "above": int((d["lo"] > 0).sum()), "below": int((d["hi"] < 0).sum())})
    return pd.DataFrame(rows)


def correlation_table(delta: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    """Main set only (README, 'Evaluation' and 'BLI link'): Spearman over the 18 directions, with
    an interval that resamples pairs (one direction each in the main set)."""
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
    data = {lang: load_centred(path) for lang, path in FEAT.items()}
    models = {}
    for lang, d in data.items():
        models[lang] = fit_classifiers(d, N_MLP_SEEDS)
        print(f"trained {lang}: C = {models[lang]['C']}", flush=True)
    transfer, preds = run_directions(pairs, data, models)
    delta = delta_table(pairs, data, preds, sizes)
    kw = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    transfer.to_csv(out / "transfer.csv", **kw)
    reference_table(data, models).to_csv(out / "reference.csv", **kw)
    delta.to_csv(out / "delta.csv", **kw)
    summary_table(delta).to_csv(out / "summary.csv", **kw)
    correlation_table(delta, transfer).to_csv(out / "correlations.csv", **kw)
    np.savez_compressed(out / "predictions.npz",
                        **{"|".join(k): v for k, v in preds.items()})
    info = {"commit": git("rev-parse", "HEAD").strip(), "smoke": bool(args.limit), "config": XCFG,
            "afrisenti": X_LANGS, "labelled": Y_LANGS, "pairs": len(pairs), "n_boot": N_BOOT,
            "classifiers": {"lr_grid": LR_GRID,
                            "lr_C_chosen": {k: v["C"] for k, v in models.items()},
                            "mlp": {"hidden": MLP_HIDDEN, "seeds": N_MLP_SEEDS,
                                    "max_epochs": MLP_MAX_EPOCHS, "patience": MLP_PATIENCE,
                                    "batch": MLP_BATCH}},
            "tweets_without_a_matched_token":
                pd.read_csv(CFG.path("downstream") / "afrisenti" / "stats.csv")
                [["lang", "split", "no_token"]].to_dict("records"),
            "bli_companion_joined": (BLI_RESULTS / "pooled.csv").exists(),
            "inputs": file_fingerprints(
                {**{f"features/{lang}": path for lang, path in FEAT.items()},
                 "afrisenti/stats.csv": CFG.path("downstream") / "afrisenti" / "stats.csv",
                 "pairs.csv": P6 / "pairs.csv", "dict_sizes.csv": P6 / "dict_sizes.csv",
                 "bli_companion/pooled.csv": BLI_RESULTS / "pooled.csv",
                 "bli_companion/run_info.json": BLI_RESULTS / "run_info.json",
                 **{f"dicts/{f.name}": f for f in sorted((P6 / "dicts").glob("*.parquet"))}})}
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
