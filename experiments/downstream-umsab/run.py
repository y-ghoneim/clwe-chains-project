"""Cross-lingual sentiment transfer on UMSAB. See README.md (approved before building).

For every ordered pair (X, Y) of the 7 UMSAB node languages: a classifier trained on Y's tweets in
Y's space is applied to X's test tweets moved into Y's space by the direct map or the chain through
the pair's hub (Procrustes on the full natural dictionaries of stage u). Reference points: no map
and X's own classifier (the in-language ceiling). Needs prep/13_umsab.py and prep/07, prep/08
`--stage u`; the BLI companion's results (experiments/downstream-umsab-bli) are joined if present.
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
    center_features,
    fit_logreg,
    fit_mlp,
    language_mean,
    macro_f1,
    map_features,
    paired_f1_bootstrap,
    predict,
    spearman_by_group,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CFG = load_config()
LANGS = CFG["downstream"]["umsab"]["languages"]
P6 = CFG.path("phase6") / "u"
FEAT = CFG.path("downstream") / "umsab"
BLI_RESULTS = HERE.parent / "downstream-umsab-bli" / "results"
RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/downstream-umsab/",
                "experiments/downstream-umsab-bli/pairs_spec.csv")
OWN_RESULTS = "experiments/downstream-umsab/results/"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
N_MLP_SEEDS, N_BOOT = XCFG["n_mlp_seeds"], XCFG["n_boot"]
R_BINS = [-np.inf, -1, 0, 1, 2, np.inf]
R_LABELS = ["r<=-1", "-1<r<=0", "0<r<=1", "1<r<=2", "r>2"]
KINDS = ("lr", "mlp")
ROMANISED = "hi"   # UMSAB's Hindi is Latin script; our Hindi vectors and dictionaries are not
HINDI = "romanised_hindi"
METHODS = ("direct", "chain", "nomap")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def load_features() -> dict:
    """Each language's features centred on its own training-tweet mean (README, 'Features'): a
    language's tweets, mapped or not, are compared with the labelled language's in the same way."""
    out = {}
    for lang in LANGS:
        with np.load(FEAT / f"{lang}.npz") as z:
            d = {k: z[k] for k in z.files}
        mu = language_mean(d["X_train"], d["n_train"])
        for split in ("train", "validation", "test"):
            d[f"X_{split}"] = center_features(d[f"X_{split}"], mu, d[f"n_{split}"])
        out[lang] = d
    return out


def train_models(data: dict) -> dict:
    """Per language: logistic regression (C by validation) and N_MLP_SEEDS MLPs."""
    models = {}
    for lang, d in data.items():
        lr, c = fit_logreg(d["X_train"], d["y_train"], d["X_validation"], d["y_validation"])
        mlps = [fit_mlp(d["X_train"], d["y_train"], d["X_validation"], d["y_validation"], seed=s)
                for s in range(N_MLP_SEEDS)]
        models[lang] = {"lr": [lr], "mlp": mlps, "C": c}
        print(f"trained {lang}: C = {c}", flush=True)
    return models


def predictions(models_y: dict, x: np.ndarray) -> dict[str, np.ndarray]:
    """kind -> (models, tweets) predicted classes."""
    return {k: np.stack([predict(m, x) for m in models_y[k]]).astype(np.int8) for k in KINDS}


def edge_indices(a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    """Vocabulary rows (of a, of b) of the pair's full natural dictionary, whichever way round the
    prep stored it."""
    f = P6 / "dicts" / f"{a}__{b}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        return d["a"].to_numpy(), d["b"].to_numpy()
    d = pd.read_parquet(P6 / "dicts" / f"{b}__{a}.parquet")
    return d["b"].to_numpy(), d["a"].to_numpy()


def edge_map(store: EmbeddingStore, a: str, b: str) -> torch.Tensor:
    ia, ib = edge_indices(a, b)
    dev = store[a].device
    return fit(store[a][torch.as_tensor(ia, device=dev)], store[b][torch.as_tensor(ib, device=dev)],
               "procrustes")


def dictionary_size(sizes: pd.Series, a: str, b: str) -> int:
    return int(sizes[f"{a}__{b}"] if f"{a}__{b}" in sizes else sizes[f"{b}__{a}"])


def run_directions(pairs: pd.DataFrame, data: dict, models: dict) -> tuple[pd.DataFrame, dict]:
    store = EmbeddingStore(CFG, max_cached=len(LANGS))
    rows, preds = [], {}
    for pair in pairs.itertuples():
        for x, y in ((pair.s, pair.t), (pair.t, pair.s)):
            x_test, y_test = data[x]["X_test"], data[x]["y_test"]
            maps = {"direct": edge_map(store, x, y),
                    "chain": compose([edge_map(store, x, pair.hub), edge_map(store, pair.hub, y)]),
                    "nomap": None}
            for method, w in maps.items():
                feats = x_test if w is None else map_features(x_test, w.cpu().numpy())
                for kind, p in predictions(models[y], feats).items():
                    preds[x, y, method, kind] = p
                    rows.append({"x": x, "y": y, "hub": pair.hub, "method": method, "model": kind,
                                 "macro_f1": float(np.mean([macro_f1(y_test, q) for q in p])),
                                 "accuracy": float(np.mean([(q == y_test).mean() for q in p]))})
            print(f"{x}->{y} via {pair.hub} done", flush=True)
    return pd.DataFrame(rows), preds


def reference_table(data: dict, models: dict) -> pd.DataFrame:
    """In-language ceiling (own classifier, own test tweets) and the majority-class floor."""
    rows = []
    for lang, d in data.items():
        for kind, p in predictions(models[lang], d["X_test"]).items():
            rows.append({"lang": lang, "model": kind, "ceiling_macro_f1":
                         float(np.mean([macro_f1(d["y_test"], q) for q in p])),
                         "floor_macro_f1": 1 / 6, "C": models[lang]["C"] if kind == "lr" else None})
    return pd.DataFrame(rows)


def stratum_of(x: str, y: str) -> str:
    """Directions with Hindi form their own stratum (README, 'Data findings'); of the rest, distant
    = a pair with ar (the only non-Indo-European language), the others the same family."""
    if ROMANISED in (x, y):
        return HINDI
    return "distant" if "ar" in (x, y) else "same_family"


def delta_table(pairs: pd.DataFrame, data: dict, preds: dict, sizes: pd.Series) -> pd.DataFrame:
    """Per direction and model: chain minus direct macro-F1 with a paired bootstrap over tweets."""
    bli = None
    if (BLI_RESULTS / "pooled.csv").exists():
        b = pd.read_csv(BLI_RESULTS / "pooled.csv")
        bli = b[b["method"] == "chain"].set_index(["src", "tgt"])
    rows = []
    for i, pair in enumerate(pairs.itertuples()):
        r = float(np.log2(min(dictionary_size(sizes, pair.s, pair.hub),
                              dictionary_size(sizes, pair.hub, pair.t))
                          / dictionary_size(sizes, pair.s, pair.t)))
        for x, y in ((pair.s, pair.t), (pair.t, pair.s)):
            for kind in KINDS:
                d, lo, hi = paired_f1_bootstrap(data[x]["y_test"], preds[x, y, "chain", kind],
                                                preds[x, y, "direct", kind], N_BOOT, seed=0)
                row = {"pair": i, "x": x, "y": y, "hub": pair.hub, "model": kind, "delta_f1": d,
                       "lo": lo, "hi": hi, "r": r,
                       "stratum": stratum_of(x, y)}
                if bli is not None and (x, y) in bli.index:
                    rec = bli.loc[x, y]
                    row.update(mrr_direct=float(rec["mrr_thin"]), mrr_chain=float(rec["mrr"]),
                               delta_mrr=float(rec["delta"]))
                rows.append(row)
    return pd.DataFrame(rows)


def binned_table(delta: pd.DataFrame) -> pd.DataFrame:
    """Mean chain advantage in macro-F1 by r (README, 'Where the chain helps')."""
    d = delta.assign(r_bin=pd.cut(delta["r"], R_BINS, labels=R_LABELS))
    out = (d.groupby(["model", "r_bin"], observed=True)["delta_f1"]
           .agg(n="size", mean_delta_f1="mean").reset_index())
    return out


def standout_table(delta: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """The k directions with the largest and the k with the smallest chain advantage (logistic
    regression), largest first, with their r."""
    d = delta[delta["model"] == "lr"].sort_values("delta_f1", ascending=False)
    cols = ["model", "pair", "x", "y", "hub", "delta_f1", "lo", "hi", "r"]
    return pd.concat([d.head(k), d.tail(k)])[cols].reset_index(drop=True)


def summary_table(delta: pd.DataFrame) -> pd.DataFrame:
    """Per model and stratum: mean over pairs of the chain's advantage with a bootstrap interval,
    and the directions above / below 0 next to a nominal count (2.5% of the directions). The
    nominal count is a floor, not a null expectation: the intervals cover test-tweet sampling
    only, not the fit of the classifier or of the maps, and directions with the same source
    language share their test tweets. "all_but_hindi" is distant + same family; the romanised-Hindi
    directions are reported on their own and are in no other row."""
    rows = []
    for kind in KINDS:
        for stratum in ("all_but_hindi", "distant", "same_family", HINDI):
            d = delta[(delta["model"] == kind) & (
                (delta["stratum"] == stratum)
                | ((stratum == "all_but_hindi") & (delta["stratum"] != HINDI)))]
            if d.empty:
                continue
            per_pair = d.groupby("pair")["delta_f1"].mean().to_numpy()
            mean, lo, hi = paired_bootstrap(per_pair, np.zeros_like(per_pair), n=N_BOOT, seed=0)
            rows.append({"model": kind, "stratum": stratum, "n_pairs": len(per_pair),
                         "n_directions": len(d), "mean_delta_f1": mean, "lo": lo, "hi": hi,
                         "above": int((d["lo"] > 0).sum()), "below": int((d["hi"] < 0).sum()),
                         "nominal_above_floor": 0.025 * len(d)})
    return pd.DataFrame(rows)


def correlation_table(delta: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations with intervals that resample pairs (README, 'BLI link' and 'Where the
    chain helps'). BLI rows need the companion run's pooled.csv."""
    rows = []
    for kind in KINDS:
        d = delta[delta["model"] == kind].reset_index(drop=True)
        rho, lo, hi = spearman_by_group(d["r"], d["delta_f1"], d["pair"], N_BOOT, 0)
        rows.append({"model": kind, "what": "advantage in F1 vs r", "n": len(d), "rho": rho,
                     "lo": lo, "hi": hi})
        if "delta_mrr" not in d:
            continue
        d = d.dropna(subset=["delta_mrr"]).reset_index(drop=True)   # a partial BLI file
        rho, lo, hi = spearman_by_group(d["delta_mrr"], d["delta_f1"], d["pair"], N_BOOT, 0)
        rows.append({"model": kind, "what": "advantage in F1 vs advantage in MRR", "n": len(d),
                     "rho": rho, "lo": lo, "hi": hi})
        t = transfer[(transfer["model"] == kind) & transfer["method"].isin(["direct", "chain"])]
        key = d[["x", "y", "pair", "mrr_direct", "mrr_chain"]]
        m = t.merge(key, on=["x", "y"])
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
    data = load_features()
    models = train_models(data)
    transfer, preds = run_directions(pairs, data, models)
    delta = delta_table(pairs, data, preds, sizes)
    main_delta = delta[delta["stratum"] != HINDI]    # no Hindi in the r, standout and BLI analyses
    kw = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    transfer.to_csv(out / "transfer.csv", **kw)
    reference_table(data, models).to_csv(out / "reference.csv", **kw)
    delta.to_csv(out / "delta.csv", **kw)
    summary_table(delta).to_csv(out / "summary.csv", **kw)
    binned_table(main_delta).to_csv(out / "binned_by_r.csv", **kw)
    standout_table(main_delta).to_csv(out / "standout.csv", **kw)
    correlation_table(main_delta, transfer).to_csv(out / "correlations.csv", **kw)
    np.savez_compressed(out / "predictions.npz",
                        **{"|".join(k): v for k, v in preds.items()})
    stats = pd.read_csv(FEAT / "stats.csv")
    info = {"commit": git("rev-parse", "HEAD").strip(), "smoke": bool(args.limit), "config": XCFG,
            "languages": LANGS, "pairs": len(pairs), "n_boot": N_BOOT,
            "classifiers": {"lr_grid": LR_GRID, "lr_C_chosen": {k: v["C"] for k, v in
                                                                 models.items()},
                            "mlp": {"hidden": MLP_HIDDEN, "seeds": N_MLP_SEEDS,
                                    "max_epochs": MLP_MAX_EPOCHS, "patience": MLP_PATIENCE,
                                    "batch": MLP_BATCH}},
            "tweets_without_a_matched_token":
                stats[["lang", "split", "no_token"]].to_dict("records"),
            "bli_companion_joined": (BLI_RESULTS / "pooled.csv").exists(),
            "inputs": file_fingerprints(
                {**{f"features/{lang}": FEAT / f"{lang}.npz" for lang in LANGS},
                 "pairs.csv": P6 / "pairs.csv", "dict_sizes.csv": P6 / "dict_sizes.csv",
                 "bli_companion/pooled.csv": BLI_RESULTS / "pooled.csv",
                 "bli_companion/run_info.json": BLI_RESULTS / "run_info.json",
                 **{f"dicts/{f.name}": f for f in sorted((P6 / "dicts").glob("*.parquet"))}})}
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
