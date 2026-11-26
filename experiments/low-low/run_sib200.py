"""Both ends low-resource: SIB-200 topic classification between low-resource languages of
different families. See README.md (approved before running).

The setup of ../downstream-sib200/ (its functions imported unchanged): src's test sentences are
mapped into tgt's space and classified by tgt's classifiers, trained on tgt's own sentences. Maps:
direct and the chain through the pair's hub (Procrustes on stage ll's full natural dictionaries),
and no map. Directions: the stage's pairs with both languages in SIB-200, both ways; a pair whose
hub is excluded (zh, ja, th) is left out and listed, as in that experiment. Writes results/sib200/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from clwe_chains.align import compose
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.runs import file_fingerprints, result_affecting
from clwe_chains.stats import paired_bootstrap

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "low_low_sib_run", HERE.parent / "downstream-sib200" / "run.py")
sib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sib)
STAGE = "ll"
OUT = HERE / "results" / "sib200"
RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/downstream-sib200/run.py",
                "experiments/downstream-sib200/config.yaml", "experiments/low-low/")
OWN_RESULTS = "experiments/low-low/results/"
METHODS = ("direct", "chain", "nomap")


def directions(pairs: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Both directions of every pair with both languages in SIB-200; and the pairs left out
    because their hub is excluded."""
    low, excluded = set(sib.SC["low"]), set(sib.SC["excluded"])
    rows, left = [], []
    for p in pairs.itertuples():
        if not {p.s, p.t} <= low:
            continue
        if p.hub in excluded:
            left.append({"s": p.s, "t": p.t, "hub": p.hub})
            continue
        for src, tgt in ((p.s, p.t), (p.t, p.s)):
            rows.append({"pair": f"{p.s}-{p.t}", "src": src, "tgt": tgt, "hub": p.hub})
    return pd.DataFrame(rows), left


def summary(transfer: pd.DataFrame, delta: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Per model: mean macro-F1 of each method over directions; chain - direct as the mean over
    pairs (each pair's two directions averaged) with a bootstrap over pairs; directions above /
    below 0 by their own bootstrap."""
    rows = []
    for kind, d in delta.groupby("model", sort=False):
        t = transfer[transfer["model"] == kind]
        v = d.groupby("pair")["delta_f1"].mean().to_numpy()
        mean, lo, hi = paired_bootstrap(v, np.zeros_like(v), n=n_boot, seed=0)
        rows.append({"model": kind, "n_pairs": len(v), "n_directions": len(d),
                     **{f"f1_{m}": t[t["method"] == m]["macro_f1"].mean() for m in METHODS},
                     **{f"acc_{m}": t[t["method"] == m]["accuracy"].mean() for m in METHODS},
                     "chain_minus_direct": mean, "lo": lo, "hi": hi,
                     "above": int((d["lo"] > 0).sum()), "below": int((d["hi"] < 0).sum())})
    return pd.DataFrame(rows)


def positive(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=positive, default=None,
                    help="smoke run on the first N directions; writes results/sib200/smoke/ and "
                         "skips the clean-tree check")
    args = ap.parse_args()
    smoke = args.limit is not None
    status = sib.git("status", "--porcelain").splitlines()
    if not smoke and (dirty := result_affecting(status, RESULT_PATHS, OWN_RESULTS)):
        sys.exit("uncommitted changes that could affect results:\n" + "\n".join(dirty))
    out = OUT / "smoke" if smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    stage_dir = sib.stage_dir(STAGE)
    sizes = pd.read_csv(stage_dir / "dict_sizes.csv").set_index("edge")["pairs"]
    dirs, left = directions(pd.read_csv(stage_dir / "pairs.csv"))
    if smoke:
        dirs = dirs.head(args.limit)
    print(f"{len(dirs)} directions; left out (hub excluded): {left}", flush=True)
    langs = sorted(set(dirs["src"]) | set(dirs["tgt"]))
    data = {lang: sib.load_centred(sib.FEAT / f"{lang}.npz") for lang in langs}
    topics = sorted(set(data[langs[0]]["labels_train"]))
    for lang, d in data.items():                       # SIB-200: one topic set for every language
        for split in ("train", "validation", "test"):
            if not set(d[f"labels_{split}"]) <= set(topics):
                sys.exit(f"{lang} {split} has topics outside {topics}")
    k = len(topics)
    data = {lang: sib.with_ids(d, topics) for lang, d in data.items()}

    models, reference = {}, []
    for lang in langs:
        models[lang] = sib.fit_classifiers(data[lang], sib.N_MLP_SEEDS, k)
        for kind, p in sib.predict_kinds(models[lang], data[lang]["X_test"], sib.KINDS).items():
            reference.append({"lang": lang, "model": kind, "ceiling_macro_f1": float(np.mean(
                [sib.macro_f1(data[lang]["y_test"], q, k) for q in p])),
                "floor_macro_f1": sib.majority_floor(data[lang]["y_test"], k),
                "C": models[lang]["C"] if kind == "lr" else None})
        print(f"trained {lang}: C = {models[lang]['C']}", flush=True)

    store = EmbeddingStore(sib.CFG, max_cached=4)
    rows, preds, deltas = [], {}, []
    for d in dirs.itertuples():
        x_test, y_test = data[d.src]["X_test"], data[d.src]["y_test"]
        maps = {"direct": sib.edge_map(store, STAGE, d.src, d.tgt),
                "chain": compose([sib.edge_map(store, STAGE, d.src, d.hub),
                                  sib.edge_map(store, STAGE, d.hub, d.tgt)]),
                "nomap": None}
        for method, w in maps.items():
            feats = x_test if w is None else sib.map_features(x_test, w.cpu().numpy())
            for kind, p in sib.predict_kinds(models[d.tgt], feats, sib.KINDS).items():
                preds[d.src, d.tgt, method, kind] = p
                rows.append({"pair": d.pair, "src": d.src, "tgt": d.tgt, "hub": d.hub,
                             "method": method, "model": kind,
                             "macro_f1": float(np.mean([sib.macro_f1(y_test, q, k) for q in p])),
                             "accuracy": float(np.mean([(q == y_test).mean() for q in p]))})
        for kind in sib.KINDS:
            delta, lo, hi = sib.paired_f1_bootstrap(
                y_test, preds[d.src, d.tgt, "chain", kind], preds[d.src, d.tgt, "direct", kind],
                sib.N_BOOT, seed=0, n_classes=k)
            deltas.append({"pair": d.pair, "src": d.src, "tgt": d.tgt, "hub": d.hub,
                           "model": kind, "delta_f1": delta, "lo": lo, "hi": hi,
                           "r": sib.chain_ratio(sizes, d.src, d.hub, d.tgt)})
        print(f"{d.src}->{d.tgt} via {d.hub} done", flush=True)
    transfer, delta = pd.DataFrame(rows), pd.DataFrame(deltas)
    kw = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    dirs.to_csv(out / "directions.csv", **kw)
    transfer.to_csv(out / "transfer.csv", **kw)
    pd.DataFrame(reference).to_csv(out / "reference.csv", **kw)
    delta.to_csv(out / "delta.csv", **kw)
    summ = summary(transfer, delta, sib.N_BOOT)
    summ.to_csv(out / "summary.csv", **kw)
    np.savez_compressed(out / "predictions.npz", **{"|".join(key): v for key, v in preds.items()})
    info = {"commit": sib.git("rev-parse", "HEAD").strip(), "smoke": smoke,
            "stage": STAGE, "left_out": left, "topics": topics, "n_boot": sib.N_BOOT,
            "config": sib.XCFG, "sib200": sib.SC,
            "classifiers": {"lr_grid": sib.LR_GRID,
                            "lr_C_chosen": {g: m["C"] for g, m in models.items()},
                            "mlp": {"hidden": sib.MLP_HIDDEN, "seeds": sib.N_MLP_SEEDS,
                                    "max_epochs": sib.MLP_MAX_EPOCHS,
                                    "patience": sib.MLP_PATIENCE, "batch": sib.MLP_BATCH}},
            "inputs": file_fingerprints(
                {**{f"features/{lang}": sib.FEAT / f"{lang}.npz" for lang in langs},
                 **{f"stage ll/{f}": stage_dir / f for f in ("pairs.csv", "dict_sizes.csv")},
                 **{f"stage ll/dicts/{f.name}": f for f in sorted((stage_dir / "dicts")
                                                                   .glob("*.parquet"))}})}
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(summ.to_string(index=False))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
