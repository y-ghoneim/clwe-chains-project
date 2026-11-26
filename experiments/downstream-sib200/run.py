"""Topic classification across languages on SIB-200 through direct and chained maps. See README.md
(approved before building).

Two sets of directions (src -> tgt: src's test sentences are moved into tgt's space and classified
by tgt's classifiers): well-resourced = stage D's pairs without zh, ja, th, both directions;
low-resource = the pairs of stages 1, A, A2 whose low-resource language is in SIB-200,
low-resource -> node only. Maps: direct and the chain through the pair's hub (Procrustes on the
stage's full natural dictionaries), and no map. Every language has the same 7 topics and the same
701 / 99 / 204 sentences (translations). Needs prep/17; BLI scores come from the stages' results.
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
SC = CFG["downstream"]["sib200"]
P6 = CFG.path("phase6")
FEAT = CFG.path("downstream") / "sib200"
RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/downstream-sib200/",
                *(f"experiments/{name}/results/pooled.csv" for name in SC["bli"].values()))
OWN_RESULTS = "experiments/downstream-sib200/results/"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
N_MLP_SEEDS, N_BOOT = XCFG["n_mlp_seeds"], XCFG["n_boot"]
KINDS = ("lr", "mlp")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def stage_dir(stage: str) -> Path:
    return P6 / stage if stage else P6


def directions(stage_pairs: dict[str, pd.DataFrame], available: set[str]) -> pd.DataFrame:
    """One row per direction: set, stage, pair (shared by a pair's two directions), src, tgt, hub,
    low (the low-resource language, low set only)."""
    excluded, low = set(SC["excluded"]), set(SC["low"])
    rows = []
    for side in ("well", "low"):
        for stage in SC["stages"][side]:
            for p in stage_pairs[stage].itertuples():
                if excluded & {p.s, p.t, p.hub} or not {p.s, p.t} <= available:
                    continue
                if side == "low" and p.s not in low:
                    continue
                pair = f"{stage or '1'}:{p.s}-{p.t}"
                ways = [(p.s, p.t)] if side == "low" else [(p.s, p.t), (p.t, p.s)]
                for src, tgt in ways:
                    rows.append({"set": side, "stage": stage, "pair": pair, "src": src,
                                 "tgt": tgt, "hub": p.hub, "low": p.s if side == "low" else ""})
    return pd.DataFrame(rows)


def with_ids(d: dict, topics: list[str]) -> dict:
    """y_<split> as topic ids (topics in sorted order)."""
    ids = {t: i for i, t in enumerate(topics)}
    for split in ("train", "validation", "test"):
        d[f"y_{split}"] = np.array([ids[t] for t in d[f"labels_{split}"]], dtype=np.int64)
    return d


def edge_map(store: EmbeddingStore, stage: str, a: str, b: str) -> torch.Tensor:
    ia, ib = edge_indices(stage_dir(stage) / "dicts", a, b)
    dev = store[a].device
    return fit(store[a][torch.as_tensor(ia, device=dev)], store[b][torch.as_tensor(ib, device=dev)],
               "procrustes")


def delta_table(dirs: pd.DataFrame, data: dict, preds: dict, bli: dict, sizes: dict,
                k: int) -> pd.DataFrame:
    rows = []
    for d in dirs.itertuples():
        for kind in KINDS:
            delta, lo, hi = paired_f1_bootstrap(data[d.src]["y_test"],
                                                preds[d.stage, d.src, d.tgt, "chain", kind],
                                                preds[d.stage, d.src, d.tgt, "direct", kind],
                                                N_BOOT, seed=0, n_classes=k)
            row = {"set": d.set, "stage": d.stage, "pair": d.pair, "low": d.low, "src": d.src,
                   "tgt": d.tgt, "hub": d.hub, "model": kind, "delta_f1": delta, "lo": lo,
                   "hi": hi, "r": chain_ratio(sizes[d.stage], d.src, d.hub, d.tgt)}
            b = bli.get(d.stage)
            if b is not None and (d.src, d.tgt) in b.index:
                rec = b.loc[d.src, d.tgt]
                row.update(mrr_direct=float(rec["mrr_thin"]), mrr_chain=float(rec["mrr"]),
                           delta_mrr=float(rec["delta"]))
            rows.append(row)
    return pd.DataFrame(rows)


def summary_table(delta: pd.DataFrame) -> pd.DataFrame:
    """Per set and model: mean over pairs with a bootstrap over pairs; in the low set also per
    low-resource language (mean over its directions, bootstrap over them)."""
    rows = []
    for (side, kind), d in delta.groupby(["set", "model"], sort=False):
        groups = [("all", d)] + ([(lang, g) for lang, g in d.groupby("low")] if side == "low"
                                 else [])
        for name, g in groups:
            v = g.groupby("pair")["delta_f1"].mean().to_numpy()
            mean, lo, hi = paired_bootstrap(v, np.zeros_like(v), n=N_BOOT, seed=0)
            rows.append({"set": side, "model": kind, "group": name, "n_pairs": len(v),
                         "n_directions": len(g), "mean_delta_f1": mean, "lo": lo, "hi": hi,
                         "above": int((g["lo"] > 0).sum()), "below": int((g["hi"] < 0).sum())})
    return pd.DataFrame(rows)


def correlation_table(delta: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    """Per set and model: Spearman of the advantage against r and against its MRR advantage, and
    F1 against MRR over directions x {direct, chain}; intervals resample pairs."""
    rows = []
    for (side, kind), d in delta.groupby(["set", "model"], sort=False):
        d = d.reset_index(drop=True)
        rho, lo, hi = spearman_by_group(d["r"], d["delta_f1"], d["pair"], N_BOOT, 0)
        rows.append({"set": side, "model": kind, "what": "advantage in F1 vs r", "n": len(d),
                     "rho": rho, "lo": lo, "hi": hi})
        if "delta_mrr" not in d or d["delta_mrr"].isna().all():
            continue
        d = d.dropna(subset=["delta_mrr"]).reset_index(drop=True)
        rho, lo, hi = spearman_by_group(d["delta_mrr"], d["delta_f1"], d["pair"], N_BOOT, 0)
        rows.append({"set": side, "model": kind, "what": "advantage in F1 vs advantage in MRR",
                     "n": len(d), "rho": rho, "lo": lo, "hi": hi})
        t = transfer[(transfer["set"] == side) & (transfer["model"] == kind)
                     & transfer["method"].isin(["direct", "chain"])]
        m = t.merge(d[["stage", "src", "tgt", "mrr_direct", "mrr_chain"]],   # pair: transfer's
                    on=["stage", "src", "tgt"])
        mrr = np.where(m["method"] == "direct", m["mrr_direct"], m["mrr_chain"])
        rho, lo, hi = spearman_by_group(mrr, m["macro_f1"], m["pair"], N_BOOT, 0)
        rows.append({"set": side, "model": kind, "what": "F1 vs MRR, directions x {direct, chain}",
                     "n": len(m), "rho": rho, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke run on the first N directions of each set; writes results/smoke/ "
                         "and skips the clean-tree check")
    args = ap.parse_args()
    status = git("status", "--porcelain").splitlines()
    if args.limit is None and (dirty := result_affecting(status, RESULT_PATHS, OWN_RESULTS)):
        sys.exit("uncommitted changes that could affect results:\n" + "\n".join(dirty))
    out = HERE / "results" / ("smoke" if args.limit else "")
    out.mkdir(parents=True, exist_ok=True)
    stages = sorted({s for v in SC["stages"].values() for s in v})
    pairs = {s: pd.read_csv(stage_dir(s) / "pairs.csv") for s in stages}
    sizes = {s: pd.read_csv(stage_dir(s) / "dict_sizes.csv").set_index("edge")["pairs"]
             for s in stages}
    dirs = directions(pairs, set(SC["codes"]))
    if args.limit:
        dirs = dirs.groupby("set", sort=False).head(args.limit).reset_index(drop=True)
    print(f"{len(dirs)} directions: {dirs['set'].value_counts().to_dict()}", flush=True)
    langs = sorted(set(dirs["src"]) | set(dirs["tgt"]))
    data = {lang: load_centred(FEAT / f"{lang}.npz") for lang in langs}
    topics = sorted(set(data[langs[0]]["labels_train"]))
    for lang, d in data.items():                      # SIB-200: one topic set for every language
        for split in ("train", "validation", "test"):
            if not set(d[f"labels_{split}"]) <= set(topics):
                sys.exit(f"{lang} {split} has topics outside {topics}")
    k = len(topics)
    data = {lang: with_ids(d, topics) for lang, d in data.items()}

    models, reference = {}, []
    for lang in langs:
        models[lang] = fit_classifiers(data[lang], N_MLP_SEEDS, k)
        for kind, p in predict_kinds(models[lang], data[lang]["X_test"], KINDS).items():
            reference.append({"lang": lang, "model": kind, "ceiling_macro_f1": float(np.mean(
                [macro_f1(data[lang]["y_test"], q, k) for q in p])),
                "floor_macro_f1": majority_floor(data[lang]["y_test"], k),
                "C": models[lang]["C"] if kind == "lr" else None})
        print(f"trained {lang}: C = {models[lang]['C']}", flush=True)

    store = EmbeddingStore(CFG, max_cached=4)
    rows, preds = [], {}
    for d in dirs.itertuples():
        x_test, y_test = data[d.src]["X_test"], data[d.src]["y_test"]
        maps = {"direct": edge_map(store, d.stage, d.src, d.tgt),
                "chain": compose([edge_map(store, d.stage, d.src, d.hub),
                                  edge_map(store, d.stage, d.hub, d.tgt)]),
                "nomap": None}
        for method, w in maps.items():
            feats = x_test if w is None else map_features(x_test, w.cpu().numpy())
            for kind, p in predict_kinds(models[d.tgt], feats, KINDS).items():
                preds[d.stage, d.src, d.tgt, method, kind] = p
                rows.append({"set": d.set, "stage": d.stage, "pair": d.pair, "src": d.src,
                             "tgt": d.tgt, "hub": d.hub, "method": method, "model": kind,
                             "macro_f1": float(np.mean([macro_f1(y_test, q, k) for q in p])),
                             "accuracy": float(np.mean([(q == y_test).mean() for q in p]))})
        print(f"{d.set} {d.src}->{d.tgt} via {d.hub} (stage {d.stage or '1'}) done", flush=True)
    transfer = pd.DataFrame(rows)

    bli = {}
    for stage, name in SC["bli"].items():
        f = REPO / "experiments" / name / "results" / "pooled.csv"
        if f.exists():
            b = pd.read_csv(f)
            bli[stage] = b[b["method"] == "chain"].set_index(["src", "tgt"])
    delta = delta_table(dirs, data, preds, bli, sizes, k)
    kw = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}

    def named(df: pd.DataFrame) -> pd.DataFrame:
        """Stage 1's data folder is "" in code; "1" in the tables (an empty field reads as NaN)."""
        return df.assign(stage=df["stage"].replace("", "1"))

    dirs, transfer, delta = named(dirs), named(transfer), named(delta)
    dirs.to_csv(out / "directions.csv", **kw)
    transfer.to_csv(out / "transfer.csv", **kw)
    pd.DataFrame(reference).to_csv(out / "reference.csv", **kw)
    delta.to_csv(out / "delta.csv", **kw)
    summary_table(delta).to_csv(out / "summary.csv", **kw)
    correlation_table(delta, transfer).to_csv(out / "correlations.csv", **kw)
    np.savez_compressed(out / "predictions.npz",
                        **{"|".join(x or "1" for x in key): v for key, v in preds.items()})
    info = {"commit": git("rev-parse", "HEAD").strip(), "smoke": bool(args.limit), "config": XCFG,
            "sib200": SC, "topics": topics, "directions": dirs["set"].value_counts().to_dict(),
            "n_boot": N_BOOT, "texts_without_a_matched_token":
                pd.read_csv(FEAT / "stats.csv").query("no_token > 0")[
                    ["lang", "split", "no_token"]].to_dict("records"),
            "classifiers": {"lr_grid": LR_GRID, "lr_C_chosen": {g: m["C"] for g, m in
                                                                 models.items()},
                            "mlp": {"hidden": MLP_HIDDEN, "seeds": N_MLP_SEEDS,
                                    "max_epochs": MLP_MAX_EPOCHS, "patience": MLP_PATIENCE,
                                    "batch": MLP_BATCH}},
            "inputs": file_fingerprints(
                {**{f"features/{lang}": FEAT / f"{lang}.npz" for lang in langs},
                 "sib200/stats.csv": FEAT / "stats.csv",
                 **{f"stage {s or '1'}/{f}": stage_dir(s) / f for s in stages
                    for f in ("pairs.csv", "dict_sizes.csv")},
                 **{f"bli/{s or '1'}": REPO / "experiments" / n / "results" / "pooled.csv"
                    for s, n in SC["bli"].items()},
                 **{f"stage {d.stage or '1'}/dicts/{a}__{b}": f
                    for d in dirs.assign(stage=dirs["stage"].replace("1", ""))
                    .drop_duplicates(["stage", "src", "tgt", "hub"]).itertuples()
                    for a, b in ((d.src, d.tgt), (d.src, d.hub), (d.hub, d.tgt))
                    for f in [stage_dir(d.stage) / "dicts" / f"{a}__{b}.parquet"]
                    if f.exists()}})}
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
