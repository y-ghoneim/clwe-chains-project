"""Tier 2: downstream transfer with the direct dictionary shrunk to fixed budgets. See README.md
(approved before building).

Two tasks, each with its natural run's classifiers (retrained here the same way, deterministic):
- sib: SIB-200 on stage D's pairs (without zh, ja, th), both directions, 7 topics; budget folders
  <phase6>/dc/b<budget>/ (prep/18), the same subsamples as experiments/controlled-scarcity;
- umsab: UMSAB sentiment on its 21 pairs, both directions, 3 classes; folders <phase6>/uc/; Hindi
  directions reported apart (romanised; decisions.md 2026-11-19).
Per budget and direction (pairs whose full direct dictionary is larger than the budget): direct
on the budget, the chain on the full hub dictionaries, joined direct (the budget plus the hub's
joined pairs, mutual best); no map once per direction. Maps on whole dictionaries (no test-word
removal). The controlled-scarcity BLI cells (its results/cells.csv) are joined for sib if present.
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
from clwe_chains.dictionaries import joined_dictionary
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.runs import file_fingerprints, result_affecting
from clwe_chains.stats import paired_bootstrap
from clwe_chains.transfer import (
    edge_indices,
    fit_classifiers,
    load_centred,
    macro_f1,
    map_features,
    paired_f1_bootstrap,
    predict_kinds,
    spearman_by_group,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CFG = load_config()
P6 = CFG.path("phase6")
BUDGETS = [*CFG["controlled_scarcity"]["budgets"], "full"]
BLI_CELLS = HERE.parent / "controlled-scarcity" / "results" / "cells.csv"
RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/downstream-tier2/",
                "experiments/controlled-scarcity/results/")
OWN_RESULTS = "experiments/downstream-tier2/results/"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
N_MLP_SEEDS, N_BOOT = XCFG["n_mlp_seeds"], XCFG["n_boot"]
KINDS = ("lr", "mlp")
TASKS = {
    "sib": {"folder": "dc", "features": CFG.path("downstream") / "sib200",
            "excluded": set(CFG["downstream"]["sib200"]["excluded"])},
    "umsab": {"folder": "uc", "features": CFG.path("downstream") / "umsab", "excluded": set()},
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def load_task(task: str, langs: list[str]) -> tuple[dict, int]:
    """Centred features per language with integer labels, and the number of classes."""
    data = {lang: load_centred(TASKS[task]["features"] / f"{lang}.npz") for lang in langs}
    if task == "umsab":
        return data, 3
    topics = sorted(set(data[langs[0]]["labels_train"]))
    for d in data.values():
        ids = {t: i for i, t in enumerate(topics)}
        for split in ("train", "validation", "test"):
            d[f"y_{split}"] = np.array([ids[t] for t in d[f"labels_{split}"]], dtype=np.int64)
    return data, len(topics)


def task_pairs(task: str, budget) -> pd.DataFrame:
    p = pd.read_csv(P6 / TASKS[task]["folder"] / f"b{budget}" / "pairs.csv")
    ex = TASKS[task]["excluded"]
    return p[~p["s"].isin(ex) & ~p["t"].isin(ex) & ~p["hub"].isin(ex)].reset_index(drop=True)


def maps_for(store: EmbeddingStore, folder: Path, s: str, t: str, x: str, src: str,
             tgt: str) -> dict[str, torch.Tensor]:
    """direct (budget), chain (full hub edges) and joined direct (budget + mutual-best joined
    pairs), each from src to tgt."""
    def fit_rows(a_rows, b_rows, a, b):
        dev = store[a].device
        return fit(store[a][torch.as_tensor(np.asarray(a_rows), device=dev)],
                   store[b][torch.as_tensor(np.asarray(b_rows), device=dev)], "procrustes")

    d = folder / "dicts"
    direct = pd.read_parquet(folder / "direct" / f"{s}__{t}.parquet")   # the budget's
    counts = pd.read_parquet(folder / "joined" / f"{s}__{x}__{t}.parquet")
    joined = joined_dictionary(direct, counts, set(), set(), filtered=True)
    out = {}
    for name, table in (("direct", direct), ("joined", joined)):
        a, b = table["a"].to_numpy(), table["b"].to_numpy()       # a rows are s, b rows are t
        out[name] = fit_rows(a, b, s, t) if src == s else fit_rows(b, a, t, s)
    out["chain"] = compose([fit_rows(*edge_indices(d, src, x), src, x),
                            fit_rows(*edge_indices(d, x, tgt), x, tgt)])
    return out


def r_of(sizes: pd.Series, src: str, x: str, tgt: str, s: str, t: str) -> float:
    """log2(narrowest full hub step / the budget's direct pairs)."""
    step = min(int(sizes.get(f"{src}__{x}", sizes.get(f"{x}__{src}"))),
               int(sizes.get(f"{x}__{tgt}", sizes.get(f"{tgt}__{x}"))))
    return float(np.log2(step / int(sizes[f"direct:{s}__{t}"])))


def run_task(task: str, store: EmbeddingStore) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    full = task_pairs(task, "full")
    langs = sorted(set(full["s"]) | set(full["t"]))
    data, k = load_task(task, langs)
    models = {lang: fit_classifiers(data[lang], N_MLP_SEEDS, k) for lang in langs}
    rows, deltas = [], []
    for budget in BUDGETS:
        folder = P6 / TASKS[task]["folder"] / f"b{budget}"
        sizes = pd.read_csv(folder / "dict_sizes.csv").set_index("edge")["pairs"]
        for i, p in enumerate(task_pairs(task, budget).itertuples()):
            for src, tgt in ((p.s, p.t), (p.t, p.s)):
                x_test, y_test = data[src]["X_test"], data[src]["y_test"]
                maps = maps_for(store, folder, p.s, p.t, p.hub, src, tgt)
                if budget == "full":
                    maps["nomap"] = None
                preds = {}
                for method, w in maps.items():
                    feats = x_test if w is None else map_features(x_test, w.cpu().numpy())
                    for kind, q in predict_kinds(models[tgt], feats, KINDS).items():
                        preds[method, kind] = q
                        rows.append({"task": task, "budget": budget, "pair": f"{p.s}-{p.t}",
                                     "src": src, "tgt": tgt, "hub": p.hub, "method": method,
                                     "model": kind, "macro_f1": float(np.mean(
                                         [macro_f1(y_test, z, k) for z in q]))})
                for kind in KINDS:
                    for method in ("chain", "joined"):
                        dd, lo, hi = paired_f1_bootstrap(y_test, preds[method, kind],
                                                         preds["direct", kind], N_BOOT, seed=0,
                                                         n_classes=k)
                        deltas.append({"task": task, "budget": budget, "pair": f"{p.s}-{p.t}",
                                       "src": src, "tgt": tgt, "hub": p.hub, "model": kind,
                                       "method": method, "delta_f1": dd, "lo": lo, "hi": hi,
                                       "direct_pairs": int(sizes[f"direct:{p.s}__{p.t}"]),
                                       "r": r_of(sizes, src, p.hub, tgt, p.s, p.t),
                                       "hindi": task == "umsab" and "hi" in (src, tgt)})
            print(f"{task} b{budget} pair {i + 1} {p.s}-{p.t} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(deltas), {g: m["C"] for g, m in models.items()}


def summary_table(delta: pd.DataFrame) -> pd.DataFrame:
    """Per task, budget, model and method (chain or joined, each minus direct): mean over pairs
    with a bootstrap over pairs, directions above / below 0. UMSAB's Hindi directions apart."""
    rows = []
    d = delta.assign(group=np.where(delta["hindi"], "romanised_hindi", "main"))
    for keys, g in d.groupby(["task", "group", "budget", "model", "method"], sort=False):
        v = g.groupby("pair")["delta_f1"].mean().to_numpy()
        mean, lo, hi = paired_bootstrap(v, np.zeros_like(v), n=N_BOOT, seed=0)
        rows.append(dict(zip(("task", "group", "budget", "model", "method"), keys, strict=True))
                    | {"n_pairs": len(v), "n_directions": len(g), "mean_delta_f1": mean,
                       "lo": lo, "hi": hi, "above": int((g["lo"] > 0).sum()),
                       "below": int((g["hi"] < 0).sum())})
    return pd.DataFrame(rows)


R_BINS = [-np.inf, 0, 1, 2, 3, 4, 5, np.inf]
R_LABELS = ["r<=0", "0-1", "1-2", "2-3", "3-4", "4-5", "r>5"]


def binned_table(delta: pd.DataFrame) -> pd.DataFrame:
    """Mean chain (and joined) advantage by r band, per task and model (Hindi apart excluded)."""
    d = delta[~delta["hindi"]]
    d = d.assign(r_bin=pd.cut(d["r"], R_BINS, labels=R_LABELS))
    return (d.groupby(["task", "model", "method", "r_bin"], observed=True)["delta_f1"]
            .agg(n="size", mean_delta_f1="mean").reset_index())


def correlation_table(delta: pd.DataFrame) -> pd.DataFrame:
    """Over budget cells (Hindi excluded): the chain's advantage against r per task, and for
    sib against the controlled-scarcity BLI advantage of the same cell; intervals resample
    pairs."""
    rows = []
    d = delta[(delta["method"] == "chain") & ~delta["hindi"]
              & (delta["budget"].astype(str) != "full")]
    bli = pd.read_csv(BLI_CELLS) if BLI_CELLS.exists() else None
    for (task, kind), g in d.groupby(["task", "model"], sort=False):
        g = g.reset_index(drop=True)
        rho, lo, hi = spearman_by_group(g["r"], g["delta_f1"], g["pair"], N_BOOT, 0)
        rows.append({"task": task, "model": kind, "what": "advantage in F1 vs r", "n": len(g),
                     "rho": rho, "lo": lo, "hi": hi})
        if task == "sib" and bli is not None:
            b = bli[bli["method"] == "chain"].assign(budget=lambda x: x["budget"].astype(str))
            m = g.assign(budget=g["budget"].astype(str)).merge(
                b[["budget", "src", "tgt", "delta"]], on=["budget", "src", "tgt"])
            if len(m) > 2:
                rho, lo, hi = spearman_by_group(m["delta"], m["delta_f1"], m["pair"], N_BOOT, 0)
                rows.append({"task": task, "model": kind,
                             "what": "advantage in F1 vs advantage in MRR (same cell)",
                             "n": len(m), "rho": rho, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="budgets 250 and full only; writes results/smoke/, no clean-tree check")
    args = ap.parse_args()
    global BUDGETS
    status = git("status", "--porcelain").splitlines()
    if not args.smoke and (dirty := result_affecting(status, RESULT_PATHS, OWN_RESULTS)):
        sys.exit("uncommitted changes that could affect results:\n" + "\n".join(dirty))
    if args.smoke:
        BUDGETS = [BUDGETS[0], "full"]
    out = HERE / "results" / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    store = EmbeddingStore(CFG, max_cached=4)
    transfer, delta, chosen = [], [], {}
    for task in TASKS:
        t, d, c = run_task(task, store)
        transfer.append(t)
        delta.append(d)
        chosen[task] = c
    transfer, delta = pd.concat(transfer, ignore_index=True), pd.concat(delta, ignore_index=True)
    kw = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
    transfer.to_csv(out / "transfer.csv", **kw)
    delta.to_csv(out / "delta.csv", **kw)
    summary_table(delta).to_csv(out / "summary.csv", **kw)
    binned_table(delta).to_csv(out / "binned_by_r.csv", **kw)
    correlation_table(delta).to_csv(out / "correlations.csv", **kw)
    info = {"commit": git("rev-parse", "HEAD").strip(), "smoke": args.smoke, "config": XCFG,
            "budgets": BUDGETS, "lr_C_chosen": chosen,
            "bli_cells_joined": BLI_CELLS.exists(),
            "inputs": file_fingerprints(
                {**{f"{t}/{f.name}": f for t, spec in TASKS.items()
                    for f in sorted(spec["features"].glob("*.npz"))},
                 **{f"{spec['folder']}/{f.relative_to(P6 / spec['folder']).as_posix()}": f
                    for spec in TASKS.values()
                    for f in sorted((P6 / spec["folder"]).rglob("*.csv"))},
                 "controlled-scarcity/cells.csv": BLI_CELLS})}
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str) + "\n",
                                       encoding="utf-8")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
