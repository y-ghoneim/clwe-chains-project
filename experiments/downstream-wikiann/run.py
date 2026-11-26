"""Cross-lingual named-entity recognition on WikiANN through direct and chained maps. See README.md
(approved before building).

Two sets of directions (src -> tgt: src's test sentences are moved into tgt's space and tagged by
tgt's taggers): well-resourced = stage D's pairs without zh, ja, th, both directions whose target
has at least downstream.wikiann.min_train training sentences; low-resource = the pairs of stages
1, A, A2 whose low-resource language has WikiANN, low-resource -> node only. Maps: direct and the
chain through the pair's hub (Procrustes on the stage's full natural dictionaries), and no map.
Taggers per language (ner.py): BiLSTM-CRF on word vectors (headline, 3 seeds), the same with
character features (secondary A, 3 seeds), window logistic regression (secondary B). Taggers and
per-direction counts are cached under <results>/downstream-wikiann/ with the commit they were made
at, so an interrupted run resumes (a cache entry from another commit is redone). Runs on the
remote PC (docs/remote.md); `--data-files` lists its inputs.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from clwe_chains import load_config
from clwe_chains.align import compose, fit
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.ner import (
    Corpus,
    char_ids,
    char_vocab,
    f1_from_counts,
    fit_tagger,
    fit_window_lr,
    paired_entity_bootstrap,
    predict_tagger,
    sentence_counts,
    split_tags,
    token_mean,
    window_features,
)
from clwe_chains.runs import file_fingerprints, result_affecting
from clwe_chains.stats import paired_bootstrap
from clwe_chains.transfer import chain_ratio, edge_indices, spearman_by_group

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CFG = load_config()
WC = CFG["downstream"]["wikiann"]
P6 = CFG.path("phase6")
FEAT = CFG.path("downstream") / "wikiann"
CACHE = CFG.path("results") / "downstream-wikiann"
RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/downstream-wikiann/",
                *(f"experiments/{name}/results/pooled.csv"          # the joined BLI scores
                  for name in CFG["downstream"]["wikiann"]["bli"].values()))
OWN_RESULTS = "experiments/downstream-wikiann/results/"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
N_SEEDS, N_BOOT = XCFG["n_seeds"], XCFG["n_boot"]
TAGGERS = ("bilstm", "bilstm_char", "window_lr")     # headline, secondary A, secondary B
METHODS = ("direct", "chain", "nomap")
SMOKE_CAP = None        # smoke runs only: train and validate on at most this many sentences


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


LOG: Path | None = None   # <cache>/run.log, set in main (docs/remote.md copies it back)


def say(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG is not None:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------------------------
# Directions (tests/test_downstream_wikiann_experiment.py)

def stage_dir(stage: str) -> Path:
    return P6 / stage if stage else P6


def directions(stage_pairs: dict[str, pd.DataFrame], train_sizes: dict[str, int]) -> pd.DataFrame:
    """One row per direction: set, stage, pair (an id shared by a pair's two directions), src,
    tgt, hub, low (the low-resource language, low set only)."""
    excluded, low, need = set(WC["excluded"]), set(WC["low"]), WC["min_train"]
    rows = []
    for side in ("well", "low"):
        for stage in WC["stages"][side]:
            for p in stage_pairs[stage].itertuples():
                if excluded & {p.s, p.t, p.hub}:
                    continue
                if side == "low" and p.s not in low:
                    continue
                pair = f"{stage or '1'}:{p.s}-{p.t}"
                ways = [(p.s, p.t)] if side == "low" else [(p.s, p.t), (p.t, p.s)]
                for src, tgt in ways:
                    if train_sizes.get(tgt, 0) >= need and src in train_sizes:
                        rows.append({"set": side, "stage": stage, "pair": pair, "src": src,
                                     "tgt": tgt, "hub": p.hub,
                                     "low": p.s if side == "low" else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Data

class Lang:
    """A language's WikiANN sentences (rows, tags, tokens per split) and its token mean."""

    def __init__(self, lang: str, store: EmbeddingStore):
        d = pd.read_parquet(FEAT / f"{lang}.parquet")
        self.lang = lang
        self.split = {s: g for s, g in d.groupby("split", sort=False)}
        emb = store[lang]
        rows = np.concatenate(self.split["train"]["rows"].to_numpy())
        self.mu = token_mean(emb.cpu().numpy(), rows)

    def corpus(self, split: str, emb: torch.Tensor, w: torch.Tensor | None,
               vocab: dict[str, int] | None) -> Corpus:
        """Centred (and mapped) vectors of a split; character ids under a tagger's vocabulary."""
        g = self.split[split]
        if SMOKE_CAP and split != "test":
            g = g.head(SMOKE_CAP)
        rows = np.concatenate(g["rows"].to_numpy()).astype(np.int64)
        lengths = np.array([len(r) for r in g["rows"]])
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        r = torch.as_tensor(rows, device=emb.device)
        v = torch.zeros((len(rows), emb.shape[1]), device=emb.device)
        hit = r >= 0
        v[hit] = emb[r[hit]] - torch.as_tensor(self.mu, device=emb.device)
        if w is not None:
            v = v @ w
        c = Corpus(v.cpu().numpy(), offsets, np.concatenate(g["tags"].to_numpy()).astype(np.int64))
        if vocab is not None:
            c.chars, c.char_lens = char_ids([t for ts in g["tokens"] for t in ts], vocab)
        return c


def cached(path: Path, commit: str):
    """A cache entry made at this commit, or None."""
    if path.exists():
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if obj["commit"] == commit:
            return obj
    return None


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    tmp.replace(path)


def taggers(lang: Lang, store: EmbeddingStore, commit: str, device) -> dict:
    """kind -> {"models": [...], "info": [...], "vocab": char vocabulary or None}, trained or
    loaded from the cache."""
    out = {}
    emb = store[lang.lang]
    vocab = char_vocab([t for ts in lang.split["train"]["tokens"] for t in ts])
    for kind in TAGGERS:
        path = CACHE / "taggers" / f"{lang.lang}_{kind}.pkl"
        hit = cached(path, commit)
        if hit is None:
            v = vocab if kind == "bilstm_char" else None
            train, val = lang.corpus("train", emb, None, v), lang.corpus("validation", emb, None, v)
            say(f"training {kind} for {lang.lang}: {len(train)} sentences")
            if kind == "window_lr":
                m, c, f1 = fit_window_lr(train, val)
                hit = {"models": [m], "info": [{"C": c, "best_val_f1": f1}]}
            else:
                n_chars = len(vocab) + 2 if v else 0
                name = f"{lang.lang} {kind}"
                fitted = [fit_tagger(train, val, seed, device, n_chars,
                                     log=lambda m, s=seed, n=name: say(f"  {n} seed {s} {m}"))
                          for seed in range(N_SEEDS)]
                hit = {"models": [m.cpu().state_dict() for m, _ in fitted],
                       "info": [i for _, i in fitted]}
            hit.update(commit=commit, vocab=v, dim=emb.shape[1])
            save(path, hit)
        out[kind] = hit
    return out


def tag_counts(kind: str, tg: dict, c: Corpus, device) -> np.ndarray:
    """(models, sentences, 3) entity counts of a tagger's predictions on a corpus."""
    from clwe_chains.ner import BiLSTMCRF
    gold = c.sentence_tags()
    out = []
    for m in tg["models"]:
        if kind == "window_lr":
            pred = split_tags(m.predict(window_features(c)), c.offsets)
        else:
            model = BiLSTMCRF(tg["dim"], len(tg["vocab"]) + 2 if tg["vocab"] else 0).to(device)
            model.load_state_dict(m)
            pred = predict_tagger(model, c, device)
        out.append(sentence_counts(gold, pred))
    return np.stack(out)


def edge_map(store: EmbeddingStore, stage: str, a: str, b: str) -> torch.Tensor:
    ia, ib = edge_indices(stage_dir(stage) / "dicts", a, b)
    dev = store[a].device
    return fit(store[a][torch.as_tensor(ia, device=dev)], store[b][torch.as_tensor(ib, device=dev)],
               "procrustes")


# ---------------------------------------------------------------------------------------------
# Tables

def delta_table(dirs: pd.DataFrame, counts: dict, bli: dict, sizes: dict) -> pd.DataFrame:
    rows = []
    for d in dirs.itertuples():
        for kind in TAGGERS:
            c = counts[d.set, d.stage, d.src, d.tgt]
            delta, lo, hi = paired_entity_bootstrap(c["chain", kind], c["direct", kind], N_BOOT)
            row = {"set": d.set, "stage": d.stage, "pair": d.pair, "low": d.low, "src": d.src,
                   "tgt": d.tgt, "hub": d.hub, "tagger": kind, "delta_f1": delta, "lo": lo,
                   "hi": hi, "r": chain_ratio(sizes[d.stage], d.src, d.hub, d.tgt)}
            b = bli.get(d.stage)
            if b is not None and (d.src, d.tgt) in b.index:
                rec = b.loc[d.src, d.tgt]
                row.update(mrr_direct=float(rec["mrr_thin"]), mrr_chain=float(rec["mrr"]),
                           delta_mrr=float(rec["delta"]))
            rows.append(row)
    return pd.DataFrame(rows)


def summary_table(delta: pd.DataFrame) -> pd.DataFrame:
    """Per set and tagger: mean over pairs with a bootstrap over pairs; in the low set also per
    low-resource language (mean over its directions, bootstrap over them)."""
    rows = []
    for (side, kind), d in delta.groupby(["set", "tagger"], sort=False):
        groups = [("all", d)] + ([(lang, g) for lang, g in d.groupby("low")] if side == "low"
                                 else [])
        for name, g in groups:
            v = g.groupby("pair")["delta_f1"].mean().to_numpy()
            mean, lo, hi = paired_bootstrap(v, np.zeros_like(v), n=N_BOOT, seed=0)
            rows.append({"set": side, "tagger": kind, "group": name, "n_pairs": len(v),
                         "n_directions": len(g), "mean_delta_f1": mean, "lo": lo, "hi": hi,
                         "above": int((g["lo"] > 0).sum()), "below": int((g["hi"] < 0).sum())})
    return pd.DataFrame(rows)


def correlation_table(delta: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    """Per set and tagger: Spearman of the chain's advantage against r and against its MRR
    advantage, and F1 against MRR over directions x {direct, chain}; intervals resample pairs."""
    rows = []
    for (side, kind), d in delta.groupby(["set", "tagger"], sort=False):
        d = d.reset_index(drop=True)
        rho, lo, hi = spearman_by_group(d["r"], d["delta_f1"], d["pair"], N_BOOT, 0)
        rows.append({"set": side, "tagger": kind, "what": "advantage in F1 vs r", "n": len(d),
                     "rho": rho, "lo": lo, "hi": hi})
        if "delta_mrr" not in d or d["delta_mrr"].isna().all():
            continue
        d = d.dropna(subset=["delta_mrr"]).reset_index(drop=True)
        rho, lo, hi = spearman_by_group(d["delta_mrr"], d["delta_f1"], d["pair"], N_BOOT, 0)
        rows.append({"set": side, "tagger": kind, "what": "advantage in F1 vs advantage in MRR",
                     "n": len(d), "rho": rho, "lo": lo, "hi": hi})
        t = transfer[(transfer["set"] == side) & (transfer["tagger"] == kind)
                     & transfer["method"].isin(["direct", "chain"])]
        m = t.merge(d[["stage", "src", "tgt", "mrr_direct", "mrr_chain"]],   # pair: transfer's
                    on=["stage", "src", "tgt"])
        mrr = np.where(m["method"] == "direct", m["mrr_direct"], m["mrr_chain"])
        rho, lo, hi = spearman_by_group(mrr, m["f1"], m["pair"], N_BOOT, 0)
        rows.append({"set": side, "tagger": kind, "what": "F1 vs MRR, directions x {direct, chain}",
                     "n": len(m), "rho": rho, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------

def load_inputs() -> tuple[dict, dict, dict, pd.DataFrame]:
    stages = sorted({s for v in WC["stages"].values() for s in v})
    pairs = {s: pd.read_csv(stage_dir(s) / "pairs.csv") for s in stages}
    sizes = {s: pd.read_csv(stage_dir(s) / "dict_sizes.csv").set_index("edge")["pairs"]
             for s in stages}
    st = pd.read_csv(FEAT / "stats.csv")
    train_sizes = dict(zip(st.loc[st["split"] == "train", "lang"],
                           st.loc[st["split"] == "train", "sentences"], strict=True))
    return pairs, sizes, train_sizes, directions(pairs, train_sizes)


def data_files(dirs: pd.DataFrame) -> list[Path]:
    """Every input the run reads, relative to the data root (for the remote copy)."""
    langs = sorted(set(dirs["src"]) | set(dirs["tgt"]) | set(dirs["hub"]))
    files = [CFG.path("v0_embeddings") / f"{lang}.npy" for lang in langs]
    files += [FEAT / f"{lang}.parquet" for lang in sorted(set(dirs["src"]) | set(dirs["tgt"]))]
    files.append(FEAT / "stats.csv")
    for stage in sorted(set(dirs["stage"])):
        files += [stage_dir(stage) / "pairs.csv", stage_dir(stage) / "dict_sizes.csv"]
        need = set()
        for d in dirs[dirs["stage"] == stage].itertuples():
            for a, b in ((d.src, d.tgt), (d.src, d.hub), (d.hub, d.tgt)):
                need |= {f"{a}__{b}", f"{b}__{a}"}
        files += sorted(f for f in (stage_dir(stage) / "dicts").glob("*.parquet")
                        if f.stem in need)
    return [f.relative_to(CFG.data_root) for f in files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke run on the first N directions of each set; writes results/smoke/ "
                         "and skips the clean-tree check")
    ap.add_argument("--data-files", action="store_true", help="list inputs and exit")
    args = ap.parse_args()
    _, sizes, train_sizes, dirs = load_inputs()
    if args.data_files:
        for f in data_files(dirs):
            print(f.as_posix())
        return
    status = git("status", "--porcelain").splitlines()
    if args.limit is None and (dirty := result_affecting(status, RESULT_PATHS, OWN_RESULTS)):
        sys.exit("uncommitted changes that could affect results:\n" + "\n".join(dirty))
    commit = git("rev-parse", "HEAD").strip()
    if args.limit:
        global SMOKE_CAP, CACHE
        SMOKE_CAP, CACHE = 300, CACHE / "smoke"         # smoke entries never touch the run's
        dirs = dirs.groupby("set", sort=False).head(args.limit).reset_index(drop=True)
        commit = "smoke-" + commit
    out = HERE / "results" / ("smoke" if args.limit else "")
    out.mkdir(parents=True, exist_ok=True)
    global LOG
    CACHE.mkdir(parents=True, exist_ok=True)
    LOG = CACHE / "run.log"
    say(f"start commit {commit}: {len(dirs)} directions "
        f"({dirs['set'].value_counts().to_dict()})")
    device = torch.device(CFG["device"])
    store = EmbeddingStore(CFG, max_cached=4)
    langs = sorted(set(dirs["src"]) | set(dirs["tgt"]))
    data = {lang: Lang(lang, store) for lang in langs}
    tg = {lang: taggers(data[lang], store, commit, device) for lang in langs}

    reference = []
    for lang in langs:
        for kind in TAGGERS:
            c = tag_counts(kind, tg[lang][kind], data[lang].corpus(
                "test", store[lang], None, tg[lang][kind]["vocab"]), device)
            reference.append({"lang": lang, "tagger": kind, "train_sentences": train_sizes[lang],
                              "ceiling_f1": float(np.mean([f1_from_counts(m) for m in c])),
                              "val_f1": float(np.mean([i["best_val_f1"]
                                                       for i in tg[lang][kind]["info"]]))})

    counts, transfer = {}, []
    for d in dirs.itertuples():
        path = CACHE / "counts" / f"{d.set}_{d.stage or '1'}_{d.src}_{d.tgt}.pkl"
        hit = cached(path, commit)
        if hit is None:
            maps = {"direct": edge_map(store, d.stage, d.src, d.tgt),
                    "chain": compose([edge_map(store, d.stage, d.src, d.hub),
                                      edge_map(store, d.stage, d.hub, d.tgt)]),
                    "nomap": None}
            hit = {"commit": commit}
            for method, w in maps.items():
                for kind in TAGGERS:
                    c = data[d.src].corpus("test", store[d.src], w, tg[d.tgt][kind]["vocab"])
                    hit[method, kind] = tag_counts(kind, tg[d.tgt][kind], c, device)
            save(path, hit)
            say(f"{d.set} {d.src}->{d.tgt} via {d.hub} (stage {d.stage or '1'}) done")
        counts[d.set, d.stage, d.src, d.tgt] = hit
        for method in METHODS:
            for kind in TAGGERS:
                f1s = [f1_from_counts(m) for m in hit[method, kind]]
                transfer.append({"set": d.set, "stage": d.stage, "pair": d.pair, "src": d.src,
                                 "tgt": d.tgt, "hub": d.hub, "method": method, "tagger": kind,
                                 "f1": float(np.mean(f1s)), "f1_sd": float(np.std(f1s))})
    transfer = pd.DataFrame(transfer)

    bli = {}
    for stage, name in WC["bli"].items():
        f = REPO / "experiments" / name / "results" / "pooled.csv"
        if f.exists():
            b = pd.read_csv(f)
            bli[stage] = b[b["method"] == "chain"].set_index(["src", "tgt"])
    delta = delta_table(dirs, counts, bli, sizes)
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
    info = {"commit": commit, "smoke": bool(args.limit), "config": XCFG, "wikiann": WC,
            "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                            else None, "cudnn_deterministic": False},
            "directions": dirs["set"].value_counts().to_dict(),
            "taggers": {lang: {k: v["info"] for k, v in t.items()} for lang, t in tg.items()},
            "inputs": file_fingerprints({f.as_posix(): CFG.data_root / f
                                         for f in data_files(dirs)}
                                        | {f"bli/{s}": REPO / "experiments" / n / "results" /
                                           "pooled.csv" for s, n in WC["bli"].items()})}
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str) + "\n",
                                       encoding="utf-8")
    say(f"results written to {out}")


if __name__ == "__main__":
    main()
