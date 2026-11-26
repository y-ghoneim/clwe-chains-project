"""External anchor (review B): our direct map on the MUSE benchmark. See README.md (approved
before running).

Per direction in config.yaml: Procrustes on MUSE's 0-5000 train pairs found in our vocabularies
(clwe_chains.muse), our vectors and normalization, then CSLS (k as configured) and plain
nearest-neighbour retrieval over the target's full vocabulary for the 5000-6500 test words; a
query counts if any of its MUSE translations is retrieved (each at its row in our vocabulary;
the *_anycase column also counts their other casings). results/results.csv sets
P@1 (in %) next to Conneau et al. 2018's Table 1. `--data-files` lists the inputs for the remote.
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
from clwe_chains.align import fit
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.evaluate import bli_multi, metrics
from clwe_chains.muse import Vocab, query_rows, read_muse, train_rows
from clwe_chains.runs import file_fingerprints, result_affecting
from clwe_chains.text import load_vocab

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NAME = HERE.name
RESULT_PATHS = ("src/", "configs/", f"experiments/{NAME}/")
OWN_RESULTS = f"experiments/{NAME}/results/"
CFG = load_config()
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def muse_file(s: str, t: str, split: str) -> Path:
    return CFG.data_root / XCFG["data"] / f"{s}-{t}.{split}.txt"


def inputs() -> list[Path]:
    langs = sorted({lang for pair in XCFG["pairs"] for lang in pair})
    emb = CFG.path("v0_embeddings")
    files = [emb / f"{lang}.{ext}" for lang in langs for ext in ("npy", "vocab.txt")]
    files += [muse_file(s, t, sp) for s, t in XCFG["pairs"] for sp in ("0-5000", "5000-6500")]
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-files", action="store_true", help="list inputs and exit")
    args = ap.parse_args()
    if args.data_files:
        for f in inputs():
            print(f.relative_to(CFG.data_root).as_posix())
        return
    dirty = result_affecting(git("status", "--porcelain").splitlines(), RESULT_PATHS, OWN_RESULTS)
    if dirty:
        sys.exit("uncommitted changes that could affect results:\n" + "\n".join(dirty))
    store = EmbeddingStore(CFG)
    ev = {"k": CFG["eval"]["csls_k"], "block": CFG["eval"]["block_rows"],
          "half": CFG["eval"]["half"]}
    vocab = {}
    rows = []
    for s, t in XCFG["pairs"]:
        for lang in (s, t):
            if lang not in vocab:
                vocab[lang] = Vocab(load_vocab(CFG.path("v0_embeddings") / f"{lang}.vocab.txt"),
                                    lang)
        train_raw = read_muse(muse_file(s, t, "0-5000"))
        train = train_rows(train_raw, vocab[s], vocab[t])
        test_raw = read_muse(muse_file(s, t, "5000-6500"))
        queries, gold, anycase, cov = query_rows(test_raw, vocab[s], vocab[t])
        es, et = store[s], store[t]
        dev = es.device
        w = fit(es[torch.as_tensor(train["a"].to_numpy(), device=dev)],
                et[torch.as_tensor(train["b"].to_numpy(), device=dev)], "procrustes")
        name = f"{s}-{t}"
        row = {"direction": name, "train_pairs_file": len(train_raw), "train_pairs": len(train),
               "train_src_words": int(train["a"].nunique()), **cov}
        for retrieval in ("csls", "nn"):
            ranks, ranks_any = bli_multi(es, et, w, np.array(queries), [gold, anycase],
                                         retrieval=retrieval, **ev)
            m = metrics(ranks)
            row |= {f"{retrieval}_p1": 100 * m["p1"], f"{retrieval}_p5": 100 * m["p5"],
                    f"{retrieval}_mrr": m["mrr"],
                    f"{retrieval}_published_p1": XCFG["published"][retrieval][name]}
            row[f"{retrieval}_diff"] = row[f"{retrieval}_p1"] - row[f"{retrieval}_published_p1"]
            row[f"{retrieval}_p1_anycase"] = 100 * metrics(ranks_any)["p1"]   # secondary gold
        print(f"{name}: CSLS P@1 {row['csls_p1']:.1f} (published {row['csls_published_p1']}), "
              f"NN {row['nn_p1']:.1f} ({row['nn_published_p1']}); "
              f"{cov['test_words_used']} of {cov['test_words']} test words", flush=True)
        rows.append(row)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False, lineterminator="\n",
                              float_format="%.4g")
    info = {"commit": git("rev-parse", "HEAD").strip(), "config": XCFG, "eval": ev,
            "normalize": CFG["embeddings"]["normalize"],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "inputs": file_fingerprints({f.relative_to(CFG.data_root).as_posix(): f
                                         for f in inputs() if f.suffix == ".txt"})}
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
