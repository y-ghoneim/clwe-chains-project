"""UMSAB tweets -> unit-length mean word-vector features -> <downstream>/umsab/<lang>.npz.

Downstream experiment (experiments/downstream-umsab/README.md). Per language and split
(train / validation / test): X_<split> (tweets x 300, float32), y_<split> (labels 0/1/2) and
n_<split> (matched tokens per tweet). Features use the project's normalized fastText vectors
(unit, centre, unit) of the language's top 200k words. stats.csv: tweets with no matched token, the
mean matched tokens and the share of letters in the language's expected script, per language and
split.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from clwe_chains import load_config
from clwe_chains.downstream import SPLITS, mean_features, read_umsab, script_share
from clwe_chains.embeddings import normalize
from clwe_chains.text import load_vocab, vocab_index

log = logging.getLogger("umsab")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    out = cfg.path("downstream") / "umsab"
    out.mkdir(parents=True, exist_ok=True)
    raw = cfg.path("raw_downstream") / "umsab"
    stats = []
    for lang in cfg["downstream"]["umsab"]["languages"]:
        emb = normalize(torch.from_numpy(np.load(cfg.path("v0_embeddings") / f"{lang}.npy")),
                        cfg["embeddings"]["normalize"]).numpy()
        index = vocab_index(load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt"), lang)
        data, arrays = read_umsab(raw, lang), {}
        for split in SPLITS:
            texts, labels = data[split]
            x, n = mean_features(texts, lang, index, emb)
            arrays.update({f"X_{split}": x, f"y_{split}": np.array(labels, dtype=np.int64),
                           f"n_{split}": n})
            # share of letters in the script the language's vectors and dictionaries use: UMSAB's
            # Hindi turned out to be romanized (Latin), not Devanagari
            share = script_share(texts, cfg["pairs"]["scripts"][lang])
            stats.append({"lang": lang, "split": split, "tweets": len(texts),
                          "no_token": int((n == 0).sum()), "mean_tokens": float(n.mean()),
                          "native_script_share": share})
            log.info("%s %s: %d tweets, %d without a matched token, %.1f matched tokens each, "
                     "%.0f%% of letters in the expected script", lang, split, len(texts),
                     stats[-1]["no_token"], stats[-1]["mean_tokens"], 100 * share)
        np.savez_compressed(out / f"{lang}.npz", **arrays)
    pd.DataFrame(stats).to_csv(out / "stats.csv", index=False)


if __name__ == "__main__":
    main()
