"""AfriSenti tweets -> unit-length mean word-vector features -> <downstream>/afrisenti/<lang>.npz.

Downstream part 2 (experiments/downstream-afrisenti/README.md). Same features and file layout as
prep/13_umsab.py (X_/y_/n_<split>, UMSAB's label numbers). stats.csv is the README's data check,
per language and split: tweets with no matched token, mean matched tokens, the share of tokens
matched, the share of letters in the expected script, the share of matched tokens with a tone mark
or dot (diacritics; meant for Yoruba) and the share of matched tokens that look English (in the
English vocabulary's top 50k and not a PanLex word of the language; code-mixing).
"""

from __future__ import annotations

import logging
import unicodedata

import numpy as np
import pandas as pd
import torch

from clwe_chains import load_config
from clwe_chains.downstream import (
    SPLITS,
    match_rows,
    mean_features,
    read_afrisenti,
    script_share,
    tokenize,
)
from clwe_chains.embeddings import normalize
from clwe_chains.text import load_vocab, vocab_index

log = logging.getLogger("afrisenti")
ENGLISH_TOP = 50_000


def has_diacritic(word: str) -> bool:
    """A combining mark after decomposition (Yoruba tone marks and the dot below)."""
    return any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", word))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    dc = cfg["downstream"]["afrisenti"]
    out = cfg.path("downstream") / "afrisenti"
    out.mkdir(parents=True, exist_ok=True)
    raw = cfg.path("raw_downstream") / "afrisenti"
    en_top = set(vocab_index(load_vocab(cfg.path("v0_embeddings") / "en.vocab.txt")[:ENGLISH_TOP],
                             "en"))
    stats = []
    for lang in dc["languages"]:
        emb = normalize(torch.from_numpy(np.load(cfg.path("v0_embeddings") / f"{lang}.npy")),
                        cfg["embeddings"]["normalize"]).numpy()
        words = load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt")
        index = vocab_index(words, lang)
        panlex_rows = set(pd.read_parquet(cfg.path("v0_panlex") / f"{lang}.parquet")["word"])
        data, arrays = read_afrisenti(raw, lang), {}
        for split in SPLITS:
            texts, labels = data[split]
            x, n = mean_features(texts, lang, index, emb)
            arrays.update({f"X_{split}": x, f"y_{split}": np.array(labels, dtype=np.int64),
                           f"n_{split}": n})
            tokens = [t for text in texts for t in tokenize(text, lang)]
            rows = match_rows(tokens, index)
            matched = [words[r] for r in rows]
            english = sum(r not in panlex_rows and w.lower() in en_top
                          for r, w in zip(rows, matched, strict=True))
            stats.append({
                "lang": lang, "split": split, "tweets": len(texts),
                "no_token": int((n == 0).sum()), "mean_tokens": float(n.mean()),
                "matched_share": len(rows) / max(len(tokens), 1),
                "native_script_share": script_share(texts, dc["scripts"][lang]),
                "diacritic_share": sum(map(has_diacritic, matched)) / max(len(matched), 1),
                "english_like_share": english / max(len(matched), 1),
                "labels": "/".join(str(int((np.array(labels) == k).sum())) for k in range(3))})
            s = stats[-1]
            log.info("%s %s: %d tweets, %d empty, %.1f matched tokens each (%.0f%% of tokens), "
                     "%.0f%% script, %.0f%% diacritics, %.0f%% English-like, labels %s", lang,
                     split, len(texts), s["no_token"], s["mean_tokens"], 100 * s["matched_share"],
                     100 * s["native_script_share"], 100 * s["diacritic_share"],
                     100 * s["english_like_share"], s["labels"])
        np.savez_compressed(out / f"{lang}.npz", **arrays)
    pd.DataFrame(stats).to_csv(out / "stats.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
