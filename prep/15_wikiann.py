"""WikiANN sentences -> tokens, tags and vocabulary rows -> <downstream>/wikiann/<lang>.parquet.

Downstream part 3 (experiments/downstream-wikiann/README.md). One row per sentence: split
(train / validation / test), tokens, tags (WikiANN's ids, ner.TAGS) and rows (each token's
vocabulary row: its exact surface form, else its normalized form, else -1). Every language
downloaded under raw/downstream/wikiann/ is processed. stats.csv, the README's check before any
transfer number: sentences, tokens, entities, and the shares of tokens matched exactly, through
the normalized form, and not at all, per language and split.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from clwe_chains import load_config
from clwe_chains.downstream import SPLITS
from clwe_chains.ner import entities, match_tokens
from clwe_chains.text import load_vocab, normalize, vocab_index

log = logging.getLogger("wikiann")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    raw = cfg.path("raw_downstream") / "wikiann"
    out = cfg.path("downstream") / "wikiann"
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for lang in sorted(p.name for p in raw.iterdir() if p.is_dir()):
        words = load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt")
        exact = {}
        for i, w in enumerate(words):
            exact.setdefault(w, i)
        normalized = vocab_index(words, lang)
        frames = []
        for split in SPLITS:
            d = pd.read_parquet(raw / lang / f"{split}.parquet", columns=["tokens", "ner_tags"])
            tokens = [list(t) for t in d["tokens"]]
            tags = [np.asarray(t, dtype=np.int8) for t in d["ner_tags"]]
            rows = [match_tokens(t, exact, normalized, lang, normalize).astype(np.int32)
                    for t in tokens]
            frames.append(pd.DataFrame({"split": split, "tokens": tokens, "tags": tags,
                                        "rows": rows}))
            flat = np.concatenate(rows)
            n_exact = sum(sum(tok in exact for tok in t) for t in tokens)
            stats.append({"lang": lang, "split": split, "sentences": len(tokens),
                          "tokens": len(flat), "entities": sum(len(entities(t)) for t in tags),
                          "exact_share": n_exact / len(flat),
                          "normalized_share": ((flat >= 0).sum() - n_exact) / len(flat),
                          "unmatched_share": (flat < 0).mean()})
            s = stats[-1]
            log.info("%s %s: %d sentences, %d tokens, %d entities; %.0f%% exact, %.0f%% "
                     "normalized, %.0f%% unmatched", lang, split, s["sentences"], s["tokens"],
                     s["entities"], 100 * s["exact_share"], 100 * s["normalized_share"],
                     100 * s["unmatched_share"])
        pd.concat(frames, ignore_index=True).to_parquet(out / f"{lang}.parquet", index=False)
    pd.DataFrame(stats).to_csv(out / "stats.csv", index=False, lineterminator="\n",
                               float_format="%.4f")


if __name__ == "__main__":
    main()
