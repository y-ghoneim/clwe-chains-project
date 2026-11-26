"""English concept candidates -> <list>/candidates.parquet. Runs after 01_wiktionary.py.

in_sources holds normalized English forms found in PanLex (v0 in-vocab table) or as Wiktionary
headwords. PanLex stores each form as the rank of its most frequent spelling ("War" for "war"), so
the forms are matched, not the ranks. en_key is that PanLex rank for each concept: Tasks 5 and 6
join PanLex rows on it, while en_rank is the concept word's own vocab row.
"""

from __future__ import annotations

import logging

import nltk
import numpy as np
import pandas as pd

from clwe_chains import load_config
from clwe_chains.concepts import english_candidates, wordnet_content
from clwe_chains.text import load_vocab, normalize, vocab_index

log = logging.getLogger("candidates")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    nltk.data.path.insert(0, str(cfg.path("nltk_data")))
    out = cfg.path("list")
    vocab = load_vocab(cfg.path("v0_embeddings") / "en.vocab.txt")
    index = vocab_index(vocab, "en")

    pl_ranks = set(pd.read_parquet(cfg.path("v0_panlex") / "en.parquet")["word"].unique().tolist())
    pl_forms = {form for form, rank in index.items() if rank in pl_ranks}
    headwords = pd.read_parquet(out / "wiktionary_en.parquet", columns=["en"])["en"].unique()
    wk_forms = {normalize(h, "en") for h in headwords}
    log.info("English forms: %d in PanLex, %d Wiktionary headwords", len(pl_forms), len(wk_forms))

    cands = english_candidates(vocab, pl_forms | wk_forms, wordnet_content,
                               cfg["list"]["n_candidates"])
    cands["en_key"] = cands["en"].map(index).astype(np.int32)
    cands.to_parquet(out / "candidates.parquet", index=False)
    last = cands.iloc[-1]
    log.info("%d candidates; last %r at rank %d; %d with en_key != en_rank", len(cands),
             last["en"], last["en_rank"], int((cands["en_key"] != cands["en_rank"]).sum()))


if __name__ == "__main__":
    main()
