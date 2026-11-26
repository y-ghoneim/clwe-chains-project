"""Per-concept sense agreement of the shared list -> <list>/sense.parquet. Runs after
03_build_list.py with a threshold set (needs shared_list.parquet).

Columns: concept_id, n_evaluable (language pairs whose words both appear in a Wiktionary sense
table of the concept's English word), n_coherent (pairs sharing a table), coherence. The
robustness check reports headline results on all test concepts and on the high-agreement subset
(brief 5.7).
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from clwe_chains import load_config
from clwe_chains.sense import sense_coherence, wiktionary_glosses
from clwe_chains.text import load_vocab, vocab_index

log = logging.getLogger("sense")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    lst = cfg.path("list")
    path = lst / "shared_list.parquet"
    if not path.exists():
        sys.exit(f"{path} not found: set list.coverage_threshold and run prep/03_build_list.py")
    wide = pd.read_parquet(path)
    langs = sorted(c for c in wide.columns
                   if c not in ("concept_id", "en") and not c.endswith("_source"))
    concepts = pd.read_parquet(lst / "candidates.parquet")
    wk = pd.read_parquet(lst / "wiktionary_en.parquet", columns=["en", "sense", "code", "word"])

    words, glosses = wide[["concept_id"]].copy(), {}
    for lang in langs:
        per = pd.read_parquet(lst / "per_lang" / f"{lang}.parquet",
                              columns=["concept_id", "word_idx"])
        words = words.merge(per.rename(columns={"word_idx": lang}), on="concept_id")
        index = vocab_index(load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt"), lang)
        glosses[lang] = wiktionary_glosses(wk, cfg["list"]["wikt_codes"].get(lang, [lang]),
                                           concepts, index, lang)
    out = sense_coherence(words, glosses)
    out.to_parquet(lst / "sense.parquet", index=False)
    pooled = out["n_coherent"].sum() / out["n_evaluable"].sum()
    q = out["coherence"].quantile([0.1, 0.25, 0.5]).round(3).tolist()
    log.info("%d concepts, %d languages; pooled coherence %.3f; per-concept quantiles 10/25/50%% "
             "%s; %d concepts with no evaluable pair", len(out), len(langs), pooled, q,
             int((out["n_evaluable"] == 0).sum()))


if __name__ == "__main__":
    main()
