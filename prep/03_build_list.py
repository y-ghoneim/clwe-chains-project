"""Per-language word choice, coverage report and (with a threshold) the shared list.

Runs after 02_candidates.py. For each node language L other than en:
  candidates (PanLex -> Wiktionary -> MT file if present) -> one-to-one choice -> round trip,
written to <list>/per_lang/<L>.parquet. Then <list>/coverage.csv, and <list>/shared_list.parquet
when list.coverage_threshold is set. English is the identity (word = en).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from anyascii import anyascii

from clwe_chains import load_config
from clwe_chains.config import node_languages
from clwe_chains.select import choose_one_to_one, reverse_top, round_trip
from clwe_chains.shared_list import assemble, coverage_table
from clwe_chains.sources import (
    merge_candidates,
    mt_candidates,
    panlex_candidates,
    wiktionary_candidates,
    wiktionary_reverse,
)
from clwe_chains.text import load_vocab, normalize, vocab_index

log = logging.getLogger("build_list")


def identical_flags(words: pd.Series, en: pd.Series, lang: str) -> np.ndarray:
    """Whether each word, normalized for its language, equals the concept's English word. Used by
    phases 3-4 to drop test concepts; from phase 5 on, text.identical_pairs compares source with
    target instead (brief 5.2; decisions.md 2026-11-16)."""
    return (words.map(lambda w: normalize(w, lang)) == en).to_numpy(dtype=bool)


def build_language(lang: str, cfg, concepts, en_pl, en_index, wk) -> tuple[pd.DataFrame, dict]:
    vocab = load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt")
    index = vocab_index(vocab, lang)
    pl_path = cfg.path("v0_panlex") / f"{lang}.parquet"
    lang_pl = (pd.read_parquet(pl_path) if pl_path.exists()
               else pd.DataFrame({"meaning": pd.Series([], dtype="int64"),
                                  "word": pd.Series([], dtype="int32")}))
    codes = cfg["list"]["wikt_codes"].get(lang, [lang])

    pl = panlex_candidates(en_pl, lang_pl, concepts)
    wk_c, drops = wiktionary_candidates(wk, codes, concepts, index, lang)
    mt = mt_candidates(cfg.path("list") / "mt" / f"{lang}.tsv", concepts, index, lang)
    chosen = choose_one_to_one(merge_candidates(pl, wk_c, mt))

    top = reverse_top(lang_pl, en_pl, wiktionary_reverse(wk, codes, en_index, index, lang),
                      k=cfg["list"]["round_trip_top_k"])
    chosen["round_trip"] = round_trip(chosen, concepts, top).to_numpy()
    en = chosen["concept_id"].map(concepts.set_index("concept_id")["en"])
    words = chosen["word_idx"].map(lambda i: vocab[i])
    out = pd.DataFrame({
        "concept_id": chosen["concept_id"], "word_idx": chosen["word_idx"], "word": words,
        "source": chosen["source"], "round_trip": chosen["round_trip"].astype(bool),
        "identical": identical_flags(words, en, lang),
        "translit": words.map(lambda w: anyascii(w).lower()),
    })
    return out.sort_values("concept_id", ignore_index=True), drops


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    out = cfg.path("list")
    (out / "per_lang").mkdir(parents=True, exist_ok=True)
    concepts = pd.read_parquet(out / "candidates.parquet")
    en_pl = pd.read_parquet(cfg.path("v0_panlex") / "en.parquet")
    en_index = vocab_index(load_vocab(cfg.path("v0_embeddings") / "en.vocab.txt"), "en")
    wk = pd.read_parquet(out / "wiktionary_en.parquet", columns=["en", "code", "word"])

    per_lang, drops = {}, {}
    for lang in node_languages(cfg):
        if lang == "en":
            continue
        df, drops[lang] = build_language(lang, cfg, concepts, en_pl, en_index, wk)
        df.to_parquet(out / "per_lang" / f"{lang}.parquet", index=False)
        per_lang[lang] = df
        log.info("%s: %d chosen, %d pass the round trip, %d multiword, %d oov", lang, len(df),
                 int(df["round_trip"].sum()), drops[lang]["n_multiword"], drops[lang]["n_oov"])

    lc = cfg["list"]
    cov = coverage_table(per_lang, lc["coverage_top"], drops, n_candidates=len(concepts))
    cov.to_csv(out / "coverage.csv", index=False, lineterminator="\n", float_format="%.4f")
    by_cov = cov.sort_values("coverage")[["lang", "coverage", "coverage_all"]]
    log.info("coverage@%d:\n%s", lc["coverage_top"], by_cov.to_string(index=False))
    if lc["coverage_threshold"] is not None:
        wide, kept = assemble(per_lang, concepts, lc["coverage_threshold"], lc["coverage_top"])
        wide.to_parquet(out / "shared_list.parquet", index=False)
        log.info("threshold %.2f keeps %d languages, %d concepts; dropped: %s",
                 lc["coverage_threshold"], len(kept), len(wide), sorted(set(per_lang) - set(kept)))


if __name__ == "__main__":
    main()
