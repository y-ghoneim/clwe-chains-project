"""Test items for the MT pivot experiment -> <phase6>/<stage>/mt_items.parquet.

Runs after 07_phase6_list.py --stage <stage>. One row per pair direction and eligible concept: the
source list word as a string, and every gold answer as a string (main gold: list word, PanLex
translations, English Wiktionary when en is an end, all casings; decisions.md 2026-11-17). MT
systems read and write strings, so the runner needs no vocab files. Columns: s, t (the pair as in
pairs.csv), src, tgt (the direction), concept_id, src_word, gold (list of strings).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from clwe_chains import load_config
from clwe_chains.text import load_vocab

log = logging.getLogger("phase6_mt_items")


def items(folds: pd.DataFrame, words: pd.DataFrame, gold: pd.DataFrame,
          vocab: dict[str, list[str]]) -> pd.DataFrame:
    """Both directions of every (s, t, concept) in folds, with strings from the vocab rows."""
    query = {(lng, int(c)): int(r) for lng, c, r, w in
             zip(words["lang"], words["concept_id"], words["row"], words["is_word"], strict=True)
             if w}
    answers = {k: sorted({vocab[k[1]][int(r)] for r in g["row"]}) for k, g in
               gold[gold["kind"] == "main"].groupby(["src", "tgt", "concept_id"])
               for k in [(k[0], k[1], int(k[2]))]}
    rows = []
    for s, t, c in zip(folds["s"], folds["t"], folds["concept_id"], strict=True):
        for src, tgt in ((s, t), (t, s)):
            rows.append({"s": s, "t": t, "src": src, "tgt": tgt, "concept_id": int(c),
                         "src_word": vocab[src][query[src, int(c)]],
                         "gold": answers[src, tgt, int(c)]})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="d", help="a phase6_stages key ('' = stage 1)")
    stage = ap.parse_args().stage
    cfg = load_config()
    out = cfg.path("phase6") / cfg["phase6_stages"][stage]["data"] if stage else cfg.path("phase6")
    folds = pd.read_parquet(out / "folds.parquet")
    words = pd.read_parquet(out / "words.parquet")
    gold = pd.read_parquet(out / "gold.parquet")
    langs = sorted(set(folds["s"]) | set(folds["t"]))
    vocab = {lang: load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt") for lang in langs}
    df = items(folds, words, gold, vocab)
    df.to_parquet(out / "mt_items.parquet", index=False)
    log.info("%d items over %d pair directions", len(df), df.groupby(["src", "tgt"]).ngroups)


if __name__ == "__main__":
    main()
