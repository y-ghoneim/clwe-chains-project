"""Gold answers for BLI evaluation (brief 5.7).

Main setting: a test concept's correct answers in target language t are its list word, plus every
t word sharing a PanLex meaning with the source word, plus (for an English source) Wiktionary's
translations of the English headword into t. This is the MUSE convention: synonyms aren't
penalized. The secondary setting uses the list word alone.

The source word is identified by its PanLex rank: en_key for English (the rank PanLex stores for
the form, prep/02_candidates.py), the list word_idx for other languages.

Each answer is one vocabulary row, the most frequent casing of its form. From phase 5 on,
expand_gold adds the other casings (fastText is cased); phases 3-4 ran without it.
"""

from __future__ import annotations

import pandas as pd


def gold_sets(list_word: pd.Series, src_key: pd.Series, pl_src: pd.DataFrame,
              pl_tgt: pd.DataFrame, wk_pairs: pd.DataFrame | None) -> dict[int, list[int]]:
    """concept_id -> sorted target rows: list word, PanLex translations, Wiktionary rows."""
    gold = {int(c): {int(w)} for c, w in list_word.items()}
    keys = pd.DataFrame({"concept_id": src_key.index, "src": src_key.to_numpy()})
    src_m = pl_src.merge(keys, left_on="word", right_on="src")[["concept_id", "meaning"]]
    pairs = src_m.merge(pl_tgt, on="meaning")[["concept_id", "word"]].drop_duplicates()
    for c, w in zip(pairs["concept_id"], pairs["word"], strict=True):
        if int(c) in gold:
            gold[int(c)].add(int(w))
    if wk_pairs is not None:
        for c, w in zip(wk_pairs["concept_id"], wk_pairs["word_idx"], strict=True):
            if int(c) in gold:
                gold[int(c)].add(int(w))
    return {c: sorted(ws) for c, ws in gold.items()}


def with_variants(rows: list[int], variants: dict[int, list[int]]) -> list[int]:
    """rows plus every other casing of each (variants from text.variant_rows), sorted."""
    out = set(rows)
    for r in rows:
        out.update(variants.get(r, ()))
    return sorted(out)


def expand_gold(gold: dict[int, list[int]], variants: dict[int, list[int]]) -> dict[int, list[int]]:
    """Every gold set with all casings of its answers (decisions.md 2026-11-16; from phase 5 on).

    gold_sets itself is unchanged, so phases 3-4 reproduce at any commit."""
    return {c: with_variants(ws, variants) for c, ws in gold.items()}


def gold_stats(gold: dict[int, list[int]]) -> dict[str, float]:
    """Accepted answers per query: n, mean, median, max (reported next to MRR from phase 5 on)."""
    sizes = sorted(len(ws) for ws in gold.values())
    n = len(sizes)
    mid = (sizes[(n - 1) // 2] + sizes[n // 2]) / 2
    return {"n": n, "mean": sum(sizes) / n, "median": float(mid), "max": sizes[-1]}
