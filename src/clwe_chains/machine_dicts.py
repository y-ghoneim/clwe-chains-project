"""Dictionaries built from machine translations (prep/11_machine_dicts.py).

A machine dictionary for an edge a-b holds (a word, b word) vocab-row pairs: each source word of a
translated into b, and each source word of b translated into a, top output only, cleaned
(mt.clean_hop) and kept if it is a single word in the target vocabulary. "Pivoted" dictionaries
translate through a third language (a -> x -> b). The same layout as prep/08's natural
dictionaries (columns a, b), so stage 1's runner uses them unchanged.
"""

from __future__ import annotations

import unicodedata

import pandas as pd

from clwe_chains.mt import clean_hop
from clwe_chains.text import normalize

COLUMNS = ["a", "b"]


def is_word(text: str) -> bool:
    """Two or more characters, all letters or combining marks (Indic vowel signs are marks)."""
    return len(text) >= 2 and all(unicodedata.category(ch)[0] in "LM" for ch in text)


def frequent_rows(vocab: list[str], n: int) -> list[int]:
    """The first n vocab rows that are words (is_word), most frequent first."""
    out = []
    for i, w in enumerate(vocab):
        if len(out) == n:
            break
        if is_word(w):
            out.append(i)
    return out


def list_rows(words: pd.DataFrame, lang: str) -> dict[int, int]:
    """concept_id -> the row of lang's list word (words.parquet rows with is_word)."""
    q = words[(words["lang"] == lang) & words["is_word"]]
    return {int(c): int(r) for c, r in zip(q["concept_id"], q["row"], strict=True)}


def source_rows(words: pd.DataFrame, lang: str, vocab: list[str], n_frequent: int) -> list[int]:
    """The rows lang's dictionaries are built from: its list words, plus its n_frequent most
    frequent words (0 = list words only), without repeats, in row order."""
    rows = set(list_rows(words, lang).values()) | set(frequent_rows(vocab, n_frequent))
    return sorted(rows)


def output_row(text: str | None, lang: str, index: dict[str, int]) -> int | None:
    """The target vocab row of an MT output: cleaned, a single word, normalized, in the vocab."""
    if not text:
        return None
    w = clean_hop(text, lang)
    if not w or " " in w:
        return None
    return index.get(normalize(w, lang))


def translated_pairs(src_rows: list[int], outputs: list[str | None], tgt: str,
                     index: dict[str, int]) -> pd.DataFrame:
    """(source row, target row) for every output that maps to a target vocab row."""
    a, b = [], []
    for r, out in zip(src_rows, outputs, strict=True):
        row = output_row(out, tgt, index)
        if row is not None:
            a.append(r)
            b.append(row)
    return pd.DataFrame({"a": pd.Series(a, dtype="int64"), "b": pd.Series(b, dtype="int64")})


def union_dictionary(ab: pd.DataFrame, ba: pd.DataFrame) -> pd.DataFrame:
    """An edge's dictionary (a rows, b rows) from both translation directions: ab = a words
    translated into b (a, b); ba = b words translated into a (b's rows in column a, a's in b)."""
    flipped = pd.DataFrame({"a": ba["b"].to_numpy(), "b": ba["a"].to_numpy()})
    both = pd.concat([ab[COLUMNS], flipped], ignore_index=True).drop_duplicates()
    return both.sort_values(COLUMNS, ignore_index=True).astype("int64")


def list_dictionary(words: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """The shared list as an edge dictionary: (a's list word, b's list word) per concept both
    have."""
    ra, rb = list_rows(words, a), list_rows(words, b)
    both = sorted(set(ra) & set(rb))
    return pd.DataFrame({"a": pd.Series([ra[c] for c in both], dtype="int64"),
                         "b": pd.Series([rb[c] for c in both], dtype="int64")})


def edges_for(pairs: pd.DataFrame, chain_edges: bool) -> list[tuple[str, str]]:
    """Distinct edges in route order: s-t, and s-hub, hub-t when chain_edges."""
    edges = []
    for s, t, x in zip(pairs["s"], pairs["t"], pairs["hub"], strict=True):
        edges.append((s, t))
        if chain_edges:
            edges += [(s, x), (x, t)]
    return list(dict.fromkeys(edges))
