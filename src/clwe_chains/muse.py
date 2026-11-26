"""MUSE ground-truth dictionaries (Conneau et al. 2018), for the external anchor
(experiments/muse-anchor/).

A file has one "source target" pair per line, separated by whitespace; a source word has one line
per translation. A word's row in the (cased) fastText vocabulary is its exact form, or, if that
is missing, the most frequent row with the same normalized form (text.normalize, lower case)
(the approved rule). Secondary gold: a translation counts at any of its casings.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clwe_chains.text import normalize


class Vocab:
    """One language's vocabulary: exact form -> first row; normalized form -> every row, most
    frequent first."""

    def __init__(self, words: list[str], lang: str):
        self.lang = lang
        self.exact: dict[str, int] = {}
        self.forms: dict[str, list[int]] = {}
        for i, w in enumerate(words):
            self.exact.setdefault(w, i)
            self.forms.setdefault(normalize(w, lang), []).append(i)

    def row(self, word: str) -> int | None:
        """The exact form's row, else the most frequent casing's, else None."""
        if word in self.exact:
            return self.exact[word]
        rows = self.forms.get(normalize(word, self.lang))
        return rows[0] if rows else None

    def casings(self, word: str) -> list[int]:
        """Every row with the word's normalized form."""
        return self.forms.get(normalize(word, self.lang), [])


def read_muse(path: Path) -> pd.DataFrame:
    """src, tgt columns, one row per line with exactly two fields, in file order."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            rows.append(parts)
    return pd.DataFrame(rows, columns=["src", "tgt"])


def train_rows(d: pd.DataFrame, vs: Vocab, vt: Vocab) -> pd.DataFrame:
    """a, b row pairs of the pairs with both words in the vocabularies, duplicates dropped."""
    out = pd.DataFrame({"a": [vs.row(w) for w in d["src"]], "b": [vt.row(w) for w in d["tgt"]]})
    return out.dropna().astype(int).drop_duplicates().reset_index(drop=True)


def query_rows(d: pd.DataFrame, vs: Vocab, vt: Vocab
               ) -> tuple[list[int], list[list[int]], list[list[int]], dict]:
    """Query rows (one per source word in the vocabulary with a translation in it, first-seen
    order), each query's gold rows (one row per translation, the main gold) and its gold at any
    casing (secondary), and coverage counts."""
    gold: dict[int, set[int]] = {}
    anycase: dict[int, set[int]] = {}
    for w, g in zip(d["src"], d["tgt"], strict=True):
        q, r = vs.row(w), vt.row(g)
        if q is None or r is None:
            continue
        gold.setdefault(q, set()).add(r)
        anycase.setdefault(q, set()).update(vt.casings(g))
    queries = list(gold)
    return (queries, [sorted(gold[q]) for q in queries], [sorted(anycase[q]) for q in queries],
            {"test_words": int(d["src"].nunique()), "test_words_used": len(queries)})
