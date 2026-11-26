"""Natural bilingual dictionaries per edge, for phase 6 (decisions.md 2026-11-16).

Each edge a-b trains on the word pairs its sources attest: PanLex in-vocab pairs (words sharing a
meaning), plus English Wiktionary translations when one side is English. Frames have int64
columns a and b, vocabulary rows of the edge's first and second language, as PanLex stores them
(the most frequent casing of each form).

Test words are removed from every edge, per fold. Joined direct adds the s-t pairs obtained by
joining s-x and x-t on the hub word, optionally filtered to mutual best: a pair is kept when each
word is the other's most-connected partner, connections counted as distinct linking hub words (a
variant of inverse consultation, Tanaka & Umemura 1994).
"""

from __future__ import annotations

import pandas as pd

COLUMNS = ["a", "b"]


def _frame(a, b) -> pd.DataFrame:
    return pd.DataFrame({"a": pd.Series(a, dtype="int64"), "b": pd.Series(b, dtype="int64")})


def _distinct(df: pd.DataFrame) -> pd.DataFrame:
    out = df[COLUMNS].drop_duplicates().sort_values(COLUMNS, ignore_index=True)
    return _frame(out["a"].to_numpy(), out["b"].to_numpy())


def panlex_pairs(pl_a: pd.DataFrame, pl_b: pd.DataFrame) -> pd.DataFrame:
    """Distinct (a word, b word) pairs sharing a PanLex meaning."""
    m = pl_a[["meaning", "word"]].merge(pl_b[["meaning", "word"]], on="meaning",
                                        suffixes=("_a", "_b"))
    return _distinct(_frame(m["word_a"].to_numpy(), m["word_b"].to_numpy()))


def edge_dictionary(pl_a: pd.DataFrame, pl_b: pd.DataFrame,
                    wk_pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """PanLex pairs plus wk_pairs (columns a, b; English Wiktionary for edges with en)."""
    parts = [panlex_pairs(pl_a, pl_b)]
    if wk_pairs is not None:
        parts.append(_frame(wk_pairs["a"].to_numpy(), wk_pairs["b"].to_numpy()))
    return _distinct(pd.concat(parts, ignore_index=True))


def remove_test(d: pd.DataFrame, test_a: set[int], test_b: set[int]) -> pd.DataFrame:
    """d without pairs whose a word is in test_a or b word in test_b (rows incl. other casings)."""
    keep = ~d["a"].isin(list(test_a)) & ~d["b"].isin(list(test_b))
    return d[keep].reset_index(drop=True)


def join_counts(sx: pd.DataFrame, xt: pd.DataFrame) -> pd.DataFrame:
    """s-t pairs joined on the hub word: columns a, b, n (distinct hub words linking them)."""
    j = sx[COLUMNS].rename(columns={"b": "w"}).merge(
        xt[COLUMNS].rename(columns={"a": "w"}), on="w")
    out = j.groupby(COLUMNS)["w"].nunique().rename("n").reset_index()
    return out.astype({"a": "int64", "b": "int64", "n": "int64"})


def mutual_best(j: pd.DataFrame) -> pd.DataFrame:
    """Rows of join_counts whose n is the maximum for both their a word and their b word (ties
    kept)."""
    if j.empty:
        return _frame([], [])
    best_a = j.groupby("a")["n"].transform("max")
    best_b = j.groupby("b")["n"].transform("max")
    kept = j[(j["n"] == best_a) & (j["n"] == best_b)]
    return _distinct(kept)


def joined_dictionary(direct: pd.DataFrame, counts: pd.DataFrame, test_s: set[int],
                      test_t: set[int], filtered: bool) -> pd.DataFrame:
    """direct (already test-removed) plus the joined pairs left after removing test words, all
    of them or (filtered) their mutual best. counts can come from full edges: removing whole a-
    and b-rows doesn't change n for the other pairs, only who is whose best."""
    j = counts[~counts["a"].isin(list(test_s)) & ~counts["b"].isin(list(test_t))]
    extra = mutual_best(j) if filtered else j[COLUMNS]
    return _distinct(pd.concat([direct[COLUMNS], extra], ignore_index=True))


def dict_sizes(d: pd.DataFrame) -> dict:
    return {"pairs": len(d), "a_words": int(d["a"].nunique()), "b_words": int(d["b"].nunique())}
