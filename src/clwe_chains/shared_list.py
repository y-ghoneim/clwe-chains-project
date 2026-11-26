"""Coverage per language and assembly of the shared concept list (brief section 5.2).

A language covers a concept when its chosen word passed the round trip. Coverage is the share of
the top `top` candidates (concept_id < top) covered. With a threshold set, languages below it are
dropped and the list keeps the concepts covered in every remaining language.
"""

from __future__ import annotations

import pandas as pd
from rapidfuzz.distance import Levenshtein

from clwe_chains.text import normalize

SOURCES = ("panlex", "wiktionary", "mt")


def _passed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["round_trip"].astype(bool)] if len(df) else df


def _coverage(df: pd.DataFrame, top: int) -> float:
    p = _passed(df)
    return float((p["concept_id"] < top).sum() / top) if len(p) else 0.0


def coverage_table(per_lang: dict[str, pd.DataFrame], top: int, drops: dict[str, dict],
                   n_candidates: int) -> pd.DataFrame:
    """One row per language: chosen, passed, coverage@top, coverage over all, source shares,
    and the Wiktionary forms dropped as multiword or out of vocabulary."""
    rows = []
    for lang, df in sorted(per_lang.items()):
        p = _passed(df)
        row = {"lang": lang, "chosen": len(df), "passed_round_trip": len(p),
               "coverage": _coverage(df, top), "coverage_all": len(p) / n_candidates}
        for s in SOURCES:
            row[f"share_{s}"] = float((p["source"] == s).mean()) if len(p) else 0.0
        row["n_multiword"] = drops[lang]["n_multiword"]
        row["n_oov"] = drops[lang]["n_oov"]
        rows.append(row)
    return pd.DataFrame(rows)


def assemble(per_lang: dict[str, pd.DataFrame], concepts: pd.DataFrame, threshold: float | None,
             top: int) -> tuple[pd.DataFrame, list[str]]:
    """(wide list, kept languages). threshold None = report only: empty frame, all languages."""
    langs = sorted(per_lang)
    if threshold is None:
        return pd.DataFrame(), langs
    kept = [lang for lang in langs if _coverage(per_lang[lang], top) >= threshold]
    wide = concepts[["concept_id", "en"]]
    for lang in kept:
        p = _passed(per_lang[lang])[["concept_id", "word", "source"]]
        wide = wide.merge(p.rename(columns={"word": lang, "source": f"{lang}_source"}),
                          on="concept_id")
    return wide.sort_values("concept_id", ignore_index=True), kept


def list_languages(wide: pd.DataFrame) -> list[str]:
    """The languages of a built shared list (English plus its language columns), sorted. These
    are the graph's nodes; configs/languages.yaml is only the candidate set the list starts from.
    """
    langs = [c for c in wide.columns if c not in ("concept_id", "en") and not c.endswith("_source")]
    return sorted(["en", *langs])


def cognate(a: str, b: str, threshold: float) -> bool:
    """Normalized Levenshtein similarity of two transliterated forms is at least threshold."""
    return bool(Levenshtein.normalized_similarity(a, b) >= threshold)


def case_support(wide: pd.DataFrame, wk: pd.DataFrame,
                 codes: dict[str, list[str]]) -> pd.DataFrame:
    """Per concept: languages whose chosen word Wiktionary gives under the lowercase headword
    (n_lower), and languages where it does so only under a capitalized headword (n_cap_only).

    wk holds Wiktionary rows (en headword as written, code, word); codes maps each list language to
    its Wiktionary codes. Languages whose word Wiktionary doesn't give (PanLex-only support, whose
    English casing isn't kept) count in neither.
    """
    words = set(wide["en"])
    kinds: dict[tuple[str, str, str], set[str]] = {}
    for h, code, word in zip(wk["en"], wk["code"], wk["word"], strict=True):
        w = normalize(h, "en")
        if w not in words:
            continue
        for lang, cs in codes.items():
            if code in cs:
                kinds.setdefault((w, lang, normalize(word, lang)), set()).add(
                    "lower" if h == w else "cap")
    langs = [c for c in codes if c in wide.columns]
    rows = []
    for cid, w, *ws in wide[["concept_id", "en", *langs]].itertuples(index=False):
        k = [kinds.get((w, lang, normalize(x, lang)), set()) for lang, x in zip(langs, ws,
                                                                               strict=True)]
        rows.append({"concept_id": cid, "n_lower": sum("lower" in s for s in k),
                     "n_cap_only": sum(s == {"cap"} for s in k)})
    return pd.DataFrame(rows)


def capitalized_senses(support: pd.DataFrame, wordnet_lower: dict[int, bool]) -> list[int]:
    """Concepts to drop from the list (decisions.md 2026-11-16; applied from the phase 5 rebuild).

    A concept goes if its English word has no WordNet content lemma spelled in lowercase
    (concepts.wordnet_lowercase: proper nouns, acronyms), or if more of its languages get their
    word only from a capitalized Wiktionary headword than from the lowercase one (may -> Mai,
    turkey -> the country). Applied to the assembled list, after the coverage rule, so the node
    set doesn't change. On the 2026-11-15 list: 42 of 883 concepts
    (analyses/2026-11-16-capitalized-senses/).
    """
    cap = support.set_index("concept_id")
    drop = {int(c) for c, ok in wordnet_lower.items() if not ok}
    drop |= {int(c) for c in cap.index[cap["n_cap_only"] > cap["n_lower"]]}
    return sorted(drop)
