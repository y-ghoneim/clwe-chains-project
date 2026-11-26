"""Per-concept sense agreement of the shared list, for the robustness check (brief 5.7).

Each language picks its word for a concept through the English word, so an ambiguous anchor can
split across languages ("crane": bird vs machine). Wiktionary groups an English word's
translations into sense tables; two languages' words for a concept are coherent when they appear
in the same table, and the pair is evaluable when both appear in some non-empty table. This is
independent of PanLex, whose per-source meaning ids understate agreement
(analyses/2026-11-15-meaning-anchored/).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from clwe_chains.text import normalize


def wiktionary_glosses(wk: pd.DataFrame, codes: list[str], concepts: pd.DataFrame,
                       index: dict[str, int], lang: str) -> pd.DataFrame:
    """(concept_id, word_idx, sense): the Wiktionary sense tables each word appears in."""
    concept_of = dict(zip(concepts["en"], concepts["concept_id"], strict=True))
    rows = wk[wk["code"].isin(codes) & (wk["sense"] != "")]
    cid = rows["en"].map(lambda h: concept_of.get(normalize(h, "en")))
    widx = rows["word"].map(lambda w: index.get(normalize(w, lang)))
    ok = cid.notna() & widx.notna()
    return (pd.DataFrame({"concept_id": cid[ok].astype(np.int64),
                          "word_idx": widx[ok].astype(np.int64), "sense": rows["sense"][ok]})
            .drop_duplicates().sort_values(["concept_id", "word_idx", "sense"],
                                           ignore_index=True))


def sense_coherence(words: pd.DataFrame, glosses: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per concept: evaluable language pairs, coherent pairs and their ratio (NaN if none).

    words has concept_id and one word_idx column per language in `glosses`.
    """
    langs = sorted(glosses)
    lookup = {lang: g.groupby(["concept_id", "word_idx"])["sense"].apply(set).to_dict()
              for lang, g in glosses.items()}
    rows = []
    for rec in words[["concept_id", *langs]].itertuples(index=False):
        c = rec[0]
        sets = [lookup[lang].get((c, w)) for lang, w in zip(langs, rec[1:], strict=True)]
        ev = co = 0
        for a, b in itertools.combinations(sets, 2):
            if a and b:
                ev += 1
                co += bool(a & b)
        rows.append({"concept_id": c, "n_evaluable": ev, "n_coherent": co,
                     "coherence": co / ev if ev else np.nan})
    return pd.DataFrame(rows)
