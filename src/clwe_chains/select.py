"""One word per concept per language, one-to-one, plus the round-trip check (brief section 5.2).

Greedy in concept frequency order (ascending concept_id): each concept takes its best unused
candidate by (PanLex score desc, Wiktionary-attested, MT-attested, more frequent word). A word
already taken by a more frequent concept is skipped, so big/large -> grande goes to "big".

English ranks here are en_key values: the rank PanLex stores for a form's most frequent
spelling (prep/02_candidates.py). The round trip compares in that space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def choose_one_to_one(cands: pd.DataFrame) -> pd.DataFrame:
    """(concept_id, word_idx, source, back) for each concept that gets a word."""
    c = cands.sort_values(["concept_id", "n_pl", "wk", "mt", "word_idx"],
                          ascending=[True, False, False, False, True], kind="mergesort")
    used, taken, keep = set(), set(), []
    for i, cid, w in zip(range(len(c)), c["concept_id"].to_numpy(), c["word_idx"].to_numpy(),
                         strict=True):
        if cid in taken or w in used:
            continue
        taken.add(cid)
        used.add(w)
        keep.append(i)
    out = c.iloc[keep].reset_index(drop=True)
    source = np.where(out["n_pl"] > 0, "panlex", np.where(out["wk"], "wiktionary", "mt"))
    return pd.DataFrame({"concept_id": out["concept_id"].to_numpy(dtype=np.int32),
                         "word_idx": out["word_idx"].to_numpy(dtype=np.int32),
                         "source": pd.Series(source, dtype=str),
                         "back": out["back"].to_numpy(dtype=str)})


def reverse_top(lang_pl: pd.DataFrame, en_pl_full: pd.DataFrame, wk_reverse: pd.DataFrame,
                k: int) -> pd.DataFrame:
    """(word_idx, en_key): each L word's k best English translations over all of PanLex and
    Wiktionary, best first.

    Order: shared PanLex meanings (desc), then Wiktionary-attested, then the lower en_key.
    """
    pl = (lang_pl.merge(en_pl_full, on="meaning", suffixes=("_l", "_en"))
          .groupby(["word_l", "word_en"]).size().rename("n_pl").reset_index()
          .rename(columns={"word_l": "word_idx", "word_en": "en_key"}))
    wk = wk_reverse[["word_idx", "en_key"]].drop_duplicates().assign(wk=True)
    both = pl.merge(wk, on=["word_idx", "en_key"], how="outer")
    both["n_pl"] = both["n_pl"].fillna(0)
    both["wk"] = both["wk"].fillna(False).astype(bool)
    best = (both.sort_values(["word_idx", "n_pl", "wk", "en_key"],
                             ascending=[True, False, False, True], kind="mergesort")
            .groupby("word_idx", sort=False).head(k))
    return pd.DataFrame({"word_idx": best["word_idx"].to_numpy(dtype=np.int64),
                         "en_key": best["en_key"].to_numpy(dtype=np.int64)})


def round_trip(chosen: pd.DataFrame, concepts: pd.DataFrame, top: pd.DataFrame) -> pd.Series:
    """True where the concept's own word is among the chosen word's top English translations
    (`top`, from reverse_top).

    An MT entry with no reverse information passes iff its back-translation equals the concept's
    English word.
    """
    c = chosen.merge(concepts[["concept_id", "en", "en_key"]], on="concept_id", how="left")
    pairs = set(zip(top["word_idx"].tolist(), top["en_key"].tolist(), strict=True))
    ok = pd.Series([(w, e) in pairs for w, e in zip(c["word_idx"], c["en_key"], strict=True)],
                   index=c.index, dtype=bool)
    mt_only = (c["source"] == "mt") & ~c["word_idx"].isin(top["word_idx"])
    ok = ok.where(~mt_only, c["back"].eq(c["en"]))
    return pd.Series(ok.to_numpy(dtype=bool), index=chosen.index)
