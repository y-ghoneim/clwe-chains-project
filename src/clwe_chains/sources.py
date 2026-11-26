"""Translation candidates per language from the tiered sources (brief section 5.2).

1. PanLex: v0's in-vocab tables, joined with English on meaning id; n_pl = shared meanings.
2. Wiktionary: English-edition translation tables (wiktextract dump).
3. MT: an optional TSV per language, for concepts still missing.

Every function returns (concept_id, word_idx) rows, word_idx being the rank of the translation's
most frequent surface form in the language's fastText vocab. PanLex English rows are matched to
concepts on en_key, the rank PanLex stores for the concept's form (see prep/02_candidates.py).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clwe_chains.text import normalize

WK_COLUMNS = ["en", "pos", "sense", "code", "word"]
MERGED_COLUMNS = ["concept_id", "word_idx", "n_pl", "wk", "mt", "back"]


def read_wiktextract(path: Path) -> pd.DataFrame:
    """Translations of English entries, from the entry and from each sense."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            # Cheap string checks first: most of the 10M+ lines are other languages.
            if '"lang_code": "en"' not in line or '"translations"' not in line:
                continue
            e = json.loads(line)
            if e.get("lang_code") != "en":
                continue
            tr = list(e.get("translations", []))
            for s in e.get("senses", []):
                tr += s.get("translations", [])
            for t in tr:
                if t.get("word") and t.get("code"):
                    rows.append((e["word"], e.get("pos", ""), t.get("sense", ""), t["code"],
                                 t["word"]))
    return pd.DataFrame(rows, columns=WK_COLUMNS).drop_duplicates(ignore_index=True)


def _pairs(concept_id, word_idx) -> pd.DataFrame:
    return pd.DataFrame({"concept_id": np.asarray(concept_id, dtype=np.int32),
                         "word_idx": np.asarray(word_idx, dtype=np.int32)})


def panlex_candidates(en_pl: pd.DataFrame, lang_pl: pd.DataFrame,
                      concepts: pd.DataFrame) -> pd.DataFrame:
    """(concept_id, word_idx, n_pl): L words sharing PanLex meanings with each concept."""
    en = en_pl.merge(concepts[["concept_id", "en_key"]], left_on="word", right_on="en_key")
    joined = en[["meaning", "concept_id"]].merge(lang_pl, on="meaning")
    out = (joined.groupby(["concept_id", "word"]).size().rename("n_pl").reset_index()
           .rename(columns={"word": "word_idx"}))
    res = _pairs(out["concept_id"], out["word_idx"])
    res["n_pl"] = out["n_pl"].to_numpy(dtype=np.int32)
    return res


def wiktionary_candidates(wk: pd.DataFrame, codes: list[str], concepts: pd.DataFrame,
                          index: dict[str, int], lang: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """(concept_id, word_idx) from Wiktionary rows in `codes`, plus drop counts.

    n_rows counts the translation rows of candidate concepts; a form with whitespace is
    multiword (it can never match a fastText token); a normalized form not in index is OOV.
    """
    concept_of = dict(zip(concepts["en"], concepts["concept_id"], strict=True))
    rows = wk[wk["code"].isin(codes)]
    cid = rows["en"].map(lambda h: concept_of.get(normalize(h, "en")))
    rows, cid = rows[cid.notna()], cid[cid.notna()]
    multi = rows["word"].str.contains(r"\s", regex=True)
    idx = rows["word"][~multi].map(lambda w: index.get(normalize(w, lang)))
    keep = idx.notna()
    counts = {"n_rows": len(rows), "n_multiword": int(multi.sum()), "n_oov": int((~keep).sum())}
    out = _pairs(cid[~multi][keep], idx[keep]).drop_duplicates(ignore_index=True)
    return out, counts


def mt_candidates(path: Path, concepts: pd.DataFrame, index: dict[str, int],
                  lang: str) -> pd.DataFrame:
    """(concept_id, word_idx, back) from an optional MT TSV (concept_id, word, back)."""
    if not Path(path).exists():
        out = _pairs([], [])
        out["back"] = pd.Series([], dtype=str)
        return out
    df = pd.read_csv(path, sep="\t", dtype={"concept_id": np.int32, "word": str, "back": str},
                     keep_default_na=False)
    df = df[df["concept_id"].isin(concepts["concept_id"])]
    idx = df["word"].map(lambda w: index.get(normalize(w, lang)))
    df, idx = df[idx.notna()], idx[idx.notna()]
    out = _pairs(df["concept_id"], idx)
    out["back"] = df["back"].to_numpy(dtype=str)
    return out.drop_duplicates(["concept_id", "word_idx"], ignore_index=True)


def merge_candidates(pl: pd.DataFrame, wk: pd.DataFrame, mt: pd.DataFrame) -> pd.DataFrame:
    """One row per (concept_id, word_idx) with the sources that attest it."""
    key = ["concept_id", "word_idx"]
    m = pl[[*key, "n_pl"]].merge(wk[key].assign(wk=True), on=key, how="outer")
    m = m.merge(mt[[*key, "back"]].assign(mt=True), on=key, how="outer")
    out = _pairs(m["concept_id"], m["word_idx"])
    out["n_pl"] = m["n_pl"].fillna(0).to_numpy(dtype=np.int32)
    out["wk"] = m["wk"].fillna(False).to_numpy(dtype=bool)
    out["mt"] = m["mt"].fillna(False).to_numpy(dtype=bool)
    out["back"] = m["back"].fillna("").to_numpy(dtype=str) if len(m) else pd.Series([], dtype=str)
    return out.sort_values(key, ignore_index=True)[MERGED_COLUMNS]


def wiktionary_reverse(wk: pd.DataFrame, codes: list[str], en_index: dict[str, int],
                       index: dict[str, int], lang: str) -> pd.DataFrame:
    """(word_idx, en_key) for every Wiktionary pair in this language whose headword is in the
    English vocab, not only the candidates. Used by the list build's round trip (prep/03) and as
    the English Wiktionary side of phase 6's en-L dictionaries."""
    rows = wk[wk["code"].isin(codes) & ~wk["word"].str.contains(r"\s", regex=True)]
    en_key = rows["en"].map(lambda h: en_index.get(normalize(h, "en")))
    word_idx = rows["word"].map(lambda w: index.get(normalize(w, lang)))
    ok = en_key.notna() & word_idx.notna()
    return pd.DataFrame({"word_idx": word_idx[ok].astype(np.int64),
                         "en_key": en_key[ok].astype(np.int64)}).drop_duplicates()
