"""Word normalization shared by every source and the fastText vocabularies (brief section 5.2).

NFKC + lowercase everywhere, then strip only the optional marks that ordinary text (and so
fastText) leaves out: Arabic-script harakat and tatweel, Hebrew niqqud and cantillation, Cyrillic
stress accents, Serbo-Croatian tone marks on vowels. Marks that make a different letter (ć č š ž
đ, й ё, ñ ü, Vietnamese tone letters) are never touched.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

_ARABIC_MARKS = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭـ]"
)
_HEBREW_MARKS = re.compile("[֑-ֽֿ-ׇ]")  # everything up to U+05C7 but maqaf
_STRESS = {"́", "̀"}  # acute and grave, after NFD
# Serbo-Croatian tone and length marks, stripped only from a vowel or syllabic r: ć is
# c + U+0301 under NFD, so the base letter decides.
_HR_TONES = {"̀", "́", "̂", "̄", "̏", "̑"}
_HR_TONE_BASES = set("aeiour")

ARABIC_SCRIPT = {"ar", "fa", "ur"}
CYRILLIC_STRESS = {"ru", "uk", "bg", "mk", "be", "kk"}


def _strip_hr_tones(word: str) -> str:
    out, base = [], ""
    for ch in unicodedata.normalize("NFD", word):
        if ch in _HR_TONES and base in _HR_TONE_BASES:
            continue
        if not unicodedata.combining(ch):
            base = ch
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def normalize(word: str, lang: str) -> str:
    """The comparison form of a word in language `lang`."""
    word = unicodedata.normalize("NFKC", word).lower()
    if lang in ARABIC_SCRIPT:
        return _ARABIC_MARKS.sub("", word)
    if lang == "he":
        return _HEBREW_MARKS.sub("", word)
    if lang in CYRILLIC_STRESS:
        nfd = unicodedata.normalize("NFD", word)
        return unicodedata.normalize("NFC", "".join(c for c in nfd if c not in _STRESS))
    if lang == "hr":
        return _strip_hr_tones(word)
    return word


def vocab_index(words: list[str], lang: str) -> dict[str, int]:
    """Normalized form -> rank of its most frequent surface form (fastText is cased)."""
    index: dict[str, int] = {}
    for i, w in enumerate(words):
        index.setdefault(normalize(w, lang), i)
    return index


def variant_rows(words: list[str], lang: str) -> dict[int, list[int]]:
    """Row of a form's most frequent surface form -> every row with that normalized form.

    fastText is cased, and vocab_index keeps only the most frequent casing, so a gold answer
    stored as that row misses retrievals of its other casings (Halk / halk). Forms with a single
    row are omitted. Uses normalize(), i.e. Unicode lowercasing, not Turkish dotted/dotless i rules.
    """
    rows: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        rows.setdefault(normalize(w, lang), []).append(i)
    return {r[0]: r for r in rows.values() if len(r) > 1}


def casing_groups(words: list[str], lang: str) -> dict[int, list[int]]:
    """Every row of a form with several casings -> all rows of that form (variant_rows keys only
    the most frequent casing, so a less frequent list word such as "center" under "Center" found
    no variants there)."""
    rows: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        rows.setdefault(normalize(w, lang), []).append(i)
    return {r: group for group in rows.values() if len(group) > 1 for r in group}


def identical_pairs(words_s: pd.Series, words_t: pd.Series, s: str, t: str) -> pd.Series:
    """concept_id -> whether the source and target words are the same normalized string.

    Brief 5.2: identical-string pairs are excluded from test words. Concepts missing from either
    side are left out.
    """
    both = words_s.index.intersection(words_t.index)
    return pd.Series({c: normalize(words_s[c], s) == normalize(words_t[c], t) for c in both},
                     dtype=bool)


def load_vocab(path: Path) -> list[str]:
    """fastText vocab file, line i = frequency rank i. Files have \\r\\n endings."""
    words = Path(path).read_text(encoding="utf-8").split("\n")
    return words[:-1] if words and words[-1] == "" else words
