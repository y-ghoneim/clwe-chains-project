"""Tests of prep/03_build_list.py's own helpers (phases 1-3 code review, 2026-11-16)."""

import importlib.util
from pathlib import Path

import pandas as pd

PREP = Path(__file__).resolve().parents[1] / "prep" / "03_build_list.py"
_spec = importlib.util.spec_from_file_location("build_list", PREP)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def test_wiktionary_reverse_maps_words_to_english_keys():
    wk = pd.DataFrame({
        "en": ["water", "water", "May", "dog", "dog", "water", "cat"],
        "code": ["de", "de", "de", "de", "fr", "de", "de"],
        "word": ["Wasser", "Wasser", "Mai", "Hund", "chien", "heißes Wasser", "Katze"],
    })
    en_index = {"water": 10, "may": 20, "dog": 30}            # no "cat"
    index = {"wasser": 1, "mai": 2, "hund": 3, "chien": 4}     # de vocab; no "katze"
    out = build.wiktionary_reverse(wk, ["de"], en_index, index, "de")
    got = set(zip(out["word_idx"], out["en_key"], strict=True))
    # duplicates dropped; fr row ignored; multiword and out-of-vocab words dropped; the
    # capitalized headword "May" attaches to the lowercase English key (the casing merge the
    # review found, kept as is for phases 1-4)
    assert got == {(1, 10), (2, 20), (3, 30)} and len(out) == 3


def test_identical_flags_compare_the_normalized_word_with_english():
    words = pd.Series(["Hotel", "hôtel", "radio", "voda", "كتاب"])
    en = pd.Series(["hotel", "hotel", "radio", "water", "كتاب"])
    assert build.identical_flags(words, en, "fr").tolist() == [True, False, True, False, True]
    # Arabic harakat are stripped before comparing
    assert build.identical_flags(pd.Series(["كِتاب"]), pd.Series(["كتاب"]), "ar").tolist() == [True]
