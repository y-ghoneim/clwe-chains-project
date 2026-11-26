"""Tests of prep/07_phase6_list.py's own helpers."""

import importlib.util
from pathlib import Path

import pandas as pd

PREP = Path(__file__).resolve().parents[1] / "prep" / "07_phase6_list.py"
_spec = importlib.util.spec_from_file_location("phase6_list", PREP)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def test_eligible_concepts_drops_capitalized_identical_and_missing():
    ws = pd.Series({1: "omo", 2: "radio", 3: "ile", 4: "may"})
    wt = pd.Series({1: "水", 2: "Radio", 4: "五月", 5: "火"})
    assert prep.eligible_concepts(ws, wt, dropped={4}, s="yo", t="de") == [1]


def test_word_rows_marks_the_list_word_and_adds_casings():
    words = pd.Series({7: 3})                       # concept 7 -> vocab row 3
    out = prep.word_rows("de", words, {3: [3, 9]})
    assert out.values.tolist() == [["de", 7, 3, True], ["de", 7, 9, False]]
    assert list(out.columns) == ["lang", "concept_id", "row", "is_word"]


def test_route_edges_in_route_order_without_duplicates():
    spec08 = importlib.util.spec_from_file_location("phase6_dicts",
                                                    PREP.parent / "08_phase6_dicts.py")
    dicts = importlib.util.module_from_spec(spec08)
    spec08.loader.exec_module(dicts)
    pairs = pd.DataFrame({"s": ["mg", "mg", "yo"], "t": ["ar", "zh", "zh"],
                          "hub": ["fr", "fr", "en"]})
    assert dicts.route_edges(pairs, ["mg"]) == [
        ("mg", "ar"), ("mg", "fr"), ("fr", "ar"), ("mg", "en"), ("en", "ar"),
        ("mg", "zh"), ("fr", "zh"), ("en", "zh"), ("yo", "zh"), ("yo", "en")]


def test_route_edges_cover_every_one_and_two_hop_step():
    spec09 = importlib.util.spec_from_file_location("phase6_route_dicts",
                                                    PREP.parent / "09_phase6_route_dicts.py")
    rd = importlib.util.module_from_spec(spec09)
    spec09.loader.exec_module(rd)
    edges = rd.route_edges(["yo", "mg"], ["fr", "en", "ja"])
    assert edges == [("yo", "fr"), ("yo", "en"), ("yo", "ja"), ("mg", "fr"), ("mg", "en"),
                     ("mg", "ja"), ("en", "fr"), ("en", "ja"), ("fr", "ja")]


def test_mt_items_turn_rows_into_strings_both_directions():
    spec10 = importlib.util.spec_from_file_location("phase6_mt_items",
                                                    PREP.parent / "10_phase6_mt_items.py")
    mt = importlib.util.module_from_spec(spec10)
    spec10.loader.exec_module(mt)
    folds = pd.DataFrame({"s": ["tr"], "t": ["ja"], "concept_id": [7], "fold": [0]})
    words = pd.DataFrame({"lang": ["tr", "tr", "ja"], "concept_id": [7, 7, 7], "row": [0, 1, 0],
                          "is_word": [True, False, True]})
    gold = pd.DataFrame({"src": ["tr", "tr", "ja", "tr"], "tgt": ["ja", "ja", "tr", "ja"],
                         "concept_id": [7, 7, 7, 7], "kind": ["main", "main", "main", "list"],
                         "row": [0, 1, 0, 0]})
    vocab = {"tr": ["su", "Su"], "ja": ["水", "お水"]}
    out = mt.items(folds, words, gold, vocab)
    assert out[["src", "tgt", "src_word"]].values.tolist() == [["tr", "ja", "su"],
                                                             ["ja", "tr", "水"]]
    assert out["gold"].tolist() == [["お水", "水"], ["su"]]


def test_casing_groups_cover_every_casing_row():
    from clwe_chains.text import casing_groups, variant_rows
    words = ["the", "Center", "center", "CENTER", "war"]
    g = casing_groups(words, "en")
    assert g[2] == [1, 2, 3] and g[1] == [1, 2, 3] and 0 not in g and 4 not in g
    assert variant_rows(words, "en") == {1: [1, 2, 3]}          # keyed by the first only


def test_forget_drops_only_the_matching_evaluations(tmp_path):
    from clwe_chains.runs import Cache, cache_key, forget
    c = Cache(tmp_path / "evals.jsonl")
    for pair in (("en", "ar"), ("ar", "en"), ("en", "ja")):
        c.put(cache_key("csls", "h", pair, ("thin",), 0, "full"), {"rank": [1]})
    assert forget(tmp_path / "evals.jsonl", lambda k: k["s"] == "en") == 2
    left = list(Cache(tmp_path / "evals.jsonl"))
    assert left == [cache_key("csls", "h", ("ar", "en"), ("thin",), 0, "full")]
    assert forget(tmp_path / "missing.jsonl", lambda k: True) == 0
