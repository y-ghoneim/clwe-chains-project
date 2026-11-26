import gzip
import json

import pandas as pd

from clwe_chains.sources import (
    merge_candidates,
    mt_candidates,
    panlex_candidates,
    read_wiktextract,
    wiktionary_candidates,
)

# en_rank is the concept word's own row; en_key is the rank PanLex stores for its form.
CONCEPTS = pd.DataFrame({"concept_id": [0, 1], "en": ["water", "dog"], "en_rank": [4, 6],
                         "en_key": [7, 6]})
WK_COLS = ["en", "pos", "sense", "code", "word"]


def test_read_wiktextract(tmp_path):
    lines = [
        {"word": "water", "lang_code": "en", "pos": "noun",
         "translations": [{"code": "ru", "word": "вода", "sense": "liquid"}],
         "senses": [{"translations": [{"code": "de", "word": "Wasser", "sense": "liquid"},
                                      {"code": "fr", "sense": "no word"}]}]},
        {"word": "Wasser", "lang_code": "de", "pos": "noun",
         "translations": [{"code": "en", "word": "water"}]},
        {"word": "dog", "lang_code": "en", "pos": "noun", "senses": [{"glosses": ["an animal"]}]},
    ]
    path = tmp_path / "dump.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for e in lines:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    df = read_wiktextract(path)
    assert list(df.columns) == WK_COLS
    assert sorted(map(tuple, df[["en", "code", "word"]].to_numpy())) == [
        ("water", "de", "Wasser"), ("water", "ru", "вода")]


def test_panlex_candidates_counts_shared_meanings():
    en_pl = pd.DataFrame({"meaning": [1, 2], "word": [7, 7]})
    lang_pl = pd.DataFrame({"meaning": [1, 2, 2], "word": [3, 3, 5]})
    out = panlex_candidates(en_pl, lang_pl, CONCEPTS)
    assert sorted(map(tuple, out[["concept_id", "word_idx", "n_pl"]].to_numpy())) == [
        (0, 3, 2), (0, 5, 1)]


def test_wiktionary_candidates_counts_drops():
    wk = pd.DataFrame({"en": ["water"] * 3, "pos": "noun", "sense": "",
                       "code": "ru", "word": ["вода́", "две воды", "zzz"]})
    out, counts = wiktionary_candidates(wk, ["ru"], CONCEPTS, {"вода": 9}, "ru")
    assert out[["concept_id", "word_idx"]].values.tolist() == [[0, 9]]
    assert counts == {"n_rows": 3, "n_multiword": 1, "n_oov": 1}


def test_empty_inputs(tmp_path):
    empty_pl = pd.DataFrame({"meaning": pd.Series([], dtype="int64"),
                             "word": pd.Series([], dtype="int32")})
    pl = panlex_candidates(empty_pl, empty_pl, CONCEPTS)
    wk, counts = wiktionary_candidates(pd.DataFrame(columns=WK_COLS), ["ru"], CONCEPTS, {}, "ru")
    mt = mt_candidates(tmp_path / "missing.tsv", CONCEPTS, {}, "ru")
    merged = merge_candidates(pl, wk, mt)
    assert merged.empty
    assert list(merged.columns) == ["concept_id", "word_idx", "n_pl", "wk", "mt", "back"]
    assert counts == {"n_rows": 0, "n_multiword": 0, "n_oov": 0}


def test_merge_marks_sources(tmp_path):
    pl = pd.DataFrame({"concept_id": [0], "word_idx": [9], "n_pl": [2]})
    wk = pd.DataFrame({"concept_id": [0, 1], "word_idx": [9, 4]})
    (tmp_path / "mt.tsv").write_text("concept_id\tword\tback\n1\tсобака\tdog\n",
                                     encoding="utf-8")
    mt = mt_candidates(tmp_path / "mt.tsv", CONCEPTS, {"собака": 11}, "ru")
    merged = merge_candidates(pl, wk, mt).set_index(["concept_id", "word_idx"])
    assert merged.loc[(0, 9), "n_pl"] == 2 and bool(merged.loc[(0, 9), "wk"])
    assert merged.loc[(1, 4), "n_pl"] == 0 and not bool(merged.loc[(1, 4), "mt"])
    assert bool(merged.loc[(1, 11), "mt"]) and merged.loc[(1, 11), "back"] == "dog"
    assert merged.loc[(0, 9), "back"] == ""


def test_wiktionary_reverse_is_in_the_library():
    from clwe_chains.sources import wiktionary_reverse
    wk = pd.DataFrame({"en": ["water", "dog"], "code": ["yo", "yo"], "word": ["omi", "aja"]})
    out = wiktionary_reverse(wk, ["yo"], {"water": 10, "dog": 30}, {"omi": 1}, "yo")
    assert out.values.tolist() == [[1, 10]]
