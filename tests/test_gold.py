import pandas as pd

from clwe_chains.gold import expand_gold, gold_sets, gold_stats, with_variants

EMPTY_PL = pd.DataFrame({"meaning": pd.Series([], dtype="int64"),
                         "word": pd.Series([], dtype="int32")})


def test_gold_unions_sources():
    list_word = pd.Series({0: 9})
    src_key = pd.Series({0: 7})
    pl_src = pd.DataFrame({"meaning": [1, 2], "word": [7, 7]})
    pl_tgt = pd.DataFrame({"meaning": [1, 2, 3], "word": [4, 5, 8]})  # 8: meaning not shared
    wk = pd.DataFrame({"concept_id": [0], "word_idx": [6]})
    assert gold_sets(list_word, src_key, pl_src, pl_tgt, wk) == {0: [4, 5, 6, 9]}


def test_gold_without_wiktionary_or_panlex():
    out = gold_sets(pd.Series({0: 9, 1: 3}), pd.Series({0: 7, 1: 8}), EMPTY_PL, EMPTY_PL, None)
    assert out == {0: [9], 1: [3]}


def test_gold_ignores_other_concepts():
    wk = pd.DataFrame({"concept_id": [1], "word_idx": [6]})
    out = gold_sets(pd.Series({0: 9, 1: 3}), pd.Series({0: 7, 1: 8}), EMPTY_PL, EMPTY_PL, wk)
    assert out == {0: [9], 1: [3, 6]}


def test_with_variants_adds_every_casing_of_each_row():
    variants = {4: [4, 7, 9], 5: [5, 6]}
    assert with_variants([4, 8], variants) == [4, 7, 8, 9]
    assert expand_gold({0: [4], 1: [5, 8]}, variants) == {0: [4, 7, 9], 1: [5, 6, 8]}


def test_gold_stats_counts_answers_per_query():
    s = gold_stats({0: [1], 1: [1, 2, 3], 2: [4, 5]})
    assert s == {"n": 3, "mean": 2.0, "median": 2.0, "max": 3}
