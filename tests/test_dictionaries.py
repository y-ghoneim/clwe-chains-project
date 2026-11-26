import pandas as pd

from clwe_chains.dictionaries import (
    dict_sizes,
    edge_dictionary,
    join_counts,
    joined_dictionary,
    mutual_best,
    panlex_pairs,
    remove_test,
)


def _set(df: pd.DataFrame) -> set[tuple[int, int]]:
    return set(map(tuple, df[["a", "b"]].values.tolist()))


def test_panlex_pairs_distinct():
    pa = pd.DataFrame({"meaning": [1, 1, 2, 3], "word": [10, 10, 11, 12]})
    pb = pd.DataFrame({"meaning": [1, 2, 2, 4], "word": [20, 21, 22, 23]})
    assert _set(panlex_pairs(pa, pb)) == {(10, 20), (11, 21), (11, 22)}


def test_edge_dictionary_adds_wiktionary_pairs():
    pa = pd.DataFrame({"meaning": [1], "word": [10]})
    pb = pd.DataFrame({"meaning": [1], "word": [20]})
    wk = pd.DataFrame({"a": [10, 11], "b": [20, 21]})
    d = edge_dictionary(pa, pb, wk)
    assert d[["a", "b"]].values.tolist() == [[10, 20], [11, 21]]
    assert d["a"].dtype == "int64" and d["b"].dtype == "int64"
    assert _set(edge_dictionary(pa, pb)) == {(10, 20)}


def test_remove_test_both_sides_and_casings():
    d = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    # a=2 is a test word's other casing (its variant row), b=8 a test target row
    assert remove_test(d, {2}, {8})[["a", "b"]].values.tolist() == [[1, 5], [3, 7]]


def test_join_counts_and_mutual_best_ties():
    sx = pd.DataFrame({"a": [1, 1, 1, 2, 3], "b": [100, 101, 102, 100, 103]})
    xt = pd.DataFrame({"a": [100, 101, 102, 100, 103], "b": [7, 7, 8, 8, 9]})
    j = join_counts(sx, xt)
    got = {(r.a, r.b): r.n for r in j.itertuples()}
    assert got == {(1, 7): 2, (1, 8): 2, (2, 7): 1, (2, 8): 1, (3, 9): 1}
    # 1 ties between 7 and 8: both kept; 2 loses both to 1
    assert _set(mutual_best(j)) == {(1, 7), (1, 8), (3, 9)}


def test_joined_dictionary_empty_join_is_direct():
    direct = pd.DataFrame({"a": [1], "b": [5]})
    counts = pd.DataFrame({"a": [2], "b": [6], "n": [1]})
    out = joined_dictionary(direct, counts, test_s={2}, test_t=set(), filtered=True)
    assert out[["a", "b"]].values.tolist() == [[1, 5]]
    out = joined_dictionary(direct, counts.iloc[:0], set(), set(), filtered=False)
    assert out[["a", "b"]].values.tolist() == [[1, 5]]
    assert len(mutual_best(counts.iloc[:0])) == 0


def test_joined_dictionary_unfiltered_keeps_all():
    direct = pd.DataFrame({"a": [1], "b": [5]})
    counts = pd.DataFrame({"a": [1, 2, 2], "b": [5, 6, 7], "n": [1, 2, 1]})
    assert len(joined_dictionary(direct, counts, set(), set(), filtered=False)) == 3
    assert _set(joined_dictionary(direct, counts, set(), set(), True)) == {(1, 5), (2, 6)}


def test_joined_dictionary_removes_test_words_before_mutual_best():
    # b=6's best partner a=2 is a test word; without it, a=3 becomes 6's best
    direct = pd.DataFrame({"a": [1], "b": [5]})
    counts = pd.DataFrame({"a": [2, 3], "b": [6, 6], "n": [3, 1]})
    assert _set(joined_dictionary(direct, counts, {2}, set(), filtered=True)) == {(1, 5), (3, 6)}


def test_dict_sizes():
    d = pd.DataFrame({"a": [1, 1, 2], "b": [5, 6, 6]})
    assert dict_sizes(d) == {"pairs": 3, "a_words": 2, "b_words": 2}
