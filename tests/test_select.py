import numpy as np
import pandas as pd

from clwe_chains.select import choose_one_to_one, reverse_top, round_trip


def _cands(rows):
    """rows: (concept_id, word_idx, n_pl, wk, mt, back)."""
    return pd.DataFrame(rows, columns=["concept_id", "word_idx", "n_pl", "wk", "mt", "back"])


def _chosen(df):
    return dict(zip(df["concept_id"], df["word_idx"], strict=True))


def test_competition_goes_to_more_frequent_concept():
    # 0 "big" and 1 "large" both want word 10; 1 also has word 11.
    cands = _cands([(0, 10, 1, False, False, ""), (1, 10, 3, False, False, ""),
                    (1, 11, 1, False, False, "")])
    assert _chosen(choose_one_to_one(cands)) == {0: 10, 1: 11}


def test_tier_preference():
    cands = _cands([(0, 4, 0, True, False, ""), (0, 6, 1, False, False, "")])
    out = choose_one_to_one(cands)
    assert _chosen(out) == {0: 6}
    assert out["source"].tolist() == ["panlex"]


def test_source_is_highest_attesting_tier():
    cands = _cands([(0, 4, 0, True, True, "x"), (1, 5, 0, False, True, "y")])
    assert choose_one_to_one(cands)["source"].tolist() == ["wiktionary", "mt"]


def test_shuffle_invariance():
    rng = np.random.default_rng(1)
    cands = _cands([(int(c), int(w), int(n), bool(k), False, "")
                    for c, w, n, k in zip(rng.integers(0, 10, 50), rng.integers(0, 15, 50),
                                          rng.integers(0, 3, 50), rng.integers(0, 2, 50),
                                          strict=True)]).drop_duplicates(["concept_id", "word_idx"])
    shuffled = cands.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(choose_one_to_one(cands), choose_one_to_one(shuffled))


def test_reverse_top_order():
    # L word 3 shares 2 meanings with en key 7 and 1 with en key 5; L word 4 is Wiktionary-only.
    lang_pl = pd.DataFrame({"meaning": [1, 2, 3], "word": [3, 3, 3]})
    en_pl = pd.DataFrame({"meaning": [1, 2, 3], "word": [7, 7, 5]})
    wk_rev = pd.DataFrame({"word_idx": [3, 4, 4], "en_key": [5, 9, 8]})
    top = reverse_top(lang_pl, en_pl, wk_rev, k=1)
    assert sorted(map(tuple, top[["word_idx", "en_key"]].to_numpy())) == [(3, 7), (4, 8)]


def test_reverse_top_k_keeps_k_best_in_order():
    lang_pl = pd.DataFrame({"meaning": [1, 2, 3], "word": [3, 3, 3]})
    en_pl = pd.DataFrame({"meaning": [1, 2, 3], "word": [7, 7, 5]})
    wk_rev = pd.DataFrame({"word_idx": [3, 3], "en_key": [2, 5]})
    top = reverse_top(lang_pl, en_pl, wk_rev, k=2)
    # word 3: en 7 (2 meanings), en 5 (1 meaning + Wiktionary), en 2 (Wiktionary only) -> top 2.
    assert list(map(tuple, top[["word_idx", "en_key"]].to_numpy())) == [(3, 7), (3, 5)]


CONCEPTS = pd.DataFrame({"concept_id": [0, 1], "en": ["big", "dog"], "en_rank": [3, 6],
                         "en_key": [3, 6]})


def _top(pairs):
    return pd.DataFrame(pairs, columns=["word_idx", "en_key"])


def test_round_trip_rejects_wrong_sense():
    chosen = pd.DataFrame({"concept_id": [0], "word_idx": [10], "source": ["panlex"], "back": [""]})
    assert round_trip(chosen, CONCEPTS, _top([(10, 99)])).tolist() == [False]
    assert round_trip(chosen, CONCEPTS, _top([(10, 99), (10, 3)])).tolist() == [True]


def test_round_trip_mt_backtranslation():
    chosen = pd.DataFrame({"concept_id": [1, 1], "word_idx": [20, 21], "source": ["mt", "mt"],
                           "back": ["dog", "hound"]})
    assert round_trip(chosen, CONCEPTS, _top([])).tolist() == [True, False]
