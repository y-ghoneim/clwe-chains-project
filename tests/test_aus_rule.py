import importlib.util
import warnings

import numpy as np
import pytest

from clwe_chains.aus_rule import cosine_distance, feature_vector, one_hop_score, path_cost


def test_missing_values_are_zero():
    assert feature_vector([1, "--", 0]).tolist() == [1.0, 0.0, 0.0]


def test_cosine_distance_zero_vector():
    assert cosine_distance(np.zeros(3), np.array([1.0, 0, 0])) == 1.0


def test_one_hop_and_path_cost():
    vecs = {"a": np.array([1.0, 0]), "b": np.array([1.0, 1]), "c": np.array([0.0, 1])}
    d_ab, d_bc = cosine_distance(vecs["a"], vecs["b"]), cosine_distance(vecs["b"], vecs["c"])
    assert one_hop_score(vecs, "b", "a", "c") == pytest.approx((d_ab + d_bc) / 2)
    assert path_cost(vecs, ["a", "b", "c"]) == pytest.approx(d_ab + d_bc)


@pytest.mark.skipif(importlib.util.find_spec("lang2vec") is None, reason="lang2vec not installed")
def test_reproduces_aus_table1():
    """AUS Table 1 with lang2vec codes eng spa gle mlt and Arabic as `arb` (`ara` doesn't
    reproduce it)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lang2vec imports the deprecated pkg_resources
        import lang2vec.lang2vec as l2v
    codes = ["eng", "spa", "gle", "mlt", "arb"]
    want = {"syntax_wals": {"spa": 0.3220, "gle": 0.3279}, "geo": {"spa": 0.0203, "gle": 0.0245}}
    for fs, expected in want.items():
        vecs = {k: feature_vector(v) for k, v in l2v.get_features(codes, fs).items()}
        for x, score in expected.items():
            assert round(one_hop_score(vecs, x, "eng", "arb"), 4) == score, (fs, x)
        if fs == "syntax_wals":
            assert cosine_distance(vecs["mlt"], vecs["arb"]) == pytest.approx(1.0)
