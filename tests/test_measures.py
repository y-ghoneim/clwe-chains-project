import math

import numpy as np
import pandas as pd
import pytest
import torch

from clwe_chains.measures import (
    econd_hm,
    effective_condition,
    pairwise,
    resource_cost,
    scale_max,
    spectrum,
    svg,
    uriel_cost,
)


def test_resource_cost_ranks():
    D = resource_cost({"a": 10.0, "b": 5.0, "c": None}, ["a", "b", "c"])
    assert D[0, 1] == pytest.approx(2 / 3) and D[1, 0] == pytest.approx(2 / 3)
    assert D[0, 2] == pytest.approx(1.0)
    assert np.all(np.diag(D) == 0)


def test_svg_zero_for_same_spectrum():
    s = np.array([3.0, 2.0, 1.0])
    assert svg(s, s.copy(), k=3) == 0.0


def test_svg_value():
    sa = np.exp([2.0, 1.0, 0.0])
    sb = np.exp([1.0, 1.0, 1.0])
    assert svg(sa, sb, k=3) == pytest.approx(2.0)
    assert svg(sa, sb, k=2) == pytest.approx(1.0)


def test_effective_condition():
    s = np.array([4.0, 2.0, 1.0, 1.0])
    p = s / s.sum()
    erank = math.floor(math.exp(-(p * np.log(p)).sum()))
    assert erank == 3
    assert effective_condition(s) == pytest.approx(s[0] / s[erank - 1])


def test_econd_hm_is_harmonic_mean():
    sa, sb = np.array([4.0, 2.0, 1.0, 1.0]), np.array([9.0, 3.0, 3.0, 1.0])
    ka, kb = effective_condition(sa), effective_condition(sb)
    assert econd_hm(sa, sb) == pytest.approx(2 * ka * kb / (ka + kb))


def test_spectrum_normalizes_and_centres():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 10, generator=g, dtype=torch.float64)
    scaled = x * (0.1 + torch.rand(200, 1, generator=g, dtype=torch.float64) * 10)
    s1, s2 = spectrum(x), spectrum(scaled)
    assert np.allclose(s1, s2) and (np.diff(s1) <= 1e-12).all()


def test_uriel_cost_symmetric():
    m = pd.DataFrame({"a": ["x", "x", "y", "x"], "b": ["y", "z", "z", "y"],
                      "measure": ["uriel_genetic"] * 3 + ["uriel_syntactic"],
                      "value": [0.3, 0.5, 0.7, 0.9]})
    D = uriel_cost(m, ["x", "y", "z"], "genetic")
    assert D.tolist() == [[0.0, 0.3, 0.5], [0.3, 0.0, 0.7], [0.5, 0.7, 0.0]]
    with pytest.raises(KeyError):
        uriel_cost(m, ["x", "y", "w"], "genetic")


def test_pairwise_and_scale_max():
    D = pairwise(["a", "bb", "ccc"], lambda u, v: abs(len(u) - len(v)))
    assert D.tolist() == [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
    assert scale_max(D).max() == 1.0 and scale_max(D)[0, 1] == 0.5


def _orth(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return q


def test_rsim_identical_and_rotation_invariant():
    from clwe_chains.measures import rsim

    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 16, generator=g, dtype=torch.float64)
    assert rsim(x, x.clone()) == pytest.approx(1.0)
    assert rsim(x, x @ _orth(16)) == pytest.approx(1.0)
    y = torch.randn(200, 16, generator=g, dtype=torch.float64)
    assert abs(rsim(x, y)) < 0.2


def test_validation_concepts_outside_list():
    from clwe_chains.measures import validation_concepts

    ok_a, ok_b, lst = set(range(1, 11)), set(range(2, 13)), {2, 3}
    out = validation_concepts(ok_a, ok_b, lst, n=4)
    assert out == [4, 5, 6, 7] and not set(out) & lst


def test_edge_quality_perfect_map():
    from clwe_chains.measures import edge_quality

    g = torch.Generator().manual_seed(1)
    src = torch.nn.functional.normalize(torch.randn(100, 16, generator=g, dtype=torch.float64),
                                        dim=1)
    w = _orth(16, seed=2)
    q = np.arange(30)
    assert edge_quality(src, src @ w, w, q, [[i] for i in q], block=8) == pytest.approx(1.0)
