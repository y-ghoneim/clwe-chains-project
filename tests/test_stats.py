import numpy as np

from clwe_chains.stats import nested_bootstrap, paired_bootstrap


def test_bootstrap_identical_is_zero():
    a = np.linspace(0, 1, 50)
    assert paired_bootstrap(a, a.copy()) == (0.0, 0.0, 0.0)


def test_bootstrap_detects_shift():
    rng = np.random.default_rng(1)
    b = rng.random(200)
    a = b + 0.1 + rng.normal(0, 0.05, 200)
    mean, lo, hi = paired_bootstrap(a, b, n=1000, seed=0)
    assert abs(mean - 0.1) < 0.02 and lo > 0 and lo < mean < hi
    assert paired_bootstrap(a, b, n=1000, seed=0) == (mean, lo, hi)


def test_nested_bootstrap_statistic_weights_items_equally():
    g = [[np.array([[1.0, 1.0]]), np.array([[3.0, 3.0, 3.0, 3.0]])], [np.array([[0.0]])]]
    stat, lo, hi = nested_bootstrap(g, n=200, seed=0)
    assert stat == (2.0 + 0.0) / 2 and lo <= stat <= hi


def test_nested_bootstrap_constant_and_shift():
    g = [[np.full((2, 30), 0.5)] for _ in range(4)]
    assert nested_bootstrap(g, n=100) == (0.5, 0.5, 0.5)
    rng = np.random.default_rng(0)
    g = [[rng.normal(0.2, 0.1, (2, 50)) for _ in range(5)] for _ in range(3)]
    stat, lo, _hi = nested_bootstrap(g, n=500, seed=0)
    assert lo > 0 and abs(stat - 0.2) < 0.05


def test_nested_bootstrap_resamples_concepts_jointly_across_directions():
    # directions are perfectly anti-correlated per concept: every joint resample averages to 0
    d = np.array([[1.0, -1.0, 2.0, -2.0], [-1.0, 1.0, -2.0, 2.0]])
    assert nested_bootstrap([d], n=50, seed=0) == (0.0, 0.0, 0.0)
