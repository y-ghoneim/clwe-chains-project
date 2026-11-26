import itertools
from collections import Counter

import numpy as np
import pytest

from clwe_chains.graph import all_routes, best_route, path_costs, ranked, transform, widest_ranked


def _hops(routes):
    return (routes >= 0).sum(axis=1) - 2


def test_route_counts():
    r = all_routes(30, 0, 1)
    assert r.shape == (20441, 5)
    assert Counter(_hops(r).tolist()) == {0: 1, 1: 28, 2: 756, 3: 19656}


def _brute_best(D, s, t, max_mid=3):
    n = len(D)
    others = [x for x in range(n) if x not in (s, t)]
    best = None
    for k in range(1, max_mid + 1):
        for mids in itertools.permutations(others, k):
            route = (s, *mids, t)
            cost = sum(D[a, b] for a, b in itertools.pairwise(route))
            key = (cost, len(route), route)
            if best is None or key < best[0]:
                best = (key, route)
    return best[1]


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    D = rng.random((7, 7))
    np.fill_diagonal(D, 0)
    for s, t in itertools.permutations(range(7), 2):
        assert best_route(D, s, t) == _brute_best(D, s, t), (s, t)


def test_path_costs():
    D = np.arange(16, dtype=float).reshape(4, 4)
    routes = np.array([[0, 1, -1, -1], [0, 2, 3, 1]])
    assert path_costs(D, routes).tolist() == [D[0, 1], D[0, 2] + D[2, 3] + D[3, 1]]


def test_tie_break_prefers_fewer_hops():
    D = np.ones((5, 5))
    np.fill_diagonal(D, 0)
    D[0, 3] = D[3, 4] = 0.5
    D[3, 1] = D[0, 4] = 5.0  # block the shortcuts 0-3-1 and 0-4-1
    # 0-2-1 and 0-3-4-1 both cost exactly 2.0, and nothing is cheaper
    assert best_route(D, 0, 1) == (0, 2, 1)


def test_exclude_direct():
    D = np.ones((4, 4))
    np.fill_diagonal(D, 0)
    D[0, 1] = 0.0
    assert best_route(D, 0, 1, exclude_direct=True) != (0, 1)
    assert best_route(D, 0, 1, exclude_direct=False) == (0, 1)
    routes, costs = ranked(D, 0, 1)
    assert not any(_hops(routes) == 0) and (np.diff(costs) >= 0).all()


def test_transform_power_and_neglog():
    D = np.array([[0.0, 0.5], [0.75, 0.0]])
    assert np.allclose(transform(D, "power", 2.0), D ** 2)
    assert np.allclose(transform(D, "neglog"), [[0.0, -np.log(0.5)], [-np.log(0.25), 0.0]])
    assert np.allclose(transform(D, "identity"), D)
    assert np.isfinite(transform(np.array([[0.0, 1.0], [1.0, 0.0]]), "neglog")).all()
    with pytest.raises(ValueError):
        transform(D, "other")


def test_matches_brute_force_with_ties_and_zero_costs():
    # integer costs give many ties and some zero-cost edges; costs are asymmetric
    rng = np.random.default_rng(1)
    for _ in range(5):
        D = rng.integers(0, 3, size=(6, 6)).astype(float)
        np.fill_diagonal(D, 0)
        for s, t in itertools.permutations(range(6), 2):
            assert best_route(D, s, t) == _brute_best(D, s, t), (s, t)


def test_widest_ranked_maximizes_the_smallest_step():
    # 0 = s, 3 = t; sizes are symmetric dictionary sizes
    S = np.array([[0, 5, 9, 1],
                  [5, 0, 8, 7],
                  [9, 8, 0, 2],
                  [1, 7, 2, 0]], dtype=float)
    routes, widths = widest_ranked(S, 0, 3, max_mid=2)
    best = tuple(int(x) for x in routes[0] if x >= 0)
    assert best == (0, 2, 1, 3) and widths[0] == 7       # min(9, 8, 7) beats one-hop (0,1,3): 5
    assert tuple(int(x) for x in routes[1] if x >= 0) == (0, 1, 3) and widths[1] == 5
    assert (np.diff(widths) <= 0).all()


def test_widest_ranked_ties_go_to_fewer_hops():
    S = np.full((4, 4), 4.0)
    routes, _ = widest_ranked(S, 0, 3, max_mid=2)
    assert tuple(int(x) for x in routes[0] if x >= 0) == (0, 1, 3)
