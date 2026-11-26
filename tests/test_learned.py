import numpy as np

from clwe_chains.graph import all_routes
from clwe_chains.learned import fit, pick, route_features, training_routes


def test_route_features_shape_and_values():
    D1 = np.arange(16, dtype=float).reshape(4, 4)
    D2 = np.ones((4, 4))
    routes = np.array([[0, 1, -1, -1, -1], [0, 2, 1, -1, -1], [0, 2, 3, 1, -1]])
    X = route_features({"m2": D2, "m1": D1}, routes)
    # columns: m1, m2 (sorted names), hops
    assert X.tolist() == [[D1[0, 1], 1.0, 0.0],
                          [D1[0, 2] + D1[2, 1], 2.0, 1.0],
                          [D1[0, 2] + D1[2, 3] + D1[3, 1], 3.0, 2.0]]


def test_training_routes_union_and_determinism():
    rng = np.random.default_rng(0)
    costs = {"a": rng.random((8, 8)), "b": rng.random((8, 8))}
    r1 = training_routes(costs, 0, 1, per_measure=5, n_random=5, seed=0)
    r2 = training_routes(costs, 0, 1, per_measure=5, n_random=5, seed=0)
    assert (r1 == r2).all()
    assert 5 <= len(r1) <= 15
    assert not ((r1 >= 0).sum(axis=1) == 2).any()  # never the direct route
    assert len({tuple(r) for r in r1}) == len(r1)


def test_fit_and_pick_learns_simple_rule():
    rng = np.random.default_rng(0)
    X = rng.random((600, 3))
    y = -X[:, 0]
    model = fit(X, y, seed=0)
    X_new = np.column_stack([np.linspace(0.05, 0.95, 10), rng.random((10, 2))])
    routes = all_routes(12, 0, 1)[1:11]
    assert pick(model, X_new, routes) == tuple(int(v) for v in routes[0] if v >= 0)
