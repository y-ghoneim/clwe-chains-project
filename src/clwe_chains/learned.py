"""Learned combination of edge measures (LangRank-style; phase 4 design).

A route is described by its path cost under every measure plus its hop count. A gradient-boosted
regressor learns, from training-fold pairs, how well routes align on concepts outside the shared
list, and picks the best predicted route for held-out pairs (folds from cv.pair_folds).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from clwe_chains.graph import all_routes, path_costs


def route_features(costs: dict[str, np.ndarray], routes: np.ndarray) -> np.ndarray:
    """One column per measure (path cost, measures sorted by name), then the hop count."""
    cols = [path_costs(costs[name], routes) for name in sorted(costs)]
    hops = (routes >= 0).sum(axis=1) - 2
    return np.column_stack([*cols, hops]).astype(float)


def training_routes(costs: dict[str, np.ndarray], s: int, t: int, per_measure: int = 20,
                    n_random: int = 20, seed: int = 0) -> np.ndarray:
    """The union of each measure's per_measure cheapest routes and n_random seeded random ones,
    direct excluded, in all_routes order."""
    n = len(next(iter(costs.values())))
    routes = all_routes(n, s, t)[1:]
    keep = set()
    for name in sorted(costs):
        cheapest = np.argsort(path_costs(costs[name], routes), kind="stable")[:per_measure]
        keep.update(cheapest.tolist())
    rng = np.random.default_rng(seed)
    keep.update(rng.choice(len(routes), size=min(n_random, len(routes)), replace=False).tolist())
    return routes[sorted(keep)]


def fit(X: np.ndarray, y: np.ndarray, seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=seed).fit(X, y)


def pick(model: HistGradientBoostingRegressor, X: np.ndarray,
         routes: np.ndarray) -> tuple[int, ...]:
    """The route with the highest predicted quality (ties: first in route order)."""
    best = int(np.argmax(model.predict(X)))
    return tuple(int(v) for v in routes[best] if v >= 0)
