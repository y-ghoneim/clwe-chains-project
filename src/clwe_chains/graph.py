"""Routes on the complete language graph (brief 5.6).

A measure gives a dense cost matrix D (n x n, D[a, b] = cost of the step a -> b). A route
s -> x1 -> ... -> t has at most max_mid intermediates (brief 5.5: 3) and costs the sum of its
steps. With 30 nodes there are 20,440 chains per pair, few enough to score all of them exactly:
no search heuristic, any non-negative cost (metric or not), and the full ranking comes for free.
Ties: fewer hops first, then the route's node indices in lexicographic order.
"""

from __future__ import annotations

import functools
import itertools
from typing import Literal

import numpy as np


@functools.lru_cache(maxsize=4096)
def _routes(n: int, s: int, t: int, max_mid: int) -> np.ndarray:
    others = [x for x in range(n) if x not in (s, t)]
    rows = [(s, *mids, t) for k in range(max_mid + 1) for mids in itertools.permutations(others, k)]
    out = np.full((len(rows), max_mid + 2), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    out.setflags(write=False)
    return out


def all_routes(n: int, s: int, t: int, max_mid: int = 3) -> np.ndarray:
    """Every route s -> ... -> t with at most max_mid intermediates and no repeated node,
    direct included, as rows of node indices padded with -1; ordered by hops, then
    lexicographically."""
    return _routes(n, s, t, max_mid)


def path_costs(D: np.ndarray, routes: np.ndarray) -> np.ndarray:
    """Sum of D over each route's consecutive steps (padding ignored)."""
    a, b = routes[:, :-1], routes[:, 1:]
    valid = b >= 0
    steps = D[np.where(valid, a, 0), np.where(valid, b, 0)]
    return np.where(valid, steps, 0.0).sum(axis=1)


def ranked(D: np.ndarray, s: int, t: int, max_mid: int = 3,
           exclude_direct: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Routes and their costs, cheapest first (ties: fewer hops, then route order)."""
    routes = all_routes(len(D), s, t, max_mid)
    if exclude_direct:
        routes = routes[1:]  # row 0 is the direct route
    costs = path_costs(D, routes)
    order = np.argsort(costs, kind="stable")  # routes are already in (hops, lexicographic) order
    return routes[order], costs[order]


def best_route(D: np.ndarray, s: int, t: int, max_mid: int = 3,
               exclude_direct: bool = True) -> tuple[int, ...]:
    routes, _ = ranked(D, s, t, max_mid, exclude_direct)
    return tuple(int(x) for x in routes[0] if x >= 0)


def widest_ranked(S: np.ndarray, s: int, t: int, max_mid: int = 3,
                  exclude_direct: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Routes ranked by their narrowest step, widest first (phase 6: S[a, b] = dictionary size, so
    the best route's smallest dictionary is as large as possible). Ties as ranked: fewer hops, then
    route order."""
    routes = all_routes(len(S), s, t, max_mid)
    if exclude_direct:
        routes = routes[1:]
    a, b = routes[:, :-1], routes[:, 1:]
    valid = b >= 0
    steps = S[np.where(valid, a, 0), np.where(valid, b, 0)]
    widths = np.where(valid, steps, np.inf).min(axis=1)
    order = np.argsort(-widths, kind="stable")
    return routes[order], widths[order]


def transform(D: np.ndarray, kind: Literal["identity", "power", "neglog"],
              p: float = 1.0) -> np.ndarray:
    """identity; d**p; or -log(1 - d) for the quality edge (d = 1 - MRR, clipped below 1)."""
    if kind == "identity":
        return D.copy()
    if kind == "power":
        return D ** p
    if kind == "neglog":
        return -np.log(1.0 - np.clip(D, 0.0, 1.0 - 1e-6))
    raise ValueError(f"unknown transform {kind!r}")
