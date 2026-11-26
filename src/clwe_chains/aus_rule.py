"""The AUS report's intermediate-selection rule, reverse-engineered (brief 2).

lang2vec feature vectors (`syntax_wals`, `geo`) with missing values ("--") set to 0, compared
by cosine distance. A one-hop intermediate x scores ½[d(x, src) + d(x, tgt)]; longer routes use
the path cost d(src, x1) + ... + d(xk, tgt) (brief 5.6), which ranks one-hop routes the same way.
"""

from __future__ import annotations

import itertools

import numpy as np


def feature_vector(values: list) -> np.ndarray:
    """lang2vec values as floats, with "--" (missing) as 0.0."""
    return np.array([0.0 if v == "--" else float(v) for v in values])


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """1 - cosine similarity; 1.0 when either vector is all zeros."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 1.0
    return float(1.0 - (u @ v) / (nu * nv))


def one_hop_score(vecs: dict[str, np.ndarray], x: str, src: str, tgt: str) -> float:
    return (cosine_distance(vecs[x], vecs[src]) + cosine_distance(vecs[x], vecs[tgt])) / 2


def path_cost(vecs: dict[str, np.ndarray], route: list[str]) -> float:
    return sum(cosine_distance(vecs[a], vecs[b]) for a, b in itertools.pairwise(route))
