"""Significance for paired comparisons (brief 5.7)."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 1000,
                     seed: int = 0) -> tuple[float, float, float]:
    """(mean(a - b), 2.5th and 97.5th percentiles of the resampled means).

    a and b are paired per item (e.g. reciprocal ranks of one test concept under two routes);
    items are resampled with replacement.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def _stat(item) -> float:
    if isinstance(item, np.ndarray):
        return float(item.mean())
    return float(np.mean([_stat(x) for x in item]))


def _resampled(item, rng: np.random.Generator) -> float:
    if isinstance(item, np.ndarray):
        cols = rng.integers(0, item.shape[-1], size=item.shape[-1])
        return float(item[..., cols].mean())
    picks = rng.integers(0, len(item), size=len(item))
    return float(np.mean([_resampled(item[i], rng) for i in picks]))


def nested_bootstrap(groups: list, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(statistic, 2.5th and 97.5th percentiles) for grouped paired differences (phase 6).

    groups nests lists down to numpy arrays (e.g. languages -> pairs -> directions x concepts).
    The statistic is the equal-weight mean over a list's items of each item's statistic; an
    array's statistic is its mean. Each resample draws items with replacement at every list
    level, and an array's last axis (concepts) jointly across its rows (directions), so
    observations that share a pair or a language aren't treated as independent.
    """
    rng = np.random.default_rng(seed)
    means = np.array([_resampled(groups, rng) for _ in range(n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return _stat(groups), float(lo), float(hi)
