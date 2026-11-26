"""Concept-level train/test splits (brief section 5.3).

A test concept is held out in every language at once, so it can't be a training row on any edge,
chain steps included. One split per seed; results are reported as mean and spread over seeds.
"""

from __future__ import annotations

import numpy as np


def concept_splits(concept_ids: np.ndarray, test_frac: float,
                   seeds: list[int]) -> dict[int, dict[str, list[int]]]:
    """seed -> {"train": sorted ids, "test": sorted ids}, |test| = round(test_frac * n)."""
    ids = np.sort(np.asarray(concept_ids))
    n_test = round(test_frac * len(ids))
    out = {}
    for seed in seeds:
        perm = np.random.default_rng(seed).permutation(len(ids))
        test = np.sort(ids[perm[:n_test]])
        train = np.sort(ids[perm[n_test:]])
        out[seed] = {"train": train.tolist(), "test": test.tolist()}
    return out


def concept_folds(concept_ids: np.ndarray, k: int = 5, seed: int = 0) -> dict[int, list[int]]:
    """fold -> sorted test concept ids: k disjoint folds, so every concept is tested exactly once
    (phase 6, decisions.md 2026-11-16). The ids are sorted before the seeded shuffle, so input
    order doesn't matter; fold sizes differ by at most one."""
    ids = np.sort(np.asarray(concept_ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {f: sorted(ids[perm[f::k]].tolist()) for f in range(k)}
