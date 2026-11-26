"""Cross-validation over pairs (brief 5.4): anything selected or tuned (the threshold rule's τ,
the exponent p, the learned combination) is chosen on the other folds' pairs and applied to the
held-out fold, so every pair is still scored, but never by a value it helped choose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pair_folds(strata: list[str], k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per pair; each stratum is spread evenly over the k folds (seeded)."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    folds = np.empty(len(strata), dtype=np.int64)
    for s in sorted(set(strata.tolist())):
        idx = np.flatnonzero(strata == s)
        folds[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % k
    return folds


def cv_choose(score: pd.DataFrame, folds: np.ndarray) -> tuple[pd.Series, dict[int, object]]:
    """score: one row per pair, one column per option. Per fold, the option with the highest mean
    over the other folds' pairs (ties: first column). Returns each pair's score under its fold's
    choice and the choice per fold."""
    chosen: dict[int, object] = {}
    out = pd.Series(np.nan, index=score.index)
    for f in sorted(set(folds.tolist())):
        train = folds != f
        best = score[train].mean(axis=0).idxmax()
        chosen[f] = best
        out[folds == f] = score.loc[folds == f, best]
    return out, chosen


def threshold_scores(direct_cost: np.ndarray, mrr_chain: np.ndarray, mrr_direct: np.ndarray,
                     taus: np.ndarray) -> pd.DataFrame:
    """Per pair x τ: the chain's MRR if the direct edge's cost exceeds τ, else direct's MRR."""
    chain = np.asarray(direct_cost)[:, None] > np.asarray(taus)[None, :]
    vals = np.where(chain, np.asarray(mrr_chain)[:, None], np.asarray(mrr_direct)[:, None])
    return pd.DataFrame(vals, columns=list(np.asarray(taus)))
