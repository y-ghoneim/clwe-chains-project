import numpy as np
import pandas as pd

from clwe_chains.cv import cv_choose, pair_folds, threshold_scores


def test_pair_folds_stratified():
    strata = ["a"] * 20 + ["b"] * 20
    folds = pair_folds(strata, k=5, seed=0)
    for f in range(5):
        in_f = np.array(strata)[folds == f]
        assert (in_f == "a").sum() == 4 and (in_f == "b").sum() == 4
    assert (pair_folds(strata, k=5, seed=0) == folds).all()
    assert not (pair_folds(strata, k=5, seed=1) == folds).all()


def _scores():
    # 10 pairs, 2 per fold; option x is better everywhere except fold 4, where y is much better
    folds = np.repeat(np.arange(5), 2)
    score = pd.DataFrame({"x": [0.6] * 10, "y": [0.5] * 8 + [0.95, 0.95]})
    return score, folds


def test_cv_choose_picks_best_on_other_folds():
    score, folds = _scores()
    per_pair, chosen = cv_choose(score, folds)
    # fold 4's choice ignores fold 4: on folds 0-3, x (0.6) beats y (0.5)
    assert chosen[4] == "x" and per_pair.iloc[8] == 0.6
    # fold 0's choice sees fold 4's y = 0.95: y averages (0.5*6 + 0.95*2) / 8 = 0.6125 > 0.6
    assert chosen[0] == "y"


def test_heldout_fold_does_not_affect_its_choice():
    score, folds = _scores()
    _, before = cv_choose(score, folds)
    changed = score.copy()
    changed.loc[folds == 2, "y"] = 100.0
    _, after = cv_choose(changed, folds)
    assert after[2] == before[2]


def test_threshold_scores():
    out = threshold_scores(np.array([0.2, 0.8]), np.array([0.5, 0.7]), np.array([0.6, 0.4]),
                           np.array([0.5, 0.9]))
    # tau 0.5: pair 0 stays direct (0.6), pair 1 chains (0.7); tau 0.9: both direct
    assert out.to_numpy().tolist() == [[0.6, 0.6], [0.7, 0.4]]
