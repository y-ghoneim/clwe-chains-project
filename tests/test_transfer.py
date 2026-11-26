import numpy as np
import pandas as pd
import pytest

from clwe_chains import transfer
from clwe_chains.transfer import (
    LR_GRID,
    center_features,
    fit_logreg,
    fit_mlp,
    language_mean,
    macro_f1,
    map_features,
    paired_f1_bootstrap,
    predict,
    spearman_by_group,
)


def toy_task(rng, centres, n=300, shift=3.0):
    """Three classes, separable along the given class centres (shared by every split of one
    task); rows unit length."""
    y = rng.integers(0, 3, n)
    x = centres[y] * shift + rng.normal(size=(n, centres.shape[1]))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), y


def random_rotation(rng, d):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q.astype(np.float32)


def test_macro_f1_matches_hand_computation():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    # class 0: tp 1, fp 1, fn 1 -> 0.5; class 1: tp 2, fp 1, fn 0 -> 0.8; class 2: tp 1, fp 0, fn 1
    assert macro_f1(y, pred) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_macro_f1_majority_class_is_one_sixth_on_balanced_classes():
    y = np.repeat([0, 1, 2], 10)
    assert macro_f1(y, np.zeros(30, dtype=int)) == pytest.approx(1 / 6)


def test_logreg_learns_a_separable_task_and_picks_c_from_the_grid():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(3, 20))
    x, y = toy_task(rng, centres)
    xv, yv = toy_task(rng, centres, n=100)
    model, c = fit_logreg(x, y, xv, yv)
    assert c in LR_GRID
    assert macro_f1(yv, predict(model, xv)) > 0.9


def test_mlp_learns_and_is_deterministic_per_seed():
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(3, 20))
    x, y = toy_task(rng, centres)
    xv, yv = toy_task(rng, centres, n=100)
    a, b = fit_mlp(x, y, xv, yv, seed=0), fit_mlp(x, y, xv, yv, seed=0)
    assert macro_f1(yv, predict(a, xv)) > 0.9
    np.testing.assert_array_equal(predict(a, xv), predict(b, xv))


def test_map_features_is_a_rotation_and_keeps_zero_rows():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 8)).astype(np.float32)
    x[3] = 0
    w = random_rotation(rng, 8)
    out = map_features(x, w)
    np.testing.assert_allclose(out, x @ w, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.linalg.norm(x, axis=1), atol=1e-5)
    assert not out[3].any()


def test_language_mean_ignores_tweets_without_a_matched_token():
    x = np.array([[1, 0], [3, 0], [0, 0]], dtype=np.float32)
    np.testing.assert_allclose(language_mean(x, np.array([2, 1, 0])), [2, 0])


def test_center_features_subtracts_the_mean_and_leaves_empty_tweets_at_zero():
    x = np.array([[1, 0], [3, 0], [0, 0]], dtype=np.float32)
    out = center_features(x, np.array([2, 0], dtype=np.float32), np.array([2, 1, 0]))
    np.testing.assert_allclose(out, [[-1, 0], [1, 0], [0, 0]])


def test_centring_removes_a_shift_that_would_skew_the_classifier():
    """A classifier trained on centred features is not thrown by a source language whose tweets sit
    at a different overall position (the mean shift seen on UMSAB before centring)."""
    rng = np.random.default_rng(6)
    d = 20
    centres = rng.normal(size=(3, d))
    x_t, y = toy_task(rng, centres, n=400)
    xv_t, yv = toy_task(rng, centres, n=100)
    n_ok = np.ones(len(y), dtype=int)
    mu_t = language_mean(x_t[150:], n_ok[150:])
    model, _ = fit_logreg(center_features(x_t[150:], mu_t, n_ok[150:]), y[150:],
                          center_features(xv_t, mu_t, np.ones(100, int)), yv)
    toward_class_1 = (centres[1] / np.linalg.norm(centres[1])).astype(np.float32)
    shifted = x_t[:150] + 1.5 * toward_class_1                  # the whole language sits elsewhere
    mu_s = language_mean(shifted, n_ok[:150])
    raw = macro_f1(y[:150], predict(model, shifted))
    centred = macro_f1(y[:150], predict(model, center_features(shifted, mu_s, n_ok[:150])))
    assert centred > 0.9 and centred - raw > 0.1


def test_the_right_map_beats_a_wrong_one_and_no_map():
    """A classifier trained in the target space, applied to source-space features moved by the map
    that relates them: the correct map must transfer, a random rotation and no map must not."""
    rng = np.random.default_rng(3)
    d = 20
    centres = rng.normal(size=(3, d))
    x_t, y = toy_task(rng, centres, n=400)
    xv_t, yv = toy_task(rng, centres, n=100)
    # the same task seen in a source space: source = target rotated by r; the map src->tgt is r.T
    r = random_rotation(rng, d)
    xs_test = (x_t[:150] @ r).astype(np.float32)
    model, _ = fit_logreg(x_t[150:], y[150:], xv_t, yv)
    f1 = {name: macro_f1(y[:150], predict(model, map_features(xs_test, w)))
          for name, w in {"right": r.T, "wrong": random_rotation(rng, d),
                          "none": np.eye(d, dtype=np.float32)}.items()}
    assert f1["right"] > 0.9
    assert f1["right"] - f1["wrong"] > 0.4
    assert f1["right"] - f1["none"] > 0.4


def test_paired_bootstrap_zero_for_identical_predictions_and_positive_for_better():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 400)
    good = y.copy()
    bad = rng.integers(0, 3, 400)
    d0, lo0, hi0 = paired_f1_bootstrap(y, good[None], good[None], n=200, seed=0)
    assert (d0, lo0, hi0) == (0.0, 0.0, 0.0)
    d, lo, hi = paired_f1_bootstrap(y, good[None], bad[None], n=200, seed=0)
    assert lo > 0.3 and lo <= d <= hi


def test_spearman_by_group_perfect_and_reversed_and_independent():
    x = np.arange(40.0)
    groups = x.astype(int) // 2                      # two directions per pair
    rho, lo, hi = spearman_by_group(x, x ** 2, groups, n=200, seed=0)
    assert (rho, lo, hi) == pytest.approx((1.0, 1.0, 1.0))
    assert spearman_by_group(x, -x, groups, n=200, seed=0)[0] == pytest.approx(-1.0)
    rng = np.random.default_rng(5)
    rho, lo, hi = spearman_by_group(x, rng.normal(size=40), groups, n=500, seed=0)
    assert lo < 0 < hi


def test_spearman_by_group_resamples_whole_groups(monkeypatch):
    """Every bootstrap resample must contain both members of each drawn pair equally often; a row
    resample would not."""
    seen = []
    real = transfer._spearman

    def spy(x, y):
        seen.append(np.asarray(x))
        return real(x, y)

    monkeypatch.setattr(transfer, "_spearman", spy)
    x = np.arange(12.0)                         # rows 2k and 2k+1 form group k
    spearman_by_group(x, x ** 2, np.arange(12) // 2, n=50, seed=0)
    assert len(seen) == 51                      # 50 resamples, then the point estimate
    for sample in seen[:-1]:
        counts = np.array([(sample == v).sum() for v in x])
        np.testing.assert_array_equal(counts[0::2], counts[1::2])


def test_paired_bootstrap_averages_over_models_within_a_resample():
    y = np.array([0, 1, 2] * 50)
    a = np.stack([y, y])                          # two seeds, both perfect
    b = np.stack([np.zeros_like(y), y])           # one seed all zeros, one perfect
    d, _, _ = paired_f1_bootstrap(y, a, b, n=100, seed=0)
    assert d == pytest.approx((1 - 1 / 6) / 2, abs=0.02)


def test_majority_floor_balanced_and_skewed():
    assert transfer.majority_floor(np.repeat([0, 1, 2], 10)) == pytest.approx(1 / 6)
    y = np.array([0] * 6 + [1] * 2 + [2] * 2)                # always 0: F1 of class 0 = 12/16
    assert transfer.majority_floor(y) == pytest.approx(0.75 / 3)


def write_dict(folder, name, a, b):
    folder.mkdir(exist_ok=True)
    pd.DataFrame({"a": np.array(a, dtype="int64"), "b": np.array(b, dtype="int64")}).to_parquet(
        folder / f"{name}.parquet", index=False)


def test_edge_indices_follow_the_stored_orientation(tmp_path):
    write_dict(tmp_path, "yo__ar", [1, 2, 3], [10, 20, 30])
    yo, ar = transfer.edge_indices(tmp_path, "yo", "ar")
    assert (yo.tolist(), ar.tolist()) == ([1, 2, 3], [10, 20, 30])
    ar, yo = transfer.edge_indices(tmp_path, "ar", "yo")          # stored the other way round
    assert (ar.tolist(), yo.tolist()) == ([10, 20, 30], [1, 2, 3])


def test_dictionary_size_and_chain_ratio():
    sizes = pd.Series({"yo__ar": 586, "yo__en": 7436, "en__ar": 131688})
    assert transfer.dictionary_size(sizes, "ar", "yo") == 586
    # narrowest step yo-en (7,436) over direct (586)
    assert transfer.chain_ratio(sizes, "yo", "en", "ar") == pytest.approx(np.log2(7436 / 586))


def test_load_centred_centres_on_the_training_mean_and_keeps_empty_tweets_zero(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {}
    for split in ("train", "validation", "test"):
        x = rng.normal(1.0, 0.1, size=(20, 4)).astype(np.float32)
        n = np.full(20, 3)
        x[0], n[0] = 0, 0                                             # an empty tweet
        arrays.update({f"X_{split}": x, f"y_{split}": np.zeros(20, int), f"n_{split}": n})
    np.savez(tmp_path / "xx.npz", **arrays)
    d = transfer.load_centred(tmp_path / "xx.npz")
    assert np.allclose(d["X_train"][1:].mean(axis=0), 0, atol=1e-6)
    assert not d["X_test"][0].any()


def test_fit_classifiers_and_predict_kinds_shapes():
    rng = np.random.default_rng(1)
    centres = np.eye(3, 8)
    (xt, yt), (xv, yv) = toy_task(rng, centres), toy_task(rng, centres)
    models = transfer.fit_classifiers({"X_train": xt, "y_train": yt, "X_validation": xv,
                                       "y_validation": yv}, n_mlp_seeds=2)
    assert models["C"] in LR_GRID
    p = transfer.predict_kinds(models, xv)
    assert p["lr"].shape == (1, len(yv)) and p["mlp"].shape == (2, len(yv))


def test_classifiers_and_scores_take_the_number_of_classes():
    rng = np.random.default_rng(2)
    centres = np.eye(7, 12)

    def task(n):
        y = rng.integers(0, 7, n)
        x = centres[y] * 3 + rng.normal(size=(n, 12))
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), y

    (xt, yt), (xv, yv) = task(700), task(200)
    models = transfer.fit_classifiers({"X_train": xt, "y_train": yt, "X_validation": xv,
                                       "y_validation": yv}, n_mlp_seeds=1, n_classes=7)
    p = transfer.predict_kinds(models, xv)
    assert set(np.unique(p["lr"])) <= set(range(7))
    assert transfer.macro_f1(yv, p["lr"][0], 7) > 0.8
    assert transfer.macro_f1(yv, p["mlp"][0], 7) > 0.8
    y = np.repeat(np.arange(7), 10)
    assert transfer.majority_floor(y, 7) == pytest.approx((2 / 8) / 7)   # 10 of 70 right
    d, lo, hi = transfer.paired_f1_bootstrap(yv, p["lr"], p["lr"], n=50, n_classes=7)
    assert (d, lo, hi) == (0.0, 0.0, 0.0)
