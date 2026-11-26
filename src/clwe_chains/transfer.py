"""Cross-lingual sentiment transfer: classifiers and scoring (experiments/downstream-*).

A classifier is trained on one language's tweet features in its own space; another language's
features are moved into that space by an alignment map and classified. Logistic regression (C from a
fixed grid) is the headline model; the MLP (one hidden layer, fixed configuration) is secondary.
Both are chosen and stopped on the *labelled* language's validation split, never on the language
being tested. Features are unit-length means of word vectors, centred on each language's own mean
(center_features); the classifiers multiply them by sqrt(d) so coordinates are about unit scale
(a constant fixed before any run).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

N_CLASSES = 3                                # sentiment; topic tasks pass their own
LR_GRID = [0.001, 0.01, 0.1, 1, 10, 100]    # widened from 0.1-100: 0.1 won for every language
MLP_HIDDEN = 256
MLP_MAX_EPOCHS = 100
MLP_PATIENCE = 10
MLP_BATCH = 64


@dataclass
class Classifier:
    clf: object
    scale: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict(x * self.scale)


def predict(model: Classifier, x: np.ndarray) -> np.ndarray:
    return model.predict(x)


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int = N_CLASSES) -> float:
    """Mean over classes of 2tp / (2tp + fp + fn); a class with no support and no predictions
    scores 0 (as sklearn's zero_division=0)."""
    cm = np.bincount(np.asarray(y) * n_classes + np.asarray(pred),
                     minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)))


def fit_logreg(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray,
               n_classes: int = N_CLASSES) -> tuple[Classifier, float]:
    """Logistic regression with C from LR_GRID by validation macro-F1 (ties: the smaller C)."""
    scale = float(np.sqrt(x.shape[1]))
    best = None
    for c in LR_GRID:
        model = Classifier(LogisticRegression(C=c, max_iter=2000).fit(x * scale, y), scale)
        score = macro_f1(yv, model.predict(xv), n_classes)
        if best is None or score > best[0]:
            best = (score, model, c)
    return best[1], best[2]


def fit_mlp(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray,
            seed: int, n_classes: int = N_CLASSES) -> Classifier:
    """One hidden layer, trained by epochs; the epoch with the lowest validation cross-entropy is
    kept, stopping after MLP_PATIENCE epochs without improvement."""
    scale = float(np.sqrt(x.shape[1]))
    clf = MLPClassifier(hidden_layer_sizes=(MLP_HIDDEN,), batch_size=MLP_BATCH, random_state=seed)
    best, best_clf, since = np.inf, None, 0
    for _ in range(MLP_MAX_EPOCHS):
        clf.partial_fit(x * scale, y, classes=np.arange(n_classes))
        score = log_loss(yv, clf.predict_proba(xv * scale), labels=np.arange(n_classes))
        if score < best:
            best, best_clf, since = score, copy.deepcopy(clf), 0
        else:
            since += 1
            if since >= MLP_PATIENCE:
                break
    return Classifier(best_clf, scale)


def fit_classifiers(d: dict, n_mlp_seeds: int, n_classes: int = N_CLASSES) -> dict:
    """A language's classifiers from its feature dict (X_/y_ train and validation): logistic
    regression with C chosen on validation, and n_mlp_seeds MLPs (seeds 0, 1, ...)."""
    lr, c = fit_logreg(d["X_train"], d["y_train"], d["X_validation"], d["y_validation"],
                       n_classes)
    mlps = [fit_mlp(d["X_train"], d["y_train"], d["X_validation"], d["y_validation"], seed=s,
                    n_classes=n_classes) for s in range(n_mlp_seeds)]
    return {"lr": [lr], "mlp": mlps, "C": c}


def predict_kinds(models: dict, x: np.ndarray, kinds=("lr", "mlp")) -> dict[str, np.ndarray]:
    """kind -> (models, tweets) predicted classes."""
    return {k: np.stack([predict(m, x) for m in models[k]]).astype(np.int8) for k in kinds}


def majority_floor(y: np.ndarray, n_classes: int = N_CLASSES) -> float:
    """Macro-F1 of always predicting the test set's most common class (1/6 on a balanced
    3-class set, higher when one class dominates)."""
    y = np.asarray(y)
    return macro_f1(y, np.full_like(y, np.bincount(y, minlength=n_classes).argmax()), n_classes)


def load_centred(path: Path) -> dict:
    """A prep/13 or prep/14 feature file with every split centred on the language's own
    training-tweet mean (no labels used)."""
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    mu = language_mean(d["X_train"], d["n_train"])
    for split in ("train", "validation", "test"):
        d[f"X_{split}"] = center_features(d[f"X_{split}"], mu, d[f"n_{split}"])
    return d


def edge_indices(dicts: Path, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    """Vocabulary rows (of a, of b) of a pair's full natural dictionary in a prep/08 dicts folder,
    whichever way round it is stored."""
    f = Path(dicts) / f"{a}__{b}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        return d["a"].to_numpy(), d["b"].to_numpy()
    d = pd.read_parquet(Path(dicts) / f"{b}__{a}.parquet")
    return d["b"].to_numpy(), d["a"].to_numpy()


def dictionary_size(sizes: pd.Series, a: str, b: str) -> int:
    """Pairs in the a-b dictionary (dict_sizes.csv 'pairs' indexed by edge), either orientation."""
    return int(sizes[f"{a}__{b}"] if f"{a}__{b}" in sizes else sizes[f"{b}__{a}"])


def chain_ratio(sizes: pd.Series, s: str, hub: str, t: str) -> float:
    """RQ2's r: log2 of the chain's narrowest step over the direct dictionary, in pairs."""
    return float(np.log2(min(dictionary_size(sizes, s, hub), dictionary_size(sizes, hub, t))
                         / dictionary_size(sizes, s, t)))


def language_mean(x: np.ndarray, n_tokens: np.ndarray) -> np.ndarray:
    """Mean feature vector of a language's tweets that have at least one matched token (its own
    unlabelled training tweets; no labels are used)."""
    return np.asarray(x, dtype=np.float32)[np.asarray(n_tokens) > 0].mean(axis=0)


def center_features(x: np.ndarray, mu: np.ndarray, n_tokens: np.ndarray) -> np.ndarray:
    """Features minus the language mean. Without it a mapped language sits at a different overall
    position than the labelled language and the classifier's bias decides the result (a smoke run
    on UMSAB sent most tweets to one class). Tweets with no matched token stay at zero."""
    out = np.asarray(x, dtype=np.float32) - np.asarray(mu, dtype=np.float32)
    out[np.asarray(n_tokens) == 0] = 0
    return out


def map_features(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Features moved by a map (rows times W). Rotations keep lengths; zero rows stay zero."""
    return np.asarray(x, dtype=np.float32) @ np.asarray(w, dtype=np.float32)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def spearman_by_group(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n: int = 1000,
                      seed: int = 0) -> tuple[float, float, float]:
    """Spearman correlation of x and y, with 2.5th / 97.5th percentiles of a bootstrap that
    resamples whole groups (the two directions of a pair share dictionaries and tweets, so they
    are not independent observations)."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = list(members)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n):
        idx = np.concatenate([members[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        rhos.append(_spearman(x[idx], y[idx]))
    lo, hi = np.nanpercentile(rhos, [2.5, 97.5])
    return _spearman(x, y), float(lo), float(hi)


def paired_f1_bootstrap(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n: int = 1000,
                        seed: int = 0, n_classes: int = N_CLASSES) -> tuple[float, float, float]:
    """(macro-F1(a) - macro-F1(b), 2.5th and 97.5th percentiles of the resampled differences).

    pred_a and pred_b are (models, tweets): the same tweets are drawn for both methods in each
    resample and the F1s are averaged over the models (the MLP's seeds) within it."""
    y = np.asarray(y)
    pred_a, pred_b = np.atleast_2d(pred_a), np.atleast_2d(pred_b)

    def mean_f1(idx: np.ndarray) -> tuple[float, float]:
        return (float(np.mean([macro_f1(y[idx], p[idx], n_classes) for p in pred_a])),
                float(np.mean([macro_f1(y[idx], p[idx], n_classes) for p in pred_b])))

    full = np.arange(len(y))
    fa, fb = mean_f1(full)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        a, b = mean_f1(rng.integers(0, len(y), len(y)))
        diffs[i] = a - b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return fa - fb, float(lo), float(hi)
