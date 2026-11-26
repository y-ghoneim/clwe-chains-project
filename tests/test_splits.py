import numpy as np

from clwe_chains.splits import concept_folds, concept_splits


def test_partition():
    ids = np.arange(1001) * 3
    splits = concept_splits(ids, test_frac=0.2, seeds=[0, 1, 2, 3, 4])
    assert sorted(splits) == [0, 1, 2, 3, 4]
    for s in splits.values():
        train, test = set(s["train"]), set(s["test"])
        assert train | test == set(ids.tolist()) and not train & test
        assert len(test) == round(0.2 * 1001)
        assert s["test"] == sorted(s["test"]) and s["train"] == sorted(s["train"])


def test_deterministic():
    ids = np.arange(500)
    a, b = concept_splits(ids, 0.2, [0, 1]), concept_splits(ids, 0.2, [0, 1])
    assert a == b
    assert a[0]["test"] != a[1]["test"]


def test_concept_folds_partition_each_concept_once():
    ids = np.arange(100, 153)                     # 53 concepts: folds of 11, 11, 11, 10, 10
    folds = concept_folds(ids, k=5, seed=0)
    assert sorted(folds) == [0, 1, 2, 3, 4]
    allc = [c for f in folds.values() for c in f]
    assert sorted(allc) == ids.tolist() and len(set(allc)) == 53
    assert sorted(len(f) for f in folds.values()) == [10, 10, 11, 11, 11]
    assert all(f == sorted(f) for f in folds.values())
    assert concept_folds(ids[::-1], k=5, seed=0) == folds      # input order doesn't matter
    assert concept_folds(ids, k=5, seed=1) != folds
