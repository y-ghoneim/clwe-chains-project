import numpy as np
import pandas as pd

from clwe_chains.runs import (
    Cache,
    cache_key,
    file_fingerprints,
    query_concepts,
    result_affecting,
    settings_hash,
    threshold_cv,
)


def test_settings_hash_is_stable_and_order_free():
    a = settings_hash({"csls_k": 10, "half": True, "block_rows": 1024})
    assert a == settings_hash({"half": True, "block_rows": 1024, "csls_k": 10})
    assert a != settings_hash({"csls_k": 5, "half": True, "block_rows": 1024})
    assert len(a) == 12


def test_cache_key_carries_the_evaluation_settings():
    s10, s5 = settings_hash({"csls_k": 10}), settings_hash({"csls_k": 5})
    k10 = cache_key("csls", s10, ("en", "ar"), ("en", "de", "ar"), 0, "full")
    k5 = cache_key("csls", s5, ("en", "ar"), ("en", "de", "ar"), 0, "full")
    assert k10 != k5 and s10 in k10


def test_cache_roundtrip_skips_a_truncated_line(tmp_path):
    path = tmp_path / "evals.jsonl"
    c = Cache(path)
    c.put("a", {"ranks": [1]})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"key": "b", "val')       # interrupted mid-write
    again = Cache(path)
    assert again.get("a") == {"ranks": [1]} and "b" not in again
    calls = []
    again.get_or_compute("b", lambda: calls.append(1) or {"ranks": [2]})
    again.get_or_compute("b", lambda: calls.append(1) or {"ranks": [2]})
    assert calls == [1] and Cache(path).get("b") == {"ranks": [2]}


def test_result_affecting_handles_renames_and_package_files():
    paths = ("src/", "prep/", "configs/", "experiments/x/")
    lines = [" M src/clwe_chains/gold.py", "?? docs/report.pdf",
             "R  docs/old.md -> src/new.py",            # renamed into a result path
             "R  src/moved.py -> docs/moved.md",        # renamed out of one
             " M pyproject.toml", " M requirements.lock.txt",
             " M experiments/x/results/metrics.csv",    # this run's own results
             ' M "experiments/x/run file.py"']
    out = result_affecting(lines, paths, own_results="experiments/x/results/")
    assert out == [" M src/clwe_chains/gold.py", "R  docs/old.md -> src/new.py",
                   "R  src/moved.py -> docs/moved.md", " M pyproject.toml",
                   " M requirements.lock.txt", ' M "experiments/x/run file.py"']


def test_file_fingerprints(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.json"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    fp = file_fingerprints({"list": a, "splits": b, "missing": tmp_path / "nope"})
    assert fp["list"]["bytes"] == 3 and len(fp["list"]["sha256"]) == 64
    assert fp["list"]["sha256"] != fp["splits"]["sha256"]
    assert fp["missing"] is None


def test_threshold_cv_can_choose_never_chain():
    # Training pairs (fold 0) always lose by chaining, so the best rule is "never chain".
    # Held-out pair 2's direct cost (0.9) is above every training cost; without an infinite
    # candidate it would be forced to chain.
    cost = np.array([[0.2], [0.5], [0.9]])
    chain = np.array([[0.1], [0.1], [0.1]])
    direct = np.array([[0.5], [0.5], [0.5]])
    folds = np.array([0, 0, 1])
    out = threshold_cv(cost, chain, direct, folds)
    assert np.isinf(out.loc[2, "tau"]) and out.loc[2, "n_chained"] == 0
    assert out.loc[2, "mrr"] == 0.5


def test_query_concepts_count_exclusions_without_evaluating():
    identical = pd.Series({1: True, 2: False, 3: True})
    kept, n = query_concepts([1, 2, 3, 4], identical)
    assert kept == [2, 4] and n == 2
