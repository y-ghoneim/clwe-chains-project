"""Tests of experiments/downstream-masakhanews/run.py's helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUN = Path(__file__).resolve().parents[1] / "experiments" / "downstream-masakhanews" / "run.py"
_spec = importlib.util.spec_from_file_location("downstream_masakhanews_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def lang(labels_by_split):
    d = {}
    for split, labels in labels_by_split.items():
        n = len(labels)
        d[f"X_{split}"] = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
        d[f"labels_{split}"] = np.array(labels)
    return d


def test_shared_topics_and_restrict():
    yo = lang({"train": ["health", "religion", "sports"], "validation": ["health", "religion"],
               "test": ["religion", "sports", "health", "sports"]})
    en = lang({"train": ["business", "health", "sports"], "validation": ["health"],
               "test": ["sports"]})
    topics = run.shared_topics({"yo": yo, "en": en}, "yo", "en")
    assert topics == ("health", "sports")                         # no religion, no business
    r = run.restrict(yo, topics)
    assert r["y_test"].tolist() == [1, 0, 1]                      # religion dropped; sorted ids
    assert r["X_test"].tolist() == [[2, 3], [4, 5], [6, 7]]       # the matching rows, unchanged
    assert r["y_validation"].tolist() == [0]


def test_summary_table_per_language_and_all():
    rows = []
    for pair, (af, d, lo, hi) in enumerate([("yo", 0.04, 0.01, 0.07), ("yo", 0.0, -0.02, 0.02),
                                            ("sw", -0.02, -0.05, 0.01)]):
        for kind in run.KINDS:
            rows.append({"set": "main", "pair": pair, "african": af, "model": kind,
                         "delta_f1": d, "lo": lo, "hi": hi})
    s = run.summary_table(pd.DataFrame(rows)).set_index(["set", "model", "group"])
    assert s.loc[("main", "lr", "all"), "n_directions"] == 3
    assert s.loc[("main", "lr", "yo"), "mean_delta_f1"] == pytest.approx(0.02)
    assert s.loc[("main", "lr", "yo"), "above"] == 1
    assert s.loc[("main", "lr", "sw"), "below"] == 0
