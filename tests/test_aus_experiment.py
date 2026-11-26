import importlib.util
from pathlib import Path

import pandas as pd
import pytest

RUN = Path(__file__).resolve().parents[1] / "experiments" / "aus-replication" / "run.py"
_spec = importlib.util.spec_from_file_location("aus_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_assert_disjoint():
    run.assert_disjoint([1, 2], [3, 4])
    with pytest.raises(AssertionError):
        run.assert_disjoint([1, 2], [2, 3])


def test_test_concepts_excludes_identical():
    kept, n_excl = run.test_concepts([1, 2, 3], pd.Series({1: False, 2: True, 3: False}))
    assert kept == [1, 3] and n_excl == 1


def test_subset_split():
    out = run.subset_split({"train": [1, 2, 3], "test": [4, 5]}, {2, 5})
    assert out == {"train": [2], "test": [5]}


def test_result_affecting_changes():
    status = [" M CLAUDE.md", "?? docs/MLR570_Project_Report.pdf",
              "?? experiments/aus-replication/results/metrics.csv", " M src/clwe_chains/gold.py",
              "?? experiments/aus-replication/notes.txt"]
    assert run.result_affecting(status) == [" M src/clwe_chains/gold.py",
                                            "?? experiments/aus-replication/notes.txt"]
