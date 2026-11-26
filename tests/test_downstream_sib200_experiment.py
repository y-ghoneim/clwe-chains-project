"""Tests of experiments/downstream-sib200/run.py's helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1] / "experiments" / "downstream-sib200" / "run.py"
_spec = importlib.util.spec_from_file_location("downstream_sib200_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_directions_rules(monkeypatch):
    monkeypatch.setattr(run, "SC", {"excluded": ["zh"], "low": ["yo"],
                                    "stages": {"well": ["d"], "low": ["", "a"]}})
    pairs = {"d": pd.DataFrame({"s": ["en", "en"], "t": ["ar", "zh"], "hub": ["fr", "fr"]}),
             "": pd.DataFrame({"s": ["yo", "yo", "mhr"], "t": ["ar", "mt", "ar"],
                               "hub": ["en", "en", "en"]}),
             "a": pd.DataFrame({"s": ["yo"], "t": ["de"], "hub": ["zh"]})}
    d = run.directions(pairs, {"en", "ar", "yo", "mt", "de", "fr"})
    got = set(zip(d["set"], d["src"], d["tgt"], strict=True))
    # both ways in the well set; mt allowed as a target; mhr not available; zh excluded as hub
    assert got == {("well", "en", "ar"), ("well", "ar", "en"), ("low", "yo", "ar"),
                   ("low", "yo", "mt")}


def test_with_ids_uses_the_sorted_topic_order():
    d = {f"labels_{s}": np.array(["sports", "health"]) for s in ("train", "validation", "test")}
    out = run.with_ids(d, ["health", "sports"])
    assert out["y_test"].tolist() == [1, 0]
