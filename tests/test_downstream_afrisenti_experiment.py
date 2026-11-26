"""Tests of experiments/downstream-afrisenti/run.py's helpers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

RUN = Path(__file__).resolve().parents[1] / "experiments" / "downstream-afrisenti" / "run.py"
_spec = importlib.util.spec_from_file_location("downstream_afrisenti_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_directions_main_is_the_afrisenti_language_into_the_umsab_one():
    pair = SimpleNamespace(s="yo", t="ar", hub="en")
    assert run.directions(pair) == [("main", "yo", "ar"), ("reverse", "ar", "yo")]


def delta_rows():
    """Two AfriSenti languages, two pairs each, both sets and models."""
    rows = []
    spec = [("yo", "ar", 0.04, 0.01, 0.07, 3.7), ("yo", "en", -0.01, -0.03, 0.01, -2.0),
            ("sw", "ar", 0.02, -0.01, 0.05, 2.9), ("sw", "en", 0.0, -0.02, 0.02, -0.7)]
    for pair, (x, y, d, lo, hi, r) in enumerate(spec):
        for side, src, tgt in (("main", x, y), ("reverse", y, x)):
            for kind in ("lr", "mlp"):
                rows.append({"set": side, "pair": pair, "afrisenti": x, "src": src, "tgt": tgt,
                             "hub": "en", "model": kind, "delta_f1": d, "lo": lo, "hi": hi,
                             "r": r})
    return pd.DataFrame(rows)


def test_summary_table_per_set_and_language_never_pooled():
    s = run.summary_table(delta_rows())
    assert set(s["afrisenti"]) == {"yo", "sw"}                      # no pooled row
    s = s.set_index(["set", "model", "afrisenti"])
    assert s.loc[("main", "lr", "yo"), "n_directions"] == 2
    assert s.loc[("main", "lr", "yo"), "mean_delta_f1"] == pytest.approx(0.015)
    assert s.loc[("main", "lr", "yo"), "above"] == 1 and s.loc[("main", "lr", "yo"), "below"] == 0
    assert ("reverse", "mlp", "sw") in s.index


def test_correlation_table_uses_the_main_set_only():
    delta = delta_rows()
    delta.loc[delta["set"] == "reverse", "r"] = -delta.loc[delta["set"] == "reverse", "r"]
    out = run.correlation_table(delta, pd.DataFrame()).set_index(["model", "what"])
    assert out.loc[("lr", "advantage in F1 vs r"), "n"] == 4
    assert out.loc[("lr", "advantage in F1 vs r"), "rho"] == pytest.approx(1.0)


def test_correlation_table_joins_bli():
    rng = np.random.default_rng(0)
    rows, transfer = [], []
    for pair in range(6):
        mrr_d = rng.uniform(0.1, 0.6)
        mrr_c = mrr_d + rng.normal(0, 0.02)
        for kind in ("lr", "mlp"):
            rows.append({"set": "main", "pair": pair, "src": f"x{pair}", "tgt": "y",
                         "model": kind, "r": rng.normal(), "delta_f1": (mrr_c - mrr_d) * 2,
                         "mrr_direct": mrr_d, "mrr_chain": mrr_c, "delta_mrr": mrr_c - mrr_d})
            for method, mrr in (("direct", mrr_d), ("chain", mrr_c)):
                transfer.append({"set": "main", "src": f"x{pair}", "tgt": "y", "model": kind,
                                 "method": method, "macro_f1": 0.3 + mrr * 0.5})
    out = run.correlation_table(pd.DataFrame(rows), pd.DataFrame(transfer))
    lr = out[out["model"] == "lr"].set_index("what")
    assert lr.loc["advantage in F1 vs advantage in MRR", "rho"] > 0.99
    assert lr.loc["F1 vs MRR, directions x {direct, chain}", "n"] == 12


class Constant:
    """A stand-in classifier that always predicts one class."""

    def __init__(self, k):
        self.k = k

    def predict(self, x):
        return np.full(len(x), self.k)


def test_run_directions_uses_the_target_classifier_and_the_source_labels(monkeypatch):
    import torch
    monkeypatch.setattr(run, "EmbeddingStore", lambda *a, **k: None)
    monkeypatch.setattr(run, "edge_map", lambda store, a, b: torch.eye(4))
    n = 10
    # yo's tweets are all class 0, ar's all class 2; ar's classifier says 0, yo's says 2. Only
    # the right pairing (target's classifier, source's labels) scores above 0 in both sets.
    data = {"yo": {"X_test": np.zeros((n, 4), np.float32), "y_test": np.zeros(n, int)},
            "ar": {"X_test": np.zeros((n, 4), np.float32), "y_test": np.full(n, 2)}}
    models = {"ar": {"lr": [Constant(0)], "mlp": [Constant(0)] * 3},
              "yo": {"lr": [Constant(2)], "mlp": [Constant(2)] * 3}}
    pairs = pd.DataFrame({"s": ["yo"], "t": ["ar"], "hub": ["en"]})
    transfer, preds = run.run_directions(pairs, data, models)
    assert len(transfer) == 2 * 3 * 2                            # sets x methods x models
    assert transfer["macro_f1"].tolist() == pytest.approx([1 / 3] * 12)   # one class, all right
    main = transfer[transfer["set"] == "main"]
    assert set(zip(main["src"], main["tgt"])) == {("yo", "ar")}
    assert preds["yo", "ar", "chain", "mlp"].shape == (3, n)
