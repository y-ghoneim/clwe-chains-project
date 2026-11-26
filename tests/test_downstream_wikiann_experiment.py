"""Tests of experiments/downstream-wikiann/run.py's helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUN = Path(__file__).resolve().parents[1] / "experiments" / "downstream-wikiann" / "run.py"
_spec = importlib.util.spec_from_file_location("downstream_wikiann_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_directions_rules(monkeypatch):
    monkeypatch.setattr(run, "WC", {"excluded": ["zh"], "low": ["yo"], "min_train": 1000,
                                    "stages": {"well": ["d"], "low": ["", "a"]}})
    pairs = {
        "d": pd.DataFrame({"s": ["en", "mt", "en"], "t": ["ar", "hi", "zh"],
                           "hub": ["fr", "en", "fr"]}),
        "": pd.DataFrame({"s": ["yo", "yo", "or"], "t": ["ar", "mt", "ar"],
                          "hub": ["en", "en", "hi"]}),
        "a": pd.DataFrame({"s": ["yo"], "t": ["de"], "hub": ["zh"]}),
    }
    sizes = {"en": 20000, "ar": 20000, "hi": 5000, "mt": 100, "yo": 100, "de": 20000}
    d = run.directions(pairs, sizes)
    got = set(zip(d["set"], d["src"], d["tgt"], strict=True))
    assert got == {("well", "en", "ar"), ("well", "ar", "en"),   # both ways
                   ("well", "mt", "hi"),                          # mt -> hi only (mt has 100)
                   ("low", "yo", "ar")}                           # yo -> mt: target too small
    # zh as endpoint or hub is excluded; or is not a WikiANN low-resource language here
    assert d.loc[d["src"] == "en", "pair"].nunique() == 1         # one pair id for both ways
    assert d.loc[d["set"] == "low", "low"].tolist() == ["yo"]


def delta_rows():
    rows = []
    spec = [("low", "1:yo-ar", "yo", "yo", "ar", 0.04, 0.01, 0.07, 3.0),
            ("low", "1:yo-de", "yo", "yo", "de", 0.02, -0.01, 0.05, 2.0),
            ("low", "1:or-ar", "or", "or", "ar", -0.02, -0.05, 0.01, 1.0),
            ("well", "d:en-ar", "", "en", "ar", 0.0, -0.01, 0.01, -1.0),
            ("well", "d:en-ar", "", "ar", "en", 0.02, 0.01, 0.03, -1.0)]
    for side, pair, low, src, tgt, d, lo, hi, r in spec:
        for kind in run.TAGGERS:
            rows.append({"set": side, "stage": "", "pair": pair, "low": low, "src": src,
                         "tgt": tgt, "hub": "en", "tagger": kind, "delta_f1": d, "lo": lo,
                         "hi": hi, "r": r})
    return pd.DataFrame(rows)


def test_summary_table_groups():
    s = run.summary_table(delta_rows()).set_index(["set", "tagger", "group"])
    assert s.loc[("low", "bilstm", "all"), "n_directions"] == 3
    assert s.loc[("low", "bilstm", "yo"), "mean_delta_f1"] == pytest.approx(0.03)
    assert s.loc[("low", "bilstm", "yo"), "above"] == 1
    w = s.loc[("well", "bilstm", "all")]
    assert w["n_pairs"] == 1 and w["n_directions"] == 2                # pair mean of 2 ways
    assert w["mean_delta_f1"] == pytest.approx(0.01)
    assert ("well", "bilstm", "yo") not in s.index                     # no language groups


def test_correlation_table_per_set_and_tagger():
    delta = delta_rows()
    out = run.correlation_table(delta, pd.DataFrame())
    assert set(zip(out["set"], out["tagger"], strict=True)) == {
        (s, k) for s in ("low", "well") for k in run.TAGGERS}
    low = out[(out["set"] == "low") & (out["tagger"] == "bilstm")].iloc[0]
    assert low["rho"] == pytest.approx(1.0)                            # advantage rises with r


def test_cache_entries_from_another_commit_are_ignored(tmp_path):
    path = tmp_path / "x.pkl"
    run.save(path, {"commit": "abc", "v": np.arange(3)})
    assert run.cached(path, "abc")["v"].tolist() == [0, 1, 2]
    assert run.cached(path, "def") is None
    assert run.cached(tmp_path / "missing.pkl", "abc") is None


def test_correlation_table_joins_bli_and_transfer():
    rng = np.random.default_rng(0)
    delta, transfer = [], []
    for i in range(8):
        mrr_d = rng.uniform(0.1, 0.5)
        mrr_c = mrr_d + rng.normal(0, 0.02)
        pair = f"1:x{i}-y"
        delta.append({"set": "low", "stage": "", "pair": pair, "low": f"x{i}", "src": f"x{i}",
                      "tgt": "y", "hub": "en", "tagger": "bilstm", "r": rng.normal(),
                      "delta_f1": (mrr_c - mrr_d) * 3, "lo": 0, "hi": 0, "mrr_direct": mrr_d,
                      "mrr_chain": mrr_c, "delta_mrr": mrr_c - mrr_d})
        for method, mrr in (("direct", mrr_d), ("chain", mrr_c), ("nomap", 0.0)):
            transfer.append({"set": "low", "stage": "", "pair": pair, "src": f"x{i}",
                             "tgt": "y", "hub": "en", "method": method, "tagger": "bilstm",
                             "f1": 0.1 + mrr})
    out = run.correlation_table(pd.DataFrame(delta), pd.DataFrame(transfer)).set_index("what")
    assert out.loc["advantage in F1 vs advantage in MRR", "rho"] == pytest.approx(1.0)
    both = out.loc["F1 vs MRR, directions x {direct, chain}"]
    assert both["n"] == 16 and both["rho"] > 0.99
