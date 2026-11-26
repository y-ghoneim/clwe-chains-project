"""Tests of prep/18_controlled_scarcity.py and experiments/downstream-tier2/run.py helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = load("prep18", "prep/18_controlled_scarcity.py")
tier2 = load("tier2_run", "experiments/downstream-tier2/run.py")


def test_nested_order_is_fixed_and_differs_by_pair():
    a = prep.nested_order(1000, 0, "en__ar")
    assert sorted(a.tolist()) == list(range(1000))                 # a permutation
    assert (a == prep.nested_order(1000, 0, "en__ar")).all()       # deterministic
    assert not (a == prep.nested_order(1000, 0, "fr__fi")).all()   # per-pair order
    small, large = set(a[:250].tolist()), set(a[:500].tolist())
    assert small <= large                                          # budgets are nested


def test_summary_table_keeps_hindi_apart_and_averages_pairs():
    rows = []
    for pair, (src, tgt, d, hindi) in enumerate([("en", "de", 0.04, False),
                                                 ("de", "en", 0.02, False),
                                                 ("en", "hi", -0.5, True)]):
        rows.append({"task": "umsab", "budget": 250, "pair": "en-de" if pair < 2 else "en-hi",
                     "src": src, "tgt": tgt, "model": "lr", "method": "chain", "delta_f1": d,
                     "lo": d - 0.01, "hi": d + 0.01, "hindi": hindi})
    s = tier2.summary_table(pd.DataFrame(rows)).set_index("group")
    assert s.loc["main", "n_pairs"] == 1 and s.loc["main", "n_directions"] == 2
    assert s.loc["main", "mean_delta_f1"] == pytest.approx(0.03)
    assert s.loc["romanised_hindi", "mean_delta_f1"] == pytest.approx(-0.5)


def test_correlation_table_skips_full_and_hindi_and_joins_bli(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows, bli = [], []
    for i in range(8):
        for budget in (250, 500, "full"):
            d = rng.normal()
            rows.append({"task": "sib", "budget": budget, "pair": f"p{i}", "src": f"a{i}",
                         "tgt": "b", "model": "lr", "method": "chain", "delta_f1": d,
                         "r": d + rng.normal(0, 0.01), "hindi": False})
            bli.append({"budget": budget, "src": f"a{i}", "tgt": "b", "method": "chain",
                        "delta": d})
    path = tmp_path / "cells.csv"
    pd.DataFrame(bli).to_csv(path, index=False)
    monkeypatch.setattr(tier2, "BLI_CELLS", path)
    out = tier2.correlation_table(pd.DataFrame(rows)).set_index("what")
    assert out.loc["advantage in F1 vs r", "n"] == 16                  # "full" left out
    assert out.loc["advantage in F1 vs advantage in MRR (same cell)", "rho"] == pytest.approx(1.0)


def test_prep_keeps_hub_edges_full_when_they_are_another_pairs_direct(tmp_path, monkeypatch):
    """en__tr is the direct dictionary of en-tr and the hub edge of ko-tr (via en)."""
    p6 = tmp_path / "phase6"
    base = p6 / "d"
    (base / "dicts").mkdir(parents=True)
    pd.DataFrame({"s": ["en", "ko"], "t": ["tr", "tr"], "hub": ["fr", "en"]}).to_csv(
        base / "pairs.csv", index=False)
    for name, n in (("en__tr", 100), ("en__fr", 100), ("fr__tr", 100), ("ko__tr", 100),
                    ("ko__en", 100)):
        pd.DataFrame({"a": np.arange(n), "b": np.arange(n)}).to_parquet(
            base / "dicts" / f"{name}.parquet", index=False)
    for name in prep.COPIED:
        pd.DataFrame({"x": [1]}).to_parquet(base / name, index=False)

    class Cfg(dict):
        def path(self, key):
            return p6

    monkeypatch.setattr(prep, "load_config",
                        lambda: Cfg(controlled_scarcity={"bases": ["d"], "budgets": [10],
                                                         "seed": 0}))
    prep.main()
    out = p6 / "dc" / "b10"
    assert len(pd.read_parquet(out / "direct" / "en__tr.parquet")) == 10      # shrunk direct
    assert len(pd.read_parquet(out / "dicts" / "en__tr.parquet")) == 100      # hub edge full
    assert len(pd.read_parquet(p6 / "dc" / "bfull" / "direct" / "en__tr.parquet")) == 100


def test_binned_table_by_r_excludes_hindi():
    d = pd.DataFrame({"task": ["sib"] * 3, "model": ["lr"] * 3, "method": ["chain"] * 3,
                      "r": [0.5, 3.5, 3.7], "delta_f1": [0.01, 0.1, 0.2],
                      "hindi": [False, False, True]})
    b = tier2.binned_table(d).set_index("r_bin")
    assert b.loc["3-4", "n"] == 1 and b.loc["3-4", "mean_delta_f1"] == pytest.approx(0.1)
    assert b.loc["0-1", "mean_delta_f1"] == pytest.approx(0.01)
