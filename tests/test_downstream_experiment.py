"""Tests of experiments/downstream-umsab/run.py's helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUN = Path(__file__).resolve().parents[1] / "experiments" / "downstream-umsab" / "run.py"
_spec = importlib.util.spec_from_file_location("downstream_umsab_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def write_dict(root, name, a, b):
    (root / "dicts").mkdir(exist_ok=True)
    pd.DataFrame({"a": np.array(a, dtype="int64"), "b": np.array(b, dtype="int64")}).to_parquet(
        root / "dicts" / f"{name}.parquet", index=False)


def test_edge_indices_follow_the_stored_orientation(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "P6", tmp_path)
    write_dict(tmp_path, "ar__de", [1, 2, 3], [10, 20, 30])          # a rows are ar, b rows are de
    ar, de = run.edge_indices("ar", "de")
    assert (ar.tolist(), de.tolist()) == ([1, 2, 3], [10, 20, 30])
    # only ar__de is stored: asking for de -> ar must swap the columns, not return ar's rows first
    de, ar = run.edge_indices("de", "ar")
    assert (de.tolist(), ar.tolist()) == ([10, 20, 30], [1, 2, 3])


def test_edge_indices_prefer_the_stored_name_when_both_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "P6", tmp_path)
    write_dict(tmp_path, "de__en", [1, 2], [5, 6])
    write_dict(tmp_path, "en__de", [5, 6, 7], [1, 2, 3])               # a different set of pairs
    en, de = run.edge_indices("en", "de")
    assert (en.tolist(), de.tolist()) == ([5, 6, 7], [1, 2, 3])


def test_dictionary_size_either_orientation():
    sizes = pd.Series({"ar__de": 28897, "de__en": 288460})
    assert run.dictionary_size(sizes, "de", "ar") == 28897
    assert run.dictionary_size(sizes, "de", "en") == 288460
    assert run.dictionary_size(sizes, "en", "de") == 288460


def delta_rows():
    """Three pairs (0, 1 distant; 2 same family), both directions, both models."""
    rows = []
    for pair, (stratum, d, lo, hi, r) in enumerate([("distant", 0.05, 0.01, 0.09, 2.0),
                                                    ("distant", -0.03, -0.06, 0.00, -1.0),
                                                    ("same_family", 0.0, -0.02, 0.02, 0.5)]):
        for x, y in (("a", "b"), ("b", "a")):
            for kind in ("lr", "mlp"):
                rows.append({"pair": pair, "x": x, "y": y, "hub": "en", "model": kind,
                             "delta_f1": d, "lo": lo, "hi": hi, "r": r, "stratum": stratum})
    return pd.DataFrame(rows)


def test_summary_table_strata_and_counts():
    s = run.summary_table(delta_rows()).set_index(["model", "stratum"])
    both = ("lr", "all_but_hindi")
    assert s.loc[both, "n_pairs"] == 3 and s.loc[both, "n_directions"] == 6
    assert s.loc[("lr", "distant"), "n_pairs"] == 2
    assert s.loc[("lr", "same_family"), "n_pairs"] == 1
    assert s.loc[both, "above"] == 2 and s.loc[both, "below"] == 0
    assert s.loc[("lr", "distant"), "mean_delta_f1"] == pytest.approx(0.01)


def test_summary_table_skips_an_empty_stratum():
    only_distant = delta_rows().query("stratum == 'distant'")
    s = run.summary_table(only_distant)
    assert set(s["stratum"]) == {"all_but_hindi", "distant"}


def test_stratum_of_puts_hindi_in_its_own_stratum():
    assert run.stratum_of("hi", "en") == run.stratum_of("ar", "hi") == run.HINDI
    assert run.stratum_of("ar", "de") == run.stratum_of("it", "ar") == "distant"
    assert run.stratum_of("en", "de") == "same_family"


def test_hindi_directions_are_kept_out_of_the_other_summaries():
    delta = delta_rows()
    hindi = delta.query("pair == 2").assign(stratum=run.HINDI, delta_f1=0.4, lo=0.3, hi=0.5)
    s = run.summary_table(pd.concat([delta.query("pair != 2"), hindi])).set_index(
        ["model", "stratum"])
    assert s.loc[("lr", run.HINDI), "n_directions"] == 2
    assert s.loc[("lr", run.HINDI), "mean_delta_f1"] == pytest.approx(0.4)
    assert s.loc[("lr", "all_but_hindi"), "n_directions"] == 4      # not the Hindi pair
    assert s.loc[("lr", "all_but_hindi"), "mean_delta_f1"] == pytest.approx(0.01)


def test_binned_and_standout_tables():
    delta = delta_rows()
    b = run.binned_table(delta).set_index(["model", "r_bin"])
    assert b.loc[("lr", "1<r<=2"), "n"] == 2 and b.loc[("lr", "1<r<=2"), "mean_delta_f1"] == 0.05
    assert ("lr", "-1<r<=0") not in b.index                  # an empty bin is not reported
    assert b.loc[("lr", "r<=-1"), "mean_delta_f1"] == -0.03
    top = run.standout_table(delta, k=1)
    assert top["delta_f1"].tolist() == [0.05, -0.03]       # the largest advantage, then the loss
    assert set(top["model"]) == {"lr"}


def test_correlation_table_joins_bli_and_uses_both_methods():
    rng = np.random.default_rng(0)
    rows, transfer = [], []
    for pair in range(6):
        for x, y in ((f"x{pair}", "y"), ("y", f"x{pair}")):
            mrr_d = rng.uniform(0.1, 0.6)
            mrr_c = mrr_d + rng.normal(0, 0.02)
            for kind in ("lr", "mlp"):
                rows.append({"pair": pair, "x": x, "y": y, "model": kind, "r": rng.normal(),
                             "delta_f1": (mrr_c - mrr_d) * 2, "mrr_direct": mrr_d,
                             "mrr_chain": mrr_c, "delta_mrr": mrr_c - mrr_d})
                for method, mrr in (("direct", mrr_d), ("chain", mrr_c)):
                    transfer.append({"x": x, "y": y, "model": kind, "method": method,
                                     "macro_f1": 0.3 + mrr * 0.5})
    out = run.correlation_table(pd.DataFrame(rows), pd.DataFrame(transfer))
    lr = out[out["model"] == "lr"].set_index("what")
    both = lr.loc["F1 vs MRR, directions x {direct, chain}"]
    assert both["n"] == 24 and both["rho"] > 0.99           # 12 directions x 2 methods
    assert lr.loc["advantage in F1 vs advantage in MRR", "rho"] > 0.99


def test_correlation_table_without_the_bli_run_only_reports_r():
    delta = delta_rows().drop(columns=["stratum"])
    out = run.correlation_table(delta, pd.DataFrame())
    assert set(out["what"]) == {"advantage in F1 vs r"}
