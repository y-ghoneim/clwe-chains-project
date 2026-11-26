"""Tests of prep/20_break_even_grid.py and experiments/break-even-grid/run.py helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = load("prep20", "prep/20_break_even_grid.py")
run = load("break_even_run", "experiments/break-even-grid/run.py")


def test_pairs_need_both_hub_steps_above_the_size():
    pairs = pd.DataFrame({"s": ["a", "a", "c"], "t": ["b", "d", "b"], "hub": ["x", "x", "x"]})
    n = {"a__x": 5000, "x__b": 3000, "x__d": 9000, "c__x": 1000}
    assert prep.kept_pairs(pairs, n, 2000)[["s", "t"]].values.tolist() == [["a", "b"],
                                                                          ["a", "d"]]
    assert prep.kept_pairs(pairs, n, 3000)[["s", "t"]].values.tolist() == [["a", "d"]]


def test_cut_is_nested_keeps_row_order_and_is_the_same_for_a_shared_edge():
    d = pd.DataFrame({"a": np.arange(500), "b": np.arange(500) + 1})
    small, large = prep.cut(d, 100, 0, "en__tr"), prep.cut(d, 200, 0, "en__tr")
    assert len(small) == 100 and set(small["a"]) <= set(large["a"])
    assert (np.diff(small["a"]) > 0).all()                 # original order kept
    assert small.equals(prep.cut(d, 100, 0, "en__tr"))     # an edge is cut one way only
    assert not small.equals(prep.cut(d, 100, 0, "ko__tr"))
    # the same permutation as controlled scarcity's direct subsamples (prep/18)
    order = prep._prep18.nested_order(500, 0, "en__tr")
    assert set(small["a"]) == set(order[:100].tolist())


def test_crossing_interpolates_on_log2_and_marks_censoring_and_rebounds():
    logs = np.log2([500, 1000, 2000, 4000])
    c = run.crossing(logs, [0.05, 0.02, -0.02, -0.03])
    assert c["status"] == "crossed" and 2 ** c["log2"] == pytest.approx(2000 / 2 ** 0.5)
    assert not c["rebound"]
    assert run.crossing(logs, [0.05, -0.01, 0.01, -0.02])["rebound"]
    below = run.crossing(logs, [-0.01, 0.0, 0.0, -0.1])
    assert below["status"] == "below" and below["log2"] == -np.inf
    above = run.crossing(logs[:3], [0.05, 0.04, 0.01])
    assert above["status"] == "above" and above["log2"] == np.inf
    assert 2 ** above["bound"] == pytest.approx(2000)
    assert run.crossing(logs, [0.03, 0.0, 0.01, 0.01])["status"] == "crossed"   # 0 is not a win


def test_median_puts_censored_crossings_at_the_ends():
    assert run.median_log(np.array([1.0, 3.0, 2.0])) == 2.0
    assert run.median_log(np.array([1.0, 2.0, 3.0, 5.0])) == 2.5
    assert run.median_log(np.array([1.0, np.inf, np.inf])) == np.inf
    assert run.median_log(np.array([-np.inf, -np.inf, 4.0])) == -np.inf
    assert np.isnan(run.median_log(np.array([-np.inf, np.inf])))
    assert run.median_log(np.array([1.0, 2.0, np.inf, np.inf])) == np.inf


def test_growth_and_verdict_follow_the_readme_rule():
    h = np.log2(run.XCFG["growth_sizes"])
    assert run.growth(list(h - 1)) == pytest.approx(4.0)          # N* = H / 2: proportional
    assert run.growth([13.0, 13.0, 13.0]) == pytest.approx(1.0)   # flat
    assert run.growth([13.0, 13.5, 14.0]) == pytest.approx(2.0)   # one doubling over 4x
    assert run.growth([13.0, 14.0, np.inf]) == np.inf             # above the grid at 64k
    assert run.growth([-np.inf, 9.0, 10.0]) == np.inf             # below the grid at 16k
    for unclear in ([np.inf, 13.0, 13.0], [13.0, np.inf, 13.0], [np.inf, np.inf, np.inf],
                    [13.0, np.inf, np.inf], [13.0, 13.0, -np.inf], [13.0, np.nan, 14.0]):
        assert np.isnan(run.growth(unclear)), unclear              # left out, never -inf
    assert run.verdict(1.1, 1.8) == "levels off"
    assert run.verdict(2.5, 5.0) == "grows in proportion"
    assert run.verdict(1.5, 3.0) == "undecided"
    assert run.verdict(0.8, 4.5) == "undecided"                    # does not exclude 1x
    # on the medians themselves: README's rule, or what the censoring says
    assert run.growth_verdict([13.0, 13.0, 13.0], 1.1, 1.8) == "levels off"
    text = run.growth_verdict([13.0, np.inf, np.inf], np.nan, np.nan)
    assert text.startswith("no break-even within direct <= 16000 at hub 32000, 64000")


def test_censored_draws_cannot_pull_the_growth_interval_down():
    """The reviewer's case: draws with N* above the grid at 16k must be left out, not count as
    -inf (that would push the interval below 2x and favour "levels off")."""
    sizes = run.XCFG["growth_sizes"]
    rows = []
    rng = np.random.default_rng(1)
    for i in range(9):
        # 4 pairs never cross at hub 16k (still positive at 16k direct): above the grid
        for hub, at in zip(sizes, [np.log2(9000), np.log2(12000), np.log2(15000)], strict=True):
            for n in run.XCFG["direct_sizes"]:
                g = 0.05 if (hub == sizes[0] and i < 4) else 0.02 * (at - np.log2(n))
                g += rng.normal(0, 1e-4)
                for src, tgt in ((f"p{i}", "z"), ("z", f"p{i}")):
                    rows.append({"hub": hub, "direct": n, "s": f"p{i}", "t": "z", "src": src,
                                 "tgt": tgt, "delta": g})
    cells = pd.DataFrame(rows)
    _, grow = run.nstar_table(cells, run.pair_crossings(cells))
    assert grow["lo"] > 0 and grow["undefined_share"] > 0       # never -inf, share reported


def test_same_pairs_check_removes_a_change_of_pairs():
    """Pairs p0-p4 cross at 8k at every hub size; p5-p10 cross at 2k and drop out above 16k. The
    main growth rises only because the pairs change; on the same pairs it is flat."""
    rows = []
    for hub in run.XCFG["growth_sizes"]:
        for i in range(11):
            if i >= 5 and hub > 16000:
                continue
            at = np.log2(8000 if i < 5 else 2000) + 0.01 * i
            rows.append({"hub": hub, "s": f"p{i}", "t": "z", "log2": at, "status": "crossed",
                         "bound": at, "rebound": False})
    cross = pd.DataFrame(rows)
    main = [run.median_log(cross[cross["hub"] == h]["log2"].to_numpy())
            for h in run.XCFG["growth_sizes"]]
    assert run.growth(main) > 2.5                       # 2k -> 8k: looks like growth
    same = run.same_pairs_growth(cross)
    assert same["n_pairs"] == 5 and same["growth"] == pytest.approx(1.0)
    assert same["verdict"] == "levels off"


def test_grid_table_counts_pairs_not_source_languages():
    rows = [{"hub": 1000, "direct": 500, "s": "tr", "t": t, "src": a, "tgt": b, "delta": 0.01,
             "mrr_chain": 0.5, "mrr_direct": 0.49, "lo": 0.0, "hi": 0.02}
            for t in ("eu", "fi", "hu") for a, b in (("tr", t), (t, "tr"))]
    g = run.grid_table(pd.DataFrame(rows))
    assert g.loc[0, "n_pairs"] == 3 and g.loc[0, "n_dirs"] == 6


def test_above_label_is_a_lower_bound_from_each_pairs_own_bound():
    log2 = np.array([12.0, np.inf, np.inf])          # 4096, then two pairs still positive
    bound = np.log2([4096.0, 8000.0, 16000.0])
    lb = run.median_bound(log2, bound)
    assert run.median_log(log2) == np.inf and 2 ** lb == pytest.approx(8000)
    assert run.describe(np.inf, lb) == "above 8000"


def test_pair_crossings_average_the_two_directions_and_skip_one_sided_sizes():
    rows = []
    for n, (d1, d2) in {500: (0.06, 0.04), 1000: (0.01, -0.03), 2000: (-0.02, None)}.items():
        rows.append({"hub": 4000, "direct": n, "s": "a", "t": "b", "src": "a", "tgt": "b",
                     "delta": d1})
        if d2 is not None:
            rows.append({"hub": 4000, "direct": n, "s": "a", "t": "b", "src": "b", "tgt": "a",
                         "delta": d2})
    out = run.pair_crossings(pd.DataFrame(rows))
    assert len(out) == 1 and out.loc[0, "status"] == "crossed"
    # gains 0.05 at 500 and -0.01 at 1000; 2000 has one direction only and is skipped
    assert out.loc[0, "n_star"] == pytest.approx(500 * 2 ** (0.05 / 0.06))
    assert out.loc[0, "gains"] == "500:0.0500;1000:-0.0100"


def test_nstar_table_reports_median_pooled_and_growth():
    rng = np.random.default_rng(0)
    rows = []
    for hub in run.XCFG["growth_sizes"]:
        for i in range(9):
            at = np.log2(8000) + rng.normal(0, 0.3)              # every pair crosses near 8k
            for n in run.XCFG["direct_sizes"]:
                for src, tgt in ((f"p{i}", "z"), ("z", f"p{i}")):
                    rows.append({"hub": hub, "direct": n, "s": f"p{i}", "t": "z", "src": src,
                                 "tgt": tgt, "delta": 0.02 * (at - np.log2(n))})
    cells = pd.DataFrame(rows)
    table, grow = run.nstar_table(cells, run.pair_crossings(cells))
    assert (table["n_crossed"] == 9).all()
    assert table["median_log2"].between(np.log2(6000), np.log2(11000)).all()
    assert table["pooled_log2"].between(np.log2(6000), np.log2(11000)).all()
    assert grow["growth"] == pytest.approx(1.0, abs=0.3) and grow["verdict"] == "levels off"


def test_the_chain_alone_ranks_as_the_combined_pass_does():
    """Controlled scarcity scored the chain inside bli_with_combination; the grid scores it with
    bli_multi. Both must give the same ranks (the check against controlled scarcity's full-route
    chain relies on it)."""
    from clwe_chains.evaluate import bli_multi, bli_with_combination
    g = torch.Generator().manual_seed(0)
    src = torch.nn.functional.normalize(torch.randn(300, 16, generator=g), dim=1)
    tgt = torch.nn.functional.normalize(torch.randn(400, 16, generator=g), dim=1)
    w_a = torch.linalg.qr(torch.randn(16, 16, generator=g))[0]
    w_b = torch.linalg.qr(torch.randn(16, 16, generator=g))[0]
    queries = np.arange(0, 300, 7)
    golds = [[[int(q) % 400] for q in queries], [[int(q + 1) % 400] for q in queries]]
    for half in (False, True):                 # production uses half: true
        both = bli_with_combination(src, tgt, w_a, w_b, queries, golds, k=5, block=64, half=half)
        alone = bli_multi(src, tgt, w_b, queries, golds, k=5, block=64, retrieval="csls",
                          half=half)
        assert (both["b"] == alone).all(), half
