"""Tests of experiments/ensemble-control/run.py's helpers."""

import importlib.util
from pathlib import Path

import pandas as pd

RUN = Path(__file__).resolve().parents[1] / "experiments" / "ensemble-control" / "run.py"
_spec = importlib.util.spec_from_file_location("ensemble_control_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_bootstrap_is_fixed_same_size_and_differs_by_fold_and_pair():
    d = pd.DataFrame({"a": range(50), "b": range(100, 150)})
    one = run.bootstrap(d, 0, "en__ar", 0)
    assert len(one) == len(d)
    assert one.equals(run.bootstrap(d, 0, "en__ar", 0))
    assert not one.equals(run.bootstrap(d, 0, "en__ar", 1))
    assert not one.equals(run.bootstrap(d, 0, "en__tr", 0))
    assert one["a"].duplicated().any()                  # drawn with replacement
    assert (one["b"] - one["a"] == 100).all()           # rows stay whole


def test_methods_for_names_every_map_and_combination():
    got = run.methods_for("fr", ["boot", "lstsq"])
    assert got == [("thin",), ("chain", "fr"), ("combo", "fr"), ("boot",), ("ensboot",),
                   ("lstsq",), ("enslstsq",)]
    assert [run.s1.label(m, "fr") for m in got] == ["thin", "chain", "combo", "boot", "ensboot",
                                                     "lstsq", "enslstsq"]


def test_evaluate_direction_caches_every_map_and_combination(monkeypatch):
    """One direction of one fold on tiny CPU vectors: every method cached, combo equal to stage
    1's two-map combination, the bootstrap and least-squares partners present."""
    import numpy as np
    import torch

    from clwe_chains.evaluate import bli_with_combination

    torch.manual_seed(0)
    n, dim = 60, 8
    emb = {lang: torch.nn.functional.normalize(torch.randn(n, dim, dtype=torch.float64), dim=1)
           for lang in ("s", "t", "x")}
    run.configure("d")
    monkeypatch.setattr(run.s1, "EVAL", {**run.s1.EVAL, "half": False, "csls_k": 3})

    class Cache(dict):
        def put(self, k, v):
            self[k] = v

    class FakeCtx:
        def __init__(self):
            self.store, self.cache = emb, Cache()
            self.folds = {("s", "t"): {0: list(range(10))}}
            self.query = {(lang, c): c for lang in ("s", "t") for c in range(10)}
            self.gold = {(a, b, kind, c): [c] for a, b in (("s", "t"), ("t", "s"))
                         for kind in ("main", "list") for c in range(10)}

        def methods(self, s, hub):
            return run.methods_for(hub, run.partners())

    rows = np.arange(10, n)
    pairs = pd.DataFrame({"a": rows, "b": rows})
    d = {"direct": pairs, "sx": pairs, "xt": pairs,
         "boot": run.bootstrap(pairs, 0, "s__t", 0)}
    ctx = FakeCtx()
    assert run.evaluate_direction(ctx, "s", "t", "x", 0, True, d, {}) == 4
    got = {k: v for k, v in ctx.cache.items()}
    assert len(got) == 7
    w_d = run.s1.edge_map(pairs, emb["s"], emb["t"], True)
    w_c = run.s1.compose([run.s1.edge_map(pairs, emb["s"], emb["x"], True),
                          run.s1.edge_map(pairs, emb["x"], emb["t"], True)])
    ref = bli_with_combination(emb["s"], emb["t"], w_d, w_c, np.arange(10),
                               [[[c] for c in range(10)]] * 2, k=3,
                               block=run.s1.CFG["eval"]["block_rows"])
    assert got[run.s1.key("s", "t", ("combo", "x"), 0)]["rank"] == ref["combo"][0].tolist()
    assert got[run.s1.key("s", "t", ("thin",), 0)]["rank"] == ref["a"][0].tolist()
    assert run.evaluate_direction(ctx, "s", "t", "x", 0, True, d, {}) == 0     # all cached
    # backward: t -> s, maps fitted the other way round, chain t -> x -> s
    assert run.evaluate_direction(ctx, "s", "t", "x", 0, False, d, {}) == 4
    w_d = run.s1.edge_map(pairs, emb["s"], emb["t"], False)
    w_c = run.s1.compose([run.s1.edge_map(pairs, emb["x"], emb["t"], False),
                          run.s1.edge_map(pairs, emb["s"], emb["x"], False)])
    ref = bli_with_combination(emb["t"], emb["s"], w_d, w_c, np.arange(10),
                               [[[c] for c in range(10)]] * 2, k=3,
                               block=run.s1.CFG["eval"]["block_rows"])
    assert ctx.cache[run.s1.key("t", "s", ("combo", "x"), 0)]["rank"] == ref["combo"][0].tolist()


def test_gap_summary_bootstraps_over_pairs_and_counts_directions():
    gap = pd.DataFrame({"stage": "d", "what": "combo - ensboot", "src": ["a", "b", "a", "c"],
                        "tgt": ["b", "a", "c", "a"], "s": ["a", "a", "a", "a"],
                        "delta": [0.02, 0.04, 0.01, 0.01], "lo": [0.01, 0.0, -0.01, 0.005],
                        "hi": [0.03, 0.05, 0.02, 0.02]})
    g = run.gap_summary(gap).iloc[0]
    assert g["n_dirs"] == 4 and abs(g["mean_over_dirs"] - 0.02) < 1e-12
    assert g["n_above"] == 2 and g["n_below"] == 0
    assert g["lo_pairs"] <= 0.02 <= g["hi_pairs"]


def test_configure_points_stage_1_runner_at_each_stage():
    run.configure("d")
    assert run.s1.DATA == "d" and run.s1.NAME == "ensemble-control/d"
    assert run.s1.HERE.name == "natural-highres-d"
    assert run.s1.evaluate_direction is run.evaluate_direction
    run.configure("s1")
    assert run.s1.DATA == "" and run.s1.HERE.name == "scarce-natural-s1"
    assert run.partners() == ["boot"]
    assert "experiments/scarce-natural-s1/pairs.csv" in run.s1.RESULT_PATHS
