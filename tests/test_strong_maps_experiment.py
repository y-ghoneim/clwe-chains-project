"""Tests of experiments/strong-maps/run.py's helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

RUN = Path(__file__).resolve().parents[1] / "experiments" / "strong-maps" / "run.py"
_spec = importlib.util.spec_from_file_location("strong_maps_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_units_and_scenarios():
    u = run.units()
    assert u[:4] == ["s1", "a", "a2", "d"] and "dc_b250" in u and "dc_b16000" in u
    assert run.scenario_of("a2") == ("a2", None)
    assert run.scenario_of("dc_b4000") == ("dc", 4000)


def test_unit_log_is_where_the_runner_writes_it():
    """run_info reads each unit's start-commit lines from here (it once looked one folder up,
    without the version, and found none)."""
    for unit in ("s1", "a", "dc_b4000"):
        run.configure(unit)
        assert run.unit_log(unit) == run.s1.out_dir() / "run.log"
        assert run.unit_log(unit).parent.parent.name == run.VERSION


def test_methods_pilot_has_every_candidate_full_only_the_chosen():
    pilot = run.vm_names(True, "pilot", None)
    assert {"thin", "chain", "combo", "joined", "w0.3", "chain2", "chain3", "hubs3", "seeded",
            "thin_shared"} <= set(pilot)
    assert run.vm_names(False, "full", "combo") == ["thin", "chain", "combo"]
    assert "seeded" in run.vm_names(False, "full", "seeded")
    assert "w0.9" in run.vm_names(False, "full", "wcombo")
    d = run.methods_for(run.XCFG["scenarios"]["d"], "pilot", None)
    assert ("proc", "thin") in d and ("rc", "combo") in d and ("vm", "joined") not in d
    assert ("rc", "thin") not in run.methods_for(run.XCFG["scenarios"]["s1"], "full", "combo")


def _pooled(scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Two directions per scenario, each with the given delta per method."""
    rows = []
    for sc, per in scores.items():
        for i in range(2):
            for m, v in per.items():
                rows.append({"unit": sc, "scenario": sc, "src": f"{sc}{i}", "tgt": "x",
                             "s": f"{sc}{i}", "method": m, "delta": v, "mrr": 0.5,
                             "mrr_thin": 0.5, "lo": v - 0.01, "hi": v + 0.01})
    return pd.DataFrame(rows)


def test_choose_weights_scenarios_equally_and_breaks_ties_to_the_simpler():
    base = {"vm_chain": 0.0, "vm_combo": 0.03, "vm_hubs3": 0.034, "vm_seeded": 0.01,
            **{f"vm_w{w}": 0.02 for w in run.XCFG["weights"]}}
    pooled = _pooled({"s1": {**base, "vm_w0.3": 0.025}, "d": dict(base)})
    choice, table = run.choose(pooled)
    assert choice["w"] == 0.3
    assert choice["candidate"] == "combo"            # hubs3 is within the 0.005 tie of combo
    pooled = _pooled({"s1": {**base, "vm_hubs3": 0.06}, "d": dict(base)})
    assert run.choose(pooled)[0]["candidate"] == "hubs3"
    assert set(table["candidate"]) == set(run.XCFG["candidates"])


def test_weight_cv_picks_w_without_the_held_out_pairs():
    rows = []
    for i in range(10):
        for w in run.XCFG["weights"]:
            rows.append({"unit": "d", "scenario": "d", "src": f"l{i}", "tgt": "x", "s": f"l{i}",
                         "method": f"vm_w{w}", "delta": 0.01 if w == 0.4 else 0.0, "mrr": 0.5,
                         "mrr_thin": 0.5, "lo": 0, "hi": 0})
    out = run.weight_cv(pd.DataFrame(rows))
    assert len(out) == 10 and (out["method"] == "vm_wcv").all() and (out["w"] == 0.4).all()


class _Cache(dict):
    def put(self, k, v):
        self[k] = v


def _fake_ctx(tmp_path, emb, n_concepts=10):
    rows = np.arange(n_concepts, len(emb["s"]))
    pairs = pd.DataFrame({"a": rows, "b": rows})
    counts = pairs.assign(n=1)

    class FakeCtx:
        def __init__(self):
            self.store, self.cache, self.maps = emb, _Cache(), tmp_path
            self.unit, self.scenario, self.budget = "d", "d", None
            self.sc = {"joined": True, "rcsls": True}
            self.folds = {("s", "t"): {0: list(range(n_concepts))}}
            self.query = {(lang, c): c for lang in ("s", "t") for c in range(n_concepts)}
            self.gold = {(a, b, kind, c): [c] for a, b in (("s", "t"), ("t", "s"))
                         for kind in ("main", "list") for c in range(n_concepts)}

        def test_rows(self, lang, concepts):
            return set(concepts)

        def dictionary(self, name):
            return counts if name.count("__") == 2 else pairs

        def vocab(self, lang):
            return np.arange(len(emb[lang]))

    return FakeCtx(), pairs


def test_evaluate_fold_every_method_and_checks(tmp_path, monkeypatch):
    """One pair and fold on small CPU vectors: every method cached in both directions; proc
    equals stage 1's combination; w0.5 equals combo; self-learning kept the test rows out; the
    hub-edge fits are stored on disk and reused."""
    from clwe_chains.evaluate import bli_with_combination

    g = torch.Generator().manual_seed(0)
    n, dim = 120, 8
    base = torch.nn.functional.normalize(torch.randn(n, dim, generator=g, dtype=torch.float64),
                                         dim=1)
    emb = {lang: torch.nn.functional.normalize(
        base @ torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=torch.float64))[0]
        + 0.1 * torch.randn(n, dim, generator=g, dtype=torch.float64), dim=1)
        for lang in ("s", "t", "x", "y", "z")}
    monkeypatch.setattr(run.s1, "EVAL", {**run.s1.EVAL, "half": False, "csls_k": 3})
    monkeypatch.setitem(run.XCFG, "selflearn", {"k": 3, "rounds": 3, "tol": 0.01,
                                                "min_full": 20, "final": "procrustes"})
    monkeypatch.setitem(run.XCFG, "rcsls", {"k": 3, "lr": 1.0, "niter": 3})
    ctx, pairs = _fake_ctx(tmp_path, emb)
    methods = run.methods_for(ctx.sc, "pilot", None)
    done = run.evaluate_fold(ctx, "s", "t", "x", ["y", "z"], 0, methods)
    assert done > 0 and len(ctx.cache) == 2 * len(methods)
    key = run.s1.key
    rec = ctx.cache[key("s", "t", ("vm", "combo"), 0)]
    assert ctx.cache[key("s", "t", ("vm", "w0.5"), 0)]["rank"] == rec["rank"]
    assert all(info["test_hits"] == 0 for info in rec["fits"].values())
    assert {"direct", "sx", "xt", "joined", "sx2", "xt3", "seeded"} <= set(rec["fits"])
    w_d = run.s1.edge_map(pairs, emb["s"], emb["t"], True)
    w_c = run.s1.compose([run.s1.edge_map(pairs, emb["s"], emb["x"], True),
                          run.s1.edge_map(pairs, emb["x"], emb["t"], True)])
    ref = bli_with_combination(emb["s"], emb["t"], w_d, w_c, np.arange(10),
                               [[[c] for c in range(10)]] * 2, k=3,
                               block=run.s1.CFG["eval"]["block_rows"])
    assert ctx.cache[key("s", "t", ("proc", "combo"), 0)]["rank"] == ref["combo"][0].tolist()
    assert ctx.cache[key("s", "t", ("proc", "thin"), 0)]["rank"] == ref["a"][0].tolist()
    assert run.evaluate_fold(ctx, "s", "t", "x", ["y", "z"], 0, methods) == 0     # all cached
    stored = sorted(tmp_path.glob("*.pt"))
    assert len(stored) == 6 + 4                      # vm: 6 hub edges; rc: 2 edges x 2 directions
    ctx2, _ = _fake_ctx(tmp_path, emb)
    run.evaluate_fold(ctx2, "s", "t", "x", ["y", "z"], 0, methods)
    assert sorted(tmp_path.glob("*.pt")) == stored
    assert ctx2.cache[key("t", "s", ("vm", "hubs3"), 0)]["rank"] == \
        ctx.cache[key("t", "s", ("vm", "hubs3"), 0)]["rank"]


@pytest.mark.data
def test_pilot_pairs_follow_the_rule():
    pilot = run.make_pilot()
    per = pilot.groupby("unit").size()
    assert per["s1"] == 6 and per["a"] == 4 and per["a2"] == 4 and per["d"] == 6
    assert pilot[pilot["unit"] == "s1"]["s"].nunique() == 6
    assert pilot[pilot["unit"] == "a"]["s"].nunique() == 4
    d = set(zip(*[pilot[pilot["unit"] == "d"][c] for c in ("s", "t")], strict=True))
    for u in ("dc_b250", "dc_b1000", "dc_b4000", "dc_b16000"):
        assert set(zip(*[pilot[pilot["unit"] == u][c] for c in ("s", "t")], strict=True)) <= d
    assert pilot.equals(run.make_pilot())


def test_finite_refuses_nan_maps_and_version_is_stable():
    with pytest.raises(FloatingPointError):
        run.finite({"ok": torch.eye(3), "bad": torch.full((3, 3), float("nan"))}, "s", "t", 0)
    run.finite({"ok": torch.eye(3)}, "s", "t", 0)
    assert run.fit_version() == run.VERSION and len(run.VERSION) == 10
    run.configure("d")
    assert run.VERSION in run.s1.NAME


def test_choose_skips_candidates_without_rows():
    base = {"vm_chain": 0.0, "vm_combo": 0.03, **{f"vm_w{w}": 0.02 for w in run.XCFG["weights"]}}
    assert run.choose(_pooled({"s1": base, "d": base}))[0]["candidate"] == "combo"
