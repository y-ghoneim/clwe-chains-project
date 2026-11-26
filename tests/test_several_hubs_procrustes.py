"""Tests of experiments/several-hubs-procrustes/run.py's helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

RUN = Path(__file__).resolve().parents[1] / "experiments" / "several-hubs-procrustes" / "run.py"
_spec = importlib.util.spec_from_file_location("several_hubs_procrustes_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_configure_uses_own_cache_and_watches_own_code():
    for unit in run.UNITS:
        run.configure(unit)
        assert run.s1.out_dir() == run.s1.CFG.path("results") / run.NAME / unit
        assert f"experiments/{run.NAME}/" in run.s1.RESULT_PATHS
        assert "experiments/strong-maps/" in run.s1.RESULT_PATHS
        assert run.unit_log(unit) == run.s1.out_dir() / "run.log"


def test_several_hubs_is_strong_maps_rule():
    """Direct plus the chains through hubs 1-3, equal weights; the combination direct + chain."""
    assert run.COMBOS == {"combo": ("thin", "chain"),
                          "hubs3": ("thin", "chain", "chain2", "chain3")}
    assert [n for _, n in run.METHODS] == ["thin", "chain", "chain2", "chain3", "combo", "hubs3"]


def test_backward_chain_is_the_forward_chain_transposed():
    """Procrustes maps are rotations: the t -> s chain must be the s -> t chain's inverse."""
    g = torch.Generator().manual_seed(0)
    es, ex, et = (torch.nn.functional.normalize(torch.randn(40, 5, generator=g), dim=1)
                  for _ in range(3))
    pairs = pd.DataFrame({"a": range(30), "b": np.random.default_rng(0).permutation(30)})
    d = {"sx": pairs, "xt": pairs.rename(columns={"a": "b", "b": "a"})[["a", "b"]]}
    fw = run.chain_map(d, "", es, ex, et, True)
    bw = run.chain_map(d, "", es, ex, et, False)
    assert torch.allclose(bw, fw.T, atol=1e-5)
    assert torch.allclose(fw @ fw.T, torch.eye(5), atol=1e-5)


def _scores(rr_a, rr_b, concepts):
    def frame(rr):
        return pd.DataFrame({"concept": concepts, "rr": rr, "rr_list": rr})
    return {"proc_hubs3": frame(rr_a), "proc_thin": frame(rr_b)}


def test_compare_pools_test_words_and_counts_directions():
    scores = {("x", "y"): _scores([1.0, 1.0, 0.5], [0.5, 0.5, 0.5], [1, 2, 3]),
              ("y", "x"): _scores([0.2, 0.2], [0.2, 0.2], [1, 2])}
    per, rows, summary = run.compare("a", scores, {"n": 200, "seed": 0})
    assert len(per) == 4 and len(rows) == 2
    s = summary[0]
    assert (s["a"], s["b"], s["n_dirs"], s["n_words"]) == ("proc_hubs3", "proc_thin", 2, 5)
    assert s["delta"] == pytest.approx(1.0 / 5)
    assert s["n_below"] == 0


def test_compare_refuses_mismatched_test_words():
    scores = {("x", "y"): _scores([1.0, 1.0], [0.5, 0.5], [1, 2])}
    scores["x", "y"]["proc_thin"]["concept"] = [1, 3]
    with pytest.raises(ValueError):
        run.compare("a", scores, {"n": 10, "seed": 0})
