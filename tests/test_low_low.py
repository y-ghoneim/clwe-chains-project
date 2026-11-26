"""Tests of experiments/low-low/ (run.py and run_sib200.py helpers)."""

import importlib.util
import itertools
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1] / "experiments" / "low-low"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, EXP / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load("low_low_run", "run.py")
sib = _load("low_low_run_sib200", "run_sib200.py")


def test_pairs_are_every_different_family_pair_of_the_ten():
    cfg = run.s1.CFG
    fam = pd.read_csv(cfg.path("v0_languages")).set_index("ft_code")["family"]
    low = cfg["phase6_stages"]["ll"]["low"]
    want = {(a, b) for a, b in itertools.combinations(sorted(low), 2) if fam[a] != fam[b]}
    spec = pd.read_csv(EXP / "pairs_spec.csv")
    assert set(zip(spec["s"], spec["t"], strict=True)) == want and len(spec) == 40
    for same in (("as", "or"), ("ckb", "or"), ("as", "ckb"), ("mhr", "myv"), ("jv", "mg")):
        assert same not in want


def test_groups_split_at_100_direct_pairs_and_50_test_words():
    pairs = pd.DataFrame({"s": ["a", "b", "c"], "t": ["x", "y", "z"], "hub": ["en"] * 3})
    sizes = {("a", "x"): 99, ("b", "y"): 100, ("c", "z"): 551}
    n_test = {("a", "x"): 49, ("b", "y"): 50, ("c", "z"): 400}
    g = run.pair_groups(pairs, sizes, n_test)
    assert list(g["group"]) == ["under 100", "100 and above", "100 and above"]
    assert list(g["included"]) == [False, True, True]


def test_stage_is_a_strong_maps_scenario_with_its_own_cache():
    assert run.sm.XCFG["scenarios"]["ll"]["data"] == "ll"
    assert run.sm.XCFG["scenarios"]["ll"]["joined"] is True
    run.configure()
    assert run.s1.out_dir() == run.s1.CFG.path("results") / "low-low" / "ll"
    assert run.unit_log() == run.s1.out_dir() / "run.log"
    for path in ("experiments/strong-maps/", "experiments/several-hubs-procrustes/",
                 "experiments/low-low/", "src/", "prep/", "configs/"):
        assert path in run.s1.RESULT_PATHS


def test_methods_add_pooled_direct_to_several_hubs():
    assert run.MAPS == ["thin", "chain", "chain2", "chain3", "joined"]
    assert [n for _, n in run.METHODS][-2:] == ["combo", "hubs3"]
    assert ("joined", "thin") in run.COMPARE and ("hubs3", "chain") in run.COMPARE


def test_sib_directions_both_ways_hub_rule_and_coverage():
    pairs = pd.DataFrame({"s": ["as", "jv", "mhr", "qu"], "t": ["yo", "or", "yo", "yo"],
                          "hub": ["en", "zh", "en", "es"]})
    dirs, left = sib.directions(pairs)
    assert left == [{"s": "jv", "t": "or", "hub": "zh"}]      # excluded hub, listed
    assert set(zip(dirs["src"], dirs["tgt"], strict=True)) == {("as", "yo"), ("yo", "as"),
                                                               ("qu", "yo"), ("yo", "qu")}
    assert "mhr" not in set(dirs["src"])                       # not in SIB-200


def test_sib_summary_averages_a_pairs_two_directions_first():
    delta = pd.DataFrame({"pair": ["p", "p", "q", "q"], "model": ["lr"] * 4,
                          "delta_f1": [0.1, 0.3, 0.0, 0.0], "lo": [0.05, 0.1, -0.1, -0.1],
                          "hi": [0.2, 0.4, 0.1, 0.1]})
    transfer = pd.DataFrame({"model": ["lr"] * 3, "method": ["direct", "chain", "nomap"],
                             "macro_f1": [0.2, 0.3, 0.1], "accuracy": [0.3, 0.4, 0.2]})
    s = sib.summary(transfer, delta, n_boot=200).iloc[0]
    assert (s["n_pairs"], s["n_directions"], s["above"], s["below"]) == (2, 4, 2, 0)
    assert abs(s["chain_minus_direct"] - 0.1) < 1e-12
    assert s["f1_chain"] == 0.3
