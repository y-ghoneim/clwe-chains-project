"""Tests of experiments/panlex-bli/run.py helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from clwe_chains.muse import Vocab
from clwe_chains.stats import paired_bootstrap

EXP = Path(__file__).resolve().parents[1] / "experiments" / "panlex-bli"
_spec = importlib.util.spec_from_file_location("panlex_bli_run", EXP / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def _frame(pairs):
    return pd.DataFrame(pairs, columns=["a", "b"], dtype="int64")


def test_read_trans_keeps_tab_separated_pairs(tmp_path):
    f = tmp_path / "x.trans"
    f.write_text("азот\tnitrogen\nbad line\nмол\tmol\n\tempty\n", encoding="utf-8")
    d = run.read_trans(f)
    assert d.values.tolist() == [["азот", "nitrogen"], ["мол", "mol"]]


def test_directions_are_all_ordered_pairs():
    d = run.directions(["a", "b", "c"])
    assert len(d) == 6 and ("a", "b") in d and ("b", "a") in d and ("a", "a") not in d
    assert len(run.directions(run.XCFG["languages"])) == 210
    assert all(isinstance(x, str) for x in run.XCFG["languages"])   # "no" is not YAML's false
    on_disk = {p.name.split("-")[0] for p in (run.CFG.data_root / run.XCFG["data"]).glob("*-bg")}
    assert set(run.XCFG["languages"]) == on_disk | {"bg"}


def test_queries_per_pair_and_per_word():
    vs, vt = Vocab(["dog", "cat", "Dog"], "en"), Vocab(["hund", "katze", "köter"], "de")
    test = pd.DataFrame({"src": ["dog", "dog", "cat", "bird"],
                         "tgt": ["hund", "köter", "katze", "vogel"]})
    q = run.test_queries(test, vs, vt)
    assert q["queries"] == [0, 0, 1]                    # bird has no row: dropped
    assert q["gold_pair"] == [[0], [2], [1]]            # each pair: its one translation
    assert q["gold_word"] == [[0, 2], [0, 2], [1]]      # each word: any of its translations
    assert q["word_first"] == [0, 2]
    assert (q["test_pairs"], q["test_pairs_used"], q["test_words"], q["test_words_used"]) == \
        (4, 3, 3, 2)


def test_test_words_removed_at_every_casing_and_from_the_right_side():
    v = Vocab(["dog", "Dog", "cat"], "en")
    rows = run.casing_rows(["dog"], v)
    assert rows == {0, 1}
    sx = _frame([[0, 5], [1, 6], [2, 7]])
    assert run.strip_test(sx, rows, "a")["a"].tolist() == [2]
    xt = _frame([[5, 0], [6, 2]])
    assert run.strip_test(xt, rows, "b")["b"].tolist() == [2]


def test_hubs_ranked_by_narrowest_step_ties_by_code():
    sizes = {"fi": (4000, 900), "eu": (950, 950), "et": (950, 3000), "hu": (5000, 4000)}
    assert run.rank_hubs(sizes, 3) == ["hu", "et", "eu"]


def test_control_adds_mutual_best_joined_pairs_summed_over_hubs():
    direct = _frame([[0, 0]])
    # hub 1 links s1-t1 (via 10) and s1-t2 (via 11); hub 2 links s1-t1 again (via 20)
    h1 = (_frame([[1, 10], [1, 11]]), _frame([[10, 1], [11, 2]]))
    h2 = (_frame([[1, 20]]), _frame([[20, 1]]))
    out = run.joined(direct, [h1, h2])
    assert out.values.tolist() == [[0, 0], [1, 1]]      # s1-t1: 2 links, s1-t2: 1, not best


def test_chunked_bootstrap_matches_its_statistic():
    rng = np.random.default_rng(1)
    d = rng.normal(0.1, 1, 5000)
    mean, lo, hi = run.pooled_bootstrap(d, 1000, 0)
    ref = paired_bootstrap(d, np.zeros_like(d), n=1000, seed=0)
    assert mean == ref[0] and lo < mean < hi
    assert abs(lo - ref[1]) < 0.01 and abs(hi - ref[2]) < 0.01


def test_methods_and_comparisons():
    assert run.METHODS == ["thin", "chain", "chain2", "chain3", "joined", "combo", "hubs3"]
    assert run.COMBOS["hubs3"] == ("thin", "chain", "chain2", "chain3")
    assert all(a in run.METHODS and b in run.METHODS for a, b in run.COMPARE)


def test_bases_ours_unchanged_and_paper_as_the_check():
    assert run.XCFG["bases"]["ours"] == {"normalize": None, "retrieval": "csls"}
    assert run.XCFG["bases"]["paper"] == {"normalize": ["unit"], "retrieval": "nn"}
