"""Tests of experiments/contrastivebli-baseline/run.py's helpers."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch

RUN = Path(__file__).resolve().parents[1] / "experiments" / "contrastivebli-baseline" / "run.py"
_spec = importlib.util.spec_from_file_location("contrastivebli_baseline_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def _pairs(n: int) -> pd.DataFrame:
    return pd.DataFrame({"s": [f"l{i:02d}" for i in range(n)], "t": ["zz"] * n})


def test_draw_skips_pilot_pairs_in_either_orientation_and_extends_in_order():
    pairs = _pairs(20)
    pilot = pd.DataFrame({"s": ["l03", "zz"], "t": ["zz", "l07"]})
    five = run.draw_pairs(pairs, pilot, 5, seed=0)
    three = run.draw_pairs(pairs, pilot, 3, seed=0)
    assert len(five) == 5 and list(five["order"]) == list(range(5))
    assert not {"l03", "l07"} & set(five["s"]) and not {"l03", "l07"} & set(
        run.draw_pairs(pairs, pilot, 18, seed=0)["s"])
    assert three[["s", "t"]].equals(five[["s", "t"]].head(3))
    assert five.equals(run.draw_pairs(pairs.sample(frac=1, random_state=1), pilot, 5, seed=0))


def test_cap_keeps_the_most_frequent_pairs_ties_in_file_order():
    d = pd.DataFrame({"a": [9, 1, 5, 2, 3], "b": [1, 8, 2, 2, 3]})
    got = run.cap_pairs(d, 3)
    assert list(zip(got["a"], got["b"], strict=True)) == [(2, 2), (3, 3), (5, 2)]
    assert run.cap_pairs(d, None) is d and run.cap_pairs(d, 10) is d


def test_drop_test_removes_induced_pairs_with_a_test_row_on_their_side():
    a, b, dropped = run.drop_test([1, 2, 3, 4], [10, 20, 30, 40], {2}, {40, 1})
    assert (a, b, dropped) == ([1, 3], [10, 30], 2)       # 1 is a test row only on the other side


def test_shared_pairs_keep_the_drawn_order_and_only_pairs_the_data_has():
    drawn = pd.DataFrame({"s": ["a", "c", "e"], "t": ["b", "d", "f"], "order": [0, 1, 2]})
    have = pd.DataFrame({"s": ["f", "a"], "t": ["e", "b"]})     # either orientation
    got = run.shared_pairs(drawn, have)
    assert list(zip(got["s"], got["order"], strict=True)) == [("a", 0), ("e", 2)]


def test_budget_groups_read_the_cut_direct_dictionary():
    class Ctx:
        budget = None
    assert run.direct_name(Ctx, "en", "ja") == "en__ja"
    Ctx.budget = 1000
    assert run.direct_name(Ctx, "en", "ja") == "direct:en__ja"


def test_groups_sharing_pairs_never_share_a_cache_or_maps():
    """The well-resourced group and its 1,000-pair cut have the same pairs and the same cache
    keys: one folder for both let a 1,000-pair fit pass as the full dictionary's (remote smoke
    test, 2026-11-24)."""
    assert run.group_root("d") != run.group_root("dc_b1000")
    assert run.group_root("d").parent == run.results_root()


def test_ours_come_from_both_pipelines_caches_for_the_group():
    strong = run.ours_cache("strong-maps", "a").path
    plain = run.ours_cache("several-hubs-procrustes", "a").path
    assert strong.parts[-4:] == ("strong-maps", run.sm.VERSION, "a", "evals.jsonl")
    assert plain.parts[-3:] == ("several-hubs-procrustes", "a", "evals.jsonl")
    names = {f"{p}_{n}" for ms in run.OURS.values() for p, n in ms}
    assert {a for a, _ in run.COMPARE} - {"c1_rule", "c1_published"} <= names


def test_published_variant_only_for_the_first_pairs_of_its_group():
    n = run.XCFG["groups"]["d"]["published_pairs"]
    assert run.variants("d", n - 1) == ["rule", "published"]
    assert run.variants("d", n) == ["rule"]
    assert run.variants("a", 0) == ["rule"]


def test_c1_args_are_the_authors_settings():
    a = run.c1_args("5k")
    assert (a.num_steps, a.num_sl, a.num_sample, a.lr, a.num_aug, a.dico_max_rank) == (
        200, 2, 150, 1.5, 10000, 60000)
    b = run.c1_args("1k")
    assert (b.num_steps, b.num_sl, b.num_sample, b.gamma, b.neg_max) == (50, 3, 60, 1.0, 60000)


@pytest.mark.skipif(not torch.cuda.is_available() or not (run.c1_root() / "C1").exists(),
                    reason="needs CUDA and the authors' code")
def test_split_batch_gives_the_full_batch_gradient():
    c1 = run.load_c1()
    args = run.c1_args("1k")
    args.D_emb, args.num_sample = 6, 2
    torch.manual_seed(0)
    src, tgt = torch.randn(7, 3, 6), torch.randn(7, 3, 6)
    grads = []
    for mini in (0, 3):
        torch.manual_seed(1)
        model = c1["models"].C1_Model(args).cuda()
        run.split_step(c1, model, args, c1["util"].get_log_loss_dict_(), src, tgt, mini)
        grads.append([p.grad.clone() for p in model.parameters()])
    for g0, g1 in zip(*grads, strict=True):
        assert torch.allclose(g0, g1, atol=1e-6)


def test_drawn_pairs_file_if_present():
    path = run.HERE / "pairs.csv"
    if not path.exists():
        pytest.skip("pairs.csv not made yet")
    p = pd.read_csv(path)
    pilot = pd.read_csv(run.sm.HERE / "pilot_pairs.csv")
    assert set(p["group"]) == set(run.XCFG["groups"])
    for g, c in run.XCFG["groups"].items():
        mine = p[p["group"] == g]
        if "pairs_from" in c:                 # a subset of the other group's pairs, same order
            src = p[p["group"] == c["pairs_from"]].set_index(["s", "t"])["order"]
            assert all(src[s, t] == o for s, t, o in zip(mine["s"], mine["t"], mine["order"],
                                                          strict=True))
        else:
            assert len(mine) == run.XCFG["n_pairs"]
    skip = {(u, *sorted(x)) for u, *x in zip(pilot["unit"], pilot["s"], pilot["t"], strict=True)}
    assert not any((g, *sorted((s, t))) in skip
                   for g, s, t in zip(p["group"], p["s"], p["t"], strict=True))
