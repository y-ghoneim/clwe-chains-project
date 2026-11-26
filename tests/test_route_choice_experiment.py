"""Tests of experiments/route-choice-s1/run.py (phase 6 stage 1E)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1] / "experiments" / "route-choice-s1" / "run.py"
_spec = importlib.util.spec_from_file_location("route_choice_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_validation_split_never_uses_excluded_rows():
    d = pd.DataFrame({"a": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      "b": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11]})
    train, val = run.validation_split(d, exclude_a={2}, exclude_b={0}, frac=0.25, cap=500)
    # 9 a words remain (2 and 9 dropped); the 2 most frequent (smallest rows) are validation
    assert sorted(val["a"].unique()) == [1, 3] and len(val) == 3
    assert not set(train["a"]) & {1, 3, 2} and 0 not in set(train["b"]) | set(val["b"])
    _, val = run.validation_split(d, set(), set(), frac=0.9, cap=2)
    assert sorted(val["a"].unique()) == [1, 2]


def test_step_test_removes_only_endpoint_words():
    ts, tt = {1}, {2}
    assert run.step_test("yo", "en", "yo", "ja", ts, tt) == ({1}, set())
    assert run.step_test("en", "ja", "yo", "ja", ts, tt) == (set(), {2})
    assert run.step_test("en", "fr", "yo", "ja", ts, tt) == (set(), set())
    assert run.step_test("ja", "en", "yo", "ja", ts, tt) == ({2}, set())     # reversed route


def test_kendall_is_positive_when_cheaper_routes_score_higher():
    assert run.kendall([1, 2, 3, 4], [0.9, 0.8, 0.5, 0.1]) == 1.0
    assert np.isnan(run.kendall([1, 1, 1], [0.1, 0.2, 0.3]))


def test_pick_widest_and_cheapest():
    sizes = np.array([[0, 9, 2, 1], [9, 0, 8, 7], [2, 8, 0, 9], [1, 7, 9, 0]], dtype=float)
    cost = {"uriel": np.ones((4, 4))}
    p = run.pick(cost, sizes, 0, 3, max_mid=2)
    assert p["size_widest"] == (0, 1, 2, 3)        # min(9, 8, 9) = 8 beats (0, 1, 3): 7
    assert p["uriel"] == (0, 1, 3)                 # equal costs per step: fewer hops win


def _world(tmp_path, monkeypatch):
    """lo (low-resource source), nd (target node), hubs ha and hb. ha's edges are clean and large;
    hb's are small and wrong (shuffled pairs), so routes through ha should win."""
    import torch

    from clwe_chains.config import load_config
    root, cdir = tmp_path / "data", tmp_path / "cfg"
    cdir.mkdir()
    (cdir / "default.yaml").write_text(
        f"data_root: {root.as_posix()}\ndevice: cpu\n"
        "embeddings: {dim: 8, max_vocab: 40, normalize: [unit, center, unit]}\n"
        "eval: {csls_k: 3, block_rows: 16, half: false}\n"
        "phase6: {en_chain: [], low: [lo]}\n", encoding="utf-8")
    cfg = load_config(config_dir=cdir)
    g = torch.Generator().manual_seed(0)
    base = torch.randn(40, 8, generator=g)
    emb = cfg.path("v0_embeddings")
    emb.mkdir(parents=True)
    for lang, noise in (("lo", 0.05), ("nd", 0.0), ("ha", 0.0), ("hb", 0.0)):
        q, _ = torch.linalg.qr(torch.randn(8, 8, generator=g))
        np.save(emb / f"{lang}.npy", (base @ q + noise * torch.randn(40, 8, generator=g)).numpy())
    p6 = cfg.path("phase6")
    for sub in ("dicts", "routes/dicts"):
        (p6 / sub).mkdir(parents=True)
    eye = lambda n: pd.DataFrame({"a": np.arange(n), "b": np.arange(n)})
    wrong = pd.DataFrame({"a": np.arange(12), "b": np.random.default_rng(0).permutation(12) + 20})
    eye(10).to_parquet(p6 / "dicts" / "lo__nd.parquet")
    edges = {("lo", "ha"): eye(40), ("lo", "hb"): wrong, ("lo", "nd"): eye(10),
             ("ha", "hb"): wrong, ("ha", "nd"): eye(40), ("hb", "nd"): wrong}
    rows = []
    for (a, b), d in edges.items():
        d.to_parquet(p6 / "routes" / "dicts" / f"{a}__{b}.parquet")
        rows.append({"a": a, "b": b, "pairs": len(d), "a_words": d["a"].nunique(),
                     "b_words": d["b"].nunique()})
    pd.DataFrame(rows).to_csv(p6 / "routes" / "dict_sizes.csv", index=False)
    langs = ["ha", "hb", "lo", "nd"]
    pd.DataFrame([{"a": a, "b": b, "uriel": 0.5} for i, a in enumerate(langs)
                  for b in langs[i + 1:]]).to_csv(p6 / "routes" / "uriel.csv", index=False)
    concepts = list(range(5, 25))
    pd.DataFrame([(lang, c, c, True) for lang in ("lo", "nd") for c in concepts],
                 columns=["lang", "concept_id", "row", "is_word"]).to_parquet(p6 / "words.parquet")
    pd.DataFrame([("lo", "nd", c, i % 5) for i, c in enumerate(concepts)],
                 columns=["s", "t", "concept_id", "fold"]).to_parquet(p6 / "folds.parquet")
    pd.DataFrame([(a, b, c, k, c) for a, b in (("lo", "nd"), ("nd", "lo")) for c in concepts
                  for k in ("main", "list")],
                 columns=["src", "tgt", "concept_id", "kind", "row"]).to_parquet(
        p6 / "gold.parquet")
    here = tmp_path / "exp"
    here.mkdir()
    pd.DataFrame({"s": ["lo"], "t": ["nd"], "direct": [10], "hub": ["ha"], "route": [40]}).to_csv(
        here / "pairs.csv", index=False)
    monkeypatch.setattr(run.s1, "CFG", cfg)
    monkeypatch.setattr(run.s1, "HERE", here)
    monkeypatch.setattr(run.s1, "EVAL", {"csls_k": 3, "half": False, "retrieval": "csls"})
    monkeypatch.setattr(run, "RESULTS", here / "results")
    monkeypatch.setattr(run, "XCFG", {**run.XCFG, "oracle_top": 2,
                                      "validation": {"frac": 0.5, "max": 500, "min": 3}})
    return here


def test_end_to_end_on_synthetic_routes(tmp_path, monkeypatch):
    here = _world(tmp_path, monkeypatch)
    ctx = run.Ctx(limit=None)
    assert ctx.nodes == ["ha", "hb", "nd"]
    plans = run.run_all(ctx)
    plan = plans["lo", "nd"]
    assert len(plan["nn"]) == 4                            # 2 one-hop + 2 two-hop routes
    assert plan["picks"]["size_widest"] == ("lo", "ha", "nd")
    assert plan["picks"]["quality"] == ("lo", "ha", "nd")
    assert plan["picks"]["size_sum"] == ("lo", "ha", "nd")    # not the longest route (review C1)
    assert plan["top"][0] == ("lo", "ha", "nd")
    # validation never touches a list word of lo (on lo -> x) or nd (on x -> nd), and edges
    # between two intermediates exclude nothing (review C2)
    m = run.edge_measure(ctx, "lo", "ha", "lo", "nd")
    assert m["quality"] > 0.9 and m["n_val"] > 0
    lo_rows, nd_rows = ctx.list_rows["lo"], ctx.list_rows["nd"]
    assert m["excluded"] == ["a"] and m["val_min_row"] not in lo_rows
    assert run.edge_measure(ctx, "ha", "nd", "lo", "nd")["excluded"] == ["b"]
    assert run.edge_measure(ctx, "ha", "hb", "lo", "nd")["excluded"] == []
    _, val = run.validation_split(ctx.edge("lo", "ha"), lo_rows, set(), 0.5, 500)
    assert not set(val["a"]) & lo_rows and nd_rows
    run.write_results(ctx, plans)
    summ = pd.read_csv(here / "results" / "summary.csv").set_index("chooser")
    choosers = {"size_widest", "size_sum", "quality", "rsim", "uriel", "hub", "oracle"}
    assert choosers <= set(summ.index)
    assert summ.at["oracle", "regret"] == 0 and (summ["regret"] >= 0).all()
    assert summ.at["quality", "mrr"] > 0.9 and summ.at["hub", "hit_rate"] == 1.0
    pooled = pd.read_csv(here / "results" / "pooled.csv")
    assert set(pooled["src"]) == {"lo", "nd"}              # both directions scored
    n = sum(1 for _ in run.Ctx(limit=None).cache)
    run.run_all(run.Ctx(limit=None))                       # resumes: nothing new
    assert sum(1 for _ in run.Ctx(limit=None).cache) == n


def test_size_sum_cost_is_non_negative():
    # review C1: -log(size) is negative, so summing it rewards extra hops
    c = run.size_cost(np.array([[0, 100, 10], [100, 0, 1000], [10, 1000, 0]], dtype=float),
                      biggest=1000)
    off = c[~np.eye(3, dtype=bool)]
    assert (off >= 0).all() and c[1, 2] == 0 and np.isclose(c[0, 2], np.log(100))


def test_route_map_fits_each_step_once_per_pair_and_fold(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    ctx = run.Ctx(limit=None)
    calls = []
    real = run.s1.edge_map
    monkeypatch.setattr(run.s1, "edge_map", lambda *a, **k: calls.append(1) or real(*a, **k))
    for r in [("lo", "ha", "nd"), ("lo", "ha", "hb", "nd"), ("lo", "hb", "ha", "nd")]:
        run.route_map(ctx, r, "lo", "nd", 0)
    # distinct steps: lo-ha, ha-nd, ha-hb, hb-nd, lo-hb, hb-ha
    assert len(calls) == 6
    run.route_map(ctx, ("lo", "ha", "nd"), "lo", "nd", 1)     # another fold refits s/t steps only
    assert len(calls) == 8
