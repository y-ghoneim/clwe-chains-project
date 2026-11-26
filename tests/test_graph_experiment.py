import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

RUN = Path(__file__).resolve().parents[1] / "experiments" / "graph-all-pairs" / "run.py"
_spec = importlib.util.spec_from_file_location("graph_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_cache_roundtrip_and_skip(tmp_path):
    path = tmp_path / "evals.jsonl"
    k1 = run.cache_key("csls", ("en", "ar"), ("en", "de", "ar"), 0, "full")
    k2 = run.cache_key("csls", ("en", "ar"), ("en", "ar"), 1, "robust")
    assert k1 != k2
    cache = run.Cache(path)
    cache.put(k1, {"ranks": [1, 3]})
    cache.put(k2, {"ranks": [2]})
    again = run.Cache(path)
    assert again.get(k1) == {"ranks": [1, 3]} and again.get(k2) == {"ranks": [2]}
    assert again.get("missing") is None

    calls = []

    def fake_stage(c: "run.Cache") -> None:
        for split in range(3):
            key = run.cache_key("csls", ("en", "ar"), ("en", "ar"), split, "full")
            c.get_or_compute(key, lambda s=split: calls.append(s) or {"ranks": [s + 1]})

    fake_stage(run.Cache(path))       # split 1 "full" isn't cached (k2 is "robust"): 3 calls
    assert calls == [0, 1, 2]
    fake_stage(run.Cache(path))       # a rerun computes nothing
    assert calls == [0, 1, 2]


def test_cache_skips_a_truncated_last_line(tmp_path):
    path = tmp_path / "evals.jsonl"
    cache = run.Cache(path)
    cache.put("a", {"ranks": [1]})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"key": "b", "value": {"ran')   # an interrupted write
    again = run.Cache(path)
    assert again.get("a") == {"ranks": [1]} and again.get("b") is None
    again.put("b", {"ranks": [2]})                # appends on a new line
    assert run.Cache(path).get("b") == {"ranks": [2]}


def test_aus_rule_skips_missing_languages():
    langs = ["a", "b", "c", "d"]
    codes = {lang: lang + "x" for lang in langs}
    # c sits between a and b (the cheapest intermediate) but lang2vec has no values for it
    feats = {"ax": [1, 0, 0], "bx": [0, 1, 0], "cx": ["--", "--", "--"], "dx": [1, 1, 1]}
    vecs, missing = run.lang2vec_vectors(feats, codes, langs)
    assert missing == ["c"] and set(vecs) == {"a", "b", "d"}
    assert run.aus_rule_routes(vecs, langs, "a", "b") == ("a", "d", "b")
    # with c present it would be chosen: the rule really is avoiding it
    full = {**vecs, "c": np.array([1.0, 1.0, 0.0])}
    assert run.aus_rule_routes(full, langs, "a", "b") == ("a", "c", "b")
    # an endpoint without values: no route, no exception
    assert run.aus_rule_routes(vecs, langs, "a", "c") is None


def test_result_affecting_includes_graph_experiment():
    status = [" M CLAUDE.md", "?? docs/MLR570_Project_Report.pdf",
              "?? experiments/graph-all-pairs/results/metrics.csv",
              " M experiments/graph-all-pairs/run.py", " M src/clwe_chains/graph.py",
              " M experiments/aus-replication/run.py"]
    assert run.result_affecting(status) == [" M experiments/graph-all-pairs/run.py",
                                            " M src/clwe_chains/graph.py",
                                            " M experiments/aus-replication/run.py"]


def test_hub_route():
    assert run.hub_route("tr", "ru", ["en", "ru"]) == ("tr", "en", "ru")
    assert run.hub_route("en", "ar", ["en", "ru"]) == ("en", "ru", "ar")


def test_threshold_cv_uses_only_training_pairs():
    # 4 pairs in 2 folds, 1 split. Chaining helps only pair 3 (fold 1), whose direct cost (0.9)
    # is the largest. Fold 0's τ comes from fold 1's costs only, and vice versa.
    folds = np.array([0, 0, 1, 1])
    cost = np.array([[0.2], [0.5], [0.4], [0.9]])
    chain = np.array([[0.1], [0.1], [0.1], [0.8]])
    direct = np.array([[0.5], [0.5], [0.5], [0.5]])
    out = run.threshold_cv(cost, chain, direct, folds)
    # fold 0 (trained on pairs 2, 3; τ ∈ {0.4, 0.9}): τ = 0.4 chains pair 3 only → mean 0.65
    # beats τ = 0.9 (nothing chained, 0.5). Applied to pairs 0, 1: only pair 1 (0.5) chains.
    assert list(out["tau"][:2]) == [0.4, 0.4] and list(out["mrr"][:2]) == [0.5, 0.1]
    # fold 1 (trained on pairs 0, 1; τ ∈ {0.2, 0.5}): τ = 0.5 chains nothing (0.5 > 0.3 at 0.2)
    assert list(out["tau"][2:]) == [0.5, 0.5]
    assert list(out["mrr"][2:]) == [0.5, 0.8]
    # changing fold 1's own results doesn't change fold 1's τ
    chain2 = chain.copy()
    chain2[2:] = 0.99
    assert list(run.threshold_cv(cost, chain2, direct, folds)["tau"][2:]) == [0.5, 0.5]


# --- Characterization tests of the results tables (phase 4 code review, 2026-11-16) -----------
# They pin down what write_results, add_oracle_columns, summary_measures and cv_table compute,
# on a tiny synthetic cache; the phase 4 code itself is frozen with its results.

def _put(cache, s, t, route, split, ranks, subset="full"):
    cache.put(run.cache_key("csls", (s, t), tuple(route.split("-")), split, subset),
              {"ranks": ranks})


def _routes_csv(path, rows):
    pd.DataFrame(rows, columns=["s", "t", "measure", "split", "route", "hops", "cost"]).to_csv(
        path / "routes.csv", index=False)


def test_write_results_metrics_bootstrap_and_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    cache = run.Cache(tmp_path / "evals.jsonl")
    _put(cache, "en", "ar", "en-ar", 0, [1, 2, 4, 1])       # MRR 0.6875
    _put(cache, "en", "ar", "en-de-ar", 0, [2, 2, 4, 1])    # 0.5625
    _put(cache, "en", "ar", "en-ar", 1, [1, 1, 2, 2])       # 0.75
    _put(cache, "en", "ar", "en-de-ar", 1, [1, 1, 1, 1])    # 1.0
    _routes_csv(tmp_path, [("en", "ar", "direct", 0, "en-ar", 0, 0.0),
                           ("en", "ar", "direct", 1, "en-ar", 0, 0.0),
                           ("en", "ar", "quality", 0, "en-de-ar", 1, 1.0),
                           ("en", "ar", "quality", 1, "en-de-ar", 1, 1.0)])
    ctx = SimpleNamespace(xcfg={"bootstrap": {"n": 200, "seed": 0}}, cache=cache,
                          splits={0: None, 1: None},
                          all_pairs=pd.DataFrame({"s": ["en"], "t": ["ar"], "stratum": ["a"]}))
    run.write_results(ctx)

    met = pd.read_csv(tmp_path / "metrics.csv").set_index(["route", "split"])
    assert len(met) == 4 and abs(met.loc[("en-de-ar", 0), "mrr"] - 0.5625) < 1e-6
    assert met.loc[("en-ar", 1), "p1"] == 0.5 and met.loc[("en-ar", 1), "p5"] == 1.0
    boot = pd.read_csv(tmp_path / "bootstrap.csv").set_index("split")
    assert len(boot) == 2   # direct isn't compared with itself
    assert abs(boot.loc[0, "d_mrr"] + 0.125) < 1e-6 and abs(boot.loc[1, "d_mrr"] - 0.25) < 1e-6
    assert (boot["lo"] <= boot["d_mrr"]).all() and (boot["d_mrr"] <= boot["hi"]).all()
    pairs = pd.read_csv(tmp_path / "summary_pairs.csv").set_index("measure")
    q = pairs.loc["quality"]
    assert q["n_splits"] == 2 and abs(q["mrr"] - 0.78125) < 1e-6
    assert abs(q["d_mrr"] - 0.0625) < 1e-6 and q["stratum"] == "a"
    assert pairs.loc["direct", "d_mrr"] == 0.0
    sm = pd.read_csv(tmp_path / "summary_measures.csv")
    qa = sm[(sm["measure"] == "quality") & (sm["group"] == "all")].iloc[0]
    assert qa["n_pairs"] == 1 and qa["pairs_won"] == 1 and abs(qa["d_mrr_mean"] - 0.0625) < 1e-6
    assert set(sm[sm["measure"] == "quality"]["group"]) == {"all", "stratum_a", "en_source"}


def test_add_oracle_columns_regret_top_pick_and_tau(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    monkeypatch.setattr(run, "MEASURES", ("resource",))
    langs = ["ar", "de", "en", "fr"]
    D = np.full((4, 4), 1.0)
    D[2, 1], D[1, 0] = 0.1, 0.1     # en-de-ar costs 0.2
    D[2, 3], D[3, 0] = 0.2, 0.3     # en-fr-ar 0.5
    D[1, 3] = 0.5                   # en-de-fr-ar 0.9
    monkeypatch.setattr(run, "matrices", lambda ctx: {"resource": D})
    pd.DataFrame({"s": "en", "t": "ar", "route": ["en-de-ar", "en-fr-ar", "en-de-fr-ar"],
                  "hops": [1, 1, 2], "mrr_mean": [0.6, 0.5, 0.4],
                  "oracle": [True, False, False]}).to_csv(tmp_path / "oracle.csv", index=False)
    routes = pd.DataFrame({"s": "en", "t": "ar", "measure": "resource", "split": [0, 1],
                           "route": ["en-de-ar", "en-fr-ar"]})
    pair_sum = pd.DataFrame({"measure": "resource", "subset": ["full", "robust"], "s": "en",
                             "t": "ar", "mrr": [0.55, 0.5]})
    ctx = SimpleNamespace(langs=langs, splits={0: None, 1: None},
                          xcfg={"oracle": {"max_mid": 2}})
    out = run.add_oracle_columns(ctx, pair_sum, routes).set_index("subset")
    assert abs(out.loc["full", "regret"] - 0.05) < 1e-9
    assert out.loc["full", "top_pick"] == 0.5
    expect = kendalltau([-0.2, -0.5, -0.9], [0.6, 0.5, 0.4]).statistic
    assert expect == 1.0 and abs(out.loc["full", "kendall_tau"] - expect) < 1e-9
    assert out.loc["robust", ["regret", "top_pick", "kendall_tau"]].isna().all()  # full list only


def test_summary_measures_groups():
    ps = pd.DataFrame({"measure": "m", "subset": "full", "s": ["en", "fr", "ka"],
                       "t": ["ar", "fi", "et"], "stratum": ["a", "a", "b"],
                       "mrr": [0.5, 0.4, 0.2], "d_mrr": [-0.02, 0.01, -0.03],
                       "splits_above_0": [0, 3, 0], "splits_below_0": [4, 0, 3],
                       "regret": [0.01, 0.0, 0.02], "top_pick": [0.2, 1.0, 0.0],
                       "kendall_tau": [0.5, 0.7, 0.1]})
    sm = run.summary_measures(ps).set_index("group")
    assert sm.loc["all", "n_pairs"] == 3 and sm.loc["all", "pairs_won"] == 1
    assert sm.loc["all", "pairs_above_3of5"] == 1 and sm.loc["all", "pairs_below_3of5"] == 2
    assert sm.loc["stratum_a", "n_pairs"] == 2 and sm.loc["stratum_b", "n_pairs"] == 1
    assert sm.loc["en_source", "n_pairs"] == 1 and sm.loc["non_en_source", "n_pairs"] == 2
    assert abs(sm.loc["all", "d_mrr_mean"] + 0.04 / 3) < 1e-12
    assert abs(sm.loc["all", "kendall_tau_mean"] - 1.3 / 3) < 1e-12


def test_cv_table_power_and_threshold_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    monkeypatch.setattr(run, "MEASURES", ("resource",))
    langs = ["a", "b", "c", "d", "x", "y"]
    pairs = [("a", "b"), ("c", "d"), ("a", "c"), ("b", "d")]
    # per pair: direct, p = 1 chain (via x), p = 2 chain (via y) ranks; one query, MRR = 1 / rank
    ranks = {("a", "b"): (1, 2, 1), ("c", "d"): (1, 4, 1), ("a", "c"): (2, 1, 4),
             ("b", "d"): (2, 1, 2)}
    cache = run.Cache(tmp_path / "evals.jsonl")
    rows = []
    for (s, t), (rd, r1, r2) in ranks.items():
        for measure, route, r in (("direct", f"{s}-{t}", rd), ("resource", f"{s}-x-{t}", r1),
                                  ("resource^2", f"{s}-y-{t}", r2)):
            rows.append((s, t, measure, 0, route, 1, 0.0))
            _put(cache, s, t, route, 0, [r])
    _routes_csv(tmp_path, rows)
    D = np.zeros((6, 6))
    for i, (s, t) in enumerate(pairs):
        D[langs.index(s), langs.index(t)] = 0.1 * (i + 1)
    folds = np.array([0, 0, 1, 1])
    ctx = SimpleNamespace(cache=cache, splits={0: None}, langs=langs, xcfg={"alt_p": [2]},
                          all_pairs=pd.DataFrame({"s": [p[0] for p in pairs],
                                                  "t": [p[1] for p in pairs]}))
    run.cv_table(ctx, {"resource": D}, pd.read_csv(tmp_path / "routes.csv"), folds)
    cv = pd.read_csv(tmp_path / "cv.csv")
    power = cv[cv["kind"] == "power"].set_index(["s", "t"])
    # fold 0 is chosen on fold 1's pairs (p = 1 mean 1.0, p = 2 mean 0.375): p = 1;
    # fold 1 on fold 0's pairs (p = 1 mean 0.375, p = 2 mean 1.0): p = 2
    assert power.loc[("a", "b"), "choice"] == 1 and power.loc[("a", "c"), "choice"] == 2
    assert power.loc[("a", "b"), "mrr"] == 0.5 and power.loc[("a", "c"), "mrr"] == 0.25
    assert power.loc[("a", "b"), "mrr_direct"] == 1.0
    th = cv[cv["kind"] == "threshold"].reset_index(drop=True)
    cost = np.array([[0.1], [0.2], [0.3], [0.4]])
    chain = np.array([[0.5], [0.25], [1.0], [1.0]])
    direct = np.array([[1.0], [1.0], [0.5], [0.5]])
    expect = run.threshold_cv(cost, chain, direct, folds)
    assert np.allclose(th["mrr"], expect["mrr"]) and np.allclose(th["choice"], expect["tau"])
