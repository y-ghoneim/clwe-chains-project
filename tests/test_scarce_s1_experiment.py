"""Tests of experiments/scarce-natural-s1/run.py's helpers (phase 6 stage 1)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from clwe_chains.runs import Cache

RUN = Path(__file__).resolve().parents[1] / "experiments" / "scarce-natural-s1" / "run.py"
_spec = importlib.util.spec_from_file_location("scarce_s1_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_alpha_cv_ties_go_to_smaller_alpha():
    score = pd.DataFrame({0.3: [0.5, 0.5, 0.9], 0.1: [0.5, 0.5, 0.4]}, index=["a", "b", "c"])
    lang = pd.Series(["yo", "yo", "mg"], index=score.index)
    per, chosen = run.alpha_cv(score, lang)
    assert chosen["mg"] == 0.1          # yo's pairs tie: the smaller alpha, whatever the order
    assert chosen["yo"] == 0.3 and per["a"] == 0.5 and per["c"] == 0.4


def test_pooled_concatenates_folds(tmp_path):
    cache = Cache(tmp_path / "e.jsonl")
    for f, (cs, rk) in enumerate([([1, 2], [1, 4]), ([3], [2])]):
        cache.put(run.key("yo", "zh", ("thin",), f), {"concepts": cs, "rank": rk,
                                                       "rank_list": rk, "n_train": {},
                                                       "seconds": 0})
    p = run.pooled(cache, "yo", "zh", ("thin",), folds=[0, 1])
    assert p["concept"].tolist() == [1, 2, 3] and p["rr"].tolist() == [1.0, 0.25, 0.5]
    assert run.pooled(cache, "yo", "zh", ("chain", "en"), folds=[0, 1]) is None


def test_language_table_is_descriptive():
    # per language: mean over pairs (each pair weighted equally), and how many pair-directions
    # have their paired bootstrap interval above / below 0; no pass/fail (user, 2026-11-16)
    diffs = {"yo": [np.full((2, 10), 0.1), np.full((2, 5), 0.3)], "mg": [np.zeros((2, 4))]}
    pooled = pd.DataFrame({"s": ["yo"] * 4 + ["mg"] * 2, "lo": [.01, .02, -.1, .05, -.1, -.2],
                           "hi": [.2, .3, .1, .2, .1, -.01]})
    t = run.language_table(diffs, pooled).set_index("lang")
    assert np.isclose(t.at["yo", "mean"], 0.2) and t.at["yo", "n_pairs"] == 2
    assert t.at["yo", "n_dirs"] == 4 and t.at["yo", "n_above"] == 3 and t.at["yo", "n_below"] == 0
    assert t.at["mg", "mean"] == 0 and t.at["mg", "n_below"] == 1
    assert list(t.columns) == ["mean", "n_pairs", "n_dirs", "n_above", "n_below"]


def test_method_labels_round_trip():
    for m in [("thin",), ("chain", "fr"), ("chain", "en"), ("joined", "mb"), ("joined", "all"),
              ("teacher", "0.3"), ("teacherd", "10"), ("combo", "fr")]:
        assert run.method_of(run.label(m, hub="fr"), hub="fr") == m
    assert run.label(("chain", "fr"), hub="fr") == "chain"
    assert run.label(("chain", "en"), hub="fr") == "chain_en"
    assert run.label(("teacher", "0.3"), hub="fr") == "teacher_0.3"


def _synthetic(tmp_path, monkeypatch):
    """A tiny world: lo, nd and hub hb share one geometry up to rotations (lo with noise); lo-nd
    has 10 dictionary pairs, the hub edges 40. Everything the runner reads, on the CPU."""
    import torch

    from clwe_chains.config import load_config
    from clwe_chains.dictionaries import join_counts

    root, cdir = tmp_path / "data", tmp_path / "cfg"
    cdir.mkdir()
    (cdir / "default.yaml").write_text(
        f"data_root: {root.as_posix()}\ndevice: cpu\n"
        "embeddings: {dim: 8, max_vocab: 40, normalize: [unit, center, unit]}\n"
        "eval: {csls_k: 3, block_rows: 16, half: false}\nphase6: {en_chain: []}\n",
        encoding="utf-8")
    cfg = load_config(config_dir=cdir)
    g = torch.Generator().manual_seed(0)
    base = torch.randn(40, 8, generator=g)
    emb = cfg.path("v0_embeddings")
    emb.mkdir(parents=True)
    for lang, noise in (("lo", 0.05), ("nd", 0.0), ("hb", 0.0)):
        q, _ = torch.linalg.qr(torch.randn(8, 8, generator=g))
        np.save(emb / f"{lang}.npy", (base @ q + noise * torch.randn(40, 8, generator=g)).numpy())
    p6 = cfg.path("phase6")
    (p6 / "dicts").mkdir(parents=True)
    (p6 / "joined").mkdir()
    eye = lambda n: pd.DataFrame({"a": np.arange(n), "b": np.arange(n)})
    eye(10).to_parquet(p6 / "dicts" / "lo__nd.parquet")
    eye(40).to_parquet(p6 / "dicts" / "lo__hb.parquet")
    eye(40).to_parquet(p6 / "dicts" / "hb__nd.parquet")
    join_counts(eye(40), eye(40)).to_parquet(p6 / "joined" / "lo__hb__nd.parquet")
    concepts = list(range(5, 25))       # concepts 5-9 are in the direct dictionary too
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
    pd.DataFrame({"s": ["lo"], "t": ["nd"], "direct": [10], "hub": ["hb"], "route": [40]}).to_csv(
        here / "pairs.csv", index=False)
    monkeypatch.setattr(run, "CFG", cfg)
    monkeypatch.setattr(run, "HERE", here)
    monkeypatch.setattr(run, "RESULTS", here / "results")
    monkeypatch.setattr(run, "EVAL", {"csls_k": 3, "half": False, "retrieval": "csls"})
    return here


def test_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    here = _synthetic(tmp_path, monkeypatch)
    ctx = run.Ctx(limit=None)
    run.run_all(ctx)
    n_methods = 15                      # thin, chain, combo, 2 joined, 5 + 5 teacher alphas
    assert sum(1 for _ in ctx.cache) == 2 * 5 * n_methods
    # every test word is gone from every edge it touches, in every fold
    d, sizes = run.fold_dicts(ctx, "lo", "nd", "hb", 0)
    test = set(ctx.folds["lo", "nd"][0])
    assert not (set(d["direct"]["a"]) & test) and not (set(d["sx"]["a"]) & test)
    assert not (set(d["xt"]["b"]) & test) and not (set(d["joined_all"]["b"]) & test)
    assert sizes["direct"]["pairs"] == 10 - len(test & set(range(10)))
    run.write_results(ctx)
    res = here / "results"
    met = pd.read_csv(res / "metrics.csv")
    assert len(met) == 2 * 5 * n_methods and set(met["src"]) == {"lo", "nd"}
    by = met.groupby("method")["mrr"].mean()
    # the hub edges see every non-test word, so the chain recovers the shared geometry
    assert by["chain"] > 0.9 and by["joined_all"] > 0.9 and by["teacher_10"] > 0.9
    langs = pd.read_csv(res / "languages.csv")
    assert set(langs["lang"]) == {"lo"} and "teacher_cv" not in set(langs["method"])
    assert (res / "run_info.json").exists() and not (res / "success.json").exists()
    assert sum(1 for _ in run.Ctx(limit=None).cache) == 150        # resumes from the cache


def test_teacher_label_maps_the_cv_choice_back_to_the_config_value():
    # alpha_cv on pd.DataFrame(score).T returns floats; the stored labels follow the YAML values
    score = pd.DataFrame({"a": {0.1: 0.1, 1: 0.9}, "b": {0.1: 0.1, 1: 0.8}}).T
    _, chosen = run.alpha_cv(score, pd.Series({"a": "yo", "b": "mg"}))
    assert run.teacher_label(chosen["yo"]) == "teacher_1"
    assert run.teacher_label(np.float64(0.1)) == "teacher_0.1"
    assert run.teacher_label(10.0) == "teacher_10"


def test_eval_commits_are_read_from_the_run_log():
    log = ("start commit 830a312aaaa\n23:00:06 pair 1/2 yo-ar via en: 90 evaluations\n"
           "start commit 830a312aaaa\nstart commit 1234567bbbb\n")
    assert run.eval_commits(log) == ["830a312aaaa", "1234567bbbb"]
    assert run.eval_commits("23:00:06 pair 1/2 yo-ar\n") == []


def test_check_clean_refuses_result_affecting_changes():
    import pytest
    run.check_clean([" M docs/remote.md", "?? experiments/scarce-natural-s1/results/x.csv"])
    with pytest.raises(SystemExit):
        run.check_clean([" M src/clwe_chains/stats.py"])


def test_teacher_lambda_families():
    assert run.teacher_lambda("teacher", 0.3, n=600, d=300) == 0.3 * 600
    assert run.teacher_lambda("teacherd", 0.3, n=600, d=300) == 0.3 * 600 / 300
    assert run.teacher_label(1.0, "teacherd") == "teacherd_1"


def test_dimension_scaled_teacher_spans_thin_to_chain_at_real_size():
    # final review I2: with lambda = alpha * n every alpha is ~ the chain at d = 300; dividing by
    # d puts the grid between thin direct (small alpha) and the chain (large alpha)
    import torch

    from clwe_chains.align import fit, fit_teacher
    g = torch.Generator().manual_seed(0)
    d, n = 300, 250
    x = torch.nn.functional.normalize(torch.randn(n, d, generator=g, dtype=torch.float64), dim=1)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    y = torch.nn.functional.normalize(x @ q + 0.5 * torch.randn(n, d, generator=g,
                                                                dtype=torch.float64), dim=1)
    wc, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    thin = fit(x, y, "procrustes")

    def rel(fam, a):     # 0 = the chain, 1 = thin direct
        w = fit_teacher(x, y, wc, run.teacher_lambda(fam, a, n=n, d=d))
        return float((w - wc).norm() / (thin - wc).norm())
    assert rel("teacher", 0.1) < 0.05                          # the old grid: already the chain
    assert rel("teacherd", 0.1) > 0.5 > rel("teacherd", 10)    # the new grid spans both ends
