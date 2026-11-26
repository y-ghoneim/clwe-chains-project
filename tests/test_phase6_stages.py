"""Tests of the later phase 6 stages' runner (experiments/scarce-moderate-a/run.py, identical in
experiments/natural-highres-d/), end to end on a tiny CPU world in the stage's data subfolder."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"stage_{name}", EXP / name / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(mod, tmp_path, monkeypatch, joined: bool, en_chain: str = "[]"):
    import torch

    from clwe_chains.config import load_config
    from clwe_chains.dictionaries import join_counts
    root, cdir = tmp_path / "data", tmp_path / "cfg"
    cdir.mkdir()
    (cdir / "default.yaml").write_text(
        f"data_root: {root.as_posix()}\ndevice: cpu\n"
        "embeddings: {dim: 8, max_vocab: 40, normalize: [unit, center, unit]}\n"
        "eval: {csls_k: 3, block_rows: 16, half: false}\n"
        f"phase6: {{en_chain: {en_chain}, low: [lo]}}\n",
        encoding="utf-8")
    cfg = load_config(config_dir=cdir)
    g = torch.Generator().manual_seed(0)
    base = torch.randn(40, 8, generator=g)
    emb = cfg.path("v0_embeddings")
    emb.mkdir(parents=True)
    for lang, noise in (("lo", 0.05), ("nd", 0.0), ("hb", 0.0)):
        q, _ = torch.linalg.qr(torch.randn(8, 8, generator=g))
        np.save(emb / f"{lang}.npy", (base @ q + noise * torch.randn(40, 8, generator=g)).numpy())
    p6 = cfg.path("phase6") / mod.XCFG["data"]
    (p6 / "dicts").mkdir(parents=True)
    (p6 / "joined").mkdir()
    eye = lambda n: pd.DataFrame({"a": np.arange(n), "b": np.arange(n)})
    eye(10).to_parquet(p6 / "dicts" / "lo__nd.parquet")
    eye(40).to_parquet(p6 / "dicts" / "lo__hb.parquet")
    eye(40).to_parquet(p6 / "dicts" / "hb__nd.parquet")
    if joined:
        join_counts(eye(40), eye(40)).to_parquet(p6 / "joined" / "lo__hb__nd.parquet")
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
    pd.DataFrame({"s": ["lo"], "t": ["nd"], "direct": [10], "hub": ["hb"], "route": [40]}).to_csv(
        here / "pairs.csv", index=False)
    mod.configure()
    monkeypatch.setattr(mod.s1, "CFG", cfg)
    monkeypatch.setattr(mod.s1, "HERE", here)
    monkeypatch.setattr(mod.s1, "RESULTS", here / "results")
    monkeypatch.setattr(mod.s1, "EVAL", {"csls_k": 3, "half": False, "retrieval": "csls"})
    return here, cfg


@pytest.mark.parametrize("name, joined, n_methods", [("scarce-moderate-a", True, 4),
                                                     ("natural-highres-d", False, 3)])
def test_stage_runs_its_methods_on_its_own_data(tmp_path, monkeypatch, name, joined, n_methods):
    mod = _load(name)
    here, cfg = _world(mod, tmp_path, monkeypatch, joined)
    ctx = mod.s1.Ctx(limit=None)
    assert type(ctx).__name__ == "Ctx" and ctx.methods("lo", "hb")[0] == ("thin",)
    mod.s1.run_all(ctx)
    assert sum(1 for _ in ctx.cache) == 2 * 5 * n_methods
    assert ctx.cache.path == cfg.path("results") / name / "evals.jsonl"     # its own cache
    mod.s1.write_results(ctx)
    met = pd.read_csv(here / "results" / "metrics.csv")
    want = {"thin", "chain", "combo"} | ({"joined_mb"} if joined else set())
    assert set(met["method"]) == want
    assert met.groupby("method")["mrr"].mean()["chain"] > 0.9
    assert (here / "results" / "run_info.json").exists()


def test_stage_configs_differ_only_in_data_and_methods():
    a, d = _load("scarce-moderate-a"), _load("natural-highres-d")
    assert (a.NAME, a.XCFG["data"]) == ("scarce-moderate-a", "a")
    assert (d.NAME, d.XCFG["data"]) == ("natural-highres-d", "d")
    assert "joined_mb" in a.XCFG["methods"] and "joined_mb" not in d.XCFG["methods"]
    for other, data in (("natural-highres-d", "d"), ("scarce-moderate-a2", "a2"),
                        ("close-pairs-b", "b")):
        assert (EXP / "scarce-moderate-a" / "run.py").read_bytes() == \
            (EXP / other / "run.py").read_bytes()
        assert _load(other).XCFG["data"] == data


def test_stage_skips_the_en_chain_dictionaries_it_does_not_score(tmp_path, monkeypatch):
    # lo is an en-chain language (as mg is), but stage A doesn't score the en chain, so its
    # lo__en / en__nd dictionaries (absent here) must not be read (review of stages A/D, item 3)
    mod = _load("scarce-moderate-a")
    _world(mod, tmp_path, monkeypatch, joined=True, en_chain="[lo]")
    ctx = mod.s1.Ctx(limit=None)
    d, _ = mod.s1.fold_dicts(ctx, "lo", "nd", "hb", 0)
    assert "sen" not in d and "ent" not in d and "joined_mb" in d


def test_run_info_records_the_stage_config(tmp_path, monkeypatch):
    import json
    mod = _load("natural-highres-d")
    here, _ = _world(mod, tmp_path, monkeypatch, joined=False)
    ctx = mod.s1.Ctx(limit=None)
    mod.s1.run_all(ctx)
    mod.s1.write_results(ctx)
    info = json.loads((here / "results" / "run_info.json").read_text(encoding="utf-8"))
    assert info["config"]["stage"] == mod.XCFG and info["config"]["bootstrap"]["n"] == 1000
