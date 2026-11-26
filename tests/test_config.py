from pathlib import Path

from clwe_chains.config import load_config, node_languages


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_local_overrides_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LG_DATA_ROOT", raising=False)
    _write(tmp_path / "default.yaml", "data_root: /a\neval:\n  csls_k: 10\n  topk: [1, 5]\n")
    _write(tmp_path / "local.yaml", "data_root: /b\neval:\n  csls_k: 5\n")
    cfg = load_config(config_dir=tmp_path)
    assert cfg.data_root == Path("/b")
    assert cfg["eval"] == {"csls_k": 5, "topk": [1, 5]}  # nested keys merge, not replace


def test_env_overrides_data_root(tmp_path, monkeypatch):
    _write(tmp_path / "default.yaml", "data_root: /a\n")
    monkeypatch.setenv("LG_DATA_ROOT", str(tmp_path / "data"))
    cfg = load_config(config_dir=tmp_path)
    assert cfg.path("list") == tmp_path / "data" / "v1" / "list"


def test_path_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("LG_DATA_ROOT", raising=False)
    _write(tmp_path / "default.yaml", "data_root: /a\n")
    cfg = load_config(config_dir=tmp_path)
    assert cfg.path("v0_panlex") == Path("/a/processed/panlex_invocab")
    assert cfg.path("results") == Path("/a/v1/results")


def test_phase6_path(tmp_path, monkeypatch):
    _write(tmp_path / "default.yaml", "data_root: /a\n")
    monkeypatch.setenv("LG_DATA_ROOT", str(tmp_path / "data"))
    cfg = load_config(config_dir=tmp_path)
    assert cfg.path("phase6") == tmp_path / "data" / "v1" / "phase6"
    assert cfg.path("v0_coverage_pairs") == (tmp_path / "data" / "processed"
                                             / "coverage_pairs.parquet")


def test_node_languages():
    langs = node_languages(load_config())
    assert len(langs) == 32
    assert "en" in langs and "mt" in langs
    assert langs == sorted(langs)


def test_repo_default_config():
    cfg = load_config()
    assert cfg["list"]["n_candidates"] == 20000
    assert cfg["list"]["coverage_threshold"] == 0.69  # provisional, user 2026-11-15
    assert cfg["list"]["round_trip_top_k"] == 5
    assert cfg["eval"]["block_rows"] == 1024
    assert cfg["eval"]["half"] is True
