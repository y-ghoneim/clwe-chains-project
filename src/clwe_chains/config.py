"""Config loading: configs/default.yaml, overridden by configs/local.yaml, then env vars."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

# Subdirectories of data_root. v0 inputs are pinned and read-only (brief section 7); everything
# v1 writes goes under v1/.
DATA_LAYOUT = {
    "v0_embeddings": "processed/embeddings",
    "v0_panlex": "processed/panlex_invocab",
    "v0_languages": "processed/languages_all.csv",
    "v0_distances": "processed/distances/measures.parquet",
    "v0_coverage_pairs": "processed/coverage_pairs.parquet",
    "raw_wiktionary": "raw/wiktionary",
    "nltk_data": "raw/nltk_data",
    "list": "v1/list",
    "results": "v1/results",
    "phase6": "v1/phase6",     # phase 6's own data (brief 5.10 isolation): never the main list
    "raw_downstream": "raw/downstream",
    "downstream": "v1/downstream",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def data_root(self) -> Path:
        return Path(self.raw["data_root"])

    def path(self, name: str) -> Path:
        """Absolute path of a named data location, e.g. cfg.path("list")."""
        return self.data_root / DATA_LAYOUT[name]

    def make_dirs(self) -> None:
        """Create the v1 output directories (never the read-only v0 inputs)."""
        for name in ("list", "results", "phase6"):
            self.path(name).mkdir(parents=True, exist_ok=True)


def load_config(*extra: str | Path, config_dir: Path = CONFIG_DIR) -> Config:
    """Load default.yaml, then local.yaml if present, then any extra YAML files in order.

    LG_DATA_ROOT, if set, overrides data_root last.
    """
    files = [config_dir / "default.yaml", config_dir / "local.yaml", *map(Path, extra)]
    merged: dict[str, Any] = {}
    for f in files:
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                merged = _deep_merge(merged, yaml.safe_load(fh) or {})
    if env_root := os.environ.get("LG_DATA_ROOT"):
        merged["data_root"] = env_root
    return Config(merged)


def node_languages(cfg: Config) -> list[str]:
    """The phase 1 *candidate* languages (configs/languages.yaml, key `nodes`), sorted: the 32
    the list build starts from, am and vi included. Not the graph's nodes: those are the built
    list's languages (shared_list.list_languages; 30 since 2026-11-15). The name and the unused
    `cfg` are kept for the prep step and the analyses that call it."""
    with open(CONFIG_DIR / "languages.yaml", encoding="utf-8") as fh:
        return sorted(yaml.safe_load(fh)["nodes"])
