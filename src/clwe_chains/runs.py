"""Helpers for experiment runners, from phase 5 on (decisions.md 2026-11-16).

Phase 4's runner (experiments/graph-all-pairs/run.py) keeps its own copies, frozen with its
results. These versions fix what the phase 4 code review found:
- the evaluation cache key carries a hash of the evaluation settings, so a changed CSLS k or
  precision can't silently reuse stale scores;
- the uncommitted-changes check reads both sides of a rename and counts package files;
- run_info can record fingerprints of its input data (the list will exist in two versions);
- the threshold rule can choose "never chain", and ties go to the larger τ;
- test-concept exclusions are counted from the data, not only for evaluations run this time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clwe_chains.cv import threshold_scores

PACKAGE_FILES = ("pyproject.toml", "requirements.lock.txt")


def settings_hash(settings: dict) -> str:
    """Short, key-order-independent hash of evaluation settings, for cache keys."""
    blob = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def cache_key(kind: str, settings: str, pair: tuple[str, str], route: tuple[str, ...],
              split: int, subset: str) -> str:
    """kind (e.g. csls, nn-test, nn-val) | settings hash | s>t | route | split | subset."""
    return f"{kind}|{settings}|{pair[0]}>{pair[1]}|{'-'.join(route)}|{split}|{subset}"


def parse_key(key: str) -> dict:
    kind, settings, pair, route, split, subset = key.split("|")
    s, t = pair.split(">")
    return {"kind": kind, "settings": settings, "s": s, "t": t, "route": route,
            "split": int(split), "subset": subset}


def forget(path: Path, drop) -> int:
    """Rewrite a Cache file without the records whose key satisfies drop(parse_key(key)); returns
    how many were dropped. For reruns after an input changed for part of a run."""
    path = Path(path)
    if not path.exists():
        return 0
    kept, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if drop(parse_key(rec["key"])):
            dropped += 1
        else:
            kept.append(line)
    path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    return dropped


class Cache:
    """Evaluation records in an append-only JSONL file, loaded at start. A line cut short by an
    interruption is ignored (its evaluation is simply redone)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self._newline = False
        if self.path.exists():
            text = self.path.read_text(encoding="utf-8")
            for line in text.splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._data[rec["key"]] = rec["value"]
            self._newline = bool(text) and not text.endswith("\n")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            if self._newline:
                fh.write("\n")
                self._newline = False
            fh.write(json.dumps({"key": key, "value": record}) + "\n")
        self._data[key] = record

    def get_or_compute(self, key: str, fn) -> dict:
        if key not in self._data:
            self.put(key, fn())
        return self._data[key]


def _status_paths(line: str) -> list[str]:
    """The path(s) of a `git status --porcelain` line: both sides of a rename or copy."""
    return [p.strip().strip('"') for p in line[3:].split(" -> ")]


def result_affecting(status_lines: list[str], result_paths: tuple[str, ...],
                     own_results: str, package_files: tuple[str, ...] = PACKAGE_FILES
                     ) -> list[str]:
    """Status lines that could change results: under result_paths, or a package file (package
    versions change results too); the run's own results folder doesn't count."""
    out = []
    for line in status_lines:
        paths = [p for p in _status_paths(line) if not p.startswith(own_results)]
        if any(p.startswith(result_paths) or p in package_files for p in paths):
            out.append(line)
    return out


def file_fingerprints(files: dict[str, Path]) -> dict[str, dict | None]:
    """name -> {path, bytes, sha256} of each input file (None if absent), for run_info.json."""
    out: dict[str, dict | None] = {}
    for name, path in files.items():
        path = Path(path)
        if not path.exists():
            out[name] = None
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda f=fh: f.read(1 << 20), b""):
                h.update(chunk)
        out[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}
    return out


def query_concepts(test_ids, identical: pd.Series) -> tuple[list[int], int]:
    """Test concepts minus identical-string pairs (text.identical_pairs), and how many went.

    Computed for every pair and split up front, so run_info's count doesn't depend on which
    evaluations the cache already held."""
    kept = [c for c in test_ids if not bool(identical.get(c, False))]
    return kept, len(test_ids) - len(kept)


def threshold_cv(direct_cost: np.ndarray, mrr_chain: np.ndarray, mrr_direct: np.ndarray,
                 folds: np.ndarray) -> pd.DataFrame:
    """Threshold rule (ii) cross-validated over pairs. Arrays are pairs x splits.

    Per fold, the candidates are the training pairs' direct costs plus +inf ("never chain"); the
    τ with the best mean MRR over the training pairs is applied to the held-out pairs. Ties go to
    the larger τ, so the rule chains only when the training pairs show a gain. (Phase 4 had no
    +inf candidate and broke ties toward the smaller τ, so held-out pairs costlier than every
    training pair always chained.) Returns per pair: fold, tau, mrr (mean over splits),
    n_chained (splits where it chains).
    """
    n_splits = direct_cost.shape[1]
    out = pd.DataFrame({"fold": folds, "tau": np.nan, "mrr": np.nan, "n_chained": 0})
    for f in sorted(set(folds.tolist())):
        train, held = folds != f, folds == f
        taus = np.append(np.unique(direct_cost[train]), np.inf)
        score = np.mean([threshold_scores(direct_cost[:, j], mrr_chain[:, j], mrr_direct[:, j],
                                          taus).to_numpy() for j in range(n_splits)], axis=0)
        train_score = score[train].mean(axis=0)
        best = int(np.flatnonzero(train_score == train_score.max())[-1])
        out.loc[held, "tau"] = taus[best]
        out.loc[held, "mrr"] = score[held, best]
        out.loc[held, "n_chained"] = (direct_cost[held] > taus[best]).sum(axis=1)
    return out
