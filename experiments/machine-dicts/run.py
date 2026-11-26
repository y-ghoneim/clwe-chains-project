"""RQ3: maps and chains trained on machine-made dictionaries. See README.md (approved first).

Runs stage 1's runner (experiments/scarce-natural-s1/run.py) unchanged in method on one version of
prep/11's data (<phase6>/md/<version>/): same folds, test removal, Procrustes maps, chains and CSLS
evaluation, with the methods in config.yaml. Results go to results/<version>/, the evaluation cache
and log to v1/results/machine-dicts/<version>/.

python run.py --version highres-nllb-a [--limit N] [--results-only] [--data-files]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
NAME = HERE.name
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    f"{NAME}_s1_run", HERE.parent / "scarce-natural-s1" / "run.py")
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)


class Ctx(s1.Ctx):
    def methods(self, s: str, hub: str) -> list[tuple[str, ...]]:
        return [s1.method_of(m, hub) for m in XCFG["methods"]]

    def dictionary(self, name: str) -> pd.DataFrame:
        """dicts/<name>, or chain/<name> for a chain edge that version b keeps only there."""
        own, chain = self.p6 / "dicts" / f"{name}.parquet", self.p6 / "chain" / f"{name}.parquet"
        if name not in self._dicts and not own.exists() and chain.exists():
            self._dicts[name] = pd.read_parquet(chain)
        return super().dictionary(name)


_fold_dicts = s1.fold_dicts


def fold_dicts(ctx, s: str, t: str, x: str, fold: int) -> tuple[dict, dict]:
    """Stage 1's fold dictionaries, with the chain's edges from <version>/chain/ when the version
    has one (version b: its direct edge can share a file name with another pair's chain edge)."""
    d, sizes = _fold_dicts(ctx, s, t, x, fold)
    sx, xt = ctx.p6 / "chain" / f"{s}__{x}.parquet", ctx.p6 / "chain" / f"{x}__{t}.parquet"
    if sx.exists() and xt.exists():                 # only version b has them
        test = ctx.folds[s, t][fold]
        ts, tt = ctx.test_rows(s, test), ctx.test_rows(t, test)
        d["sx"] = s1.remove_test(pd.read_parquet(sx), ts, set())
        d["xt"] = s1.remove_test(pd.read_parquet(xt), set(), tt)
        sizes = {name: s1.dict_sizes(v) for name, v in d.items()}
    return d, sizes


def check_stamp(version_dir: Path, cache_dir: Path) -> None:
    """The evaluation cache is keyed by method and fold, not by the dictionaries: refuse to reuse
    a cache written for other dictionaries (a rebuilt version, or a smoke test's)."""
    stamp = hashlib.sha256((version_dir / "dict_sizes.csv").read_bytes()).hexdigest()[:16]
    f = cache_dir / "dictionaries.stamp"
    if f.exists() and f.read_text(encoding="utf-8").strip() != stamp:
        sys.exit(f"{cache_dir} holds evaluations of other dictionaries: delete it and rerun")
    cache_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(stamp + "\n", encoding="utf-8")


def configure(version: str) -> None:
    """Point stage 1's runner at one version's data, results, cache and methods."""
    if version not in XCFG["versions"]:
        sys.exit(f"unknown version {version}; config.yaml lists {XCFG['versions']}")
    s1.NAME, s1.DATA = f"{NAME}/{version}", f"md/{version}"
    s1.HERE = s1.p6_dir()                          # pairs.csv comes from the version's folder
    s1.RESULTS = HERE / "results" / version
    s1.RESULTS.parent.mkdir(exist_ok=True)         # stage 1's writer makes only the last level
    s1.RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/scarce-natural-s1/run.py",
                       "experiments/scarce-natural-s1/config.yaml", f"experiments/{NAME}/")
    s1.OWN_RESULTS = f"experiments/{NAME}/results/"
    s1.Ctx = Ctx
    s1.fold_dicts = fold_dicts
    s1.XCFG = {**s1.XCFG, "stage": {**XCFG, "version": version}}   # recorded in run_info.json


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--version", required=True)
    args, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    configure(args.version)
    if "--data-files" not in rest:
        check_stamp(s1.p6_dir(), s1.out_dir())
    s1.main()
