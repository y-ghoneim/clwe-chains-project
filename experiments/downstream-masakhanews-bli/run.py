"""A later phase 6 stage, run with stage 1's runner. See README.md (approved before running).

Reuses experiments/scarce-natural-s1/run.py unchanged in method: same folds, test removal, maps
and CSLS evaluation, on this stage's data (v1/phase6/<data>, prep/07 and prep/08 --stage <data>)
and only the methods in config.yaml. This file is identical in every such stage's folder; the
folder name is the experiment's name (cache and log under v1/results/<name>/).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
        """This stage's methods, as stage 1 labels them (thin, chain, combo, joined_mb, ...)."""
        return [s1.method_of(m, hub) for m in XCFG["methods"]]


def configure() -> None:
    """Point stage 1's runner at this stage's data, results, cache and methods."""
    s1.NAME, s1.DATA = NAME, XCFG["data"]
    s1.HERE, s1.RESULTS = HERE, HERE / "results"
    s1.RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/scarce-natural-s1/run.py",
                       "experiments/scarce-natural-s1/config.yaml", f"experiments/{NAME}/")
    s1.OWN_RESULTS = f"experiments/{NAME}/results/"
    s1.Ctx = Ctx
    s1.XCFG = {**s1.XCFG, "stage": XCFG}         # recorded in run_info.json


if __name__ == "__main__":
    configure()
    s1.main()
