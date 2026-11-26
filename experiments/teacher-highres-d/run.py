"""Chain-as-teacher on stage D's well-resourced pairs, with stage 1's runner. See README.md.

The same runner as experiments/natural-highres-d/ (stage D's data, folds, test removal, maps and
CSLS evaluation), with the methods and teacher weights (alphas) in config.yaml. The teacher fits
the direct map with a pull towards the chain's map, lambda = alpha x n_direct / d ("teacherd");
alpha is chosen by cross-validation leaving one source language out, as in stage 1.
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


def teacher_methods() -> list[str]:
    return [f"teacherd_{a}" for a in XCFG["alphas"]]


class Ctx(s1.Ctx):
    def methods(self, s: str, hub: str) -> list[tuple[str, ...]]:
        return [s1.method_of(m, hub) for m in XCFG["methods"] + teacher_methods()]


def configure() -> None:
    """Point stage 1's runner at stage D's data, this experiment's results, cache and alphas."""
    s1.NAME, s1.DATA = NAME, XCFG["data"]
    s1.HERE, s1.RESULTS = HERE, HERE / "results"
    s1.RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/scarce-natural-s1/run.py",
                       "experiments/scarce-natural-s1/config.yaml", f"experiments/{NAME}/")
    s1.OWN_RESULTS = f"experiments/{NAME}/results/"
    s1.ALPHAS = sorted(XCFG["alphas"])           # write_results' cross-validation grid
    s1.Ctx = Ctx
    s1.XCFG = {**s1.XCFG, "stage": XCFG}         # recorded in run_info.json


if __name__ == "__main__":
    configure()
    s1.main()
