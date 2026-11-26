"""Break-even grid: stage folders with both hub steps cut to fixed sizes.

experiments/break-even-grid/README.md (approved 2026-11-23). For the base stage (stage D) and every
size in break_even_grid.hub_sizes, writes <phase6>/<break_even_grid.data>/h<size>/ with:
- pairs.csv: the base pairs whose two full hub steps (s__x and x__t) both have more pairs than the
  size;
- dicts/: those pairs' hub steps, each as its first <size> rows in a fixed random order (seed
  break_even_grid.seed with the edge's name), so smaller sizes are subsets of larger ones and an
  edge shared by two pairs is cut the same way in both;
- folds.parquet, gold.parquet, words.parquet: copied from the base (the runner removes each fold's
  test words from both steps after cutting, as controlled scarcity does for direct);
- dict_sizes.csv for these dictionaries.
The direct dictionaries are not written: the experiment reuses controlled scarcity's direct ranks.
Deterministic; the base stage's own folder is not touched.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from clwe_chains import load_config
from clwe_chains.dictionaries import dict_sizes

log = logging.getLogger("break_even")
COPIED = ("folds.parquet", "gold.parquet", "words.parquet")
_spec = importlib.util.spec_from_file_location(
    "prep18", Path(__file__).resolve().parent / "18_controlled_scarcity.py")
_prep18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep18)
nested_order = _prep18.nested_order       # the same fixed permutation as controlled scarcity's


def steps(s: str, t: str, x: str) -> tuple[str, str]:
    return f"{s}__{x}", f"{x}__{t}"


def kept_pairs(pairs: pd.DataFrame, n_rows: dict[str, int], size: int) -> pd.DataFrame:
    """The pairs whose two hub steps both have more than `size` pairs."""
    keep = [all(n_rows[e] > size for e in steps(s, t, x))
            for s, t, x in zip(pairs["s"], pairs["t"], pairs["hub"], strict=True)]
    return pairs[keep].reset_index(drop=True)


def cut(d: pd.DataFrame, size: int, seed: int, name: str) -> pd.DataFrame:
    """The edge's first `size` rows in its fixed order, kept in their original order."""
    return d.iloc[np.sort(nested_order(len(d), seed, name)[:size])].reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    bg = cfg["break_even_grid"]
    p6 = cfg.path("phase6")
    # one order per edge, shared with controlled scarcity's direct cuts (README build notes)
    assert bg["seed"] == cfg["controlled_scarcity"]["seed"]
    src = p6 / bg["base"]
    pairs = pd.read_csv(src / "pairs.csv")
    names = sorted({e for s, t, x in zip(pairs["s"], pairs["t"], pairs["hub"], strict=True)
                    for e in steps(s, t, x)})
    dicts = {n: pd.read_parquet(src / "dicts" / f"{n}.parquet") for n in names}
    n_rows = {n: len(d) for n, d in dicts.items()}
    for size in bg["hub_sizes"]:
        out = p6 / bg["data"] / f"h{size}"
        if out.exists():
            shutil.rmtree(out)
        (out / "dicts").mkdir(parents=True)
        kept = kept_pairs(pairs, n_rows, size)
        kept.to_csv(out / "pairs.csv", index=False, lineterminator="\n")
        for name in COPIED:
            shutil.copy2(src / name, out / name)
        edges = sorted({e for s, t, x in zip(kept["s"], kept["t"], kept["hub"], strict=True)
                        for e in steps(s, t, x)})
        sizes = []
        for name in edges:
            d = cut(dicts[name], size, bg["seed"], name)
            d.to_parquet(out / "dicts" / f"{name}.parquet", index=False)
            sizes.append({"edge": name, **dict_sizes(d)})
        pd.DataFrame(sizes).to_csv(out / "dict_sizes.csv", index=False, lineterminator="\n")
        log.info("%s/h%s: %d of %d pairs, %d edges", bg["data"], size, len(kept), len(pairs),
                 len(edges))


if __name__ == "__main__":
    main()
