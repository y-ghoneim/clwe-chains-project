"""Controlled scarcity: stage folders with the direct dictionary shrunk to fixed budgets.

experiments/controlled-scarcity/README.md and experiments/downstream-tier2/README.md (approved
2026-11-20). For a base stage (d: stage D's 40 pairs; u: the UMSAB pairs, Tier 2 only) and every
budget in controlled_scarcity.budgets, plus "full", writes <phase6>/<base>c/b<budget>/ with:
- pairs.csv: the base pairs whose full direct dictionary has more pairs than the budget (all
  pairs for "full");
- dicts/: every edge of the base, unchanged (full). A pair's hub edge can be another pair's
  direct dictionary (stage D: en__tr is direct for en-tr and the hub edge of ko-tr), so the
  shrunk direct dictionaries cannot live here;
- direct/<s>__<t>.parquet: each pair's direct dictionary as its first <budget> rows in a fixed
  random order (seed controlled_scarcity.seed with the pair's name), so smaller budgets are
  subsets of larger ones (the full dictionary for "full");
- joined/<s>__<x>__<t>.parquet: s-t pairs joined on the hub word over the full hub edges (the
  same for every budget; stage D was built without them);
- folds.parquet, gold.parquet, words.parquet: copied from the base (the runner removes each fold's
  test words from every edge after subsampling, as in stage D);
- dict_sizes.csv for these dictionaries.
Deterministic; the base stage's own folder is not touched.
"""

from __future__ import annotations

import logging
import shutil
import zlib

import numpy as np
import pandas as pd

from clwe_chains import load_config
from clwe_chains.dictionaries import dict_sizes, join_counts

log = logging.getLogger("controlled")
COPIED = ("folds.parquet", "gold.parquet", "words.parquet")


def nested_order(n: int, seed: int, name: str) -> np.ndarray:
    """A fixed permutation of n rows for one dictionary; its prefixes are the budgets."""
    return np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))]).permutation(n)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    cc = cfg["controlled_scarcity"]
    p6 = cfg.path("phase6")
    for base in cc["bases"]:
        src = p6 / base
        pairs = pd.read_csv(src / "pairs.csv")
        dicts = {f.stem: pd.read_parquet(f) for f in sorted((src / "dicts").glob("*.parquet"))}
        joined = {f"{s}__{x}__{t}": join_counts(dicts[f"{s}__{x}"], dicts[f"{x}__{t}"])
                  for s, t, x in zip(pairs["s"], pairs["t"], pairs["hub"], strict=True)}
        order = {f"{s}__{t}": nested_order(len(dicts[f"{s}__{t}"]), cc["seed"], f"{s}__{t}")
                 for s, t in zip(pairs["s"], pairs["t"], strict=True)}
        for budget in [*cc["budgets"], "full"]:
            out = p6 / f"{base}c" / f"b{budget}"
            if out.exists():
                shutil.rmtree(out)
            (out / "dicts").mkdir(parents=True)
            (out / "direct").mkdir()
            (out / "joined").mkdir()
            keep = [budget == "full" or len(dicts[f"{s}__{t}"]) > budget
                    for s, t in zip(pairs["s"], pairs["t"], strict=True)]
            kept = pairs[keep].reset_index(drop=True)
            kept.to_csv(out / "pairs.csv", index=False, lineterminator="\n")
            for name in COPIED:
                shutil.copy2(src / name, out / name)
            sizes = []
            for name, d in dicts.items():
                d.to_parquet(out / "dicts" / f"{name}.parquet", index=False)
                sizes.append({"edge": name, **dict_sizes(d)})
            for s, t in zip(kept["s"], kept["t"], strict=True):
                name = f"{s}__{t}"
                d = dicts[name]
                if budget != "full":
                    d = d.iloc[np.sort(order[name][:budget])].reset_index(drop=True)
                d.to_parquet(out / "direct" / f"{name}.parquet", index=False)
                sizes.append({"edge": f"direct:{name}", **dict_sizes(d)})
            for s, t, x in zip(kept["s"], kept["t"], kept["hub"], strict=True):
                j = joined[f"{s}__{x}__{t}"]
                j.to_parquet(out / "joined" / f"{s}__{x}__{t}.parquet", index=False)
                sizes.append({"edge": f"{s}__{x}__{t}", **dict_sizes(j)})
            pd.DataFrame(sizes).to_csv(out / "dict_sizes.csv", index=False, lineterminator="\n")
            log.info("%sc/b%s: %d of %d pairs", base, budget, len(kept), len(pairs))


if __name__ == "__main__":
    main()
