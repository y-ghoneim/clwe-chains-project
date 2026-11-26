"""Concept splits for the shared list -> <list>/splits.json. Runs after 03_build_list.py with a
coverage threshold set; refuses to run without shared_list.parquet."""

from __future__ import annotations

import json
import logging
import sys

import pandas as pd

from clwe_chains import load_config
from clwe_chains.splits import concept_splits

log = logging.getLogger("splits")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    path = cfg.path("list") / "shared_list.parquet"
    if not path.exists():
        sys.exit(f"{path} not found: set list.coverage_threshold and run prep/03_build_list.py")
    ids = pd.read_parquet(path, columns=["concept_id"])["concept_id"].to_numpy()
    lc = cfg["list"]
    splits = concept_splits(ids, lc["test_frac"], lc["split_seeds"])
    payload = {"test_frac": lc["test_frac"], "n_concepts": len(ids),
               "splits": {str(s): v for s, v in splits.items()}}
    (cfg.path("list") / "splits.json").write_text(json.dumps(payload, indent=1) + "\n",
                                                  encoding="utf-8")
    log.info("%d concepts; %d seeds; %d test concepts each", len(ids), len(splits),
             len(splits[lc["split_seeds"][0]]["test"]))


if __name__ == "__main__":
    main()
