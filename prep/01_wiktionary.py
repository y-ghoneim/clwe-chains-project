"""English-entry translation tables from the wiktextract dump -> <list>/wiktionary_en.parquet.

Input: <raw_wiktionary>/raw-wiktextract-data.jsonl.gz (wiktextract 2026-10-30, Wiktionary dump
2026-10-01). Takes about 15 minutes; rows keep the dump's order, so reruns are identical.
"""

from __future__ import annotations

import logging

from clwe_chains import load_config
from clwe_chains.sources import read_wiktextract

log = logging.getLogger("wiktionary")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    out = cfg.path("list")
    out.mkdir(parents=True, exist_ok=True)
    df = read_wiktextract(cfg.path("raw_wiktionary") / "raw-wiktextract-data.jsonl.gz")
    df.to_parquet(out / "wiktionary_en.parquet", index=False)
    log.info("%d translation rows from %d English headwords, %d language codes", len(df),
             df["en"].nunique(), df["code"].nunique())


if __name__ == "__main__":
    main()
