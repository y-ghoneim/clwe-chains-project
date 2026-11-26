"""MasakhaNEWS articles -> unit-length mean word-vector features -> <downstream>/masakhanews/.

Downstream MasakhaNEWS (experiments/downstream-masakhanews/README.md). Per language (yo, sw, am,
so, en, fr) and split: X_, n_ and labels_ (topic names; the label sets differ across languages and
the runner keeps each direction's shared topics). Text = headline and body. stats.csv is the
README's data check.
"""

from __future__ import annotations

import logging

from clwe_chains import load_config
from clwe_chains.downstream import write_topic_features


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    mc = cfg["downstream"]["masakhanews"]
    raw = cfg.path("raw_downstream") / "masakhanews"
    langs = {lang: raw / mc["folders"][lang] for lang in mc["languages"] + mc["labelled"]}
    write_topic_features(cfg, langs, mc["scripts"], cfg.path("downstream") / "masakhanews",
                         ("headline", "text"), logging.getLogger("masakhanews"))


if __name__ == "__main__":
    main()
