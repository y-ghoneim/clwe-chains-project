"""SIB-200 sentences -> unit-length mean word-vector features -> <downstream>/sib200/.

Downstream SIB-200 (experiments/downstream-sib200/README.md). Per project language (mapped to its
SIB-200 code in configs/default.yaml) and split: X_, n_ and labels_ (the 7 topics, the same for
every language). stats.csv is the README's data check. The text stays in the data root.
"""

from __future__ import annotations

import logging

from clwe_chains import load_config
from clwe_chains.downstream import write_topic_features


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    sc = cfg["downstream"]["sib200"]
    raw = cfg.path("raw_downstream") / "sib200"
    langs = {lang: raw / code for lang, code in sc["codes"].items()}
    names = {"Latn": "Latin", "Arab": "Arabic", "Cyrl": "Cyrillic", "Grek": "Greek",
             "Hebr": "Hebrew", "Deva": "Devanagari", "Geor": "Georgian", "Hang": "Hangul",
             "Taml": "Tamil", "Orya": "Oriya", "Beng": "Bengali"}      # Unicode name prefixes
    scripts = {lang: names[code.split("_")[1]] for lang, code in sc["codes"].items()}
    write_topic_features(cfg, langs, scripts, cfg.path("downstream") / "sib200", ("text",),
                         logging.getLogger("sib200"))


if __name__ == "__main__":
    main()
