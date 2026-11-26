"""Strong maps: each pair's three widest hubs and the second and third hubs' dictionaries.

Runs after 08_phase6_dicts.py (experiments/strong-maps/README.md, candidate 4 "several hubs").
For each stage in STAGES, the pair's hubs are ranked by phase 6's hub rule applied repeatedly:
the largest min(size[s, x], size[x, t]) over the nodes other than s and t, ties to the first in
the node order (pairs.widest_hub); the first must be the stage's own hub. Writes
<phase6>/hubs/<stage>/hubs.csv (s, t, hub1-3, route1-3) and, for hubs 2 and 3, the edges' natural
dictionaries built as prep/08 builds the first hub's (PanLex in-vocab pairs, plus English
Wiktionary translations when one side is en): dicts/<a>__<b>.parquet in route order (s__x,
x__t) and dict_sizes.csv. Full dictionaries: the runner removes each fold's test words.
Controlled scarcity uses stage D's (same pairs, full hub edges). The default builds s1, a, a2 and
d; `--stage <key>` builds that stage only, the only way to build ll (experiments/low-low, which
needs prep/07 and 08 with --stage ll first).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from clwe_chains import load_config
from clwe_chains.dictionaries import dict_sizes, edge_dictionary
from clwe_chains.shared_list import list_languages
from clwe_chains.sources import wiktionary_reverse
from clwe_chains.text import load_vocab, vocab_index

log = logging.getLogger("strong_hubs")
# stage -> data subfolder of <phase6>; ll: both ends low-resource (experiments/low-low), built only
# with --stage ll, so the default needs nothing beyond the earlier stages
STAGES = {"s1": "", "a": "a", "a2": "a2", "d": "d", "ll": "ll"}
DEFAULT = ("s1", "a", "a2", "d")
N_HUBS = 3


def ranked_hubs(size: pd.DataFrame, s: str, t: str, nodes: list[str], n: int
                ) -> list[tuple[str, int]]:
    """The n nodes x (not s or t) with the largest min(size[s, x], size[x, t]), widest first,
    ties to the first in `nodes` (widest_hub's rule, applied repeatedly), with that bottleneck."""
    via = [(x, int(min(size.at[s, x], size.at[x, t]))) for x in nodes if x not in (s, t)]
    return sorted(via, key=lambda v: -v[1])[:n]      # sorted is stable: ties keep node order


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), default=None,
                    help="one stage (default: s1, a, a2, d)")
    only = ap.parse_args().stage
    todo = DEFAULT if only is None else (only,)
    cfg = load_config()
    p6 = cfg.path("phase6")
    nodes = list_languages(pd.read_parquet(cfg.path("list") / "shared_list.parquet"))
    cp = pd.read_parquet(cfg.path("v0_coverage_pairs"))
    cp["d"] = cp[["n_src_words", "n_tgt_words"]].min(axis=1)
    size = cp.pivot(index="src", columns="tgt", values="d").fillna(0).astype(int)
    wk = pd.read_parquet(cfg.path("list") / "wiktionary_en.parquet", columns=["en", "code", "word"])
    en_index = vocab_index(load_vocab(cfg.path("v0_embeddings") / "en.vocab.txt"), "en")
    panlex: dict[str, pd.DataFrame] = {}

    def pl(lang: str) -> pd.DataFrame:
        if lang not in panlex:
            panlex[lang] = pd.read_parquet(cfg.path("v0_panlex") / f"{lang}.parquet")
        return panlex[lang]

    def wiktionary(a: str, b: str) -> pd.DataFrame | None:      # as prep/08
        if "en" not in (a, b):
            return None
        other = b if a == "en" else a
        index = vocab_index(load_vocab(cfg.path("v0_embeddings") / f"{other}.vocab.txt"), other)
        rev = wiktionary_reverse(wk, cfg["list"]["wikt_codes"].get(other, [other]), en_index,
                                 index, other)
        if a == "en":
            return pd.DataFrame({"a": rev["en_key"], "b": rev["word_idx"]})
        return pd.DataFrame({"a": rev["word_idx"], "b": rev["en_key"]})

    for stage in todo:
        data = STAGES[stage]
        pairs = pd.read_csv((p6 / data if data else p6) / "pairs.csv")
        out = p6 / "hubs" / stage
        (out / "dicts").mkdir(parents=True, exist_ok=True)
        rows, edges = [], []
        for s, t, hub in zip(pairs["s"], pairs["t"], pairs["hub"], strict=True):
            ranked = ranked_hubs(size, s, t, nodes, N_HUBS)
            if ranked[0][0] != hub:
                raise ValueError(f"{stage} {s}-{t}: first hub {ranked[0][0]}, stage has {hub}")
            rows.append({"s": s, "t": t, **{f"hub{i + 1}": x for i, (x, _) in enumerate(ranked)},
                         **{f"route{i + 1}": r for i, (_, r) in enumerate(ranked)}})
            edges += [e for x, _ in ranked[1:] for e in ((s, x), (x, t))]
        pd.DataFrame(rows).to_csv(out / "hubs.csv", index=False, lineterminator="\n")
        sizes = []
        for a, b in dict.fromkeys(edges):
            d = edge_dictionary(pl(a), pl(b), wiktionary(a, b))
            d.to_parquet(out / "dicts" / f"{a}__{b}.parquet", index=False)
            sizes.append({"edge": f"{a}__{b}", **dict_sizes(d)})
        pd.DataFrame(sizes).to_csv(out / "dict_sizes.csv", index=False, lineterminator="\n")
        log.info("%s: %d pairs, %d new edges", stage, len(pairs), len(sizes))


if __name__ == "__main__":
    main()
