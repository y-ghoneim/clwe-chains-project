"""Phase 6 stage 1E: every edge a one- or two-hop route can use, and URIEL -> <phase6>/routes/.

Runs after 07_phase6_list.py (decisions.md 2026-11-17). For the stage 1 pairs, routes go
s -> x -> t and s -> x1 -> x2 -> t through the 30 nodes, so the edges are s-x for each low-resource
language s and node x, and x-y for any two nodes. Each gets its natural dictionary as in
08_phase6_dicts.py (PanLex in-vocab pairs, plus English Wiktionary on edges with en), written
once per unordered pair as dicts/<a>__<b>.parquet (s first for s-x; nodes in alphabetical order)
with sizes in dict_sizes.csv. uriel.csv: cosine distances between lang2vec `syntax_knn` vectors
(URIEL syntax, kNN-imputed: it covers all 36 languages; v0's URIEL+ table has none of the six new
ones). Full dictionaries: the runner removes each fold's test words.
"""

from __future__ import annotations

import itertools
import logging
import warnings

import pandas as pd

from clwe_chains import load_config
from clwe_chains.aus_rule import cosine_distance, feature_vector
from clwe_chains.dictionaries import dict_sizes, edge_dictionary
from clwe_chains.shared_list import list_languages
from clwe_chains.sources import wiktionary_reverse
from clwe_chains.text import load_vocab, vocab_index

log = logging.getLogger("phase6_route_dicts")
LANG2VEC_CODES = {"ar": "arb"}    # as phase 4 (graph-all-pairs/config.yaml)


def route_edges(low: list[str], nodes: list[str]) -> list[tuple[str, str]]:
    """Unordered edges in storage order: (s, x) for each low-resource s and node x, then (x, y)
    for each pair of nodes, alphabetical."""
    return [(s, x) for s in low for x in nodes] + list(itertools.combinations(sorted(nodes), 2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    out = cfg.path("phase6") / "routes"
    (out / "dicts").mkdir(parents=True, exist_ok=True)
    nodes = list_languages(pd.read_parquet(cfg.path("list") / "shared_list.parquet"))
    low = cfg["phase6"]["low"]
    langs = sorted(set(nodes) | set(low))
    pl = {lang: pd.read_parquet(cfg.path("v0_panlex") / f"{lang}.parquet") for lang in langs}
    wk = pd.read_parquet(cfg.path("list") / "wiktionary_en.parquet", columns=["en", "code", "word"])
    en_index = vocab_index(load_vocab(cfg.path("v0_embeddings") / "en.vocab.txt"), "en")

    def wiktionary(a: str, b: str) -> pd.DataFrame | None:
        if "en" not in (a, b):
            return None
        other = b if a == "en" else a
        index = vocab_index(load_vocab(cfg.path("v0_embeddings") / f"{other}.vocab.txt"), other)
        rev = wiktionary_reverse(wk, cfg["list"]["wikt_codes"].get(other, [other]), en_index,
                                 index, other)
        if a == "en":
            return pd.DataFrame({"a": rev["en_key"], "b": rev["word_idx"]})
        return pd.DataFrame({"a": rev["word_idx"], "b": rev["en_key"]})

    edges = route_edges(low, nodes)
    sizes = []
    for i, (a, b) in enumerate(edges):
        d = edge_dictionary(pl[a], pl[b], wiktionary(a, b))
        d.to_parquet(out / "dicts" / f"{a}__{b}.parquet", index=False)
        sizes.append({"a": a, "b": b, **dict_sizes(d)})
        if i % 50 == 0:
            log.info("%d/%d edges", i, len(edges))
    pd.DataFrame(sizes).to_csv(out / "dict_sizes.csv", index=False, lineterminator="\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lang2vec.lang2vec as l2v
    iso = pd.read_csv(cfg.path("v0_languages")).set_index("ft_code")["iso3"].to_dict()
    codes = {lang: LANG2VEC_CODES.get(lang, iso[lang]) for lang in langs}
    feats = l2v.get_features(sorted(set(codes.values())), "syntax_knn")
    vec = {lang: feature_vector(feats[codes[lang]]) for lang in langs}
    pd.DataFrame([{"a": a, "b": b, "uriel": cosine_distance(vec[a], vec[b])}
                  for a, b in itertools.combinations(langs, 2)]).to_csv(
        out / "uriel.csv", index=False, lineterminator="\n", float_format="%.6g")
    log.info("%d edges, uriel for %d languages", len(edges), len(langs))


if __name__ == "__main__":
    main()
