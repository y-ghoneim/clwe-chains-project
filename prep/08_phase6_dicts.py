"""Phase 6 stage 1: each edge's natural dictionary and the joined dictionaries -> <phase6>/.

Runs after 07_phase6_list.py. Every edge a-b of a stage 1 route gets the word pairs its sources
attest (decisions.md 2026-11-16): PanLex in-vocab pairs, plus English Wiktionary translations
when one side is en. Writes dicts/<a>__<b>.parquet (a, b: vocab rows, in route order: s__t,
s__x, x__t, and s__en, en__t for phase6.en_chain languages), joined/<s>__<x>__<t>.parquet (a, b,
n: s-t pairs joined on the hub word, n = distinct linking hub words) and dict_sizes.csv. These are
full dictionaries: the runner removes each fold's test words.

`--stage a|d` (configs/default.yaml `phase6_stages`) reads and writes that stage's subfolder;
a stage with `joined: false` (D) gets no joined-count files.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from clwe_chains import load_config
from clwe_chains.dictionaries import dict_sizes, edge_dictionary, join_counts
from clwe_chains.sources import wiktionary_reverse
from clwe_chains.text import load_vocab, vocab_index

log = logging.getLogger("phase6_dicts")


def route_edges(pairs: pd.DataFrame, en_chain: list[str]) -> list[tuple[str, str]]:
    """Distinct edges (a, b) in route order: direct s-t, s-hub, hub-t, and s-en, en-t for the
    en_chain languages."""
    edges = []
    for s, t, x in zip(pairs["s"], pairs["t"], pairs["hub"], strict=True):
        edges += [(s, t), (s, x), (x, t)]
        if s in en_chain:
            edges += [(s, "en"), ("en", t)]
    return list(dict.fromkeys(edges))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="a phase6_stages key; omitted = stage 1")
    stage = ap.parse_args().stage
    cfg = load_config()
    st = cfg["phase6_stages"][stage] if stage else {}
    out = cfg.path("phase6") / st["data"] if stage else cfg.path("phase6")
    with_joined = st.get("joined", True)
    (out / "dicts").mkdir(parents=True, exist_ok=True)
    (out / "joined").mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(out / "pairs.csv")
    edges = route_edges(pairs, cfg["phase6"]["en_chain"])
    langs = sorted({lang for e in edges for lang in e})
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

    d, sizes = {}, []
    for a, b in edges:
        d[a, b] = edge_dictionary(pl[a], pl[b], wiktionary(a, b))
        d[a, b].to_parquet(out / "dicts" / f"{a}__{b}.parquet", index=False)
        sizes.append({"edge": f"{a}__{b}", **dict_sizes(d[a, b])})
    log.info("%d edges", len(edges))
    for s, t, x in (zip(pairs["s"], pairs["t"], pairs["hub"], strict=True) if with_joined
                    else []):
        j = join_counts(d[s, x], d[x, t])
        j.to_parquet(out / "joined" / f"{s}__{x}__{t}.parquet", index=False)
        sizes.append({"edge": f"{s}__{x}__{t}", **dict_sizes(j)})
    pd.DataFrame(sizes).to_csv(out / "dict_sizes.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
