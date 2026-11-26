"""RQ3: translate prep/11's source words with an LLM (Message Batches API) into its cache.

Builds exactly prep/11's translation jobs for a set (direct, both directions of every edge, and
through the pair's hub) and fills v1/results/machine-dicts/hops_llm.jsonl (prep/11.cache_file),
depth by depth with src/clwe_chains/llm.BatchTranslator (one batch per depth covering every
language pair, `words_per_request` words per request, JSON out, one probe request first).
prep/11_machine_dicts.py --system llm then builds the dictionaries from that cache without
calling the API. Needs ANTHROPIC_API_KEY (set by the user; never printed). Failed requests are
retried once; if more than 5% of a depth's words still have no translation, nothing is cached
and it stops. Prints the token usage and cost at the end.

python prep/12_llm_translate.py --set highres [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

from clwe_chains import load_config
from clwe_chains.llm import MAX_FAILED, BatchTranslator, cache_tag
from clwe_chains.mt import HopCache, run_routes
from clwe_chains.text import load_vocab

log = logging.getLogger("llm_translate")
_spec = importlib.util.spec_from_file_location(
    "prep11", Path(__file__).resolve().parent / "11_machine_dicts.py")
p11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p11)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["highres", "low"])
    ap.add_argument("--dry-run", action="store_true", help="count requests, call nothing")
    args = ap.parse_args()
    cfg = load_config()
    lc = cfg["machine_dicts"]["llm"]
    st, pairs, _, words, _ = p11.set_data(cfg, args.set)
    edges = p11.edges_for(pairs, st["chain_edges"])
    ends = sorted({lang for e in edges for lang in e})
    vocab = {lang: load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt")
             for lang in sorted(set(ends) | set(pairs["hub"]))}
    src = {lang: p11.source_rows(words, lang, vocab[lang], st["frequent_words"]) for lang in ends}
    jobs = p11.jobs_for(pairs, edges, src, vocab)
    log.info("%d translation jobs; first depth is one batch", len(jobs))
    cache_path = p11.cache_file(cfg, "llm")
    llm = BatchTranslator(lc, n=1, dry_run=args.dry_run, state_dir=cache_path.parent)

    def one_at_a_time(ws, a, b):                    # never used: translate_many covers it
        raise AssertionError

    run_routes(jobs, one_at_a_time, HopCache(cache_path), cache_tag(lc["model"], "dict-v1"),
               translate_many=llm)
    cost = llm.cost(lc["price_per_mtok"])
    failed, total = llm.failed_share()
    if failed > max(MAX_FAILED * total, 50):
        raise SystemExit(f"{failed} of {total} words failed (not cached): rerun to retry them")
    log.info("%d requests, %d input / %d output tokens, about $%.2f", llm.requests,
             llm.tokens["input"], llm.tokens["output"], cost)


if __name__ == "__main__":
    main()
