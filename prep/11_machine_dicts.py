"""RQ3: dictionaries made by machine translation -> <phase6>/md/<version>/.

Runs after the base stage's prep (07, 08; stage D for the "highres" set, stage 1 for "low").
For a set (configs/default.yaml `machine_dicts.sets`) and a system (nllb, on the GPU; llm, from
the hop cache that prep/12_llm_translate.py fills), every source word of each language is
translated (src/clwe_chains/machine_dicts.py: list words, plus the most frequent words for "low"):
- version <set>-<system>-a: each edge's dictionary from direct translations, both directions;
- version <set>-<system>-b: the s-t dictionary translated through the pair's hub (s -> hub -> t,
  both directions); the chain's edges as in version a ("highres") or natural ("low");
- version <set>-list ("highres" with list_version): every edge's dictionary is the shared list.
Each version folder gets the base stage's pairs, folds, words and gold (filtered to the set's
pairs) and dicts/<a>__<b>.parquet, so experiments/machine-dicts/run.py runs stage 1's runner on
it unchanged (test words removed per fold there). Translations are cached by (model, from, to,
input) in v1/results/mt-pivot/hops.jsonl, shared with the MT pivot pilots; a rerun resumes.

python prep/11_machine_dicts.py --set highres --system nllb
python prep/11_machine_dicts.py --set highres --data-files      # inputs, for the remote copy
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil

import pandas as pd

from clwe_chains import load_config
from clwe_chains.dictionaries import dict_sizes
from clwe_chains.llm import cache_tag
from clwe_chains.machine_dicts import (
    edges_for,
    list_dictionary,
    source_rows,
    translated_pairs,
    union_dictionary,
)
from clwe_chains.mt import CODES, NLLB, HopCache, run_routes
from clwe_chains.text import load_vocab, normalize, vocab_index

log = logging.getLogger("machine_dicts")


def set_data(cfg, name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                        pd.DataFrame]:
    """The set's settings, and its base stage's pairs, folds, words and gold, filtered to the
    set's pairs (both ends covered by the MT system)."""
    st = cfg["machine_dicts"]["sets"][name]
    base = cfg.path("phase6") / st["base"] if st["base"] else cfg.path("phase6")
    pairs = pd.read_csv(base / "pairs.csv")
    if "low" in st:
        pairs = pairs[pairs["s"].isin(st["low"])]
    pairs = pairs[pairs["s"].isin(CODES) & pairs["t"].isin(CODES)].reset_index(drop=True)
    keep = set(zip(pairs["s"], pairs["t"], strict=True))
    folds = pd.read_parquet(base / "folds.parquet")
    folds = folds[[(s, t) in keep for s, t in zip(folds["s"], folds["t"], strict=True)]]
    dirs = keep | {(t, s) for s, t in keep}
    gold = pd.read_parquet(base / "gold.parquet")
    gold = gold[[(a, b) in dirs for a, b in zip(gold["src"], gold["tgt"], strict=True)]]
    words = pd.read_parquet(base / "words.parquet")
    langs = set(pairs["s"]) | set(pairs["t"]) | set(pairs["hub"])
    return {**st, "base_dir": base}, pairs, folds, words[words["lang"].isin(langs)], gold


def jobs_for(pairs: pd.DataFrame, edges: list[tuple[str, str]], src: dict[str, list[int]],
             vocab: dict[str, list[str]]) -> list[tuple]:
    """Direct translation jobs for both directions of every edge, and pivoted jobs (through the
    pair's hub) for both directions of every pair. Job id: (kind, from, to, source row)."""
    jobs = []
    for a, b in edges:
        for x, y in ((a, b), (b, a)):
            jobs += [(("d", x, y, r), (x, y), vocab[x][r]) for r in src[x]]
    for s, t, h in zip(pairs["s"], pairs["t"], pairs["hub"], strict=True):
        for x, y in ((s, t), (t, s)):
            jobs += [(("p", x, y, r), (x, h, y), vocab[x][r]) for r in src[x]]
    return list({j[0]: j for j in jobs}.values())


MIN_PAIRS = 100        # a smaller machine dictionary means the translations failed: stop


def cache_file(cfg, system: str):
    """The hop cache: NLLB shares the MT pilots' file; the LLM has its own (prep/12 writes it on
    the laptop, so the remote's NLLB file can be copied back without losing it)."""
    if system == "nllb":
        return cfg.path("results") / "mt-pivot" / "hops.jsonl"
    return cfg.path("results") / "machine-dicts" / "hops_llm.jsonl"


def identical_share(d: pd.DataFrame, va: list[str], vb: list[str], a: str, b: str) -> float:
    """Share of pairs whose two words are the same string (MT often copies a word it can't
    translate)."""
    if d.empty:
        return 0.0
    same = [normalize(va[x], a) == normalize(vb[y], b)
            for x, y in zip(d["a"], d["b"], strict=True)]
    return round(sum(same) / len(same), 4)


def write_version(out, pairs, folds, words, gold, dicts: dict[tuple[str, str], pd.DataFrame],
                  vocab: dict[str, list[str]],
                  chain: dict[tuple[str, str], pd.DataFrame] | None = None) -> list[dict]:
    """One version folder in the stage layout; returns its dictionary sizes. chain: the chain's
    edges, written to chain/ when they must differ from a direct edge of the same name (version
    b; experiments/machine-dicts/run.py reads the chain from chain/ when it exists)."""
    if out.exists():
        shutil.rmtree(out)
    (out / "dicts").mkdir(parents=True)
    pairs.to_csv(out / "pairs.csv", index=False, lineterminator="\n")
    folds.to_parquet(out / "folds.parquet", index=False)
    words.to_parquet(out / "words.parquet", index=False)
    gold.to_parquet(out / "gold.parquet", index=False)
    sizes = []
    for folder, ds in (("dicts", dicts), ("chain", chain or {})):
        if not ds:
            continue                                # no empty chain/ folder for versions a
        (out / folder).mkdir(exist_ok=True)
        for (a, b), d in ds.items():
            d.to_parquet(out / folder / f"{a}__{b}.parquet", index=False)
            sizes.append({"folder": folder, "edge": f"{a}__{b}", **dict_sizes(d),
                          "identical": identical_share(d, vocab[a], vocab[b], a, b)})
    pd.DataFrame(sizes).to_csv(out / "dict_sizes.csv", index=False, lineterminator="\n")
    return sizes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["highres", "low"])
    ap.add_argument("--system", default="nllb", choices=["nllb", "llm"])
    ap.add_argument("--data-files", action="store_true", help="list inputs and exit")
    ap.add_argument("--limit", type=int, default=None, help="first N pairs only (smoke test)")
    ap.add_argument("--model", default=None, help="another NLLB model (smoke test)")
    args = ap.parse_args()
    cfg = load_config()
    st, pairs, folds, words, gold = set_data(cfg, args.set)
    if args.limit:
        pairs = pairs.head(args.limit)
        keep = set(zip(pairs["s"], pairs["t"], strict=True))
        folds = folds[[(s, t) in keep for s, t in zip(folds["s"], folds["t"], strict=True)]]
    edges = edges_for(pairs, st["chain_edges"])
    langs = sorted({lang for e in edges for lang in e} | set(pairs["hub"]))
    cache_path = cache_file(cfg, args.system)
    if args.data_files:
        base, root = st["base_dir"], cfg.data_root
        files = [base / f for f in ("pairs.csv", "folds.parquet", "words.parquet",
                                    "gold.parquet")]
        files += [cfg.path("v0_embeddings") / f"{lang}.vocab.txt" for lang in langs]
        if not st["chain_edges"]:
            files += [base / "dicts" / f"{a}__{b}.parquet" for s, t, x in
                      zip(pairs["s"], pairs["t"], pairs["hub"], strict=True)
                      for a, b in ((s, x), (x, t))]
        for f in dict.fromkeys(files):
            print(f.relative_to(root).as_posix())
        return

    vocab = {lang: load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt") for lang in langs}
    index = {lang: vocab_index(vocab[lang], lang) for lang in langs}
    ends = sorted({lang for e in edges for lang in e})
    src = {lang: source_rows(words, lang, vocab[lang], st["frequent_words"]) for lang in ends}
    md = cfg.path("phase6") / "md"
    sizes = {}
    if st.get("list_version"):
        dicts = {(a, b): list_dictionary(words, a, b) for a, b in edges}
        sizes[f"{args.set}-list"] = write_version(md / f"{args.set}-list", pairs, folds, words,
                                                  gold, dicts, vocab)

    nl = {**cfg["machine_dicts"]["nllb"], **({"model": args.model} if args.model else {})}
    if args.system == "nllb":
        tag = nl["model"].split("/")[-1]
        model = None

        def translate(ws, a, b):
            nonlocal model
            if model is None:
                model = NLLB(nl["model"], nl["beams"], nl["n_return"], nl["max_new_tokens"],
                             nl["batch"])
            return model(ws, a, b)
    else:
        tag = cache_tag(cfg["machine_dicts"]["llm"]["model"], "dict-v1")    # as prep/12

        missing: dict[str, int] = {}

        def translate(ws, a, b):
            # words prep/12 couldn't get (their requests failed twice; not cached): no pair,
            # like a multi-word answer. More than 1% stops the build (rerun prep/12 first).
            missing[f"{a}>{b}"] = missing.get(f"{a}>{b}", 0) + len(ws)
            return [None] * len(ws)

    jobs = jobs_for(pairs, edges, src, vocab)
    log.info("%d translation jobs (%s)", len(jobs), tag)
    final = run_routes(jobs, translate, HopCache(cache_path), tag)
    if args.system == "llm" and missing:
        n_missing = sum(missing.values())
        log.info("%d translations missing from the LLM cache (no pair): %s", n_missing, missing)
        if n_missing > 0.01 * len(jobs):
            raise SystemExit("more than 1% of the LLM translations are missing: rerun "
                             "prep/12_llm_translate.py to retry them first")

    def directed(kind: str, x: str, y: str) -> pd.DataFrame:
        rows = src[x]
        outs = [(final[kind, x, y, r] or [None])[0] for r in rows]
        return translated_pairs(rows, outs, y, index[y])

    dict_a = {(a, b): union_dictionary(directed("d", a, b), directed("d", b, a))
              for a, b in edges}
    small = {f"{a}__{b}": len(d) for (a, b), d in dict_a.items() if len(d) < MIN_PAIRS}
    if small:
        raise SystemExit(f"machine dictionaries under {MIN_PAIRS} pairs (failed translations?): "
                         f"{small}")
    chain_edges = list(dict.fromkeys(e for s, t, x in zip(pairs["s"], pairs["t"], pairs["hub"],
                                                          strict=True) for e in ((s, x), (x, t))))
    if st["chain_edges"]:
        chain = {e: dict_a[e] for e in chain_edges}
    else:                                           # the chain keeps its natural dictionaries
        chain = {e: pd.read_parquet(st["base_dir"] / "dicts" / f"{e[0]}__{e[1]}.parquet")
                 for e in chain_edges}
        dict_a |= chain
    # version b: each direct dictionary translated through the hub. A pair's direct edge can be
    # another pair's chain edge under the same name (en__ja: en-ja direct, en-zh via ja), so b's
    # chain (version a's edges) goes to chain/ and b's dicts/ holds only direct edges.
    dict_b = {(s, t): union_dictionary(directed("p", s, t), directed("p", t, s))
              for s, t in zip(pairs["s"], pairs["t"], strict=True)}
    for v, dicts, ch in (("a", dict_a, None), ("b", dict_b, chain)):
        name = f"{args.set}-{args.system}-{v}"
        sizes[name] = write_version(md / name, pairs, folds, words, gold, dicts, vocab, ch)
    # share of source words whose translation maps to a single in-vocab word, per kind and
    # direction (d = direct, p = through the hub)
    kept = {f"{k}:{x}>{y}": round(len(directed(k, x, y)) / max(len(src[x]), 1), 3)
            for k, x, y in sorted({j[0][:3] for j in jobs})}
    manifest = {"set": args.set, "system": args.system, "model": tag, "n_pairs": len(pairs),
                "n_jobs": len(jobs), "source_words": {k: len(v) for k, v in src.items()},
                "share_kept": kept, "sizes": sizes}
    (md / f"{args.set}-{args.system}.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                                        encoding="utf-8")
    log.info("wrote %s", ", ".join(sizes))


if __name__ == "__main__":
    main()
