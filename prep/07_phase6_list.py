"""Phase 6 stage 1: pairs, the new languages' list words, folds and gold -> <phase6>/.

Runs after the shared list exists (decisions.md 2026-11-16; spec
docs/superpowers/specs/2026-11-16-phase6-stage1-design.md). Writes:
- pairs.csv (also experiments/scarce-natural-s1/pairs.csv): the low-resource x node pairs with
  direct PanLex size below phase6.max_direct, different family, and their hub;
- list/<lang>.parquet: each low-resource language's list words, built exactly as the 30 nodes'
  (prep/03_build_list.build_language over all candidates: PanLex -> Wiktionary -> one-to-one ->
  round trip);
- words.parquet (lang, concept_id, row, is_word): every endpoint's list word per list concept
  (round trip passed; capitalized-sense concepts dropped), with all casing variants of it, for
  test removal; is_word marks the list word's own row, the query;
- folds.parquet (s, t, concept_id, fold): each pair's eligible concepts in 5 disjoint folds;
- gold.parquet (src, tgt, concept_id, kind, row): per direction, kind "main" (list word, PanLex
  translations of the source word, casing variants) and "list" (the list word and its casings).
The main list and its files are only read (brief 5.10 isolation).

Later stages (`--stage a|d`, configs/default.yaml `phase6_stages`) write the same files to their
own subfolder of <phase6> and experiment folder: A takes the distant low-resource x node pairs with
direct in [min_direct, max_direct); D takes a fixed pair list (phase 4's 40 pairs) with the same
hub rule. An English endpoint gets the concept's English word and English Wiktionary answers.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging

import nltk
import pandas as pd

from clwe_chains import load_config
from clwe_chains.concepts import wordnet_lowercase
from clwe_chains.config import REPO_ROOT
from clwe_chains.gold import expand_gold, gold_sets, with_variants
from clwe_chains.pairs import scarce_pairs, widest_hub
from clwe_chains.shared_list import capitalized_senses, case_support, list_languages
from clwe_chains.sources import wiktionary_reverse
from clwe_chains.splits import concept_folds
from clwe_chains.text import casing_groups, identical_pairs, load_vocab, variant_rows, vocab_index

log = logging.getLogger("phase6_list")

_spec = importlib.util.spec_from_file_location("build_list",
                                               REPO_ROOT / "prep" / "03_build_list.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def eligible_concepts(words_s: pd.Series, words_t: pd.Series, dropped: set[int], s: str,
                      t: str) -> list[int]:
    """Concepts both languages have (concept_id -> word), minus dropped ones and identical-string
    pairs (text.identical_pairs), sorted."""
    ident = identical_pairs(words_s, words_t, s, t)
    return sorted(int(c) for c, same in ident.items() if not same and int(c) not in dropped)


def word_rows(lang: str, words: pd.Series, variants: dict[int, list[int]]) -> pd.DataFrame:
    """(lang, concept_id, row, is_word) for each concept's list word (concept_id -> vocab row) and
    its other casings."""
    rows = []
    for c, w in words.items():
        for r in with_variants([int(w)], variants):
            rows.append((lang, int(c), r, r == int(w)))
    rows.sort(key=lambda x: (x[1], not x[3], x[2]))
    return pd.DataFrame(rows, columns=["lang", "concept_id", "row", "is_word"])


def stage_settings(cfg, stage: str | None) -> dict:
    """Output subfolder, experiment folder and pair rule of a stage (None = stage 1)."""
    pc = cfg["phase6"]
    if stage is None:
        return {"data": "", "experiment": "scarce-natural-s1", "low": pc["low"], "min_direct": 0,
                "max_direct": pc["max_direct"], "pairs_from": None}
    st = cfg["phase6_stages"][stage]
    return {"data": st["data"], "experiment": st["experiment"], "low": st.get("low", []),
            "min_direct": st.get("min_direct", 0), "max_direct": st.get("max_direct"),
            "pairs_from": st.get("pairs_from")}


def wiktionary_answers(rev: pd.DataFrame, src: str, rows_src: pd.Series) -> pd.DataFrame:
    """English Wiktionary answers (concept_id, word_idx) for a pair with English at one end.
    rev: (word_idx of the other language, en_key) from sources.wiktionary_reverse; rows_src:
    concept -> the source word's row (en_key when src is English)."""
    keys = pd.DataFrame({"concept_id": rows_src.index, "src": rows_src.to_numpy()})
    if src == "en":
        m = keys.merge(rev, left_on="src", right_on="en_key")
        return m[["concept_id", "word_idx"]].reset_index(drop=True)
    m = keys.merge(rev, left_on="src", right_on="word_idx")
    return pd.DataFrame({"concept_id": m["concept_id"], "word_idx": m["en_key"]})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="a phase6_stages key; omitted = stage 1")
    cfg = load_config()
    st = stage_settings(cfg, ap.parse_args().stage)
    pc, lst = cfg["phase6"], cfg.path("list")
    out = cfg.path("phase6") / st["data"] if st["data"] else cfg.path("phase6")
    experiment = REPO_ROOT / "experiments" / st["experiment"]
    (out / "list").mkdir(parents=True, exist_ok=True)
    wide = pd.read_parquet(lst / "shared_list.parquet")
    nodes = list_languages(wide)
    concepts = pd.read_parquet(lst / "candidates.parquet")
    wk = pd.read_parquet(lst / "wiktionary_en.parquet", columns=["en", "code", "word"])
    en_pl = pd.read_parquet(cfg.path("v0_panlex") / "en.parquet")
    en_index = vocab_index(load_vocab(cfg.path("v0_embeddings") / "en.vocab.txt"), "en")

    # The capitalized-sense rule, on the shared list as in its analysis (42 of 883 concepts).
    nltk.data.path.insert(0, str(cfg.path("nltk_data")))
    codes = {lang: cfg["list"]["wikt_codes"].get(lang, [lang]) for lang in nodes if lang != "en"}
    support = case_support(wide, wk, codes)
    dropped = set(capitalized_senses(support, {int(c): wordnet_lowercase(w) for c, w in
                                               zip(wide["concept_id"], wide["en"], strict=True)}))
    list_ids = [int(c) for c in wide["concept_id"] if int(c) not in dropped]
    log.info("%d list concepts after dropping %d capitalized-sense concepts", len(list_ids),
             len(dropped))

    langs = pd.read_csv(cfg.path("v0_languages")).set_index("ft_code")
    cp = pd.read_parquet(cfg.path("v0_coverage_pairs"))
    cp["d"] = cp[["n_src_words", "n_tgt_words"]].min(axis=1)
    size = cp.pivot(index="src", columns="tgt", values="d").fillna(0).astype(int)
    if st["pairs_from"]:
        fixed = pd.read_csv(REPO_ROOT / st["pairs_from"])
        pairs = pd.DataFrame([{"s": s, "t": t, "direct": int(size.at[s, t]),
                               **dict(zip(("hub", "route"), widest_hub(size, s, t, nodes),
                                          strict=True))}
                              for s, t in zip(fixed["s"], fixed["t"], strict=True)])
    else:
        pairs = scarce_pairs(size, langs["family"].to_dict(), st["low"], nodes, st["max_direct"])
        pairs = pairs[pairs["direct"] >= st["min_direct"]].reset_index(drop=True)
    pairs.to_csv(out / "pairs.csv", index=False, lineterminator="\n")
    experiment.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(experiment / "pairs.csv", index=False, lineterminator="\n")
    log.info("%d pairs: %s", len(pairs), pairs["s"].value_counts().to_dict())

    # List words (concept_id -> vocab row and word) of every endpoint, round trip passed. key: the
    # row PanLex and Wiktionary use for a word (its most frequent casing; same as row except for
    # English, whose list word keeps its own row, en_rank, as in phase 4: decisions.md 2026-11-19).
    row, key, word, vocab = {}, {}, {}, {}
    for lang in st["low"]:
        df, _ = build.build_language(lang, cfg, concepts, en_pl, en_index, wk)
        df.to_parquet(out / "list" / f"{lang}.parquet", index=False)
        df = df[df["round_trip"] & df["concept_id"].isin(list_ids)].set_index("concept_id")
        row[lang], word[lang] = df["word_idx"].astype(int), df["word"]
        log.info("%s: %d list words pass the round trip", lang, len(df))
    for lang in sorted((set(pairs["s"]) | set(pairs["t"])) - set(st["low"])):
        if lang == "en":           # the concept's English word: its own row (center, not Center)
            c = concepts.set_index("concept_id").loc[list_ids]
            row[lang], word[lang] = c["en_rank"].astype(int), c["en"]
            key[lang] = c["en_key"].astype(int)
            continue
        p = pd.read_parquet(lst / "per_lang" / f"{lang}.parquet")
        p = p[p["round_trip"].astype(bool) & p["concept_id"].isin(list_ids)].set_index(
            "concept_id")
        row[lang], word[lang] = p["word_idx"].astype(int), p["word"]
    for lang, rows_l in row.items():
        key.setdefault(lang, rows_l)
        words_l = load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt")
        # English: casings found from any row (its list word isn't always the most frequent
        # casing); other languages unchanged, so stages without English reproduce exactly
        vocab[lang] = casing_groups(words_l, lang) if lang == "en" else variant_rows(words_l, lang)
    pd.concat([word_rows(lang, row[lang], vocab[lang]) for lang in sorted(row)],
              ignore_index=True).to_parquet(out / "words.parquet", index=False)

    folds, gold = [], []
    pl = {lang: pd.read_parquet(cfg.path("v0_panlex") / f"{lang}.parquet") for lang in row}
    rev = {}
    for lang in row:
        if lang != "en" and "en" in row:
            rev[lang] = wiktionary_reverse(
                wk, cfg["list"]["wikt_codes"].get(lang, [lang]), en_index,
                vocab_index(load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt"), lang),
                lang)
    for s, t in zip(pairs["s"], pairs["t"], strict=True):
        ids = eligible_concepts(word[s], word[t], dropped, s, t)
        for f, test in concept_folds(pd.Series(ids).to_numpy(), pc["folds"],
                                     pc["fold_seed"]).items():
            folds += [(s, t, c, f) for c in test]
        for a, b in ((s, t), (t, s)):
            wk_ans = (wiktionary_answers(rev[b if a == "en" else a], a, key[a].loc[ids])
                      if "en" in (a, b) else None)
            main = expand_gold(gold_sets(row[b].loc[ids], key[a].loc[ids], pl[a], pl[b], wk_ans),
                               vocab[b])
            for c in ids:
                gold += [(a, b, c, "main", r) for r in main[c]]
                gold += [(a, b, c, "list", r) for r in with_variants([int(row[b][c])], vocab[b])]
        log.info("%s-%s: %d eligible concepts", s, t, len(ids))
    pd.DataFrame(folds, columns=["s", "t", "concept_id", "fold"]).to_parquet(
        out / "folds.parquet", index=False)
    pd.DataFrame(gold, columns=["src", "tgt", "concept_id", "kind", "row"]).to_parquet(
        out / "gold.parquet", index=False)


if __name__ == "__main__":
    main()
