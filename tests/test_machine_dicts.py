"""Tests of src/clwe_chains/machine_dicts.py, prep/11's job list and the RQ3 / teacher runners'
configuration (no model, no data root)."""

import importlib.util
from pathlib import Path

import pandas as pd

from clwe_chains import machine_dicts as md
from clwe_chains.text import vocab_index

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_is_word_accepts_letters_and_combining_marks_only():
    assert md.is_word("haus") and md.is_word("ଓଡ଼ିଆ") and md.is_word("Вода")
    assert not md.is_word("a") and not md.is_word("x2") and not md.is_word("e-mail")


def test_frequent_rows_skips_non_words():
    vocab = [",", "the", "1999", "a", "house", "co-op", "dog"]
    assert md.frequent_rows(vocab, 3) == [1, 4, 6]
    assert md.frequent_rows(vocab, 0) == []


def test_source_rows_joins_list_words_and_frequent_words():
    words = pd.DataFrame({"lang": ["fi", "fi", "fi", "et"], "concept_id": [0, 0, 1, 0],
                          "row": [5, 9, 7, 3], "is_word": [True, False, True, True]})
    vocab = ["ja", "on", "ei", "talo", "x", "y", "z", "kissa", "w", "Talo"]
    assert md.source_rows(words, "fi", vocab, 0) == [5, 7]          # list words only
    assert md.source_rows(words, "fi", vocab, 2) == [0, 1, 5, 7]


def test_output_row_cleans_and_needs_one_in_vocab_word():
    index = vocab_index(["koira", "Kissa", "talo"], "fi")
    assert md.output_row("Kissa.", "fi", index) == 1
    assert md.output_row("The dog .", "en", vocab_index(["dog"], "en")) == 0
    assert md.output_row("iso talo", "fi", index) is None           # two words
    assert md.output_row("auto", "fi", index) is None               # not in the vocab
    assert md.output_row(None, "fi", index) is None


def test_union_dictionary_orients_both_directions_as_a_b():
    ab = pd.DataFrame({"a": [1, 2], "b": [10, 20]})                  # a words -> b
    ba = pd.DataFrame({"a": [10, 30], "b": [1, 3]})                  # b words -> a
    u = md.union_dictionary(ab, ba)
    assert u.values.tolist() == [[1, 10], [2, 20], [3, 30]]


def test_list_dictionary_pairs_list_words_by_concept():
    words = pd.DataFrame({"lang": ["fi", "fi", "et", "et", "et"], "concept_id": [0, 1, 0, 2, 0],
                          "row": [5, 7, 3, 4, 8], "is_word": [True, True, True, True, False]})
    assert md.list_dictionary(words, "fi", "et").values.tolist() == [[5, 3]]


def test_edges_for_adds_chain_edges_only_when_asked():
    pairs = pd.DataFrame({"s": ["fi", "ko"], "t": ["ja", "tr"], "hub": ["en", "en"]})
    assert md.edges_for(pairs, False) == [("fi", "ja"), ("ko", "tr")]
    assert md.edges_for(pairs, True) == [("fi", "ja"), ("fi", "en"), ("en", "ja"),
                                         ("ko", "tr"), ("ko", "en"), ("en", "tr")]


def test_prep11_jobs_cover_both_directions_and_the_hub_route_once():
    p11 = _load("prep11", ROOT / "prep" / "11_machine_dicts.py")
    pairs = pd.DataFrame({"s": ["fi"], "t": ["ja"], "hub": ["en"]})
    vocab = {"fi": ["talo", "kissa"], "ja": ["家"], "en": ["house"]}
    src = {"fi": [0, 1], "ja": [0]}
    jobs = p11.jobs_for(pairs, [("fi", "ja")], src, vocab)
    ids = {j[0]: j[1] for j in jobs}
    assert ids[("d", "fi", "ja", 1)] == ("fi", "ja")
    assert ids[("d", "ja", "fi", 0)] == ("ja", "fi")
    assert ids[("p", "fi", "ja", 0)] == ("fi", "en", "ja")
    assert ids[("p", "ja", "fi", 0)] == ("ja", "en", "fi")
    assert len(jobs) == len(ids) == 6


def test_machine_dicts_runner_points_stage1_at_the_version():
    run = _load("md_run", ROOT / "experiments" / "machine-dicts" / "run.py")
    run.configure("highres-nllb-a")
    s1 = run.s1
    assert s1.DATA == "md/highres-nllb-a" and s1.NAME == "machine-dicts/highres-nllb-a"
    assert s1.HERE == s1.p6_dir() and s1.RESULTS.name == "highres-nllb-a"
    assert s1.Ctx.methods(None, "x", "en") == [("thin",), ("chain", "en"), ("combo", "en")]


def test_teacher_runner_uses_its_own_alpha_grid():
    run = _load("teacher_run", ROOT / "experiments" / "teacher-highres-d" / "run.py")
    run.configure()
    s1 = run.s1
    assert s1.ALPHAS == [0.03, 0.1, 0.3, 1] and s1.DATA == "d"
    assert ("teacherd", "0.1") in s1.Ctx.methods(None, "x", "en")
    assert s1.teacher_label(1.0, "teacherd") == "teacherd_1"


def test_llm_chunks_and_parses_by_echoed_source():
    from clwe_chains import llm
    c = llm.chunks({("fi", "ja"): ["a", "b", "c"], ("en", "ko"): ["x"]}, 2)
    assert c == {"fi-ja-0": ("fi", "ja", ["a", "b"]), "fi-ja-1": ("fi", "ja", ["c"]),
                 "en-ko-0": ("en", "ko", ["x"])}
    text = ('{"translations": [{"source": "b", "words": ["B", "B2", "B3", "B4"]}, '
            '{"source": "a", "words": [""]}]}')
    # "a" answered with no word -> []; "z" not in the reply -> None (asked again later)
    assert llm.parse(text, ["a", "b", "z"]) == [[], ["B"], None]
    assert llm.parse(text, ["b"], n=3) == [["B", "B2", "B3"]]
    assert llm.parse("not json", ["a"]) == [None]
    # echo in another Unicode form or casing still matches; same-length replies fall back to order
    nfd = '{"translations": [{"source": "Café", "words": ["x"]}]}'
    assert llm.parse(nfd, ["café"]) == [["x"]]
    order = '{"translations": [{"source": "?", "words": ["x"]}, {"source": "?", "words": ["y"]}]}'
    assert llm.parse(order, ["p", "q"]) == [["x"], ["y"]]
    one, three = llm.prompt(["talo"], "fi", "ja"), llm.prompt(["talo"], "fi", "ja", 3)
    assert "Finnish" in one and "Japanese" in one and '"talo"' in one
    assert "single most common" in one and "up to 3" in three
    p12 = _load("prep12", ROOT / "prep" / "12_llm_translate.py")    # still imports cleanly
    assert p12.BatchTranslator is llm.BatchTranslator


def test_version_b_chain_comes_from_chain_folder_even_when_names_clash(tmp_path):
    """en__ja is en-ja's direct edge (pivoted in version b) and en-zh-via-ja's chain edge."""
    run = _load("md_run_b", ROOT / "experiments" / "machine-dicts" / "run.py")
    run.configure("highres-nllb-b")
    (tmp_path / "dicts").mkdir()
    (tmp_path / "chain").mkdir()
    pivoted = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    direct_a = pd.DataFrame({"a": [1, 3], "b": [11, 30]})
    pivoted.to_parquet(tmp_path / "dicts" / "en__zh.parquet")
    pivoted.to_parquet(tmp_path / "dicts" / "en__ja.parquet")
    direct_a.to_parquet(tmp_path / "chain" / "en__ja.parquet")
    direct_a.to_parquet(tmp_path / "chain" / "ja__zh.parquet")
    ctx = run.Ctx.__new__(run.Ctx)
    ctx.p6, ctx._dicts = tmp_path, {}
    ctx.folds = {("en", "zh"): {0: [7]}}
    ctx.rows = {("en", 7): [3], ("zh", 7): [99]}
    d, sizes = run.fold_dicts(ctx, "en", "zh", "ja", 0)
    assert d["direct"].values.tolist() == [[1, 10], [2, 20]]       # version b's own direct edge
    assert d["sx"].values.tolist() == [[1, 11]]   # version a edge; test row 3 out
    assert sizes["sx"]["pairs"] == 1
    # a chain edge that exists only in chain/ is found by Ctx.dictionary too
    assert ctx.dictionary("ja__zh").values.tolist() == direct_a.values.tolist()


def test_stamp_refuses_a_cache_built_from_other_dictionaries(tmp_path):
    run = _load("md_run_stamp", ROOT / "experiments" / "machine-dicts" / "run.py")
    v, cache = tmp_path / "v", tmp_path / "cache"
    v.mkdir()
    (v / "dict_sizes.csv").write_text("edge,pairs\nen__ja,100\n", encoding="utf-8")
    run.check_stamp(v, cache)
    run.check_stamp(v, cache)                                      # same dictionaries: fine
    (v / "dict_sizes.csv").write_text("edge,pairs\nen__ja,101\n", encoding="utf-8")
    try:
        run.check_stamp(v, cache)
    except SystemExit:
        pass
    else:
        raise AssertionError("expected a refusal")


def test_prep11_identical_share_and_cache_files():
    p11 = _load("prep11b", ROOT / "prep" / "11_machine_dicts.py")
    d = pd.DataFrame({"a": [0, 1], "b": [0, 1]})
    assert p11.identical_share(d, ["Taxi", "kissa"], ["taxi", "neko"], "fi", "ja") == 0.5
    from clwe_chains import load_config
    cfg = load_config()
    assert p11.cache_file(cfg, "nllb").parent.name == "mt-pivot"
    assert p11.cache_file(cfg, "llm").name == "hops_llm.jsonl"


def test_run_routes_caches_each_language_pair_before_the_next(tmp_path):
    from clwe_chains.mt import HopCache, run_routes

    def flaky(ws, a, b):
        if b == "ko":
            raise RuntimeError("crash")
        return [["x"] for _ in ws]

    jobs = [("1", ("en", "ja"), "dog"), ("2", ("en", "ko"), "dog")]
    try:
        run_routes(jobs, flaky, HopCache(tmp_path / "h.jsonl"), "m")
    except RuntimeError:
        pass
    assert HopCache(tmp_path / "h.jsonl").get(("m", "en", "ja", "dog")) == ["x"]


class _FakeBatches:
    """Stands in for client.messages.batches: answers every word with WORD.upper(), except
    requests listed in `cut` (stop_reason max_tokens) on their first submission."""

    def __init__(self, cut=()):
        import itertools
        self.cut, self.created, self.ids, self.store = set(cut), [], itertools.count(), {}

    def create(self, requests):
        from types import SimpleNamespace as NS
        bid = f"b{next(self.ids)}"
        self.store[bid] = requests
        self.created.append(bid)
        return NS(id=bid)

    def retrieve(self, bid):
        from types import SimpleNamespace as NS
        return NS(processing_status="ended", request_counts=NS(processing=0))

    def results(self, bid):
        import json as js
        from types import SimpleNamespace as NS
        for r in self.store[bid]:
            words = js.loads(r["params"]["messages"][0]["content"].split("\n\n", 1)[1])
            stop = "max_tokens" if r["custom_id"] in self.cut else "end_turn"
            text = js.dumps({"translations": [{"source": w, "words": [w.upper()]}
                                              for w in words]})
            msg = NS(stop_reason=stop, content=[NS(type="text", text=text)],
                     usage=NS(input_tokens=10, output_tokens=20))
            yield NS(custom_id=r["custom_id"], result=NS(type="succeeded", message=msg))


def _translator(tmp_path, batches):
    from types import SimpleNamespace as NS

    from clwe_chains import llm
    cfg = {"model": "m", "max_tokens": 100, "effort": "low", "words_per_request": 2}
    t = llm.BatchTranslator(cfg, n=1, dry_run=True, state_dir=tmp_path)
    t.dry_run, t.probed = False, True
    t.client = NS(messages=NS(batches=batches))
    return t


def test_batch_translator_splits_cut_off_requests_and_counts_usage(tmp_path):
    batches = _FakeBatches(cut={"fi-ja-0"})
    t = _translator(tmp_path, batches)
    out = t({("fi", "ja"): ["a", "b", "c"]})
    assert out == {("fi", "ja"): [["A"], ["B"], ["C"]]}         # fi-ja-0 retried as two halves
    assert len(batches.created) == 2 and t.failed_share() == (0, 3)
    assert t.tokens == {"input": 40, "output": 80}              # 2 + 2 requests, all counted
    assert len((tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_batch_translator_resumes_a_saved_batch_instead_of_paying_again(tmp_path):
    batches = _FakeBatches()
    _translator(tmp_path, batches)({("fi", "ja"): ["a", "b"]})
    again = _translator(tmp_path, batches)({("fi", "ja"): ["a", "b"]})
    assert again == {("fi", "ja"): [["A"], ["B"]]} and len(batches.created) == 1


def test_failed_words_are_not_cached_so_a_rerun_asks_again(tmp_path):
    from clwe_chains.mt import HopCache, run_routes

    def many(need):
        return {ab: [None if w == "bad" else [w.upper()] for w in ws] for ab, ws in need.items()}

    cache = HopCache(tmp_path / "h.jsonl")
    out = run_routes([("1", ("fi", "en", "ja"), "bad"), ("2", ("fi", "ja"), "ok")], None, cache,
                     "m", translate_many=many)
    assert out == {"1": [], "2": ["OK"]}
    assert cache.get(("m", "fi", "en", "bad")) is None
    assert HopCache(tmp_path / "h.jsonl").get(("m", "fi", "ja", "ok")) == ["OK"]


def test_an_empty_chain_folder_falls_back_to_dicts(tmp_path):
    """highres-list and version a have no chain/ files (a remote run crashed on an empty
    chain/ folder): the chain then comes from dicts/."""
    run = _load("md_run_a", ROOT / "experiments" / "machine-dicts" / "run.py")
    run.configure("highres-list")
    (tmp_path / "dicts").mkdir()
    (tmp_path / "chain").mkdir()
    d = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    for name in ("en__zh", "en__ja", "ja__zh"):
        d.to_parquet(tmp_path / "dicts" / f"{name}.parquet")
    ctx = run.Ctx.__new__(run.Ctx)
    ctx.p6, ctx._dicts = tmp_path, {}
    ctx.folds = {("en", "zh"): {0: [7]}}
    ctx.rows = {("en", 7): [2], ("zh", 7): [99]}
    out, _ = run.fold_dicts(ctx, "en", "zh", "ja", 0)
    assert out["sx"].values.tolist() == [[1, 10]]                 # from dicts/, test row 2 out
