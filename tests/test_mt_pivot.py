"""Tests of experiments/mt-pivot/run.py's logic, with a fake translator (no model)."""

import importlib.util
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "experiments" / "mt-pivot" / "run.py"
_spec = importlib.util.spec_from_file_location("mt_pivot_run", RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_rank_matches_whole_output_or_first_word_after_normalizing():
    gold = {run.norm_answer("water", "en")}
    assert run.rank(["Sea", "Water.", "liquid"], gold, "en") == 2
    assert run.rank(["water bottle"], gold, "en") == 1          # first word of a phrase
    assert run.rank(["sea", "ocean"], gold, "en") == 0
    assert run.norm_answer(" «Вода» ", "ru") == "вода"


def test_routes_for_adds_one_hop_routes_and_fixed_ones_once():
    r = run.routes_for("tr", "ja", ["en", "fr", "ja", "tr"],
                       [("tr", "en", "ja"), ("tr", "fr", "en", "ja")])
    assert r == [("tr", "ja"), ("tr", "en", "ja"), ("tr", "fr", "ja"), ("tr", "fr", "en", "ja")]


def test_run_routes_chains_top_outputs_and_caches_each_hop_once(tmp_path):
    table = {("tr", "en", "su"): ["water", "aqua"],
             ("en", "ja", "water"): ["水", "お水"],
             ("tr", "ja", "su"): ["海"], ("tr", "fr", "su"): ["eau"],
             ("fr", "en", "eau"): ["water"]}
    calls = []

    def fake(words, a, b):
        calls.append((a, b, tuple(words)))
        return [table.get((a, b, w), []) for w in words]

    cache = run.HopCache(tmp_path / "hops.jsonl")
    jobs = [("d", ("tr", "ja"), "su"), ("e", ("tr", "en", "ja"), "su"),
            ("f", ("tr", "fr", "en", "ja"), "su"), ("x", ("tr", "fr", "ja"), "su")]
    out = run.run_routes(jobs, fake, cache, "m")
    assert out == {"d": ["海"], "e": ["水", "お水"], "f": ["水", "お水"], "x": []}
    # en->ja "water" is needed by two routes at different depths but translated once
    assert sum(1 for a, b, ws in calls if (a, b) == ("en", "ja")) == 1
    n = len(calls)
    again = run.run_routes(jobs, fake, run.HopCache(tmp_path / "hops.jsonl"), "m")
    assert again == out and len(calls) == n                    # resumed entirely from the cache


def test_language_codes_cover_every_node_and_phase6_language():
    assert len(run.NODES) == 30 and "en" in run.NODES
    assert all(code.count("_") == 1 for code in run.CODES.values())


def test_distinct_keeps_the_first_of_each_normalized_form():
    cands = ["Paholainen", "paholainen", "paholainen.", "piru", "Piru!"]
    assert run.distinct(cands, "fi") == ["Paholainen", "piru"]


def test_clean_hop_strips_phrase_padding_and_articles():
    assert run.clean_hop("The fox .", "en") == "fox"
    assert run.clean_hop("It 's a fan .", "en") == "fan"
    assert run.clean_hop("La volpe.", "it") == "volpe"
    assert run.clean_hop("l'acqua", "it") == "acqua"
    assert run.clean_hop("il-kelb", "mt") == "kelb"
    assert run.clean_hop("The", "en") == "The"                  # nothing left: keep the output
    assert run.clean_hop("Kettu.", "fi") == "Kettu"


def test_rank_lenient_matches_cleaned_output_or_any_word():
    gold = {run.norm_answer("volpe", "it")}
    assert run.rank(["La volpe."], gold, "it") == 0              # strict misses the article
    assert run.rank_lenient(["La volpe."], gold, "it") == 1
    assert run.rank_lenient(["una piccola volpe", "volpe"], gold, "it") == 1
    assert run.rank_lenient(["cane"], gold, "it") == 0


def test_next_hop_gets_the_cleaned_output(tmp_path):
    seen = []

    def fake(words, a, b):
        seen.append((a, b, tuple(words)))
        return [["The fox ."] if a == "ru" else ["Kettu"] for _ in words]

    out = run.run_routes([("j", ("ru", "en", "fi"), "лиса")], fake,
                         run.HopCache(tmp_path / "hops.jsonl"), "m")
    assert out == {"j": ["Kettu"]}
    assert seen[1] == ("en", "fi", ("fox",))


def test_best_route_picks_highest_one_hop_p1():
    import pandas as pd
    sel = pd.DataFrame({"route": ["a-b", "a-x-b", "a-y-b"], "hops": [0, 1, 1],
                        "p1": [0.9, 0.4, 0.6]})
    assert run.best_route(sel) == "a-y-b"


def test_covered_drops_pairs_with_a_language_nllb_lacks():
    assert run.covered([("yo", "de"), ("mhr", "fi"), ("fi", "myv"), ("or", "en")]) == \
        [("yo", "de"), ("or", "en")]


def test_named_routes_basic_keeps_direct_en_and_hub_only():
    fixed = {"quality": ("ko", "en", "fi"), "svg": ("ko", "ja", "es", "fi")}
    basic = run.named_routes("ko", "fi", "ru", fixed, basic=True)
    assert basic == {"direct": ("ko", "fi"), "via_en": ("ko", "en", "fi"),
                     "via_hub": ("ko", "ru", "fi")}
    full = run.named_routes("ko", "fi", "ru", fixed, basic=False)
    assert full["p4_svg"] == ("ko", "ja", "es", "fi") and len(full) == 5
    assert "via_en" not in run.named_routes("en", "fi", "ru", {}, basic=True)
