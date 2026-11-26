import pandas as pd
import pytest

from clwe_chains.shared_list import (
    assemble,
    capitalized_senses,
    case_support,
    cognate,
    coverage_table,
)

PER_LANG_COLS = ["concept_id", "word_idx", "word", "source", "round_trip", "identical", "translit"]


def _lang(passed: list[int], failed: list[int] = (), source: str = "panlex") -> pd.DataFrame:
    ids = [*passed, *failed]
    return pd.DataFrame({"concept_id": ids, "word_idx": [100 + i for i in ids],
                         "word": [f"w{i}" for i in ids], "source": source,
                         "round_trip": [True] * len(passed) + [False] * len(failed),
                         "identical": False, "translit": [f"w{i}" for i in ids]})


CONCEPTS = pd.DataFrame({"concept_id": range(4), "en": ["a", "b", "c", "d"]})
NO_DROPS = {"n_multiword": 0, "n_oov": 0}


def test_assemble_intersection():
    per_lang = {"aa": _lang([0, 1, 2]), "bb": _lang([1, 2, 3]), "cc": _lang([0])}
    wide, kept = assemble(per_lang, CONCEPTS, threshold=0.5, top=4)
    assert kept == ["aa", "bb"]
    assert wide["concept_id"].tolist() == [1, 2]
    assert wide["aa"].tolist() == ["w1", "w2"] and wide["bb_source"].tolist() == ["panlex"] * 2
    assert "cc" not in wide.columns


def test_coverage_uses_top():
    per_lang = {"aa": _lang([0, 1], list(range(2, 10)))}
    cov = coverage_table(per_lang, top=2, drops={"aa": NO_DROPS}, n_candidates=10).set_index("lang")
    assert cov.loc["aa", "coverage"] == pytest.approx(1.0)
    assert cov.loc["aa", "coverage_all"] == pytest.approx(0.2)


def test_assemble_report_only():
    per_lang = {"aa": _lang([0]), "bb": _lang([1])}
    wide, kept = assemble(per_lang, CONCEPTS, threshold=None, top=4)
    assert wide.empty and kept == ["aa", "bb"]


def test_failed_round_trip_excluded():
    per_lang = {"aa": _lang([0], [1, 2, 3])}
    cov = coverage_table(per_lang, top=4, drops={"aa": NO_DROPS}, n_candidates=4).set_index("lang")
    assert cov.loc["aa", "chosen"] == 4 and cov.loc["aa", "passed_round_trip"] == 1
    assert cov.loc["aa", "coverage"] == pytest.approx(0.25)


def test_zero_coverage_language():
    empty = pd.DataFrame({c: pd.Series([], dtype=object) for c in PER_LANG_COLS})
    per_lang = {"aa": _lang([0, 1, 2, 3]), "zz": empty}
    drops = {"aa": NO_DROPS, "zz": {"n_multiword": 3, "n_oov": 5}}
    cov = coverage_table(per_lang, top=4, drops=drops, n_candidates=4).set_index("lang")
    assert cov.loc["zz", "coverage"] == 0.0 and cov.loc["zz", "n_oov"] == 5
    wide, kept = assemble(per_lang, CONCEPTS, threshold=0.5, top=4)
    assert kept == ["aa"] and len(wide) == 4


def test_cognate():
    assert cognate("informacion", "information", 0.8) is True  # similarity 0.909
    assert cognate("voda", "water", 0.8) is False


def test_list_languages_from_wide():
    from clwe_chains.shared_list import list_languages

    wide = pd.DataFrame(columns=["concept_id", "en", "de", "de_source", "ar", "ar_source"])
    assert list_languages(wide) == ["ar", "de", "en"]


def test_case_support_counts_lowercase_and_capitalized_only_headwords():
    wide = pd.DataFrame({"concept_id": [0, 1], "en": ["may", "water"],
                         "de": ["Mai", "Wasser"], "fr": ["mai", "eau"], "hr": ["svibanj", "voda"]})
    wk = pd.DataFrame({"en": ["May", "May", "may", "water", "Water"],
                       "code": ["de", "fr", "hr", "de", "sh"],
                       "word": ["Mai", "mai", "svibanj", "Wasser", "voda"]})
    out = case_support(wide, wk, {"de": ["de"], "fr": ["fr"], "hr": ["hr", "sh"]})
    assert out.set_index("concept_id")[["n_lower", "n_cap_only"]].to_dict("index") == {
        0: {"n_lower": 1, "n_cap_only": 2},   # de, fr only under "May"; hr under "may"
        1: {"n_lower": 1, "n_cap_only": 1},   # fr "eau" has no Wiktionary support at all
    }


def test_capitalized_senses_rule():
    support = pd.DataFrame({"concept_id": [0, 1, 2, 3], "n_lower": [0, 5, 3, 10],
                            "n_cap_only": [20, 5, 4, 0]})
    wordnet_lower = {0: True, 1: True, 2: True, 3: False}
    # 0: majority capitalized; 1: a tie is kept; 2: majority by one; 3: no lowercase lemma
    assert capitalized_senses(support, wordnet_lower) == [0, 2, 3]
