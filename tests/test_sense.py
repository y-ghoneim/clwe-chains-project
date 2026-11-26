import math

import pandas as pd

from clwe_chains.sense import sense_coherence, wiktionary_glosses

CONCEPTS = pd.DataFrame({"concept_id": [0, 1], "en": ["crane", "water"]})


def _g(rows):
    return pd.DataFrame(rows, columns=["concept_id", "word_idx", "sense"])


def test_wiktionary_glosses_maps_and_skips_empty():
    wk = pd.DataFrame({"en": ["crane", "crane", "Water", "crane"],
                       "sense": ["bird", "machine", "liquid", ""],
                       "code": ["de", "de", "de", "de"],
                       "word": ["Kranich", "Kran", "Wasser", "Kran"]})
    index = {"kranich": 5, "kran": 6, "wasser": 7}
    out = wiktionary_glosses(wk, ["de"], CONCEPTS, index, "de")
    assert sorted(map(tuple, out.to_numpy())) == [(0, 5, "bird"), (0, 6, "machine"),
                                                  (1, 7, "liquid")]


def test_sense_coherence_counts_pairs():
    # concept 0: aa and bb share "bird", cc is "machine"; concept 1: only aa has a gloss.
    words = pd.DataFrame({"concept_id": [0, 1], "aa": [1, 4], "bb": [2, 5], "cc": [3, 6]})
    glosses = {"aa": _g([(0, 1, "bird"), (1, 4, "liquid")]), "bb": _g([(0, 2, "bird")]),
               "cc": _g([(0, 3, "machine")])}
    out = sense_coherence(words, glosses).set_index("concept_id")
    assert out.loc[0, "n_evaluable"] == 3 and out.loc[0, "n_coherent"] == 1
    assert math.isclose(out.loc[0, "coherence"], 1 / 3)
    assert out.loc[1, "n_evaluable"] == 0 and math.isnan(out.loc[1, "coherence"])


def test_sense_coherence_any_shared_gloss_counts():
    words = pd.DataFrame({"concept_id": [0], "aa": [1], "bb": [2]})
    glosses = {"aa": _g([(0, 1, "bird"), (0, 1, "machine")]), "bb": _g([(0, 2, "machine")])}
    out = sense_coherence(words, glosses)
    assert out["n_coherent"].tolist() == [1] and out["coherence"].tolist() == [1.0]
