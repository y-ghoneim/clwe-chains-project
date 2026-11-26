"""Tests of clwe_chains.muse."""

import pandas as pd

from clwe_chains.muse import Vocab, query_rows, read_muse, train_rows


def test_read_muse_keeps_two_field_lines(tmp_path):
    p = tmp_path / "en-de.txt"
    p.write_text("okay ok\nokay okay\nbad line here\n\nwaves\twelle\n", encoding="utf-8",
                 newline="\n")
    d = read_muse(p)
    assert d.values.tolist() == [["okay", "ok"], ["okay", "okay"], ["waves", "welle"]]


def test_rows_prefer_exact_form_then_most_frequent_casing():
    vs = Vocab(["The", "house", "the", "dog"], "en")
    vt = Vocab(["Haus", "der", "haus", "Hund"], "de")
    assert vs.row("the") == 2                       # exact form, though "The" is more frequent
    assert vt.row("hund") == 3                      # no exact form: its casing
    assert vt.row("katze") is None
    assert vt.casings("haus") == [0, 2]
    d = pd.DataFrame({"src": ["house", "house", "dog", "cat"],
                      "tgt": ["haus", "Haus", "hund", "katze"]})
    assert train_rows(d, vs, vt).values.tolist() == [[1, 2], [1, 0], [3, 3]]
    q, gold, anycase, cov = query_rows(d, vs, vt)
    assert q == [1, 3] and gold == [[0, 2], [3]] and anycase == [[0, 2], [3]]
    assert cov == {"test_words": 3, "test_words_used": 2}
    q, gold, anycase, _ = query_rows(d.iloc[[0]], vs, vt)
    assert gold == [[2]] and anycase == [[0, 2]]    # main gold: the translation's own row only
