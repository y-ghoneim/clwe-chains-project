import json

import numpy as np
import pytest

from clwe_chains.downstream import (
    match_rows,
    mean_features,
    read_afrisenti,
    read_umsab,
    script_share,
    tokenize,
)


def test_tokenize_drops_placeholders_retweet_marker_and_hash():
    assert tokenize("RT @user: I LOVE #Python http", "en") == ["i", "love", "python"]


def test_tokenize_keeps_other_mentions_out_and_numbers_in():
    assert tokenize("@someone_else 2 apples, 10 pears!", "en") == ["2", "apples", "10", "pears"]


def test_tokenize_keeps_devanagari_vowel_signs():
    # Hindi matras are combining marks: a \w split would cut these words apart.
    assert tokenize("मुझे बहुत पसंद है", "hi") == ["मुझे", "बहुत", "पसंद", "है"]


def test_tokenize_normalizes_like_the_vocabulary():
    # Arabic harakat are stripped by text.normalize, as in vocab_index.
    assert tokenize("مَاء", "ar") == ["ماء"]


def test_tokenize_keeps_inner_apostrophes_straight():
    assert tokenize("Don’t l’été", "fr") == ["don't", "l'été"]

BIDI_FSI, BIDI_PDI = chr(0x2068), chr(0x2069)   # bidi isolate start and end
ZWSP, ZWNJ = chr(0x200B), chr(0x200C)


def test_tokenize_drops_bidi_and_zero_width_format_marks_but_keeps_joiners():
    # bidi isolates and zero-width spaces must not glue onto a word (the token would miss the
    # vocabulary); the zero-width (non-)joiner is part of some Persian / Indic words and stays.
    assert tokenize(f"{BIDI_FSI}الموصل{BIDI_PDI} مرحبا{ZWSP}", "ar") == ["الموصل", "مرحبا"]
    assert tokenize(f"می{ZWNJ}خواهم", "fa") == [f"می{ZWNJ}خواهم"]


def test_tokenize_drops_the_url_placeholder_and_a_lone_at_sign():
    assert tokenize("great game http/URL @ tonight", "en") == ["great", "game", "tonight"]


def test_script_share_counts_letters_of_the_expected_script():
    assert script_share(["abc", "def"], "Latin") == 1.0
    # letters only: a, b, c and the two consonants म, झ (vowel signs are marks, not letters)
    assert script_share(["abc", "मुझे"], "Devanagari") == 2 / 5
    assert script_share([""], "Latin") == 0.0


def test_match_rows_whole_token_first_then_apostrophe_pieces():
    index = {"don't": 0, "l'": 1, "été": 2}
    # fastText keeps don't whole but splits French elision after the apostrophe.
    assert match_rows(["don't", "l'été", "unknown"], index) == [0, 1, 2]


def test_match_rows_skips_out_of_vocabulary():
    assert match_rows(["zzz", "qqq"], {"a": 0}) == []


def test_mean_features_unit_length_and_counts():
    emb = np.array([[1, 0], [0, 1], [3, 4]], dtype=np.float32)
    index = {"a": 0, "b": 1}
    x, n = mean_features(["a b", "a zzz", "zzz"], "en", index, emb)
    assert n.tolist() == [2, 1, 0]
    np.testing.assert_allclose(x[0], [2 ** -0.5, 2 ** -0.5], rtol=1e-6)
    np.testing.assert_allclose(x[1], [1, 0], rtol=1e-6)
    assert not x[2].any()                       # no matched token: zero vector, not NaN
    assert x.dtype == np.float32


def test_read_umsab_returns_texts_and_int_labels(tmp_path):
    d = tmp_path / "data" / "english"
    d.mkdir(parents=True)
    for split, rows in {"train": [("good", "2")], "validation": [("meh", "1")],
                        "test": [("bad", "0"), ("ok", "1")]}.items():
        with open(d / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps({"text": text, "label": label}) + "\n"
                          for text, label in rows)
    out = read_umsab(tmp_path, "en", {"en": "english"})
    assert out["train"] == (["good"], [2])
    assert out["test"] == (["bad", "ok"], [0, 1])


def write_tsv(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_afrisenti_maps_labels_and_keeps_quotes(tmp_path):
    for name in ("train", "dev", "test"):
        write_tsv(tmp_path / "yor" / f"{name}.tsv",
                  ["tweet	label", "'@user ẹ kú iṣẹ́'	positive",
                   'ó dára """"..àyà	negative', "ko si	neutral"])
    d = read_afrisenti(tmp_path, "yo")
    assert set(d) == {"train", "validation", "test"}
    texts, labels = d["validation"]
    assert labels == [2, 0, 1]                                 # UMSAB: 0 neg, 1 neutral, 2 pos
    assert texts[1] == 'ó dára """"..àyà'                       # quotes are text, not quoting
    assert tokenize(texts[0], "yo") == ["ẹ", "kú", "iṣẹ́"]      # wrapping quotes stripped


def test_read_afrisenti_rejects_a_malformed_row(tmp_path):
    for name in ("train", "dev", "test"):
        write_tsv(tmp_path / "swa" / f"{name}.tsv", ["tweet	label", "a	b	positive"])
    with pytest.raises(ValueError):
        read_afrisenti(tmp_path, "sw")


def test_script_share_ethiopic():
    assert script_share(["ሰላም abc"], "Ethiopic") == pytest.approx(3 / 6)


def test_read_topic_tsv_joins_columns_and_keeps_multiline_fields(tmp_path):
    for name in ("train", "dev", "test"):
        (tmp_path / f"{name}.tsv").write_text(
            'category\theadline\ttext\turl\nsports\tGoal!\t"line one\nline two"\tu\n'
            "health\tFlu\tshort\tu\n", encoding="utf-8", newline="\n")
    from clwe_chains.downstream import read_topic_tsv
    d = read_topic_tsv(tmp_path, text_columns=("headline", "text"))
    texts, labels = d["validation"]
    assert labels == ["sports", "health"]
    assert texts == ["Goal!\nline one\nline two", "Flu\nshort"]
