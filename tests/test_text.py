import pandas as pd
import pytest

from clwe_chains.text import identical_pairs, load_vocab, normalize, variant_rows, vocab_index

# Combining marks are written as escapes so the test data can't be silently recomposed.
CASES = [
    ("ru", "вода́", "вода"),          # stress accent stripped
    ("ru", "ёж", "ёж"),                    # ё is a letter: kept
    ("ru", "йод", "йод"),                  # й is a letter: kept
    ("hr", "vòda", "voda"),           # tone mark on a vowel stripped
    ("hr", "kuća", "kuća"),                # ć = c + U+0301, base c: kept
    ("hr", "ćevapi", "ćevapi"),
    ("hr", "čovjȅk", "čovjek"),       # ȅ = e + U+030F stripped; č kept
    ("hr", "Žena", "žena"),
    ("ar", "مَاء", "ماء"),            # fatha stripped
    ("ur", "پانی", "پانی"),
    ("he", "מַיִם", "מים"),       # patah and hiriq stripped
    ("es", "Niño", "niño"),
    ("de", "Über", "über"),
    ("vi", "nước", "nước"),                # Vietnamese tone letters kept
    ("en", "ﬁsh", "fish"),            # NFKC: the fi ligature
]


@pytest.mark.parametrize("lang,raw,want", CASES)
def test_normalize(lang, raw, want):
    assert normalize(raw, lang) == want


def test_vocab_index_keeps_most_frequent_form():
    assert vocab_index(["The", "the", "вода"], "ru") == {"the": 0, "вода": 2}


def test_load_vocab_crlf(tmp_path):
    (tmp_path / "v.txt").write_bytes(b",\r\nthe\r\n")
    assert load_vocab(tmp_path / "v.txt") == [",", "the"]


def test_variant_rows_groups_casings_under_the_most_frequent_row():
    words = ["Halk", "proje", "halk", "Proje", "HALK", "su"]
    assert variant_rows(words, "tr") == {0: [0, 2, 4], 1: [1, 3]}  # "su" has one form: omitted


def test_identical_pairs_compare_source_with_target():
    words_s = pd.Series({1: "hotel", 2: "kitap", 3: "Radio"})
    words_t = pd.Series({1: "Hotel", 2: "buku", 3: "radio", 4: "x"})
    out = identical_pairs(words_s, words_t, "hu", "id")
    assert out.to_dict() == {1: True, 2: False, 3: True}  # concept 4 has no source word
