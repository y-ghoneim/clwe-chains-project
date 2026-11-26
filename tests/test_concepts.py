import pytest

from clwe_chains.concepts import english_candidates, wordnet_content, wordnet_lowercase
from clwe_chains.config import load_config

VOCAB = ["the", "Paris", "water", "x", "dog", "run2", "cat"]
NLTK_DATA = load_config().path("nltk_data")


def test_filters_and_order():
    out = english_candidates(VOCAB, set(VOCAB), lambda w: w != "the", n=2)
    assert out["en"].tolist() == ["water", "dog"]
    assert out["en_rank"].tolist() == [2, 4]
    assert out["concept_id"].tolist() == [0, 1]
    assert str(out["concept_id"].dtype) == "int32" and str(out["en_rank"].dtype) == "int32"


def test_requires_source():
    out = english_candidates(VOCAB, set(VOCAB) - {"dog"}, lambda w: w != "the", n=2)
    assert out["en"].tolist() == ["water", "cat"]


@pytest.mark.skipif(not (NLTK_DATA / "corpora" / "wordnet.zip").exists(),
                    reason="WordNet data not found")
def test_wordnet_content(monkeypatch):
    import nltk

    monkeypatch.setattr(nltk.data, "path", [str(NLTK_DATA), *nltk.data.path])
    assert wordnet_content("water") is True
    assert wordnet_content("the") is False


@pytest.mark.skipif(not (NLTK_DATA / "corpora" / "wordnet.zip").exists(),
                    reason="WordNet data not found")
def test_wordnet_content_requires_base_form(monkeypatch):
    """Brief 5.2: a concept is anchored on the English *base form*. WordNet's synsets() maps
    inflections to their lemma, so the word itself must be one of the synset's lemmas."""
    import nltk

    monkeypatch.setattr(nltk.data, "path", [str(NLTK_DATA), *nltk.data.path])
    assert wordnet_content("child") is True and wordnet_content("go") is True
    for inflected in ("children", "was", "went", "books", "started"):
        assert wordnet_content(inflected) is False, inflected


@pytest.mark.skipif(not (NLTK_DATA / "corpora" / "wordnet.zip").exists(),
                    reason="WordNet data not found")
def test_wordnet_lowercase_rejects_capitalized_only_lemmas(monkeypatch):
    """Proper nouns and acronyms (Mexico, WHO) pass wordnet_content, which lowercases lemma
    names; wordnet_lowercase needs a content lemma spelled exactly as the word."""
    import nltk

    monkeypatch.setattr(nltk.data, "path", [str(NLTK_DATA), *nltk.data.path])
    for word in ("water", "may", "china"):  # may: the hawthorn; china: porcelain
        assert wordnet_lowercase(word) is True, word
    for word in ("mexico", "april", "who"):
        assert wordnet_content(word) is True and wordnet_lowercase(word) is False, word
