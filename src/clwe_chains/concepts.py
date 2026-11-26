"""English concept candidates for the shared list (brief section 5.2).

A concept is a meaning anchored on one English word. Candidates are English fastText words in
frequency order (line number = Common Crawl rank) that are plain lowercase words, WordNet content
words, and present in PanLex or Wiktionary; the first n are kept.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd

_WORD = re.compile(r"^[a-z]{2,}$")
_CONTENT_POS = {"n", "v", "a", "s", "r"}


def english_candidates(vocab: list[str], in_sources: set[str], is_content: Callable[[str], bool],
                       n: int) -> pd.DataFrame:
    """The first n vocab words that match ^[a-z]{2,}$, are in in_sources and pass is_content."""
    words, ranks = [], []
    for rank, w in enumerate(vocab):
        if _WORD.match(w) and w in in_sources and is_content(w):
            words.append(w)
            ranks.append(rank)
            if len(words) == n:
                break
    return pd.DataFrame({
        "concept_id": np.arange(len(words), dtype=np.int32),
        "en": pd.Series(words, dtype=str),
        "en_rank": np.asarray(ranks, dtype=np.int32),
    })


def wordnet_content(word: str) -> bool:
    """True iff word is itself a lemma of a WordNet noun, verb, adjective or adverb synset.

    synsets() maps inflections to their lemma ("children" -> child), so the lemma check is what
    keeps candidates to base forms (brief 5.2).
    """
    from nltk.corpus import wordnet as wn

    return any(s.pos() in _CONTENT_POS and any(lem.name().lower() == word for lem in s.lemmas())
               for s in wn.synsets(word))


def wordnet_lowercase(word: str) -> bool:
    """True iff some WordNet content synset has a lemma spelled exactly `word`.

    wordnet_content lowercases lemma names, so proper nouns and acronyms (Mexico, April, WHO, US)
    pass it although the lowercase candidate has no such content sense. Used by the
    capitalized-senses rule (shared_list.capitalized_senses; decisions.md 2026-11-16).
    """
    from nltk.corpus import wordnet as wn

    return any(s.pos() in _CONTENT_POS and any(lem.name() == word for lem in s.lemmas())
               for s in wn.synsets(word))
