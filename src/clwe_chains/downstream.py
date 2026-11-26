"""Tweet features for the downstream sentiment experiment (experiments/downstream-umsab/README.md).

A tweet becomes the mean of its tokens' normalized fastText vectors, rescaled to unit length. The
text cleaning is fixed in the README: placeholders, the retweet marker and the hash sign are
dropped; a token is a run of letters, combining marks and digits (so Devanagari vowel signs stay
inside their word); words are matched through text.normalize like every other word in the project.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from clwe_chains.text import normalize

UMSAB_FOLDERS = {"en": "english", "ar": "arabic", "de": "german", "es": "spanish",
                 "fr": "french", "hi": "hindi", "it": "italian", "pt": "portuguese"}
SPLITS = ("train", "validation", "test")
AFRISENTI_FOLDERS = {"yo": "yor", "sw": "swa", "am": "amh"}
AFRISENTI_SPLITS = {"train": "train", "validation": "dev", "test": "test"}
LABELS = {"negative": 0, "neutral": 1, "positive": 2}   # UMSAB's numbering

_MENTION = re.compile(r"@\S+")          # @user, and any other handle with the text after it
_URL = re.compile(r"http\S*")           # UMSAB replaces links by "http"
_RETWEET = re.compile(r"(?<!\w)RT(?!\w)")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_JOINERS = {chr(0x200C), chr(0x200D)}   # zero-width non-joiner and joiner
_ZWSP = chr(0x200B)                    # zero-width space: separates words


def tokenize(text: str, lang: str) -> list[str]:
    """Cleaned, normalized tokens of a tweet. Inner apostrophes are kept (fastText has don't)."""
    text = _RETWEET.sub(" ", _URL.sub(" ", _MENTION.sub(" ", text)))
    text = normalize(text, lang).translate(_APOSTROPHES)
    return [t for t in (w.strip("'") for w in "".join(map(_char, text)).split()) if t]


def _char(c: str) -> str:
    """What a character becomes: kept inside words (letters, combining marks, digits, apostrophes
    and the zero-width (non-)joiner, which belongs to some Persian and Indic words), dropped (other
    invisible format marks such as bidi isolates, which would glue onto a word and miss the
    vocabulary), or a separator."""
    cat = unicodedata.category(c)
    if c == "'" or cat[0] in "LMN" or c in _JOINERS:
        return c
    return "" if cat == "Cf" and c != _ZWSP else " "


def script_share(texts: list[str], script: str) -> float:
    """Share of the letters in the texts that belong to a script (Latin, Arabic, Devanagari...)
    by their Unicode names; 0 when there are no letters. Used to check that a language's tweets
    are written in the script its vectors and dictionaries use."""
    letters = [c for t in texts for c in t if unicodedata.category(c)[0] == "L"]
    if not letters:
        return 0.0
    prefix = script.upper() + " "
    return sum(unicodedata.name(c, "").startswith(prefix) for c in letters) / len(letters)


def match_rows(tokens: list[str], index: dict[str, int]) -> list[int]:
    """Vocabulary rows of the tokens. A token missing as a whole is retried in pieces split after
    each apostrophe (l'été -> l' + été, how fastText tokenizes elision); other misses are
    skipped."""
    rows = []
    for tok in tokens:
        if tok in index:
            rows.append(index[tok])
        elif "'" in tok:
            parts = tok.split("'")
            pieces = [p + "'" for p in parts[:-1]] + [parts[-1]]
            rows.extend(index[p] for p in pieces if p in index)
    return rows


def mean_features(texts: list[str], lang: str, index: dict[str, int],
                  emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-length mean vector per text, and how many tokens matched. A text with no matched token
    gets a zero vector."""
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    counts = np.zeros(len(texts), dtype=np.int64)
    for i, text in enumerate(texts):
        rows = match_rows(tokenize(text, lang), index)
        counts[i] = len(rows)
        if rows:
            v = emb[rows].mean(axis=0)
            out[i] = v / max(float(np.linalg.norm(v)), 1e-8)
    return out, counts


def read_afrisenti(root: Path, lang: str, folders: dict[str, str] = AFRISENTI_FOLDERS) -> dict:
    """split -> (texts, integer labels) from <root>/<folder>/{train,dev,test}.tsv (columns tweet,
    label), with UMSAB's label numbers. Read without quote handling: some tweets are wrapped in
    quotes or contain runs of them, and a tab never occurs inside a tweet."""
    out = {}
    for split, name in AFRISENTI_SPLITS.items():
        path = Path(root) / folders[lang] / f"{name}.tsv"
        with open(path, encoding="utf-8", newline="") as fh:
            header, *rows = [line.rstrip("\r\n").split("\t") for line in fh if line.strip()]
        if header != ["tweet", "label"] or any(len(r) != 2 for r in rows):
            raise ValueError(f"{path}: expected two tab-separated columns, tweet and label")
        out[split] = ([r[0] for r in rows], [LABELS[r[1]] for r in rows])
    return out


def read_topic_tsv(folder: Path, splits: dict[str, str] = AFRISENTI_SPLITS,
                   text_columns: tuple[str, ...] = ("text",)) -> dict:
    """split -> (texts, label names) from <folder>/{train,dev,test}.tsv with a "category" column
    (MasakhaNEWS, SIB-200). Several text columns (MasakhaNEWS: headline, text) are joined with a
    newline. Fields may span lines, so the files are read as TSV with quoting."""
    import pandas as pd
    out = {}
    for split, name in splits.items():
        d = pd.read_csv(Path(folder) / f"{name}.tsv", sep="\t", dtype=str, keep_default_na=False)
        texts = d[list(text_columns)].agg("\n".join, axis=1).tolist()
        out[split] = (texts, d["category"].tolist())
    return out


def read_umsab(root: Path, lang: str, folders: dict[str, str] = UMSAB_FOLDERS) -> dict:
    """split -> (texts, integer labels) from <root>/data/<language>/<split>.jsonl."""
    out = {}
    for split in SPLITS:
        path = Path(root) / "data" / folders[lang] / f"{split}.jsonl"
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        out[split] = ([r["text"] for r in rows], [int(r["label"]) for r in rows])
    return out


def write_topic_features(cfg, langs: dict[str, Path], scripts: dict[str, str], out: Path,
                         text_columns: tuple[str, ...], log) -> None:
    """Features of a topic dataset (MasakhaNEWS, SIB-200) -> <out>/<lang>.npz and stats.csv.

    langs maps a project language to its dataset folder. Per split: X_ (texts x 300, float32), n_
    (matched tokens) and labels_ (label names). stats.csv: texts, texts with no matched token,
    mean matched tokens, script share and label counts per language and split."""
    import pandas as pd
    import torch

    from clwe_chains.embeddings import normalize as normalize_vectors
    from clwe_chains.text import load_vocab, vocab_index

    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for lang, folder in langs.items():
        raw = np.load(cfg.path("v0_embeddings") / f"{lang}.npy")
        emb = normalize_vectors(torch.from_numpy(raw), cfg["embeddings"]["normalize"]).numpy()
        index = vocab_index(load_vocab(cfg.path("v0_embeddings") / f"{lang}.vocab.txt"), lang)
        data, arrays = read_topic_tsv(folder, text_columns=text_columns), {}
        for split in SPLITS:
            texts, labels = data[split]
            x, n = mean_features(texts, lang, index, emb)
            arrays.update({f"X_{split}": x, f"n_{split}": n,
                           f"labels_{split}": np.array(labels, dtype=str)})
            counts = pd.Series(labels).value_counts().sort_index()
            stats.append({"lang": lang, "split": split, "texts": len(texts),
                          "no_token": int((n == 0).sum()), "mean_tokens": float(n.mean()),
                          "native_script_share": script_share(texts, scripts[lang]),
                          "labels": "; ".join(f"{k} {v}" for k, v in counts.items())})
            s = stats[-1]
            log.info("%s %s: %d texts, %d empty, %.1f matched tokens each, %.0f%% script; %s",
                     lang, split, s["texts"], s["no_token"], s["mean_tokens"],
                     100 * s["native_script_share"], s["labels"])
        np.savez_compressed(out / f"{lang}.npz", **arrays)
    pd.DataFrame(stats).to_csv(out / "stats.csv", index=False, lineterminator="\n",
                               float_format="%.4f")
