"""Named-entity recognition for the downstream experiment (experiments/downstream-wikiann/).

A tagger is trained on one language's WikiANN sentences in its own vector space; another language's
sentences are moved into that space by an alignment map and tagged. Inputs are the project's
normalized fastText vectors, centred on the language's mean over its own training tokens (no
labels used); a token with no vocabulary match is a zero vector, before and after mapping.

Taggers (settings fixed in the README before any run):
- BiLSTM-CRF on the frozen word vectors (headline), optionally with a character BiLSTM per word
  (secondary A); early stopping on the training language's validation entity F1;
- window logistic regression (secondary B): a token's vector with its two neighbours'.
Scoring is entity-level F1, micro over entities, exact span and type, with the CoNLL reading of
the tags (an I- tag after O or after another type starts a new entity).
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

from clwe_chains.transfer import LR_GRID, Classifier

TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]    # WikiANN's ids
N_TAGS = len(TAGS)
LSTM_HIDDEN = 256
DROPOUT = 0.5
LEARNING_RATE = 1e-3
BATCH = 32
MAX_EPOCHS = 30
PATIENCE = 5
CHAR_DIM = 30
CHAR_HIDDEN = 50
MAX_WORD_CHARS = 20          # longer words keep their first 20 characters
PAD_CHAR, UNK_CHAR = 0, 1


# ---------------------------------------------------------------------------------------------
# Entities and scores

def entities(tags) -> set[tuple[int, int, str]]:
    """(start, end, type) spans of a tag sequence, end exclusive, by the CoNLL rule."""
    out, start, kind = set(), None, None
    for i, t in enumerate(list(tags) + [0]):
        name = TAGS[int(t)]
        if name.startswith("I-") and name[2:] == kind:
            continue                                    # the current entity goes on
        if start is not None:
            out.add((start, i, kind))
        start, kind = (None, None) if name == "O" else (i, name[2:])
    return out


def sentence_counts(gold_seqs, pred_seqs) -> np.ndarray:
    """(sentences, 3) array of true positives, false positives and false negatives."""
    out = np.zeros((len(gold_seqs), 3), dtype=np.int64)
    for i, (g, p) in enumerate(zip(gold_seqs, pred_seqs, strict=True)):
        eg, ep = entities(g), entities(p)
        tp = len(eg & ep)
        out[i] = tp, len(ep) - tp, len(eg) - tp
    return out


def f1_from_counts(counts: np.ndarray) -> float:
    """Micro entity F1 from summed counts; 0 when there are no gold and no predicted entities."""
    tp, fp, fn = np.asarray(counts).reshape(-1, 3).sum(axis=0)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def paired_entity_bootstrap(counts_a: np.ndarray, counts_b: np.ndarray, n: int = 1000,
                            seed: int = 0) -> tuple[float, float, float]:
    """(F1(a) - F1(b), 2.5th and 97.5th percentiles over resampled test sentences).

    counts_a and counts_b are (models, sentences, 3): the same sentences are drawn for both in each
    resample, and F1 is averaged over the models (seeds) within it."""
    a, b = np.asarray(counts_a), np.asarray(counts_b)
    if a.ndim == 2:                                     # one model
        a, b = a[None], b[None]

    def diff(idx: np.ndarray) -> float:
        return (np.mean([f1_from_counts(m[idx]) for m in a])
                - np.mean([f1_from_counts(m[idx]) for m in b]))

    full = diff(np.arange(a.shape[1]))
    rng = np.random.default_rng(seed)
    d = np.array([diff(rng.integers(0, a.shape[1], a.shape[1])) for _ in range(n)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(full), float(lo), float(hi)


# ---------------------------------------------------------------------------------------------
# Inputs

def match_tokens(tokens, exact: dict[str, int], normalized: dict[str, int], lang: str,
                 normalize) -> np.ndarray:
    """Vocabulary row of each token: its exact surface form, else its normalized form, else -1."""
    out = np.full(len(tokens), -1, dtype=np.int64)
    for i, tok in enumerate(tokens):
        if tok in exact:
            out[i] = exact[tok]
        else:
            out[i] = normalized.get(normalize(tok, lang), -1)
    return out


def token_mean(emb: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Mean vector of a language's matched training tokens (token frequency weighted)."""
    rows = np.asarray(rows)
    return np.asarray(emb, dtype=np.float32)[rows[rows >= 0]].mean(axis=0)


def token_vectors(emb: np.ndarray, rows: np.ndarray, mu: np.ndarray,
                  w: np.ndarray | None = None) -> np.ndarray:
    """(tokens, d): centred vectors of matched tokens, optionally mapped by W (rows times W);
    unmatched tokens are zero."""
    rows = np.asarray(rows)
    out = np.zeros((len(rows), emb.shape[1]), dtype=np.float32)
    hit = rows >= 0
    out[hit] = np.asarray(emb, dtype=np.float32)[rows[hit]] - np.asarray(mu, dtype=np.float32)
    return out if w is None else out @ np.asarray(w, dtype=np.float32)


def char_vocab(words) -> dict[str, int]:
    """Character -> id over a language's training tokens (0 padding, 1 unknown)."""
    chars = sorted({c for w in words for c in w})
    return {c: i + 2 for i, c in enumerate(chars)}


def char_ids(words, vocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """(tokens, MAX_WORD_CHARS) character ids and each token's length (at least 1)."""
    ids = np.zeros((len(words), MAX_WORD_CHARS), dtype=np.int64)
    lens = np.ones(len(words), dtype=np.int64)
    for i, w in enumerate(words):
        cs = [vocab.get(c, UNK_CHAR) for c in w[:MAX_WORD_CHARS]] or [UNK_CHAR]
        ids[i, :len(cs)] = cs
        lens[i] = len(cs)
    return ids, lens


@dataclass
class Corpus:
    """Sentences as one token table: vectors (tokens, d), sentence offsets (sentences + 1), tags
    (tokens), and optionally character ids and lengths."""
    vectors: np.ndarray
    offsets: np.ndarray
    tags: np.ndarray
    chars: np.ndarray | None = None
    char_lens: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def sentence_tags(self) -> list[np.ndarray]:
        return [self.tags[a:b] for a, b in itertools.pairwise(self.offsets)]


def window_features(c: Corpus) -> np.ndarray:
    """(tokens, 3d): left neighbour, token, right neighbour; zero at sentence edges."""
    v = np.asarray(c.vectors, dtype=np.float32)
    left, right = np.zeros_like(v), np.zeros_like(v)
    left[1:], right[:-1] = v[:-1], v[1:]
    starts, ends = c.offsets[:-1], c.offsets[1:] - 1
    left[starts], right[ends] = 0, 0
    return np.concatenate([left, v, right], axis=1)


# ---------------------------------------------------------------------------------------------
# BiLSTM-CRF

class BiLSTMCRF(nn.Module):
    def __init__(self, dim: int, n_chars: int = 0):
        super().__init__()
        self.n_chars = n_chars
        if n_chars:
            self.char_emb = nn.Embedding(n_chars, CHAR_DIM, padding_idx=PAD_CHAR)
            self.char_lstm = nn.LSTM(CHAR_DIM, CHAR_HIDDEN, batch_first=True, bidirectional=True)
            dim += 2 * CHAR_HIDDEN
        self.drop = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(dim, LSTM_HIDDEN, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * LSTM_HIDDEN, N_TAGS)
        self.crf = CRF(N_TAGS, batch_first=True)

    def _chars(self, ids: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """(batch, length, W) character ids -> (batch, length, 2 * CHAR_HIDDEN)."""
        b, n, w = ids.shape
        packed = pack_padded_sequence(self.char_emb(ids.view(b * n, w)), lens.view(-1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h, _) = self.char_lstm(packed)
        return torch.cat([h[0], h[1]], dim=1).view(b, n, -1)

    def emissions(self, x: torch.Tensor, lengths: torch.Tensor, chars=None) -> torch.Tensor:
        if self.n_chars:
            x = torch.cat([x, self._chars(*chars)], dim=2)
        packed = pack_padded_sequence(self.drop(x), lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        return self.out(self.drop(h))


def _batch(c: Corpus, idx: np.ndarray, device) -> tuple:
    """Padded tensors of sentences idx: vectors, tags, mask, lengths, (chars, char lengths)."""
    spans = [(int(c.offsets[i]), int(c.offsets[i + 1])) for i in idx]
    lengths = np.array([b - a for a, b in spans])
    n = int(lengths.max())
    x = np.zeros((len(idx), n, c.vectors.shape[1]), dtype=np.float32)
    y = np.zeros((len(idx), n), dtype=np.int64)
    ch = None
    if c.chars is not None:
        ids = np.zeros((len(idx), n, MAX_WORD_CHARS), dtype=np.int64)
        lens = np.ones((len(idx), n), dtype=np.int64)
    for j, (a, b) in enumerate(spans):
        x[j, :b - a], y[j, :b - a] = c.vectors[a:b], c.tags[a:b]
        if c.chars is not None:
            ids[j, :b - a], lens[j, :b - a] = c.chars[a:b], c.char_lens[a:b]
    if c.chars is not None:
        ch = (torch.from_numpy(ids).to(device), torch.from_numpy(lens).to(device))
    mask = torch.from_numpy(np.arange(n)[None, :] < lengths[:, None]).to(device)
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device), mask,
            torch.from_numpy(lengths), ch)


@torch.no_grad()
def predict_tagger(model: BiLSTMCRF, c: Corpus, device, batch: int = 256) -> list[np.ndarray]:
    """Viterbi tag sequence of every sentence, in order."""
    model.eval()
    out = []
    for s in range(0, len(c), batch):
        x, _, mask, lengths, ch = _batch(c, np.arange(s, min(s + batch, len(c))), device)
        out += [np.array(p) for p in model.crf.decode(model.emissions(x, lengths, ch), mask)]
    return out


def fit_tagger(train: Corpus, val: Corpus, seed: int, device, n_chars: int = 0,
               log=None) -> tuple[BiLSTMCRF, dict]:
    """Train with Adam on the CRF's negative log-likelihood; keep the epoch with the best
    validation entity F1, stopping after PATIENCE epochs without improvement."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BiLSTMCRF(train.vectors.shape[1], n_chars).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gold = val.sentence_tags()
    best, best_state, since, history = -1.0, None, 0, []
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(train))
        for s in range(0, len(order), BATCH):
            x, y, mask, lengths, ch = _batch(train, order[s:s + BATCH], device)
            loss = -model.crf(model.emissions(x, lengths, ch), y, mask, reduction="mean")
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = f1_from_counts(sentence_counts(gold, predict_tagger(model, val, device)))
        history.append(f1)
        if log:
            log(f"epoch {epoch + 1}: validation entity F1 {f1:.4f}")
        if f1 > best:
            best, best_state, since = f1, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_f1": best, "epochs": len(history), "val_f1": history}


# ---------------------------------------------------------------------------------------------
# Window logistic regression

def fit_window_lr(train: Corpus, val: Corpus) -> tuple[Classifier, float, float]:
    """Logistic regression on window features, C from LR_GRID by validation entity F1 (ties: the
    smaller C). Returns the classifier, C and its validation F1."""
    xt, xv = window_features(train), window_features(val)
    scale = float(np.sqrt(train.vectors.shape[1]))
    gold, best = val.sentence_tags(), None
    for c in LR_GRID:
        model = Classifier(LogisticRegression(C=c, max_iter=1000).fit(xt * scale, train.tags),
                           scale)
        f1 = f1_from_counts(sentence_counts(gold, split_tags(model.predict(xv), val.offsets)))
        if best is None or f1 > best[0]:
            best = (f1, model, c)
    return best[1], best[2], best[0]


def split_tags(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """A flat per-token tag array cut into sentences."""
    return [np.asarray(flat[a:b]) for a, b in itertools.pairwise(offsets)]
