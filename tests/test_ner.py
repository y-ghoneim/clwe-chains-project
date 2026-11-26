import itertools

import numpy as np
import pytest
import torch
from torchcrf import CRF

from clwe_chains import ner
from clwe_chains.text import normalize

O, BP, IP, BO, IO, BL, IL = range(7)


def test_entities_conll_rule():
    assert ner.entities([BP, IP, O, BL]) == {(0, 2, "PER"), (3, 4, "LOC")}
    assert ner.entities([IP, IP]) == {(0, 2, "PER")}                  # I- after O starts one
    assert ner.entities([BP, IL]) == {(0, 1, "PER"), (1, 2, "LOC")}   # type change starts one
    assert ner.entities([BP, BP]) == {(0, 1, "PER"), (1, 2, "PER")}
    assert ner.entities([BO, IO, IO]) == {(0, 3, "ORG")}               # entity at the end
    assert ner.entities([O, O]) == set()


def test_sentence_counts_and_f1_need_exact_span_and_type():
    gold = [[BP, IP, O, BL], [BO]]
    pred = [[BP, O, O, BL], [BL]]          # PER span too short; ORG tagged as LOC
    c = ner.sentence_counts(gold, pred)
    assert c.tolist() == [[1, 1, 1], [0, 1, 1]]
    assert ner.f1_from_counts(c) == pytest.approx(2 * 1 / (2 * 1 + 2 + 2))
    assert ner.f1_from_counts(np.zeros((3, 3))) == 0.0


def test_paired_entity_bootstrap_zero_for_identical_and_sign_otherwise():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=(3, 50, 3))
    d, lo, hi = ner.paired_entity_bootstrap(a, a, n=200)
    assert (d, lo, hi) == (0.0, 0.0, 0.0)
    better = a.copy()
    better[:, :, 0] += 2                                              # more true positives
    d, lo, hi = ner.paired_entity_bootstrap(better, a, n=200)
    assert d > 0 and lo > 0
    d1, _, _ = ner.paired_entity_bootstrap(better[0], a[0], n=10)    # one model, 2-d input
    assert d1 > 0


def test_match_tokens_exact_form_first_then_normalized():
    exact = {"Berlin": 3, "berlin": 7, "Haus": 5}
    normalized = {"berlin": 7, "haus": 5, "straße": 9}
    rows = ner.match_tokens(["Berlin", "BERLIN", "haus", "Straße", "xyz"], exact, normalized,
                            "de", normalize)
    assert rows.tolist() == [3, 7, 5, 9, -1]


def test_token_vectors_centred_mapped_and_zero_when_unmatched():
    emb = np.eye(4, dtype=np.float32)
    mu = ner.token_mean(emb, np.array([0, 1, -1]))
    assert mu.tolist() == [0.5, 0.5, 0, 0]
    w = np.roll(np.eye(4, dtype=np.float32), 1, axis=1)               # a permutation
    v = ner.token_vectors(emb, np.array([0, -1]), mu, w)
    assert v[0].tolist() == pytest.approx((np.array([0.5, -0.5, 0, 0]) @ w).tolist())
    assert not v[1].any()


def test_window_features_zero_at_sentence_edges():
    v = np.arange(1, 6, dtype=np.float32)[:, None]                    # 5 tokens, d = 1
    c = ner.Corpus(v, np.array([0, 2, 5]), np.zeros(5, int))          # sentences [0,2) [2,5)
    f = ner.window_features(c)
    assert f.tolist() == [[0, 1, 2], [1, 2, 0], [0, 3, 4], [3, 4, 5], [4, 5, 0]]


def test_char_ids_unknown_padding_and_truncation():
    vocab = ner.char_vocab(["ab"])
    ids, lens = ner.char_ids(["ba", "az", "x" * 30, ""], vocab)
    assert ids[0, :2].tolist() == [vocab["b"], vocab["a"]] and lens[0] == 2
    assert ids[1, 1] == ner.UNK_CHAR
    assert lens[2] == ner.MAX_WORD_CHARS
    assert lens[3] == 1 and ids[3, 0] == ner.UNK_CHAR


def test_crf_package_matches_brute_force_on_tiny_sentences():
    """torchcrf's likelihood and Viterbi against enumerating every tag sequence."""
    torch.manual_seed(0)
    k, n = 3, 4
    crf = CRF(k, batch_first=True)
    em = torch.randn(1, n, k)
    mask = torch.ones(1, n, dtype=torch.bool)

    def score(seq):
        s = crf.start_transitions[seq[0]] + em[0, 0, seq[0]] + crf.end_transitions[seq[-1]]
        for i in range(1, n):
            s = s + crf.transitions[seq[i - 1], seq[i]] + em[0, i, seq[i]]
        return s

    seqs = list(itertools.product(range(k), repeat=n))
    scores = torch.stack([score(s) for s in seqs])
    log_z = torch.logsumexp(scores, 0)
    tags = torch.tensor([[2, 0, 1, 1]])
    assert crf(em, tags, mask).item() == pytest.approx((score((2, 0, 1, 1)) - log_z).item(),
                                                       abs=1e-5)
    assert tuple(crf.decode(em, mask)[0]) == seqs[int(scores.argmax())]


def toy_corpus(rng, n_sent, centres):
    """Sentences whose tag is readable from the token vector (a separable toy task)."""
    lens = rng.integers(2, 6, n_sent)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    tags = np.zeros(offsets[-1], dtype=np.int64)
    for a, b in itertools.pairwise(offsets):
        tags[a:b] = rng.choice([O, BP, BL], b - a)
        tags[a + 1:b][tags[a:b - 1] == BP] = IP if rng.random() < 0.5 else BP
    v = centres[tags] * 3 + rng.normal(size=(len(tags), centres.shape[1]))
    return ner.Corpus(v.astype(np.float32), offsets, tags)


@pytest.mark.parametrize("chars", [False, True])
def test_fit_tagger_learns_a_separable_toy_task(chars):
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(7, 16))
    train, val = toy_corpus(rng, 300, centres), toy_corpus(rng, 80, centres)
    n_chars = 0
    if chars:
        vocab = ner.char_vocab(["ab", "cd"])
        n_chars = len(vocab) + 2
        for c in (train, val):
            c.chars, c.char_lens = ner.char_ids(["ab"] * len(c.tags), vocab)
    model, info = ner.fit_tagger(train, val, seed=0, device="cpu", n_chars=n_chars)
    assert info["best_val_f1"] > 0.9
    pred = ner.predict_tagger(model, val, "cpu")
    assert [len(p) for p in pred] == [len(t) for t in val.sentence_tags()]


def test_fit_window_lr_learns_the_toy_task():
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(7, 16))
    train, val = toy_corpus(rng, 300, centres), toy_corpus(rng, 80, centres)
    model, c, f1 = ner.fit_window_lr(train, val)
    assert c in ner.LR_GRID and f1 > 0.8
    pred = ner.split_tags(model.predict(ner.window_features(val)), val.offsets)
    assert ner.f1_from_counts(ner.sentence_counts(val.sentence_tags(), pred)) == f1
