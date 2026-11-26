import numpy as np
import pytest
import torch

from clwe_chains.evaluate import (
    bli,
    bli_multi,
    bli_with_combination,
    bli_with_combinations,
    metrics,
)

torch.manual_seed(0)


def _orth(d):
    q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    return q


def _setup(n=100, d=16):
    src = torch.nn.functional.normalize(torch.randn(n, d, dtype=torch.float64), dim=1)
    r = _orth(d)
    return src, src @ r, r


def test_bli_ranks_and_metrics():
    src, tgt, r = _setup()
    ranks = bli(src, tgt, r, np.arange(10), [[i] for i in range(10)], k=5, block=4)
    assert (ranks == 1).all()
    wrong = bli(src, tgt, r, np.array([0]), [[1]], k=5)
    assert wrong[0] > 1
    m = metrics(np.array([1, 2, 10, 1]), identical=np.array([False, False, False, True]))
    assert m["p1"] == 0.5 and m["p5"] == 0.75 and abs(m["mrr"] - (1 + 0.5 + 0.1 + 1) / 4) < 1e-12
    assert m["n_no_identical"] == 3 and abs(m["p1_no_identical"] - 1 / 3) < 1e-12


def test_nn_vs_csls_both_run():
    src, tgt, r = _setup()
    ranks = bli(src, tgt, r, np.arange(10), [[i] for i in range(10)], k=5, retrieval="nn")
    assert (ranks == 1).all()


def test_candidates_restrict_retrieval():
    src, tgt, r = _setup()
    # Query 50 maps exactly onto target 50; its gold is row 3, which only wins without row 50.
    candidates = np.arange(10)
    with_c = bli(src, tgt, r, np.array([50]), [[3]], k=5, retrieval="nn", candidates=candidates)
    without = bli(src, tgt, r, np.array([50]), [[3]], k=5, retrieval="nn")
    assert without[0] > 1
    assert with_c[0] < without[0]
    # Within the candidates, the best-ranked gold among rows 0-9 is what is ranked.
    nearest = int((src[50] @ r @ tgt[:10].T).argmax())
    assert bli(src, tgt, r, np.array([50]), [[nearest]], k=5, retrieval="nn",
               candidates=candidates)[0] == 1


def test_candidates_require_gold_inside():
    src, tgt, r = _setup()
    with pytest.raises(ValueError):
        bli(src, tgt, r, np.array([0]), [[50]], k=5, candidates=np.arange(10))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_half_matches_full():
    g = torch.Generator().manual_seed(1)
    d, n = 64, 2000
    src = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=1).cuda()
    w = _orth(d).float().cuda()
    tgt = torch.nn.functional.normalize(src @ w + 0.3 * torch.randn(n, d, generator=g).cuda(),
                                        dim=1)
    q, gold = np.arange(300), [[i] for i in range(300)]
    full = bli(src, tgt, w, q, gold, k=10, block=256, half=False)
    half = bli(src, tgt, w, q, gold, k=10, block=256, half=True)
    assert (full == half).all()


def test_half_matches_full_on_cpu_with_near_ties():
    """The same check without a GPU, on data with near-tied neighbours: every target has a
    near-duplicate (noise 1e-3), the case where fp16 scores alone shift ranks."""
    g = torch.Generator().manual_seed(2)
    d, n = 32, 600
    base = torch.nn.functional.normalize(torch.randn(n // 2, d, generator=g), dim=1)
    twins = base + 1e-3 * torch.randn(n // 2, d, generator=g)
    tgt = torch.nn.functional.normalize(torch.cat([base, twins]), dim=1)
    w = _orth(d).float()
    src = torch.nn.functional.normalize(tgt @ w.T + 0.2 * torch.randn(n, d, generator=g), dim=1)
    q, gold = np.arange(200), [[i] for i in range(200)]
    full = bli(src, tgt, w, q, gold, k=10, block=64, half=False)
    half = bli(src, tgt, w, q, gold, k=10, block=64, half=True)
    assert (full == half).all() and (full > 1).any()


def _csls_reference(src, tgt, w, queries, gold, k):
    """Brute-force CSLS ranks in float64 numpy, for checking bli."""
    def unit(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    m, t = unit(src.numpy() @ w.numpy()), unit(tgt.numpy())
    sim = m @ t.T
    r_t = np.sort(sim, axis=1)[:, -k:].mean(1)      # each mapped source word's k nearest targets
    r_s = np.sort(sim.T, axis=1)[:, -k:].mean(1)    # each target word's k nearest mapped sources
    csls = 2 * sim - r_t[:, None] - r_s[None, :]
    return np.array([int((csls[q] > csls[q, g].max()).sum()) + 1 for q, g in zip(queries, gold)])


def test_csls_matches_reference():
    src, _, r = _setup(n=80)
    tgt = torch.nn.functional.normalize(src @ r + 0.3 * torch.randn_like(src), dim=1)
    queries = np.arange(30)
    gold = [[i, (i + 7) % 80] for i in range(30)]
    ranks = bli(src, tgt, r, queries, gold, k=5, block=8)
    assert (ranks == _csls_reference(src, tgt, r, queries, gold, k=5)).all()
    assert ranks.max() > 1  # the noise makes some queries miss


def test_bli_multi_matches_separate_calls():
    src, _, r = _setup(n=80)
    tgt = torch.nn.functional.normalize(src @ r + 0.3 * torch.randn_like(src), dim=1)
    queries = np.arange(20)
    full = [[i, (i + 3) % 80] for i in range(20)]
    single = [[i] for i in range(20)]
    for retrieval in ("csls", "nn"):
        both = bli_multi(src, tgt, r, queries, [full, single], k=5, block=8, retrieval=retrieval)
        assert both.shape == (2, 20)
        assert (both[0] == bli(src, tgt, r, queries, full, k=5, retrieval=retrieval)).all()
        assert (both[1] == bli(src, tgt, r, queries, single, k=5, retrieval=retrieval)).all()
        assert (both[0] <= both[1]).all()  # more answers can only help


def _combined_reference(src, tgt, w_a, w_b, queries, gold, k):
    """Brute-force ranks of the equal-weight average of two maps' CSLS scores (float64 numpy)."""
    def unit(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    t = unit(tgt.numpy())
    total = 0
    for w in (w_a, w_b):
        sim = unit(src.numpy() @ w.numpy()) @ t.T
        r_t = np.sort(sim, axis=1)[:, -k:].mean(1)
        r_s = np.sort(sim.T, axis=1)[:, -k:].mean(1)
        total = total + (2 * sim - r_t[:, None] - r_s[None, :]) / 2
    return np.array([int((total[q] > total[q, g].max()).sum()) + 1 for q, g in zip(queries, gold)])


def test_bli_with_combination_matches_separate_calls_and_reference():
    src, _, r = _setup(n=80)
    tgt = torch.nn.functional.normalize(src @ r + 0.3 * torch.randn_like(src), dim=1)
    q = np.arange(25)
    gold, single = [[i, (i + 3) % 80] for i in range(25)], [[i] for i in range(25)]
    w2 = _orth(16)
    out = bli_with_combination(src, tgt, r, w2, q, [gold, single], k=5, block=8)
    assert out["a"].shape == out["b"].shape == out["combo"].shape == (2, 25)
    assert (out["a"][0] == bli(src, tgt, r, q, gold, k=5)).all()
    assert (out["b"][1] == bli(src, tgt, w2, q, single, k=5)).all()
    ref = _combined_reference(src, tgt, r, w2, q, gold, k=5)
    assert (out["combo"][0] == ref).all()
    assert (out["combo"][0] != out["a"][0]).any()      # the average is its own ranking
    same = bli_with_combination(src, tgt, r, r, q, [gold], k=5, block=8)
    assert (same["combo"][0] == same["a"][0]).all()


def test_bli_with_combinations_matches_pairwise_calls():
    src, _, r = _setup(n=80)
    tgt = torch.nn.functional.normalize(src @ r + 0.3 * torch.randn_like(src), dim=1)
    q = np.arange(25)
    golds = [[[i, (i + 3) % 80] for i in range(25)], [[i] for i in range(25)]]
    w2, w3 = _orth(16), _orth(16)
    out = bli_with_combinations(src, tgt, {"d": r, "c": w2, "b": w3},
                                {"dc": ("d", "c"), "db": ("d", "b")}, q, golds, k=5, block=8)
    assert set(out) == {"d", "c", "b", "dc", "db"}
    for w, combo in ((w2, "dc"), (w3, "db")):
        pair = bli_with_combination(src, tgt, r, w, q, golds, k=5, block=8)
        assert (out[combo] == pair["combo"]).all()
        assert (out[combo[1]] == pair["b"]).all()
    assert (out["d"] == bli_multi(src, tgt, r, q, golds, k=5)).all()


def test_weighted_and_multi_map_combinations():
    """Weights 0.5 / 0.5 rank as the equal-weight pair; weight 1 on one map ranks as that map;
    a three-map mean matches a reference built from each map's CSLS scores."""
    src, _, r = _setup(n=80)
    tgt = torch.nn.functional.normalize(src @ r + 0.3 * torch.randn_like(src), dim=1)
    q = np.arange(25)
    golds = [[[i, (i + 3) % 80] for i in range(25)]]
    w2, w3 = _orth(16), _orth(16)
    out = bli_with_combinations(
        src, tgt, {"d": r, "c": w2, "b": w3},
        {"pair": ("d", "c"), "half": {"d": 0.5, "c": 0.5}, "only_c": {"d": 0.0, "c": 1.0},
         "three": ("d", "c", "b"), "tilt": {"d": 0.7, "c": 0.3}}, q, golds, k=5, block=8)
    assert (out["half"] == out["pair"]).all()
    assert (out["only_c"] == out["c"]).all()
    assert (out["tilt"] != out["pair"]).any()

    def csls(w):
        m = torch.nn.functional.normalize(src @ w, dim=1).numpy()
        t = tgt.numpy()
        sim = m @ t.T
        r_t, r_s = np.sort(sim, 1)[:, -5:].mean(1), np.sort(sim.T, 1)[:, -5:].mean(1)
        return 2 * sim - r_t[:, None] - r_s[None, :]
    total = (csls(r) + csls(w2) + csls(w3)) / 3
    ref = [int((total[i] > total[i, g].max()).sum()) + 1 for i, g in zip(q, golds[0])]
    assert (out["three"][0] == np.array(ref)).all()


def test_combinations_with_nearest_neighbour_retrieval():
    """retrieval="nn": each map ranks as bli_multi's nn; a combination ranks by the mean cosine."""
    src, _, r = _setup(n=80)
    tgt = torch.nn.functional.normalize(src @ r + 0.3 * torch.randn_like(src), dim=1)
    q = np.arange(25)
    golds = [[[i, (i + 3) % 80] for i in range(25)]]
    w2 = _orth(16)
    out = bli_with_combinations(src, tgt, {"d": r, "c": w2}, {"dc": ("d", "c")}, q, golds,
                                block=8, retrieval="nn")
    for name, w in (("d", r), ("c", w2)):
        assert (out[name] == bli_multi(src, tgt, w, q, golds, retrieval="nn")).all()
    cos = sum(torch.nn.functional.normalize(src @ w, dim=1).numpy() @ tgt.numpy().T
              for w in (r, w2)) / 2
    ref = [int((cos[i] > cos[i, golds[0][i]].max()).sum()) + 1 for i in q]
    assert out["dc"][0].tolist() == ref
    csls = bli_with_combinations(src, tgt, {"d": r, "c": w2}, {"dc": ("d", "c")}, q, golds,
                                 k=5, block=8)
    assert (csls["dc"] != out["dc"]).any()
    with pytest.raises(ValueError):
        bli_with_combinations(src, tgt, {"d": r}, {}, q, golds, retrieval="cos")
