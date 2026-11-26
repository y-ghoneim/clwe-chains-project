import torch

from clwe_chains.align import (
    compose,
    csls_mutual,
    fit,
    fit_rcsls,
    fit_teacher,
    fit_vecmap,
    rcsls_loss,
    self_learn,
    side_map,
)
from clwe_chains.embeddings import normalize

torch.manual_seed(0)


def _orth(d):
    q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    return q


def test_procrustes_recovers_rotation():
    x, r = torch.randn(500, 20, dtype=torch.float64), _orth(20)
    assert torch.allclose(fit(x, x @ r, "procrustes"), r, atol=1e-8)


def test_lstsq_recovers_linear_map():
    x, m = torch.randn(500, 20, dtype=torch.float64), torch.randn(20, 20, dtype=torch.float64)
    assert torch.allclose(fit(x, x @ m, "lstsq"), m, atol=1e-8)


def test_aus_scheme_collapses_to_direct():
    """The earlier report fitted step 2 from the *mapped source* vectors over the same rows.
    For both solvers the chain then equals direct alignment exactly."""
    a, b, c = (torch.randn(800, 30, dtype=torch.float64) for _ in range(3))
    for solver in ("lstsq", "procrustes"):
        w1 = fit(a, b, solver)
        w2 = fit(a @ w1, c, solver)
        assert torch.allclose(w1 @ w2, fit(a, c, solver), atol=1e-8), solver


def test_chain_with_own_dictionaries_differs_from_direct():
    """Our scheme: each step is fitted on its own language pair, so a chain is a different map."""
    n, d = 600, 30
    a = torch.randn(n, d, dtype=torch.float64)
    b = a @ _orth(d) + 0.3 * torch.randn(n, d, dtype=torch.float64)
    c = b @ _orth(d) + 0.3 * torch.randn(n, d, dtype=torch.float64)
    direct = fit(a, c, "procrustes")
    chain = fit(a, b, "procrustes") @ fit(b, c, "procrustes")
    assert not torch.allclose(direct, chain, atol=1e-3)


def test_compose_order():
    a, b, c = (torch.randn(5, 5, dtype=torch.float64) for _ in range(3))
    assert torch.allclose(compose([a, b, c]), a @ b @ c)
    assert not torch.allclose(compose([a, b, c]), c @ b @ a)


def test_normalize_raw_is_identity():
    e = torch.randn(50, 8, dtype=torch.float64)
    assert torch.equal(normalize(e, []), e)


def test_normalize_unit_center_unit():
    e = 3.0 + 5.0 * torch.randn(200, 8, dtype=torch.float64)
    out = normalize(e, ["unit", "center", "unit"])
    assert torch.allclose(out.norm(dim=1), torch.ones(200, dtype=torch.float64))


def test_teacher_endpoints_and_orthogonality():
    x = torch.nn.functional.normalize(torch.randn(50, 16, dtype=torch.float64), dim=1)
    y = x @ _orth(16) + 0.1 * torch.randn(50, 16, dtype=torch.float64)
    wc = _orth(16)
    assert torch.allclose(fit_teacher(x, y, wc, 0.0), fit(x, y, "procrustes"))
    assert torch.allclose(fit_teacher(x, y, wc, 1e9), wc, atol=1e-6)
    w = fit_teacher(x, y, wc, 3.0)
    assert torch.allclose(w.T @ w, torch.eye(16, dtype=torch.float64), atol=1e-10)
    # in between: closer to the chain than direct is, closer to direct than the chain is
    wd = fit(x, y, "procrustes")
    assert (w - wc).norm() < (wd - wc).norm() and (w - wd).norm() < (wc - wd).norm()


def test_fit_with_fewer_pairs_than_dims_is_orthogonal():
    x, y = torch.randn(5, 16, dtype=torch.float64), torch.randn(5, 16, dtype=torch.float64)
    for w in (fit(x, y, "procrustes"), fit_teacher(x, y, _orth(16), 2.0)):
        assert torch.allclose(w.T @ w, torch.eye(16, dtype=torch.float64), atol=1e-10)


# ---------------------------------------------------------------------------------------------
# Stronger maps (experiments/strong-maps/)

def _unit(x):
    return x / x.norm(dim=1, keepdim=True)


def _toy(n=400, d=16, noise=0.05, seed=1):
    g = torch.Generator().manual_seed(seed)
    src = _unit(torch.randn(n, d, generator=g, dtype=torch.float64))
    q = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0]
    tgt = _unit(src @ q + noise * torch.randn(n, d, generator=g, dtype=torch.float64))
    return src, tgt, q


def test_vecmap_recovers_rotation_and_shares_one_space():
    x, q = torch.randn(500, 20, dtype=torch.float64), _orth(20)
    wx, wz = fit_vecmap(x, x @ q)
    assert torch.allclose(x @ wx, x @ q @ wz, atol=1e-8)
    assert torch.allclose(side_map(wx, wz), q, atol=1e-8)
    assert torch.allclose(side_map(wx, wz) @ side_map(wz, wx),
                          torch.eye(20, dtype=torch.float64), atol=1e-8)


def test_csls_mutual_pairs_identical_rows():
    x = _unit(torch.randn(50, 8, dtype=torch.float64))
    i, j = csls_mutual(x, x, k=3, block=16)
    assert (i == j).all() and len(i) == 50


def test_self_learning_grows_a_tiny_seed_and_never_uses_excluded_rows():
    src, tgt, q = _toy()
    seed = torch.arange(20)
    excl_a, excl_b = set(range(300, 320)), set(range(330, 350))
    va = torch.tensor([r for r in range(400) if r not in excl_a])
    vb = torch.tensor([r for r in range(400) if r not in excl_b])
    out = self_learn(src, tgt, seed, seed, va, vb, k=5, min_full=40, block=64)
    assert out.final == "vecmap" and out.n_induced > 200
    assert not set(out.a.tolist()) & excl_a and not set(out.b.tolist()) & excl_b
    assert (out.a == out.b).float().mean() > 0.95          # induced pairs are the true ones
    assert set(seed.tolist()) <= set(out.a[out.a == out.b].tolist())   # the seed is kept
    assert (side_map(out.wx, out.wz) - q).norm() < 0.1 * q.norm()


def test_self_learning_falls_back_to_procrustes_on_small_dictionaries():
    src, tgt, _ = _toy()
    seed = torch.arange(20)
    out = self_learn(src, tgt, seed, seed, torch.arange(400), torch.arange(400), k=5,
                     min_full=10_000, block=64)
    assert out.final == "procrustes"
    assert torch.equal(out.wz, torch.eye(16, dtype=torch.float64))
    assert torch.allclose(out.wx, fit(src[out.a], tgt[out.b], "procrustes"))


def test_self_learning_init_proposes_round_zero_pairs():
    """A seed of 3 pairs is too small to start from, a good init map is not: with it the
    induced pairs are the true ones."""
    src, tgt, q = _toy(noise=0.02)
    seed = torch.arange(3)
    out = self_learn(src, tgt, seed, seed, torch.arange(400), torch.arange(400), k=5,
                     min_full=40, block=64, init=q, rounds=1)
    assert (out.a == out.b).float().mean() > 0.95 and out.n_induced > 300


def test_rcsls_lowers_its_loss_from_the_procrustes_start():
    src, tgt, _ = _toy(noise=0.3)
    x, y = src[:150], tgt[:150]
    w = fit_rcsls(x, y, src, tgt, k=5, block=64)
    start, _ = rcsls_loss(x, y, src, tgt, fit(x, y, "procrustes").T, 5, 64)
    end, _ = rcsls_loss(x, y, src, tgt, w.T, 5, 64)
    assert w.shape == (16, 16) and end < start


def test_rcsls_gradient_matches_autograd():
    src, tgt, _ = _toy(n=60, d=6, noise=0.3)
    r = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))[0].requires_grad_(True)
    f, df = rcsls_loss(src[:20], tgt[:20], src, tgt, r.detach(), 3, 7)
    f2, _ = rcsls_loss(src[:20], tgt[:20], src, tgt, r, 3, 7)
    (g,) = torch.autograd.grad(f2, r)
    assert torch.allclose(f, f2.detach()) and torch.allclose(df, g, atol=1e-10)


def test_self_learning_final_procrustes_is_a_rotation():
    src, tgt, _ = _toy()
    seed = torch.arange(20)
    out = self_learn(src, tgt, seed, seed, torch.arange(400), torch.arange(400), k=5,
                     min_full=40, block=64, final="procrustes")
    assert out.final == "procrustes" and out.n_induced > 200
    assert torch.equal(out.wz, torch.eye(16, dtype=torch.float64))
    assert torch.allclose(out.wx.T @ out.wx, torch.eye(16, dtype=torch.float64), atol=1e-10)
