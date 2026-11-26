"""Linear maps between embedding spaces, one per chain step, composed into one map.

Every step is fitted on its *own* pair of languages (for s->x, the training concepts' s and x
words). The AUS report fitted later steps from the mapped source vectors instead, which makes any
chain equal to direct alignment (tests/test_align.py, brief section 2).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import torch

from clwe_chains.evaluate import knn_mean_sim


def fit(x: torch.Tensor, y: torch.Tensor, solver: Literal["procrustes", "lstsq"]) -> torch.Tensor:
    """Map W (d x d) with x @ W ~ y. Rows of x and y are translation pairs."""
    if solver == "procrustes":
        u, _, vt = torch.linalg.svd(x.T @ y)
        return u @ vt
    if solver == "lstsq":  # pseudo-inverse least squares, as in the AUS report
        return torch.linalg.pinv(x) @ y
    raise ValueError(f"unknown solver {solver!r}")


def compose(maps: list[torch.Tensor]) -> torch.Tensor:
    """One map for a route: the step maps multiplied in route order (s->x1 first)."""
    return functools.reduce(torch.matmul, maps)


def fit_teacher(x: torch.Tensor, y: torch.Tensor, w_chain: torch.Tensor,
                lam: float) -> torch.Tensor:
    """Chain-as-teacher (phase 6): the orthogonal W minimizing ||xW - y||^2 + lam ||W - w_chain||^2.

    For orthogonal W both terms are constants minus a trace, so W maximizes
    tr(W^T (x^T y + lam w_chain)): W = U V^T from that matrix's SVD. lam = 0 is Procrustes on
    (x, y); a large lam returns w_chain."""
    u, _, vt = torch.linalg.svd(x.T @ y + lam * w_chain)
    return u @ vt


# ---------------------------------------------------------------------------------------------
# Stronger maps (experiments/strong-maps/): VecMap's supervised pipeline with self-learning, and
# RCSLS. Each mirrors its published code (artetxem/vecmap map_embeddings.py; fastText
# alignment/align.py); the differences are listed in the experiment's README.

def _whitening(m: torch.Tensor) -> torch.Tensor:
    """(m^T m)^(-1/2), from the SVD of m (VecMap's whitening_transformation)."""
    _, s, vt = torch.linalg.svd(m, full_matrices=False)
    return vt.T @ torch.diag(1 / s) @ vt


def fit_vecmap(x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """VecMap's supervised mapping (Artetxe et al. 2018, AAAI) on dictionary rows x, z: whitening
    of each side, the orthogonal map of the whitened cross-covariance, re-weighting of both sides
    by its singular values ** 0.5, de-whitening of each side by its own whitening; no dimension
    reduction. Returns (wx, wz), d x d and invertible: x @ wx and z @ wz share one space. Needs
    more distinct rows than dimensions on each side. Computed in float64."""
    dtype = x.dtype
    x, z = x.double(), z.double()
    wx1, wz1 = _whitening(x), _whitening(z)
    wx2, s, wz2t = torch.linalg.svd((x @ wx1).T @ (z @ wz1))
    wz2, r = wz2t.T, torch.diag(s.sqrt())
    wx = wx1 @ wx2 @ r @ wx2.T @ torch.linalg.inv(wx1) @ wx2
    wz = wz1 @ wz2 @ r @ wz2.T @ torch.linalg.inv(wz1) @ wz2
    return wx.to(dtype), wz.to(dtype)


def side_map(wx: torch.Tensor, wz: torch.Tensor) -> torch.Tensor:
    """The map from the source's own space into the target's own space, wx wz^-1 (its inverse,
    wz wx^-1, maps back): what chain steps compose."""
    return (wx.double() @ torch.linalg.inv(wz.double())).to(wx.dtype)


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def csls_mutual(xm: torch.Tensor, zm: torch.Tensor, k: int, block: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutual CSLS nearest neighbours between the rows of xm and zm (unit length): positions
    (i, j) where j is i's best target and i is j's best source. As in VecMap's induction, each
    side's neighbourhood term is taken over the other side's rows only."""
    r_x = knn_mean_sim(xm, zm, k, block, False)
    r_z = knn_mean_sim(zm, xm, k, block, False)
    fwd = torch.cat([(xm[i:i + block] @ zm.T - r_z[None, :] / 2).argmax(1)
                     for i in range(0, len(xm), block)])
    bwd = torch.cat([(zm[i:i + block] @ xm.T - r_x[None, :] / 2).argmax(1)
                     for i in range(0, len(zm), block)])
    i = torch.arange(len(xm), device=xm.device)
    keep = bwd[fwd] == i
    return i[keep], fwd[keep]


@dataclass
class SelfLearned:
    """A self-learned fit: side transforms (wx, wz; route map side_map(wx, wz)), the final
    dictionary's rows (a, b), the rounds run, the induced pairs in the final dictionary that are
    not seed pairs, and which fit made the final map ("vecmap", or "procrustes" when the
    dictionary was too small)."""
    wx: torch.Tensor
    wz: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    rounds: int
    n_induced: int
    final: str


def self_learn(src: torch.Tensor, tgt: torch.Tensor, seed_a: torch.Tensor, seed_b: torch.Tensor,
               vocab_a: torch.Tensor, vocab_b: torch.Tensor, *, k: int = 10, rounds: int = 10,
               tol: float = 0.01, min_full: int = 600, block: int = 1024,
               init: torch.Tensor | None = None, final: str = "vecmap") -> SelfLearned:
    """VecMap's semi-supervised self-learning (Artetxe et al. 2018) on vectors src, tgt from the
    seed dictionary rows (seed_a, seed_b), inducing pairs only among the rows vocab_a, vocab_b.

    Each round: Procrustes on the current dictionary, then the CSLS mutual nearest neighbours of
    the mapped vocab_a rows and the vocab_b rows; the new dictionary is the seed plus them (the
    seed is never dropped). Stops when under `tol` of the induced pairs are new since the last
    round, or after `rounds` rounds. The final map is fit_vecmap on the final dictionary, or
    Procrustes (wz = I) if it has fewer than min_full pairs or no more distinct words than
    dimensions on a side. init (a src -> tgt map) proposes
    round 0's extra pairs the same way; later rounds use only the seed and the current map.
    final="procrustes" makes the final map Procrustes on the final dictionary always (every map
    a rotation: Procrustes with self-learning, XLING's PROC-B; experiments/strong-maps/)."""
    dev = src.device
    n_b = len(tgt)
    seed_a, seed_b = seed_a.to(dev).long(), seed_b.to(dev).long()
    vocab_a, vocab_b = vocab_a.to(dev).long(), vocab_b.to(dev).long()
    xv, zv = _unit(src[vocab_a]), _unit(tgt[vocab_b])
    seed_code = torch.unique(seed_a * n_b + seed_b)

    def induce(w: torch.Tensor) -> torch.Tensor:
        i, j = csls_mutual(_unit(xv @ w), zv, k, block)
        return torch.unique(vocab_a[i] * n_b + vocab_b[j])

    def with_seed(code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        both = torch.unique(torch.cat([seed_code, code]))
        return both // n_b, both % n_b

    a, b = with_seed(induce(init)) if init is not None else (seed_a, seed_b)
    prev, n_round, induced = None, 0, torch.empty(0, dtype=torch.long, device=dev)
    for n_round in range(1, rounds + 1):
        induced = induce(fit(src[a], tgt[b], "procrustes"))
        a, b = with_seed(induced)
        if prev is not None:
            new = int((~torch.isin(induced, prev)).sum().item())
            if new < tol * max(len(induced), 1):
                break
        prev = induced
    n_induced = int((~torch.isin(induced, seed_code)).sum().item())
    dim = src.shape[1]       # whitening also needs more distinct words than dimensions per side
    if (final == "vecmap" and len(a) >= min_full
            and min(len(torch.unique(a)), len(torch.unique(b))) > dim):
        wx, wz = fit_vecmap(src[a], tgt[b])
        final = "vecmap"
    else:
        wx = fit(src[a], tgt[b], "procrustes")
        wz = torch.eye(wx.shape[0], dtype=wx.dtype, device=dev)
        final = "procrustes"
    return SelfLearned(wx, wz, a, b, n_round, n_induced, final)


def rcsls_loss(x: torch.Tensor, y: torch.Tensor, z_src: torch.Tensor, z_tgt: torch.Tensor,
               r: torch.Tensor, k: int = 10, block: int = 1024
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """align.py's rcsls(): the RCSLS loss of R (x @ R.T ~ y) and its gradient with respect to R,
    over the dictionary rows x, y with negatives z_src, z_tgt; scores in blocks of rows."""
    xt = x @ r.T
    zs = z_src @ r.T
    f = 2 * (xt * y).sum()
    df = 2 * y.T @ x
    fk0 = fk1 = torch.zeros((), dtype=x.dtype, device=x.device)
    dfk0, dfk1 = torch.zeros_like(df), torch.zeros_like(df)
    for i in range(0, len(x), block):
        v, idx = (xt[i:i + block] @ z_tgt.T).topk(k, dim=1)
        fk0 = fk0 + v.sum()
        dfk0 += z_tgt[idx].sum(1).T @ x[i:i + block]
        v, idx = (y[i:i + block] @ zs.T).topk(k, dim=1)
        fk1 = fk1 + v.sum()
        dfk1 += z_src[idx].sum(1).T @ y[i:i + block]
    f = f - fk0 / k - fk1 / k
    df = df - dfk0 / k - dfk1.T / k
    return -f / len(x), -df / len(x)


def fit_rcsls(x: torch.Tensor, y: torch.Tensor, z_src: torch.Tensor, z_tgt: torch.Tensor, *,
              k: int = 10, lr: float = 1.0, niter: int = 10, block: int = 1024) -> torch.Tensor:
    """RCSLS (Joulin et al. 2018) as fastText's alignment/align.py runs with its defaults:
    full-batch gradient descent from Procrustes, unconstrained, niter + 1 steps, the learning
    rate halved whenever the loss rises (stopping once it is below 1e-4). align.py's step back to
    the previous map does nothing (its Rold aliases R, which is updated in place), so it is left
    out here too. Returns W with x @ W ~ y."""
    r = fit(x, y, "procrustes").T
    f_old = 0.0
    for it in range(niter + 1):
        if lr < 1e-4:
            break
        f, df = rcsls_loss(x, y, z_src, z_tgt, r, k, block)
        r = r - lr * df
        f = f.item()
        if f > f_old and it > 0:
            lr /= 2
            f = f_old
        f_old = f
    return r.T
