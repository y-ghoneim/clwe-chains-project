"""The AUS report's pipeline and metrics, for the phase 3 "AUS-exact" base case (decisions.md).

Not a method: the scheme equals direct least squares (brief 2). It is kept so one base case does
exactly what AUS did. Definitions follow the report's section 5.1:
  - P@k: share of words whose translation is among the k nearest target vectors of the mapped
    source vector (the distance isn't stated; cosine and Euclidean are both offered);
  - error: squared Frobenius norm ||XW - Y||²;
  - Hausdorff distance: max over both directions of max_u min_v ||u - v||;
  - neighbourhood overlap: (1 / nk) Σ_i |NN_U(i) ∩ NN_V(i)|, the k nearest neighbours of word i
    among the aligned vectors and among the target vectors (k isn't stated; cosine).
"""

from __future__ import annotations

from typing import Literal

import torch


def aus_scheme_map(xs: list[torch.Tensor]) -> torch.Tensor:
    """AUS Algorithm 1 along a route [X_src, X_1, ..., X_tgt] (same word rows in each).

    W_i = pinv(X_cur) · X_next, then X_cur <- X_cur · W_i: every step after the first is fitted
    from the mapped source vectors. Returns the product of the W_i.
    """
    cur, total = xs[0], None
    for nxt in xs[1:]:
        w = torch.linalg.pinv(cur) @ nxt
        cur = cur @ w
        total = w if total is None else total @ w
    return total


def frobenius_error(u: torch.Tensor, v: torch.Tensor) -> float:
    return float(((u - v) ** 2).sum())


def hausdorff(u: torch.Tensor, v: torch.Tensor) -> float:
    d = torch.cdist(u, v)
    return float(max(d.min(dim=1).values.max(), d.min(dim=0).values.max()))


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def neighbourhood_overlap(u: torch.Tensor, v: torch.Tensor, k: int) -> float:
    """Mean share of each word's k nearest neighbours (cosine, itself excluded) shared by the
    aligned space u and the target space v."""
    def knn(x: torch.Tensor) -> torch.Tensor:
        s = _unit(x) @ _unit(x).T
        s.fill_diagonal_(-float("inf"))
        return s.topk(k, dim=1).indices
    nu, nv = knn(u), knn(v)
    shared = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(nu, nv, strict=True))
    return shared / (len(u) * k)


def precision_at_k(u: torch.Tensor, v: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10),
                   metric: Literal["cosine", "euclidean"] = "cosine") -> dict[int, float]:
    """Share of rows i whose target row i is among the k nearest rows of v to u[i]."""
    if metric == "cosine":
        score = _unit(u) @ _unit(v).T
    elif metric == "euclidean":
        score = -torch.cdist(u, v)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    own = score.diagonal()[:, None]
    rank = (score > own).sum(dim=1) + 1
    return {k: float((rank <= k).double().mean()) for k in ks}
