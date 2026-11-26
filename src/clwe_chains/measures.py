"""Edge measures for the language graph (phase 4 design; decisions.md 2026-11-15).

Each measure becomes a cost matrix over the nodes (lower = closer, zero diagonal).
Language-level: resource (CC-100 size rank), SVG and econd-hm (Dubossarsky et al. 2020), URIEL+
distances (v0's table). Split-level: RSIM (Vulić et al. 2020) and the quality edge.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch


def pairwise(langs: list[str], fn: Callable[[str, str], float]) -> np.ndarray:
    """Symmetric matrix of fn over unordered pairs, zero diagonal."""
    n = len(langs)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = fn(langs[i], langs[j])
    return out


def scale_max(D: np.ndarray) -> np.ndarray:
    """D divided by its largest off-diagonal value."""
    off = D[~np.eye(len(D), dtype=bool)]
    return D / off.max()


def resource_cost(sizes: dict[str, float | None], langs: list[str]) -> np.ndarray:
    """(max(rank_a, rank_b) + 1) / n, rank 0 = largest size; None (not measured) ranks last."""
    order = sorted(langs, key=lambda x: (sizes.get(x) is None, -(sizes.get(x) or 0.0), x))
    rank = {x: i for i, x in enumerate(order)}
    return pairwise(langs, lambda a, b: (max(rank[a], rank[b]) + 1) / len(langs))


def spectrum(emb: torch.Tensor) -> np.ndarray:
    """Singular values (descending) of the length-normalized, column-mean-centred matrix."""
    x = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    x = x - x.mean(dim=0, keepdim=True)
    return torch.linalg.svdvals(x).double().cpu().numpy()


def svg(sa: np.ndarray, sb: np.ndarray, k: int = 40) -> float:
    """Singular value gap: sum over the first k of (log sa_i - log sb_i)^2."""
    return float(((np.log(sa[:k]) - np.log(sb[:k])) ** 2).sum())


def effective_condition(s: np.ndarray) -> float:
    """sigma_1 / sigma_erank, erank = floor(exp(entropy of s / sum(s))) (1-indexed)."""
    p = s / s.sum()
    p = p[p > 0]
    erank = max(1, math.floor(math.exp(-(p * np.log(p)).sum())))
    return float(s[0] / s[erank - 1])


def econd_hm(sa: np.ndarray, sb: np.ndarray) -> float:
    ka, kb = effective_condition(sa), effective_condition(sb)
    return 2 * ka * kb / (ka + kb)


def rsim(xa: torch.Tensor, xb: torch.Tensor) -> float:
    """Relational similarity (Vulić et al. 2020): Pearson correlation between the pairwise
    cosine similarities of translation pairs' words in the two languages (row i of xa and xb are
    a translation pair). The edge cost is 1 - rsim."""
    def sims(x: torch.Tensor) -> torch.Tensor:
        u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        s = u @ u.T
        i, j = torch.triu_indices(len(x), len(x), offset=1, device=x.device)
        return s[i, j].double()
    a, b = sims(xa), sims(xb)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def validation_concepts(ok_a: set[int], ok_b: set[int], list_ids: set[int],
                        n: int = 500) -> list[int]:
    """The n most frequent concepts (smallest ids) both languages cover that are *not* in the
    shared list: never trained or tested on, so they can score an edge without using up data."""
    return sorted((ok_a & ok_b) - list_ids)[:n]


def edge_quality(src_emb: torch.Tensor, tgt_emb: torch.Tensor, w: torch.Tensor,
                 queries: np.ndarray, gold: list[list[int]], block: int) -> float:
    """MRR of the map w on validation queries, nearest-neighbour retrieval over the target's full
    vocabulary (it only has to rank edges). The edge cost is 1 - MRR."""
    from clwe_chains.evaluate import bli, metrics

    return metrics(bli(src_emb, tgt_emb, w, queries, gold, block=block, retrieval="nn"))["mrr"]


def uriel_cost(measures: pd.DataFrame, langs: list[str], kind: str) -> np.ndarray:
    """URIEL+ distance matrix from v0's long table (rows a, b, measure, value; one direction)."""
    rows = measures[measures["measure"] == f"uriel_{kind}"]
    value = {frozenset((a, b)): v for a, b, v in zip(rows["a"], rows["b"], rows["value"],
                                                     strict=True)}

    def get(a: str, b: str) -> float:
        key = frozenset((a, b))
        if key not in value:
            raise KeyError(f"no uriel_{kind} value for {a}-{b}")
        return float(value[key])

    return pairwise(langs, get)
