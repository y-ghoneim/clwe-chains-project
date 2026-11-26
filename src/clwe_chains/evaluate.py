"""Bilingual lexicon induction with CSLS retrieval (Conneau et al. 2018), ported from v0.

CSLS(x, y) = 2 cos(x, y) - r_T(x) - r_S(y): r_T(x) is the mean cosine of mapped source word x to
its k nearest target words, r_S(y) that of target word y to its k nearest mapped source words
(over the whole mapped source vocabulary). Metrics: P@1, P@5 and MRR, the mean of 1 / rank of
the best-ranked correct answer (Glavaš et al. 2019; v0 called it MAP).

half=True uses fp16 (the GPU's tensor cores) only to find each word's nearest-neighbour
candidates in the expensive vocabulary-by-vocabulary products, then recomputes those candidates'
cosines in full precision. fp16 scores alone shift near-tied ranks by 1-2 places; with the
refinement, ranks match full precision (tests/test_evaluate.py::test_half_matches_full).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def knn_mean_sim(queries: torch.Tensor, keys: torch.Tensor, k: int, block: int,
                 half: bool) -> torch.Tensor:
    """Mean cosine of each query to its k nearest keys (both already unit length).

    half=True: fp16 picks each query's 2k nearest candidates; their cosines are recomputed in
    the inputs' precision and the k best are averaged.
    """
    if not half:
        return torch.cat([(queries[i:i + block] @ keys.T).topk(k, dim=1).values.mean(1)
                          for i in range(0, len(queries), block)])
    q16, k16 = queries.half(), keys.half()
    out = []
    for i in range(0, len(queries), block):
        cand = (q16[i:i + block] @ k16.T).topk(min(2 * k, len(keys)), dim=1).indices
        exact = torch.einsum("bd,bcd->bc", queries[i:i + block], keys[cand])
        out.append(exact.topk(k, dim=1).values.mean(1))
    return torch.cat(out)


def bli(src_emb: torch.Tensor, tgt_emb: torch.Tensor, w: torch.Tensor, queries: np.ndarray,
        gold: list[list[int]], *, k: int = 10, block: int = 1024,
        retrieval: Literal["csls", "nn"] = "csls", candidates: np.ndarray | None = None,
        half: bool = False) -> np.ndarray:
    """Rank (1 = top) of the best-ranked gold answer for each query.

    candidates restricts retrieval, and CSLS's target-side term, to those target rows; every gold
    row must be among them. r_S(y), each target word's neighbourhood among mapped source words,
    still uses the whole mapped source vocabulary. No experiment so far combines candidates with
    CSLS (phase 3's list-only retrieval uses nn). retrieval="nn" ranks by plain cosine. A gold
    answer tied with other words gets the best of their ranks (ties are negligible in fp32).
    """
    return bli_multi(src_emb, tgt_emb, w, queries, [gold], k=k, block=block,
                     retrieval=retrieval, candidates=candidates, half=half)[0]


def bli_multi(src_emb: torch.Tensor, tgt_emb: torch.Tensor, w: torch.Tensor, queries: np.ndarray,
              golds: list[list[list[int]]], *, k: int = 10, block: int = 1024,
              retrieval: Literal["csls", "nn"] = "csls", candidates: np.ndarray | None = None,
              half: bool = False) -> np.ndarray:
    """bli for several gold sets over one retrieval pass: ranks of shape (len(golds), n).

    E.g. the main gold and the list word alone (brief 5.7's secondary gold, recorded from phase 5
    on) cost one set of similarity scores instead of two.
    """
    if retrieval not in ("csls", "nn"):
        raise ValueError(f"unknown retrieval {retrieval!r}")
    mapped = _unit(src_emb @ w)
    tgt = _unit(tgt_emb)
    dev = tgt.device
    if candidates is not None:
        pos = {int(r): i for i, r in enumerate(np.asarray(candidates))}
        missing = sorted({g for gold in golds for gs in gold for g in gs} - pos.keys())
        if missing:
            raise ValueError(f"{len(missing)} gold rows are not among the candidates, "
                             f"e.g. {missing[:5]}")
        golds = [[[pos[g] for g in gs] for gs in gold] for gold in golds]
        tgt = tgt[torch.as_tensor(np.asarray(candidates), device=dev, dtype=torch.long)]
    q = mapped[torch.as_tensor(np.asarray(queries), device=dev, dtype=torch.long)]
    if retrieval == "csls":
        r_tgt = knn_mean_sim(tgt, mapped, k, block, half)   # r_S(y) for every target word
        r_q = knn_mean_sim(q, tgt, k, block, half)          # r_T(x) for the queries
    ranks = np.empty((len(golds), len(q)), dtype=np.int64)
    for i in range(0, len(q), block):  # queries x targets is small: always full precision
        scores = q[i:i + block] @ tgt.T
        if retrieval == "csls":
            scores = 2 * scores - r_q[i:i + block, None] - r_tgt[None, :]
        for n, gold in enumerate(golds):
            for j, g in enumerate(gold[i:i + block]):
                row = scores[j]
                best = row[torch.as_tensor(g, device=dev, dtype=torch.long)].max()
                ranks[n, i + j] = int((row > best).sum().item()) + 1
    return ranks


def bli_with_combination(src_emb: torch.Tensor, tgt_emb: torch.Tensor, w_a: torch.Tensor,
                         w_b: torch.Tensor, queries: np.ndarray, golds: list[list[list[int]]], *,
                         k: int = 10, block: int = 1024, half: bool = False
                         ) -> dict[str, np.ndarray]:
    """CSLS ranks under two maps and under their combination, in one pass (phase 6).

    Keys "a", "b", "combo", each of shape (len(golds), n). The combination ranks targets by the
    equal-weight average of the two maps' CSLS scores, each map with its own r_S and r_T terms;
    those terms are computed once and serve all three, so the cost is about two evaluations.
    Ranking as bli_multi: best-ranked gold, ties optimistic.
    """
    return bli_with_combinations(src_emb, tgt_emb, {"a": w_a, "b": w_b}, {"combo": ("a", "b")},
                                 queries, golds, k=k, block=block, half=half)


def bli_with_combinations(src_emb: torch.Tensor, tgt_emb: torch.Tensor,
                          maps: dict[str, torch.Tensor],
                          combos: dict[str, tuple[str, ...] | dict[str, float]],
                          queries: np.ndarray, golds: list[list[list[int]]], *, k: int = 10,
                          block: int = 1024, half: bool = False,
                          retrieval: Literal["csls", "nn"] = "csls") -> dict[str, np.ndarray]:
    """bli_with_combination for several maps and averages of their CSLS scores.

    Keys: every name in `maps` and in `combos`, each of shape (len(golds), n). A combination is a
    tuple of map names (equal weights; two names give bli_with_combination's combo) or a dict of
    map name -> weight (strong maps' weighted combination). Each map's CSLS terms are computed
    once, so the cost is about one evaluation per map (the ensemble control,
    experiments/ensemble-control/). retrieval="nn" ranks by plain cosine and averages cosines
    (PanLex-BLI on its paper's basis); k and half are then unused.
    """
    if retrieval not in ("csls", "nn"):
        raise ValueError(f"unknown retrieval {retrieval!r}")
    if set(maps) & set(combos):
        raise ValueError(f"names used for both a map and a combination: {set(maps) & set(combos)}")
    tgt = _unit(tgt_emb)
    dev = tgt.device
    qi = torch.as_tensor(np.asarray(queries), device=dev, dtype=torch.long)
    parts = {}
    for name, w in maps.items():
        mapped = _unit(src_emb @ w)
        q = mapped[qi]
        parts[name] = (q, None, None) if retrieval == "nn" else (
            q, knn_mean_sim(tgt, mapped, k, block, half), knn_mean_sim(q, tgt, k, block, half))
    names = [*maps, *combos]
    out = {name: np.empty((len(golds), len(queries)), dtype=np.int64) for name in names}
    for i in range(0, len(queries), block):
        if retrieval == "nn":
            sc = {name: q[i:i + block] @ tgt.T for name, (q, _, _) in parts.items()}
        else:
            sc = {name: 2 * (q[i:i + block] @ tgt.T) - r_q[i:i + block, None] - r_tgt[None, :]
                  for name, (q, r_tgt, r_q) in parts.items()}
        for name in names:      # a combination's scores exist only while it is ranked
            scores = sc[name] if name in sc else _combined(sc, combos[name])
            for n, gold in enumerate(golds):
                for j, g in enumerate(gold[i:i + block]):
                    row = scores[j]
                    best = row[torch.as_tensor(g, device=dev, dtype=torch.long)].max()
                    out[name][n, i + j] = int((row > best).sum().item()) + 1
    return out


def _combined(sc: dict[str, torch.Tensor], combo: tuple[str, ...] | dict[str, float]
              ) -> torch.Tensor:
    """A combination's scores: the mean of the named maps' scores, or their weighted sum."""
    if isinstance(combo, dict):
        return sum(w * sc[name] for name, w in combo.items())
    return sum(sc[name] for name in combo) / len(combo)


def metrics(ranks: np.ndarray, identical: np.ndarray | None = None) -> dict[str, float]:
    """n, P@1, P@5 and MRR; with `identical`, also over non-identical queries (*_no_identical)."""
    def m(r: np.ndarray) -> dict[str, float]:
        if not len(r):
            return {"n": 0, "p1": float("nan"), "p5": float("nan"), "mrr": float("nan")}
        return {"n": len(r), "p1": float((r <= 1).mean()), "p5": float((r <= 5).mean()),
                "mrr": float((1.0 / r).mean())}
    out = m(np.asarray(ranks))
    if identical is not None:
        keep = ~np.asarray(identical, dtype=bool)
        out.update({f"{key}_no_identical": v for key, v in m(np.asarray(ranks)[keep]).items()})
    return out
