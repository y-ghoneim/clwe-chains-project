"""Phase 4 pair selection (brief 5.4; decisions.md 2026-11-15).

40 distant pairs in two strata, drawn by fixed rules with one seeded generator, before any
results. Distant = different top-level family and not a heavy-contact pair. Low-resource = the
bottom third of the nodes by CC-100 size (a language missing from CC-100 ranks last).

Stratum (a), both high-resource: en→ar, en→zh, en→ja fixed; 3 XLING partners each for fi and tr;
6 PanLex-BLI pairs among its high-resource languages; one pair each for ja, zh, fa, el, hi.
Stratum (b): 8 PanLex-BLI pairs with et/eu/ka; 8 pairs of one other low-resource language with a
high-resource partner; 4 pairs of two low-resource languages; at most `cap` pairs per language.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

XLING_PARTNERS = ["hr", "en", "de", "it", "fr", "ru"]
FIXED = [("en", "ar", "fixed"), ("en", "zh", "covering"), ("en", "ja", "covering")]
COVER = ["ja", "zh", "fa", "el", "hi"]
PANLEX_LOW = ["et", "eu", "ka"]


def low_resource(sizes: dict[str, float | None], nodes: list[str],
                 frac: float = 1 / 3) -> list[str]:
    """The round(frac * n) smallest nodes by size; None (not measured) ranks smallest."""
    order = sorted(nodes, key=lambda x: (sizes.get(x) is not None, sizes.get(x) or 0.0, x))
    return order[:round(frac * len(nodes))]


def eligible(a: str, b: str, family: dict[str, str], contact: set[frozenset]) -> bool:
    return a != b and family[a] != family[b] and frozenset((a, b)) not in contact


def select_pairs(nodes: list[str], family: dict[str, str], sizes: dict[str, float | None],
                 pools: dict, contact: set[frozenset], cap: int, seed: int) -> pd.DataFrame:
    """(s, t, stratum, pool) for the 40 pairs, in draw order."""
    rng = np.random.default_rng(seed)
    low = set(low_resource(sizes, nodes))
    high = sorted(set(nodes) - low)
    rows: list[dict] = []
    taken: set[frozenset] = set()
    count_b: dict[str, int] = dict.fromkeys(nodes, 0)

    def ok(a: str, b: str) -> bool:
        return eligible(a, b, family, contact) and frozenset((a, b)) not in taken

    def add(a: str, b: str, stratum: str, pool: str, fixed: bool = False) -> None:
        taken.add(frozenset((a, b)))
        rows.append({"a": a, "b": b, "stratum": stratum, "pool": pool, "fixed": fixed})
        if stratum == "b":
            count_b[a] += 1
            count_b[b] += 1

    def draw(cands: list[tuple[str, str]], n: int, stratum: str, pool: str) -> None:
        got = 0
        for i in rng.permutation(len(cands)):
            a, b = cands[i]
            if got == n:
                break
            if not ok(a, b) or (stratum == "b" and max(count_b[a], count_b[b]) >= cap):
                continue
            add(a, b, stratum, pool)
            got += 1
        if got < n:
            raise ValueError(f"pool {pool!r}: only {got} of {n} pairs could be drawn")

    def unordered(xs, ys) -> list[tuple[str, str]]:
        return sorted({tuple(sorted((x, y))) for x in xs for y in ys
                       if eligible(x, y, family, contact)})

    for s, t, pool in FIXED:
        add(s, t, "a", pool, fixed=True)
    for lang in ("fi", "tr"):
        draw(unordered([lang], [p for p in XLING_PARTNERS if p in pools["xling"]]), 3, "a",
             "xling")
    panlex_high = sorted(set(pools["panlex_bli"]) & set(high))
    draw([p for p in itertools.combinations(panlex_high, 2) if eligible(*p, family, contact)],
         6, "a", "panlex_bli")
    for lang in COVER:
        draw(unordered([lang], [h for h in high if h != lang]), 1, "a", "covering")

    draw(unordered(PANLEX_LOW, pools["panlex_bli"]), 8, "b", "panlex_bli")
    draw(unordered(sorted(low - set(PANLEX_LOW)), high), 8, "b", "low")
    draw([p for p in itertools.combinations(sorted(low), 2) if eligible(*p, family, contact)],
         4, "b", "both_low")

    out = []
    for r in rows:
        s, t = (r["a"], r["b"]) if r["fixed"] or rng.random() < 0.5 else (r["b"], r["a"])
        out.append({"s": s, "t": t, "stratum": r["stratum"], "pool": r["pool"]})
    return pd.DataFrame(out)


def scarce_pairs(size: pd.DataFrame, family: dict[str, str], low: list[str], nodes: list[str],
                 max_direct: int) -> pd.DataFrame:
    """Phase 6 stage 1 pairs (decisions.md 2026-11-16): a low-resource language s with a node t
    of another top-level family whose direct dictionary size is below max_direct. The hub is the
    node x (not s or t) maximizing min(size[s, x], size[x, t]), ties to the first in `nodes`;
    route is that bottleneck. size is symmetric (PanLex min(distinct words) per pair)."""
    rows = []
    for s in low:
        for t in nodes:
            if t == s or family[s] == family[t] or size.at[s, t] >= max_direct:
                continue
            hub, route = widest_hub(size, s, t, nodes)
            rows.append({"s": s, "t": t, "direct": int(size.at[s, t]), "hub": hub,
                         "route": route})
    return pd.DataFrame(rows, columns=["s", "t", "direct", "hub", "route"])


def widest_hub(size: pd.DataFrame, s: str, t: str, nodes: list[str]) -> tuple[str, int]:
    """The node x (not s or t) maximizing min(size[s, x], size[x, t]), ties to the first in
    `nodes`, and that bottleneck (phase 6's hub rule)."""
    via = pd.Series({x: min(size.at[s, x], size.at[x, t]) for x in nodes if x not in (s, t)})
    return via.idxmax(), int(via.max())
