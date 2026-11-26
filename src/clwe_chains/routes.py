"""Routes between two languages: the AUS report's chains (brief 2)."""

from __future__ import annotations

import itertools


def aus_chains(src: str, tgt: str, mids: list[str]) -> list[list[str]]:
    """Every ordering of every non-empty subset of mids, as routes [src, ..., tgt].

    Sorted by length, then lexicographically; direct (src -> tgt) is not included. For three
    intermediates that is 3 one-hop + 6 two-hop + 6 three-hop = 15 chains, as in AUS.
    """
    routes = [[src, *p, tgt] for size in range(1, len(mids) + 1)
              for p in itertools.permutations(sorted(mids), size)]
    return sorted(routes, key=lambda r: (len(r), r))
