"""Phase 4 pairs -> <list>/pairs.csv and experiments/graph-all-pairs/pairs.csv.

Runs after the shared list exists (its languages are the graph's nodes). Selection rules in
clwe_chains.pairs and the `pairs:` config block; every pair gets a generated one-line reason. Run
and committed before any phase 4 result exists.
"""

from __future__ import annotations

import logging

import pandas as pd

from clwe_chains import load_config
from clwe_chains.config import REPO_ROOT
from clwe_chains.pairs import low_resource, select_pairs
from clwe_chains.shared_list import list_languages

log = logging.getLogger("pairs")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    pc = cfg["pairs"]
    nodes = list_languages(pd.read_parquet(cfg.path("list") / "shared_list.parquet"))
    langs = pd.read_csv(cfg.path("v0_languages")).set_index("ft_code")
    family = langs.loc[nodes, "family"].to_dict()
    macro = langs.loc[nodes, "macroarea"].to_dict()
    sizes = pc["sizes_gib"]
    contact = {frozenset(p) for p in pc["contact"]}
    df = select_pairs(nodes, family, sizes, pc, contact, cap=pc["cap_b"], seed=pc["seed"])

    m = pd.read_parquet(cfg.path("v0_distances"))
    asjp = m[m["measure"] == "asjp_ldnd"].set_index(["a", "b"])["value"]
    low = set(low_resource(sizes, nodes))

    def size(x: str) -> str:
        return "n/a" if sizes[x] is None else f"{sizes[x]:g}"

    def dist(s: str, t: str) -> float:
        return float(asjp.get((s, t), asjp.get((t, s), float("nan"))))

    sc = pc["scripts"]
    df["family_s"], df["family_t"] = df["s"].map(family), df["t"].map(family)
    df["script_s"], df["script_t"] = df["s"].map(sc), df["t"].map(sc)
    df["macroarea_s"], df["macroarea_t"] = df["s"].map(macro), df["t"].map(macro)
    df["size_s"], df["size_t"] = df["s"].map(sizes), df["t"].map(sizes)
    df["asjp"] = [round(dist(s, t), 3) for s, t in zip(df["s"], df["t"], strict=True)]
    def lows(s: str, t: str, r: str) -> str:
        return f" (low-resource: {', '.join(x for x in (s, t) if x in low)})" if r == "b" else ""

    df["reason"] = [
        f"{s}→{t}: {family[s]} vs {family[t]}; {sc[s]} vs {sc[t]} script; {macro[s]} / {macro[t]};"
        f" CC-100 {size(s)} / {size(t)} GiB{lows(s, t, r)}; pool {p}; ASJP {a:.2f}"
        for s, t, r, p, a in zip(df["s"], df["t"], df["stratum"], df["pool"], df["asjp"],
                                 strict=True)]
    out_list = cfg.path("list") / "pairs.csv"
    out_exp = REPO_ROOT / "experiments" / "graph-all-pairs" / "pairs.csv"
    out_exp.parent.mkdir(parents=True, exist_ok=True)
    for path in (out_list, out_exp):
        df.to_csv(path, index=False, lineterminator="\n")
    log.info("%d pairs (%s); written to %s and %s", len(df),
             df["stratum"].value_counts().to_dict(), out_list, out_exp)


if __name__ == "__main__":
    main()
