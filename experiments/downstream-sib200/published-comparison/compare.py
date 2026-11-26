"""Our SIB-200 transfer next to the published XLM-R transfer (Adelani et al. 2024, Table 8).

Published: XLM-R fine-tuned on English / Arabic and tested on the target language (accuracy, %),
copied into published.csv from arXiv v3 (2309.07445v3), Table 8. Ours: the directions of
experiments/downstream-sib200 whose classifier language is en or ar and whose source is a
low-resource language (our "src -> tgt" = their "trained on tgt, tested on src"), accuracy of
logistic regression, direct, chain and no map. Writes out/comparison.csv.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
COL = {"en": "xlmr_from_eng", "ar": "xlmr_from_ara"}


def main() -> None:
    pub = pd.read_csv(HERE / "published.csv").set_index("lang")
    t = pd.read_csv(REPO / "experiments/downstream-sib200/results/transfer.csv")
    t = t[(t["model"] == "lr") & t["src"].isin(pub.index) & t["tgt"].isin(list(COL))]
    ours = t.pivot_table(index=["src", "tgt", "hub"], columns="method",
                         values="accuracy").reset_index()
    rows = []
    for r in ours.itertuples():
        p = pub.loc[r.src]
        rows.append({"src": r.src, "classifier": r.tgt, "hub": r.hub, "in_xlmr": p["in_xlmr"],
                     "ours_direct": 100 * r.direct, "ours_chain": 100 * r.chain,
                     "ours_nomap": 100 * r.nomap, "xlmr_transfer": p[COL[r.tgt]],
                     "xlmr_in_language": p["xlmr_in_language"],
                     "chain_minus_xlmr": 100 * r.chain - p[COL[r.tgt]]})
    out = pd.DataFrame(rows).sort_values(["in_xlmr", "src", "classifier"])
    (HERE / "out").mkdir(exist_ok=True)
    out.to_csv(HERE / "out/comparison.csv", index=False, lineterminator="\n", float_format="%.1f")
    print(out.round(1).to_string(index=False))
    print(out.groupby("in_xlmr")[["ours_direct", "ours_chain", "xlmr_transfer"]].mean().round(1))


if __name__ == "__main__":
    main()
