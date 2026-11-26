"""The chain's gain against its supervision advantage, over phase 6 stages 1, A, A2, B, D.

Implements README.md (specified before stages A, A2, B, D were run). Per pair: gain = mean over
its two directions of the pooled delta MRR (method - thin direct) from each stage's pooled.csv;
r = log2(route pairs / direct pairs), route pairs = min(|s-hub|, |hub-t|), direct pairs = |s-t|,
distinct word pairs of the full natural dictionaries (<phase6>/<stage>/dict_sizes.csv). Fit
gain = a + b r on the distant pairs (stages 1, A, A2, D) with a cluster bootstrap over languages
(the low-resource language for 1/A/A2, each pair its own cluster for D; 1,000 resamples, seed 0);
crossing point r0 = -a/b. Stage B's residuals from that fit. Secondary: log10(direct pairs) as
predictor; joined direct (mutual best) and the combination against thin direct.

Run: python gain_curve.py out/ > out/report.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from clwe_chains import load_config

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[3]
P6 = load_config().path("phase6")
STAGES = {"1": ("scarce-natural-s1", ""), "A": ("scarce-moderate-a", "a"),
          "A2": ("scarce-moderate-a2", "a2"), "B": ("close-pairs-b", "b"),
          "D": ("natural-highres-d", "d")}
N_BOOT, SEED = 1000, 0


def pair_table(stage: str) -> pd.DataFrame:
    exp, data = STAGES[stage]
    p6 = P6 / data if data else P6
    pairs = pd.read_csv(p6 / "pairs.csv")
    sizes = pd.read_csv(p6 / "dict_sizes.csv").set_index("edge")["pairs"]
    pooled = pd.read_csv(REPO / "experiments" / exp / "results" / "pooled.csv")
    rows = []
    for s, t, hub in zip(pairs["s"], pairs["t"], pairs["hub"], strict=True):
        direct = sizes[f"{s}__{t}"]
        route = min(sizes[f"{s}__{hub}"], sizes[f"{hub}__{t}"])
        sel = pooled[((pooled["src"] == s) & (pooled["tgt"] == t))
                     | ((pooled["src"] == t) & (pooled["tgt"] == s))]
        for method, g in sel.groupby("method"):
            if method not in ("chain", "joined_mb", "combo"):
                continue
            rows.append({"stage": stage, "s": s, "t": t, "hub": hub, "method": method,
                         "direct_pairs": direct, "route_pairs": route,
                         "r": np.log2(route / direct), "log10_direct": np.log10(direct),
                         "gain": g["delta"].mean(), "n_dirs": len(g),
                         "cluster": f"D:{s}-{t}" if stage == "D" else s})
    return pd.DataFrame(rows)


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b, a = np.polyfit(x, y, 1)
    return a, b


def cluster_boot(df: pd.DataFrame, xcol: str) -> dict:
    a, b = fit(df[xcol].to_numpy(), df["gain"].to_numpy())
    clusters = df["cluster"].unique()
    groups = {c: df[df["cluster"] == c] for c in clusters}
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        d = pd.concat([groups[c] for c in pick])
        if d[xcol].nunique() < 2:
            continue
        ba, bb = fit(d[xcol].to_numpy(), d["gain"].to_numpy())
        boots.append((ba, bb, -ba / bb if bb > 0 else np.nan))
    boots = np.array(boots)
    ci = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5]))
    r0 = -a / b if b > 0 else np.nan
    return {"a": a, "b": b, "a_ci": ci(boots[:, 0]), "b_ci": ci(boots[:, 1]), "r0": r0,
            "r0_ci": ci(boots[:, 2]) if b > 0 else (np.nan, np.nan),
            "r0_undefined_share": float(np.mean(np.isnan(boots[:, 2]))),
            "n_pairs": len(df), "n_clusters": len(clusters)}


def main() -> None:
    pd.set_option("display.width", 200)
    allp = pd.concat([pair_table(s) for s in STAGES], ignore_index=True)
    allp.to_csv(OUT / "pairs.csv", index=False, lineterminator="\n", float_format="%.6g")
    chain = allp[allp["method"] == "chain"]
    distant = chain[chain["stage"] != "B"]

    print("1. Description: chain gain by bins of r (one unit), distant pairs and stage B")
    for name, d in (("distant", distant), ("B", chain[chain["stage"] == "B"])):
        b = d.assign(bin=np.floor(d["r"]).astype(int)).groupby("bin")["gain"] \
            .agg(["mean", "size"]).round(4)
        print(f"  {name}:\n{b.to_string()}")
    print("  stage means (pairs): r, gain")
    print(chain.groupby("stage")[["r", "gain"]].agg(["mean", "min", "max"]).round(3).to_string())

    res = {}
    print("\n2. Fit on distant pairs: gain = a + b r (cluster bootstrap over languages)")
    res["chain_r"] = cluster_boot(distant, "r")
    print(" ", {k: (np.round(v, 4) if not isinstance(v, tuple) else tuple(np.round(v, 4)))
                for k, v in res["chain_r"].items()})

    print("\n3. Relatedness: stage B residuals from the distant fit")
    a, b = res["chain_r"]["a"], res["chain_r"]["b"]
    bp = chain[chain["stage"] == "B"].assign(resid=lambda d: d["gain"] - (a + b * d["r"]))
    rng = np.random.default_rng(SEED)
    resid = bp["resid"].to_numpy()
    boots = [rng.choice(resid, size=len(resid)).mean() for _ in range(N_BOOT)]
    print(f"  mean residual {resid.mean():.4f}, 95% [{np.percentile(boots, 2.5):.4f}, "
          f"{np.percentile(boots, 97.5):.4f}] over {len(resid)} pairs")
    print(bp[["s", "t", "r", "gain", "resid"]].round(4).to_string(index=False))
    bp.to_csv(OUT / "b_residuals.csv", index=False, lineterminator="\n", float_format="%.6g")

    print("\n4. Secondary")
    res["chain_log10_direct"] = cluster_boot(distant, "log10_direct")
    print("  chain vs log10(direct pairs):", {k: np.round(v, 4) if not isinstance(v, tuple)
                                              else tuple(np.round(v, 4))
                                              for k, v in res["chain_log10_direct"].items()})
    for m in ("joined_mb", "combo"):
        d = allp[(allp["method"] == m) & (allp["stage"] != "B")]
        res[f"{m}_r"] = cluster_boot(d, "r")
        print(f"  {m} vs r:", {k: np.round(v, 4) if not isinstance(v, tuple)
                               else tuple(np.round(v, 4)) for k, v in res[f"{m}_r"].items()})

    print("\n5. Stage means of the gain (pairs), by method")
    print(allp.groupby(["stage", "method"])["gain"].agg(["mean", "size"]).round(4).to_string())
    pd.DataFrame([{"fit": k, **{kk: (str(tuple(np.round(vv, 5))) if isinstance(vv, tuple)
                                     else vv) for kk, vv in v.items()}}
                  for k, v in res.items()]).to_csv(OUT / "fits.csv", index=False,
                                                    lineterminator="\n", float_format="%.6g")


if __name__ == "__main__":
    main()
