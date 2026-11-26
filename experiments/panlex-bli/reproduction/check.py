"""Which setting reproduces Vulić et al.'s orthogonal map on PanLex-BLI (1k:orthg-super, 0.071)?

Our direct map (Procrustes on the benchmark's 1k file) on all 210 directions, MRR per test pair
(one translation per query, as the paper), under three settings besides the experiment's own
(unit-centre-unit, CSLS; experiments/panlex-bli/results/d1000, 0.131): A length normalization only
and nearest-neighbour retrieval; B length only and CSLS; C unit-centre-unit and nearest neighbour.
Files and queries are read by experiments/panlex-bli/run.py's own functions. Writes out/.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clwe_chains.align import fit
from clwe_chains.embeddings import EmbeddingStore
from clwe_chains.evaluate import bli_multi

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("panlex_run", REPO / "experiments/panlex-bli/run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)
SETTINGS = {"A_unit_nn": (["unit"], "nn"), "B_unit_csls": (["unit"], "csls"),
            "C_ucu_nn": (None, "nn")}       # None: the configured unit, centre, unit


def main() -> None:
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    stores = {"unit": EmbeddingStore(run.CFG, max_cached=2, steps=["unit"]),
              "ucu": EmbeddingStore(run.CFG, max_cached=2)}
    data = run.Data(1000)
    k, block, half = run.CFG["eval"]["csls_k"], run.XCFG["block_rows"], run.CFG["eval"]["half"]
    rows, start = [], time.time()
    for i, (s, t) in enumerate(run.directions(run.XCFG["languages"])):
        test = run.read_trans(run.trans_file(s, t, "test.2000"))
        tq = run.test_queries(test, data.vocab(s), data.vocab(t))
        d = data.rows(s, t, "train.1000")
        for name, (steps, retrieval) in SETTINGS.items():
            store = stores["unit" if steps == ["unit"] else "ucu"]
            es, et = store[s], store[t]
            dev = es.device
            w = fit(es[torch.as_tensor(d["a"].to_numpy(), device=dev)],
                    et[torch.as_tensor(d["b"].to_numpy(), device=dev)], "procrustes")
            r = bli_multi(es, et, w, np.array(tq["queries"]), [tq["gold_pair"]], k=k,
                          block=block, retrieval=retrieval, half=half)[0]
            rows.append({"s": s, "t": t, "setting": name, "mrr": float((1.0 / r).mean()),
                         "p1": float((r == 1).mean()), "n": len(r)})
        print(f"{i + 1}/210 {s}-{t}: {(time.time() - start) / 60:.1f} min", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_direction.csv", index=False, lineterminator="\n", float_format="%.6g")
    avg = per.groupby("setting")[["mrr", "p1"]].mean().reset_index()
    avg.to_csv(out / "averages.csv", index=False, lineterminator="\n", float_format="%.6g")
    (out / "info.json").write_text(json.dumps({"commit": run.git("rev-parse", "HEAD").strip(),
                                               "k": k, "block": block, "half": half,
                                               "published_orthg_super_1k": 0.071,
                                               "published_full_super_1k": 0.101}, indent=1)
                                   + "\n", encoding="utf-8")
    print(avg.to_string(index=False))


if __name__ == "__main__":
    main()
