# Chains between related languages (phase 6, stage B)

**Question.** Stages 1, A and A2 pair a low-resource language with a *distant* node. When the
partner is related (same branch or family), aligning directly may be easier (more similar spaces,
shared words), so thin direct could do better at the same dictionary size. Does relatedness shrink
or remove the chain's advantage?

decisions.md 2026-11-17; same design, code and evaluation as `experiments/scarce-natural-s1/`
(stage 1's runner; `run.py` here only points it at this stage's data and methods). This README is
written and approved before any run. Results are descriptive.

## Data (`prep/07` and `prep/08 --stage b`, `v1/phase6/b/`)
- **Pairs:** fixed by hand before any result in `pairs_spec.csv`: each low-resource language with
  the nodes of its own branch or family (no node is related to Quechua or Nahuatl). Hub = the node
  maximizing min(|s–x|, |x–t|), as everywhere. **16 pairs**, 32 pair-directions:

  | pair | relation | hub | direct pairs | s–hub pairs | hub–t pairs | eligible concepts |
  |---|---|---|---|---|---|---|
  | mhr–fi / et / hu | Uralic | en | 690 / 679 / 694 | 29,461 | 158,466 / 53,267 / 109,574 | 762 |
  | myv–fi / et / hu | Uralic | en | 1,146 / 974 / 1,042 | 4,758 | 158,466 / 53,267 / 109,574 | 591 |
  | or–hi | Indo-Aryan | ta | 3,125 | 2,100 | 5,237 | 328 |
  | or–ur | Indo-Aryan | hi | 249 | 3,125 | 9,260 | 328 |
  | as–hi / ur | Indo-Aryan | en | 1,549 / 787 | 4,912 | 104,320 / 38,160 | 660 |
  | mg–id / tl | Malayo-Polynesian | fr | 614 / 375 | 4,713 | 119,363 / 6,395 | 277–278 |
  | jv–id / tl | Malayo-Polynesian | en | 2,041 / 1,632 | 3,661 | 150,838 / 28,268 | 348 / 452 |
  | ckb–fa | Iranian | en | 1,301 | 8,184 | 69,921 | 626 |
  | yo–sw | Atlantic-Congo | en | 573 | 7,436 | 35,235 | 511 |

  Odia–Hindi is a natural oddity: its direct dictionary (3,125 pairs) is larger than any chain's
  first step (Hindi is excluded as hub because it is the partner; the hub is Tamil, 2,100).
- Natural dictionaries, test words, 5 disjoint folds, test removal: exactly as stage 1.

## Methods
As stage A: thin direct, the chain through the hub, the combination, joined direct (mutual best).

## Evaluation
As stage A. The question is answered in the combined analysis
(`analyses/2026-11-17-phase6-gain-curve/`): B's pairs against the distant pairs' gain curve at the
same supervision ratio.

## Expected outcomes (written before any run)
- **The chain still beats thin direct where its first step is much larger than direct** (the
  Meadow Mari pairs: 29,461 against ~690; Yoruba–Swahili; Odia–Urdu), **but by less than distant
  pairs at the same ratio**, because related spaces align better directly.
- **Thin direct wins where direct is close to or above the chain's first step:** Odia–Hindi (3,125
  against 2,100), Javanese–Indonesian (2,041 against 3,661), Assamese–Hindi.
- Absolute MRR higher than for the same languages' distant pairs.

## Compute
About 480 CSLS evaluations (32 pair-directions × 5 folds × 3) ≈ 10 min at 1.1 s. No new vectors.

## Results (run on the remote at 1741f92, 2026-11-18; `results/`)
Per pair, the mean ΔMRR over its two directions:

| pair | thin direct MRR | chain MRR | chain Δ | combination Δ | joined (mb) Δ |
|---|---|---|---|---|---|
| or–ur | 0.007 | 0.099 | +0.092 | +0.042 | +0.063 |
| mg–id | 0.059 | 0.146 | +0.087 | +0.087 | +0.080 |
| mhr–hu / fi / et | 0.036 / 0.034 / 0.018 | 0.103 / 0.093 / 0.073 | +0.067 / +0.059 / +0.056 | +0.052 / +0.046 / +0.043 | +0.050 / +0.043 / +0.063 |
| as–hi / ur | 0.113 / 0.023 | 0.171 / 0.075 | +0.057 / +0.052 | +0.093 / +0.053 | +0.061 / +0.046 |
| ckb–fa | 0.080 | 0.138 | +0.057 | +0.087 | +0.056 |
| mg–tl | 0.036 | 0.082 | +0.046 | +0.049 | +0.037 |
| jv–tl | 0.109 | 0.151 | +0.042 | +0.086 | +0.021 |
| myv–et / fi / hu | 0.023 / 0.038 / 0.040 | 0.043 / 0.044 / 0.045 | +0.020 / +0.006 / +0.006 | +0.017 / +0.017 / +0.017 | +0.012 / +0.003 / +0.005 |
| yo–sw | 0.016 | 0.028 | +0.012 | +0.012 | +0.003 |
| jv–id | 0.299 | 0.284 | −0.014 | +0.072 | −0.048 |
| or–hi | 0.245 | 0.104 | −0.141 | +0.004 | −0.010 |

Overall (mean of languages): chain **+0.031** (22 of 32 pair-directions above, 2 below: both
Odia–Hindi), combination **+0.051** (30 / 0), joined **+0.030** (21 / 1: jv→id).

**Against the expected outcomes:**
- The chain beats thin direct where its first step is much larger than direct (Meadow Mari,
  Odia–Urdu; Yoruba–Swahili weakly), yes. **By less than distant pairs at the same ratio, no:**
  B's mean residual from the distant pairs' gain curve is +0.005 (interval −0.020 to +0.025;
  `analyses/2026-11-17-phase6-gain-curve/`).
- Thin direct wins where direct is close to or above the first step: Odia–Hindi, yes (−0.141);
  Javanese–Indonesian, narrowly (−0.014, neither direction's interval below 0); Assamese–Hindi,
  **no** (+0.057).
- Absolute MRR higher than for the same languages' distant pairs: where the related dictionary is
  large, yes (jv–id direct 0.299, or–hi 0.245, as–hi 0.113, against those languages' stage A / A2
  direct of 0.03–0.11); otherwise about the same (mhr, myv, yo, mg).
- The combination never falls below thin direct, including Odia–Hindi (+0.004), where the chain
  loses 0.14.
