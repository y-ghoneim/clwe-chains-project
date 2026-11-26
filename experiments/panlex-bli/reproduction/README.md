# Which setting reproduces Vulić et al.'s orthogonal map on PanLex-BLI?

**Question.** The PanLex-BLI experiment's reproduction check (`experiments/panlex-bli/README.md`)
compares our direct map with Vulić et al. 2019's plain orthogonal map at 1k seeds (Table 3,
"1k:orthg-super", average MRR over the 210 directions: **0.071**). Ours scored **0.131**
(`experiments/panlex-bli/results/d1000/reproduction.csv`, per test pair, the paper's convention).
The paper's orthg-super uses length normalization only (its Table 1); its test-time retrieval is
not stated (CSLS is named for training). Does one of these differences account for the gap?

**Design.** Our direct map (Procrustes on the benchmark's 1k file) on all 210 directions, MRR per
test pair, under three settings: A length normalization + nearest neighbour; B length
normalization + CSLS (k = 10); C our unit-centre-unit + nearest neighbour. Everything else as the
experiment (its file reading and queries). `python experiments/panlex-bli/reproduction/check.py`
(user approved, 2026-11-25: "yeah sure run that"); about 40 min on the laptop.

**Answer** (run 2026-11-25, commit e32a5e9, 26 min; `out/averages.csv`, `out/per_direction.csv`).
Average MRR per test pair over the 210 directions:

| setting | normalization | retrieval | MRR | P@1 |
|---|---|---|---|---|
| A | length only | nearest neighbour | 0.089 | 4.5% |
| C | unit, centre, unit | nearest neighbour | 0.106 | 5.6% |
| B | length only | CSLS | 0.125 | 7.6% |
| the experiment | unit, centre, unit | CSLS | 0.131 | 8.0% |
| Vulić et al. 1k:orthg-super | length only | not stated | 0.071 | |
| Vulić et al. 1k:full-super | full preprocessing | not stated | 0.101 | |

- Most of the gap is retrieval: CSLS adds +0.025 to +0.037 over nearest neighbour; centring adds
  +0.006 to +0.017. The paper's numbers line up with nearest neighbour (A nearest its orthg-super,
  C next to its full-super), so it most likely scored with nearest neighbour.
- Under the paper's likely setting (A) our direct map still scores 0.089, 0.018 above 0.071 (a
  residual, presumably implementation details of their code). **The reproduction is explained in
  large part, not matched exactly.**
- Consequence: the experiment's comparisons between our own methods stand (same vectors,
  evaluation and test words); its absolute numbers are on a more generous basis than the paper's
  and are not set next to the published averages as they are.
