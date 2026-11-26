# 2026-11-19: the gain curve after the English list-word fix

**Question.** Stages D, A and A2 were regenerated with English list words on their own rows
("center", not "Center"; decisions.md 2026-11-19) and their English-source directions recomputed
(D 4, A 1, A2 3). Does the pre-registered gain-curve analysis
(`analyses/2026-11-17-phase6-gain-curve/`, run before the fix and left as it was) change?

**Method.** Its `gain_curve.py`, copied unchanged, on the updated `pooled.csv` files (commit
0a3d5f4 plus the recomputed results). `python gain_curve.py out/ > out/report.txt`.

## Answer: no change worth reporting
| | before the fix | after |
|---|---|---|
| slope b (chain gain per doubling of route / direct) | 0.0067 [0.0031, 0.0156] | 0.0066 [0.0031, 0.0154] |
| intercept a | 0.0079 | 0.0080 |
| crossing point r₀ | −1.18 [−6.17, 1.12] | −1.20 [−6.22, 1.14] |
| against log10(direct): slope | −0.0305 [−0.038, −0.022] | −0.0304 [−0.038, −0.022] |
| stage B mean residual | +0.0046 [−0.020, +0.025] | +0.0046 [−0.020, +0.025] |
| combination against r: slope | −0.0030 | −0.0030 |

The report uses the numbers after the fix. Stage-level changes: stage A identical at the
precision reported; A2 chain +0.0262 → +0.0265, combination +0.0403 → +0.0405; D gains identical
to three decimals, direct MRR from English up by 0.01–0.03 (English → Turkish 0.807 → 0.839).
