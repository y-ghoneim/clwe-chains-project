# Controlled scarcity: shrinking the direct dictionary on fixed pairs (BLI)

**Status: APPROVED by the user, 2026-11-20 (open questions as proposed). RUN 2026-11-20 on the
remote at 9baf6ac; results below (2026-11-21).** Add-on in the
sense of brief §5.10: it changes nothing in the natural experiments; its own folder, results and
report section; descriptive results (decisions.md 2026-11-16). Approved as a plan on 2026-11-19
(decisions.md: "controlled-scarcity BLI experiment on stage D's 40 pairs").

**Question.** RQ2's gain curve (`analyses/2026-11-17-phase6-gain-curve/`) compares different
language pairs: pairs with small direct dictionaries are also low-resource languages with weaker
vectors, so dictionary size and language are confounded (review concern A2), and the break-even
near 10,000 direct pairs was found after looking. Here the languages are fixed: on stage D's 40
well-resourced pairs, only the direct dictionary is shrunk to fixed budgets while the chain keeps
its full dictionaries. Does the chain's advantage appear as the direct dictionary shrinks, and
where does it cross zero? Does a direct map given the hub's joined pairs (pooled data, no
chaining) do as well as the chain?

## Design
- **Pairs:** stage D's 40 pairs (`experiments/natural-highres-d/pairs.csv`), both directions; hubs,
  test words, folds, gold and CSLS evaluation exactly as stage D.
- **Budgets (fixed now): 250, 500, 1,000, 2,000, 4,000, 8,000, 16,000 direct pairs.** A (pair,
  budget) cell is skipped when the budget is at least the pair's full direct dictionary (then it
  is stage D itself); the full dictionary is the reference row, taken from a rerun at the same
  commit, not from the old stage D files.
- **Subsampling:** the direct s–t dictionary is subsampled uniformly over its pairs, seed 0, nested
  (the 500-pair dictionary contains the 250-pair one, and so on), before the folds' test words are
  removed from every edge, as stage D removes them. One subsample per budget (the folds already
  vary the training words; a second seed is possible if time allows).
- **Methods (stage D's runner):** direct on the budget (thin), the chain through the hub on the full
  s–hub and hub–t dictionaries, their combination, and **joined direct** (`joined_mb`: the budget
  plus the s–t pairs obtained by joining the full hub dictionaries, mutual best; brief §5.10
  fairness: direct also gets every pair recoverable by joining). The chain does not depend on the
  budget, so it is evaluated once per pair and fold.
- **r per cell:** log2(narrowest chain step / budget).
- **Outputs:** per pair, direction and budget: MRR of each method, ΔMRR (method − thin) with the
  paired bootstrap over test words; per budget: mean over directions and counts above / below 0;
  the controlled curve (ΔMRR against budget, and against r) next to the natural cross-pair curve.

## Expected outcomes (written before any run; guesses, to be marked wrong where wrong)
1. **The chain's advantage rises as the budget shrinks:** clearly positive at 250–1,000 pairs
   (about +0.05 to +0.15 MRR), still positive at 2,000, and **crossing zero between 4,000 and
   16,000 pairs**, a prospective test of RQ2's post-hoc break-even near 10,000.
2. **Against r, the controlled curve lies on the natural one** within noise; if it lies clearly
   above, the natural low-resource gains were understated by weak vectors, if below, overstated.
3. **Joined direct ≈ chain at every budget** (both use the hub's full supervision); joined
   slightly ahead at large budgets, where the chain's second step adds error.
4. **The combination beats both at every budget**, its gain over thin shrinking with the budget.

## Changes and findings while building (2026-11-20, before any result)
- **The chain is evaluated in every budget's pass, not once:** stage D's runner scores direct,
  chain and their combination in one CSLS pass, so evaluating the chain separately would save
  little. Its scores are the same at every budget (same full hub dictionaries and folds), which
  doubles as a check.
- **Pairs per budget** (a pair stays while its full direct dictionary has more pairs than the
  budget): 40 at 250–2,000, 35 at 4,000, 27 at 8,000, 16 at 16,000; "full" has all 40. The
  16,000 row is therefore a subset of pairs with large dictionaries.
- **Data:** `prep/18_controlled_scarcity.py` writes `v1/phase6/dc/b<budget>/` (and `uc/` for
  Tier 2): nested seed-0 subsamples of the direct dictionaries, the hub edges unchanged, joined
  hub dictionaries, stage D's folds, test words and gold.
- **Smoke run** (laptop, en–ar at 250 and full): runs end to end; r for en–ar at full is −1.24, as
  in downstream part 1.
- **Run:** on the remote (docs/remote.md), `python experiments\controlled-scarcity\run.py`;
  resumable per budget; `--budgets` runs a subset.

## Compute and build (estimates are the assistant's)
- A prep step builds, per budget, the subsampled direct dictionaries and the joined dictionaries
  (stage D was built without joined files); a runner wraps stage D's runner per budget with the
  chain evaluated once (about 2 h to build, with tests).
- About 80 directions × 5 folds × (1 chain + 7 budgets × 3 methods) ≈ 8,800 evaluations at about
  3.5 s: about 9 h on the laptop or 3–4 h on the remote PC.

## Questions settled with the approval (user, 2026-11-20, as proposed)
1. Budgets 250 … 16,000 as above? (Proposed: yes.)
2. Include joined direct? (Proposed: yes; it is the brief's fairness rule and answers review
   concern A2's main question; it costs a third more.)
3. Laptop overnight or the remote? (Proposed: the remote.)

## Results (run 2026-11-20 on the remote, commit 9baf6ac; written up 2026-11-21)
**Run.** All eight budgets on the remote (RTX 4500 Ada, 1.25 s per evaluation, about 3 h);
every cached evaluation is from 9baf6ac (`results/b*/run_info.json`). Pairs per budget as
planned (40, 40, 40, 40, 35, 27, 16; 40 at full). Tables: `results/curve.csv` (per budget),
`results/cells.csv` (per budget and direction), `results/curve_by_r.csv`; the comparison with the
natural curve is `compare_natural.py` (written after the results; see the last section).

**Checks.** The chain's MRR is the same at every budget for every direction (largest difference
0), and the full budget reproduces stage D's `pooled.csv` and `overall.csv` exactly (chain,
combination and direct, all 80 directions).

**ΔMRR against direct on the budget** (mean over directions; above / below: directions whose
paired-bootstrap interval is above / below 0):

| budget | directions | direct MRR | chain | joined direct | combination | chain above / below |
|---|---|---|---|---|---|---|
| 250 | 80 | 0.096 | +0.417 | +0.408 | +0.366 | 80 / 0 |
| 500 | 80 | 0.194 | +0.319 | +0.311 | +0.296 | 80 / 0 |
| 1,000 | 80 | 0.302 | +0.210 | +0.204 | +0.209 | 80 / 0 |
| 2,000 | 80 | 0.395 | +0.117 | +0.114 | +0.134 | 79 / 0 |
| 4,000 | 70 | 0.509 | +0.058 | +0.060 | +0.086 | 60 / 0 |
| 8,000 | 54 | 0.598 | +0.020 | +0.027 | +0.053 | 22 / 2 |
| 16,000 | 32 | 0.710 | +0.001 | +0.015 | +0.031 | 7 / 5 |
| full | 80 | 0.518 | −0.005 | +0.006 | +0.037 | 9 / 22 |

The chain's MRR is 0.513 on all 40 pairs; the rows from 4,000 on cover fewer pairs, those with
larger full dictionaries (hence the higher direct MRR there).

- **On fixed pairs** (post hoc; `results/fixed_pairs.csv`): over the 16 pairs present at 16,000,
  the chain's gain is +0.58 at 250, +0.08 at 4,000, +0.03 at 8,000, +0.001 at 16,000 and −0.020
  with their full dictionaries; the curve is not an artefact of pairs dropping out.
- **Where it crosses zero, per direction** (post hoc; `results/crossing.csv`): the first budget
  at which the chain's change is 0 or below is 8,000 for 10 directions, 16,000 for 13 and the full
  dictionary for 30; for 27 directions (full dictionaries of median 5,500 pairs) the chain stays
  ahead even with the full dictionary.
- **Joined direct ≈ chain:** within 0.015 at every budget (within 0.01 up to 8,000); slightly
  behind at 250–2,000 (−0.009 at 250), slightly ahead from 4,000 (+0.008 at 8,000, +0.014 at
  16,000, +0.012 at full).
- **The combination** (equal-weight average of the direct and chain CSLS scores) is below the
  chain at 250 (−0.051) and 500 (−0.023), level at 1,000 and the best of the three from 2,000 on;
  it is clearly above direct in every direction up to 8,000, in 31 of 32 at 16,000 and 67 of 80
  at full, and clearly below in none. At small
  budgets the direct map is so weak that the equal weights pull the chain down.

**Against the natural cross-pair curve** (`compare_natural.py`; same units as the gain-curve
analysis after the English list-word fix, `experiments/natural-highres-d/gain-curve/`: a pair
is the mean of its two directions, bins of one unit of r; natural = its 285 distant pairs):

| r (floor) | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| controlled chain gain (pair-budget cells) | −0.021 (2) | −0.008 (21) | +0.008 (34) | +0.036 (46) | +0.088 (40) | +0.159 (39) | +0.255 (38) |
| natural chain gain (pairs) | +0.029 (7) | +0.020 (49) | +0.028 (96) | +0.030 (94) | +0.026 (10) | +0.042 (22) | +0.035 (1) |

Controlled counts are (pair, budget) cells: a pair appears at up to eight budgets, so they are not
independent. Not shown: bins below 0 (4 controlled cells, 6 natural pairs, all negative) and the
controlled bins 7–9 (54 cells, +0.37 to +0.67; no natural pair has r ≥ 7). Natural bins 0–2 hold
29 of stage D's pairs, which are also the controlled pairs at the full budget, so part of the
comparison below r = 3 sets a pair against itself.

By direct dictionary size (post hoc, the bins of the natural analysis's post-hoc table):

| direct pairs | controlled chain gain (share above 0) | natural chain gain (share above 0) |
|---|---|---|
| < 500 | +0.417 (100%) | +0.051 (100%) |
| 500–1,000 | +0.319 (100%) | +0.033 (99%) |
| 1,000–3,000 | +0.159 (100%) | +0.028 (100%) |
| 3,000–10,000 | +0.035 (82%) | −0.002 (41%) |
| ≥ 10,000 | −0.010 (28%) | −0.017 (9%) |

**Against the expected outcomes** (written before the run):
1. **Partly right.** The advantage rises as the budget shrinks, clearly positive at 250–2,000 and
   gone by 16,000 (+0.001, 7 above and 5 below); the crossing is at the top of the predicted
   4,000–16,000 range, consistent with RQ2's post-hoc break-even near 10,000 (from the natural
   log-linear fit). By direct-size bins the controlled curve changes sign later than the natural
   one: between 3,000–10,000 (+0.035, 82% above) and ≥ 10,000, where the natural bins already turn
   at 3,000–10,000 (−0.002, 41% above). **The size was wrong:** +0.21 to +0.42 at 250–1,000, not +0.05 to +0.15.
2. **Wrong: the controlled curve is not on the natural one.** Below r = 3 it is under the natural
   curve (by 0.02 to 0.05; these are mostly stage D's well-resourced pairs at large budgets, which
   lost in stage D too), at r 3–4 the two meet, and from r = 4 it is far above (+0.088 against +0.026 at
   r 4–5, +0.159 against +0.042 at r 5–6), and at the same direct size below 3,000 pairs it is
   about 6 to 10 times the natural gain (8.2×, 9.5×, 5.7×; the controlled cells sit at the bottom
   of each bin, 250, 500 and 1,000–2,000 pairs, against natural medians of 289, 676 and 1,296).
   This is consistent with the README's reading that weak vectors held the natural low-resource
   gains down, but the two sets also differ in dictionary quality (a random subset of a large,
   clean dictionary), test words and hubs, so it does not isolate the vectors.
3. **Right.** Joined direct ≈ chain at every budget (within 0.015), joined slightly ahead at the
   large budgets. With the languages fixed, the chain's gain is the hub's extra supervision; how it
   is used (composed maps or pooled pairs) hardly matters here. (In the natural runs the chain was
   ahead of joined direct, 4.2 of the report; why the two differ is not tested here.)
4. **Wrong below 2,000 pairs.** The combination beats both from 2,000 on and beats direct
   everywhere, but at 250 and 500 it is below the chain (fixed equal weights; a weight chosen per
   budget would avoid this, not tried). Its gain over direct shrinks with the budget, as expected.
