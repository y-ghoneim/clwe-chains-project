# Tier 2: downstream transfer with the direct dictionary shrunk (artificial scarcity)

**Status: APPROVED by the user, 2026-11-20 (open questions as proposed). RUN 2026-11-21 on the
laptop at 126a143; results below.** Add-on (brief
§5.10 isolation): own folder, results and report section; descriptive. Approved as a plan on
2026-11-19 (decisions.md: "Tier 2: the same sentiment pairs with the direct dictionary shrunk to
fixed budgets; artificial scarcity; its BLI companion comes from the controlled-scarcity
experiment").

**Question.** The natural downstream runs found the chain level with direct on well-resourced
pairs and far ahead where direct dictionaries are small, but those small-dictionary cases are
different, weaker languages. Holding the languages fixed and shrinking only the direct dictionary:
does the chain's downstream advantage appear, and does it track the BLI advantage measured on the
same dictionaries (`../controlled-scarcity/`)?

## Design
- **Tasks and pairs (proposed):**
  1. **SIB-200, stage D's pairs** (the well-resourced set of `../downstream-sib200/`: 31 pairs, 62
     directions). These are exactly the pairs of the controlled-scarcity BLI experiment, so
     every downstream cell has its BLI cell, and SIB-200 proved the more map-sensitive task.
  2. **UMSAB sentiment, stage u's pairs** (the decided "sentiment pairs"; 30 directions without
     Hindi, Hindi apart as in part 1), without a BLI link unless the controlled-scarcity run also
     covers stage u.
- **Budgets:** the controlled-scarcity budgets (250 … 16,000 direct pairs) and the same nested,
  seed-0 subsamples (for stage u, built the same way).
- **Maps per direction and budget:** direct on the budget; the chain on the full hub dictionaries
  (as in the natural runs); joined direct (budget plus joined hub pairs, mutual best); no map.
  Fitted on the whole dictionaries (no test-word removal; the downstream text is different).
- **Classifiers:** the natural runs' (trained once per labelled language, unchanged), so only the
  maps differ from the natural results.
- **Outputs:** chain − direct and joined − direct per direction and budget with the paired
  bootstrap over test items; per budget: mean over pairs, counts above / below 0; the downstream
  curve against budget and r; Spearman of downstream against BLI advantage over cells (SIB-200).

## Expected outcomes (written before any run; guesses, to be marked wrong where wrong)
1. **SIB-200: the chain's advantage rises as the budget shrinks**, from level at 16,000 to about
   +5 to +15 macro-F1 points at 250–500 (the natural low-resource pairs gained 12.5 with 300–1,650
   direct pairs).
2. **UMSAB: the same direction, smaller** (+1 to +4 points at 250–500; sentiment moved less with
   map quality in part 1).
3. **Joined direct ≈ chain** downstream, as in BLI.
4. **Downstream and BLI advantages correlate across cells** (SIB-200, ρ about 0.5), since budget
   drives both.

## Findings while building (2026-11-20, before any result)
- **Check against the natural runs:** at the full budget, Tier 2's direct, chain and no-map
  macro-F1 equal the natural SIB-200 run exactly (372 rows) and the UMSAB run to within 0.0005
  (only MLP direct-map scores differ, in the fourth decimal; the same maps up to GPU arithmetic).
- **Pairs per budget:** as controlled scarcity for SIB-200 (31 of stage D's pairs without zh, ja,
  th); UMSAB keeps all 21 pairs up to 8,000 and 19 at 16,000.
- **Order:** Tier 2 runs after the controlled-scarcity BLI run, so its BLI correlation can be
  computed in the same run.

## Compute and build (estimates are the assistant's)
Laptop, minutes per run (the classifiers are reused; each cell refits one or two 300 × 300 maps).
Build about 1.5 h on top of the controlled-scarcity data step. Runs after controlled scarcity's
data step; its BLI link waits for that run.

## Questions settled with the approval (user, 2026-11-20, as proposed)
1. SIB-200 (stage D pairs) as the main task and UMSAB as the second? (Proposed: yes; the decision
   of 2026-11-19 named the sentiment pairs, which stay in, but SIB-200 matches the BLI pairs and
   is more sensitive to the maps.)
2. Extend the controlled-scarcity BLI run to stage u too, so UMSAB also gets a BLI link? (Proposed:
   no; about half as much again in compute for a task where BLI and downstream did not track.)

## Results (run 2026-11-21, laptop, commit 126a143)
**Run.** `python experiments/downstream-tier2/run.py`, every budget of both tasks, with the
controlled-scarcity BLI cells joined (`results/run_info.json`). Tables: `results/summary.csv` (per
task, group, budget, classifier and map), `delta.csv` (per direction), `transfer.csv` (macro-F1),
`binned_by_r.csv`, `correlations.csv`. Check: at the full budget, direct, chain and no-map
macro-F1 equal the natural SIB-200 run in all 372 rows and the UMSAB run in 251 of 252 (one MLP
direct-map score differs by 0.0005, as in the smoke check). Intervals below are a bootstrap over
pairs of the pair means.

**Mean macro-F1, logistic regression** (UMSAB without Hindi; the chain does not depend on the
budget, its value moves only where pairs drop out):

| budget | SIB-200 direct | joined | chain | UMSAB direct | joined | chain |
|---|---|---|---|---|---|---|
| 250 | 0.456 | 0.702 | 0.705 | 0.462 | 0.526 | 0.537 |
| 1,000 | 0.609 | 0.700 | 0.705 | 0.496 | 0.526 | 0.537 |
| 4,000 | 0.686 | 0.720 | 0.721 | 0.515 | 0.527 | 0.537 |
| 16,000 | 0.720 | 0.726 | 0.729 | 0.524 | 0.527 | 0.537 |
| full | 0.702 | 0.702 | 0.705 | 0.530 | 0.530 | 0.537 |

**Chain − direct, macro-F1 points, logistic regression** (mean over directions [interval];
directions whose paired bootstrap over test items is clearly above / below 0):

| budget | SIB-200 (directions) | above / below | UMSAB main (30) | above / below | UMSAB Hindi (12) |
|---|---|---|---|---|---|
| 250 | +24.9 [+22.5, +27.6] (62) | 61 / 0 | +7.5 [+6.1, +9.0] | 27 / 0 | +2.7 |
| 500 | +16.5 [+14.7, +18.6] (62) | 58 / 0 | +6.2 [+4.7, +7.9] | 25 / 0 | +1.4 |
| 1,000 | +9.6 [+8.1, +11.3] (62) | 41 / 0 | +4.1 [+2.7, +5.6] | 18 / 0 | +0.3 |
| 2,000 | +5.7 [+4.3, +7.2] (62) | 30 / 0 | +2.5 [+1.4, +3.6] | 14 / 2 | +0.1 |
| 4,000 | +3.5 [+2.3, +4.7] (52) | 15 / 0 | +2.2 [+1.4, +3.1] | 12 / 0 | −0.9 |
| 8,000 | +1.0 [+0.1, +2.0] (42) | 6 / 0 | +1.6 [+0.8, +2.4] | 10 / 3 | −0.4 |
| 16,000 | +0.9 [−0.8, +2.6] (24) | 4 / 0 | +1.3 [+0.6, +2.2] | 8 / 2 | +0.3 (8) |
| full | +0.3 [−0.5, +1.2] (62) | 6 / 2 | +0.7 [−0.0, +1.5] | 3 / 1 | −0.6 |

- **Joined direct:** SIB-200 within 0.5 points of the chain at every budget (+24.7 at 250, +3.3 at
  4,000, +0.0 at full); UMSAB about 1 point behind the chain at every budget (+6.4 at 250, +1.2 at
  4,000, +0.0 at full).
- **The neural network** agrees: SIB-200 chain +24.6 at 250, +2.8 at 4,000, +0.4 at full; UMSAB
  +7.5, +1.6, +0.4.
- **Against r and BLI** (Spearman over budget cells, the full budget left out; intervals resample
  pairs): SIB-200 F1 advantage vs r 0.78 [0.69, 0.85], vs the same cell's MRR advantage 0.76
  [0.66, 0.84] (366 cells; MLP 0.81 and 0.81); UMSAB vs r 0.53 [0.41, 0.67] (210 cells; MLP 0.70).
  Post hoc (reviewer, after the results): SIB-200's F1 advantage against the budget alone is
  ρ 0.83 (MLP 0.89), higher than against the MRR advantage; within one budget, F1 against MRR
  advantage is only 0.14–0.35 (MLP 0.06–0.38). The pooled correlation is the budget acting on
  both, not word-translation quality picking out the better direction.
- **UMSAB keeps a small chain advantage at large budgets** (+1.3 at 16,000, interval above 0;
  +0.7 at full, as in part 1), where BLI on stage D's pairs was level; the UMSAB pairs differ from
  stage D's, so the two are not directly comparable.
- **Hindi** (romanised tweets, apart): +2.7 at 250, about 0 from 1,000 on, as level as in part 1
  (romanised tweets against Devanagari vectors and dictionaries).

**Against the expected outcomes** (written before any run):
1. **Right in direction, too small in size.** SIB-200's advantage rises as the budget shrinks, is
   level at 16,000 (+0.9, interval across 0) and at full, but reaches +24.9 at 250 and +16.5 at
   500, above the predicted +5 to +15.
2. **Right in direction, too small in size.** UMSAB moves the same way and less, but +7.5 at 250
   and +6.2 at 500, above the predicted +1 to +4.
3. **Right for SIB-200, partly wrong for UMSAB:** joined direct trails the chain by about a point
   at every budget on sentiment.
4. **Right, and stronger:** downstream and BLI advantages correlate across cells, ρ 0.76 (0.81 with
   the neural network), not about 0.5, as predicted because budget drives both (see the post-hoc
   note above: within a budget the link is weak, 0.14–0.35). For comparison, the natural SIB-200
   run gave 0.75 on its low-resource pairs (across different pairs).
