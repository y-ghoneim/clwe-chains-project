# Ensemble control: is the combination's gain just averaging two maps? (A3)

**Status: APPROVED by the user, 2026-11-21 (questions as proposed, but run on the remote, not
the laptop). Run 2026-11-21; results at the end.** Add-on (brief §5.10 isolation): own folder,
results and report paragraph; descriptive. Raised by the 2026-11-19 review (concern A3,
`docs/review-2026-11-19-open-concerns.md`); the user asked for it on 2026-11-21.

**Question.** The combination (equal-weight average of the direct and chain CSLS scores) beats
direct almost everywhere, and its gain is flat in the supervision ratio (about +0.04 MRR at every
direct size; +0.037 on stage D, where the chain alone is −0.005). That is what an ensemble of any
two different maps would do. Does averaging direct with a second map trained on **the same direct
dictionary** gain as much? If yes, the combination's gain is an ensemble effect and says nothing
about the pivot; if the chain gains clearly more, the pivot's different supervision is what helps.

## Design
- **Stages and pairs (proposed):** stage D (`../natural-highres-d/`, 40 well-resourced pairs, 80
  directions) and stage 1 (`../scarce-natural-s1/`, 77 low-resource pairs, 154 directions): the
  case where the chain adds nothing on its own and the case where it adds supervision. Same folds,
  test words, gold, test removal and CSLS evaluation as those stages (stage 1's runner, wrapped as
  in `../controlled-scarcity/`).
- **Maps per direction and fold:** direct (Procrustes on the fold's direct dictionary) and the
  chain, as in the stages; plus the second direct-type maps:
  - **boot:** Procrustes on a bootstrap resample of the same direct dictionary (rows drawn with
    replacement, as many as the dictionary has, after the fold's test words are removed; one
    resample per pair and fold, used by both directions as the direct dictionary is; seeded from
    seed 0, the pair's name and the fold). Same supervision, same model, different errors;
  - **lstsq** (stage D only): unconstrained least squares on the same dictionary. A different
    model on the same data. Not on stage 1: its direct dictionaries have 66–499 pairs (18 of 77
    under the 300 dimensions, where least squares is underdetermined), and least squares was 2–3×
    worse than Procrustes on small dictionaries (brief §3), so it would be a straw partner.
- **Combinations** (equal-weight average of CSLS scores, as the stages' combination): direct +
  chain (reproduces the stages' `combo`, a check), **direct + boot**, **direct + lstsq**; each
  against direct alone.
- **Library:** `bli_with_combination` generalised to several maps and the combinations of pairs
  of them in one pass (each map's CSLS terms computed once), with a test that it reproduces the
  two-map function.
- **Outputs:** per stage and direction, ΔMRR of each combination against direct with the paired
  bootstrap over test words; per stage, the mean over directions and counts above / below 0, and
  the gap combo − ensemble with its interval (stage 1: also the mean over languages, as its README
  reports).

## Expected outcomes (written before any run; guesses, to be marked wrong where wrong)
1. **direct + boot gains little:** +0.005 MRR or less on both stages (a bootstrap map is nearly
   the same rotation; its errors overlap direct's).
2. **direct + lstsq gains a little on stage D:** about +0.01, below the chain's +0.037.
3. **The chain combination stays clearly ahead of both** on both stages (combo − best ensemble at
   least +0.02), so the combination's gain is mostly not a generic ensemble effect: the chain's
   errors differ from direct's more than a second direct map's do.
4. Check: direct + chain reproduces the stages' `combo` exactly (stage 1; stage D ran on the
   laptop's GPU, so there identical up to near-tied ranks from fp16 candidate search; wording
   clarified before the run, reviewer, 2026-11-21).

## Build notes (2026-11-21, before any run)
- `run.py` writes per stage `results/<stage>/` (stage 1's tables: pooled.csv etc., with methods
  thin, chain, combo, boot, ensboot, lstsq, enslstsq) and `gaps.csv` (combo − each ensemble per
  direction, paired bootstrap over test words); `results/summary.csv` collects every stage with
  results: each map and combination against direct, and combo − each ensemble with an interval
  from a bootstrap over pairs.
- **Check 4** needs the stages' own caches, which are on the laptop only: after the remote run,
  `python experiments/ensemble-control/run.py --check` writes `results/check.csv` (share of
  queries with the same rank as the stage's cache, per method).
- A CPU test runs one direction end to end on small random vectors and checks that direct +
  chain equals stage 1's two-map combination.

## Compute and build (estimates are the assistant's)
- Build: about 1.5 h (wrapper, generalised combination, tests).
- Run, laptop: stage D 4 maps × 80 directions × 5 folds, stage 1 3 maps × 154 × 5, at about 3.5 s
  per map: about 1.6 h + 2.2 h, overnight-free. Remote about a third of that.

## Questions settled with the approval (user, 2026-11-21)
1. Partners: bootstrap (both stages) and least squares (stage D only)? (Proposed: yes.) An
   alternative is a bag of several bootstrap maps, a stronger ensemble but no longer a two-map
   average like the combination.
2. Stages D and 1? (Proposed: yes. Stages A, A2 and B behave like stage 1 and are left out.)
3. Laptop? (Proposed: yes, about 4 h.) **User: the remote instead** (the laptop's GPU stays free).

## Results (run 2026-11-21 on the remote, commit f3ff2bb; written up 2026-11-21)
**Run.** Both stages on the remote (RTX 4500 Ada): stage D 2,800 records in 34 min, stage 1
3,850 in 44 min (1.1–1.3 s per map); every record from f3ff2bb (`results/<stage>/run_info.json`).
Tables: `results/summary.csv`, `results/gaps.csv`, `results/<stage>/pooled.csv`.

**Check 4** (`results/check.csv`, `run.py --check` on the laptop): stage 1's thin, chain and
combo ranks equal its cache exactly (770 records each); stage D's equal the laptop's cache for
99.994% of queries (largest rank difference 5), the near-ties expected from a different GPU.

**ΔMRR against direct** (mean over directions; above / below: directions whose paired-bootstrap
interval over test words is above / below 0):

| | stage D (80 directions) | stage 1 (154 directions) |
|---|---|---|
| chain alone | −0.005 (9 / 22) | +0.037 (134 / 0) |
| boot alone | −0.024 (0 / 59) | −0.005 (0 / 49) |
| lstsq alone | +0.027 (50 / 7) | not run |
| **direct + chain (combo)** | **+0.037 (67 / 0)** | **+0.032 (146 / 0)** |
| direct + boot (ensboot) | +0.004 (8 / 0) | −0.000 (5 / 7) |
| direct + lstsq (enslstsq) | +0.039 (72 / 0) | not run |

Means over languages (as the stages report them): stage D combo +0.041, ensboot +0.005,
enslstsq +0.037; stage 1 combo +0.031, ensboot −0.000. The same picture, except that combo −
enslstsq on stage D changes sign (+0.003 over languages, −0.002 over directions; both within
noise).

**combo − ensemble** (paired per test word per direction; interval from a bootstrap over pairs):

| | mean | 95% interval over pairs | directions above / below |
|---|---|---|---|
| stage D, combo − ensboot | +0.033 | +0.026 to +0.040 | 63 / 0 |
| stage D, combo − enslstsq | −0.002 | −0.011 to +0.006 | 20 / 30 |
| stage 1, combo − ensboot | +0.032 | +0.028 to +0.036 | 149 / 0 |

- **A bootstrap map adds nothing.** It is a slightly worse copy of direct (a resample with
  replacement keeps about 1 − 1/e ≈ 63% of the distinct pairs; 67–87% of the source words on
  stage D, `results/d/dict_sizes.csv`) and averaging with it gains +0.005 at most.
- **Least squares, a different model on the same data, gains as much as the chain on stage D.**
  It is also better than Procrustes on its own there (+0.027, 50 directions above, 7 below).
- **By direct dictionary size on stage D** (post hoc; quartiles of the 80 directions' direct
  pairs, mean over folds):

  | direct pairs | direct MRR | lstsq alone | direct + lstsq | direct + chain |
  |---|---|---|---|---|
  | 2,158–6,059 | 0.230 | +0.013 | +0.028 | +0.061 |
  | 6,059–11,346 | 0.420 | +0.042 | +0.051 | +0.052 |
  | 11,346–33,278 | 0.621 | +0.035 | +0.044 | +0.023 |
  | 33,278–280,432 | 0.800 | +0.016 | +0.034 | +0.010 |

  The chain combination's gain falls with the direct size (Spearman ρ −0.81 over directions);
  the least-squares ensemble's does not (ρ 0.06). The chain combination is ahead on the smallest
  quarter of stage D's dictionaries and behind on the two largest; the stage D average hides that.
  Per direction, combo − enslstsq is clearly above / below 0 in 16 / 0 directions of the smallest
  quarter, 3 / 3, 0 / 11 and 1 / 16 of the others. Across pairs only: the pairs with small direct
  dictionaries also have much lower direct scores and different languages.

**Against the expected outcomes** (written before the run):
1. **Right.** direct + boot +0.004 (stage D) and −0.000 (stage 1), under +0.005 on both.
2. **Wrong.** direct + lstsq gains +0.039 on stage D, not about +0.01, and slightly more than the
   chain's +0.037; least squares alone also beats Procrustes there, which brief §3 (least squares
   2–3× worse on small dictionaries) did not lead us to expect for dictionaries of 2,000 pairs
   and up.
3. **Partly right:** right against boot on both stages, wrong against lstsq on stage D (not
   testable on stage 1), so its conclusion ("mostly not a generic ensemble effect") fails on
   stage D. combo − ensboot is +0.033 (stage D) and +0.032 (stage 1), above +0.02 with intervals
   clear of it on both; combo − enslstsq on stage D is −0.002 (interval −0.011 to +0.006), not
   +0.02.
4. **Right.** Stage 1 identical; stage D identical up to near-tied ranks.

**What this says about the combination.** Averaging direct with a near-copy of itself does
nothing, so the combination's gain is not a by-product of the averaging step. But on stage D,
where the chain alone is no better than direct, a second map fitted differently on the same
dictionary gives the same average gain: there the combination's gain is consistent with a generic
two-different-maps ensemble effect, and the pivot's extra supervision is not needed for it, except
on the smallest direct dictionaries (post hoc). On stage 1 the chain alone already gains +0.037
from the hub's larger dictionary; least squares was not run there (design), so whether a
different direct map would match the combination there is not tested.

**Not settled here.** Least squares beating Procrustes on stage D's large dictionaries means
stage D's direct baseline is not the strongest single map available. Every comparison in the
project is Procrustes against Procrustes, so they are like for like; a stronger solver used for
both the direct map and the chain's steps might change them (not tested; in v0, on shared word
triples, properly fitted least-squares chains beat direct in 8 of 8 pairs against 5 of 8 with
Procrustes, brief §3).
Noted for the report (raised with the user, 2026-11-21).
