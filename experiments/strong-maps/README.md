# Stronger maps: can our method stay ahead of a strong direct map?

**Status: APPROVED by the user, 2026-11-23: the pilot on the remote; the pilot pairs stay in
the headline numbers; questions 3–5 as proposed. Pilot and full run done (2026-11-24); results
at the end ("Results").** A second "strong
pipeline" next to the current one (decisions.md 2026-11-22: "Higher scores: both word
translation and the real tasks"); the current numbers stay as they are. Real tasks (report 4.4)
get their own README once this one's pilot has run.

**Question.** Every map in the project is a single Procrustes rotation. Our direct maps match the
published Procrustes baselines (XLING 1K: 0.299 MRR; ours at 1,000 pairs 0.302), and published
stronger methods beat those by a wide margin, most of all on small dictionaries (PanLex-BLI, 1K
pairs: rotation 0.071 MRR, the full VecMap pipeline with self-learning 0.274; Vulić et al. 2019,
Table 3).
1. How much higher are our scores when every map, direct and chain steps alike, is fitted with a
   stronger method?
2. Do RQ2's conclusions hold then: does the chain still beat a thin direct dictionary, and where
   does its advantage stop?
3. Which pivot-based method does best against the strong direct map, and by how much? Candidates:
   the chain, the combination, a weighted combination, several hubs, and the chain as the seed
   for self-learning.

## Design
- **Everything except the maps and the new methods is the stages' own:**
  - pairs, hubs, folds (seed 0), test words and test removal from every edge;
  - gold, CSLS (k = 10) evaluation and the paired bootstrap.
  - Stage 1's runner, wrapped as in `../ensemble-control/`; own cache and results.
- **Scenarios:**

  | report section | stage | pairs (directions) | pooled direct |
  |---|---|---|---|
  | 4.2, under 500 pairs | stage 1 (`../scarce-natural-s1/`) | 77 (154) | yes |
  | 4.2, 500–1,000 | stage A (`../scarce-moderate-a/`) | 125 (250) | yes |
  | 4.2, 1,000–3,000 | stage A2 (`../scarce-moderate-a2/`) | 43 (86) | yes |
  | 4.2, well-resourced | stage D (`../natural-highres-d/`) | 40 (80) | no (as stage D) |
  | 4.2, shrinking direct | controlled scarcity (`../controlled-scarcity/`) | 40 (80) × 7 | yes |

  The gain curve (`analyses/2026-11-17-phase6-gain-curve/`) is recomputed from these results.
- **Not in scope:**
  - RQ1: self-learning adds different word pairs to each step, which breaks its same-words design;
  - stages B, teacher and 1E; chain via en; `joined_all`; RQ3a/b.

## Maps
### VecMap's supervised pipeline + self-learning (every map, unless stated)
Artetxe et al. 2018 (AAAI: the multi-step framework; ACL: self-learning), as in VecMap's
`--semi_supervised` defaults, reimplemented in torch. The published code is checked before the
build, and any difference is recorded here. Per edge and fold, one fit on the edge's training
dictionary (the seed):
1. **Fit** (vectors already unit, centred, unit; brief §3):
   - whitening of each side on the dictionary rows;
   - an orthogonal map from the SVD of the whitened cross-covariance;
   - re-weighting by the singular values^0.5 on both sides;
   - de-whitening of each side by its own whitening.

   **No dimension reduction**, so both side transforms `Wx`, `Wz` (300 × 300) stay invertible.
2. **Self-learning:** induce new pairs by CSLS (k = 10) mutual nearest neighbours between `x Wx`
   and `z Wz`, over each language's 20,000 most frequent words (`frequent_rows`,
   machine_dicts.py).
   - **The fold's test words of that language are removed in every casing:** source-side test
     words from the source, target-side ones from the target, as fold_dicts removes them.
   - New dictionary = seed ∪ induced pairs. The seed is never dropped: VecMap drops it, but we
     keep it because it is the only human-checked supervision.
   - Refit, and repeat.
3. **Stopping:** when under 1% of the induced pairs change between rounds, or after 10 rounds.
4. **Small dictionaries:** whitening needs more rows than dimensions.
   - In any round whose dictionary has under 600 pairs (2 × 300), that round's fit is plain
     Procrustes (`Wx` = the rotation, `Wz` = I).
   - Stage 1's seeds (66–499 pairs) and the 250-pair budget start this way. The full pipeline
     takes over once the induced dictionary is large enough.
5. **Map used for routes:** `M = Wx Wz⁻¹`, from the source into the target's own space.
   - Chain steps compose (`M_sx M_xt`), and every route is retrieved in the target's own
     vectors, as now.
   - One fit per edge and fold serves both directions (`M⁻¹ = Wz Wx⁻¹`).

### RCSLS (contrast)
Joulin et al. 2018: a map that directly optimizes the CSLS retrieval criterion.
- It starts from Procrustes and is unconstrained (not orthogonal).
- Settings are the official `fastText/alignment/align.py` defaults (learning rate, epochs,
  k = 10, negatives over the 200,000 vocabulary words), checked against the code before the build.
- No self-learning; each direction is fitted separately.
- **Only on stage D and controlled scarcity, and only for direct, chain and combination.** XLING
  found it strongest on larger dictionaries (5K: 0.437 against Procrustes 0.405); it is too slow
  to run everywhere.

## Methods (all with VecMap + self-learning maps)
**Baseline:** strong direct, the VecMap + self-learning map on the thin direct dictionary.

**Controls** (as in the stages):
- **pooled direct:** the seed is direct ∪ `joined_mb`. It tests whether a gain is only the hub's
  extra word pairs.
- **the single chain**, the single-best-route control that Mohammadshahi et al. 2024 call for
  (`analyses/2026-11-16-related-work-recent/`, RF1).

**Candidates for "our method":**
1. **Chain:** strong steps through the stage's hub, composed.
2. **Combination:** the equal-weight average of the direct and chain CSLS scores, as now.
3. **Weighted combination:** `(1 − w) · direct + w · chain`, with w from {0.1, 0.2, …, 0.9}
   chosen on the pilot pairs (below).
4. **Several hubs:** the equal-weight average of direct and the chains through the 3 widest hubs.
   - The hubs are ranked by `widest_hub`'s rule, the largest min(|s–x|, |x–t|), applied
     repeatedly, ties as there.
   - k = 3, fixed now; the brief's add-on lists k ∈ {1, 3, 5}, and k = 1 is candidate 2.
   - The second and third hubs' dictionaries are built by a new `prep/19_strong_hubs.py`, the
     same way prep/08 builds the first hub's: PanLex, plus Wiktionary on edges with en, and the
     same test removal.
   - The phase 6 data has only the first hub's edges (and en's for stage 1), so this needs new
     data.
5. **Chain-seeded direct:** the chain proposes the starting pairs for the direct map.
   - Round 0's dictionary is thin direct ∪ the CSLS mutual nearest neighbours of `x M_chain`
     and `z`, over the same 20,000-word vocabularies with test words removed.
   - Later rounds go back to thin direct ∪ the pairs induced by the current direct map, so the
     chain's proposals are a starting point only.
   - Self-learning fails from weak seeds (Vulić et al. 2019: unsupervised on 87 of 210 pairs), and
     the chain is a stronger seed than a thin dictionary.

## What is chosen, and how
- **The candidate is chosen on the pilot pairs, and the headline numbers use every pair, the
  pilot pairs included** (user, 2026-11-23).
  - The pilot pairs are about 7% of all pairs (20 of 285), so choosing on them inflates the
    headline only a little. The write-up says so, and gives the chosen candidate's mean on the
    other pairs as a one-line check.
  - In the pilot, w is set on the pilot pairs. In the full run, w is chosen by 5-fold
    cross-validation over all pairs (brief §5.4), so every pair's score uses a w chosen without it.
- **Selection rule** (fixed now): the candidate with the highest mean ΔMRR against strong direct
  over the pilot's directions.
  - Each of the five scenarios weighs equally, with controlled scarcity's budgets pooled.
  - Ties within 0.005 go to the simpler candidate, in the order 1, 2, 3, 4, 5.
- **Everything else** is a published default or fixed in this README. If a bug fix changes a
  method, it is dated in the build notes and the pilot is rerun.

## Pilot first (the "MVP" run)
A small slice of every scenario with every candidate, run before the full run.
- **Pilot pairs** (rule fixed now; `pilot_pairs.csv` written by the build from a seed-0 draw
  before any run):
  - stage 1: one pair per language (6);
  - stages A and A2: 4 pairs each, from different source languages;
  - stage D: 6 pairs;
  - controlled scarcity: the same 6 pairs at budgets 250, 1,000, 4,000 and 16,000, where the
    budget is below the pair's full dictionary.
  - All five folds, both directions; RCSLS on D and scarcity only.
- **Checks** (all must pass):
  - with the fit set to Procrustes, the wrapper reproduces the stages' cached ranks;
  - no induced or chain-proposed pair contains a test word (an assertion in the run, plus a count
    in the outputs);
  - every map is finite;
  - `M` and the reverse fit's inverse agree;
  - with w = 0.5, the weighted combination equals candidate 2.
- **Also measured:** seconds per fit, induced dictionary sizes and rounds, for an honest estimate
  of the full run.
- **Diagnostic** (pilot only, direct map only): retrieval in VecMap's shared space (`x Wx` against
  `z Wz`) next to retrieval in the target's own space (point 5). It shows how much the
  composable form gives up against VecMap as published.
- **Go / no-go** (fixed now):
  - The full run goes ahead if the checks pass and strong direct beats Procrustes direct on mean
    MRR over the pilot's directions (RCSLS likewise, on its scenarios).
  - If not, the pilot is written up as a negative result and raised with the user before
    anything else runs.

## Full run
- **On every pair and every scenario:** strong direct, the single chain, the combination, pooled
  direct, and the chosen candidate (if not among them).
- **RCSLS:** direct, chain and combination on stage D and controlled scarcity.
- **Headline numbers:** on all pairs, next to the standard pipeline's numbers.
- **The other candidates:** reported on the pilot pairs, labelled as such. If the pilot shows
  time to spare, they run on all pairs too, as secondary rows.

## Outputs
- **Per stage and pipeline** (standard = the stages' committed results, strong = this run):
  - direct MRR;
  - ΔMRR of each method against direct, with the paired bootstrap over test words;
  - counts of directions above and below 0;
  - means over directions and over languages, as each stage reports them.
- **Strong against standard:** per method, the paired difference strong − standard, by direction.
- **Pilot table:** every candidate's ΔMRR against strong direct per scenario on the pilot pairs,
  the chosen w, and the chosen candidate.
- **Controlled scarcity:** the curve (ΔMRR against budget) for both pipelines, and the first
  budget at which the chain, and the chosen candidate, stop winning, per direction.
- **Gain curve:** recomputed from stages 1, A, A2 and D under the strong pipeline.
- **Diagnostics:**
  - per fit: induced dictionary sizes, rounds, seconds, and the test words found in induced pairs
    (expected 0);
  - P@1 next to MRR, for the comparison with published tables.

## Expected outcomes (written before any run; guesses, to be marked wrong where wrong)
1. **Strong direct beats standard direct everywhere, most on small dictionaries.**
   - Scarcity at 250 pairs: direct 0.096 → at least 0.25; at 1,000: 0.302 → about 0.40;
     at 16,000: +0.01 to +0.03.
   - Stages 1, A and A2: direct +0.03 to +0.10. That's less than on scarcity, because their
     languages' weak vectors limit what self-learning can find.
2. **The single chain gains less than direct**, because its steps already have thousands of
   pairs. So its advantage shrinks:
   - stage 1's +0.035 (over languages) falls to between 0 and +0.02;
   - in controlled scarcity, the budget where the chain stops winning drops from 8,000–16,000 to
     1,000–4,000.
3. **The combination still beats strong direct** on every stage, by less than now: about +0.02,
   against +0.031 to +0.041.
4. **Several hubs is chosen.** It beats the combination by about +0.01 on the pilot pairs,
   because each hub makes different mistakes. Chain-seeded direct is close behind on stages 1 and
   A and at small budgets, but not on stage D. The chosen w is 0.3–0.5.
5. **The chosen candidate beats strong direct on every scenario**, by
   +0.02 to +0.04, smallest on stage D.
6. **Pooled direct stays about level with the single chain**, as now.
7. **RCSLS on stage D:** direct +0.02 to +0.04 over Procrustes (XLING: +0.03 at 5K). Chain against
   direct about as now (−0.002 over languages).
8. **Self-learning hurts in some low-resource directions**: those with the weakest vectors (yo,
   myv: X→hub MRR 0.08–0.11). At least a few directions fall below standard.
9. **The shared-space diagnostic** is within 0.02 of retrieval in the target's own space.

## Build and compute (estimates are the assistant's)
- **Library:**
  - `src/clwe_chains/align.py`: `fit_vecmap` (returns `Wx`, `Wz`), `self_learn` (with an optional
    round-0 seed), and `fit_rcsls`;
  - `src/clwe_chains/evaluate.py`: the combination generalized to weights and to more than two
    maps, every weight scored in one pass.

  Tests:
  - the VecMap fit on a random orthogonal toy problem recovers the map;
  - with re-weighting and whitening off, it equals Procrustes;
  - self-learning never induces an excluded row;
  - RCSLS doesn't lower its own objective from the Procrustes start;
  - the weighted and multi-map combinations reproduce the two-map function at equal weights.
- **Data:** `prep/19_strong_hubs.py`, dictionaries for the second and third hubs (pilot pairs
  first, the rest for the full run if several hubs is chosen).
- **Runner:** `experiments/strong-maps/run.py`, wrapping stage 1's runner as ensemble-control
  does; `--pilot`, `--full`, `--units`, `--pipelines`, `--limit` (build notes).
- **Build time:** about 6–7 h, plus a reviewer pass.
- **Fits** (one per edge and fold for VecMap; one per direction for RCSLS):
  - **Pilot:** about 1,100 VecMap fits (including 2 extra hubs × 2 steps, and chain-seeded direct)
    and 900 RCSLS fits.
  - **Full run, VecMap:** about 7,900 fits for the baseline and controls (stage 1 1,540, A 2,500,
    A2 860, D 600, scarcity 2,380), plus up to about 6,000 more if several hubs is chosen.
  - **Full run, RCSLS:** about 6,000 fits, mostly small budgets.
- **Time:**
  - **Pilot:** about 2–3 h on the remote, or 3–5 h on the laptop GPU.
  - **VecMap full run:** about 2–4 s per fit (up to 10 rounds of a 20k × 20k CSLS search). That's
    4–9 h for the baseline and controls, and up to 15 h if several hubs is chosen, which would
    need two remote nights.
  - **RCSLS:** unknown until the pilot measures it; up to a minute per fit on the largest stage D
    dictionaries (280k pairs).

## Build notes (2026-11-23, before any run)
**Against the published code** (artetxem/vecmap `map_embeddings.py`; fastText
`alignment/align.py`, both read before the build):
- **VecMap's self-learning rounds use a plain rotation.** The whitening, re-weighting and
  de-whitening run once, on the final dictionary (its loop maps orthogonally until it ends). Ours
  does the same, so the 600-pair rule (point 4) applies to that final fit only. The final
  dictionaries are thousands of pairs, so the fallback should rarely fire; fits.csv records it.
- **VecMap's `--semi_supervised` default uses stochastic induction.** Candidate pairs are dropped
  at random with keep probability 0.1, doubled after every 50 rounds without improvement, so it
  runs hundreds of rounds (its README: about 10 minutes per fit on a GPU). For about 8,000 fits
  that's out of reach. Ours induces deterministically and stops by point 3's rule (under 1% new
  pairs, or 10 rounds).
- **VecMap induces the union of both directions' best matches and replaces the seed with them.**
  Ours keeps mutual best matches plus the seed, as approved (point 2, question 3).
- **VecMap's induction vocabulary is the first 20,000 rows;** ours is the first 20,000 that are
  words (`frequent_rows`: letters and marks only, two or more characters).
- **RCSLS settings are align.py's defaults:** learning rate 1.0, 10 iterations (11 gradient
  steps), k = 10, 200,000 negatives, full batch, no constraint. When the loss rises, align.py
  halves the learning rate and means to step back, but the step back does nothing (its `Rold`
  aliases `R`, which it updates in place); ours behaves the same.
- **Dropped check:** "`M` and the reverse fit's inverse agree" holds by construction, since one
  VecMap fit per edge serves both directions; RCSLS fits each direction separately.

**Runner** (`run.py`, config.yaml):
- `--make-pilot` writes `pilot_pairs.csv`, committed before the run.
- `--pilot` runs every method on the pilot pairs: vm (VecMap + self-learning), proc (Procrustes,
  as the stages) and rc (RCSLS, on D and the budgets). It writes `results/pilot/`:
  - `candidates.csv` and `choice.json`: the chosen candidate, w, and go / no-go;
  - `pipelines.csv`: strong − standard per direction, paired per test word;
  - `pooled.csv`, `summary.csv`, `fits.csv`.
- `--full` runs every pair with vm thin, chain, combo, pooled direct, the chosen candidate and rc
  (D, budgets).
  - Strong against standard uses each direction's mean MRR against the stages' committed
    `pooled.csv` (`standard.csv`), since the stages' caches are on the laptop.
  - `summary_without_pilot.csv` is the one-line check that leaves the pilot pairs out.
- `--check` (laptop) compares proc's ranks with the stages' caches.
- **The weighted combination uses one w for every scenario.** In the pilot it's chosen on the
  pilot pairs. In the full run it's cross-validated over pairs (5 folds, seed 0; a pair's stage D
  and budget cells share a fold), written as `vm_wcv`.
- **Hub-edge fits are stored on disk**, keyed by their inputs, so the budgets reuse stage D's
  chain steps.
- **Extra hubs:** `prep/19_strong_hubs.py` writes `v1/phase6/hubs/<stage>/` (hubs.csv, the
  second and third hubs' dictionaries). Controlled scarcity uses stage D's.

**Reviewer pass** (agent, 2026-11-23, before any run): no test-word leakage and no maths errors
against VecMap and align.py. Fixed:
- **Every map is checked for finite values before ranking.** A NaN map would rank every gold
  answer first. Self-learning's final VecMap fit also needs more distinct words than dimensions
  on each side, else it falls back to Procrustes.
- **Caches and stored fits live under a version tag:** a hash of align.py, the map and
  vocabulary settings and the normalization. A change starts them afresh.
- **Stored fits are saved atomically.**
- **The uncommitted-changes check covers every unit's pair list.**
- **The selection skips candidates without rows.**

Noted:
- On CPU on the three smallest stage 1 pairs (seeds of 79–205 pairs), self-learning ran all 10
  rounds and ended with about 12,000 pairs, about 99% of them induced. So the 1% stopping rule
  may rarely fire, and the seed is a small share of the final dictionary.
- **Go is provisional until `--check`** (proc against the stages' caches) has run on the laptop.
- **RCSLS's negatives are the whole vocabularies, test words included,** as in align.py and in
  CSLS evaluation itself. No test translation pair is used, but the method is transductive; the
  write-up says so.

**Laptop run** (decisions.md 2026-11-23: the remote is not available tonight): the pilot runs
`--pipelines vm,proc` first and RCSLS after, so the candidates' results come first.

## Change after the stage 1 pilot: every map a rotation (user, 2026-11-23)
**This change was made after seeing pilot results, and is labelled as such.**

The first pilot (VecMap's final step, as designed above) ran on the laptop from 03:16 to 05:45 and
was stopped. It covered stages 1, A and A2 (14 pairs, all folds) and stage D's first pair. What it
showed:
- **Procrustes reproduces the stages** exactly, in every direction of stages 1 and A.
- **VecMap's final step, in the form chains can use (`M = Wx Wz⁻¹`), distorts the maps.** On
  yo–ar the condition number is 36 for direct and 175 for the chain (1 for a rotation).
- **Chains collapse.** Stage 1 chain MRR 0.046 → 0.017 (12 directions), with about 0.000 for yo,
  mhr, mg and myv. Stage A 0.065 → 0.019. Stage D en–zh direct 0.895 → 0.646 (partial folds).
- **VecMap in its own shared space** is about level with Procrustes on en–zh (0.898) and
  stronger on stage A's direct maps (0.031 → 0.061). It can't be composed into a chain.
- **Self-learning itself is not the cause.** On one fold of yo→ar (CPU check),
  Procrustes on the self-learned dictionary gives direct 0.013 → 0.019 and chain 0.043 →
  0.041.

**Change** (config `selflearn.final: procrustes`):
- The self-learning rounds are unchanged, but the final map is Procrustes on the final
  dictionary. Every map is a rotation, which is Procrustes with self-learning (XLING's PROC-B).
- `thin_shared` is now a reference row: VecMap's full pipeline fitted on the same self-learned
  dictionary, scored in its own shared space. It is the strongest direct map, and it can't be
  chained.
- Everything else as approved: candidates, selection rule, go / no-go, RCSLS.
- **The pilot is rerun from scratch** under a new version tag. The VecMap run's cache stays under
  `v1/results/strong-maps/5e558240b2/` as the record of the negative result. Its log is
  `v1/results/strong-maps-pilot.log`.

## Questions settled with the approval (user, 2026-11-23)
1. Pilot pairs as proposed; **they stay in the headline numbers** (proposed: left out).
2. **The pilot runs on the remote** (proposed: the laptop GPU, to save a night).
3. The seed is kept in every self-learning round. 4. The 600-pair whitening threshold.
5. k = 3 hubs.

## Results (2026-11-24)
**Pilot** (laptop, 2026-11-23; `results/pilot/`): every check passed. Go for the rotation
pipeline (strong direct − Procrustes direct +0.119 over the pilot's directions, mostly from
controlled scarcity); RCSLS stopped after one stage D pair (user, decisions.md 2026-11-23), so it
has no go / no-go. **Chosen: several hubs (hubs3)**, mean ΔMRR against strong direct +0.044 over
the pilot's scenarios (weighted combination +0.033 at w = 0.7, combination +0.029, chain +0.021,
chain-seeded +0.016).

**Full run** (`results/full/`; stages 1, D, A2 and A, every pair, both directions, all folds;
controlled scarcity pilot-only). ΔMRR against strong direct, mean over directions (over
languages in brackets), and directions above / below 0 by the paired bootstrap:

| stage | dirs | strong direct MRR | chain | combination | several hubs (chosen) |
|---|---|---|---|---|---|
| 1 | 154 | 0.015 | +0.049 (+0.045), 147 / 0 | +0.037 (+0.036), 150 / 0 | **+0.063 (+0.059), 153 / 0** |
| A | 250 | 0.031 | +0.035 (+0.040), 230 / 0 | +0.035 (+0.041), 244 / 0 | **+0.049 (+0.056), 244 / 0** |
| A2 | 86 | 0.078 | +0.030 (+0.024), 71 / 4 | +0.052 (+0.040), 84 / 0 | **+0.077 (+0.049), 83 / 0** |
| D | 80 | 0.530 | −0.011 (−0.007), 11 / 39 | +0.028 (+0.032), 62 / 0 | **+0.042 (+0.048), 70 / 0** |

- **The pivot methods stay ahead of the self-learned direct map on every stage**, and several
  hubs beats it in 550 of 570 directions, below it in none. The chain alone still loses on stage
  D, as in the standard pipeline.
- **Without the pilot pairs** (`summary_without_pilot.csv`, the one-line check): several hubs
  +0.063, +0.048, +0.083 and +0.044 on stages 1, A, A2 and D, so choosing on the pilot pairs did
  not inflate the headline.
- **Strong against standard** (`standard.csv`, paired by direction): **self-learning barely moves
  direct on the real stages**: stage 1 −0.0008, A +0.0005, A2 +0.006, D +0.012. The chain gains
  more: +0.011, +0.004, +0.010, +0.006. In controlled scarcity (pilot pairs only) direct gains a
  lot: 0.117 → 0.577 at 250 pairs, 0.352 → 0.592 at 1,000, +0.085 at 4,000, +0.021 at 16,000;
  there several hubs still beats strong direct at every budget (+0.069, +0.055, +0.041, +0.023)
  and the chain alone stops winning at 16,000 (−0.002).
- **So "stronger maps" helps where the vectors are good and the dictionary small**, and hardly at
  all for the low-resource languages, whose weak vectors give self-learning little to find.
- **Pooled direct** (stages 1, A, A2): 0.010–0.017 below the chain, as in the standard pipeline.
- **The shared-space reference** (pilot, direct only): VecMap's own space is +0.003 to +0.031
  above the rotation; several hubs beats it on every pilot scenario.
- **Record notes:**
  - stage 1 was scored on the laptop (RTX 4050), D, A2 and A on the remote (RTX 4500); run_info's
    `gpu` names the machine that wrote the results (the laptop, rebuilt with `--results-only`;
    every results file byte-identical to the remote's). Self-learned maps differ slightly
    between machines (analyses/2026-11-24-selflearn-across-machines: per-direction gains move
    by about 0.002, the mean gain by at most 0.002);
  - the full results show three RCSLS rows for stage D (one direction, en → zh, scored before
    RCSLS was stopped). They are not a result;
  - `results/pilot/run_info.json` lists no `eval_commits` (a path bug, fixed in 791a6f2); the
    pilot's run logs show commit 44fc359. Kept as committed (user, 2026-11-24): a rebuild would
    also pick up the late RCSLS direction and record a "go" for RCSLS on one direction.

**Expected outcomes, marked:**
1. Strong direct beats standard everywhere, most on small dictionaries: **right on controlled
   scarcity** (250: 0.117 → 0.577, beyond the guess; 16,000: +0.021), **wrong on stages 1, A and
   A2** (guess +0.03 to +0.10; got −0.001 to +0.006).
2. The chain gains less than direct, so its advantage shrinks: **wrong.** The chain gained more;
   stage 1's chain advantage grew (+0.045 over languages, against +0.035). In scarcity it stops
   winning at 16,000, not 1,000–4,000.
3. The combination beats strong direct on every stage: **right**; by less than now (about
   +0.02): **wrong**, +0.032 to +0.041 over languages, about as now.
4. Several hubs is chosen: **right**, +0.014 over the combination on the pilot. w = 0.7 (guess
   0.3–0.5) and chain-seeded last, not close behind: **wrong.**
5. The chosen candidate beats strong direct on every scenario, smallest on D: **right**; by more
   than guessed (+0.042 to +0.077, guess +0.02 to +0.04).
6. Pooled direct level with the chain, as now: **as now, but not level** (0.010–0.017 below, in
   both pipelines).
7. RCSLS: **not run** (stopped).
8. Self-learning hurts some low-resource directions: **right**; stage 1 direct is not above
   standard in 84 of 154 directions (70 above).
9. Shared space within 0.02 of the target's own space: **partly**: within on 4 of 8 pilot
   scenarios, 0.024–0.031 above on D and scarcity 250–4,000.
