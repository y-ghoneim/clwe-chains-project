# Several hubs with plain Procrustes maps: where does our best result come from?

**Status: APPROVED by the user, 2026-11-24** ("yes commit and push"). Agreed in principle the same
day (decisions.md: "Several hubs with plain Procrustes maps: to run on the remote tonight"); the
user asked for it before deciding how strong maps is reported ("let's go several hubs with
procrustes first then evaluate afterwards").

**Question.** Strong maps' best method, several hubs (direct plus the chains through the 3 widest
hubs, CSLS scores averaged), beat its self-learned direct map in 550 of 570 directions. Two things
changed at once there: the maps (self-learned rotations instead of plain Procrustes) and the method
(three hubs instead of one). Which one gives the gain?
1. With plain Procrustes maps, how much does several hubs gain over direct, over the one-hub chain
   and over the combination?
2. How much does self-learning add on top, method for method (several hubs with self-learning
   minus several hubs with Procrustes)?

## Design
- **Everything except the maps is strong maps' own**, through its runner (`../strong-maps/run.py`,
  imported unchanged), as strong maps took stage 1's: the units (low-resource under 500, 500–1,000,
  1,000–3,000, well-resourced), pairs, hubs 1–3 (widest-path rule), folds (seed 0), test words and
  their removal from every edge, gold, CSLS (k = 10) and the paired bootstrap.
- **570 directions** (285 pairs × 2), all 5 folds, as strong maps' full run.
- **Maps: plain Procrustes** (a rotation fitted on each edge's dictionary, test words removed), as
  the stages; no self-learning. The chain composes its two rotations.
- **Methods**, all in one pass per direction and fold (`bli_with_combinations`):
  - direct (`thin`), the chain through hub 1, 2 and 3 (`chain`, `chain2`, `chain3`);
  - the combination (direct + chain through hub 1), several hubs (direct + the 3 chains), equal
    weights on CSLS scores, as strong maps.
- **Check:** direct, chain and combination here against each stage's own cache (rank by rank, as
  strong maps' `--check`): they are the same maps on the same data, so the ranks should match.
- Own cache and results (`<results>/several-hubs-procrustes/<unit>/`, `results/`).

## Measures (fixed before the run)
Per unit, pooled over the unit's directions' test words, with the paired bootstrap; per direction,
above / below 0:
1. several hubs minus direct, minus the chain, minus the combination (plain Procrustes);
2. self-learning's share: strong maps' several hubs minus this several hubs; strong maps' direct
   minus this direct; strong maps' chain minus this chain (read from strong maps' committed cache,
   `55ca51e319`);
3. the split of strong maps' headline gain (its several hubs minus plain direct) into the part from
   several hubs (1, against direct) and the part from self-learning (2, several hubs).

## Expected outcomes (before the run)
From strong maps: self-learning moved direct by −0.001 to +0.012 MRR on the real stages and the
chain by +0.004 to +0.011, while several hubs gained +0.04 to +0.08 over self-learned direct.
1. Several hubs beats plain direct by about as much as with self-learning (+0.04 to +0.08) and in
   nearly every direction.
2. Self-learning adds little on top of several hubs (about +0.01).
3. If instead self-learning adds as much as several hubs, the report keeps strong maps as the main
   pipeline; if it adds little, the report's main line is plain Procrustes with several hubs and
   strong maps a robustness section (the user's decision after this run).

## Cost
Four map evaluations per direction and fold (direct and 3 chains; the combinations are free), 11,400
in all; Procrustes fits are cheap, the CSLS evaluation over the 200k-word vocabularies is the cost.
The laptop's smoke test (2026-11-24, one under-500 and one well-resourced pair) took about 4 s per
map; the remote is about twice as fast. **Estimate: 3–5 h on the remote**, measured first with
`--limit 1`.

**Smoke test, check against the stages' caches:** on the well-resourced pair 99.3–99.8% of ranks
were identical; on the under-500 pair 72–79% (largest difference 5–9 ranks). Below 300 dictionary
pairs a Procrustes rotation is not unique (300 dimensions), so the GPU's SVD picks slightly
different maps from run to run (a known limitation of the stages' results, not of this run).

## Files
- `run.py`: the default runs every unit, resumable; `--limit N` the first N pairs per unit;
  `--units`; `--check` (against the stages' caches); `--results-only`; `--data-files` lists every
  input (for the remote copy).
- `results/`: `pooled.csv` (per direction and method), `summary.csv` (per unit and comparison),
  `check.csv`, `run_info.json` (commit, config, inputs' fingerprints).
- Tests in `tests/test_several_hubs_procrustes.py`.

## Results (run 2026-11-25 on the remote at b206a8e; `results/`)
All units complete (570 directions, all 5 folds), plus the 1,000-pair budget (`dc_b1000`, 40 pairs,
80 directions; run for the ContrastiveBLI baseline). MRR pooled over test words, paired bootstrap;
above / below: directions whose own interval excludes 0.

| plain Procrustes | under 500 (154) | 500–1,000 (250) | 1,000–3,000 (86) | well-resourced (80) | 1,000-pair budget (80) |
|---|---|---|---|---|---|
| direct MRR | 0.016 | 0.030 | 0.068 | 0.517 | 0.302 |
| several hubs MRR | 0.063 | 0.078 | 0.135 | 0.569 | 0.559 |
| several hubs − direct | +0.047 [+0.045, +0.048], 150 / 0 | +0.048 [+0.047, +0.049], 242 / 0 | +0.067 [+0.065, +0.070], 80 / 0 | +0.051 [+0.050, +0.053], 71 / 0 | +0.257 [+0.254, +0.259], 80 / 0 |
| several hubs − chain | +0.013, 50 / 0 | +0.016, 109 / 2 | +0.042, 66 / 1 | +0.056, 79 / 0 | +0.046, 76 / 0 |
| several hubs − combination | +0.017, 84 / 0 | +0.012, 122 / 1 | +0.020, 57 / 1 | +0.015, 46 / 0 | +0.047, 80 / 0 |
| chain − direct | +0.034, 134 / 0 | +0.032, 183 / 0 | +0.025, 56 / 3 | −0.005, 9 / 22 | +0.210, 80 / 0 |

**Self-learning's share** (strong maps' cache `55ca51e319` minus this run, same method):

| | under 500 | 500–1,000 | 1,000–3,000 | well-resourced |
|---|---|---|---|---|
| several hubs | +0.008, 45 / 6 | +0.003, 54 / 44 | +0.010, 33 / 12 | +0.002, 12 / 8 |
| direct | −0.000, 11 / 19 | +0.002, 45 / 51 | +0.005, 20 / 11 | +0.012, 35 / 2 |
| chain | +0.009, 51 / 2 | +0.005, 49 / 9 | +0.009, 25 / 3 | +0.006, 22 / 3 |

- Split of strong maps' headline (its several hubs minus plain direct: +0.055, +0.051, +0.077,
  +0.054): the hubs give +0.047, +0.048, +0.067, +0.051 of it; self-learning the rest.
- On the 1,000-pair budget, strong maps exists only for its 6 pilot pairs (12 directions): there
  self-learning lifts direct a lot (+0.240), several hubs by +0.006.

**Against the expected outcomes.**
1. Several hubs beats plain direct by about as much as with self-learning, in nearly every
   direction: yes (+0.047 to +0.067 on the natural units; no direction below).
2. Self-learning adds little on top of several hubs: yes (+0.002 to +0.010).
3. Hence the report's main line is plain Procrustes with several hubs, strong maps and
   ContrastiveBLI as checks (the user's decision, decisions.md 2026-11-25).
