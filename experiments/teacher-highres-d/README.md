# The chain as a teacher for well-resourced pairs (stage D)

**Question.** On phase 4's 40 pairs with natural dictionaries (stage D), the chain lost to direct
but averaging the two (the combination) beat direct on every pair. Does training the direct map
with the chain as a teacher, instead of averaging at search time, help too?

decisions.md 2026-11-18 (deferred item, now prepared). This README is written and approved before
any run. Results are descriptive.

## Design
- **Data:** stage D unchanged (`v1/phase6/d`): 40 pairs, natural dictionaries, 5 folds, test
  removal, gold.
- **Teacher ("teacherd", as stage 1):** the direct map is fitted on the direct dictionary with an
  added pull towards the chain's map; weight λ = α · n_direct / d (d = 300).
- **α:** chosen by cross-validation over source languages (leave one out), from
  {0.03, 0.1, 0.3, 1}. The first grid, {3, 10, 30} (stage 1's choice was 10), returned the chain's
  own map in the laptop smoke test on one pair (English–Arabic: all three α 0.773–0.774, chain
  0.773, direct 0.788): with stage D's large dictionaries the pull towards the chain dominates.
  The grid was moved down after that one-pair check, before any full run.
- **References in the same pass:** direct, chain, combination (identical to stage D's numbers).
- Reported: per stratum and pair, teacher − direct and teacher − combination, with standout pairs.

## Expected outcomes (written before any run)
- The teacher is at or slightly above direct (about +0.005), more in stratum (b) than (a),
  because the chain carries some useful signal there.
- It stays below the combination (+0.018 in (a), +0.056 in (b)): the teacher pulls one map
  towards the chain, while the combination keeps both maps' different errors.

## Compute
About 400 direction-folds × (one combined pass + 4 teacher maps) ≈ 45 min at 1.1 s per
evaluation, about 30 min with stage D's cache copied. Copying stage D's evaluation cache first skips the references:
`copy v1\results\natural-highres-d\evals.jsonl v1\results\teacher-highres-d\evals.jsonl`.

## Results (remote, 4390b56, 2026-11-19; `results/`)
References (direct, chain, combination) equal stage D's exactly (its cache, refreshed after the
English fix). α chosen by cross-validation over source languages: 0.3 for 15 languages, 0.1 for
7. Gains over direct, mean over pairs:

| stratum | chain | combination | teacherd 0.03 | 0.1 | 0.3 | 1 | **teacherd (cross-validated)** | teacher above / below direct |
|---|---|---|---|---|---|---|---|---|
| (a) high-resource | −0.018 | +0.018 | +0.003 | +0.002 | −0.003 | −0.009 | **−0.003** | 2 / 7 of 40 |
| (b) one weaker language | +0.007 | +0.056 | +0.011 | +0.022 | +0.027 | +0.023 | **+0.021** | 25 / 0 of 40 |

- The teacher helps where the chain carries some useful signal (stratum (b): +0.021, no
  direction below direct; largest Japanese–Kazakh +0.048, Estonian–Thai +0.044, Irish–Turkish
  +0.041), and is neutral to slightly negative where it doesn't (stratum (a): Arabic–English
  −0.025, Finnish–Chinese −0.024).
- It never beats the combination (0 of 80 directions): pulling one map towards the chain loses
  what averaging two maps with different mistakes gains.

**Against the expected outcomes:** at or slightly above direct, more in (b) than (a): yes, and
more in (b) than expected (+0.021 against about +0.005); below the combination: yes, everywhere.
