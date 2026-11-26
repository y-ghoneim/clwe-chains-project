# Chains on naturally scarce pairs (phase 6, stage 1)

**Question.** When a pair's direct dictionary is naturally small but one hub has large
dictionaries with both endpoints, does the chain through that hub beat direct alignment trained
on the small dictionary? And is any gain the hub's extra supervision (joined direct does as well)
or something composing maps adds?

Spec: `docs/superpowers/specs/2026-11-16-phase6-stage1-design.md`; plan:
`docs/superpowers/plans/2026-11-16-phase6-stage1.md`; decisions.md (2026-11-16, "what comes after
phase 4"). This README is written and approved before any run (CLAUDE.md). Stage 1 is the
**favourable case** (user): it is reported as such, not as a general result.

## Data
- **Pairs** (`pairs.csv`, prep/07, from PanLex sizes alone): a low-resource language (median
  dictionary with the 30 nodes under ~1,000 words, one hub with at least 2,000) paired with a node
  of another top-level family whose direct PanLex size (min distinct words) is below 500. The hub
  is the node maximizing min(|s–x|, |x–t|). 77 pairs, both directions evaluated (154):

  | language | pairs | hub | direct pairs, median (range) | s–hub pairs | hub–t pairs, median | joined, mutual best / all, median | eligible concepts |
  |---|---|---|---|---|---|---|---|
  | yo Yoruba | 20 | en | 591 (448–921) | 7,436 | 72,614 | 10,641 / 44,488 | 509–511 |
  | or Odia | 17 | hi | 263 (86–329) | 3,125 | 6,904 | 1,989 / 2,240 | 328 |
  | mg Malagasy | 16 | fr | 557 (289–627) | 4,713 | 21,802 | 6,061 / 9,949 | 277–278 |
  | mhr Meadow Mari | 13 | en | 575 (421–839) | 29,461 | 50,762 | 13,678 / 135,285 | 762 |
  | as Assamese | 7 | en | 478 (442–579) | 4,912 | 35,235 | 14,337 / 23,175 | 660 |
  | myv Erzya | 4 | en | 624 (596–670) | 4,758 | 44,461 | 7,494 / 21,786 | 591 |

  Dictionary sizes are distinct word pairs before test removal (`v1/phase6/dict_sizes.csv`); the
  run reports them per fold after removal.
- **Dictionaries** (prep/08): each edge trains on its own natural dictionary: every distinct
  in-vocab PanLex pair, plus English Wiktionary translations on edges with en. Nothing is shrunk by
  design.
- **Test words** (prep/07): the list's 883 concepts minus 42 capitalized-sense concepts (841). The
  six languages get list words exactly as the 30 nodes did (PanLex → Wiktionary → one-to-one →
  round trip). A pair's eligible concepts: both languages have a word, not identical strings.
  `analyses/2026-11-16-phase6-test-words/`.
- **Folds:** each pair's eligible concepts in 5 disjoint folds (seed 0); every concept is tested
  once. Per fold, a dictionary pair is removed from every edge if its source-side word is a test
  word of that edge's source language or its target-side word one of the target language
  (all casings): s–t loses both, s–hub the s words, hub–t the t words.

## Methods (per pair, direction and fold)
1. **Thin direct:** Procrustes on the s–t dictionary.
2. **Chain:** Procrustes on s–hub and on hub–t, composed.
3. **Chain via en** (secondary): the mg and or pairs, whose hubs are fr and hi (33 pairs).
4. **Joined direct:** Procrustes on the s–t dictionary plus the s–t pairs obtained by joining
   s–hub and hub–t on the hub word, filtered to **mutual best** (each word the other's
   most-connected partner, counting distinct linking hub words; ties kept); **unfiltered** as a
   check.
5. **Chain-as-teacher:** the orthogonal W minimizing ‖XW − Y‖² + λ‖W − W_chain‖² on the direct
   dictionary, λ = α · (direct training pairs), α ∈ {0.1, 0.3, 1, 3, 10}; α chosen by
   cross-validation leaving one low-resource language out (ties to the smaller α). Every α is
   reported too.
6. **Combination:** the equal-weight average of thin direct's and the chain's CSLS scores.

## Evaluation
- CSLS (k = 10) over the target's full vocabulary, half precision with exact rescoring. Gold:
  the target's list word, every target word sharing a PanLex meaning with the source word, all
  casings. Secondary: the list word alone; P@1.
- Per pair-direction: the 5 folds' test concepts pooled; ΔMRR = method − thin direct, with a
  paired bootstrap 95% interval over concepts (1,000 resamples, seed 0).
- **Per language:** mean ΔMRR over its pairs with a 95% interval from a nested bootstrap (pairs
  within the language, then concepts within each pair, both directions together). Overall: the
  same over languages → pairs → concepts, reported but not part of the rule.
- Share of the gap recovered: (method − thin) / (joined mutual best − thin), where joined beats
  thin.

## Success rule (withdrawn before any result)
As approved: the plain chain beats thin direct in at least 2 of the 6 languages, each with its 95%
interval above 0. **Withdrawn by the user on 2026-11-17, before any result was viewed**: the final
review found that the per-language interval resampled concepts independently per pair although a
language's pairs share their test words and the s–hub map, so it was too narrow. Results are
reported descriptively instead (per pair-direction ΔMRR with its paired bootstrap interval; per
language and overall, the mean over pairs and how many pair-directions lie above or below 0), and
interpreted afterwards. The plain chain stays the main comparison; chain-as-teacher, the
combination, joined direct and the en chain are secondary.

## Changes after approval, before any result
- **Chain-as-teacher rescaled** (user, 2026-11-16; final review I2): with λ = α · n_direct the
  penalty outweighs the data term by about d = 300, so every α gives essentially the chain. A
  second family, `teacherd`, uses λ = α · n_direct / d with the same α grid and cross-validation;
  the original `teacher` scores are kept and reported as the pre-registered version. The teacher
  expectation below applies to `teacherd`.
- **Provenance:** stage 1's evaluations were made at commit 830a312 (the remote's checkout);
  `teacherd` and the results files come from a later commit, which also fixes a results-writing
  crash (the cross-validated α's label). `run_info.json` lists `eval_commits` from `run.log`
  (830a312 is recorded by hand: that run predates the log line).

## Expected outcomes (written before any run)
- **Chain vs thin direct (the main comparison): expected to win.** The chain's weak step (s–hub) has 8–50× the
  pairs of the direct dictionary (Meadow Mari 29,461 against a median 575, Malagasy 4,713 against
  557), and 90–920 pairs are few for Procrustes between distant languages. Largest gains expected for mhr and yo (largest hub dictionaries),
  smallest for or (hi–t dictionaries are the smallest, 6,904 median) and mg (fewest test words,
  noisiest). If the chain does **not** beat thin direct here, chains don't help even where the
  hub has far more supervision: vector quality and the error added by a second step outweigh it.
  That would be a strong negative result for the thesis question.
- **Joined direct vs chain: expected about equal, or joined slightly ahead.** Both use the hub's
  supervision; joined direct trains one map on it, avoiding the second step's error, but its
  pairs are noisier. If joined ≈ chain, the gain is supervision, not composing maps (the
  reframed question's answer); if the chain clearly beats joined, composing adds something
  joining loses (e.g. the joined dictionary's noise); if joined clearly wins, pivoting
  dictionaries is the better way to use a hub. Unfiltered joined expected below mutual best.
- **Chain-as-teacher: expected at least as good as the better of thin and chain**, with a mid
  or large α chosen (the chain carries most of the signal); the smallest α should be close to
  thin, the largest close to the chain.
- **Combination: expected between thin and chain, closer to the chain**, possibly above both
  where thin direct is not much worse than the chain.
- **Chain via en (mg, or): expected below the best-hub chain** for or (or–en is 1,481 pairs
  against or–hi 3,125) and mixed for mg (mg–en 1,233 against mg–fr 4,713, but en–t
  dictionaries are larger).
- **Secondary measure (list word only)** expected to move with the main gold.

## Compute
About 7,300 CSLS evaluation equivalents: 154 pair-directions × 5 folds × 9 (thin + chain +
combination in one pass ≈ 2, joined 2, teacher 5), plus 330 for the en chain. At the measured
3.9 s per evaluation (laptop RTX 4050) ≈ 7.9 h; the RTX 4500 is measured by a timing run
(`--limit 2`, the first two Yoruba pairs, 180 evaluations) before the full run.

## Running
On the remote PC, per `docs/remote.md`: `python experiments\scarce-natural-s1\run.py --limit 2`,
then without `--limit`. The run refuses result-affecting uncommitted changes, caches every
evaluation and resumes after an interruption; `results/` is rebuilt from the cache. Results
files: `metrics.csv`, `pooled.csv`, `languages.csv`, `overall.csv`, `alpha_cv.csv`, `gap.csv`,
`dict_sizes.csv`, `success.json`, `run_info.json` (commit, configs, input fingerprints, GPU,
seconds per evaluation).

## Results (run 2026-11-17 on the remote RTX 4500)
Evaluations: thin, chain, combination, joined, en chain and `teacher` at 830a312; `teacherd` and
the results files at 66200d0 (decisions.md 2026-11-17). 11,880 cache records, 1.11 s per CSLS
evaluation; every input fingerprint in `run_info.json` matches the laptop's prep outputs. The
830a312 run's own summary line (the withdrawn rule, with its too-narrow intervals) read "chain above
thin direct in 6 of 6 languages"; it is kept out of `results/`.

**The chain beats thin direct, consistently, but from a very low base.** Mean pooled MRR per
language over its pair-directions (`pooled.csv`):

| language | thin direct | chain | joined (mutual best) | teacher, d-scaled (CV α) | combination |
|---|---|---|---|---|---|
| as | 0.024 | 0.069 | 0.055 | 0.075 | 0.075 |
| mg | 0.024 | 0.078 | 0.069 | 0.080 | 0.073 |
| mhr | 0.017 | 0.048 | 0.044 | 0.051 | 0.045 |
| myv | 0.015 | 0.023 | 0.019 | 0.025 | 0.024 |
| or | 0.006 | 0.062 | 0.044 | 0.062 | 0.039 |
| yo | 0.015 | 0.031 | 0.023 | 0.033 | 0.032 |

ΔMRR against thin direct, mean over languages (each pair weighted equally), and the
pair-directions (of 154) whose paired bootstrap interval lies above / below 0 (`overall.csv`):

| method | ΔMRR | above / below |
|---|---|---|
| chain | +0.035 | 134 / 0 |
| joined, mutual best | +0.025 | 118 / 0 |
| joined, unfiltered | +0.023 | 117 / 0 |
| teacher, d-scaled, CV α (α = 10, or 3 for or) | +0.037 | 138 / 0 |
| teacher, pre-registered λ = α·n (CV α = 0.1) | +0.037 | 138 / 0 |
| combination | +0.031 | 146 / 0 |
| chain via en (mg, or only; 66 pair-directions) | +0.043 | 55 / 0 |

- **Every method helps and none ever hurts:** no pair-direction is significantly below thin
  direct for any method. The chain's gain is largest for Odia (+0.056, the thinnest direct
  dictionaries: median 263 pairs) and Malagasy (+0.054), smallest for Erzya (+0.008; 4 pairs, 4 of 8
  pair-directions above 0) and Yoruba (+0.016; 26 of 40). P@1: thin 0.007, chain 0.028.
- **Absolute accuracy is very low for every method.** Thin direct finds the right word first 0.7% of
  the time, the chain 2.8%; MRR stays under 0.08 everywhere. These are distant pairs with weak
  vectors (Meadow Mari and Yoruba have 150k-word vocabularies), noisy many-to-many dictionaries and
  retrieval over the target's full 200k vocabulary. The gains are real and consistent, but they are
  gains between maps that mostly fail.
- **Chaining beats the pivoted dictionary.** The chain is above joined direct (mutual best) in 126 of
  154 pair-directions (mean +0.010), and above it in every language; the chain recovers more than
  the joined dictionary's whole gain (median share 1.1–1.7 per language, `gap.csv`). Expected was
  "about equal or joined slightly ahead": **not as expected**. The gain isn't only the extra
  supervision as joining uses it; composing two maps through the hub adds to it, or joining's noise
  (polysemous hub words) costs more than a second step does. Joined direct is an imperfect control
  (it keeps only pairs that line up through one hub word), so this separates the two explanations
  only partly.
- **Chain-as-teacher:** the d-scaled family behaves as designed: α = 0.1 stays near thin (+0.005),
  α = 1 is +0.031, α = 3–10 reach +0.038, just above the chain; the cross-validation picks α = 10
  (3 for Odia). The pre-registered family is the chain for every α (+0.035 to +0.037), as the final
  review predicted. The teacher adds at most ~0.003 over the chain: the chain carries the signal.
- **Combination:** below the chain on average (+0.031) but the most consistent (146 of 154 above 0);
  above the chain for Assamese, Erzya and Yoruba, well below it for Odia (0.039 vs 0.062), where thin
  direct is the weakest and averaging with it hurts.
- **Chain via en:** for Odia, en is as good as its best hub hi (+0.058 vs +0.056); for Malagasy, en is
  much worse than fr (+0.027 vs +0.054).
- **Direction:** gains are larger from the low-resource language into the node for Assamese and Odia
  (e.g. or +0.070 vs +0.042), about equal elsewhere.

**Against the expected outcomes:** chain beats thin direct, yes (largest for or and mg, as expected;
mhr is mid-range, not among the largest as predicted); joined ≈ chain, **no** (the chain is ahead);
teacher ≥ the better of thin and chain, yes for the d-scaled family, by a hair; combination between
thin and chain, yes on average; en chain below the best hub for or, **no** (equal), worse for mg, yes.
