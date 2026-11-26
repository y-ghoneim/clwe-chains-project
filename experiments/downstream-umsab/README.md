# Cross-lingual sentiment transfer through direct and chained maps (downstream, part 1: UMSAB)

**Status: APPROVED by the user, 2026-11-19 (Portuguese left out). RUN 2026-11-19 at commit
afe1979; see "Results" near the end.** Results are descriptive (decisions.md 2026-11-16: no success rule). The user's view:
gains in only some cases, not overall, would already be a good result; see "Where the chain
helps" below for how they are reported.

**Question.** Brief §1 asks whether a chain "improves real cross-lingual transfer". Word-translation
tests (BLI) showed that with large dictionaries the chain is level with, or slightly below, the direct
map (stage D: −0.005 MRR over 40 pairs). Does that carry over to a real task? A classifier is trained
on labelled tweets in one language and applied to tweets in another language whose vectors are moved
into the first language's space by the direct map or by the chain. Second question: does a map's
word-translation score predict its sentiment score (Glavaš et al. 2019 found BLI often doesn't)?

**Framing (user, 2026-11-19).** A language X with little labelled data is moved into the space of a
language Y with plenty, where a classifier is trained: X's text is mapped into Y's space (map X → Y),
a classifier trained on Y's labelled tweets (in Y's native space) is applied to the mapped X tweets.
This is the same direction as BLI's source → target for the pair (s = X, t = Y), and both directions
of every pair are run. Caveat: in UMSAB every language has the same amount of labelled data, so
"Y has plenty" is the premise here, not a fact; X's own training tweets give the in-language ceiling
and are not used for transfer.

AUS's own sentiment test is not rerun (decisions.md 2026-11-19).

## Data
- **UMSAB** (`raw/downstream/umsab`; tweets, 3 classes 0/1/2, balanced): 613 / 108 / 290 per class in
  train / validation / test, so **1,839 / 324 / 870** tweets per language. Tweets in every language,
  but not the same sources: see "Data findings from the review" below.
- **Languages: the 7 UMSAB languages that are graph nodes: en, ar, de, es, fr, hi, it.** Portuguese
  (also in UMSAB) is not a node: it has no shared-list words, so no BLI test and no hub rule. It
  could be added for the downstream numbers alone; left out unless you want it. This corrects my
  earlier "8 languages, 56 pairs".
- **Pairs: all ordered pairs, 7 × 6 = 42 directions (21 unordered), fixed now.** Strata, by the
  project's rule (different top-level Glottolog family): **distant = the 6 pairs with ar**
  (12 directions; ar is Afro-Asiatic, the others Indo-European); **same family = the other 15**
  (30 directions; includes en–de, es–it–fr and the distant-script en–hi). Only en–ar is also one
  of phase 4's 40 pairs. (Updated 2026-11-19: the 12 directions with hi are a separate stratum,
  leaving 5 distant and 10 same-family pairs; see "Data findings from the review".)
- **Text preparation (fixed):** lowercase; `@user` and `http…` placeholders, `RT` and the `#` sign
  dropped; tokens = runs of letters/digits (Unicode); each token matched to the language's
  200k-word vocabulary through `text.normalize` and `vocab_index`; out-of-vocabulary tokens skipped.
  A tweet's feature = the mean of its tokens' vectors (unit → centre → unit normalized, as the
  rest of the project), then unit length. Tweets with no in-vocabulary token get a zero vector;
  their count is reported per language. **Then centred on the language's own mean** (the mean
  over its training tweets that have a matched token; no labels used); tweets with no matched
  token stay at zero. Maps are applied to the centred features (added after the first smoke run:
  see "Changes after building").

## Maps
- **Direct:** Procrustes on the pair's **full natural dictionary** (PanLex in-vocab pairs plus
  English Wiktionary when en is an end), X → Y. **Chain:** the same through the pair's hub, one
  Procrustes per step on that step's full natural dictionary. Hub rule and dictionaries as stage D
  (hub = the node maximizing min(|s–x|, |x–t|), never an endpoint).
- No test removal: the downstream test text differs from the dictionaries' words, and test
  tweets carry no dictionary labels. (BLI's per-fold removal does not apply.)
- The combination (average of CSLS scores) is a retrieval method and is left out here.

## Classifiers
- Features as above; **logistic regression (headline)** and **a one-hidden-layer MLP
  (secondary, the AUS comparison point)** (sklearn). Reason (user, 2026-11-19): logistic regression
  is deterministic, has fewer knobs and is more sensitive to the geometry of the mapped space; the
  MLP can fit quirks of Y's space that don't survive the map. **Seeds:** 3 for the MLP.
- **Fixed in advance:** logistic regression C ∈ {0.001, 0.01, 0.1, 1, 10, 100} (widened, see
  "Changes after building"), chosen once per labelled language
  Y on **Y's validation split**, never on X's data. The MLP has one fixed configuration (hidden
  size 256, early stopping on Y's validation split), no grid.
- Each Y's classifier is trained once and applied to every X ≠ Y and method (only the test features
  differ).

## Reference points
- **Floor:** majority class (macro-F1 0.167, accuracy 0.333 on this balanced test set).
- **No map:** the Y classifier applied to X's vectors unmapped (the two spaces are unrelated); shows
  that the maps carry the transfer.
- **In-language ceiling:** a classifier trained on X's own training tweets, tested on X's test tweets
  (same features and models).

## Evaluation
- **Metrics:** macro-F1 (headline, logistic regression) and accuracy, per direction, method and
  model; the MLP's value is the mean over its 3 seeds and is reported as secondary, never pooled with
  logistic regression.
- **Chain − direct** per direction, with a paired bootstrap over the 870 test tweets (1,000
  resamples; each resample's F1 averaged over the MLP seeds). Per pair: the mean of its two
  directions. Overall and per stratum: mean over pairs, count of directions above / below 0
  (interval excludes 0), and an interval from a bootstrap that resamples pairs. Directions
  share tweets and dictionaries, so counts are descriptive only.
- **BLI link (brief §5.8):** BLI MRR for these 42 directions comes from a companion run of stage D's
  runner on these pairs (new stage `u`: same dictionaries, test words, folds and CSLS evaluation
  as stage D; about 42 × 5 × 2 = 420 evaluations, ~10 min). Reported: (i) Spearman between BLI MRR
  and macro-F1 over the 42 directions × 2 methods; (ii) the one that matters, Spearman between
  the chain's advantage in MRR and its advantage in macro-F1 over the 42 directions. (i) mostly
  reflects how easy a pair is; (ii) asks whether a better map means a better classifier.
- **Where the chain helps (pre-declared, so that wins in some cases are explained, not picked
  out afterwards).** Per direction, the chain's advantage in macro-F1 is set against
  r = log2(narrowest chain step / direct dictionary), the variable of RQ2's gain curve
  (`analyses/2026-11-17-phase6-gain-curve/`): a Spearman correlation over the 42 directions with a
  bootstrap interval over pairs, and the binned means. Also reported: the directions with the
  largest chain advantage (standout cases, with their r), and the number of directions above 0
  next to a nominal floor (2.5% of the directions, about 1 of 42). The floor is not a null
  expectation: the tweet bootstrap covers test sampling only, not the fit of the classifier or the
  maps, and directions with the same source share their test tweets, so more "above" directions
  than the floor can occur with no real effect. Also written: `binned_by_r.csv` (mean advantage by
  r: ≤ −1, −1–0, 0–1, 1–2, > 2) and `standout.csv` (the 5 largest and 5 smallest advantages).
  Intervals for the stratum means and the Spearman correlations resample pairs; each language is in
  six pairs, and every distant pair contains ar, so these intervals are narrower than a
  language-level resample would give.
- **Precision to expect:** macro-F1 on 870 tweets has a standard error of about 1.6 points; paired
  chain − direct differences about 1 point. Effects under about 2 points will not be resolved.

## Expected outcomes (written before any run; guesses, to be marked wrong where wrong)
1. **No map ≈ floor** (macro-F1 near 0.2–0.35), mapped transfer clearly above it for most pairs.
2. **Ceilings** (averaged fastText vectors, 3 classes) around 0.45–0.60 macro-F1; transfer below the
   ceiling by 5–15 points, less for the same-family pairs. Same-family pairs transfer better than
   the 12 ar directions.
3. **Chain vs direct: no resolvable difference for most pairs** (BLI differences are 0–0.02 MRR
   with these large dictionaries). I expect fewer than a fifth of directions to have an interval
   off zero, with the chain more often below than above.
4. **BLI link: (i) positive, ρ about 0.3–0.6** (pair difficulty); **(ii) weak, near 0** (the
   differences are below the noise).
6. **Where the chain does win, it should be where direct is smallest relative to the chain's
   steps** (RQ2's gain curve), not scattered; a positive Spearman between r and the chain's
   advantage. If wins are scattered with no relation to r, they are consistent with noise.
5. A direction where the chain is clearly better downstream while worse in BLI, or the reverse, would
   be reported as a case where BLI misleads.

## Changes after building (user, 2026-11-19; before any real run)
A smoke run on the first 3 pairs (6 directions; `--limit 3`, outputs not kept) showed two things,
both about preprocessing, not about the question:
- **Centring.** Without it the classifier sent most tweets to one class on a balanced test set
  (for example de → ar with the direct map: 653 of 870 tweets in class 1), so chain − direct
  swings of 10–14 F1 points came from a shift in the mapped features' overall position, not from
  the maps (BLI gaps for these pairs are about 0.01 MRR). Centring each language on its own
  unlabelled training mean gave near-balanced predictions and swings of −3 to +3 points, with
  in-language ceilings unchanged. It is standard preprocessing and uses no labels.
- **C grid.** Logistic regression picked C = 0.1, the smallest value, for every language, so the
  grid was widened downward. After centring the best C is 0.001–0.01 for most languages; validation
  F1 falls again below 0.001 (checked down to 1e-5), so the optimum is inside the grid.
- I saw the smoke run's test-set F1 for those 3 pairs. The two changes rest on the class skew and
  the validation curve, not on which method won. The expected outcomes above were written before
  the smoke run and are unchanged.
- The bootstrap intervals cover test-tweet sampling only. Each language has a single fitted
  logistic regression, so the variability of fitting it is not in the intervals (the MLP's three
  seeds partly show it); the write-up says so.

## Data findings from the review (2026-11-19, before any real run)
A reviewer agent went through commit 40c192f (no correctness bug found in the code); its data
checks, which I re-checked:
- **UMSAB's Hindi is romanized, not Devanagari.** 1 of 2,709 Hindi train + test tweets contains a
  Devanagari letter; `stats.csv` shows 0% of the test letters in Devanagari (`native_script_share`;
  1.0 for en, de, es, fr, it, 0.92 for ar). Tweets read like "CONGRESS na ye party kabhi bani hoti".
  The Hindi vectors and dictionaries are Devanagari, so for the 12 directions that involve hi the
  maps are applied to text they were mostly not fitted on (only the Latin-script words that
  happen to be in the Hindi vocabulary and dictionaries), and the BLI companion measures
  Devanagari words. Hindi's in-language ceiling (0.34–0.35 against 0.55–0.62 elsewhere) fits this.
  **Decided (user, 2026-11-19): keep hi as a separate, flagged stratum** (`romanised_hindi`: the
  12 directions with hi, including ar–hi). Those directions are run and written to `transfer.csv`
  and `delta.csv`, and get their own rows in `summary.csv`; they are left out of the distant and
  same-family rows, the `all_but_hindi` row, `binned_by_r.csv`, `standout.csv` and the BLI and r
  correlations. So the strata have 5 distant pairs (10 directions: ar with de, en, es, fr, it) and
  10 same-family pairs (20 directions); the 42-direction counts and the "6 pairs with ar" above
  are replaced by these wherever a stratum or an analysis is reported. Hindi's numbers are not
  read as evidence about chains.
- **Italian is dominated by one topic:** 797 of its 1,838 training tweets (43%) mention "monti"
  (the Monti government), so Italian classifiers learn topic as much as sentiment. Not changed;
  stated in the write-up.
- Smaller, fixed: zero-width and bidi format characters no longer glue onto words (15 of 41k Arabic
  tokens); the zero-width (non-)joiner stays inside words; `native_script_share` added to
  `stats.csv`; binned and standout tables, `config.yaml`, a fingerprint of the BLI companion's
  results in `run_info.json`, runner tests and a Spearman test that checks whole pairs are
  resampled were added.

## Results (run 2026-11-19, commit afe1979; `results/`)
Main analyses use the 30 directions without Hindi (15 pairs: 5 distant, 10 same family).
Logistic regression (LR) is the headline; the MLP (mean of 3 seeds) is secondary.

| mean macro-F1, LR | direct | chain | no map | X's own ceiling |
|---|---|---|---|---|
| distant (10 directions) | 0.525 | 0.538 | 0.306 | |
| same family (20) | 0.532 | 0.536 | 0.330 | |
| all but Hindi (30) | 0.530 | 0.537 | 0.322 | 0.603 |

- **Transfer works:** either map recovers most of the in-language score (0.53 of 0.60, from 0.32
  with no map); direct beats no map in 30 of 30 directions.
- **Chain − direct (`summary.csv`):** LR +0.0075 [−0.0003, +0.015], 3 directions above 0 and 1
  below (nominal floor 0.75); MLP +0.0035 [−0.003, +0.010], 7 above and 2 below. Distant: LR
  +0.013 [−0.008, +0.029]; same family: +0.005 [−0.001, +0.010]. In BLI on the same 42 directions
  the chain was slightly below direct (−0.0086 MRR, BLI companion `overall.csv`), so the small
  BLI losses do not carry over.
- **Where the chain helps (`correlations.csv`, `binned_by_r.csv`):** the chain's advantage rises
  with r; Spearman ρ = 0.54 [0.12, 0.81] for LR, 0.41 [−0.01, 0.70] for the MLP. Mean advantage
  (LR) by r: ≤ −1: −0.028; −1–0: +0.002; 0–1: +0.007; 1–2: +0.026; > 2: +0.014. The two end bins
  are one pair each (r ≤ −1 is ar–en via fr).
- **BLI link:** F1 against MRR over directions × {direct, chain}: ρ = 0.08 [−0.32, 0.50] (MLP
  0.11); the chain's F1 advantage against its MRR advantage: ρ = −0.07 [−0.46, 0.35] (MLP −0.19).
- **Standouts (`standout.csv`):** it → ar via en +0.055 [0.031, 0.081] for the chain while BLI
  rates the chain clearly worse there (−0.037 MRR [−0.056, −0.018]); es → ar and fr → ar
  (MLP) show the same pattern. Largest loss: en → ar via fr −0.035 [−0.060, −0.010] (BLI −0.018).
- **Romanised Hindi (12 directions, own stratum; not evidence about chains):** Hindi ceiling 0.35;
  mean direct 0.372, chain 0.366, no map 0.334; LR chain − direct −0.006 [−0.023, +0.011], 2
  above and 2 below.
- **LR's C** chose 0.001 (the grid's lowest value) for ar, de and fr, 0.01 for en, es, it and 1
  for hi.

**Expected outcomes, marked:**
1. No map near the floor, mapped clearly above: **right** (0.31–0.33; 30 of 30 above).
2. Ceilings 0.45–0.60: **about right** (0.59–0.62, at or just above the top). Transfer 5–15
   points below the ceiling: **right** (about 7). Same family transfers better than the ar
   directions: **wrong** (no difference).
3. Under a fifth of directions resolved, chain more often below: LR count **right** (4 of 30),
   sign **wrong** (3 above, 1 below); MLP **wrong** on both (9 of 30; 7 above).
4. (i) F1 vs MRR ρ about 0.3–0.6: **wrong** (about 0.1, interval includes 0). (ii) near 0:
   **right**.
5. A direction where BLI misleads: **found** (it → ar; es → ar and fr → ar with the MLP).
6. Wins related to r, positive Spearman: **right** (ρ = 0.54 LR; the MLP's interval just
   includes 0).

**Reading.** With full natural dictionaries the chain is a safe substitute for the direct map
downstream (level to slightly ahead), and where it gains follows RQ2's variable r. BLI does not
predict which map transfers better. Distant vs same family says little here: ar is the only
distant language.

**Caveats.** 15 pairs, every distant pair shares ar, and intervals resample pairs, so the
correlation intervals are optimistic; the tweet bootstrap covers test sampling only, not the fit
of the classifier or the maps; Italian is topic-dominated (43% "monti") and it → ar is the top
standout; the downstream maps use the full dictionaries while the BLI maps drop each fold's test
words; RQ2's break-even near 10,000 pairs was itself found post hoc.

## Compute and build
Laptop only; the transfer step is cheap (classifiers are trained once per language, the test features
need one 300 × 300 multiplication). The cost is building the data: UMSAB tables, dictionaries for 21
pairs plus hubs (prep stage `u`), map refits, runner and tests. Estimates are the assistant's; see the
plan in the message that accompanies this README.

## What comes next (separate READMEs, each approved before its run)
2. **AfriSenti (yo, sw)** tested with UMSAB languages as the labelled side; the real low-resource case.
3. **WikiANN NER**: covers the project's own pairs.
4. **Tier 2** (add-on, user 2026-11-19): the same sentiment pairs with the direct dictionary shrunk
   to fixed budgets; artificial scarcity, own folder and report section, budgets fixed before its
   run. Its BLI companion comes from the controlled-scarcity experiment (decisions.md 2026-11-19).
