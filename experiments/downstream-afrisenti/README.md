# Sentiment transfer into Yoruba, Swahili and Amharic through direct and chained maps (downstream, part 2: AfriSenti)

**Status: APPROVED by the user, 2026-11-19, with Amharic added at the user's request, Hindi left
out as the labelled side and the reverse directions as a secondary result. RUN 2026-11-19 at commit ae489e8; see "Results".** Results are
descriptive (decisions.md 2026-11-16: no success rule). Gains in only some cases count as a good
result (user, 2026-11-19).

**Question.** Part 1 (UMSAB, `experiments/downstream-umsab/`) found the chain level with direct or
slightly ahead on well-resourced pairs, with its advantage rising with r (RQ2's variable: log2 of
the chain's narrowest step over the direct dictionary). Part 2 asks the same question where RQ2 says
the chain should matter: a low-resource language whose direct dictionaries are small. Yoruba (yo)
has 586–1,652 direct dictionary pairs with the UMSAB languages other than English, against about
7,400 on its English step; Swahili (sw) has about 12,000 with French and Spanish and 35,000 with
English, so it is the moderate-resource contrast. Amharic (am, added by the user) has small
dictionaries on every side (about 1,800–2,300 pairs with each UMSAB language and with English), so
its chain has small steps too: r near 0, the RQ1-like case of equal supervision.

**Framing (user, 2026-11-19).** As in part 1: X = yo, sw or am is the low-resource side; its test
tweets are mapped into the space of Y, a UMSAB language with labelled tweets, and classified by Y's
classifier. Caveat: AfriSenti's Yoruba has more labelled tweets than each UMSAB language (see Data),
so "X has little labelled data" is again the premise, not a fact; X's own training tweets give the
in-language ceiling only.

## Data
- **AfriSenti** (Muhammad et al. 2023, SemEval-2023 Task 12): tweets labelled positive / neutral /
  negative, the same three classes as UMSAB. Languages used: **yo and sw** (decisions.md
  2026-11-19, downstream scope) **and am** (added by the user, 2026-11-19; the brief lists am).
  Amharic is written in Ethiopic script, as are its fastText vectors; the data check reports the
  script share. Source: the task's GitHub repository (train / dev / test TSV
  files); licence and exact sizes checked on download and logged. Approximate sizes from the
  paper, to be confirmed: yo about 8,500 / 2,100 / 4,500 tweets, sw about 1,800 / 450 / 750,
  am about 6,000 / 1,500 / 2,000.
  Unlike UMSAB, the classes are not balanced.
- Downloaded into its own empty folder under `raw/downstream/afrisenti/`; Python that reads it runs
  with `-I`.
- **Labelled side Y: en, ar, de, es, fr, it**, with the part 1 UMSAB features and classifiers
  (rebuilt with the same settings; deterministic). **Hindi is left out as Y** (user,
  2026-11-19): its classifier was trained on romanised tweets while the maps go into the
  Devanagari space.
- **Pairs: X ∈ {yo, sw, am} × Y ∈ {en, ar, de, es, fr, it} = 18 directions, fixed now.** Hubs by the
  project's rule (the node maximizing min(|s–x|, |x–t|), never an endpoint), from a new prep
  stage `af` built like stage `u` (prep/07 with a pairs spec, prep/08). The yo pairs with ar, de,
  es, fr, it are already pairs of stages 1, A and A2 (hub en); the same sources and code should give
  the same dictionaries, checked when built. yo–en and the sw and am pairs are new. By the
  project's family rule every direction is distant except am–ar (both Afro-Asiatic).

## Method (as part 1 unless stated)
- **Text and features:** the part 1 tokenizer (`clwe_chains.downstream.tokenize`) and features:
  mean of normalized vectors over matched tokens, unit length, centred on the language's own
  training-tweet mean (no labels used).
- **Data check before any transfer number is seen:** per language and split, tweets with no
  matched token, mean matched tokens, the share of letters in the expected script (Latin for yo
  and sw, Ethiopic for am), and for yo the share of matched tokens carrying tone marks
  or dots. Yoruba is written both with and without diacritics and the fastText vocabulary mixes
  both; both forms are matched as they are, without stripping. If matched tokens are very few
  (say under 3 per tweet, or over 20% of tweets empty), the assistant reports it and the user
  decides before the run. Also reported, not filtered: the share of X's tokens that are English
  words (code-mixing is common in AfriSenti), measured as tokens found in the English vocabulary's
  top 50k and not in the X side of X's dictionaries.
- **Maps:** direct and chain through the hub, Procrustes on the full natural dictionaries of
  stage `af` (no test-word removal; the downstream test data is different text). No map as a
  reference.
- **Classifiers:** Y's logistic regression (C chosen on Y's validation macro-F1, grid
  {0.001 … 100}) as the headline; the MLP (3 seeds) as secondary. X's own classifiers, trained on
  AfriSenti train with C chosen on AfriSenti dev, give the ceiling.
- **Reference points:** no map; the majority-class floor computed from X's test labels (macro-F1 of
  always predicting the most common class, which is not 1/6 here); X's own ceiling.
- **Evaluation:** macro-F1 per direction; chain − direct with a paired bootstrap over test tweets
  (1,000 resamples). Summaries per X language (yo: 6 directions; sw: 6), not pooled into one mean,
  because yo, sw and am are three regimes (yo: small direct, large chain steps; sw: large
  direct; am: small everywhere). Spearman of the chain's advantage against r over the 18
  directions, reported with the caveat that there are only 18 and they share three test sets.
- **BLI link:** a BLI companion run on stage `af` (stage D's runner, as part 1; 18 pairs × 2
  directions, minutes). Reported as in part 1: the chain's F1 advantage against its MRR advantage.
- **Secondary (user, 2026-11-19):** the reverse directions Y → X (UMSAB test tweets mapped into
  X's space and classified by X's own classifier), 18 more directions, reported separately and
  never pooled with the main ones.
- **Precision:** about 4,500 yo test tweets give paired differences of roughly ±0.5–1 point; about
  750 sw tweets roughly ±1.5–2 points; about 2,000 am tweets roughly ±1 point.

## Expected outcomes (written before any run; guesses, to be marked wrong where wrong)
1. **Transfer is weak in absolute terms:** UMSAB and AfriSenti differ in topic, country and class
   balance, and Yoruba's maps are poor (BLI near the floor in stages 1 and A). Transfer into yo a
   few points above no map; into sw and am clearly above no map; all well below their ceilings
   (ceilings around 0.45–0.60).
2. **yo: the chain beats direct** in most of the 5 directions with a small direct dictionary (ar,
   de, es, fr, it; r between about 2.2 and 3.7, from 586–1,652 direct pairs against 7,436 on
   the yo–en step), by 1–4 points, in line with RQ2's BLI gains; at
   least 2 of the 5 resolved above 0. yo → en (large direct, negative r): no gain or a loss.
3. **sw: no resolvable difference** in most directions (dictionaries of 12,000+ pairs, past RQ2's
   break-even near 10,000), as in part 1.
4. **am: no resolvable difference, or the chain slightly below** (direct and chain steps are
   about equally small, r near 0; RQ1's equal-supervision case, where the chain loses).
5. **The chain's advantage rises with r** across the 18 directions (positive Spearman), mostly
   because yo has high r and sw and am low.
6. **BLI and downstream differences agree in sign for yo** (large BLI gains there); for sw, as in
   part 1, little relation.

## Data findings while building (2026-11-19, before any transfer number was computed)
- **Source:** GitHub `afrisenti-semeval/afrisent-semeval-2023`, `data/{yor,swa,amh}/`, commit
  5aec3cf, recorded in `raw/downstream/afrisenti/SOURCE.txt`. No licence file in the repository.
  Sizes (train / dev / test): yo 8,522 / 2,090 / 4,515; sw 1,810 / 453 / 748; am 5,984 / 1,497 /
  1,999.
- **Stage `af` (prep/07, prep/08):** the yo dictionaries are identical to stages 1, A and A2. Hubs:
  en for every pair except yo–en (es), sw–en (it) and the am pairs (fr; am–fr via en). Amharic's
  hub is chosen on PanLex word counts, where fr (1,522) edges out en (1,520), although with
  English Wiktionary am–en has more pairs (3,330 against 2,110 for am–fr).
- **Correction to the Question and prediction 3:** sw is not large-direct everywhere. Direct pairs
  are sw–ar 4,612 (r ≈ 2.9, below RQ2's break-even, like yo), sw–de 13,070, sw–es 11,858, sw–fr
  12,153, sw–it 22,376, sw–en 35,235. Prediction 3 is read as covering sw → de, es, fr, it, en;
  for **sw → ar the expectation is that of yo** (chain ahead). Written before any transfer or BLI
  number for these pairs was computed.
- **Data check (`v1/downstream/afrisenti/stats.csv`):** passes the README's rule. Matched tokens
  per tweet: yo 15–17, sw 15–16, am 12; tokens matched 79–80% (yo), 90% (sw), 82–84% (am); at most
  9 empty tweets per split. Script share: yo, sw 100% Latin; am 98% Ethiopic (91% on test).
  Matched yo tokens with diacritics: 49–52%. English-looking matched tokens: yo 11–12%, sw 8%,
  am 2%. (As computed by prep/14: matched tokens whose surface form, lowercased, is
  among the English vocabulary's top 50k and that are not PanLex words of the language, over all
  matched tokens; the README's plan said the stage `af` dictionaries.)
- **Amharic's test labels are shifted:** negative / neutral / positive = 1,337 / 224 / 438 on
  test against 1,548 / 3,104 / 1,332 on train (dev like train), and every test tweet is wrapped in
  quotes. The test split was evidently collected or formatted apart. Chain − direct stays a paired
  comparison on the same tweets, but am's absolute scores, floor and ceiling are not comparable
  with yo's and sw's; stated with the results.

## Results (run 2026-11-19, commit ae489e8; `results/`)
Logistic regression (LR) is the headline; the MLP (mean of 3 seeds) is secondary. Main set: X's
AfriSenti test tweets mapped into Y's space, Y's UMSAB classifier.

| mean macro-F1, main set, LR | direct | chain | no map | X's own ceiling | X's floor |
|---|---|---|---|---|---|
| yo (6 directions) | 0.341 | 0.379 | 0.329 | 0.612 | 0.199 |
| sw (6) | 0.397 | 0.384 | 0.288 | 0.464 | 0.248 |
| am (6) | 0.475 | 0.477 | 0.303 | 0.464 | 0.267 |

- **Chain − direct (`summary.csv`, main, LR):** yo **+0.038 [+0.026, +0.050], 6 of 6 directions
  above 0**, from +0.020 (yo → de, yo → en) to +0.056 (yo → ar, yo → it); MLP +0.027
  [+0.004, +0.049], 4 above, 1 below. sw −0.013 [−0.023, −0.003], 0 above, 1 below (sw → en via it
  −0.039); MLP −0.002. am +0.002 [−0.014, +0.017], 2 above (de, es), 1 below (en); MLP +0.007.
- **Transfer into yo is weak:** direct is 1 point above no map on average, and for yo → de, es, fr
  no map (0.375–0.387) scores above both maps. The chain lifts yo from about the no-map level to a
  few points above it, still far below yo's own classifier (0.61).
- **Amharic transfers above its own ceiling** (0.475 against 0.464): its classifier learns the
  neutral-heavy training labels, while its test set is mostly negative (Data findings); the UMSAB
  classifiers, trained on balanced classes, suffer less from the shift.
- **r and BLI (`correlations.csv`, 18 main directions):** advantage vs r ρ = 0.45 [−0.09, 0.83]
  (MLP 0.50 [0.02, 0.83]), mostly yo (high r) against sw and am; advantage in F1 vs advantage in
  MRR ρ = 0.37 [−0.23, 0.82]; F1 vs MRR over directions × methods ρ = 0.27 [−0.24, 0.68].
- **Reverse set (secondary; UMSAB tweets into X's space, X's classifier):** yo −0.034 [−0.059,
  −0.009], 0 above, 2 below (ar → yo −0.084, it → yo −0.051); sw −0.001; am +0.002. ar → yo is a
  case where BLI misleads: the chain is better in BLI (+0.016 MRR [+0.005, +0.028]) and clearly
  worse in sentiment.
- **LR's C:** 100 (the grid's top) for yo and am, 10 for sw; this affects their ceilings and the
  reverse set, not the main set (the UMSAB classifiers are part 1's).

**Expected outcomes, marked:**
1. Transfer weak; into yo a few points above no map: **partly** (direct +1, chain +5; for 3 of 6
   directions no map beats both maps). Into sw and am clearly above no map: **right** (+11, +17).
   All well below their ceilings: **right for yo, partly for sw** (7 points), **wrong for am**
   (above its ceiling; label shift). Ceilings 0.45–0.60: **right** (0.46–0.61).
2. yo: the chain beats direct in most of the 5 small-direct directions by 1–4 points, at least 2
   resolved: **right** (5 of 5 resolved, +2.0 to +5.6, a little above the range for ar, es, it).
   yo → en no gain or a loss: **wrong** (+0.020, resolved).
3. sw → de, es, fr, it, en no resolvable difference: **mostly right** (4 of 5; sw → en clearly
   worse for the chain). The later sw → ar expectation (chain ahead): **wrong** (−0.009,
   unresolved).
4. am no difference or the chain slightly below: **right** (+0.002, interval around 0).
5. Advantage rises with r: **right in sign** (ρ = 0.45; the LR interval includes 0, the MLP's
   does not).
6. BLI and downstream agree in sign for yo: **partly** (4 of 6 main directions; yo → en and
   yo → es disagree).

**Reading.** In the real low-resource case the chain helps where RQ2 says it should: Yoruba, whose
direct dictionaries are 586–1,652 pairs against 7,436 on its English step, gains in every
direction (+3.8 points). Swahili and Amharic, whose chains bring no more data than direct, gain
nothing; Swahili–Arabic, small direct but large steps, did not gain either. The gain is real but
lifts a weak transfer: Yoruba stays far below its own classifier. In the opposite direction
(classifying UMSAB tweets with the Yoruba classifier) the chain hurts, so the benefit is not
symmetric.

**Caveats.** Three test sets, 6 directions each; each language's directions share their test
tweets and its correlation intervals resample directions as if independent; the tweet bootstrap
covers test sampling only; Amharic's test labels are shifted; LR's C is at the grid's top for yo
and am; the downstream maps use full dictionaries while the BLI maps drop each fold's test words;
the sw → ar expectation was added after the README's approval (from dictionary sizes, before any
result).

## Build (estimates are the assistant's)
1. Download AfriSenti yo, sw and am; `prep/14_afrisenti.py` writes features and `stats.csv` as
   prep/13 does (about 1 h).
2. Stage `af`: pairs spec, prep/07 and prep/08, check the yo dictionaries against stages 1, A, A2
   (30 min). `configs/default.yaml` gets `downstream.afrisenti` (languages, labelled side,
   scripts for the script check) and stage `af`.
3. Runner `experiments/downstream-afrisenti/run.py`: the part 1 helpers it shares (dictionary
   lookup, maps, bootstrap tables) move into `src/clwe_chains/transfer.py` with tests, so the two
   runners don't copy each other (about 2 h).
4. Data check reported to the user; BLI companion run; reviewer pass; the run itself (minutes on
   the laptop).

## Decisions taken with the README (user, 2026-11-19)
- Hindi left out as the labelled side; reverse directions run as a secondary result; Amharic added.
