# News topic transfer into African languages through direct and chained maps (downstream: MasakhaNEWS)

**Status: APPROVED by the user, 2026-11-20. RUN 2026-11-20 at commit 072a094; see "Results".** Results will be
descriptive (decisions.md 2026-11-16: no success rule); gains in only some cases count as a good
result (user, 2026-11-19). Replaces WikiANN NER as the second downstream task (user, 2026-11-20;
see decisions.md).

**Question.** The sentiment parts found the chain level with direct on well-resourced pairs and
ahead for Yoruba, whose direct dictionaries are small. Does that hold for a second task, news
topic classification, on real low-resource languages classified by an English or French model?
Within Yoruba the two labelled languages give opposite predictions: yo–fr has a small direct
dictionary (907 pairs) and a large English step (r ≈ 3.0), yo–en a large direct one (7,436
pairs, r ≈ −2.2).

**Framing (as parts 1–2).** X (yo, sw, am, so) has little labelled data; its articles are mapped
into the space of Y (en, fr) and classified by Y's classifier. X's own training articles give the
ceiling only.

## Data
- **MasakhaNEWS** (Adelani et al. 2023), Hugging Face `masakhane/masakhanews`, licence AFL-3.0;
  BBC and VOA articles with topic labels shared across languages. Downloaded into its own empty
  folder, read with `-I`; revision pinned and logged.
- **Languages:** X = **yo, sw, am, so** (the MasakhaNEWS languages with fastText vectors and PanLex
  entries; the other 10 African languages have no fastText vectors). Y = **en, fr** (the only
  high-resource languages in the dataset).
- **Sizes (train / validation / test, from the dataset server):** en 3,309 / 472 / 948; fr 1,476 /
  211 / 422; yo 1,433 / 206 / 411; sw 1,658 / 237 / 476; am 1,311 / 188 / 376; so 1,021 / 148 /
  294.
- **Text:** headline and article body joined (the dataset's combined field if it has one; checked
  on download). Labels: the dataset's topic set, the same for all languages; classes missing in a
  language are reported.
- **Pairs: X × Y = 8 main directions, fixed now; the reverse Y → X (8) as a secondary result**
  reported apart, as in part 2.
- **Dictionaries and hubs:** a new prep stage `mn` built like stage `af` (prep/07 with a pairs spec,
  prep/08). The yo, sw and am pairs with en and fr are pairs of stage `af`, so the same sources
  and code give the same dictionaries and hubs (checked when built); so–en and so–fr are new.
  A BLI companion run on stage `mn` (8 pairs, both directions, minutes).

## Method (as parts 1–2 unless stated)
- **Features:** the part 1 tokenizer and features: mean of normalized vectors over matched tokens,
  unit length, centred on the language's own training-article mean. Data check before any transfer
  number: matched tokens per article, empty articles, script share (am Ethiopic, the rest Latin).
- **Maps:** direct and chain through the pair's hub (Procrustes on the full natural dictionaries of
  stage `mn`); no map as a reference.
- **Classifiers:** Y's logistic regression (C from {0.001 … 100} on Y's validation macro-F1),
  headline; the MLP (hidden 256, 3 seeds, early stopping on validation cross-entropy), secondary.
  The library's classifiers are written for 3 classes; they get the number of classes as a
  parameter (tested) before this run.
- **Reference points:** no map; majority-class floor from X's test labels; X's own ceiling.
- **Evaluation:** macro-F1 per direction; chain − direct with a paired bootstrap over test articles
  (1,000 resamples). Per X language (2 directions each) and over the 8 directions; Spearman of the
  advantage against r over the 8 directions, descriptive only.
- **Precision:** 294–476 test articles give paired differences of roughly ±1.5–2.5 points.

## Expected outcomes (written before any run; guesses, to be marked wrong where wrong)
1. **Transfer clearly above no map and the floor**: topic words (sport, election, health) are
   frequent and well translated; ceilings 0.70–0.90 macro-F1, transfer 10–25 points below.
2. **yo → fr: the chain beats direct** (small direct, large English step), by 2–5 points.
   **yo → en (via es): no gain or a loss.**
3. **sw → fr (r ≈ 1.5) a small gain or none; sw → en (via it), am → en and am → fr (r near 0):
   no resolvable difference.**
4. **so:** following RQ2, a gain where its direct dictionary is under about 10,000 pairs and its
   chain's narrowest step is at least twice as large; otherwise none (sizes known only after the
   stage is built; this rule is fixed now).
5. **Advantage rises with r** over the 8 directions (descriptive).
6. **Reverse set:** no consistent sign (part 2 found the chain hurt Yoruba in reverse).

## Data findings while building (2026-11-20, before any transfer number was computed)
- **Source:** Hugging Face `masakhane/masakhanews`, revision fa3b5ff (`raw/downstream/
  masakhanews/SOURCE.txt`); TSV files with fields that span lines, read with a quoting TSV reader;
  row counts match the dataset card.
- **The label sets differ across languages** (the README assumed one set): en business,
  entertainment, health, politics, sports, technology; fr without entertainment; yo entertainment,
  health, politics, religion, sports; sw and so all seven; am business, health, politics, sports.
  **Decided (user, 2026-11-20): each direction keeps only the topics both languages have**
  (yo–en 4, yo–fr 3, sw–en 6, sw–fr 5, am–en 4, am–fr 4, so–en 6, so–fr 5); classifier, test
  articles, ceiling and floor all use those topics. Absolute F1s are therefore not comparable
  across directions; chain − direct is paired within a direction.
- **Stage `mn`:** yo, sw and am dictionaries and hubs identical to stage `af`. Somali: so–en 2,033
  pairs (hub fr, so–fr 1,323), so–fr 1,323 (hub en); r ≈ ±0.6, so the rule in prediction 4
  (direct under 10,000 and the narrowest chain step at least twice as large) predicts **no gain
  for so** in either direction. Written from dictionary sizes before any number was seen.
- **Data check (`v1/downstream/masakhanews/stats.csv`):** no empty article; 330–880 matched
  tokens per article; script share 100% (am 99.9% Ethiopic).
- **BLI companion** (`../downstream-masakhanews-bli/`, commit 3776fa6): run before the downstream
  run and joined; its summary line was seen when it finished (chain +0.002 MRR over 16
  directions, 4 above and 4 below 0). Nothing downstream had been computed then.

## Results (run 2026-11-20, commit 072a094; `results/`)
Logistic regression (LR) headline, MLP (3 seeds) secondary. Each direction is scored on the topics
both languages have, so absolute F1s differ in difficulty across directions.

| main set, LR | topics | direct | chain | no map | chain − direct [interval] | r |
|---|---|---|---|---|---|---|
| yo → en (via es) | 4 | 0.707 | 0.545 | 0.121 | **−0.162** [−0.209, −0.112] | −2.2 |
| yo → fr (via en) | 3 | 0.708 | 0.823 | 0.418 | **+0.116** [+0.064, +0.164] | +3.0 |
| sw → en (via it) | 6 | 0.721 | 0.707 | 0.137 | −0.015 [−0.057, +0.026] | −0.7 |
| sw → fr (via en) | 5 | 0.773 | 0.773 | 0.096 | −0.000 [−0.036, +0.033] | +1.5 |
| am → en (via fr) | 4 | 0.803 | 0.846 | 0.244 | **+0.043** [+0.008, +0.082] | −0.7 |
| am → fr (via en) | 4 | 0.884 | 0.882 | 0.345 | −0.002 [−0.023, +0.019] | +0.7 |
| so → en (via fr) | 6 | 0.485 | 0.472 | 0.155 | −0.013 [−0.075, +0.050] | −0.6 |
| so → fr (via en) | 5 | 0.530 | 0.576 | 0.172 | **+0.046** [+0.006, +0.092] | +0.6 |

- **Yoruba shows RQ2 in both directions:** with French, whose direct dictionary is small (907
  pairs) and whose English step is large, the chain gains 11.6 points; with English, whose direct
  dictionary is large (7,436) and whose hub step (yo–es, 1,652 pairs) is small, the chain loses
  16.2. The MLP agrees in sign (`summary.csv`).
- **Over the 8 directions the mean is about zero** (LR +0.001; 3 above 0, 1 below); advantage vs r
  ρ = 0.64 (8 directions; interval −0.2 to 1), advantage in F1 vs advantage in MRR ρ = 0.67.
- **Unexpected gains:** am → en (+4.3, r −0.7) and so → fr (+4.6, r +0.6), where the chain brings
  little or no extra data.
- **Transfer works well:** direct reaches 0.49–0.88 macro-F1 against 0.10–0.42 with no map and
  ceilings of 0.72–0.91 (`reference.csv`).
- **Reverse set (secondary):** mean −0.032 (2 above, 3 below); en → yo via es −0.177 and en → so
  via fr −0.152 against fr → so via en +0.116: the same pattern by hub as the main set.

**Expected outcomes, marked:**
1. Transfer clearly above no map and the floor: **right**. Ceilings 0.70–0.90: **right**
   (0.72–0.91). Transfer 10–25 points below: **partly** (3–24).
2. yo → fr chain ahead by 2–5 points: **right in sign, far larger** (+11.6). yo → en no gain or a
   loss: **right** (a large loss, −16.2).
3. sw → fr, sw → en, am → en, am → fr no resolvable difference: **right for 3 of 4**; am → en
   **wrong** (+4.3).
4. so, no gain by the RQ2 rule: **right for so → en, wrong for so → fr** (+4.6).
5. Advantage rises with r: **right in sign** (ρ = 0.64, 8 directions, not resolved).
6. Reverse set no consistent sign: **right**.

**Reading.** With 8 directions the overall mean says little; the pair of Yoruba directions says
the most: the same language and the same test articles, and the chain is far better or far worse
depending on whether its weakest step is larger or smaller than the direct dictionary.

**Caveats.** 8 directions, 2 per language sharing test articles; topic sets differ per direction;
the LR intervals cover test sampling only; the BLI companion's summary line was seen before the
run (Data findings).

## Build (estimates are the assistant's)
1. Classifiers take the number of classes (library change with tests; parts 1–2 unchanged at 3).
2. Download; `prep/16_masakhanews.py` writes features and the data check (as prep/14).
3. Stage `mn` (pairs spec, prep/07, prep/08), BLI companion folder and run.
4. Runner `experiments/downstream-masakhanews/run.py` from part 2's, on the shared helpers; smoke
   run; reviewer pass; run (minutes, laptop).
About 2–3 hours in all.
