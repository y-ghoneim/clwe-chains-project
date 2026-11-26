# Topic classification across languages through direct and chained maps (downstream: SIB-200)

**Status: APPROVED by the user, 2026-11-20. RUN 2026-11-20 at commit 072a094; see "Results".** Results will be
descriptive (decisions.md 2026-11-16: no success rule); gains in only some cases count as a good
result (user, 2026-11-19). With MasakhaNEWS, replaces WikiANN NER (decisions.md 2026-11-20).
SIB-200's sentences are FLORES-200 sentences, but SIB-200 is its own CC BY-SA 4.0 dataset without
the FLORES+ terms, so training on its train split is fine (decisions.md 2026-11-20); its text and
features stay in the data root, only scores are committed.

**Question.** Does the chain's advantage on downstream tasks, level on well-resourced pairs and
positive where the direct dictionary is small (sentiment parts 1–2), hold on the project's own
pairs (stage D's well-resourced pairs and the low-resource pairs of stages 1, A, A2), which already
have BLI results? SIB-200 is the one task that covers nearly all of them, and its sentences are the
same in every language, so every direction is tested on the same content.

**Framing (as before).** X's test sentences are mapped into Y's space and classified by Y's
classifier, trained on Y's own sentences.

## Data
- **SIB-200** (Adelani et al. 2024), Hugging Face `Davlan/sib200`, licence CC BY-SA 4.0: 1,004
  FLORES-200 sentences in 205 languages with 7 topic labels (science/technology, travel, politics,
  sports, health, entertainment, geography). Every language: 701 / 99 / 204 sentences (train /
  validation / test), the same sentences translated. Downloaded into its own empty folder, read
  with `-I`, revision pinned and logged.
- **Codes:** each project language is matched to its SIB-200 configuration (for example ar →
  arb_Arab, fa → pes_Arab, sw → swh_Latn, mg → plt_Latn (Plateau Malagasy), qu → quy_Latn
  (Ayacucho Quechua; our fastText "qu" is Quechua without a variety), or → ory_Orya); the table is
  written in the config.
- **Left out:** zh, ja, th (no spaces between words, so a sentence would be one unmatched token;
  proposed, as for WikiANN); mhr, myv and nah (not in SIB-200).
- **Pairs (fixed now, from the stages' pairs.csv; hubs and dictionaries as in those stages):**
  - **Well-resourced set:** stage D's 31 pairs without zh, ja, th, both directions: 62 directions.
  - **Low-resource set (headline for RQ2):** every pair of stages 1, A, A2 whose low-resource
    language is in SIB-200 (yo, or, mg, as, ckb, qu, jv) and whose node is not zh, ja or th,
    low-resource → node: 142 directions (as 15, ckb 15, jv 25, mg 24, or 15, qu 23, yo 25). Every
    language has 701 training sentences, so mt can be a labelled language here.
- BLI scores come from the stages' existing results (stage D, 1, A, A2 `pooled.csv`); no new BLI run.

## Method (as the sentiment parts unless stated)
- **Features:** the part 1 tokenizer and features: mean of normalized vectors over matched tokens,
  unit length, centred on the language's own training-sentence mean. Data check before any
  transfer number: matched tokens per sentence, empty sentences, per language.
- **Maps:** direct and chain through the pair's hub, Procrustes on the stage's full natural
  dictionaries; no map as a reference.
- **Classifiers:** per labelled language, logistic regression (C from {0.001 … 100} on its
  validation macro-F1), headline; the MLP (hidden 256, 3 seeds), secondary; 7 classes (the library
  takes the number of classes as a parameter, shared with MasakhaNEWS).
- **Reference points:** no map; majority-class floor (the same test labels for every language);
  each language's own classifier.
- **Evaluation:** macro-F1 per direction; chain − direct with a paired bootstrap over test
  sentences (1,000 resamples). Summaries per set, and in the low-resource set per low-resource
  language; Spearman of the advantage against r and the BLI link (F1 against MRR; advantage
  against advantage), per set, intervals resampling pairs.
- **Precision:** 204 test sentences and 7 classes: one direction's macro-F1 is uncertain by several
  points and paired differences by about 2–3; few single directions will resolve. A language's
  directions share its test sentences, so the per-language means are the readable numbers. With
  only 701 training sentences per language the classifiers are weak, which lowers all scores but
  is the same for both maps.

## Expected outcomes (written before any run; guesses, to be marked wrong where wrong)
1. **Ceilings** 0.55–0.80 macro-F1 for the well-resourced languages, lower for the low-resource
   ones; transfer 10–25 points below its target's ceiling; no map near the floor.
2. **Well-resourced set: chain level with direct or slightly below** (stage D's BLI −0.005 MRR;
   sentiment part 1 level to slightly ahead); few directions resolved.
3. **Low-resource set: chain above direct on average** and in most of the 7 low-resource
   languages' means, as in BLI and AfriSenti's Yoruba; single directions mostly unresolved.
4. **Advantage rises with r** in the low-resource set (positive Spearman).
5. **BLI link:** F1 against MRR positive over directions × methods in both sets (ρ about
   0.2–0.5); advantage against advantage weak.

## Data findings while building (2026-11-20, before any transfer number was computed)
- **Source:** Hugging Face `Davlan/sib200`, revision 38977a6, 34 languages
  (`raw/downstream/sib200/SOURCE.txt`); 7 topics, identical in every language.
- **Directions:** low-resource set 142 (as 15, ckb 15, jv 25, mg 24, or 15, qu 23, yo 25),
  well-resourced set 62 over 31 pairs, as the README says.
- **Data check (`v1/downstream/sib200/stats.csv`, test):** no empty sentence except one for qu;
  10–27 matched tokens per sentence (qu lowest at 10); script share 96–100% (ko 96%).

## Results (run 2026-11-20, commit 072a094; `results/`)
Logistic regression (LR) headline, MLP (3 seeds) secondary; 7 topics; floor 0.057 macro-F1.

| mean macro-F1, LR | direct | chain | no map |
|---|---|---|---|
| well-resourced set (62 directions) | 0.702 | 0.705 | 0.123 |
| low-resource set (142 directions) | 0.378 | 0.502 | 0.119 |

| low-resource language (directions) | direct | chain | chain − direct [interval] | above / below 0 |
|---|---|---|---|---|
| or (15) | 0.281 | 0.555 | +0.274 [+0.226, +0.306] | 14 / 0 |
| as (15) | 0.458 | 0.645 | +0.187 [+0.161, +0.212] | 15 / 0 |
| mg (24) | 0.289 | 0.460 | +0.170 [+0.140, +0.197] | 21 / 0 |
| yo (25) | 0.302 | 0.411 | +0.109 [+0.094, +0.124] | 18 / 0 |
| ckb (15) | 0.531 | 0.623 | +0.092 [+0.075, +0.110] | 12 / 0 |
| jv (25) | 0.536 | 0.598 | +0.062 [+0.042, +0.081] | 8 / 0 |
| qu (23) | 0.294 | 0.337 | +0.044 [+0.031, +0.055] | 3 / 0 |

- **Low-resource set: the chain beats direct by 12.5 points** (LR, [+0.110, +0.139]); 91 of 142
  directions above 0 and none below; every one of the 7 languages gains. The MLP agrees (+11.6;
  99 above, 1 below).
- **Well-resourced set: level** (+0.003 [−0.005, +0.012]; 6 above, 2 below; MLP +0.004).
- **r:** in the low-resource set the advantage rises with r (ρ = 0.21 [0.01, 0.38]); by r band:
  0–1 +0.033 (3 directions), 1–2 +0.060 (23), 2–3 +0.134 (33), above 3 +0.145 (82). Well set:
  none (ρ = 0.02).
- **BLI link, much stronger than for sentiment:** F1 against MRR over directions × methods
  ρ = 0.75 [0.69, 0.80] (low) and 0.56 [0.34, 0.72] (well); the chain's F1 advantage against its
  MRR advantage ρ = 0.48 [0.34, 0.60] (low), 0.10 (well; MLP 0.32).
- **Ceilings** (each language's own classifier, 701 training sentences): well-resourced 0.65–0.83
  (mean 0.76), low-resource 0.44–0.73. With the chain the low-resource languages reach about two
  thirds of a node's ceiling; with the direct map about half.

**Expected outcomes, marked:**
1. Ceilings 0.55–0.80 well-resourced, lower for low-resource: **right** (max 0.83). Transfer
   10–25 points below the target's ceiling: **wrong** (about 6 in the well set; about 25–38 in
   the low set). No map near the floor: **about right** (0.12 against 0.06).
2. Well set level or slightly below: **right (level)**.
3. Low set chain above direct on average and in most languages: **right** (all 7, +12.5);
   single directions mostly unresolved: **wrong** (91 of 142 resolved).
4. Advantage rises with r in the low set: **right** (ρ = 0.21, interval just above 0).
5. BLI link ρ 0.2–0.5: **wrong, stronger** (0.56 and 0.75); advantage against advantage weak:
   **right for the well set, wrong for the low set** (0.48).

**Reading.** On the project's own pairs, with test sentences identical in every language, the
downstream result repeats RQ1/RQ2 closely: chains are level with direct where direct dictionaries
are large, and far better where they are small (the low-resource pairs have about 300–1,650 direct
pairs against several thousand on the English step). Here BLI does predict downstream quality,
unlike for tweet sentiment: short sentences of content words depend on word translation more
directly than tweets do.

**Caveats.** A low-resource language's directions share its 204 test sentences, and its
per-language interval resamples directions, not sentences; the per-direction bootstraps cover
test sampling only; classifiers trained on 701 sentences are weak; SIB-200's sentences are
FLORES-200 translations (decisions.md 2026-11-20); qu is Ayacucho Quechua in SIB-200 while our
vectors are Quechua without a stated variety.

## Build (estimates are the assistant's; shared steps with MasakhaNEWS done once)
1. Classifiers take the number of classes (library change with tests; shared).
2. Download; `prep/17_sib200.py` writes features and the data check per language.
3. Runner `experiments/downstream-sib200/run.py`: both sets from the stages' pairs (as the WikiANN
   runner selects them), maps per stage, tables as the sentiment runners; smoke run; reviewer pass;
   run (laptop, minutes: the classifiers train on 701 sentences).
About 2–3 hours, after MasakhaNEWS.

## Open questions for the user
1. Leave out zh, ja, th? (Proposed: yes; segmenting them would need per-language tools.)
2. Both sets, with the low-resource set as the headline? (Proposed: yes.)
