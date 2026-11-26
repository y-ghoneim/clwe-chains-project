# Both ends low-resource: chains between low-resource languages of different families

**Status: APPROVED by the user, 2026-11-25** ("yes, then reviewer it then run it here"). Scope
approved by the user the same day (decisions.md, "New experiment: both ends low-resource"; the
professor asked for pairs of low-resource languages from different families). Handed over from the
report session, which writes it up (Results 4.2, one row in Table 2, one row in Table 4).

**Data built 2026-11-25** (prep/07, 08, 19 with `--stage ll`, 4 min): 40 pairs; every pair has
at least 173 eligible test words (none under the 50-word rule); the 10 languages' list words are
identical to stage A's; widest hubs en 26, hi 7, fr 6, es 1 (none zh, ja, th), narrowest hub step
451 pairs (hub 1) and 271 (hub 3). Code reviewed before the run; its findings fixed.

**Question.** Every low-resource pair so far had a well-resourced language at one end. When both
languages are low-resource and unrelated, the direct dictionary is tiny and both vector spaces are
weak. Does translating through a well-resourced hub still beat the direct map, and how far does it
get?

## Pairs (fixed by rule before any result)
- The 10 low-resource languages of the earlier stages: yo, or, mg, mhr, as, myv, nah, ckb, qu, jv.
- **Every pair of different families** (the languages table's family): 45 pairs minus the 5
  same-family ones (or-as, or-ckb, as-ckb: Indo-European; mhr-myv: Uralic; mg-jv: Austronesian)
  = **40 pairs, 80 directions**. Written to `pairs_spec.csv` (alphabetical order within a pair).
- **Hubs:** phase 6's rule over the 30 nodes: the widest hub (largest min(size[s, x], size[x, t]),
  PanLex coverage sizes), and for several hubs the 3 widest (prep/19's rule).
- **Rules fixed now:**
  - pairs whose direct dictionary (distinct word pairs, before test removal) has **under 100
    pairs are their own group**, reported next to the others, not dropped (Procrustes is barely
    determined there);
  - a pair needs **at least 50 test words** (eligible list concepts over its 5 folds); any pair
    below is listed and left out of the summaries.
- A scratch census (other session, not a result) found direct dictionaries of 2 to 551 pairs:
  9 under 100, 16 at 100–300, 13 at 300–500, 2 above 500.

## Data (as the natural-dictionary stages)
A new phase 6 stage `ll` (`configs/default.yaml`, `phase6_stages`), built by the existing prep:
- `prep/07_phase6_list.py --stage ll`: list words of the 10 languages (built as before, so the
  same words as stages 1, A, A2), eligible concepts, 5 folds (seed 0), gold (main and list word);
- `prep/08_phase6_dicts.py --stage ll`: natural dictionaries of every edge (direct s-t, s-hub,
  hub-t) and the joined counts (pooled direct);
- `prep/19_strong_hubs.py --stage ll` (a new option; the other stages unchanged): hubs 2 and 3 and
  their dictionaries.
- Each fold's test words are removed from every edge, as in every experiment.

## Word translation
- **Maps:** plain Procrustes rotations (no self-learning).
- **Methods**, per direction and fold, in one pass: direct; the chain through the widest hub;
  the average of direct and chain (combination); several hubs (direct + the chains through the 3
  widest hubs, CSLS scores averaged); pooled direct (the direct pairs plus the s-t pairs joined
  through the hub, mutual best, as the stages' `joined_mb`).
- **Evaluation:** CSLS (k = 10), MRR and P@1 on the main gold, all 5 folds; the stages' runner
  (through `../several-hubs-procrustes/`), so the code is the one already checked.
- **Measures**, per group (all pairs; under 100; 100 and above), pooled over test words with the
  paired bootstrap (1,000 resamples), and directions above / below 0 by their own bootstrap:
  direct MRR; chain − direct; combination − direct; several hubs − direct; pooled − direct;
  several hubs − chain.

## Topic classification (SIB-200)
- **Directions:** the pairs with both languages in SIB-200 (or, as, mg, yo, ckb, jv, qu): **17
  pairs, both directions, 34 directions**.
- **Setup as `../downstream-sib200/`** (its functions imported unchanged): the source language's
  test sentences mapped into the target language's space and classified by the target language's
  classifier, trained on its own 701 sentences (logistic regression, C on validation macro-F1; the
  MLP, 3 seeds, secondary). Maps: direct and the chain (Procrustes on the full natural
  dictionaries), and no map. As in that experiment, a pair whose hub is zh, ja or th is left out
  and listed (none expected).
- **Measures:** macro-F1 of direct, chain and no map per direction; chain − direct with the paired
  bootstrap over test sentences; mean over pairs per set with a bootstrap over pairs.

## Expected outcomes (before the run)
1. The chain above direct in nearly every direction, by a wide margin (both direct ends weak,
   each hub step a large dictionary to a well-resourced node).
2. Absolute MRR very low for every method (both vector spaces weak), lowest in the under-100 group.
3. Several hubs above the chain; pooled direct between direct and the chain.
4. SIB-200: the chain above direct, but macro-F1 close to the no-map floor on many directions.

## Cost
Laptop. Prep about 30–60 min (list words of 10 languages, dictionaries). Word translation: 40
pairs x 5 folds x 2 directions x 5 maps = 2,000 map evaluations, about 2 h at the laptop's 3–4 s each.
SIB-200: about 10 min.

## Files
- `pairs_spec.csv` (the rule's 40 pairs), `pairs.csv` (with hubs, written by prep/07).
- `config.yaml`: the two rules (100 direct pairs, 50 test words).
- `run.py` (word translation): resumable; `--limit N` (first N pairs, results to
  `results/smoke/`); `--results-only`. It records the stage's input fingerprints beside its cache
  at the first run and refuses to resume if they change.
- `run_sib200.py` (topics).
- `results/`: per direction, per group, run_info.json (commit, inputs' fingerprints).
- Tests in `tests/test_low_low.py`: the pair rule, the groups, the 50-word rule.
- Results summary sent back to the report session (Table 2 and Table 4 rows).

## Results (run 2026-11-25; word translation evaluated at 56c5a0b, SIB-200 at 557f3ad; `results/`)
All 40 pairs included (at least 173 test words each); 9 under 100 direct pairs. MRR, pooled over
test words, paired bootstrap; above / below: directions whose own interval excludes 0.

| word translation | all (80 directions) | under 100 (18) | 100 and above (62) |
|---|---|---|---|
| direct MRR | 0.0045 | 0.0013 | 0.0052 |
| chain − direct | +0.0099 [+0.0087, +0.0110], 56 / 0 | +0.0081, 16 / 0 | +0.0102, 40 / 0 |
| combination − direct | +0.0085 [+0.0076, +0.0094], 63 / 0 | +0.0062, 18 / 0 | +0.0090, 45 / 0 |
| several hubs − direct | +0.0131 [+0.0121, +0.0142], 67 / 0 | +0.0143, 18 / 0 | +0.0129, 49 / 0 |
| pooled direct − direct | +0.0075 [+0.0065, +0.0085], 41 / 0 | +0.0057, 14 / 0 | +0.0079, 27 / 0 |
| several hubs − chain | +0.0033 [+0.0021, +0.0044], 15 / 2 | +0.0062, 7 / 0 | +0.0026, 8 / 2 |

- P@1 (mean over directions): direct 0.17%, chain 0.52%, several hubs 0.71%. Best directions
  (several hubs MRR): as→jv 0.050, or→jv 0.047, jv→or 0.045; worst into or and from nah, yo
  (about 0.003).

| SIB-200 (34 directions, 17 pairs) | direct | chain | no map | chain − direct (over pairs) | above / below |
|---|---|---|---|---|---|
| logistic regression, macro-F1 | 0.227 | 0.410 | 0.126 | +0.184 [+0.150, +0.220] | 28 / 0 |
| MLP (3 seeds), macro-F1 | 0.246 | 0.420 | 0.135 | +0.175 [+0.147, +0.204] | 29 / 0 |

- No pair left out (no hub zh, ja, th). Reference: the target's own classifier on its own test
  set 0.632 (LR); majority-class floor 0.057. The 6 directions not above 0 (LR) all involve qu or
  mg: qu→ckb, jv→qu, qu→yo, yo→qu, mg→or, mg→yo.

**Against the expected outcomes.**
1. Chain above direct nearly everywhere: yes (56 of 80 above, none below).
2. Absolute MRR very low, lowest under 100: yes (direct 0.0013 there, 0.0052 above).
3. Several hubs above the chain, pooled direct between direct and the chain: yes.
4. SIB-200, chain above direct but near the no-map floor: only the first half. The chain reaches
   about two thirds of the in-language score (0.410 against 0.632); direct is the one near the
   floor.
