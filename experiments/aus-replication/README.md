# AUS replication on the shared list (phase 3)

**Question.** The AUS report claimed that chaining en→ar alignment through Spanish, Irish and
Maltese beats direct alignment. With its flaws fixed (each step fitted on its own language pair,
held-out concepts, 5 splits, significance tests), does any of its 15 chains beat direct en→ar?

Spec: `docs/superpowers/specs/2026-11-15-phase3-aus-replication-design.md`; brief sections 2,
5.5, 5.7. This README was written and approved before any run (CLAUDE.md).

## Data
- The shared list (phase 1/2, threshold 0.69): 883 concepts over English + 29 languages, all of
  en, ar, es, ga, mt included. Splits: seeds 0–4, 706 training / 177 test concepts each.
- Robustness subset (brief 5.7): the 534 concepts with Wiktionary sense coherence ≥ 0.9 over at
  least 10 evaluable language pairs; each split filtered to it (about 425 / 105).
- fastText vectors (top 200k; mt 198,664), v0's PanLex tables, the Wiktionary extract.

## Routes
Direct en→ar, and AUS's 15 chains: every ordering of every non-empty subset of {es, ga, mt}
(3 one-hop, 6 two-hop, 6 three-hop). In the faithful and proper settings each step s→x is fitted
on the split's training concepts' (s word, x word) rows, and the route's map is the product of
its steps.

## Base case: AUS-exact
AUS's pipeline as published (report section 4.3, Algorithm 1; section 5.1), on our list instead
of their 823 Google-translated words (user decision, decisions.md 2026-11-15):
- **fitting:** pseudo-inverse least squares on raw vectors; every step after the first is fitted
  from the *mapped* English vectors (`aus_metrics.aus_scheme_map`). This scheme equals direct
  alignment mathematically (`tests/test_align.py::test_aus_scheme_collapses_to_direct`);
- **words:** all 883 concepts are both the training and the evaluation words; one run, no split;
- **metrics:** P@1, P@5, P@10 among the 883 Arabic list words; squared Frobenius error
  ‖XW − Y‖²; Hausdorff distance; neighbourhood overlap. The report doesn't state P@k's distance
  (both cosine and Euclidean are reported) or overlap's k (10 is used).

## Settings

| | AUS-faithful | proper (main) |
|---|---|---|
| solver | least squares | Procrustes |
| vectors | raw | unit → centre → unit |
| retrieval | cosine nearest neighbour among the 883 Arabic list words | CSLS, k = 10, over Arabic's full vocabulary (200k) |
| correct answers | the list translation | the list translation plus every PanLex/Wiktionary translation of the English word in Arabic's vocabulary |
| evaluated on | test concepts; separately the training concepts (train = test, labelled, only for comparison with AUS Table 2) | test concepts |
| robustness rerun | no | yes: trained and tested on the subset |

Test concepts whose Arabic word equals the English string are excluded (none expected).

## Metrics and significance
- P@1, P@5, MRR per route × split; mean ± SD over the 5 splits.
- Per split, a paired bootstrap over test concepts (1,000 resamples, seed 0) of MRR(chain) −
  MRR(direct), 95% interval. Per chain: mean ΔMRR over splits, and in how many of the 5 splits the
  interval lies above 0 / below 0. Splits share concepts, so they aren't pooled.
- AUS's selection rule (lang2vec `syntax_wals` and `geo`, missing = 0, cosine distance): its Table
  1 is reproduced exactly (`tests/test_aus_rule.py`); all 15 chains are ranked by path cost under
  each feature set.

## Expected outcomes (written before running)
1. **Proper setting: no chain reliably beats direct.** At most 2 of the 15 chains have their
   interval above 0 in 3 or more splits; most chains have mean ΔMRR below 0, and longer chains
   lose more. (v0: with full dictionaries chains almost never beat direct; on equal shared word
   triples Procrustes chains beat direct in 5/8 pairs by at most +0.026.)
2. **Faithful setting, held out:** least-squares direct is clearly below Procrustes direct
   (v0: 2–3× lower on small dictionaries). Some least-squares chains may beat least-squares
   direct, as in v0 (8/8 pairs): that would reflect a weak solver being regularized by the chain,
   not a better route.
3. **Faithful setting, train = test:** P@1 far above the held-out P@1 for every route (AUS
   reported 71.7 for direct on its training words); direct at or above every chain.
4. **Robustness:** the subset doesn't change the sign of any chain's mean ΔMRR in the proper
   setting. If chains win on the full list but not on the subset, that win is reported as an
   artifact of the list.
5. **AUS-exact base case: every chain equals direct up to rounding.** All 16 routes have the same
   P@k, error, Hausdorff and overlap to within numerical noise (AUS's own Table 2 shows the same
   pattern: P@1 70.2–71.9 for chains vs 71.7 for direct, identical error 7.24 for several). P@1
   is high, because the evaluation words are the training words.
6. **AUS rule:** by syntax it ranks routes through es and ga first (Table 1); by geography it
   ranks mt first (0.0137 vs es 0.0203; Maltese is absent from AUS's Table 1). No relation is
   expected between the rule's ranking and the chains' measured MRR.

Any outcome that differs from these is reported as found.

## Run
`python experiments/aus-replication/run.py` from a clean working tree (it refuses otherwise; each
results file records the commit hash). About 10 minutes on the laptop GPU.

Outputs in `results/`: `aus_exact.csv`, `metrics.csv`, `bootstrap.csv`, `summary.csv`,
`aus_rule.csv`, `run_info.json`.

## Results
Run 2026-11-15 at commit `48fcf80` (`results/run_info.json`); no test concept had an Arabic word
identical to its English one. Means over the 5 splits; ΔMRR = chain − direct; "below 0" = splits
whose 95% bootstrap interval lies below 0.

**Proper setting (main):** direct en→ar MRR 0.714 (P@1 0.616). **Every chain is below direct.**

| route | MRR | ΔMRR | splits above / below 0 |
|---|---|---|---|
| en→ar (direct) | 0.714 | | |
| en→es→ar (best chain) | 0.695 | −0.019 | 0 / 0 |
| other one-hop (ga, mt) | 0.587–0.590 | −0.124, −0.127 | 0 / 5 |
| two-hop (6 routes) | 0.398–0.606 | −0.108 to −0.316 | 0 / 5 each |
| three-hop (6 routes) | 0.376–0.469 | −0.245 to −0.339 | 0 / 5 each |

Mean MRR falls with every intermediate: 0.714 (direct), 0.624, 0.523, 0.418.

**Robustness (high-agreement subset):** direct MRR 0.721; still every chain below direct
(en→es→ar −0.008, 0 / 0; the rest −0.050 to −0.200, significant in 3–5 splits). The chains'
deficits are about half as large as on the full list.

**AUS-faithful setting, held out:** least-squares direct MRR 0.392 (P@1 0.254), about half of
Procrustes direct. Every chain is below it (en→es→ar −0.021, 0 / 0; the other 14 −0.118 to −0.231,
significant in all 5 splits).

**AUS-faithful setting, train = test:** direct P@1 0.963 (vs 0.254 held out); every chain below
direct (0.416–0.872).

**AUS-exact base case** (`results/aus_exact.csv`): all 16 routes give identical numbers to 4
decimals: P@1/5/10 0.941 / 0.986 / 0.997 (cosine) and 0.598 / 0.687 / 0.703 (Euclidean); overlap
0.330; Hausdorff 3.21; error 977.8; the routes' maps differ from direct by less than 5e-5.

**AUS selection rule** (`results/aus_rule.csv`): Table 1 reproduced exactly. By syntax path cost
direct ranks first (a path through an intermediate can't be shorter under this measure here),
then en→es→ar, en→ga→ar. By geography, mt is the best one-hop intermediate and en→es→mt→ar the
cheapest route overall; direct ranks 7th. Over all 16 routes, syntax cost correlates with MRR
(Spearman −0.83, proper setting); within routes of equal length the direction holds but isn't
significant (one-hop −1.00 over 3 routes; two-hop −0.71, p = 0.11; three-hop −0.31, p = 0.54).
Geography cost shows no relation (−0.21, p = 0.43).

### Against the expected outcomes
1. **Proper: confirmed, more strongly than expected.** No chain beats direct; 14 of 15 are
   significantly worse in all 5 splits; longer chains lose more.
2. **Faithful held out: half confirmed.** Least squares is well below Procrustes (0.39 vs 0.71).
   But **no** least-squares chain beat least-squares direct; the v0 pattern (8/8 on shared word
   triples) did not recur here.
3. **Train = test: confirmed.** P@1 0.96 on training words vs 0.25 held out; direct at the top.
4. **Robustness: confirmed.** No sign changes.
5. **AUS-exact: confirmed, exactly.** All routes identical, not merely within noise. AUS's
   tables show differences between chains (823 words: P@1 46.4–56.7 vs direct 54.2); with this
   scheme they can only come from numerical precision or from per-route differences in their
   word rows, not from the routes. Against their 823-word table (Table 2): our P@1 brackets
   theirs depending on the distance (cosine 0.94, Euclidean 0.60, theirs 54.2); Hausdorff 3.21
   vs 2.76 and overlap 0.33 vs 31.7% are comparable; their error (23.5) is on a different scale.

   *Correction (after the run):* the expected outcomes above (items 3 and 5) quote "71.7 for
   direct" and "P@1 70.2–71.9 for chains" as AUS's training-word numbers. Those are from their
   1K FLORES table (Table 3); their 823-word table has direct at 54.2. The expectations
   themselves are unchanged.

### AUS's own best chains in this replication
| AUS's claim | chain | proper MRR (Δ vs 0.714) | faithful held-out MRR (Δ vs 0.392) |
|---|---|---|---|
| best BLI, 823 words (P@1 56.7 vs 54.2) | en→mt→ga→es→ar | 0.422 (−0.292, 0 / 5) | 0.165 (−0.227) |
| best sentiment, neural net (67.35% vs 65.29%) | en→mt→es→ga→ar | 0.469 (−0.245, 0 / 5) | 0.166 (−0.226) |
| best sentiment, logistic regression (64.27% vs 63.83%) | en→es→mt→ga→ar | 0.385 (−0.329, 0 / 5) | 0.173 (−0.219) |

All three are three-intermediate routes, among the worst here, and identical to direct in the
AUS-exact base case.
6. **AUS rule: partly wrong.** The rankings are as expected (syntax: es then ga; geography: mt
   first among one-hop routes). But "no relation to MRR" was wrong for syntax overall (−0.83);
   that correlation mostly reflects route length, since both path cost and the MRR loss grow
   with each hop.

### Observation (not in the expected outcomes)
On the high-agreement subset the chains' deficits roughly halve (e.g. en→ga→ar −0.124 → −0.060).
For an English source, sense noise sits on the chains' non-English steps (es→ar, ga→ar, mt→ar),
while the direct en→ar rows are clean by construction, so the noise penalizes chains here. This
is the mirror image of the bias expected for non-English pairs (brief 5.7), where it favours
routes through English.

**Why the absolute numbers can't match AUS's.** Their 823-word list, its Google translations,
their code and the fastText files they used are not public (the report has no appendix or code
link), so their exact numbers can't be reproduced; the pattern their scheme guarantees (every
chain equal to direct) is reproduced exactly. Their 823-word table also suggests noisy data: P@10
is only 63.7% for retrieval among 823 words (ours: 99.7%), as if about a third of their pairs
could never be matched (our reading, not stated in the report).

**Answer to the question:** no. With per-step fitting, held-out concepts and repeated splits,
none of AUS's 15 chains beats direct en→ar alignment, in either setting; their reported gains are
not reproduced.

### Known issues (found in code review, 2026-11-16; not rerun)
The user decided not to rerun phase 3 and to fix these from phase 5 on (`docs/decisions.md`,
2026-11-16). None changes the answer above.
- **Gold casing:** gold sets hold one vocabulary row per word (its most frequent casing) while
  retrieval ranks the cased vocabulary. Arabic has no case, so every en→ar number here is
  unaffected.
- **English-side sense mismatches:** lowercasing attaches the translations of capitalized senses
  to lowercase English words (*may* gets the month, *turkey* the country; about 10–12 of 883
  concepts). That is 1–3 test words per split, shared by direct and every chain, so ΔMRR barely
  moves; but the direct en→ar rows above are not fully "clean by construction".
- **Gold breadth:** en→ar accepts about 16 answers per word on average (list word, PanLex and
  Wiktionary; up to 152 for "more"). The main MRR is therefore much higher than with the list
  word alone (reviewer, split 0: 0.714 vs 0.474). Comparisons within this experiment are
  unaffected; comparisons with MUSE-style single-answer numbers are not like for like.
- **Records:** the Spearman figures in the AUS-rule paragraph were computed ad hoc from
  `results/` (the reviewer reproduced them); `summary.csv` gives the SD of each route's MRR over
  splits, not of its ΔMRR.
