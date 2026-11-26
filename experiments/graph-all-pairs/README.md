# The language graph on 40 distant pairs (phase 4)

**Question.** For distant source–target pairs, can a language graph pick an intermediate chain
whose composed map beats direct alignment, and which edge measures make that work (brief 1)?

Spec: `docs/superpowers/specs/2026-11-15-phase4-graph-design.md`; plan:
`docs/superpowers/plans/2026-11-15-phase4-graph.md`; decisions.md (2026-11-15). This README was
written and approved before any run (CLAUDE.md).

## Data
- The shared list: 883 concepts over English + 29 languages (the graph's 30 nodes); splits seeds
  0–4 (706 / 177); robustness subset 534 concepts (sense coherence ≥ 0.9).
- **Pairs:** `pairs.csv`, 40 distant pairs drawn by seeded rules before any result (prep/06):
  - stratum (a), both high-resource (20): en→ar, en→zh, en→ja; 6 XLING pairs (fi and tr);
    6 PanLex-BLI pairs; one pair each for ja, zh, fa, el, hi;
  - stratum (b), at least one low-resource (20): 8 PanLex-BLI pairs with et/eu/ka; 8 pairs of
    mt/ga/sw/tl/ur/kk/ta with a high-resource partner; 4 pairs of two low-resource languages;
    at most 3 per language.
  - Low-resource = bottom third by CC-100 size: mt ga sw eu tl ur et kk ka ta.

## Routes, measures, baselines
- **Alignment:** as phase 3's proper setting (Procrustes per step on the split's training
  concepts, normalized vectors, CSLS k = 10 over the target's full vocabulary, gold = list word +
  PanLex + Wiktionary).
- **Search:** every route with at most 3 intermediates is scored exactly (20,440 per pair); a
  route costs the sum of its edges; ties go to fewer hops.
- **Edge measures (14 graphs):** quality edge (1 − MRR on each edge's 500 most frequent concepts
  outside the list) and its −log variant; resource (CC-100 size rank); SVG and econd-hm
  (Dubossarsky et al. 2020); RSIM (Vulić et al. 2020); URIEL+ (7 kinds); learned combination
  (gradient-boosted trees over all path costs and hop count, cross-validated over pairs).
- **Headline (i), always chain:** each measure's cheapest route with the direct edge excluded.
- **Baselines:** direct; via hub (en, or ru when en is an endpoint); the AUS rule (lang2vec
  syntax path cost; geography reported too). Random chains are deferred.
- **Oracle:** all 784 one- and two-hop routes per pair shortlisted by nearest-neighbour MRR on
  split 0, the top 10 rescored with exact CSLS on all 5 splits; an exhaustive exact check on 2
  pairs measures how often the true best is in the shortlist. An upper bound, never a method.
- **Cross-validated over pairs (5 folds, stratified):** the threshold rule (ii) (chain only when
  the direct edge's cost exceeds τ), the exponent p of dᵖ (p ∈ {1.5, 2, 3}), the learned
  combination.
- **Robustness:** every headline route refitted and re-evaluated on the high-agreement subset.

## Metrics
- Per pair and route: P@1, P@5, MRR per split; ΔMRR vs direct with per-split paired bootstrap
  (1,000 resamples) and the splits above / below 0.
- Across pairs: Wilcoxon signed-rank on per-pair mean ΔMRR, per stratum and overall;
  English-source pairs (en→ar, en→zh, en→ja, en–tr) reported separately.
- Per measure: pairs won against direct; regret vs the oracle; Kendall τ between path cost and
  MRR over each pair's 10 rescored routes; top-pick hit rate.
- Best-hub frequency: how often the oracle's best route goes through en, next to Anastasopoulos
  & Neubig (2020): en best in 17 of 90 pairs.

## Expected outcomes (written before running)
1. **Always chain loses.** For every measure, mean ΔMRR < 0 overall and in each stratum; no
   measure beats direct significantly across pairs (Wilcoxon, p < 0.05). Basis: phase 3 (every
   chain below direct, each hop costing ~0.1 MRR) and v0 (3 of ~540 chain runs won).
2. **The oracle rarely finds a winner.** Its best 1–2-hop route beats direct (interval above 0 in
   ≥ 3 of 5 splits) in fewer than a quarter of the pairs, with mean gains under 0.02 MRR where it
   does. The shortlist contains the true best route in both checked pairs.
3. **Measures rank as:** quality edge (lowest regret, highest Kendall τ) > resource and spectral
   > RSIM > URIEL+ (τ near 0, as in v0's ≤ 0.2). The −log variant and dᵖ change little; no p beats
   p = 1 by more than 0.01 MRR in cross-validation.
4. **The learned combination is no better than the quality edge alone.**
5. **The threshold rule doesn't beat "always direct".**
6. **Routes through English do relatively well among chains:** via hub is among the better chains
   for non-English pairs, and en is the oracle's hub in more than half of the pairs, unlike
   Anastasopoulos & Neubig's 17 of 90, because our list is built through English (clean rows on
   every step to and from en).
7. **Stratum (b) loses less than (a)** (smaller mean ΔMRR deficit), if any difference shows.
8. **English-source pairs favour direct most** (their direct rows are clean by construction).
9. **Robustness:** no measure's mean ΔMRR changes sign on the subset; the advantage of routes
   through English among chains shrinks there.

Any outcome that differs is reported as found.

## Run
Three stages, each resumable from the cache (`<data_root>/v1/results/graph-all-pairs/`: large,
so not in the repo) and run overnight on the laptop from a clean
working tree (`run.py` refuses result-affecting uncommitted changes and records the commit):
- **A** measures + headline routes + baselines, full list and subset: about 4–6 h (max 8.2 h);
- **B′** oracle: about 4.5 h;
- **C** learned combination targets (about 2.5 h) + dᵖ graphs (about 1–4 h).

`python experiments/graph-all-pairs/run.py --stage A` (then B, C). Outputs in `results/`.

## Results
Run 2026-11-16 (`results/run_info.json`): stage A at commit `ebb7494` (257 min), B′ at `8f77da5`
(237 min), C at `8eb84c7` (86 min); the later commits add only `results/`, so all three stages ran
the same code. At most 13 test concepts per split were dropped for a target word identical to the
English one. lang2vec has no syntax values for et, hr and kk, so 6 pairs (hr>fi, et>he, ka>et,
et>th, kk>ja, kk>tl) have no AUS-syntax route. Means over the 5 splits; ΔMRR = route − direct;
"won" = pairs with mean ΔMRR > 0; "above / below" = pairs whose 95% bootstrap interval lies above /
below 0 in at least 3 of 5 splits; Wilcoxon two-sided over the per-pair mean ΔMRR. Files:
`summary_measures.csv`, `summary_pairs.csv`, `metrics.csv`, `bootstrap.csv`, `routes.csv`,
`oracle.csv`, `oracle_check.csv`, `best_hub.csv`, `cv.csv`.

**Answer to the question: no.** No measure's chain beats direct alignment. Every chain choice is
below direct on average, significantly so across pairs (p < 0.001 for each), and no measure's
route is significantly above direct in any pair. Even the oracle (the best of 784 one- and two-hop
routes, chosen in hindsight) is 0.011 MRR below direct. The best measures do find nearly the best
chain (regret 0.002–0.004), mostly a single hop through English, but that chain still loses.

**Why (post hoc, `analyses/2026-11-16-phase4-diagnostics/`).**
- *Direct is optimal on the training data by construction.* The list is multi-parallel and every
  split uses the same 706 training concepts in every language, so direct Procrustes is the best
  orthogonal map s→t on them. A chain's composed map is also orthogonal and can't fit the
  training concepts better; it can win only through generalisation, a margin of about ±0.01
  here. Pivots help when the direct dictionary is small, noisy or missing, which this setting
  never has (that is phase 6, starved supervision).
- *Route differences are below split-to-split noise.* One split's 95% interval for ΔMRR has median
  width 0.083, and a route's ΔMRR varies with SD 0.020 across splits. On split 0, th-fa-it-eu
  was the best of all 784 routes (+0.031 over direct); over 5 splits it is 0.025 below.
  Per-pair tests can't see 0.01 effects; the Wilcoxon across pairs can, and says chains lose.

**All 40 pairs (full list, headline (i): always chain).** Kendall τ, regret and top pick are against
the oracle shortlist (base measures only for τ).

| route choice | pairs | MRR | ΔMRR mean | Wilcoxon p | won | above / below (≥ 3 of 5) | regret | Kendall τ | top pick |
|---|---|---|---|---|---|---|---|---|---|
| direct | 40 | 0.393 | | | | | | | |
| oracle | 40 | 0.382 | −0.0109 | < 0.001 | 10 | 0 / 0 | | | |
| quality (−log) | 40 | 0.380 | −0.0133 | < 0.001 | 5 | 0 / 2 | 0.002 | 0.646 | 0.46 |
| quality | 40 | 0.379 | −0.0138 | < 0.001 | 5 | 0 / 2 | 0.003 | 0.623 | 0.44 |
| RSIM | 40 | 0.378 | −0.0146 | < 0.001 | 5 | 0 / 4 | 0.004 | 0.600 | 0.40 |
| via hub | 40 | 0.378 | −0.0150 | < 0.001 | 5 | 0 / 3 | 0.004 | | 0.40 |
| resource | 40 | 0.369 | −0.0244 | < 0.001 | 4 | 0 / 10 | 0.013 | 0.424 | 0.12 |
| AUS rule (syntax) | 34 | 0.371 | −0.0328 | < 0.001 | 0 | 0 / 11 | 0.022 | | 0.00 |
| URIEL+ phonological | 40 | 0.358 | −0.0346 | < 0.001 | 1 | 0 / 14 | 0.024 | 0.356 | 0.05 |
| learned combination | 40 | 0.356 | −0.0369 | < 0.001 | 2 | 0 / 16 | 0.026 | | 0.12 |
| URIEL+ inventory | 40 | 0.355 | −0.0382 | < 0.001 | 1 | 0 / 16 | 0.027 | 0.366 | 0.03 |
| URIEL+ syntactic | 40 | 0.354 | −0.0393 | < 0.001 | 0 | 0 / 19 | 0.028 | 0.448 | 0.00 |
| URIEL+ geographic | 40 | 0.352 | −0.0410 | < 0.001 | 1 | 0 / 22 | 0.030 | 0.298 | 0.00 |
| URIEL+ featural | 40 | 0.351 | −0.0419 | < 0.001 | 1 | 0 / 16 | 0.031 | 0.416 | 0.00 |
| URIEL+ genetic | 40 | 0.348 | −0.0453 | < 0.001 | 0 | 0 / 22 | 0.034 | 0.443 | 0.05 |
| URIEL+ morphological | 40 | 0.345 | −0.0482 | < 0.001 | 1 | 0 / 22 | 0.037 | 0.298 | 0.00 |
| SVG | 40 | 0.337 | −0.0558 | < 0.001 | 0 | 0 / 26 | 0.045 | 0.158 | 0.05 |
| econd-hm | 40 | 0.333 | −0.0596 | < 0.001 | 0 | 0 / 35 | 0.049 | 0.148 | 0.00 |
| AUS rule (geography) | 40 | 0.307 | −0.0857 | < 0.001 | 0 | 0 / 35 | 0.075 | | 0.00 |

- **Routes chosen** (`routes.csv`, 200 pair-splits per measure). quality, its −log variant, RSIM,
  resource, econd-hm, every URIEL+ kind and the AUS syntax rule always pick a single intermediate.
  quality goes through en in 82% (es 16%), RSIM 80%, via hub 90% (ru for the 4 English-source
  pairs). SVG, AUS geography and the learned combination often pick 2–3 intermediates (SVG 45%,
  geography 78%, learned 51%) and lose most.
- **Post hoc (not in the plan):** ranking each pair's shortlist by hop count alone (fewer is
  better) gives Kendall τ 0.51, because every shortlist mixes one- and two-hop routes and the
  two-hop routes are worse. Only quality (both variants) and RSIM rank routes better than hop
  count alone; resource, URIEL+ and the spectral measures rank them worse.

**Oracle (B′).** The oracle's best route has one intermediate in 37 pairs, two in 3. Its mean ΔMRR
is above 0 in 10 pairs (mean gain 0.004; largest mt>hi +0.012), but in no pair is its interval
above 0 in 3 or more splits (mt>hi: 1 split). Exhaustive exact check (all 784 routes on split 0,
pairs drawn with seed 0): the true best route was in the shortlist in both pairs: th>eu
(th-fa-it-eu, MRR 0.217, nearest-neighbour rank 5) and tr>ga (tr-ru-hu-ga, 0.118, rank 4).

**By stratum and source** (Wilcoxon p per stratum; the 4 English-source pairs can't go below
p = 0.125 and every measure has that value).

| route choice | (a) ΔMRR | p | (b) ΔMRR | p | en source (4) | other sources (36) |
|---|---|---|---|---|---|---|
| direct MRR | 0.563 | | 0.223 | | 0.767 | 0.351 |
| oracle | −0.0136 | < 0.001 | −0.0083 | 0.007 | −0.0203 | −0.0099 |
| quality (−log) | −0.0156 | < 0.001 | −0.0109 | < 0.001 | −0.0227 | −0.0122 |
| quality | −0.0153 | < 0.001 | −0.0122 | < 0.001 | −0.0228 | −0.0127 |
| RSIM | −0.0174 | < 0.001 | −0.0117 | < 0.001 | −0.0286 | −0.0130 |
| via hub | −0.0187 | < 0.001 | −0.0112 | < 0.001 | −0.0352 | −0.0127 |
| resource | −0.0233 | < 0.001 | −0.0255 | < 0.001 | −0.0352 | −0.0232 |
| AUS rule (syntax) | −0.0353 | < 0.001 | −0.0296 | < 0.001 | −0.0325 | −0.0328 |
| URIEL+ phonological | −0.0363 | < 0.001 | −0.0328 | < 0.001 | −0.0514 | −0.0327 |
| learned combination | −0.0335 | < 0.001 | −0.0403 | < 0.001 | −0.0398 | −0.0366 |
| URIEL+ inventory | −0.0406 | < 0.001 | −0.0359 | < 0.001 | −0.0483 | −0.0371 |
| URIEL+ syntactic | −0.0443 | < 0.001 | −0.0343 | < 0.001 | −0.0392 | −0.0393 |
| URIEL+ geographic | −0.0492 | < 0.001 | −0.0328 | < 0.001 | −0.0430 | −0.0407 |
| URIEL+ featural | −0.0469 | < 0.001 | −0.0368 | < 0.001 | −0.0575 | −0.0401 |
| URIEL+ genetic | −0.0573 | < 0.001 | −0.0333 | < 0.001 | −0.0575 | −0.0439 |
| URIEL+ morphological | −0.0613 | < 0.001 | −0.0350 | < 0.001 | −0.0678 | −0.0460 |
| SVG | −0.0473 | < 0.001 | −0.0643 | < 0.001 | −0.0461 | −0.0569 |
| econd-hm | −0.0644 | < 0.001 | −0.0549 | < 0.001 | −0.0555 | −0.0601 |
| AUS rule (geography) | −0.1021 | < 0.001 | −0.0694 | < 0.001 | −0.1360 | −0.0801 |

**Cross-validated over pairs (stage C, 5 folds).** The exponent p and the threshold τ are chosen
on the training folds and applied to the held-out pairs.

| measure | p chosen (pairs: 1 / 1.5 / 2 / 3) | MRR at chosen p − at p = 1 | threshold rule − always direct | pair-splits chained (of 200) |
|---|---|---|---|---|
| quality (−log) | 40 / 0 / 0 / 0 | +0.0000 | −0.0002 | 30 |
| quality | 8 / 0 / 32 / 0 | −0.0001 | −0.0004 | 25 |
| RSIM | 0 / 8 / 24 / 8 | +0.0004 | −0.0003 | 20 |
| resource | 40 / 0 / 0 / 0 | +0.0000 | −0.0001 | 5 |
| URIEL+ phonological | 40 / 0 / 0 / 0 | +0.0000 | −0.0008 | 5 |
| URIEL+ inventory | 0 / 0 / 40 / 0 | +0.0009 | −0.0035 | 15 |
| URIEL+ syntactic | 0 / 0 / 8 / 32 | +0.0008 | −0.0011 | 10 |
| URIEL+ geographic | 40 / 0 / 0 / 0 | +0.0000 | −0.0007 | 5 |
| URIEL+ featural | 40 / 0 / 0 / 0 | +0.0000 | −0.0003 | 5 |
| URIEL+ genetic | 40 / 0 / 0 / 0 | +0.0000 | +0.0000 | 0 |
| URIEL+ morphological | 0 / 0 / 32 / 8 | +0.0018 | −0.0014 | 5 |
| SVG | 40 / 0 / 0 / 0 | +0.0000 | −0.0021 | 5 |
| econd-hm | 40 / 0 / 0 / 0 | +0.0000 | −0.0001 | 5 |

The threshold rule mostly learns never to chain. Its small losses come from held-out pairs whose
direct cost exceeds every training pair's: τ is chosen among the training pairs' costs, which
can't express "never chain", so those pairs always chain (all of the −0.0035, −0.0021 and −0.0011
of URIEL+ inventory, SVG and URIEL+ syntactic). Post hoc:
a weaker direct edge does go with a smaller chain deficit (Spearman 0.36, p = 0.02), but the
largest oracle gain is +0.012, so there is nothing to win. The learned combination
(cross-validated the same way) is in the first table: ΔMRR −0.037, worse than the quality edge
alone. Post hoc: it overrates its picks (predicted MRR 0.237, true validation MRR 0.190 where
known), and 69 of its 200 picks are 3-intermediate routes outside its training set (ΔMRR −0.053);
its validation target agrees with the test (Spearman 0.95 over 2,825 routes). Picking over all
20,440 chains, rather than among the routes it was trained on, was an implementation choice
(plan ledger) and drives this result.

**Robustness (high-agreement subset, 534 concepts).** No measure's mean ΔMRR changes sign; the
deficits shrink to about half, as in phase 3, while direct MRR is unchanged (0.394 vs 0.393). This
fits senses drifting through the intermediate on low-agreement concepts, but it isn't a causal
estimate: the subset changes both the training concepts (about 427 instead of 706) and the test
concepts. Restricting only the test concepts, with maps fitted on the full list, would isolate the
effect. The oracle and dᵖ rows of `summary_measures.csv` on
the subset cover only the pairs where their route coincides with a headline route (oracle: 20
pairs), so they aren't compared here.

| route choice | ΔMRR full | ΔMRR subset | below (≥ 3 of 5), subset |
|---|---|---|---|
| quality (−log) | −0.0133 | −0.0077 | 0 |
| quality | −0.0138 | −0.0082 | 0 |
| RSIM | −0.0146 | −0.0071 | 0 |
| via hub | −0.0150 | −0.0081 | 0 |
| resource | −0.0244 | −0.0129 | 0 |
| AUS rule (syntax) | −0.0328 | −0.0189 | 2 |
| URIEL+ phonological | −0.0346 | −0.0192 | 3 |
| learned combination | −0.0369 | −0.0245 | 3 |
| URIEL+ inventory | −0.0382 | −0.0181 | 1 |
| URIEL+ syntactic | −0.0393 | −0.0217 | 1 |
| URIEL+ geographic | −0.0410 | −0.0207 | 3 |
| URIEL+ featural | −0.0419 | −0.0220 | 4 |
| URIEL+ genetic | −0.0453 | −0.0236 | 6 |
| URIEL+ morphological | −0.0482 | −0.0274 | 5 |
| SVG | −0.0558 | −0.0315 | 6 |
| econd-hm | −0.0596 | −0.0331 | 9 |
| AUS rule (geography) | −0.0857 | −0.0516 | 20 |

**Best-hub frequency** (`best_hub.csv`). The oracle's best route goes through en in 18 of 40 pairs
(9 per stratum), i.e. 18 of the 36 pairs without en as an endpoint, against Anastasopoulos &
Neubig's 17 of 90. Next: es (10), it (3). The comparison is loose: their "hub" is the centre of a
jointly trained multilingual space, ours the intermediate of a composed chain, and their
dictionaries for non-English pairs are triangulated through English
(`analyses/2026-11-16-related-work/`).

### Against the expected outcomes
1. **Always chain loses: confirmed.** Mean ΔMRR < 0 for every measure overall and in each stratum;
   every measure is significantly *below* direct (p < 0.001 overall, ≤ 0.007 per stratum); no
   measure's route is above direct in 3 of 5 splits in any pair.
2. **The oracle rarely finds a winner: confirmed.** 0 of 40 pairs with the interval above 0 in
   ≥ 3 splits; mean gain 0.004 in the 10 pairs where the mean is positive. The shortlist contained
   the true best route in both checked pairs.
3. **Measure ranking: partly confirmed.** quality is first (lowest regret, highest τ; the −log
   variant marginally better), as expected. But RSIM is level with it (regret 0.004, τ 0.60), not
   below resource; the spectral measures (SVG, econd-hm) are the worst, not second (τ 0.15–0.16);
   and URIEL+ τ is 0.30–0.45, not near 0. Observed order: quality ≈ RSIM > resource ≈ URIEL+ >
   spectral. Given the post hoc hop-count τ of 0.51, only quality and RSIM carry information
   beyond the number of hops. The −log variant and dᵖ change little: confirmed (no p gains more
   than 0.002 over p = 1). Resource's place comes from an alphabetical tie-break (see Known
   issues); with ties broken by size it equals via hub and sits level with quality and RSIM.
4. **The learned combination is no better than quality alone: confirmed, and it is clearly worse**
   (−0.037 vs −0.014); it picks two or three intermediates in half the pair-splits.
5. **The threshold rule doesn't beat always direct: confirmed** (every measure 0 to −0.004).
6. **Routes through English do relatively well: partly confirmed.** Via hub is among the best
   chains for non-English sources (−0.0127, level with quality), and quality and RSIM pick en
   about 80% of the time. But en is the oracle's hub in exactly half of the 36 pairs without en as
   an endpoint (18 of 40 overall), not more than half; well above Anastasopoulos & Neubig's
   17 of 90, though the settings differ (see Best-hub frequency).
7. **Stratum (b) loses less than (a): confirmed in absolute ΔMRR** for 14 of the 17 chain choices
   and the oracle (not resource, SVG, learned). Relative to its much lower direct MRR (0.223 vs
   0.563), (b) loses a larger share. Not tested formally.
8. **English-source pairs favour direct most: confirmed for the oracle and the best chains**
   (oracle −0.020 vs −0.010 for other sources; quality −0.023 vs −0.013), not for every measure
   (AUS syntax and URIEL+ syntactic are level). Only 4 pairs.
9. **Robustness: first half confirmed, second half not.** No sign changes. The advantage of routes
   through English doesn't shrink on the subset: via hub moves from 4th to 3rd among chains and
   draws level with quality (−0.0081 vs −0.0082). The oracle wasn't run on the subset, so its hub
   can't be compared.

### Known issues (found in code review, 2026-11-16; not rerun)
Two reviewer agents read the code after the runs. No issue changes the conclusion that no chain
beats direct; the user decided not to rerun phases 3–4 and to fix these from phase 5 on
(`docs/decisions.md`, 2026-11-16). The numbers above are as run at the commits in `run_info.json`.
1. **Gold sets miss other casings.** Each gold word is one vocabulary row (its most frequent
   casing), but retrieval ranks the whole cased vocabulary, so retrieving `halk` for gold `Halk`
   counts as a miss. Every phase 4 target with case is affected (el, et, eu, fi, ga, hu, id, it,
   ru, sw, tl, tr: 24 of the 40 pairs); the uncased targets (ar, he, hi, ja, ka, ko, ta, th, ur, zh)
   aren't. The reviewer measured direct en→tr MRR
   0.667 → 0.802 with all casings counted, and in a small sample of chains the ΔMRR moved by up to
   −0.02 (en→de→tr −0.037 → −0.054; fr→en→fi −0.005 → −0.014): the fix would make chains look
   worse, not better. Absolute MRRs, stratum means and the English-source comparison are
   understated. The quality edge and the learned combination's validation answers share the issue.
2. **English-side sense mismatches in the list.** Lowercasing WordNet lemmas and Wiktionary
   headwords attaches the translations of capitalized senses to lowercase English words: *may*
   (modal) carries the month in every other language, *turkey* the country; also march, august,
   jordan, guinea, eve, mercury, smith, bush, cork (about 10–12 of 883 concepts, 1–3 test words per
   split on English edges; some are in the robustness subset, whose check compares only
   non-English languages). So English rows aren't fully "clean by construction" (expected
   outcomes 6 and 8).
3. **Identical-word exclusion compares with English.** Test concepts whose target word equals the
   English word were dropped; brief 5.2 asks for source–target identical pairs. Right for the 4
   English-source pairs; for the others it drops some non-identical concepts (22 for ru→fi) and
   keeps some identical ones (22 for hu→id). A few test words per split per pair.
4. **Resource ties.** The resource cost, (larger size rank + 1)/30, ties every intermediate larger
   than both endpoints; same-hop ties go to the lowest node index, i.e. alphabetical order. 34 of 40
   pairs have tied cheapest routes (2–27 routes each), and the routes run went via de 19 times, ar
   9, en 8, ru 4 on split 0. Breaking ties by intermediate size gives the via-hub route in all 40
   pairs (checked). Post hoc row (the via-hub numbers, already evaluated):

   | route choice | MRR | ΔMRR mean | Wilcoxon p | won | above / below | regret | top pick |
   |---|---|---|---|---|---|---|---|
   | resource, as run | 0.369 | −0.0244 | < 0.001 | 4 | 0 / 10 | 0.013 | 0.12 |
   | resource, ties by size (post hoc) | 0.378 | −0.0150 | < 0.001 | 5 | 0 / 3 | 0.004 | 0.40 |

   Its Kendall τ (0.42) doesn't change, and every resource^p graph is identical to p = 1 for the
   same reason.
5. **Gold breadth varies by pair.** Accepted answers per word range from about 16 (en→ar mean) to 1
   (ta→tl median), and only English sources get Wiktionary answers. ΔMRR within a pair is fair
   (direct and chain share the gold); comparisons across pairs and strata partly reflect answer
   counts.
6. **Smaller:** the robustness rows of `summary_measures.csv` for the oracle and dᵖ cover only some
   pairs (noted above); `routes.csv`'s `cost` column holds predicted MRR (higher is better) for
   `learned` and a path cost (lower is better) for everything else; the evaluation cache key holds
   no evaluation settings, which is safe here because all stages ran the same code.
