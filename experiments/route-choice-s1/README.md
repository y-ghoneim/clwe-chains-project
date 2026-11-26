# Route choice under natural dictionaries (phase 6, stage 1E)

**Question.** On stage 1's 77 naturally scarce pairs, with every edge trained on its own natural
dictionary: does a language graph pick routes (one or two intermediates) that beat thin direct and
the stage 1 hub rule, which edge measures pick well, and how far is the best possible route (the
oracle) from those picks? If the oracle barely beats the stage 1 hub, route choice doesn't matter
once supervision is natural and "use the best-resourced hub" is enough; if it does, the graph
matters again.

decisions.md 2026-11-17 ("stage 1E"); data and evaluation as `experiments/scarce-natural-s1/`
(spec `docs/superpowers/specs/2026-11-16-phase6-stage1-design.md`). This README is written and
approved before any run. Results are descriptive (no success rule; user, 2026-11-17).

## Data (as stage 1)
- The 77 pairs (`../scarce-natural-s1/pairs.csv`), their eligible concepts, the same 5 folds, test
  words, gold and query rows (`v1/phase6/`).
- **Edges:** every edge a route can use, each with its own natural dictionary (PanLex in-vocab
  pairs, plus English Wiktionary on edges with en): s–x and x–t for every node x ≠ t, and x–y
  between any two nodes (two-hop routes). Built on the laptop by `prep/09_phase6_route_dicts.py`
  into `v1/phase6/routes/`.
- **Test removal per fold:** a pair is removed from an edge if its s-side word is a test word of s
  or its t-side word a test word of t (all casings); edges between two nodes other than t carry no
  test words of the pair and are unchanged.

## Routes
- Intermediates: the 30 nodes except t (and never another low-resource language). One-hop routes
  s→x→t (29 per pair) and two-hop routes s→x₁→x₂→t (812 per pair): 841 routes per pair.
- Each step is Procrustes on its own fold's dictionary; steps are composed (as stage 1's chain).

## Edge measures (computed without test words) and the graph's pick
Each edge's size is min(distinct a words, distinct b words) of its full natural dictionary.
Validation pairs (for measures 3–4) come from the step's own dictionary: on a step s → x, every
pair touching a list word of s is dropped first; on x → t, every pair touching a list word of t;
on a step between two intermediates, nothing (those carry no test words of the pair). So no fold's
test word is ever used. Validation = the most frequent 20% of the remaining source words (at most
500) with all their translations; the map is fitted on the rest. A step with fewer than 10
validation words gets the worst cost (quality 0, RSIM −1).

1. **Size, widest path:** the route whose smallest edge is largest (stage 1's rule, extended to
   two intermediates). It equals the stage 1 hub for nearly every pair; where two-hop routes tie
   on the same bottleneck, the tie goes to node order.
2. **Size, sum:** the smallest sum of log(M / size) over the route's edges, M = the largest route
   edge (about 51,000 words). Never negative, so an extra hop can't lower a route's cost (the
   first draft's −log(size) could, and picked two hops for every pair; final review C1).
3. **Quality edge:** cost 1 − MRR (nearest neighbour over the target's full vocabulary) on the
   edge's validation words, per direction; the smallest sum.
4. **RSIM** (Vulić et al. 2020): cost 1 − Pearson correlation of the two languages' similarity
   matrices on up to 500 validation pairs (one per source word); the smallest sum.
5. **URIEL baseline:** cosine distance between lang2vec `syntax_knn` vectors (URIEL syntax with
   kNN imputation, which covers all 36 languages; v0's URIEL+ table has none of the six new
   ones); the smallest sum.

Ties go to fewer hops, then to the first route in node order. Nothing is tuned, so no
cross-validation is needed.

## Oracle (the ceiling)
Every route is scored by nearest-neighbour MRR on fold 0's test words, s→t only; the 5 best are
rescored like the picks. The **oracle** of a pair-direction is the best of every rescored route
(the picks, the stage 1 hub route and the 5 screened routes) by pooled CSLS MRR; regret and the
top-pick hit rate are against it. It uses test words to choose, so it is an upper bound on any
route choice, not a method (as in phase 4).

## Evaluation
Every distinct route a measure picks, the stage 1 hub route, the oracle's 5 and thin direct are
scored with exact CSLS (k = 10, full target vocabulary) on all 5 folds and both directions (the
route reversed for t→s), pooled over folds per pair-direction. Reported, descriptively:
- per measure: mean MRR and ΔMRR against thin direct and against the stage 1 hub route; pairs where
  it beats thin direct (paired bootstrap interval above 0); regret against the oracle's best;
  top-pick hit rate; how often it picks one or two intermediates and which hubs;
- the oracle: its best route's gain over thin direct and over the stage 1 hub; how often the
  stage 1 hub is the oracle's best;
- Kendall τ between each measure's route cost and CSLS MRR over each pair-direction's rescored
  routes (costs are the s→t ones for both directions, as the picks are).

## Expected outcomes (written before any run)
- **The stage 1 hub (en, fr or hi) is usually the oracle's best one-hop route, or close to it.** Its
  edges are by far the largest; other hubs' s–x edges are small (often a few hundred pairs).
- **Two-hop routes win for some pairs** whose hub has a small edge to the target: or→hi→en→t and
  mg→fr→en→t could beat or→hi→t and mg→fr→t (hi–t and fr–t are much smaller than en–t). Elsewhere
  two hops add error and lose, as in phase 4.
- **Quality edge picks best among the measures, then the size measures; URIEL picks worst.** Quality
  reflects both how much supervision an edge has and how well it aligns; URIEL ignores supervision,
  which is now the main thing that varies between edges.
- **The oracle gap over the stage 1 hub is small (under ~0.02 MRR) for most pairs.** A large gap
  would mean route choice matters even when supervision is natural.

## Results (run 2026-11-17 on the remote RTX 4500, commit 66200d0)
73,769 cache records (screening, edge measures, 7,160 CSLS evaluations at 1.11 s); 156 min in all.
Every input fingerprint matches the laptop. Cross-check: thin direct equals stage 1's exactly, and
the hub route equals stage 1's chain to within 3e-5 MRR (float noise of composing cached maps).

Per chooser over the 154 pair-directions (`summary.csv`; ΔMRR against thin direct and against the
stage 1 hub route; regret against the oracle; τ = Kendall between route cost and MRR over each
pair-direction's rescored routes):

| chooser | MRR | Δ thin | above / below 0 | Δ hub | regret | top pick | two-hop | τ |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.059 | +0.043 | 141 / 0 | +0.0055 | 0 | — | 49% | — |
| stage 1 hub | 0.053 | +0.037 | 134 / 0 | 0 | 0.0055 | 37% | 0% | — |
| size, widest | 0.053 | +0.037 | 134 / 0 | −0.0001 | 0.0056 | 36% | 1% | 0.53 |
| size, sum | 0.053 | +0.037 | 134 / 0 | −0.0006 | 0.0061 | 32% | 21% | 0.45 |
| quality edge | 0.041 | +0.025 | 101 / 4 | −0.012 | 0.018 | 21% | 0% | 0.00 |
| URIEL (syntax_knn) | 0.020 | +0.004 | 37 / 9 | −0.033 | 0.039 | 5% | 0% | −0.12 |
| RSIM | 0.019 | +0.003 | 29 / 21 | −0.035 | 0.040 | 3% | 0% | −0.28 |

- **Route choice adds little beyond the best-resourced hub.** The oracle beats the stage 1 hub by
  a mean 0.0055 MRR (median 0.002; 90th percentile 0.018; at most 0.036), and the hub is the
  oracle's best route in 57 of 154 pair-directions. Where the oracle wins, it mostly adds a second
  hub to the same one (en-it, en-fr, hi-en, hi-fr) or swaps en and fr; half its best routes have
  two intermediates. The largest gap is Odia's (+0.013 over hi alone): its best routes go hi→en (8 of
  34 pair-directions) or hi→fr (5), as expected, but also en alone (7).
- **Dictionary size is the graph signal that works; similarity measures mislead.** Both size
  measures pick the hub (widest: 76 of 77 pairs; sum: en 61, fr→en 16 two-hop) and rank routes
  best (τ 0.53, 0.45). The quality edge picks a different route from the hub for 55 of 77 pairs
  (en for many mg and or pairs, it, id or es elsewhere) and loses 0.012 against the hub. RSIM and URIEL pick linguistically or structurally close but
  supervision-poor hubs (tl, et, ta; he, hi, ka) and lose nearly all of the chain's gain; their τ is
  negative (cheaper routes score worse).
- **Against the expected outcomes:** the hub is usually the oracle's best or close, yes; two-hop wins
  for some pairs (or, mg), yes; quality best among measures, **no** (third, behind both size
  measures); URIEL worst, nearly (RSIM is slightly worse); the oracle gap small for most pairs, yes.
- **Why quality fails here (post hoc, not tested):** its validation words are the most frequent
  words *outside* the list, and on s → x steps with a few hundred pairs it has few of them, so the
  measure is noisy exactly on the steps that decide the route; it also scores how well an edge
  aligns, not how much the chain gains from it.

## Compute (estimates; the run logs real times)
NN screening ≈ 64,800 routes on fold 0 (≈ 0.03 s each, ≈ 35 min on the RTX 4500); quality edge and
RSIM on ≈ 1,800 step variants (minutes); exact CSLS on ≈ 10 distinct routes per pair (thin direct
included) × 77 pairs × 5 folds × 2 directions ≈ 7,700 evaluations (≈ 2.1 h at the measured 1.0 s).
About 3 h in all. The remote needs three more vectors (de, es, it) and the route dictionaries
(0.8 GB).
