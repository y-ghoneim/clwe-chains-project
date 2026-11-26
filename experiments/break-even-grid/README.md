# Break-even grid: direct dictionary size × hub-route size (BLI)

**Status: APPROVED by the user, 2026-11-23 (measures as proposed: the per-pair median crossing as
N\*(H), the pooled mean as a check). Run 2026-11-24 on the remote; results at the end.** Add-on
in the sense of brief §5.10, like controlled scarcity: it changes nothing in the other
experiments; its own folder, results and report paragraph. Decisions so far: decisions.md 2026-11-23 ("break-even grid").

**Question (user).** Is there a direct dictionary size above which the chain cannot beat direct,
whatever the size of the hub routes? Controlled scarcity cut only the direct dictionary; the hub
routes kept their full size (narrowest step: median 59k pairs, 7k–156k). Real low-resource routes
are about 10× smaller (narrowest step: median 4.5k–6.8k in stages 1, A and A2). Here both are cut,
on the same fixed language pairs.

For each hub size H, **N\*(H)** is the direct size at which the chain's gain over direct reaches
zero: below it the chain wins, above it direct does.
- **N\*(H) levels off as H grows:** there is such a direct size, and N\*(64k) estimates it.
- **N\*(H) grows in proportion to H:** the gain depends only on the ratio r = log2(H / direct
  size), and there is no such size within the range tested.

## Design
- **Pairs:** stage D's 40 pairs (`experiments/natural-highres-d/pairs.csv`), both directions, each
  through its stage D hub; test words, folds, gold and CSLS evaluation exactly as stage D.
  Procrustes maps, as in controlled scarcity. Self-learning (strong maps) is not used: it grows
  every dictionary to about 10k induced pairs, which undoes the sizes.
- **Direct sizes: 500, 1k, 2k, 4k, 8k, 16k.** These are controlled scarcity's subsamples; its
  cached direct ranks (`v1/results/controlled-scarcity/b<size>/`) are reused, not rerun. 250 is
  left out (the grid's lowest hub size is 1k), and 32k is left out (8–10 pairs).
- **Hub sizes: 1k, 2k, 4k, 8k, 16k, 32k, 64k, both steps cut to the same size.** Steps of 2× on
  both axes, so cells with the same r lie on a diagonal. The full routes are only a reference
  (controlled scarcity's cached chain ranks): they are not one size (0.45× to 9.7× of 16k across
  pairs), and their two steps are unequal.
- **Cutting a hub step:** the edge's first H rows in a fixed random order (seed 0 with the edge's
  name), so smaller sizes are subsets of larger ones. The order belongs to the edge, so an edge
  shared by two pairs (stage D: en__tr is direct for en–tr and a hub step of ko–tr) is cut the
  same way in both. The fold's test words are removed after cutting, as controlled scarcity does
  for direct, so a step can be slightly below H.
- **Cells:** a pair is in the cell (direct N, hub H) if its full direct dictionary has more than N
  pairs and both of its full hub steps have more than H.

  | direct \ hub | 1k | 2k | 4k | 8k | 16k | 32k | 64k |
  |---|---|---|---|---|---|---|---|
  | 500–2k | 40 | 40 | 40 | 39 | 38 | 29 | 20 |
  | 4k | 35 | 35 | 35 | 35 | 35 | 28 | 20 |
  | 8k | 27 | 27 | 27 | 27 | 27 | 23 | 20 |
  | 16k | 16 | 16 | 16 | 16 | 16 | 16 | 15 |

  Exact counts from prep/20's pairs.csv and controlled scarcity's (the draft's fold-0 count had
  19 at 64k and 14 at 16k × 64k).
- **Method:** the chain alone. The chain's ranks depend only on H, and direct's only on N, so each
  is scored once per direction and fold; a cell's gain is chain − direct per test word, from the
  two sets of ranks (same folds and test words). The combination is left for later (it needs its
  own pass per cell, about 27 h).

## Measures (fixed now)
- **Per cell:** ΔMRR (chain − direct) per direction with the paired bootstrap over test words, the
  mean over the cell's directions, and the counts clearly above and below 0.
- **N\*(H), main estimate: the median over pairs of each pair's own crossing.**
  - A pair's gain at a direct size is the mean ΔMRR of its two directions.
  - Its crossing is the first direct size where the gain goes from positive to ≤ 0, placed by
    linear interpolation on log2(direct size) between the two sizes either side.
  - A pair whose gain is ≤ 0 already at 500 counts as "below 500"; one still positive at its
    largest available direct size counts as "above" that size. The median is taken with these
    placed at the low and high ends; when the middle value is one of them, N\*(H) is reported as
    "below 500" or "above X".
  - Pairs whose gain goes back above 0 after crossing are counted and reported.
  - 90% interval: 1,000 bootstrap resamples of the pairs, seed 0.
- **N\*(H), check: the pooled-mean crossing.** At each direct size, the mean ΔMRR over the pairs
  present, and its crossing by the same interpolation. It mixes pair sets (40 pairs at 2k, 16 at
  16k), so it is reported beside the main estimate. Where the two disagree, that is said.
- **Levels off or grows in proportion: how much N\* grows when the hub routes grow 4×, from 16k to
  64k.** Computed as the slope of log2 N\*(H) against log2 H over the three largest hub sizes
  (16k, 32k, 64k), turned into a growth factor (4^slope), with a 90% bootstrap interval over pairs.
  - No change is 1×; proportional growth (r only) is 4×.
  - **"Levels off" is claimed only if the interval lies below 2×** (user, 2026-11-23: halfway
    between 1× and 4× in doublings, and about the precision the grid's 2× direct steps allow);
    "grows in proportion" only if it contains 4× and excludes 1×; otherwise "undecided".
  - If N\*(H) is "above 16k" for any of the three, the slope cannot be computed. The answer is
    then "no break-even within direct ≤ 16k at that hub size".
- **Same-pairs check (user, 2026-11-23, added after the review, before any run):** the growth
  again on the pairs present at all three growth sizes only (20 pairs), with the same rule, so a
  change of pairs between 16k (38 pairs) and 64k (20) cannot move N\*. Reported beside the main
  result (growth.json `same_pairs`); if the two disagree, that is said.
- **Reproduction check:** the full-route chain rescored by this runner must equal controlled
  scarcity's cached chain ranks exactly, on a few pairs.

## Figures
1. **Main:** ΔMRR (chain − direct) against direct size (log scale), one line per hub size (7 lines,
   light to dark), the full-route line dashed, and a line at zero. Each line's zero crossing is
   N\*(H).
2. **The test:** N\*(H) against H, both log scales, with 90% intervals, and the slope-1 diagonal
   that proportional growth predicts. Censored values ("above 16k") are drawn as arrows.
3. **Optional:** a heatmap of the mean ΔMRR over the grid, with the zero contour.

## Expected outcomes (written before any run; guesses, to be marked wrong where wrong)
1. **At hub 1k the chain barely beats direct, if at all:** N\*(1k) is below 500 or near it.
2. **N\*(H) grows about in proportion to H at small hub sizes** (N\* ≈ H/2, r ≈ 1) **and levels
   off from about 16k:** N\*(16k), N\*(32k) and N\*(64k) all lie between 8k and 16k, near
   controlled scarcity's full-route break-even (between 8k, +0.020, and 16k, +0.001). The slope
   over 16k–64k lies below 0.5.
3. **At a fixed r the gain shrinks as both sizes grow:** the diagonals are not flat, so r alone
   does not predict the gain.
4. **The median crossing and the pooled-mean crossing agree within one grid step (2×).**

## Limits (for the write-up)
- **The cut steps keep strong vectors.** All 28 languages are well-resourced, so the grid
  separates dictionary size from vector quality but does not reproduce low-resource conditions.
- **Both steps have the same size, which real routes do not:** in stages 1, A and A2 the wide
  step is about 10× the narrow one.
- **"Whatever the hub size" means up to 64k**, beyond which the claim rests on the levelling-off
  between 16k and 64k.
- **One subsample per size (seed 0).** The folds vary the training words, but not which pairs are
  kept.
- **The chain alone:** the combination may behave differently (at full routes it beats direct
  at every direct size, +0.037 at full stage D).

## Build and compute (estimates are the assistant's)
- **Prep (`prep/20_break_even_grid.py`):** writes `v1/phase6/dh/h<size>/` with the cut hub steps
  for every pair, pairs.csv per hub size, and dict_sizes.csv. Deterministic, output path from the
  config. Tests: nesting, shared edges cut the same way, the pair rule. Builds in about 1 h.
- **Runner:** wraps stage D's runner (as controlled scarcity does) with the hub steps read from
  `dh/h<size>/` and the method list [chain]; then joins the cached direct ranks, computes the
  measures, and draws the figures.
- **Compute:** about 2,450 chain evaluations (245 pair-hub cells × 2 directions × 5 folds) at
  about 3.5 s: **about 2.5 h on the laptop, under 1 h on the remote.** Procrustes fits take
  seconds.
- **Resumable per hub size**, with results recording the commit hash and the config.

## Build notes (2026-11-23, before any result)
- **prep/20 ran** (`v1/phase6/dh/`, 51 MB, 6 s): 40, 40, 40, 39, 38, 29 and 20 pairs at 1k–64k.
  It uses prep/18's permutation, so a hub step cut here and a direct dictionary cut there share
  one order per edge.
- **The chain is scored alone** (`bli_multi`), not inside the combined pass controlled scarcity
  used; a test checks the two give the same ranks.
- **The chain is scored on the laptop; the direct ranks came from the remote.** With 300 or more
  training pairs the map is unique and the machines agree within 0.0001 MRR (strong-maps check,
  decisions.md 2026-11-23); every map here has at least 500 pairs. `--check` rescores the
  full-route chain for 2 pairs and compares it with controlled scarcity's (about 2 min of GPU).
- **Medians with an even number of pairs:** the mean of the two middle crossings on log2 scale;
  censored if either is, undefined if one is "below" and the other "above".
- **The growth with censored medians:** if a median at 16k, 32k or 64k is censored, the growth
  is not computed and growth.json gives the README's wording ("no break-even within direct ≤ 16k
  at hub …"). In the bootstrap, a draw counts as unbounded growth only when it plainly is: N\*
  above the grid at 64k (or below it at 16k) with the other two sizes numbers; every other
  censored draw is left out, and the shares of both are reported (`unbounded_share`,
  `undefined_share`). A censored draw is never counted as shrinking growth, which would pull the
  interval towards "levels off" (reviewer).
- **"Above X":** X is a lower bound for the median, each pair still positive at its largest
  direct size placed at that size.
- **`--check` passes** if every record's MRR is within 0.0001 of controlled scarcity's (the
  laptop–remote tolerance above), with the same test words; the tree must be clean.
- **Record:** run_info.json fingerprints the cut hub steps, this experiment's caches and every
  reused controlled-scarcity cache (b500–b16000 and bfull), with the commits each was scored at.
- **Reviewer pass (2026-11-23, before any run):** two bugs fixed (censored growth draws counted as
  −∞; grid.csv counted source languages, not pairs), plus the four items above and the folder name
  read from the config. Design notes kept for the write-up: pairs with small direct dictionaries
  can only be censored at 4k–8k, which pushes medians up; the pairs change across hub sizes (38,
  29, 20), so the growth mixes which pairs with how large the hub is; N\* above 16k direct cannot
  be measured.
- **Outputs:** results/cells.csv, grid.csv, crossings.csv, nstar.csv, growth.json, check.csv,
  run_info.json. Tests: `tests/test_break_even_grid.py`.

## Questions settled with the user (2026-11-23)
1. The measures as proposed; the levelling-off rule as a growth factor below 2×.
2. Run on the remote (user, 2026-11-23 evening), not the laptop.

## Results (run 2026-11-24, remote RTX 4500, commit 504c9ad)
**Record.** `--check` passed: the full-route chain gave exactly controlled scarcity's ranks (2
pairs, 20 records, every rank equal). All 367 input fingerprints in run_info.json match the
laptop's files. Cell counts as in the table above. Direct ranks reused from controlled scarcity
(commit 9baf6ac).

**Answer: no direct size beats the chain at every hub size within the grid. N\*(H) grows in
proportion to H: direct wins once it is about a quarter of each hub step's size.**

| hub size H | 1k | 2k | 4k | 8k | 16k | 32k | 64k | full routes |
|---|---|---|---|---|---|---|---|---|
| pairs | 40 | 40 | 40 | 39 | 38 | 29 | 20 | 40 |
| N\*(H), median over pairs | below 500 | below 500 | 927 | 1,818 | 3,734 | 6,051 | 15,476 | above 8,000 |
| 90% interval | | | 838–1,087 | 1,769–2,308 | 3,386–4,507 | 4,878–8,717 | 8,687–above 16k | |
| pooled-mean crossing | below 500 | below 500 | 985 | 1,983 | 3,719 | 6,815 | 11,901 | above 16,000 |
| pairs crossed / below / above | 0 / 40 / 0 | 14 / 26 / 0 | 38 / 2 / 0 | 37 / 0 / 2 | 34 / 0 / 4 | 25 / 0 / 4 | 11 / 0 / 9 | 13 / 0 / 27 |

- **Growth from 16k to 64k: 4.1× (90% interval 2.4× to unbounded; 38% of draws above the grid at
  64k): "grows in proportion"** by the rule fixed before the run. **Same-pairs check** (the 20
  pairs present at 16k, 32k and 64k): 4.3× (2.5× to unbounded), the same verdict.
- **N\*/H is about 0.23 at every hub size from 4k to 64k** (0.19 at 32k). No pair's gain went
  back above 0 after crossing.
- **Mean ΔMRR (chain − direct) over the grid** (`grid.csv`):

  | direct \ hub | 1k | 2k | 4k | 8k | 16k | 32k | 64k |
  |---|---|---|---|---|---|---|---|
  | 500 | −0.118 | −0.016 | +0.106 | +0.205 | +0.271 | +0.339 | +0.392 |
  | 1k | −0.227 | −0.124 | −0.002 | +0.094 | +0.158 | +0.210 | +0.254 |
  | 2k | −0.320 | −0.217 | −0.095 | −0.001 | +0.061 | +0.102 | +0.137 |
  | 4k | −0.425 | −0.310 | −0.175 | −0.072 | −0.007 | +0.030 | +0.061 |
  | 8k | −0.502 | −0.371 | −0.222 | −0.114 | −0.046 | −0.009 | +0.016 |
  | 16k | −0.597 | −0.444 | −0.266 | −0.146 | −0.072 | −0.033 | −0.012 |

- **The break-even lies on one diagonal.** At hub = 4 × direct (r = 2) the mean gain is between
  −0.016 and −0.001 at every size, from 500 against 2k to 16k against 64k. Above that diagonal
  the gain at a fixed ratio shrinks as both sizes grow (r = 3: +0.106 at 500, +0.016 at 8k);
  below it the loss shrinks.

**Expected outcomes, marked:**
1. At hub 1k the chain barely beats direct, N\*(1k) below 500 or near it: **right**, and more so:
   the chain loses in every pair at hub 1k, even against 500 direct pairs.
2. N\* grows with H at small sizes and levels off from about 16k: **wrong.** It keeps growing in
   proportion up to 64k (N\* ≈ H / 4, not H / 2), and the slope over 16k–64k is about 1, not
   below 0.5.
3. At a fixed ratio the gain shrinks as both sizes grow: **partly.** Where the chain wins, yes;
   at the break-even ratio (r = 2) it stays at zero.
4. Median and pooled-mean crossings within 2× of each other: **right** (within 1.3×).

**For the write-up** (limits above apply): within dictionaries up to 64k a side and well-resourced
languages, the chain's advantage is set by the ratio of hub-step size to direct size, not by the
direct size alone. With both steps cut to the same size, direct overtakes the chain once it has
about a quarter as many pairs as each step.
