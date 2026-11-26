# Figure: the chain's gain as the direct dictionary is cut

**Status: requested by the user, 2026-11-25.** It carries the reduced-dictionary word-translation
results in the report (no table; decisions.md 2026-11-25). It replaces an earlier
`direct_size_sweep.pdf` in the report folder that had no script and showed only the 16 pairs
present at every size. A report figure from committed results; no new runs or numbers.

**What it shows.** The 40 distant pairs with their direct dictionary cut to 250 ... 16,000 word
pairs while the chain keeps its full dictionaries: the gain over the cut direct map of the chain,
pooled direct and the combination. The chain's gain falls from +0.417 at 250 to +0.001 at 16,000
and is -0.005 with the full dictionary; the combination is behind the chain below 1,000 and ahead
of it from 2,000.

## Design
- **Data:** `experiments/controlled-scarcity/results/curve.csv` (commit 126a143): per size, the
  mean over the directions present at that size.
- **Pairs:** all pairs present at each size (a pair is present when its full direct dictionary is
  larger than the size): 40, 40, 40, 40, 35, 27, 16, and 40 at full. This matches the report's
  text. The counts are printed under each size.
- **No direct MRR row** (user, 2026-11-25: too cluttered). The direct map's MRR (0.096 at 250)
  goes in the text instead.
- **x-axis:** sizes evenly spaced (each doubles the last), "full" set apart after a dashed break.
- **No intervals** (user, 2026-11-25).
- **Colours:** the dataviz reference palette's categorical slots 1-3 in fixed order (chain blue,
  pooled direct orange, combination aqua), with distinct markers so identity is not colour alone.
- A two-panel version (absolute MRR on the left) was tried and dropped: with the pair set shrinking
  above 2,000, its left panel showed the unchanged chain rising from 0.51 to 0.71.

## Run
```
python figures/direct-size-sweep/make_figure.py
```
Then copy `direct_size_sweep.pdf` to `To Chain or Not to Chain_/report/figures/`.
