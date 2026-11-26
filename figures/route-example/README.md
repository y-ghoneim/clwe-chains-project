# Figure: one language pair's direct dictionary and its three widest hubs

**Status: requested by the user, 2026-11-25.** It replaces the full language-graph figure
(`../language-graph/`, 615 edges) in the report and doubles as the methods figure: the direct
map, the chain (widest hub) and several hubs (three widest hubs) on one real pair, with each
dictionary's size. Drafts tried first (the graph's maximum spanning tree; per-language direct
against chain dictionary sizes; Yoruba's dictionaries as spokes and bars) were not taken. A
report figure from existing data; no new runs or numbers.

## Design
- **Data:** `<phase6>/routes/dict_sizes.csv` (distinct word pairs per dictionary; the 30 nodes and
  stage 1's six low-resource languages); `experiments/scarce-natural-s1/pairs.csv`.
- **Pair, chosen by rule:** among the under-500 pairs, the one whose ratio (the widest hub's
  smaller dictionary / direct) is closest, in log, to the group's median (11.8): Odia → Finnish,
  direct 264, ratio 11.8.
- **Hubs** ranked as `prep/19_strong_hubs.py` ranks them (largest smaller dictionary first): Hindi
  (3,125 / 10,007), Tamil (2,100 / 3,632), English (1,481 / 158,466). The experiment's own hub file
  (`<phase6>/hubs/s1/hubs.csv`, in min-distinct-words units) has the same three in the same order.
- **Five hubs shown** (user, 2026-11-25): the three several hubs uses (blue, light blue) and the
  next two (grey, not used): Chinese (329 / 90,988), Arabic (303 / 36,939).
- **One line width** (user, 2026-11-25: the numbers are printed, so width would only repeat
  them); colour carries the role: the chain blue, the other two hubs light blue, the direct
  dictionary dashed grey. Odia, low-resource, in yellow.

## Run
```
python figures/route-example/make_figure.py
```
Then copy `route_example.pdf` to `To Chain or Not to Chain_/report/figures/`.
