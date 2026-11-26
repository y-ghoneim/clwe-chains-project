# Figure: chain minus direct by how many times larger the chain's dictionaries are

**Status: requested by the user, 2026-11-25.** The professor and the partner found the N\*(H)
figure (`../break-even-grid/`) hard to read. After several drafts (lines per hub size, a heatmap,
the gain against the ratio, learning curves, flat direct lines), the user chose this form, from
their own sketch. It replaces the N\*(H) figure in the report. A report figure from committed
results; no new runs or numbers.

**What it shows.** Both chain dictionaries cut to H word pairs, the direct one to N. x is H / N,
how many times larger each chain dictionary is than the direct one; y is MRR, one line for the
chain and one for the direct map. The lines cross at 4×: the chain is better when each of its
dictionaries is more than about four times the direct dictionary, the direct map below that.

## Design
- **Data:** `experiments/break-even-grid/results/cells.csv` (commit 7c08d0a).
- **Pairs:** the 15 pairs present in every size combination, so a change of pairs cannot move the
  lines. Each pair's two directions are averaged, then the pairs per combination, then the
  combinations sharing a ratio (1 to 6 of them).
- **The heights mix sizes.** Each point averages combinations of different absolute sizes (at 4×:
  500 vs 2k up to 16k vs 64k), so the heights depend on that mix; the crossing does not: at 4×
  every combination's chain - direct is between -0.012 and +0.024 on these pairs (over all
  pairs present, -0.016 to -0.001). The caption says so.
- **No bands** (user, 2026-11-25): a range band over MRR shows the spread from absolute size, not
  doubt about which method wins, and made the crossing look uncertain.
- **Marks:** a dotted line at 4× with "direct is better" / "chain is better" arrows; colours are
  the dataviz reference palette's categorical slots 1 and 2, with distinct markers.

## Run
```
python figures/break-even-ratio/make_figure.py
```
Then copy `break_even_ratio.pdf` to `To Chain or Not to Chain_/report/figures/`.

## Second figure: six sample pairs (`break_even_ratio_pairs`)
The professor asked for graphs of individual pairs. For 6 pairs (2×3 panels, shared y-axis):
chain - direct (MRR) against the same ratio, each pair's two directions averaged; the line is the
mean over the pair's size combinations at each ratio, the band their range (drawn where there are
two or more). Redrawn 2026-11-25 (user) from two MRR lines to this gain-and-band form, after the
draft `proto_ratio3_band`; y label "difference between chain and direct (MRR)" (user). Lines
only where the pair's dictionaries allow. **Sample rule, fixed before any pair's curve was seen
(decisions.md 2026-11-25):** the 38 pairs present at hub 16k, ranked by their break-even direct
size at hub 16k (pairs still above zero at their largest direct size rank highest, ties by the
gain there), the pairs at the 0th, 20th, 40th, 60th, 80th and 100th percentile (nearest rank):
Chinese–Tamil (1,497), Korean–Finnish (2,625), Turkish–Russian (3,467), Hebrew–Greek (4,026),
Turkish–Irish (7,016), Kazakh–Tagalog (above 2,000, its largest direct size). At hub 16k these
break-even sizes give ratios of about 2× to about 11×.

**Dotted lines at each pair's own crossing** (user, 2026-11-26): each panel's dotted line marks
where the pair's mean line crosses zero (interpolated on log2 of the ratio, as the main figure's
4.33x), labelled with its value, instead of a fixed line at 4×: Chinese–Tamil 9.6×,
Korean–Finnish 6.2×, Turkish–Russian 4.9×, Hebrew–Greek 3.6×, Turkish–Irish 2.6×,
Kazakh–Tagalog 3.8×. The report quotes these (2.6× to 9.6×). They pool all of a pair's size
combinations, so they differ from the hub-16k ratios above, and their order is not the sample
order (Kazakh–Tagalog ranks last at hub 16k but crosses at 3.8×).

**Sample redrawn by stratum** (user, 2026-11-26; replaces the six-percentile sample above, after
its curves had been seen): within each stratum of `experiments/graph-all-pairs/pairs.csv`, (a) two
high-resource languages (20 pairs at hub 16k) and (b) at least one weaker (18), the pairs at the
0th, 50th and 100th percentile of the same hub-16k ranking (nearest rank); (a) on the top row, (b)
below, the stratum in each title. Picks and their crossings: (a) Japanese–Hungarian 7.3×,
Korean–Turkish 4.7×, English–Japanese 1.9×; (b) Chinese–Tamil 9.6×, Estonian–Hebrew 4.3×,
Kazakh–Tagalog 3.8×. The report quotes 1.9× to 9.6×.

## Redrawn 2026-11-25 (user): gain and band, like the sample pairs
The main figure now shows chain - direct (MRR) against the ratio, not the two MRR lines: the line
is the mean over the size combinations with that ratio (each over the pairs present: all pairs,
not only the 15 present everywhere, since a difference is not moved by the mix of sizes the way
the MRR heights were), the band their range. The mean crosses zero at 4.33x; at 4x the six
combinations are -0.016 to -0.001. The section "Design" above describes the earlier two-line
version (kept for the record).
