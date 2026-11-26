# Figure: the break-even direct size against the hub-step size

**Status: requested by the user, 2026-11-24** ("a figure of the break-even thing"). It is figure 2
of `experiments/break-even-grid/README.md` ("Figures", fixed before the run); figures 1 and 3
come later. A report figure from committed results; no new runs or numbers.

**What it shows.** For each hub-step size H (both steps cut to H word pairs), the direct
dictionary size N\*(H) at which the chain stops beating direct. The medians rise along the
slope-1 line: the break-even grows in proportion to H (about H / 4), with no sign of levelling
off up to 64k.

## Design
- **Data:** `experiments/break-even-grid/results/nstar.csv` (commit 7c08d0a); the direct sizes'
  range from the experiment's `config.yaml`. The full-route row is left out (its routes are not
  one size).
- **Marks:** the median over pairs (blue line and dots) with its 90% pair-bootstrap interval
  (vertical bars); the pooled-mean crossing as hollow orange circles (the check); the dashed
  slope-1 line drawn at N\* = H / 4, a reference for growth in proportion to H, not a fitted line
  (the README's test is the growth factor, 4.1×).
- **Censoring:** at hubs 1k and 2k the median is "below 500" (down arrows from 500); at 64k the
  interval's upper end is open (an arrow up from 16,000). The bands below 500 and above 16,000
  are outside the direct sizes measured.
- **Axes:** both log2. The number of language pairs at each hub size is under its tick.
- **Style:** the dataviz skill's reference palette, light mode (categorical slots 1 and 2, its
  text and grid tokens). The skill's palette validator could not be run (no Node.js on the
  laptop); slots 1 and 2 are the pair its documentation records as passing, and the two series
  also differ by marker (filled against hollow), so colour is never the only cue.
- **Size:** 3.4 × 2.9 in, single column.

## Built (2026-11-24)
`python figures/break-even-grid/make_figure.py` writes `break_even.pdf` (for the report; the
same bytes on every run) and `break_even.png` (for reading in the repo). Plotted values checked
against `nstar.csv`: medians 927, 1,818, 3,734, 6,051, 15,476; pooled 985, 1,983, 3,719, 6,815,
11,901.
