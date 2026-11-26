# Word-translation scores for the UMSAB pairs (companion of the downstream sentiment run)

Stage `u` of phase 6's machinery, run by stage D's runner (`run.py` here is identical to
`../natural-highres-d/run.py`). Its only purpose is to give the downstream experiment
(`../downstream-umsab/README.md`, approved 2026-11-19) a BLI score for each of its 42 directions,
for the Spearman correlations specified there.

- **Pairs:** the 21 unordered pairs among the 7 UMSAB languages that are graph nodes (en, ar, de,
  es, fr, hi, it), listed in `pairs_spec.csv` (fixed before any result; both directions run).
- **Everything else as stage D** (`../natural-highres-d/README.md`): natural dictionaries (PanLex
  in-vocab pairs plus English Wiktionary on pairs with en), hub = the node maximizing
  min(|s–x|, |x–t|) and never an endpoint, test words = the shared list's concepts (capitalized
  senses dropped) both languages have minus identical strings, 5 disjoint folds with test words
  removed from every edge, CSLS (k = 10) over the target's full vocabulary, gold = list word plus
  PanLex (and English Wiktionary with en) translations with casing variants, methods thin
  (here the full direct dictionary), chain and combination in one pass, ΔMRR with a paired
  bootstrap interval. Descriptive results; the expected outcomes of stage D apply (direct beats the
  chain for most pairs with large dictionaries).
- **Compute:** about 420 CSLS evaluations (42 directions × 5 folds × 2), ~10 min on the laptop.
- Not a result in its own right: it is reported only through the downstream comparison.
