# Word-translation scores for the AfriSenti pairs (companion of the downstream sentiment run)

Stage `af` of phase 6's machinery, run by stage D's runner (`run.py` here is identical to
`../natural-highres-d/run.py`). Its only purpose is to give the downstream experiment
(`../downstream-afrisenti/README.md`, approved 2026-11-19) a BLI score for each of its 18
directions (and their reverses), and the dictionaries and hubs the downstream maps use.

- **Pairs:** X ∈ {yo, sw, am} × Y ∈ {en, ar, de, es, fr, it}, listed in `pairs_spec.csv` (fixed
  before any result; both directions run). yo and am are not graph nodes: their list words are
  built as stage 1 builds the low-resource languages' (`low` in the stage config).
- **Everything else as stage D** (`../natural-highres-d/README.md`) and stage `u`
  (`../downstream-umsab-bli/README.md`): natural dictionaries, hub = the node maximizing
  min(|s–x|, |x–t|) and never an endpoint, 5 disjoint folds with test words removed from every
  edge, CSLS (k = 10), methods thin (the full direct dictionary), chain and combination, ΔMRR with a
  paired bootstrap interval. Descriptive.
- Expected (from stages 1, A, A2 and D): the chain beats direct for yo with ar, de, es, fr, it;
  little difference for sw and am.
- Not a result in its own right: it is reported only through the downstream comparison.
