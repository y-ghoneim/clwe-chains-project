# Word-translation scores for the MasakhaNEWS pairs (companion of the downstream run)

Stage `mn` of phase 6's machinery, run by stage D's runner (`run.py` here is identical to
`../natural-highres-d/run.py`). It gives the downstream experiment
(`../downstream-masakhanews/README.md`, approved 2026-11-20) the dictionaries, hubs and a BLI
score for each of its 8 directions (and their reverses).

- **Pairs:** X ∈ {yo, sw, am, so} × Y ∈ {en, fr}, in `pairs_spec.csv` (fixed before any result;
  both directions run). yo, am and so are not graph nodes: their list words are built as stage 1
  builds the low-resource languages' (`low` in the stage config).
- **Everything else as stage D and stages `u`, `af`:** natural dictionaries, hub = the node
  maximizing min(|s–x|, |x–t|), never an endpoint, 5 folds with test words removed from every
  edge, CSLS (k = 10), methods thin (the full direct dictionary), chain and combination. The yo,
  sw and am pairs are pairs of stage `af` and should reproduce its numbers.
- Not a result in its own right: it is reported only through the downstream comparison.
