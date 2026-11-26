# Our SIB-200 transfer next to the published XLM-R transfer

**Question.** The professor asked for baselines. SIB-200's paper (Adelani et al. 2024) reports
cross-lingual transfer with XLM-R per language. Where our downstream SIB-200 directions match its
setting, how does our transfer (static vectors, direct or chained map, logistic regression)
compare? (User, 2026-11-25: "lets get started on 1", the published reference block.)

**Design.** Published: XLM-R fine-tuned on English or Arabic, tested on the target language,
accuracy (%), Table 8 of arXiv v3 (2309.07445v3), copied into `published.csv` (read through a web
extraction of the HTML page; **to check against the PDF before submission**). Ours: the
`experiments/downstream-sib200` directions whose classifier language is en or ar and whose source
is one of our 7 low-resource languages (our "src → tgt" is their "trained on tgt, tested on src");
logistic regression's accuracy, direct, chain, no map; the same SIB-200 test sentences (204).
9 directions (the paper has no per-language French column). `python compare.py` →
`out/comparison.csv`.

**Answer** (2026-11-25). Mean accuracy (%):

| | our direct | our chain | XLM-R transfer |
|---|---|---|---|
| languages in XLM-R's pretraining (or, mg, as, jv; 6 directions) | 48.7 | 58.4 | 81.6 |
| languages not in it (yo, ckb, qu; 3 directions) | 40.2 | 50.8 | 36.3 |

- Languages XLM-R saw: XLM-R ahead by 17 to 31 points in every direction.
- Languages it did not see: our chain ahead for ckb (71.1 against 25.7; above XLM-R trained on
  Kurdish itself, 50.1) and yo (45.1 against 32.5); behind for qu (36.3 against 50.8).
- A different resource class (a 550M-parameter pretrained model against static vectors and
  dictionaries): reference rows, not a head-to-head baseline. 9 directions, one test set, no
  intervals for the published numbers.
