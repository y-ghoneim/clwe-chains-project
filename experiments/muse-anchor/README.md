# External anchor: our direct alignment on the MUSE benchmark (review B)

**Status: APPROVED by the user, 2026-11-21 (the four pairs as proposed). Run 2026-11-21 on the
remote with the ensemble control (the laptop's GPU stays free); results at the end.** A sanity
check, not a research
question: it changes nothing in the experiments. Raised by the 2026-11-19 review (section B, "no
external anchor (MUSE dropped)"); the user asked for it on 2026-11-21. Brief §3 lists MUSE test
sets under "not carried over" from v0 as an evaluation; this uses them only to place our numbers
next to the literature's.

**Question.** Does our pipeline's direct map (our vectors, normalization, Procrustes and CSLS)
score what the literature reports on the standard benchmark? If it does, our absolute MRR / P@1
numbers on PanLex can be read with some confidence; if it is far off, something in the pipeline
or the vectors needs explaining.

## Design
- **Benchmark:** the MUSE ground-truth dictionaries (Conneau et al. 2018): train `0-5000` (5,000
  source words with all their translations) and test `5000-6500` (1,500 source words), from
  `https://dl.fbaipublicfiles.com/arrival/dictionaries/{src}-{tgt}.{0-5000,5000-6500}.txt`.
- **Pairs (proposed):** en–de, en–es, en–fr, en–ru, both directions (8 directions; MUSE's are
  directional files, so 16 files). Published reference (Conneau et al. 2018, Table 1, supervised
  Procrustes, P@1): CSLS en→de 73.5, de→en 72.4, en→es 81.4, es→en 82.9, en→fr 81.1, fr→en 82.4,
  en→ru 51.7, ru→en 63.7; NN retrieval 68.4, 67.7, 77.4, 77.3, 74.9, 76.1, 47.0, 58.2.
- **Ours, unchanged:** fastText Common Crawl vectors, top 200k words, unit → centre → unit;
  Procrustes on every train pair whose two words are in our vocabularies (exact form, MUSE's
  words are lower case; then the lower-cased form of our vocabulary if the exact form is
  missing); CSLS (k = 10) over the full 200k target vocabulary; a test word counts if any of its
  MUSE translations is retrieved. Reported: P@1 (as the paper), P@5, MRR, NN retrieval too, and
  coverage (train pairs and test words in our vocabulary).
- **Differences from the paper, stated in the results:** Common Crawl instead of Wikipedia
  vectors; our normalization (MUSE's default is none); test words missing from our vocabulary are
  dropped (the paper's vocabularies contain all of them).
- **Data handling:** downloaded into a new empty folder `C:/data/language-graph/external/muse/`,
  read only as plain text by `run.py`, which lives in the repo (no code runs from the download
  folder; first inspection done with Python run with `-I`); not committed or re-hosted (the MUSE
  repository's licence is CC BY-NC 4.0). Only scores are committed.
- **Code:** a reader for MUSE dictionary files in `src/clwe_chains/` (with a test) and
  `run.py` here; the map and evaluation are the library's `fit` and `bli_multi`.

## Expected outcomes (written before any run; guesses)
1. CSLS P@1 within 5 points of the published value on en–de, en–es and en–fr in both
   directions; within 8 on en–ru.
2. Common Crawl vectors slightly below Wikipedia's on this benchmark (its test words are frequent
   Wikipedia words), so most of our numbers a little under the paper's.
3. CSLS above NN by 4–7 points, as in the paper.
4. Coverage above 95% of test words.

## Findings while building (2026-11-21, before any score)
- **Coverage** (CPU check of the vocabularies, no vectors): test words with an in-vocabulary
  translation 1,499 / 1,500 (en→de, en→es, en→fr), 1,479 (de→en), 1,489 (es→en), 1,484 (fr→en),
  1,456 (en→ru), 1,452 (ru→en); train pairs kept 97.5% (fr→en) to 99.8%. Prediction 4 is about
  the scores' setup, so it is checked here: above 95% everywhere.
- **Matching, as approved:** a word's row is its exact form, or, if that is missing from our
  cased vocabulary, the most frequent row with the same normalized (lower-case) form; each gold
  translation counts at that one row (1.4–2.4 gold rows per query, as in the files). Coverage is
  the same under this rule. A first build matched every word by its normalized form and counted a
  gold translation at any casing (2.6–4.3 rows per query); the reviewer flagged that as a change
  from the approved design and more lenient than "slightly", so the approved rule is the main
  result and the any-casing gold is reported as a secondary column (`*_p1_anycase`).
- Queries with no translation in our vocabulary are dropped (1–48 per direction), probably among
  the harder words; stated with the results.

## Compute and build (estimates are the assistant's)
About 1 h to build with the test; the run takes minutes on the laptop.

## Questions settled with the approval (user, 2026-11-21)
1. The four pairs above, both directions? (Proposed: yes; en–zh is left out, as SIB-200 left out
   languages written without spaces: word segmentation would be a second source of difference.)

## Results (run 2026-11-21 on the remote, commit f3ff2bb; written up 2026-11-21)
`results/results.csv`, `results/run_info.json` (RTX 4500 Ada; every input fingerprinted).
P@1 in %, CSLS retrieval; "any casing" counts a retrieved word as correct if it is any casing of
a gold translation (secondary, more lenient than the approved rule):

| direction | ours (approved rule) | published | difference | ours, any casing | difference | CSLS − NN, ours / published |
|---|---|---|---|---|---|---|
| en→de | 62.0 | 73.5 | −11.5 | 73.0 | −0.5 | 3.7 / 5.1 |
| de→en | 63.1 | 72.4 | −9.3 | 69.8 | −2.6 | 2.4 / 4.7 |
| en→es | 75.0 | 81.4 | −6.4 | 76.9 | −4.6 | 2.5 / 4.0 |
| es→en | 73.5 | 82.9 | −9.4 | 81.3 | −1.6 | 3.6 / 5.6 |
| en→fr | 74.0 | 81.1 | −7.1 | 75.7 | −5.4 | 3.3 / 6.2 |
| fr→en | 71.3 | 82.4 | −11.1 | 77.6 | −4.8 | 1.4 / 6.3 |
| en→ru | 52.9 | 51.7 | +1.2 | 53.9 | +2.2 | 2.2 / 4.7 |
| ru→en | 60.6 | 63.7 | −3.1 | 62.7 | −1.0 | 0.4 / 5.5 |

Test words used 1,452–1,499 of 1,500; train pairs 97.5–99.8% (as found while building).

- **Under the approved rule our P@1 is 6–12 points below the paper on de, es and fr**, and close
  on ru (+1.2 and −3.1).
- **Casing accounts for much of the gap in five directions, little in two.** Our Common Crawl
  vocabulary is cased and MUSE's dictionaries are lower case. When a gold word exists in our
  vocabulary in lower case, the approved rule puts the gold at that row, so retrieving a
  capitalised form of it counts as wrong. Counting any casing adds 11.0 points into German (nouns
  are capitalised), 6.3–7.8 into English from de, es and fr, 2.1 from ru, 1.7–1.9 into Spanish and
  French and 1.0 into Russian. That closes most of the gap on en→de, de→en, es→en, fr→en and
  ru→en, but only a quarter to a third on en→es and en→fr, and leaves every direction within 5.4
  points of the paper (−5.4 to +2.2). Any casing is more lenient than the paper's matching, which
  is not checked here, so the share closed is an upper bound on what casing explains.
- **CSLS beats NN by less than in the paper:** by 0.4–3.7 points against 4.0–6.3.

**Against the expected outcomes** (written before the run):
1. **Wrong for de, es and fr under the approved rule** (−6.4 to −11.5, not within 5); **right for
   ru** (within 8). With any casing counted, within 5 points everywhere except en→fr (−5.4).
2. **Partly right.** 7 of 8 directions are below the paper, by more than "slightly" under the
   approved rule; en→ru is above.
3. **Wrong.** CSLS beats NN in every direction, but by 0.4–3.7 points, not 4–7.
4. **Right.** Coverage above 95% (checked while building).

**What it means for the project.** The PanLex evaluation counts every casing of a gold
translation (`prep/07_phase6_list.py`, gold "main" and "list" with casing variants), which is the
any-casing column here, not the approved MUSE rule. On that like-for-like footing our direct map
is within 5.4 points of the published P@1 in every direction (−5.4 to +2.2), so the absolute
levels on PanLex are not explained by a broken map or retrieval (the PanLex gold, test words and
low-resource vectors are not tested here). The approved rule stays the
main MUSE number, as approved; it shows that exact-form gold on a cased vocabulary costs up to
11 points against lower-case gold lists. Untested reasons for the remaining gap: Common Crawl
instead of Wikipedia vectors, and our normalization.
