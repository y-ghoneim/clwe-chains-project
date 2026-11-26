# Cross-lingual named-entity recognition through direct and chained maps (downstream, part 3: WikiANN)

**Status: DROPPED 2026-11-20, not run** (about 50 h on the remote at the approved scale; replaced by MasakhaNEWS and SIB-200, decisions.md 2026-11-20). Kept as the design record. Results will be
descriptive (decisions.md 2026-11-16: no success rule); gains in only some cases count as a good
result (user, 2026-11-19).

**Question.** Parts 1 and 2 used sentiment, where a tweet is an average of word vectors, so
single-word errors of a map wash out. Named-entity recognition (NER) labels every token from its
own vector and its neighbours', so it should be more sensitive to map quality. It also covers the
project's own pairs, which already have BLI results: stage D's 40 well-resourced distant pairs
and the low-resource pairs of stages 1, A and A2 (decisions.md 2026-11-19, downstream scope:
"WikiANN NER: covers the project's own pairs"). Does the chain help NER where it helped BLI (small
direct dictionaries), and is it level elsewhere?

**Framing (as parts 1 and 2).** A tagger trained on a language Y's labelled sentences in Y's space
is applied to a language X's test sentences, whose vectors are moved into Y's space by the direct
map or the chain through the pair's hub.

## Data
- **WikiANN** (Pan et al. 2017) with the balanced splits of Rahimi et al. (2019), Hugging Face
  `unimelb-nlp/wikiann` (parquet files; downloaded into their own empty folder, read with `-I`).
  Tags: O, B-/I-PER, B-/I-ORG, B-/I-LOC. Sentences come pre-tokenized.
- **Split sizes (from the dataset server, 2026-11-20):** most nodes 20,000 / 10,000 / 10,000
  (train / validation / test); et 15,000, ka and eu 10,000 train; hi, tl, ta, ur 1,000 test;
  ga, kk, sw 1,000 / 1,000 / 1,000; mt 100 / 100 / 100. Low-resource languages: yo, or, mg, mhr,
  as, qu, jv 100 / 100 / 100; ckb 1,000 / 1,000 / 1,000; **no WikiANN for myv and nah**.
- **Left out: zh, ja, th** (proposed). Their WikiANN tokens are single characters ("香", "港"),
  which do not match the word vectors. This removes 9 of stage D's 40 pairs (en–zh, en–ja,
  zh–fi, ja–hu, et–th, th–eu, tl–zh, zh–ta, kk–ja) and the low-resource pairs with these nodes.
- **Training side needs at least 1,000 training sentences** (proposed), so mt and the
  low-resource languages other than ckb are only test languages.

## Pairs (fixed now, from the stages' pairs.csv; hubs and dictionaries as in those stages)
- **Well-resourced set:** stage D's 31 remaining pairs, both directions where the target has
  enough training data (mt–hi and mt–sw: only mt → hi, mt → sw). About 60 directions; test sets
  1,000 or 10,000 sentences (mt 100).
- **Low-resource set (the headline for RQ2):** every pair of stages 1, A and A2 whose low-resource
  language has WikiANN (yo, or, mg, mhr, as, ckb, qu, jv) and whose node is not zh, ja or th,
  direction low-resource → node only. 165 directions (as 15, ckb 15, jv 25, mg 24, mhr 23, or 15,
  qu 23, yo 25); each low-resource language's
  directions share one test set of 100 sentences (ckb 1,000).
- BLI scores for every direction come from the stages' existing results (stage D, 1, A, A2
  `pooled.csv`); no new BLI run.

## Method
- **Word vectors:** the project's normalized fastText vectors, each centred on the language's mean
  over its own training tokens, then mapped. **Matching (proposed): the token's exact surface
  form first** (fastText is cased, and case matters for names), **then the normalized form** used
  everywhere else in the project; no match gives a zero vector.
- **Maps:** direct and chain through the pair's hub, Procrustes on the stage's full natural
  dictionaries (no test-word removal; NER test data is different text). No map as a reference.
- **Three taggers, trained once per labelled language Y on Y's full training split** (user,
  2026-11-20: the standard model, no cap; runs on the remote PC, docs/remote.md). Model selection
  only on Y's validation split, never on the tested language.
  1. **Headline: BiLSTM-CRF on the word vectors only.** The vectors are frozen inputs (fine-tuning
     would move Y's vectors away from the space the maps point into). 1 BiLSTM layer, 256 hidden
     units per direction, dropout 0.5, a CRF output layer (`pytorch-crf`, user 2026-11-20), Adam
     (learning rate 0.001), batch 32, at most 30 epochs, early stopping on Y's validation entity F1
     with patience 5, 3 seeds. Settings fixed now, close to the usual BiLSTM-CRF NER setup
     (Lample et al. 2016), not tuned.
  2. **Secondary A: the same BiLSTM-CRF plus character features** (a character BiLSTM, 30-d
     character embeddings, 50 hidden units per direction, concatenated with the word vector;
     characters unseen in Y's training data share one unknown symbol). The standard,
     better-performing system; spelling cues do not go through the maps, so they help direct and
     chain alike within a script and not across scripts. It asks whether the chain's advantage
     survives in a realistic system.
  3. **Secondary B: window logistic regression** (brief §5.8, as parts 1–2): each token tagged on
     its own from its vector and its two neighbours' (900 dimensions, zero at sentence edges), 7
     tags, C from {0.001 … 100} on Y's validation entity F1.
  Which of the secondary taggers go into the report is decided after the run (user, 2026-11-20);
  the headline is fixed now.
- **Metric:** entity-level F1, micro-averaged over entities, exact span and type (the CoNLL rule:
  an I- tag after O or a different type starts a new entity; the window tagger's output is read
  the same way), written and tested in the library. Floor: tagging everything O gives 0. Ceiling:
  X's own tagger where X has at least 1,000 training sentences; for the 100-sentence languages
  their own tagger is reported but is a weak ceiling.
- **Evaluation:** chain − direct per direction and tagger with a paired bootstrap over test
  sentences (1,000 resamples; the BiLSTM's F1 averaged over its seeds within each resample).
  Summaries per set; in the low-resource set also per low-resource language (its directions
  share a test set and the dictionaries on its side). Spearman of the chain's advantage against r,
  and the BLI link (F1 against MRR; advantage against advantage), per set, with intervals
  resampling pairs.
- **Precision:** 100 test sentences hold about 150–250 entities, so one direction's F1 is uncertain
  by several points and few single low-resource directions will resolve; more training data
  does not change that (the noise is on the test side). The per-language means over 15–25
  directions are the readable numbers. 10,000-sentence test sets resolve differences of under a
  point.

## Expected outcomes (written before any run; guesses, to be marked wrong where wrong)
1. **Ceilings** (headline BiLSTM-CRF, word vectors only): 0.55–0.75 entity F1 for the
   well-resourced languages; secondary A a few points higher, the window tagger lower (0.40–0.65).
   **Transfer far below**: 0.15–0.40 in the well-resourced set, lower in the low-resource set; no
   map near 0.
2. **Well-resourced set: chain level with direct or slightly below**, as stage D's BLI (−0.005
   MRR); most directions unresolved or slightly negative.
3. **Low-resource set: chain above direct on average**, in most of the 8 low-resource languages'
   means, as in BLI (stages 1, A, A2) and AfriSenti's Yoruba; single directions mostly unresolved.
4. **Advantage rises with r** (positive Spearman) in the low-resource set.
5. **BLI link stronger than for sentiment:** F1 against MRR positive in both sets (ρ about
   0.3–0.6), since NER uses single-word vectors.
6. **Character features shrink the chain's advantage** (secondary A against the headline), most
   in same-script pairs.

## Data findings while building (2026-11-20, before any transfer number was computed)
- **Source:** Hugging Face `unimelb-nlp/wikiann`, revision f0a3be6, 35 languages
  (`raw/downstream/wikiann/SOURCE.txt`); the dataset card gives the licence as "unknown". Split
  sizes as listed under Data.
- **Directions after the rules (`run.py --data-files` / `directions.csv`):** low-resource set 157
  (as 14, ckb 14, jv 24, mg 23, mhr 22, or 14, qu 22, yo 24; the 8 pairs into mt drop out because
  mt has 100 training sentences); well-resourced set 60 directions over 31 pairs. 26 labelled
  languages; 35 languages in all.
- **Token matching (`v1/downstream/wikiann/stats.csv`, test splits):** 76–96% of tokens match the
  vocabulary, almost all by their exact form (normalized-form matches 0–3%). The low-resource
  test sets are short: about 110 entities in 100 sentences of about 5 tokens.
- **Timing (laptop, English, 20,000 sentences):** one BiLSTM-CRF epoch 27 s plus 3 s validation;
  one window logistic regression fit 27 s. The full run is estimated at 8–12 hours on the remote.

## Build (estimates are the assistant's)
1. Download WikiANN for the ~38 languages; `prep/15_wikiann.py`: tokens, tags and vocabulary rows
   (exact, normalized, none) per language and split, with match rates in a stats file reported
   before any transfer number (about 1.5 h).
2. Library: entity extraction and entity F1, the BiLSTM-CRF (with and without characters) and the
   window features, with tests, including a brute-force check of the CRF on tiny sentences
   (about 3 h). `pytorch-crf` installed as CLAUDE.md says and added to `requirements.lock.txt`.
3. Runner `experiments/downstream-wikiann/run.py` on the shared transfer helpers: taggers, maps
   per stage, both sets, tables as parts 1–2, `--data-files` for the remote copy, a cache so an
   interrupted run resumes (about 2 h); smoke run on the laptop; reviewer pass.
4. Run on the remote PC (docs/remote.md; needs a push to GitHub, with the user's approval): about
   30 labelled languages × (2 BiLSTM-CRFs × 3 seeds + the window tagger). Estimate a few hours on
   the RTX 4500.

## Questions settled with the approval (user, 2026-11-20: "all fine", as proposed)
1. Both sets, with the low-resource set as the headline? (Proposed: yes.)
2. Leave out zh, ja, th? (Proposed: yes. Keeping them would mean re-segmenting their character
   tokens into words the way fastText's training text was segmented, with per-language tools,
   and moving the character tags onto words; not worth it for 9 of 40 well-resourced pairs.)
3. Training side at least 1,000 sentences, so mt and the 100-sentence languages are test-only?
   (Proposed: yes.)
4. Exact surface form first, then the normalized form? (Proposed: yes.)

## Remote run (docs/remote.md, with this experiment's names)
After the one-time setup there, plus `python -m pip install pytorch-crf==0.7.2`:
```
$src = "\\tsclient\C\data\language-graph"; $dst = "C:\data\language-graph"
Get-Content "$src\remote_files_wikiann.txt" | ForEach-Object {
    $to = Join-Path $dst $_
    New-Item -ItemType Directory -Force (Split-Path $to) | Out-Null
    if (-not (Test-Path $to) -or (Get-Item $to).Length -ne (Get-Item (Join-Path $src $_)).Length) {
        Copy-Item (Join-Path $src $_) $to -Force
    }
}
conda activate clwe; cd C:\clwe-chains; git fetch; git checkout <commit>
$env:LG_DATA_ROOT = "C:\data\language-graph"; $env:PYTHONPATH = "C:\clwe-chains\src"
$env:PYTHONUTF8 = "1"
python experiments\downstream-wikiann\run.py --limit 1     # smoke: minutes, own cache folder
python experiments\downstream-wikiann\run.py               # the full run; rerun to resume
```
Bringing results back (the counts let the bootstraps be redone later; about 200 MB):
```
$out = "\\tsclient\C\data\language-graph\remote-out\downstream-wikiann"
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item -Recurse -Force experiments\downstream-wikiann\results $out
Copy-Item -Force C:\data\language-graph\v1\results\downstream-wikiann\run.log $out
Copy-Item -Recurse -Force C:\data\language-graph\v1\results\downstream-wikiann\counts $out
```
