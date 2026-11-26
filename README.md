# To Chain or Not to Chain? When Pivot Languages Help Align Word Embeddings

Yahia Abdelghafur, Youssef Ghoneim

Code and results for the MLR570 project. The report is
[MLR570-Language-Project-Submission.pdf](MLR570-Language-Project-Submission.pdf).

The project maps word embeddings from one language into another, either directly or through a
chain of pivot languages (source → pivot → target), and asks when the chain helps. Maps are
orthogonal Procrustes maps on fastText vectors, trained on bilingual dictionaries and scored by
word translation (MRR, P@1) and by cross-lingual text classification.

## Layout

| folder | holds |
|---|---|
| `src/clwe_chains/` | the library: embeddings, alignment, chains and route search, dictionaries, evaluation, statistics, downstream classifiers |
| `tests/` | unit tests for the library and the experiment runners |
| `prep/NN_*.py` | data-building steps, numbered in run order; outputs go under the data root |
| `experiments/<name>/` | one experiment each: `README.md` (question, design, expected outcomes, then results), `config.yaml`, `run.py`, `results/` |
| `figures/<name>/` | the report's figures: `make_figure.py` (reads committed results), README, PDF/PNG |
| `configs/` | `default.yaml` (all research settings), `languages.yaml`, `local.example.yaml` |

## Setup

Python 3.12 and a CUDA GPU (tested on 6 GB; set `eval.block_rows` lower on less memory, or
`device: cpu`).

```sh
uv sync                         # or: pip install -r requirements.lock.txt
cp configs/local.example.yaml configs/local.yaml   # set data_root and device for your machine
python -m pytest -q             # unit tests; no data needed
```

`data_root` can also be set with the `LG_DATA_ROOT` environment variable.

## Data

Large data is not in the repository. Everything lives under `data_root`:

- `processed/` (inputs, read-only): `embeddings/<lang>.npy` and `<lang>.vocab.txt` (fastText
  vectors, top 200k words), `panlex_invocab/<lang>.parquet` (PanLex meanings for in-vocabulary
  words, from the Hugging Face dump `cointegrated/panlex-meanings`, 2024-03-01),
  `languages_all.csv`, `distances/measures.parquet` and `coverage_pairs.parquet`. These were built
  by an earlier stage of the project whose scripts are not part of this repository.
- `raw/wiktionary/raw-wiktextract-data.jsonl.gz`: the wiktextract English dump (2026-10-30).
- `raw/downstream/<dataset>/`: UMSAB, AfriSenti, MasakhaNEWS and SIB-200, downloaded as described
  in each downstream experiment's README.
- `v1/`: everything the prep steps and experiments write.

The machine-made dictionaries use NLLB-200 (3.3B, from Hugging Face) and Claude Haiku 5.5 through
the Anthropic Message Batches API (`prep/12_llm_translate.py`; needs `ANTHROPIC_API_KEY`).

## Running

1. `python prep/01_wiktionary.py` through `prep/20_break_even_grid.py`, in order. Each step's
   docstring says what it reads, what it writes, and which earlier step it needs; a few take
   `--stage` or `--set` arguments documented there.
2. `python experiments/<name>/run.py` for each experiment (`--help` lists its options; most
   READMEs give the command used and the run time).
3. `python figures/<name>/make_figure.py` to redraw a figure from the committed results.

Prep and runs are deterministic (fixed seeds in `configs/default.yaml`). Each run writes
`run_info.json` in its `results/` folder with the commit and config it was produced with (WikiANN
never ran).

## Report sections and experiments

| report section | experiment folders |
|---|---|
| Chains under equal supervision | `aus-replication` (replication of Lifiya et al.), `graph-all-pairs` (choosing the pivot on 40 distant pairs) |
| When chains help: natural dictionaries | `scarce-natural-s1` (under 500 words), `scarce-moderate-a` (500–1,000), `scarce-moderate-a2` (1,000–3,000), `natural-highres-d` (the 40 distant pairs; `gain-curve/` fits the gain against dictionary size), `close-pairs-b` (related pairs), `low-low` (both ends low-resource), `route-choice-s1` (pivot choice under 500) |
| Reduced dictionaries and the break-even point | `controlled-scarcity`, `break-even-grid` |
| Several hubs | `several-hubs-procrustes` |
| Comparison with other methods | `strong-maps` (self-learning), `contrastivebli-baseline`, `panlex-bli` (the PanLex-BLI benchmark; `reproduction/` checks our scoring against the published one) |
| Machine-made dictionaries and translators | `machine-dicts`, `mt-pivot` |
| Cross-lingual text classification | `downstream-umsab`, `downstream-afrisenti`, `downstream-sib200` (`published-comparison/` sets it next to the published XLM-R transfer), `downstream-masakhanews`, `downstream-tier2` (reduced direct dictionaries); the `*-bli` folders score the same pairs by word translation |

Not in the report, kept as the record of what was run: `ensemble-control` (is the combination's
gain just averaging two maps?), `teacher-highres-d` (the chain as a teacher for a direct map),
`muse-anchor` (our direct map on the MUSE benchmark), and `downstream-wikiann` (named-entity
recognition; designed but dropped before it ran, which the report only mentions).

Figures used in the report: `route-example`, `direct-size-sweep`, `break-even-ratio`.
`language-graph` and `break-even-grid` are earlier figures from the same results.

## A note on references

The experiment READMEs were written during the project and are kept as written. Some mention
`decisions.md`, `docs/…` or `analyses/…`: these are the project's internal working records (the
dated decision log, design notes and one-off checks), which are not part of this submission. The
commit hashes in `run_info.json` and in the READMEs refer to the original working repository.
