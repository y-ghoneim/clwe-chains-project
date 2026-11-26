# Chains when direct is less scarce still (phase 6, stage A2)

**Question.** One more step up from stage A: where a low-resource language's direct dictionary with
a distant node has 1,000–3,000 words, does the chain still beat thin direct? Stage 1 (< 500), A
(500–1,000) and A2 (1,000–3,000) together trace the chain's gain against direct size for the same
kind of pair.

decisions.md 2026-11-17; same design, code and evaluation as `experiments/scarce-natural-s1/` and
`experiments/scarce-moderate-a/` (stage 1's runner; `run.py` here only points it at this stage's
data and methods). This README is written and approved before any run. Results are descriptive.

## Data (`prep/07` and `prep/08 --stage a2`, `v1/phase6/a2/`)
- **Pairs:** as stage A (the ten low-resource languages × the 30 nodes, different top-level family,
  hub = the node maximizing min(|s–x|, |x–t|)), with direct PanLex size (min distinct words) in
  [1,000, 3,000). Few low-resource pairs have dictionaries this large: **43 pairs**, 86
  pair-directions, mostly Javanese and Quechua:

  | language | pairs | hub | direct pairs, median (range) | s–hub pairs | hub–t pairs, median | eligible concepts |
  |---|---|---|---|---|---|---|
  | jv Javanese | 19 | en / fr | 1,508 (1,225–3,661) | 3,661 | 127,742 | 461 |
  | qu Quechua | 11 | es | 1,574 (1,269–5,571) | 17,783 | 70,241 | 650 |
  | nah Nahuatl | 4 | es | 1,679 (1,493–2,376) | 6,971 | 157,052 | 369 |
  | ckb Sorani | 3 | en | 1,706 (1,380–2,140) | 8,184 | 267,078 | 658 |
  | mhr Meadow Mari | 2 | en | 1,854 (1,640–2,068) | 29,461 | 244,272 | 705 |
  | myv Erzya | 2 | en / ru | 3,060 (1,362–4,758) | 3,060 | 221,465 | 538 |
  | or Odia | 1 | hi | 2,100 | 3,125 | 5,237 | 328 |
  | yo Yoruba | 1 | en | 1,652 | 7,436 | 235,379 | 511 |

  Dictionary sizes are distinct word pairs before test removal. In several pairs the direct
  dictionary is as large as, or larger than, the chain's first step (Javanese: s–en 3,661; Erzya's
  pair with direct 4,758).
- Natural dictionaries, test words, 5 disjoint folds, test removal: exactly as stage 1.

## Methods
As stage A: thin direct, the chain through the hub, the combination, joined direct (mutual best).

## Evaluation
As stage A (per pair-direction ΔMRR with a paired bootstrap interval; per language and overall
means; counts above / below 0). Read together with stage 1 and A in the combined analysis
(`analyses/2026-11-17-phase6-gain-curve/`, specified before any of these results).

## Expected outcomes (written before any run)
- **The chain's gain is small on average (about +0.005 to +0.01 MRR) and no longer consistent:**
  many pair-directions are indistinguishable from 0 and some below it.
- **It stays positive where the first step is still much larger than direct** (Quechua: s–es
  17,783 against ~1,600 direct) **and turns negative where it isn't** (Javanese, whose s–en
  dictionary is only about twice its direct ones; Erzya's 4,758-pair direct pair).
- **Joined direct stays below the chain** where the chain gains, and near thin direct elsewhere.

## Compute
About 1,300 CSLS evaluations (86 pair-directions × 5 folds × 3) ≈ 25 min at 1.1 s. No new vectors;
the data comes in `remote_files_a2b.txt` with stage B's.

## Results (run on the remote at 1741f92, 2026-11-18; `results/`)

| language | pair-dirs | thin direct MRR | chain Δ | above / below | combination Δ | above / below | joined (mb) Δ | above / below |
|---|---|---|---|---|---|---|---|---|
| or | 2 | 0.041 | +0.073 | 2 / 0 | +0.079 | 2 / 0 | +0.030 | 2 / 0 |
| jv | 38 | 0.108 | +0.036 | 28 / 2 | +0.078 | 38 / 0 | +0.012 | 8 / 1 |
| ckb | 6 | 0.061 | +0.034 | 6 / 0 | +0.058 | 6 / 0 | +0.026 | 4 / 0 |
| nah | 8 | 0.042 | +0.030 | 5 / 0 | +0.027 | 8 / 0 | +0.009 | 2 / 0 |
| mhr | 4 | 0.058 | +0.023 | 3 / 0 | +0.037 | 4 / 0 | +0.021 | 2 / 0 |
| qu | 22 | 0.033 | +0.011 | 9 / 0 | +0.018 | 20 / 0 | +0.006 | 6 / 0 |
| yo | 2 | 0.050 | +0.008 | 1 / 0 | +0.013 | 2 / 0 | −0.000 | 0 / 0 |
| myv | 4 | 0.058 | −0.002 | 2 / 1 | +0.013 | 2 / 0 | −0.004 | 0 / 1 |
| **overall** (mean of languages) | 86 | | **+0.026** | 56 / 3 | **+0.040** | 82 / 0 | **+0.012** | 24 / 2 |

The three pair-directions where the chain is below thin direct are the low-resource language's
pair **with English** (jv→en −0.126, en→jv −0.046, myv→en −0.044): direct there is the language's
English dictionary (3,661 and 4,758 pairs, with Wiktionary), larger than any route's narrowest step.

**Against the expected outcomes:**
- A small (+0.005 to +0.01), inconsistent gain, **no**: +0.026, above 0 in 56 of 86
  pair-directions and below in only 3.
- Positive where the first step is much larger than direct (Quechua), yes, but small (+0.011).
  Negative for Javanese in general, **no** (+0.036, 28 / 2): only its pair with English is
  negative, as is Erzya's 4,758-pair pair (also with English).
- Joined direct below the chain, yes (+0.012).
- The combination is above thin direct for every language and in 82 of 86 pair-directions, never
  below; it adds most for Javanese (+0.078), where direct is strongest.

*Updated 2026-11-19: English list words fixed in prep (decisions.md 2026-11-19); English-source directions recomputed; numbers above are after the fix (`experiments/natural-highres-d/gain-curve/`).*
