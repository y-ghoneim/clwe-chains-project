# Chains on well-resourced pairs with natural dictionaries (phase 6, stage D)

**Question.** Phase 4 found no chain beating direct on 40 distant pairs among the 30 graph
languages, with every edge trained on the same 706 shared-list concepts. Stage 1 found chains
winning where direct is naturally scarce, but every map there was weak (MRR < 0.08). With each edge
trained on its **own natural dictionary**, do chains help on phase 4's well-resourced pairs, where
the vectors are good and maps work? This separates "chains help because of natural, unequal
supervision" from "chains help only when everything is weak".

decisions.md 2026-11-17; same design, code and evaluation as `experiments/scarce-natural-s1/`, run
by stage 1's runner (`run.py` here only points it at this stage's data and methods). This README is
written and approved before any run. Results are descriptive.

## Data (`prep/07` and `prep/08 --stage d`, `v1/phase6/d/`)
- **Pairs:** phase 4's 40 pairs (`../graph-all-pairs/pairs.csv`: stratum (a) high-resource and
  stratum (b) with a low-resource node such as mt, ga, sw, eu, tl, ur, et, kk, ka, ta). Hub = the
  node maximizing min(|s–x|, |x–t|), never an endpoint: en for 33 pairs, ru 2, fr 2, zh, ja, ar 1
  each (en→zh goes through ja, en→ar through fr).
- **Dictionaries** (distinct pairs, before test removal): direct median 11,939 (2,339–289,452);
  s–hub median 91,237; hub–t median 81,908. Direct is larger than the chain's narrowest step in 4 of
  40 pairs. Edges with en also get English Wiktionary, as in stage 1; gold for an English
  endpoint uses English Wiktionary in both directions (phase 4 used it only for an English
  source), a difference to keep in mind when comparing with phase 4.
- **Test words:** the shared list's 841 concepts (capitalized senses dropped) that both nodes have,
  minus identical strings: 823–841 per pair. 5 disjoint folds; test words removed from every edge
  as in stage 1.

## Methods (trimmed; decisions.md 2026-11-17)
Thin direct (here not thin: the full natural direct dictionary), the chain through the hub and the
combination, in one CSLS pass. Joined direct is left out: for pairs like de–ja through en the
joined dictionary would run to tens of millions of pairs.

## Evaluation
As stage 1: CSLS (k = 10) over the target's full vocabulary; gold = list word + PanLex (+ English
Wiktionary when en is an end) with casing variants; per pair-direction ΔMRR (chain − direct) with
a paired bootstrap interval. **Summary, computed from `pooled.csv`** (the runner's
`languages.csv` / `overall.csv` group by phase 4's source language, whose orientation was a
coin flip, so they aren't used): per pair, the mean of its two directions; overall and per
stratum (from `../graph-all-pairs/pairs.csv`), the mean over pairs and the count of
pair-directions above / below 0. Compared descriptively with phase 4's numbers (different
dictionaries and test words, so not directly).

## Expected outcomes (written before any run)
- **Direct beats the chain for most pairs.** Direct dictionaries here are large (median about
  12,000 pairs), and the chain adds a second step's error. This is phase 4's null with natural
  dictionaries.
- **The chain is closest, or wins, where direct is smallest relative to the hub's edges**:
  stratum (b) pairs (direct median 6,443, down to 2,339, against stratum (a)'s 35,516), consistent
  with stage 1.
- **Absolute MRR is far higher than stage 1's** (phase 4's direct maps reached about 0.38).
- **The combination is close to direct**, above it for a few pairs.
- If the chain beats direct on most pairs here, the chain's advantage is not specific to scarcity,
  and phase 4's null came from its shared-list control.

## Compute
About 800 CSLS evaluations (80 pair-directions × 5 folds × 2) ≈ 15 min at 1.1 s. The data comes with
stage A's (`remote_files_ad.txt`).

## Results (run on the remote at 1741f92, 2026-11-18; `results/`)
Summary from `pooled.csv` as specified: per pair, the mean of its two directions; per stratum, the
mean over pairs and the pair-directions whose interval is above / below 0.

| | pairs | direct MRR | chain MRR | chain Δ | above / below | pairs with Δ > 0 | combination Δ | above / below |
|---|---|---|---|---|---|---|---|---|
| stratum (a) | 20 | 0.698 | 0.680 | −0.018 | 0 / 19 | 1 | +0.018 | 27 / 0 |
| stratum (b) | 20 | 0.338 | 0.345 | +0.007 | 9 / 3 | 10 | +0.056 | 40 / 0 |
| all | 40 | 0.518 | 0.513 | −0.005 | 9 / 22 | 11 | +0.037 | 67 / 0 |

Largest chain gains (all stratum (b)): kk–tl +0.058, ga–he +0.044, ga–ta +0.040, sw–ur +0.033.
Largest losses (all stratum (a)): fi–zh −0.038, fi–ko −0.036, ar–en −0.034, he–tr −0.033. Over the
40 pairs the chain's gain correlates with log10(direct pairs) at −0.64.

**Against the expected outcomes:**
- Direct beats the chain for most pairs, yes: 29 of 40 pairs below 0, including every stratum (a)
  pair but one (el–he, +0.002).
- The chain is closest, or wins, where direct is smallest, yes: stratum (b) is even on average
  (+0.007, 10 of 20 pairs above 0).
- Absolute MRR far higher than stage 1's, yes (direct 0.70 in (a), 0.34 in (b); stage 1 < 0.08).
- The combination close to direct, above it for a few pairs: **no, it is above direct for all 40
  pairs** (67 of 80 pair-directions above, none below): +0.018 in (a), +0.056 in (b).
- The chain does not beat direct on most pairs, so its advantage in stages 1–A2 is specific to
  scarce direct dictionaries; combining it with direct helps here too.

*Updated 2026-11-19: English list words fixed in prep (decisions.md 2026-11-19); English-source directions recomputed; numbers above are after the fix (`experiments/natural-highres-d/gain-curve/`).*
