# Chains when direct is moderately scarce (phase 6, stage A)

**Question.** Stage 1's chain beat thin direct in every language where the direct dictionary had
under 500 words. Does that hold, shrink or vanish one step up, where direct has 500–1,000 words?
Together with stage 1 this gives the chain's gain against how scarce direct is, with no edge
shrunk by design.

decisions.md 2026-11-17; same design, code and evaluation as `experiments/scarce-natural-s1/`
(its README and spec), run by stage 1's runner (`run.py` here only points it at this stage's data and
methods). This README is written and approved before any run. Results are descriptive (no success
rule; user, 2026-11-17).

## Data (`prep/07` and `prep/08 --stage a`, `v1/phase6/a/`)
- **Pairs:** a low-resource language paired with one of the 30 nodes, different top-level family,
  direct PanLex size (min distinct words) in [500, 1,000); hub = the node maximizing min(|s–x|,
  |x–t|). Low-resource languages: stage 1's six plus Nahuatl, Sorani Kurdish, Quechua and Javanese
  (the ten from 2026-11-16's count). **125 pairs**, both directions (250 pair-directions):

  | language | pairs | hub | direct pairs, median (range) | s–hub pairs | hub–t pairs, median | mutual-best joined pairs added to direct, median | eligible concepts |
  |---|---|---|---|---|---|---|---|
  | nah Nahuatl | 25 | es | 1,075 (703–2,067) | 6,971 | 30,128 | 7,478 | 369 |
  | myv Erzya | 21 | en | 890 (662–1,379) | 4,758 | 91,237 | 10,635 | 591 |
  | ckb Sorani | 15 | en | 1,088 (830–1,283) | 8,184 | 53,267 | 13,922 | 658 |
  | qu Quechua | 15 | en / es | 1,067 (725–1,505) | 17,783 | 21,817 | 7,835 | 650 |
  | mg Malagasy | 11 | fr | 646 (603–1,233) | 4,713 | 135,065 | 18,765 | 278 |
  | mhr Meadow Mari | 11 | en | 721 (629–745) | 29,461 | 127,742 | 20,684 | 762 |
  | as Assamese | 11 | en | 766 (642–1,152) | 4,912 | 91,237 | 27,287 | 660 |
  | jv Javanese | 9 | en | 1,099 (818–1,412) | 3,661 | 38,160 | 9,380 | 463 |
  | yo Yoruba | 7 | en | 717 (626–907) | 7,436 | 158,466 | 14,888 | 511 |

  Odia has no pair in this band (its direct dictionaries with distant nodes are under 500, except
  or–ta at 1,158, above the band). Dictionary sizes are
  distinct word pairs before test removal.
- Natural dictionaries, test words, 5 disjoint folds, test removal: exactly as stage 1.

## Methods (trimmed; decisions.md 2026-11-17)
Thin direct, the chain through the pair's hub, the combination (both in one CSLS pass), and joined
direct with the mutual-best filter. Chain-as-teacher, unfiltered joined and the en chain are left
out: stage 1 found them adding nothing over the chain, or matching it.

## Evaluation
As stage 1: CSLS (k = 10) over the target's full vocabulary; gold with casing variants; per
pair-direction ΔMRR against thin direct with a paired bootstrap interval; per language, the mean
over its pairs (each pair weighted equally) and the count of pair-directions above / below 0;
overall, the mean of the language means (`overall.csv`, as stage 1). Reported next to stage 1's
numbers, as gain against direct size.

## Expected outcomes (written before any run)
- **The chain still beats thin direct in most pair-directions, with a smaller gain than stage 1.**
  About +0.02 MRR on average instead of +0.035, because direct has about twice as many pairs while
  the chain's weak step is the same edge.
- **The gain shrinks as direct grows within the band**, and is smallest for pairs with the largest
  direct dictionaries (Nahuatl's up to 2,067 pairs).
- **For the four new languages, absolute MRR stays low** (the ceiling check's pattern), unless
  Quechua's large s–hub dictionary (17,783) lifts it.
- **Joined direct is below the chain again,** as in stage 1.
- If the chain does not beat thin direct here, its advantage is confined to the severely scarce
  regime of stage 1.

## Compute
About 3,750 CSLS evaluations (250 pair-directions × 5 folds × 3: thin + chain + combination ≈ 2,
joined 1) ≈ 70 min at the remote's 1.1 s. The remote needs four more vectors (ckb, jv, nah, qu) and
this stage's data (`remote_files_ad.txt` with stage D, 0.95 GB).

## Results (run on the remote at 1741f92, 2026-11-18; `results/`)
Per language: mean ΔMRR against thin direct over its pairs, and pair-directions whose bootstrap
interval is above / below 0.

| language | pair-dirs | thin direct MRR | chain Δ | above / below | combination Δ | above / below | joined (mb) Δ | above / below |
|---|---|---|---|---|---|---|---|---|
| mg | 22 | 0.048 | +0.068 | 21 / 0 | +0.072 | 22 / 0 | +0.051 | 20 / 0 |
| as | 22 | 0.027 | +0.058 | 22 / 0 | +0.062 | 22 / 0 | +0.041 | 21 / 0 |
| mhr | 22 | 0.026 | +0.050 | 22 / 0 | +0.042 | 22 / 0 | +0.044 | 22 / 0 |
| ckb | 30 | 0.042 | +0.048 | 30 / 0 | +0.058 | 30 / 0 | +0.041 | 30 / 0 |
| jv | 18 | 0.060 | +0.037 | 17 / 0 | +0.067 | 18 / 0 | +0.028 | 15 / 0 |
| yo | 14 | 0.026 | +0.019 | 12 / 0 | +0.021 | 11 / 0 | +0.003 | 2 / 0 |
| nah | 50 | 0.022 | +0.014 | 20 / 0 | +0.014 | 37 / 0 | +0.008 | 11 / 0 |
| myv | 42 | 0.026 | +0.013 | 22 / 0 | +0.017 | 41 / 0 | +0.007 | 9 / 0 |
| qu | 30 | 0.018 | +0.012 | 17 / 0 | +0.016 | 30 / 0 | +0.007 | 13 / 0 |
| **overall** (mean of languages) | 250 | | **+0.035** | 183 / 0 | **+0.041** | 233 / 0 | **+0.026** | 143 / 0 |

**Against the expected outcomes:**
- The chain beats thin direct in most pair-directions, yes (183 of 250 above, none below), but
  **not with a smaller gain**: +0.035, the same as stage 1's +0.035. That prediction was wrong.
- The gain shrinks as direct grows within the band, roughly: over pairs, the gain correlates with
  log10(direct pairs) at −0.40; by direct size in distinct pairs (post hoc bins) ≤ 750: +0.051,
  750–1,000: +0.024, 1,000–1,500: +0.026.
- Absolute MRR stays low for the four new languages (chain 0.030–0.097); Quechua's large s–hub
  dictionary did not lift it (chain 0.030, gain +0.012, the smallest).
- Joined direct is below the chain again, yes, for every language.
- The combination is the best method for 7 of 9 languages (tied with the chain for Nahuatl,
  below it for Meadow Mari) and never below thin direct.

*Updated 2026-11-19: English list words fixed in prep (decisions.md 2026-11-19); English-source directions recomputed; numbers above are after the fix (`experiments/natural-highres-d/gain-curve/`).*
