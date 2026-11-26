# MT pivots: direct word translation against routes through other languages (pilots)

**Question.** Phase 4 found that no route through intermediate languages beat direct alignment of
embeddings on its 40 pairs. With a strong multilingual MT system instead of embedding maps, does
translating a test word through a pivot (English, the best-resourced hub, the routes phase 4's
measures chose, or the best pivot found on other words) beat translating it directly?

**Status: exploratory pilots; RQ3b below** (user, 2026-11-18: the pilots are there to avoid wasting time and
compute, not as blind tests). Plan: `docs/superpowers/plans/2026-11-17-mt-pivot.md`; decisions.md
2026-11-18.

## Design
- **System:** NLLB-200 3.3B (fp16, beams 5, 5 returned, at most 8 new tokens), on the remote.
- **Items:** stage D's test words and gold (`prep/10`; phase 4's 40 pairs, both directions, the
  shared list's concepts, gold = list word + PanLex (+ English Wiktionary for en)).
- **Routes:** each hop translates the previous hop's top output; the last hop keeps 5 candidates.
- **Scoring:** P@1 over distinct candidates. Strict = the whole output or its first word is a gold
  answer. Lenient (pilot 2's headline) = the output, cleaned of punctuation, "it's / this is" and
  leading articles, or any of its words is a gold answer.

| | pilot 1 (0eba2af) | pilot 2 (9c14949) |
|---|---|---|
| pairs | 8 of the 40 | all 40 (80 directions) |
| words per direction | 200, every one-hop route | 400: 100 selection words get every one-hop route; 300 evaluation words get the named routes |
| named routes | direct, via en, phase 4's split-0 routes | direct, via en, via stage D's hub, phase 4's split-0 routes, the selection's best one-hop route (held-out) |
| next hop's input | the raw top output | the cleaned top output |
| round trip | yes | no |
| translations, time | 223k, 27 min | 561k, 66 min (142/s) |

## Results
**Pilot 1** found that NLLB answers single words with phrases ("The fox .", "It 's a fan .", "La
volpe"), which strict scoring misses and which went into the next hop as they were. Strict P@1:
direct 0.51, via en 0.34, best one-hop route (chosen after the fact) 0.54, with Hindi, whose
outputs are bare words, the best pivot in 13 of 16 directions. Rescored leniently from the cache:
direct 0.58, via en 0.48, best one-hop route 0.57. Pilot 2 fixed both.

**Pilot 2**, lenient P@1 on the 300 evaluation words, mean over directions; "beats / loses" counts
directions above / below direct (point estimates, no intervals):

| route | directions | P@1 | − direct | beats / loses | strict P@1 |
|---|---|---|---|---|---|
| direct | 80 | 0.575 | | | 0.498 |
| held-out best one-hop route | 80 | 0.549 | −0.026 | 21 / 58 | 0.465 |
| phase 4 "hub" route | 80 | 0.518 | −0.057 | 18 / 62 | 0.455 |
| phase 4 "quality" route | 80 | 0.517 | −0.057 | 20 / 59 | 0.473 |
| via stage D's hub | 80 | 0.514 | −0.061 | 18 / 61 | 0.452 |
| via en (en not an endpoint) | 72 | 0.504 | −0.047 | 18 / 54 | 0.470 |
| phase 4 "oracle" route (its best on embeddings) | 80 | 0.503 | −0.072 | 10 / 69 | 0.451 |
| phase 4 "resource" route | 80 | 0.497 | −0.078 | 7 / 72 | 0.414 |
| phase 4 URIEL routes (range over the variants) | 80 | 0.476–0.494 | −0.08 to −0.10 | | |
| phase 4 "svg", "aus_geo" routes | 80 | 0.41–0.45 | −0.13 to −0.16 | | |

- **Direct is best.** No way of choosing a pivot beats direct on average; the best, a pivot chosen
  on 100 other words, is 0.026 below and loses in 58 of 80 directions. Both strata look the same
  (high-resource (a): direct 0.669, held-out best 0.644, via en 0.594; (b): 0.480, 0.454, 0.432).
- **On the selection words** (100 per direction), the best of the 28 one-hop routes picked after the
  fact is 0.563 against direct's 0.550, and on average 2.7 of the 28 pivots beat direct; the
  median pivot is 0.456.
- **Phase 4's measures rank as they did for embeddings:** quality and size-based routes lose least
  (about −0.06), URIEL distances more (−0.08 to −0.10), geography and SVG-based ones most.
- **Hindi is still the best pivot on average** after cleaning (selection P@1 0.542 against en
  0.481, ru 0.488) and the held-out winner in 24 of 80 directions; whether this is translation
  quality or Hindi's bare-word outputs is not separated here.

## Further runs (prepared 2026-11-18; the experiment is stashed from the report for now)
Same code and settings as pilot 2 (config.yaml `runs`). Written before running.
- **full:** pilot 2 on every test word of the 40 pairs (about 840 per direction). A seeded sample's
  first 400 rows are pilot 2's sample, so the same 100 words choose the pivot and pilot 2's
  translations come from the cache. Expected: pilot 2's numbers with narrower uncertainty (direct
  about 0.57, the held-out best pivot about 0.03 below).
- **lowres:** stage 1's pairs whose languages NLLB has (Yoruba, Odia, Malagasy, Assamese: 60
  pairs, 120 directions; Meadow Mari and Erzya are not in NLLB), all of each pair's words
  (277–660 per direction; 100 choose the pivot, the rest are scored), routes: direct, via en, via stage 1's hub, the held-out
  best pivot. Test items from `prep/10_phase6_mt_items.py --stage ""`. Expected: direct lower than
  for the 40 pairs (about 0.2–0.4), and pivots closer to direct or above it for some directions out
  of a low-resource language, where English is NLLB's strongest partner.
- Compute: full about 45 min, lowres about 2 h on the remote; both resume from the cache.

## RQ3b: NLLB against an English-centric LLM (written 2026-11-18, before running)
Back from the stash as RQ3b (user, 2026-11-18): does pivoting through English help an
English-centric translator (an LLM) more than a many-to-many one (NLLB)?
- **LLM:** Claude Haiku 5.5 through the batch API (`--system llm`; src/clwe_chains/llm.py), asked
  for up to 3 translations per word, most likely first; its own cache
  (`v1/results/mt-pivot/hops_llm.jsonl`). Between hops the top answer is cleaned as for NLLB.
- **Runs:** `llm40` (every word, in the `full` run's seeded order, so the same words as NLLB's
  full run) and `llmlow` (the `lowres` run's words: all of them). Routes: direct, via en, via the stage's
  hub; no pivot search (cost).
- **Comparison:** within each system, the gain from pivoting (via en − direct, via hub − direct,
  lenient P@1), then the two systems' gains side by side, per direction and per stratum, paired
  per word (`ranks.csv`); absolute levels are reported but are not the question. Every word is
  compared (NLLB's selection words also have the three routes), the same words for both
  systems. P@5 and MRR are compared within a system only (the LLM gives up to 3
  answers, NLLB up to 5).
- **Expected:** NLLB's gain from the English pivot is negative (pilot 2: −0.05); the LLM's is
  closer to zero or positive, most for pairs without English and with a low-resource language
  (Jiao et al. 2023 found pivot prompting through English helped ChatGPT on distant pairs).
- **Cost:** at most about 360k translations, about 9,000 requests of 40 words, about $4–7 at
  batch prices. Every batch's id is saved before polling (a rerun picks it up instead of paying
  again) and its usage logged (`hops_llm.jsonl`'s folder, `usage.jsonl`).
- **Caveats:** a small LLM; a larger one may be less English-centric. The LLM is not
  deterministic (its sampling can't be fixed); the cache fixes its answers once made.

## Results of the further runs and RQ3b (2026-11-19)
NLLB `full` and `lowres` on the remote at 4390b56 (after the English list-word fix); LLM `llm40`
and `llmlow` on the laptop at 837ee55 (Claude Haiku 5.5, $2.08; no missing words). Check: `full`
reproduces pilot 2 exactly on pilot 2's 300 evaluation words (all 1,081 non-English
direction-routes identical).

**NLLB, `full` (every word, 80 directions; evaluation words) and `lowres` (120 directions):**

| route | `full` P@1 | − direct | above direct | `lowres` P@1 | − direct | above direct |
|---|---|---|---|---|---|---|
| direct | 0.568 | | | 0.380 | | |
| held-out best one-hop | 0.540 | −0.028 | 19 / 80 | 0.381 | +0.001 | 47 / 120 |
| via en | 0.494 | −0.051 | 14 / 72 | 0.368 | −0.013 | 44 / 120 |
| via the stage's hub | 0.504 | −0.064 | 14 / 80 | 0.369 | −0.011 | 50 / 120 |

Hindi is again the most frequent held-out winner (24 of 80 in `full`, 29 of 120 in `lowres`).

**RQ3b, the gain from the English pivot (via en − direct, lenient P@1), the same words for both
systems** (all words, both splits):

| | NLLB | LLM |
|---|---|---|
| direct, 40 pairs | 0.566 | 0.784 |
| via en − direct, 40 pairs (72 directions without en) | −0.050 (14 above) | −0.005 (21 above) |
| via hub − direct, 40 pairs (14 directions, hub not en) | −0.135 (1 above) | −0.027 (1 above) |
| direct, low-resource pairs | 0.378 | 0.489 |
| via en − direct, low-resource pairs (120) | −0.011 (51 above) | +0.009 (78 above) |
| — from a low-resource language (60) | −0.019 (15 above) | +0.012 (43 above) |
| — from a graph language (60) | −0.003 (36 above) | +0.007 (35 above) |
| via hub − direct, low-resource pairs (66) | −0.023 (17 above) | −0.015 (15 above) |

- The LLM gains more from the English pivot than NLLB in 50 of 72 (40 pairs) and 83 of 120
  (low-resource) directions; mean difference +0.045 and +0.020.
- Standouts: the LLM gains most through English for pairs with Basque as target (Thai → Basque
  +0.14, Turkish → Basque +0.10) and for Malagasy → Basque, Georgian → Yoruba, Odia → Korean
  (+0.05); NLLB loses most on Finnish → Russian (−0.18), Hungarian → Georgian (−0.16), Irish →
  Turkish (−0.16), Assamese → Georgian (−0.14).
- The non-English hub hurts both systems.

**Against the expected outcomes:** `full` as pilot 2 (direct about 0.57, held-out best about 0.03
below): yes (0.568, −0.028). `lowres`: direct 0.2–0.4, yes (0.380); pivots close to or above
direct for some directions out of a low-resource language: the held-out best is level with direct
(+0.001), but via en is below it from a low-resource language (−0.019). RQ3b: NLLB's English-pivot
gain negative, yes; the LLM's closer to zero or positive, yes; most for pairs without English and
with a low-resource language, yes (+0.012 from a low-resource language).

## Files
`results/pilot1/`, `results/pilot2/`: `metrics.csv` (per direction, split and route),
`choosers.csv` (per direction and named route), `run_info.json`. The hop cache
(`v1/results/mt-pivot/hops.jsonl`, keyed by model, languages and input) is reused by later runs.
