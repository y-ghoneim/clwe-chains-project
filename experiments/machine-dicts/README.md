# Maps and chains trained on machine-made dictionaries (RQ3)

**Question.** Doe et al. trained their maps on word lists made by machine translation (Google
Translate). Do maps and chains behave differently when their dictionaries are made by a machine
instead of collected from human dictionaries? And can a machine-made direct dictionary replace the
chain for low-resource languages, where chains helped (stage 1)?

decisions.md 2026-11-18 (RQ3 redirected; choices: both versions, both pair sets). This README is
written and approved before any run. Results are descriptive.

## Data (prep/11_machine_dicts.py, `v1/phase6/md/<version>/`)
- **Translators:** NLLB-200 3.3B (beams 5, top output; same decoding as the MT pilots, whose cache
  is reused); and an LLM, Claude Haiku 5.5 (prep/12, batch API, one word per source word, JSON
  output; run when the API key is set, budget $20, estimated $3–5).
- **A machine dictionary for an edge a–b:** each source word of a translated into b and each source
  word of b translated into a; the top output, cleaned of punctuation, fillers and articles (as in
  the MT pilots), kept if it is a single word in the target's vocabulary. Both directions are
  merged into one dictionary (the stage runner uses one dictionary per edge for both directions).
- **Version a:** each dictionary from direct translations. **Version b:** the s–t dictionary
  translated through the pair's hub (s → hub → t, both directions); chain edges as in version a
  (kept in the version's `chain/` folder, because 4 pairs' direct edges share a file name with
  other pairs' chain edges, e.g. en__ja).
- **Checks:** a machine dictionary under 100 pairs stops the build (failed translations); each
  version's `dict_sizes.csv` reports the share of identical-string pairs (NLLB copies words it
  can't translate); the runner refuses an evaluation cache built from other dictionaries.
- **Set "highres":** phase 4's 40 pairs, on stage D's test words, folds and gold. Source words =
  each language's list words only (one word list, as Doe et al.). Every edge (s–t, s–hub, hub–t)
  is machine-made. A third version, **highres-list**, uses the shared list itself as every edge's
  dictionary (the human reference with the same words).
- **Set "low":** stage 1's pairs of the low-resource languages NLLB covers: Yoruba (20 pairs),
  Odia (17), Malagasy (16), Assamese (7) = 60 pairs (Meadow Mari and Erzya are not in NLLB).
  Source words = list words plus each language's 1,000 most frequent words. Only s–t is
  machine-made; the chain keeps its natural dictionaries (stage 1's), so the question is whether a
  machine direct dictionary beats the natural chain.
- **Test words, folds, test removal, gold:** the base stage's, unchanged (5 disjoint folds; test
  words removed from every edge, all casings; gold = list word + PanLex (+ Wiktionary with en)).
  Machine output never enters the gold.

## Methods (run.py → stage 1's runner, per version)
Direct ("thin": the version's s–t dictionary), the chain through the hub, and their combination,
in one CSLS pass; CSLS k = 10 over the target's full vocabulary; paired bootstrap per
pair-direction.

## Comparisons (reported descriptively)
- **highres:** list vs nllb-a vs nllb-b (vs llm-a, llm-b): direct MRR, chain − direct, combination
  − direct; next to stage D (natural dictionaries) for scale.
- **low:** nllb-a direct (machine) vs stage 1's thin direct (natural, < 500 pairs) and stage 1's
  chain (natural); chain and combination within each version; a vs b.
- Per stratum / per language, with standout pairs (user, 2026-11-18).

## Expected outcomes (written before any run)
- **highres, machine vs list:** machine dictionaries are noisier but cover both directions, so
  about 1,000–1,500 pairs per edge against about 670 list pairs per fold. Direct maps on them reach
  about the list version's MRR (within ±0.03).
- **highres, chain vs direct:** as with the list (RQ1), the chain stays below direct, because
  every edge has about the same amount of data.
- **highres, version b:** the hub-translated direct dictionary is slightly worse than version a
  (the MT pilots: direct translation beat translating through a pivot).
- **low:** a machine direct dictionary (about 1,000–1,800 pairs) beats the natural thin direct
  (< 500) clearly, and reaches or beats the natural chain for Odia, Assamese and Malagasy; less so
  for Yoruba, where NLLB is weaker. The combination stays the best method.
- **LLM vs NLLB:** similar for the 40 pairs; the LLM weaker for the low-resource languages.

## Compute
- NLLB translations (remote, cache on the remote): highres about 105k new, low about 400k new:
  about 1 h at 140/s. Don't run it at the same time as the MT pivot runs (same cache file).
- Evaluations: highres 3 versions × 400 direction-folds, low 2 × 600, about 2.2 s each: about
  90 min.
- LLM (laptop, API): `prep/12 --set highres` then `--set low`; two batches per set; then
  `prep/11 --system llm` and the runner on the llm versions.

## Results
NLLB versions and `highres-list` run on the remote at 4390b56 (2026-11-19); LLM versions on the
laptop at 72eeb01 / 837ee55 (same evaluation code). LLM translations: Claude Haiku 5.5, batch
API, $4.10 for both sets; 1 (highres) and 55 (low) of the translations missing, no pair.

**Checks.** The low-resource chains equal stage 1's exactly (NLLB versions; LLM versions within
0.0003, laptop GPU). `highres-list` direct against phase 4's direct over the 40 pairs: mean 0.398
vs 0.393, correlation 0.994.

**The 40 pairs** (per stratum: (a) both high-resource, (b) one weaker; mean over pairs):

| training dictionary | pairs per edge (median) | direct MRR (a / b) | chain − direct (a / b) | chain above / below (of 80) | combination − direct (a / b) | combination above / below |
|---|---|---|---|---|---|---|
| human list | about 670 per fold | 0.575 / 0.223 | −0.015 / −0.010 | 0 / 24 | +0.030 / +0.031 | 75 / 0 |
| NLLB, direct (a) | about 1,200 | 0.583 / 0.237 | −0.015 / +0.002 | 14 / 17 | +0.053 / +0.062 | 79 / 0 |
| LLM, direct (a) | 1,176 | 0.621 / 0.273 | −0.016 / −0.011 | 5 / 33 | +0.034 / +0.044 | 79 / 0 |
| NLLB, through the hub (b) | about 1,200 | 0.551 / 0.221 | +0.016 / +0.018 | 43 / 4 | +0.067 / +0.064 | 80 / 0 |
| LLM, through the hub (b) | 1,198 | 0.608 / 0.263 | −0.004 / −0.001 | 6 / 9 | +0.039 / +0.045 | 80 / 0 |
| natural (stage D, for scale) | median about 12,000 | 0.698 / 0.338 | −0.018 / +0.007 | 9 / 22 | +0.018 / +0.056 | 67 / 0 |

- LLM dictionaries give a better direct map than the human list for all 40 pairs (largest
  Korean–Turkish, Tagalog–Chinese, Hungarian–Korean, Tamil–Chinese: +0.08 to +0.09); NLLB for
  24 of 40 (largest Korean–Turkish +0.037, Tamil–Chinese +0.033).
- Building the direct dictionary through the hub makes it worse for both systems (NLLB −0.032 /
  −0.016, LLM −0.013 / −0.010). With NLLB's, the chain (built from directly translated edges)
  then beats direct in 43 of 80 directions (largest Finnish–Croatian +0.06, Basque–Finnish +0.05).
- Machine dictionaries cover both translation directions, so they have more pairs than the list;
  "at least as useful as the list" is about the dictionary as made, not per pair.

**The low-resource pairs** (60 pairs, 120 directions; MRR, mean over each language's directions):

| language | human direct (stage 1) | NLLB direct (a) | LLM direct (a) | NLLB via hub (b) | LLM via hub (b) | chain (human data) | NLLB (a) + chain | LLM (a) + chain |
|---|---|---|---|---|---|---|---|---|
| Assamese | 0.024 | 0.056 | 0.069 | 0.049 | 0.061 | 0.069 | 0.102 | 0.111 |
| Malagasy | 0.024 | 0.058 | 0.064 | 0.050 | 0.060 | 0.078 | 0.101 | 0.102 |
| Odia | 0.006 | 0.085 | 0.094 | 0.079 | 0.086 | 0.062 | 0.123 | 0.128 |
| Yoruba | 0.015 | 0.018 | 0.022 | 0.018 | 0.022 | 0.031 | 0.038 | 0.039 |
| **all** | **0.016** | **0.052** | **0.059** | 0.048 | 0.055 | **0.057** | **0.086** | **0.089** |

- A machine direct dictionary (median 1,770 pairs for NLLB) beats the human one (< 500) in 108
  (NLLB) and 112 (LLM) of 120 directions, and the chain in 36 and 46: about level with the chain
  on average.
- Standouts: Odia's LLM direct maps beat its chain clearly (Odia → Indonesian 0.246 vs 0.134,
  Odia → Thai 0.187 vs 0.091); Malagasy's stay below (Malagasy → Persian 0.072 vs 0.111).
- Combined with the chain, a machine direct map is the best method: above the chain in 110 (NLLB)
  and 114 (LLM) of 120 directions.

**Against the expected outcomes:**
- Machine vs list within ±0.03: NLLB yes (+0.008 / +0.014); the LLM better than expected (+0.046
  / +0.050).
- The chain below direct as with the list: yes for the list and the LLM; with NLLB's direct
  dictionaries about even (14 above / 17 below).
- Version b slightly worse than a: yes, for both systems.
- Low-resource: a machine direct dictionary beats the human one clearly, yes; reaches or beats
  the chain for Odia (yes), Assamese (LLM level, NLLB below) and Malagasy (no); Yoruba weakest,
  yes; the combination best, yes.
- LLM similar to NLLB on the 40 pairs and weaker for low-resource languages: no, the LLM is better
  in both.
