# Our method on a standard benchmark: chains and several hubs on PanLex-BLI

**Status: APPROVED by the user, 2026-11-25, with the proposals below** ("go with your proposals
as long as you think these make for a good baseline"). Requested by the user (2026-11-24: baselines against our method only; decisions.md "Baselines only against our
method": PanLex-BLI is the standard benchmark, XLING dropped).

**Question.** On the standard low-resource word translation benchmark, with its own dictionaries
and test sets, does our method (the chain and several hubs, through other benchmark languages)
beat the published methods trained on the direct dictionary alone?

## The benchmark (checked 2026-11-25)
PanLex-BLI (Vulić et al. 2019, EMNLP; github.com/cambridgeltl/panlex-bli), downloaded to
`C:/data/language-graph/external/panlex-bli/` (17 MB):
- **15 languages:** bg, ca, eo, et, eu, fi, he, hu, id, ka, ko, lt, no, th, tr; all **210 directed
  pairs**, each built from PanLex separately (not through English).
- **Per directed pair:** training dictionaries of 500, 1,000, 2,000 and 5,000 pairs (the most
  frequent source words first) and a test set of 2,000 pairs (about 1,670 source words; some have
  several translations). 17 of the 5,000 files are smaller (2,458–4,650 pairs; all involve th).
- **Vectors: exactly ours.** fastText Common Crawl, 200k words. Every word of every file (1.6M
  rows) is in our vocabularies; the paper's normalization (unit, centre, unit) is ours.
- **Its own overlap between training and test:** exact test pairs almost never appear in the 1k
  training set (median 0.05%, at most 1.2%), but 3–15% of the test source words (median 6.5%) do,
  with other translations. The published numbers include that; we keep the benchmark as published
  for the direct dictionary.

## Design
- **Directions:** all 210. **Direct seed:** the benchmark's 1,000-pair file, as published (the
  setting with the most published numbers); 500 as a second setting if time allows.
- **Hub steps:** the benchmark's 5,000-pair files of s-x and x-t, with the test words removed
  (source test words from s-x, target test words from x-t, as in all our experiments). Without the
  removal a chain could rebuild an exact test pair from two hub files, which no direct method
  can. The removal keeps a median 82% of s-x (61–96%) and 98% of x-t (73–100%). The direct
  dictionary keeps the benchmark's own overlap, so this rule can only work against us.
- **Hubs, fixed before any result:** the 3 other benchmark languages with the widest hub steps
  (the largest min(|s-x|, |x-t|) after the removal; ties by language code), the analogue of our
  widest-path rule.
- **Maps and methods:** plain Procrustes (unit, centre, unit), as our main experiments: direct;
  chain through hub 1; combination; several hubs (direct + the 3 chains, CSLS scores averaged).
  The self-learned version (strong maps) only if the user keeps strong maps as the main
  pipeline after tonight's run.
- **The same-data control:** our method uses pivot dictionaries the published methods don't
  have. The control gives direct the same data: the 1k direct pairs pooled with the s-t pairs
  composed through the 3 hubs' dictionaries (joined on the hub word, mutual best, test words
  removed), one Procrustes map. If several hubs beats this too, the gain is not only more data.
- **Evaluation:** CSLS (k = 10), our evaluator. Two conventions, because the papers differ:
  - **per source word, any gold translation correct:** P@1 and MRR (ContrastiveBLI, BLICEr, the
    LLM papers);
  - **per test pair, its one translation** (Vulić et al. 2019: "MAP is equal to MRR", one correct
    translation per query). Which one Vulić et al. used is settled by the reproduction check below
    and stated in the report.
- **Reproduction checks (before comparing):**
  1. our plain Procrustes direct at 1k against Vulić et al.'s orthogonal map at 1k (0.071 MRR, the
     average over the 210 directions);
  2. ContrastiveBLI stage 1 through our wrapper (`../contrastivebli-baseline/`) on its 6 published
     PanLex-BLI directions (bg→ca, ca→he, he→bg, et→hu, hu→eu, eu→et; P@1 41.88, 30.56, 33.49,
     40.35, 20.09, 13.00), 6 fits.

## Published numbers to compare with (each re-read from its paper's table before use)
All at 1k seeds, the benchmark's test sets.

| paper | directions | metric | methods |
|---|---|---|---|
| Vulić et al. 2019, Table 3 | all 210 (average) | MRR | orthogonal 0.071; + self-learning 0.132; VecMap pipeline 0.101; + self-learning 0.212 / 0.255 / 0.274 (best); unsupervised 0.140 |
| Li et al. 2022 (ContrastiveBLI), Table 2 | 6 | P@1 | VecMap, FIPP, C1, C2 (+ mBERT) |
| Li et al. 2022 (BLICEr), Table 2 | 30 (bg, ca, et, ka, he, hu) | P@1 | RCSLS, VecMap, C1, C2, each + BLICEr (XLM-R reranker) |
| Li et al. 2023 (LLMs) | 6 (bg, ca, hu) | P@1 | mT5, mT0, XGLM, mGPT, LLaMA; ContrastiveBLI, VecMap |
| Li et al. 2024 (SAIL) | 6 (bg, ca, hu) | P@1 | unsupervised: VecMap, C1, C2, LLaMA zero-shot and SAIL |

Rows are grouped by resources: static vectors only; + a multilingual language model (C2, BLICEr);
+ an LLM. Every row of ours says it also uses pivot dictionaries.

## Measures (fixed before the run)
- Per direction: P@1 and MRR of each method (both conventions).
- **Headline:** several hubs and the chain against the published methods, on exactly each paper's
  directions (averages over the same directions); and several hubs minus our direct, minus the
  combination, minus the same-data control, pooled over the 210 directions with the paired
  bootstrap over test words, directions above / below 0.

## Expected outcomes (before the run)
1. Our direct about at Vulić et al.'s 0.071 MRR (the reproduction check).
2. The chain and several hubs well above our direct (each hub step has 5k pairs against 1k
   direct), likely above the self-learning methods' 0.27 MRR average, since the 5k direct
   setting reaches about 0.30; and above the same-data control by less.
3. Against ContrastiveBLI and BLICEr on their directions: unclear; they are much stronger direct
   maps, and BLICEr adds XLM-R.

## Cost
Laptop. Each direction: direct, 3 chains and the control, about 5 evaluations; 210 directions,
about 1 h; the 500 setting another hour; ContrastiveBLI's 6 reproduction fits about 30 min (or on
the remote).

## Questions settled (user, 2026-11-25: the proposals)
1. Hub steps at 5,000 (the benchmark's largest training file), direct at 1,000.
2. Test words removed from the hub files; the benchmark's direct file as published.
3. 1k first; 500 if time allows.
4. Plain Procrustes; the self-learned version decided after the several-hubs-with-Procrustes run.

## Amendment 2026-11-25: the paper's basis, and 500 pairs (approved: "do a, yes")
The reproduction check (`experiments/panlex-bli/reproduction/`) found our direct map at 1k
scoring 0.131 MRR against Vulić et al.'s 0.071, mostly from CSLS and centring (length
normalization with nearest-neighbour retrieval gives 0.089; a 0.018 residual remains). So:
- **A second basis (`--basis paper`):** every method rescored with length normalization only and
  nearest-neighbour retrieval; the combination and several hubs average cosine scores instead of
  CSLS scores. Everything else as above. Our methods are set next to the published averages only
  on this basis, with the 0.018 residual stated. Its direct map must equal the check's setting A
  direction by direction.
- **The 500-pair direct file**, on both bases, after the 1k paper-basis run.
- Each basis and size has its own cache and `results/d<size>[-paper]/`.

## Results at 1k (run 2026-11-25; ours at 9c3c058, the paper's basis at 1565faa; `results/d1000*/`)
All 210 directions. MRR per test pair (Vulić et al.'s convention), average over directions;
differences pooled over test pairs, paired bootstrap; above / below: directions whose own interval
excludes 0. Hubs: the 3 widest per direction (fi the most common).

| method | our basis (CSLS, unit-centre-unit) | the paper's basis (nearest neighbour, length only) |
|---|---|---|
| direct (the benchmark's 1k file) | 0.131 | 0.089 |
| chain | 0.196 | 0.152 |
| combination | 0.224 | 0.154 |
| same-data control | 0.250 | 0.199 |
| several hubs | 0.276 | 0.191 |
| several hubs − direct | +0.148 [+0.147, +0.149], 210 / 0 | +0.105 [+0.104, +0.105], 210 / 0 |
| several hubs − same-data control | +0.025 [+0.025, +0.026], 179 / 0 | −0.009 [−0.010, −0.009], 32 / 101 |
| chain − direct | +0.067, 207 / 0 | +0.066, 210 / 0 |

Per source word (any translation; ContrastiveBLI's convention): several hubs 0.298 against direct
0.141 (our basis), 0.206 against 0.096 (the paper's).

**Next to the published averages (the paper's basis only; Vulić et al. 2019, Table 3, 1k).** Their
methods: orthogonal 0.071; full preprocessing 0.101; orthogonal + self-learning 0.132; unsupervised
0.140; full + self-learning 0.212 / 0.255 / 0.274 (best); and orthogonal with 5k seeds 0.201. Ours:
direct 0.089, several hubs 0.191, same-data control 0.199. Our direct map is 0.018 above theirs on
this basis (the reproduction check's residual), so the gaps are if anything flattering to us.

**Against the expected outcomes.**
1. Our direct about at 0.071: not on our basis (0.131); 0.089 on the paper's (the check).
2. The chain and several hubs well above our direct: yes on both bases; above the self-learning
   methods' 0.27: only on our basis (0.276), not on the paper's (0.191). Above the same-data control
   by less: yes on our basis (+0.025), no on the paper's (−0.009).
3. Against ContrastiveBLI and BLICEr: not compared (their per-direction numbers not re-read; the
   reproduction of ContrastiveBLI on its 6 directions deferred).

**Reading.** On the paper's basis, chains and pooling turn the 1k file into about what a 5k direct
file gives (0.191–0.199 against 0.201), above every published method without self-learning and
below the self-learning pipelines (0.212–0.274). Several hubs' edge over pooling the same pairs
depends on CSLS: averaging plain cosines barely helps (the combination 0.154 against the chain
0.152).

## Results at 500 (run 2026-11-25 at 1565faa; `results/d500*/`)
The benchmark's 500-pair direct file; hub steps unchanged (5k), so the chains are the 1k runs'.

| method | our basis | the paper's basis |
|---|---|---|
| direct | 0.052 | 0.031 |
| chain | 0.196 | 0.152 |
| same-data control | 0.249 | 0.197 |
| several hubs | 0.265 | 0.178 |
| several hubs − direct | +0.219 [+0.218, +0.220], 210 / 0 | +0.152 [+0.151, +0.153], 210 / 0 |
| several hubs − same-data control | +0.015 [+0.015, +0.016], 134 / 10 | −0.020 [−0.021, −0.020], 14 / 139 |

- Against Vulić et al.'s Table 5 (full + self-learning, 500 seeds; the paper's basis): fi→ko 0.192,
  eu→tr 0.189, ko→tr 0.264; ours several hubs 0.127, 0.163, 0.206 (same-data control 0.145,
  0.165, 0.199). Below, as at 1k.
- The gain over direct grows as the direct file shrinks (+0.105 at 1k to +0.152 at 500, the
  paper's basis); several hubs' edge over pooling the same pairs depends on CSLS, more clearly
  than at 1k.
