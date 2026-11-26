# A published competitor on our data: ContrastiveBLI against our direct, chain and several hubs

**Status: APPROVED by the user, 2026-11-24** ("yes go ahead"): their vectors; our test-word
rule, with the published variant on 3 well-resourced pairs; 6 pairs per group; a fifth group,
well-resourced pairs with the direct dictionary cut to 1,000 pairs. The comparisons that matter are
our method's (chain, several hubs) against C1 (user: "direct doesnt matter").

**Amended 2026-11-25, after the remote smoke test (approved by the user the same day):**
1. **Bug fixed: each group has its own cache and maps** (`<results>/contrastivebli-baseline/
   <group>/`). The well-resourced group and its 1,000-pair cut share pairs, and the cache keys
   carried no group, so the smoke test's 1,000-pair fit for Estonian–Hebrew was read as the full
   dictionary's. The smoke test's outputs are discarded.
2. **Ours in both versions:** strong maps (self-learned direct, chain, several hubs) and several
   hubs with plain Procrustes (`../several-hubs-procrustes/`: chain, combination, several hubs),
   each where it covers the group. Which version leads the report is decided after the
   several-hubs-with-Procrustes run (user, 2026-11-24).
3. **The 1,000-pair group:** strong maps covers it only on its 6 pilot pairs, so for
   the drawn pairs our method comes from several hubs with plain Procrustes, run on that budget
   too (`run.py --units dc_b1000` there; about 30 min on the remote, after its main run).
Requested by the user
(2026-11-24, from the professor's feedback: "recreate the latest published competing method and
use our data and then compare results"). Timing and a first look on two pairs:
`analyses/2026-11-24-contrastivebli-timing/`.

**Question.** Our strongest results (strong maps: several hubs over self-learned rotations) beat
our own direct maps. Do they also beat a recent published word-translation method trained on the
same direct dictionary?
1. On low-resource pairs, does several hubs beat ContrastiveBLI's direct map, and by how much?
2. On well-resourced pairs, where ContrastiveBLI should be strongest, where does several hubs stand
   against it?

**The competitor.** ContrastiveBLI, stage C1 (Li et al., ACL 2022, "Improving Word Translation via
Two-Stage Contrastive Learning"; code github.com/cambridgeltl/ContrastiveBLI, commit
31cf6888e0998b7c8cde69b7619dbaa919cf574e). Its stage C1 uses static word vectors only:
- two linear maps (general, not rotations), one per language, into a shared space;
- initialized with a VecMap-style whitening, SVD and re-weighting solution;
- trained with a contrastive loss with hard negatives;
- inside self-learning rounds that add induced word pairs.

Stage C2 (adding mBERT) is not run. It is explained in the report instead: mBERT covers few of our
low-resource languages, and C2's cost (a cross-encoder fine-tuned per pair) is far beyond the time
left.

## Design
- **The data are strong maps' own** (`../strong-maps/`, cache `55ca51e319`): the pairs, hubs,
  folds, test words, gold and evaluation are unchanged. Ours (amendment 2) means:
  - strong maps: self-learned direct (`vm-thin`), the chain through the best hub (`vm-chain`),
    several hubs (`vm-hubs3`);
  - plain Procrustes (`../several-hubs-procrustes/`): the chain (`proc-chain`), the combination
    (`proc-combo`), several hubs (`proc-hubs3`).
- **Plain Procrustes direct** is refitted in the runner, a check that the runner's test words and
  evaluation match the cache: the timing check matched it exactly on 4 directions.
- **Pairs, fixed by rule before any result.** Per pair group: a seed-0 random draw from the
  group's pairs, leaving out strong maps' pilot pairs (several hubs was chosen on them).

  | pair group | data | pairs left after the pilot | drawn (directions) | settings |
  |---|---|---|---|---|
  | low-resource, under 500 | `../scarce-natural-s1/` | 71 | 6 (12) | the authors' "1k" |
  | low-resource, 500–1,000 | `../scarce-moderate-a/` | 121 | 6 (12) | "1k" |
  | low-resource, 1,000–3,000 | `../scarce-moderate-a2/` | 39 | 6 (12) | "1k" |
  | well-resourced | `../natural-highres-d/` | 34 | 6 (12) | "5k", seeds capped at 5,000 |
  | well-resourced, direct cut to 1,000 | `../controlled-scarcity/` (budget 1,000) | the same 6 | 6 (12) | "1k" |

  The last group reuses the well-resourced group's pairs (where the budget folder has them), so
  the two differ only in the direct dictionary's size. It is the closest match on our data to the
  authors' own 1k setting. `pairs.csv` holds the draw (made 2026-11-24, before any run).

- **Fold 0 only.** All 5 folds are about five times the cost (timing below).
- **One fit per pair serves both directions.** C1 maps both languages into one shared space.
- **C1's training data is our direct dictionary for the fold, test words removed** (as every
  method).
  - **Well-resourced pairs:** at most the 5,000 most frequent word pairs (smallest larger row
    number, ties in file order). Those dictionaries hold 2k–267k pairs, and the authors' "5k"
    setting puts every seed pair into one batch. Ours keep the full dictionaries. This works
    against C1, which still beat our direct map in the timing check.
- **C1's settings are the authors' published ones, not tuned on our data:**
  - "1k" (for 1k seed pairs): 3 rounds × 50 steps, 60 negatives, up to 6,000 induced pairs a side
    from the 20,000 most frequent words.
  - "5k": 2 rounds × 200 steps, 150 negatives, up to 10,000 induced pairs from 60,000 words.

  Tuning them would need cross-validation over pairs (brief 5.4), with no time for it. The report
  says so.
- **The authors' code, imported unchanged.** The runner runs main.py's training loop without its
  logging evaluation, which doesn't change training.
  - **Each batch is split into parts** (3,000 for "1k", 1,000 for "5k"). Each part's loss is
    weighted by its share, so the gradient is the full batch's. Only the memory changes: the
    timing check gave identical scores with and without the split.
- **The rule: what belongs to their method follows the authors; what belongs to the evaluation is
  the same for every method** (user, 2026-11-24).
- **Input vectors: the authors' preprocessing** (question 1). C1 gets the raw fastText vectors
  (the same 200k-word files as ours) and unit-normalizes them as its code does, without our
  centring. It is part of their method.
- **Test words are removed from C1's induced pairs** (question 2), as strong maps removes them from
  its own self-learning (brief 5.10): after each induction round, any induced pair with a test
  word of the fold on its side is dropped, in every casing. It is part of the evaluation: the
  published code would induce pairs among all 20,000 most frequent words, test words included.
  C1's hard negatives may still include test words: they are not training pairs.
  - **Both variants on the first 3 well-resourced pairs**, where C1 is strongest: also the
    authors' self-learning as published (no filter). This shows how much the rule moves C1's score
    (about +36 min). The headline uses the rule.
- **Seeds:** `torch.manual_seed(0)`; the authors' NumPy seed 1234 (set in their main.py). The GPU's
  top-k search is not bit-reproducible across machines (as strong maps).
- **Kept for a possible follow-up:** each pair's two maps (2 × 300 × 300), in
  `<results>/contrastivebli-baseline/maps/`. A follow-up could build chains and several hubs from
  C1's own maps; it is not part of this experiment.

## Measures (fixed before the run)
- **Per direction:** CSLS (k = 10) MRR and P@1 on the main gold, on fold 0's test words.
- **Per pair group, pooled over its directions' test words:**
  - mean MRR of plain Procrustes, C1, and our direct, chain and several hubs;
  - **the headline:** ΔMRR, several hubs minus C1, with the 95% paired bootstrap interval over test
    words (as strong maps), and directions above / below 0; for each version of ours (amendment 2),
    and the chain (and, with Procrustes, the combination) minus C1 likewise;
  - also ΔMRR of C1 minus our direct, and of C1 minus plain Procrustes: how strong a direct map C1
    is on our data;
  - well-resourced only: ΔMRR of C1 as published (no filter) minus C1 with the rule, with its
    interval.
- **Not compared:** XLING or the other published benchmarks (a separate README, if approved).

## Expected outcomes (before the run)
From the timing check's two pairs, so weak expectations:
1. **Low-resource:** C1 only a little above plain Procrustes, at or below our direct, and clearly
   below several hubs (Kurdish → Turkish: C1 0.087 MRR, several hubs 0.310).
2. **Well-resourced:** C1 above our direct by about +0.1 MRR, and level with several hubs or above
   it (Turkish → Basque: C1 0.622, several hubs 0.564; Basque → Turkish 0.699 each).
3. **If C1 beats several hubs on low-resource pairs too,** the report says our method doesn't beat
   a strong published direct map there, and the claim narrows.

## Cost
Laptop, per pair and fold (timing check): about 4.5 minutes for "1k" and about 12 minutes for "5k".
- 18 low-resource pairs: about 1.4 h;
- 6 well-resourced pairs, plus 3 as published: about 1.8 h;
- 6 well-resourced pairs cut to 1,000: about 0.5 h;
- **about 3.7 h in total** at the laptop's speed.

## Files
- `config.yaml`: the pair draw, C1 settings, the cap, batch parts, the path to the authors' code.
- `run.py`:
  - `--make-pairs` writes `pairs.csv` (the draw; committed before the run);
  - the default runs every pair, resumable, a cache per pair;
  - `--results-only` builds `results/` (per direction, per group, the bootstrap).
- `results/`: the tables, `run_info.json` (commit, config, the authors' commit, GPU, timings).
- **The remote** needs the authors' repository at the pinned commit in
  `C:\data\language-graph\external\contrastivebli\repo` (a `git clone` and `git checkout`),
  plus strong maps' data as before.
- Tests in `tests/test_contrastivebli_baseline.py`:
  - the pair draw (rule, seed, no pilot pairs);
  - the cap (most frequent, test words already gone);
  - induced test pairs dropped;
  - the split batch gives the full batch's gradient.

## Questions settled (user, 2026-11-24)
1. Input vectors: the authors' preprocessing.
2. Test words in C1's self-learning: removed (the rule); the published variant on 3
   well-resourced pairs.
3. Pair count: 6 per group ("enough to have some baselines"); more can follow with the same draw.
4. The under-500 group: kept; its floor is reported.
5. Added: well-resourced pairs with the direct dictionary cut to 1,000 pairs.

## Results (fits on the remote: four natural groups at b206a8e, the 1,000-pair group at 351b88c; tables built at 56c5a0b; `results/`)
6 pairs per group, both directions (12), fold 0. MRR pooled over test words, paired bootstrap;
above / below: directions whose own interval excludes 0. C1 = ContrastiveBLI stage 1, test words
dropped from its induced pairs (the rule). Check: plain Procrustes direct refitted here matches the
stages' caches rank for rank in every group (12 of 12 records).

| group | C1 | our direct | several hubs − C1 | chain − C1 | C1 − our direct |
|---|---|---|---|---|---|
| low-resource, under 500 | 0.009 | 0.017 | +0.059 [+0.048, +0.070], 9 / 0 | +0.043, 9 / 0 | −0.007 [−0.013, −0.002], 0 / 3 |
| low-resource, 500–1,000 | 0.021 | 0.031 | +0.050 [+0.039, +0.061], 10 / 0 | +0.033, 7 / 0 | −0.010, 0 / 6 |
| low-resource, 1,000–3,000 | 0.044 | 0.081 | +0.126 [+0.108, +0.143], 11 / 0 | +0.068, 10 / 0 | −0.037, 0 / 7 |
| well-resourced, full dictionaries | 0.544 | 0.491 | +0.014 [+0.001, +0.027], 4 / 3 | −0.055, 0 / 6 | +0.053, 8 / 1 |
| well-resourced, direct cut to 1,000 | 0.369 | 0.257 | +0.176 [+0.160, +0.191], 11 / 0 | +0.120, 10 / 1 | +0.112, 8 / 0 |

(Our methods: plain Procrustes, several-hubs-procrustes' cache. With self-learned maps, strong
maps' cache, several hubs − C1: +0.063, +0.048, +0.142, +0.017; not run on the 1,000-pair group.)
- The published variant (test words left in C1's induced pairs, 3 well-resourced pairs) against
  the rule: −0.005 [−0.011, +0.001], 0 / 1: the rule costs C1 nothing.

**Against the expected outcomes.**
1. Low-resource: C1 at or below our direct and clearly below several hubs: yes, and C1 is below
   our direct in every low-resource group (its self-learning drifts from so few seed pairs).
2. Well-resourced: C1 above our direct (+0.053; expected about +0.1) and level with several hubs
   (+0.014 for several hubs, 4 above, 3 below): yes, in the main.
3. C1 does not beat several hubs on low-resource pairs; the claim stands.
