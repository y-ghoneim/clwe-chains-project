# Figure: the language graph with natural dictionaries (review C3)

**Status: APPROVED by the user, 2026-11-21 (design as below; report figures live in
`figures/`).** A report figure from existing data; no new runs
and no new numbers. Raised by the 2026-11-19 review (section C, item 3); the user asked for it on
2026-11-21. The graph subsection it belongs to (C1, C2, C4, C5) is writing, decided separately.

**What it shows.** The project's graph as phase 6 used it: languages as nodes, natural
dictionaries (PanLex, plus English Wiktionary on edges with English) as edges, and the routes the
RQ2 rule picks (the widest path: the hub whose smaller dictionary to the two endpoints is
largest). The point to make visible: English (and a few large languages) sit at the centre of
the graph's supervision, so the widest path almost always goes through them, and the
low-resource languages' direct edges are thin next to their edges to the hub.

## Design
- **Data:** `v1/phase6/routes/dict_sizes.csv` (stage 1E's edges: 615 edges among 36 languages,
  the 30 nodes and stage 1's six low-resource languages as, mg, mhr, myv, or, yo); edge size =
  distinct word pairs, as the report quotes dictionary sizes. Hubs from
  `experiments/scarce-natural-s1/pairs.csv` (stage 1's pairs: en 44, hi 17, fr 16).
- **Panel (a), the graph:** the 30 nodes on a circle ordered by CC-100 size (the resource order
  of decisions.md 2026-11-15), the six low-resource languages on an inner ring next to their
  hub; every edge drawn grey with width and opacity rising with log10(pairs); each low-resource
  language's edge to its chosen hub highlighted in colour (yo–en, mhr–en, myv–en, as–en, or–hi,
  mg–fr).
- **Panel (b), one route:** Yoruba–Arabic (stage 1's first pair, chosen before plotting): the
  direct edge (586 pairs) against the route through English (yo–en 7,436, en–ar from the same
  file), sizes printed on the edges.
- **Output:** `language_graph.pdf` (for the report) and `.png` (for reading in the repo), written
  next to the script; matplotlib (in the venv). Deterministic (fixed layout).
- **Units:** edge sizes are distinct word pairs (as the gain-curve analysis counts them); stage
  1's `pairs.csv` sizes are min(distinct words) (yo–ar direct 433, route 3,079: about 7× rather
  than 13×). The report must say which unit a number is in and not mix them.
- mt has no CC-100 size and is placed last (just left of en); the caption says so.

## Built (2026-11-21)
`python figures/language-graph/make_figure.py` writes `language_graph.pdf` and `.png` (the PDF is
byte-identical across runs). Each low-resource language uses one hub for all its stage 1 pairs
(yo, mhr, myv, as: en; or: hi; mg: fr), so panel (a) has one highlighted edge per language.
Panel (b): yo–ar direct 586 pairs, yo–en 7,436, en–ar 131,688 (the narrower step 13× direct).

## Where this lives (settled: user, 2026-11-21, as proposed)
The repo had no place for report figures. Proposed and approved: `figures/<name>/` with the script, this
README and the small output files, committed; the CLAUDE.md layout table gets a `figures/` row
("report figures from committed or v1 data; no new results"). Alternative: an `analyses/` folder,
but those are for checks that inform a decision.
