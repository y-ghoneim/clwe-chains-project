"""Strong maps: can our method stay ahead of a strong direct map? See README.md (approved before
running).

Units: stages 1, A, A2, D and each controlled-scarcity budget (dc_b<budget>), each with stage 1's
runner's data handling (experiments/scarce-natural-s1/run.py: pairs, folds, test words, gold,
dictionaries, cache; unchanged) and this experiment's maps. Per pair, fold and direction, ranked in
one pass per pipeline:
- vm, VecMap + self-learning (align.self_learn; one fit per edge and fold serves both directions):
  thin (direct), chain, combo, joined (pooled direct, stages with joined dictionaries) and the
  candidates w<w> (weighted combinations), hubs3 (direct + the chains through the 3 widest hubs;
  chain2 and chain3 are the second and third), seeded (chain-seeded direct); thin_shared
  (retrieval in VecMap's shared space, a diagnostic) in the pilot only;
- proc, Procrustes as the stages (pilot only: the reproduction check and go / no-go);
- rc, RCSLS (align.fit_rcsls; one fit per direction): thin, chain, combo, where the scenario has
  rcsls (stage D, the budgets).
`--make-pilot` writes pilot_pairs.csv; `--pilot` runs the pilot pairs with every method and writes
results/pilot/ (candidates, the choice, go / no-go); `--full` runs every pair with the baseline, the
controls and the chosen candidate and writes results/full/. Hub-edge fits are kept on disk
(<results>/strong-maps/maps/), keyed by their inputs, so the budgets reuse stage D's. `--check`
(laptop) compares proc with the stages' caches; `--results-only` rebuilds the tables;
`--data-files` lists every input for the remote copy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from clwe_chains.align import compose, fit_rcsls, fit_vecmap, self_learn, side_map
from clwe_chains.dictionaries import dict_sizes, joined_dictionary, remove_test
from clwe_chains.evaluate import bli_multi, bli_with_combination, bli_with_combinations
from clwe_chains.machine_dicts import frequent_rows
from clwe_chains.runs import Cache, file_fingerprints
from clwe_chains.stats import paired_bootstrap
from clwe_chains.text import load_vocab

HERE = Path(__file__).resolve().parent
NAME = HERE.name
RESULTS = HERE / "results"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    f"{NAME}_s1_run", HERE.parent / "scarce-natural-s1" / "run.py")
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)
PIPELINES = ("vm", "proc", "rc")


def fit_version() -> str:
    """A tag of everything that changes a fit or a ranking beyond stage 1's EVAL: align.py's
    source, the map and vocabulary settings, the embeddings' normalization. It names the cache
    folder and keys the stored fits, so a change starts both afresh."""
    src = (HERE.parents[1] / "src" / "clwe_chains" / "align.py").read_bytes()
    parts = [src.replace(bytes([13, 10]), bytes([10])),     # same tag whatever the line ends
             json.dumps({k: XCFG[k] for k in ("vocab", "selflearn", "rcsls")}, sort_keys=True),
             json.dumps(s1.CFG["embeddings"]["normalize"])]
    return hashlib.sha1(b"|".join(p if isinstance(p, bytes) else p.encode() for p in parts)
                        ).hexdigest()[:10]


VERSION = fit_version()
KW = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}


# ---------------------------------------------------------------------------------------------
# Units, pairs and methods

def units() -> list[str]:
    """Every unit: s1, a, a2, d, dc_b<budget>."""
    out = []
    for sc, c in XCFG["scenarios"].items():
        out += [f"{sc}_b{b}" for b in c["budgets"]] if "budgets" in c else [sc]
    return out


def scenario_of(unit: str) -> tuple[str, int | None]:
    sc, _, b = unit.partition("_b")
    return sc, int(b) if b else None


def configure(unit: str) -> None:
    """Point stage 1's runner at one unit's data, cache and log."""
    sc, b = scenario_of(unit)
    c = XCFG["scenarios"][sc]
    s1.NAME = f"{NAME}/{VERSION}/{unit}"
    s1.DATA = f"{c['data']}/b{b}" if b is not None else c["data"]
    s1.HERE = HERE.parent / c["experiment"]
    s1.RESULT_PATHS = ("src/", "prep/", "configs/", "experiments/scarce-natural-s1/run.py",
                       "experiments/scarce-natural-s1/config.yaml",
                       f"experiments/{c['experiment']}/pairs.csv", f"experiments/{NAME}/")
    s1.OWN_RESULTS = f"experiments/{NAME}/results/"


def vm_names(joined: bool, mode: str, choice: str | None) -> list[str]:
    """The vm methods of a run: the baseline and controls, and every candidate (pilot) or the
    chosen one (full)."""
    names = ["thin", "chain", "combo"] + (["joined"] if joined else [])
    cands = set(XCFG["candidates"]) if mode == "pilot" else {choice}
    if "wcombo" in cands:
        names += [f"w{w}" for w in XCFG["weights"]]
    if "hubs3" in cands:
        names += ["chain2", "chain3", "hubs3"]
    if "seeded" in cands:
        names += ["seeded"]
    if mode == "pilot":
        names += ["thin_shared"]
    return names


def methods_for(sc: dict, mode: str, choice: str | None) -> list[tuple[str, str]]:
    out = [("vm", n) for n in vm_names(sc["joined"], mode, choice)]
    if mode == "pilot":
        out += [("proc", n) for n in ("thin", "chain", "combo")]
    if sc["rcsls"]:
        out += [("rc", n) for n in ("thin", "chain", "combo")]
    return out


def make_pilot() -> pd.DataFrame:
    """The pilot pairs (README rule, seed 0): stage 1 one pair per language, A and A2 one pair
    from each of 4 source languages, D 6 pairs, and D's pilot pairs at the pilot budgets where the
    budget folder has them."""
    rng = np.random.default_rng(XCFG["seed"])
    rows = []
    for sc in ("s1", "a", "a2", "d"):
        configure(sc)
        pairs = pd.read_csv(s1.HERE / "pairs.csv").sort_values(["s", "t"], ignore_index=True)
        rule = XCFG["pilot"][sc]
        if "per_language" in rule:
            langs = sorted(set(pairs["s"]))
        elif "languages" in rule:
            langs = sorted(rng.choice(sorted(set(pairs["s"])), rule["languages"], replace=False))
        else:
            langs = None
        if langs is None:
            picked = pairs.iloc[sorted(rng.choice(len(pairs), rule["pairs"], replace=False))]
        else:
            picked = pd.concat([g.iloc[[int(rng.integers(len(g)))]] for lang in langs
                                for g in [pairs[pairs["s"] == lang]]])
        rows += [{"unit": sc, "s": s, "t": t} for s, t in zip(picked["s"], picked["t"],
                                                                 strict=True)]
    d_pilot = {(r["s"], r["t"]) for r in rows if r["unit"] == "d"}
    for b in XCFG["pilot"]["dc_budgets"]:
        configure(f"dc_b{b}")
        have = pd.read_csv(s1.p6_dir() / "pairs.csv")
        rows += [{"unit": f"dc_b{b}", "s": s, "t": t}
                 for s, t in zip(have["s"], have["t"], strict=True) if (s, t) in d_pilot]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Context and dictionaries

class Ctx(s1.Ctx):
    """Stage 1's context on one unit's data, with the extra hubs, induction vocabularies and the
    on-disk store of hub-edge fits."""

    store_shared = None          # one EmbeddingStore for every unit (GPU memory)

    def __init__(self, unit: str):
        super().__init__(None)
        if Ctx.store_shared is None:
            Ctx.store_shared = self.store
        self.store = Ctx.store_shared
        self.unit = unit
        self.scenario, self.budget = scenario_of(unit)
        self.sc = XCFG["scenarios"][self.scenario]
        if self.budget is not None:
            self.all_pairs = pd.read_csv(self.p6 / "pairs.csv")
        hub_root = s1.CFG.path("phase6") / "hubs" / self.sc["hubs"]
        hubs = pd.read_csv(hub_root / "hubs.csv")
        self.all_pairs = self.all_pairs.merge(hubs[["s", "t", "hub1", "hub2", "hub3"]],
                                              on=["s", "t"], how="left", validate="one_to_one")
        if not (self.all_pairs["hub"] == self.all_pairs["hub1"]).all():
            raise ValueError(f"{unit}: hubs.csv's first hubs differ from the stage's")
        self.pairs = self.all_pairs
        self.hub_dir = hub_root / "dicts"
        self._vocab: dict[str, np.ndarray] = {}
        self.maps = s1.CFG.path("results") / NAME / VERSION / "maps"
        self.maps.mkdir(parents=True, exist_ok=True)

    def dictionary(self, name: str) -> pd.DataFrame:
        """dicts/ and joined/ as stage 1; "direct:<s>__<t>" from a budget folder's direct/;
        "hub:<a>__<b>" from the extra hubs' dicts/."""
        kind, _, edge = name.rpartition(":")
        if not kind:
            return super().dictionary(name)
        if name not in self._dicts:
            folder = self.p6 / "direct" if kind == "direct" else self.hub_dir
            self._dicts[name] = pd.read_parquet(folder / f"{edge}.parquet")
        return self._dicts[name]

    def vocab(self, lang: str) -> np.ndarray:
        """The induction vocabulary: the language's most frequent word rows."""
        if lang not in self._vocab:
            words = load_vocab(s1.CFG.path("v0_embeddings") / f"{lang}.vocab.txt")
            self._vocab[lang] = np.array(frequent_rows(words, XCFG["vocab"]), dtype=np.int64)
        return self._vocab[lang]


def fold_dicts(ctx: Ctx, s: str, t: str, x: str, extra: list[str], fold: int
               ) -> tuple[dict, set, set, dict]:
    """The fold's test-removed dictionaries, as the stages remove them (stage 1's fold_dicts;
    controlled scarcity's direct from the budget folder): direct, sx, xt, joined (mutual best,
    where the scenario has it) and sx<i>, xt<i> for the extra hubs; with the test rows."""
    test = ctx.folds[s, t][fold]
    ts, tt = ctx.test_rows(s, test), ctx.test_rows(t, test)
    name = f"direct:{s}__{t}" if ctx.budget is not None else f"{s}__{t}"
    direct = remove_test(ctx.dictionary(name), ts, tt)
    d = {"direct": direct,
         "sx": remove_test(ctx.dictionary(f"{s}__{x}"), ts, set()),
         "xt": remove_test(ctx.dictionary(f"{x}__{t}"), set(), tt)}
    if ctx.sc["joined"]:
        d["joined"] = joined_dictionary(direct, ctx.dictionary(f"{s}__{x}__{t}"), ts, tt,
                                        filtered=True)
    for i, h in enumerate(extra, start=2):
        d[f"sx{i}"] = remove_test(ctx.dictionary(f"hub:{s}__{h}"), ts, set())
        d[f"xt{i}"] = remove_test(ctx.dictionary(f"hub:{h}__{t}"), set(), tt)
    return d, ts, tt, {k: dict_sizes(v) for k, v in d.items()}


# ---------------------------------------------------------------------------------------------
# Fits

def digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def stored(ctx: Ctx, key: str | None, fn, device: torch.device):
    """fn()'s result, kept on disk under key (None: not kept), loaded onto device."""
    path = ctx.maps / f"{key}.pt" if key else None
    if path is not None and path.exists():
        return torch.load(path, map_location=device)
    out = fn()
    if path is not None:                    # atomic: an interrupted save leaves no .pt behind
        tmp = path.with_suffix(".tmp")
        torch.save(out, tmp)
        os.replace(tmp, path)
    return out


def vm_fit(ctx: Ctx, a: str, b: str, d: pd.DataFrame, excl_a: set, excl_b: set, *,
           init: torch.Tensor | None = None, keep: bool = False, reference: bool = False
           ) -> dict:
    """self_learn on an edge dictionary (a, b rows), each side's test rows left out of its
    induction vocabulary; asserts that the final dictionary holds no test row."""
    va = np.setdiff1d(ctx.vocab(a), np.array(sorted(excl_a), dtype=np.int64))
    vb = np.setdiff1d(ctx.vocab(b), np.array(sorted(excl_b), dtype=np.int64))
    sl = XCFG["selflearn"]

    def run() -> dict:
        t0 = time.perf_counter()
        out = self_learn(ctx.store[a], ctx.store[b], torch.tensor(d["a"].to_numpy()),
                         torch.tensor(d["b"].to_numpy()), torch.as_tensor(va),
                         torch.as_tensor(vb), k=sl["k"], rounds=sl["rounds"], tol=sl["tol"],
                         min_full=sl["min_full"], block=s1.CFG["eval"]["block_rows"], init=init,
                         final=sl["final"])
        hits = (int(np.isin(out.a.cpu().numpy(), list(excl_a)).sum())
                + int(np.isin(out.b.cpu().numpy(), list(excl_b)).sum()))
        if hits:
            raise AssertionError(f"{a}-{b}: {hits} test rows in the self-learned dictionary")
        ref = None                  # VecMap's full pipeline on the final dictionary (reference)
        if reference and min(len(torch.unique(out.a)), len(torch.unique(out.b))) > out.wx.shape[0]:
            ref = fit_vecmap(ctx.store[a][out.a], ctx.store[b][out.b])
        return {"wx": out.wx, "wz": out.wz, "ref": ref,
                "info": {"edge": f"{a}__{b}", "seed": len(d), "pairs": len(out.a),
                         "induced": out.n_induced, "rounds": out.rounds, "final": out.final,
                         "test_hits": hits, "seconds": time.perf_counter() - t0}}

    key = (digest("vm", a, b, d["a"].to_numpy(), d["b"].to_numpy(), va, vb,
                  json.dumps(sl, sort_keys=True)) if keep and init is None else None)
    return stored(ctx, key, run, ctx.store[a].device)


def vm_map(rec: dict, forward: bool) -> torch.Tensor:
    return side_map(rec["wx"], rec["wz"]) if forward else side_map(rec["wz"], rec["wx"])


def rc_fit(ctx: Ctx, a: str, b: str, d: pd.DataFrame, forward: bool, keep: bool) -> dict:
    """RCSLS on an edge dictionary, a -> b if forward, else b -> a; negatives: the two
    languages' whole vocabularies."""
    rc = XCFG["rcsls"]

    def run() -> dict:
        t0 = time.perf_counter()
        ea, eb = ctx.store[a], ctx.store[b]
        xa, xb = ea[s1._idx(d["a"], ea.device)], eb[s1._idx(d["b"], eb.device)]
        x, y, zs, zt = (xa, xb, ea, eb) if forward else (xb, xa, eb, ea)
        w = fit_rcsls(x, y, zs, zt, k=rc["k"], lr=rc["lr"], niter=rc["niter"],
                      block=s1.CFG["eval"]["block_rows"])
        return {"w": w, "info": {"edge": f"{a}__{b}", "forward": forward, "seed": len(d),
                                 "seconds": time.perf_counter() - t0}}

    key = (digest("rc", a, b, forward, d["a"].to_numpy(), d["b"].to_numpy(),
                  json.dumps(rc, sort_keys=True)) if keep else None)
    return stored(ctx, key, run, ctx.store[a].device)


# ---------------------------------------------------------------------------------------------
# Evaluation

def evaluate_fold(ctx: Ctx, s: str, t: str, x: str, extra: list[str], fold: int,
                  methods: list[tuple[str, str]]) -> int:
    """Every method of one pair and fold, both directions; returns the maps evaluated."""
    todo = {fw: [m for m in methods if s1.key(*((s, t) if fw else (t, s)), m, fold)
                 not in ctx.cache] for fw in (True, False)}
    if not any(todo.values()):
        return 0
    names = [n for p, n in methods if p == "vm"]
    d, ts, tt, sizes = fold_dicts(ctx, s, t, x, extra if "chain2" in names else [], fold)
    hubs = {"": x, "2": extra[0] if extra else None, "3": extra[1] if len(extra) > 1 else None}
    fits: dict[str, dict] = {}

    def vm(name: str) -> dict:
        if name not in fits:
            if name in ("direct", "joined"):
                fits[name] = vm_fit(ctx, s, t, d[name], ts, tt, reference=name == "direct"
                                    and ("vm", "thin_shared") in methods)
            elif name == "seeded":
                fits[name] = vm_fit(ctx, s, t, d["direct"], ts, tt, init=chain("", True))
            elif name.startswith("sx"):
                fits[name] = vm_fit(ctx, s, hubs[name[2:]], d[name], ts, set(), keep=True)
            else:
                fits[name] = vm_fit(ctx, hubs[name[2:]], t, d[name], set(), tt, keep=True)
        return fits[name]

    def chain(i: str, forward: bool) -> torch.Tensor:
        first, second = vm(f"sx{i}"), vm(f"xt{i}")
        if forward:
            return compose([vm_map(first, True), vm_map(second, True)])
        return compose([vm_map(second, False), vm_map(first, False)])

    done = 0
    for forward in (True, False):
        if todo[forward]:
            done += evaluate_direction(ctx, s, t, x, fold, forward, d, sizes, methods,
                                       todo[forward], vm, chain, hubs)
    return done


def finite(maps: dict, src: str, tgt: str, fold: int) -> None:
    """Refuse non-finite maps: a NaN map ranks every gold first (row > NaN is all False)."""
    bad = [n for n, w in maps.items() if not torch.isfinite(w).all()]
    if bad:
        raise FloatingPointError(f"{src}>{tgt} fold {fold}: non-finite maps {bad}")


def evaluate_direction(ctx: Ctx, s: str, t: str, x: str, fold: int, forward: bool, d: dict,
                       sizes: dict, methods: list, todo: list, vm, chain, hubs) -> int:
    src, tgt = (s, t) if forward else (t, s)
    es, et = ctx.store[s], ctx.store[t]
    e_src, e_tgt = (es, et) if forward else (et, es)
    concepts = ctx.folds[s, t][fold]
    queries = np.array([ctx.query[src, c] for c in concepts])
    golds = [[ctx.gold[src, tgt, kind, c] for c in concepts] for kind in ("main", "list")]
    ev = {"k": s1.EVAL["csls_k"], "block": s1.CFG["eval"]["block_rows"], "half": s1.EVAL["half"]}
    done = 0

    def put(method, ranks, seconds, fit_info):
        ctx.cache.put(s1.key(src, tgt, method, fold),
                      {"concepts": concepts, "rank": ranks[0].tolist(),
                       "rank_list": ranks[1].tolist(), "n_train": sizes, "seconds": seconds,
                       "fits": fit_info})

    if any(m[0] == "vm" for m in todo):
        names = [n for p, n in methods if p == "vm"]
        t0 = time.perf_counter()
        maps = {"thin": vm_map(vm("direct"), forward), "chain": chain("", forward)}
        used = ["direct", "sx", "xt"]
        if "joined" in names:
            maps["joined"] = vm_map(vm("joined"), forward)
            used.append("joined")
        if "chain2" in names:
            maps["chain2"], maps["chain3"] = chain("2", forward), chain("3", forward)
            used += ["sx2", "xt2", "sx3", "xt3"]
        if "seeded" in names:
            maps["seeded"] = vm_map(vm("seeded"), forward)
            used.append("seeded")
        combos = {"combo": ("thin", "chain"), **{n: {"thin": 1 - float(n[1:]),
                                                     "chain": float(n[1:])}
                                                 for n in names if n.startswith("w")}}
        if "hubs3" in names:
            combos["hubs3"] = ("thin", "chain", "chain2", "chain3")
        finite(maps, src, tgt, fold)
        out = bli_with_combinations(e_src, e_tgt, maps, combos, queries, golds, **ev)
        if "thin_shared" in names:
            ref = vm("direct")["ref"]       # None (too few distinct words to whiten): thin itself
            if ref is None:
                out["thin_shared"] = out["thin"]
            else:
                wx, wz = ref if forward else ref[::-1]
                finite({"wx": wx, "wz": wz}, src, tgt, fold)
                out["thin_shared"] = bli_multi(e_src, e_tgt @ wz, wx, queries, golds,
                                               retrieval="csls", **ev)
        dt = (time.perf_counter() - t0) / len(maps)
        info = {name: vm(name)["info"] for name in used}
        for n in names:
            put(("vm", n), out[n], dt if n in maps else 0.0, info)
        done += len(maps)
    if any(m[0] == "proc" for m in todo):
        t0 = time.perf_counter()
        ex = ctx.store[x]
        if forward:
            w_chain = compose([s1.edge_map(d["sx"], es, ex, True),
                               s1.edge_map(d["xt"], ex, et, True)])
        else:
            w_chain = compose([s1.edge_map(d["xt"], ex, et, False),
                               s1.edge_map(d["sx"], es, ex, False)])
        out = bli_with_combination(e_src, e_tgt, s1.edge_map(d["direct"], es, et, forward),
                                   w_chain, queries, golds, **ev)
        dt = (time.perf_counter() - t0) / 2
        for n, k in (("thin", "a"), ("chain", "b"), ("combo", "combo")):
            put(("proc", n), out[k], dt if n != "combo" else 0.0, {})
        done += 2
    if any(m[0] == "rc" for m in todo):
        t0 = time.perf_counter()
        direct = rc_fit(ctx, s, t, d["direct"], forward, keep=False)
        first, second = (rc_fit(ctx, s, x, d["sx"], forward, keep=True),
                         rc_fit(ctx, x, t, d["xt"], forward, keep=True))
        w_chain = compose([first["w"], second["w"]] if forward else [second["w"], first["w"]])
        finite({"rc thin": direct["w"], "rc chain": w_chain}, src, tgt, fold)
        out = bli_with_combination(e_src, e_tgt, direct["w"], w_chain, queries, golds, **ev)
        dt = (time.perf_counter() - t0) / 2
        info = {"direct": direct["info"], "sx": first["info"], "xt": second["info"]}
        for n, k in (("thin", "a"), ("chain", "b"), ("combo", "combo")):
            put(("rc", n), out[k], dt if n != "combo" else 0.0, info)
        done += 2
    return done


def unit_log(unit: str) -> Path:
    """A unit's run.log, beside its cache (configure's folder)."""
    return s1.CFG.path("results") / NAME / VERSION / unit / "run.log"


def run_unit(ctx: Ctx, pairs: pd.DataFrame, methods: list[tuple[str, str]]) -> None:
    log = unit_log(ctx.unit)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"start commit {s1.git('rev-parse', 'HEAD').strip()}\n")
    start, total = time.time(), 0
    for i, r in enumerate(pairs.itertuples()):
        for fold in sorted(ctx.folds[r.s, r.t]):
            total += evaluate_fold(ctx, r.s, r.t, r.hub, [r.hub2, r.hub3], fold, methods)
        el = time.time() - start
        msg = (f"{time.strftime('%H:%M:%S')} {ctx.unit} pair {i + 1}/{len(pairs)} {r.s}-{r.t} "
               f"via {r.hub}: {total} maps, {el / 60:.1f} min, {el / max(total, 1):.2f} s each")
        print(msg, flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


# ---------------------------------------------------------------------------------------------
# Results

def unit_rows(ctx: Ctx, pairs: pd.DataFrame, methods: list[tuple[str, str]]
              ) -> tuple[list[dict], list[dict]]:
    """Per direction and method with every fold cached: MRR, P@1 and the change against its
    pipeline's thin (vm methods against vm thin), with the paired bootstrap over test words; and
    the fits' records (one row per fit and fold)."""
    bs, rows, fits = s1.XCFG["bootstrap"], [], []
    for r in pairs.itertuples():
        folds = sorted(ctx.folds[r.s, r.t])
        for src, tgt in ((r.s, r.t), (r.t, r.s)):
            got = {m: p for m in methods
                   if (p := s1.pooled(ctx.cache, src, tgt, m, folds)) is not None}
            for m, p in got.items():
                base = got.get((m[0], "thin"))
                if base is None:
                    continue
                mean, lo, hi = paired_bootstrap(p["rr"].to_numpy(), base["rr"].to_numpy(), **bs)
                rows.append({"unit": ctx.unit, "scenario": ctx.scenario, "budget": ctx.budget,
                             "src": src, "tgt": tgt, "s": r.s, "method": f"{m[0]}_{m[1]}",
                             "mrr": p["rr"].mean(), "p1": (p["rr"] == 1).mean(),
                             "mrr_list": p["rr_list"].mean(), "mrr_thin": base["rr"].mean(),
                             "delta": mean, "lo": lo, "hi": hi, "n": len(p)})
            for m in (("vm", "thin"), ("rc", "thin")):      # each record lists every fit used
                if m not in methods:
                    continue
                for f in folds:
                    rec = ctx.cache.get(s1.key(src, tgt, m, f))
                    for name, info in ((rec or {}).get("fits") or {}).items():
                        fits.append({"unit": ctx.unit, "src": src, "tgt": tgt, "fold": f,
                                     "pipeline": m[0], "fit": name, **info})
    return rows, fits


def pipeline_rows(pooled: pd.DataFrame, ctxs: dict, bs: dict) -> pd.DataFrame:
    """Strong against standard per direction, paired per test word: vm and rc against proc for
    thin, chain and combo (pilot pairs only, where proc runs)."""
    rows = []
    for (unit, src, tgt), _ in pooled.groupby(["unit", "src", "tgt"], sort=False):
        ctx = ctxs[unit]
        s, t = next((s, t) for s, t in zip(ctx.all_pairs["s"], ctx.all_pairs["t"], strict=True)
                    if {s, t} == {src, tgt})
        folds = sorted(ctx.folds[s, t])
        for n in ("thin", "chain", "combo"):
            ref = s1.pooled(ctx.cache, src, tgt, ("proc", n), folds)
            for p in ("vm", "rc"):
                got = s1.pooled(ctx.cache, src, tgt, (p, n), folds)
                if ref is None or got is None:
                    continue
                mean, lo, hi = paired_bootstrap(got["rr"].to_numpy(), ref["rr"].to_numpy(), **bs)
                rows.append({"unit": unit, "scenario": ctx.scenario, "src": src, "tgt": tgt,
                             "what": f"{p}_{n} - proc_{n}", "mrr": got["rr"].mean(),
                             "mrr_proc": ref["rr"].mean(), "delta": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def summarize(pooled: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group and method: directions, mean MRR, mean change against thin over directions and
    over languages, directions above / below 0."""
    return (pooled.groupby([*by, "method"], sort=False, dropna=False)
            .apply(lambda g: pd.Series({
                "n_dirs": len(g), "mrr": g["mrr"].mean(), "mrr_thin": g["mrr_thin"].mean(),
                "mean_delta": g["delta"].mean(),
                "mean_over_languages": g.groupby("s")["delta"].mean().mean(),
                "n_above": int((g["lo"] > 0).sum()), "n_below": int((g["hi"] < 0).sum())}),
                include_groups=False)
            .reset_index())


def scenario_mean(pooled: pd.DataFrame, method: str) -> float:
    """Mean change against vm thin, each scenario weighing equally (the budgets pooled as one)."""
    g = pooled[pooled["method"] == method]
    return float(g.groupby("scenario")["delta"].mean().mean()) if len(g) else float("nan")


def choose(pooled: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """README selection: the weighted combination's w on these pairs (first best, smaller w on
    ties), then the candidate with the highest scenario-weighted mean change against strong
    direct; candidates within `tie` of the best go to the simplest (config order)."""
    w_scores = {w: scenario_mean(pooled, f"vm_w{w}") for w in XCFG["weights"]}
    w = max(w_scores, key=lambda k: (w_scores[k], -k))
    method = {"chain": "vm_chain", "combo": "vm_combo", "wcombo": f"vm_w{w}",
              "hubs3": "vm_hubs3", "seeded": "vm_seeded"}
    score = {c: scenario_mean(pooled, method[c]) for c in XCFG["candidates"]}
    have = {c: v for c, v in score.items() if not np.isnan(v)}       # partial runs
    best = max(have.values())
    chosen = next(c for c in XCFG["candidates"] if c in have and have[c] >= best - XCFG["tie"])
    table = pd.DataFrame([{"candidate": c, "method": method[c], "score": score[c],
                           **{f"scenario_{sc}": g["delta"].mean() for sc, g in
                              pooled[pooled["method"] == method[c]].groupby("scenario")}}
                          for c in XCFG["candidates"]])
    return {"candidate": chosen, "w": w, "scores": score, "w_scores": w_scores}, table


def weight_cv(pooled: pd.DataFrame) -> pd.DataFrame:
    """vm_wcv rows: each direction gets vm_w<w> with w chosen on the other folds of pairs
    (XCFG weight_folds, seed 0; a pair's stage D and budget cells share a fold)."""
    groups = sorted({" ".join(sorted(p)) for p in zip(pooled["src"], pooled["tgt"], strict=True)})
    rng = np.random.default_rng(XCFG["seed"])
    fold_of = {g: i % XCFG["weight_folds"] for i, g in enumerate(rng.permutation(groups))}
    pair = pd.Series([fold_of[" ".join(sorted(p))]
                      for p in zip(pooled["src"], pooled["tgt"], strict=True)], index=pooled.index)
    out = []
    for f in range(XCFG["weight_folds"]):
        train = pooled[pair != f]
        scores = {w: scenario_mean(train, f"vm_w{w}") for w in XCFG["weights"]}
        w = max(scores, key=lambda k: (scores[k], -k))
        out.append(pooled[(pair == f) & (pooled["method"] == f"vm_w{w}")].assign(
            method="vm_wcv", w=w))
    return pd.concat(out, ignore_index=True)


def standard_rows(pooled: pd.DataFrame) -> pd.DataFrame:
    """The standard pipeline's committed per-direction MRR (the stages' pooled.csv) next to the
    strong one, per unit and method (thin, chain, combo, joined)."""
    rows = []
    std_name = {"thin": "thin", "chain": "chain", "combo": "combo", "joined": "joined_mb"}
    for unit in pooled["unit"].unique():
        sc, b = scenario_of(unit)
        exp = HERE.parent / XCFG["scenarios"][sc]["experiment"] / "results"
        path = exp / f"b{b}" / "pooled.csv" if b is not None else exp / "pooled.csv"
        if not path.exists():
            continue
        std = pd.read_csv(path)
        thin = std.groupby(["src", "tgt"])["mrr_thin"].first()
        mine = pooled[pooled["unit"] == unit]
        for n, sn in std_name.items():
            got = mine[mine["method"] == f"vm_{n}"].set_index(["src", "tgt"])["mrr"]
            ref = thin if n == "thin" else std[std["method"] == sn].set_index(["src", "tgt"])["mrr"]
            both = pd.concat({"strong": got, "standard": ref}, axis=1).dropna()
            if len(both):
                rows.append({"unit": unit, "scenario": sc, "method": n, "n_dirs": len(both),
                             "standard": both["standard"].mean(), "strong": both["strong"].mean(),
                             "change": (both["strong"] - both["standard"]).mean(),
                             "n_up": int((both["strong"] > both["standard"]).sum())})
    return pd.DataFrame(rows)


def run_info(ctxs: dict, mode: str) -> dict:
    files = {}
    for ctx in ctxs.values():
        files.update({str(f.relative_to(s1.CFG.data_root)): f for f in needed_files(ctx)
                      if s1.CFG.path("v0_embeddings") not in f.parents})
    logs = [unit_log(u) for u in ctxs]
    commits = [c for log in logs if log.exists()
               for c in s1.eval_commits(log.read_text(encoding="utf-8"))]
    return {"mode": mode, "commit": s1.git("rev-parse", "HEAD").strip(),
            "eval_commits": list(dict.fromkeys(commits)), "config": XCFG,
            "eval": s1.EVAL, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
            else None, "n_records": {u: sum(1 for _ in c.cache) for u, c in ctxs.items()},
            "inputs": file_fingerprints(dict(sorted(files.items())))}


def write_results(ctxs: dict, chosen_pairs: dict, mode: str, methods: dict) -> None:
    out = RESULTS / mode
    out.mkdir(parents=True, exist_ok=True)
    rows, fits = [], []
    for unit, ctx in ctxs.items():
        r, f = unit_rows(ctx, chosen_pairs[unit], methods[unit])
        rows += r
        fits += f
    if not rows:
        print("no complete directions yet")
        return
    pooled = pd.DataFrame(rows)
    if mode == "full" and pooled["method"].str.startswith("vm_w").any():
        pooled = pd.concat([pooled, weight_cv(pooled)], ignore_index=True)
    pooled.to_csv(out / "pooled.csv", **KW)
    pd.DataFrame(fits).to_csv(out / "fits.csv", **KW)
    summarize(pooled, ["unit"]).to_csv(out / "summary.csv", **KW)
    summarize(pooled, ["scenario"]).to_csv(out / "summary_scenarios.csv", **KW)
    bs = s1.XCFG["bootstrap"]
    if mode == "pilot":
        pipes = pipeline_rows(pooled, ctxs, bs)
        pipes.to_csv(out / "pipelines.csv", **KW)
        choice, table = choose(pooled)
        table.to_csv(out / "candidates.csv", **KW)
        go = {p: float((pipes[pipes["what"] == f"{p}_thin - proc_thin"]["delta"]).mean())
              for p in ("vm", "rc")}
        choice["go"] = {p: bool(v > 0) for p, v in go.items()}
        choice["strong_minus_standard_direct"] = go
        (out / "choice.json").write_text(json.dumps(choice, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(choice, indent=1))
    else:
        standard_rows(pooled).to_csv(out / "standard.csv", **KW)
        pilot = pd.read_csv(HERE / "pilot_pairs.csv")
        in_pilot = {(u, *sorted((s, t))) for u, s, t in zip(pilot["unit"], pilot["s"], pilot["t"],
                                                             strict=True)}
        keep = [(u, *sorted((a, b))) not in in_pilot
                for u, a, b in zip(pooled["unit"], pooled["src"], pooled["tgt"], strict=True)]
        summarize(pooled[keep], ["scenario"]).to_csv(out / "summary_without_pilot.csv", **KW)
    (out / "run_info.json").write_text(json.dumps(run_info(ctxs, mode), indent=1) + "\n",
                                       encoding="utf-8")
    print(f"results written to {out}")


# ---------------------------------------------------------------------------------------------
# Inputs and the reproduction check

def needed_files(ctx: Ctx) -> list[Path]:
    """Every input a unit reads, absolute: the route languages' vectors and vocabularies, the
    unit's folds, gold, test words and pairs, each pair's direct, hub, joined and extra-hub
    dictionaries, and hubs.csv."""
    p6, pairs = ctx.p6, ctx.all_pairs
    langs = set(pairs["s"]) | set(pairs["t"]) | set(pairs["hub"]) | set(pairs["hub2"]) | set(
        pairs["hub3"])
    emb = s1.CFG.path("v0_embeddings")
    files = [emb / f"{lang}{ext}" for lang in langs for ext in (".npy", ".vocab.txt")]
    files += [p6 / f for f in ("folds.parquet", "gold.parquet", "words.parquet")]
    if ctx.budget is not None:
        files.append(p6 / "pairs.csv")
    files.append(ctx.hub_dir.parent / "hubs.csv")
    for s, t, x, x2, x3 in zip(pairs["s"], pairs["t"], pairs["hub"], pairs["hub2"],
                               pairs["hub3"], strict=True):
        files.append(p6 / "direct" / f"{s}__{t}.parquet" if ctx.budget is not None
                     else p6 / "dicts" / f"{s}__{t}.parquet")
        files += [p6 / "dicts" / f"{s}__{x}.parquet", p6 / "dicts" / f"{x}__{t}.parquet"]
        if ctx.sc["joined"]:
            files.append(p6 / "joined" / f"{s}__{x}__{t}.parquet")
        files += [ctx.hub_dir / f"{a}__{b}.parquet" for h in (x2, x3)
                  for a, b in ((s, h), (h, t))]
    return sorted(set(files))


def stage_cache(unit: str) -> Path:
    sc, b = scenario_of(unit)
    exp = XCFG["scenarios"][sc]["experiment"]
    return s1.CFG.path("results") / (f"{exp}/b{b}" if b is not None else exp) / "evals.jsonl"


def reproduction_check(ctx: Ctx, pairs: pd.DataFrame) -> pd.DataFrame:
    """proc thin, chain and combo against the stage's own cache: records compared, share of
    queries with the same rank, largest rank difference. Empty without the stage's cache."""
    path = stage_cache(ctx.unit)
    if not path.exists():
        return pd.DataFrame()
    theirs, stats = Cache(path), {}
    for r in pairs.itertuples():
        for src, tgt in ((r.s, r.t), (r.t, r.s)):
            for n, old in (("thin", ("thin",)), ("chain", ("chain", r.hub)),
                           ("combo", ("combo", r.hub))):
                for f in sorted(ctx.folds[r.s, r.t]):
                    a = ctx.cache.get(s1.key(src, tgt, ("proc", n), f))
                    b = theirs.get(s1.key(src, tgt, old, f))
                    if a is None or b is None:
                        continue
                    a, b = np.asarray(a["rank"]), np.asarray(b["rank"])
                    stats.setdefault(n, []).append((len(a), int((a == b).sum()),
                                                    int(np.abs(a - b).max())))
    return pd.DataFrame([{"unit": ctx.unit, "method": n, "records": len(v),
                          "same_rank_share": sum(x[1] for x in v) / sum(x[0] for x in v),
                          "max_rank_diff": max(x[2] for x in v)} for n, v in stats.items()])


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--make-pilot", action="store_true", help="write pilot_pairs.csv")
    mode.add_argument("--pilot", action="store_true", help="run the pilot pairs, every method")
    mode.add_argument("--full", action="store_true", help="run every pair, the chosen candidate")
    ap.add_argument("--units", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs per unit")
    ap.add_argument("--results-only", action="store_true", help="rebuild results from caches")
    ap.add_argument("--data-files", action="store_true", help="list inputs and exit")
    ap.add_argument("--check", action="store_true", help="proc against the stages' caches")
    ap.add_argument("--pipelines", default=None,
                    help="run only these pipelines now, e.g. vm,proc (default: all)")
    args = ap.parse_args()
    if args.make_pilot:
        make_pilot().to_csv(HERE / "pilot_pairs.csv", index=False, lineterminator="\n")
        print(pd.read_csv(HERE / "pilot_pairs.csv").groupby("unit").size().to_string())
        return
    run_mode = "full" if args.full else "pilot"
    choice = None
    if run_mode == "full":
        choice = json.loads((RESULTS / "pilot" / "choice.json").read_text(encoding="utf-8"))
        choice = choice["candidate"]
    pilot = pd.read_csv(HERE / "pilot_pairs.csv")
    todo = units() if args.units is None else args.units.split(",")
    ctxs, chosen, methods, paths = {}, {}, {}, set()
    for unit in todo:
        configure(unit)
        ctx = Ctx(unit)
        if run_mode == "pilot":
            mine = pilot[pilot["unit"] == unit][["s", "t"]]
            pairs = ctx.all_pairs.merge(mine, on=["s", "t"])
        else:
            pairs = ctx.all_pairs
        if args.limit is not None:
            pairs = pairs.head(args.limit)
        if pairs.empty:
            continue
        ctxs[unit], chosen[unit] = ctx, pairs
        methods[unit] = methods_for(ctx.sc, run_mode, choice)
        paths |= set(s1.RESULT_PATHS)
    if args.data_files:
        files = {f for ctx in ctxs.values() for f in needed_files(ctx)}
        for f in sorted(files):
            print(f.relative_to(s1.CFG.data_root).as_posix())
        return
    if args.check:
        out = pd.concat([reproduction_check(ctxs[u], chosen[u]) for u in ctxs],
                        ignore_index=True)
        out.to_csv(RESULTS / "check.csv", **KW)
        print(out.to_string(index=False))
        return
    s1.RESULT_PATHS = tuple(sorted(paths))        # every unit's inputs, not only the last one's
    s1.check_clean(s1.git("status", "--porcelain").splitlines())
    if not args.results_only:
        keep = None if args.pipelines is None else set(args.pipelines.split(","))
        for unit, ctx in ctxs.items():
            configure(unit)
            now = [m for m in methods[unit] if keep is None or m[0] in keep]
            print(f"unit {unit}: {len(chosen[unit])} pairs, {len(now)} methods", flush=True)
            run_unit(ctx, chosen[unit], now)
    write_results(ctxs, chosen, run_mode, methods)


if __name__ == "__main__":
    main()
