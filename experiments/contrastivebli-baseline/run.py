"""A published competitor on our data: ContrastiveBLI stage C1 (Li et al., ACL 2022) against our
direct, chain and several hubs. See README.md (approved before running).

Per pair (config groups: strong-maps units; a seed-0 draw without strong maps' pilot pairs) and
fold: C1 is trained on the fold's direct dictionary, test words removed, with the authors' code
imported unchanged and main.py's training loop run here without its logging evaluation. The
authors' preprocessing (raw vectors, unit-normalized by their code); our evaluation (CSLS, k = 10,
both gold sets) on C1's two maps, and the fold's test words dropped from C1's induced pairs
("rule"; "published" leaves them, where the group asks for both). Plain Procrustes direct is
refitted as a check against the stage's own cache. Ours is read from two caches: strong maps'
(self-learned direct, chain, several hubs) and several-hubs-procrustes' (plain Procrustes chain,
combination, several hubs), where each has the group. One fit per pair serves both directions.
Each group has its own cache and maps (<results>/<name>/<group>/): groups may share pairs.
`--make-pairs` writes pairs.csv; the default runs every pair; `--results-only` builds results/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from clwe_chains.dictionaries import remove_test
from clwe_chains.evaluate import bli_multi
from clwe_chains.runs import Cache
from clwe_chains.stats import paired_bootstrap

HERE = Path(__file__).resolve().parent
NAME = HERE.name
RESULTS = HERE / "results"
XCFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    f"{NAME}_sm_run", HERE.parent / "strong-maps" / "run.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)
s1 = sm.s1
KW = {"index": False, "lineterminator": "\n", "float_format": "%.6g"}
# the comparisons (a minus b), our method first; "published" only where the group runs it
COMPARE = [("vm_hubs3", "c1_rule"), ("vm_chain", "c1_rule"),
           ("proc_hubs3", "c1_rule"), ("proc_chain", "c1_rule"), ("proc_combo", "c1_rule"),
           ("c1_rule", "vm_thin"), ("c1_rule", "proc_thin"), ("c1_published", "c1_rule")]
OURS = {"strong-maps": [("vm", "thin"), ("vm", "chain"), ("vm", "hubs3")],
        "several-hubs-procrustes": [("proc", "chain"), ("proc", "combo"), ("proc", "hubs3")]}


# ---------------------------------------------------------------------------------------------
# Pairs, dictionaries, the induced-pair rule

def draw_pairs(pairs: pd.DataFrame, pilot: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """The first n of a seed-`seed` permutation of the pairs (sorted by s, t) that are not pilot
    pairs (either orientation); more pairs later extend the same order."""
    skip = {tuple(sorted(p)) for p in zip(pilot["s"], pilot["t"], strict=True)}
    left = (pairs[[tuple(sorted(p)) not in skip for p in zip(pairs["s"], pairs["t"], strict=True)]]
            .sort_values(["s", "t"], ignore_index=True))
    order = np.random.default_rng(seed).permutation(len(left))[:n]
    return left.iloc[order][["s", "t"]].reset_index(drop=True).assign(order=range(len(order)))


def shared_pairs(drawn: pd.DataFrame, have: pd.DataFrame) -> pd.DataFrame:
    """Another group's drawn pairs (`pairs_from`) that this group's data has, in their order."""
    ok = {tuple(sorted(p)) for p in zip(have["s"], have["t"], strict=True)}
    keep = [tuple(sorted(p)) in ok for p in zip(drawn["s"], drawn["t"], strict=True)]
    return drawn[keep][["s", "t", "order"]].reset_index(drop=True)


def direct_name(ctx, s: str, t: str) -> str:
    """The direct dictionary: a budget folder's cut one (controlled scarcity) or the stage's."""
    return f"direct:{s}__{t}" if ctx.budget is not None else f"{s}__{t}"


def cap_pairs(d: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    """The cap most frequent pairs (smallest larger row number; ties in file order)."""
    if not cap or len(d) <= cap:
        return d
    order = np.maximum(d["a"].to_numpy(), d["b"].to_numpy()).argsort(kind="stable")
    return d.iloc[order[:cap]].reset_index(drop=True)


def drop_test(aug1: list[int], aug2: list[int], excl1: set[int], excl2: set[int]
              ) -> tuple[list[int], list[int], int]:
    """Induced pairs without a test row on their side; and how many were dropped."""
    keep = [(a, b) for a, b in zip(aug1, aug2, strict=True) if a not in excl1 and b not in excl2]
    return [a for a, _ in keep], [b for _, b in keep], len(aug1) - len(keep)


# ---------------------------------------------------------------------------------------------
# The authors' code and its training loop

def c1_root() -> Path:
    return s1.CFG.data_root / XCFG["c1_repo"]


def load_c1() -> dict:
    """The authors' C1 modules, unchanged. They import each other by name ("from util import
    *"), so each is put in sys.modules first; their folder never goes on sys.path."""
    src = c1_root() / "C1" / "src"
    mods = {}
    for m, name in (("util", "util"), ("models", "models"), ("dataloader", "dataloader"),
                    ("forward", "forward"), ("main", "c1main")):
        spec = importlib.util.spec_from_file_location(name, src / f"{m}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods[m] = mod
    return mods


def c1_commit() -> str:
    return subprocess.run(["git", "-C", str(c1_root()), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def c1_args(settings: str) -> argparse.Namespace:
    """main.py's defaults, overwritten by its settings for the seed size."""
    a = argparse.Namespace(**XCFG["fixed"], **XCFG["settings"][settings], train_size=settings,
                           device=torch.device("cuda:0"), sup_batch_size=0, mini_batch_size=0)
    return a


def split_step(c1: dict, model, args, loss_dict, src: torch.Tensor, tgt: torch.Tensor,
               mini: int) -> None:
    """One batch's gradient, accumulated over parts of `mini` rows, each part's loss weighted by
    its share: the full batch's gradient with less memory."""
    n = len(src)
    step = mini or n
    for i in range(0, n, step):
        part = (src[i:i + step].cuda(), tgt[i:i + step].cuda())
        loss = c1["forward"].forward_joint(part, model, loss_dict, args, mode="train")
        (loss * (len(part[0]) / n)).backward()


def train_c1(c1: dict, args, embs1: np.ndarray, embs2: np.ndarray, sup1: list[int],
             sup2: list[int], mini: int, excl: tuple[set, set] | None) -> dict:
    """main.py's self-learning loop (lines 479-589) without its logging evaluation; excl: the
    test rows dropped from each round's induced pairs (None: as published)."""
    U, C, D = c1["util"], c1["main"], c1["dataloader"]
    data1, data2 = torch.from_numpy(embs1.copy()), torch.from_numpy(embs2.copy())
    model = c1["models"].C1_Model(args).cuda()
    params = list(model.parameters())
    opt = torch.optim.SGD(params, lr=args.lr)
    loss_dict = U.get_log_loss_dict_()
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.gamma)
    rounds = []
    for it in range(args.num_sl):
        t0 = time.perf_counter()
        aug1, aug2, dropped = [], [], 0
        if it > 0:
            with torch.no_grad():
                model.eval()
                aug1, aug2 = C.high_conf_pairs(model, args, data1, data2, sup1, sup2)
            model.train()
            if excl is not None:
                aug1, aug2, dropped = drop_test(aug1, aug2, *excl)
        cur1, cur2 = sup1 + aug1, sup2 + aug2
        w1, w2 = U.AdvancedMapping(embs1.copy()[cur1, :], embs2.copy()[cur2, :])
        model.load_state_dict({"s2m_mapping.weight": torch.from_numpy(w1).float().cuda(),
                               "t2m_mapping.weight": torch.from_numpy(w2).float().cuda()})
        if args.train_size == "5k":            # 5k: the induced pairs only seed the init
            cur1, cur2 = sup1, sup2
        args.sup_batch_size = args.mini_batch_size = len(cur1)
        opt.param_groups[0]["lr"] = args.lr
        torch.cuda.reset_peak_memory_stats()
        for _epoch in range(args.num_steps + 1):
            with torch.no_grad():
                model.eval()
                cs, ct = C.neg_sample(model, args, data1, data2, cur1, cur2)
            model.train()
            src, tgt = D.next_batch_joint_sup_confuse(cs, ct, data1, data2, args.sup_batch_size,
                                                      args)
            opt.zero_grad()
            split_step(c1, model, args, loss_dict, src, tgt, mini)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            sched.step()
        rounds.append({"round": it, "train_pairs": len(cur1), "induced": len(aug1),
                       "dropped_test": dropped, "seconds": round(time.perf_counter() - t0, 1),
                       "peak_mib": int(torch.cuda.max_memory_allocated() // 2**20)})
    return {"w_s": model.s2m_mapping.weight.detach().float().cpu(),
            "w_t": model.t2m_mapping.weight.detach().float().cpu(), "rounds": rounds}


# ---------------------------------------------------------------------------------------------
# Running

def results_root() -> Path:
    return s1.CFG.path("results") / NAME


class RawVectors:
    """The raw fastText vectors, made unit length by the authors' normed_input, kept on the CPU
    (C1 copies batches to the GPU itself); the last `keep` languages stay loaded."""

    def __init__(self, norm, keep: int = 4):
        self.norm, self.keep, self._have = norm, keep, {}

    def __getitem__(self, lang: str) -> np.ndarray:
        if lang not in self._have:
            arr = np.load(s1.CFG.path("v0_embeddings") / f"{lang}.npy").astype(np.float32)
            self._have[lang] = self.norm(arr).astype(np.float32)
            while len(self._have) > self.keep:
                self._have.pop(next(iter(self._have)))
        return self._have[lang]


def contexts(units: list[str]) -> dict:
    """Strong maps' context per group (pairs, folds, test words, dictionaries, vectors)."""
    out = {}
    for u in units:
        sm.configure(u)
        out[u] = sm.Ctx(u)
    return out


def group_root(group: str) -> Path:
    """A group's cache and maps (groups may share pairs, so never one folder for all)."""
    return results_root() / group


def ours_cache(source: str, group: str) -> Cache:
    """Our method's records for the group: strong maps' (by its version) or several hubs with
    plain Procrustes'."""
    root = s1.CFG.path("results") / source
    return Cache((root / sm.VERSION if source == "strong-maps" else root) / group / "evals.jsonl")


def variants(group: str, order: int) -> list[str]:
    """"rule" always; "published" too for the group's first `published_pairs` pairs."""
    return ["rule"] + (["published"] if order < XCFG["groups"][group].get("published_pairs", 0)
                       else [])


def fit_pair(c1: dict, ctx, raw: RawVectors, s: str, t: str, fold: int, variant: str,
             log) -> dict:
    """One pair's C1 fit for the fold, kept on disk (atomic) and reused."""
    path = group_root(ctx.unit) / "maps" / f"{s}__{t}__f{fold}__{variant}.pt"
    if path.exists():
        return torch.load(path)
    g = XCFG["groups"][ctx.unit]
    test = ctx.folds[s, t][fold]
    ts, tt = ctx.test_rows(s, test), ctx.test_rows(t, test)
    direct = remove_test(ctx.dictionary(direct_name(ctx, s, t)), ts, tt)
    seed = cap_pairs(direct, g.get("cap"))
    torch.manual_seed(XCFG["torch_seed"])
    t0 = time.perf_counter()
    out = train_c1(c1, c1_args(g["settings"]), raw[s], raw[t], seed["a"].astype(int).tolist(),
                   seed["b"].astype(int).tolist(), g["mini"],
                   (ts, tt) if variant == "rule" else None)
    out["info"] = {"s": s, "t": t, "fold": fold, "variant": variant, "settings": g["settings"],
                   "direct_pairs": len(direct), "seed_pairs": len(seed),
                   "seconds": round(time.perf_counter() - t0, 1), "rounds": out.pop("rounds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(out, tmp)
    tmp.replace(path)
    log(f"{s}-{t} f{fold} {variant}: {out['info']['seconds'] / 60:.1f} min, "
        f"rounds {[r['train_pairs'] for r in out['info']['rounds']]}")
    return out


def evaluate_pair(c1: dict, ctx, raw: RawVectors, cache: Cache, s: str, t: str, order: int,
                  fold: int, log) -> None:
    """C1 (each variant) and plain Procrustes, both directions, into the cache."""
    test = ctx.folds[s, t][fold]
    ev = {"k": s1.EVAL["csls_k"], "block": s1.CFG["eval"]["block_rows"], "half": s1.EVAL["half"]}
    fits = {v: fit_pair(c1, ctx, raw, s, t, fold, v, log) for v in variants(ctx.unit, order)
            if any(s1.key(a, b, ("c1", v), fold) not in cache for a, b in ((s, t), (t, s)))}
    es, et = ctx.store[s], ctx.store[t]          # on the GPU only after the fits
    maps = {}
    for v, f in fits.items():
        ms = torch.from_numpy(raw[s]).to(es.device) @ f["w_s"].to(es.device).T
        mt = torch.from_numpy(raw[t]).to(es.device) @ f["w_t"].to(es.device).T
        maps[v] = (ms, mt)
    direct = remove_test(ctx.dictionary(direct_name(ctx, s, t)), ctx.test_rows(s, test),
                         ctx.test_rows(t, test))
    for src, tgt, fw in ((s, t, True), (t, s, False)):
        q = np.array([ctx.query[src, c] for c in test])
        golds = [[ctx.gold[src, tgt, kind, c] for c in test] for kind in ("main", "list")]
        todo = {("c1", v): (m[0], m[1]) if fw else (m[1], m[0]) for v, m in maps.items()}
        for m, (a, b) in todo.items():
            if s1.key(src, tgt, m, fold) in cache:
                continue
            eye = torch.eye(a.shape[1], device=a.device)
            r = bli_multi(a, b, eye, q, golds, retrieval="csls", **ev)
            cache.put(s1.key(src, tgt, m, fold), {"concepts": test, "rank": r[0].tolist(),
                                                  "rank_list": r[1].tolist()})
        if s1.key(src, tgt, ("proc", "thin"), fold) not in cache:
            w = s1.edge_map(direct, es, et, fw)
            r = bli_multi(es if fw else et, et if fw else es, w, q, golds, retrieval="csls", **ev)
            cache.put(s1.key(src, tgt, ("proc", "thin"), fold),
                      {"concepts": test, "rank": r[0].tolist(), "rank_list": r[1].tolist()})


def run_all(pairs: pd.DataFrame) -> None:
    got = c1_commit()
    if got != XCFG["c1_commit"]:
        raise SystemExit(f"the authors' code is at {got}, not {XCFG['c1_commit']}")
    c1 = load_c1()
    ctxs = contexts(list(dict.fromkeys(pairs["group"])))
    raw = RawVectors(c1["main"].normed_input)   # the authors' preprocessing: unit length only
    root = results_root()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "run.log"
    caches = {}
    for g in ctxs:
        group_root(g).mkdir(parents=True, exist_ok=True)
        caches[g] = Cache(group_root(g) / "evals.jsonl")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"start commit {s1.git('rev-parse', 'HEAD').strip()}\n")
    start = time.time()
    for i, r in enumerate(pairs.itertuples()):
        evaluate_pair(c1, ctxs[r.group], raw, caches[r.group], r.s, r.t, r.order, XCFG["fold"],
                      log)
        log(f"pair {i + 1}/{len(pairs)} {r.group} {r.s}-{r.t}: "
            f"{(time.time() - start) / 60:.1f} min in all")


# ---------------------------------------------------------------------------------------------
# Results

def direction_scores(caches: dict, pairs: pd.DataFrame) -> dict:
    """(group, src, tgt) -> method -> the test words' reciprocal ranks (sorted by concept): C1 and
    plain Procrustes direct from this experiment's cache, ours from OURS' caches."""
    fold, out = XCFG["fold"], {}
    for r in pairs.itertuples():
        if r.group not in caches:
            continue
        sources = [(caches[r.group], [("c1", v) for v in variants(r.group, r.order)]
                    + [("proc", "thin")])]
        sources += [(ours_cache(src, r.group), ms) for src, ms in OURS.items()]
        for src, tgt in ((r.s, r.t), (r.t, r.s)):
            got = {}
            for cache, methods in sources:
                for m in methods:
                    p = s1.pooled(cache, src, tgt, m, [fold])
                    if p is not None:
                        got[f"{m[0]}_{m[1]}"] = p
            out[r.group, src, tgt] = got
    return out


def tables(scores: dict, bs: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per direction and method; per direction and comparison; per group and comparison (pooled
    over the group's test words, paired; directions above / below 0)."""
    per, cmp_rows = [], []
    for (g, src, tgt), got in scores.items():
        for m, p in got.items():
            per.append({"group": g, "src": src, "tgt": tgt, "method": m, "mrr": p["rr"].mean(),
                        "p1": (p["rr"] == 1).mean(), "mrr_list": p["rr_list"].mean(),
                        "n": len(p)})
        for a, b in COMPARE:
            if a in got and b in got:
                if not (got[a]["concept"].to_numpy() == got[b]["concept"].to_numpy()).all():
                    raise ValueError(f"{src}->{tgt}: {a} and {b} differ in test words")
                mean, lo, hi = paired_bootstrap(got[a]["rr"].to_numpy(), got[b]["rr"].to_numpy(),
                                                **bs)
                cmp_rows.append({"group": g, "src": src, "tgt": tgt, "a": a, "b": b,
                                 "mrr_a": got[a]["rr"].mean(), "mrr_b": got[b]["rr"].mean(),
                                 "delta": mean, "lo": lo, "hi": hi})
    groups = []
    for (g, a, b), d in pd.DataFrame(cmp_rows).groupby(["group", "a", "b"], sort=False):
        ra = np.concatenate([scores[g, s, t][a]["rr"].to_numpy() for s, t in zip(d["src"],
                             d["tgt"], strict=True)])
        rb = np.concatenate([scores[g, s, t][b]["rr"].to_numpy() for s, t in zip(d["src"],
                             d["tgt"], strict=True)])
        mean, lo, hi = paired_bootstrap(ra, rb, **bs)
        groups.append({"group": g, "a": a, "b": b, "n_dirs": len(d), "n_words": len(ra),
                       "mrr_a": ra.mean(), "mrr_b": rb.mean(), "delta": mean, "lo": lo, "hi": hi,
                       "n_above": int((d["lo"] > 0).sum()), "n_below": int((d["hi"] < 0).sum())})
    return pd.DataFrame(per), pd.DataFrame(cmp_rows), pd.DataFrame(groups)


def reproduction(caches: dict, pairs: pd.DataFrame) -> pd.DataFrame:
    """Plain Procrustes here against the stage's own cache (thin): records, same-rank share."""
    rows = []
    for g, cache in caches.items():
        path = sm.stage_cache(g)
        if not path.exists():
            continue
        theirs, n, same, diff, recs = Cache(path), 0, 0, 0, 0
        for r in pairs[pairs["group"] == g].itertuples():
            for src, tgt in ((r.s, r.t), (r.t, r.s)):
                a = cache.get(s1.key(src, tgt, ("proc", "thin"), XCFG["fold"]))
                b = theirs.get(s1.key(src, tgt, ("thin",), XCFG["fold"]))
                if a is None or b is None:
                    continue
                a, b = np.asarray(a["rank"]), np.asarray(b["rank"])
                n, same, recs = n + len(a), same + int((a == b).sum()), recs + 1
                diff = max(diff, int(np.abs(a - b).max()))
        if recs:
            rows.append({"group": g, "records": recs, "same_rank_share": same / n,
                         "max_rank_diff": diff})
    return pd.DataFrame(rows)


def write_results(pairs: pd.DataFrame) -> None:
    root = results_root()
    caches = {g: Cache(group_root(g) / "evals.jsonl") for g in dict.fromkeys(pairs["group"])
              if (group_root(g) / "evals.jsonl").exists()}
    per, cmp_dir, groups = tables(direction_scores(caches, pairs), s1.XCFG["bootstrap"])
    if per.empty:
        print("no results yet")
        return
    RESULTS.mkdir(exist_ok=True)
    per.to_csv(RESULTS / "per_direction.csv", **KW)
    cmp_dir.to_csv(RESULTS / "comparisons.csv", **KW)
    groups.to_csv(RESULTS / "summary.csv", **KW)
    reproduction(caches, pairs).to_csv(RESULTS / "reproduction.csv", **KW)
    fits = [{"group": p.parent.parent.name, **torch.load(p)["info"]}
            for p in sorted(root.glob("*/maps/*.pt"))]
    log = root / "run.log"
    info = {"commit": s1.git("rev-parse", "HEAD").strip(),
            "eval_commits": s1.eval_commits(log.read_text(encoding="utf-8"))
            if log.exists() else [],
            "config": XCFG, "eval": s1.EVAL, "c1_commit": c1_commit(),
            "strong_maps_cache": sm.VERSION, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "n_records": {g: sum(1 for _ in c) for g, c in caches.items()}, "fits": fits}
    (RESULTS / "run_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(groups.to_string())
    print(f"results written to {RESULTS}")


def make_pairs() -> pd.DataFrame:
    """Each group's draw; a group with `pairs_from` takes that group's drawn pairs that its own
    data has (a budget folder's pairs.csv)."""
    pilot = pd.read_csv(sm.HERE / "pilot_pairs.csv")
    drawn = {}
    for g, c in XCFG["groups"].items():
        sm.configure(g)
        if "pairs_from" in c:
            drawn[g] = shared_pairs(drawn[c["pairs_from"]], pd.read_csv(s1.p6_dir() / "pairs.csv"))
        else:
            drawn[g] = draw_pairs(pd.read_csv(s1.HERE / "pairs.csv"), pilot[pilot["unit"] == g],
                                  XCFG["n_pairs"], XCFG["seed"])
    return pd.concat([d.assign(group=g) for g, d in drawn.items()],
                     ignore_index=True)[["group", "order", "s", "t"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-pairs", action="store_true", help="write pairs.csv")
    ap.add_argument("--results-only", action="store_true", help="build results/ from the cache")
    ap.add_argument("--groups", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs per group")
    args = ap.parse_args()
    if args.make_pairs:
        make_pairs().to_csv(HERE / "pairs.csv", index=False, lineterminator="\n")
        print(pd.read_csv(HERE / "pairs.csv").to_string())
        return
    pairs = pd.read_csv(HERE / "pairs.csv")
    if args.groups:
        pairs = pairs[pairs["group"].isin(args.groups.split(","))]
    if args.limit:
        pairs = pairs.groupby("group", sort=False).head(args.limit)
    if not args.results_only:
        run_all(pairs.reset_index(drop=True))
    write_results(pd.read_csv(HERE / "pairs.csv"))


if __name__ == "__main__":
    main()
