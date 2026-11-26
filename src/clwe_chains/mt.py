"""Word translation with an MT system, hop by hop, with a resumable cache.

Shared by experiments/mt-pivot/ (translating test words) and prep/11_machine_dicts.py (building
dictionaries from machine translations). Language codes for NLLB-200; output matching (strict and
lenient), output cleaning (clean_hop); HopCache (model, from, to, input) -> candidates in JSONL;
run_routes translates routes depth by depth; NLLB wraps the model (loaded on first use).
"""

from __future__ import annotations

import json
import re
import string
import time
from pathlib import Path

from clwe_chains.text import normalize

CODES = {"ar": "arb_Arab", "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn",
         "es": "spa_Latn", "et": "est_Latn", "eu": "eus_Latn", "fa": "pes_Arab",
         "fi": "fin_Latn", "fr": "fra_Latn", "ga": "gle_Latn", "he": "heb_Hebr",
         "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn", "id": "ind_Latn",
         "it": "ita_Latn", "ja": "jpn_Jpan", "ka": "kat_Geor", "kk": "kaz_Cyrl",
         "ko": "kor_Hang", "mt": "mlt_Latn", "ru": "rus_Cyrl", "sw": "swh_Latn",
         "ta": "tam_Taml", "th": "tha_Thai", "tl": "tgl_Latn", "tr": "tur_Latn",
         "ur": "urd_Arab", "zh": "zho_Hans", "yo": "yor_Latn", "or": "ory_Orya",
         "mg": "plt_Latn", "as": "asm_Beng", "ckb": "ckb_Arab", "qu": "quy_Latn",
         "jv": "jav_Latn"}
NODES = sorted(set(CODES) - {"yo", "or", "mg", "as", "ckb", "qu", "jv"})
_PUNCT = string.punctuation + "«»„“”‘’…。、！？：；（）「」《》"


# ---------------------------------------------------------------------------------------------
# Matching and cleaning (tests/test_mt_pivot.py)

def norm_answer(text: str, lang: str) -> str:
    """Comparable form of an MT output or gold string: normalized, edge punctuation stripped."""
    return normalize(text.strip().strip(_PUNCT).strip(), lang)


def distinct(cands: list[str], lang: str) -> list[str]:
    """Candidates without repeats of the same normalized form (casing or punctuation variants
    that beam search often returns), first occurrence kept, empty outputs dropped."""
    seen, out = set(), []
    for c in cands:
        n = norm_answer(c, lang)
        if n and n not in seen:
            seen.add(n)
            out.append(c)
    return out


def rank(cands: list[str], gold: set[str], lang: str) -> int:
    """Strict: 1-based rank of the first candidate that equals a gold answer (whole output or its
    first word), 0 if none does. gold holds normalized answers."""
    for i, c in enumerate(cands, 1):
        n = norm_answer(c, lang)
        if n in gold or (" " in n and n.split()[0] in gold):
            return i
    return 0


# NLLB often answers a single word with a short phrase ("The fox .", "It 's a fan .", "La volpe").
# Leading fillers and articles are stripped before the next hop and before lenient scoring.
_FILLERS = {"en": ["it's", "it is", "this is", "that's", "there's", "there is", "the", "a", "an",
                   "to"],
            "fr": ["c'est", "le", "la", "les", "un", "une", "des", "du"],
            "it": ["è", "il", "lo", "la", "i", "gli", "le", "un", "uno", "una"],
            "es": ["es", "el", "la", "los", "las", "un", "una", "unos", "unas"],
            "de": ["es ist", "das ist", "der", "die", "das", "den", "dem", "ein", "eine", "einen"],
            "el": ["ο", "η", "το", "οι", "τα", "ένας", "μια", "ένα"],
            "hu": ["a", "az", "egy"]}
_ELIDED = {"fr": ("l'", "d'"), "it": ("l'", "un'")}
_MT_ARTICLE = re.compile(r"^(?:i[lsxżċdnrtz]|l)-", re.IGNORECASE)
_EDGE = re.compile(rf"^[\s{re.escape(_PUNCT)}]+|[\s{re.escape(_PUNCT)}]+$")


def clean_hop(text: str, lang: str) -> str:
    """The head of an MT output: edge punctuation, leading fillers ("it's", "this is") and
    articles removed; the output itself if nothing would be left."""
    t = _EDGE.sub("", re.sub(r"\s+'\s*", "'", text))
    changed = True
    while changed:
        changed = False
        low = t.lower()
        for f in _FILLERS.get(lang, []):
            if low.startswith(f + " ") and len(t) > len(f) + 1:
                t, changed = t[len(f) + 1:].lstrip(), True
                break
        else:
            for e in _ELIDED.get(lang, ()):
                if low.startswith(e) and len(t) > len(e):
                    t, changed = t[len(e):], True
                    break
            if not changed and lang == "mt" and _MT_ARTICLE.match(t):
                t, changed = _MT_ARTICLE.sub("", t), True
        t = _EDGE.sub("", t)
    return t or text.strip()


def rank_lenient(cands: list[str], gold: set[str], lang: str) -> int:
    """Lenient: 1-based rank of the first candidate whose cleaned form (clean_hop), or any of its
    words, equals a gold answer; 0 if none does."""
    for i, c in enumerate(cands, 1):
        n = norm_answer(clean_hop(c, lang), lang)
        if n in gold or any(w.strip(_PUNCT) in gold for w in n.split()):
            return i
    return 0


class HopCache:
    """(model, from, to, word) -> candidates, in an append-only JSONL file."""

    def __init__(self, path: Path):
        self.path, self.data = Path(path), {}
        if self.path.exists():
            # split on \n only: splitlines() also breaks at U+2028 and friends, which
            # json.dumps(ensure_ascii=False) leaves inside strings
            for line in self.path.read_text(encoding="utf-8").split("\n"):
                try:
                    k, v = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                self.data[tuple(k)] = v

    def get(self, key):
        return self.data.get(key)

    def put_many(self, items: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for k, v in items.items():
                fh.write(json.dumps([list(k), v], ensure_ascii=False) + "\n")
                self.data[k] = v


def run_routes(jobs: list[tuple], translate, cache: HopCache, model: str,
               translate_many=None) -> dict:
    """jobs: (job id, route, source word). Translates hop by hop, depth by depth, batching all
    uncached hops of one depth by language pair; each hop's input is the previous hop's top
    output, cleaned (clean_hop). Returns job id -> the final hop's candidates ([] if a hop
    produced nothing).

    translate(words, a, b) -> candidates per word, called once per language pair and depth; or
    translate_many({(a, b): words}) -> {(a, b): candidates per word}, called once per depth (an
    API batch covering every language pair at once). A word's candidates may be None (its request
    failed): it counts as no output in this run and is not cached, so a rerun asks again."""
    state = {j: (route, 0, word) for j, route, word in jobs}
    final: dict = {}
    while state:
        need: dict[tuple[str, str], set[str]] = {}
        for route, d, word in state.values():
            if cache.get((model, route[d], route[d + 1], word)) is None:
                need.setdefault((route[d], route[d + 1]), set()).add(word)
        need_sorted = {ab: sorted(ws) for ab, ws in need.items()}
        failed: set = set()

        def keep(a, b, ws, cands, failed=failed):
            failed.update((a, b, w) for w, c in zip(ws, cands, strict=True) if c is None)
            cache.put_many({(model, a, b, w): c for w, c in zip(ws, cands, strict=True)
                            if c is not None})

        if translate_many is not None and need_sorted:
            got = translate_many(need_sorted)
            for (a, b), ws in need_sorted.items():
                keep(a, b, ws, got[a, b])
        else:
            for (a, b), ws in need_sorted.items():         # cached per pair: a crash keeps them
                keep(a, b, ws, translate(ws, a, b))
        nxt = {}
        for j, (route, d, word) in state.items():
            if (route[d], route[d + 1], word) in failed:
                final[j] = []
                continue
            cands = cache.get((model, route[d], route[d + 1], word)) or []
            if d + 2 == len(route) or not cands:
                final[j] = cands
            else:
                nxt[j] = (route, d + 1, clean_hop(cands[0], route[d + 1]))
        state = nxt
    return final


# ---------------------------------------------------------------------------------------------
# NLLB

class NLLB:
    def __init__(self, name: str, beams: int, n_return: int, max_new: int, batch: int):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
        self.model.generation_config.max_length = None      # max_new_tokens governs length
        self.beams, self.n_return, self.max_new, self.batch = beams, n_return, max_new, batch
        self.done, self.t0, self.next_log = 0, time.time(), 5000

    def __call__(self, words: list[str], src: str, tgt: str) -> list[list[str]]:
        self.tok.src_lang = CODES[src]
        bos = self.tok.convert_tokens_to_ids(CODES[tgt])
        out = []
        for i in range(0, len(words), self.batch):
            chunk = words[i:i + self.batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to("cuda")
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, forced_bos_token_id=bos, num_beams=self.beams,
                                          num_return_sequences=self.n_return,
                                          max_new_tokens=self.max_new)
            text = self.tok.batch_decode(gen, skip_special_tokens=True)
            out += [distinct(text[k * self.n_return:(k + 1) * self.n_return], tgt)
                    for k in range(len(chunk))]
        self.done += len(words)
        if self.done >= self.next_log:
            el = time.time() - self.t0
            print(f"{time.strftime('%H:%M:%S')} {self.done} translations, {el / 60:.1f} min, "
                  f"{self.done / el:.0f}/s", flush=True)
            self.next_log += 5000
        return out
