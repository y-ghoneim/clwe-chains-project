"""Word translation with an LLM through the Message Batches API (RQ3).

Shared by prep/12_llm_translate.py (machine-made dictionaries: one word per source word) and
experiments/mt-pivot/ (the LLM as the translator: up to n candidates). BatchTranslator is a
`translate_many` for mt.run_routes: one batch per depth covering every language pair, requests of
`words_per_request` words with JSON output matched back by the echoed source word, one ordinary
probe request before the first batch, requests that fail or are cut off retried once split in
half. Words still without an answer come back as None: run_routes doesn't cache them, so a rerun
asks again, and callers stop before scoring when more than MAX_FAILED of the words failed. Each
batch's id is saved before polling (a rerun retrieves it instead of paying again) and its usage
appended to usage.jsonl. Cache tags carry the use and a prompt version (cache_tag). Needs
ANTHROPIC_API_KEY (set by the user; never printed).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import unicodedata
from pathlib import Path

log = logging.getLogger("llm")

NAMES = {"ar": "Arabic", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
         "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fr": "French",
         "ga": "Irish", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian",
         "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ka": "Georgian",
         "kk": "Kazakh", "ko": "Korean", "mt": "Maltese", "ru": "Russian", "sw": "Swahili",
         "ta": "Tamil", "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "ur": "Urdu",
         "zh": "Chinese (Simplified)", "yo": "Yoruba", "or": "Odia", "mg": "Malagasy",
         "as": "Assamese", "ckb": "Sorani Kurdish", "qu": "Quechua", "jv": "Javanese"}
MAX_FAILED = 0.05      # callers stop before scoring if more words than this failed
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["translations"],
          "properties": {"translations": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["source", "words"],
              "properties": {"source": {"type": "string"},
                             "words": {"type": "array", "items": {"type": "string"}}}}}}}


def cache_tag(model: str, use: str) -> str:
    """The hop-cache model tag: the use and prompt version, so a changed prompt never reuses old
    answers ("dict-v1": one word for dictionaries; "top3-v1": up to 3 as the translator)."""
    return f"{model}/{use}"


def prompt(words: list[str], a: str, b: str, n: int = 1) -> str:
    """The request text: one most common translation (n = 1) or up to n, most likely first."""
    want = (f"the single most common {NAMES[b]} word with the same meaning and part of speech"
            if n == 1 else
            f"up to {n} {NAMES[b]} words with the same meaning and part of speech, most likely "
            f"first")
    return (f"Translate each {NAMES[a]} word below into {NAMES[b]}. For each, give {want}: "
            f"single words, no articles, no explanation, in `words`. Copy each source word "
            f"exactly into `source`.\n\n" + json.dumps(words, ensure_ascii=False))


def chunks(need: dict[tuple[str, str], list[str]], size: int) -> dict[str, tuple]:
    """custom_id -> (a, b, words) for every language pair's words, `size` words per request."""
    out = {}
    for (a, b), ws in need.items():
        for i in range(0, len(ws), size):
            out[f"{a}-{b}-{i // size}"] = (a, b, ws[i:i + size])
    return out


def _key(word: str) -> str:
    return unicodedata.normalize("NFC", word).strip().casefold()


def parse(text: str, words: list[str], n: int = 1) -> list[list[str] | None]:
    """Candidates per word (at most n, empty strings dropped) from a JSON reply, matched by the
    echoed source word (NFC, case-folded), or by position when the reply has one entry per word.
    None: the reply has no entry for the word (or isn't valid JSON), so it is asked again; []:
    the model answered with no word."""
    try:
        entries = [(t["source"], t["words"]) for t in json.loads(text)["translations"]]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [None for _ in words]
    got = {_key(s): ws for s, ws in entries if isinstance(s, str)}
    out = []
    for i, w in enumerate(words):
        ws = got.get(_key(w))
        if ws is None and len(entries) == len(words):
            ws = entries[i][1]
        out.append(None if not isinstance(ws, list) else
                   [c for c in ws if isinstance(c, str) and c.strip()][:n])
    return out


class BatchTranslator:
    """translate_many for mt.run_routes (see the module docstring). cfg: model, max_tokens,
    effort, words_per_request (configs/default.yaml machine_dicts.llm; words_per_request can be
    overridden). state_dir: where each submitted batch's id and requests are saved before polling,
    so a rerun after a crash retrieves the paid batch instead of submitting it again, and where
    every batch's usage is appended (usage.jsonl)."""

    def __init__(self, cfg: dict, n: int = 1, dry_run: bool = False,
                 state_dir: Path | None = None, words_per_request: int | None = None):
        self.cfg, self.n, self.dry_run = cfg, n, dry_run
        self.size = words_per_request or cfg["words_per_request"]
        self.state_dir = Path(state_dir) if state_dir else None
        if not dry_run:
            import anthropic
            self.client = anthropic.Anthropic()
        self.tokens = {"input": 0, "output": 0}
        self.requests, self.probed = 0, False
        self.failed: dict[tuple[str, str], list[int]] = {}    # (a, b) -> [words, failed]

    def params(self, a: str, b: str, words: list[str]) -> dict:
        return {"model": self.cfg["model"], "max_tokens": self.cfg["max_tokens"],
                "output_config": {"effort": self.cfg["effort"],
                                  "format": {"type": "json_schema", "schema": SCHEMA}},
                "messages": [{"role": "user", "content": prompt(words, a, b, self.n)}]}

    def cost(self, price: dict) -> float:
        return (self.tokens["input"] * price["input"]
                + self.tokens["output"] * price["output"]) / 1e6

    def failed_share(self) -> tuple[int, int]:
        """(failed words, words) over everything translated so far."""
        return (sum(f for _, f in self.failed.values()), sum(n for n, _ in self.failed.values()))

    def _state_file(self, todo: dict[str, tuple]) -> Path | None:
        if self.state_dir is None:
            return None
        digest = hashlib.sha256(json.dumps(sorted(todo.items()), ensure_ascii=False)
                                .encode("utf-8")).hexdigest()[:16]
        return self.state_dir / "batches" / f"{digest}.json"

    def run_batch(self, todo: dict[str, tuple]) -> dict[str, list]:
        """custom_id -> candidates per word for the requests that ended normally (end_turn)."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        state = self._state_file(todo)
        if state is not None and state.exists():
            batch_id = json.loads(state.read_text(encoding="utf-8"))["batch_id"]
            log.info("resuming saved batch %s (%d requests)", batch_id, len(todo))
        else:
            reqs = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                **self.params(a, b, ws))) for cid, (a, b, ws) in todo.items()]
            batch_id = self.client.messages.batches.create(requests=reqs).id
            log.info("batch %s: %d requests", batch_id, len(reqs))
            if state is not None:
                state.parent.mkdir(parents=True, exist_ok=True)
                state.write_text(json.dumps({"batch_id": batch_id, "requests": len(todo)}) + "\n",
                                 encoding="utf-8")
        while (batch := self.client.messages.batches.retrieve(batch_id)).processing_status \
                != "ended":
            log.info("  %s: %d processing", batch.processing_status,
                     batch.request_counts.processing)
            time.sleep(30)
        out, usage = {}, {"input": 0, "output": 0, "max_output": 0, "not_end_turn": 0}
        for r in self.client.messages.batches.results(batch_id):
            if r.result.type != "succeeded" or r.custom_id not in todo:
                continue
            msg = r.result.message
            usage["input"] += msg.usage.input_tokens
            usage["output"] += msg.usage.output_tokens
            usage["max_output"] = max(usage["max_output"], msg.usage.output_tokens)
            if msg.stop_reason != "end_turn":
                usage["not_end_turn"] += 1
                continue
            text = next((blk.text for blk in msg.content if blk.type == "text"), "")
            out[r.custom_id] = parse(text, todo[r.custom_id][2], self.n)
        self.tokens["input"] += usage["input"]
        self.tokens["output"] += usage["output"]
        log.info("batch %s done: %d of %d ok, largest reply %d tokens", batch_id, len(out),
                 len(todo), usage["max_output"])
        if self.state_dir is not None:
            with open(self.state_dir / "usage.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                     "batch_id": batch_id, "requests": len(todo),
                                     "model": self.cfg["model"], **usage}) + "\n")
        return out

    def probe(self, a: str, b: str, words: list[str]) -> None:
        """One ordinary request before the first batch: a wrong parameter or key fails here, not
        as thousands of failed batch requests."""
        msg = self.client.messages.create(**self.params(a, b, words))
        self.tokens["input"] += 2 * msg.usage.input_tokens       # standard price: twice batch
        self.tokens["output"] += 2 * msg.usage.output_tokens
        text = next((blk.text for blk in msg.content if blk.type == "text"), "")
        got = parse(text, words, self.n)
        log.info("probe %s->%s (%s, %d output tokens): %s", a, b, msg.stop_reason,
                 msg.usage.output_tokens, dict(zip(words, got, strict=True)))
        if msg.stop_reason != "end_turn" or not any(got):
            raise SystemExit("probe request failed; nothing submitted")

    def __call__(self, need: dict[tuple[str, str], list[str]]) -> dict:
        todo = chunks(need, self.size)
        self.requests += len(todo)
        n_words = sum(len(w) for w in need.values())
        if self.dry_run:
            log.info("dry run: %d requests for %d words at this depth", len(todo), n_words)
            raise SystemExit(0)
        if not self.probed:
            a, b, ws = next(iter(todo.values()))
            self.probe(a, b, ws[:10])
            self.probed = True
        got = self.run_batch(todo)
        # requests that failed, were cut off or refused: once more, split in half
        missing = {}
        for cid, (a, b, ws) in todo.items():
            if cid not in got:
                half = (len(ws) + 1) // 2
                missing[f"{cid}h0"] = (a, b, ws[:half])
                if ws[half:]:
                    missing[f"{cid}h1"] = (a, b, ws[half:])
        if missing:
            log.info("retrying %d failed requests as %d halves", len(todo) - len(got),
                     len(missing))
            again = self.run_batch(missing)
            for cid in todo:
                if cid not in got and f"{cid}h0" in again:
                    got[cid] = again[f"{cid}h0"] + again.get(f"{cid}h1", [None] * (
                        len(todo[cid][2]) - len(again[f"{cid}h0"])))
        result: dict[tuple[str, str], list] = {ab: [] for ab in need}
        for cid, (a, b, ws) in todo.items():
            result[a, b] += got.get(cid, [None] * len(ws))
        for ab, cands in result.items():
            stat = self.failed.setdefault(ab, [0, 0])
            stat[0] += len(cands)
            stat[1] += sum(c is None for c in cands)
            if cands and sum(c is None for c in cands) > 0.1 * len(cands):
                log.warning("%s->%s: %d of %d words failed (not cached; rerun to retry)",
                            ab[0], ab[1], sum(c is None for c in cands), len(cands))
        return result
