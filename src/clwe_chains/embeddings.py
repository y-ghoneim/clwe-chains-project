"""Load fastText embeddings onto the GPU and normalize them.

Main setting: unit length, mean centering, unit length again (Artetxe et al. 2018, as used by
Glavaš et al. 2019), applied once per language over its full top-N vocabulary. The AUS-faithful
variant uses raw vectors (steps=[]).
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from clwe_chains.config import Config


def normalize(emb: torch.Tensor, steps: list[str]) -> torch.Tensor:
    for step in steps:
        if step == "unit":
            emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
        elif step == "center":
            emb = emb - emb.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"unknown normalization step {step!r}")
    return emb


class EmbeddingStore:
    """Normalized embeddings by language, with a small cache (each is ~240 MB on the GPU).

    steps=None uses cfg["embeddings"]["normalize"]; steps=[] gives raw vectors.
    """

    def __init__(self, cfg: Config, max_cached: int = 6, steps: list[str] | None = None):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.steps = cfg["embeddings"]["normalize"] if steps is None else steps
        self.max_cached = max_cached
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __getitem__(self, lang: str) -> torch.Tensor:
        if lang in self._cache:
            self._cache.move_to_end(lang)
            return self._cache[lang]
        arr = np.load(self.cfg.path("v0_embeddings") / f"{lang}.npy")
        emb = normalize(torch.from_numpy(arr).to(self.device), self.steps)
        self._cache[lang] = emb
        if len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return emb
