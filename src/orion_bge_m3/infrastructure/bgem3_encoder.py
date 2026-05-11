from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class BgeM3SparseEncoder:
    """FlagEmbedding adapter for SparseEncoder. Loads BGEM3FlagModel
    once at startup; HF_HUB_OFFLINE=1 in the container forces lookup
    against the mounted host cache (no network)."""

    def __init__(self, model_id: str, use_fp16: bool = False) -> None:
        from FlagEmbedding import BGEM3FlagModel

        log.info("loading BGE-M3 from %s (use_fp16=%s)", model_id, use_fp16)
        self._model = BGEM3FlagModel(model_id, use_fp16=use_fp16)
        log.info("BGE-M3 loaded")

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        out = self._model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [{int(k): float(v) for k, v in w.items()} for w in out["lexical_weights"]]
