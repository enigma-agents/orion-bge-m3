from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Resolve "auto" to the best available backend: mps > cuda > cpu.
    mps requires a native macOS process (Metal); it is absent inside
    Linux containers, so "auto" safely degrades to cpu there."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # torch import / probe failure -> cpu
        pass
    return "cpu"


class BgeM3SparseEncoder:
    """FlagEmbedding adapter for SparseEncoder. Loads BGEM3FlagModel
    once at startup; HF_HUB_OFFLINE=1 in the container forces lookup
    against the mounted host cache (no network)."""

    def __init__(
        self, model_id: str, use_fp16: bool = False, device: str = "auto"
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        resolved = _resolve_device(device)
        # fp16 halves memory and is well-supported on Metal; force it on
        # mps even if config left use_fp16 False. Keep the caller's choice
        # on cuda; keep fp32 on cpu (fp16 is slow/partial there).
        if resolved == "mps":
            use_fp16 = True

        log.info(
            "loading BGE-M3 from %s (device=%s, use_fp16=%s)",
            model_id,
            resolved,
            use_fp16,
        )
        self._model = BGEM3FlagModel(model_id, use_fp16=use_fp16, devices=resolved)
        log.info("BGE-M3 loaded on %s", resolved)

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        out = self._model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [{int(k): float(v) for k, v in w.items()} for w in out["lexical_weights"]]
