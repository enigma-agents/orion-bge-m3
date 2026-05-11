from __future__ import annotations

from orion_bge_m3.domain.ports.sparse_encoder import SparseEncoder


class EncodeSparseUseCase:
    def __init__(self, encoder: SparseEncoder) -> None:
        self._encoder = encoder

    def execute(self, texts: list[str]) -> list[dict[int, float]]:
        if not texts:
            return []
        return self._encoder.encode(texts)
