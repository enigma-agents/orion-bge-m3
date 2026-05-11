from __future__ import annotations

from typing import Protocol


class SparseEncoder(Protocol):
    """Encodes text into BGE-M3 lexical_weights — a token_id → weight map
    Milvus stores in its sparse vector field."""

    def encode(self, texts: list[str]) -> list[dict[int, float]]: ...
