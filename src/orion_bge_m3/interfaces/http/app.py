from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orion_bge_m3.application.encode_sparse import EncodeSparseUseCase
from orion_bge_m3.config import get_settings
from orion_bge_m3.infrastructure.bgem3_encoder import BgeM3SparseEncoder
from orion_bge_m3.interfaces.http.routers import embed, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    encoder = BgeM3SparseEncoder(
        model_id=s.model_id, use_fp16=s.use_fp16, device=s.device
    )
    app.state.use_case = EncodeSparseUseCase(encoder)
    yield


def create_app() -> FastAPI:
    s = get_settings()
    logging.basicConfig(
        level=s.log_level,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )
    app = FastAPI(
        title="orion-bge-m3",
        version="0.1.0",
        description="BGE-M3 sparse embedding service. Returns lexical_weights for hybrid Milvus retrieval.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(embed.router)
    return app
