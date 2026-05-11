from __future__ import annotations

from typing import Union

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class SparseRequest(BaseModel):
    inputs: Union[str, list[str]] = Field(
        ..., description="Single string or list of strings to encode."
    )


class SparseResponse(BaseModel):
    # token_id (str — JSON keys are strings) → weight
    results: list[dict[str, float]]


@router.post("/sparse", response_model=SparseResponse)
def sparse(req: SparseRequest, request: Request) -> SparseResponse:
    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    weights = request.app.state.use_case.execute(texts)
    return SparseResponse(results=[{str(k): v for k, v in w.items()} for w in weights])
