from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@router.get("/readyz")
def readyz(request: Request) -> dict[str, bool]:
    return {"ok": hasattr(request.app.state, "use_case")}
