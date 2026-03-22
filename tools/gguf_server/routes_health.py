"""Health check endpoint."""

import torch
from fastapi import APIRouter

from .schemas import HealthResponse
from .state import state

router = APIRouter()


@router.get("/v1/health")
async def health():
    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
    return HealthResponse(
        status=(
            "ok"
            if state.ready
            else "loading" if state.active_model_name else "empty"
        ),
        model=state.active_model_name or "",
        device=state.device,
        vram_used_gb=round(vram, 2),
    )


@router.post("/v1/health")
async def health_post():
    return await health()
