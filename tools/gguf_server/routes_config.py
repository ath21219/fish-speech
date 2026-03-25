"""Runtime configuration endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .idle_offload import start_idle_offload_daemon, stop_idle_offload_daemon
from .state import save_server_state, state

router = APIRouter()


class IdleOffloadConfig(BaseModel):
    timeout_seconds: int = 0  # 0 = disabled


class IdleOffloadStatus(BaseModel):
    enabled: bool
    timeout_seconds: int
    offloaded: bool


@router.post("/v1/config/idle-offload")
async def set_idle_offload(config: IdleOffloadConfig):
    if config.timeout_seconds < 0:
        raise HTTPException(400, "timeout_seconds must be >= 0")

    # Stop existing daemon
    stop_idle_offload_daemon()

    # Start new daemon if timeout > 0
    if config.timeout_seconds > 0:
        start_idle_offload_daemon(config.timeout_seconds)

    save_server_state()
    return {"status": "ok", "timeout_seconds": config.timeout_seconds}


@router.get("/v1/config/idle-offload")
async def get_idle_offload():
    return IdleOffloadStatus(
        enabled=state._idle_offload_timeout > 0,
        timeout_seconds=state._idle_offload_timeout,
        offloaded=state._offloaded,
    )
