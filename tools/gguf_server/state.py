"""
Global server state singleton.

Every other module in gguf_server imports `state` and `app` from here,
avoiding circular dependencies.
"""

import json
from pathlib import Path
from threading import Lock

from fastapi import FastAPI


class GGUFServerState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.codec = None
        self.sample_rate = 44100
        self.device = "cuda"
        self.lock = Lock()
        self.max_seq_len = 2048
        self.ready = False
        self.active_model_name = None
        self.codec_gpu_resident = False
        self.sliding_window_size = 0  # server default; 0 = disabled

        # ── Idle VRAM offload state ──
        self._offloaded = False
        self._last_request_time = 0.0
        self._idle_offload_timeout = 0       # seconds; 0 = disabled
        self._offload_thread = None
        self._offload_stop_event = None
        self._codec_was_gpu_resident = False  # remember placement before offload


# Singleton instances
state = GGUFServerState()
app = FastAPI(title="Fish Speech GGUF API", version="1.0.0")

# Shared directory paths
MODELS_DIR = Path("models")
REFERENCES_DIR = Path("references")

# Persistent server state file (inside models/ volume)
_STATE_FILE = MODELS_DIR / ".server_state.json"

# Will be set by the entrypoint after argument parsing.
# Used by routes_models.load_model_endpoint() to pass args to load_models().
_server_args = None


def save_server_state():
    """Persist current model load state and settings to disk."""
    data = {
        "active_model": state.active_model_name,
        "idle_offload_timeout": state._idle_offload_timeout,
    }
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        from loguru import logger

        logger.warning(f"Failed to save server state to {_STATE_FILE}")


def load_server_state() -> dict | None:
    """Load previously persisted server state.

    Returns dict with keys 'active_model', 'idle_offload_timeout', etc.
    Returns None if missing or unreadable.
    """
    try:
        if _STATE_FILE.exists():
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None
