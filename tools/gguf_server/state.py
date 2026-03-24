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
    """Persist current model load state to disk."""
    data = {"active_model": state.active_model_name}
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        from loguru import logger

        logger.warning(f"Failed to save server state to {_STATE_FILE}")


def load_server_state() -> str | None:
    """Load previously persisted model name. Returns None if unset or missing."""
    try:
        if _STATE_FILE.exists():
            data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            return data.get("active_model")
    except Exception:
        pass
    return None
