"""
Global server state singleton.

Every other module in gguf_server imports `state` and `app` from here,
avoiding circular dependencies.
"""

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

# Will be set by the entrypoint after argument parsing.
# Used by routes_models.load_model_endpoint() to pass args to load_models().
_server_args = None
