"""
Fish Speech GGUF API Server — modular package.

Submodules:
  logging_setup   Loguru/uvicorn unified logging
  schemas         Pydantic request/response models
  state           Global server state singleton
  codec_manager   Codec GPU/CPU lifecycle
  model_loader    GGUF model + codec loading
  tts_engine      Streaming token generation + codec decode
  routes_*        FastAPI route modules
"""
