"""Pydantic request/response models and shared constants."""

import base64
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ── Streaming constants ──
STREAM_CHUNK_TOKENS = 21
STREAM_MIN_FIRST_CHUNK = 21
AMPLITUDE = 32768

# ── Codec limits ──
# The codec's Transformer buffers (causal_mask, freqs_cis) are shrunk from
# 8192 to 2048 frames in load_codec_from_gguf() to save VRAM.
# The bottleneck is the encoder's internal Transformer (encoder_transformer_layers
# [0,0,0,4]): it operates at encoder hop resolution (512x downsampling), so it
# receives  samples / 512  frames.  The shrunk limit of 2048 frames therefore
# caps input audio at  2048 * 512 / 44100 ≈ 23.8 s.
# (The quantizer's Transformer is fine — it gets 4x fewer frames via
# downsample_factor [2,2].)
# We use 23 s as a practical limit with safety margin.
CODEC_MAX_SECONDS = 23


# ── Request / Response schemas ──

class TTSReference(BaseModel):
    audio: bytes = b""
    text: str = ""

    @model_validator(mode="before")
    @classmethod
    def decode_audio(cls, values):
        audio = values.get("audio")
        if isinstance(audio, str) and len(audio) > 255:
            try:
                values["audio"] = base64.b64decode(audio)
            except Exception:
                pass
        return values


class TTSRequest(BaseModel):
    text: str
    references: List[TTSReference] = []
    reference_id: Optional[str] = None
    format: str = Field(default="wav", pattern="^(wav|flac|mp3)$")
    streaming: bool = False
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 30
    repetition_penalty: float = 1.2
    chunk_length: int = 200
    sentence_split: bool = False
    sliding_window_size: int = 0  # 0=disabled, >0=window token count (recommended: 512)


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = ""
    device: str = ""
    vram_used_gb: float = 0.0
    offloaded: bool = False
