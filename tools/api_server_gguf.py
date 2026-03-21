"""
Fish Speech GGUF API Server
Triton-accelerated Q6_K inference with OpenAI-compatible TTS endpoint.

Usage:
  python tools/api_server_gguf.py \
    --gguf-path ./models/gguf/s2-pro-q6_k.gguf \
    --codec-path ./checkpoints/s2-pro/codec.pth \
    --listen 0.0.0.0:7820
"""

import argparse
import io
import sys
import time
import base64
from pathlib import Path
from threading import Lock


import numpy as np
import soundfile as sf
import torch
import uvicorn
import json
import shutil
from fastapi import UploadFile, File, Form, FastAPI, HTTPException, Header
from fastapi.responses import Response, JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from contextlib import contextmanager

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@contextmanager
def codec_on_gpu():
    """
    Offload GGUF model to CPU, move codec to GPU,
    then restore after encode/decode.
    """
    model = state.model
    codec = state.codec
    device = state.device

    # 1. Offload GGUF model to CPU
    logger.info("Offloading GGUF model to CPU...")
    t0 = time.perf_counter()
    model.cpu()
    for name, module in model.named_modules():
        if hasattr(module, 'qparam') and hasattr(module.qparam, 'data'):
            if module.qparam.data.is_cuda:
                module.qparam.data = module.qparam.data.cpu()
        for attr_name in ['weight', 'qweight', 'data']:
            attr = getattr(module, attr_name, None)
            if isinstance(attr, torch.Tensor) and attr.is_cuda:
                setattr(module, attr_name, attr.cpu())
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Model offloaded in {time.perf_counter() - t0:.1f}s (VRAM: {vram_after:.2f} GB)")

    # 2. Move codec to GPU
    logger.info("Moving codec to GPU...")
    t0 = time.perf_counter()
    codec.to(device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Codec on GPU in {time.perf_counter() - t0:.1f}s (VRAM: {vram:.2f} GB)")

    try:
        yield codec
    finally:
        pass  # Caller must call restore_model() explicitly


def restore_after_codec():
    """Move codec back to CPU and restore GGUF model to GPU."""
    model = state.model
    codec = state.codec
    device = state.device

    logger.info("Offloading codec to CPU...")
    codec.to(device="cpu", dtype=torch.float32)
    torch.cuda.empty_cache()

    logger.info("Restoring GGUF model to GPU...")
    t0 = time.perf_counter()
    model.to(device)
    for name, module in model.named_modules():
        if hasattr(module, 'qparam') and hasattr(module.qparam, 'data'):
            if not module.qparam.data.is_cuda:
                module.qparam.data = module.qparam.data.to(device)
        for attr_name in ['weight', 'qweight', 'data']:
            attr = getattr(module, attr_name, None)
            if isinstance(attr, torch.Tensor) and not attr.is_cuda:
                setattr(module, attr_name, attr.to(device))
    torch.cuda.synchronize()

    model._cache_setup_done = False
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=torch.float16,
        )
    model._cache_setup_done = True
    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Model restored in {time.perf_counter() - t0:.1f}s (VRAM: {vram:.2f} GB)")


# ============================================================
# Request / Response schemas (compatible with official API)
# ============================================================

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

class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = ""
    device: str = ""
    vram_used_gb: float = 0.0


# ============================================================
# Global state
# ============================================================

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

state = GGUFServerState()
app = FastAPI(title="Fish Speech GGUF API", version="1.0.0")


# ============================================================
# Startup (from working code, unchanged)
# ============================================================

def load_models(args):
    """Load GGUF model and codec at startup."""
    from fish_speech.gguf import load_gguf_model
    from fish_speech.models.text2semantic.inference import load_codec_model

    logger.info("Loading GGUF model...")
    t0 = time.perf_counter()
    model = load_gguf_model(
        gguf_path=args.gguf_path,
        device=args.device,
        compute_dtype=torch.float16,
        max_seq_len=args.max_seq_len,
    )
    logger.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # Setup KV caches
    model.setup_caches(
        max_batch_size=1,
        max_seq_len=args.max_seq_len,
        dtype=torch.float16,
    )
    model = model.to(args.device)
    model._cache_setup_done = True

    logger.info("Loading codec...")
    t0 = time.perf_counter()
    # load_codec_model has @torch.inference_mode() decorator,
    # so we must re-create parameters as normal tensors
    codec = load_codec_model(args.codec_path, "cpu", precision=torch.float32)

    # Fix: convert all inference-mode tensors to normal tensors
    for name, param in list(codec.named_parameters()):
        parts = name.split('.')
        module = codec
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1],
                torch.nn.Parameter(param.data.clone(), requires_grad=False))

    for name, buf in list(codec.named_buffers()):
        parts = name.split('.')
        module = codec
        for part in parts[:-1]:
            module = getattr(module, part)
        module.register_buffer(parts[-1], buf.data.clone())

    logger.info(f"Codec loaded in {time.perf_counter() - t0:.1f}s")

    state.model = model
    state.tokenizer = model.tokenizer
    state.codec = codec
    state.sample_rate = codec.sample_rate
    state.device = args.device
    state.max_seq_len = args.max_seq_len
    state.ready = True

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Ready. VRAM: {vram:.2f} GB")


# ============================================================
# TTS generation core (from working code, unchanged)
# ============================================================

def generate_speech(req: TTSRequest) -> np.ndarray:
    """Generate speech from text, returns numpy audio array."""
    logger.info(f"generate_speech called: text={req.text[:50]}..., "
                f"reference_id={req.reference_id}, "
                f"references={len(req.references)}")
    from fish_speech.content_sequence import TextPart, VQPart
    from fish_speech.conversation import Conversation, Message
    from fish_speech.models.text2semantic.inference import (
        decode_one_token_ar,
        generate,
        decode_to_audio,
        encode_audio,
    )
    import tempfile
    import os
    import re
    from copy import deepcopy
    from fish_speech.models.text2semantic.inference import (
        split_text_by_speaker,
        group_turns_into_batches,
    )

    model = state.model
    tokenizer = state.tokenizer
    device = state.device

    # -------------------------------------------------------
    # Load reference audio
    # -------------------------------------------------------
    prompt_tokens_list = []
    prompt_texts = []

    # Reference encoding
    if req.references:
        logger.info(f"Encoding {len(req.references)} inline reference(s)...")
        with codec_on_gpu():
            for ref in req.references:
                if not ref.audio or not ref.text:
                    continue
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(ref.audio)
                    f.flush()
                    try:
                        codes = encode_audio(f.name, state.codec, state.device)
                        prompt_tokens_list.append(codes.cpu())
                        prompt_texts.append(ref.text)
                    finally:
                        os.unlink(f.name)
        logger.info(f"Reference encoding done: {len(prompt_tokens_list)} ref(s)")

    elif req.reference_id:
        ref_dir = Path("references") / req.reference_id
        if not ref_dir.exists():
            raise ValueError(f"Reference '{req.reference_id}' not found")

        codes_cache = ref_dir / "codes.pt"
        if not codes_cache.exists():
            raise ValueError(
                f"Reference '{req.reference_id}' has no cached codes. "
                f"Please re-register the voice via POST /v1/voices"
            )
        codes = torch.load(codes_cache, map_location="cpu", weights_only=True)
        logger.info(f"Loaded cached VQ codes: {codes.shape}")
        prompt_tokens_list.append(codes)

        ref_text = ""
        meta_file = ref_dir / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            ref_text = meta.get("description", "")
        if not ref_text:
            txt_files = list(ref_dir.glob("*.txt"))
            if txt_files:
                ref_text = txt_files[0].read_text(encoding="utf-8").strip()
        if not ref_text:
            raise ValueError(f"No description/text for reference '{req.reference_id}'")
        prompt_texts.append(ref_text)

    use_prompt = bool(prompt_texts) and bool(prompt_tokens_list)

    # -------------------------------------------------------
    # Build conversation
    # -------------------------------------------------------
    base_conversation = Conversation()

    if use_prompt:
        system_parts = [
            TextPart(
                text="convert the provided text to speech reference to the following:\n\nText:\n",
                cal_loss=False,
            ),
        ]
        tagged = []
        for i, t in enumerate(prompt_texts):
            if not re.search(r"<\|speaker:\d+\|>", t):
                tagged.append(f"<|speaker:{i}|>{t}")
            else:
                tagged.append(t)
        system_parts.append(TextPart(text="\n".join(tagged), cal_loss=False))
        system_parts.append(TextPart(text="\n\nSpeech:\n", cal_loss=False))
        all_codes = torch.cat(prompt_tokens_list, dim=1)
        system_parts.append(VQPart(codes=all_codes, cal_loss=False))
    else:
        system_parts = [
            TextPart(text="convert the provided text to speech", cal_loss=False)
        ]

    base_conversation.append(
        Message(
            role="system",
            parts=system_parts,
            cal_loss=False,
            add_im_start=True,
            add_im_end=True,
        )
    )

    # -------------------------------------------------------
    # Split text into batches and generate
    # -------------------------------------------------------
    turns = split_text_by_speaker(req.text)
    batches = (
        group_turns_into_batches(turns, max_speakers=5, max_bytes=req.chunk_length)
        if turns
        else [req.text]
    )

    all_segments = []
    conversation = deepcopy(base_conversation)

    for batch_idx, batch_text in enumerate(batches):
        logger.info(f"Batch {batch_idx}: {len(batch_text.encode('utf-8'))} bytes")

        conversation.append(
            Message(
                role="user",
                parts=[TextPart(text=batch_text, cal_loss=False)],
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )

        conversation_gen = deepcopy(conversation)
        conversation_gen.append(
            Message(
                role="assistant",
                parts=[],
                cal_loss=False,
                modality="voice",
                add_im_start=True,
                add_im_end=False,
            )
        )

        num_codebooks = model.config.num_codebooks
        encoded, audio_masks, audio_parts = conversation_gen.encode_for_inference(
            tokenizer, num_codebooks=num_codebooks
        )
        encoded = encoded.to(device=device)
        prompt_len = encoded.shape[1]

        if prompt_len > state.max_seq_len - 128:
            raise ValueError(f"Prompt too long: {prompt_len} tokens")

        t0 = time.perf_counter()
        with torch.inference_mode():
            y = generate(
                model=model,
                prompt=encoded,
                max_new_tokens=req.max_new_tokens,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
                decode_one_token=decode_one_token_ar,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
            )
        t_gen = time.perf_counter() - t0
        gen_tokens = y.shape[1] - prompt_len
        logger.info(f"Generated {gen_tokens} tokens in {t_gen:.1f}s "
                    f"({gen_tokens/t_gen:.1f} tok/s)")

        codes = y[1:, prompt_len:-1].clone().clamp(min=0)
        if codes.shape[1] == 0:
            logger.warning(f"Batch {batch_idx}: no codes generated, skipping")
            continue

        # Decode on CPU
        with torch.inference_mode():
            audio_segment = decode_to_audio(codes.cpu(), state.codec)
        all_segments.append(audio_segment.float().cpu().numpy())

        # Add back to conversation for multi-batch context
        conversation.append(
            Message(
                role="assistant",
                parts=[VQPart(codes=codes.cpu(), cal_loss=False)],
                cal_loss=False,
                modality="voice",
                add_im_start=True,
                add_im_end=True,
            )
        )
        del y, encoded

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_segments:
        raise ValueError("No audio generated")

    return np.concatenate(all_segments, axis=0)


# ============================================================
# API endpoints
# ============================================================

@app.get("/v1/health")
async def health():
    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
    return HealthResponse(
        status="ok" if state.ready else "loading",
        model="s2-pro-q6_k-gguf",
        device=state.device,
        vram_used_gb=round(vram, 2),
    )

@app.post("/v1/health")
async def health_post():
    return await health()


# ============================================================
# Voice management endpoints
# ============================================================

REFERENCES_DIR = Path("references")


@app.post("/v1/voices")
async def create_voice(
    name: str = Form(...),
    transcription: str = Form(""),
    audio: UploadFile = File(...),
):
    """Register a reference voice."""
    # Validate name
    if not name or not name.strip():
        raise HTTPException(400, "Voice name cannot be empty")
    name = name.strip()
    if not all(c.isalnum() or c in "-_ " for c in name):
        raise HTTPException(400, "Voice name can only contain alphanumeric, hyphen, underscore, space")

    voice_dir = REFERENCES_DIR / name
    if voice_dir.exists():
        raise HTTPException(409, f"Voice '{name}' already exists")

    # Read and validate audio
    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(400, "Audio file too small or empty")

    # Determine extension from filename
    ext = Path(audio.filename).suffix.lower() if audio.filename else ".wav"
    if ext not in (".wav", ".mp3", ".flac", ".ogg"):
        ext = ".wav"

    # Save
    voice_dir.mkdir(parents=True, exist_ok=True)
    try:
        audio_path = voice_dir / f"audio{ext}"
        audio_path.write_bytes(audio_bytes)

        meta = {"name": name, "transcription": transcription}
        (voice_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Also write transcription as .txt for compatibility with official format
        if transcription:
            (voice_dir / "audio.txt").write_text(transcription, encoding="utf-8")

        # Pre-encode VQ codes at registration time
        if state.ready:
            try:
                import torchaudio
                logger.info(f"Pre-encoding voice '{name}'...")

                # Offload model, load codec to GPU
                codec_ctx = codec_on_gpu()
                codec = codec_ctx.__enter__()

                # Encode with no_grad (not inference_mode)
                with torch.no_grad():
                    wav, sr = torchaudio.load(str(audio_path))
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    wav = torchaudio.functional.resample(
                        wav.to(state.device), sr, codec.sample_rate
                    )[0]
                    model_dtype = next(codec.parameters()).dtype
                    audios = wav[None, None].to(dtype=model_dtype)
                    audio_lengths = torch.tensor(
                        [len(wav)], device=state.device, dtype=torch.long
                    )
                    indices, feature_lengths = codec.encode(audios, audio_lengths)
                    codes = indices[0, :, :feature_lengths[0]]

                torch.save(codes.cpu(), voice_dir / "codes.pt")
                logger.info(f"Voice '{name}': pre-encoded {codes.shape[1]} VQ frames")

            except Exception as e:
                logger.warning(f"Voice '{name}': pre-encoding failed: {e}")
            finally:
                # Restore model (outside no_grad scope)
                restore_after_codec()

        logger.info(f"Voice '{name}' registered: {audio_path} ({len(audio_bytes)} bytes)")

        return JSONResponse(
            status_code=201,
            content={
                "name": name,
                "transcription": transcription,
                "audio_file": str(audio_path.name),
                "audio_size": len(audio_bytes),
                "message": f"Voice '{name}' created successfully",
            },
        )

    except Exception as e:
        # Cleanup on failure
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
        raise HTTPException(500, f"Failed to save voice: {e}")


@app.get("/v1/voices")
async def list_voices():
    """List all registered reference voices."""
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    voices = []
    for d in sorted(REFERENCES_DIR.iterdir()):
        if not d.is_dir():
            continue

        meta_file = d / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        else:
            meta = {"name": d.name, "transcription": ""}

        # Find audio file
        audio_file = None
        for ext in (".wav", ".mp3", ".flac", ".ogg"):
            candidates = list(d.glob(f"*{ext}"))
            if candidates:
                audio_file = candidates[0].name
                break

        voices.append({
            "name": d.name,
            "transcription": meta.get("transcription", ""),
            "audio_file": audio_file,
            "has_audio": audio_file is not None,
        })

    return {"voices": voices, "total": len(voices)}


@app.get("/v1/voices/{name}")
async def get_voice(name: str):
    """Get details for a specific voice."""
    voice_dir = REFERENCES_DIR / name
    if not voice_dir.exists():
        raise HTTPException(404, f"Voice '{name}' not found")

    meta_file = voice_dir / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    else:
        meta = {"name": name, "transcription": ""}

    audio_file = None
    audio_size = 0
    for ext in (".wav", ".mp3", ".flac", ".ogg"):
        candidates = list(voice_dir.glob(f"*{ext}"))
        if candidates:
            audio_file = candidates[0].name
            audio_size = candidates[0].stat().st_size
            break

    return {
        "name": name,
        "transcription": meta.get("transcription", ""),
        "audio_file": audio_file,
        "audio_size": audio_size,
    }


@app.delete("/v1/voices/{name}")
async def delete_voice(name: str):
    """Delete a registered voice."""
    voice_dir = REFERENCES_DIR / name
    if not voice_dir.exists():
        raise HTTPException(404, f"Voice '{name}' not found")

    shutil.rmtree(voice_dir)
    logger.info(f"Voice '{name}' deleted")
    return {"message": f"Voice '{name}' deleted successfully"}


@app.post("/v1/tts")
async def tts(req: TTSRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if req.streaming:
        raise HTTPException(
            status_code=400,
            detail="Streaming not supported in GGUF mode. Set streaming=false."
        )

    # Serialize generation (single GPU, no concurrent inference)
    with state.lock:
        try:
            t0 = time.perf_counter()
            audio_np = generate_speech(req)
            t_total = time.perf_counter() - t0
            duration = len(audio_np) / state.sample_rate
            logger.info(f"TTS complete: {duration:.1f}s audio in {t_total:.1f}s "
                        f"(RTF={t_total/duration:.2f})")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.opt(exception=True).error("Generation failed: {}", str(e))
            raise HTTPException(status_code=500, detail="Generation failed")

    # Encode to requested format
    buffer = io.BytesIO()
    try:
        sf.write(buffer, audio_np, state.sample_rate, format=req.format)
    except Exception:
        # Fallback to wav if format not supported
        sf.write(buffer, audio_np, state.sample_rate, format="wav")
        req.format = "wav"

    content_type_map = {
        "wav": "audio/wav",
        "flac": "audio/flac",
        "mp3": "audio/mpeg",
    }

    return Response(
        content=buffer.getvalue(),
        media_type=content_type_map.get(req.format, "audio/wav"),
        headers={
            "Content-Disposition": f"attachment; filename=speech.{req.format}",
            "X-Audio-Duration": f"{duration:.2f}",
            "X-Generation-Time": f"{t_total:.2f}",
        },
    )


@app.get("/v1/models")
async def list_models():
    return {
        "data": [{
            "id": "fish-speech-s2-pro-gguf",
            "object": "model",
            "owned_by": "fishaudio",
            "meta": {
                "quantization": "Q6_K",
                "backend": "triton-fused-gemv",
                "device": str(state.device),
            }
        }]
    }


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fish Speech GGUF API Server")
    parser.add_argument("--gguf-path", type=str, required=True,
                        help="Path to Q6_K GGUF model file")
    parser.add_argument("--codec-path", type=str, required=True,
                        help="Path to codec.pth")
    parser.add_argument("--listen", type=str, default="0.0.0.0:7820",
                        help="Host:port to listen on")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Max sequence length (lower = less VRAM)")
    parser.add_argument("--max-text-length", type=int, default=0,
                        help="Max input text length (0 = unlimited)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Optional API key for authentication")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of workers (keep 1 for single GPU)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # API key middleware
    if args.api_key:
        @app.middleware("http")
        async def auth_middleware(request, call_next):
            if request.url.path == "/v1/health":
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != args.api_key:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
            return await call_next(request)

    # Max text length check
    if args.max_text_length > 0:
        state._max_text_length = args.max_text_length

    # Check port availability before expensive model loading
    import socket
    host, port = args.listen.rsplit(":", 1)
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    port = int(port)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host if host != "0.0.0.0" else "", port))
    except OSError as e:
        logger.error(f"Port {port} is already in use: {e}")
        sys.exit(1)

    # Load models (heavy operation)
    load_models(args)

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=args.workers, log_level="info")
