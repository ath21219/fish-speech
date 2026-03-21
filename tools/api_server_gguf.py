"""
Fish Speech GGUF API Server
Triton-accelerated Q6_K inference with OpenAI-compatible TTS endpoint.

Usage:
  python tools/api_server_gguf.py \
    --model-name s2-pro-q6_k \
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
    Provide codec on GPU for encode/decode operations.

    When codec_gpu_resident is enabled (default for Q3_K and other small models),
    codec stays on GPU permanently alongside the model — no swap needed.
    Falls back to model↔codec swap when VRAM is too tight.
    """
    if state.codec_gpu_resident:
        # Codec already on GPU, just yield it
        yield state.codec
        return

    # Legacy swap path for large models that don't fit with codec
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
    codec.to(device=device)
    torch.cuda.synchronize()
    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Codec on GPU in {time.perf_counter() - t0:.1f}s (VRAM: {vram:.2f} GB)")

    try:
        yield codec
    finally:
        pass


def restore_after_codec():
    """Move codec back to CPU and restore GGUF model to GPU."""
    if state.codec_gpu_resident:
        # Nothing to restore — both model and codec stay on GPU
        return

    model = state.model
    codec = state.codec
    device = state.device

    logger.info("Offloading codec to CPU...")
    codec.to(device="cpu")
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
        self.active_model_name = None
        self.codec_gpu_resident = False  # True when codec stays on GPU permanently

state = GGUFServerState()
app = FastAPI(title="Fish Speech GGUF API", version="1.0.0")
MODELS_DIR = Path("models")


# ============================================================
# Startup (from working code, unchanged)
# ============================================================

def load_models(args, name=None):
    """Load GGUF model and codec dynamically."""
    from fish_speech.gguf import load_gguf_model
    from fish_speech.gguf.dequant import dequantize_tensor

    if not name:
        logger.info("No model specified to load at startup.")
        return

    gguf_path = MODELS_DIR / name / "model.gguf"
    if not gguf_path.exists():
        raise FileNotFoundError(f"Model file not found for {name}")

    # Unload existing
    if state.model is not None:
        logger.info("Unloading current model...")
        del state.model
        del state.codec
        state.model = None
        state.codec = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Loading GGUF model from {gguf_path}...")
    t0 = time.perf_counter()
    model = load_gguf_model(
        gguf_path=str(gguf_path),
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

    # -------------------------------------------------------
    # Load codec from GGUF tensors (no codec.pth needed)
    # -------------------------------------------------------
    if args.codec_path:
        # Legacy path: load from external codec.pth
        logger.info(f"Loading codec from {args.codec_path}...")
        t0 = time.perf_counter()
        from fish_speech.models.text2semantic.inference import load_codec_model
        codec = load_codec_model(args.codec_path, "cpu", precision=torch.float32)
        # Fix inference-mode tensors (existing code)
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
        logger.info(f"Codec loaded from .pth in {time.perf_counter() - t0:.1f}s")
    else:
        # New path: load from GGUF-embedded tensors
        logger.info("Loading codec from GGUF tensors...")
        t0 = time.perf_counter()
        codec = _load_codec_from_gguf(model, device="cpu")
        logger.info(f"Codec loaded from GGUF in {time.perf_counter() - t0:.1f}s")

    state.model = model
    state.tokenizer = model.tokenizer
    state.codec = codec
    state.sample_rate = codec.sample_rate
    state.device = args.device
    state.max_seq_len = args.max_seq_len
    state.active_model_name = name
    state.ready = True

    # Try to keep codec on GPU alongside model (avoids costly CPU↔GPU swap)
    if torch.cuda.is_available() and args.device == "cuda":
        model_vram = torch.cuda.memory_allocated() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Estimate codec GPU size: params→FP16 (2B), buffers keep dtype (bool=1B, float=2-4B)
        n_params = sum(p.numel() for p in codec.parameters())
        param_size_f16 = n_params * 2  # params will be cast to FP16
        buffer_size_gpu = sum(b.numel() * b.element_size() for b in codec.buffers())  # buffers keep dtype
        codec_size_gpu = (param_size_f16 + buffer_size_gpu) / 1e9
        codec_size_current = (sum(p.numel() * p.element_size() for p in codec.parameters())
                              + buffer_size_gpu) / 1e9
        n_buffers = sum(b.numel() for b in codec.buffers())
        logger.info(f"Codec size: {n_params/1e6:.1f}M params + {n_buffers/1e6:.1f}M buffers, "
                    f"current={codec_size_current:.2f} GB, GPU(F16)≈{codec_size_gpu:.2f} GB")
        headroom = 1.0  # 1 GB headroom for KV cache, activations, etc.

        if model_vram + codec_size_gpu + headroom < total_vram:
            logger.info(f"Moving codec to GPU (model={model_vram:.2f} GB + "
                        f"codec≈{codec_size_gpu:.2f} GB, total VRAM={total_vram:.1f} GB)")
            t0 = time.perf_counter()
            codec = codec.to(device="cuda", dtype=torch.float16)
            codec.eval()
            state.codec = codec
            state.codec_gpu_resident = True
            vram_now = torch.cuda.memory_allocated() / 1e9
            logger.info(f"Codec on GPU in {time.perf_counter() - t0:.1f}s "
                        f"(VRAM: {vram_now:.2f} GB)")
        else:
            logger.info(f"Codec stays on CPU (model={model_vram:.2f} GB + "
                        f"codec≈{codec_size_gpu:.2f} GB would exceed "
                        f"{total_vram:.1f} GB VRAM)")
            state.codec_gpu_resident = False

    # Pre-compile Triton kernels to avoid JIT latency on first inference
    from fish_speech.gguf.dequant import warmup_triton_kernels
    logger.info("Warming up Triton kernels...")
    t0 = time.perf_counter()
    warmup_triton_kernels(model, dtype=torch.float16)
    logger.info(f"Triton warmup done in {time.perf_counter() - t0:.1f}s")

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Ready. VRAM: {vram:.2f} GB")


def _load_codec_from_gguf(model, device="cpu"):
    """
    Build codec model structure via Hydra config, then load weights
    from GGUF tensors stored in model._gguf_codec_tensors.

    This replaces load_codec_model() + codec.pth, saving ~3.5 GB of
    disk/memory overhead (codec.pth is FP32, GGUF has F16).
    """
    import gguf as gguf_lib
    from fish_speech.gguf.dequant import dequantize_tensor, NATIVE_TORCH_QTYPES, DEQUANT_FN
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    codec_tensors = getattr(model, '_gguf_codec_tensors', None)
    if not codec_tensors:
        raise RuntimeError(
            "No codec tensors found in GGUF. "
            "Make sure the GGUF file contains c.* tensors."
        )

    # 1. Instantiate empty codec model from config
    config_candidates = [
        project_root / "fish_speech" / "configs" / "modded_dac_vq.yaml",
        project_root / "configs" / "modded_dac_vq.yaml",
        Path("/app/fish_speech/configs/modded_dac_vq.yaml"),
        Path("/app/configs/modded_dac_vq.yaml"),
    ]

    # Also try resolving from the fish_speech package if possible
    try:
        import fish_speech
        if hasattr(fish_speech, '__file__') and fish_speech.__file__ is not None:
            config_candidates.insert(0,
                Path(fish_speech.__file__).parent / "configs" / "modded_dac_vq.yaml"
            )
        elif hasattr(fish_speech, '__path__'):
            for p in fish_speech.__path__:
                config_candidates.insert(0,
                    Path(p) / "configs" / "modded_dac_vq.yaml"
                )
    except Exception:
        pass

    config_path = None
    for candidate in config_candidates:
        if candidate.exists():
            config_path = candidate
            break

    if config_path is None:
        searched = "\n  ".join(str(c) for c in config_candidates)
        raise FileNotFoundError(
            f"Codec config modded_dac_vq.yaml not found. Searched:\n  {searched}"
        )

    logger.debug(f"Using codec config: {config_path}")
    cfg = OmegaConf.load(str(config_path))
    codec = instantiate(cfg)
    codec.eval()

    # 2. Remove all weight_norm parametrizations so state_dict keys become plain "weight"
    #    The codec uses TWO weight_norm APIs:
    #    - New API (torch.nn.utils.parametrizations.weight_norm) -> parametrizations.weight.original0/1
    #    - Old API (torch.nn.utils.weight_norm) -> weight_g/weight_v
    #    GGUF stores combined "weight" tensors, so we remove weight_norm to match.
    from torch.nn.utils.parametrize import remove_parametrizations
    wn_removed = 0
    for name, module in codec.named_modules():
        # New parametrizations API: check for parametrizations attribute
        if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
            try:
                remove_parametrizations(module, 'weight')
                wn_removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove parametrization from {name}: {e}")
        # Old weight_norm API: check for weight_g/weight_v attributes
        elif hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
            try:
                torch.nn.utils.remove_weight_norm(module)
                wn_removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove weight_norm from {name}: {e}")
    logger.info(f"Removed weight_norm from {wn_removed} modules")

    # 3. Build state dict from GGUF codec tensors
    gguf_state_dict = {}
    for gguf_name, (raw_data, qtype, gguf_shape) in codec_tensors.items():
        # Strip "c." prefix: "c.decoder.model.0.conv.weight" -> "decoder.model.0.conv.weight"
        if gguf_name.startswith("c."):
            param_name = gguf_name[2:]
        else:
            param_name = gguf_name

        if qtype in NATIVE_TORCH_QTYPES:
            if qtype == gguf_lib.GGMLQuantizationType.F16:
                tensor = raw_data.view(torch.float16).reshape(gguf_shape)
            elif qtype == gguf_lib.GGMLQuantizationType.F32:
                tensor = raw_data.view(torch.float32).reshape(gguf_shape)
            else:
                tensor = raw_data.view(torch.bfloat16).reshape(gguf_shape)
        elif qtype in DEQUANT_FN:
            tensor = dequantize_tensor(raw_data, qtype, gguf_shape, dtype=torch.float16)
        else:
            logger.warning(f"Skipping codec tensor {param_name}: unsupported qtype {qtype}")
            continue

        gguf_state_dict[param_name] = tensor.to(device)

    # 4. Load into codec model (weight_norm removed, so keys should match directly)
    codec_state = codec.state_dict()
    loaded = 0
    skipped = []

    for param_name, param_tensor in gguf_state_dict.items():
        if param_name in codec_state:
            target_shape = codec_state[param_name].shape
            if target_shape == param_tensor.shape:
                codec_state[param_name] = param_tensor
                loaded += 1
            elif param_tensor.numel() == codec_state[param_name].numel():
                codec_state[param_name] = param_tensor.reshape(target_shape)
                loaded += 1
            elif target_shape == param_tensor.T.shape and param_tensor.dim() == 2:
                codec_state[param_name] = param_tensor.T
                loaded += 1
            else:
                skipped.append(
                    f"{param_name}: GGUF {param_tensor.shape} ({param_tensor.numel()}) vs "
                    f"model {target_shape} ({codec_state[param_name].numel()})"
                )
        else:
            skipped.append(f"{param_name}: not in model state_dict")

    codec.load_state_dict(codec_state, strict=False)

    logger.info(
        f"Codec: loaded {loaded}/{len(gguf_state_dict)} tensors from GGUF, "
        f"{len(skipped)} skipped"
    )
    if skipped:
        for s in skipped[:30]:
            logger.warning(f"  Codec skip: {s}")
        if len(skipped) > 30:
            logger.warning(f"  ... and {len(skipped) - 30} more")

    # 5. Shrink oversized Transformer buffers (causal_mask, freqs_cis)
    # The Transformer class hardcodes 32768×32768 causal masks (~1 GB each × 3 instances).
    # For codec inference we only need a few hundred frames max.
    max_codec_frames = 2048  # ~24s at 44100Hz/512hop, more than enough
    shrunk_buffers = 0
    for name, module in codec.named_modules():
        if hasattr(module, 'causal_mask') and module.causal_mask is not None:
            old_size = module.causal_mask.shape[0]
            if old_size > max_codec_frames:
                new_mask = module.causal_mask[:max_codec_frames, :max_codec_frames].clone()
                module.register_buffer("causal_mask", new_mask, persistent=False)
                shrunk_buffers += 1
        if hasattr(module, 'freqs_cis') and module.freqs_cis is not None:
            if module.freqs_cis.shape[0] > max_codec_frames:
                new_freqs = module.freqs_cis[:max_codec_frames].clone()
                module.register_buffer("freqs_cis", new_freqs, persistent=False)
                shrunk_buffers += 1
    if shrunk_buffers > 0:
        logger.info(f"Shrunk {shrunk_buffers} Transformer buffers to max_frames={max_codec_frames}")

    # 6. Move to target device/dtype
    # CPU: float32 (CPUs lack efficient float16 compute)
    # GPU: float16 (saves VRAM, GPU has native FP16 support)
    target_dtype = torch.float32 if device == "cpu" else torch.float16
    codec = codec.to(device=device, dtype=target_dtype)

    # Verify essential attributes
    if not hasattr(codec, 'sample_rate'):
        # Fall back to GGUF metadata
        meta = getattr(model, '_gguf_metadata', {})
        sr = meta.get('fish_speech.codec.sample_rate', 44100)
        codec.sample_rate = int(sr) if isinstance(sr, (int, float)) else 44100
        logger.info(f"Set codec.sample_rate = {codec.sample_rate} from GGUF metadata")

    return codec


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

        # Decode audio from VQ codes
        with torch.inference_mode():
            codec_dev = next(state.codec.parameters()).device
            logger.info(f"Decoding: codes shape={codes.shape}, "
                        f"codec device={codec_dev}, "
                        f"codec dtype={next(state.codec.parameters()).dtype}")
            t_dec = time.perf_counter()
            codes_for_decode = codes.to(codec_dev)
            audio_segment = decode_to_audio(codes_for_decode, state.codec)
            logger.info(f"Codec decode in {time.perf_counter() - t_dec:.2f}s")
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
        status="ok" if state.ready else "loading" if state.active_model_name else "empty",
        model=state.active_model_name or "",
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




@app.post("/v1/models")
async def create_model(
    name: str = Form(...),
    config: str = Form(""),
    tokenizer: str = Form(""),
    model: UploadFile = File(...),
):
    """Register a reference model."""
    # Validate name
    if not name or not name.strip():
        raise HTTPException(400, "Model name cannot be empty")
    name = name.strip()
    if not all(c.isalnum() or c in "-_ " for c in name):
        raise HTTPException(400, "Model name can only contain alphanumeric, hyphen, underscore, space")

    model_dir = MODELS_DIR / name
    if model_dir.exists():
        raise HTTPException(409, f"Model '{name}' already exists")

    # Read and validate model
    model_bytes = await model.read()
    if len(model_bytes) < 1000:
        raise HTTPException(400, "Model file too small or empty")

    # Determine extension from filename
    ext = Path(model.filename).suffix.lower() if model.filename else ".safetensors"
    if ext not in (".safetensors", ".gguf"):
        ext = ".safetensors"

    # Save
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        model_path = model_dir / f"model{ext}"
        model_path.write_bytes(model_bytes)

        (model_dir / "config.json").write_text(config)
        (model_dir / "tokenizer.json").write_text(tokenizer)

        logger.info(f"Model '{name}' registered: {model_path} ({len(model_bytes)} bytes)")

        return JSONResponse(
            status_code=201,
            content={
                "name": name,
                "model_file": str(model_path.name),
                "model_size": len(model_bytes),
                "message": f"Model '{name}' created successfully",
            },
        )

    except Exception as e:
        # Cleanup on failure
        if model_dir.exists():
            shutil.rmtree(model_dir)
        raise HTTPException(500, f"Failed to save model: {e}")

@app.get("/v1/models")
async def list_models():
    models = []
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir():
                models.append({
                    "id": d.name,
                    "object": "model"
                })
    return {"data": models}

@app.get("/v1/models/{name}")
async def get_model(name: str):
    d = MODELS_DIR / name
    meta = {}
    if d.is_dir():
        meta_file = d / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "id": name,
            "object": "model",
            "owned_by": "fishaudio",
            "meta": meta
        }
    elif (MODELS_DIR / f"{name}.gguf").exists():
        return {
            "id": name,
            "object": "model",
            "owned_by": "fishaudio",
            "meta": {}
        }
    raise HTTPException(404, f"Model '{name}' not found")

@app.post("/v1/models/{name}/load")
async def load_model_endpoint(name: str):
    global args
    try:
        load_models(args, name)
        return {"message": f"Model '{name}' loaded successfully", "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0}
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{name}' not found")
    except Exception as e:
        logger.exception("Failed to load model")
        raise HTTPException(500, f"Failed to load model: {str(e)}")

@app.post("/v1/models/{name}/unload")
async def unload_model_endpoint(name: str):
    if state.active_name != name:
        raise HTTPException(400, f"Model '{name}' is not currently loaded")
    
    logger.info("Unloading current model...")
    state.ready = False
    state.active_name = None
    if state.model is not None:
        del state.model
        del state.codec
        state.model = None
        state.codec = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return {"message": f"Model '{name}' unloaded successfully"}

@app.delete("/v1/models/{name}")
async def delete_model(name: str):
    d = MODELS_DIR / name
    if d.is_dir():
        shutil.rmtree(d)
        if state.active_name == name:
            await unload_model_endpoint(name)
        return {"message": f"Model '{name}' deleted successfully"}
    elif (MODELS_DIR / f"{name}.gguf").exists():
        (MODELS_DIR / f"{name}.gguf").unlink()
        if state.active_name == name:
            await unload_model_endpoint(name)
        return {"message": f"Model '{name}' deleted successfully"}
    raise HTTPException(404, f"Model '{name}' not found")




# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fish Speech GGUF API Server")
    parser.add_argument("--model-name", type=str, default="s2-pro-q6_k",
                        help="Model to load on startup")
    parser.add_argument("--codec-path", type=str, default=None,
                        help="Path to codec.pth (optional: uses GGUF-embedded codec if omitted)")
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

    # Initialize models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load models (heavy operation)
    if args.model_name:
        load_models(args, args.model_name)
    else:
        logger.info("Starting without an active model. Use POST /v1/models/{id}/load to load one.")

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=args.workers, log_level="info")
