"""
GGUF model + codec loading and VRAM management.
"""

import gc
import time

import torch
import gguf
from loguru import logger

from .codec_manager import load_codec_from_gguf
from .state import MODELS_DIR, save_server_state, state
from .tts_engine import invalidate_decode_cache


def load_models(args, name=None):
    """Load GGUF model and codec dynamically."""
    from fish_speech.gguf import load_gguf_model

    if not name:
        logger.info("No model specified to load at startup.")
        return

    gguf_path = MODELS_DIR / name / "model.gguf"
    if not gguf_path.exists():
        raise FileNotFoundError(f"Model file not found for {name}")

    # Unload existing
    if state.model is not None:
        logger.info("Unloading current model...")
        invalidate_decode_cache()
        del state.model
        del state.codec
        state.model = None
        state.codec = None
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

    # Load codec
    if args.codec_path:
        logger.info(f"Loading codec from {args.codec_path}...")
        t0 = time.perf_counter()
        from fish_speech.models.text2semantic.inference import load_codec_model

        codec = load_codec_model(args.codec_path, "cpu", precision=torch.float32)
        # Fix inference-mode tensors
        for pname, param in list(codec.named_parameters()):
            parts = pname.split(".")
            module = codec
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(
                module,
                parts[-1],
                torch.nn.Parameter(param.data.clone(), requires_grad=False),
            )
        for bname, buf in list(codec.named_buffers()):
            parts = bname.split(".")
            module = codec
            for part in parts[:-1]:
                module = getattr(module, part)
            module.register_buffer(parts[-1], buf.data.clone())
        logger.info(f"Codec loaded from .pth in {time.perf_counter() - t0:.1f}s")
    else:
        logger.info("Loading codec from GGUF tensors...")
        t0 = time.perf_counter()
        codec = load_codec_from_gguf(model, device="cpu")
        logger.info(f"Codec loaded from GGUF in {time.perf_counter() - t0:.1f}s")

    # Update global state
    state.model = model
    state.tokenizer = model.tokenizer
    state.codec = codec
    state.sample_rate = codec.sample_rate
    state.device = args.device
    state.max_seq_len = args.max_seq_len
    state.active_model_name = name
    state.ready = True
    save_server_state()

    # Try to keep codec on GPU alongside model
    _try_codec_gpu_resident(codec, args.device)

    # Enable custom_op mode for GGUFLinear
    try:
        from fish_speech.gguf.dequant import enable_custom_op_mode

        enable_custom_op_mode(model)
        logger.info("Custom op mode enabled for GGUFLinear")
    except Exception as e:
        logger.warning(f"Could not enable custom_op mode: {e}")

    # Pre-compile Triton kernels
    from fish_speech.gguf.dequant import warmup_triton_kernels

    # ---- kernel warmup (CUDA kernels or Triton) ----
    try:
        from fish_speech.gguf.cuda_kernels import _CUDA_KERNELS_AVAILABLE
    except ImportError:
        _CUDA_KERNELS_AVAILABLE = False

    if _CUDA_KERNELS_AVAILABLE:
        logger.info("CUDA kernels available — warming up CUDA GEMV kernels...")
        t0 = time.perf_counter()
        _warmup_cuda_kernels(model)
        logger.info(f"CUDA kernel warmup done in {time.perf_counter() - t0:.1f}s")
    else:
        # Fallback to Triton warmup
        from fish_speech.gguf.dequant import warmup_triton_kernels
        logger.info("Warming up Triton kernels...")
        t0 = time.perf_counter()
        warmup_triton_kernels(model, dtype=torch.float16)
        logger.info(f"Triton warmup done in {time.perf_counter() - t0:.1f}s")

    # ---- Full pipeline warmup: dummy prefill + CUDA Graph pre-capture ----
    # This triggers all Triton JIT compilations (batch-mode dequant, etc.)
    # and pre-captures the CUDA Graph, eliminating first-request latency.
    t0 = time.perf_counter()
    _warmup_full_pipeline(model, args.device, torch.float16)
    logger.info(f"Full pipeline warmup done in {time.perf_counter() - t0:.1f}s")

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Ready. VRAM: {vram:.2f} GB")


def _try_codec_gpu_resident(codec, device):
    """Move codec to GPU if VRAM allows, avoiding costly CPU↔GPU swap."""
    if not (torch.cuda.is_available() and device == "cuda"):
        return

    model_vram = torch.cuda.memory_allocated() / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    n_params = sum(p.numel() for p in codec.parameters())
    param_size_f16 = n_params * 2
    buffer_size_gpu = sum(
        b.numel() * b.element_size() for b in codec.buffers()
    )
    codec_size_gpu = (param_size_f16 + buffer_size_gpu) / 1e9
    codec_size_current = (
        sum(p.numel() * p.element_size() for p in codec.parameters())
        + buffer_size_gpu
    ) / 1e9
    n_buffers = sum(b.numel() for b in codec.buffers())

    logger.info(
        f"Codec size: {n_params / 1e6:.1f}M params + {n_buffers / 1e6:.1f}M buffers, "
        f"current={codec_size_current:.2f} GB, GPU(F16)≈{codec_size_gpu:.2f} GB"
    )

    headroom = 1.0  # 1 GB for KV cache, activations, etc.

    if model_vram + codec_size_gpu + headroom < total_vram:
        logger.info(
            f"Moving codec to GPU (model={model_vram:.2f} GB + "
            f"codec≈{codec_size_gpu:.2f} GB, total VRAM={total_vram:.1f} GB)"
        )
        t0 = time.perf_counter()
        codec = codec.to(device="cuda", dtype=torch.float16)
        codec.eval()
        state.codec = codec
        state.codec_gpu_resident = True
        vram_now = torch.cuda.memory_allocated() / 1e9
        logger.info(
            f"Codec on GPU in {time.perf_counter() - t0:.1f}s "
            f"(VRAM: {vram_now:.2f} GB)"
        )
    else:
        logger.info(
            f"Codec stays on CPU (model={model_vram:.2f} GB + "
            f"codec≈{codec_size_gpu:.2f} GB would exceed "
            f"{total_vram:.1f} GB VRAM)"
        )
        state.codec_gpu_resident = False

def _warmup_full_pipeline(model, device, dtype):
    """Run dummy prefill + CUDA Graph capture at startup.

    This eliminates first-request latency by:
      1. Triggering all Triton JIT compilations (batch-mode dequant, embedding, etc.)
      2. Running CUDAGraphRunner.warmup_and_capture() which:
         - Phase 0/0b: Materializes CPU tensors on GPU, caches GGUFEmbedding weights
         - Phase 0c: Creates Attention._neg_inf on GPU
         - Phase 1+2: Warms up and captures the CUDA Graph
      3. Storing the captured graph in the decode cache for immediate reuse
    """
    from fish_speech.models.text2semantic.inference import (
        CUDAGraphRunner,
        decode_one_token_ar,
    )
    from fish_speech.tokenizer import IM_END_TOKEN
    from .tts_engine import _cache

    codebook_dim = 1 + model.config.num_codebooks
    vocab_size = model.config.vocab_size

    # Build semantic_logit_bias (same logic as _DecodeCache.ensure_for_model)
    semantic_logit_bias = torch.full(
        (1, 1, vocab_size), float("-inf"), device=device, dtype=dtype
    )
    semantic_logit_bias[
        0, 0,
        model.config.semantic_begin_id : model.config.semantic_end_id + 1,
    ] = 0.0
    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)
    semantic_logit_bias[0, 0, im_end_id] = 0.0

    # Dummy prefill — triggers Triton batch-mode JIT compilation
    warmup_len = 32
    dummy_tokens = torch.zeros(
        (1, codebook_dim, warmup_len), dtype=torch.long, device=device
    )
    input_pos = torch.arange(warmup_len, device=device, dtype=torch.long)
    temperature = torch.tensor(1.0, device=device, dtype=dtype)
    top_p = torch.tensor(0.9, device=device, dtype=dtype)
    previous_tokens = torch.zeros(
        (codebook_dim, 10), dtype=torch.int, device=device
    )

    logger.info("Running dummy prefill for Triton JIT warmup...")
    with torch.inference_mode():
        first_token = decode_one_token_ar(
            model=model,
            x=dummy_tokens,
            input_pos=input_pos,
            temperature=temperature,
            top_p=top_p,
            top_k=30,
            semantic_logit_bias=semantic_logit_bias,
            audio_masks=None,
            audio_parts=None,
            previous_tokens=previous_tokens,
        )

    # Pre-capture CUDA Graph
    fixed_cur_token = first_token.view(1, codebook_dim, 1).clone()
    fixed_input_pos = torch.tensor([warmup_len], device=device, dtype=torch.long)

    try:
        graph_runner = CUDAGraphRunner(
            model=model,
            decode_fn=decode_one_token_ar,
            codebook_dim=codebook_dim,
            device=torch.device(device),
            dtype=dtype,
            vocab_size=vocab_size,
            top_k=30,
            semantic_logit_bias=semantic_logit_bias,
            audio_masks=None,
            audio_parts=None,
        )
        graph_runner.static_x.copy_(fixed_cur_token)
        graph_runner.static_input_pos.copy_(fixed_input_pos)
        graph_runner.static_temperature.copy_(temperature)
        graph_runner.static_top_p.copy_(top_p)
        graph_runner.warmup_and_capture()

        # Store in decode cache for immediate reuse by first TTS request
        _cache.model_id = id(model)
        _cache.semantic_logit_bias = semantic_logit_bias
        _cache.im_end_id = im_end_id
        _cache.graph_runner = graph_runner
        logger.info("[Warmup] CUDA Graph pre-captured and cached")
    except Exception as e:
        logger.warning(f"[Warmup] CUDA Graph pre-capture failed: {e}")

    # Zero KV caches — remove warmup artifacts
    for layer in model.layers:
        kv = layer.attention.kv_cache
        if kv is not None:
            kv.k_cache.zero_()
            kv.v_cache.zero_()

    # Also zero fast transformer caches if present
    if hasattr(model, 'fast_layers'):
        for layer in model.fast_layers:
            if hasattr(layer, 'attention') and layer.attention.kv_cache is not None:
                layer.attention.kv_cache.k_cache.zero_()
                layer.attention.kv_cache.v_cache.zero_()

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _warmup_cuda_kernels(model):
    """Pre-run CUDA GEMV kernels for each unique (D_out, D_in, qtype) shape.

    This warms up:
      1. The JIT-compiled CUDA extension (already compiled at import time)
      2. The persistent Q8_1 buffer pool (first call grows to max D_in)
      3. CUDA driver caches for multi-warp kernel launches (4 warps/row)
    """
    from fish_speech.gguf.dequant import GGUFLinear
    from fish_speech.gguf.cuda_kernels import (
        fused_gemv_q6k,
        fused_gemv_q3k,
    )

    seen = set()
    for module in model.modules():
        if not isinstance(module, GGUFLinear):
            continue
        qp = module.qparam
        if not qp.data.is_cuda:
            continue
        key = (qp.shape[0], qp.shape[1], qp.qtype.name)
        if key in seen:
            continue
        seen.add(key)

        D_out, D_in = qp.shape
        if D_in % 256 != 0:
            continue

        x_dummy = torch.randn(D_in, dtype=torch.float32, device=qp.data.device)

        if qp.qtype == gguf.GGMLQuantizationType.Q6_K:
            fused_gemv_q6k(x_dummy, qp.data, D_in, D_out)
        elif qp.qtype == gguf.GGMLQuantizationType.Q3_K:
            fused_gemv_q3k(x_dummy, qp.data, D_in, D_out)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info(f"CUDA kernel warmup: {len(seen)} unique shapes")
