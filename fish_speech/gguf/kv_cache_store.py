"""
Persistent KV cache storage for reference voice prefixes.

Saves the KV cache state after processing the system prompt + reference
audio tokens, enabling subsequent requests to skip this expensive prefill step.

Typical savings:
  - Reference with ~200 VQ tokens → ~14 MB cache file
  - Prefill skip: ~0.3-0.5s saved per batch

Cache format (kv_cache.pt):
  {
      "cache_key": str,         # SHA-256 hash of encoded prefix tokens
      "prefix_len": int,        # Number of tokens in cached prefix
      "k_caches": [Tensor, ...],  # Per-layer K cache (batch, heads, prefix_len, head_dim)
      "v_caches": [Tensor, ...],  # Per-layer V cache
  }
"""

import hashlib
from pathlib import Path
from typing import Optional

import torch
from loguru import logger


REFERENCES_DIR = Path("references")


def compute_cache_key(encoded_prefix: torch.Tensor) -> str:
    """
    Hash encoded prefix tokens to create a cache key.

    Args:
        encoded_prefix: (num_codebooks+1, prefix_len) tensor of token IDs + VQ codes
    """
    data = encoded_prefix.cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def save_kv_cache(
    ref_id: str,
    model,
    prefix_len: int,
    cache_key: str,
):
    """
    Extract KV cache for positions 0..prefix_len-1 from all slow transformer
    layers and save to disk.

    Args:
        ref_id: Reference voice ID (directory name under references/)
        model: DualARTransformer with populated KV caches
        prefix_len: Number of prefix tokens to save
        cache_key: Hash of the encoded prefix for cache validation
    """
    cache_path = REFERENCES_DIR / ref_id / "kv_cache.pt"

    k_caches = []
    v_caches = []

    for layer in model.layers:
        kv = layer.attention.kv_cache
        if kv is None:
            logger.warning("KV cache not initialized, cannot save")
            return
        # Extract prefix portion, preserving the model's native KV dtype
        k_caches.append(kv.k_cache[:, :, :prefix_len, :].cpu().clone())
        v_caches.append(kv.v_cache[:, :, :prefix_len, :].cpu().clone())

    cache_data = {
        "cache_key": cache_key,
        "prefix_len": prefix_len,
        "k_caches": k_caches,
        "v_caches": v_caches,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_data, cache_path)

    size_mb = cache_path.stat().st_size / 1e6
    logger.info(
        f"KV cache saved: ref={ref_id}, "
        f"{prefix_len} tokens, {len(k_caches)} layers, {size_mb:.1f} MB"
    )


def load_kv_cache(
    ref_id: str,
    model,
    cache_key: str,
    device: torch.device,
) -> Optional[int]:
    """
    Load KV cache from disk into model's KV cache buffers.

    Returns:
        prefix_len on success, None on cache miss or validation failure.
    """
    cache_path = REFERENCES_DIR / ref_id / "kv_cache.pt"
    if not cache_path.exists():
        return None

    try:
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception as e:
        logger.warning(f"KV cache load failed for {ref_id}: {e}")
        return None

    # Validate cache key
    if data.get("cache_key") != cache_key:
        logger.info(
            f"KV cache key mismatch for {ref_id} "
            f"(stored={data.get('cache_key')}, expected={cache_key}), "
            f"will regenerate"
        )
        return None

    prefix_len = data["prefix_len"]
    k_caches = data["k_caches"]
    v_caches = data["v_caches"]

    # Validate layer count
    n_layers = len(list(model.layers))
    if len(k_caches) != n_layers:
        logger.warning(
            f"KV cache layer count mismatch: "
            f"cached={len(k_caches)}, model={n_layers}"
        )
        return None

    # Validate shape compatibility with first layer
    first_kv = list(model.layers)[0].attention.kv_cache
    if first_kv is None:
        logger.warning("Model KV cache not initialized, cannot load")
        return None

    expected_shape = first_kv.k_cache[:, :, :prefix_len, :].shape
    if k_caches[0].shape != expected_shape:
        logger.warning(
            f"KV cache shape mismatch: "
            f"cached={k_caches[0].shape}, expected={expected_shape}"
        )
        return None

    # Load into model's KV cache buffers
    model_dtype = first_kv.k_cache.dtype
    for layer, k_saved, v_saved in zip(model.layers, k_caches, v_caches):
        kv = layer.attention.kv_cache
        kv.k_cache[:, :, :prefix_len, :] = k_saved.to(
            device=device, dtype=model_dtype
        )
        kv.v_cache[:, :, :prefix_len, :] = v_saved.to(
            device=device, dtype=model_dtype
        )

    logger.info(f"KV cache loaded: ref={ref_id}, {prefix_len} tokens")
    return prefix_len


def invalidate_kv_cache(ref_id: str):
    """Remove cached KV data for a reference (e.g. when voice is re-registered)."""
    cache_path = REFERENCES_DIR / ref_id / "kv_cache.pt"
    if cache_path.exists():
        cache_path.unlink()
        logger.info(f"KV cache invalidated: {ref_id}")


def precompile_kv_cache(
    ref_id: str,
    codes: torch.Tensor,
    ref_text: str,
    model,
    tokenizer,
    device,
):
    """
    Pre-compute and save KV cache for a reference voice at registration time.

    Builds the system prompt + reference conversation, runs prefill through
    the model, and saves the resulting KV cache to disk.

    Args:
        ref_id: Reference voice ID
        codes: VQ codes tensor (num_codebooks, T) on CPU
        ref_text: Reference transcription text
        model: DualARTransformer on GPU with caches set up
        tokenizer: FishTokenizer
        device: Target device (e.g. "cuda")
    """
    import re
    import time

    from fish_speech.content_sequence import TextPart, VQPart
    from fish_speech.conversation import Conversation, Message

    # Build the same system prompt structure used during TTS generation
    if not re.search(r"<\|speaker:\d+\|>", ref_text):
        ref_text = f"<|speaker:0|>{ref_text}"

    system_parts = [
        TextPart(
            text="convert the provided text to speech reference to the following:\n\nText:\n",
            cal_loss=False,
        ),
        TextPart(text=ref_text, cal_loss=False),
        TextPart(text="\n\nSpeech:\n", cal_loss=False),
        VQPart(codes=codes, cal_loss=False),
    ]

    conv = Conversation()
    conv.append(
        Message(
            role="system",
            parts=system_parts,
            cal_loss=False,
            add_im_start=True,
            add_im_end=True,
        )
    )

    num_codebooks = model.config.num_codebooks
    encoded, audio_masks, audio_parts = conv.encode_for_inference(
        tokenizer, num_codebooks=num_codebooks
    )
    prefix_len = encoded.shape[1]
    cache_key = compute_cache_key(encoded)

    logger.info(
        f"Pre-compiling KV cache for '{ref_id}': "
        f"{prefix_len} tokens, cache_key={cache_key}"
    )

    # Ensure model caches are ready
    dtype = next(model.parameters()).dtype
    if not getattr(model, "_cache_setup_done", False):
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=dtype,
            )
        model._cache_setup_done = True

    # Run prefill (forward pass through all layers to populate KV cache)
    encoded = encoded.to(device=device)
    input_pos = torch.arange(0, prefix_len, device=device, dtype=torch.long)
    prompt_3d = encoded[None].contiguous()  # (1, codebook_dim, prefix_len)

    t0 = time.perf_counter()
    with torch.inference_mode():
        model.forward_generate(
            prompt_3d,
            input_pos,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
        )
    dt = (time.perf_counter() - t0) * 1000
    logger.info(f"Prefill for KV cache: {prefix_len} tokens in {dt:.0f}ms")

    # Save the KV cache
    save_kv_cache(ref_id, model, prefix_len, cache_key)
