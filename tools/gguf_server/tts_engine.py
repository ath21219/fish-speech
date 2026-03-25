"""
TTS generation core — streaming token generation + codec decode.

Public API:
  generate_speech_streaming(req) -> generator of audio events
  generate_speech(req)           -> np.ndarray (non-streaming wrapper)
"""

import json
import os
import re
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from loguru import logger

from .codec_manager import codec_on_gpu
from .schemas import (
    STREAM_CHUNK_TOKENS,
    STREAM_MIN_FIRST_CHUNK,
    TTSRequest,
)
from .state import state

# ── Crossfade constant for sentence-split boundaries ──
CROSSFADE_SAMPLES = 256  # ~6ms at 44.1kHz

from fish_speech.gguf.kv_cache_store import (
    compute_cache_key,
    load_kv_cache,
    save_kv_cache,
)


# ── Decode cache: reusable objects across TTS requests ──
# Invalidated on model load/unload.

class _DecodeCache:
    __slots__ = (
        "model_id", "semantic_logit_bias", "im_end_id",
        "graph_runner", "streaming_decoder",
    )

    def __init__(self):
        self.model_id = None
        self.semantic_logit_bias = None
        self.im_end_id = None
        self.graph_runner = None
        self.streaming_decoder = None

    def clear(self):
        if self.graph_runner is not None:
            try:
                del self.graph_runner.graph
            except Exception:
                pass
        self.model_id = None
        self.semantic_logit_bias = None
        self.im_end_id = None
        self.graph_runner = None
        self.streaming_decoder = None

    def ensure_for_model(self, model, device, dtype):
        """Lazily build model-dependent constants. No-op if already valid."""
        mid = id(model)
        if self.model_id == mid:
            return
        self.clear()
        self.model_id = mid

        from fish_speech.tokenizer import IM_END_TOKEN

        vocab_size = model.config.vocab_size
        bias = torch.full(
            (1, 1, vocab_size), float("-inf"), device=device, dtype=dtype
        )
        bias[
            0, 0,
            model.config.semantic_begin_id : model.config.semantic_end_id + 1,
        ] = 0.0
        im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)
        bias[0, 0, im_end_id] = 0.0

        self.semantic_logit_bias = bias
        self.im_end_id = im_end_id


_cache = _DecodeCache()


def invalidate_decode_cache():
    """Clear cached decode objects. Called on model load/unload."""
    _cache.clear()
    logger.info("Decode cache invalidated")


# ── Sentence splitting for long-text voice quality preservation ──

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[。！？!?…\n])"        # Split after CJK/ASCII sentence-ending punctuation
    r"|(?<=[.!?])\s+"            # Split after English punctuation + whitespace
)


def _split_into_sentences(text: str) -> list[str]:
    """Split text at natural sentence boundaries.

    Preserves speaker tags attached to the following sentence.
    Merges very short fragments (<10 UTF-8 bytes) with the preceding sentence
    to avoid generating degenerate single-word utterances.
    """
    raw_parts = _SENTENCE_SPLIT_RE.split(text)

    sentences: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # Merge very short fragments with previous sentence
        if sentences and len(part.encode("utf-8")) < 10:
            sentences[-1] = sentences[-1] + part
        else:
            sentences.append(part)

    return sentences if sentences else [text]


def _crossfade_audio(
    prev_tail: np.ndarray | None,
    new_audio: np.ndarray,
    n_crossfade: int = CROSSFADE_SAMPLES,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply linear crossfade between the tail of the previous sentence
    and the head of the new sentence.

    Returns (output_audio, new_tail_to_hold).
    - output_audio: audio ready to yield (crossfaded head + middle, tail withheld)
    - new_tail_to_hold: last n_crossfade samples to blend with next sentence
    """
    if len(new_audio) < n_crossfade * 2:
        # Too short to crossfade — just concatenate
        if prev_tail is not None:
            combined = np.concatenate([prev_tail, new_audio])
            return combined, None
        return new_audio, None

    if prev_tail is not None and len(prev_tail) == n_crossfade:
        # Blend: prev_tail fades out, new_audio[:n_crossfade] fades in
        fade_out = np.linspace(1.0, 0.0, n_crossfade, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, n_crossfade, dtype=np.float32)
        blended = prev_tail * fade_out + new_audio[:n_crossfade] * fade_in
        # Output: blended + body (minus tail to hold)
        body = new_audio[n_crossfade:-n_crossfade]
        tail = new_audio[-n_crossfade:]
        output = np.concatenate([blended, body])
        return output, tail
    else:
        # First sentence or no prev_tail — just hold the tail
        output = new_audio[:-n_crossfade]
        tail = new_audio[-n_crossfade:]
        return output, tail


def generate_speech_streaming(req: TTSRequest):
    """
    Generator that yields event dicts as semantic tokens are produced.

    Event types:
      {"type": "audio", "data": np.ndarray, "chunk_idx": int}
    """
    from fish_speech.content_sequence import TextPart, VQPart
    from fish_speech.conversation import Conversation, Message
    from fish_speech.models.text2semantic.inference import (
        decode_one_token_ar,
        encode_audio,
        group_turns_into_batches,
        split_text_by_speaker,
    )

    model = state.model
    tokenizer = state.tokenizer
    device = state.device

    # ── Reference loading ──
    prompt_tokens_list = []
    prompt_texts = []

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
        prompt_tokens_list.append(codes)
        ref_text = ""
        meta_file = ref_dir / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            ref_text = meta.get("transcription", "")
        if not ref_text:
            txt_files = list(ref_dir.glob("*.txt"))
            if txt_files:
                ref_text = txt_files[0].read_text(encoding="utf-8").strip()
        if not ref_text:
            raise ValueError(
                f"No description/text for reference '{req.reference_id}'"
            )
        prompt_texts.append(ref_text)

    use_prompt = bool(prompt_texts) and bool(prompt_tokens_list)

    # ── Build conversation ──
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

    # ── Compute prefix for KV cache ──
    # Encode the system message (with reference) alone to determine the
    # reusable prefix length.  This prefix is identical across all batches
    # and across requests that share the same reference_id.
    prefix_len = 0
    cache_key = ""
    ref_id = req.reference_id or ""

    if use_prompt and ref_id:
        num_codebooks = model.config.num_codebooks
        prefix_conv = deepcopy(base_conversation)
        prefix_encoded, _, _ = prefix_conv.encode_for_inference(
            tokenizer, num_codebooks=num_codebooks
        )
        prefix_len = prefix_encoded.shape[1]
        cache_key = compute_cache_key(prefix_encoded)
        logger.info(
            f"Reference prefix: {prefix_len} tokens, "
            f"cache_key={cache_key}, ref_id={ref_id}"
        )

    # ── Split text and generate ──
    turns = split_text_by_speaker(req.text)
    batches = (
        group_turns_into_batches(
            turns, max_speakers=5, max_bytes=req.chunk_length
        )
        if turns
        else [req.text]
    )

    total_audio_samples = 0
    chunk_count = 0

    # ── Resolve sliding window size: request > server default > disabled ──
    sw_size = req.sliding_window_size or state.sliding_window_size or 0

    # ── Sentence-split mode: each sentence gets a fresh context ──
    if req.sentence_split:
        crossfade_tail: np.ndarray | None = None  # held tail for crossfade
        sentence_idx = 0

        for batch_idx, batch_text in enumerate(batches):
            sentences = _split_into_sentences(batch_text)
            logger.info(
                f"Batch {batch_idx}: {len(batch_text.encode('utf-8'))} bytes "
                f"→ {len(sentences)} sentence(s)"
            )

            for sent_text in sentences:
                logger.info(
                    f"  Sentence {sentence_idx}: "
                    f"{len(sent_text.encode('utf-8'))} bytes"
                )

                # Fresh conversation per sentence (reference always dominant)
                sent_conv = deepcopy(base_conversation)
                sent_conv.append(
                    Message(
                        role="user",
                        parts=[TextPart(text=sent_text, cal_loss=False)],
                        cal_loss=False,
                        add_im_start=True,
                        add_im_end=True,
                    )
                )
                sent_conv.append(
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
                encoded, audio_masks, audio_parts = (
                    sent_conv.encode_for_inference(
                        tokenizer, num_codebooks=num_codebooks
                    )
                )
                encoded = encoded.to(device=device)
                prompt_len = encoded.shape[1]

                if prompt_len > state.max_seq_len - 128:
                    raise ValueError(
                        f"Prompt too long: {prompt_len} tokens "
                        f"(sentence: {sent_text[:50]}...)"
                    )

                # Collect all audio from this sentence for crossfade
                sent_audio_chunks: list[np.ndarray] = []

                with torch.inference_mode():
                    y = _generate_streaming(
                        model=model,
                        prompt=encoded,
                        max_new_tokens=req.max_new_tokens,
                        audio_masks=audio_masks,
                        audio_parts=audio_parts,
                        decode_one_token=decode_one_token_ar,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        codec=state.codec,
                        chunk_size=STREAM_CHUNK_TOKENS,
                        min_first_chunk=STREAM_MIN_FIRST_CHUNK,
                        streaming=req.streaming,
                        ref_id=ref_id,
                        prefix_len=prefix_len,
                        cache_key=cache_key,
                        sliding_window_size=sw_size,
                    )

                    for event in y:
                        if event["type"] == "audio_chunk":
                            sent_audio_chunks.append(event["audio"])
                        elif event["type"] == "stats":
                            logger.info(
                                f"  Sentence {sentence_idx}: "
                                f"{event['tokens']} tokens in "
                                f"{event['time']:.1f}s "
                                f"({event['tok_per_sec']:.1f} tok/s)"
                            )

                # Apply crossfade and yield
                if sent_audio_chunks:
                    sent_audio = np.concatenate(sent_audio_chunks)
                    output, crossfade_tail = _crossfade_audio(
                        crossfade_tail, sent_audio
                    )
                    if len(output) > 0:
                        chunk_count += 1
                        total_audio_samples += len(output)
                        yield {
                            "type": "audio",
                            "data": output,
                            "chunk_idx": chunk_count,
                        }

                sentence_idx += 1

        # Flush remaining crossfade tail
        if crossfade_tail is not None and len(crossfade_tail) > 0:
            chunk_count += 1
            total_audio_samples += len(crossfade_tail)
            yield {
                "type": "audio",
                "data": crossfade_tail,
                "chunk_idx": chunk_count,
            }

    else:
        # ── Legacy mode: accumulated conversation across batches ──
        conversation = deepcopy(base_conversation)

        for batch_idx, batch_text in enumerate(batches):
            logger.info(
                f"Batch {batch_idx}: {len(batch_text.encode('utf-8'))} bytes"
            )

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
            encoded, audio_masks, audio_parts = (
                conversation_gen.encode_for_inference(
                    tokenizer, num_codebooks=num_codebooks
                )
            )
            encoded = encoded.to(device=device)
            prompt_len = encoded.shape[1]

            if prompt_len > state.max_seq_len - 128:
                raise ValueError(f"Prompt too long: {prompt_len} tokens")

            t0 = time.perf_counter()
            all_batch_codes = []

            with torch.inference_mode():
                y = _generate_streaming(
                    model=model,
                    prompt=encoded,
                    max_new_tokens=req.max_new_tokens,
                    audio_masks=audio_masks,
                    audio_parts=audio_parts,
                    decode_one_token=decode_one_token_ar,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    codec=state.codec,
                    chunk_size=STREAM_CHUNK_TOKENS,
                    min_first_chunk=STREAM_MIN_FIRST_CHUNK,
                    streaming=req.streaming,
                    ref_id=ref_id,
                    prefix_len=prefix_len,
                    cache_key=cache_key,
                    sliding_window_size=sw_size,
                )

                for event in y:
                    if event["type"] == "codes":
                        all_batch_codes.append(event["codes"])
                    elif event["type"] == "audio_chunk":
                        chunk_count += 1
                        audio_np = event["audio"]
                        total_audio_samples += len(audio_np)
                        yield {
                            "type": "audio",
                            "data": audio_np,
                            "chunk_idx": chunk_count,
                        }
                    elif event["type"] == "stats":
                        logger.info(
                            f"Batch {batch_idx}: {event['tokens']} tokens in "
                            f"{event['time']:.1f}s "
                            f"({event['tok_per_sec']:.1f} tok/s)"
                        )

            if all_batch_codes:
                merged = torch.cat(all_batch_codes, dim=1)
                conversation.append(
                    Message(
                        role="assistant",
                        parts=[VQPart(codes=merged.cpu(), cal_loss=False)],
                        cal_loss=False,
                        modality="voice",
                        add_im_start=True,
                        add_im_end=True,
                    )
                )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    duration = (
        total_audio_samples / state.sample_rate
        if total_audio_samples > 0
        else 0
    )
    logger.info(
        f"Streaming complete: {chunk_count} chunks, {duration:.1f}s audio"
    )

    if chunk_count == 0:
        raise ValueError("No audio generated")


def generate_speech(req: TTSRequest) -> np.ndarray:
    """Non-streaming: collect all chunks, return concatenated audio."""
    req_copy = req.model_copy()
    req_copy.streaming = False

    all_audio = []
    for event in generate_speech_streaming(req_copy):
        if event["type"] == "audio":
            all_audio.append(event["data"])

    if not all_audio:
        raise ValueError("No audio generated")

    return np.concatenate(all_audio, axis=0)


# ============================================================
# Internal: streaming token generation loop
# ============================================================


def _generate_streaming(
    model,
    prompt,
    max_new_tokens,
    audio_masks,
    audio_parts,
    decode_one_token,
    temperature,
    top_p,
    top_k,
    codec,
    chunk_size=21,
    min_first_chunk=21,
    streaming=True,
    ref_id="",
    prefix_len=0,
    cache_key="",
    sliding_window_size=0,
):
    """
    Token-by-token generation with streaming codec decode.

    [OPT-F19] Batched im_end check, no tqdm.
    [OPT-B6]  Fixed-address buffers for CUDA Graph compatibility.
    [OPT-KVC] Reference KV cache: skip prefill for cached prefix tokens.
    """
    from fish_speech.models.text2semantic.inference import (
        CUDAGraphRunner,
        decode_one_token_ar,
    )

    device = prompt.device
    dtype = next(model.parameters()).dtype
    T = prompt.size(1)
    codebook_dim = 1 + model.config.num_codebooks

    if T >= model.config.max_seq_len:
        raise ValueError(f"Input sequence length {T} exceeds max_seq_len")

    effective_max = (
        min(max_new_tokens, model.config.max_seq_len - T)
        if max_new_tokens
        else model.config.max_seq_len - T
    )

    # Setup LLM caches if needed
    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=dtype,
            )
        model._cache_setup_done = True

    # [OPT-CACHE] Reuse model-dependent constants across requests
    _cache.ensure_for_model(model, device, dtype)
    semantic_logit_bias = _cache.semantic_logit_bias
    im_end_id = _cache.im_end_id

    temperature_t = torch.tensor(temperature, device=device, dtype=dtype)
    top_p_t = torch.tensor(top_p, device=device, dtype=dtype)

    # [OPT-CACHE] Reuse streaming codec decoder (avoid KV cache re-setup)
    from fish_speech.models.dac.streaming_codec import StreamingCodecDecoder

    if _cache.streaming_decoder is None:
        _cache.streaming_decoder = StreamingCodecDecoder(
            codec=codec,
            device=str(next(codec.parameters()).device),
            max_frames=state.max_seq_len + 64,
        )
    else:
        _cache.streaming_decoder.reset()
    streaming_decoder = _cache.streaming_decoder

    # ── Zero KV caches to prevent stale data from prior generations ──
    for layer in model.layers:
        kv = layer.attention.kv_cache
        if kv is not None:
            kv.k_cache.zero_()
            kv.v_cache.zero_()

    # ── [OPT-KVC] Try to load cached KV for reference prefix ──
    cached_prefix_len = 0
    should_save_cache = False

    if ref_id and prefix_len > 0 and cache_key:
        # Verify prefix tokens of the full prompt match the standalone prefix
        actual_prefix_key = compute_cache_key(prompt[:, :prefix_len])
        if actual_prefix_key != cache_key:
            logger.warning(
                f"[KV Cache] Prefix token mismatch! "
                f"standalone={cache_key}, actual={actual_prefix_key}. "
                f"Skipping cache."
            )
        else:
            loaded = load_kv_cache(ref_id, model, cache_key, device)
            if loaded is not None:
                cached_prefix_len = loaded
                logger.info(
                    f"[KV Cache HIT] ref={ref_id}, "
                    f"skipping {cached_prefix_len}/{T} prefix tokens"
                )
            else:
                # Cache miss — will save after full prefill
                should_save_cache = True
                logger.info(
                    f"[KV Cache MISS] ref={ref_id}, "
                    f"will save after prefill ({prefix_len} tokens)"
                )

    # ── Prefill ──
    RAS_WIN_SIZE = 10
    previous_tokens = torch.zeros(
        (codebook_dim, RAS_WIN_SIZE), dtype=torch.int, device=device
    )
    ras_pos = 0

    if cached_prefix_len > 0 and cached_prefix_len <= T:
        # [OPT-KVC] Partial prefill: only process suffix tokens.
        # KV cache for positions 0..cached_prefix_len-1 is already loaded.
        t_prefill = time.perf_counter()
        suffix_start = cached_prefix_len
        input_pos = torch.arange(
            suffix_start, T, device=device, dtype=torch.long
        )
        prompt_suffix = prompt[None, :, suffix_start:].contiguous()

        # Slice audio_masks/audio_parts for suffix only
        suffix_audio_masks = None
        suffix_audio_parts = None
        if audio_masks is not None:
            suffix_audio_masks = audio_masks[:, suffix_start:]
            # Count how many audio positions are in the prefix vs suffix
            prefix_audio_count = audio_masks[:, :suffix_start].sum().item()
            total_audio_count = audio_masks.sum().item()
            suffix_audio_count = int(total_audio_count - prefix_audio_count)
            if suffix_audio_count > 0 and audio_parts is not None:
                suffix_audio_parts = audio_parts[int(prefix_audio_count):]
            else:
                suffix_audio_parts = None

        first_token = decode_one_token(
            model=model,
            x=prompt_suffix,
            input_pos=input_pos,
            temperature=temperature_t,
            top_p=top_p_t,
            top_k=top_k,
            semantic_logit_bias=semantic_logit_bias,
            audio_masks=suffix_audio_masks,
            audio_parts=suffix_audio_parts,
            previous_tokens=previous_tokens,
        )
        dt_prefill = (time.perf_counter() - t_prefill) * 1000
        logger.info(
            f"[KV Cache] Partial prefill: {T - suffix_start} tokens "
            f"(skipped {suffix_start}) in {dt_prefill:.0f}ms"
        )
    else:
        # Full prefill (no cache or cache unusable)
        t_prefill = time.perf_counter()
        input_pos = torch.arange(0, T, device=device, dtype=torch.long)
        prompt_3d = prompt[None].repeat(1, 1, 1)

        first_token = decode_one_token(
            model=model,
            x=prompt_3d,
            input_pos=input_pos,
            temperature=temperature_t,
            top_p=top_p_t,
            top_k=top_k,
            semantic_logit_bias=semantic_logit_bias,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
            previous_tokens=previous_tokens,
        )
        dt_prefill = (time.perf_counter() - t_prefill) * 1000
        logger.info(f"Full prefill: {T} tokens in {dt_prefill:.0f}ms")

        # [OPT-KVC] Save KV cache for prefix after full prefill
        if should_save_cache and prefix_len > 0:
            try:
                save_kv_cache(ref_id, model, prefix_len, cache_key)
            except Exception as e:
                logger.warning(f"Failed to save KV cache: {e}")

    previous_tokens[:, ras_pos % RAS_WIN_SIZE] = first_token.view(
        codebook_dim, -1
    )[:, 0]
    ras_pos += 1

    # ── Token accumulation state ──
    all_new_tokens = [first_token]
    pending_tokens = [first_token]
    tokens_since_last_chunk = 1
    total_tokens = 1
    t0 = time.perf_counter()
    is_first_chunk = True
    finished = first_token[0, 0] == im_end_id

    # [OPT-F19] GPU-side im_end comparison
    im_end_tensor = torch.tensor(im_end_id, dtype=torch.int, device=device)

    # [OPT-B6] Fixed-address buffers
    fixed_input_pos = torch.tensor([T], device=device, dtype=torch.long)
    fixed_cur_token = first_token.view(1, codebook_dim, 1).clone()

    # [OPT-F19] Batched im_end check
    CHECK_INTERVAL = 8
    semantic_id_buffer = torch.empty(
        CHECK_INTERVAL, dtype=torch.int, device=device
    )
    buf_pos = 0

    # [OPT-B6+CACHE] CUDA Graph: reuse across requests
    use_cuda_graph = (
        device.type == "cuda"
        if isinstance(device, torch.device)
        else str(device).startswith("cuda")
    )
    graph_runner = None
    if use_cuda_graph:
        if _cache.graph_runner is not None and _cache.graph_runner.captured:
            # Reuse cached graph — just update static buffers
            graph_runner = _cache.graph_runner
            graph_runner.static_x.copy_(fixed_cur_token)
            graph_runner.static_input_pos.copy_(fixed_input_pos)
            graph_runner.static_temperature.copy_(temperature_t)
            graph_runner.static_top_p.copy_(top_p_t)
            logger.debug("[CUDA Graph] Reusing cached graph runner")
        else:
            try:
                graph_runner = CUDAGraphRunner(
                    model=model,
                    decode_fn=decode_one_token,
                    codebook_dim=codebook_dim,
                    device=(
                        torch.device(device)
                        if isinstance(device, str)
                        else device
                    ),
                    dtype=dtype,
                    vocab_size=model.config.vocab_size,
                    top_k=top_k,
                    semantic_logit_bias=semantic_logit_bias,
                    audio_masks=audio_masks,
                    audio_parts=audio_parts,
                )
                graph_runner.static_x.copy_(fixed_cur_token)
                graph_runner.static_input_pos.copy_(fixed_input_pos)
                graph_runner.static_temperature.copy_(temperature_t)
                graph_runner.static_top_p.copy_(top_p_t)
                graph_runner.warmup_and_capture()
                _cache.graph_runner = graph_runner
                logger.info("[CUDA Graph] Captured and cached for reuse")
            except Exception as e:
                logger.warning(
                    f"[CUDA Graph] Streaming capture failed ({e}), using eager"
                )
                graph_runner = None

    # ── Sliding window: pin reference tokens, window generated tokens ──
    _use_sliding_window = sliding_window_size > 0 and prefix_len > 0
    if _use_sliding_window:
        model.enable_sliding_window(prefix_len, sliding_window_size)
        logger.info(
            f"[Sliding Window] enabled: prefix={prefix_len}, "
            f"window={sliding_window_size}"
        )

    # ── Decode loop ──
    with sdpa_kernel(SDPBackend.MATH):
        while not finished and total_tokens < effective_max:
            if graph_runner is not None and graph_runner.captured:
                next_token = graph_runner.replay(
                    x=fixed_cur_token,
                    input_pos=fixed_input_pos,
                    temperature=temperature_t,
                    top_p=top_p_t,
                    previous_tokens=previous_tokens,
                )
            else:
                next_token = decode_one_token(
                    model=model,
                    x=fixed_cur_token,
                    input_pos=fixed_input_pos,
                    temperature=temperature_t,
                    top_p=top_p_t,
                    top_k=top_k,
                    semantic_logit_bias=semantic_logit_bias,
                    audio_masks=audio_masks,
                    audio_parts=audio_parts,
                    previous_tokens=previous_tokens,
                )

            fixed_input_pos.add_(1)
            fixed_cur_token.copy_(next_token.view(1, codebook_dim, 1))

            previous_tokens[:, ras_pos % RAS_WIN_SIZE] = next_token.view(
                codebook_dim, -1
            )[:, 0]
            ras_pos += 1
            total_tokens += 1

            all_new_tokens.append(next_token)
            pending_tokens.append(next_token)
            tokens_since_last_chunk += 1

            # [OPT-F19] Batched im_end check
            semantic_id_buffer[buf_pos] = next_token[0, 0]
            buf_pos += 1

            if buf_pos >= CHECK_INTERVAL:
                if (semantic_id_buffer[:buf_pos] == im_end_tensor).any().item():
                    match_mask = semantic_id_buffer[:buf_pos] == im_end_tensor
                    first_match = match_mask.nonzero(as_tuple=False)[0, 0].item()
                    tokens_to_discard = buf_pos - first_match - 1
                    if tokens_to_discard > 0:
                        all_new_tokens = all_new_tokens[:-tokens_to_discard]
                        pending_tokens = pending_tokens[:-tokens_to_discard]
                        tokens_since_last_chunk -= tokens_to_discard
                        total_tokens -= tokens_to_discard
                    finished = True
                buf_pos = 0

            # ── Emit chunk ──
            threshold = min_first_chunk if is_first_chunk else chunk_size
            should_emit = (
                streaming and tokens_since_last_chunk >= threshold
            ) or finished

            if should_emit and pending_tokens:
                chunk_all = torch.cat(pending_tokens, dim=1)
                codes_new = chunk_all[1:, :].clone().clamp(min=0)

                if codes_new.shape[1] > 0:
                    t_dec = time.perf_counter()
                    audio_np = streaming_decoder.decode_chunk(codes_new)
                    t_dec_ms = (time.perf_counter() - t_dec) * 1000

                    if len(audio_np) > 0:
                        chunk_duration = len(audio_np) / state.sample_rate
                        logger.debug(
                            f"Streaming decode: {codes_new.shape[1]} frames → "
                            f"{chunk_duration:.2f}s audio in {t_dec_ms:.0f}ms"
                        )
                        yield {"type": "codes", "codes": codes_new.cpu()}
                        yield {"type": "audio_chunk", "audio": audio_np}

                pending_tokens = []
                tokens_since_last_chunk = 0
                is_first_chunk = False

    # ── Post-loop buffer flush ──
    if not finished and buf_pos > 0:
        if (semantic_id_buffer[:buf_pos] == im_end_tensor).any().item():
            match_mask = semantic_id_buffer[:buf_pos] == im_end_tensor
            first_match = match_mask.nonzero(as_tuple=False)[0, 0].item()
            tokens_to_discard = buf_pos - first_match - 1
            if tokens_to_discard > 0 and pending_tokens:
                pending_tokens = pending_tokens[:-tokens_to_discard]

        if pending_tokens:
            chunk_all = torch.cat(pending_tokens, dim=1)
            codes_new = chunk_all[1:, :].clone().clamp(min=0)
            if codes_new.shape[1] > 0:
                audio_np = streaming_decoder.decode_chunk(codes_new)
                if len(audio_np) > 0:
                    yield {"type": "codes", "codes": codes_new.cpu()}
                    yield {"type": "audio_chunk", "audio": audio_np}

    # graph_runner is cached in _cache — do NOT delete

    # ── Stats ──
    t_total = time.perf_counter() - t0
    yield {
        "type": "stats",
        "tokens": total_tokens,
        "time": t_total,
        "tok_per_sec": total_tokens / t_total if t_total > 0 else 0,
    }

    # ── Cleanup ──
    if _use_sliding_window:
        model.disable_sliding_window()
    streaming_decoder.reset()
    del fixed_cur_token, fixed_input_pos
    del all_new_tokens, pending_tokens, semantic_id_buffer
