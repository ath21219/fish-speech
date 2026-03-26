import os
import queue
import re
import threading
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._inductor.config
from torch.nn.attention import SDPBackend, sdpa_kernel
from loguru import logger
from tqdm import tqdm

from fish_speech.content_sequence import (
    TextPart,
    VQPart,
)
from fish_speech.conversation import Conversation, Message
from fish_speech.tokenizer import IM_END_TOKEN
from fish_speech.models.text2semantic.llama import Attention

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True

# ============================================================
# [OPT-B6] Manual CUDA Graph capture for decode step
# ============================================================

class CUDAGraphRunner:
    """
    Wraps decode_one_token_ar in a manually captured CUDA Graph.
    
    Unlike torch.compile(mode="reduce-overhead"), manual capture via
    torch.cuda.CUDAGraph does NOT check for mutated inputs — KV cache
    in-place updates work correctly.
    
    Usage:
        runner = CUDAGraphRunner(model, decode_fn, warmup_kwargs)
        # In decode loop:
        output = runner.replay(x, input_pos, temperature, top_p, ...)
    """

    def __init__(
        self,
        model,
        decode_fn,
        *,
        codebook_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        vocab_size: int,
        top_k: int = 30,
        semantic_logit_bias: torch.Tensor,
        audio_masks: torch.Tensor,
        audio_parts: torch.Tensor,
    ):
        self.model = model
        self.decode_fn = decode_fn
        self.device = device
        self.graph = torch.cuda.CUDAGraph()
        self.captured = False

        # --- Static input buffers (fixed addresses for capture/replay) ---
        self.static_x = torch.zeros(
            1, codebook_dim, 1, dtype=torch.long, device=device
        )
        self.static_input_pos = torch.zeros(1, dtype=torch.long, device=device)
        self.static_temperature = torch.zeros(1, dtype=dtype, device=device)
        self.static_top_p = torch.zeros(1, dtype=dtype, device=device)
        self.static_previous_tokens = torch.zeros(
            codebook_dim, RAS_WIN_SIZE, dtype=torch.int, device=device
        )

        # These don't change between calls but must have fixed addresses
        self.static_semantic_logit_bias = semantic_logit_bias.clone()
        self.static_audio_masks = audio_masks
        self.static_audio_parts = audio_parts
        self.top_k = top_k

        # Static output buffer (will be filled during capture)
        self.static_output = None

    def _run_fn(self):
        """The function to capture — calls decode_one_token_ar with static buffers."""
        return self.decode_fn(
            model=self.model,
            x=self.static_x,
            input_pos=self.static_input_pos,
            temperature=self.static_temperature,
            top_p=self.static_top_p,
            top_k=self.top_k,
            semantic_logit_bias=self.static_semantic_logit_bias,
            audio_masks=self.static_audio_masks,
            audio_parts=self.static_audio_parts,
            previous_tokens=self.static_previous_tokens,
        )

    def warmup_and_capture(self):
        """
        Warmup the function (JIT Triton kernels etc.) then capture as CUDA Graph.
        Must be called AFTER model.setup_caches() and AFTER prefill.
        """
        logger.info("[CUDA Graph] Pre-materializing CPU tensors on GPU...")

        # ── Phase 0: Ensure ALL model tensors are on GPU ──
        # Some tensors (e.g. GGUFEmbedding buffers, bias, freqs_cis) may
        # still be on CPU after GGUF loading. CUDA Graph capture forbids
        # any CPU↔GPU copy, so we must move them now.
        model = self.model
        device = self.device

        moved = 0
        for name, param in model.named_parameters():
            if not param.is_cuda:
                # Can't reassign parameters easily, but we can move data
                param.data = param.data.to(device, non_blocking=True)
                moved += 1
        for name, buf in model.named_buffers():
            if not buf.is_cuda:
                # Re-register buffer on GPU
                parts = name.split('.')
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                mod.register_buffer(parts[-1], buf.to(device, non_blocking=True),
                                    persistent=False)
                moved += 1

        # Also handle GGUFLinear bias tensors and GGUFEmbedding internals
        for name, module in model.named_modules():
            if hasattr(module, 'bias') and module.bias is not None:
                if isinstance(module.bias, torch.Tensor) and not module.bias.is_cuda:
                    module.bias = module.bias.to(device, non_blocking=True)
                    moved += 1
            # GGUFParameter.data should already be on GPU, but double-check
            if hasattr(module, 'qparam') and hasattr(module.qparam, 'data'):
                if isinstance(module.qparam.data, torch.Tensor) and not module.qparam.data.is_cuda:
                    module.qparam.data = module.qparam.data.to(device, non_blocking=True)
                    moved += 1

        if moved > 0:
            torch.cuda.synchronize(device)
            logger.info(f"[CUDA Graph] Moved {moved} CPU tensors to GPU")

        # ── Phase 0b: Pre-cache dequantized embedding weights ──
        # GGUFEmbedding.forward() calls dequantize() every time, which
        # may create intermediate CPU tensors. We pre-cache the result.
        import torch.nn.functional as _F  # ★ CUDA Graph 用ローカル import

        for name, module in model.named_modules():
            if hasattr(module, '__class__') and module.__class__.__name__ == 'GGUFEmbedding':
                with torch.no_grad():
                    cached_w = module.qparam.dequantize(dtype=torch.float16).to(device)

                # forward を差し替え
                def _make_forward(w):
                    def _forward(x):
                        return _F.embedding(x, w)
                    return _forward
                module.forward = _make_forward(cached_w)

                # weight プロパティ用キャッシュ（tie_word_embeddings 対応）
                module._weight_override = cached_w

                moved += 1
                logger.debug(f"[CUDA Graph] Cached GGUFEmbedding: {name} "
                             f"({cached_w.shape}, {cached_w.nbytes/1e6:.1f} MB)")

        # ── Phase 0c: Attention._neg_inf を GPU 上に作成 ──
        neg_inf_count = 0
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                module._neg_inf = torch.tensor(
                    float("-inf"), device=device, dtype=torch.float16
                )
                neg_inf_count += 1
        logger.info(f"[CUDA Graph] Phase 0c: Set _neg_inf on {neg_inf_count} Attention modules (device={device})")

        # ── Phase 1: Warmup on side stream ──
        logger.info("[CUDA Graph] Warming up decode function...")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self._run_fn()
        torch.cuda.current_stream().wait_stream(s)

        # Debug: detect any hidden CPU-GPU syncs
        torch.cuda.set_sync_debug_mode(2)  # Error on any sync

        # ★ RNG 状態を CUDA Graph に登録
        # これにより replay ごとに RNG が正しく進行し、
        # 毎回異なる乱数が生成される（サンプリングの多様性を維持）
        device_idx = device.index if device.index is not None else 0
        self.graph.register_generator_state(
            torch.cuda.default_generators[device_idx]
        )

        # ── Phase 2: Capture ──
        logger.info("[CUDA Graph] Capturing graph...")
        try:
            torch.cuda.set_sync_debug_mode(2)
            with torch.cuda.graph(self.graph):
                self.static_output = self._run_fn()
            self.captured = True
            logger.info("[CUDA Graph] Capture complete.")
        except RuntimeError as e:
            import traceback
            logger.error(f"[CUDA Graph] Capture error: {e}")
            logger.error(f"[CUDA Graph] Traceback:\n{traceback.format_exc()}")
            self.captured = False
        finally:
            torch.cuda.set_sync_debug_mode(0)

    def replay(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: torch.Tensor,
        top_p: torch.Tensor,
        previous_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Copy inputs into static buffers and replay the captured graph.
        Returns a CLONE of the output (static_output address is reused).
        """
        # Copy new values into static buffers (preserves addresses)
        self.static_x.copy_(x)
        self.static_input_pos.copy_(input_pos)
        self.static_temperature.copy_(temperature)
        self.static_top_p.copy_(top_p)
        self.static_previous_tokens.copy_(previous_tokens)

        # Replay
        self.graph.replay()

        # MUST clone — static_output memory will be overwritten on next replay
        return self.static_output.clone()

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.rand_like(probs_sort)
    q = -torch.log(q)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


RAS_WIN_SIZE = 10
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    neg_inf: torch.Tensor = None,  # ★ GPU上の -inf 定数
) -> torch.Tensor:
    if neg_inf is None:
        # eager パスでのみ到達。CUDA Graph パスでは bufs['neg_inf'] が渡される
        neg_inf = logits.new_tensor(float("-inf"))

    # ★ Temperature を top-p/top-k フィルタリングの前に適用
    # これにより top-p の累積確率が temperature 適用後の分布に基づいて
    # 正しく計算される（サンプリング品質の安定化）
    logits = logits / torch.clip(temperature, min=1e-5)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    indices = torch.arange(sorted_logits.shape[-1], device=sorted_logits.device)
    top_k_mask = indices >= top_k
    sorted_indices_to_remove = (cum_probs > top_p) | top_k_mask
    first_mask = (indices == 0)
    sorted_indices_to_remove = sorted_indices_to_remove & ~first_mask

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = torch.where(indices_to_remove, neg_inf, logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    neg_inf: torch.Tensor = None,  # ★ 追加
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        neg_inf=neg_inf,  # ★ 追加
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def _ensure_decode_buffers(model, device, dtype):
    """Pre-allocate reusable tensors for decode_one_token_ar (called once)."""
    if hasattr(model, '_decode_bufs'):
        return
    model._decode_bufs = {
        'high_temp': torch.tensor(RAS_HIGH_TEMP, device=device, dtype=dtype),
        'high_top_p': torch.tensor(RAS_HIGH_TOP_P, device=device, dtype=dtype),
        'fast_pos': [
            torch.tensor([i], device=device, dtype=torch.long)
            for i in range(model.config.num_codebooks)
        ],
        'neg_inf': torch.tensor(float("-inf"), device=device, dtype=dtype),  # ★ 追加
    }



def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Lazy-init reusable buffers (once per session)
    _ensure_decode_buffers(model, x.device, temperature.dtype)
    bufs = model._decode_bufs

    forward_result = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = forward_result.logits  # (1, 1, vocab_size)
    hidden_states = forward_result.hidden_states

    # Apply constrained decoding: only allow semantic tokens + im_end
    biased_logits = logits + semantic_logit_bias

    # Normal sample
    main_token_normal = sample(
        biased_logits, temperature=temperature, top_p=top_p, top_k=top_k,
        neg_inf=bufs['neg_inf']  # ★ 追加
    )[0]

    # RAS: also sample with high temp to use as fallback if token repeats
    main_token_high = sample(
        biased_logits, temperature=bufs['high_temp'], top_p=bufs['high_top_p'], top_k=top_k,
        neg_inf=bufs['neg_inf']  # ★ 追加
    )[0]

    # Use high-temp sample if: token is semantic AND token is in previous window
    if previous_tokens is not None:
        in_window = (previous_tokens[0] == main_token_normal).any()
        # Use tensor ops (&, torch.where) instead of Python (and, if) — torch.compile requires no data-dependent branching
        is_semantic = (main_token_normal >= model.config.semantic_begin_id) & (
            main_token_normal <= model.config.semantic_end_id
        )
        should_use_high = in_window & is_semantic
        main_token_normal = torch.where(
            should_use_high, main_token_high, main_token_normal
        )

    codebooks = [main_token_normal]

    model.forward_generate_fast(hidden_states, bufs['fast_pos'][0])

    a = codebooks[0] - model.config.semantic_begin_id
    a = torch.clamp(a, min=0, max=model.config.codebook_size - 1)

    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        logits = model.forward_generate_fast(hidden_states, bufs['fast_pos'][codebook_idx])

        # Convert logits to probs (no constrain for fast codebooks)
        a = sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            neg_inf=bufs['neg_inf'],  # ★ 追加
        )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    return torch.stack(codebooks, dim=1).T


def decode_n_tokens(
    model: DualARTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
):
    """
    Optimized decode loop with manual CUDA Graph capture.

    [OPT-B6]  Manual CUDA Graph via torch.cuda.CUDAGraph (not torch.compile).
              Unlike torch.compile's reduce-overhead mode, manual capture allows
              KV cache in-place updates without graph breaks.
    [OPT-F19] No tqdm, sdpa_kernel outside loop, batched im_end check.
    """
    codebook_dim = model.config.num_codebooks + 1
    device = cur_token.device

    # Rolling window for RAS
    previous_tokens = torch.zeros(
        (codebook_dim, RAS_WIN_SIZE), dtype=torch.int, device=device,
    )

    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)
    im_end_tensor = torch.tensor(im_end_id, dtype=torch.int, device=device)

    # ── [OPT-B6] Build CUDA Graph runner ──────────────────────
    use_cuda_graph = (device.type == "cuda" if isinstance(device, torch.device)
                      else str(device).startswith("cuda"))

    graph_runner = None
    if use_cuda_graph:
        try:
            graph_runner = CUDAGraphRunner(
                model=model,
                decode_fn=decode_one_token,
                codebook_dim=codebook_dim,
                device=torch.device(device),
                dtype=next(model.parameters()).dtype,
                vocab_size=model.config.vocab_size,
                top_k=top_k,
                semantic_logit_bias=semantic_logit_bias,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
            )
            # Feed initial token for warmup
            graph_runner.static_x.copy_(cur_token.view(1, codebook_dim, 1))
            graph_runner.static_input_pos.copy_(input_pos)
            graph_runner.static_temperature.copy_(temperature)
            graph_runner.static_top_p.copy_(top_p)
            graph_runner.warmup_and_capture()
        except Exception as e:
            logger.warning(f"[CUDA Graph] Capture failed ({e}), falling back to eager")
            graph_runner = None

    # ── Fixed-address buffers (fallback path uses these) ──────
    fixed_input_pos = torch.empty(1, dtype=torch.long, device=device)
    fixed_input_pos.copy_(input_pos)
    fixed_cur_token = cur_token.view(1, codebook_dim, 1).clone()

    new_tokens = []
    ras_pos = 0
    CHECK_INTERVAL = 8
    semantic_id_buffer = torch.empty(CHECK_INTERVAL, dtype=torch.int, device=device)
    buf_pos = 0
    finished = False

    with sdpa_kernel(SDPBackend.MATH):
        for i in range(num_new_tokens):
            if graph_runner is not None and graph_runner.captured:
                # ── CUDA Graph replay path ──
                next_token = graph_runner.replay(
                    x=fixed_cur_token,
                    input_pos=fixed_input_pos,
                    temperature=temperature,
                    top_p=top_p,
                    previous_tokens=previous_tokens,
                )
            else:
                # ── Eager fallback path ──
                next_token = decode_one_token(
                    model=model,
                    x=fixed_cur_token,
                    input_pos=fixed_input_pos,
                    previous_tokens=previous_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    semantic_logit_bias=semantic_logit_bias,
                    audio_masks=audio_masks,
                    audio_parts=audio_parts,
                )

            # Update fixed buffers in-place
            fixed_input_pos.add_(1)
            fixed_cur_token.copy_(next_token.view(1, codebook_dim, 1))

            # RAS circular buffer
            previous_tokens[:, ras_pos % RAS_WIN_SIZE] = next_token.view(
                codebook_dim, -1
            )[:, 0]
            ras_pos += 1
            new_tokens.append(next_token)

            # Batched im_end check
            semantic_id_buffer[buf_pos] = next_token[0, 0]
            buf_pos += 1

            if buf_pos >= CHECK_INTERVAL:
                if (semantic_id_buffer[:buf_pos] == im_end_tensor).any().item():
                    match_mask = semantic_id_buffer[:buf_pos] == im_end_tensor
                    first_match = match_mask.nonzero(as_tuple=False)[0, 0].item()
                    tokens_to_remove = buf_pos - first_match - 1
                    if tokens_to_remove > 0:
                        new_tokens = new_tokens[:-tokens_to_remove]
                    finished = True
                    break
                buf_pos = 0

    if not finished and buf_pos > 0:
        if (semantic_id_buffer[:buf_pos] == im_end_tensor).any().item():
            match_mask = semantic_id_buffer[:buf_pos] == im_end_tensor
            first_match = match_mask.nonzero(as_tuple=False)[0, 0].item()
            tokens_to_remove = buf_pos - first_match - 1
            if tokens_to_remove > 0:
                new_tokens = new_tokens[:-tokens_to_remove]

    # Cleanup
    if graph_runner is not None:
        del graph_runner.graph
        del graph_runner
    del fixed_cur_token, fixed_input_pos, semantic_id_buffer

    return torch.cat(new_tokens, dim=1)


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: DualARTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    num_samples: int = 1,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device = prompt.device
    dtype = next(
        model.parameters()
    ).dtype  # model weight dtype (bfloat16), NOT prompt dtype (int32)

    # Critical fix: Only set up cache on first run or when necessary
    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,  # Fixed to 1, avoid dynamic changes
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True

    codebook_dim = 1 + model.config.num_codebooks

    # Create new tensor each time, but try to reuse memory
    input_pos = torch.arange(0, T, device=device, dtype=torch.long)
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=prompt.dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty

    temp_val = sampling_kwargs.get("temperature", 1.0)
    top_p_val = sampling_kwargs.get("top_p", 0.9)
    top_k_val = sampling_kwargs.get("top_k", 30)

    temperature = torch.tensor(temp_val, device=device, dtype=dtype)
    top_p = torch.tensor(top_p_val, device=device, dtype=dtype)

    # Build semantic logit bias: 0 for semantic tokens + im_end, -inf for all others
    vocab_size = model.config.vocab_size
    semantic_logit_bias = torch.full(
        (1, 1, vocab_size), float("-inf"), device=device, dtype=dtype
    )

    # [MODIFIED] Use config for semantic range
    semantic_logit_bias[
        0, 0, model.config.semantic_begin_id : model.config.semantic_end_id + 1
    ] = 0.0

    # [MODIFIED] Use tokenizer.get_token_id (Wrapper method)
    semantic_logit_bias[0, 0, model.tokenizer.get_token_id(IM_END_TOKEN)] = 0.0

    prefill_decode = decode_one_token_ar

    first_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        top_k_val,
        semantic_logit_bias,
        audio_masks,
        audio_parts,
    )
    seq[:, T : T + 1] = first_token

    # Recreate input_pos
    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    x = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k_val,
        semantic_logit_bias=semantic_logit_bias,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
    )
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    # Clean up temporary variables
    del first_token, x, prompt, empty, input_pos

    return seq


def init_model(checkpoint_path, device, precision, compile=False):
    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        raise ValueError("Unsupported model type")

    # Pre-create fixed parameter tensors to avoid runtime creation
    model.fixed_temperature = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_top_p = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_repetition_penalty = torch.tensor(1.5, device=device, dtype=torch.float)

    # Mark whether cache has been initialized
    model._cache_setup_done = False

    if compile:
        # [OPT-B6] Manual CUDA Graph capture is now handled inside
        # decode_n_tokens(), so torch.compile is no longer needed.
        # We still enable custom_op mode for potential future use.
        try:
            from fish_speech.gguf.dequant import enable_custom_op_mode
            enable_custom_op_mode(model)
            logger.info("Custom op mode enabled")
        except Exception as e:
            logger.warning(f"Could not enable custom_op mode: {e}")

        logger.info("CUDA Graph will be captured at first decode call (manual capture)")
        # decode_one_token remains the raw function — no torch.compile wrapper

    return model.eval(), decode_one_token


@torch.inference_mode()
def load_codec_model(codec_checkpoint_path, device, precision=torch.bfloat16):
    """Load the DAC codec model for audio encoding/decoding."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = Path(__file__).parent.parent.parent / "configs" / "modded_dac_vq.yaml"
    cfg = OmegaConf.load(str(config_path))
    codec = instantiate(cfg)

    state_dict = torch.load(codec_checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }
    codec.load_state_dict(state_dict, strict=False)
    codec.eval()
    codec.to(device=device, dtype=precision)
    return codec


@torch.inference_mode()
def encode_audio(audio_path, codec, device):
    """Encode an audio file to VQ codes."""
    import torchaudio

    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav.to(device), sr, codec.sample_rate)[0]

    # Match codec model dtype (e.g. bfloat16)
    model_dtype = next(codec.parameters()).dtype
    audios = wav[None, None].to(dtype=model_dtype)  # (1, 1, T)
    audio_lengths = torch.tensor([len(wav)], device=device, dtype=torch.long)

    indices, feature_lengths = codec.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]]  # (num_codebooks, T)


@torch.inference_mode()
def decode_to_audio(codes, codec):
    """Decode VQ codes to audio waveform."""
    # codes: (num_codebooks, T) -> (1, num_codebooks, T)
    audio = codec.from_indices(codes[None])
    return audio[0, 0]  # (T,) mono waveform


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def split_text_by_speaker(text: str) -> list[str]:
    """
    Split text into turns based on <|speaker:X|> tags.

    Args:
        text: The full text with speaker tags

    Returns:
        List of speaker turns, each starting with <|speaker:X|>
    """
    pattern = r"(<\|speaker:\d+\|>)"
    parts = re.split(pattern, text)

    turns = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if re.match(pattern, part):
            if i + 1 < len(parts):
                turn = part + parts[i + 1]
                turns.append(turn.strip())
                i += 2
            else:
                turns.append(part)
                i += 1
        else:
            i += 1

    return turns


def group_turns_into_batches(
    turns: list[str], max_speakers: int = 3, max_bytes: int = 300
) -> list[str]:
    """
    Group turns into batches based on speaker count or byte limit.

    Args:
        turns: List of speaker turns
        max_speakers: Maximum number of speakers per batch (default 3)
        max_bytes: Maximum UTF-8 bytes per batch (default 300)

    Returns:
        List of batched text strings
    """
    batches = []
    current_batch = []
    current_bytes = 0

    for turn in turns:
        turn_bytes = len(turn.encode("utf-8"))

        would_exceed_speakers = len(current_batch) >= max_speakers
        would_exceed_bytes = current_bytes + turn_bytes > max_bytes and current_batch

        if would_exceed_speakers or would_exceed_bytes:
            batches.append("\n".join(current_batch))
            current_batch = [turn]
            current_bytes = turn_bytes
        else:
            current_batch.append(turn)
            current_bytes += turn_bytes

    if current_batch:
        batches.append("\n".join(current_batch))

    return batches


def generate_long(
    *,
    model,
    device: Union[str, torch.device],
    decode_one_token: Callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.9,
    top_k: int = 30,
    repetition_penalty: float = 1.1,
    temperature: float = 1.0,
    compile: bool = False,
    iterative_prompt: bool = True,
    chunk_length: int = 512,
    prompt_text: Optional[Union[str, list[str]]] = None,
    prompt_tokens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = bool(prompt_text) and bool(prompt_tokens)
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    if use_prompt:
        assert len(prompt_text) == len(
            prompt_tokens
        ), "Prompt text and tokens must have the same length"

    if prompt_tokens:
        prompt_tokens = [i.cpu() for i in prompt_tokens]

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    max_length = model.config.max_seq_len

    # Build base conversation with system message
    base_conversation = Conversation()

    if use_prompt:
        # Auto-add speaker tags to prompt texts that don't have them
        tagged_prompt_text = []
        for i, t in enumerate(prompt_text):
            if not re.search(r"<\|speaker:\d+\|>", t):
                tagged_prompt_text.append(f"<|speaker:{i}|>{t}")
            else:
                tagged_prompt_text.append(t)

        system_parts = [
            TextPart(
                text="convert the provided text to speech reference to the following:\n\nText:\n",
                cal_loss=False,
            ),
        ]
        reference_text = "\n".join(tagged_prompt_text)
        system_parts.append(TextPart(text=reference_text, cal_loss=False))
        system_parts.append(TextPart(text="\n\nSpeech:\n", cal_loss=False))
        all_codes = torch.cat([c for c in prompt_tokens], dim=1)
        system_parts.append(VQPart(codes=all_codes, cal_loss=False))
        # torch.save(all_codes, "debug_vq_codes.pt")
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

    # Split text by speaker and group into batches
    turns = split_text_by_speaker(text)
    if turns:
        batches = group_turns_into_batches(
            turns, max_speakers=5, max_bytes=chunk_length
        )
    else:
        batches = [text]

    logger.info(f"Split into {len(turns)} turns, grouped into {len(batches)} batches")

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # Deep copy base conversation for this sample
        conversation = deepcopy(base_conversation)

        for batch_idx, batch_text in enumerate(batches):
            logger.info(
                f"--- Sample {sample_idx}, Batch {batch_idx} "
                f"({len(batch_text.encode('utf-8'))} bytes) ---"
            )
            logger.info(f"Batch text: {batch_text}")

            # Add user message
            conversation.append(
                Message(
                    role="user",
                    parts=[TextPart(text=batch_text, cal_loss=False)],
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            # Deep copy for generation (don't pollute original conversation)
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

            logger.info("Visualizing prompt structure:")
            conversation_gen.visualize(
                tokenizer,
                merge_audio_tokens=True,
                merge_semantic_tokens=True,
            )

            encoded, audio_masks, audio_parts = conversation_gen.encode_for_inference(
                tokenizer, num_codebooks=model.config.num_codebooks
            )

            logger.info(f"Encoded prompt shape: {encoded.shape}")
            if audio_parts is not None:
                logger.info(f"Audio parts shape: {audio_parts.shape}")
            if audio_masks is not None:
                logger.info(
                    f"Audio masks non-zero count: {torch.count_nonzero(audio_masks)}"
                )

            if encoded.size(1) > max_length - 2048:
                raise ValueError(
                    f"Prompt is too long: {encoded.size(1)} > {max_length - 2048}"
                )

            encoded = encoded.to(device=device)
            prompt_length = encoded.size(1)

            y = generate(
                model=model,
                prompt=encoded,
                max_new_tokens=max_new_tokens,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            if sample_idx == 0 and batch_idx == 0 and compile:
                logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_batch = time.perf_counter() - t0
            tokens_generated = y.size(1) - prompt_length
            tokens_sec = tokens_generated / t_batch if t_batch > 0 else 0
            logger.info(
                f"Batch {batch_idx}: Generated {tokens_generated} tokens in "
                f"{t_batch:.02f} seconds, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )

            # Extract generated codes
            codes = y[1:, prompt_length:-1].clone()
            assert (codes >= 0).all(), f"Negative code found: {codes}"

            # Add assistant message with generated codes back to conversation
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

            yield GenerateResponse(action="sample", codes=codes, text=batch_text)

            # Cleanup
            del y, encoded

        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[Union[GenerateResponse, Exception]] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = init_model(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )

                # Only clear cache after complete request batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))
                # Clear cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


@click.command()
@click.option(
    "--text",
    type=str,
    default="<|speaker:0|>你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option(
    "--prompt-audio",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.9)
@click.option("--top-k", type=int, default=30)
@click.option("--temperature", type=float, default=1.0)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/s2-pro",
)
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=300)
@click.option("--output-dir", type=Path, default="output")
def main(
    text: str,
    prompt_text: Optional[tuple[str, ...]],
    prompt_tokens: Optional[tuple[Path, ...]],
    prompt_audio: Optional[tuple[Path, ...]],
    output: Optional[Path],
    num_samples: int,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16

    if prompt_text and not prompt_audio and not prompt_tokens:
        raise ValueError(
            "--prompt-text requires either --prompt-audio or --prompt-tokens"
        )
    if prompt_text and prompt_tokens and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )
    if prompt_text and prompt_audio and len(prompt_text) != len(prompt_audio):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt audio ({len(prompt_audio)}) should be the same"
        )

    logger.info("Loading model ...")
    t0 = time.time()
    model, decode_one_token = init_model(
        checkpoint_path, device, precision, compile=compile
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    codec = None
    codec_checkpoint = checkpoint_path / "codec.pth"

    # Handle prompt: --prompt-audio takes priority over --prompt-tokens
    prompt_tokens_list = None
    if prompt_audio:
        logger.info("Loading codec model for audio encoding...")
        codec = load_codec_model(codec_checkpoint, device, precision)
        prompt_tokens_list = [
            encode_audio(p, codec, device).cpu() for p in prompt_audio
        ]
        logger.info(f"Encoded {len(prompt_audio)} audio file(s) to VQ codes")
    elif prompt_tokens is not None:
        prompt_tokens_list = [torch.from_numpy(np.load(p)) for p in prompt_tokens]

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=list(prompt_text) if prompt_text else None,
        prompt_tokens=prompt_tokens_list,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                merged_codes = torch.cat(codes, dim=1)
                codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                np.save(codes_npy_path, merged_codes.cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")

                # Decode to wav if --output is specified
                if output:
                    if codec is None:
                        logger.info("Loading codec model for audio decoding...")
                        codec = load_codec_model(codec_checkpoint, device, precision)
                    audio = decode_to_audio(merged_codes.to(device), codec)
                    import soundfile as sf

                    out_path = (
                        str(output)
                        if num_samples == 1
                        else str(output.with_stem(f"{output.stem}_{idx}"))
                    )
                    sf.write(out_path, audio.cpu().float().numpy(), codec.sample_rate)
                    logger.info(f"Saved audio to {out_path}")

            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()
