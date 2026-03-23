"""
Runtime patches for fish-speech model compatibility.

These patches fix issues in the upstream llama.py code that arise
in specific environments (e.g., Windows, Turing GPUs, PyTorch nightly).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from loguru import logger


def patch_attention_sdpa():
    """Remove flash-attention-only constraint from Attention.forward.

    llama.py wraps the mask=None SDPA call in
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    which fails when flash attention is not compiled (Windows builds,
    some PyTorch versions). This patch removes that restriction and
    lets PyTorch auto-select the best available backend.

    Safe to call multiple times (idempotent).
    """
    from fish_speech.models.text2semantic.llama import Attention, apply_rotary_emb

    # Guard: don't patch twice
    if getattr(Attention, '_gguf_sdpa_patched', False):
        return

    def _patched_forward(self, x, freqs_cis, mask, input_pos=None):
        bsz, seqlen, _ = x.shape

        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.attention_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.use_sdpa:
            if mask is None:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        else:
            y = self.eq_scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, q_size)
        return self.wo(y)

    Attention.forward = _patched_forward
    Attention._gguf_sdpa_patched = True
    logger.debug("  Patched Attention.forward (removed flash-only SDPA constraint)")
