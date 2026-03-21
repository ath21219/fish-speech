"""
GGUF on-the-fly dequantization for PyTorch.

Weights are stored in quantized form on GPU. At each forward pass,
they are dequantized just before matmul, then the dequantized buffer
is discarded. This keeps VRAM usage close to the quantized size.

Reference implementations:
  - llama.cpp gguf-py/gguf/quants.py (NumPy, MIT license)
  - city96/ComfyUI-GGUF/dequant.py (PyTorch, Apache-2.0)
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf
from loguru import logger

QK_K = 256
K_SCALE_SIZE = 12

NATIVE_TORCH_QTYPES = frozenset({
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
})


# ---------- Triton kernels (optional acceleration) ----------
from fish_speech.gguf.triton_kernels import (
    _TRITON_AVAILABLE,
    triton_dequant_q6k as _triton_dequant_q6k,
    fused_dequant_gemv_q6k,
)

if _TRITON_AVAILABLE:
    logger.info("Triton Q6_K kernels loaded (dequant + fused GEMV)")
else:
    _triton_dequant_q6k = None
    fused_dequant_gemv_q6k = None
    logger.debug("Triton not available, using PyTorch dequant for Q6_K")


# ================================================================
#  Low-level dequant functions (operate on raw uint8 blocks)
# ================================================================

def _split(blocks: torch.Tensor, *sizes: int) -> list[torch.Tensor]:
    remaining = blocks.shape[1] - sum(sizes)
    return list(torch.split(blocks, list(sizes) + [remaining], dim=1))


def _to_uint32(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | (x[:, 1] << 8) | (x[:, 2] << 16) | (x[:, 3] << 24)).unsqueeze(1)


def _get_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = scales.shape[0]
    s = scales.view(torch.uint8).reshape(n, 3, 4)
    d, m, md = s[:, 0], s[:, 1], s[:, 2]
    sc = torch.cat([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape(n, 8), mn.reshape(n, 8)


def _dq_bf16(blocks, bs, ts, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

def _dq_q8_0(blocks, bs, ts, dtype=None):
    d, x = _split(blocks, 2)
    d = d.view(torch.float16).to(dtype or torch.float32)
    return d * x.view(torch.int8)

def _dq_q4_0(blocks, bs, ts, dtype=None):
    n = blocks.shape[0]; dev = blocks.device
    d, qs = _split(blocks, 2)
    d = d.view(torch.float16).to(dtype or torch.float32)
    qs = qs.reshape(n, -1, 1, bs // 2) >> torch.tensor([0, 4], device=dev, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n, -1).to(torch.int8) - 8
    return d * qs

def _dq_q4_1(blocks, bs, ts, dtype=None):
    n = blocks.shape[0]; dev = blocks.device; dt = dtype or torch.float32
    d, m, qs = _split(blocks, 2, 2)
    d = d.view(torch.float16).to(dt); m = m.view(torch.float16).to(dt)
    qs = qs.reshape(n, -1, 1, bs // 2) >> torch.tensor([0, 4], device=dev, dtype=torch.uint8).reshape(1, 1, 2, 1)
    return d * (qs & 0x0F).reshape(n, -1) + m

def _dq_q5_0(blocks, bs, ts, dtype=None):
    n = blocks.shape[0]; dev = blocks.device
    d, qh, qs = _split(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype or torch.float32)
    qh = _to_uint32(qh).reshape(n, 1) >> torch.arange(32, device=dev, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n, -1, 1, bs // 2) >> torch.tensor([0, 4], device=dev, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8); ql = (ql & 0x0F).reshape(n, -1)
    return d * ((ql | (qh << 4)).to(torch.int8) - 16)

def _dq_q5_1(blocks, bs, ts, dtype=None):
    n = blocks.shape[0]; dev = blocks.device; dt = dtype or torch.float32
    d, m, qh, qs = _split(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dt); m = m.view(torch.float16).to(dt)
    qh = _to_uint32(qh).reshape(n, 1) >> torch.arange(32, device=dev, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n, -1, 1, bs // 2) >> torch.tensor([0, 4], device=dev, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8); ql = (ql & 0x0F).reshape(n, -1)
    return d * (ql | (qh << 4)) + m

def _dq_q2_k(blocks, bs, ts, dtype=None):
    n = blocks.shape[0]; dev = blocks.device; dt = dtype or torch.float32
    sc, qs, d, dmin = _split(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dt); dmin = dmin.view(torch.float16).to(dt)
    dl = (d * (sc & 0xF).to(dt)).reshape(n, QK_K // 16, 1)
    ml = (dmin * (sc >> 4).to(dt)).reshape(n, QK_K // 16, 1)
    shift = torch.tensor([0, 2, 4, 6], device=dev, dtype=torch.uint8).reshape(1, 1, 4, 1)
    qs = ((qs.reshape(n, -1, 1, 32) >> shift) & 3).reshape(n, QK_K // 16, 16).to(dt)
    return (dl * qs - ml).reshape(n, -1)

def _dq_q3_k(blocks, bs, ts, dtype=None):
    n = blocks.shape[0]; dev = blocks.device; dt = dtype or torch.float32
    hmask, qs, sc_raw, d = _split(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dt)
    ls, hs = sc_raw[:, :8], sc_raw[:, 8:]
    ls = ls.reshape(n, 1, 8) >> torch.tensor([0, 4], device=dev, dtype=torch.uint8).reshape(1, 2, 1)
    ls = ls.reshape(n, 16)
    hs = hs.reshape(n, 1, 4) >> torch.tensor([0, 2, 4, 6], device=dev, dtype=torch.uint8).reshape(1, 4, 1)
    hs = hs.reshape(n, 16)
    sc = ((ls & 0x0F) | ((hs & 0x03) << 4)).to(torch.int8) - 32
    dl = (d * sc.to(dt)).reshape(n, 16, 1)
    ql = qs.reshape(n, -1, 1, 32) >> torch.tensor([0, 2, 4, 6], device=dev, dtype=torch.uint8).reshape(1, 1, 4, 1)
    qh = hmask.reshape(n, -1, 1, 32) >> torch.tensor(list(range(8)), device=dev, dtype=torch.uint8).reshape(1, 1, 8, 1)
    ql = (ql.reshape(n, 16, QK_K // 16) & 3)
    qh = (qh.reshape(n, 16, QK_K // 16) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q.to(dt)).reshape(n, QK_K)

def _dq_q4_k(blocks, bs, ts, dtype=None):
    n = blocks.shape[0]; dev = blocks.device; dt = dtype or torch.float32
    d, dmin, scales, qs = _split(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dt); dmin = dmin.view(torch.float16).to(dt)
    sc, m = _get_scale_min(scales)
    d = (d * sc.to(dt)).reshape(n, -1, 1); dm = (dmin * m.to(dt)).reshape(n, -1, 1)
    qs = qs.reshape(n, -1, 1, 32) >> torch.tensor([0, 4], device=dev, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n, -1, 32).to(dt)
    return (d * qs - dm).reshape(n, QK_K)

def _dq_q5_k(blocks, bs, ts, dtype=None):
    n = blocks.shape[0]; dev = blocks.device; dt = dtype or torch.float32
    d, dmin, scales, qh, qs = _split(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dt); dmin = dmin.view(torch.float16).to(dt)
    sc, m = _get_scale_min(scales)
    d = (d * sc.to(dt)).reshape(n, -1, 1); dm = (dmin * m.to(dt)).reshape(n, -1, 1)
    ql = qs.reshape(n, -1, 1, 32) >> torch.tensor([0, 4], device=dev, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = qh.reshape(n, -1, 1, 32) >> torch.tensor(list(range(8)), device=dev, dtype=torch.uint8).reshape(1, 1, 8, 1)
    ql = (ql & 0x0F).reshape(n, -1, 32); qh = (qh & 0x01).reshape(n, -1, 32)
    return (d * (ql | (qh << 4)).to(dt) - dm).reshape(n, QK_K)

def _dq_q6_k(blocks, bs, ts, dtype=None):
    n = blocks.shape[0]; dev = blocks.device; dt = dtype or torch.float32
    ql, qh, scales, d = _split(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dt)
    d = d.view(torch.float16).to(dt)
    d = (d * scales).reshape(n, QK_K // 16, 1)
    ql = ql.reshape(n, -1, 1, 64) >> torch.tensor([0, 4], device=dev, dtype=torch.uint8).reshape(1, 1, 2, 1)
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = qh.reshape(n, -1, 1, 32) >> torch.tensor([0, 2, 4, 6], device=dev, dtype=torch.uint8).reshape(1, 1, 4, 1)
    qh = (qh & 0x03).reshape(n, -1, 32)
    q = (ql | (qh << 4)).to(torch.int8) - 32
    return (d * q.reshape(n, QK_K // 16, -1).to(dt)).reshape(n, QK_K)


DEQUANT_FN = {
    gguf.GGMLQuantizationType.BF16:  _dq_bf16,
    gguf.GGMLQuantizationType.Q8_0:  _dq_q8_0,
    gguf.GGMLQuantizationType.Q4_0:  _dq_q4_0,
    gguf.GGMLQuantizationType.Q4_1:  _dq_q4_1,
    gguf.GGMLQuantizationType.Q5_0:  _dq_q5_0,
    gguf.GGMLQuantizationType.Q5_1:  _dq_q5_1,
    gguf.GGMLQuantizationType.Q2_K:  _dq_q2_k,
    gguf.GGMLQuantizationType.Q3_K:  _dq_q3_k,
    gguf.GGMLQuantizationType.Q4_K:  _dq_q4_k,
    gguf.GGMLQuantizationType.Q5_K:  _dq_q5_k,
    gguf.GGMLQuantizationType.Q6_K:  _dq_q6_k,
}


def dequantize_tensor(
    data: torch.Tensor,
    qtype: gguf.GGMLQuantizationType,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize raw GGUF bytes into a float tensor."""
    if qtype == gguf.GGMLQuantizationType.F32:
        return data.view(torch.float32).reshape(shape).to(dtype)
    if qtype == gguf.GGMLQuantizationType.F16:
        return data.view(torch.float16).reshape(shape).to(dtype)
    if qtype not in DEQUANT_FN:
        raise NotImplementedError(f"No dequant for {qtype.name}")

    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rows = data.reshape(-1, data.shape[-1]).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape(n_blocks, type_size)
    return DEQUANT_FN[qtype](blocks, block_size, type_size).reshape(shape).to(dtype)


# ================================================================
#  GGUFLinear: Drop-in replacement for nn.Linear
#  Keeps weights quantized, dequantizes on each forward()
# ================================================================

class GGUFParameter:
    """Wrapper holding quantized weight data + metadata on GPU."""
    __slots__ = ('data', 'qtype', 'shape', 'block_size', 'type_size')

    def __init__(self, data: torch.Tensor, qtype: gguf.GGMLQuantizationType,
                 shape: tuple[int, ...]):
        self.data = data            # uint8 tensor on GPU
        self.qtype = qtype
        self.shape = shape          # original (out_features, in_features)
        self.block_size, self.type_size = gguf.GGML_QUANT_SIZES[qtype]

    @property
    def device(self):
        return self.data.device

    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()

    def dequantize(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        if (
            _TRITON_AVAILABLE
            and _triton_dequant_q6k is not None
            and self.qtype == 14  # Q6_K
            and self.data.is_cuda
        ):
            return _triton_dequant_q6k(self.data, self.shape, dtype)

        # Fallback: PyTorch dequant
        fn = DEQUANT_FN[self.qtype]
        rows = self.data.view(torch.uint8)
        n_blocks = rows.numel() // self.type_size
        blocks = rows.reshape(n_blocks, self.type_size)
        return fn(blocks, self.block_size, self.type_size).reshape(self.shape).to(dtype)


class GGUFLinear(nn.Module):
    """Linear layer with GGUF quantized weights and generation-phase caching."""

    def __init__(self, qparam: GGUFParameter, bias=None):
        super().__init__()
        self.qparam = qparam
        self.bias = bias
        self._cached_weight: torch.Tensor | None = None
        self._cache_dtype: torch.dtype | None = None
        self._cache_enabled: bool = False  # Only cache during generation

    def forward(self, x):
        if self._cache_enabled and self._cached_weight is not None and self._cache_dtype == x.dtype:
            w = self._cached_weight
            return F.linear(x, w, self.bias)

        # Fused GEMV path: batch=1 の Q6_K 層で使用
        if (
            _TRITON_AVAILABLE
            and self.qparam.qtype == 14  # Q6_K
            and self.qparam.data.is_cuda
            and x.shape[0] == 1 and x.dim() == 3 and x.shape[1] == 1  # [1,1,D]
            and self.qparam.shape[1] % 256 == 0
        ):
            out = fused_dequant_gemv_q6k(
                x.view(-1),
                self.qparam.data,
                self.qparam.shape[1],  # D_in
                self.qparam.shape[0],  # D_out
                dtype=x.dtype,
            )
            out = out.view(1, 1, -1)
            if self.bias is not None:
                out = out + self.bias
            return out

        # Fallback: dequant + F.linear
        w = self.qparam.dequantize(dtype=x.dtype)
        if w.device != x.device:
            w = w.to(x.device)
        if self._cache_enabled:
            self._cached_weight = w
            self._cache_dtype = x.dtype
        return F.linear(x, w, self.bias)

    def clear_cache(self):
        self._cached_weight = None
        self._cache_dtype = None

    def enable_cache(self):
        self._cache_enabled = True

    def disable_cache(self):
        self._cache_enabled = False
        self.clear_cache()


class GGUFEmbedding(nn.Module):
    """Embedding that stores the weight table in GGUF quantized format.

    On forward(), the full table is dequantized, indexed, then discarded.
    For very large vocabs this is still cheaper than storing float16.
    """

    def __init__(self, qparam: GGUFParameter):
        super().__init__()
        self.qparam = qparam
        self.num_embeddings, self.embedding_dim = qparam.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qparam.dequantize(dtype=torch.float16)  # embedding always f16
        return F.embedding(x, w)

    @property
    def weight(self):
        """For compatibility with F.linear(x, embeddings.weight) in tie_word_embeddings."""
        return self.qparam.dequantize(dtype=torch.float16)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"qtype={self.qparam.qtype.name}"
        )


def warmup_all_caches(model: nn.Module, dtype: torch.dtype = torch.float16) -> int:
    """Pre-dequantize all GGUFLinear weights. Returns count of cached modules."""
    count = 0
    for module in model.modules():
        if isinstance(module, GGUFLinear):
            module.warmup_cache(dtype)
            count += 1
    return count


def clear_all_caches(model: nn.Module) -> int:
    """Release all dequant caches. Returns count of cleared modules."""
    count = 0
    for module in model.modules():
        if isinstance(module, GGUFLinear):
            module.clear_cache()
            count += 1
    return count


def release_quantized_data(model: nn.Module) -> float:
    """After caching, release the raw quantized data to save VRAM.
    Returns GB freed."""
    freed = 0
    for module in model.modules():
        if isinstance(module, GGUFLinear) and module._cached_weight is not None:
            freed += module.qparam.data.nbytes
            # Replace with tiny placeholder to keep structure intact
            module.qparam.data = torch.empty(0, dtype=torch.uint8,
                                              device=module.qparam.data.device)
    return freed / 1e9

def setup_layer_cache_hooks(model: nn.Module) -> list:
    """Install forward hooks that cache/clear dequant per TransformerBlock.

    During decode, only one layer's weights are cached at a time.
    Each layer has 5 GGUFLinear modules (wqkv, wo, w1, w3, w2).
    5 weights × ~25MB avg ≈ 125 MB temporary VRAM per layer.
    """
    from fish_speech.models.text2semantic.llama import TransformerBlock

    hooks = []

    def make_pre_hook(block):
        def pre_hook(module, input):
            for m in module.modules():
                if isinstance(m, GGUFLinear):
                    m.enable_cache()
        return pre_hook

    def make_post_hook(block):
        def post_hook(module, input, output):
            for m in module.modules():
                if isinstance(m, GGUFLinear):
                    m.disable_cache()
        return post_hook

    for name, module in model.named_modules():
        if isinstance(module, TransformerBlock):
            h1 = module.register_forward_pre_hook(make_pre_hook(module))
            h2 = module.register_forward_hook(make_post_hook(module))
            hooks.append(h1)
            hooks.append(h2)

    count = len(hooks) // 2
    logger.debug(f"Installed layer-cache hooks on {count} TransformerBlocks")
    return hooks


def warmup_all_caches(model: nn.Module, dtype: torch.dtype = torch.float16) -> int:
    """DEPRECATED for 8GB GPUs. Use setup_layer_cache_hooks instead.
    Pre-dequantize all GGUFLinear weights. Only use if VRAM > 16GB."""
    count = 0
    for module in model.modules():
        if isinstance(module, GGUFLinear):
            module._cache_enabled = True
            w = module.qparam.dequantize(dtype=dtype)
            if module.qparam.data.is_cuda:
                module._cached_weight = w
            else:
                module._cached_weight = w.to(module.qparam.data.device)
            module._cache_dtype = dtype
            count += 1
    return count


def clear_all_caches(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, GGUFLinear):
            module.disable_cache()
            count += 1
    return count
