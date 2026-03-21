"""
Triton GPU kernels for GGUF Q6_K dequantization and fused GEMV.

Requires:
  - triton >= 3.2.0 (tested with triton-windows 3.2.0.post21)
  - NVIDIA GPU with compute capability >= 7.5 (Turing+)

These kernels are optional. If Triton is unavailable, the system
falls back to PyTorch-based dequantization automatically.
"""

from __future__ import annotations
from typing import Optional
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

if _TRITON_AVAILABLE:

    # ================================================================
    #  Q6_K chunk decoders (shared by both dequant and fused GEMV)
    # ================================================================
    # Each chunk decodes 32 values from a 210-byte Q6_K block.
    # 8 chunks × 32 = 256 values per block.
    #
    # Q6_K block layout (210 bytes):
    #   ql[128]   bytes 0..127    lower 4 bits (two nibbles per byte)
    #   qh[64]    bytes 128..191  upper 2 bits (four 2-bit fields per byte)
    #   scales[16] bytes 192..207  per-chunk scales (int8)
    #   d[2]      bytes 208..209  super-block scale (float16)

    @triton.jit
    def _decode_q6k_chunk_0(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE: tl.constexpr):
        ql = tl.load(block_ptr + 0 + offsets_32)
        ql_val = (ql & 0x0F).to(tl.int8, bitcast=True)
        qh = tl.load(block_ptr + 128 + offsets_32)
        qh_val = (qh.to(tl.int8, bitcast=True)) & 0x03
        q_val = ((ql_val | (qh_val << 4)) - 32).to(DTYPE)
        mask_16 = offsets_32 < 16
        s0 = tl.load(scales_ptr).to(tl.int8, bitcast=True).to(DTYPE)
        s1 = tl.load(scales_ptr + 1).to(tl.int8, bitcast=True).to(DTYPE)
        return q_val * (d_scale * tl.where(mask_16, s0, s1))

    @triton.jit
    def _decode_q6k_chunk_1(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE: tl.constexpr):
        ql = tl.load(block_ptr + 32 + offsets_32)
        ql_val = (ql & 0x0F).to(tl.int8, bitcast=True)
        qh = tl.load(block_ptr + 128 + offsets_32)
        qh_val = (qh.to(tl.int8, bitcast=True) >> 2) & 0x03
        q_val = ((ql_val | (qh_val << 4)) - 32).to(DTYPE)
        mask_16 = offsets_32 < 16
        s0 = tl.load(scales_ptr + 2).to(tl.int8, bitcast=True).to(DTYPE)
        s1 = tl.load(scales_ptr + 3).to(tl.int8, bitcast=True).to(DTYPE)
        return q_val * (d_scale * tl.where(mask_16, s0, s1))

    @triton.jit
    def _decode_q6k_chunk_2(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE: tl.constexpr):
        ql = tl.load(block_ptr + 0 + offsets_32)
        ql_val = ((ql >> 4) & 0x0F).to(tl.int8, bitcast=True)
        qh = tl.load(block_ptr + 128 + offsets_32)
        qh_val = (qh.to(tl.int8, bitcast=True) >> 4) & 0x03
        q_val = ((ql_val | (qh_val << 4)) - 32).to(DTYPE)
        mask_16 = offsets_32 < 16
        s0 = tl.load(scales_ptr + 4).to(tl.int8, bitcast=True).to(DTYPE)
        s1 = tl.load(scales_ptr + 5).to(tl.int8, bitcast=True).to(DTYPE)
        return q_val * (d_scale * tl.where(mask_16, s0, s1))

    @triton.jit
    def _decode_q6k_chunk_3(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE: tl.constexpr):
        ql = tl.load(block_ptr + 32 + offsets_32)
        ql_val = ((ql >> 4) & 0x0F).to(tl.int8, bitcast=True)
        qh = tl.load(block_ptr + 128 + offsets_32)
        qh_val = (qh.to(tl.int8, bitcast=True) >> 6) & 0x03
        q_val = ((ql_val | (qh_val << 4)) - 32).to(DTYPE)
        mask_16 = offsets_32 < 16
        s0 = tl.load(scales_ptr + 6).to(tl.int8, bitcast=True).to(DTYPE)
        s1 = tl.load(scales_ptr + 7).to(tl.int8, bitcast=True).to(DTYPE)
        return q_val * (d_scale * tl.where(mask_16, s0, s1))

    @triton.jit
    def _decode_q6k_chunk_4(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE: tl.constexpr):
        ql = tl.load(block_ptr + 64 + offsets_32)
        ql_val = (ql & 0x0F).to(tl.int8, bitcast=True)
        qh = tl.load(block_ptr + 160 + offsets_32)
        qh_val = (qh.to(tl.int8, bitcast=True)) & 0x03
        q_val = ((ql_val | (qh_val << 4)) - 32).to(DTYPE)
        mask_16 = offsets_32 < 16
        s0 = tl.load(scales_ptr + 8).to(tl.int8, bitcast=True).to(DTYPE)
        s1 = tl.load(scales_ptr + 9).to(tl.int8, bitcast=True).to(DTYPE)
        return q_val * (d_scale * tl.where(mask_16, s0, s1))

    @triton.jit
    def _decode_q6k_chunk_5(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE: tl.constexpr):
        ql = tl.load(block_ptr + 96 + offsets_32)
        ql_val = (ql & 0x0F).to(tl.int8, bitcast=True)
        qh = tl.load(block_ptr + 160 + offsets_32)
        qh_val = (qh.to(tl.int8, bitcast=True) >> 2) & 0x03
        q_val = ((ql_val | (qh_val << 4)) - 32).to(DTYPE)
        mask_16 = offsets_32 < 16
        s0 = tl.load(scales_ptr + 10).to(tl.int8, bitcast=True).to(DTYPE)
        s1 = tl.load(scales_ptr + 11).to(tl.int8, bitcast=True).to(DTYPE)
        return q_val * (d_scale * tl.where(mask_16, s0, s1))

    @triton.jit
    def _decode_q6k_chunk_6(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE: tl.constexpr):
        ql = tl.load(block_ptr + 64 + offsets_32)
        ql_val = ((ql >> 4) & 0x0F).to(tl.int8, bitcast=True)
        qh = tl.load(block_ptr + 160 + offsets_32)
        qh_val = (qh.to(tl.int8, bitcast=True) >> 4) & 0x03
        q_val = ((ql_val | (qh_val << 4)) - 32).to(DTYPE)
        mask_16 = offsets_32 < 16
        s0 = tl.load(scales_ptr + 12).to(tl.int8, bitcast=True).to(DTYPE)
        s1 = tl.load(scales_ptr + 13).to(tl.int8, bitcast=True).to(DTYPE)
        return q_val * (d_scale * tl.where(mask_16, s0, s1))

    @triton.jit
    def _decode_q6k_chunk_7(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE: tl.constexpr):
        ql = tl.load(block_ptr + 96 + offsets_32)
        ql_val = ((ql >> 4) & 0x0F).to(tl.int8, bitcast=True)
        qh = tl.load(block_ptr + 160 + offsets_32)
        qh_val = (qh.to(tl.int8, bitcast=True) >> 6) & 0x03
        q_val = ((ql_val | (qh_val << 4)) - 32).to(DTYPE)
        mask_16 = offsets_32 < 16
        s0 = tl.load(scales_ptr + 14).to(tl.int8, bitcast=True).to(DTYPE)
        s1 = tl.load(scales_ptr + 15).to(tl.int8, bitcast=True).to(DTYPE)
        return q_val * (d_scale * tl.where(mask_16, s0, s1))

    # ================================================================
    #  Standalone Q6_K dequantization kernel
    # ================================================================

    @triton.jit
    def _dequantize_q6_k_kernel(
        q_ptr, out_ptr, n_total_blocks, DTYPE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        if pid >= n_total_blocks:
            return
        block_start_ptr = q_ptr + pid * 210
        out_start_ptr = out_ptr + pid * 256
        offsets_32 = tl.arange(0, 32)
        d_super_scale = tl.load(
            (block_start_ptr + 208).to(tl.pointer_type(tl.float16))
        ).to(DTYPE)
        scales_ptr = block_start_ptr + 192

        w0 = _decode_q6k_chunk_0(block_start_ptr, offsets_32, scales_ptr, d_super_scale, DTYPE)
        tl.store(out_start_ptr + 0 * 32 + offsets_32, w0)
        w1 = _decode_q6k_chunk_1(block_start_ptr, offsets_32, scales_ptr, d_super_scale, DTYPE)
        tl.store(out_start_ptr + 1 * 32 + offsets_32, w1)
        w2 = _decode_q6k_chunk_2(block_start_ptr, offsets_32, scales_ptr, d_super_scale, DTYPE)
        tl.store(out_start_ptr + 2 * 32 + offsets_32, w2)
        w3 = _decode_q6k_chunk_3(block_start_ptr, offsets_32, scales_ptr, d_super_scale, DTYPE)
        tl.store(out_start_ptr + 3 * 32 + offsets_32, w3)
        w4 = _decode_q6k_chunk_4(block_start_ptr, offsets_32, scales_ptr, d_super_scale, DTYPE)
        tl.store(out_start_ptr + 4 * 32 + offsets_32, w4)
        w5 = _decode_q6k_chunk_5(block_start_ptr, offsets_32, scales_ptr, d_super_scale, DTYPE)
        tl.store(out_start_ptr + 5 * 32 + offsets_32, w5)
        w6 = _decode_q6k_chunk_6(block_start_ptr, offsets_32, scales_ptr, d_super_scale, DTYPE)
        tl.store(out_start_ptr + 6 * 32 + offsets_32, w6)
        w7 = _decode_q6k_chunk_7(block_start_ptr, offsets_32, scales_ptr, d_super_scale, DTYPE)
        tl.store(out_start_ptr + 7 * 32 + offsets_32, w7)

    # ================================================================
    #  Fused Q6_K dequant + GEMV kernel (batch=1 decode)
    # ================================================================

    @triton.jit
    def _fused_dequant_gemv_q6k_kernel(
        input_ptr,
        q_weight_ptr,
        output_ptr,
        D_in: tl.constexpr,
        D_out,
        BLOCK_SIZE_Q6K: tl.constexpr,
        VALS_PER_BLOCK: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= D_out:
            return

        n_blocks_per_row: tl.constexpr = D_in // VALS_PER_BLOCK
        offsets_32 = tl.arange(0, 32)
        acc_32 = tl.zeros([32], dtype=tl.float32)

        for block_idx in range(n_blocks_per_row):
            block_byte_offset = (row_idx * n_blocks_per_row + block_idx) * BLOCK_SIZE_Q6K
            block_ptr = q_weight_ptr + block_byte_offset
            scales_ptr = block_ptr + 192
            d_ptr = block_ptr + 208
            d_scale = tl.load(d_ptr.to(tl.pointer_type(tl.float16))).to(DTYPE)
            input_base = block_idx * VALS_PER_BLOCK

            w0 = _decode_q6k_chunk_0(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE)
            x0 = tl.load(input_ptr + input_base + 0 * 32 + offsets_32)
            acc_32 += w0.to(tl.float32) * x0.to(tl.float32)

            w1 = _decode_q6k_chunk_1(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE)
            x1 = tl.load(input_ptr + input_base + 1 * 32 + offsets_32)
            acc_32 += w1.to(tl.float32) * x1.to(tl.float32)

            w2 = _decode_q6k_chunk_2(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE)
            x2 = tl.load(input_ptr + input_base + 2 * 32 + offsets_32)
            acc_32 += w2.to(tl.float32) * x2.to(tl.float32)

            w3 = _decode_q6k_chunk_3(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE)
            x3 = tl.load(input_ptr + input_base + 3 * 32 + offsets_32)
            acc_32 += w3.to(tl.float32) * x3.to(tl.float32)

            w4 = _decode_q6k_chunk_4(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE)
            x4 = tl.load(input_ptr + input_base + 4 * 32 + offsets_32)
            acc_32 += w4.to(tl.float32) * x4.to(tl.float32)

            w5 = _decode_q6k_chunk_5(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE)
            x5 = tl.load(input_ptr + input_base + 5 * 32 + offsets_32)
            acc_32 += w5.to(tl.float32) * x5.to(tl.float32)

            w6 = _decode_q6k_chunk_6(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE)
            x6 = tl.load(input_ptr + input_base + 6 * 32 + offsets_32)
            acc_32 += w6.to(tl.float32) * x6.to(tl.float32)

            w7 = _decode_q6k_chunk_7(block_ptr, offsets_32, scales_ptr, d_scale, DTYPE)
            x7 = tl.load(input_ptr + input_base + 7 * 32 + offsets_32)
            acc_32 += w7.to(tl.float32) * x7.to(tl.float32)

        result = tl.sum(acc_32, axis=0).to(DTYPE)
        tl.store(output_ptr + row_idx, result)

    # ================================================================
    #  Python wrappers
    # ================================================================

    def triton_dequant_q6k(
        data: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Dequantize Q6_K raw bytes to float tensor using Triton kernel."""
        raw = data.view(torch.uint8)
        n_blocks = raw.numel() // 210
        out = torch.empty(n_blocks * 256, dtype=dtype, device=data.device)
        _dequantize_q6_k_kernel[(n_blocks,)](
            raw, out, n_blocks, DTYPE=tl.float32,
        )
        return out.reshape(shape)

    def fused_dequant_gemv_q6k(
        input_vec: torch.Tensor,
        q_weight_data: torch.Tensor,
        D_in: int,
        D_out: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Fused Q6_K dequant + GEMV for batch=1 decode.

        Args:
            input_vec: [D_in] or [1, D_in] activation vector (FP16)
            q_weight_data: raw Q6_K bytes on GPU
            D_in: input dimension (must be multiple of 256)
            D_out: output dimension
            dtype: output dtype

        Returns:
            [D_out] output vector
        """
        assert D_in % 256 == 0, f"D_in must be multiple of 256, got {D_in}"
        input_flat = input_vec.view(-1).contiguous()
        output = torch.empty(D_out, device=input_vec.device, dtype=dtype)
        TRITON_DTYPE = tl.float16 if dtype == torch.float16 else tl.bfloat16

        _fused_dequant_gemv_q6k_kernel[(D_out,)](
            input_flat,
            q_weight_data,
            output,
            D_in=D_in,
            D_out=D_out,
            BLOCK_SIZE_Q6K=210,
            VALS_PER_BLOCK=256,
            DTYPE=TRITON_DTYPE,
        )
        return output

else:
    # Triton not available — provide None stubs
    triton_dequant_q6k = None
    fused_dequant_gemv_q6k = None
