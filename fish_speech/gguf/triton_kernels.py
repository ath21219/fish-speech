"""
Triton GPU kernels for GGUF Q6_K/Q3_K dequantization and fused GEMV.

Requires:
  - triton >= 3.2.0 (tested with triton-windows 3.2.0.post21)
  - NVIDIA GPU with compute capability >= 7.5 (Turing+)

These kernels are optional. If Triton is unavailable, the system
falls back to PyTorch-based dequantization automatically.

v2 improvements:
  - Q6_K GEMV: eliminated redundant ql/qh loads (6→4 loads per block),
    pre-loaded all 16 scales, autotuned num_warps
  - Q3_K GEMV: autotuned num_warps
  - Both: maintained numerical correctness with original kernels
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
    #  Q3_K block layout (110 bytes per 256 values)
    #  hmask[32]  bytes 0..31    high-bit mask
    #  qs[64]     bytes 32..95   2-bit quantized values
    #  scales[12] bytes 96..107  packed 6-bit scales (16 values)
    #  d[2]       bytes 108..109 super-block scale (float16)
    # ================================================================

    @triton.jit
    def _q3k_decode_scale(scales_ptr, idx: tl.constexpr, DTYPE: tl.constexpr):
        """Decode one 6-bit scale from packed 12 scale bytes."""
        ls = (tl.load(scales_ptr + (idx % 8)) >> ((idx // 8) * 4)) & 0x0F
        hs = (tl.load(scales_ptr + 8 + (idx % 4)) >> ((idx // 4) * 2)) & 0x03
        return (ls | (hs << 4)).to(tl.int8) - 32

    # ================================================================
    #  Q3_K fused dequant + GEMV (v2: autotuned num_warps)
    # ================================================================

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
        ],
        key=["D_in"],
    )
    @triton.jit
    def _fused_dequant_gemv_q3k_kernel(
        input_ptr,
        q_weight_ptr,
        output_ptr,
        D_in: tl.constexpr,
        D_out,
        BLOCK_SIZE_Q3K: tl.constexpr,  # 110
        VALS_PER_BLOCK: tl.constexpr,  # 256
        DTYPE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= D_out:
            return

        n_blocks_per_row: tl.constexpr = D_in // VALS_PER_BLOCK
        offsets_32 = tl.arange(0, 32)
        mask_16 = offsets_32 < 16
        acc = tl.zeros([32], dtype=tl.float32)

        for block_idx in range(n_blocks_per_row):
            block_off = (row_idx * n_blocks_per_row + block_idx) * BLOCK_SIZE_Q3K
            block_ptr = q_weight_ptr + block_off
            input_base = block_idx * VALS_PER_BLOCK

            d_scale = tl.load(
                (block_ptr + 108).to(tl.pointer_type(tl.float16))
            ).to(DTYPE)

            hmask = tl.load(block_ptr + offsets_32)
            qs_lo = tl.load(block_ptr + 32 + offsets_32)
            qs_hi = tl.load(block_ptr + 64 + offsets_32)
            scales_ptr = block_ptr + 96

            # Super-chunks 0-3: qs_lo
            for sc in tl.static_range(4):
                ql = (qs_lo >> (sc * 2)) & 0x03
                hbit_inv = ((hmask >> sc) & 0x01) ^ 0x01
                q_val = ql.to(tl.int8) - (hbit_inv << 2).to(tl.int8)

                sc_a = _q3k_decode_scale(scales_ptr, sc * 2, DTYPE)
                sc_b = _q3k_decode_scale(scales_ptr, sc * 2 + 1, DTYPE)
                final_scale = d_scale * tl.where(
                    mask_16, sc_a.to(DTYPE), sc_b.to(DTYPE)
                )

                w = final_scale * q_val.to(DTYPE)
                x = tl.load(input_ptr + input_base + sc * 32 + offsets_32)
                acc += w.to(tl.float32) * x.to(tl.float32)

            # Super-chunks 4-7: qs_hi
            for sc in tl.static_range(4):
                sc_idx = sc + 4
                ql = (qs_hi >> (sc * 2)) & 0x03
                hbit_inv = ((hmask >> sc_idx) & 0x01) ^ 0x01
                q_val = ql.to(tl.int8) - (hbit_inv << 2).to(tl.int8)

                sc_a = _q3k_decode_scale(scales_ptr, sc_idx * 2, DTYPE)
                sc_b = _q3k_decode_scale(scales_ptr, sc_idx * 2 + 1, DTYPE)
                final_scale = d_scale * tl.where(
                    mask_16, sc_a.to(DTYPE), sc_b.to(DTYPE)
                )

                w = final_scale * q_val.to(DTYPE)
                x = tl.load(
                    input_ptr + input_base + sc_idx * 32 + offsets_32
                )
                acc += w.to(tl.float32) * x.to(tl.float32)

        result = tl.sum(acc, axis=0).to(DTYPE)
        tl.store(output_ptr + row_idx, result)

    # ================================================================
    #  Q6_K chunk decoders (shared by standalone dequant kernel)
    # ================================================================

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
    #  Standalone Q6_K dequantization kernel (unchanged)
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
    #  Fused Q6_K dequant + GEMV kernel (v2: reduced loads + autotune)
    #
    #  Key improvements over v1:
    #   1. ql bytes loaded once, reused for low/high nibble chunks:
    #      - ql_0 (offset 0):  chunk 0 (low nibble) + chunk 2 (high nibble)
    #      - ql_1 (offset 32): chunk 1 (low nibble) + chunk 3 (high nibble)
    #      - ql_2 (offset 64): chunk 4 (low nibble) + chunk 6 (high nibble)
    #      - ql_3 (offset 96): chunk 5 (low nibble) + chunk 7 (high nibble)
    #   2. qh bytes loaded once, reused for 4 chunks each:
    #      - qh_0 (offset 128): chunks 0,1,2,3 (shift 0,2,4,6)
    #      - qh_1 (offset 160): chunks 4,5,6,7 (shift 0,2,4,6)
    #   3. Total loads per block: 4(ql) + 2(qh) + 16(scales) + 1(d) + 8(input)
    #      vs v1: 8(ql) + 8(qh) + 16(scales) + 1(d) + 8(input)
    #      → 10 fewer vector loads per block (saves ~640 bytes bandwidth)
    #   4. Autotuned num_warps for Turing SM75
    # ================================================================

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
        ],
        key=["D_in"],
    )
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
        mask_16 = offsets_32 < 16
        acc = tl.zeros([32], dtype=tl.float32)

        for block_idx in range(n_blocks_per_row):
            block_byte_offset = (row_idx * n_blocks_per_row + block_idx) * BLOCK_SIZE_Q6K
            block_ptr = q_weight_ptr + block_byte_offset
            scales_ptr = block_ptr + 192
            d_scale = tl.load(
                (block_ptr + 208).to(tl.pointer_type(tl.float16))
            ).to(DTYPE)
            input_base = block_idx * VALS_PER_BLOCK

            # ---- Pre-load: 4 ql segments + 2 qh segments (6 loads vs 16) ----
            ql_0 = tl.load(block_ptr + 0 + offsets_32)
            ql_1 = tl.load(block_ptr + 32 + offsets_32)
            ql_2 = tl.load(block_ptr + 64 + offsets_32)
            ql_3 = tl.load(block_ptr + 96 + offsets_32)
            qh_0 = tl.load(block_ptr + 128 + offsets_32)
            qh_1 = tl.load(block_ptr + 160 + offsets_32)

            # Pre-load all 16 int8 scales
            s0  = tl.load(scales_ptr + 0).to(tl.int8, bitcast=True).to(DTYPE)
            s1  = tl.load(scales_ptr + 1).to(tl.int8, bitcast=True).to(DTYPE)
            s2  = tl.load(scales_ptr + 2).to(tl.int8, bitcast=True).to(DTYPE)
            s3  = tl.load(scales_ptr + 3).to(tl.int8, bitcast=True).to(DTYPE)
            s4  = tl.load(scales_ptr + 4).to(tl.int8, bitcast=True).to(DTYPE)
            s5  = tl.load(scales_ptr + 5).to(tl.int8, bitcast=True).to(DTYPE)
            s6  = tl.load(scales_ptr + 6).to(tl.int8, bitcast=True).to(DTYPE)
            s7  = tl.load(scales_ptr + 7).to(tl.int8, bitcast=True).to(DTYPE)
            s8  = tl.load(scales_ptr + 8).to(tl.int8, bitcast=True).to(DTYPE)
            s9  = tl.load(scales_ptr + 9).to(tl.int8, bitcast=True).to(DTYPE)
            s10 = tl.load(scales_ptr + 10).to(tl.int8, bitcast=True).to(DTYPE)
            s11 = tl.load(scales_ptr + 11).to(tl.int8, bitcast=True).to(DTYPE)
            s12 = tl.load(scales_ptr + 12).to(tl.int8, bitcast=True).to(DTYPE)
            s13 = tl.load(scales_ptr + 13).to(tl.int8, bitcast=True).to(DTYPE)
            s14 = tl.load(scales_ptr + 14).to(tl.int8, bitcast=True).to(DTYPE)
            s15 = tl.load(scales_ptr + 15).to(tl.int8, bitcast=True).to(DTYPE)

            # Cast qh once to int8 for bit extraction
            qh_0_i8 = qh_0.to(tl.int8, bitcast=True)
            qh_1_i8 = qh_1.to(tl.int8, bitcast=True)

            # ---- Chunk 0: ql_0 low, qh_0 bits[1:0] ----
            q_val = ((ql_0 & 0x0F).to(tl.int8, bitcast=True) | ((qh_0_i8 & 0x03) << 4)) - 32
            w = (d_scale * tl.where(mask_16, s0, s1)) * q_val.to(DTYPE)
            acc += w.to(tl.float32) * tl.load(input_ptr + input_base + offsets_32).to(tl.float32)

            # ---- Chunk 1: ql_1 low, qh_0 bits[3:2] ----
            q_val = ((ql_1 & 0x0F).to(tl.int8, bitcast=True) | (((qh_0_i8 >> 2) & 0x03) << 4)) - 32
            w = (d_scale * tl.where(mask_16, s2, s3)) * q_val.to(DTYPE)
            acc += w.to(tl.float32) * tl.load(input_ptr + input_base + 32 + offsets_32).to(tl.float32)

            # ---- Chunk 2: ql_0 high, qh_0 bits[5:4] ----
            q_val = (((ql_0 >> 4) & 0x0F).to(tl.int8, bitcast=True) | (((qh_0_i8 >> 4) & 0x03) << 4)) - 32
            w = (d_scale * tl.where(mask_16, s4, s5)) * q_val.to(DTYPE)
            acc += w.to(tl.float32) * tl.load(input_ptr + input_base + 64 + offsets_32).to(tl.float32)

            # ---- Chunk 3: ql_1 high, qh_0 bits[7:6] ----
            q_val = (((ql_1 >> 4) & 0x0F).to(tl.int8, bitcast=True) | (((qh_0_i8 >> 6) & 0x03) << 4)) - 32
            w = (d_scale * tl.where(mask_16, s6, s7)) * q_val.to(DTYPE)
            acc += w.to(tl.float32) * tl.load(input_ptr + input_base + 96 + offsets_32).to(tl.float32)

            # ---- Chunk 4: ql_2 low, qh_1 bits[1:0] ----
            q_val = ((ql_2 & 0x0F).to(tl.int8, bitcast=True) | ((qh_1_i8 & 0x03) << 4)) - 32
            w = (d_scale * tl.where(mask_16, s8, s9)) * q_val.to(DTYPE)
            acc += w.to(tl.float32) * tl.load(input_ptr + input_base + 128 + offsets_32).to(tl.float32)

            # ---- Chunk 5: ql_3 low, qh_1 bits[3:2] ----
            q_val = ((ql_3 & 0x0F).to(tl.int8, bitcast=True) | (((qh_1_i8 >> 2) & 0x03) << 4)) - 32
            w = (d_scale * tl.where(mask_16, s10, s11)) * q_val.to(DTYPE)
            acc += w.to(tl.float32) * tl.load(input_ptr + input_base + 160 + offsets_32).to(tl.float32)

            # ---- Chunk 6: ql_2 high, qh_1 bits[5:4] ----
            q_val = (((ql_2 >> 4) & 0x0F).to(tl.int8, bitcast=True) | (((qh_1_i8 >> 4) & 0x03) << 4)) - 32
            w = (d_scale * tl.where(mask_16, s12, s13)) * q_val.to(DTYPE)
            acc += w.to(tl.float32) * tl.load(input_ptr + input_base + 192 + offsets_32).to(tl.float32)

            # ---- Chunk 7: ql_3 high, qh_1 bits[7:6] ----
            q_val = (((ql_3 >> 4) & 0x0F).to(tl.int8, bitcast=True) | (((qh_1_i8 >> 6) & 0x03) << 4)) - 32
            w = (d_scale * tl.where(mask_16, s14, s15)) * q_val.to(DTYPE)
            acc += w.to(tl.float32) * tl.load(input_ptr + input_base + 224 + offsets_32).to(tl.float32)

        result = tl.sum(acc, axis=0).to(DTYPE)
        tl.store(output_ptr + row_idx, result)

    # ================================================================
    #  Standalone Q3_K dequantization kernel (unchanged from original)
    # ================================================================

    @triton.autotune(
        configs=[
            triton.Config({"N_BLOCKS_PER_PROG": n}, num_warps=w)
            for n in [1, 2, 4] for w in [2, 4, 8]
        ],
        key=["n_total_blocks"],
    )
    @triton.jit
    def _dequantize_q3_k_kernel(
        q_ptr, out_ptr, n_total_blocks,
        DTYPE: tl.constexpr,
        N_BLOCKS_PER_PROG: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        start_block_idx = pid * N_BLOCKS_PER_PROG
        n_blocks = n_total_blocks - start_block_idx

        if n_blocks > 0:
            for i in tl.static_range(N_BLOCKS_PER_PROG):
                if i < n_blocks:
                    block_offset = start_block_idx + i
                    block_start_ptr = q_ptr + block_offset * 110
                    output_base = out_ptr + block_offset * 256

                    offsets_16 = tl.arange(0, 16)
                    hmask_ptr = block_start_ptr
                    qs_ptr = block_start_ptr + 32
                    scales_ptr = block_start_ptr + 96
                    d_ptr = block_start_ptr + 108

                    d_super_scale = tl.load(
                        d_ptr.to(tl.pointer_type(tl.float16))
                    ).to(DTYPE)

                    for chunk_idx in tl.static_range(16):
                        lscale_byte_index = chunk_idx % 8
                        lscale_shift = (chunk_idx // 8) * 4
                        lscale_byte = tl.load(scales_ptr + lscale_byte_index)
                        lscale_nibble = (lscale_byte >> lscale_shift) & 0x0F

                        hscale_byte_index = chunk_idx % 4
                        hscale_shift_index = chunk_idx // 4
                        hscale_byte = tl.load(scales_ptr + 8 + hscale_byte_index)
                        hscale_2bit = (hscale_byte >> (hscale_shift_index * 2)) & 0x03

                        scale_6bit = lscale_nibble | (hscale_2bit << 4)
                        final_scale = d_super_scale * (scale_6bit.to(tl.int8) - 32).to(DTYPE)

                        flat_indices = chunk_idx * 16 + offsets_16

                        ql_source_row = flat_indices // 32
                        ql_source_col = flat_indices % 32
                        ql_segment = ql_source_row // 4
                        ql_shift_group = ql_source_row % 4

                        ql_byte = tl.load(qs_ptr + ql_segment * 32 + ql_source_col)
                        ql_vec = ((ql_byte >> (ql_shift_group * 2)) & 3).to(tl.int8)

                        qh_source_row = flat_indices // 32
                        qh_source_col = flat_indices % 32
                        qh_byte = tl.load(hmask_ptr + qh_source_col)
                        qh_vec = (((qh_byte >> qh_source_row) & 1) ^ 1).to(tl.int8)

                        q_vec = ql_vec - (qh_vec << 2)
                        dequant_16 = final_scale * q_vec.to(DTYPE)
                        tl.store(output_base + chunk_idx * 16 + offsets_16, dequant_16)

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
        """Fused Q6_K dequant + GEMV for batch=1 decode (v2)."""
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

    def fused_dequant_gemv_q3k(
        input_vec: torch.Tensor,
        q_weight_data: torch.Tensor,
        D_in: int,
        D_out: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Fused Q3_K dequant + GEMV for batch=1 decode (v2)."""
        assert D_in % 256 == 0, f"D_in must be multiple of 256, got {D_in}"
        input_flat = input_vec.view(-1).contiguous()
        output = torch.empty(D_out, device=input_vec.device, dtype=dtype)
        TRITON_DTYPE = tl.float16 if dtype == torch.float16 else tl.bfloat16

        _fused_dequant_gemv_q3k_kernel[(D_out,)](
            input_flat,
            q_weight_data,
            output,
            D_in=D_in,
            D_out=D_out,
            BLOCK_SIZE_Q3K=110,
            VALS_PER_BLOCK=256,
            DTYPE=TRITON_DTYPE,
        )
        return output

    def triton_dequant_q3k(
        data: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Dequantize Q3_K raw bytes to float tensor using Triton."""
        raw = data.view(torch.uint8)
        n_blocks = raw.numel() // 110
        out = torch.empty(n_blocks * 256, dtype=dtype, device=data.device)
        grid = lambda meta: (triton.cdiv(n_blocks, meta["N_BLOCKS_PER_PROG"]),)
        _dequantize_q3_k_kernel[grid](
            raw, out, n_blocks, DTYPE=tl.float32,
        )
        return out.reshape(shape)

else:
    triton_dequant_q6k = None
    fused_dequant_gemv_q6k = None
    fused_dequant_gemv_q3k = None
    triton_dequant_q3k = None
