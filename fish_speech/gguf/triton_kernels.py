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
    #  Q3_K block layout (110 bytes per 256 values)
    #  hmask[32]  bytes 0..31    high-bit mask
    #  qs[64]     bytes 32..95   2-bit quantized values
    #  scales[12] bytes 96..107  packed 6-bit scales (16 values)
    #  d[2]       bytes 108..109 super-block scale (float16)
    # ================================================================

    @triton.jit
    def _decode_q3k_scales(block_ptr):
        """Decode 16 x 6-bit scales from 12 packed bytes.

        Layout:
          ls[0..7]:   bytes 96..103  — low nibbles
          hs[0..3]:   bytes 104..107 — high 2-bit pairs

        Scale[i] = (ls_low[i] & 0x0F) | ((hs_bits[i] & 0x03) << 4) - 32
        """
        # Load low-scale bytes (8 bytes → 16 nibbles)
        ls0 = tl.load(block_ptr + 96).to(tl.uint8)
        ls1 = tl.load(block_ptr + 97).to(tl.uint8)
        ls2 = tl.load(block_ptr + 98).to(tl.uint8)
        ls3 = tl.load(block_ptr + 99).to(tl.uint8)
        ls4 = tl.load(block_ptr + 100).to(tl.uint8)
        ls5 = tl.load(block_ptr + 101).to(tl.uint8)
        ls6 = tl.load(block_ptr + 102).to(tl.uint8)
        ls7 = tl.load(block_ptr + 103).to(tl.uint8)

        # Load high-scale bytes (4 bytes → 16 x 2-bit)
        hs0 = tl.load(block_ptr + 104).to(tl.uint8)
        hs1 = tl.load(block_ptr + 105).to(tl.uint8)
        hs2 = tl.load(block_ptr + 106).to(tl.uint8)
        hs3 = tl.load(block_ptr + 107).to(tl.uint8)

        # Low nibbles: scale[0]=ls0&0xF, scale[8]=ls0>>4, etc.
        s0  = ls0 & 0x0F;  s8  = (ls0 >> 4) & 0x0F
        s1  = ls1 & 0x0F;  s9  = (ls1 >> 4) & 0x0F
        s2  = ls2 & 0x0F;  s10 = (ls2 >> 4) & 0x0F
        s3  = ls3 & 0x0F;  s11 = (ls3 >> 4) & 0x0F
        s4  = ls4 & 0x0F;  s12 = (ls4 >> 4) & 0x0F
        s5  = ls5 & 0x0F;  s13 = (ls5 >> 4) & 0x0F
        s6  = ls6 & 0x0F;  s14 = (ls6 >> 4) & 0x0F
        s7  = ls7 & 0x0F;  s15 = (ls7 >> 4) & 0x0F

        # High 2-bit pairs
        h0  = hs0 & 0x03;        h4  = (hs0 >> 2) & 0x03
        h8  = (hs0 >> 4) & 0x03; h12 = (hs0 >> 6) & 0x03
        h1  = hs1 & 0x03;        h5  = (hs1 >> 2) & 0x03
        h9  = (hs1 >> 4) & 0x03; h13 = (hs1 >> 6) & 0x03
        h2  = hs2 & 0x03;        h6  = (hs2 >> 2) & 0x03
        h10 = (hs2 >> 4) & 0x03; h14 = (hs2 >> 6) & 0x03
        h3  = hs3 & 0x03;        h7  = (hs3 >> 2) & 0x03
        h11 = (hs3 >> 4) & 0x03; h15 = (hs3 >> 6) & 0x03

        # Combine: 6-bit scale = low4 | (high2 << 4), then subtract 32
        # Return as tuple of 16 int8-range values
        return (
            (s0  | (h0  << 4)).to(tl.int8) - 32,
            (s1  | (h1  << 4)).to(tl.int8) - 32,
            (s2  | (h2  << 4)).to(tl.int8) - 32,
            (s3  | (h3  << 4)).to(tl.int8) - 32,
            (s4  | (h4  << 4)).to(tl.int8) - 32,
            (s5  | (h5  << 4)).to(tl.int8) - 32,
            (s6  | (h6  << 4)).to(tl.int8) - 32,
            (s7  | (h7  << 4)).to(tl.int8) - 32,
            (s8  | (h8  << 4)).to(tl.int8) - 32,
            (s9  | (h9  << 4)).to(tl.int8) - 32,
            (s10 | (h10 << 4)).to(tl.int8) - 32,
            (s11 | (h11 << 4)).to(tl.int8) - 32,
            (s12 | (h12 << 4)).to(tl.int8) - 32,
            (s13 | (h13 << 4)).to(tl.int8) - 32,
            (s14 | (h14 << 4)).to(tl.int8) - 32,
            (s15 | (h15 << 4)).to(tl.int8) - 32,
        )

    @triton.jit
    def _decode_q3k_chunk(
        block_ptr, chunk_idx: tl.constexpr, offsets_16, d_scale, sc, DTYPE: tl.constexpr,
    ):
        """Decode 16 values from one of 16 sub-blocks within a Q3_K block.

        Q3_K packing for qs (2-bit, 64 bytes for 256 values):
          qs[i] contains 4 x 2-bit values via shifts 0,2,4,6
          Byte index = chunk_idx // 4 * 32 + offsets_16  (for first 16 of 32)
                     or chunk_idx // 4 * 32 + offsets_16 + 16 (for second 16)

        hmask (1-bit, 32 bytes for 256 values):
          Bit position = chunk_idx % 8
          Byte index = chunk_idx // 8 * 32 + value_offset_within_32

        q_val = (ql & 3) - 4 * (1 - hbit)
        """
        # Which of the 4 shift groups (0,2,4,6) and which 32-byte row
        shift = (chunk_idx % 4) * 2
        qs_row = chunk_idx // 4  # 0..3

        # For chunks 0-7: values map to first/second half within a 32-byte row
        is_upper = (chunk_idx % 2)  # 0 or 1 (first/second 16 of 32)

        # qs byte offset
        qs_byte_off = 32 + qs_row * 32 + is_upper * 16 + offsets_16

        # hmask byte offset: bit (chunk_idx % 8) of byte at hmask_row * 32 + offset
        hmask_row = chunk_idx // 8  # 0..1
        hmask_bit = chunk_idx % 8
        hmask_byte_off = hmask_row * 32 + is_upper * 16 + offsets_16

        # Load and decode
        qs_bytes = tl.load(block_ptr + qs_byte_off)
        ql = (qs_bytes >> shift) & 0x03

        hmask_bytes = tl.load(block_ptr + hmask_byte_off)
        hbit = (hmask_bytes >> hmask_bit) & 0x01
        # hbit=1 means subtract 0 (no offset), hbit=0 means subtract 4
        hbit_inv = hbit ^ 0x01  # invert: 1->0, 0->1

        q_val = ql.to(tl.int8) - (hbit_inv << 2).to(tl.int8)

        return (d_scale * sc.to(DTYPE)) * q_val.to(DTYPE)

    # ================================================================
    #  Q3_K fused dequant + GEMV (batch=1 decode)
    #  Optimized: 32-wide vectors, pre-loaded hmask/qs, 8 super-chunks
    #
    #  Key optimizations over the naive 16-wide / 16-chunk version:
    #   1. 32-element accumulator (matching Q6_K parallelism)
    #   2. 8 super-chunks of 32 values instead of 16 chunks of 16
    #   3. Pre-load hmask (32 bytes) once per block (was loaded 8x)
    #   4. Pre-load qs in two 32-byte halves, reuse with shifts
    #   5. tl.where for dual-scale application (same pattern as Q6_K)
    #
    #  Q3_K data mapping for super-chunk sc (0..7):
    #   sc 0-3: qs bytes 0..31  with shift sc*2, hmask bit sc
    #   sc 4-7: qs bytes 32..63 with shift (sc-4)*2, hmask bit sc
    #   Each super-chunk has 2 scales: scale[sc*2] and scale[sc*2+1]
    # ================================================================

    @triton.jit
    def _q3k_decode_scale(scales_ptr, idx: tl.constexpr, DTYPE: tl.constexpr):
        """Decode one 6-bit scale from packed 12 scale bytes.

        Low nibble:  scale_bytes[idx % 8] >> ((idx // 8) * 4) & 0xF
        High 2-bit:  scale_bytes[8 + idx % 4] >> ((idx // 4) * 2) & 0x3
        Result:      (low | (high << 4)) - 32
        """
        ls = (tl.load(scales_ptr + (idx % 8)) >> ((idx // 8) * 4)) & 0x0F
        hs = (tl.load(scales_ptr + 8 + (idx % 4)) >> ((idx // 4) * 2)) & 0x03
        return (ls | (hs << 4)).to(tl.int8) - 32

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

            # Super-block scale (float16 at byte offset 108)
            d_scale = tl.load(
                (block_ptr + 108).to(tl.pointer_type(tl.float16))
            ).to(DTYPE)

            # Pre-load hmask[0..31] — reused across all 8 super-chunks
            hmask = tl.load(block_ptr + offsets_32)
            # Pre-load qs in two 32-byte halves
            qs_lo = tl.load(block_ptr + 32 + offsets_32)   # qs[0..31]
            qs_hi = tl.load(block_ptr + 64 + offsets_32)   # qs[32..63]

            scales_ptr = block_ptr + 96

            # ---- Super-chunks 0-3: qs_lo with shifts 0, 2, 4, 6 ----
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

            # ---- Super-chunks 4-7: qs_hi with shifts 0, 2, 4, 6 ----
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
    #  Python wrappers for Q3_K
    # ================================================================

    def fused_dequant_gemv_q3k(
        input_vec: torch.Tensor,
        q_weight_data: torch.Tensor,
        D_in: int,
        D_out: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
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

    # ================================================================
    #  Standalone Q3_K dequantization kernel (blepping方式)
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
