/*
 * GGUF Fused Dequant + GEMV CUDA Kernels for fish-speech (v4: multi-warp dp4a)
 *
 * Faithfully reproduces llama.cpp's mmvq algorithm with multi-warp optimization:
 *   1. Input vector quantized to Q8_1 (int8 + scale)
 *   2. GEMV via dp4a with 4 warps per output row (shared-memory reduction)
 *
 * Key improvements over v3 (single-warp):
 *   - 4 warps per row: each warp processes nblocks/4 Q-blocks → ~3x fewer
 *     inner-loop iterations per warp for typical D_in (2560–10240)
 *   - __launch_bounds__(128, 1): compiler uses more registers, fewer spills
 *   - Matches llama.cpp GENERIC (NVIDIA) configuration exactly
 *
 * Target: NVIDIA Turing SM75+ (RTX 2070)
 *
 * Block layouts (identical to llama.cpp / GGML standard):
 *   Q6_K (210 bytes per 256 values):
 *     ql[128]    offset 0   : low 4-bit quantized values
 *     qh[64]     offset 128 : high 2-bit quantized values
 *     scales[16] offset 192 : int8 per-16-value scales
 *     d          offset 208 : float16 super-block scale
 *
 *   Q3_K (110 bytes per 256 values):
 *     hmask[32]  offset 0   : high-bit mask
 *     qs[64]     offset 32  : 2-bit quantized values
 *     scales[12] offset 96  : packed 6-bit scales
 *     d          offset 108 : float16 super-block scale
 *
 *   Q8_1 (36 bytes per 32 values):
 *     ds         offset 0   : half2(d=scale, s=sum)
 *     qs[32]     offset 4   : int8 quantized values
 *
 * References:
 *   - llama.cpp ggml-cuda/mmvq.cu  (mul_mat_vec_q kernel, calc_nwarps)
 *   - llama.cpp ggml-cuda/vecdotq.cuh  (vec_dot_q6_K_q8_1_impl_mmvq)
 *   - llama.cpp ggml-cuda/quantize.cu
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "gguf_gemv.h"

/* ================================================================
 *  Constants
 * ================================================================ */
#define QK_K            256
#define QK8_1           32
#define Q6K_BLOCK_SIZE  210
#define Q3K_BLOCK_SIZE  110
#define WARP_SIZE       32

/*
 * Multi-warp configuration (matching llama.cpp GENERIC for ncols_dst=1):
 *   nwarps = 4  →  blockDim = (32, 4) = 128 threads
 *   1 row per CUDA block (normal path)
 *
 * vs. v3: 8 warps, each on a separate row, no shared-memory reduction.
 * The multi-warp approach reduces inner-loop iterations per warp by ~4x,
 * improving per-row latency at the cost of a small shared-memory sync.
 */
#define NWARPS          4
#define MW_THREADS      (WARP_SIZE * NWARPS)   /* 128 */

/* ================================================================
 *  Q8_1 block structure (matches llama.cpp block_q8_1)
 * ================================================================ */
struct __align__(4) block_q8_1 {
    __half2 ds;       /* x=d (scale), y=s (sum); only d used for MMVQ */
    int8_t  qs[32];   /* quantized values */
};

/* ================================================================
 *  Warp-level reductions
 * ================================================================ */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

/* ================================================================
 *  Integer read helpers (matching llama.cpp vecdotq.cuh)
 *  b2 = 2-byte aligned, b4 = 4-byte aligned
 * ================================================================ */
__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    return __ldg(x16 + 2*i32) | (__ldg(x16 + 2*i32 + 1) << 16);
}

__device__ __forceinline__ int get_int_b4(const void* x, int i32) {
    return __ldg((const int*)x + i32);
}


/* ================================================================
 *  Q8_1 Quantization Kernel  (unchanged from v3)
 *
 *  Each warp (32 threads) quantizes one block of 32 float values:
 *    1. Find max absolute value (warp reduce)
 *    2. Compute scale d = max / 127
 *    3. Quantize each value to int8
 *
 *  Grid: ceil(n/256) blocks × 256 threads (8 warps/block)
 * ================================================================ */
__global__ void __launch_bounds__(256)
quantize_q8_1_kernel(
    const float* __restrict__ x,
    block_q8_1*  __restrict__ y,
    int n)
{
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int base = warp_global * QK8_1;

    if (base >= n) return;

    float val = (base + lane < n) ? __ldg(x + base + lane) : 0.0f;
    float amax = fabsf(val);
    amax = warp_reduce_max(amax);

    float d = amax / 127.0f;
    int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(val / d);

    y[warp_global].qs[lane] = q;
    if (lane == 0) {
        y[warp_global].ds = __halves2half2(__float2half(d), __float2half(0.0f));
    }
}


/* ================================================================
 *  Q6_K GEMV — Multi-Warp (4 warps per output row)
 *
 *  Matching llama.cpp GENERIC: nwarps=4, blockDim=(32,4)=128 threads.
 *
 *  Each warp processes blocks with stride NWARPS:
 *    warp 0 → blocks 0, 4, 8, ...
 *    warp 1 → blocks 1, 5, 9, ...
 *    warp 2 → blocks 2, 6, 10, ...
 *    warp 3 → blocks 3, 7, 11, ...
 *
 *  After the loop, partial sums from warps 1-3 are written to shared
 *  memory.  Warp 0 accumulates everything and does warp_reduce_sum.
 *
 *  Thread mapping (32 lanes per warp, same as v3):
 *    Each lane reads 4 bytes from ql (get_int_b2) and qh,
 *    then performs 2 dp4a calls (QR6_K=2) per Q6_K block.
 *    32 lanes × 2 dp4a × 4 values/dp4a = 256 values/block. ✓
 *
 *  Index derivation (from llama.cpp, QI6_K=32, QR6_K=2):
 *    bq8_offset   = 4*(lane/16) + (lane%16)/8
 *    scale_offset = 8*(lane/16) + (lane%16)/4
 *    vh_shift     = 2*((lane%16)/8)
 *    qh_index     = 8*(lane/16) + lane%8
 * ================================================================ */
__global__ void __launch_bounds__(MW_THREADS, 1)
gemv_q6k_q8_1_kernel(
    const block_q8_1* __restrict__ q8,
    const uint8_t*    __restrict__ qw,
    float*            __restrict__ y,
    int D_in, int D_out)
{
    const int lane = threadIdx.x;          /* 0..31 */
    const int warp = threadIdx.y;          /* 0..NWARPS-1 */
    const int row  = blockIdx.x;

    if (row >= D_out) return;

    const int nblocks = D_in / QK_K;
    const uint8_t* row_data = qw + (size_t)row * nblocks * Q6K_BLOCK_SIZE;

    /* Per-lane constants (invariant across blocks) */
    const int bq8_off  = 4 * (lane / 16) + (lane % 16) / 8;
    const int sc_off   = 8 * (lane / 16) + (lane % 16) / 4;
    const int vh_shift = 2 * ((lane % 16) / 8);
    const int qh_idx   = 8 * (lane / 16) + lane % 8;

    float acc = 0.0f;

    for (int b = warp; b < nblocks; b += NWARPS) {
        const uint8_t* bp = row_data + b * Q6K_BLOCK_SIZE;
        const block_q8_1* q8_b = q8 + b * 8; /* 8 Q8_1 blocks per Q6_K block */

        /* Super-block scale */
        float d = __half2float(__ldg((const __half*)(bp + 208)));

        /* Read weight data: 4 bytes from ql and qh */
        int vl = get_int_b2(bp, lane);                        /* ql[lane] */
        int vh = get_int_b2(bp + 128, qh_idx) >> vh_shift;    /* qh[qh_idx] */

        const int8_t* scales = (const int8_t*)(bp + 192);

        float sumf = 0.0f;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {  /* QR6_K = 2 */
            int sc = (int)__ldg(scales + sc_off + 4*i);

            /* Decode 4 weight values as packed int8x4 */
            int vil = (vl >> (4*i)) & 0x0F0F0F0F;
            int vih = ((vh >> (4*i)) << 4) & 0x30303030;
            int vi  = __vsubss4(vil | vih, 0x20202020); /* subtract 32 */

            /* Read 4 Q8_1 input values + scale */
            int   u  = get_int_b4(q8_b[bq8_off + 2*i].qs, lane & 7);
            float d8 = __low2float(__ldg(&q8_b[bq8_off + 2*i].ds));

            sumf += d8 * (float)(__dp4a(vi, u, 0) * sc);
        }

        acc += d * sumf;
    }

    /* ---- Inter-warp reduction via shared memory ---- */
    __shared__ float smem[NWARPS - 1][WARP_SIZE];

    if (warp > 0) {
        smem[warp - 1][lane] = acc;
    }
    __syncthreads();
    if (warp > 0) return;

    /* Warp 0: accumulate partial sums from all warps */
    #pragma unroll
    for (int w = 0; w < NWARPS - 1; ++w) {
        acc += smem[w][lane];
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[row] = acc;
}


/* ================================================================
 *  Q3_K GEMV — Multi-Warp (4 warps per output row)
 *
 *  Matching llama.cpp GENERIC: nwarps=4, blockDim=(32,4)=128 threads.
 *
 *  Thread mapping: QI3_K=16, so each half-warp (16 threads) processes
 *  one Q3_K block.  Within each warp, 2 half-warps handle 2 blocks.
 *  With NWARPS=4, stride = 2*NWARPS = 8 blocks per iteration.
 *
 *  Index derivation (QI3_K=16, QR3_K=4, QI8_1=8):
 *    iqs          = lane % 16
 *    block_offset = lane / 16  (0 or 1)
 *    bq8_offset   = 4*(iqs/8)
 *    scale_base   = iqs - (iqs%8) + (iqs%8)/4
 * ================================================================ */
__global__ void __launch_bounds__(MW_THREADS, 1)
gemv_q3k_q8_1_kernel(
    const block_q8_1* __restrict__ q8,
    const uint8_t*    __restrict__ qw,
    float*            __restrict__ y,
    int D_in, int D_out)
{
    const int lane = threadIdx.x;          /* 0..31 */
    const int warp = threadIdx.y;          /* 0..NWARPS-1 */
    const int row  = blockIdx.x;

    if (row >= D_out) return;

    const int nblocks = D_in / QK_K;
    const uint8_t* row_data = qw + (size_t)row * nblocks * Q3K_BLOCK_SIZE;

    /* Q3_K: 16 threads per block → 2 blocks per warp */
    const int iqs       = lane & 15;       /* 0..15 */
    const int block_ofs = lane >> 4;        /* 0 or 1 */

    /* Per-thread constants */
    const int bq8_off = 4 * (iqs / 8);     /* 0 or 4 */
    const int sc_base = iqs - (iqs & 7) + ((iqs & 7) >> 2);
    /* iqs 0-3:0, 4-7:1, 8-11:8, 12-15:9 */

    float acc = 0.0f;

    /* 2 blocks per warp × NWARPS warps = 2*NWARPS blocks per iteration */
    const int start_block = warp * 2 + block_ofs;

    for (int b = start_block; b < nblocks; b += NWARPS * 2) {
        const uint8_t* bp = row_data + b * Q3K_BLOCK_SIZE;
        const block_q8_1* q8_b = q8 + b * 8; /* 8 Q8_1 blocks per Q3_K block */

        /* Super-block scale */
        float d3 = __half2float(__ldg((const __half*)(bp + 108)));

        /* Read weight data */
        int vl = get_int_b2(bp + 32, iqs);               /* qs[iqs] */
        int vh = ~get_int_b2(bp, iqs & 7) >> bq8_off;    /* ~hmask[iqs%8] >> offset */

        const uint8_t* sc_ptr = bp + 96;

        float sumf = 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {  /* QR3_K = 4 */
            /* Read Q8_1 input */
            int   u  = get_int_b4(q8_b[bq8_off + i].qs, iqs & 7);
            float d8 = __low2float(__ldg(&q8_b[bq8_off + i].ds));

            /* Decode 6-bit scale from packed 12 bytes */
            const int isc = sc_base + 2*i;

            const int isc_low      = isc & 7;
            const int sc_shift_low = 4 * (isc >> 3);
            const int sc_low = (__ldg(sc_ptr + isc_low) >> sc_shift_low) & 0xF;

            const int isc_high      = isc & 3;
            const int sc_shift_high = 2 * (isc >> 2);
            const int sc_high = ((__ldg(sc_ptr + 8 + isc_high) >> sc_shift_high) & 3) << 4;

            const int sc = (sc_low | sc_high) - 32;

            /* Decode 4 weight values as packed int8x4 */
            int vil = (vl >> (2*i)) & 0x03030303;
            int vih = ((vh >> i) << 2) & 0x04040404;
            int vi  = __vsubss4(vil, vih);

            sumf += d8 * (float)(__dp4a(vi, u, 0) * sc);
        }

        acc += d3 * sumf;
    }

    /* ---- Inter-warp reduction via shared memory ---- */
    __shared__ float smem[NWARPS - 1][WARP_SIZE];

    if (warp > 0) {
        smem[warp - 1][lane] = acc;
    }
    __syncthreads();
    if (warp > 0) return;

    /* Warp 0: accumulate partial sums from all warps */
    #pragma unroll
    for (int w = 0; w < NWARPS - 1; ++w) {
        acc += smem[w][lane];
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[row] = acc;
}


/* ================================================================
 *  Host wrapper functions
 * ================================================================ */
extern "C" {

void quantize_q8_1_cuda(
    const float* input, void* output, int n, cudaStream_t stream)
{
    const int n_q8_blocks = (n + QK8_1 - 1) / QK8_1;
    const int threads = 256;  /* 8 warps per CUDA block */
    const int blocks  = (n_q8_blocks * WARP_SIZE + threads - 1) / threads;
    quantize_q8_1_kernel<<<blocks, threads, 0, stream>>>(
        input, (block_q8_1*)output, n);
}

void gemv_q6k_q8_1(
    const void* q8_input, const uint8_t* q_weight,
    float* output, int D_in, int D_out, cudaStream_t stream)
{
    /* 1 row per CUDA block, NWARPS warps per block */
    const dim3 grid(D_out);
    const dim3 block(WARP_SIZE, NWARPS);
    gemv_q6k_q8_1_kernel<<<grid, block, 0, stream>>>(
        (const block_q8_1*)q8_input, q_weight, output, D_in, D_out);
}

void gemv_q3k_q8_1(
    const void* q8_input, const uint8_t* q_weight,
    float* output, int D_in, int D_out, cudaStream_t stream)
{
    const dim3 grid(D_out);
    const dim3 block(WARP_SIZE, NWARPS);
    gemv_q3k_q8_1_kernel<<<grid, block, 0, stream>>>(
        (const block_q8_1*)q8_input, q_weight, output, D_in, D_out);
}

/* Backward-compatible wrappers (not used by current bindings.cpp,
   but kept for potential direct C usage) */
void fused_gemv_q6k(
    const float* input, const uint8_t* q_weight,
    float* output, int D_in, int D_out, cudaStream_t stream)
{
    const int n_q8_blocks = D_in / QK8_1;
    block_q8_1* q8_buf;
    cudaMallocAsync(&q8_buf, n_q8_blocks * sizeof(block_q8_1), stream);
    quantize_q8_1_cuda(input, q8_buf, D_in, stream);
    gemv_q6k_q8_1(q8_buf, q_weight, output, D_in, D_out, stream);
    cudaFreeAsync(q8_buf, stream);
}

void fused_gemv_q3k(
    const float* input, const uint8_t* q_weight,
    float* output, int D_in, int D_out, cudaStream_t stream)
{
    const int n_q8_blocks = D_in / QK8_1;
    block_q8_1* q8_buf;
    cudaMallocAsync(&q8_buf, n_q8_blocks * sizeof(block_q8_1), stream);
    quantize_q8_1_cuda(input, q8_buf, D_in, stream);
    gemv_q3k_q8_1(q8_buf, q_weight, output, D_in, D_out, stream);
    cudaFreeAsync(q8_buf, stream);
}

} /* extern "C" */
