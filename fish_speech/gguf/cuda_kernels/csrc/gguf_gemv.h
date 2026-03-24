/*
 * GGUF Fused Dequant + GEMV CUDA Kernels — public API (v3: dp4a)
 *
 * Two-phase approach matching llama.cpp:
 *   1. quantize_q8_1_cuda: quantize float32 input to Q8_1 (int8)
 *   2. gemv_q6k_q8_1 / gemv_q3k_q8_1: dp4a-based GEMV
 *
 * The fused_gemv_* wrappers handle Q8_1 allocation internally
 * for backward compatibility with existing Python bindings.
 */

#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Phase 1: Quantize float32 input vector to Q8_1 format */
void quantize_q8_1_cuda(
    const float* input, void* q8_output, int n, cudaStream_t stream);

/* Phase 2: dp4a GEMV with Q8_1 input */
void gemv_q6k_q8_1(
    const void* q8_input, const uint8_t* q_weight,
    float* output, int D_in, int D_out, cudaStream_t stream);

void gemv_q3k_q8_1(
    const void* q8_input, const uint8_t* q_weight,
    float* output, int D_in, int D_out, cudaStream_t stream);

/* Backward-compatible: float input, handles Q8_1 internally */
void fused_gemv_q6k(
    const float* input, const uint8_t* q_weight,
    float* output, int D_in, int D_out, cudaStream_t stream);

void fused_gemv_q3k(
    const float* input, const uint8_t* q_weight,
    float* output, int D_in, int D_out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
