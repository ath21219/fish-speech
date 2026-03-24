#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

// Forward declarations (implemented in gguf_gemv.cu)
extern "C" {
void quantize_q8_1_cuda(
    const float* input, void* q8_output, int n, cudaStream_t stream);
void gemv_q6k_q8_1(
    const void* q8_input, const uint8_t* q_weight,
    float* output, int D_in, int D_out, cudaStream_t stream);
void gemv_q3k_q8_1(
    const void* q8_input, const uint8_t* q_weight,
    float* output, int D_in, int D_out, cudaStream_t stream);
}

// Q8_1 block size: half2 (4 bytes) + int8[32] = 36 bytes per 32 values
static constexpr int64_t Q8_1_BLOCK_BYTES = 36;
static constexpr int64_t QK8_1 = 32;


/* ================================================================
 *  Persistent Q8_1 buffer pool
 *
 *  Avoids per-call torch::empty() overhead on the hot decode path.
 *  The buffer grows monotonically to fit the largest D_in seen.
 *  Since all calls happen on the same CUDA stream (sequentially),
 *  a single buffer is safe to reuse without synchronization.
 * ================================================================ */
static torch::Tensor g_q8_buf;
static int64_t g_q8_buf_bytes = 0;

static uint8_t* get_q8_buffer(int64_t needed_bytes, const torch::Device& device) {
    if (g_q8_buf_bytes < needed_bytes || !g_q8_buf.defined() || g_q8_buf.device() != device) {
        // Grow with 25% headroom to avoid frequent re-allocations
        int64_t alloc_bytes = needed_bytes + needed_bytes / 4;
        g_q8_buf = torch::empty(
            {alloc_bytes},
            torch::dtype(torch::kUInt8).device(device));
        g_q8_buf_bytes = alloc_bytes;
    }
    return g_q8_buf.data_ptr<uint8_t>();
}


torch::Tensor fused_gemv_q6k_torch(
    torch::Tensor input,
    torch::Tensor q_weight,
    int64_t D_in,
    int64_t D_out
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(q_weight.is_cuda(), "q_weight must be on CUDA");
    TORCH_CHECK(D_in % 256 == 0, "D_in must be multiple of 256");

    auto input_f32 = input.to(torch::kFloat32).contiguous();
    auto output = torch::empty({D_out}, torch::dtype(torch::kFloat32).device(input.device()));

    // Reuse persistent Q8_1 buffer
    int64_t q8_bytes = (D_in / QK8_1) * Q8_1_BLOCK_BYTES;
    uint8_t* q8_ptr = get_q8_buffer(q8_bytes, input.device());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    quantize_q8_1_cuda(
        input_f32.data_ptr<float>(), q8_ptr,
        (int)D_in, stream);
    gemv_q6k_q8_1(
        q8_ptr, q_weight.data_ptr<uint8_t>(),
        output.data_ptr<float>(), (int)D_in, (int)D_out, stream);

    return output;
}


torch::Tensor fused_gemv_q3k_torch(
    torch::Tensor input,
    torch::Tensor q_weight,
    int64_t D_in,
    int64_t D_out
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(q_weight.is_cuda(), "q_weight must be on CUDA");
    TORCH_CHECK(D_in % 256 == 0, "D_in must be multiple of 256");

    auto input_f32 = input.to(torch::kFloat32).contiguous();
    auto output = torch::empty({D_out}, torch::dtype(torch::kFloat32).device(input.device()));

    // Reuse persistent Q8_1 buffer
    int64_t q8_bytes = (D_in / QK8_1) * Q8_1_BLOCK_BYTES;
    uint8_t* q8_ptr = get_q8_buffer(q8_bytes, input.device());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    quantize_q8_1_cuda(
        input_f32.data_ptr<float>(), q8_ptr,
        (int)D_in, stream);
    gemv_q3k_q8_1(
        q8_ptr, q_weight.data_ptr<uint8_t>(),
        output.data_ptr<float>(), (int)D_in, (int)D_out, stream);

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemv_q6k", &fused_gemv_q6k_torch, "Fused Q6_K dequant + GEMV (dp4a, multi-warp)");
    m.def("fused_gemv_q3k", &fused_gemv_q3k_torch, "Fused Q3_K dequant + GEMV (dp4a, multi-warp)");
}
