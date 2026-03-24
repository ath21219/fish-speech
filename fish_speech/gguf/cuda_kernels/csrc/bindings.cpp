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

    // Allocate Q8_1 workspace via PyTorch caching allocator (fast after first call)
    int64_t n_q8_blocks = D_in / QK8_1;
    auto q8_buf = torch::empty(
        {n_q8_blocks * Q8_1_BLOCK_BYTES},
        torch::dtype(torch::kUInt8).device(input.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    quantize_q8_1_cuda(
        input_f32.data_ptr<float>(), q8_buf.data_ptr<uint8_t>(),
        (int)D_in, stream);
    gemv_q6k_q8_1(
        q8_buf.data_ptr<uint8_t>(), q_weight.data_ptr<uint8_t>(),
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

    int64_t n_q8_blocks = D_in / QK8_1;
    auto q8_buf = torch::empty(
        {n_q8_blocks * Q8_1_BLOCK_BYTES},
        torch::dtype(torch::kUInt8).device(input.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    quantize_q8_1_cuda(
        input_f32.data_ptr<float>(), q8_buf.data_ptr<uint8_t>(),
        (int)D_in, stream);
    gemv_q3k_q8_1(
        q8_buf.data_ptr<uint8_t>(), q_weight.data_ptr<uint8_t>(),
        output.data_ptr<float>(), (int)D_in, (int)D_out, stream);

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemv_q6k", &fused_gemv_q6k_torch, "Fused Q6_K dequant + GEMV (dp4a)");
    m.def("fused_gemv_q3k", &fused_gemv_q3k_torch, "Fused Q3_K dequant + GEMV (dp4a)");
}
