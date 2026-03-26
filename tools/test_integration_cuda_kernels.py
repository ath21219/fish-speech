import torch
import gc

QK_K = 256
Q6K_BYTES = 210
Q3K_BYTES = 110

from fish_speech.gguf.triton_kernels import fused_dequant_gemv_q6k, fused_dequant_gemv_q3k
from fish_speech.gguf.cuda_kernels import fused_gemv_q6k, fused_gemv_q3k

def make_valid_blocks(n_blocks, block_size, d_offset, device='cuda'):
    raw = torch.randint(0, 256, (n_blocks, block_size), dtype=torch.uint8, device='cpu')
    for i in range(n_blocks):
        d_val = torch.tensor([0.5], dtype=torch.float16)
        d_bytes = d_val.view(torch.uint8)
        raw[i, d_offset] = d_bytes[0]
        raw[i, d_offset + 1] = d_bytes[1]
    return raw.reshape(-1).to(device)

def benchmark_pair(D_out, D_in, block_size, d_off, cuda_fn, triton_fn, warmup=200, runs=500):
    n_blocks = (D_out * D_in) // QK_K
    raw = make_valid_blocks(n_blocks, block_size, d_off)
    x_f32 = torch.randn(D_in, dtype=torch.float32, device='cuda')
    x_f16 = x_f32.half()
    
    # Warmup both
    for _ in range(warmup):
        cuda_fn(x_f32, raw, D_in, D_out)
        triton_fn(x_f16, raw, D_in, D_out, dtype=torch.float16)
    torch.cuda.synchronize()
    
    # Benchmark CUDA
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(runs)]
    for s, e in events:
        s.record()
        cuda_fn(x_f32, raw, D_in, D_out)
        e.record()
    torch.cuda.synchronize()
    cuda_times = sorted([s.elapsed_time(e) for s, e in events])
    
    # Benchmark Triton
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(runs)]
    for s, e in events:
        s.record()
        triton_fn(x_f16, raw, D_in, D_out, dtype=torch.float16)
        e.record()
    torch.cuda.synchronize()
    triton_times = sorted([s.elapsed_time(e) for s, e in events])
    
    del raw, x_f32, x_f16
    gc.collect()
    torch.cuda.empty_cache()
    
    return cuda_times[len(cuda_times)//2], triton_times[len(triton_times)//2]

# Actual model decode shapes
SHAPES = [
    ('wqkv', 6144, 2560),
    ('wo',   2560, 4096),
    ('w1',   9728, 2560),
    ('w2',   2560, 9728),
]

for qname, bs, d_off, cfn, tfn in [
    ('Q6_K', Q6K_BYTES, 208, fused_gemv_q6k, fused_dequant_gemv_q6k),
    ('Q3_K', Q3K_BYTES, 108, fused_gemv_q3k, fused_dequant_gemv_q3k),
]:
    print(f'\n{"="*70}')
    print(f'  {qname} — batch=1 decode GEMV (median of 500 runs, warmup=200)')
    print(f'{"="*70}')
    print(f'{"Layer":>8s}  {"Shape":>14s}  {"CUDA":>10s}  {"Triton":>10s}  {"Speedup":>8s}')
    print('-' * 58)
    
    total_c = 0; total_t = 0
    
    for name, Dout, Din in SHAPES:
        tc, tt = benchmark_pair(Dout, Din, bs, d_off, cfn, tfn)
        sp = tt / tc if tc > 0 else 0
        count = 2 if name == 'w1' else 1  # w1+w3 have same shape
        total_c += tc * count
        total_t += tt * count
        label = f'{name}(x{count})' if count > 1 else name
        print(f'  {label:>6s}  {Dout:5d}x{Din:<5d}  {tc*1000:8.1f}us  {tt*1000:8.1f}us  {sp:6.2f}x')
    
    sp_total = total_t / total_c if total_c > 0 else 0
    print('-' * 58)
    print(f'  {"LAYER":>6s}  {"5 GEMVs":>14s}  {total_c*1000:8.1f}us  {total_t*1000:8.1f}us  {sp_total:6.2f}x')
    print(f'  40 layers total:  CUDA {total_c*40:.2f}ms  Triton {total_t*40:.2f}ms')
