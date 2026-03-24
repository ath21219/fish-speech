from fish_speech.gguf.dequant import (
    GGUFLinear,
    GGUFParameter,
    GGUFEmbedding,
    warmup_all_caches,
    clear_all_caches,
    setup_layer_cache_hooks,
    _TRITON_AVAILABLE,
)
from fish_speech.gguf.loader import load_gguf_model, load_gguf_into_model
from fish_speech.gguf.patches import patch_attention_sdpa

# NEW: expose CUDA kernel availability
try:
    from fish_speech.gguf.cuda_kernels import _CUDA_KERNELS_AVAILABLE
except ImportError:
    _CUDA_KERNELS_AVAILABLE = False
