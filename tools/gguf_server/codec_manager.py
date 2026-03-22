"""
Codec GPU/CPU lifecycle management.

Provides:
  - codec_on_gpu()       context manager for GPU codec access
  - restore_after_codec() restore model to GPU after codec swap
  - load_codec_from_gguf() build codec from GGUF-embedded tensors
"""

import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from loguru import logger

from .state import state

# Project root (tools/gguf_server/../../ = project root)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@contextmanager
def codec_on_gpu():
    """
    Provide codec on GPU for encode/decode operations.

    When codec_gpu_resident is enabled (default for small quant models),
    codec stays on GPU permanently — no swap needed.
    Falls back to model↔codec swap when VRAM is too tight.
    """
    if state.codec_gpu_resident:
        yield state.codec
        return

    model = state.model
    codec = state.codec
    device = state.device

    # 1. Offload GGUF model to CPU
    logger.info("Offloading GGUF model to CPU...")
    t0 = time.perf_counter()
    model.cpu()
    for _name, module in model.named_modules():
        if hasattr(module, "qparam") and hasattr(module.qparam, "data"):
            if module.qparam.data.is_cuda:
                module.qparam.data = module.qparam.data.cpu()
        for attr_name in ("weight", "qweight", "data"):
            attr = getattr(module, attr_name, None)
            if isinstance(attr, torch.Tensor) and attr.is_cuda:
                setattr(module, attr_name, attr.cpu())
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / 1e9
    logger.info(
        f"Model offloaded in {time.perf_counter() - t0:.1f}s "
        f"(VRAM: {vram_after:.2f} GB)"
    )

    # 2. Move codec to GPU
    logger.info("Moving codec to GPU...")
    t0 = time.perf_counter()
    codec.to(device=device)
    torch.cuda.synchronize()
    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(
        f"Codec on GPU in {time.perf_counter() - t0:.1f}s (VRAM: {vram:.2f} GB)"
    )

    try:
        yield codec
    finally:
        pass


def restore_after_codec():
    """Move codec back to CPU and restore GGUF model to GPU."""
    if state.codec_gpu_resident:
        return

    model = state.model
    codec = state.codec
    device = state.device

    logger.info("Offloading codec to CPU...")
    codec.to(device="cpu")
    torch.cuda.empty_cache()

    logger.info("Restoring GGUF model to GPU...")
    t0 = time.perf_counter()
    model.to(device)
    for _name, module in model.named_modules():
        if hasattr(module, "qparam") and hasattr(module.qparam, "data"):
            if not module.qparam.data.is_cuda:
                module.qparam.data = module.qparam.data.to(device)
        for attr_name in ("weight", "qweight", "data"):
            attr = getattr(module, attr_name, None)
            if isinstance(attr, torch.Tensor) and not attr.is_cuda:
                setattr(module, attr_name, attr.to(device))
    torch.cuda.synchronize()

    model._cache_setup_done = False
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=torch.float16,
        )
    model._cache_setup_done = True
    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(
        f"Model restored in {time.perf_counter() - t0:.1f}s "
        f"(VRAM: {vram:.2f} GB)"
    )


def load_codec_from_gguf(model, device="cpu"):
    """
    Build codec model structure via Hydra config, then load weights
    from GGUF tensors stored in model._gguf_codec_tensors.
    """
    import gguf as gguf_lib
    from fish_speech.gguf.dequant import (
        DEQUANT_FN,
        NATIVE_TORCH_QTYPES,
        dequantize_tensor,
    )
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.nn.utils.parametrize import remove_parametrizations

    codec_tensors = getattr(model, "_gguf_codec_tensors", None)
    if not codec_tensors:
        raise RuntimeError(
            "No codec tensors found in GGUF. "
            "Make sure the GGUF file contains c.* tensors."
        )

    # 1. Find codec config
    config_candidates = [
        project_root / "fish_speech" / "configs" / "modded_dac_vq.yaml",
        project_root / "configs" / "modded_dac_vq.yaml",
        Path("/app/fish_speech/configs/modded_dac_vq.yaml"),
        Path("/app/configs/modded_dac_vq.yaml"),
    ]
    try:
        import fish_speech

        if hasattr(fish_speech, "__file__") and fish_speech.__file__ is not None:
            config_candidates.insert(
                0,
                Path(fish_speech.__file__).parent / "configs" / "modded_dac_vq.yaml",
            )
        elif hasattr(fish_speech, "__path__"):
            for p in fish_speech.__path__:
                config_candidates.insert(
                    0, Path(p) / "configs" / "modded_dac_vq.yaml"
                )
    except Exception:
        pass

    config_path = None
    for candidate in config_candidates:
        if candidate.exists():
            config_path = candidate
            break

    if config_path is None:
        searched = "\n  ".join(str(c) for c in config_candidates)
        raise FileNotFoundError(
            f"Codec config modded_dac_vq.yaml not found. Searched:\n  {searched}"
        )

    logger.debug(f"Using codec config: {config_path}")
    cfg = OmegaConf.load(str(config_path))
    codec = instantiate(cfg)
    codec.eval()

    # 2. Remove weight_norm parametrizations
    wn_removed = 0
    for name, module in codec.named_modules():
        if hasattr(module, "parametrizations") and hasattr(
            module.parametrizations, "weight"
        ):
            try:
                remove_parametrizations(module, "weight")
                wn_removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove parametrization from {name}: {e}")
        elif hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            try:
                torch.nn.utils.remove_weight_norm(module)
                wn_removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove weight_norm from {name}: {e}")
    logger.info(f"Removed weight_norm from {wn_removed} modules")

    # 3. Build state dict from GGUF codec tensors
    gguf_state_dict = {}
    for gguf_name, (raw_data, qtype, gguf_shape) in codec_tensors.items():
        param_name = gguf_name[2:] if gguf_name.startswith("c.") else gguf_name

        if qtype in NATIVE_TORCH_QTYPES:
            if qtype == gguf_lib.GGMLQuantizationType.F16:
                tensor = raw_data.view(torch.float16).reshape(gguf_shape)
            elif qtype == gguf_lib.GGMLQuantizationType.F32:
                tensor = raw_data.view(torch.float32).reshape(gguf_shape)
            else:
                tensor = raw_data.view(torch.bfloat16).reshape(gguf_shape)
        elif qtype in DEQUANT_FN:
            tensor = dequantize_tensor(
                raw_data, qtype, gguf_shape, dtype=torch.float16
            )
        else:
            logger.warning(
                f"Skipping codec tensor {param_name}: unsupported qtype {qtype}"
            )
            continue

        gguf_state_dict[param_name] = tensor.to(device)

    # 4. Load into codec model
    codec_state = codec.state_dict()
    loaded = 0
    skipped = []

    for param_name, param_tensor in gguf_state_dict.items():
        if param_name in codec_state:
            target_shape = codec_state[param_name].shape
            if target_shape == param_tensor.shape:
                codec_state[param_name] = param_tensor
                loaded += 1
            elif param_tensor.numel() == codec_state[param_name].numel():
                codec_state[param_name] = param_tensor.reshape(target_shape)
                loaded += 1
            elif (
                target_shape == param_tensor.T.shape and param_tensor.dim() == 2
            ):
                codec_state[param_name] = param_tensor.T
                loaded += 1
            else:
                skipped.append(
                    f"{param_name}: GGUF {param_tensor.shape} "
                    f"({param_tensor.numel()}) vs "
                    f"model {target_shape} ({codec_state[param_name].numel()})"
                )
        else:
            skipped.append(f"{param_name}: not in model state_dict")

    codec.load_state_dict(codec_state, strict=False)
    logger.info(
        f"Codec: loaded {loaded}/{len(gguf_state_dict)} tensors from GGUF, "
        f"{len(skipped)} skipped"
    )
    if skipped:
        for s in skipped[:30]:
            logger.warning(f"  Codec skip: {s}")
        if len(skipped) > 30:
            logger.warning(f"  ... and {len(skipped) - 30} more")

    # 5. Shrink oversized Transformer buffers
    max_codec_frames = 2048
    shrunk_buffers = 0
    for _name, module in codec.named_modules():
        if hasattr(module, "causal_mask") and module.causal_mask is not None:
            if module.causal_mask.shape[0] > max_codec_frames:
                new_mask = module.causal_mask[
                    :max_codec_frames, :max_codec_frames
                ].clone()
                module.register_buffer("causal_mask", new_mask, persistent=False)
                shrunk_buffers += 1
        if hasattr(module, "freqs_cis") and module.freqs_cis is not None:
            if module.freqs_cis.shape[0] > max_codec_frames:
                new_freqs = module.freqs_cis[:max_codec_frames].clone()
                module.register_buffer("freqs_cis", new_freqs, persistent=False)
                shrunk_buffers += 1
    if shrunk_buffers > 0:
        logger.info(
            f"Shrunk {shrunk_buffers} Transformer buffers "
            f"to max_frames={max_codec_frames}"
        )

    # 6. Move to target device/dtype
    target_dtype = torch.float32 if device == "cpu" else torch.float16
    codec = codec.to(device=device, dtype=target_dtype)

    # Verify essential attributes
    if not hasattr(codec, "sample_rate"):
        meta = getattr(model, "_gguf_metadata", {})
        sr = meta.get("fish_speech.codec.sample_rate", 44100)
        codec.sample_rate = int(sr) if isinstance(sr, (int, float)) else 44100
        logger.info(
            f"Set codec.sample_rate = {codec.sample_rate} from GGUF metadata"
        )

    return codec
