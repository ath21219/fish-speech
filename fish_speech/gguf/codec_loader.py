"""
Load the DAC codec model from GGUF codec tensors (c.* prefix).

The GGUF bundles the codec, so a separate codec.pth is optional.
This loader extracts c.* tensors and loads them into the DAC model.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import gguf as gguf_lib
from loguru import logger

from fish_speech.gguf.dequant import dequantize_tensor, NATIVE_TORCH_QTYPES, _is_quantized


def build_codec_state_dict(
    codec_tensors: dict[str, tuple],
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> OrderedDict[str, torch.Tensor]:
    """Convert GGUF codec tensor dict to a state_dict for the DAC codec.

    The GGUF stores codec tensors with 'c.' prefix:
        c.encoder.block.0.conv.weight  →  encoder.block.0.conv.weight
        c.decoder.model.0.conv.weight  →  decoder.model.0.conv.weight
        c.quantizer.*                  →  quantizer.*

    These map to the DAC model's 'generator.' prefix that fish-speech strips
    during normal loading (see load_codec_model in inference.py).

    Args:
        codec_tensors: Dict from load_gguf_into_model, keys like "c.decoder.model.0..."
        dtype: Target dtype
        device: Target device

    Returns:
        OrderedDict suitable for codec.load_state_dict()
    """
    state_dict = OrderedDict()

    for gguf_name, (raw, qtype, shape) in codec_tensors.items():
        # Strip "c." prefix
        if gguf_name.startswith("c."):
            model_name = gguf_name[2:]
        else:
            model_name = gguf_name

        # Dequantize (all codec tensors are F16 in this GGUF, but handle generically)
        if qtype == gguf_lib.GGMLQuantizationType.F32:
            tensor = raw.view(torch.float32).reshape(shape).to(dtype=dtype, device=device)
        elif qtype == gguf_lib.GGMLQuantizationType.F16:
            tensor = raw.view(torch.float16).reshape(shape).to(dtype=dtype, device=device)
        elif qtype in NATIVE_TORCH_QTYPES:
            tensor = raw.reshape(shape).to(dtype=dtype, device=device)
        else:
            tensor = dequantize_tensor(raw, qtype, shape, dtype=dtype).to(device=device)

        state_dict[model_name] = tensor

    logger.info(f"Built codec state_dict: {len(state_dict)} tensors")
    return state_dict


def load_codec_from_gguf(
    codec_tensors: dict[str, tuple],
    config_name: str = "modded_dac_vq",
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
):
    """Instantiate DAC codec and load weights from GGUF codec tensors.

    Args:
        codec_tensors: From load_gguf_into_model
        config_name: Hydra config name for the codec
        dtype: Model dtype
        device: Device

    Returns:
        Loaded codec model ready for decode
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = (
        Path(__file__).parent.parent / "configs" / f"{config_name}.yaml"
    )
    cfg = OmegaConf.load(str(config_path))
    codec = instantiate(cfg)

    state_dict = build_codec_state_dict(codec_tensors, dtype=dtype, device="cpu")

    err = codec.load_state_dict(state_dict, strict=False)
    logger.info(f"Codec load from GGUF: {err}")

    codec.eval()
    codec.to(device=device, dtype=dtype)
    return codec
