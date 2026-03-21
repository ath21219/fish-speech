"""
GGUF loader: reads .gguf, builds state dict with GGUFLinear/GGUFEmbedding.
Applies on-the-fly dequant layers for quantized tensors,
and standard nn.Parameter for unquantized ones (norms, biases).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import gguf as gguf_lib
from loguru import logger

from fish_speech.gguf.dequant import (
    GGUFParameter, GGUFLinear, GGUFEmbedding,
    DEQUANT_FN, NATIVE_TORCH_QTYPES, dequantize_tensor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_metadata(reader: gguf_lib.GGUFReader) -> dict:
    """Extract metadata key-value pairs from GGUF reader."""
    meta = {}
    for name, field in reader.fields.items():
        try:
            if hasattr(field, 'parts') and len(field.parts) > 0:
                raw = bytes(field.parts[-1])
                try:
                    meta[name] = raw.decode('utf-8')
                except Exception:
                    meta[name] = raw
            elif len(field.data) == 1:
                val = field.data[0]
                meta[name] = int(val) if val == int(val) else val
            elif len(field.data) > 1:
                meta[name] = [int(x) if x == int(x) else x for x in field.data]
        except Exception:
            pass
    return meta


def _gguf_tensor_to_raw(tensor) -> tuple[torch.Tensor, gguf_lib.GGMLQuantizationType, tuple[int, ...]]:
    """Convert a GGUFReader tensor to (raw_uint8_tensor, qtype, pytorch_shape).

    GGUF stores tensor shapes in GGML convention [inner_dim, outer_dim, ...]
    while PyTorch uses [outer_dim, inner_dim, ...].
    For 2D tensors this means we need to reverse the shape.
    For 1D tensors (norms, biases) the shape is unchanged.
    """
    qtype = tensor.tensor_type
    gguf_shape = tuple(int(x) for x in tensor.shape)

    # Reverse shape for 2D+ tensors (GGML → PyTorch convention)
    if len(gguf_shape) >= 2:
        shape = tuple(reversed(gguf_shape))
    else:
        shape = gguf_shape

    raw_np = tensor.data  # numpy.memmap

    # Convert to contiguous uint8 bytes regardless of source dtype
    contiguous = np.ascontiguousarray(raw_np)
    raw_bytes = contiguous.view(np.uint8).flatten()
    data_tensor = torch.from_numpy(raw_bytes.copy())

    return data_tensor, qtype, shape


# ---------------------------------------------------------------------------
# Name mapping (GGUF → fish-speech parameter names)
# ---------------------------------------------------------------------------
# inspect_gguf.py confirmed that tensor names in this GGUF file are
# IDENTICAL to fish-speech parameter names (e.g. layers.0.attention.wqkv.weight).
# The mapping below is kept for compatibility with other GGUF exporters
# that may use llama.cpp-style naming (blk.0.attn_qkv.weight, etc.).

_STATIC_NAME_MAP = {
    "token_embd.weight":    "embeddings.weight",
    "output_norm.weight":   "norm.weight",
    "output.weight":        "output.weight",
    "codebook_embd.weight": "codebook_embeddings.weight",
    "fast_embd.weight":     "fast_embeddings.weight",
    "fast_norm.weight":     "fast_norm.weight",
    "fast_output.weight":   "fast_output.weight",
}

_SUFFIX_MAP = {
    "attn_qkv.weight":     "attention.wqkv.weight",
    "attn_qkv.bias":       "attention.wqkv.bias",
    "attn_output.weight":  "attention.wo.weight",
    "attn_output.bias":    "attention.wo.bias",
    "attn_norm.weight":    "attention_norm.weight",
    "attn_q_norm.weight":  "attention.q_norm.weight",
    "attn_k_norm.weight":  "attention.k_norm.weight",
    "ffn_norm.weight":     "ffn_norm.weight",
    "ffn_gate.weight":     "feed_forward.w1.weight",
    "ffn_up.weight":       "feed_forward.w3.weight",
    "ffn_down.weight":     "feed_forward.w2.weight",
}


def _map_tensor_name(gguf_name: str) -> str:
    """Map a GGUF tensor name to fish-speech parameter name."""

    # 1. Static map
    if gguf_name in _STATIC_NAME_MAP:
        return _STATIC_NAME_MAP[gguf_name]

    # 2. Block patterns: blk.{i}.xxx / fast_blk.{i}.xxx
    m = re.match(r'^(fast_)?blk\.(\d+)\.(.+)$', gguf_name)
    if m:
        prefix = "fast_layers" if m.group(1) else "layers"
        idx = m.group(2)
        suffix = _SUFFIX_MAP.get(m.group(3), m.group(3))
        return f"{prefix}.{idx}.{suffix}"

    # 3. Passthrough (names already match fish-speech)
    return gguf_name


def _is_quantized(qtype: gguf_lib.GGMLQuantizationType) -> bool:
    return qtype not in NATIVE_TORCH_QTYPES and qtype in DEQUANT_FN


def _should_use_gguf_linear(name: str, shape: tuple) -> bool:
    """2D weight matrices (not norms) get on-the-fly dequant."""
    return len(shape) == 2 and "norm" not in name


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_gguf_into_model(
    model: nn.Module,
    gguf_path: str | Path,
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Load GGUF weights into an existing DualARTransformer.

    Quantized 2D weight tensors  → GGUFLinear  (on-the-fly dequant)
    Quantized embedding tensors  → GGUFEmbedding
    Unquantized / 1D tensors     → standard nn.Parameter

    Args:
        model: Initialized DualARTransformer (random weights)
        gguf_path: Path to .gguf file
        device: Target device
        compute_dtype: dtype for dequantized params (float16 for RTX 2070)

    Returns:
        Model with GGUF weights loaded
    """
    gguf_path = Path(gguf_path)
    reader = gguf_lib.GGUFReader(str(gguf_path))
    metadata = _read_metadata(reader)

    # ---- Read and classify all tensors ----
    transformer_tensors: dict[str, tuple[torch.Tensor, gguf_lib.GGMLQuantizationType, tuple[int, ...]]] = {}
    codec_tensors: dict[str, tuple[torch.Tensor, gguf_lib.GGMLQuantizationType, tuple[int, ...]]] = {}

    for t in reader.tensors:
        raw, qtype, shape = _gguf_tensor_to_raw(t)
        fs_name = _map_tensor_name(t.name)

        # Codec tensors start with "c." based on gguf_tensors.txt
        if fs_name.startswith("c.") or fs_name.startswith("codec."):
            codec_tensors[fs_name] = (raw, qtype, shape)
        else:
            transformer_tensors[fs_name] = (raw, qtype, shape)

    logger.info(
        f"GGUF: {len(transformer_tensors)} transformer tensors, "
        f"{len(codec_tensors)} codec tensors"
    )

    # ---- Walk model and replace / load ----
    replaced = 0
    loaded_as_param = 0
    skipped = 0
    skipped_names = []

    for fs_name, (raw, qtype, shape) in transformer_tensors.items():
        # Split "layers.0.attention.wqkv.weight" → parent="layers.0.attention.wqkv", attr="weight"
        parts = fs_name.rsplit(".", 1)
        if len(parts) == 2:
            parent_path, attr_name = parts
        else:
            parent_path, attr_name = "", parts[0]

        # Locate parent module
        try:
            parent = model.get_submodule(parent_path) if parent_path else model
        except (AttributeError, torch.nn.modules.module.ModuleAttributeError):
            skipped += 1
            skipped_names.append(fs_name)
            continue

        is_quant = _is_quantized(qtype)

        # --- Case 1: Quantized 2D weight → GGUFLinear ---
        if is_quant and attr_name == "weight" and isinstance(parent, nn.Linear):
            qp = GGUFParameter(raw.to(device), qtype, shape)  # ← .to(device) 追加
            bias_data = parent.bias.data.to(compute_dtype).to(device) if parent.bias is not None else None
            new_module = GGUFLinear(qp, bias=bias_data)

            if "." in parent_path:
                gp_path, parent_attr = parent_path.rsplit(".", 1)
                grandparent = model.get_submodule(gp_path)
            else:
                gp_path, parent_attr = "", parent_path
                grandparent = model
            setattr(grandparent, parent_attr, new_module)
            replaced += 1
            continue

        # --- Case 2: Quantized embedding → GGUFEmbedding ---
        if is_quant and attr_name == "weight" and isinstance(parent, nn.Embedding):
            qp = GGUFParameter(raw.to(device), qtype, shape)  # ← .to(device) 追加
            new_module = GGUFEmbedding(qp)

            if "." in parent_path:
                gp_path, parent_attr = parent_path.rsplit(".", 1)
                grandparent = model.get_submodule(gp_path)
            else:
                gp_path, parent_attr = "", parent_path
                grandparent = model
            setattr(grandparent, parent_attr, new_module)
            replaced += 1
            continue

        # --- Case 3: Dequantize and load as nn.Parameter ---
        if is_quant:
            weight = dequantize_tensor(raw, qtype, shape, dtype=compute_dtype)
        elif qtype == gguf_lib.GGMLQuantizationType.F16:
            weight = raw.view(torch.float16).reshape(shape).to(compute_dtype)
        elif qtype == gguf_lib.GGMLQuantizationType.F32:
            weight = raw.view(torch.float32).reshape(shape).to(compute_dtype)
        else:
            # BF16 or other
            weight = raw.view(torch.bfloat16).reshape(shape).to(compute_dtype)

        try:
            existing = getattr(parent, attr_name)
            if isinstance(existing, nn.Parameter):
                setattr(parent, attr_name, nn.Parameter(weight.to(device), requires_grad=False))
            else:
                setattr(parent, attr_name, nn.Parameter(weight.to(device), requires_grad=False))
            loaded_as_param += 1
        except Exception as e:
            logger.debug(f"  skip (set failed): {fs_name}: {e}")
            skipped += 1
            skipped_names.append(fs_name)

    logger.info(
        f"GGUF load complete: {replaced} replaced (GGUFLinear/Embedding), "
        f"{loaded_as_param} standard params, {skipped} skipped"
    )
    if skipped_names and skipped <= 20:
        for name in skipped_names:
            logger.debug(f"  skipped: {name}")

    # Store codec tensors and metadata on the model for later use
    model._gguf_codec_tensors = codec_tensors
    model._gguf_metadata = metadata

    return model


def load_gguf_model(
    gguf_path: str | Path,
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.float16,
    *,
    config_path: Optional[str | Path] = None,
    tokenizer_path: Optional[str | Path] = None,
    max_seq_len: Optional[int] = None,
):
    """High-level: create DualARTransformer and load GGUF weights.

    Args:
        gguf_path: Path to .gguf file
        config_path: config.json (defaults to same dir as gguf)
        tokenizer_path: path to tokenizer.json or directory containing it
        device: target device
        compute_dtype: float16 recommended for RTX 2070
        max_seq_len: override max sequence length (reduce for VRAM savings)

    Returns:
        Model ready for inference (eval mode)
    """
    from fish_speech.models.text2semantic.llama import (
        BaseModelArgs, DualARTransformer,
    )

    gguf_path = Path(gguf_path)
    gguf_dir = gguf_path.parent

    # ---- Config ----
    cfg_path = Path(config_path) if config_path else gguf_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config.json not found at {cfg_path}. "
            f"Download it from the model repository."
        )
    config = BaseModelArgs.from_pretrained(str(cfg_path))

    # Override max_seq_len if specified (VRAM saving)
    if max_seq_len is not None:
        logger.info(f"Overriding max_seq_len: {config.max_seq_len} → {max_seq_len}")
        config.max_seq_len = max_seq_len

    # ---- Tokenizer ----
    # For GGUF models we load tokenizer.json directly via the tokenizers
    # library, bypassing FishTokenizer / AutoTokenizer which requires
    # HuggingFace-compatible config files (tokenizer_config.json etc.)
    # that the GGUF distribution doesn't include.
    tokenizer, semantic_begin_id, semantic_end_id = _load_tokenizer_for_gguf(gguf_path)

    config.semantic_begin_id = semantic_begin_id
    config.semantic_end_id = semantic_end_id
    logger.info(
        f"Semantic token ID range: {semantic_begin_id} -> {semantic_end_id}"
    )

    logger.info(
        f"Creating model: {config.model_type}, "
        f"dim={config.dim}, n_layer={config.n_layer}, "
        f"max_seq_len={config.max_seq_len}"
    )

    # ---- Create model on meta (saves VRAM during init) ----
    with torch.device("meta"):
        model = DualARTransformer(config)

    # Attach tokenizer as a lightweight wrapper
    model.tokenizer = tokenizer

    # ---- Load GGUF weights ----
    model = load_gguf_into_model(
        model, gguf_path, device=device, compute_dtype=compute_dtype,
    )

    # ---- Move to device (meta テンソルが残っていれば実体化) ----
    for name, param in list(model.named_parameters()):
        if param.device == torch.device("meta"):
            parts = name.split(".")
            submodule = model
            for p in parts[:-1]:
                submodule = getattr(submodule, p)
            setattr(submodule, parts[-1],
                    nn.Parameter(torch.zeros(param.shape, dtype=compute_dtype, device=device),
                                 requires_grad=False))
    for name, buf in list(model.named_buffers()):
        if buf.device == torch.device("meta"):
            parts = name.split(".")
            submodule = model
            for p in parts[:-1]:
                submodule = getattr(submodule, p)
            setattr(submodule, parts[-1],
                    torch.zeros(buf.shape, dtype=buf.dtype, device=device))

    # ---- Recompute buffers that were skipped during meta init ----
    from fish_speech.models.text2semantic.llama import precompute_freqs_cis

    # freqs_cis (Slow-AR RoPE)
    model.freqs_cis = precompute_freqs_cis(
        config.max_seq_len,
        config.head_dim,
        config.rope_base,
    ).to(compute_dtype)

    # causal_mask
    model.causal_mask = torch.tril(
        torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
    )

    # fast_freqs_cis (Fast-AR RoPE)
    model.fast_freqs_cis = precompute_freqs_cis(
        config.num_codebooks,
        config.fast_head_dim,
        config.rope_base,
    ).to(compute_dtype)

    model = model.to(device)

    # ---- Fix bf16 buffers for non-bf16 GPUs ----
    if compute_dtype != torch.bfloat16:
        _convert_bf16_buffers(model, compute_dtype)

    # ---- Patch SDPA for compatibility ----
    from fish_speech.gguf.patches import patch_attention_sdpa
    patch_attention_sdpa()

    logger.info(f"Model ready on {device}")
    return model.eval()


def _convert_bf16_buffers(model: nn.Module, target_dtype: torch.dtype):
    """Convert all bfloat16 buffers to target dtype.

    Turing, sm_75 does not support bfloat16 in SDPA kernels.
    freqs_cis and fast_freqs_cis are created as bf16 by precompute_freqs_cis().
    """
    converted = 0
    for name, buf in model.named_buffers():
        if buf is not None and buf.dtype == torch.bfloat16:
            # named_buffers gives dotted names like "freqs_cis" or "layers.0.something"
            # We need to set the buffer on the correct submodule
            parts = name.split(".")
            submodule = model
            for part in parts[:-1]:
                submodule = getattr(submodule, part)
            setattr(submodule, parts[-1], buf.to(target_dtype))
            converted += 1
    if converted:
        logger.debug(f"  Converted {converted} bf16 buffers → {target_dtype}")


def _load_tokenizer_for_gguf(gguf_path: str) -> tuple:
    """Load tokenizer for GGUF model."""
    gguf_dir = Path(gguf_path).parent
    tokenizer_path = gguf_dir / "tokenizer.json"
    
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"tokenizer.json not found in {gguf_dir}. "
            "Download from https://huggingface.co/rodrigomt/s2-pro-gguf"
        )
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    
    # Detect semantic token range from tokenizer vocab
    from tokenizers import Tokenizer as HFTokenizer
    temp_tok = HFTokenizer.from_file(str(tokenizer_path))
    vocab = temp_tok.get_vocab()
    
    valid_ids = []
    for code_idx in range(4096):
        token = f"<|semantic:{code_idx}|>"
        if token in vocab:
            valid_ids.append(vocab[token])
    
    if not valid_ids:
        raise ValueError("No semantic tokens found in tokenizer vocabulary!")
    
    semantic_begin_id = min(valid_ids)
    semantic_end_id = max(valid_ids)
    logger.info(f"Found {len(valid_ids)} semantic tokens in vocab")
    
    # Create wrapper — only 3 args, it builds vocab internally
    wrapper = _GGUFTokenizerWrapper(
        str(tokenizer_path), semantic_begin_id, semantic_end_id
    )
    
    return wrapper, semantic_begin_id, semantic_end_id


class _GGUFTokenizerWrapper:
    """Lightweight tokenizer wrapper for GGUF-loaded models."""
    
    def __init__(self, tokenizer_path: str, semantic_begin_id: int, semantic_end_id: int):
        from tokenizers import Tokenizer as HFTokenizer
        self._tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        self._semantic_begin_id = semantic_begin_id
        self._semantic_end_id = semantic_end_id
        
        # Build vocab and reverse vocab
        self._vocab = self._tokenizer.get_vocab()
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        
        # Build semantic_id_to_token_id map (code_idx -> token_id)
        self.semantic_id_to_token_id = {}
        for code_idx in range(4096):
            token = f"<|semantic:{code_idx}|>"
            if token in self._vocab:
                self.semantic_id_to_token_id[code_idx] = self._vocab[token]
        
        # Build semantic_map_tensor
        self.semantic_map_tensor = torch.zeros(4096, dtype=torch.long)
        for k, v in self.semantic_id_to_token_id.items():
            self.semantic_map_tensor[k] = v

    @property
    def semantic_begin_id(self) -> int:
        return self._semantic_begin_id
    
    @property
    def semantic_end_id(self) -> int:
        return self._semantic_end_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
    
    @property
    def pad_token_id(self) -> int:
        token = "<|pad|>"
        return self._vocab.get(token, 0)
    
    @property
    def eos_token_id(self) -> int:
        token = "<|endoftext|>"
        return self._vocab.get(token, 0)
    
    def get_token_id(self, token: str) -> int:
        """Get token ID from token string. Used by generate() for IM_END_TOKEN."""
        return self._vocab.get(token, 0)
    
    def convert_tokens_to_ids(self, token: str) -> int:
        """Compatibility with HuggingFace tokenizer interface."""
        if isinstance(token, list):
            return [self._vocab.get(t, 0) for t in token]
        return self._vocab.get(token, 0)
    
    def get_vocab(self) -> dict:
        """Return full vocabulary dict."""
        return self._vocab
    
    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> list[int]:
        """Encode text to token IDs.
        
        Handles the allowed_special kwarg used by FishTokenizer for Qwen/Tiktoken backends.
        The tokenizers library handles special tokens differently, so we just encode normally.
        """
        # The tokenizers library doesn't use allowed_special; it uses add_special_tokens
        encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoded.ids
    
    def decode(self, tokens, **kwargs) -> str:
        """Decode token IDs to text."""
        if isinstance(tokens, int):
            tokens = [tokens]
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self._tokenizer.decode(tokens, skip_special_tokens=False)
    
    def save_pretrained(self, path: str):
        """Save tokenizer (no-op for GGUF wrapper)."""
        pass
    
    def __getattr__(self, name):
        """Forward unknown attributes to underlying tokenizer."""
        return getattr(self._tokenizer, name)
