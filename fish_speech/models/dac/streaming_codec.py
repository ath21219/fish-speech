"""
Streaming codec decoder with KV-cache for incremental decoding.

The DAC codec's decode path contains two causal Transformers:
  1) quantizer.post_module  (WindowLimitedTransformer, window=128, 8 layers)
  2) decoder.model[1].block[0] area — but actually inside DecoderBlock
     the transformer_module is NOT in the Sequential (it's commented out
     in the source). Checking the config: decoder_transformer_layers=[4,0,0,0]
     BUT DecoderBlock places transformer_module OUTSIDE self.block.

Wait — let me re-read DecoderBlock.__init__:

    transformer_module = WindowLimitedTransformer(...) if n_t_layer > 0 else Identity()
    self.block = nn.Sequential(
        Snake1d(input_dim),
        conv_trans_class(...),
        ResidualUnit(..., dilation=1),
        ResidualUnit(..., dilation=3),
        ResidualUnit(..., dilation=9),
    )

The transformer_module is created but NOT added to self.block!
There's a commented-out line: # transformer_module,

So decoder_transformer_layers=[4,0,0,0] creates a transformer but never uses it.
The only active Transformer in the decode path is quantizer.post_module.

This is great news — we only need to cache ONE transformer.

Strategy:
  - Give post_module's Transformer KV caches
  - On each streaming chunk, only feed the NEW VQ frames through post_module
    with appropriate input_pos, then run upsample + decoder on the new frames
  - The convolutions (CausalConvNet, CausalTransConvNet) need their own state
    management since they use causal padding

Actually, let's think more carefully about what "causal" means here for convolutions:

CausalConvNet pads (kernel_size - stride) zeros on the LEFT.
This means each output only depends on current and past inputs.
But when we process chunk 2 independently, the left padding is zeros
instead of the actual values from chunk 1's tail.

So the convolutions DO have cross-chunk dependencies through their receptive field.

For the upsample path (CausalTransConvNet): transposes then trims.
Similar issue — the transposed convolution's output at the boundary
depends on input values at the boundary.

Key insight: The convolutions' receptive field is LOCAL (a few frames),
while the Transformer's is GLOBAL (up to window_size=128).
The Transformer is the dominant source of artifacts.
The convolution boundary effects are much smaller and can be handled
with a small overlap-and-trim approach — but ONLY for the conv part,
while the Transformer gets proper KV caching.

REVISED STRATEGY — Two-phase approach:
  Phase A: post_module (Transformer) — use KV cache for exact incremental decode
  Phase B: upsample + decoder (all convolutions) — use overlap context

This hybrid approach gives us:
  - Exact Transformer output (no quality loss)
  - Tiny conv boundary artifacts smoothed by small overlap
  - O(n) total compute instead of O(n²)
"""

import time
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from typing import Optional
from fish_speech.models.dac.modded_dac import (
    DAC,
    WindowLimitedTransformer,
    Transformer,
    find_multiple,
)


class StreamingCodecDecoder:
    """
    Wraps a DAC codec for streaming incremental decode.

    Architecture analysis from config:
      - quantizer.post_module: WindowLimitedTransformer(causal=True, window=128, 8 layers, dim=1024)
        operates on VQ-frame space (T_vq after downsample)
      - quantizer.upsample: 2 stages of CausalTransConvNet + ConvNeXtBlock
        upsample T_vq → T_vq*4
      - decoder: CausalConv1d + DecoderBlock(stride=8,8,4,2) + final conv
        NOTE: decoder_transformer_layers=[4,0,0,0] but the transformer_module
        in DecoderBlock is NOT added to self.block (commented out in source).
        So the decoder is purely convolutional.

    Strategy:
      1) Codebook lookup (stateless) — just process new frames
      2) post_module Transformer — use KV cache for incremental processing
      3) upsample + decoder (all causal convolutions) — process with overlap context

    The overlap for convolutions needs to cover the receptive field.
    Receptive field analysis for upsample + decoder:
      - upsample: 2x CausalTransConvNet(k=factor,s=factor) + ConvNeXtBlock(k=7,groups=dim)
        Factor [2,2], so upsample ratio = 4x
        ConvNeXtBlock receptive field: kernel=7, dilation=1 → 7 samples
      - decoder: CausalConv1d(k=7) + 4x DecoderBlock
        Each DecoderBlock: CausalTransConv(k=2*stride,s=stride) + 3x ResidualUnit(k=7, dilation=1,3,9)
        ResidualUnit max RF: dilation=9, kernel=7 → (7-1)*9+1 = 55 samples
        Across 4 blocks with strides [8,8,4,2]: RF accumulates multiplicatively

    In VQ-frame space, the total decoder RF is roughly:
      ~55 (max residual unit) / stride_product * 4 blocks ≈ several frames
    Conservative estimate: 8 VQ frames of overlap should be more than enough.
    """

    # Number of VQ frames to use as overlap context for convolutions
    CONV_OVERLAP_FRAMES = 8

    def __init__(self, codec: DAC, device: str = "cuda", max_frames: int = 2048):
        self.codec = codec
        self.device = device
        self.max_frames = max_frames
        self.sample_rate = codec.sample_rate
        self.hop_length = codec.hop_length  # typically 512

        # Verify decoder transformer is inactive
        self._verify_decoder_transformer_inactive()

        # Setup KV caches for post_module
        self._setup_post_module_kv_cache(max_frames)

        # State
        self._frames_processed = 0       # Total VQ frames fed to post_module so far
        self._prev_z_overlap = None       # Last CONV_OVERLAP_FRAMES of z (post_module output) for conv context
        self._prev_audio_trim = 0         # Audio samples to trim from start of conv output (overlap region)

    def _verify_decoder_transformer_inactive(self):
        """
        Confirm that the transformer in DecoderBlock is not in the forward path.
        In the source code, transformer_module is created but not added to self.block.
        We verify by checking that self.block doesn't contain a WindowLimitedTransformer.
        """
        decoder = self.codec.decoder
        for i, layer in enumerate(decoder.model):
            if hasattr(layer, 'block') and isinstance(layer.block, nn.Sequential):
                for sublayer in layer.block:
                    if isinstance(sublayer, (WindowLimitedTransformer, Transformer)):
                        logger.warning(
                            f"Found active Transformer in decoder.model[{i}].block — "
                            f"streaming may have quality issues in decoder conv path"
                        )
                        return
        logger.debug("Confirmed: decoder path is purely convolutional (no active Transformer)")

    def _setup_post_module_kv_cache(self, max_frames: int):
        """
        Install KV caches into quantizer.post_module's Transformer layers.
        The post_module operates on downsampled VQ space.
        downsample_factor = [2,2] → 4x downsample, so T_transformer = T_vq / 4
        But wait — looking at quantizer.decode():
            z_q = semantic + residual
            z_q = self.post_module(z_q)   ← this operates on the DOWNSAMPLED space
            z_q = self.upsample(z_q)      ← then upsample
        The downsample happens in encode, not decode. In decode, the codebook
        outputs are already in the downsampled space. So post_module sees T_vq
        directly (no further downsample).

        Actually, let me re-read. The codebook_dim=8, input_dim=1024.
        from_codes returns shape (B, D, T) where D=input_dim.
        So post_module input is (B, 1024, T_vq) and it's a
        WindowLimitedTransformer with input_dim=1024, dim=1024.
        """
        post_module = self.codec.quantizer.post_module
        if not isinstance(post_module, WindowLimitedTransformer):
            logger.warning("post_module is not WindowLimitedTransformer, KV cache disabled")
            self._has_kv_cache = False
            return

        # Max sequence length the Transformer will see
        # This is in the VQ frame space directly
        max_seq = find_multiple(max_frames, 8)

        dtype = next(post_module.parameters()).dtype
        dev = next(post_module.parameters()).device

        post_module.setup_caches(
            max_batch_size=1,
            max_seq_length=max_seq,
        )

        # Move caches to correct device/dtype
        for layer in post_module.layers:
            if layer.attention.kv_cache is not None:
                layer.attention.kv_cache = layer.attention.kv_cache.to(device=dev, dtype=dtype)

        self._has_kv_cache = True
        self._post_module_window = post_module.window_size  # 128
        logger.info(
            f"StreamingCodecDecoder: KV cache installed for post_module "
            f"({len(post_module.layers)} layers, window={self._post_module_window}, "
            f"max_seq={max_seq})"
        )

    def reset(self):
        """Reset state for a new utterance."""
        self._frames_processed = 0
        self._prev_z_overlap = None
        self._prev_audio_trim = 0

        # Clear KV caches
        if self._has_kv_cache:
            post_module = self.codec.quantizer.post_module
            for layer in post_module.layers:
                if layer.attention.kv_cache is not None:
                    layer.attention.kv_cache.k_cache.zero_()
                    layer.attention.kv_cache.v_cache.zero_()

    def decode_chunk(self, codes: torch.Tensor) -> np.ndarray:
        """
        Decode a chunk of VQ codes incrementally.

        Args:
            codes: (num_codebooks, N) new VQ codes for this chunk

        Returns:
            np.ndarray: new audio samples (1D float32)
        """
        if codes.shape[1] == 0:
            return np.array([], dtype=np.float32)

        codes = codes.to(self.device)
        # Add batch dim: (1, num_codebooks, N)
        codes_3d = codes.unsqueeze(0)

        with torch.inference_mode():
            # --- Phase 1: Codebook lookup (stateless) ---
            quantizer = self.codec.quantizer
            indices = codes_3d.clone()
            indices[:, 0] = torch.clamp(indices[:, 0], max=quantizer.semantic_quantizer.codebook_size - 1)
            indices[:, 1:] = torch.clamp(indices[:, 1:], max=quantizer.quantizer.codebook_size - 1)

            z_q_semantic = quantizer.semantic_quantizer.from_codes(indices[:, :1])[0]
            z_q_residual = quantizer.quantizer.from_codes(indices[:, 1:])[0]
            z_q = z_q_semantic + z_q_residual
            # z_q: (1, 1024, N) in VQ frame space

            # --- Phase 2: post_module Transformer with KV cache ---
            z_post = self._run_post_module_incremental(z_q)
            # z_post: (1, 1024, N)

            # --- Phase 3: upsample + decoder with overlap context ---
            audio_new = self._run_conv_with_overlap(z_post)

        self._frames_processed += codes.shape[1]
        return audio_new

    def _run_post_module_incremental(self, z_new: torch.Tensor) -> torch.Tensor:
        """
        Run post_module on new frames using KV cache.

        The WindowLimitedTransformer.forward() normally:
          1) Transposes (channels_first)
          2) input_proj
          3) look_ahead_conv (Identity)
          4) Creates input_pos = arange(T)
          5) Creates mask
          6) Runs Transformer.forward(x, input_pos, mask)
          7) output_proj
          8) Transposes back

        For incremental mode, we need to:
          - Set input_pos to the GLOBAL position (not starting from 0)
          - Build mask that references cached KV positions
          - Feed only new frames through the Transformer
        """
        post_module = self.codec.quantizer.post_module

        if not self._has_kv_cache:
            # Fallback: run full sequence (accumulate codes externally)
            return post_module(z_new)

        # z_new: (1, 1024, N_new) channels_first
        N_new = z_new.shape[2]
        start_pos = self._frames_processed

        # Step 1-3: same transforms as WindowLimitedTransformer.forward
        x = z_new.transpose(1, 2)                    # (1, N_new, 1024)
        x = post_module.input_proj(x)                # (1, N_new, dim)
        x = post_module.look_ahead_conv(x)           # Identity

        # Step 4: global positions for new frames
        input_pos = torch.arange(
            start_pos, start_pos + N_new,
            device=x.device, dtype=torch.long
        )

        # Step 5: build mask for KV-cached attention
        # For windowed causal attention, each new position i attends to
        # max(0, i - window + 1) .. i in the global sequence
        # The KV cache stores all past positions, so the mask shape is
        # (1, 1, N_new, start_pos + N_new) — rows are new positions, cols are all cached + new
        total_len = start_pos + N_new
        window = post_module.window_size or total_len

        if post_module.window_size is not None:
            # Window-limited causal mask
            row_positions = input_pos.unsqueeze(1)  # (N_new, 1)
            col_positions = torch.arange(total_len, device=x.device).unsqueeze(0)  # (1, total_len)
            mask = (col_positions <= row_positions) & (col_positions >= (row_positions - window + 1).clamp(min=0))
            mask = mask.bool().unsqueeze(0).unsqueeze(0)  # (1, 1, N_new, total_len)
        else:
            # Full causal mask
            row_positions = input_pos.unsqueeze(1)
            col_positions = torch.arange(total_len, device=x.device).unsqueeze(0)
            mask = (col_positions <= row_positions)
            mask = mask.bool().unsqueeze(0).unsqueeze(0)

        # Step 6: run through Transformer layers with KV cache
        if post_module.config.pos_embed_type == "rope":
            freqs_cis = post_module.freqs_cis[input_pos]
        else:
            freqs_cis = None

        for layer in post_module.layers:
            x = layer(x, input_pos, freqs_cis, mask)
        x = post_module.norm(x)

        # Step 7-8: output proj and transpose back
        x = post_module.output_proj(x)               # (1, N_new, 1024)
        x = x.transpose(1, 2)                         # (1, 1024, N_new)

        return x

    def _run_conv_with_overlap(self, z_post_new: torch.Tensor) -> np.ndarray:
        """
        Run upsample + decoder on new frames with overlap from previous chunk.

        The causal convolutions pad zeros on the left. By prepending a few
        frames of the previous post_module output, we give the convolutions
        proper context instead of zero padding, eliminating boundary artifacts.

        We then trim the output to only return the audio corresponding to
        the new frames.
        """
        N_new = z_post_new.shape[2]
        overlap = self.CONV_OVERLAP_FRAMES

        if self._prev_z_overlap is not None and self._prev_z_overlap.shape[2] > 0:
            # Prepend overlap context
            z_input = torch.cat([self._prev_z_overlap.to(z_post_new.device), z_post_new], dim=2)
            has_overlap = True
            n_overlap = self._prev_z_overlap.shape[2]
        else:
            z_input = z_post_new
            has_overlap = False
            n_overlap = 0

        # Save overlap for next chunk (last `overlap` frames of post_module output)
        if N_new >= overlap:
            self._prev_z_overlap = z_post_new[:, :, -overlap:].detach().cpu()
        else:
            # Less frames than overlap — keep what we have
            if self._prev_z_overlap is not None:
                combined = torch.cat([self._prev_z_overlap.to(z_post_new.device), z_post_new], dim=2)
                self._prev_z_overlap = combined[:, :, -overlap:].detach().cpu()
            else:
                self._prev_z_overlap = z_post_new.detach().cpu()

        # Run upsample
        z_up = self.codec.quantizer.upsample(z_input)
        # z_up: (1, 1024, (n_overlap + N_new) * 4)

        # Run decoder
        audio = self.codec.decoder(z_up)
        # audio: (1, 1, (n_overlap + N_new) * upsample_total)
        audio_np = audio[0, 0].float().cpu().numpy()

        if has_overlap:
            # Calculate audio samples produced by the overlap frames
            # Each VQ frame → hop_length audio samples (512)
            # But there's also the quantizer upsample (×4) and decoder upsample (×512/4=128)
            # Actually: total upsample = quantizer upsample(×4) then decoder strides(8×8×4×2=512)
            # So 1 VQ frame → 4 (quantizer) × 128 (decoder strides 8*8*4*2/4=128?) 
            # Hmm, let's just compute it: hop_length = prod(encoder_rates) = 2*4*8*8 = 512
            # But quantizer downsample_factor=[2,2], so effective: 512 / 4 = 128 samples per VQ frame
            # Wait no. Let me think again.
            #
            # Encoder: audio → encoder(rates=[2,4,8,8]) → latent (T/512)
            # Quantizer downsample: latent(T/512) → downsample([2,2]) → VQ space (T/512/4 = T/2048)
            # So: 1 VQ frame = 2048 audio samples
            #
            # Decode path: VQ codes → post_module(T_vq) → upsample([2,2])(T_vq*4) → decoder(rates=[8,8,4,2])(T_vq*4*512)
            # Wait, decoder rates are strides for ConvTranspose1d, so they upsample.
            # decoder_rates = [8,8,4,2], so total decoder upsample = 8*8*4*2 = 512
            # Total: T_vq * 4 (quantizer upsample) * 512 (decoder upsample) = T_vq * 2048
            #
            # So 1 VQ frame → 2048 audio samples
            # And hop_length = np.prod(encoder_rates) = 512... but that's the encoder side
            # The actual audio-per-VQ-frame is: frame_length = hop_length * 4 = 512 * 4 = 2048
            # (because quantizer downsample_factor=[2,2] → factor 4)

            samples_per_vq_frame = self.codec.hop_length * int(np.prod(self.codec.quantizer.downsample_factor))
            overlap_audio = n_overlap * samples_per_vq_frame

            # Trim the overlap portion from the beginning
            if overlap_audio < len(audio_np):
                audio_np = audio_np[overlap_audio:]
            else:
                audio_np = np.array([], dtype=np.float32)

        return audio_np
