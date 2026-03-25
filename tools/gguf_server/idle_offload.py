"""
Idle VRAM offload: moves model/codec to CPU after inactivity timeout.

Public API:
  start_idle_offload_daemon(timeout_seconds)  — start background monitor
  stop_idle_offload_daemon()                  — stop monitor
  touch_last_request()                        — record request completion
  ensure_on_gpu()                             — restore before generation
                                                (caller must hold state.lock)
"""

import gc
import threading
import time

import torch
from loguru import logger

from .state import state
from .tts_engine import invalidate_decode_cache


# ── Public API ──


def touch_last_request():
    """Record that a request just completed (resets idle timer)."""
    state._last_request_time = time.monotonic()


def start_idle_offload_daemon(timeout_seconds: int):
    """Start background thread that monitors idle time."""
    if timeout_seconds <= 0:
        return

    # Stop existing daemon if running
    stop_idle_offload_daemon()

    state._idle_offload_timeout = timeout_seconds
    state._last_request_time = time.monotonic()
    stop_event = threading.Event()
    state._offload_stop_event = stop_event

    def _poll_loop():
        poll_interval = min(30.0, timeout_seconds / 2.0)
        while not stop_event.wait(timeout=poll_interval):
            if state._offloaded or not state.ready or state.model is None:
                continue

            idle_seconds = time.monotonic() - state._last_request_time
            if idle_seconds >= state._idle_offload_timeout:
                _do_offload()

    thread = threading.Thread(target=_poll_loop, name="idle-offload", daemon=True)
    thread.start()
    state._offload_thread = thread
    logger.info(f"Idle offload daemon started (timeout={timeout_seconds}s)")


def stop_idle_offload_daemon():
    """Stop the background polling thread."""
    if state._offload_stop_event is not None:
        state._offload_stop_event.set()
    if state._offload_thread is not None:
        state._offload_thread.join(timeout=5)
        state._offload_thread = None
        state._offload_stop_event = None
    state._idle_offload_timeout = 0


def ensure_on_gpu():
    """Restore model to GPU if it was offloaded.

    MUST be called while holding state.lock.
    Safe to call when not offloaded (returns immediately).
    """
    if not state._offloaded:
        return

    logger.info("Restoring model from CPU to GPU for incoming request...")
    t0 = time.perf_counter()

    model = state.model
    device = state.device

    # Move model to GPU (same pattern as codec_manager.restore_after_codec)
    model.to(device)
    for _name, module in model.named_modules():
        if hasattr(module, "qparam") and hasattr(module.qparam, "data"):
            if not module.qparam.data.is_cuda:
                module.qparam.data = module.qparam.data.to(device)
        for attr_name in ("weight", "qweight", "data"):
            attr = getattr(module, attr_name, None)
            if isinstance(attr, torch.Tensor) and not attr.is_cuda:
                setattr(module, attr_name, attr.to(device))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Re-setup KV caches
    model._cache_setup_done = False
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=torch.float16,
        )
    model._cache_setup_done = True

    # Restore codec to GPU if it was previously GPU-resident
    if state._codec_was_gpu_resident and state.codec is not None:
        state.codec.to(device="cuda", dtype=torch.float16)
        state.codec.eval()
        state.codec_gpu_resident = True

    # Re-enable custom_op mode for torch.compile compatibility
    try:
        from fish_speech.gguf.dequant import GGUFLinear

        for _name, module in model.named_modules():
            if isinstance(module, GGUFLinear):
                module._use_custom_op = True
    except Exception:
        pass

    state._offloaded = False

    dt = time.perf_counter() - t0
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    logger.info(f"Model restored to GPU in {dt:.1f}s (VRAM: {vram:.2f} GB)")


# ── Internal ──


def _do_offload():
    """Move model + codec to CPU, free VRAM.  Acquires lock internally."""
    acquired = state.lock.acquire(timeout=1)
    if not acquired:
        return  # Request in progress, skip this cycle

    try:
        if state._offloaded or not state.ready or state.model is None:
            return

        logger.info("Idle timeout reached, offloading model to CPU...")
        t0 = time.perf_counter()
        model = state.model

        # Invalidate decode cache (CUDA Graph, streaming decoder, etc.)
        invalidate_decode_cache()

        # Move model to CPU (same pattern as codec_manager.codec_on_gpu)
        model.cpu()
        for _name, module in model.named_modules():
            if hasattr(module, "qparam") and hasattr(module.qparam, "data"):
                if module.qparam.data.is_cuda:
                    module.qparam.data = module.qparam.data.cpu()
            for attr_name in ("weight", "qweight", "data"):
                attr = getattr(module, attr_name, None)
                if isinstance(attr, torch.Tensor) and attr.is_cuda:
                    setattr(module, attr_name, attr.cpu())

        model._cache_setup_done = False

        # Move codec to CPU if it was GPU-resident
        state._codec_was_gpu_resident = state.codec_gpu_resident
        if state.codec_gpu_resident and state.codec is not None:
            state.codec.to(device="cpu")
            state.codec_gpu_resident = False

        torch.cuda.empty_cache()
        gc.collect()

        state._offloaded = True
        vram = torch.cuda.memory_allocated() / 1e9
        dt = time.perf_counter() - t0
        logger.info(f"Model offloaded to CPU in {dt:.1f}s (VRAM: {vram:.2f} GB)")
    finally:
        state.lock.release()
