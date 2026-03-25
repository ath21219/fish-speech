"""
Fish Speech GGUF API Server
CUDA-accelerated GGUF inference with OpenAI-compatible TTS endpoint.

Usage:
  python tools/api_server_gguf.py \
    --model-name s2-pro-q3_k_s \
    --listen 0.0.0.0:7820
"""

import argparse
import sys
from pathlib import Path

import uvicorn

# Ensure project root is on sys.path before any fish_speech imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ── Package imports ──
from gguf_server.logging_setup import print_startup_banner, setup_logging
from gguf_server.model_loader import load_models
from gguf_server.state import app

# Import route modules so their routers get registered
from gguf_server import (
    routes_config,
    routes_health,
    routes_models,
    routes_tts,
    routes_voices,
)

# Register all routers
app.include_router(routes_health.router)
app.include_router(routes_tts.router)
app.include_router(routes_voices.router)
app.include_router(routes_models.router)
app.include_router(routes_config.router)


def parse_args():
    parser = argparse.ArgumentParser(description="Fish Speech GGUF API Server")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model to load on startup (omit to restore previous state)",
    )
    parser.add_argument(
        "--codec-path",
        type=str,
        default=None,
        help="Path to codec.pth (optional: uses GGUF-embedded codec if omitted)",
    )
    parser.add_argument(
        "--listen",
        type=str,
        default="0.0.0.0:7820",
        help="Host:port to listen on",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max sequence length for KV cache allocation",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=0,
        help="Max input text length (0 = unlimited)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for authentication",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers (keep 1 for single GPU)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--no-docker-log",
        action="store_true",
        help="Include timestamps in logs (for local dev)",
    )
    parser.add_argument(
        "--idle-offload-timeout",
        type=int,
        default=0,
        help="Seconds of inactivity before offloading model to CPU (0 = disabled)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Logging (must be first) ──
    setup_logging(
        log_level=args.log_level,
        docker_mode=not args.no_docker_log,
    )

    # ── Startup banner ──
    codec_source = "External" if args.codec_path else "Embedded"
    print_startup_banner(args, codec_source)

    from loguru import logger

    # ── Store args globally for dynamic model loading ──
    import gguf_server.state as _state_mod

    _state_mod._server_args = args

    # ── Restore saved state: CLI args take priority ──
    from gguf_server.state import load_server_state

    saved = load_server_state()

    model_to_load = args.model_name
    if model_to_load is None and saved:
        model_to_load = saved.get("active_model")
        if model_to_load:
            logger.info(f"Restoring previous model: {model_to_load}")

    if not model_to_load:
        logger.info("No saved state found — starting without model")

    # ── Load models ──
    if model_to_load:
        load_models(args, name=model_to_load)

    # ── Idle offload daemon: CLI arg > saved state > disabled ──
    idle_timeout = args.idle_offload_timeout
    if idle_timeout == 0 and saved:
        idle_timeout = saved.get("idle_offload_timeout", 0)
        if idle_timeout > 0:
            logger.info(f"Restoring idle offload timeout: {idle_timeout}s")

    if idle_timeout > 0:
        from gguf_server.idle_offload import start_idle_offload_daemon

        start_idle_offload_daemon(idle_timeout)

    # ── Start server ──
    host, port = args.listen.rsplit(":", 1)
    logger.info(f"Starting server on {args.listen}")

    uvicorn.run(
        app,
        host=host,
        port=int(port),
        log_level=args.log_level.lower(),
        access_log=True,  # Keep access log but health checks are filtered
        log_config=None,  # Disable uvicorn's default log config (we use loguru)
    )