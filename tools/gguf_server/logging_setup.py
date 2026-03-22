"""
Unified logging configuration.

Goals:
  1. Single timestamp (loguru only, no Docker double-timestamp)
  2. Consistent format across loguru / uvicorn / warnings
  3. Health-check noise suppression
"""

import logging
import sys
import warnings

import torch
from loguru import logger


def setup_logging(log_level: str = "INFO", docker_mode: bool = True):
    """
    Configure loguru as the sole logging backend.

    Args:
        log_level: "DEBUG", "INFO", "WARNING", etc.
        docker_mode: If True, omit timestamp (Docker adds its own).
    """
    logger.remove()

    if docker_mode:
        fmt = (
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    else:
        fmt = (
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    logger.add(
        sys.stderr,
        format=fmt,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    # Route Python warnings through loguru
    def _warning_handler(message, category, filename, lineno, file=None, line=None):
        logger.opt(depth=2).warning(
            f"{category.__name__}: {message} ({filename}:{lineno})"
        )

    warnings.showwarning = _warning_handler

    # Route stdlib logging (uvicorn etc.) through loguru
    class _LoguruHandler(logging.Handler):
        _LEVEL_MAP = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL",
        }

        def emit(self, record):
            msg = record.getMessage()
            if "/v1/health" in msg:
                return
            level = self._LEVEL_MAP.get(record.levelno, "INFO")
            logger.opt(depth=6, exception=record.exc_info).log(level, msg)

    loguru_handler = _LoguruHandler()
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers.clear()
        stdlib_logger.addHandler(loguru_handler)
        stdlib_logger.setLevel(getattr(logging, log_level))
        stdlib_logger.propagate = False

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(loguru_handler)
    root_logger.setLevel(logging.WARNING)

    logger.info("Logging initialized")


def print_startup_banner(args, codec_source: str = "Embedded"):
    """Print a clean startup banner."""
    logger.info("=" * 50)
    logger.info("Fish Speech GGUF API Server")
    logger.info("=" * 50)
    logger.info(f"  Model:       {args.model_name or '(none)'}")
    logger.info(f"  Codec:       {codec_source}")
    logger.info(f"  Listen:      {args.listen}")
    logger.info(f"  Max seq len: {args.max_seq_len}")
    logger.info(f"  Device:      {args.device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  GPU:         {gpu_name} ({vram_total:.1f} GB)")
    logger.info("=" * 50)
