"""TTS endpoint with streaming support."""

import asyncio
import io
import struct
import threading
import time

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from loguru import logger

from .schemas import AMPLITUDE, TTSRequest
from .state import state
from .tts_engine import generate_speech, generate_speech_streaming

router = APIRouter()


@router.post("/v1/tts")
async def tts(req: TTSRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if req.streaming:
        return StreamingResponse(
            _stream_tts(req),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        with state.lock:
            try:
                t0 = time.perf_counter()
                audio_np = generate_speech(req)
                t_total = time.perf_counter() - t0
                duration = len(audio_np) / state.sample_rate
                logger.info(
                    f"TTS complete: {duration:.1f}s audio in {t_total:.1f}s "
                    f"(RTF={t_total / duration:.2f})"
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.opt(exception=True).error("Generation failed: {}", str(e))
                raise HTTPException(status_code=500, detail="Generation failed")

        buffer = io.BytesIO()
        try:
            sf.write(buffer, audio_np, state.sample_rate, format=req.format)
        except Exception:
            sf.write(buffer, audio_np, state.sample_rate, format="wav")
            req.format = "wav"

        content_type_map = {
            "wav": "audio/wav",
            "flac": "audio/flac",
            "mp3": "audio/mpeg",
        }
        return Response(
            content=buffer.getvalue(),
            media_type=content_type_map.get(req.format, "audio/wav"),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.format}",
                "X-Audio-Duration": f"{duration:.2f}",
                "X-Generation-Time": f"{t_total:.2f}",
            },
        )


async def _stream_tts(req: TTSRequest):
    """Async generator for streaming TTS response."""
    loop = asyncio.get_event_loop()
    sr = state.sample_rate
    header_sent = False
    total_samples = 0

    result_queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    def _worker():
        try:
            with state.lock:
                for event in generate_speech_streaming(req):
                    loop.call_soon_threadsafe(result_queue.put_nowait, event)
            loop.call_soon_threadsafe(result_queue.put_nowait, None)
        except Exception as e:
            loop.call_soon_threadsafe(
                result_queue.put_nowait, {"type": "error", "error": str(e)}
            )

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()

    try:
        while True:
            event = await result_queue.get()
            if event is None:
                break
            if event.get("type") == "error":
                raise HTTPException(status_code=500, detail=event["error"])
            if event["type"] == "audio":
                audio_np = event["data"]

                if not header_sent:
                    header = _make_wav_header(sr, 0x7FFFFFFF)
                    yield header
                    header_sent = True

                audio_int16 = (
                    (audio_np * AMPLITUDE)
                    .clip(-AMPLITUDE, AMPLITUDE - 1)
                    .astype(np.int16)
                )
                yield audio_int16.tobytes()
                total_samples += len(audio_int16)

    except asyncio.CancelledError:
        stop_event.set()
    finally:
        worker_thread.join(timeout=5)

    if not header_sent:
        raise HTTPException(status_code=400, detail="No audio generated")

    logger.info(
        f"Streamed {total_samples} samples "
        f"({total_samples / sr:.1f}s) in {total_samples // sr + 1} chunks"
    )


def _make_wav_header(sample_rate: int, data_size: int) -> bytes:
    """Create a minimal WAV header for 16-bit mono PCM."""
    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
