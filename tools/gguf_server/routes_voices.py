"""Voice management CRUD endpoints."""

import json
import shutil
from pathlib import Path

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from .codec_manager import codec_on_gpu, restore_after_codec
from .schemas import CODEC_MAX_SECONDS
from .state import REFERENCES_DIR, state

router = APIRouter()


@router.post("/v1/voices")
async def create_voice(
    name: str = Form(...),
    transcription: str = Form(""),
    audio: UploadFile = File(...),
):
    """Register a reference voice."""
    if not name or not name.strip():
        raise HTTPException(400, "Voice name cannot be empty")
    name = name.strip()
    if not all(c.isalnum() or c in "-_ " for c in name):
        raise HTTPException(
            400,
            "Voice name can only contain alphanumeric, hyphen, underscore, space",
        )

    voice_dir = REFERENCES_DIR / name
    if voice_dir.exists():
        raise HTTPException(409, f"Voice '{name}' already exists")

    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(400, "Audio file too small or empty")

    ext = Path(audio.filename).suffix.lower() if audio.filename else ".wav"
    if ext not in (".wav", ".mp3", ".flac", ".ogg"):
        ext = ".wav"

    voice_dir.mkdir(parents=True, exist_ok=True)
    try:
        audio_path = voice_dir / f"audio{ext}"
        audio_path.write_bytes(audio_bytes)

        meta = {"name": name, "transcription": transcription}
        (voice_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if transcription:
            (voice_dir / "audio.txt").write_text(transcription, encoding="utf-8")

        # Pre-encode VQ codes
        if state.ready:
            try:
                import torchaudio

                logger.info(f"Pre-encoding voice '{name}'...")
                codec_ctx = codec_on_gpu()
                codec = codec_ctx.__enter__()

                with torch.no_grad():
                    wav, sr = torchaudio.load(str(audio_path))
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    wav = torchaudio.functional.resample(
                        wav.to(state.device), sr, codec.sample_rate
                    )[0]

                    # Truncate to codec's max frame limit to avoid
                    # Transformer buffer overflow (causal_mask/freqs_cis)
                    max_samples = int(CODEC_MAX_SECONDS * codec.sample_rate)
                    if len(wav) > max_samples:
                        original_dur = len(wav) / codec.sample_rate
                        wav = wav[:max_samples]
                        logger.warning(
                            f"Voice '{name}': audio truncated from "
                            f"{original_dur:.1f}s to {CODEC_MAX_SECONDS}s "
                            f"(codec frame limit)"
                        )

                    model_dtype = next(codec.parameters()).dtype
                    audios = wav[None, None].to(dtype=model_dtype)
                    audio_lengths = torch.tensor(
                        [len(wav)], device=state.device, dtype=torch.long
                    )
                    indices, feature_lengths = codec.encode(
                        audios, audio_lengths
                    )
                    codes = indices[0, :, : feature_lengths[0]]

                torch.save(codes.cpu(), voice_dir / "codes.pt")
                logger.info(
                    f"Voice '{name}': pre-encoded {codes.shape[1]} VQ frames"
                )

            except Exception as e:
                logger.warning(f"Voice '{name}': pre-encoding failed: {e}")
            finally:
                restore_after_codec()

        logger.info(
            f"Voice '{name}' registered: {audio_path} ({len(audio_bytes)} bytes)"
        )

        return JSONResponse(
            status_code=201,
            content={
                "name": name,
                "transcription": transcription,
                "audio_file": str(audio_path.name),
                "audio_size": len(audio_bytes),
                "message": f"Voice '{name}' created successfully",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
        raise HTTPException(500, f"Failed to save voice: {e}")


@router.get("/v1/voices")
async def list_voices():
    """List all registered reference voices."""
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    voices = []
    for d in sorted(REFERENCES_DIR.iterdir()):
        if not d.is_dir():
            continue

        meta_file = d / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        else:
            meta = {"name": d.name, "transcription": ""}

        audio_file = None
        for ext in (".wav", ".mp3", ".flac", ".ogg"):
            candidates = list(d.glob(f"*{ext}"))
            if candidates:
                audio_file = candidates[0].name
                break

        voices.append(
            {
                "name": d.name,
                "transcription": meta.get("transcription", ""),
                "audio_file": audio_file,
                "has_audio": audio_file is not None,
            }
        )

    return {"voices": voices, "total": len(voices)}


@router.get("/v1/voices/{name}")
async def get_voice(name: str):
    """Get details for a specific voice."""
    voice_dir = REFERENCES_DIR / name
    if not voice_dir.exists():
        raise HTTPException(404, f"Voice '{name}' not found")

    meta_file = voice_dir / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    else:
        meta = {"name": name, "transcription": ""}

    audio_file = None
    audio_size = 0
    for ext in (".wav", ".mp3", ".flac", ".ogg"):
        candidates = list(voice_dir.glob(f"*{ext}"))
        if candidates:
            audio_file = candidates[0].name
            audio_size = candidates[0].stat().st_size
            break

    return {
        "name": name,
        "transcription": meta.get("transcription", ""),
        "audio_file": audio_file,
        "audio_size": audio_size,
    }


@router.delete("/v1/voices/{name}")
async def delete_voice(name: str):
    """Delete a registered voice."""
    voice_dir = REFERENCES_DIR / name
    if not voice_dir.exists():
        raise HTTPException(404, f"Voice '{name}' not found")

    shutil.rmtree(voice_dir)
    logger.info(f"Voice '{name}' deleted")
    return {"message": f"Voice '{name}' deleted successfully"}
