"""Model management CRUD + load/unload endpoints."""

import gc
import json
import shutil
from pathlib import Path

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from .model_loader import load_models
from .state import MODELS_DIR, state

router = APIRouter()


@router.post("/v1/models")
async def create_model(
    name: str = Form(...),
    config: str = Form(""),
    tokenizer: str = Form(""),
    model: UploadFile = File(...),
):
    """Register a model."""
    if not name or not name.strip():
        raise HTTPException(400, "Model name cannot be empty")
    name = name.strip()
    if not all(c.isalnum() or c in "-_ " for c in name):
        raise HTTPException(
            400,
            "Model name can only contain alphanumeric, hyphen, underscore, space",
        )

    model_dir = MODELS_DIR / name
    if model_dir.exists():
        raise HTTPException(409, f"Model '{name}' already exists")

    model_bytes = await model.read()
    if len(model_bytes) < 1000:
        raise HTTPException(400, "Model file too small or empty")

    ext = Path(model.filename).suffix.lower() if model.filename else ".safetensors"
    if ext not in (".safetensors", ".gguf"):
        ext = ".safetensors"

    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        model_path = model_dir / f"model{ext}"
        model_path.write_bytes(model_bytes)

        (model_dir / "config.json").write_text(config)
        (model_dir / "tokenizer.json").write_text(tokenizer)

        logger.info(
            f"Model '{name}' registered: {model_path} ({len(model_bytes)} bytes)"
        )

        return JSONResponse(
            status_code=201,
            content={
                "name": name,
                "model_file": str(model_path.name),
                "model_size": len(model_bytes),
                "message": f"Model '{name}' created successfully",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        raise HTTPException(500, f"Failed to save model: {e}")


@router.get("/v1/models")
async def list_models():
    models = []
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir():
                models.append({"id": d.name, "object": "model"})
    return {"data": models}


@router.get("/v1/models/{name}")
async def get_model(name: str):
    d = MODELS_DIR / name
    meta = {}
    if d.is_dir():
        meta_file = d / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "id": name,
            "object": "model",
            "owned_by": "fishaudio",
            "meta": meta,
        }
    elif (MODELS_DIR / f"{name}.gguf").exists():
        return {
            "id": name,
            "object": "model",
            "owned_by": "fishaudio",
            "meta": {},
        }
    raise HTTPException(404, f"Model '{name}' not found")


@router.post("/v1/models/{name}/load")
async def load_model_endpoint(name: str):
    # `args` is injected by the entrypoint via router state
    from .state import _server_args

    try:
        load_models(_server_args, name)
        return {
            "message": f"Model '{name}' loaded successfully",
            "vram_gb": (
                round(torch.cuda.memory_allocated() / 1e9, 2)
                if torch.cuda.is_available()
                else 0
            ),
        }
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{name}' not found")
    except Exception as e:
        logger.exception("Failed to load model")
        raise HTTPException(500, f"Failed to load model: {str(e)}")


@router.post("/v1/models/{name}/unload")
async def unload_model_endpoint(name: str):
    # BUG FIX: original code used state.active_name (nonexistent)
    if state.active_model_name != name:
        raise HTTPException(400, f"Model '{name}' is not currently loaded")

    logger.info("Unloading current model...")
    state.ready = False
    state.active_model_name = None
    if state.model is not None:
        del state.model
        del state.codec
        state.model = None
        state.codec = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"message": f"Model '{name}' unloaded successfully"}


@router.delete("/v1/models/{name}")
async def delete_model(name: str):
    d = MODELS_DIR / name
    if d.is_dir():
        # BUG FIX: original code used state.active_name (nonexistent)
        if state.active_model_name == name:
            await unload_model_endpoint(name)
        shutil.rmtree(d)
        return {"message": f"Model '{name}' deleted successfully"}
    elif (MODELS_DIR / f"{name}.gguf").exists():
        if state.active_model_name == name:
            await unload_model_endpoint(name)
        (MODELS_DIR / f"{name}.gguf").unlink()
        return {"message": f"Model '{name}' deleted successfully"}
    raise HTTPException(404, f"Model '{name}' not found")
