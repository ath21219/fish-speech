#!/bin/bash
set -euo pipefail

log() { echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" >&2; }

# Defaults (overridable via environment variables)
GGUF_MODEL_PATH="${GGUF_MODEL_PATH:-/app/models/gguf/s2-pro-q6_k.gguf}"
CODEC_PATH="${CODEC_PATH:-/app/checkpoints/s2-pro/codec.pth}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-7820}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MAX_TEXT_LENGTH="${MAX_TEXT_LENGTH:-0}"

# Validate required files
if [ ! -f "${GGUF_MODEL_PATH}" ]; then
    log "ERROR: GGUF model not found at ${GGUF_MODEL_PATH}"
    log "Mount your model: -v /path/to/models/gguf:/app/models/gguf"
    exit 1
fi

if [ ! -f "${CODEC_PATH}" ]; then
    log "ERROR: Codec not found at ${CODEC_PATH}"
    log "Mount your codec: -v /path/to/checkpoints/s2-pro:/app/checkpoints/s2-pro"
    exit 1
fi

log "Fish Speech GGUF API Server"
log "  Model:    ${GGUF_MODEL_PATH}"
log "  Codec:    ${CODEC_PATH}"
log "  Listen:   ${API_HOST}:${API_PORT}"
log "  SeqLen:   ${MAX_SEQ_LEN}"

# Build command
CMD="python3 tools/api_server_gguf.py \
    --gguf-path ${GGUF_MODEL_PATH} \
    --codec-path ${CODEC_PATH} \
    --listen ${API_HOST}:${API_PORT} \
    --max-seq-len ${MAX_SEQ_LEN}"

if [ "${MAX_TEXT_LENGTH}" -gt 0 ] 2>/dev/null; then
    CMD="${CMD} --max-text-length ${MAX_TEXT_LENGTH}"
fi

if [ -n "${API_KEY:-}" ]; then
    CMD="${CMD} --api-key ${API_KEY}"
fi

log "Executing: ${CMD}"
exec ${CMD}
