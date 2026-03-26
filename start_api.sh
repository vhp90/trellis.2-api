#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-flex_gemm}"
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-xformers}"
export TRELLIS_LOW_VRAM="${TRELLIS_LOW_VRAM:-0}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

bash "$ROOT_DIR/install_lightning.sh"

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

cd "$ROOT_DIR"
python -m uvicorn api_service.main:app --host "$HOST" --port "$PORT"
