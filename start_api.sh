#!/usr/bin/env bash
set -euo pipefail

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-flex_gemm}"
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-xformers}"
export TRELLIS_LOW_VRAM="${TRELLIS_LOW_VRAM:-0}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"

uvicorn api_service.main:app --host "$HOST" --port "$PORT"
