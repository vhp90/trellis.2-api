#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
STATE_DIR="${BUILD_STATE_DIR:-$ROOT_DIR/.build_state}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch torchvision}"
XFORMERS_PACKAGE="${XFORMERS_PACKAGE:-xformers>=0.0.28}"
NVDIFFRAST_PACKAGE="${NVDIFFRAST_PACKAGE:-git+https://github.com/NVlabs/nvdiffrast.git}"
CUMESH_PACKAGE="${CUMESH_PACKAGE:-git+https://github.com/JeffreyXiang/CuMesh.git}"
FLEX_GEMM_PACKAGE="${FLEX_GEMM_PACKAGE:-git+https://github.com/JeffreyXiang/FlexGEMM.git}"

mkdir -p "$STATE_DIR" "$HF_HOME"

hash_file() {
  local path="$1"
  python - "$path" <<'PY'
from pathlib import Path
import hashlib
import sys

path = Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
}

hash_paths() {
  python - "$@" <<'PY'
from pathlib import Path
import hashlib
import sys

h = hashlib.sha256()
for raw in sys.argv[1:]:
    path = Path(raw)
    if path.is_file():
        h.update(path.read_bytes())
print(h.hexdigest())
PY
}

create_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

activate_venv() {
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

write_marker() {
  local name="$1"
  local version="$2"
  printf "%s" "$version" > "$STATE_DIR/$name.done"
}

read_marker() {
  local name="$1"
  local marker="$STATE_DIR/$name.done"
  if [ -f "$marker" ]; then
    cat "$marker"
  fi
}

run_step() {
  local name="$1"
  local version="$2"
  shift 2
  local current
  current="$(read_marker "$name" || true)"
  if [ "$current" = "$version" ]; then
    echo "[build] Skipping $name ($version)"
    return 0
  fi
  echo "[build] Running $name ($version)"
  "$@"
  write_marker "$name" "$version"
}

has_import() {
  local code="$1"
  python - <<PY >/dev/null 2>&1
$code
PY
}

create_venv
activate_venv

REQ_HASH="$(hash_file "$ROOT_DIR/requirements.txt")"
OVOXEL_HASH="$(hash_paths "$ROOT_DIR/o-voxel/pyproject.toml" "$ROOT_DIR/o-voxel/setup.py")"

run_step "pip-bootstrap" "v2" python -m pip install --upgrade pip setuptools wheel

if has_import "import torch, torchvision"; then
  echo "[build] torch/torchvision already available"
else
  run_step "torch" "${TORCH_INDEX_URL}|${TORCH_PACKAGES}" \
    python -m pip install --index-url "$TORCH_INDEX_URL" $TORCH_PACKAGES
fi

if has_import "import fastapi, uvicorn, huggingface_hub, transformers, safetensors, PIL, cv2, easydict, tqdm, trimesh, plyfile, imageio, zstandard"; then
  echo "[build] Python API dependencies already available"
else
  run_step "python-deps" "$REQ_HASH" python -m pip install -r "$ROOT_DIR/requirements.txt"
fi

if has_import "import xformers"; then
  echo "[build] xformers already available"
else
  run_step "xformers" "$XFORMERS_PACKAGE" python -m pip install "$XFORMERS_PACKAGE"
fi

if has_import "import nvdiffrast.torch"; then
  echo "[build] nvdiffrast already available"
else
  run_step "nvdiffrast" "$NVDIFFRAST_PACKAGE" python -m pip install "$NVDIFFRAST_PACKAGE"
fi

if has_import "import cumesh"; then
  echo "[build] cumesh already available"
else
  run_step "cumesh" "$CUMESH_PACKAGE" python -m pip install "$CUMESH_PACKAGE"
fi

if has_import "import flex_gemm"; then
  echo "[build] flex_gemm already available"
else
  run_step "flex-gemm" "$FLEX_GEMM_PACKAGE" python -m pip install "$FLEX_GEMM_PACKAGE"
fi

if has_import "import o_voxel"; then
  echo "[build] o_voxel already available"
else
  run_step "o-voxel" "$OVOXEL_HASH" python -m pip install "$ROOT_DIR/o-voxel"
fi

python - <<'PY'
import importlib

modules = [
    "fastapi",
    "uvicorn",
    "torch",
    "torchvision",
    "xformers",
    "nvdiffrast.torch",
    "cumesh",
    "flex_gemm",
    "o_voxel",
    "trellis2",
]

for name in modules:
    importlib.import_module(name)

print("[build] Runtime import verification passed")
PY
