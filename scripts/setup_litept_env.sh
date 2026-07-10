#!/usr/bin/env bash
#
# setup_litept_env.sh — reproducibly recreate the conda `litept` environment.
#
# This is a hand-written installer (NOT a `pip freeze` replay) because the env mixes
# four install channels plus three CUDA extensions built from this repo's source:
#   - PyTorch cu128 index        : torch / torchvision / torchaudio
#   - PyG wheel index            : torch_scatter / torch_sparse / torch_cluster / torch_geometric
#   - special wheels             : spconv-cu128, flash-attn (source build)
#   - repo source (libs/)        : pointops, pointgroup_ops, pointrope
# A plain `pip install -r <freeze>` would hard-fail on the local packages and the
# +cu128 / +pt27cu128 local-version tags.
#
# Target hardware for the pinned versions below: CUDA 12.8 toolkit, RTX 5090 (sm_120).
#
# Usage:
#   bash scripts/setup_litept_env.sh                 # create/populate env named `litept`
#   ENV_NAME=litept_repro bash scripts/setup_litept_env.sh   # into a differently-named env
#
# Overridable via environment variables (defaults match the live env):
#   ENV_NAME, CUDA_HOME, TORCH_CUDA_ARCH_LIST
#
set -euo pipefail

# ----------------------------------------------------------------------------------
# Configuration — pinned to the exact versions in the live `litept` env (2026-07-10).
# ----------------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-litept}"
PY_VERSION="3.11.15"

TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
TORCHAUDIO_VERSION="2.7.1"
CUDA_TAG="cu128"                                   # PyTorch / PyG wheel channel
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

TORCH_SCATTER_VERSION="2.1.2"
TORCH_SPARSE_VERSION="0.6.18"
TORCH_CLUSTER_VERSION="1.6.3"
TORCH_GEOMETRIC_VERSION="2.5.3"

# spconv-cu128 / cumm-cu128 are NOT on PyPI — the only cu128 spconv wheels are the
# third-party builds published by github.com/rathaROG. These URLs are cp311 / linux
# x86_64 (matching the live env); change them if you target a different py/platform.
SPCONV_VERSION="2.4.1"
CUMM_VERSION="0.9.1"
CUMM_WHEEL_URL="https://github.com/rathaROG/cumm-gpu/releases/download/v${CUMM_VERSION}/cumm_cu128-${CUMM_VERSION}-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
SPCONV_WHEEL_URL="https://github.com/rathaROG/spconv-gpu/releases/download/v${SPCONV_VERSION}/spconv_cu128-${SPCONV_VERSION}-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
NVIDIA_ARCH_VERSION="7.1.0"                        # spconv dep, not a torch CUDA runtime lib
FLASH_ATTN_VERSION="2.8.0.post2"
SPARSEHASH_VERSION="2.0.3"                         # google-sparsehash, bioconda

# Build toolchain for the source-built CUDA extensions.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"   # RTX 5090 = sm_120

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${REPO_ROOT}/scripts/requirements-litept-lock.txt"

# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
log()  { printf '\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# Run a command inside the target conda env with live (unbuffered) output.
crun() { conda run --no-capture-output -n "$ENV_NAME" "$@"; }
# pip inside the target env.
cpip() { crun python -m pip "$@"; }

# ----------------------------------------------------------------------------------
# Step 1 — Preflight checks (warn, do not silently proceed on a broken host).
# ----------------------------------------------------------------------------------
log "Step 1/11: Preflight checks"

command -v conda >/dev/null 2>&1 || die "conda not found on PATH. Install miniconda/anaconda first."
echo "conda    : $(conda --version)"

[ -f "$LOCK_FILE" ] || die "Lock file not found: $LOCK_FILE"

if command -v nvcc >/dev/null 2>&1; then
    NVCC_VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
    echo "nvcc     : ${NVCC_VER}"
    case "$CUDA_TAG" in
        cu128) [ "$NVCC_VER" = "12.8" ] || warn "nvcc is ${NVCC_VER} but wheels target cu12.8 — the source-built libs link against the system CUDA, so a mismatch may fail." ;;
    esac
else
    warn "nvcc not on PATH. The custom CUDA extensions (steps 10) need the CUDA 12.8 toolkit."
fi

[ -d "$CUDA_HOME" ] || warn "CUDA_HOME does not exist: $CUDA_HOME (needed to build libs/). Override with CUDA_HOME=..."
echo "CUDA_HOME: $CUDA_HOME"
echo "arch list: $TORCH_CUDA_ARCH_LIST"

command -v gcc >/dev/null 2>&1 && echo "gcc      : $(gcc --version | head -1)" || warn "gcc not found — required to compile the extensions."

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu      : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not found — no GPU visible. Extensions will build but torch.cuda will be unavailable at runtime."
fi

# ----------------------------------------------------------------------------------
# Step 2 — Create the conda env.
# ----------------------------------------------------------------------------------
log "Step 2/11: Create conda env '${ENV_NAME}' (python ${PY_VERSION})"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    warn "conda env '${ENV_NAME}' already exists — reusing it (pip steps are idempotent)."
else
    conda create -y -n "$ENV_NAME" "python=${PY_VERSION}"
fi

# ----------------------------------------------------------------------------------
# Step 3 — conda channel dependency (must precede the pointgroup_ops build).
# ----------------------------------------------------------------------------------
log "Step 3/11: Install google-sparsehash ${SPARSEHASH_VERSION} (bioconda) for pointgroup_ops"
conda install -y -n "$ENV_NAME" -c bioconda "google-sparsehash=${SPARSEHASH_VERSION}"

# ----------------------------------------------------------------------------------
# Step 4 — Sanity: report the resolved interpreter.
# ----------------------------------------------------------------------------------
log "Step 4/11: Env interpreter"
crun python -c "import sys; print(sys.version); print(sys.executable)"
cpip install --upgrade "pip==26.0.1" >/dev/null 2>&1 || true

# ----------------------------------------------------------------------------------
# Step 5 — PyTorch stack (cu128 index).
# ----------------------------------------------------------------------------------
log "Step 5/11: PyTorch ${TORCH_VERSION} stack (${CUDA_TAG})"
cpip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "$TORCH_INDEX"

# ----------------------------------------------------------------------------------
# Step 6 — PyTorch Geometric stack (data.pyg.org wheel index for the compiled ops).
#          The lock file is passed as a pip CONSTRAINTS file (-c) here so transitive
#          deps resolve to their exact pinned versions. Without it, torch_geometric
#          pulls scikit-learn unpinned -> latest (1.9.0) -> which drags in narwhals;
#          step 9 then downgrades scikit-learn to 1.8.0 but leaves narwhals orphaned.
# ----------------------------------------------------------------------------------
log "Step 6/11: PyG stack (torch_scatter/sparse/cluster + torch_geometric)"
cpip install \
    "torch_scatter==${TORCH_SCATTER_VERSION}" \
    "torch_sparse==${TORCH_SPARSE_VERSION}" \
    "torch_cluster==${TORCH_CLUSTER_VERSION}" \
    -f "$PYG_FIND_LINKS" -c "$LOCK_FILE"
cpip install "torch_geometric==${TORCH_GEOMETRIC_VERSION}" -c "$LOCK_FILE"

# ----------------------------------------------------------------------------------
# Step 7 — spconv (SparseUNet backbone) + its cumm backend, from the rathaROG cu128
#          wheels (not on PyPI). nvidia-arch is pinned here so it isn't left to float.
#          The wheels are given as --find-links and resolved by name==version (rather
#          than passing the URLs as positional args) so `pip freeze` records them as
#          plain `spconv-cu128==2.4.1` / `cumm-cu128==0.9.1`, matching the source env
#          exactly instead of an embedded direct-URL string. Other deps (pccm, ccimport,
#          pybind11, fire, numpy) come from PyPI, constrained by the lock file.
# ----------------------------------------------------------------------------------
log "Step 7/11: spconv-cu128 2.4.1 + cumm-cu128 0.9.1 (rathaROG wheels)"
cpip install -c "$LOCK_FILE" \
    "nvidia-arch==${NVIDIA_ARCH_VERSION}" \
    "cumm-cu128==${CUMM_VERSION}" \
    "spconv-cu128==${SPCONV_VERSION}" \
    -f "$CUMM_WHEEL_URL" -f "$SPCONV_WHEEL_URL"

# ----------------------------------------------------------------------------------
# Step 8 — flash-attention (source build; needs torch + ninja + packaging present).
#          PyPI ships only an sdist for flash-attn, so this compiles from source and
#          can take 10-30+ minutes. --no-build-isolation reuses the env's torch.
# ----------------------------------------------------------------------------------
log "Step 8/11: flash-attn ${FLASH_ATTN_VERSION} (source build — this is slow)"
cpip install -c "$LOCK_FILE" "ninja==1.13.0" "packaging==26.0" "psutil==7.2.2"   # flash-attn build deps
cpip install -c "$LOCK_FILE" "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

# ----------------------------------------------------------------------------------
# Step 9 — Remaining pinned pure-Python / transitive deps.
# ----------------------------------------------------------------------------------
log "Step 9/11: Remaining pinned dependencies (${LOCK_FILE##*/})"
cpip install -r "$LOCK_FILE"

# ----------------------------------------------------------------------------------
# Step 10 — Build the three custom CUDA extensions from repo source.
#           Order: pointops, then pointgroup_ops (needs sparsehash from step 3),
#           then pointrope (its setup.py already targets sm_120).
# ----------------------------------------------------------------------------------
log "Step 10/11: Build custom CUDA extensions (CUDA_HOME=$CUDA_HOME, arch=$TORCH_CUDA_ARCH_LIST)"
for lib in pointops pointgroup_ops pointrope; do
    log "  building libs/${lib}"
    crun bash -c "cd '${REPO_ROOT}/libs/${lib}' && CUDA_HOME='${CUDA_HOME}' TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' python setup.py install"
done
# NOTE: for a non-RTX-5090 GPU, edit `all_cuda_archs` in libs/pointrope/setup.py and set
#       TORCH_CUDA_ARCH_LIST accordingly (e.g. 8.6 for RTX 3090, 9.0 for H100).

# ----------------------------------------------------------------------------------
# Step 11 — Verify the environment imports and sees the GPU.
# ----------------------------------------------------------------------------------
log "Step 11/11: Verifying environment"
crun python - <<'PY'
import importlib
import torch

print(f"torch                {torch.__version__}")
print(f"torch.cuda available {torch.cuda.is_available()}")
print(f"torch cuda build     {torch.version.cuda}")
print(f"arch list            {torch.cuda.get_arch_list()}")
if torch.cuda.is_available():
    print(f"device               {torch.cuda.get_device_name(0)}")

import spconv, spconv.pytorch          # noqa: F401
print(f"spconv               {spconv.__version__}")

import flash_attn                       # noqa: F401
print(f"flash_attn           {flash_attn.__version__}")

import torch_geometric                  # noqa: F401
import torch_scatter, torch_sparse, torch_cluster  # noqa: F401
print(f"torch_geometric      {torch_geometric.__version__}")

for m in ("pointops", "pointgroup_ops", "pointrope"):
    importlib.import_module(m)
    print(f"{m:20} OK")

print("\n\033[1;32mAll imports OK — litept environment is ready.\033[0m")
PY

log "Done. Activate with:  conda activate ${ENV_NAME}"
