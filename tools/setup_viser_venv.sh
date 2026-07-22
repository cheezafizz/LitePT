#!/usr/bin/env bash
# Create a lightweight, standalone venv for the viser viewer (tools/viser_insseg_viewer.py).
# Kept separate from the litept training env on purpose: the viewer only reads the .npz
# results and needs nothing from the repo / torch / spconv -- just viser + numpy.
#
#   bash tools/setup_viser_venv.sh
#   .venv-viser/bin/python tools/viser_insseg_viewer.py --port 8080
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV=".venv-viser"
PY_BIN="$(command -v python3.11 || command -v python3)"
echo "[setup] base python: $PY_BIN ($("$PY_BIN" --version 2>&1))"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PY_BIN" -m venv "$VENV"
  echo "[setup] created $VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/pip" install "viser>=0.2.7" numpy

echo "[setup] viser version:"
"$VENV/bin/python" -c "import viser, numpy; print('  viser', viser.__version__, '| numpy', numpy.__version__)"
echo "[setup] done. Run:"
echo "  $VENV/bin/python tools/viser_insseg_viewer.py --results-dir viz/vggt_insseg --port 8080"
