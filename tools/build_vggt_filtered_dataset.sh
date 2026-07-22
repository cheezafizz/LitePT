#!/usr/bin/env bash
# Build the `scannet-vggt-default-filtered` dataset: the default filtered VGGT pipeline.
#
# Stage 1 converts every VGGT export under $SRC into a LitePT scene with
# tools/vggt_to_scene.py --valid-mask filtered (applies filtered_valid_mask; defaults
# otherwise: 2 mm voxel, k=16 normals, no extra confidence floor).
# Stage 2 clamps the whole split to the trained working-volume AABB with
# tools/crop_vggt_dataset.py (in-place, idempotent) — the mask already keeps ~98% inside
# the box, so this only trims boundary outliers.
#
#   bash tools/build_vggt_filtered_dataset.sh
#
# Resumable: a scene whose coord.npy already exists is skipped; the crop is idempotent.
# Output: data/scannet-vggt-default-filtered/test/<loc>_<frame>/{coord,color,normal}.npy
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/fai/miniconda3/envs/litept/bin/python
SRC=/home/fai/workspace/jhp/dataset/val_sample_output
# OUT / CONF_PCTILE are env-overridable (defaults reproduce the original build):
#   OUT=data/scannet-vggt-default-filtered-conf25 CONF_PCTILE=25 bash tools/build_vggt_filtered_dataset.sh
OUT=${OUT:-data/scannet-vggt-default-filtered}
CONF_PCTILE=${CONF_PCTILE:-0}   # >0 adds --conf-percentile (within the filtered-valid region)
SPLIT=test
CROP_MIN=-0.4,-0.5,-0.2
CROP_MAX=0.6,0.7,0.7

# extra vggt_to_scene.py args (confidence percentile floor, when requested)
CONF_ARGS=()
if awk "BEGIN{exit !($CONF_PCTILE > 0)}"; then
  CONF_ARGS=(--conf-percentile "$CONF_PCTILE")
fi

mapfile -t NPZ < <(find "$SRC" -mindepth 3 -maxdepth 3 -name data.npz | sort)
echo "[info] ${#NPZ[@]} source scenes -> $OUT/$SPLIT (conf-percentile=${CONF_PCTILE})"

n_ok=0; n_skip=0; n_fail=0
for npz in "${NPZ[@]}"; do
  frame="$(basename "$(dirname "$npz")")"
  loc="$(basename "$(dirname "$(dirname "$npz")")")"
  scene="${loc}_${frame}"
  if [[ -f "$OUT/$SPLIT/$scene/coord.npy" ]]; then
    echo "[skip] $scene (exists)"; n_skip=$((n_skip + 1)); continue
  fi
  if "$PY" tools/vggt_to_scene.py \
      --input "$npz" \
      --output-root "$OUT" --split "$SPLIT" --scene "$scene" \
      --valid-mask filtered "${CONF_ARGS[@]}"; then
    echo "[ok]   $scene"; n_ok=$((n_ok + 1))
  else
    echo "[FAIL] $scene"; n_fail=$((n_fail + 1))
  fi
done
echo "[convert] ok=$n_ok skip=$n_skip fail=$n_fail"

echo "[crop] clamping $OUT/$SPLIT to AABB min=$CROP_MIN max=$CROP_MAX (in-place)"
# NOTE: --crop-min/max values begin with '-', so use the =form; passed as separate
# tokens argparse mistakes them for option flags.
"$PY" tools/crop_vggt_dataset.py \
    --data-root "$OUT" --splits "$SPLIT" \
    --crop-min="$CROP_MIN" --crop-max="$CROP_MAX"

total=$(find "$OUT/$SPLIT" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "[done] $total scenes under $OUT/$SPLIT"
