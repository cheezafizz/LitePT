#!/usr/bin/env bash
# Build a VGGT scene dataset WITH per-point 2D-mask labels for mask-constrained
# clustering (configs/.../insseg-litept-small-v1m2-2of3-embed-maskclust.py).
#
# For every VGGT export under $SRC (<loc>/<frame>/data.npz) this runs
# tools/vggt_to_scene.py --valid-mask filtered, additionally passing the D-FINE-seg
# 2D masks (--coco) and stereo calibration (--camera). Each point's origin (view, pixel)
# is mapped to the 2D instance covering it, written as mask_instance.npy / mask_view.npy
# alongside coord/color/normal.npy in ONE pass (so all arrays stay perfectly aligned --
# no separate crop stage, which would desync the labels from coord.npy).
#
#   bash tools/build_mask_labels.sh
#
# IMPORTANT: this build needs cv2 + torch + torchvision + PIL (VisionSCO's preprocess_view)
# which the litept env lacks, so it runs under $PYB (a cv2-capable env, default `dfine`).
# Inference (tools/infer_insseg.py) still runs under the litept env. Resumable: a scene
# whose mask_instance.npy already exists is skipped.
#
# Output: $OUT/test/<loc>_<frame>/{coord,color,normal,mask_instance,mask_view}.npy
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# cv2-capable python for the mask preprocessing (NOT litept; litept has no cv2).
PYB=${PYB:-/home/fai/miniconda3/envs/dfine/bin/python}
SRC=${SRC:-/home/fai/workspace/jhp/dataset/val_sample_output}
MASK_ROOT=${MASK_ROOT:-/home/fai/workspace/jhp/D-FINE-seg/val_sample_pred}
CAM_ROOT=${CAM_ROOT:-/home/fai/workspace/jhp/D-FINE-seg/val_sample}
OUT=${OUT:-data/scannet-v1.1.1-2of3-vggt-mask}
SPLIT=test

mapfile -t NPZ < <(find "$SRC" -mindepth 3 -maxdepth 3 -name data.npz | sort)
echo "[info] ${#NPZ[@]} source scenes -> $OUT/$SPLIT  (py=$PYB)"

n_ok=0; n_skip=0; n_fail=0
for npz in "${NPZ[@]}"; do
  frame="$(basename "$(dirname "$npz")")"
  loc="$(basename "$(dirname "$(dirname "$npz")")")"
  scene="${loc}_${frame}"
  coco="$MASK_ROOT/$loc/$frame/coco_annotations.json"
  camera="$CAM_ROOT/$loc/$frame/camera.json"
  out_dir="$OUT/$SPLIT/$scene"

  if [[ -f "$out_dir/mask_instance.npy" ]]; then
    echo "[skip] $scene (mask_instance.npy exists)"; n_skip=$((n_skip + 1)); continue
  fi
  if [[ ! -f "$coco" ]]; then
    echo "[FAIL] $scene: missing coco ($coco)"; n_fail=$((n_fail + 1)); continue
  fi
  if [[ ! -f "$camera" ]]; then
    echo "[FAIL] $scene: missing camera ($camera)"; n_fail=$((n_fail + 1)); continue
  fi

  if "$PYB" tools/vggt_to_scene.py \
      --input "$npz" --output-root "$OUT" --split "$SPLIT" --scene "$scene" \
      --valid-mask filtered --coco "$coco" --camera "$camera"; then
    echo "[ok]   $scene"; n_ok=$((n_ok + 1))
  else
    echo "[FAIL] $scene (see output above)"; n_fail=$((n_fail + 1))
  fi
done

echo "[done] ok=$n_ok skip=$n_skip fail=$n_fail"
echo "[done] $(find "$OUT/$SPLIT" -name mask_instance.npy 2>/dev/null | wc -l) scenes labeled under $OUT/$SPLIT"
