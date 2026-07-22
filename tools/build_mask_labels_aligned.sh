#!/usr/bin/env bash
# Build a VGGT scene dataset WITH per-point 2D-mask labels from the ALIGNED D-FINE-seg
# masks, for mask-constrained clustering
# (configs/.../insseg-litept-small-v1m2-2of3-embed-maskclust-aligned.py).
#
# This is the aligned-mask twin of tools/build_mask_labels.sh. The difference is the
# 2D-mask source: instead of the raw 640x480 masks (val_sample_pred), it uses the
# ALIGNED masks under $MASK_ROOT (val_sample_output_aligned), which are already on the
# VGGT image grid (518x392 == world_points (S,392,518,3)). They were produced by
# D-FINE-seg/scripts/align_coco_to_world_points.py, which already bakes in the full
# preprocess_view pipeline (undistort -> rotate -> center_pp -> crop-resize). So
# vggt_to_scene.py runs with --mask-pre-aligned: it indexes the decoded masks DIRECTLY
# (NO preprocess_view -- re-applying the baked 180deg rotation / center_pp would corrupt
# 5 of 7 views) and needs NO camera.json. Each point's origin (view, pixel) indexes
# straight into the aligned 2D instance covering it.
#
#   bash tools/build_mask_labels_aligned.sh
#
# IMPORTANT: needs cv2 + torch + torchvision + PIL (VisionSCO's preprocess_view) which
# the litept env lacks, so it runs under $PYB (default `dfine`). Inference
# (tools/infer_insseg.py) still runs under the litept env. Resumable: a scene whose
# mask_instance.npy already exists is skipped.
#
# Output: $OUT/test/<loc>_<frame>/{coord,color,normal,mask_instance,mask_view}.npy
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# cv2-capable python for the mask preprocessing (NOT litept; litept has no cv2).
PYB=${PYB:-/home/fai/miniconda3/envs/dfine/bin/python}
SRC=${SRC:-/home/fai/workspace/jhp/dataset/val_sample_output}
# ALIGNED COCO masks (518x392, already on the VGGT grid).
MASK_ROOT=${MASK_ROOT:-/home/fai/workspace/jhp/dataset/val_sample_output_aligned}
OUT=${OUT:-data/scannet-v1.1.1-2of3-vggt-mask-aligned}
SPLIT=test
# aligned masks are already on the VGGT image grid; pass the grid size as W,H.
MASK_ORIGINAL_SIZE=${MASK_ORIGINAL_SIZE:-518,392}

mapfile -t NPZ < <(find "$SRC" -mindepth 3 -maxdepth 3 -name data.npz | sort)
echo "[info] ${#NPZ[@]} source scenes -> $OUT/$SPLIT  (py=$PYB)"
echo "[info] aligned masks: $MASK_ROOT (size=$MASK_ORIGINAL_SIZE, --mask-pre-aligned: direct index, no preprocess_view)"

n_ok=0; n_skip=0; n_fail=0
for npz in "${NPZ[@]}"; do
  frame="$(basename "$(dirname "$npz")")"
  loc="$(basename "$(dirname "$(dirname "$npz")")")"
  scene="${loc}_${frame}"
  coco="$MASK_ROOT/$loc/$frame/coco_annotations.json"
  out_dir="$OUT/$SPLIT/$scene"

  if [[ -f "$out_dir/mask_instance.npy" ]]; then
    echo "[skip] $scene (mask_instance.npy exists)"; n_skip=$((n_skip + 1)); continue
  fi
  if [[ ! -f "$coco" ]]; then
    echo "[FAIL] $scene: missing aligned coco ($coco)"; n_fail=$((n_fail + 1)); continue
  fi

  # --mask-pre-aligned: index the already-aligned masks directly (no preprocess_view,
  # no camera.json). Re-applying undistort/rotate/center_pp would corrupt them.
  if "$PYB" tools/vggt_to_scene.py \
      --input "$npz" --output-root "$OUT" --split "$SPLIT" --scene "$scene" \
      --valid-mask filtered --coco "$coco" \
      --mask-pre-aligned --mask-original-size "$MASK_ORIGINAL_SIZE"; then
    echo "[ok]   $scene"; n_ok=$((n_ok + 1))
  else
    echo "[FAIL] $scene (see output above)"; n_fail=$((n_fail + 1))
  fi
done

echo "[done] ok=$n_ok skip=$n_skip fail=$n_fail"
echo "[done] $(find "$OUT/$SPLIT" -name mask_instance.npy 2>/dev/null | wc -l) scenes labeled under $OUT/$SPLIT"
