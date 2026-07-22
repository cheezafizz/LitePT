#!/usr/bin/env bash
# Build a VGGT scene dataset with ALL-VIEWS per-point 2D-mask labels (mask_per_view.npy)
# for multi-view mask-constrained clustering
# (configs/.../insseg-litept-small-v1m2-2of3-embed-maskclust-allviews.py).
#
# All-views twin of tools/build_mask_labels_aligned.sh. Same ALIGNED mask source
# ($MASK_ROOT, 518x392, already on the VGGT grid -> --mask-pre-aligned, no camera.json),
# but tools/vggt_to_scene.py is run with --mask-all-views: every fused point is
# reprojected into ALL 7 views (z-buffer occlusion via world_points + the aligned camera
# intrinsic_aligned/extrinsic_pnp) and the 2D mask it lands in is recorded PER VIEW into
# mask_per_view.npy (N,S). The legacy mask_instance/mask_view (source-view column) are
# also written for A/B against the single-view -maskclust-aligned run.
#
# The point set is the same filtered_valid_mask as -maskclust-aligned, additionally
# cleaned with a 25th-percentile confidence floor (--valid-mask filtered
# --conf-percentile 25).
#
#   bash tools/build_mask_labels_allviews.sh
#
# PREREQUISITE: each scene's data.npz must already carry `intrinsic_aligned` and
# `extrinsic_pnp` (the aligned camera used to reproject the cloud). Run
#   /home/fai/miniconda3/envs/da3/bin/python tools/align_vggt_cameras.py
# first if those keys are missing.
#
# IMPORTANT: needs cv2 + torch + torchvision + PIL (VisionSCO's preprocess_view) which
# the litept env lacks, so it runs under $PYB (default `dfine`). Inference
# (tools/infer_insseg.py) still runs under the litept env. Resumable: a scene whose
# mask_per_view.npy already exists is skipped.
#
# Output: $OUT/test/<loc>_<frame>/{coord,color,normal,mask_per_view,mask_instance,mask_view}.npy
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# cv2-capable python for the mask preprocessing (NOT litept; litept has no cv2).
PYB=${PYB:-/home/fai/miniconda3/envs/dfine/bin/python}
SRC=${SRC:-/home/fai/workspace/jhp/dataset/val_sample_output}
# ALIGNED COCO masks (518x392, already on the VGGT grid).
MASK_ROOT=${MASK_ROOT:-/home/fai/workspace/jhp/dataset/val_sample_output_aligned}
OUT=${OUT:-data/scannet-v1.1.1-2of3-vggt-mask-allviews}
SPLIT=test
# aligned masks are already on the VGGT image grid; pass the grid size as W,H.
MASK_ORIGINAL_SIZE=${MASK_ORIGINAL_SIZE:-518,392}
# cloud point set: filtered_valid_mask + 25th-pct confidence floor.
CONF_PERCENTILE=${CONF_PERCENTILE:-25}
# z-buffer depth tolerance (m) for the all-views occlusion test.
OCCLUSION_TOL=${OCCLUSION_TOL:-0.005}

mapfile -t NPZ < <(find "$SRC" -mindepth 3 -maxdepth 3 -name data.npz | sort)
echo "[info] ${#NPZ[@]} source scenes -> $OUT/$SPLIT  (py=$PYB)"
echo "[info] aligned masks: $MASK_ROOT (size=$MASK_ORIGINAL_SIZE, --mask-pre-aligned)"
echo "[info] all-views reprojection: --valid-mask filtered --conf-percentile $CONF_PERCENTILE"
echo "[info]   --mask-all-views --mask-occlusion-tol $OCCLUSION_TOL (needs intrinsic_aligned + extrinsic_pnp)"

n_ok=0; n_skip=0; n_fail=0
for npz in "${NPZ[@]}"; do
  frame="$(basename "$(dirname "$npz")")"
  loc="$(basename "$(dirname "$(dirname "$npz")")")"
  scene="${loc}_${frame}"
  coco="$MASK_ROOT/$loc/$frame/coco_annotations.json"
  out_dir="$OUT/$SPLIT/$scene"

  if [[ -f "$out_dir/mask_per_view.npy" ]]; then
    echo "[skip] $scene (mask_per_view.npy exists)"; n_skip=$((n_skip + 1)); continue
  fi
  if [[ ! -f "$coco" ]]; then
    echo "[FAIL] $scene: missing aligned coco ($coco)"; n_fail=$((n_fail + 1)); continue
  fi

  # --mask-pre-aligned: index the already-aligned masks directly (no preprocess_view,
  # no camera.json). --mask-all-views reprojects with intrinsic_aligned/extrinsic_pnp.
  if "$PYB" tools/vggt_to_scene.py \
      --input "$npz" --output-root "$OUT" --split "$SPLIT" --scene "$scene" \
      --valid-mask filtered --conf-percentile "$CONF_PERCENTILE" \
      --coco "$coco" --mask-pre-aligned --mask-original-size "$MASK_ORIGINAL_SIZE" \
      --mask-all-views --mask-occlusion-tol "$OCCLUSION_TOL"; then
    echo "[ok]   $scene"; n_ok=$((n_ok + 1))
  else
    echo "[FAIL] $scene (see output above)"; n_fail=$((n_fail + 1))
  fi
done

echo "[done] ok=$n_ok skip=$n_skip fail=$n_fail"
echo "[done] $(find "$OUT/$SPLIT" -name mask_per_view.npy 2>/dev/null | wc -l) scenes labeled under $OUT/$SPLIT"
