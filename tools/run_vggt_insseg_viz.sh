#!/usr/bin/env bash
# Run GT-free InsSeg inference + GLB viz for the three 2of3 augmentation runs
# (last-epoch checkpoints) over every scene in scannet-v1.1.1-2of3-vggt.
#
# Drives tools/infer_insseg.py once per (model, scene). fp32 only (the tool
# never autocasts; bf16 crashes the spconv autotuner in eval). Resumable: a
# (model, scene) whose *_pred.glb already exists is skipped.
#
#   bash tools/run_vggt_insseg_viz.sh
#
# Output (per (model, scene)):
#   viz/vggt_insseg/rgb/<scene>_rgb.glb        (RGB reference, one per scene)
#   viz/vggt_insseg/{normaug,rigidaug,softaug}/<scene>_pred.glb   (instance-colored GLB)
#   viz/vggt_insseg/{normaug,rigidaug,softaug}/<scene>_pred.npz   (structured result for
#                                                                 tools/viser_insseg_viewer.py)
# Resumable on the .npz (the canonical result): a (model, scene) whose *_pred.npz exists is
# skipped. Delete the .npz files (or the per-model dirs) to force a fresh re-run.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/fai/miniconda3/envs/litept/bin/python
DATA_DIR=data/scannet-v1.1.1-2of3-vggt/test
EXP_DIR=exp/scannet-v1.1.1-2of3
OUT_DIR=viz/vggt_insseg
LOG="$OUT_DIR/run.log"

# model tag  ->  run dir name (under $EXP_DIR)
TAGS=(normaug rigidaug softaug)
RUNS=(
  "insseg-litept-small-v1m2-2of3-normaug"
  "v1.1.1-2of3-ep1600-eval1600-lr6e-3-rigidaug-cluster-5mm"
  "v1.1.1-2of3-ep1600-eval1600-lr6e-3-softaug-cluster-5mm"
)

mkdir -p "$OUT_DIR/rgb"
: > "$LOG"

# scenes = every subdir of $DATA_DIR
mapfile -t SCENES < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
echo "[info] ${#SCENES[@]} scenes x ${#TAGS[@]} models" | tee -a "$LOG"

n_ok=0; n_fail=0; n_skip=0
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  run="${RUNS[$i]}"
  cfg="$EXP_DIR/$run/config.py"
  wgt="$EXP_DIR/$run/model/model_last.pth"
  if [[ ! -f "$cfg" || ! -f "$wgt" ]]; then
    echo "[FAIL] $tag: missing config or weight ($cfg / $wgt)" | tee -a "$LOG"
    n_fail=$((n_fail + ${#SCENES[@]}))
    continue
  fi
  # save the RGB reference cloud only on the first (normaug) pass
  save_rgb=""; [[ "$i" -eq 0 ]] && save_rgb="--save-rgb"

  echo "=== model: $tag ($run) ===" | tee -a "$LOG"
  for scene in "${SCENES[@]}"; do
    out="$OUT_DIR/$tag/${scene}_pred.glb"
    npz="$OUT_DIR/$tag/${scene}_pred.npz"
    if [[ -f "$npz" ]]; then
      echo "[skip] $tag/$scene (exists)" | tee -a "$LOG"
      n_skip=$((n_skip + 1)); continue
    fi
    if "$PY" tools/infer_insseg.py \
        --scene-dir   "$DATA_DIR/$scene" \
        --config-file "$cfg" \
        --weight      "$wgt" \
        --out         "$out" \
        --save-npz --model-tag "$tag" \
        $save_rgb >>"$LOG" 2>&1; then
      preds=$(grep -aoE '\[pred\] [0-9]+ instance' "$LOG" | tail -1 | grep -oE '[0-9]+' | head -1)
      echo "[ok]   $tag/$scene  (${preds:-?} proposals)" | tee -a "$LOG"
      n_ok=$((n_ok + 1))
    else
      echo "[FAIL] $tag/$scene  (see $LOG)" | tee -a "$LOG"
      n_fail=$((n_fail + 1))
    fi
  done
done

# consolidate the normaug-pass RGB siblings into a shared rgb/ folder
shopt -s nullglob
for f in "$OUT_DIR/normaug/"*_pred_rgb.glb; do
  base="$(basename "$f")"; scene="${base%_pred_rgb.glb}"
  mv -f "$f" "$OUT_DIR/rgb/${scene}_rgb.glb"
done
shopt -u nullglob

echo "[done] ok=$n_ok fail=$n_fail skip=$n_skip" | tee -a "$LOG"
total=$(find "$OUT_DIR" -name '*.glb' | wc -l)
echo "[done] $total GLBs under $OUT_DIR" | tee -a "$LOG"
