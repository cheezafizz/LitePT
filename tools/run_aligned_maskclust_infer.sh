#!/usr/bin/env bash
# GT-free InsSeg inference + GLB/npz viz for the ALIGNED-mask-constrained config over every
# scene in scannet-v1.1.1-2of3-vggt-mask-aligned. Runs an A/B per scene: the 2D-mask
# cannot-link constraint ON (uses the baked mask_instance.npy) vs OFF (--no-mask-constraint).
#
#   bash tools/run_aligned_maskclust_infer.sh
#
# fp32 only (tools/infer_insseg.py never autocasts; bf16 crashes the spconv autotuner in eval).
# CUDA must be reachable (nvidia-smi alone is not enough -- a sandbox that blocks torch CUDA
# init will fail at model build).
#
# Output (per scene):
#   viz/aligned_maskclust/rgb/<scene>_rgb.glb            RGB reference cloud (one per scene)
#   viz/aligned_maskclust/on/<scene>_pred.{glb,npz}      constraint ON  (model-tag maskclust-on)
#   viz/aligned_maskclust/off/<scene>_pred.{glb,npz}     constraint OFF (model-tag maskclust-off)
# Resumable on the .npz (the canonical result): a (scene, pass) whose *_pred.npz exists is skipped.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=${PY:-/home/fai/miniconda3/envs/litept/bin/python}
CFG=${CFG:-configs/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-embed-maskclust-aligned.py}
WGT=${WGT:-exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-embed/model/model_best.pth}
DATA_DIR=${DATA_DIR:-data/scannet-v1.1.1-2of3-vggt-mask-aligned/test}
OUT_DIR=${OUT_DIR:-viz/aligned_maskclust}
LOG="$OUT_DIR/run.log"

if [[ ! -f "$CFG" ]]; then echo "[FATAL] missing config: $CFG"; exit 1; fi
if [[ ! -f "$WGT" ]]; then echo "[FATAL] missing weight: $WGT"; exit 1; fi

mkdir -p "$OUT_DIR/rgb" "$OUT_DIR/on" "$OUT_DIR/off"
: > "$LOG"

mapfile -t SCENES < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
echo "[info] ${#SCENES[@]} scenes x 2 passes (ON/OFF)" | tee -a "$LOG"
echo "[info] cfg=$CFG" | tee -a "$LOG"
echo "[info] wgt=$WGT" | tee -a "$LOG"

# extract the "[pred] N instance proposals" count from the tail of the log
last_pred() { grep -aoE '\[pred\] [0-9]+ instance' "$LOG" | tail -1 | grep -oE '[0-9]+' | head -1; }

n_ok=0; n_fail=0; n_skip=0
declare -A ON_CNT OFF_CNT
for scene in "${SCENES[@]}"; do
  sd="$DATA_DIR/$scene"

  # --- ON: constraint active, plus the one-per-scene RGB reference ---
  npz_on="$OUT_DIR/on/${scene}_pred.npz"
  if [[ -f "$npz_on" ]]; then
    echo "[skip] $scene ON (exists)" | tee -a "$LOG"; n_skip=$((n_skip + 1))
  else
    echo "=== $scene : ON ===" | tee -a "$LOG"
    if "$PY" tools/infer_insseg.py \
        --scene-dir "$sd" --config-file "$CFG" --weight "$WGT" \
        --out "$OUT_DIR/on/${scene}_pred.glb" \
        --save-npz --model-tag maskclust-on --save-rgb >>"$LOG" 2>&1; then
      ON_CNT["$scene"]="$(last_pred)"
      echo "[ok]   $scene ON  (${ON_CNT[$scene]:-?} proposals)" | tee -a "$LOG"
      n_ok=$((n_ok + 1))
    else
      echo "[FAIL] $scene ON  (see $LOG)" | tee -a "$LOG"; n_fail=$((n_fail + 1))
    fi
  fi

  # --- OFF: --no-mask-constraint baseline (no RGB; ON pass already wrote it) ---
  npz_off="$OUT_DIR/off/${scene}_pred.npz"
  if [[ -f "$npz_off" ]]; then
    echo "[skip] $scene OFF (exists)" | tee -a "$LOG"; n_skip=$((n_skip + 1))
  else
    echo "=== $scene : OFF ===" | tee -a "$LOG"
    if "$PY" tools/infer_insseg.py \
        --scene-dir "$sd" --config-file "$CFG" --weight "$WGT" --no-mask-constraint \
        --out "$OUT_DIR/off/${scene}_pred.glb" \
        --save-npz --model-tag maskclust-off >>"$LOG" 2>&1; then
      OFF_CNT["$scene"]="$(last_pred)"
      echo "[ok]   $scene OFF (${OFF_CNT[$scene]:-?} proposals)" | tee -a "$LOG"
      n_ok=$((n_ok + 1))
    else
      echo "[FAIL] $scene OFF (see $LOG)" | tee -a "$LOG"; n_fail=$((n_fail + 1))
    fi
  fi
done

# consolidate the ON-pass RGB siblings into a shared rgb/ folder
shopt -s nullglob
for f in "$OUT_DIR/on/"*_pred_rgb.glb; do
  base="$(basename "$f")"; scene="${base%_pred_rgb.glb}"
  mv -f "$f" "$OUT_DIR/rgb/${scene}_rgb.glb"
done
shopt -u nullglob

echo "[done] ok=$n_ok fail=$n_fail skip=$n_skip" | tee -a "$LOG"
echo "[summary] scene                              ON   OFF" | tee -a "$LOG"
for scene in "${SCENES[@]}"; do
  printf '[summary] %-34s %4s %4s\n' "$scene" "${ON_CNT[$scene]:-_}" "${OFF_CNT[$scene]:-_}" | tee -a "$LOG"
done
g=$(find "$OUT_DIR" -name '*.glb' | wc -l); z=$(find "$OUT_DIR" -name '*_pred.npz' | wc -l)
echo "[done] $g GLBs, $z npz under $OUT_DIR" | tee -a "$LOG"
