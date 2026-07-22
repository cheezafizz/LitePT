#!/usr/bin/env bash
# GT-free InsSeg inference (--save-npz) over every scene in
# data/scannet-vggt-default-filtered/test, for the three 2of3 augmentation runs
# at their LAST epoch (model_last.pth). fp32 only (the tool never autocasts;
# bf16 crashes the spconv autotuner in eval). Resumable on the .npz.
#
#   bash tools/run_default_filtered_infer.sh
#
# Output (per (model, scene)):
#   viz/vggt_default_filtered/{normaug,rigidaug,softaug}/<scene>_pred.glb  (instance-colored)
#   viz/vggt_default_filtered/{normaug,rigidaug,softaug}/<scene>_pred.npz  (for the viser viewer)
# The viser viewer keys the model off the parent dir name (= the tag below). No --save-rgb:
# the viewer derives the RGB-input cloud from any model's npz.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/fai/miniconda3/envs/litept/bin/python
# DATA_DIR / OUT_DIR are env-overridable (defaults reproduce the original run):
#   DATA_DIR=data/scannet-vggt-default-filtered-conf25/test \
#   OUT_DIR=viz/vggt_default_filtered_conf25 bash tools/run_default_filtered_infer.sh
DATA_DIR=${DATA_DIR:-data/scannet-vggt-default-filtered/test}
EXP_DIR=exp/scannet-v1.1.1-2of3
OUT_DIR=${OUT_DIR:-viz/vggt_default_filtered}
LOG="$OUT_DIR/run.log"
# WEIGHT = checkpoint filename under <run>/model/ (model_last.pth default; model_best.pth
# for "best ckpt"). TAGS / RUNS are space-separated env overrides (kept paired) so a single
# extra model can be appended into an existing OUT_DIR, e.g. the embed best-ckpt model:
#   TAGS="embed" RUNS="insseg-litept-small-v1m2-2of3-embed" WEIGHT=model_best.pth \
#   DATA_DIR=data/scannet-vggt-default-filtered-conf25/test \
#   OUT_DIR=viz/vggt_default_filtered_conf25 bash tools/run_default_filtered_infer.sh
WEIGHT=${WEIGHT:-model_last.pth}

# model tag  ->  run dir name (under $EXP_DIR); overridable via the TAGS/RUNS env strings
if [[ -n "${TAGS:-}" ]]; then read -r -a TAGS <<< "$TAGS"; else
  TAGS=(normaug rigidaug softaug); fi
if [[ -n "${RUNS:-}" ]]; then read -r -a RUNS <<< "$RUNS"; else
  RUNS=(
    "insseg-litept-small-v1m2-2of3-normaug"
    "v1.1.1-2of3-ep1600-eval1600-lr6e-3-rigidaug-cluster-5mm"
    "v1.1.1-2of3-ep1600-eval1600-lr6e-3-softaug-cluster-5mm"
  ); fi
if [[ ${#TAGS[@]} -ne ${#RUNS[@]} ]]; then
  echo "[FAIL] TAGS (${#TAGS[@]}) and RUNS (${#RUNS[@]}) must have equal length" >&2; exit 2; fi

mkdir -p "$OUT_DIR"
: > "$LOG"

# scenes = every subdir of $DATA_DIR
mapfile -t SCENES < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
echo "[info] ${#SCENES[@]} scenes x ${#TAGS[@]} models ($WEIGHT)" | tee -a "$LOG"

n_ok=0; n_fail=0; n_skip=0
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  run="${RUNS[$i]}"
  cfg="$EXP_DIR/$run/config.py"
  wgt="$EXP_DIR/$run/model/$WEIGHT"
  mkdir -p "$OUT_DIR/$tag"
  if [[ ! -f "$cfg" || ! -f "$wgt" ]]; then
    echo "[FAIL] $tag: missing config or weight ($cfg / $wgt)" | tee -a "$LOG"
    n_fail=$((n_fail + ${#SCENES[@]}))
    continue
  fi
  echo "=== model: $tag ($run) ===" | tee -a "$LOG"
  for scene in "${SCENES[@]}"; do
    out="$OUT_DIR/$tag/${scene}_pred.glb"
    npz="$OUT_DIR/$tag/${scene}_pred.npz"
    if [[ -f "$npz" ]]; then
      echo "[skip] $tag/$scene (exists)" | tee -a "$LOG"; n_skip=$((n_skip + 1)); continue
    fi
    if "$PY" tools/infer_insseg.py \
        --scene-dir   "$DATA_DIR/$scene" \
        --config-file "$cfg" \
        --weight      "$wgt" \
        --out         "$out" \
        --save-npz --model-tag "$tag" >>"$LOG" 2>&1; then
      preds=$(grep -aoE '\[pred\] [0-9]+ instance' "$LOG" | tail -1 | grep -oE '[0-9]+' | head -1)
      echo "[ok]   $tag/$scene  (${preds:-?} proposals)" | tee -a "$LOG"; n_ok=$((n_ok + 1))
    else
      echo "[FAIL] $tag/$scene  (see $LOG)" | tee -a "$LOG"; n_fail=$((n_fail + 1))
    fi
  done
done

echo "[done] ok=$n_ok fail=$n_fail skip=$n_skip" | tee -a "$LOG"
echo "[done] $(find "$OUT_DIR" -name '*_pred.npz' | wc -l) npz under $OUT_DIR" | tee -a "$LOG"
