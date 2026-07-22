#!/usr/bin/env bash
# GT-free InsSeg inference + npz/GLB viz for an A/B of two MULTI-VIEW (all-views) 2D-mask
# cannot-link CLUSTERING strategies, over every scene in scannet-v1.1.1-2of3-vggt-mask-allviews.
# Both passes share ONE network/checkpoint (the instance-embedding model) and differ ONLY in the
# inference-time clustering config (--cluster-config):
#
#   maskclust : clustering/embed-maskclust-allviews.py  -> multi-view cannot-link, NON-strict
#   strong    : clustering/embed-maskclust-strong.py    -> multi-view cannot-link, transitively STRICT
#
# Both auto-consume <scene>/mask_per_view.npy (z-buffered per-view 2D masks); they diverge only at
# touching-object seams, where the non-strict path can re-merge via a detour and strict cannot.
#
#   bash tools/run_maskclust_strong_infer.sh
#
# fp32 only (tools/infer_insseg.py never autocasts; bf16 crashes the spconv autotuner in eval).
# CUDA must be reachable (nvidia-smi alone is not enough -- a sandbox that blocks torch CUDA init
# will fail at model build).
#
# Output (per (tag, scene)):
#   viz/maskclust_strong/maskclust/<scene>_pred.{glb,npz}   multi-view non-strict (tag maskclust)
#   viz/maskclust_strong/strong/<scene>_pred.{glb,npz}      multi-view strict     (tag strong)
# The viser viewer keys the Model dropdown off the parent dir name (= the tag) and derives the
# RGB-input cloud from any npz, so no --save-rgb pass is needed. Resumable on the .npz: a
# (tag, scene) whose *_pred.npz exists is skipped.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=${PY:-/home/fai/miniconda3/envs/litept/bin/python}
CFG=${CFG:-exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-embed/config.py}
WGT=${WGT:-exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-embed/model/model_best.pth}
DATA_DIR=${DATA_DIR:-data/scannet-v1.1.1-2of3-vggt-mask-allviews/test}
OUT_DIR=${OUT_DIR:-viz/maskclust_strong}
LOG="$OUT_DIR/run.log"

# tag  ->  inference-time clustering config (kept paired by index). Both are space-separated
# env overrides (kept paired) so an extra pass can be appended into an existing OUT_DIR, e.g.:
#   TAGS="strong-filter" \
#   CLUSTER_CFGS="configs/scannet-v1.1.1-2of3/clustering/embed-maskclust-strong-filter.py" \
#   bash tools/run_maskclust_strong_infer.sh
if [[ -n "${TAGS:-}" ]]; then read -r -a TAGS <<< "$TAGS"; else
  TAGS=(maskclust strong); fi
if [[ -n "${CLUSTER_CFGS:-}" ]]; then read -r -a CLUSTER_CFGS <<< "$CLUSTER_CFGS"; else
  CLUSTER_CFGS=(
    "configs/scannet-v1.1.1-2of3/clustering/embed-maskclust-allviews.py"
    "configs/scannet-v1.1.1-2of3/clustering/embed-maskclust-strong.py"
  ); fi
if [[ ${#TAGS[@]} -ne ${#CLUSTER_CFGS[@]} ]]; then
  echo "[FATAL] TAGS (${#TAGS[@]}) and CLUSTER_CFGS (${#CLUSTER_CFGS[@]}) must have equal length" >&2
  exit 2; fi

if [[ ! -f "$CFG" ]]; then echo "[FATAL] missing config: $CFG"; exit 1; fi
if [[ ! -f "$WGT" ]]; then echo "[FATAL] missing weight: $WGT"; exit 1; fi
if [[ ! -d "$DATA_DIR" ]]; then echo "[FATAL] missing data dir: $DATA_DIR"; exit 1; fi
for cc in "${CLUSTER_CFGS[@]}"; do
  if [[ ! -f "$cc" ]]; then echo "[FATAL] missing cluster-config: $cc"; exit 1; fi
done

mkdir -p "$OUT_DIR"
for tag in "${TAGS[@]}"; do mkdir -p "$OUT_DIR/$tag"; done
: > "$LOG"

mapfile -t SCENES < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
echo "[info] ${#SCENES[@]} scenes x ${#TAGS[@]} configs (tags: ${TAGS[*]})" | tee -a "$LOG"
echo "[info] cfg=$CFG" | tee -a "$LOG"
echo "[info] wgt=$WGT" | tee -a "$LOG"
echo "[info] data=$DATA_DIR  out=$OUT_DIR" | tee -a "$LOG"

# extract the "[pred] N instance proposals" count from the tail of the log
last_pred() { grep -aoE '\[pred\] [0-9]+ instance' "$LOG" | tail -1 | grep -oE '[0-9]+' | head -1; }

n_ok=0; n_fail=0; n_skip=0
declare -A CNT  # CNT["tag/scene"] = proposal count
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  ccfg="${CLUSTER_CFGS[$i]}"
  echo "=== config: $tag ($ccfg) ===" | tee -a "$LOG"
  for scene in "${SCENES[@]}"; do
    npz="$OUT_DIR/$tag/${scene}_pred.npz"
    if [[ -f "$npz" ]]; then
      echo "[skip] $tag/$scene (exists)" | tee -a "$LOG"; n_skip=$((n_skip + 1)); continue
    fi
    echo "=== $tag : $scene ===" | tee -a "$LOG"
    if "$PY" tools/infer_insseg.py \
        --scene-dir   "$DATA_DIR/$scene" \
        --config-file "$CFG" --weight "$WGT" \
        --cluster-config "$ccfg" \
        --out         "$OUT_DIR/$tag/${scene}_pred.glb" \
        --save-npz --model-tag "$tag" >>"$LOG" 2>&1; then
      CNT["$tag/$scene"]="$(last_pred)"
      echo "[ok]   $tag/$scene  (${CNT[$tag/$scene]:-?} proposals)" | tee -a "$LOG"
      n_ok=$((n_ok + 1))
    else
      echo "[FAIL] $tag/$scene  (see $LOG)" | tee -a "$LOG"; n_fail=$((n_fail + 1))
    fi
  done
done

echo "[done] ok=$n_ok fail=$n_fail skip=$n_skip" | tee -a "$LOG"
# summary table: one proposal-count column per tag (generic over TAGS)
hdr=$(printf '%-34s' "scene"); for tag in "${TAGS[@]}"; do hdr+=$(printf ' %12s' "$tag"); done
echo "[summary] $hdr" | tee -a "$LOG"
for scene in "${SCENES[@]}"; do
  row=$(printf '%-34s' "$scene")
  for tag in "${TAGS[@]}"; do row+=$(printf ' %12s' "${CNT[$tag/$scene]:-_}"); done
  echo "[summary] $row" | tee -a "$LOG"
done
z=$(find "$OUT_DIR" -name '*_pred.npz' | wc -l); g=$(find "$OUT_DIR" -name '*.glb' | wc -l)
echo "[done] $z npz, $g GLBs under $OUT_DIR" | tee -a "$LOG"
