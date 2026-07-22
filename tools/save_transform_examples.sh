#!/usr/bin/env bash
# Regenerate GLB examples of the training transform pipeline.
#
# For each scene this writes, under logs/transform_viz/<scene>/:
#   baseline/  -> deterministic val pipeline (no augmentation), GT + RGB
#   seed{0,1,2}/ -> full augmented train pipeline at distinct seeds, GT + RGB
# The seed lives in the output dir (the tool does not put it in the filename),
# so different draws never overwrite each other.
#
# CPU-only. Reuses tools/visualize_transformed_scene.py as-is.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/fai/miniconda3/envs/litept/bin/python
CFG=configs/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3.py
OUT=logs/transform_viz
SCENES=(scene70_0040 scene70_0155 scene70_0074)

for s in "${SCENES[@]}"; do
  # un-augmented baseline (eval/val pipeline)
  "$PY" tools/visualize_transformed_scene.py --config-file "$CFG" \
    --scene "$s" --split train --pipeline val --save-rgb \
    --out "$OUT/$s/baseline"
  # augmented train pipeline, several seeds
  for seed in 0 1 2; do
    "$PY" tools/visualize_transformed_scene.py --config-file "$CFG" \
      --scene "$s" --split train --pipeline train --seed "$seed" --save-rgb \
      --out "$OUT/$s/seed$seed"
  done
done

echo
echo "GLB count: $(find "$OUT" -name '*.glb' | wc -l)"
