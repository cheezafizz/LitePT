#!/usr/bin/env bash
# Watch the ssl_labels_v2 export container; when the export process exits,
# convert ssl_labels_out_v2 -> data/real-ssl-v2 (resumable) and launch the
# -query-realft-muon-v2 finetune detached.
#
# Launch (detached — survives the launching session, NOT a host reboot):
#   setsid nohup bash tools/watch_v2_then_train.sh > logs/watch_v2_then_train.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

PY=/home/fai/miniconda3/envs/litept/bin/python
SSL_ROOT=/home/fai/workspace/jhp/ssl_labels_out_v2
CAM_ROOT=/home/fai/workspace/jhp/dataset/SSL_dataset/dataset_ssl_1_80k
OUT_ROOT=data/real-ssl-v2
CONFIG=insseg-litept-small-v1m2-2of3-query-realft-muon-v2
MIN_SCENES=60000   # abort if the export "finished" with far fewer than 80k scenes

log() { echo "[$(date '+%F %T')] $*"; }

log "watching container ssl_labels_v2 for save_instance_labels exit..."
while docker exec ssl_labels_v2 pgrep -f save_instance_labels.py > /dev/null 2>&1; do
    sleep 600
done
log "export process gone; settling 120s"
sleep 120

N_NPZ=$(ls "$SSL_ROOT"/*.npz 2>/dev/null | wc -l)
log "npz count: $N_NPZ"
if [ "$N_NPZ" -lt "$MIN_SCENES" ]; then
    log "ABORT: only $N_NPZ scenes (< $MIN_SCENES) — export likely crashed, not finished. NOT converting/training."
    exit 1
fi

log "converting to $OUT_ROOT (resumable)..."
"$PY" tools/convert_ssl_scenes.py \
    --ssl-root "$SSL_ROOT" --out-root "$OUT_ROOT" \
    --camera-root "$CAM_ROOT" --val-stores 001,004 --workers 16 \
    || { log "ABORT: conversion failed"; exit 1; }

N_TRAIN=$(ls "$OUT_ROOT/train" | wc -l); N_VAL=$(ls "$OUT_ROOT/val" | wc -l)
log "converted: $N_TRAIN train / $N_VAL val"
[ "$N_TRAIN" -gt 10000 ] || { log "ABORT: suspiciously few train scenes"; exit 1; }

log "launching training: $CONFIG"
setsid nohup bash scripts/train.sh -p "$PY" \
    -d scannet-v1.1.1-2of3 -c "$CONFIG" -n "$CONFIG" -g 1 \
    > "logs/${CONFIG}.launch.log" 2>&1 &
log "training launched (pid $!), log: logs/${CONFIG}.launch.log"
