#!/usr/bin/env bash
# Supervise the ssl_labels_v2 export to completion, then convert
# ssl_labels_out_v2 -> data/real-ssl-v2 and launch -query-realft-muon-v2.
#
# The export dies randomly every few thousand scenes with native heap
# corruption (double free / munmap_chunk — different scenes each time, so not
# scene-deterministic). It is resumable (skips scenes with npz+dets), so this
# script relaunches it after every crash as long as each leg makes progress
# (>= MIN_PROGRESS new scenes), and treats the exporter's "DONE:" log line as
# the real completion signal.
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
EXPORT_LOG="$SSL_ROOT/full_run.log"
MIN_PROGRESS=50    # a retry leg must add at least this many scenes, else abort
MAX_RETRIES=100

log() { echo "[$(date '+%F %T')] $*"; }
count_npz() { ls "$SSL_ROOT"/*.npz 2>/dev/null | wc -l; }
export_alive() { docker exec ssl_labels_v2 pgrep -f save_instance_labels.py > /dev/null 2>&1; }
export_done() { grep -q "DONE:" "$EXPORT_LOG"; }

launch_export() {
    # HANDOFF.md Step 2, resume mode (do NOT wipe outputs)
    setsid nohup docker exec -w /workspace \
        -e SSL_DATA_ROOT=/data -e SSL_OUTPUT_ROOT=/out -e SSL_PRODUCTS_DIR=/products \
        -e "SSL_CLIP_BOX=-0.4,0.6,-0.5,0.7,-0.2,0.7" -e SSL_BG_VOXEL_MM=10 \
        ssl_labels_v2 python3 save_instance_labels.py \
        >> "$EXPORT_LOG" 2>&1 &
}

retries=0
last_count=$(count_npz)
if ! export_alive && ! export_done; then
    log "export not running (count $last_count); launching"
    launch_export
fi

newest_npz_age() {
    local f
    f=$(ls -t "$SSL_ROOT"/*.npz 2>/dev/null | head -1) || { echo 999999; return; }
    echo $(( $(date +%s) - $(stat -c %Y "$f") ))
}

until export_done; do
    # Stall detector: the process can wedge in an unkillable kernel deadlock
    # (D-state, seen 2026-07-24) — alive but writing nothing. We cannot kill it
    # (SIGKILL undeliverable; docker kill fails; host reboot required), so alert
    # loudly and keep waiting.
    if export_alive && [ "$(newest_npz_age)" -gt 1800 ]; then
        log "ALERT: export process alive but NO new scene for 30+ min — likely kernel-deadlocked (D-state). A HOST REBOOT + tools/recover_v2_after_reboot.sh is needed."
        sleep 1800
        continue
    fi
    if ! export_alive; then
        n=$(count_npz)
        if [ $((n - last_count)) -lt "$MIN_PROGRESS" ]; then
            log "ABORT: export died at $n scenes with < $MIN_PROGRESS progress since last relaunch ($last_count) — crash-looping, needs a human."
            exit 1
        fi
        retries=$((retries + 1))
        [ "$retries" -le "$MAX_RETRIES" ] || { log "ABORT: exceeded $MAX_RETRIES relaunches."; exit 1; }
        log "export died at $n scenes (+$((n - last_count)) this leg); relaunch #$retries"
        last_count=$n
        launch_export
        sleep 60
    fi
    sleep 300
done

log "export DONE ($(count_npz) scenes, $retries relaunches); settling 60s"
sleep 60

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
