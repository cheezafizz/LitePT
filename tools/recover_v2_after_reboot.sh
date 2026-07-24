#!/usr/bin/env bash
# One-shot recovery after rebooting the host to clear the deadlocked v2 export
# (D-state python3, mmap rwsem deadlock, 2026-07-24 14:05). Run as fai:
#   bash tools/recover_v2_after_reboot.sh
set -eu
cd "$(dirname "$0")/.."

SSL_ROOT=/home/fai/workspace/jhp/ssl_labels_out_v2

# 1. Drop the scene that was (possibly mid-)written when the process wedged —
#    resume skips scenes whose outputs exist, so a truncated npz would never heal.
LAST=1_003_022_000208
echo "removing possibly-truncated outputs of $LAST"
rm -f "$SSL_ROOT/$LAST.npz" "$SSL_ROOT/dets/${LAST}_dets.npz"

# 2. Bring the export container back (it does not auto-start).
docker start ssl_labels_v2
sleep 5
docker ps --filter name=ssl_labels_v2 --format 'container: {{.Status}}'

# 3. Relaunch the supervisor: it relaunches the export in resume mode and, on
#    completion, converts to data/real-ssl-v2 and launches -query-realft-muon-v2.
mkdir -p logs
setsid nohup bash tools/watch_v2_then_train.sh >> logs/watch_v2_then_train.log 2>&1 &
sleep 90
tail -3 logs/watch_v2_then_train.log
docker exec ssl_labels_v2 pgrep -f save_instance_labels.py > /dev/null \
    && echo "export: RUNNING (resumed)" || echo "export: NOT RUNNING — check logs"
