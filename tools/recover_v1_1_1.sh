#!/bin/bash
# recover_v1_1_1.sh — re-download v1.1.1 scenes recorded in the recovery manifest.
# Companion to dataset/v1.0.1-sample-every-four/script.sh (same SFTP host/creds).
#
# Usage: ./recover_v1_1_1.sh <deleted|rerun|full> [dest_dir]
#   deleted : re-fetch the subsampled-out parent-dir scenes (manifest["deleted"])
#   rerun   : re-fetch the deleted rerun_failed_cases scenes
#   full    : re-fetch ALL original parent-dir scene ids (restore full v1.1.1)
#   dest_dir defaults to /home/fai/workspace/jhp/dataset/v1.1.1
# Env: PYTHON (default python3), DRYRUN=1 (only build batch files, don't download).
set -euo pipefail

MODE="${1:?usage: recover_v1_1_1.sh <deleted|rerun|full> [dest_dir]}"
DEST="${2:-/home/fai/workspace/jhp/dataset/v1.1.1}"
MANIFEST="/home/fai/workspace/jhp/dataset/v1.1.1_recovery_manifest.json"
[ -f "$MANIFEST" ] || MANIFEST="/home/fai/workspace/jhp/LitePT/tools/v1.1.1_recovery_manifest.json"
[ -f "$MANIFEST" ] || { echo "manifest not found"; exit 1; }

HOST="192.168.0.200"
USER="junhyeok.park"
PASS='vkdlsejtm!984'
REMOTE_BASE="/datasets/fainders/vco_synthetic_3d_datasets/v1.1.1"
PYTHON="${PYTHON:-python3}"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# 1) Build one sftp batch file per target dir from the manifest's recorded ids.
#    Emits "<dir>\t<batchfile>\t<count>" lines for the runner loop.
mapfile -t TARGETS < <("$PYTHON" - "$MANIFEST" "$MODE" "$DEST" "$REMOTE_BASE" "$WORK" <<'PY'
import json, os, sys
manifest, mode, dest, remote_base, work = sys.argv[1:6]
m = json.load(open(manifest))
sel = {"deleted": ("parent", "deleted"),
       "full":    ("parent", "parent_dirs"),
       "rerun":   ("rerun",  "rerun_failed_cases")}
if mode not in sel:
    sys.exit(f"unknown mode: {mode}")
kind, key = sel[mode]
for d, v in m[key].items():
    ids = v["scene_ids"]
    if not ids:
        continue
    if kind == "parent":
        remote, localdir = f"{remote_base}/{d}", os.path.join(dest, d)
    else:
        remote = f"{remote_base}/rerun_failed_cases/{d}"
        localdir = os.path.join(dest, "rerun_failed_cases", d)
    os.makedirs(localdir, exist_ok=True)
    bf = os.path.join(work, f"batch_{kind}_{d}.sftp")
    todo = 0
    with open(bf, "w") as f:
        f.write(f"cd {remote}\nlcd {localdir}\n")
        for sid in ids:
            if not os.path.isdir(os.path.join(localdir, str(sid))):  # resume-friendly
                f.write(f"get -r {sid}\n"); todo += 1
        f.write("bye\n")
    print(f"{d}\t{bf}\t{todo}")
PY
)

echo "Recovery mode=$MODE -> $DEST  (${#TARGETS[@]} target dirs)"
for line in "${TARGETS[@]}"; do
    DIR="${line%%$'\t'*}"; rest="${line#*$'\t'}"
    BF="${rest%%$'\t'*}"; N="${rest##*$'\t'}"
    echo "=== dir $DIR : $N scenes to fetch ==="
    [ "${DRYRUN:-0}" = "1" ] && { echo "  (dry-run) batch: $BF"; continue; }
    [ "$N" = "0" ] && { echo "  nothing to fetch (all present)"; continue; }
    expect <<EOF
set timeout -1
spawn sftp -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -b $BF $USER@$HOST
expect {
    -re "(?i)password:" { send "$PASS\r"; exp_continue }
    eof
}
EOF
done
echo "Recovery ($MODE) complete."
