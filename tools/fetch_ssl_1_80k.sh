#!/usr/bin/env bash
#
# fetch_ssl_1_80k.sh — list-driven, resumable copy of the 80k SSL scenes from
# vco_dataset_2606/dataset_ssl_1 into the local dataset dir.
#
# Auth: the account on 192.168.0.200 accepts the password stored in the
# reference procedure (v1.0.1/script.sh) over the SFTP protocol, but plain
# ssh/rsync password auth is rejected on this host. So transfers use lftp
# (native SFTP, parallel + resumable). The password is read from the reference
# script into $LFTP_PW at runtime and is NEVER printed; it is passed to lftp via
# -u (env-var expansion, so it does not appear in this repo or the transcript).
#
# Throughput comes from running $NPROC parallel lftp workers, each mirroring a
# chunk of the id list, each worker mirroring $PTC files concurrently per scene.
#
# Usage:
#   tools/fetch_ssl_1_80k.sh smoke [N]   # transfer first N ids (default 200) to a temp dest
#   tools/fetch_ssl_1_80k.sh transfer    # full 80k transfer (resumable; safe to re-run)
#   tools/fetch_ssl_1_80k.sh verify      # count/diff copied scenes vs the list
set -uo pipefail

REMOTE_USER="junhyeok.park"
REMOTE_HOST="192.168.0.200"
REMOTE_BASE="/datasets/fainders/vco_dataset_2606/dataset_ssl_1"
LIST="/home/fai/workspace/jhp/dataset/SSL_dataset/dataset_ssl_1_subset/keep_80k.txt"
DEST="/home/fai/workspace/jhp/dataset/SSL_dataset/dataset_ssl_1_80k"
SCRIPT_REF="/home/fai/workspace/jhp/dataset/v1.0.1/script.sh"
LOGDIR="/home/fai/workspace/jhp/LitePT/logs"
WORKDIR="${LOGDIR}/ssl_1_80k_work"
# Throughput note: this host/link caps ~20 MB/s per connection and does NOT
# scale with more connections (measured: 3 workers were slower than 1 due to
# contention). So default to a single connection with a higher intra-scene
# parallel-transfer-count. Override with NPROC=/PTC= if the server changes.
NPROC="${NPROC:-1}"     # parallel lftp workers (1 is fastest on this host)
PTC="${PTC:-8}"         # files transferred concurrently within one scene mirror

mkdir -p "$LOGDIR"

# --- load password from the reference script into LFTP_PW (never echoed) ------
load_password() {
  [ -f "$SCRIPT_REF" ] || { echo "ERROR: reference script not found: $SCRIPT_REF" >&2; exit 2; }
  LFTP_PW="$(grep -A2 'expect "password:"' "$SCRIPT_REF" | grep -m1 'send "' | sed -E 's/.*send "(.*)\\r".*/\1/')"
  export LFTP_PW
  [ -n "${LFTP_PW:-}" ] || { echo "ERROR: failed to extract password from $SCRIPT_REF" >&2; exit 2; }
  echo "[auth] password loaded (len=${#LFTP_PW})"
}

# generate an lftp command file that mirrors every id in $1 into $2/<id>
gen_chunk_script() {   # $1=chunk_list  $2=out_script  $3=dest_root
  local list="$1" out="$2" dest="$3"
  {
    echo "set sftp:auto-confirm yes"
    echo "set net:timeout 20"
    echo "set net:max-retries 5"
    echo "set net:reconnect-interval-base 5"
    echo "set net:reconnect-interval-max 30"
    echo "set mirror:parallel-transfer-count ${PTC}"
    echo "set xfer:clobber on"
    while read -r id; do
      [ -n "$id" ] && echo "mirror -c --no-perms --no-umask ${REMOTE_BASE}/${id} ${dest}/${id}"
    done < "$list"
    echo "bye"
  } > "$out"
}

# split $1 into NPROC chunks and run one lftp worker per chunk into $2
run_transfer() {   # $1=id_list  $2=dest_root  $3=tag
  local list="$1" dest="$2" tag="$3"
  local wd="${WORKDIR}/${tag}"
  mkdir -p "$dest" "$wd"
  rm -f "$wd"/chunk_*
  local total; total=$(grep -c . "$list")
  local n=$NPROC; [ "$total" -lt "$n" ] && n=$total; [ "$n" -lt 1 ] && n=1
  split -n "l/${n}" -d "$list" "$wd/chunk_"
  load_password
  echo "[transfer:$tag] $(date '+%F %T') $total ids, $n workers x ${PTC} files -> $dest"
  local pids=() cf sf lg
  for cf in "$wd"/chunk_*; do
    [ -f "$cf" ] && [[ "$cf" != *.lftp && "$cf" != *.log ]] || continue
    sf="${cf}.lftp"; lg="${cf}.log"
    gen_chunk_script "$cf" "$sf" "$dest"
    lftp -u "${REMOTE_USER},${LFTP_PW}" -e "source $sf" "sftp://${REMOTE_HOST}" > "$lg" 2>&1 &
    pids+=($!)
  done
  echo "[transfer:$tag] launched ${#pids[@]} workers (pids: ${pids[*]})"
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "[transfer:$tag] $(date '+%F %T') all workers finished (agg rc=$rc)"
  return 0
}

smoke() {
  local n="${1:-200}"
  local slist="${WORKDIR}/smoke_ids.txt"
  local sdest="/home/fai/workspace/jhp/dataset/SSL_dataset/.smoke_ssl_1"
  mkdir -p "$WORKDIR"; rm -rf "$sdest"
  head -n "$n" "$LIST" > "$slist"
  local t0; t0=$(date +%s)
  run_transfer "$slist" "$sdest" "smoke" 2>&1 | tee -a "${LOGDIR}/fetch_ssl_1_80k.log"
  local t1; t1=$(date +%s)
  local got; got=$(ls -1 "$sdest" 2>/dev/null | wc -l)
  local bytes; bytes=$(du -sb "$sdest" 2>/dev/null | cut -f1)
  local secs=$((t1 - t0))
  echo "[smoke] copied $got/$n scenes, ${bytes} bytes, in ${secs}s"
  awk -v s="$secs" -v b="$bytes" -v g="$got" 'BEGIN{
    if(g>0 && s>0){ rate=g/s; printf "[smoke] %.1f scenes/s -> ETA for 80000: %.1f min\n", rate, (80000/rate)/60;
                    printf "[smoke] %.2f MB/s\n", (b/1048576)/s }}'
}

transfer() {
  run_transfer "$LIST" "$DEST" "full" 2>&1 | tee -a "${LOGDIR}/fetch_ssl_1_80k.log"
  verify
}

verify() {
  echo "[verify] $(date '+%F %T')"
  local want got
  want=$(sort -u "$LIST" | grep -c .)
  got=$(ls -1 "$DEST" 2>/dev/null | wc -l)
  echo "[verify] wanted ids        : $want"
  echo "[verify] dirs in dest       : $got"
  local missing miss_n
  missing="$(comm -23 <(sort -u "$LIST") <(ls -1 "$DEST" 2>/dev/null | sort -u))"
  miss_n=$(printf '%s' "$missing" | grep -c . || true)
  echo "[verify] missing ids        : $miss_n"
  if [ "$miss_n" -ne 0 ]; then
    printf '%s\n' "$missing" > "${LOGDIR}/missing_ids.txt"
    printf '%s\n' "$missing" | head -10
    echo "[verify] full missing list -> ${LOGDIR}/missing_ids.txt"
  fi
  local empties
  empties=$(find "$DEST" -mindepth 1 -maxdepth 1 -type d -empty 2>/dev/null | wc -l)
  echo "[verify] empty scene dirs   : $empties"
  echo "[verify] total size on disk : $(du -sh "$DEST" 2>/dev/null | cut -f1)"
}

case "${1:-}" in
  smoke)    smoke "${2:-200}" ;;
  transfer) transfer ;;
  verify)   verify ;;
  *) echo "usage: $0 {smoke [N]|transfer|verify}" >&2; exit 1 ;;
esac
