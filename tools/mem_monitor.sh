#!/bin/sh
# Samples host + GPU memory every 5s. Logs: epoch_time, MemAvailable(MB),
# train-proc RSS(MB), GPU used(MB). Run alongside a training job for leak diagnosis.
OUT="${1:-logs/mem_monitor.log}"
PAT="${2:-tools/train.py}"
echo "ts,mem_avail_mb,train_rss_mb,gpu_used_mb" > "$OUT"
while true; do
  TS=$(date +%H:%M:%S)
  AVAIL=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  RSS=$(ps -eo rss,args | grep -F "$PAT" | grep -v grep | awk '{s+=$1} END{printf "%d", s/1024}')
  GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$TS,$AVAIL,${RSS:-0},${GPU:-NA}" >> "$OUT"
  sleep 5
done
