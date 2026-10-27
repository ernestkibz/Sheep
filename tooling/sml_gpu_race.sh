#!/usr/bin/env bash
# GPU decode race, temperature-gated. The laptop 3050 loses about two thirds of
# its speed between 79 and 89 C, and the reference engine throttles with it, so
# EVERY run, ours and theirs, starts only once the GPU is back under COOL degrees
# (or after a 300 s cap), and the start temperature is printed beside each number.
#
#   bash tooling/sml_gpu_race.sh SPEC [COOL_C]
#
# SPEC lives outside the repo. First line "bench PATH" names the reference CUDA
# bench binary; then one line per model, files in C:/Users/earne/models:
#   OURS.ssf THEIRS_FILE ids,comma,separated
#
# Refuses to measure while Chrome holds the GPU -- it keeps using it after its
# window closes (tray icon -> Exit). Waits up to 45 minutes for it to leave.
set -u
cd "$(dirname "$0")/../sml"
M=C:/Users/earne/models
SPEC="$1"
COOL="${2:-70}"
B=$(awk '$1 == "bench" { print $2 }' "$SPEC")
mapfile -t MODELS < <(grep -v "^bench" "$SPEC" | grep -v "^ *$")

t=$(date +%s)
until ! nvidia-smi | grep -qi chrome || [ $(( $(date +%s) - t )) -gt 2700 ]; do sleep 15; done
if nvidia-smi | grep -qi chrome; then echo "ABORTED: Chrome still holds the GPU after 45 min"; exit 1; fi
echo "GPU clear of Chrome at $(date +%H:%M:%S)"

cool () { t0=$(date +%s); until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le "$COOL" ] || [ $(( $(date +%s) - t0 )) -gt 300 ]; do sleep 5; done; nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader; }
ours () { T=$(cool); printf "  start %3s C  %-6s %-22s " "$T" "$1" "$3"; SML_FUSE2=$2 SPEED_ONLY=1 ./sml-gpu-fwd.exe "$M/$3" "$4" 0 </dev/null 2>&1 | grep -oE "[0-9.]+ tok/s, [0-9]+ launches" || echo "failed"; }
theirs () { T=$(cool); printf "  start %3s C  %-6s %-22s " "$T" "ref" "${1:0:22}"; "$B" -m "$M/$1" -ngl 99 -p 0 -n 64 -r 1 </dev/null 2>&1 | grep tg64 | awk -F"|" "{ print \$(NF-1) }"; }

for round in 1 2; do
  echo "--- round $round"
  for line in "${MODELS[@]}"; do
    set -- $line
    if [ "$round" = 1 ]; then ours fuse2 1 "$1" "$3"; ours dp4a 0 "$1" "$3"; theirs "$2"
    else theirs "$2"; ours dp4a 0 "$1" "$3"; ours fuse2 1 "$1" "$3"; fi
  done
done
echo "--- end"
