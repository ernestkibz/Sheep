#!/usr/bin/env bash
# Settle a tuned GPU configuration: default vs tuned vs the reference engine,
# two rounds in mirrored order, then the tuned path graded against the CPU engine
# on every model -- the tuner grades it on one.
#
#   bash tooling/sml_gpu_settle.sh SPEC "SML_FUSE2=1 SML_GRID=48 ..." [COOL_C]
#
# SPEC is the race spec (outside the repo, same format as sml_gpu_race.sh):
# first line "bench PATH" names the reference CUDA bench binary, then one line
# per model, files in C:/Users/earne/models:  OURS.ssf THEIRS_FILE ids,comma,separated
#
# COOL_C (default 99, i.e. no wait) starts each run only once the GPU is back
# under that temperature, 10 minute cap. Written after the autotuner "won" 1.03x
# on a GPU throttled to ~11 tok/s at 90 C: at the heat floor every configuration
# runs at the same clamped speed and the race measures the throttle.
set -u
cd "$(dirname "$0")/../sml"
M=C:/Users/earne/models
SPEC="$1"
TUNED="$2"
COOL="${3:-99}"
B=$(awk '$1 == "bench" { print $2 }' "$SPEC")
mapfile -t MODELS < <(grep -v "^bench" "$SPEC" | grep -v "^ *$")

temp () { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader; }
cool () {
  t0=$(date +%s)
  until [ "$(temp)" -le "$COOL" ] || [ $(( $(date +%s) - t0 )) -gt 600 ]; do sleep 5; done
  temp
}
ours () {
  T=$(cool)
  printf "  start %3s C  %-7s %-22s " "$T" "$1" "$3"
  env $2 SPEED_ONLY=1 ./sml-gpu-fwd.exe "$M/$3" "$4" 0 </dev/null 2>&1 \
    | grep -oE "[0-9.]+ tok/s, [0-9]+ launches" || echo "failed"
}
theirs () {
  T=$(cool)
  printf "  start %3s C  %-7s %-22s " "$T" "ref" "${1:0:22}"
  "$B" -m "$M/$1" -ngl 99 -p 0 -n 64 -r 1 </dev/null 2>&1 | grep tg64 | awk -F"|" "{ print \$(NF-1) }"
}

echo "tuned: $TUNED"
echo "GPU now $(temp) C, runs start at or below $COOL C"
for round in 1 2; do
  echo "--- round $round"
  for line in "${MODELS[@]}"; do
    set -- $line
    if [ "$round" = 1 ]; then
      ours default "" "$1" "$3"; ours tuned "$TUNED" "$1" "$3"; theirs "$2"
    else
      theirs "$2"; ours tuned "$TUNED" "$1" "$3"; ours default "" "$1" "$3"
    fi
  done
done
echo "--- tuned path graded against the CPU engine"
for line in "${MODELS[@]}"; do
  set -- $line
  printf "  %-22s " "$1"
  env $TUNED ./sml-gpu-fwd.exe "$M/$1" "$3" 0 </dev/null 2>&1 \
    | grep -oE "top-1 agree [0-9]+/[0-9]+|mean logit corr [0-9.]+|tokens (identical to|DIFFER FROM)" | tr "\n" " "
  echo
done
echo "--- end"
