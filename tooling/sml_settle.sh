#!/usr/bin/env bash
# settle.sh with the reference model as an argument, so the same method --
# randomized order, reference at its own best -t 3, bootstrap CI on the median
# ratio -- races any model rather than only the 3B it was written for.
#
#   bash settle2.sh OURS.ssf REPS THEIRS.gguf
set -u
cd "/c/Users/earne/Downloads/Stallion v2/sheep"

SS="${1:?ours.ssf}"
REPS="${2:-12}"
GG="${3:?theirs.gguf}"

run_l() { ./build-ctl/bin/sheep-bench.exe -m "$GG" -p 0 -n 48 -t 3 -r 1 -o csv 2>/dev/null \
          | tail -1 | awk -F, '{print $(NF-1)}' | tr -d '"'; }
run_s() { ./sml/sml-cli.exe "$SS" --gen 48 --threads 6 2>&1 \
          | grep -oE '^  [0-9]+\.[0-9]+ tok/s' | grep -oE '[0-9]+\.[0-9]+'; }

echo "=== warm both, discard ==="
run_l >/dev/null; run_s >/dev/null

L=""; S=""
for i in $(seq 1 "$REPS"); do
  if [ $((RANDOM % 2)) -eq 0 ]; then
    l=$(run_l); s=$(run_s)
  else
    s=$(run_s); l=$(run_l)
  fi
  L="$L $l"; S="$S $s"
  printf "rep%-3d llama %-8s sheep %s\n" "$i" "$l" "$s"
done

python - "$L" "$S" <<'PY'
import sys, random
def q(v, p):
    v = sorted(v); k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)
l = [float(x) for x in sys.argv[1].split()]
s = [float(x) for x in sys.argv[2].split()]
print()
print(f"  {'engine':<12}{'p10':>8}{'median':>9}{'p90':>8}{'n':>5}")
print(f"  {'llama.cpp':<12}{q(l,.1):8.2f}{q(l,.5):9.2f}{q(l,.9):8.2f}{len(l):5d}")
print(f"  {'Sheep':<12}{q(s,.1):8.2f}{q(s,.5):9.2f}{q(s,.9):8.2f}{len(s):5d}")
print(f"  median ratio      {q(s,.5)/q(l,.5):.4f}x")
random.seed(0)
rs = []
for _ in range(20000):
    bl = [random.choice(l) for _ in l]
    bs = [random.choice(s) for _ in s]
    rs.append(q(bs, .5) / q(bl, .5))
rs.sort()
lo, hi = rs[int(.025 * len(rs))], rs[int(.975 * len(rs))]
print(f"  median ratio 95% CI   [{lo:.4f}, {hi:.4f}]   (bootstrap, 20k resamples)")
if lo > 1.0:
    print("  VERDICT: real win -- the interval excludes 1.0")
elif hi < 1.0:
    print("  VERDICT: real loss -- the interval excludes 1.0")
else:
    print("  VERDICT: not settled -- the interval still contains 1.0")
PY
