#!/usr/bin/env python3
"""Count where an SSF file's bytes actually go, before trying to remove any.

Bytes are the one axis this engine still loses on, and every previous attempt to
fix that was aimed at a guess. This counts: codes, per-group scales, per-group
mins, superblock scales, permutations, and everything else, as bits per weight.

    python tooling/sml_bytes.py MODEL.ssf

Bits per weight is the comparable unit -- a file size difference between two
formats is mostly a statement about which tensors each chose to widen.
"""
import argparse
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sml_read import SmlFile      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ssf")
    ap.add_argument("--per-tensor", action="store_true",
                    help="also list the widest tensors")
    args = ap.parse_args()

    f = SmlFile(args.ssf)
    total_file = os.path.getsize(args.ssf)

    cat = defaultdict(int)
    weights = 0
    by_bits = defaultdict(lambda: [0, 0])      # nbits -> [weights, code bytes]
    rows = []

    for name, m in f.tensors.items():
        if m.get("kind") != "ladder" or not m.get("lv"):
            continue
        r, c, g = m["rows"], m["cols"], m["group"]
        w = r * c
        weights += w
        ng = c // g
        nbits = m.get("nbits") or m["lv"][0]["bits"]

        # Every *_n field in the record is a BYTE COUNT, not an element count.
        # Multiplying by the element size again inflated scales 2x and the
        # permutation 4x, and the components then summed to 2027 MB against a
        # 1973 MB file. Components that exceed the file are the tell.
        codes = sum(e["codes_n"] for e in m["lv"])
        sd = m.get("sd_n", 0)                         # fp16 superblock scales
        sq = m.get("sq_n", 0)                         # uint8 per-group scales
        md = m.get("md_n", 0)                         # fp16 superblock mins
        mq = m.get("mq_n", 0)                         # uint8 per-group mins
        perm = m.get("perm_n", 0)                     # int32 column permutation

        cat["codes"] += codes
        cat["group scales (uint8)"] += sq
        cat["group mins (uint8)"] += mq
        cat["superblock scales (fp16)"] += sd
        cat["superblock mins (fp16)"] += md
        cat["column permutation (int32)"] += perm

        by_bits[nbits][0] += w
        by_bits[nbits][1] += codes
        rows.append((name, w, (codes + sd + sq + md + mq + perm) * 8.0 / w, nbits))

    accounted = sum(cat.values())
    cat["header, names, vocab, other"] += total_file - accounted
    if accounted > total_file:
        print("  !! components exceed the file by "
              f"{(accounted - total_file) / 1e6:.1f} MB -- the count is WRONG")

    print(f"\n  {os.path.basename(args.ssf)}")
    print(f"  {total_file / 1e6:.1f} MB, {weights / 1e6:.1f}M quantized weights\n")
    print(f"  {'component':<32}{'MB':>9}{'bits/weight':>14}{'share':>9}")
    for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32}{v / 1e6:9.1f}{v * 8.0 / weights:14.4f}"
              f"{100.0 * v / total_file:8.1f}%")
    print(f"  {'-' * 62}")
    print(f"  {'TOTAL':<32}{total_file / 1e6:9.1f}{total_file * 8.0 / weights:14.4f}"
          f"{100.0:8.1f}%")

    print(f"\n  by code width:")
    for nb in sorted(by_bits):
        w, cb = by_bits[nb]
        print(f"    {nb}-bit  {w / 1e6:7.1f}M weights ({100.0 * w / weights:4.1f}%)"
              f"   {cb / 1e6:7.1f} MB of codes")

    # The comparison that matters: per-group metadata is where the two formats
    # actually differ, so state it in the unit the other format is specified in.
    meta = (cat["group scales (uint8)"] + cat["group mins (uint8)"]
            + cat["superblock scales (fp16)"] + cat["superblock mins (fp16)"])
    print(f"\n  per-group metadata   {meta * 8.0 / weights:.4f} bits/weight")
    print(f"  (uint8 scale + uint8 min per 32 = 16 bits, plus two fp16 per 8"
          f" groups = 4 bits -> 0.625 b/w by construction)")
    print(f"  Q4_K for comparison: 6-bit scale + 6-bit min per 32, two fp16 per")
    print(f"  256 -> 0.500 b/w. The gap is 0.125 b/w, which is 2.8% of a 4.5 b/w")
    print(f"  format -- and it is the entire size deficit.")

    if args.per_tensor:
        rows.sort(key=lambda t: -t[2])
        print(f"\n  {'widest tensors':<34}{'Mw':>8}{'b/w':>8}{'nbits':>7}")
        for name, w, bw, nb in rows[:12]:
            print(f"  {name:<34}{w / 1e6:8.2f}{bw:8.3f}{nb:7d}")


if __name__ == "__main__":
    main()
