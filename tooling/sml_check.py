#!/usr/bin/env python3
"""Verify the C kernel against the format spec, on a real .ssf file.

sml/sml-check.c writes y = W.x at every read depth. This recomputes the same
thing independently from the format description and compares.

Four things have to agree for this to pass, and each is somewhere a mistake
still produces plausible-looking output rather than an obvious crash:

  * group-planar packing   -- bit-plane g is group g
  * two-level scales       -- scale[g] = d[g/8] * q8[g]
  * per-level shifts       -- levels are bit slices, so level k carries weight
                              2^(bits still unread below it)
  * the once-only bias     -- (midpoint - offset) must be added on the last
                              level read, not once per level

A bug in any of them would surface later as "the model seems a bit dumb", which
is the most expensive kind of bug to chase.

    python tooling/sml_check.py MODEL.ssf TENSOR out.bin
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sml_read import SmlFile   # noqa: E402


def main():
    if len(sys.argv) < 4:
        raise SystemExit("usage: sml_check.py MODEL.ssf TENSOR out.bin")
    path, name, binf = sys.argv[1], sys.argv[2], sys.argv[3]

    f = SmlFile(path)
    m = f.tensors[name]
    n = m["cols"]

    raw = np.fromfile(binf, dtype=np.int32, count=3)
    rows, nlev, n_chk = int(raw[0]), int(raw[1]), int(raw[2])
    if n_chk != n:
        raise SystemExit(f"width mismatch: C says {n_chk}, file says {n}")
    ys = np.fromfile(binf, dtype=np.float32, offset=12).reshape(nlev, rows)

    # Exactly the input sml-check.c builds.
    i = np.arange(n, dtype=np.float32)
    x = np.sin(0.001 * i) * (1.0 + 0.5 * (i.astype(np.int64) % 7))

    # The weights were quantized in permuted column order, so the activation
    # must be gathered the same way: (W P)(P^T x) = W x.
    perm = f.perm(name)
    xp = x[perm].astype(np.float32)

    # Match the C activation quantizer bit for bit, including its int8 rounding.
    G = m["group"]
    ng = n // G
    xg = xp.reshape(ng, G)
    amax = np.max(np.abs(xg), axis=1)
    xs = amax / 127.0
    inv = np.where(xs > 0, 1.0 / np.where(xs > 0, xs, 1), 0.0)
    xq = np.clip(np.rint(xg * inv[:, None]), -127, 127).astype(np.int32)

    scale = f.group_scales(name)          # (rows, ng)
    codes = f.codes(name)                 # (rows, n) full integer codes
    total = int(m.get("nbits") or sum(e["bits"] for e in m["lv"]))
    offset = (2 ** total - 1) / 2.0     # ladder only; flat carries a min instead

    print(f"{name}  rows={m['rows']} cols={n} group={G} levels={nlev}")
    print(f"{'depth':>6s} {'C |y|':>14s} {'python |y|':>14s} {'rel-RMS':>10s} {'max rel':>9s}")
    # A flat tensor stores the code whole and reconstructs it as
    # code*scale + min -- asymmetric, with no midpoint offset and no per-level
    # shift. Running it through the ladder branch below applies a symmetric
    # offset it never had and drops the min entirely, which is why this gate
    # reported MISMATCH on the shipping layout for as long as flat existed.
    flat = f.is_flat(name)
    mins = f.group_mins(name) if flat else None

    ok = True
    for depth in range(1, nlev + 1):
        if flat:
            w = codes[:rows].astype(np.float32) * np.repeat(scale[:rows], G, axis=1)
            if mins is not None:
                w = w + np.repeat(mins[:rows], G, axis=1)
        else:
            rem = total - depth * 2
            view = (codes >> rem) << rem
            if rem:
                view = view + (1 << (rem - 1))
            # dequantize row-block by row-block to stay inside RAM
            w = (view[:rows].astype(np.float32) - offset) * np.repeat(scale[:rows], G, axis=1)
        # the C side dots against the QUANTIZED activation, so we must too
        y = (w.reshape(rows, ng, G) * (xq[None] * xs[None, :, None])).sum(axis=(1, 2))

        c = ys[depth - 1].astype(np.float64)
        p = y.astype(np.float64)

        # rel-RMS is the metric that decides. Per-element max relative error is
        # reported but NOT gated on: both sides accumulate ~2048 fp32 terms, so
        # an output that lands near zero through cancellation has a huge relative
        # error while carrying no information. Gating on it would flag numerics
        # as a packing bug.
        rms = float(np.sqrt(np.mean(p * p)))
        rel_rms = float(np.sqrt(np.mean((c - p) ** 2)) / (rms + 1e-12))
        den = np.maximum(np.abs(p), 0.01 * rms)
        max_rel = float(np.max(np.abs(c - p) / den))

        good = rel_rms < 1e-4
        ok &= good
        print(f"{depth:6d} {np.linalg.norm(c):14.5f} {np.linalg.norm(p):14.5f} "
              f"{rel_rms:10.2e} {max_rel:9.2e} {'OK' if good else 'MISMATCH'}")

    print()
    if ok:
        print("PASS - C kernel and format spec agree at every read depth")
        print("       (packing, two-level scales, per-level shifts, once-only bias)")
    else:
        print("FAIL - kernel and format disagree; check packing, scales, shift, bias")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
