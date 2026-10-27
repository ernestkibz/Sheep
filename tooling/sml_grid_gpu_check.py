"""The GPU searches against the numpy ones on a real tensor: time, the weighted
error of the codes each produces, and how many fields agree.

    python tooling/sml_grid_gpu_check.py CHECKPOINT_DIR [tensor] [cuda|cpu]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sml_grid_gpu as G  # noqa: E402
import sml_quantize as Q  # noqa: E402
from sml_hf import open_weights  # noqa: E402


def codes_err(wp, eff_s, eff_m, group, nmax, imp):
    S = np.repeat(eff_s, group, axis=1)
    M = np.repeat(eff_m, group, axis=1)
    c = np.clip(np.rint((wp - M) / np.maximum(S, 1e-12)), 0, nmax)
    return float(np.sqrt(np.mean(((c * S + M - wp) ** 2) * imp)) / np.sqrt(np.mean(wp ** 2 * imp)))


def main():
    src = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "blk.10.ffn_down.weight"
    dev = sys.argv[3] if len(sys.argv) > 3 else (G.available() or "cpu")
    w = open_weights(src).get(name).astype(np.float32)
    rows, cols = w.shape
    group, nbits, gps = 32, 4, Q.SCALE_SUPER // 32
    rng = np.random.default_rng(0)
    imp = (rng.random(cols).astype(np.float32) ** 2) + 0.1
    perm = np.argsort(-imp).astype(np.int32)
    wp = np.ascontiguousarray(w[:, perm])
    impp = imp[perm]
    print(f"{name} {w.shape} ({w.size / 1e6:.1f}M weights), searches on {dev}", flush=True)
    res = {}
    os.environ["SML_GRID_GPU"] = "0"           # the numpy row is numpy: no fast path behind it
    os.environ["SML_GRID_C"] = "0"
    import sml_grid_c as C
    ways = [("numpy", Q.Quantizer.asym_grid, Q.Quantizer.field_search)]
    if C.available():
        ways.append(("c", C.asym_grid, C.field_search))
    if dev != "none":
        ways.append((dev, lambda *a: G.asym_grid(*a, dev=dev), lambda *a: G.field_search(*a, dev=dev)))
    for label, grid, field in ways:
        t0 = time.time()
        mins, steps = grid(wp, group, nbits, nbits, impp)
        t1 = time.time()
        d16, q8, _ = Q.Quantizer.two_level(steps, gps, signed=False, bits=6)
        md16, mq8, _ = Q.Quantizer.two_level(mins, gps, signed=True, bits=6, neg=True)
        q8b, mq8b, es, em = field(wp, d16, q8, md16, mq8, group, gps, 15, 63, 63, 63, impp, 2)
        t2 = time.time()
        res[label] = (q8b, mq8b, codes_err(wp, es, em, group, 15, impp))
        print(f"  {label:6} grid {t1 - t0:6.2f} s   field {t2 - t1:6.2f} s   weighted rel-RMS {res[label][2]:.6f}",
              flush=True)
    a = res["numpy"]
    for label, b in res.items():
        if label == "numpy":
            continue
        same = float(np.mean((a[0] == b[0]) & (a[1] == b[1])))
        print(f"  {label}: fields identical {100 * same:.2f}%, error ratio to numpy {b[2] / a[2]:.5f}")


if __name__ == "__main__":
    main()
