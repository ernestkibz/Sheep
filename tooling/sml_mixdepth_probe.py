#!/usr/bin/env python3
"""Mixed-depth rows: read the rows that matter deeply, the rest coarsely.

Zeroing unimportant FFN rows fails on a SiLU model -- the oracle itself misses
the error budget, because the tail is small but not zero (see
sml_sparsity_probe.py). This is the graceful version, and it is only possible
because the ladder stores bit slices: a row can be read at depth 1 or depth 2
independently of its neighbours, so nothing is ever discarded.

    20% of rows at depth 2 (4.31 b/w) + 80% at depth 1 (2.31 b/w)
      = 2.71 b/w average, against 4.31 uniform -- 1.59x fewer FFN bytes

The question is whether that costs materially more error than uniform depth 2.
If it lands near depth-2 error, the FFN gets 1.6x cheaper for nothing.

Compared against the two uniform baselines at their true bit rates, so the
trade is visible rather than asserted.

    python tooling/sml_mixdepth_probe.py MODEL.ssf --ref HF_CHECKPOINT
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sml_common import Bpe, CALIB_TEXT, rms_norm         # noqa: E402
from sml_hf import open_weights                          # noqa: E402
from sml_read import SmlFile                             # noqa: E402
from sml_sparsity_probe import collect                   # noqa: E402


def unpermuted(f, name, depth):
    perm = f.perm(name)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return f.dequant(name, depth=depth)[:, inv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ssf")
    ap.add_argument("--ref", default=r"C:\Users\earne\models\Qwen2.5-3B-Instruct")
    ap.add_argument("--layers", default="0,17,35")
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()

    W = open_weights(args.ref)
    f = SmlFile(args.ssf)
    toks = W.field("tokenizer.tokens") or []
    merges = W.field("tokenizer.merges") or []
    ids = Bpe(toks, merges).encode(CALIB_TEXT)[:args.tokens]
    layers = [int(x) for x in args.layers.split(",")]

    print(f"{os.path.basename(args.ssf)}  |  {len(ids)} tokens, layers {layers}")
    data = collect(W, ids, set(layers))

    print()
    print(f"  {'layer':>6s} {'scheme':>22s} {'b/w':>6s} {'act err':>9s}")
    print("  " + "-" * 48)

    for L in layers:
        hf, act_ref = data[L]
        p = f"blk.{L}."

        g1 = unpermuted(f, p + "ffn_gate.weight", 1)
        u1 = unpermuted(f, p + "ffn_up.weight", 1)
        g2 = unpermuted(f, p + "ffn_gate.weight", 2)
        u2 = unpermuted(f, p + "ffn_up.weight", 2)

        def act_of(G, U):
            g = hf @ G.T
            u = hf @ U.T
            return (g / (1.0 + np.exp(-g))) * u

        def err(a):
            return float(np.sqrt(np.mean((a - act_ref) ** 2))
                         / np.sqrt(np.mean(act_ref ** 2)))

        a1, a2 = act_of(g1, u1), act_of(g2, u2)
        print(f"  {L:6d} {'uniform depth 1':>22s} {2.31:6.2f} {err(a1):9.4f}")
        print(f"  {L:6d} {'uniform depth 2':>22s} {4.31:6.2f} {err(a2):9.4f}")

        # Rows are ranked by how much they contribute across the calibration
        # tokens -- a static per-row choice, baked in at build time, not a
        # per-token decision. That matters: a per-token choice would need the
        # rows before knowing which to fetch.
        rank = np.mean(np.abs(act_ref), axis=0)
        order = np.argsort(-rank)
        for pct in (10, 20, 40):
            k = int(len(order) * pct / 100)
            deep = order[:k]
            G = g1.copy(); G[deep] = g2[deep]
            U = u1.copy(); U[deep] = u2[deep]
            bw = 2.31 + (4.31 - 2.31) * pct / 100
            print(f"  {L:6d} {f'{pct}% deep, rest coarse':>22s} {bw:6.2f} "
                  f"{err(act_of(G, U)):9.4f}")
        print()

    print("  Uniform depth 2 is the bar to match at 4.31 b/w. A mixed row that")
    print("  lands near it for fewer bits makes the FFN -- 79% of streamed")
    print("  bytes -- cheaper at no quality cost.")


if __name__ == "__main__":
    main()
