#!/usr/bin/env python3
"""The quantization gate: 4.3 bits/weight at under 5% output error.

This is the number everything is blocked on. A ladder level that misses it
compounds to noise over 36 layers, which takes the depth gate and
self-speculation down with it.

INDEPENDENT of the build on purpose:

  * fresh text, not the calibration corpus -- GPTQ fits a Hessian to specific
    activations, and scoring on those showed 4.80x where held-out showed 0.87x
  * activations come from the SOURCE model, so the inputs are the ones the real
    model would see rather than ones the quantized model produced
  * bits/weight is counted from the FILE, not from the config that was requested

    python tooling/sml_quality_gate.py MODEL.ssf --ref HF_CHECKPOINT
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sml_common import Bpe, rms_norm, rope, bias   # noqa: E402
from sml_hf import open_weights                  # noqa: E402
from sml_read import SmlFile                     # noqa: E402

THRESHOLD = 0.05      # where a 36-layer model stops compounding into noise

# Deliberately not the calibration corpus.
HOLDOUT_TEXT = """Marie Curie was awarded Nobel Prizes in two different sciences.
def quicksort(items):
    if len(items) <= 1:
        return items
    pivot = items[len(items) // 2]
    return (quicksort([i for i in items if i < pivot])
            + [i for i in items if i == pivot]
            + quicksort([i for i in items if i > pivot]))
The Amazon basin produces a substantial share of the world's freshwater discharge.
A transformer replaces recurrence with attention over the entire input sequence.
Bread rises because yeast releases carbon dioxide as it consumes the sugars.
The judge ruled that the evidence had been obtained without a valid warrant.
UPDATE inventory SET quantity = quantity - 1 WHERE sku = 'AX-4410' AND quantity > 0;
Mount Everest sits on the border between Nepal and the Tibet Autonomous Region.
Convection carries heat upward through the mantle over geological timescales.
He wrote that the argument rested on an assumption nobody had bothered to check.
"""


def rel(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-12))


def capture_inputs(W, ids, layers_wanted):
    """Forward the SOURCE model, keeping the inputs each linear actually sees."""
    a = str(W.field("general.architecture"))
    n_layer = int(W.field(f"{a}.block_count"))
    n_embd = int(W.field(f"{a}.embedding_length"))
    n_head = int(W.field(f"{a}.attention.head_count"))
    n_kv = int(W.field(f"{a}.attention.head_count_kv", n_head))
    eps = float(W.field(f"{a}.attention.layer_norm_rms_epsilon", 1e-6))
    theta = float(W.field(f"{a}.rope.freq_base", 10000.0))
    hd = n_embd // n_head

    emb = W.get("token_embd.weight")
    x = emb[np.asarray(ids)].astype(np.float32).copy()
    T = len(ids)
    pos = np.arange(T, dtype=np.float32)
    caught = {}

    for L in range(n_layer):
        p = f"blk.{L}."
        h = rms_norm(x, W.get(p + "attn_norm.weight"), eps)
        if L in layers_wanted:
            caught[p + "attn_q.weight"] = h.copy()
            caught[p + "attn_k.weight"] = h.copy()

        q = h @ W.get(p + "attn_q.weight").T + bias(W, p + "attn_q.bias")
        k = h @ W.get(p + "attn_k.weight").T + bias(W, p + "attn_k.bias")
        v = h @ W.get(p + "attn_v.weight").T + bias(W, p + "attn_v.bias")
        q = rope(q.reshape(T, n_head, hd), pos, theta)
        k = rope(k.reshape(T, n_kv, hd), pos, theta)
        v = v.reshape(T, n_kv, hd)
        rep = n_head // n_kv
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)

        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd)
        att = att + np.triu(np.full((T, T), -1e30, dtype=np.float32), 1)[None]
        att -= att.max(axis=-1, keepdims=True)
        att = np.exp(att)
        att /= att.sum(axis=-1, keepdims=True)
        o = np.einsum("hts,shd->thd", att, v).reshape(T, n_embd)
        if L in layers_wanted:
            caught[p + "attn_output.weight"] = o.copy()
        x = x + o @ W.get(p + "attn_output.weight").T

        h = rms_norm(x, W.get(p + "ffn_norm.weight"), eps)
        if L in layers_wanted:
            caught[p + "ffn_gate.weight"] = h.copy()
            caught[p + "ffn_up.weight"] = h.copy()
        g = h @ W.get(p + "ffn_gate.weight").T
        u = h @ W.get(p + "ffn_up.weight").T
        act = (g / (1.0 + np.exp(-g))) * u
        if L in layers_wanted:
            caught[p + "ffn_down.weight"] = act.copy()
        x = x + act @ W.get(p + "ffn_down.weight").T

    return caught


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ssf")
    ap.add_argument("--ref",
                    default=r"C:\Users\earne\models\Qwen2.5-3B-Instruct",
                    help="full-precision checkpoint. Supplies BOTH the activations each linear is scored on and the ground truth compared against. It must not be quantized: a 4-bit reference is already 5.31%% from fp16, so scoring against one hides the real error and can report a pass that does not exist.")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--layers", default="0,8,17,26,35")
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()

    f = SmlFile(args.ssf)
    # activations and ground truth both come from full precision
    W = R = open_weights(args.ref)
    layers = [int(x) for x in args.layers.split(",")]

    # bits/weight counted from the file, not from what was requested
    total_w = sum(m["rows"] * m["cols"] for m in f.tensors.values()
                  if m["kind"] == "ladder")
    bits = f.bytes_at_depth(args.depth) * 8.0 / total_w
    print(f"{os.path.basename(args.ssf)}")
    print(f"  depth {args.depth}: {bits:.2f} bits/weight, "
          f"{f.bytes_at_depth(args.depth) / 1e9:.3f} GB/token")
    print(f"  gptq={f.head.get('gptq')}  calib_samples={f.head.get('calib_samples')}  "
          f"lossy_source={f.lossy_source}")
    print(f"  reference: {os.path.basename(args.ref)} (full precision)")
    if f.lossy_source:
        print("  WARNING: built from a quantized source, so its own ceiling is")
        print("           above the gate regardless of what follows")

    toks = W.field("tokenizer.tokens") or []
    merges = W.field("tokenizer.merges") or []
    ids = Bpe(toks, merges).encode(HOLDOUT_TEXT)[:args.tokens]
    print(f"  holdout: {len(ids)} tokens of text NOT in the calibration corpus\n")

    wanted = set(layers)
    caught = capture_inputs(W, ids, wanted)

    print(f"  {'tensor':30s} {'out err':>9s}")
    print("  " + "-" * 41)
    errs = []
    for name in sorted(caught, key=lambda n: (int(n.split('.')[1]), n)):
        X = caught[name]
        src = np.asarray(R.get(name), dtype=np.float32)
        perm = f.perm(name)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        got = f.dequant(name, depth=args.depth)[:, inv]
        e = rel(got @ X.T, src @ X.T)
        errs.append(e)
        flag = "" if e < THRESHOLD else "  OVER"
        print(f"  {name[:30]:30s} {e:9.5f}{flag}")

    mean = float(np.mean(errs))
    worst = float(np.max(errs))
    print("  " + "-" * 41)
    print(f"  {'mean':30s} {mean:9.5f}")
    print(f"  {'worst':30s} {worst:9.5f}")
    print()
    ok = mean < THRESHOLD
    print(f"GATE: {bits:.2f} b/w at {mean * 100:.2f}% mean output error "
          f"(threshold {THRESHOLD * 100:.0f}%)")
    if ok:
        print(f"PASS - depth {args.depth} should survive 36 layers of compounding")
    else:
        print(f"FAIL - {mean * 100:.2f}% compounds into noise; the ladder does not pay yet")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
