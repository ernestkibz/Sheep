#!/usr/bin/env python3
"""Can FFN activity be predicted before the weights that reveal it are read?

The FFN is 79% of the bytes a token moves, and most of its neurons contribute
almost nothing to any given token. If the dead ones can be identified from the
layer input alone, their rows never cross the bus -- and unlike the ladder, this
does not depend on the coarse level becoming usable.

The catch that makes or breaks it: `act = silu(W_gate x) * (W_up x)`, so knowing
which entries of `act` are small normally requires reading W_gate and W_up --
the very bytes we are trying to skip. So the predictor must work from `x` alone,
and be small enough that its own bytes do not eat the saving.

Measured here, in order:

  1. How concentrated is `act`? If the top 20% of neurons do not carry most of
     the output, there is nothing to skip and the idea is dead.
  2. Does a cheap low-rank predictor from `x` find them? Compared against an
     ORACLE that knows the true magnitudes -- the oracle bounds what any
     predictor could achieve, so if the oracle is not good enough, stop.
  3. What does the resulting output error cost, against the ~5% that a
     36-layer model tolerates?

    python tooling/sml_sparsity_probe.py --ref HF_CHECKPOINT
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sml_common import Bpe, CALIB_TEXT, rms_norm, rope, bias   # noqa: E402
from sml_hf import open_weights                          # noqa: E402


def collect(W, ids, layers):
    """Run the model, keeping (ffn input, activation) for the chosen layers."""
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
    out = {}

    for L in range(n_layer):
        p = f"blk.{L}."
        h = rms_norm(x, W.get(p + "attn_norm.weight"), eps)
        q = h @ W.get(p + "attn_q.weight").T + bias(W, p + "attn_q.bias")
        k = h @ W.get(p + "attn_k.weight").T + bias(W, p + "attn_k.bias")
        v = h @ W.get(p + "attn_v.weight").T + bias(W, p + "attn_v.bias")
        q = rope(q.reshape(T, n_head, hd), pos, theta)
        k = rope(k.reshape(T, n_kv, hd), pos, theta)
        v = v.reshape(T, n_kv, hd)
        rep = n_head // n_kv
        k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        att = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd)
        att = att + np.triu(np.full((T, T), -1e30, dtype=np.float32), 1)[None]
        att -= att.max(axis=-1, keepdims=True)
        att = np.exp(att)
        att /= att.sum(axis=-1, keepdims=True)
        o = np.einsum("hts,shd->thd", att, v).reshape(T, n_embd)
        x = x + o @ W.get(p + "attn_output.weight").T

        hf = rms_norm(x, W.get(p + "ffn_norm.weight"), eps)
        g = hf @ W.get(p + "ffn_gate.weight").T
        u = hf @ W.get(p + "ffn_up.weight").T
        act = (g / (1.0 + np.exp(-g))) * u
        if L in layers:
            out[L] = (hf.copy(), act.copy())
        x = x + act @ W.get(p + "ffn_down.weight").T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=r"C:\Users\earne\models\Qwen2.5-3B-Instruct")
    ap.add_argument("--layers", default="0,17,35")
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--rank", type=int, default=64)
    args = ap.parse_args()

    W = open_weights(args.ref)
    toks = W.field("tokenizer.tokens") or []
    merges = W.field("tokenizer.merges") or []
    ids = Bpe(toks, merges).encode(CALIB_TEXT)[:args.tokens]
    layers = [int(x) for x in args.layers.split(",")]
    print(f"collecting {len(ids)} tokens, layers {layers}")
    data = collect(W, ids, set(layers))

    print()
    print("[1] how concentrated is the activation?")
    print(f"    {'layer':>6s} {'n_ff':>7s} " + " ".join(f"{f'top{p}%':>8s}" for p in (10, 20, 40)))
    for L in layers:
        hf, act = data[L]
        mag = np.abs(act)
        share = []
        for pct in (10, 20, 40):
            k = int(act.shape[1] * pct / 100)
            idx = np.argsort(-mag, axis=1)[:, :k]
            kept = np.take_along_axis(act, idx, axis=1)
            share.append(float((kept ** 2).sum() / (act ** 2).sum()))
        print(f"    {L:6d} {act.shape[1]:7d} " + " ".join(f"{s:8.3f}" for s in share))
    print("    (fraction of total activation energy in the top-k neurons per token)")

    print()
    print("[2] can a rank-%d predictor from x find them? oracle bounds it." % args.rank)
    print(f"    {'layer':>6s} {'keep':>6s} {'oracle err':>11s} {'pred err':>10s} {'recall':>8s}")
    for L in layers:
        hf, act = data[L]
        n_ff = act.shape[1]
        mag = np.abs(act)

        # Low-rank predictor of |act| from the layer input, least squares.
        # Trained on the first half, scored on the second -- a predictor scored
        # on its own training data measures memorisation, which this project has
        # already been burned by once.
        h = len(hf) // 2
        Xtr, Ytr = hf[:h], mag[:h]
        Xte, Yte = hf[h:], mag[h:]
        U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
        r = min(args.rank, len(S))
        # regularised pseudo-inverse solve: B maps x -> predicted |act|
        Sr = np.diag(1.0 / (S[:r] + 1e-3))
        B = Vt[:r].T @ Sr @ U[:, :r].T @ Ytr
        Pte = Xte @ B

        for pct in (20, 40):
            k = int(n_ff * pct / 100)
            true_idx = np.argsort(-np.abs(Yte), axis=1)[:, :k]
            pred_idx = np.argsort(-np.abs(Pte), axis=1)[:, :k]

            # oracle: keep the truly-largest k, zero the rest
            oracle = np.zeros_like(Yte)
            np.put_along_axis(oracle, true_idx,
                              np.take_along_axis(act[h:], true_idx, axis=1), axis=1)
            # predictor: keep what it selected
            pred = np.zeros_like(Yte)
            np.put_along_axis(pred, pred_idx,
                              np.take_along_axis(act[h:], pred_idx, axis=1), axis=1)

            ref = act[h:]
            oe = float(np.sqrt(np.mean((oracle - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))
            pe = float(np.sqrt(np.mean((pred - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))
            rec = float(np.mean([len(set(a) & set(b)) / k
                                 for a, b in zip(true_idx, pred_idx)]))
            print(f"    {L:6d} {pct:5d}% {oe:11.4f} {pe:10.4f} {rec:8.3f}")

    print()
    print("Read: 'keep' is the fraction of FFN rows fetched. Skipping 60% of the")
    print("FFN would cut ~0.5 GB/token at depth 2, worth roughly 1.4x on its own.")
    print("The oracle column is the floor any predictor could reach; if that is")
    print("already above ~5% the idea does not pay at this keep rate.")


if __name__ == "__main__":
    main()
