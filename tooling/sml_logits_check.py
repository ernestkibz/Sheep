#!/usr/bin/env python3
"""Gate G1: does the SML engine actually compute the right thing?

The kernel is already checked against the format spec (tooling/sml_check.py),
but that only proves one matvec is right. It says nothing about whether the
forward pass is wired correctly -- RoPE convention, GQA head repetition,
residual placement, tied output head, norm epsilon. Any of those can be wrong
while every individual matvec is perfect, and the result is a model that runs at
full speed and produces confident nonsense.

So the whole pass is compared against an independent numpy implementation built
straight from the architecture definition, reading the original checkpoint.
The two differ only by quantization, so at full read depth they should agree
closely; disagreement means a wiring bug, not a precision limit.

    sml-cli MODEL.ssf --tokens 9707,11,358,1079 --depth 3 --logits out.bin
    python tooling/sml_logits_check.py HF_CHECKPOINT out.bin --tokens 9707,11,358,1079
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sml_common import rms_norm, norm, rope, bias, opt   # noqa: E402
from sml_hf import open_weights                     # noqa: E402


class SsfWeights:
    """Same interface as Weights, but reading SSF at a chosen ladder depth.

    Running the numpy reference on SSF weights separates two very different
    explanations for bad output: a depth-specific bug in the C engine, or honest
    error compounding through 36 layers. Same code path, only the weight source
    changes.
    """

    def __init__(self, path, depth):
        from sml_read import SmlFile
        self.f = SmlFile(path)
        self.depth = depth
        self._cache = {}

    def field(self, k, d=None):
        h = self.f
        return {"general.architecture": h.arch,
                f"{h.arch}.block_count": h.n_layer,
                f"{h.arch}.embedding_length": h.n_embd,
                f"{h.arch}.attention.head_count": h.n_head,
                f"{h.arch}.attention.head_count_kv": h.n_head_kv,
                f"{h.arch}.attention.layer_norm_rms_epsilon": h.rms_eps,
                f"{h.arch}.rope.freq_base": h.rope_theta}.get(k, d)

    def flags(self):
        h = self.f
        return {"norm_type": h.norm_type, "rope_type": h.rope_type,
                "n_rot": h.n_rot,
                "parallel_residual": bool(h.arch_bits & 1)}

    def get(self, name):
        m = self.f.tensors[name]
        if m["kind"] == "f32":
            return self.f.f32(name).reshape(-1) if m["rows"] == 1 else self.f.f32(name)
        perm = self.f.perm(name)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        return self.f.dequant(name, depth=self.depth)[:, inv]


def reference_logits(path, ids, depth=None, all_positions=False):
    """Independent numpy forward pass. Full-precision weights, or SSF at `depth`.

    By default return next-token logits after the full input (the historical
    behavior). `all_positions=True` returns one next-token distribution per
    input prefix, which is useful for a sequence-level quality measurement.
    """
    # `depth is not None`, not `depth` -- a flat model has no depth axis and
    # passes 0, which the truthiness test sent to open_weights() and made it
    # try to read an .ssf as an HF checkpoint directory.
    W = SsfWeights(path, depth) if depth is not None else open_weights(path)
    a = W.field("general.architecture")
    a = a.decode() if isinstance(a, (bytes, bytearray)) else a
    n_layer = int(W.field(f"{a}.block_count"))
    n_embd = int(W.field(f"{a}.embedding_length"))
    n_head = int(W.field(f"{a}.attention.head_count"))
    n_kv = int(W.field(f"{a}.attention.head_count_kv", n_head))
    eps = float(W.field(f"{a}.attention.layer_norm_rms_epsilon", 1e-6))
    theta = float(W.field(f"{a}.rope.freq_base", 10000.0))
    hd = n_embd // n_head
    fl = W.flags()
    nt, n_rot = fl["norm_type"], fl["n_rot"]
    interlv, par_res = fl["rope_type"] == 1, fl["parallel_residual"]
    gated = True
    try:
        W.get("blk.0.ffn_gate.weight")
    except KeyError:
        gated = False
    try:                       # rotate by the frequencies the FILE carries
        freqs = W.get("rope_freqs")
    except KeyError:
        freqs = None

    emb = W.get("token_embd.weight")
    x = emb[np.asarray(ids)].astype(np.float32).copy()
    T = len(ids)
    pos = np.arange(T, dtype=np.float32)

    for L in range(n_layer):
        p = f"blk.{L}."
        h = norm(x, W.get(p + "attn_norm.weight"),
                 opt(W, p + "attn_norm.bias"), eps, nt)
        q = h @ W.get(p + "attn_q.weight").T + bias(W, p + "attn_q.bias")
        k = h @ W.get(p + "attn_k.weight").T + bias(W, p + "attn_k.bias")
        v = h @ W.get(p + "attn_v.weight").T + bias(W, p + "attn_v.bias")

        q = rope(q.reshape(T, n_head, hd), pos, theta, n_rot, interlv, freqs)
        k = rope(k.reshape(T, n_kv, hd), pos, theta, n_rot, interlv, freqs)
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
        attn_out = (o @ W.get(p + "attn_output.weight").T
                    + bias(W, p + "attn_output.bias"))

        ffn_in = x if par_res else x + attn_out
        h = norm(ffn_in, W.get(p + "ffn_norm.weight"),
                 opt(W, p + "ffn_norm.bias"), eps, nt)
        u = h @ W.get(p + "ffn_up.weight").T + bias(W, p + "ffn_up.bias")
        if gated:
            g = h @ W.get(p + "ffn_gate.weight").T + bias(W, p + "ffn_gate.bias")
            act = (g / (1.0 + np.exp(-g))) * u
        else:
            act = 0.5 * u * (1.0 + np.tanh(0.7978845608028654 *
                                           (u + 0.044715 * u ** 3)))
        ffn_out = (act @ W.get(p + "ffn_down.weight").T
                   + bias(W, p + "ffn_down.bias"))
        x = x + attn_out + ffn_out if par_res else ffn_in + ffn_out

    hf = norm(x, W.get("output_norm.weight"),
              opt(W, "output_norm.bias"), eps, nt)
    # untied models carry a separate head; assuming tied reads the wrong matrix
    try:
        head = W.get("output.weight")
    except KeyError:
        head = emb
    logits = hf @ head.T
    return logits if all_positions else logits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="HF checkpoint directory")
    ap.add_argument("logits_bin")
    ap.add_argument("--tokens", default="9707,11,358,1079")
    ap.add_argument("--ssf", help="run the numpy reference on SSF weights instead, "
                                  "to separate a C bug from error compounding")
    ap.add_argument("--ssf-depth", type=int, default=0,
                    help="ladder depth for --ssf; ignored for a flat model")
    args = ap.parse_args()

    ids = [int(t) for t in args.tokens.split(",")]

    raw = np.fromfile(args.logits_bin, dtype=np.int32, count=2)
    n_vocab, depth = int(raw[0]), int(raw[1])
    sml = np.fromfile(args.logits_bin, dtype=np.float32, offset=8, count=n_vocab)

    print(f"tokens={ids}  vocab={n_vocab}  sml read depth={depth}")
    if args.ssf:
        print(f"  numpy reference running on SSF weights at depth {args.ssf_depth}")
        ref = reference_logits(args.ssf, ids, depth=args.ssf_depth).astype(np.float64)
    else:
        ref = reference_logits(args.ref, ids).astype(np.float64)
    got = sml.astype(np.float64)

    # Logits carry an arbitrary additive constant through the softmax, so centre
    # both before comparing.
    ref -= ref.mean()
    got -= got.mean()

    corr = float(np.corrcoef(ref, got)[0, 1])
    rel = float(np.sqrt(np.mean((ref - got) ** 2)) / np.sqrt(np.mean(ref ** 2)))

    kref = np.argsort(-ref)[:10]
    kgot = np.argsort(-got)[:10]
    top1 = int(kref[0] == kgot[0])
    overlap5 = len(set(kref[:5].tolist()) & set(kgot[:5].tolist()))

    print(f"  correlation        {corr:.5f}")
    print(f"  rel-RMS difference {rel:.5f}")
    print(f"  top-1 agrees       {'yes' if top1 else 'no'}   "
          f"(ref {kref[0]}, sml {kgot[0]})")
    print(f"  top-5 overlap      {overlap5}/5")
    print(f"  ref  top5 {kref[:5].tolist()}")
    print(f"  sml  top5 {kgot[:5].tolist()}")

    # A wiring bug (RoPE convention, GQA repeat, residual placement, tied head)
    # destroys correlation outright. Quantization at these bit rates costs a few
    # percent, not the structure of the distribution.
    ok = corr > 0.99 and overlap5 >= 4
    print()
    print("PASS - forward pass matches the reference architecture" if ok else
          "FAIL - logits diverge; this is a wiring bug, not a precision limit")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
