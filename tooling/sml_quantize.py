#!/usr/bin/env python3
"""Layer-sequential SSF build: calibrate and quantize in one pass.

Replaces calibrate-then-convert. GPTQ needs a Hessian per linear, and storing
activations for enough tokens does not scale -- 1600 tokens already cost 1982 MB,
and the held-out gain only cleared the bar at 800 training tokens (0.87x at 96,
2.11x at 800). So nothing is stored: the model is walked one layer at a time,
each layer's inputs are accumulated into H = X^T X, that layer is quantized, and
the activations are dropped before moving on.

Two properties this buys beyond memory:

  * Token count stops being bounded by RAM, which is the variable the quality
    gain actually depends on.
  * Each layer sees the activations produced by the ALREADY-QUANTIZED layers
    before it, so the feedback compensates accumulated error rather than
    pretending earlier layers were exact. This is the part that matters for a
    36-layer model where 9.8% error compounds into noise.

Calibration uses many short sequences rather than one long one: attention is
per-sequence, so memory stays flat while the sample count grows.

    python tooling/sml_build.py HF_CHECKPOINT out.ssf --seqs 48 --seq-len 96
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sml_common import (Bpe, CALIB_TEXT, rms_norm, norm, bias, opt,   # noqa: E402
                        rope_freqs, rope as rope_np)
from sml_hf import open_weights                                # noqa: E402
from sml_format import pack_bits1  # noqa: E402  (the 1-bit plane of Q3/Q5/Q7)
from sml_format import (MAGIC, VERSION, HEADER_BYTES, GROUP, SCALE_SUPER,   # noqa: E402
                         LAYOUT_LADDER, LAYOUT_FLAT,
                         MAX_LEVELS, NAME_LEN, HEADER_FMT, REC_FMT, REC_SIZE,
                         KIND_F32, KIND_LADDER, GROUPS_PER_SUPER, build_vocab_blob,
                         pack_superblocks, split_levels, super_bytes,
                         quantize_ladder, GROUP_FMT, GROUP_MAX, GROUP_KIND_ACT,
                         pack_flat4, MODEL_HADAMARD, MODEL_SCALE6, MODEL_MINNEG,
                         MODEL_EXACT, pack6, rotate32)
from sml_gptq import gptq_block                                 # noqa: E402

MAX_H = 12288    # was 4096. That cap was regularising a rank-deficient
                 # Hessian, not saving memory: at 3072 samples against 11008
                 # columns the inverse was mostly damping and long sweeps
                 # propagated noise, so blocking contained the damage (measured
                 # 7.2x worse unblocked). With 16384 samples the Hessian is full
                 # rank and the cap only costs error propagation.

def _gpu_call(fn, *a, **kw):
    """The search in fused C (sml/sml-quant.dll, every core) when it is built and
    SML_GRID_C is not 0; else on the GPU (sml_grid_gpu.py) when torch sees a CUDA
    card and SML_GRID_GPU is not 0; None to run the numpy one. Measured on a real
    13.8M-weight tensor (2026-10-06): numpy grid 16.8 s + field 5.0 s, C 0.47 +
    0.09 s with every field identical, torch on the GPU 7.0 + 0.6 s (99.7%
    identical). Out of GPU memory, the GPU path tries smaller blocks first."""
    if os.environ.get("SML_GRID_C", "1") != "0":
        try:
            import sml_grid_c as C
            if C.available():
                return getattr(C, fn)(*a, **kw)
        except ImportError:
            pass
    if os.environ.get("SML_GRID_GPU", "1") == "0":
        return None
    try:
        import sml_grid_gpu as G
    except ImportError:
        return None
    if not G.available():
        return None
    for per in (1 << 22, 1 << 20, 1 << 18):
        os.environ["SML_GRID_GPU_BLOCK"] = str(per)
        try:
            return getattr(G, fn)(*a, **kw)
        except RuntimeError as e:              # CUDA out of memory: smaller blocks
            if "out of memory" not in str(e).lower():
                raise
            import torch
            torch.cuda.empty_cache()
    return None


def grid_threads():
    """Threads for the grid search's row blocks: SML_GRID_THREADS, else the cores."""
    e = os.environ.get("SML_GRID_THREADS")
    return max(1, int(e)) if e else max(1, min(6, os.cpu_count() or 1))


# Retargeting the objective at the depth a tensor is READ was measured and does
# NOT work, despite looking like a 16x win in isolation. Kept as a parameter
# because the machinery is right; the default is not to use it. See RESULTS.md.
DEEP = 3


# The planes of a FLAT tensor of each width, lowest bits first, as the engine
# reads them (Q6 keeps its historical order: the 2-bit plane, then the nibble).
FLAT_PLANES = {2: [2], 3: [2, 1], 4: [4], 5: [4, 1], 6: [2, 4], 7: [4, 2, 1], 8: [4, 4]}


class Quantizer:
    """Quantizes one tensor and writes it, with or without error feedback."""

    def __init__(self, f, meta, level_bits, group, target_level, use_gptq,
                 layout=LAYOUT_LADDER, hadamard=False, scale6=False):
        # new 6-bit files store negative-only mins (MODEL_MINNEG)
        self.minneg = scale6
        self.layout = layout
        self.hadamard = hadamard
        self.scale6 = scale6
        self.gptq_check = False
        self.choice = {"gptq": 0, "rounding": 0}
        self.f, self.meta = f, meta
        self.level_bits, self.group = level_bits, group
        self.target_level = target_level
        self.use_gptq = use_gptq
        self.total_bits = sum(level_bits)
        self.target_bits = sum(level_bits[:target_level])
        self.depth_of = {}      # name -> the depth its objective targeted

    @staticmethod
    def asym_grid(wp, group, nbits, target_bits=None, imp=None):
        """Per-group (min, step) for an ASYMMETRIC grid.

        We rejected asymmetric once on price: a per-group fp16 min costs 1.0
        b/w against 0.3125 for a symmetric scale. That priced the min as if it
        had to be stored exactly. Quantizing it the same two-level way the scale
        already is drops the cost to 0.19 b/w, and the measured error falls 1.8x
        at depth 2. The earlier rejection was right about the price and wrong
        that the price was fixed.

        Asymmetry does not break the ladder: value = min + code * step still
        truncates correctly, so a prefix of the code is still a coarser read of
        the same number.

        2026-09-29, three changes (SML_GRID_V1=1 keeps the old search):
        - the range factor is chosen PER GROUP; one factor for the whole tensor
          fitted the average group and no actual one
        - errors are weighted by the column's activation energy (imp), so the
          grid minimizes what the output sees, not the raw weight error
        - a weighted least-squares refit of min and step once the codes are
          chosen (step = cov_w(q, g) / var_w(q)), kept per group only where the
          error fell. On a random-weight Llama head plain rounding went from
          12% behind llama.cpp's Q4_K (KL) to 2.4% behind with the refit alone.
        """
        if os.environ.get("SML_GRID_V1") == "1":
            return Quantizer.asym_grid_v1(wp, group, nbits, target_bits)
        gpu = _gpu_call("asym_grid", wp, group, nbits, target_bits, imp,
                        ncand=int(os.environ.get("SML_GRID_CAND", "20")), iters=int(os.environ.get("SML_GRID_ITERS", "3")),
                        refit=os.environ.get("SML_LS_REFIT", "1") != "0")
        if gpu is not None:
            return gpu
        rows, cols = wp.shape
        ng = cols // group
        n = (1 << nbits) - 1
        wt = None
        if imp is not None:
            wt = np.asarray(imp, dtype=np.float32).reshape(1, ng, group)
            wt = wt / max(float(wt.mean()), 1e-30) + np.float32(1e-6)
        rem = 0 if target_bits is None else nbits - target_bits
        refit = not rem and os.environ.get("SML_LS_REFIT", "1") != "0"
        # Candidate ranges, each refitted BEFORE the choice (the best range before
        # a refit is not reliably the best after one). float32, a block of rows at
        # a time. On the 193M-weight Llama head: 23 candidates x 3 refits in
        # float64 over the whole tensor, 15 min, KL 0.0279; 20 x 3 here, 7 min,
        # KL 0.0279; 12 x 2, 3.6 min, KL 0.0284 -- too thin a margin over Q4_K_M's
        # 0.0286 to build a family's verdict on.
        ncand = int(os.environ.get("SML_GRID_CAND", "20"))
        iters = int(os.environ.get("SML_GRID_ITERS", "3"))
        facs = np.concatenate([np.linspace(0.6, 0.85, max(2, ncand // 3)),
                               np.linspace(0.88, 1.04, max(2, ncand - ncand // 3))]).astype(np.float32)
        out_l = np.empty((rows, ng), np.float32)
        out_s = np.empty((rows, ng), np.float32)
        # ~1M weights per block (stays in cache), and at least one block per
        # thread; rows never interact, so the split changes no result
        blk = max(1, min((1 << 20) // max(cols, 1), -(-rows // grid_threads())))

        def one_block(r0):
            g = wp[r0:r0 + blk].reshape(-1, ng, group).astype(np.float32)
            lo = g.min(-1, keepdims=True)
            hi = g.max(-1, keepdims=True)
            c = (lo + hi) * np.float32(0.5)
            w = wt if wt is not None else np.ones((1, ng, group), np.float32)
            ws = np.broadcast_to(w, g.shape).sum(-1, keepdims=True)

            def err(l, step):
                q = np.clip(np.rint((g - l) / step), 0, n)
                qv = (((q.astype(np.int32) >> rem) << rem) + (1 << (rem - 1))) if rem else q
                d2 = (qv * step + l - g) ** 2
                return q, (d2 * w).sum(-1, keepdims=True)

            def ls(l, step, e):
                # weighted least-squares refit of (min, step) with the codes
                # fixed; per group, kept only where the error fell
                q, _ = err(l, step)
                for _ in range(iters):
                    qm = (w * q).sum(-1, keepdims=True) / ws
                    gm = (w * g).sum(-1, keepdims=True) / ws
                    dq = q - qm
                    var = (w * dq * dq).sum(-1, keepdims=True)
                    cov = (w * dq * (g - gm)).sum(-1, keepdims=True)
                    s2 = np.where(var > 1e-12, cov / np.maximum(var, 1e-12), step)
                    s2 = np.maximum(s2, np.float32(1e-12))
                    l2 = gm - s2 * qm
                    q2, e2 = err(l2, s2)
                    better = e2 < e
                    l = np.where(better, l2, l); step = np.where(better, s2, step)
                    q = np.where(better, q2, q); e = np.where(better, e2, e)
                return l, step, e

            bl = bs = be = None
            for sfac in facs:
                half = (hi - lo) * np.float32(0.5) * sfac
                l = c - half
                step = np.maximum(2.0 * half, np.float32(1e-12)) / n
                _, e = err(l, step)
                if refit:
                    l, step, e = ls(l, step, e)
                if be is None:
                    bl, bs, be = l, step, e
                else:
                    better = e < be
                    bl = np.where(better, l, bl); bs = np.where(better, step, bs); be = np.where(better, e, be)
            out_l[r0:r0 + blk] = bl[..., 0]
            out_s[r0:r0 + blk] = bs[..., 0]

        # Row blocks are independent and numpy releases the GIL inside each
        # array operation, so blocks run on a thread pool: the grid was 7 of a
        # race head's build minutes on one core. Bit-identical to the serial
        # loop -- each block's arithmetic is unchanged, only which core runs it.
        starts = list(range(0, rows, blk))
        nt = min(len(starts), grid_threads())
        if nt > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(nt) as ex:
                list(ex.map(one_block, starts))
        else:
            for r0 in starts:
                one_block(r0)
        return out_l.astype(wp.dtype), out_s.astype(wp.dtype)

    @staticmethod
    def asym_grid_v1(wp, group, nbits, target_bits=None):
        """Per-group (min, step) for an ASYMMETRIC grid.

        We rejected asymmetric once on price: a per-group fp16 min costs 1.0
        b/w against 0.3125 for a symmetric scale. That priced the min as if it
        had to be stored exactly. Quantizing it the same two-level way the scale
        already is drops the cost to 0.19 b/w, and the measured error falls 1.8x
        at depth 2. The earlier rejection was right about the price and wrong
        that the price was fixed.

        Asymmetry does not break the ladder: value = min + code * step still
        truncates correctly, so a prefix of the code is still a coarser read of
        the same number.
        """
        rows, cols = wp.shape
        g = wp.reshape(rows, cols // group, group)
        n = (1 << nbits) - 1
        lo = g.min(-1, keepdims=True)
        hi = g.max(-1, keepdims=True)
        c = (lo + hi) / 2.0
        # Score the search at the depth that will actually be READ, not at full
        # precision. The symmetric ladder already does this and it is the whole
        # reason its depth-2 view is usable; scoring at 6 bits instead cost
        # 0.974 -> 0.892 end-to-end correlation before this was carried over.
        rem = 0 if target_bits is None else nbits - target_bits
        best_e = None; best = None
        for sfac in np.linspace(0.75, 1.0, 11):
            half = (hi - lo) / 2.0 * sfac
            l = c - half
            step = np.maximum(2.0 * half, 1e-12) / n
            q = np.clip(np.rint((g - l) / step), 0, n)
            qv = (((q.astype(np.int32) >> rem) << rem) + (1 << (rem - 1))
                  if rem else q)
            e = ((qv * step + l - g) ** 2).sum()
            if best_e is None or e < best_e:
                best_e = e; best = (l[..., 0], step[..., 0])
        if rem or os.environ.get("SML_LS_REFIT", "1") == "0":
            return best        # (rows, ng) each
        # Least-squares refit (2026-09-29). The search above only shrinks the range
        # symmetrically about its middle. With the codes q fixed, the best min and
        # step for a group are a straight-line fit of g on q: step = cov(q, g) /
        # var(q), min = mean(g) - step * mean(q). Refit, re-round, and keep a
        # group's new grid only where its error actually fell. On a random-weight
        # Llama head the symmetric search alone left plain rounding 12% (KL) behind
        # llama.cpp's Q4_K, whose block search refits the same way.
        L = best[0][..., None].astype(np.float64)
        S = best[1][..., None].astype(np.float64)
        gd = g.astype(np.float64)

        def err(Lx, Sx):
            q = np.clip(np.rint((gd - Lx) / Sx), 0, n)
            return q, ((q * Sx + Lx - gd) ** 2).sum(-1, keepdims=True)

        q, e0 = err(L, S)
        for _ in range(3):
            qm = q.mean(-1, keepdims=True)
            gm = gd.mean(-1, keepdims=True)
            var = ((q - qm) ** 2).mean(-1, keepdims=True)
            cov = ((q - qm) * (gd - gm)).mean(-1, keepdims=True)
            S2 = np.where(var > 1e-12, cov / np.maximum(var, 1e-12), S)
            S2 = np.maximum(S2, 1e-12)
            L2 = gm - S2 * qm
            q2, e2 = err(L2, S2)
            better = e2 < e0
            L = np.where(better, L2, L); S = np.where(better, S2, S)
            q = np.where(better, q2, q); e0 = np.where(better, e2, e0)
        return L[..., 0].astype(g.dtype), S[..., 0].astype(g.dtype)

    @staticmethod
    def two_level(v, gps, signed, bits=8, neg=False):
        """fp16 per superblock + uint8 per group, the encoding the file uses.

        Returned as (d16, q8, effective) so the caller can re-derive codes
        against the value that will actually be READ, never against the ideal
        one -- encoder and decoder disagreeing is the whole bug class this
        avoids.
        """
        rows, ng = v.shape
        nsb = (ng + gps - 1) // gps
        pad = nsb * gps - ng
        vp = np.pad(v, ((0, 0), (0, pad)), mode="edge") if pad else v
        b = vp.reshape(rows, nsb, gps)
        # The min is stored biased into an unsigned field, so the bias is half
        # the field and moves with its width: 127 at 8 bits, 31 at 6. The C side
        # carries the same pair in SML_MIN_BIAS8/6.
        # bias = 127 at 8 bits and 31 at 6, upper code = 2*bias. An earlier
        # edit here derived the 8-bit bias as 128 while the C side subtracts
        # 127 for v8 files -- every min of every NEW v8 build would have
        # decoded one step off. Stated as literals so it cannot drift again.
        hi = (1 << bits) - 1
        bias = 127 if bits == 8 else 31
        if signed and neg:
            # negative-only (MODEL_MINNEG): every level on the side mins are on;
            # a positive min (a group entirely above zero) becomes 0
            bn = np.minimum(b, 0.0)
            d16 = (np.max(-bn, axis=2) / hi).astype(np.float16)
            dv = d16.astype(np.float32); dv[dv == 0] = 1e-8
            q8 = np.clip(np.rint(bn / dv[:, :, None]) + hi, 0, hi).astype(np.uint8)
            eff = ((q8.astype(np.float32) - hi) * dv[:, :, None])
        elif signed:
            d16 = (np.max(np.abs(b), axis=2) / bias).astype(np.float16)
            dv = d16.astype(np.float32); dv[dv == 0] = 1e-8
            q8 = np.clip(np.rint(b / dv[:, :, None]) + bias, 0, 2 * bias).astype(np.uint8)
            eff = ((q8.astype(np.float32) - bias) * dv[:, :, None])
        else:
            d16 = (np.max(b, axis=2) / hi).astype(np.float16)
            dv = d16.astype(np.float32); dv[dv == 0] = 1e-8
            q8 = np.clip(np.rint(b / dv[:, :, None]), 1, hi).astype(np.uint8)
            eff = (q8.astype(np.float32) * dv[:, :, None])
        eff = eff.reshape(rows, nsb * gps)[:, :ng]
        q8 = q8.reshape(rows, nsb * gps)[:, :ng]
        return d16, q8, eff

    @staticmethod
    def field_search(w, d16, q8, md16, mq8, group, gps, nmax, hi, moff, mmax, imp, radius=1):
        """Round the stored group fields to the best NEIGHBOUR, not the nearest.

        two_level rounds each group's ideal step and min to the nearest 6-bit
        field on its superblock's scale, and the codes are then re-derived
        against those. Nearest is not best: with the codes re-derived, a field one
        level up or down can reconstruct the group better. Each group tries
        (step field +-radius) x (min field +-radius) and keeps the pair with the
        least activation-weighted error. Nothing is stored beyond the fields
        themselves -- the file format is unchanged."""
        gpu = _gpu_call("field_search", w, d16, q8, md16, mq8, group, gps, nmax, hi, moff, mmax, imp, radius)
        if gpu is not None:
            return gpu
        rows, cols = w.shape
        ng = cols // group
        dv = np.repeat(d16.astype(np.float32), gps, axis=1)[:, :ng]
        dm = np.repeat(md16.astype(np.float32), gps, axis=1)[:, :ng]
        if imp is not None:
            wt = np.asarray(imp, dtype=np.float32).reshape(1, ng, group)
            wt = wt / max(float(wt.mean()), 1e-30) + np.float32(1e-6)
        else:
            wt = np.ones((1, ng, group), np.float32)
        q8o = q8.astype(np.int32).copy()
        mq8o = mq8.astype(np.int32).copy()
        blk = max(1, min((1 << 20) // max(cols, 1), -(-rows // grid_threads())))

        def one_block(r0):
            g = w[r0:r0 + blk].reshape(-1, ng, group).astype(np.float32)
            qs0, qm0 = q8o[r0:r0 + blk].copy(), mq8o[r0:r0 + blk].copy()
            d, m = dv[r0:r0 + blk][..., None], dm[r0:r0 + blk][..., None]
            be = bs = bm = None
            for a in range(-radius, radius + 1):
                qs = np.clip(qs0 + a, 1, hi)
                S = np.maximum(qs[..., None] * d, np.float32(1e-12))
                for b in range(-radius, radius + 1):
                    qm = np.clip(qm0 + b, 0, mmax)
                    M = (qm[..., None] - moff) * m
                    c = np.clip(np.rint((g - M) / S), 0, nmax)
                    e = (((c * S + M - g) ** 2) * wt).sum(-1)
                    if be is None:
                        be, bs, bm = e, qs, qm
                    else:
                        better = e < be
                        be = np.where(better, e, be)
                        bs = np.where(better, qs, bs)
                        bm = np.where(better, qm, bm)
            q8o[r0:r0 + blk] = bs
            mq8o[r0:r0 + blk] = bm

        starts = list(range(0, rows, blk))
        nt = min(len(starts), grid_threads())
        if nt > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(nt) as ex:
                list(ex.map(one_block, starts))
        else:
            for r0 in starts:
                one_block(r0)
        return (q8o.astype(np.uint8), mq8o.astype(np.uint8),
                q8o.astype(np.float32) * dv, (mq8o.astype(np.float32) - moff) * dm)

    def run(self, name, w, X, imp, perm=None, target_level=None):
        """w: (rows, cols) original order. X: (N, cols) inputs. Returns W_hat.

        `perm` lets a whole group share one column order. Members of a
        shared-activation group then read the SAME prepared activation, so the
        runtime gathers and quantizes once for the group instead of once per
        tensor. The permutation is worth about 0.6% on its own -- measured,
        9.141% against 9.196% unpermuted -- so sharing it is close to free, and
        the activation residency it buys is not.
        """
        m = self.meta[name]
        # Rotate BOTH the weights and the activations into the Hadamard basis
        # before anything else, so the permutation and the grid are chosen in
        # the basis the codes are stored in. The engine rotates the activation
        # in the same order -- rotate, then gather.
        if self.hadamard:
            w = rotate32(w).astype(np.float32)
            if X is not None:
                X = rotate32(X).astype(np.float32)
        rows, cols = w.shape
        group = self.group
        ng = cols // group

        # Per-tensor read depth. A tensor read at depth 3 must have its error
        # feedback TARGETED at depth 3, or the extra plane is worse than useless:
        # measured on blk.12, GPTQ targeted at the 4-bit view and read at 6 bits
        # gives 7.461% on ffn_down against 2.118% for simply reading 4 bits.
        # Targeted correctly the same 6-bit read gives 0.460%.
        #
        # Buying a deeper read while optimising the shallow one is how the third
        # plane became dead weight, and it is why mixed allocation previously
        # measured only +0.0009.
        tl = self.target_level if target_level is None else target_level
        tb = sum(self.level_bits[:tl])
        # FLAT reads every stored bit, so there is no truncated view to target:
        # the objective is exactly the code that will be read, at this tensor's
        # own width. That is the whole point of the flat path -- the grid is
        # fitted to what is read rather than to a prefix of it.
        tot = self.total_bits
        if self.layout == LAYOUT_FLAT:
            tot = int(m.get("nbits", 4))
            tb = tot

        # permutation: cluster similar magnitudes so a group's shared scale is
        # not set by one outlier
        if perm is None:
            if self.layout == LAYOUT_FLAT and X is not None:
                # Most important column FIRST. Error feedback pushes each
                # column's residual into the columns AFTER it, so the last
                # column quantized has nothing left to absorb its mistake.
                # Ordering by ascending |w| put the largest-magnitude columns
                # exactly there. Measured 1.028x held out, at zero cost.
                perm = np.argsort(-np.einsum("ij,ij->j", X, X)).astype(np.int32)
            else:
                perm = np.argsort(np.max(np.abs(w), axis=0)).astype(np.int32)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(cols)
        wp = np.ascontiguousarray(w[:, perm])
        impp = imp[perm] if imp is not None else None

        # --gptq-check (Genesis 41, discernment): GPTQ fits on three quarters of
        # the calibration rows, and it and plain rounding are both scored on the
        # held-out quarter by OUTPUT error, ||(W_hat - W) X^T||. The better one is
        # kept. Scored on its own training rows GPTQ always wins -- the trap that
        # once showed 4.80x in-sample for a 0.87x held-out result. On models with
        # massive activations (OpenReasoning-Nemotron-1.5B: residual ~8,900) GPTQ
        # measured 0.61 rel-RMS against 0.30 for plain rounding, model-wide.
        check = self.gptq_check and self.use_gptq and X is not None and X.shape[0] >= 64
        w0p = wp.copy() if check else None
        Xfit = X
        Xho = None
        if check:
            nfit = (X.shape[0] * 3) // 4
            Xfit, Xho = X[:nfit], np.ascontiguousarray(X[nfit:][:, perm])

        codes = scales = gmins = None
        if self.use_gptq and X is not None:
            Xp = np.ascontiguousarray(Xfit[:, perm])
            codes = np.zeros((rows, cols), dtype=np.int32)
            scales = np.zeros((rows, ng), dtype=np.float32)
            gmins = np.zeros((rows, ng), dtype=np.float32)
            ok = True
            for c0 in range(0, cols, MAX_H):
                c1 = min(c0 + MAX_H, cols)
                sub = np.ascontiguousarray(wp[:, c0:c1])
                cc, ss, mm = gptq_block(sub, np.ascontiguousarray(Xp[:, c0:c1]),
                                        group, tot, tb,
                                        impp[c0:c1] if impp is not None else None)
                if cc is None:
                    ok = False
                    break
                codes[:, c0:c1] = cc
                scales[:, c0 // group:c1 // group] = ss
                gmins[:, c0 // group:c1 // group] = mm
            if not ok:
                codes = scales = gmins = None
            del Xp

        # No feedback (no activations, --no-gptq, or the factorisation failed):
        # encode() below fits the grid itself (asym_grid). A quantize_ladder pass
        # used to run here first; encode never read its codes or scales, and it
        # cost a tenth of each tensor's time (2026-10-06, profiled).

        # ASYMMETRIC grid. Min and step are both stored the same two-level way,
        # and the codes are re-derived against the values that will actually be
        # READ rather than the ideal ones -- an encoder and decoder disagreeing
        # by one rounding step is the bug class that costs days here.
        gps = SCALE_SUPER // group
        nmax = (1 << tot) - 1
        sb = 6 if self.scale6 else 8
        rem = tot - tb

        def encode(wsrc, gm, sc):
            if gm is not None:
                # The feedback already chose this grid while compensating error;
                # re-deriving it here would throw that away.
                mins, steps = gm, sc
            else:
                mins, steps = self.asym_grid(wsrc, group, tot, tb, impp)
            d16, q8, eff_s = self.two_level(steps, gps, signed=False, bits=sb)
            neg = self.minneg and sb == 6
            md16, mq8, eff_m = self.two_level(mins, gps, signed=True, bits=sb, neg=neg)
            rad = int(os.environ.get("SML_FIELD_SEARCH", "2"))
            if rad and not rem:
                fhi = (1 << sb) - 1
                fb = 127 if sb == 8 else 31
                q8, mq8, eff_s, eff_m = self.field_search(
                    wsrc, d16, q8, md16, mq8, group, gps, nmax, fhi,
                    fhi if neg else fb, fhi if neg else 2 * fb, impp, rad)
            # In slices of rows (each row's grid stands alone, so the result is identical): done
            # whole, a 128256 x 3072 output head needed four 1.5 GB temporaries at once and the
            # build of Llama-3.1-Nemotron-Nano-4B ran out of memory on its last tensor (2026-10-21)
            rows = wsrc.shape[0]
            step = max(1, (1 << 25) // max(1, wsrc.shape[1]))
            cds = np.empty(wsrc.shape, np.uint8)
            what = None
            for r0 in range(0, rows, step):
                r1 = min(rows, r0 + step)
                S = np.repeat(eff_s[r0:r1], group, axis=1)
                M = np.repeat(eff_m[r0:r1], group, axis=1)
                c = np.clip(np.rint((wsrc[r0:r1] - M) / np.maximum(S, 1e-12)), 0, nmax).astype(np.uint8)
                cds[r0:r1] = c
                # what the next layer will actually see
                view = ((c.astype(np.int32) >> rem) << rem) + (1 << (rem - 1)) if rem \
                    else c.astype(np.int32)
                part = view.astype(np.float32) * S + M
                if what is None:
                    what = np.empty(wsrc.shape, part.dtype)
                what[r0:r1] = part
            return cds, d16, q8, md16, mq8, what

        codes, d16, q8, md16, mq8, what = encode(wp, gmins, scales)
        if check and gmins is not None:
            r_codes, r_d16, r_q8, r_md16, r_mq8, r_what = encode(w0p, None, None)
            e_g = float(np.sum(((what - w0p) @ Xho.T) ** 2))
            e_r = float(np.sum(((r_what - w0p) @ Xho.T) ** 2))
            if e_r < e_g:
                codes, d16, q8, md16, mq8, what = r_codes, r_d16, r_q8, r_md16, r_mq8, r_what
            self.choice["rounding" if e_r < e_g else "gptq"] += 1

        self._write(name, codes, d16, q8, md16, mq8, perm)
        self.depth_of[name] = tl
        out = what[:, inv]
        # SML_DUMP_Q=DIR saves every tensor AS QUANTIZED, in HF column order, so
        # a reference implementation can run on exactly the weights the file
        # holds: that separates a quantization loss from an engine bug.
        dump = os.environ.get("SML_DUMP_Q")
        if dump:
            os.makedirs(dump, exist_ok=True)
            np.save(os.path.join(dump, name + ".npy"), out.astype(np.float16))
        return out

    def _write(self, name, codes, d16, q8, md16, mq8, perm):
        m = self.meta[name]
        rows, cols = m["rows"], m["cols"]
        codes, d16, q8, md16, mq8, perm = pad_groups(
            codes, d16, q8, md16, mq8, perm, cols, self.group,
            (63 if self.minneg else 31) if self.scale6 else 127)
        ng = cols // self.group
        npack = cols // (GROUPS_PER_SUPER * self.group)
        f = self.f
        f.seek(m["perm_off"]);  f.write(perm.tobytes())
        f.seek(m["sd_off"]);    f.write(d16.tobytes())
        f.seek(m["sq_off"]);    f.write(pack6(q8).tobytes() if self.scale6
                                        else q8.tobytes())
        f.seek(m["md_off"]);    f.write(md16.tobytes())
        f.seek(m["mq_off"]);    f.write(pack6(mq8).tobytes() if self.scale6
                                        else mq8.tobytes())
        if self.layout == LAYOUT_FLAT:
            nbw = int(m.get("nbits", 4))
            if nbw in (2, 3, 5, 7):
                # each plane takes its bits of the code, lowest first (FLAT_PLANES)
                c = codes.astype(np.uint16)
                shift = 0
                for i, bits in enumerate(FLAT_PLANES[nbw]):
                    part = ((c >> shift) & ((1 << bits) - 1)).astype(np.uint8)
                    packed = (pack_superblocks(part, self.group) if bits == 2 else
                              pack_flat4(part, self.group) if bits == 4 else pack_bits1(part, self.group))
                    f.seek(m["lv"][i]["codes_off"])
                    f.write(packed.tobytes())
                    shift += bits
                return
            if int(m.get("nbits", 4)) == 8:
                # Q8: code = lo + 16*hi, two nibble planes in the 4-bit layout; the
                # engine sums dot(lo) + 16*dot(hi) with the 4-bit kernel's maths
                c8 = codes.astype(np.uint8)
                f.seek(m["lv"][0]["codes_off"])
                f.write(pack_flat4(c8 & 0x0F, self.group).tobytes())
                f.seek(m["lv"][1]["codes_off"])
                f.write(pack_flat4((c8 >> 4) & 0x0F, self.group).tobytes())
                return
            if int(m.get("nbits", 4)) == 6:
                # code = base + 16*extra. Store `extra` (the top 2 bits) in the
                # group-planar 2-bit layout the existing kernel reads, and
                # `base` (the low nibble) in the flat layout. The engine sums
                # 16*dot(extra) + dot(base), which is exact because the code is
                # linear -- no prefix of it has to mean anything.
                hi = (codes.astype(np.uint8) >> 4) & 0x03
                lo = codes.astype(np.uint8) & 0x0F
                f.seek(m["lv"][0]["codes_off"])
                f.write(pack_superblocks(hi, self.group).tobytes())
                f.seek(m["lv"][1]["codes_off"])
                f.write(pack_flat4(lo, self.group).tobytes())
                return
            f.seek(m["lv"][0]["codes_off"])
            f.write(pack_flat4(codes, self.group).tobytes())
            return
        for lv, lc in enumerate(split_levels(codes, self.level_bits)):
            f.seek(m["lv"][lv]["codes_off"])
            f.write(pack_superblocks(lc, self.group).tobytes())



# Licence obligations we are sure of, keyed by the licence id a model card
# declares. Anything else is recorded as "read the licence" rather than guessed:
# a wrong obligation written into a shipped file is worse than none.
LICENCE_DUTIES = {
    "llama3.2": ["model name must begin with 'Llama'", "display 'Built with Llama'",
                 "ship the Llama 3.2 Community License beside the file"],
    "llama3.1": ["model name must begin with 'Llama'", "display 'Built with Llama'",
                 "ship the Llama 3.1 Community License beside the file"],
    "llama3":   ["model name must begin with 'Llama'", "display 'Built with Meta Llama 3'",
                 "ship the Meta Llama 3 License beside the file"],
    "apache-2.0": ["ship the Apache-2.0 licence and any NOTICE file beside the file"],
    "mit":        ["ship the MIT licence text beside the file"],
    # Qwen2.5-3B (verified on its model card, 2026-09-23). A research licence:
    # a demo is research, a product is not, so the file says to check first.
    "qwen-research": ["research licence: check its commercial-use terms before any "
                      "commercial distribution", "ship the Qwen licence beside the file"],
    # NVIDIA's OpenReasoning-Nemotron models (cc-by-4.0 on the model card): the
    # duty is attribution, and saying that the weights were changed.
    "cc-by-4.0": ["credit the original creator and link the source model",
                  "state that the weights were modified (re-quantized to SSF)",
                  "link the CC BY 4.0 licence"],
}


def promote_order(ladder, n_layer):
    """Which 4-bit tensors a spare byte budget widens first: the roles the
    reference's own larger mixes widen first (attn_v, then ffn_down, then the
    rest of attention, then the FFN), each from the outermost layers inward --
    the first and last layers are where its use_more_bits already looks. One
    order for every model; nothing is chosen per family."""
    roles = ("attn_v", "ffn_down", "attn_k", "attn_output", "attn_q", "ffn_gate", "ffn_up")
    lay = []
    lo, hi = 0, n_layer - 1
    while lo <= hi:
        lay.append(lo)
        if hi != lo:
            lay.append(hi)
        lo, hi = lo + 1, hi - 1
    have = set(ladder)
    out = []
    for r in roles:
        for L in lay:
            n = f"blk.{L}.{r}.weight"
            if n in have:
                out.append(n)
    return out


def pad_groups(codes, d16, q8, md16, mq8, perm, cols, group, mbias):
    """Pad a quantized tensor out to `cols` columns with WHOLE groups whose scale
    and min are both zero, so they contribute exactly nothing whatever activation
    they meet. Both packed layouts read four groups at a time; a 576-wide model
    (18 groups) could not be stored at all before this.

    The padded slots of the permutation repeat real columns from the same
    256-wide superblock. The prompt path quantizes activations per superblock,
    so repeating values already there leaves its scale -- and therefore every
    real activation code -- exactly as it would have been."""
    rows, real = codes.shape
    if real == cols:
        return codes, d16, q8, md16, mq8, perm
    extra, eg = cols - real, (cols - real) // group
    codes = np.concatenate([codes, np.zeros((rows, extra), codes.dtype)], 1)
    q8 = np.concatenate([q8, np.zeros((rows, eg), q8.dtype)], 1)
    mq8 = np.concatenate([mq8, np.full((rows, eg), mbias, mq8.dtype)], 1)
    nsb = -(-(cols // group) // (SCALE_SUPER // group))
    if d16.shape[1] < nsb:
        z = np.zeros((rows, nsb - d16.shape[1]), d16.dtype)
        d16, md16 = np.concatenate([d16, z], 1), np.concatenate([md16, z], 1)
    start = (real // SCALE_SUPER) * SCALE_SUPER
    src = perm[start:real]
    perm = np.concatenate([perm, src[np.arange(extra) % len(src)]]).astype(perm.dtype)
    return codes, d16, q8, md16, mq8, perm


def alibi_slopes(n_head):
    """ALiBi slopes as BLOOM builds them: a geometric series over the nearest
    power of two of heads, then every other slope of the next power for any
    heads left over. The engine computes the same series."""
    p = 1 << (n_head.bit_length() - 1)
    base = 2.0 ** (-(2.0 ** -(np.log2(p) - 3)))
    s = [base ** i for i in range(1, p + 1)]
    if p != n_head:
        eb = 2.0 ** (-(2.0 ** -(np.log2(2 * p) - 3)))
        s += [eb ** i for i in range(1, 1 + 2 * min(p, n_head - p), 2)]
    return np.array(s, dtype=np.float32)


def moe_forward(Q, W, p, h, fl, imp_of):
    """Mixture-of-experts FFN for calibration, run as the engine runs it: the
    router picks n_expert_used experts per token, each expert is quantized
    against the tokens routed to IT, and outputs are summed by the router's
    weights. Every expert is quantized even if no calibration token chose it
    -- an unwritten expert would sit in the file as zeros -- against all
    tokens in that case, since that is what it would be asked on."""
    router = W.get(p + "ffn_gate_inp.weight").astype(np.float32)
    E, k = router.shape[0], fl["n_expert_used"] or 2
    logits = h @ router.T
    if (p + "ffn_gate_inp.bias") in W.t:                       # gpt-oss: chooses and weights
        logits = logits + W.get(p + "ffn_gate_inp.bias").reshape(-1).astype(np.float32)
    bias_ = W.get(p + "exp_probs_b.weight").reshape(-1) if (p + "exp_probs_b.weight") in W.t else None
    if fl.get("moe_sparsemixer"):
        # PhiMoE, as sml_moe_route runs it
        sel = np.zeros((h.shape[0], k), dtype=np.int64)
        w = np.zeros((h.shape[0], k), dtype=np.float32)
        lg = logits.astype(np.float64).copy()
        eps2 = 2.0 * fl["moe_jitter"]
        for j in range(k):
            best = np.argmax(lg, axis=1)
            mx = lg[np.arange(len(lg)), best][:, None]
            factor = np.maximum(np.abs(logits), mx)
            keep = ((mx - lg) / factor <= eps2) & np.isfinite(lg)
            w[:, j] = 1.0 / np.where(keep, np.exp(lg - mx), 0.0).sum(1)
            sel[:, j] = best
            lg[np.arange(len(lg)), best] = -np.inf
        return _moe_experts(Q, W, p, h, sel, w, E, imp_of)
    if fl.get("moe_sigmoid") or bias_ is not None or fl.get("moe_groups", 0) > 1:
        # the engine's V3-style router: choose on score + bias (within the best
        # groups), weight by the score
        if fl.get("moe_sigmoid"):
            sc = 1.0 / (1.0 + np.exp(-logits))
        else:
            sc = np.exp(logits - logits.max(1, keepdims=True))
            sc /= sc.sum(1, keepdims=True)
        ch = sc + (bias_[None, :] if bias_ is not None else 0.0)
        ng, tg = fl.get("moe_groups", 0), fl.get("moe_topk_group", 0)
        if ng > 1 and tg < ng and E % ng == 0:
            g = np.sort(ch.reshape(-1, ng, E // ng), axis=-1)[:, :, -2:].sum(-1)
            keep = np.argsort(-g, axis=1, kind="stable")[:, :tg]
            m = np.zeros_like(g, dtype=bool)
            np.put_along_axis(m, keep, True, axis=1)
            ch = np.where(np.repeat(m, E // ng, axis=1), ch, 0.0)
        sel = np.argsort(-ch, axis=1, kind="stable")[:, :k]
        w = np.take_along_axis(sc, sel, axis=1)
        if fl["moe_norm"]:
            w = w / (w.sum(1, keepdims=True) + 1e-20)
        if fl.get("moe_scale"):
            w = w * fl["moe_scale"]
        return _moe_experts(Q, W, p, h, sel, w, E, imp_of)
    sel = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(logits, sel, axis=1)
    if fl["moe_norm"]:
        w = np.exp(top - top[:, :1])
        w /= w.sum(1, keepdims=True)
    else:
        full = np.exp(logits - logits.max(1, keepdims=True))
        full /= full.sum(1, keepdims=True)
        w = np.take_along_axis(full, sel, axis=1)
    return _moe_experts(Q, W, p, h, sel, w, E, imp_of)


def _moe_experts(Q, W, p, h, sel, w, E, imp_of):
    out = np.zeros_like(h)
    # a shared expert (Qwen2-MoE) sees every token, times sigmoid(its gate)
    if (p + "ffn_up_shexp.weight") in W.t and (p + "ffn_gate_shexp.weight") not in W.t:
        # Nemotron-H: an ungated shared expert, relu^2
        ih = imp_of(h)
        su = Q.run(p + "ffn_up_shexp.weight", W.get(p + "ffn_up_shexp.weight"), h, ih)
        act = np.maximum(h @ su.T, 0.0) ** 2
        sd = Q.run(p + "ffn_down_shexp.weight", W.get(p + "ffn_down_shexp.weight"), act, imp_of(act))
        out += act @ sd.T
        del su, sd, act
    if (p + "ffn_gate_shexp.weight") in W.t:
        ih = imp_of(h)
        sg = Q.run(p + "ffn_gate_shexp.weight", W.get(p + "ffn_gate_shexp.weight"), h, ih)
        su = Q.run(p + "ffn_up_shexp.weight", W.get(p + "ffn_up_shexp.weight"), h, ih)
        g = h @ sg.T
        act = (g / (1.0 + np.exp(-g))) * (h @ su.T)
        sd = Q.run(p + "ffn_down_shexp.weight", W.get(p + "ffn_down_shexp.weight"), act, imp_of(act))
        sh = act @ sd.T
        if (p + "ffn_gate_inp_shexp.weight") in W.t:
            gw = W.get(p + "ffn_gate_inp_shexp.weight").reshape(-1).astype(np.float32)
            sh = sh / (1.0 + np.exp(-(h @ gw)))[:, None]
        out += sh
        del sg, su, sd, g, act, sh
    for e in range(E):
        rows_e, slot = np.nonzero(sel == e)
        Xe = h[rows_e] if len(rows_e) else h
        ie = imp_of(Xe)
        if (p + f"ffn_gate.{e}.weight") not in W.t:
            # ungated (Nemotron-H): up, relu^2, down
            wu = Q.run(p + f"ffn_up.{e}.weight", W.get(p + f"ffn_up.{e}.weight"), Xe, ie)
            act = np.maximum(Xe @ wu.T, 0.0) ** 2
            wd = Q.run(p + f"ffn_down.{e}.weight", W.get(p + f"ffn_down.{e}.weight"), act, imp_of(act))
            if len(rows_e):
                out[rows_e] += w[rows_e, slot][:, None] * (act @ wd.T)
            del wu, act, wd
            continue
        wg = Q.run(p + f"ffn_gate.{e}.weight", W.get(p + f"ffn_gate.{e}.weight"), Xe, ie)
        wu = Q.run(p + f"ffn_up.{e}.weight", W.get(p + f"ffn_up.{e}.weight"), Xe, ie)
        g = Xe @ wg.T
        if (p + f"ffn_gate.{e}.bias") in W.t:
            # gpt-oss: biased gate and up, the clamped SwiGLU, a biased down (added below)
            g = np.minimum(g + W.get(p + f"ffn_gate.{e}.bias").reshape(-1), 7.0)
            u_ = np.clip(Xe @ wu.T + W.get(p + f"ffn_up.{e}.bias").reshape(-1), -7.0, 7.0)
            act = g / (1.0 + np.exp(-1.702 * g)) * (u_ + 1.0)
        else:
            act = (g / (1.0 + np.exp(-g))) * (Xe @ wu.T)
        del g, wg, wu
        wd = Q.run(p + f"ffn_down.{e}.weight", W.get(p + f"ffn_down.{e}.weight"),
                   act, imp_of(act))
        if len(rows_e):
            y_ = act @ wd.T
            if (p + f"ffn_down.{e}.bias") in W.t:
                y_ = y_ + W.get(p + f"ffn_down.{e}.bias").reshape(-1)
            out[rows_e] += w[rows_e, slot][:, None] * y_
        del act, wd
    return out


def gdn_np(Q, W, p, h, Xa, ia, fl, eps, imp_of):
    """Gated DeltaNet for calibration, the recurrence the engine runs."""
    Bn, S, d = h.shape
    nk, dk = fl["gdn_nk"], fl["gdn_dk"]
    wq = Q.run(p + "gdn_qkvz.weight", W.get(p + "gdn_qkvz.weight"), Xa, ia)
    wb = Q.run(p + "gdn_ba.weight", W.get(p + "gdn_ba.weight"), Xa, ia)
    pr = h @ wq.T
    ba = h @ wb.T
    A = -np.exp(W.get(p + "gdn_a_log.weight").reshape(-1).astype(np.float32))
    nv = A.shape[0]
    nw = W.get(p + "gdn_norm.weight").reshape(-1).astype(np.float32)
    dv = nw.shape[0]
    kd, vd = nk * dk, nv * dv
    cd = 2 * kd + vd
    mix, z = pr[..., :cd], pr[..., cd:]
    cw = W.get(p + "gdn_conv.weight").astype(np.float32).reshape(cd, -1)
    K = cw.shape[1]
    u = np.zeros_like(mix)
    for k in range(K):
        sh = K - 1 - k
        u[:, sh:, :] += (mix[:, :S - sh, :] if sh else mix) * cw[:, k]
    u = u / (1.0 + np.exp(-u))
    q = u[..., :kd].reshape(Bn, S, nk, dk)
    kk = u[..., kd:2 * kd].reshape(Bn, S, nk, dk)
    v = u[..., 2 * kd:].reshape(Bn, S, nv, dv)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(dk)
    kk = kk / np.sqrt((kk * kk).sum(-1, keepdims=True) + 1e-6)
    rep = nv // nk
    q, kk = np.repeat(q, rep, axis=2), np.repeat(kk, rep, axis=2)
    beta = 1.0 / (1.0 + np.exp(-ba[..., :nv]))
    a = ba[..., nv:] + W.get(p + "gdn_dt_b.weight").reshape(-1)
    g = A * np.where(a > 20.0, a, np.log1p(np.exp(np.minimum(a, 20.0))))
    st = np.zeros((Bn, nv, dk, dv), dtype=np.float32)
    y = np.empty((Bn, S, nv, dv), dtype=np.float32)
    for t in range(S):
        st = st * np.exp(g[:, t])[:, :, None, None]
        mem = (st * kk[:, t][..., None]).sum(-2)
        delta = (v[:, t] - mem) * beta[:, t][..., None]
        st = st + kk[:, t][..., None] * delta[:, :, None, :]
        y[:, t] = (st * q[:, t][..., None]).sum(-2)
    y = y / np.sqrt((y * y).mean(-1, keepdims=True) + eps) * nw
    zz = z.reshape(Bn, S, nv, dv)
    y = (y * (zz / (1.0 + np.exp(-zz)))).reshape(Bn, S, vd)
    Xy = y.reshape(-1, vd)
    wo = Q.run(p + "gdn_out.weight", W.get(p + "gdn_out.weight"), Xy, imp_of(Xy))
    return (y @ wo.T).astype(np.float32)


def mamba1_np(Q, W, p, h, Xa, ia, fl, eps, imp_of):
    """Mamba-1 mixer for calibration, the recurrence the engine runs."""
    Bn, S, d = h.shape
    win = Q.run(p + "ssm1_in.weight", W.get(p + "ssm1_in.weight"), Xa, ia)
    pr = h @ win.T
    di = pr.shape[-1] // 2
    xin, z = pr[..., :di], pr[..., di:]
    cw = W.get(p + "ssm1_conv.weight").astype(np.float32)
    K = cw.shape[1]
    u = np.zeros_like(xin)
    for k in range(K):
        sh = K - 1 - k
        u[:, sh:, :] += (xin[:, :S - sh, :] if sh else xin) * cw[:, k]
    if (p + "ssm1_conv.bias") in W.t:
        u += W.get(p + "ssm1_conv.bias").reshape(-1)
    u = u / (1.0 + np.exp(-u))
    Xu = u.reshape(-1, di)
    wx = Q.run(p + "ssm1_x.weight", W.get(p + "ssm1_x.weight"), Xu, imp_of(Xu))
    xp = u @ wx.T
    A = -np.exp(W.get(p + "ssm1_a_log.weight").astype(np.float32).reshape(di, -1))
    N = A.shape[1]
    R = xp.shape[-1] - 2 * N
    tsr, Bm, Cm = xp[..., :R], xp[..., R:R + N], xp[..., R + N:]

    def rms(v, w, e):
        v = v / np.sqrt((v * v).mean(-1, keepdims=True) + e)
        return v * w if w is not None else v
    if fl.get("ssm_bcdt_eps"):
        e = fl["ssm_bcdt_eps"]
        Bm, Cm, tsr = rms(Bm, None, e), rms(Cm, None, e), rms(tsr, None, e)
    for nm, which in (("ssm1_dt_norm", 0), ("ssm1_b_norm", 1), ("ssm1_c_norm", 2)):
        if (p + nm + ".weight") in W.t:
            w_ = W.get(p + nm + ".weight").reshape(-1)
            if which == 0:
                tsr = rms(tsr, w_, eps)
            elif which == 1:
                Bm = rms(Bm, w_, eps)
            else:
                Cm = rms(Cm, w_, eps)
    wdt = W.get(p + "ssm1_dt.weight").astype(np.float32).reshape(di, R)
    dt = tsr @ wdt.T
    if (p + "ssm1_dt.bias") in W.t:
        dt = dt + W.get(p + "ssm1_dt.bias").reshape(-1)
    dt = np.where(dt > 20.0, dt, np.log1p(np.exp(np.minimum(dt, 20.0))))
    D = W.get(p + "ssm1_d.weight").reshape(-1)
    st = np.zeros((Bn, di, N), dtype=np.float32)
    y = np.empty((Bn, S, di), dtype=np.float32)
    for t in range(S):
        st = np.exp(A[None] * dt[:, t][..., None]) * st + dt[:, t][..., None] * Bm[:, t][:, None, :] * u[:, t][..., None]
        y[:, t] = (st * Cm[:, t][:, None, :]).sum(-1) + D * u[:, t]
    y = y * (z / (1.0 + np.exp(-z)))
    Xy = y.reshape(-1, di)
    wo = Q.run(p + "ssm1_out.weight", W.get(p + "ssm1_out.weight"), Xy, imp_of(Xy))
    return (y @ wo.T).astype(np.float32)


def mamba2_np(Q, W, p, h, Xa, ia, fl, eps, imp_of):
    """Mamba-2 mixer for calibration, the recurrence the engine runs."""
    Bn, S, d = h.shape
    win = Q.run(p + "ssm_in.weight", W.get(p + "ssm_in.weight"), Xa, ia)
    pr = h @ win.T
    di = W.shape(p + "ssm_out.weight")[1]
    cw = W.get(p + "ssm_conv.weight").astype(np.float32)
    cd, K = cw.shape
    A = -np.exp(W.get(p + "ssm_a_log.weight").reshape(-1).astype(np.float32))
    H = A.shape[0]
    dm = (pr.shape[-1] - di - cd - H) // 2
    z = pr[..., 2 * dm:2 * dm + di]
    xbc = pr[..., 2 * dm + di:2 * dm + di + cd]
    dt = pr[..., 2 * dm + di + cd:]
    conv = np.zeros_like(xbc)
    for k in range(K):
        sh = K - 1 - k
        conv[:, sh:, :] += (xbc[:, :S - sh, :] if sh else xbc) * cw[:, k]
    if (p + "ssm_conv.bias") in W.t:
        conv += W.get(p + "ssm_conv.bias").reshape(-1)
    conv = conv / (1.0 + np.exp(-conv))
    G = fl.get("ssm_groups") or 1
    N = fl["ssm_dstate"]
    P = di // H
    x = conv[..., :di].reshape(Bn, S, H, P)
    Bm = np.repeat(conv[..., di:di + G * N].reshape(Bn, S, G, N), H // G, axis=2)
    Cm = np.repeat(conv[..., di + G * N:di + 2 * G * N].reshape(Bn, S, G, N), H // G, axis=2)
    dt = dt + W.get(p + "ssm_dt_b.weight").reshape(-1)
    dt = np.where(dt > 20.0, dt, np.log1p(np.exp(np.minimum(dt, 20.0))))
    if fl.get("ssm_dt_hi"):
        dt = np.clip(dt, fl.get("ssm_dt_lo", 0.0), fl["ssm_dt_hi"])
    D = W.get(p + "ssm_d.weight").reshape(-1)
    st = np.zeros((Bn, H, P, N), dtype=np.float32)
    y = np.empty((Bn, S, H, P), dtype=np.float32)
    for t in range(S):
        dA = np.exp(dt[:, t] * A)
        st = st * dA[:, :, None, None] + (dt[:, t][:, :, None, None] * x[:, t][..., None]
                                          * Bm[:, t][:, :, None, :])
        y[:, t] = (st * Cm[:, t][:, :, None, :]).sum(-1) + D[None, :, None] * x[:, t]
    y = y.reshape(Bn, S, di) * (z / (1.0 + np.exp(-z)))
    ng = fl.get("ssm_norm_group") or di
    yg = y.reshape(Bn, S, di // ng, ng)
    yg = yg / np.sqrt((yg * yg).mean(-1, keepdims=True) + eps)
    y = yg.reshape(Bn, S, di) * W.get(p + "ssm_norm.weight").reshape(-1) if (p + "ssm_norm.weight") in W.t \
        else y.reshape(Bn, S, di)            # Falcon-H1: gated, not normed
    Xy = y.reshape(-1, di)
    wo = Q.run(p + "ssm_out.weight", W.get(p + "ssm_out.weight"), Xy, imp_of(Xy))
    return (y @ wo.T).astype(np.float32)


def qk_norm_np(x, w, n_heads, hd, eps, b=None, heads=False):
    """Q/K norm before RoPE, as the engine applies it: per head when the weight is
    one head long, over the whole projection when it is n_heads * head_dim long;
    with heads, per head with each head's own weight and bias row (a LayerNorm)."""
    if w is None:
        return x
    w = np.asarray(w, dtype=np.float32)
    if heads:
        y = x.reshape(x.shape[0], n_heads, hd)
        if b is not None:
            y = y - y.mean(-1, keepdims=True)
        y = y / np.sqrt((y * y).mean(-1, keepdims=True) + eps) * w.reshape(n_heads, hd)
        if b is not None:
            y = y + np.asarray(b, dtype=np.float32).reshape(n_heads, hd)
        return y.reshape(x.shape[0], n_heads * hd).astype(np.float32)
    if w.shape[0] == hd:
        y = x.reshape(x.shape[0], n_heads, hd)
        y = y / np.sqrt((y * y).mean(-1, keepdims=True) + eps) * w
        return y.reshape(x.shape[0], n_heads * hd).astype(np.float32)
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w).astype(np.float32)


def chat_block(src, toks, n_rows):
    """The model's OWN chat frame and stop tokens, read from its files and stored
    in the SSF (STANDARDS.md section 2): the chat template exactly as the
    publisher wrote it, BOS/EOS and every end-of-turn id from generation_config.
    Also the vocabulary gate: a tokenizer with more ids than the embedding has
    rows, or a special id outside the vocabulary, maps token 450 to the wrong
    row -- the "gibberish" failure of other tools -- so the build refuses."""
    def load(name):
        p = os.path.join(src, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {}
    tc, gc = load("tokenizer_config.json"), load("generation_config.json")
    tmpl = tc.get("chat_template")
    if isinstance(tmpl, list):     # [{"name": "default", "template": ...}, ...]
        named = {t.get("name"): t.get("template") for t in tmpl if isinstance(t, dict)}
        tmpl = named.get("default") or next(iter(named.values()), None)
    jp = os.path.join(src, "chat_template.jinja")
    if not tmpl and os.path.isfile(jp):
        tmpl = open(jp, encoding="utf-8").read()
    lookup = {t: i for i, t in enumerate(toks)}

    def tok_id(v):
        v = v.get("content") if isinstance(v, dict) else v
        return lookup.get(v) if isinstance(v, str) else None

    bos, eos = tok_id(tc.get("bos_token")), tok_id(tc.get("eos_token"))
    stops = gc.get("eos_token_id")
    stops = [stops] if isinstance(stops, int) else list(stops or [])
    if eos is not None and eos not in stops:
        stops.append(eos)
    if len(toks) > n_rows:
        raise SystemExit(f"vocabulary gate: the tokenizer has {len(toks)} ids but the embedding "
                         f"only {n_rows} rows -- ids past {n_rows - 1} would read another token's row")
    bad = [i for i in stops + [bos] if i is not None and not (0 <= i < max(len(toks), n_rows))]
    if bad:
        raise SystemExit(f"vocabulary gate: special token ids {bad} are outside the vocabulary")
    if not tmpl:
        print("  NOTE: no chat template in this checkpoint: it will be served as plain text "
              "(flagged amber)", flush=True)
    if not stops:
        print("  NOTE: no end-of-turn tokens found (flagged amber)", flush=True)
    return {"template": tmpl or "", "bos_id": bos, "eos_id": eos, "stop_ids": stops,
            "add_bos": bool(tc.get("add_bos_token", False)),
            "bos_token": toks[bos] if bos is not None and bos < len(toks) else "",
            "eos_token": toks[eos] if eos is not None and eos < len(toks) else ""}


def provenance(src, repo, licence):
    """Where the weights came from and what their licence asks of whoever ships
    them. The engine never reads this; a person distributing the file does. The
    weights are the publisher's, not ours, so the file carries their paperwork."""
    card = {}
    for name in ("README.md", "readme.md"):
        path = os.path.join(src, name)
        if os.path.isfile(path):
            text = open(path, encoding="utf-8", errors="replace").read()
            if text.startswith("---"):
                for line in text.split("---", 2)[1].splitlines():
                    if ":" in line and not line.startswith(" "):
                        k, v = line.split(":", 1)
                        card[k.strip()] = v.strip().strip("'\"")
            break
    lic = licence or card.get("license") or ""
    files = sorted(f for f in os.listdir(src)
                   if f.lower().startswith(("license", "licence", "notice", "use_policy")))
    out = {
        "checkpoint": os.path.basename(os.path.normpath(src)),
        "repo_id": repo or card.get("base_model") or "",
        "license": lic or "UNKNOWN",
        "license_files": files,
        "duties": LICENCE_DUTIES.get(lic.lower(), ["read the model licence before distributing"]
                                     if lic else ["licence unknown: do not distribute until it is"]),
        "built_by": "Sheep tooling/sml_build.py",
    }
    if not lic:
        print("  WARNING: no licence found for this checkpoint. Pass --license ID "
              "(e.g. llama3.2, apache-2.0) before distributing the file.")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="HF checkpoint directory")
    ap.add_argument("dst")
    ap.add_argument("--levels", default="2,2,2")
    ap.add_argument("--group", type=int, default=GROUP)
    ap.add_argument("--seqs", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--no-gptq", action="store_true")
    ap.add_argument("--gptq-check", action="store_true",
                    help="per tensor, keep GPTQ or plain rounding by held-out output error")
    ap.add_argument("--scale6", action="store_true",
                    help="v9: pack per-group scales and mins at 6 bits. Costs "
                         "0.31%% on weight error, saves 0.125 bits/weight -- "
                         "which is the entire size deficit vs the reference.")
    ap.add_argument("--hadamard", action="store_true",
                    help="quantize in a blockwise Hadamard-32 basis; the "
                         "engine rotates the activation to match. Zero bytes, "
                         "worth 1.103x end to end held out")
    ap.add_argument("--alloc", default="",
                    help="per-role code width for a flat build, e.g. "
                         "'attn_v=6,ffn_down=6'; anything unnamed stays at 4")
    ap.add_argument("--budget", type=int, default=0,
                    help="flat builds: a file-size ceiling in bytes; bytes the "
                         "--alloc leaves under it are spent widening more tensors "
                         "to 6 bits (see promote_order), never exceeding it")
    ap.add_argument("--layout", choices=["ladder", "flat"], default="ladder",
                    help="flat drops the prefix-readable ladder: 1.552x better "
                         "end-to-end at the same bytes, and ~2x faster per "
                         "thread, but --depth stops meaning anything")
    ap.add_argument("--exact", action="store_true",
                    help="store every supported checkpoint weight as f32 without quantization")
    ap.add_argument("--limit-layers", type=int, default=0)
    ap.add_argument("--repo", default="",
                    help="source repository id, recorded in the file (e.g. meta-llama/Llama-3.2-1B-Instruct)")
    ap.add_argument("--license", default="",
                    help="licence id of the source weights (llama3.2, apache-2.0, ...)")
    args = ap.parse_args()

    if args.exact and (args.hadamard or args.scale6 or args.budget or args.alloc):
        ap.error("--exact cannot be combined with quantization layout, scale, allocation, or budget options")
    level_bits = [int(b) for b in args.levels.split(",")]
    layout = LAYOUT_FLAT if args.exact or args.layout == "flat" else LAYOUT_LADDER
    if args.exact or layout == LAYOUT_FLAT:
        level_bits = [4]      # header records the base plane; width is per tensor

    # Per-role width for a flat build. "role=bits", comma separated, e.g.
    #   --alloc attn_v=6,ffn_down=6
    # Anything unnamed stays at 4.
    # "role=bits" or "role=bits:stride" -- stride N promotes every Nth layer.
    # The reference mix does exactly this: it spends 6 bits on attn_v and
    # ffn_down in HALF the layers, and on the embedding everywhere. Spending 6
    # bits on ffn_down in every layer, as the first attempt did, costs more than
    # they do on the wrong tensor while starving the head that produces the
    # logits being graded.
    _wide = {}
    for part in (args.alloc or "").split(","):
        if not part.strip():
            continue
        role, _, b = part.partition("=")
        if "@" in b:
            # role=bits@1;2;3 -- an EXPLICIT layer list, which is the whole
            # point: which layers damage the logits is a property of the model,
            # measured, not a stride guessed in advance. On Qwen2.5-3B the
            # measured top half overlaps "every other layer" by 3 of 18.
            bits, _, lst = b.partition("@")
            _wide[role.strip()] = (int(bits),
                                   set(int(v) for v in lst.split(";") if v.strip()))
        else:
            bits, _, stride = b.partition(":")
            _wide[role.strip()] = (int(bits), int(stride) if stride else 1)

    _promote = set()

    def alloc_bits(name):
        if name in _promote:
            return 6
        # token_embd was pinned at 4 while the row-dequant path only handled
        # nibbles. On a TIED model that tensor is the output head, so it
        # produces the logits being graded -- and the reference format stores
        # it at 6 bits. Quantizing the head coarser than they do and then
        # measuring logit correlation was self-inflicted. Now promotable.
        for role, (b, sel) in _wide.items():
            # a role names whole parts of the tensor name: "output" is the head
            # (output.weight), not every blk.N.attn_output that contains the word
            if role not in name.split(".") and name != role:
                continue
            if isinstance(sel, set):
                try:
                    return b if int(name.split(".")[1]) in sel else 4
                except (IndexError, ValueError):
                    return b
            if sel > 1:
                try:
                    layer = int(name.split(".")[1])
                except (IndexError, ValueError):
                    return b
                return b if layer % sel == 0 else 4
            return b
        # A TIED head defaults to 6 bits, as the reference stores it. Measured
        # on GPT-2 (2026-09-24): HF running our 4-bit weights got 6/20 top-1;
        # the same weights with only the tied embedding at full precision, 19/20.
        # One tensor was the whole loss. An --alloc naming it still decides.
        if name == "token_embd.weight" and getattr(W, "tied", False):
            return 6
        return 4
    group = args.group
    t0 = time.time()

    W = open_weights(args.src)

    # Refuse what we cannot compute, with the reason. Every architecture bug
    # this project has had produced PLAUSIBLE TEXT rather than an error -- the
    # RoPE convention inverted across five families, the rotated embedding
    # lookup, a flat plane read as a ladder. A model we cannot run correctly has
    # to say so at build time.
    from sml_archs import build_gate as _arch_build_gate, explain as _arch_explain  # noqa
    _mt = str((getattr(W, "vision", None) or {}).get("arch") or W.cfg.get("model_type", ""))
    _row, _arch_allowed = _arch_build_gate(
        _mt, W.cfg, override=os.environ.get("SML_UNGRADED_ARCH"))
    if not _arch_allowed:
        raise SystemExit("  " + _arch_explain(_mt, W.cfg))
    if _row is None or _row[0] != "ok":
        # Development only: building the file is how a new architecture gets
        # graded, and the row may only say "ok" AFTER the grade.
        print(f"  arch: {_mt} is NOT graded -- built on SML_UNGRADED_ARCH for "
              f"grading; do not ship this file", flush=True)
    else:
        print(f"  arch: {_mt} supported", flush=True)

    a = W.field("general.architecture")
    if isinstance(a, (bytes, bytearray)):
        a = a.decode()
    a = str(a)   # some sources hand this back as a 0-d ndarray that
                 # so every f-string worked and only json.dumps noticed -- 45
                 # minutes after the weights were already written
    hp = {
        "n_layer": int(W.field(f"{a}.block_count")),
        "n_embd": int(W.field(f"{a}.embedding_length")),
        "n_head": int(W.field(f"{a}.attention.head_count")),
        "n_head_kv": int(W.field(f"{a}.attention.head_count_kv",
                                 W.field(f"{a}.attention.head_count"))),
        "n_ff": int(W.field(f"{a}.feed_forward_length")),
        "n_ctx_train": int(W.field(f"{a}.context_length", 32768)),
        "rope_theta": float(W.field(f"{a}.rope.freq_base", 10000.0)),
        "rms_eps": float(W.field(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)),
    }
    if args.limit_layers:
        hp["n_layer"] = args.limit_layers
    n_layer, n_embd = hp["n_layer"], hp["n_embd"]
    fl = W.flags()   # needed before the layout is decided: the stored
                     # RoPE frequencies are a tensor like any other
    if fl.get("mla"):
        hp["n_head_kv"] = hp["n_head"]       # latent attention expands K and V for every head
    n_head, n_kv = hp["n_head"], hp["n_head_kv"]
    # head_dim from the checkpoint's config, not n_embd // n_head: Qwen3 has
    # 16 heads of 128 over a 1024 stream. Calibrating with the divided value
    # would quantize every layer against activations the model never makes.
    hd = int(fl.get("head_dim") or n_embd // n_head)
    n_attn = n_head * hd
    eps, theta = hp["rms_eps"], hp["rope_theta"]

    # ---- calibration batch: many short sequences keeps attention memory flat
    toks = W.field("tokenizer.tokens") or []
    merges = W.field("tokenizer.merges") or []
    uni = W.field("tokenizer.unigram")
    if uni is not None:
        # a Unigram tokenizer: calibrate on the ids the engine will actually feed it
        from sml_unigram import Unigram
        ids = Unigram(toks, uni).encode(CALIB_TEXT)
    else:
        ids = Bpe(toks, merges).encode(CALIB_TEXT)
    S, B = args.seq_len, args.seqs
    need = S * B
    if len(ids) < need:
        ids = (ids * (need // len(ids) + 1))
    seqs = np.asarray(ids[:need], dtype=np.int64).reshape(B, S)
    N = B * S
    print(f"arch={a} layers={n_layer}  calibration {B} x {S} = {N} samples", flush=True)
    print(f"gptq={'off' if args.no_gptq else 'on'}  levels={level_bits} group={group}")

    # ---- file layout (identical to sml_convert)
    names = [n for n in W.t.keys()
             if (not n.startswith("blk.") or int(n.split(".")[1]) < n_layer)
             # skip only when the head really is the embedding table
             and not (n == "output.weight" and getattr(W, "tied", True))]
    # RoPE frequencies are computed here and stored, rather than recomputed by
    # the engine from theta. That is what lets a context-extension scheme the
    # engine has never heard of (Llama 3.x, linear, dynamic) work with no engine
    # change at all, and it removes a whole class of builder/runtime drift --
    # there is only one place the frequencies are ever calculated.
    _rot = fl["n_rot"] or hd
    extra = {"rope_freqs": rope_freqs(float(hp["rope_theta"]), _rot // 2,
                                      W.cfg.get("rope_scaling"))}
    # per-layer attention, stored as small tensors the engine reads if present
    from sml_hf import layer_attn
    _lwin, _lrope, _lbase = layer_attn(W.cfg, hp["n_layer"])
    if _lwin is not None:
        extra["attn_window"] = np.asarray(_lwin, dtype=np.float32)
    if _lrope is not None:
        extra["attn_rope"] = np.asarray(_lrope, dtype=np.float32)
    if _lbase:
        extra["rope_freqs_local"] = rope_freqs(_lbase, _rot // 2, None)
    names = list(names) + list(extra)

    ladder, plain = [], []
    shapes = {}
    for n in names:
        # Shapes from the header where the weights can give them: loading every
        # tensor just to learn its shape converted a stacked-expert tensor once
        # per expert. A 1-D tensor reads as (1, n), so rows > 1 still means 2-D.
        # NOT `a` for the tensor -- that holds the architecture name, and
        # shadowing it silently put a weight tensor in the header.
        if n not in extra and hasattr(W, "shape"):
            rows, cols = W.shape(n)
        else:
            t = extra[n] if n in extra else W.get(n)
            rows, cols = (int(t.shape[0]), int(t.shape[1])) if t.ndim > 1 \
                else (1, int(t.shape[0]))
            del t
        shapes[n] = (rows, cols)
        # The router stays full precision: a coarser router picks different
        # experts, which is a different computation, not a small error.
        (ladder if (not args.exact and rows > 1 and cols % group == 0
                    and not n.endswith("ffn_gate_inp.weight")
                    # a norm is scales, never a matrix: Chameleon's per-head Q/K norms are
                    # [heads, head_dim] and were quantized as one (2026-10-19)
                    and not n.endswith(("_norm.weight", "_norm.bias")))
         and not n.endswith("position_embd.weight")   # added, never multiplied
         and not n.startswith("v.")   # vision tower: full precision, for now
         else plain).append(n)

    def key(n):
        if n.startswith("blk."):
            return (1, int(n.split(".")[1]), n)
        return (0, -1, n) if n.startswith("token_embd") else (2, 1 << 30, n)

    ladder.sort(key=key)
    plain.sort(key=key)

    # the model's own chat frame and stop tokens, and the vocabulary gate
    _chat = chat_block(args.src, toks, shapes["token_embd.weight"][0])

    # Two passes when there is a byte budget: the first lays the file out with
    # the --alloc widths and measures it; the bytes left under the budget buy
    # 6-bit widths in promote_order; the second pass lays out the final file.
    for _pass in range(2 if (args.budget and layout == LAYOUT_FLAT) else 1):
        meta, off = {}, HEADER_BYTES
        for n in plain:
            rows, cols = shapes[n]
            cnt = rows * cols
            meta[n] = {"kind": "f32", "rows": rows if args.exact else 1,
                       "cols": cols if args.exact else cnt, "off": off, "nbytes": cnt * 4}
            off += cnt * 4
        for n in ladder:
            rows, cols = shapes[n]
            # whole groups of zeros up to a multiple of four groups: see pad_groups
            cols = -(-cols // (4 * group)) * (4 * group)
            meta[n] = {"kind": "ladder", "rows": rows, "cols": cols, "group": group,
                       "lv": [], "perm_off": off, "perm_n": cols * 4}
            off += cols * 4
        gps = SCALE_SUPER // group
        for n in ladder:
            m = meta[n]; ng = m["cols"] // group
            # ceiling: a tensor whose group count is not a multiple of the
            # scale-superblock (Qwen2.5-0.5B has n_embd 896 -> 28 groups) would
            # otherwise silently allocate too few scales and read garbage
            nsb = (ng + gps - 1) // gps
            m["sd_off"], m["sd_n"] = off, m["rows"] * nsb * 2
            off += m["sd_n"]
            gbytes = (ng // 4) * 3 if args.scale6 else ng
            m["sq_off"], m["sq_n"] = off, m["rows"] * gbytes
            off += m["sq_n"]
            # The per-group MIN, stored exactly like the scale. This is what makes
            # the grid asymmetric for 0.19 b/w instead of the 1.0 b/w an fp16 min
            # would have cost -- the reason asymmetric was rejected the first time.
            m["md_off"], m["md_n"] = off, m["rows"] * nsb * 2
            off += m["md_n"]
            m["mq_off"], m["mq_n"] = off, m["rows"] * gbytes
            off += m["mq_n"]
        regions = []
        if layout == LAYOUT_FLAT:
            # Per tensor, not per level. A 4-bit tensor is one nibble plane; a 6-bit
            # tensor is a 2-bit plane plus a nibble plane, because a 6-bit code is
            # base + 16*extra and that decomposes into two dot products the kernels
            # already do. Level-major existed so a depth-1 read touched one
            # contiguous region; flat never does a partial read, so keeping a
            # tensor's planes together is the better locality.
            start = off
            for n in ladder:
                m = meta[n]
                nb = alloc_bits(n)
                m["layout"] = layout
                m["nbits"] = nb
                m["lv"] = []
                # the planes of each width, in the order the engine reads them
                npack = m["cols"] // (GROUPS_PER_SUPER * group)
                size = {2: m["rows"] * npack * super_bytes(group), 4: m["rows"] * (m["cols"] // 2),
                        1: m["rows"] * (m["cols"] // group) * 4}
                for bits in FLAT_PLANES[nb]:
                    m["lv"].append({"bits": bits, "codes_off": off, "codes_n": size[bits]})
                    off += size[bits]
            regions.append({"level": 0, "bits": 4, "group": group,
                            "off": start, "nbytes": off - start})
        else:
            for lv, bits in enumerate(level_bits):
                start = off
                for n in ladder:
                    m = meta[n]
                    npack = m["cols"] // (GROUPS_PER_SUPER * group)
                    cn = m["rows"] * npack * super_bytes(group)
                    m["lv"].append({"bits": bits, "codes_off": off, "codes_n": cn})
                    m["layout"] = layout
                    m["nbits"] = 0
                    off += cn
                regions.append({"level": lv, "bits": bits, "group": group,
                                "off": start, "nbytes": off - start})

        tokblob = build_vocab_blob(toks, merges,
                                   W.field("tokenizer.bos"),
                                   W.field("tokenizer.eos"),
                                   W.field("tokenizer.unigram"), W.field("tokenizer.pretok"))
        order = plain + ladder
        table_off = off; off += len(order) * REC_SIZE
        tok_off = off;   off += len(tokblob)
        json_off = off
        jblob = json.dumps({"arch": a, "hparams": hp, "group": group, "levels": regions,
                            "lossy_source": getattr(W, "lossy", True),
                            "precision": "exact-f32" if args.exact else "quantized",
                            "source": os.path.basename(args.src),
                            "provenance": provenance(args.src, args.repo, args.license),
                            "chat": _chat,
                            "gptq": not args.no_gptq, "calib_samples": N,
                            "arch_flags": W.flags(),
                            "vision": getattr(W, "vision", None),
                            "scale_super": SCALE_SUPER, "tensors": meta},
                           default=lambda o: o.item() if hasattr(o, "item") else str(o)
                           ).encode("utf-8")
        # ---- shared-activation groups: the scheduling facts the file can state
        #
        # Tensors that read the same activation and should be issued together. The
        # engine can infer these for a Llama-shaped model, but only because it was
        # taught Llama's shape; declaring them lets an architecture the engine has
        # never seen carry its own grouping.
        idx = {n: i for i, n in enumerate(order)}
        groups = []
        for L in range(n_layer):
            p_ = f"blk.{L}."
            qkv = [p_ + t for t in ("attn_q.weight", "attn_k.weight", "attn_v.weight")]
            if all(t in idx for t in qkv):
                groups.append([idx[t] for t in qkv])
            ff = [p_ + t for t in ("ffn_gate.weight", "ffn_up.weight")]
            if all(t in idx for t in ff):          # absent in a GELU FFN
                groups.append([idx[t] for t in ff])

        group_off = json_off + len(jblob)
        gblob = b"".join(
            struct.pack(GROUP_FMT, GROUP_KIND_ACT, len(g),
                        *(g + [-1] * (GROUP_MAX - len(g))))
            for g in groups)
        total = group_off + len(gblob)
        if _pass == 0 and args.budget and layout == LAYOUT_FLAT:
            left = args.budget - total
            for n in promote_order(ladder, n_layer):
                if left <= 0:
                    break
                if alloc_bits(n) != 4:
                    continue
                m = meta[n]
                cost = m["rows"] * (m["cols"] // (GROUPS_PER_SUPER * group)) * super_bytes(group) + 160
                if cost <= left:
                    _promote.add(n)
                    left -= cost
            print(f"budget {args.budget / 1e6:.2f} MB: alloc lays out {total / 1e6:.2f} MB; "
                  f"{len(_promote)} more tensors at 6 bits", flush=True)
    if args.budget and total > args.budget:
        raise SystemExit(f"layout {total} bytes is over the budget {args.budget}")

    lb = list(level_bits) + [0] * (MAX_LEVELS - len(level_bits))
    head = struct.pack(HEADER_FMT, MAGIC, VERSION, hp["n_layer"], n_embd, n_head,
                       n_kv, hp["n_ff"], meta["token_embd.weight"]["rows"],
                       hp["n_ctx_train"], theta, eps, group, len(level_bits), *lb,
                       len(order), table_off, json_off, len(jblob), tok_off,
                       len(tokblob), group_off if groups else 0,
                       fl["norm_type"], fl["rope_type"],
                       fl["attn_scale"], fl["logit_softcap"], fl["n_swa"],
                       (1 if fl["parallel_residual"] else 0)
                       | (2 if fl["gelu_ffn"] else 0)
                       | (4 if fl["alibi"] else 0)
                       | (8 if fl.get("gelu_erf") else 0)
                       | (16 if fl["gelu_ffn"] and "blk.0.ffn_gate.weight" in W.t else 0)
                       | (32 if fl.get("qknorm_late") else 0)
                       | (64 if fl.get("relu2") else 0)
                       | (128 if fl.get("alibi_scaled") else 0)
                       | (256 if fl.get("relu") else 0)
                       | (512 if fl.get("qknorm_heads") else 0)
                       | (1024 if fl.get("swiglu_oai") else 0)
                       | (2048 if fl.get("mla") else 0),
                       fl["n_rot"],
                       (MODEL_HADAMARD if args.hadamard else 0)
                       | (MODEL_SCALE6 if args.scale6 else 0)
                       | (MODEL_MINNEG if args.scale6 else 0)
                       | (MODEL_EXACT if args.exact else 0),
                       fl["emb_scale"], fl["res_scale"], fl["logit_scale"],
                       fl["n_expert_used"], (1 if fl["moe_norm"] else 0)
                       | (2 if fl.get("moe_sigmoid") else 0) | (4 if fl.get("moe_sparsemixer") else 0)
                       | (8 if fl.get("moe_group_max") else 0),
                       fl.get("attn_softcap", 0.0), fl.get("clip_qkv", 0.0),
                       # with the sparsemixer (bit 2) this field holds its jitter, not a scale
                       fl.get("moe_jitter", 0.0) if fl.get("moe_sparsemixer") else fl.get("moe_scale", 0.0), fl.get("moe_groups", 0), fl.get("moe_topk_group", 0),
                       fl.get("ssm_groups", 0), fl.get("ssm_dstate", 0), fl.get("ssm_norm_group", 0),
                       fl.get("ssm_dt_lo", 0.0), fl.get("ssm_dt_hi", 0.0),
                       fl.get("ssm_bcdt_eps", 0.0), fl.get("gdn_nk", 0), fl.get("gdn_dk", 0))
    print(f"writing {total / 1e9:.3f} GB -> {args.dst}", flush=True)

    # Written under a working name and renamed at the end, so a build that is
    # stopped (a session closing, a power cut) never leaves something that looks
    # like a finished model: the 7B truncated that way read as all-zero logits
    # (2026-10-06), and anything that scans the folder -- the model list, the
    # vision check -- would otherwise pick a half-written file up.
    part = args.dst + ".part"
    f = open(part, "wb")
    f.truncate(total)

    # Header, tensor table, vocab and JSON go down FIRST. None of them depend on
    # the quantization, and writing them last meant a trivial serialization error
    # destroyed a 45-minute build that had already computed every weight.
    f.seek(0); f.write(head)
    f.seek(table_off)
    for n in order:
        m = meta[n]
        coff = [e["codes_off"] for e in m.get("lv", [])] + [0] * MAX_LEVELS
        cn = [e["codes_n"] for e in m.get("lv", [])] + [0] * MAX_LEVELS
        f.write(struct.pack(REC_FMT, n.encode()[:NAME_LEN],
                            KIND_LADDER if m["kind"] == "ladder" else KIND_F32,
                            m["rows"], m["cols"], m.get("group", 0),
                            m.get("off", 0), m.get("nbytes", 0),
                            m.get("sd_off", 0), m.get("sd_n", 0),
                            m.get("sq_off", 0), m.get("sq_n", 0),
                            m.get("perm_off", 0), m.get("perm_n", 0),
                            m.get("md_off", 0), m.get("md_n", 0),
                            m.get("mq_off", 0), m.get("mq_n", 0),
                            *coff[:MAX_LEVELS], *cn[:MAX_LEVELS],
                            m.get("layout", LAYOUT_LADDER), m.get("nbits", 0)))
    f.seek(tok_off);  f.write(tokblob)
    f.seek(json_off); f.write(jblob)
    if gblob:
        f.seek(group_off); f.write(gblob)

    for n in plain:
        f.seek(meta[n]["off"])
        src = extra[n] if n in extra else W.get(n)
        f.write(np.asarray(src, dtype=np.float32).tobytes())

    if args.exact:
        f.flush()
        f.close()
        os.replace(part, args.dst)
        print(f"exact weights written without quantization: {args.dst} "
              f"({os.path.getsize(args.dst) / 1e9:.3f} GB)", flush=True)
        return

    Q = Quantizer(f, meta, level_bits, group, min(2, len(level_bits)),
                  not args.no_gptq, layout=layout, hadamard=args.hadamard,
                  scale6=args.scale6)
    Q.gptq_check = args.gptq_check

    # ---- walk the model one layer at a time
    emb = W.get("token_embd.weight")
    x = emb[seqs.reshape(-1)].astype(np.float32).reshape(B, S, n_embd)
    if fl["emb_scale"]:
        x = x * np.float32(fl["emb_scale"])
    # GPT-2 lineage: a learned row per position is added to the embedding,
    # and there is no rotary anywhere.
    abspos = "position_embd.weight" in W.t
    if abspos:
        x = x + W.get("position_embd.weight")[:S].astype(np.float32)[None]
    if "token_embd_norm.weight" in W.t:   # BLOOM: LayerNorm on the embedding
        x = norm(x, W.get("token_embd_norm.weight"), opt(W, "token_embd_norm.bias"),
                 eps, fl["norm_type"])
    alibi = alibi_slopes(n_head) if fl["alibi"] else None
    no_rope = abspos or alibi is not None
    rsc = np.float32(fl["res_scale"] or 1.0)
    attn_mul = fl["attn_scale"] or 1.0 / np.sqrt(hd)
    moe = "blk.0.ffn_gate_inp.weight" in W.t
    pos = np.arange(S, dtype=np.float32)
    mask = np.triu(np.full((S, S), -1e30, dtype=np.float32), 1)

    def imp_of(X):
        return np.sqrt(np.mean(X.astype(np.float64) ** 2, axis=0)).astype(np.float32)

    nt        = fl["norm_type"]
    n_rot     = fl["n_rot"]
    interlv   = fl["rope_type"] == 1
    par_res   = fl["parallel_residual"]
    gated     = "blk.0.ffn_gate.weight" in W.t

    for L in range(n_layer):
        p = f"blk.{L}."
        # OLMo 2 has no input norms: attention and FFN read the raw stream
        h = (norm(x, W.get(p + "attn_norm.weight"), opt(W, p + "attn_norm.bias"), eps, nt)
             if (p + "attn_norm.weight") in W.t else x)
        Xa = h.reshape(N, n_embd)
        ia = imp_of(Xa)
        # Each tensor keeps its OWN column order. Sharing one across the group
        # would let the runtime prepare the activation once instead of three
        # times -- worth about 1% -- and it was measured at 0.0038 correlation
        # on the full 3B, which is larger than the entire 0.0027 gap to Q4_K_M.
        # The 0.6% estimate that made it look free came from a single tensor.
        # Falcon-H1 runs Mamba beside attention on the same input and sums the two
        ssm_part = mamba2_np(Q, W, p, h, Xa, ia, fl, eps, imp_of) \
            if (p + "ssm_in.weight") in W.t and (p + "attn_q.weight") in W.t else None
        if (p + "attn_q.weight") not in W.t and not any(
                (p + t) in W.t for t in ("gdn_qkvz.weight", "ssm1_in.weight", "ssm_in.weight",
                                         "shortconv_in.weight", "attn_kv_a_mqa.weight")):
            attn_out = np.zeros_like(x)         # an MLP-only block (Nemotron-H)
        elif (p + "gdn_qkvz.weight") in W.t:
            attn_out = gdn_np(Q, W, p, h, Xa, ia, fl, eps, imp_of)
        elif (p + "ssm1_in.weight") in W.t:
            attn_out = mamba1_np(Q, W, p, h, Xa, ia, fl, eps, imp_of)
        elif (p + "ssm_in.weight") in W.t and ssm_part is None:
            attn_out = mamba2_np(Q, W, p, h, Xa, ia, fl, eps, imp_of)
        elif (p + "shortconv_in.weight") in W.t:
            # LFM2: B, C, x from one projection; causal depthwise conv of B*x
            win = Q.run(p + "shortconv_in.weight", W.get(p + "shortconv_in.weight"), Xa, ia)
            bcx = h @ win.T + bias(W, p + "shortconv_in.bias")
            Bq, Cq, Xq = np.split(bcx, 3, axis=-1)
            bx = Bq * Xq
            cw = W.get(p + "shortconv_conv.weight").astype(np.float32)   # [d, K]
            Kc = cw.shape[1]
            conv_ = np.zeros_like(bx)
            for kk in range(Kc):
                sh_ = Kc - 1 - kk
                if sh_ == 0:
                    conv_ += bx * cw[:, kk]
                else:
                    conv_[:, sh_:, :] += bx[:, :-sh_, :] * cw[:, kk]
            if (p + "shortconv_conv.bias") in W.t:
                conv_ += W.get(p + "shortconv_conv.bias").reshape(-1)
            yv = Cq * conv_
            Xy = yv.reshape(N, n_embd)
            wso = Q.run(p + "shortconv_out.weight", W.get(p + "shortconv_out.weight"), Xy, imp_of(Xy))
            attn_out = yv @ wso.T + bias(W, p + "shortconv_out.bias")
            del win, bcx, Bq, Cq, Xq, bx, conv_, yv, Xy, wso
        else:
            mla = (p + "attn_kv_a_mqa.weight") in W.t
            if mla:
                # latent attention as the engine expands it: heads [rotary | rest], V padded
                rot = fl["n_rot"]
                nope = hd - rot
                if (p + "attn_q_a.weight") in W.t:
                    wqa = Q.run(p + "attn_q_a.weight", W.get(p + "attn_q_a.weight"), Xa, ia)
                    ql = norm(h @ wqa.T, W.get(p + "attn_q_a_norm.weight"), None, eps, nt)
                    Xl = ql.reshape(N, -1)
                    wq = Q.run(p + "attn_q_b.weight", W.get(p + "attn_q_b.weight"), Xl, imp_of(Xl))
                    q = ql @ wq.T
                    del wqa, ql, Xl
                else:
                    wq = Q.run(p + "attn_q.weight", W.get(p + "attn_q.weight"), Xa, ia)
                    q = h @ wq.T
                wka = Q.run(p + "attn_kv_a_mqa.weight", W.get(p + "attn_kv_a_mqa.weight"), Xa, ia)
                ck = h @ wka.T
                rkv = ck.shape[-1] - rot
                kl = norm(ck[..., :rkv], W.get(p + "attn_kv_a_norm.weight"), None, eps, nt)
                Xk = kl.reshape(N, -1)
                wkb = Q.run(p + "attn_kv_b.weight", W.get(p + "attn_kv_b.weight"), Xk, imp_of(Xk))
                kv_ = (kl @ wkb.T).reshape(B, S, n_head, -1)
                vd = kv_.shape[-1] - nope
                kr = np.broadcast_to(ck[..., None, rkv:], (B, S, n_head, rot))
                k = np.concatenate([kr, kv_[..., :nope]], -1).reshape(B, S, n_head * hd).astype(np.float32)
                v = np.concatenate([kv_[..., nope:], np.zeros((B, S, n_head, hd - vd), np.float32)],
                                   -1).reshape(B, S, n_head * hd).astype(np.float32)
                gate_ = None
                del h, wq, wka, ck, kl, Xk, wkb, kv_, kr
            else:
                wq = Q.run(p + "attn_q.weight", W.get(p + "attn_q.weight"), Xa, ia)
                wk = Q.run(p + "attn_k.weight", W.get(p + "attn_k.weight"), Xa, ia)
                wv = Q.run(p + "attn_v.weight", W.get(p + "attn_v.weight"), Xa, ia)
                gate_ = None
                if (p + "attn_gate.weight") in W.t:
                    wg_ = Q.run(p + "attn_gate.weight", W.get(p + "attn_gate.weight"), Xa, ia)
                    gate_ = 1.0 / (1.0 + np.exp(-(h @ wg_.T)))
                    del wg_

                q = h @ wq.T + bias(W, p + "attn_q.bias")
                k = h @ wk.T + bias(W, p + "attn_k.bias")
                v = h @ wv.T + bias(W, p + "attn_v.bias")
                del h, wq, wk, wv

            qn = opt(W, p + "attn_q_norm.weight")
            kn = opt(W, p + "attn_k_norm.weight")
            o = np.empty((B, S, n_attn), dtype=np.float32)
            rep = n_head // n_kv
            for b in range(B):
                late = fl.get("qknorm_late")
                hq = fl.get("qknorm_heads")
                qb = (q[b] if late else qk_norm_np(q[b], qn, n_head, hd, eps, opt(W, p + "attn_q_norm.bias") if hq else None,
                                                   hq)).reshape(S, n_head, hd)
                kb = (k[b] if late else qk_norm_np(k[b], kn, n_kv, hd, eps, opt(W, p + "attn_k_norm.bias") if hq else None,
                                                   hq)).reshape(S, n_kv, hd)
                if not no_rope and (_lrope is None or _lrope[L]):
                    fq = (extra["rope_freqs_local"] if "rope_freqs_local" in extra
                          and _lwin is not None and _lwin[L] else extra["rope_freqs"])
                    qb = rope_np(qb, pos, theta, n_rot, interlv, fq)
                    kb = rope_np(kb, pos, theta, n_rot, interlv, fq)
                if late:
                    qb = qk_norm_np(qb.reshape(S, -1), qn, n_head, hd, eps).reshape(S, n_head, hd)
                    kb = qk_norm_np(kb.reshape(S, -1), kn, n_kv, hd, eps).reshape(S, n_kv, hd)
                vb = v[b].reshape(S, n_kv, hd)
                kb = np.repeat(kb, rep, axis=1)
                vb = np.repeat(vb, rep, axis=1)
                att = np.einsum("thd,shd->hts", qb, kb) * np.float32(attn_mul)
                if fl.get("attn_softcap"):
                    cap_ = np.float32(fl["attn_softcap"])
                    att = cap_ * np.tanh(att / cap_)
                att = att + mask[None]
                if alibi is not None:   # slope * (key - query): softmax-equal to HF's slope * key
                    sl = alibi * np.float32(attn_mul) if fl.get("alibi_scaled") else alibi
                    att = att + sl[:, None, None] * (pos[None, None, :] - pos[None, :, None])
                att -= att.max(axis=-1, keepdims=True)
                np.exp(att, out=att)
                att /= att.sum(axis=-1, keepdims=True)
                o[b] = np.einsum("hts,shd->thd", att, vb).reshape(S, n_attn)
            del q, k, v

            if gate_ is not None:
                o = o * gate_
            if mla:
                o = np.ascontiguousarray(o.reshape(B, S, n_head, hd)[..., :vd].reshape(B, S, n_head * vd))
            Xo = o.reshape(N, -1)
            wo = Q.run(p + "attn_output.weight", W.get(p + "attn_output.weight"),
                       Xo, imp_of(Xo))
            attn_out = o @ wo.T + bias(W, p + "attn_output.bias")
            if (p + "attn_post_norm.weight") in W.t:
                attn_out = norm(attn_out, W.get(p + "attn_post_norm.weight"), None, eps, nt)
            del o, Xo, wo

        if ssm_part is not None:
            attn_out = attn_out + ssm_part
        # Parallel residual (GPT-NeoX, Falcon, Phi): attention and FFN both read
        # the SAME residual and both add into it, rather than the FFN reading the
        # attention's output. Running it sequentially still produces fluent text,
        # so this has to come from the file rather than be assumed.
        ffn_in = x if par_res else x + rsc * attn_out
        if (p + "ffn_down.weight") not in W.t and (p + "ffn_gate_inp.weight") not in W.t:
            x = ffn_in                  # a block with no FFN (pure Mamba-2)
            del attn_out
            continue
        if (p + "ffn_norm.weight") in W.t:
            h = norm(ffn_in, W.get(p + "ffn_norm.weight"), opt(W, p + "ffn_norm.bias"), eps, nt)
        elif par_res and (p + "attn_norm.weight") in W.t:
            # one LayerNorm shared by both branches (Phi, GPT-J, Cohere)
            h = norm(ffn_in, W.get(p + "attn_norm.weight"), opt(W, p + "attn_norm.bias"), eps, nt)
        else:
            h = ffn_in
        Xf = h.reshape(N, n_embd)
        if (p + "ffn_gate_inp.weight") in W.t:     # per layer: some mix dense layers in
            ffn_out = moe_forward(Q, W, p, Xf, fl, imp_of).reshape(B, S, n_embd)
            if (p + "ffn_post_norm.weight") in W.t:
                ffn_out = norm(ffn_out, W.get(p + "ffn_post_norm.weight"), None, eps, nt)
            x = ffn_in + rsc * ffn_out
            del h, Xf, attn_out, ffn_in, ffn_out
            if L % 4 == 0 or L == n_layer - 1:
                print(f"  layer {L + 1}/{n_layer}  {time.time() - t0:.0f}s", flush=True)
            continue
        i_f = imp_of(Xf)
        wu = Q.run(p + "ffn_up.weight", W.get(p + "ffn_up.weight"), Xf, i_f)
        u = h @ wu.T + bias(W, p + "ffn_up.bias")
        del wu
        if gated:
            wg = Q.run(p + "ffn_gate.weight", W.get(p + "ffn_gate.weight"),
                       Xf, i_f)
            g = h @ wg.T
            del wg
            if fl["gelu_ffn"]:      # GeGLU (Gemma): the gate through GELU
                act = 0.5 * g * (1.0 + np.tanh(0.7978845608028654 *
                                               (g + 0.044715 * g ** 3))) * u
            else:
                act = (g / (1.0 + np.exp(-g))) * u
            del g, u
        else:
            # GELU FFN has no gate at all -- one projection, activated in place.
            xp = (p + "ffn_xielu.weight")
            if xp in W.t:
                # xIELU (Apertus), with the layer's own four numbers
                ap, an, be, ep = [float(v) for v in W.get(xp).reshape(-1)[:4]]
                act = np.where(u > 0, ap * u * u + be * u,
                               (np.expm1(np.minimum(u, ep)) - u) * an + be * u).astype(np.float32)
            elif fl.get("relu2"):
                act = np.maximum(u, 0.0) ** 2
            elif fl.get("relu"):
                act = np.maximum(u, 0.0)
            else:
                act = 0.5 * u * (1.0 + np.tanh(0.7978845608028654 *
                                               (u + 0.044715 * u ** 3)))
            del u
        del h, Xf

        Xd = act.reshape(N, hp["n_ff"])
        wd = Q.run(p + "ffn_down.weight", W.get(p + "ffn_down.weight"),
                   Xd, imp_of(Xd))
        ffn_out = act @ wd.T + bias(W, p + "ffn_down.bias")
        if (p + "ffn_post_norm.weight") in W.t:
            ffn_out = norm(ffn_out, W.get(p + "ffn_post_norm.weight"), None, eps, nt)
        x = x + attn_out + ffn_out if par_res else ffn_in + rsc * ffn_out
        del act, Xd, wd, attn_out, ffn_in, ffn_out

        if L % 4 == 0 or L == n_layer - 1:
            print(f"  layer {L + 1}/{n_layer}  {time.time() - t0:.0f}s", flush=True)

    # norm(), not rms_norm(): a LayerNorm model calibrated its head against
    # RMS-normed activations until 2026-09-24.
    hf = norm(x, W.get("output_norm.weight"), opt(W, "output_norm.bias"),
              eps, nt).reshape(N, n_embd)

    # The head takes the final hidden state, so it is quantized against hf. When
    # the model is UNTIED these are two different tensors and both must be
    # written -- quantizing only token_embd left output.weight as a table entry
    # full of zeros, which produced flat logits and looked like a wiring bug.
    if "output.weight" in meta:
        Q.run("output.weight", W.get("output.weight"), hf, imp_of(hf))
        # token_embd is then a lookup table only: no matvec consumes it, so
        # there are no activations to weight it by.
        Q.run("token_embd.weight", emb, None, None)
    else:
        Q.run("token_embd.weight", emb, hf, imp_of(hf))

    f.close()
    os.replace(part, args.dst)
    print(f"done in {time.time() - t0:.0f}s -> {os.path.getsize(args.dst) / 1e9:.3f} GB")
    if args.gptq_check:
        print(f"gptq-check: kept GPTQ on {Q.choice['gptq']} tensors, plain rounding on "
              f"{Q.choice['rounding']} (held-out output error decided)")


if __name__ == "__main__":
    main()
