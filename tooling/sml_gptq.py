#!/usr/bin/env python3
"""GPTQ-style error feedback for the SSF ladder.

Every quantizer lever tried so far -- MSE scale search, column permutation,
importance weighting -- decides each weight independently. That is what caps SSF
at Q4_0 quality (0.098 error at 4.31 b/w) when the model needs better than ~5%
to survive 36 layers of compounding.

Error feedback breaks that independence. Quantizing column j leaves a residual;
instead of discarding it, push it into the columns not yet quantized, weighted by
how correlated they are with j under the calibration Hessian H = X^T X. Later
columns then absorb earlier mistakes, and the OUTPUT error falls far more than
the weight error does.

Two adaptations the textbook version does not have:

  * DEPTH AWARE. The residual fed back is measured against the reconstruction at
    the read depth we actually care about (the 4.31 b/w view), not against the
    full stored code. Compensating full-depth error would optimise the level we
    already have working.
  * GROUP SCALES ARE CHOSEN LATE. Each group's scale is picked from the
    error-compensated weights at the moment the sweep reaches it, not from the
    original tensor -- otherwise the scale is fitted to weights that feedback is
    about to change.

Cost is real: an 11008-column Hessian is 485 MB, so the sweep is blocked. Within
a lazy-update block the updates are per column; across blocks they are one
matmul, which is what keeps this to seconds per tensor rather than hours.
"""
import numpy as np

LAZY = 128      # columns updated per-column before a batched flush


def _cholesky_inv_upper(H, damp_frac=0.01):
    """Upper-triangular Cholesky factor of H^-1, with damping.

    Damping is not optional: H from a few hundred calibration tokens is rank
    deficient for any layer wider than the token count, and an undamped inverse
    is noise.
    """
    n = H.shape[0]
    d = np.mean(np.diag(H))
    if not np.isfinite(d) or d <= 0:
        d = 1.0
    H = H + np.eye(n, dtype=H.dtype) * (damp_frac * d)

    # dead channels: a column with no activation energy cannot be compensated,
    # and would otherwise make the factorisation singular
    dead = np.diag(H) <= 0
    if dead.any():
        H[dead, dead] = 1.0

    for _ in range(5):
        try:
            L = np.linalg.cholesky(H)
            break
        except np.linalg.LinAlgError:
            H += np.eye(n, dtype=H.dtype) * (damp_frac * d)
    else:
        return None

    Hinv = np.linalg.inv(L).T @ np.linalg.inv(L)      # = H^-1
    Hinv = (Hinv + Hinv.T) * 0.5
    try:
        U = np.linalg.cholesky(Hinv).T.astype(np.float32)   # upper
    except np.linalg.LinAlgError:
        return None
    # a factor that is not finite would inject noise, not compensate error:
    # refuse it, and the caller rounds this block plainly instead
    return U if np.isfinite(U).all() else None


def gptq_block(w, X, group, total_bits, target_bits, imp=None, scale_search=None):
    """Quantize one column block with error feedback.

    w : (rows, cols) float32, permuted order, MODIFIED IN PLACE
    X : (T, cols) float32 activations in the same order
    Returns the integer codes (rows, cols) and the per-group scales (rows, ng).
    """
    rows, cols = w.shape
    ng = cols // group
    maxcode = 2 ** total_bits - 1
    offset = maxcode / 2.0
    rem = total_bits - target_bits          # bits unread at the target depth
    mid = (1 << (rem - 1)) if rem else 0

    # float64: with massive activations (OpenReasoning-Nemotron-1.5B's residual
    # reaches ~8,900) the diagonal spans 1e2..1e11, past float32's seven digits,
    # and the inverse came back as noise that error feedback then wrote into the
    # weights: rel-RMS 0.695 with GPTQ against 0.301 with plain rounding.
    H = X.T.astype(np.float64) @ X.astype(np.float64)
    U = _cholesky_inv_upper(H)
    if U is None:
        # Three Nones: the caller unpacks codes, scales AND mins, then falls back
        # to plain rounding on None. Two made the fallback itself crash, the
        # first time a Hessian failed (OpenReasoning-Nemotron-1.5B, ffn_down).
        return None, None, None

    # WHICH Hessian term weights the grid search.
    #
    # `imp` is the column norm, so imp**2 is diag(H). That is the wrong term.
    # Feedback compensates part of each column's error from the columns after
    # it, so the cost of an error in column j is not (dw)^2 * H_jj but
    # (dw)^2 / [H^-1]_jj -- the UNCOMPENSABLE share. diag(H) over-protects
    # columns whose error will be absorbed anyway and under-protects the ones
    # whose error is permanent.
    #
    # The engine already has the number: U[j,j] is sqrt([H^-1]_jj), used for
    # propagation and, until now, not for choosing the grid. Measured 1.015x
    # across four tensors, all moving the same way, at identical cost.
    wcol = 1.0 / np.maximum(np.diag(U), 1e-12) ** 2
    wcol = (wcol / max(float(wcol.mean()), 1e-30)).astype(np.float32)

    codes = np.zeros((rows, cols), dtype=np.int32)
    scales = np.zeros((rows, ng), dtype=np.float32)
    # Per-group offset. Error feedback and asymmetry are orthogonal and both are
    # needed: asymmetry decides HOW MUCH error a group carries, the feedback
    # decides WHERE it lands. Measured separately, asymmetry alone cut weight
    # error 1.7x at depth 2 and still lost 0.974 -> 0.892 on output, because the
    # error stopped being steered away from the columns that matter.
    mins = np.zeros((rows, ng), dtype=np.float32)

    for b0 in range(0, cols, LAZY):
        b1 = min(b0 + LAZY, cols)
        Wb = w[:, b0:b1].copy()
        Eb = np.zeros_like(Wb)

        for j in range(b0, b1):
            jl = j - b0
            gi = j // group
            if j % group == 0:
                # Pick this group's scale from the error-compensated weights, at
                # the moment the sweep reaches it.
                gj = j - b0
                blk = Wb[:, gj:gj + group] if gj + group <= Wb.shape[1] \
                    else w[:, j:j + group]
                sc, mn = _choose_grid(blk, maxcode, rem, mid,
                                      wcol[j:j + group])
                scales[:, gi] = sc
                mins[:, gi] = mn

            s = scales[:, gi]
            m0 = mins[:, gi]
            col = Wb[:, jl]
            c = np.clip(np.rint((col - m0) / s), 0, maxcode)
            codes[:, j] = c.astype(np.int32)

            # reconstruct at the READ DEPTH, not at full precision
            view = ((c.astype(np.int32) >> rem) << rem) + mid if rem else c
            recon = view * s + m0

            d = U[j, j]
            err = (col - recon) / (d if d != 0 else 1.0)
            Eb[:, jl] = err
            if jl + 1 < Wb.shape[1]:
                Wb[:, jl + 1:] -= np.outer(err, U[j, j + 1:b1])

        # one batched flush of this block's error into everything after it
        if b1 < cols:
            w[:, b1:] -= Eb @ U[b0:b1, b1:]
        w[:, b0:b1] = Wb

    return codes, scales, mins


def _choose_grid(blk, maxcode, rem, mid, imp):
    """MSE-optimal (step, min) for one group, scored at the target read depth.

    The range IS searched, and shrinking it below [lo, hi] is the point rather
    than a flaw: clipping a couple of outliers buys finer steps for the other
    thirty weights, which is the same trade the symmetric path makes when it
    lands near 1.0 sigma instead of amax's 1.8. Skipping this search on the
    argument that lo and hi are the true bounds cost 0.974 -> 0.944 at depth 2,
    where only 16 levels exist and two outliers can waste most of them. At depth
    3 it barely registers, which is exactly why testing at full depth hid it.
    """
    lo = np.min(blk, axis=1)
    hi = np.max(blk, axis=1)
    c = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    half[half == 0] = 1e-8
    wt = (imp.astype(np.float32) ** 2)[None, :] if imp is not None else None
    best_e = best_s = best_m = None
    for frac in np.linspace(0.20, 1.10, 19, dtype=np.float32):
        h = half * frac
        m0 = (c - h).astype(np.float32)
        s = (2.0 * h / maxcode).astype(np.float32)
        s[s == 0] = 1e-8
        q = np.clip(np.rint((blk - m0[:, None]) / s[:, None]), 0, maxcode)
        view = ((q.astype(np.int32) >> rem) << rem) + mid if rem else q
        d = blk - (view * s[:, None] + m0[:, None])
        d = d * d
        if wt is not None:
            d = d * wt[:, :d.shape[1]]
        e = np.sum(d, axis=1)
        if best_e is None:
            best_e, best_s, best_m = e, s, m0
        else:
            better = e < best_e
            best_e = np.where(better, e, best_e)
            best_s = np.where(better, s, best_s)
            best_m = np.where(better, m0, best_m)
    return best_s, best_m
