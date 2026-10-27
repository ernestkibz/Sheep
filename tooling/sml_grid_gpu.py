"""The quantizer's two searches on the GPU (torch): the per-group asymmetric grid
and the field search. The same arithmetic as Quantizer.asym_grid and
Quantizer.field_search in sml_quantize.py, a block of rows at a time, in float32.

On numpy they ran at about 0.5M weights a second on this laptop's 6 cores: a 7B
took about four hours to quantize (2026-10-06, Ernest: "someone just quantizes
their choice, plugs it in and off the bat can use Sheep"). Every group is
independent, so the GPU does them all at once.

Not byte-identical to numpy: reductions sum in another order, so a near-tie
between two candidates can fall the other way. The quality is the same (the
weighted error is compared in tooling/sml_grid_gpu_check.py).
"""
import os

import numpy as np

_T = {}


def available():
    """'cuda' when torch sees a CUDA card, else None (the numpy path is used)."""
    if "dev" not in _T:
        try:
            import torch
            _T["dev"] = "cuda" if torch.cuda.is_available() else None
        except Exception:
            _T["dev"] = None
    return _T["dev"]


def _weights(imp, ng, group, dev):
    import torch
    if imp is None:
        return torch.ones((1, ng, group), dtype=torch.float32, device=dev)
    wt = np.asarray(imp, dtype=np.float32).reshape(1, ng, group)
    wt = wt / max(float(wt.mean()), 1e-30) + np.float32(1e-6)
    return torch.from_numpy(wt).to(dev)


def _blocks(rows, cols):
    """Row blocks of about SML_GRID_GPU_BLOCK weights (read at each call: the
    quantizer lowers it when the card runs out of memory)."""
    per = int(os.environ.get("SML_GRID_GPU_BLOCK", 1 << 22))
    blk = max(1, per // max(cols, 1))
    return [(r0, min(rows, r0 + blk)) for r0 in range(0, rows, blk)]


def asym_grid(wp, group, nbits, target_bits=None, imp=None, ncand=20, iters=3, refit=True, dev=None):
    """(mins, steps) per group, as Quantizer.asym_grid."""
    import torch
    dev = dev or available() or "cpu"
    rows, cols = wp.shape
    ng = cols // group
    n = float((1 << nbits) - 1)
    rem = 0 if target_bits is None else nbits - target_bits
    refit = refit and not rem
    facs = np.concatenate([np.linspace(0.6, 0.85, max(2, ncand // 3)),
                           np.linspace(0.88, 1.04, max(2, ncand - ncand // 3))]).astype(np.float32)
    W = _weights(imp, ng, group, dev)
    ws = W.sum(-1, keepdim=True)
    out_l = np.empty((rows, ng), np.float32)
    out_s = np.empty((rows, ng), np.float32)
    tiny = torch.tensor(1e-12, dtype=torch.float32, device=dev)
    with torch.no_grad():
        for r0, r1 in _blocks(rows, cols):
            g = torch.from_numpy(np.ascontiguousarray(wp[r0:r1], dtype=np.float32)).to(dev).reshape(-1, ng, group)
            lo = g.amin(-1, keepdim=True)
            hi = g.amax(-1, keepdim=True)
            c = (lo + hi) * 0.5

            def err(l, step):
                q = torch.clamp(torch.round((g - l) / step), 0, n)
                if rem:
                    qv = (((q.to(torch.int32) >> rem) << rem) + (1 << (rem - 1))).to(torch.float32)
                else:
                    qv = q
                d2 = (qv * step + l - g) ** 2
                return q, (d2 * W).sum(-1, keepdim=True)

            def ls(l, step, e):
                q, _ = err(l, step)
                for _ in range(iters):
                    qm = (W * q).sum(-1, keepdim=True) / ws
                    gm = (W * g).sum(-1, keepdim=True) / ws
                    dq = q - qm
                    var = (W * dq * dq).sum(-1, keepdim=True)
                    cov = (W * dq * (g - gm)).sum(-1, keepdim=True)
                    s2 = torch.where(var > 1e-12, cov / torch.clamp(var, min=1e-12), step)
                    s2 = torch.clamp(s2, min=1e-12)
                    l2 = gm - s2 * qm
                    q2, e2 = err(l2, s2)
                    better = e2 < e
                    l = torch.where(better, l2, l)
                    step = torch.where(better, s2, step)
                    q = torch.where(better, q2, q)
                    e = torch.where(better, e2, e)
                return l, step, e

            bl = bs = be = None
            for sfac in facs:
                half = (hi - lo) * 0.5 * float(sfac)
                l = c - half
                step = torch.maximum(2.0 * half, tiny) / n
                _, e = err(l, step)
                if refit:
                    l, step, e = ls(l, step, e)
                if be is None:
                    bl, bs, be = l, step, e
                else:
                    better = e < be
                    bl = torch.where(better, l, bl)
                    bs = torch.where(better, step, bs)
                    be = torch.where(better, e, be)
            out_l[r0:r1] = bl[..., 0].cpu().numpy()
            out_s[r0:r1] = bs[..., 0].cpu().numpy()
    return out_l.astype(wp.dtype), out_s.astype(wp.dtype)


def field_search(w, d16, q8, md16, mq8, group, gps, nmax, hi, moff, mmax, imp, radius=1, dev=None):
    """As Quantizer.field_search: each group's stored step and min fields moved to
    the best neighbour (+-radius each) by activation-weighted error."""
    import torch
    dev = dev or available() or "cpu"
    rows, cols = w.shape
    ng = cols // group
    dv = np.repeat(d16.astype(np.float32), gps, axis=1)[:, :ng]
    dm = np.repeat(md16.astype(np.float32), gps, axis=1)[:, :ng]
    W = _weights(imp, ng, group, dev)
    q8o = q8.astype(np.int32).copy()
    mq8o = mq8.astype(np.int32).copy()
    with torch.no_grad():
        for r0, r1 in _blocks(rows, cols):
            g = torch.from_numpy(np.ascontiguousarray(w[r0:r1], dtype=np.float32)).to(dev).reshape(-1, ng, group)
            qs0 = torch.from_numpy(q8o[r0:r1]).to(dev)
            qm0 = torch.from_numpy(mq8o[r0:r1]).to(dev)
            d = torch.from_numpy(dv[r0:r1]).to(dev)[..., None]
            m = torch.from_numpy(dm[r0:r1]).to(dev)[..., None]
            be = bs = bm = None
            for a in range(-radius, radius + 1):
                qs = torch.clamp(qs0 + a, 1, hi)
                S = torch.clamp(qs[..., None].to(torch.float32) * d, min=1e-12)
                for b in range(-radius, radius + 1):
                    qm = torch.clamp(qm0 + b, 0, mmax)
                    M = (qm[..., None].to(torch.float32) - moff) * m
                    c = torch.clamp(torch.round((g - M) / S), 0, nmax)
                    e = (((c * S + M - g) ** 2) * W).sum(-1)
                    if be is None:
                        be, bs, bm = e, qs, qm
                    else:
                        better = e < be
                        be = torch.where(better, e, be)
                        bs = torch.where(better, qs, bs)
                        bm = torch.where(better, qm, bm)
            q8o[r0:r1] = bs.cpu().numpy()
            mq8o[r0:r1] = bm.cpu().numpy()
    return (q8o.astype(np.uint8), mq8o.astype(np.uint8),
            q8o.astype(np.float32) * dv, (mq8o.astype(np.float32) - moff) * dm)
