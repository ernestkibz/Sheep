"""The quantizer's two searches through sml/sml-quant.dll (fused C, every core):
the same results as Quantizer.asym_grid and Quantizer.field_search in
sml_quantize.py, group by group. None from available() when the library is not
built; the quantizer then uses the GPU (sml_grid_gpu.py) or numpy.
"""
import ctypes
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = {}
_f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_i32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")


def available():
    if "lib" not in _LIB:
        path = os.path.join(_HERE, "..", "sml", "sml-quant.dll" if os.name == "nt" else "libsml-quant.so")
        try:
            lib = ctypes.CDLL(os.path.abspath(path))
            lib.sml_asym_grid.argtypes = [_f32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_void_p, _f32, ctypes.c_int, ctypes.c_int, ctypes.c_int, _f32, _f32]
            lib.sml_field_search.argtypes = [_f32, ctypes.c_int, ctypes.c_int, ctypes.c_int, _f32, _f32, _i32, _i32,
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_int]
            _LIB["lib"] = lib
        except OSError:
            _LIB["lib"] = None
    return _LIB["lib"]


def _weights(imp, ng, group):
    if imp is None:
        return None
    wt = np.asarray(imp, dtype=np.float32).reshape(ng * group)
    return np.ascontiguousarray(wt / max(float(wt.mean()), 1e-30) + np.float32(1e-6), dtype=np.float32)


def asym_grid(wp, group, nbits, target_bits=None, imp=None, ncand=20, iters=3, refit=True):
    lib = available()
    rows, cols = wp.shape
    ng = cols // group
    rem = 0 if target_bits is None else nbits - target_bits
    facs = np.ascontiguousarray(np.concatenate([np.linspace(0.6, 0.85, max(2, ncand // 3)),
                                                np.linspace(0.88, 1.04, max(2, ncand - ncand // 3))]), np.float32)
    w = np.ascontiguousarray(wp, dtype=np.float32)
    wt = _weights(imp, ng, group)
    out_l = np.empty((rows, ng), np.float32)
    out_s = np.empty((rows, ng), np.float32)
    lib.sml_asym_grid(w, rows, cols, group, nbits, rem, wt.ctypes.data if wt is not None else None, facs, len(facs),
                      iters, int(bool(refit and not rem)), out_l, out_s)
    return out_l.astype(wp.dtype), out_s.astype(wp.dtype)


def field_search(w, d16, q8, md16, mq8, group, gps, nmax, hi, moff, mmax, imp, radius=1):
    lib = available()
    rows, cols = w.shape
    ng = cols // group
    dv = np.ascontiguousarray(np.repeat(d16.astype(np.float32), gps, axis=1)[:, :ng])
    dm = np.ascontiguousarray(np.repeat(md16.astype(np.float32), gps, axis=1)[:, :ng])
    q8o = np.ascontiguousarray(q8, dtype=np.int32).copy()
    mq8o = np.ascontiguousarray(mq8, dtype=np.int32).copy()
    wt = _weights(imp, ng, group)
    lib.sml_field_search(np.ascontiguousarray(w, dtype=np.float32), rows, cols, group, dv, dm, q8o, mq8o,
                         int(nmax), int(hi), int(moff), int(mmax), wt.ctypes.data if wt is not None else None,
                         int(radius))
    return (q8o.astype(np.uint8), mq8o.astype(np.uint8),
            q8o.astype(np.float32) * dv, (mq8o.astype(np.float32) - moff) * dm)
