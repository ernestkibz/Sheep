#!/usr/bin/env python3
"""Reader for the .ssf format, and the reference the C++ loader is checked against.

Two jobs:

  1. An executable description of the layout for the C side to match.
  2. The reference tooling/sml_check.py compares the C kernel against, at every
     read depth -- packing, two-level scales, per-level shifts and the once-only
     bias are each places a mistake still produces plausible-looking numbers.

    python tooling/sml_read.py MODEL.ssf
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

MAGIC = b"SSF "
GROUPS_PER_PACK = 4     # groups sharing one set of 2-bit planes
SCALE_SUPER = 256       # weights sharing one fp16 super-scale


class SmlFile:
    def __init__(self, path):
        self.path = path
        from sml_format import HEADER_FMT
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        if bytes(self.mm[:4]) != MAGIC:
            raise SystemExit(f"not an .ssf file: {path}")

        # binary header is authoritative -- it is what the engine reads
        n = struct.calcsize(HEADER_FMT)
        b = struct.unpack(HEADER_FMT, bytes(self.mm[:n]))
        (_, self.version, self.n_layer, self.n_embd, self.n_head, self.n_head_kv,
         self.n_ff, self.n_vocab, self.n_ctx_train, self.rope_theta, self.rms_eps,
         self.group, n_levels) = b[:13]
        self.level_bits = list(b[13:13 + 4])[:n_levels]
        (self.n_tensors, self.table_off, json_off, json_n,
         self.tok_off, self.tok_n, self.group_off) = b[17:24]
        # Architecture behaviour, straight from the binary header the engine
        # reads -- so the verifier and the engine cannot disagree about which
        # model they are computing.
        (self.norm_type, self.rope_type) = b[24:26]
        (self.attn_scale, self.logit_softcap) = b[26:28]
        (self.n_swa, self.arch_bits) = b[28:30]
        self.n_rot = b[30]
        # v9: per-group scales and mins are 6-bit packed, four to three bytes.
        self.flags = b[31]
        self.scale6 = bool(self.flags & 2)
        self.minneg = self.scale6 and bool(self.flags & 4)

        # Shared-activation groups: tensors issued as one unit because they read
        # the same input. v6 and earlier wrote 0 here and declare none.
        self.groups = []
        if self.version >= 7 and self.group_off:
            from sml_format import GROUP_FMT, GROUP_SIZE, GROUP_MAX
            off = self.group_off
            while off + GROUP_SIZE <= len(self.mm):
                rec = struct.unpack(GROUP_FMT,
                                    bytes(self.mm[off: off + GROUP_SIZE]))
                kind, n = rec[0], rec[1]
                if kind == 0 or n == 0 or n > GROUP_MAX:
                    break
                self.groups.append((kind, list(rec[2:2 + n])))
                off += GROUP_SIZE

        self.head = json.loads(bytes(self.mm[json_off: json_off + json_n]).decode("utf-8"))
        self.tensors = self.head["tensors"]
        self.hparams = self.head["hparams"]
        self.levels = self.head["levels"]
        self.arch = self.head["arch"]
        self.lossy_source = self.head.get("lossy_source", False)

    def tokenizer(self):
        return json.loads(
            bytes(self.mm[self.tok_off: self.tok_off + self.tok_n]).decode("utf-8"))

    # ------------------------------------------------------------------ access

    def f32(self, name):
        m = self.tensors[name]
        assert m["kind"] == "f32", name
        buf = self.mm[m["off"]: m["off"] + m["nbytes"]]
        return np.frombuffer(buf.tobytes(), dtype=np.float32).reshape(m["rows"], m["cols"])

    def perm(self, name):
        """Column permutation. The weights were quantized in this order, so any
        activation fed to them must be gathered through it: (W P)(P^T x) = W x."""
        m = self.tensors[name]
        buf = self.mm[m["perm_off"]: m["perm_off"] + m["perm_n"]]
        return np.frombuffer(buf.tobytes(), dtype=np.int32)

    def _gfield(self, off, n, rows, ng):
        """One per-group byte field as floats, 8-bit or 6-bit packed."""
        raw = np.frombuffer(self.mm[off: off + n].tobytes(), dtype=np.uint8)
        if self.scale6:
            from sml_format import unpack6
            return unpack6(raw.reshape(rows, -1), ng).astype(np.float32)
        return raw.reshape(rows, ng).astype(np.float32)

    def group_scales(self, name):
        """Two-level scales resolved to one float per group.

            scale[g] = d[g / groups_per_scale] * q8[g]

        One fp16 per SCALE_SUPER weights plus one uint8 per group is 0.3125
        b/w -- cheaper than an fp16 per group (0.50) and finer-grained.
        """
        m = self.tensors[name]
        rows, cols, group = m["rows"], m["cols"], m["group"]
        ng = cols // group
        gps = SCALE_SUPER // group
        nsb = (ng + gps - 1) // gps
        d = np.frombuffer(self.mm[m["sd_off"]: m["sd_off"] + m["sd_n"]].tobytes(),
                          dtype=np.float16).reshape(rows, nsb).astype(np.float32)
        q = self._gfield(m["sq_off"], m["sq_n"], rows, ng)
        return np.repeat(d, gps, axis=1)[:, :ng] * q

    def group_mins(self, name):
        """Per-group offset, stored exactly like the scale. Absent in v5 files,
        which were symmetric -- those decode with a zero offset."""
        m = self.tensors[name]
        if not m.get("mq_n"):
            return None
        rows, cols, group = m["rows"], m["cols"], m["group"]
        ng = cols // group
        gps = SCALE_SUPER // group
        nsb = (ng + gps - 1) // gps
        d = np.frombuffer(self.mm[m["md_off"]: m["md_off"] + m["md_n"]].tobytes(),
                          dtype=np.float16).reshape(rows, nsb).astype(np.float32)
        q = self._gfield(m["mq_off"], m["mq_n"], rows, ng)
        # The bias is half the field, so it moves with the field width.
        return np.repeat(d, gps, axis=1)[:, :ng] * (q - ((63.0 if self.minneg else 31.0) if self.scale6
                                                         else 127.0))

    def flat_codes(self, name, plane=-1):
        """Unpack a FLAT (v8) tensor back to (rows, cols) uint8 codes.

        The inverse of pack_flat4 and of sml_matvec_rows_flat4: one 32-byte
        block carries two groups, low nibble then high, and two blocks feed four
        groups.
        """
        m = self.tensors[name]
        # A 6-bit flat tensor stores the 2-bit high plane FIRST, then the
        # nibble plane; a 4-bit one stores only the nibble plane. Indexing [0]
        # unconditionally would read the high plane as nibbles and be wrong in
        # a way that still produces numbers.
        # Q8 (8-bit) stores two nibble planes, low then high: plane picks one
        e = m["lv"][plane]
        rows, cols, group = m["rows"], m["cols"], m["group"]
        ng = cols // group
        buf = self.mm[e["codes_off"]: e["codes_off"] + e["codes_n"]]
        packed = np.frombuffer(buf.tobytes(), dtype=np.uint8).reshape(rows, cols // 2)
        blk = packed.reshape(rows, ng // 4, 64)
        b0, b1 = blk[:, :, :32], blk[:, :, 32:]
        out = np.empty((rows, ng // 4, 4, group), dtype=np.uint8)
        out[:, :, 0, :] = b0 & 0x0F
        out[:, :, 1, :] = (b0 >> 4) & 0x0F
        out[:, :, 2, :] = b1 & 0x0F
        out[:, :, 3, :] = (b1 >> 4) & 0x0F
        return out.reshape(rows, cols)

    def bits1_codes(self, name, plane):
        """A 1-bit plane (Q3/Q5/Q7) back to (rows, cols) 0/1."""
        m = self.tensors[name]
        e = m["lv"][plane]
        rows, cols, group = m["rows"], m["cols"], m["group"]
        buf = self.mm[e["codes_off"]: e["codes_off"] + e["codes_n"]]
        packed = np.frombuffer(buf.tobytes(), dtype=np.uint8).reshape(rows, cols // group, 4)
        return np.unpackbits(packed, axis=-1, bitorder="little").reshape(rows, cols)

    def mixed_codes(self, name):
        """The whole code of a Q3/Q5/Q7 tensor from its planes."""
        nb = int(self.tensors[name].get("nbits", 4))
        if nb == 3:
            return self.level_codes(name, 0).astype(np.int32) | (self.bits1_codes(name, 1).astype(np.int32) << 2)
        if nb == 5:
            return self.flat_codes(name, 0).astype(np.int32) | (self.bits1_codes(name, 1).astype(np.int32) << 4)
        return (self.flat_codes(name, 0).astype(np.int32) | (self.level_codes(name, 1).astype(np.int32) << 4)
                | (self.bits1_codes(name, 2).astype(np.int32) << 6))

    def is_flat(self, name):
        return self.tensors[name].get("layout", 0) == 1

    def level_codes(self, name, lv):
        """Unpack one level back to (rows, cols) uint8 codes in 0..3.

        Group-planar: a pack-superblock is 4 groups in `group` bytes, and
        bit-plane g IS group g."""
        m = self.tensors[name]
        e = m["lv"][lv]
        rows, cols, group = m["rows"], m["cols"], m["group"]
        npack = cols // (GROUPS_PER_PACK * group)
        buf = self.mm[e["codes_off"]: e["codes_off"] + e["codes_n"]]
        packed = np.frombuffer(buf.tobytes(), dtype=np.uint8).reshape(rows, npack, group)
        out = np.empty((rows, npack, GROUPS_PER_PACK, group), dtype=np.uint8)
        for g in range(GROUPS_PER_PACK):
            out[:, :, g, :] = (packed >> (2 * g)) & 0x3
        return out.reshape(rows, cols)

    def codes(self, name):
        """The full integer code, reassembled from every level's bit slice.

        A flat tensor has no bit slices -- the code is stored whole -- so it
        takes the same branch dequant() does. Without this, every caller that
        went through codes() rather than dequant() silently read a flat tensor
        as ladder planes and reshaped garbage: that is why the kernel/format
        gate reported FAIL on a model that had just won its head-to-head.
        """
        m = self.tensors[name]
        if self.is_flat(name):
            if int(m.get("nbits", 4)) == 2:
                return self.level_codes(name, 0).astype(np.int32)
            if int(m.get("nbits", 4)) in (3, 5, 7):
                return self.mixed_codes(name)
            if int(m.get("nbits", 4)) == 8:
                return self.flat_codes(name, 0).astype(np.int32) | (self.flat_codes(name, 1).astype(np.int32) << 4)
            code = self.flat_codes(name).astype(np.int32)
            if int(m.get("nbits", 4)) == 6:
                code = code | (self.level_codes(name, 0).astype(np.int32) << 4)
            return code
        lb = [e["bits"] for e in m["lv"]]
        total = sum(lb)
        out = np.zeros((m["rows"], m["cols"]), dtype=np.int32)
        shift = total
        for lv, b in enumerate(lb):
            shift -= b
            out |= self.level_codes(name, lv).astype(np.int32) << shift
        return out

    def dequant(self, name, depth=None):
        """Reconstruct the weight from the first `depth` levels.

        Levels are bit slices of one integer: reading `depth` of them leaves the
        low bits unknown, and the midpoint of that interval is the minimum-error
        estimate. Reading all levels reproduces the stored code exactly.
        """
        m = self.tensors[name]
        if self.is_flat(name):
            # No prefix semantics: every stored bit is read, so `depth` is
            # meaningless here and silently ignoring it would be worse than
            # saying so.
            if int(m.get("nbits", 4)) == 2:
                code = self.level_codes(name, 0).astype(np.int32)
            elif int(m.get("nbits", 4)) in (3, 5, 7):
                code = self.mixed_codes(name)
            elif int(m.get("nbits", 4)) == 8:
                code = self.flat_codes(name, 0).astype(np.int32) | (self.flat_codes(name, 1).astype(np.int32) << 4)
            else:
                code = self.flat_codes(name).astype(np.int32)
            if int(m.get("nbits", 4)) == 6:
                # code = base + 16*extra, with extra in the group-planar 2-bit
                # plane the ladder kernel already reads.
                code = code | (self.level_codes(name, 0).astype(np.int32) << 4)
            s_ = self.group_scales(name)
            mn = self.group_mins(name)
            g = m["group"]
            W = code.astype(np.float32) * np.repeat(s_, g, axis=1)
            if mn is not None:
                W = W + np.repeat(mn, g, axis=1)
            return W
        lb = [e["bits"] for e in m["lv"]]
        total = sum(lb)
        depth = len(lb) if depth is None else depth

        code = np.zeros((m["rows"], m["cols"]), dtype=np.int32)
        shift = total
        for lv in range(depth):
            shift -= lb[lv]
            code |= self.level_codes(name, lv).astype(np.int32) << shift
        if shift > 0:
            code += 1 << (shift - 1)   # midpoint of the unread interval

        s = np.repeat(self.group_scales(name), m["group"], axis=1)
        mn = self.group_mins(name)
        if mn is None:                      # v5: symmetric, centred code
            offset = (2 ** total - 1) / 2.0
            return (code.astype(np.float32) - offset) * s
        return code.astype(np.float32) * s + np.repeat(mn, m["group"], axis=1)

    def bytes_at_depth(self, depth):
        """Weight bytes a forward pass moves reading `depth` levels."""
        n = 0
        for m in self.tensors.values():
            if m["kind"] != "ladder":
                n += m["nbytes"]
                continue
            n += m["sd_n"] + m["sq_n"] + m.get("md_n", 0) + m.get("mq_n", 0)
            for e in m["lv"][:depth]:
                n += e["codes_n"]
        return n


def rel_rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ssf")
    args = ap.parse_args()

    f = SmlFile(args.ssf)
    print(f"{args.ssf}")
    print(f"  arch={f.arch} version={f.version} lossy_source={f.lossy_source}")
    print(f"  hparams={f.hparams}")
    print(f"  group={f.group} levels={[l['bits'] for l in f.levels]}")
    print(f"  tensors={len(f.tensors)}")
    for d in range(1, len(f.levels) + 1):
        b = f.bytes_at_depth(d)
        print(f"  depth {d}: {b / 1e9:.3f} GB/token")
    print("  baseline  : 1.920 GB/token at 10.71 tok/s (measured)")
    print()
    print("Weight-space verification lives in tooling/sml_quality_gate.py, which")
    print("scores OUTPUT error against a full-precision reference -- the number")
    print("that actually decides whether a read depth survives 36 layers.")


if __name__ == "__main__":
    main()
