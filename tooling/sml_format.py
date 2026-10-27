"""The SSF format: constants, packing, and the ladder quantizer.

Everything the format is, with no dependency on any other model format. This is
what the engine's binary header, tensor table and vocab are written against, and
what tooling/sml_read.py reads back.

Layout and rationale: sml/FORMAT.md.
"""
import struct

import numpy as np

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

MAGIC = b"SSF "
VERSION = 9
HEADER_BYTES = 4096
GROUP = 32          # scale granularity
SCALE_SUPER = 256   # weights sharing one fp16 super-scale
ROW_CHUNK = 2048    # rows dequantized at a time
MAX_LEVELS = 4
NAME_LEN = 64

# The engine reads a fixed binary table, never the JSON. Parsing JSON in C buys
# nothing and costs a dependency; the JSON stays for tooling and inspection.
#
#   header : magic, version, hparams, level table, section offsets
#   record : one per tensor, fixed size, directly mmap-able
# Architecture description, learned by reading how the reference engine handles
# its ~150 architectures. The lesson worth taking is which differences are
# STRUCTURAL and which are BEHAVIOURAL:
#
#   structural  -- QKV bias, tied head, Q/K norm, MoE experts. These are just
#                  "is the tensor present", so SML derives them from the tensor
#                  table and needs no flag at all. The reference engine declares
#                  them per architecture; that is ~150 places to keep in sync.
#   behavioural -- same tensors, different maths. These genuinely need declaring.
#
# So this block is small on purpose. Adding an architecture should be a row in
# the converter's config reader, not a new graph builder.
NORM_RMS, NORM_LAYER = 0, 1
ROPE_HALVED, ROPE_INTERLEAVED = 0, 1

HEADER_FMT = ("<4s"      # magic
              "I"        # version
              "7I"       # n_layer n_embd n_head n_head_kv n_ff n_vocab n_ctx_train
              "2f"       # rope_theta rms_eps
              "2I"       # group n_levels
              f"{MAX_LEVELS}I"   # level_bits
              "Q"        # n_tensors
              "6Q"       # table_off json_off json_n tok_off tok_n group_off
              "2I"       # norm_type rope_type
              "2f"       # attn_scale (0 = 1/sqrt(head_dim))  logit_softcap (0 = none)
              "2I"       # n_swa (0 = full attention)  arch_bits
              "2I"       # n_rot (0 = full head_dim rotated)  flags
              "3f"       # emb_scale res_scale logit_scale (0 = none; Granite)
              "2I"       # n_expert_used  moe_flags (bit 0: renormalize the chosen)
              "2f"       # attn_softcap (0 = none)  clip_qkv (0 = none)
              "f2I"      # moe_scale (0 = 1)  moe_groups  moe_topk_group (0/1 = none)
              "3I2f"     # ssm_groups ssm_dstate ssm_norm_group  ssm_dt_lo ssm_dt_hi
              "f"        # ssm_bcdt_eps: Falcon-Mamba's weightless dt/B/C norm (0 = none)
              "2I")      # gdn_nk gdn_dk: Gated DeltaNet key heads and width

# arch_bits
ARCH_PARALLEL_RESIDUAL = 1 << 0   # attention and FFN read the same input
ARCH_GELU_FFN          = 1 << 1   # no gate projection; up -> GELU -> down
ARCH_ALIBI             = 1 << 2   # linear distance bias, no rotary
ARCH_GELU_ERF          = 1 << 3   # the exact erf GELU (HF 'gelu'), not tanh's
ARCH_GELU_GATE         = 1 << 4   # gated FFN activated by GELU (GeGLU), not SiLU
ARCH_QKNORM_LATE       = 1 << 5   # Q/K norm after the rotary (HunYuan), not before
ARCH_RELU2             = 1 << 6   # ungated FFN activated by relu(x)^2
ARCH_RELU              = 1 << 8   # ungated FFN activated by plain relu(x) (OPT)
ARCH_SWIGLU_OAI        = 1 << 10  # gpt-oss experts' clamped SwiGLU
ARCH_QKNORM_HEADS      = 1 << 9   # Q/K norm stats per head, one weight/bias row per head (Chameleon)
ARCH_ALIBI_SCALED      = 1 << 7   # ALiBi added before the 1/sqrt(d) scale (Falcon)
REC_FMT = (f"<{NAME_LEN}s"  # name
           "4I"             # kind rows cols group
           "2Q"             # off nbytes            (f32 payload)
           "2Q"             # sd_off  sd_n          (fp16 per scale-superblock)
           "2Q"             # sq_off  sq_n          (uint8 per group)
           "2Q"             # perm_off perm_n       (int32 column permutation)
           "2Q"             # md_off  md_n          (fp16 min per superblock)
           "2Q"             # mq_off  mq_n          (uint8 min per group)
           f"{MAX_LEVELS}Q" # codes_off
           f"{MAX_LEVELS}Q"  # codes_n
           "2I")            # v8: layout nbits
REC_SIZE = struct.calcsize(REC_FMT)

# Code layout, per tensor (v8).
#
# The ladder stores 2-bit level-major planes and forces reconstruction to be
# LINEAR in the code, so that truncating the code is a coarser read of the same
# number. That property is what makes --depth auto possible and nothing else in
# this space has it. It is also not free: measured against a flat code of the
# same width at the same bytes, end to end and held out, the ladder costs
# 1.552x of quality (19.917% -> 12.834% logit error, top-1 83.6% -> 86.9%).
#
# So the layout is a per-tensor decision. Use FLAT unless a tensor is actually
# read at more than one depth.
LAYOUT_LADDER = 0
LAYOUT_FLAT   = 1

# Model-wide header flags. Bit 0: the quantized weights live in a blockwise
# Hadamard-32 basis, so the runtime rotates the activation to match. H is
# orthogonal and symmetric, so (W H)(H x) = W x exactly; blocks of 32 match the
# scale group so the rotation never crosses a scale boundary.
#
# Rejected once at 1.06x measured PER TENSOR. Re-measured end to end, held out,
# it is worth 1.103x at zero bytes -- the rotation is arithmetic, and decode
# leaves 75-90% of the FLOPs idle.
MODEL_HADAMARD = 1
# v9: per-group scales and mins are 6-bit, four groups packed into three bytes.
# This is the whole size deficit against the reference format -- our per-group
# metadata cost 0.625 bits/weight against Q4_K's 0.500, and the file was 2.25%
# larger. See RESULTS.md; the measured quality cost is 0.31%.
MODEL_SCALE6 = 2
MODEL_MINNEG = 4    # with SCALE6: mins <= 0, read as (field - 63) * md
MODEL_EXACT = 8     # every weight matrix is stored as an unquantized f32 payload


def pack6(q):
    """(rows, ng) values in 0..63 -> (rows, ng/4*3) bytes.

    Four 6-bit fields per three bytes, in the order the kernels consume groups,
    so an epilogue that reads four groups never straddles a pack boundary.
    """
    rows, ng = q.shape
    assert ng % 4 == 0, "group count must be a multiple of 4 to pack 6-bit"
    v = q.reshape(rows, ng // 4, 4).astype(np.uint32)
    out = np.empty((rows, ng // 4, 3), dtype=np.uint8)
    out[:, :, 0] = ( v[:, :, 0]        | (v[:, :, 1] << 6)) & 0xFF
    out[:, :, 1] = ((v[:, :, 1] >> 2)  | (v[:, :, 2] << 4)) & 0xFF
    out[:, :, 2] = ((v[:, :, 2] >> 4)  | (v[:, :, 3] << 2)) & 0xFF
    return out.reshape(rows, -1)


def unpack6(b, ng):
    """Inverse of pack6, for the reader and the gates."""
    rows = b.shape[0]
    t = b.reshape(rows, ng // 4, 3).astype(np.uint32)
    v = np.empty((rows, ng // 4, 4), dtype=np.uint8)
    v[:, :, 0] =  (t[:, :, 0]        & 63)
    v[:, :, 1] = ((t[:, :, 0] >> 6) | (t[:, :, 1] << 2)) & 63
    v[:, :, 2] = ((t[:, :, 1] >> 4) | (t[:, :, 2] << 4)) & 63
    v[:, :, 3] =  (t[:, :, 2] >> 2) & 63
    return v.reshape(rows, ng)


def hadamard32():
    h = np.array([[1.0]])
    while h.shape[0] < GROUP:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(GROUP)


def rotate32(A):
    """Apply the Hadamard blockwise along the last axis, in blocks of GROUP."""
    r, c = A.shape
    n = c - c % GROUP
    out = A.astype(np.float64).copy()
    out[:, :n] = (out[:, :n].reshape(r, n // GROUP, GROUP)
                  @ hadamard32()).reshape(r, n)
    return out

# ---- shared-activation groups (v7) -------------------------------------------
#
# A GROUP names tensors that read the SAME activation and should be issued as
# one unit. Three matrices being adjacent in a file is a storage fact; three
# matrices sharing an activation is a scheduling fact, and only the second lets
# a runtime decide anything.
#
# This is the first thing in the format that is neither weights nor metadata
# about weights: it is guidance. Measured worth on Qwen2.5-3B is 1.026x for the
# Q/K/V group and 1.030x once ffn_gate/ffn_up is added -- and the second gain is
# smaller precisely because those kernels were already efficient, which is the
# prediction the idea has to keep making to stay honest.
#
# The engine already infers both groups for Llama-shaped models. Putting them in
# the file does not make Qwen faster; it makes the knowledge portable to
# architectures whose grouping differs -- GPT-NeoX's fused QKV, MoE experts, a
# gate/up pair that is absent entirely in a GELU FFN.
#
# `group_off` is 0 in v6 files, which reads as "no groups declared" and falls
# back to what the engine works out for itself.
GROUP_MAX      = 6
GROUP_KIND_ACT = 1        # members share one input activation

GROUP_FMT  = f"<2I{GROUP_MAX}i"     # kind, n_members, tensor indices (-1 = unused)
GROUP_SIZE = struct.calcsize(GROUP_FMT)
KIND_F32, KIND_LADDER = 0, 1


def build_vocab_blob(tokens, merges, bos, eos, unigram=None, pretok=None):
    """Binary vocab for the engine. See sml/sml-vocab.h. `unigram`: the description
    sml_unigram reads from a Unigram tokenizer, packed as a trailer after the merges.

    The engine parses no JSON anywhere, and a tokenizer that needed a JSON parser
    would drag one in for the sake of a table of strings. Merges are resolved
    from string pairs to token ids HERE, so the encoder never does a string
    comparison in its inner loop.
    """
    ids = {t: i for i, t in enumerate(tokens)}
    raw = [t.encode("utf-8") for t in tokens]
    offs = np.zeros(len(raw) + 1, dtype=np.uint32)
    for i, b in enumerate(raw):
        offs[i + 1] = offs[i] + len(b)

    pairs = []
    for m in merges:
        parts = m.split(" ") if isinstance(m, str) else list(m)
        if len(parts) != 2:
            continue
        a, b = ids.get(parts[0]), ids.get(parts[1])
        if a is not None and b is not None:
            pairs.append((a, b))

    out = bytearray()
    out += struct.pack("<4I", len(tokens), len(pairs),
                       int(bos) if bos is not None else 0,
                       int(eos) if eos is not None else 0)
    out += offs.tobytes()
    out += b"".join(raw)
    out += np.asarray(pairs, dtype=np.uint32).tobytes()
    if unigram is not None:
        from sml_unigram import pack
        out += pack(unigram, len(tokens))
    elif pretok is not None:
        # a BPE tokenizer's own pipeline (sml_pretok.py); engines without it read up to here
        from sml_pretok import pack as pack_pretok
        out += pack_pretok(pretok)
    return bytes(out)


# --------------------------------------------------------------- quantization

SCALE_SEARCH = np.linspace(0.35, 1.10, 13, dtype=np.float32)


def _codes_for(g, scale, offset, maxcode):
    c = np.rint(g / scale[:, :, None] + offset)
    np.clip(c, 0, maxcode, out=c)
    return c


def slice_view(codes, level_bits, k, total_bits):
    """Reconstruct the integer implied by reading only the first `k` levels.

    The ladder is not a stack of independent quantizers -- that was measured and
    it loses. Levels are BIT SLICES of one well-scaled integer, so:

      * the scale is stored once, not once per level (saves 0.25 b/w per level)
      * a partial read is a prefix of the bits, never a residual that can blow up
      * reading every level reproduces the full-precision code exactly

    Reading k levels leaves `rem` low bits unknown; the midpoint of that interval
    is the minimum-error estimate.
    """
    taken = sum(level_bits[:k])
    rem = total_bits - taken
    if rem == 0:
        return codes.astype(np.float32)
    hi = (codes.astype(np.int32) >> rem) << rem
    return (hi + (1 << (rem - 1))).astype(np.float32)


def quantize_ladder(w, level_bits, group, target_level=None, imp=None,
                    two_level=True):
    """Quantize `w` (rows, cols) once, then bit-slice it into ladder levels.

    Three measured decisions are baked in here (numbers in the plan):

    SCALE SEARCH, not amax. Scaling a group by its amax sets the step near 1.8
    sigma when MSE-optimal is near 1.0 -- one outlier stretches the range for the
    other 31 weights. The search is scored at `target_level`, the depth a typical
    read stops at; optimizing full depth is the wrong objective when most reads
    stop early.

    IMPORTANCE WEIGHTING (the largest single lever, ~2x). Weight MSE is the wrong
    objective: what reaches the logits is OUTPUT error, which weights each input
    channel by its activation magnitude. LLM activations are dominated by a few
    outlier channels, so an unweighted objective spends bits uniformly on channels
    that contribute wildly unequally. `imp` is per-column activation RMS from a
    calibration pass.

    TWO-LEVEL SCALES. One fp16 per 256-weight scale-superblock plus one uint8 per
    group costs 0.3125 b/w, against 0.50 for fp16-per-group at g32 -- finer
    granularity for fewer bits.

    Returns (codes_uint8, scale_d_f16, scale_q8, [recon_after_each_level]).
    """
    rows, cols = w.shape
    ngroups = cols // group
    total_bits = sum(level_bits)
    maxcode = 2 ** total_bits - 1
    offset = maxcode / 2.0
    if target_level is None:
        target_level = len(level_bits)

    g = w.reshape(rows, ngroups, group)
    amax = np.max(np.abs(g), axis=2)  # (rows, ngroups)
    base = (amax / offset).astype(np.float32)
    base[base == 0] = 1e-8  # dead group: every code lands on `offset`, recon 0

    wt = None
    if imp is not None:
        wt = (np.asarray(imp, dtype=np.float32) ** 2).reshape(1, ngroups, group)

    best_err = None
    best_scale = None
    for frac in SCALE_SEARCH:
        s = base * frac
        c = _codes_for(g, s, offset, maxcode)
        view = slice_view(c, level_bits, target_level, total_bits)
        d = g - (view - offset) * s[:, :, None]
        d = d * d
        if wt is not None:
            d = d * wt
        err = np.sum(d, axis=2)
        if best_err is None:
            best_err, best_scale = err, s
        else:
            m = err < best_err
            best_err = np.where(m, err, best_err)
            best_scale = np.where(m, s, best_scale)

    # Encode the scale exactly as the file will hold it, THEN pick codes against
    # that value, so encoder and decoder never disagree.
    gps = SCALE_SUPER // group
    if two_level and ngroups % gps == 0:
        b = best_scale.reshape(rows, ngroups // gps, gps)
        d16 = (np.max(b, axis=2) / 255.0).astype(np.float16)
        dv = d16.astype(np.float32)
        dv[dv == 0] = 1e-8
        q8 = np.clip(np.rint(b / dv[:, :, None]), 1, 255).astype(np.uint8)
        scale = (q8.astype(np.float32) * dv[:, :, None]).reshape(rows, ngroups)
        q8 = q8.reshape(rows, ngroups)
    else:
        d16 = best_scale.astype(np.float16)
        scale = d16.astype(np.float32)
        scale[scale == 0] = 1e-8
        q8 = None

    codes = _codes_for(g, scale, offset, maxcode).astype(np.uint8)
    recons = []
    for k in range(1, len(level_bits) + 1):
        view = slice_view(codes, level_bits, k, total_bits)
        recons.append(((view - offset) * scale[:, :, None]).reshape(rows, cols))
    return codes, d16, q8, recons


def column_permutation(w):
    """Order input channels so each group holds weights of similar magnitude.

    A group's shared scale is set by its largest weight, so a group mixing one
    big weight with 31 small ones spends the scale on an outlier. Sorting by
    column max|w| makes groups homogeneous -- measured 1.55x on its own.

    Exact, not an approximation: (W P)(P^T x) = W x, so the matvec is unchanged
    provided the activation is gathered the same way. That gather is on a
    2048-element vector, nothing against streaming millions of weight bytes.

    Measured negative: ordering by activation*|w| instead is WORSE (0.133 vs
    0.048). The scale is set by weight magnitude, so that is what the ordering
    must cluster; importance belongs in the objective, not the ordering.
    """
    return np.argsort(np.max(np.abs(w), axis=0)).astype(np.int32)


def split_levels(codes, level_bits):
    """Split the full codes into per-level bit fields, most significant first."""
    total = sum(level_bits)
    out = []
    shift = total
    for b in level_bits:
        shift -= b
        out.append(((codes.astype(np.int32) >> shift) & ((1 << b) - 1)).astype(np.uint8))
    return out


GROUPS_PER_SUPER = 4  # 4 groups share one 2-bit-plane set


def pack_superblocks(level_codes, group):
    """Group-planar packing: bit-plane g IS group g.

    A superblock is 4 groups = 4*group weights, stored in `group` bytes:

        byte[j] = g0[j] | g1[j]<<2 | g2[j]<<4 | g3[j]<<6      j = 0..group-1

    Masking the bytes with 0x03 yields group 0's codes, shifting right by 2 and
    masking yields group 1, and so on. One read produces four groups with no
    shuffles, no gathers and no cross-lane movement. Must stay in lockstep with
    sml/sml-kernel.h -- the layout was chosen for that kernel.

    level_codes: (rows, cols) uint8 in 0..3
    returns:     (rows, nsuper, group) uint8
    """
    rows, cols = level_codes.shape
    nsuper = cols // (GROUPS_PER_SUPER * group)
    c = level_codes.reshape(rows, nsuper, GROUPS_PER_SUPER, group)
    out = np.zeros((rows, nsuper, group), dtype=np.uint8)
    for g in range(GROUPS_PER_SUPER):
        out |= c[:, :, g, :] << (2 * g)
    return out


def pack_flat4(codes, group=GROUP):
    """Pack 4-bit codes for SML_LAYOUT_FLAT, the inverse of sml_matvec_rows_flat4.

    One 32-byte block carries TWO whole groups: byte i's low nibble is the first
    group's column i, its high nibble is the second group's column i. Two blocks
    (64 bytes) therefore feed four groups, which is what lets the kernel keep the
    vectorised epilogue on the same four-group stride the ladder uses.

    Written from the layout rule rather than by mirroring the C, so that the two
    are independent readings of the format and agreeing means something. Graded
    against a scalar reference at 3.275e-07.

    codes: (rows, cols) uint8 in 0..15, permuted order. Returns (rows, cols/2).
    """
    rows, cols = codes.shape
    ng = cols // group
    if ng % 4:
        raise ValueError(f"flat4 needs a multiple of 4 groups, got {ng}")
    g = codes.astype(np.uint8).reshape(rows, ng, group)
    lo0, hi0 = g[:, 0::4, :], g[:, 1::4, :]     # groups 4q, 4q+1 -> first block
    lo1, hi1 = g[:, 2::4, :], g[:, 3::4, :]     # groups 4q+2, 4q+3 -> second
    b0 = (lo0 & 0x0F) | ((hi0 & 0x0F) << 4)
    b1 = (lo1 & 0x0F) | ((hi1 & 0x0F) << 4)
    out = np.concatenate([b0, b1], axis=2)      # (rows, nquad, 64)
    return np.ascontiguousarray(out.reshape(rows, cols // 2))


def pack_bits1(bits, group=GROUP):
    """The 1-bit plane of Q3/Q5/Q7: per row, per group, 4 bytes; bit j (least
    significant first) is weight j. The inverse of sml_matvec_bits1."""
    rows, cols = bits.shape
    b = (np.asarray(bits) & 1).astype(np.uint8).reshape(rows, cols // group, group)
    return np.packbits(b, axis=-1, bitorder="little")


def super_bytes(group):
    """Bytes per superblock per level."""
    return group

