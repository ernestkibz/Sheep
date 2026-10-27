# The `.ssf` format  (Sheep Standard Format)

SML does not store tensors. It stores **levels of a residual ladder**, laid out so that reading
only the coarse level is a pure sequential stream.

## Why level-major

Conventional formats are tensor-major: each tensor's bytes are contiguous, and a
forward pass reads all of them. That makes "read a cheap approximation of the whole model" a
strided walk across the entire file, which is the one access pattern DRAM and the prefetcher
handle worst.

`.ssf` is **level-major, then layer-major within a level**:

```
[ header ][  L0: layer 0..N  ][  L1: layer 0..N  ][  L2: layer 0..N  ]
   4 KB      ~0.85 GB             ~0.85 GB            ~1.1 GB
```

An L0-only forward pass touches exactly one contiguous 0.85 GB region and never pulls an L1 byte
into a cache line it did not need. That property is the whole point of the file.

## The ladder

Each 2-D weight is quantized as a sum of residuals. For level *l* with `bits_l` and group size G:

```
offset_l = (2^bits_l - 1) / 2          # 2-bit -> 1.5, 3-bit -> 3.5
scale_l  = amax(residual, per group) / offset_l
code_l   = clip(round(residual / scale_l + offset_l), 0, 2^bits_l - 1)   # UNSIGNED
residual = residual - (code_l - offset_l) * scale_l
```

Reconstruction is a sum over the levels read:

```
w ~= SUM_l (code_l - offset_l) * scale_l
```

Codes are **unsigned** on purpose. The dot product then decomposes into an unsigned integer
accumulation plus a correction that costs one multiply per group:

```
dot = SUM_i (code_i - offset) * scale * x_i
    = scale * ( SUM_i code_i * x_i  -  offset * SUM_i x_i )
      \____________________/     \_____________/
       vpmaddubsw-friendly        precomputed once per group, reused by every row
```

`SUM_i x_i` depends only on the activation, so it is computed once per token and amortized across
every row of every matrix. This is what makes a 2-bit level cheap in arithmetic as well as bytes.

## Bit packing (planar within a group)

G = 64 weights per group. For 2-bit codes that is 16 bytes:

```
byte[j] = c[j] | c[j+16]<<2 | c[j+32]<<4 | c[j+48]<<6      for j in 0..15
```

One 16-byte load yields four 16-code vectors by shift-and-mask alone — no shuffles, no gathers.
The layout is chosen for the kernel rather than the kernel written for the layout.

## Per-tensor level payload

Codes and scales are stored as **separate contiguous arrays**, not interleaved blocks:

```
codes[rows][ngroups][G*bits/8]     uint8
scales[rows][ngroups]              float16
```

Two sequential streams, both prefetcher-friendly, and the scale array stays SIMD-aligned. Scales
are 12.5% of the code bytes at G=64/2-bit, so keeping them separate costs nothing in traffic.

## Header

4 KB, JSON: architecture, hyperparameters, the level table (bits, group, file offset, byte
count), the tensor table (name, shape, kind, per-level offsets), and the tokenizer.

The C runtime **parses no other model format**. Conversion happens offline; the runtime reads
only this format. That removes an entire format-parsing surface from the engine.

### Exact F32 files

Header flag bit 3 (`MODEL_EXACT`) marks an exact-weight file. Its weight tensors
are `KIND_F32` records containing the original row/column shape and contiguous
float32 values; no ladder or flat-code quantization is applied. The file uses
the flat layout and the runtime computes those matrices with the dense F32 CPU
path.

---

## v7: shared-activation groups

Everything above describes what the model **is**. A group describes how its
computation **relates**.

```
GROUP { kind = shared_activation, members = [attn_q, attn_k, attn_v] }
GROUP { kind = shared_activation, members = [ffn_gate, ffn_up] }
```

Three matrices being adjacent in a file is a storage fact. Three matrices
reading the same activation is a **scheduling fact**, and only the second lets a
runtime decide anything. The engine issues a group as one dispatch.

### Why this is worth a section in the format

It was measured, and it made a prediction that held.

`attn_k` and `attn_v` are 256 rows against `attn_q`'s 2048. Dispatched alone
they ran *slower the more threads were added* — 6.6 GB/s on one thread, 3.8 on
six — while wide tensors scaled 6.5 → 20.7 over the same range. One thread
reached the same rate as everything else, so the kernel was never at fault: the
barrier cost more than the work it divided. Grouping them with Q gave **1.026x**
end to end.

The principle then predicted a second group — `ffn_gate` and `ffn_up` also read
the post-`ffn_norm` vector — and predicted it would be worth *less*, because
those two already ran at 22 GB/s and had little dispatch overhead left to
reclaim. Measured: **1.030x**. Right in direction and roughly right in
magnitude, which is the part that makes it a rule rather than an anecdote.

### What it does not do

**It does not make Qwen faster.** The fusion did that, and the engine can infer
Q/K/V and gate/up for itself on a Llama-shaped model. Declaring them in the file
makes the knowledge *portable*: GPT-NeoX's fused QKV, MoE expert groups, or any
grouping this engine has never met can come from the model instead of requiring
the engine to have been taught that architecture first.

A file that declares no groups — every v6 and earlier — reads `group_off = 0`,
and the engine falls back to inference. Declared groups supersede it.

### Layout

`group_off` in the header, occupying the slot `reserved` held through v6 (always
written as 0, so old files mean "none"). The section is a run of fixed 32-byte
records terminated by a zero `kind` or end of file:

```
uint32  kind        1 = members share one input activation
uint32  n           members present, 1..6
int32   member[6]   tensor-table indices, -1 unused
```

### Verified end to end

Not asserted — the engine reports which schedule it used, because a file can
declare groups while the runtime quietly ignores them, and that difference is
the entire claim.

```
v6 file (none declared)    groups:  0 declared by the file, 72 inferred by the engine
v7 file                    groups: 48 declared by the file,  0 inferred by the engine
```

48 is 24 layers x 2 groups on Qwen2.5-0.5B; 72 is 36 x 2 on the 3B. Quality
unchanged, top-1 matching. The first group's members are tensor indices
`[125, 123, 126]` — deliberately non-consecutive, because Q, K and V are not
adjacent in the tensor table. The grouping is a relationship, not proximity,
which is precisely what a storage-only format cannot express.

### The principle

> **SML = data + relationships + guidance.**

Precision guidance was tried and does not pay — thirteen mechanisms, recorded in
`RESULTS.md`, from contextual sparsity to Jacobian-weighted rounding. Scheduling
guidance does. The centre of this format is not a smarter quantizer; it is a
description of computational relationships a generic model file does not carry.


## v8 — the code layout becomes a per-tensor decision

Every version up to v7 stored one thing: the ladder. Level-major 2-bit planes,
with reconstruction forced to be **linear in the code** so that truncating the
code is a coarser read of the same number. That property is the whole basis of
`--depth auto`, and nothing else in this space has it.

It was never free, and until now it had never been priced.

| | ladder | flat, same width, same bytes |
|---|---|---|
| logit error, held out, end to end | 19.917% | **12.834%** |
| top-1 agreement | 83.6% | **86.9%** |
| single-thread throughput, 400 MB working set | 5.12 GB/s | **10.11 GB/s** |

**1.552x of quality and 1.975x of per-thread throughput.** Two passes over two
level arrays with two epilogues, where a flat code is one of each. Every tensor
paid it, including the ones never read at more than one depth — which is nearly
all of them.

So v8 moves the choice into the record:

```c
uint32_t layout;   // SML_LAYOUT_LADDER | SML_LAYOUT_FLAT
uint32_t nbits;    // FLAT: code width. LADDER: 0, header level_bits rules
```

Both appended at the END of `sml_rec`, so a v7 record is a byte-exact prefix of
a v8 one and migration is a prefix copy plus defaults. Everything built before
v8 is a ladder by construction, and still loads.

### FLAT packing

One 32-byte block carries **two whole groups**: byte *i*'s low nibble is the
first group's column *i*, its high nibble is the second group's column *i*. Two
blocks — 64 bytes — therefore feed four groups.

That grouping is not arbitrary. The vectorised epilogue runs once per four
groups, and an epilogue running twice as often at group 32 as at group 64 has
already cost real bandwidth in this project once. Flat keeps the same stride.

`pack_flat4` in `tooling/sml_format.py` and `sml_matvec_rows_flat4` in
`sml/sml-kernel.c` are written from this rule independently rather than one
mirroring the other, so that agreeing between them means something. They agree
to 3.275e-07.

### When to keep the ladder

When a tensor is genuinely read at more than one depth — a model being fitted to
available RAM, where `--depth auto` is doing real work. That is a capability the
reference format cannot express at all, and it remains the reason this format
exists. It is now a decision made per tensor with its price known, instead of a
default paid everywhere by everything.
