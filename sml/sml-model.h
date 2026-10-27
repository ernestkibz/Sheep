// SSF model loading.
//
// The file is memory-mapped, never read into a buffer. That is not an
// optimisation detail, it is the whole point of the level-major layout: an
// L0-only forward pass touches one contiguous region and the OS never faults in
// a byte of L1 or L2. Reading the file would defeat the format.
//
// The engine parses the fixed binary header and tensor table only. It never
// touches the JSON block, which exists for tooling.

#pragma once

#include <stdint.h>
#include <stddef.h>

#include "sml-kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SML_NAME_LEN 64

// Highest format version this engine can parse. v6 added a per-group MIN
// (asymmetric grid), which grows the tensor record -- reading a newer file with
// an older struct misparses every record, so the loader checks the version and
// says so rather than crashing somewhere unrelated.
//
// v8 adds a per-tensor CODE LAYOUT and WIDTH, because the ladder turned out to
// have a price. Bit-slicing forces reconstruction to be linear in the code so
// that a prefix is a coarser read of the same number; that is what buys
// --depth auto, and it costs 1.552x of end-to-end quality against a flat code
// of the same width at the same bytes. So the choice belongs per tensor: flat
// wherever no runtime depth change is ever made, ladder where it is.
#define SML_MAX_VERSION 9
#define SML_FILE_HEADER_BYTES 4096u

enum { SML_KIND_F32 = 0, SML_KIND_LADDER = 1 };

// Code layout (v8, per tensor).
//   LADDER  level-major bit planes, 2 bits each, prefix-readable. Depth d reads
//           planes 0..d-1 and reconstructs sum(4^l * code_l) * scale + min.
//   FLAT    one plane of nbits-wide codes. No prefix semantics, so every depth
//           argument other than "all" is meaningless for this tensor -- and in
//           exchange the grid is chosen for the width actually read.
#define SML_MODEL_HADAMARD 1u
// v9: per-group scales and mins are 6-bit packed, four groups to three
// bytes. Set per FILE rather than per tensor -- a mixed file would put a
// branch in the epilogue for no measurable gain.
#define SML_MODEL_SCALE6   2u
// with SCALE6: every min is <= 0 and its field is read as (field - 63) * md --
// all 64 levels on the side a group's min is on, as Q4_K's subtracted min does,
// instead of half of them spent on positive mins that essentially never occur
#define SML_MODEL_MINNEG   4u
#define SML_MODEL_EXACT    8u  // raw f32 weight matrices; CPU dense matvec only
#define SML_LAYOUT_LADDER 0u
#define SML_LAYOUT_FLAT   1u

// arch_bits
#define SML_ARCH_PARALLEL_RESIDUAL (1u << 0)  // attention and FFN share an input
#define SML_ARCH_GELU_FFN          (1u << 1)  // no gate projection
#define SML_ARCH_ALIBI             (1u << 2)  // linear distance bias, no rotary
#define SML_ARCH_GELU_ERF          (1u << 3)  // GELU is the exact erf form (HF "gelu"),
                                              // not the tanh approximation
#define SML_ARCH_GELU_GATE         (1u << 4)  // gated FFN with GELU, not SiLU (Gemma's GeGLU)
#define SML_ARCH_QKNORM_LATE       (1u << 5)  // Q/K norm AFTER the rotary (HunYuan), not before
#define SML_ARCH_RELU2             (1u << 6)  // ungated FFN activated by relu(x)^2, not GELU
#define SML_ARCH_ALIBI_SCALED      (1u << 7)  // ALiBi added BEFORE the 1/sqrt(d) scale (Falcon)
#define SML_ARCH_RELU              (1u << 8)  // ungated FFN activated by plain relu(x) (OPT)
#define SML_ARCH_SWIGLU_OAI        (1u << 10) // gpt-oss experts: clamp(gate, 7) * sigmoid(1.702 gate)
                                              // * (clamp(up, -7, 7) + 1)
#define SML_ARCH_MLA               (1u << 11) // multi-head latent attention (DeepSeek V2/V3, Kimi K2)
#define SML_ARCH_QKNORM_HEADS      (1u << 9)  // Q/K norm statistics per head, each head its own
                                              // weight and bias row (Chameleon's LayerNorm)

// Mirrors REC_FMT in tooling/sml_convert.py. Packed, little-endian, and the two
// definitions must move together.
#pragma pack(push, 1)
typedef struct {
    char     name[SML_NAME_LEN];
    uint32_t kind, rows, cols, group;
    uint64_t off, nbytes;          // f32 payload
    uint64_t sd_off, sd_n;         // fp16 per scale-superblock
    uint64_t sq_off, sq_n;         // uint8 per group
    uint64_t perm_off, perm_n;     // int32 column permutation
    // v6: per-group MIN, stored exactly like the scale. The grid is asymmetric
    // -- w = min + code * scale -- which fits a group's real [lo, hi] instead of
    // a symmetric range set by its largest magnitude.
    uint64_t md_off, md_n;         // fp16 per scale-superblock
    uint64_t mq_off, mq_n;         // uint8 per group, biased by 127
    uint64_t codes_off[SML_MAX_LEVELS];
    uint64_t codes_n[SML_MAX_LEVELS];
    // v8: appended at the END so a v7 record is a byte-exact prefix of this
    // one and migration is a prefix copy plus defaults.
    uint32_t layout;      // SML_LAYOUT_LADDER | SML_LAYOUT_FLAT
    uint32_t nbits;       // FLAT: code width. LADDER: 0, header level_bits rule
} sml_rec;

typedef struct {
    char     magic[4];
    uint32_t version;
    uint32_t n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab, n_ctx_train;
    float    rope_theta, rms_eps;
    uint32_t group, n_levels;
    uint32_t level_bits[SML_MAX_LEVELS];
    uint64_t n_tensors;
    uint64_t table_off, json_off, json_n, tok_off, tok_n;
    // v7: offset of the shared-activation group table, 0 if none declared.
    // Tensors in a group read the SAME activation and are issued as one unit.
    // This is the first field in the format that is guidance rather than data:
    // it tells the runtime how to lead the computation, not what the numbers are.
    uint64_t group_off;

    // Architecture description. Only BEHAVIOURAL differences live here --
    // same tensors, different maths. Structural ones (QKV bias, tied head,
    // Q/K norm, experts) are read from the tensor table instead, which is one
    // source of truth rather than a per-architecture declaration to maintain.
    uint32_t norm_type;      // 0 = RMSNorm, 1 = LayerNorm
    uint32_t rope_type;      // 0 = halved pairs (Llama, Qwen, GPT-NeoX,
                             //     Falcon, Phi), 1 = interleaved (GPT-J only)
    float    attn_scale;     // 0 = default 1/sqrt(head_dim)
    float    logit_softcap;  // 0 = none
    uint32_t n_swa;          // 0 = full attention
    uint32_t arch_bits;      // SML_ARCH_* below
    uint32_t n_rot;          // rotated dims per head; 0 = the whole head
    // v8: model-wide flags. Bit 0 says the quantized weights live in a
    // blockwise Hadamard-32 basis and the activation must be rotated to match.
    // Zero in every earlier file, which reads as "no rotation".
    uint32_t flags;
    // Appended, and zero in every older file, which reads as "none". The
    // header region is a fixed zero-filled 4096 bytes, so no older file changes.
    // Granite scales embeddings, each residual branch and the logits by
    // config constants; ignoring them still runs, as a different model.
    float    emb_scale;      // embeddings times this (0 = 1)
    float    res_scale;      // each residual branch times this (0 = 1)
    float    logit_scale;    // logits divided by this (0 = 1)
    // Mixture of experts: experts per token, and bit 0 = renormalize the
    // chosen experts' weights (softmax over the chosen, as Granite and Mixtral).
    uint32_t n_expert_used;
    uint32_t moe_flags;
    // Appended 2026-09-28, zero in every older file ("none"): Gemma 2 caps
    // attention logits with a tanh; OLMo and MPT clamp Q, K and V.
    float    attn_softcap;   // scores = cap * tanh(scores / cap), 0 = none
    float    clip_qkv;       // |q|, |k|, |v| <= clip after the projection, 0 = none
    // DeepSeek-V3-style routing (GLM-4.5, dots.llm1): moe_flags bit 1 = sigmoid
    // scores instead of softmax; the chosen weights times moe_scale (0 = 1);
    // experts in moe_groups groups of which the best moe_topk_group compete.
    float    moe_scale;
    uint32_t moe_groups, moe_topk_group;
    // Mamba-2 state-space layers (2026-09-28): B and C come in ssm_groups groups
    // of ssm_dstate; the gated output norm runs over ssm_norm_group channels at a
    // time (0 = all of them); the step size is clamped to [dt_lo, dt_hi] (hi 0 = none).
    uint32_t ssm_groups, ssm_dstate, ssm_norm_group;
    float    ssm_dt_lo, ssm_dt_hi;
    // Falcon-Mamba RMS-normalizes dt, B and C with no weight, at this eps (0 = no)
    float    ssm_bcdt_eps;
    // Gated DeltaNet (Qwen3-Next, Qwen 3.5): key heads and their width; the
    // value heads and width are read from the tensors
    uint32_t gdn_nk, gdn_dk;
} sml_hdr;
#pragma pack(pop)

// The v5 record, kept so older files still load. v6 inserted the per-group min
// between the permutation and the codes, which changes the stride -- reading a
// v5 table with the v6 struct misparses every entry and crashes. Converting at
// load costs a few hundred entries of memory and keeps every file ever built
// readable, which is the point of owning the format.
#pragma pack(push, 1)
typedef struct {
    char     name[SML_NAME_LEN];
    uint32_t kind, rows, cols, group;
    uint64_t off, nbytes;
    uint64_t sd_off, sd_n;
    uint64_t sq_off, sq_n;
    uint64_t perm_off, perm_n;
    uint64_t codes_off[SML_MAX_LEVELS];
    uint64_t codes_n[SML_MAX_LEVELS];
} sml_rec_v5;
#pragma pack(pop)

// The v6/v7 record. v8 appended layout and nbits, so this is exactly the v8
// struct minus its last eight bytes; migration copies the prefix and defaults
// the rest to LADDER, which is what every file before v8 in fact contains.
#pragma pack(push, 1)
typedef struct {
    char     name[SML_NAME_LEN];
    uint32_t kind, rows, cols, group;
    uint64_t off, nbytes;
    uint64_t sd_off, sd_n;
    uint64_t sq_off, sq_n;
    uint64_t perm_off, perm_n;
    uint64_t md_off, md_n;
    uint64_t mq_off, mq_n;
    uint64_t codes_off[SML_MAX_LEVELS];
    uint64_t codes_n[SML_MAX_LEVELS];
} sml_rec_v7;
#pragma pack(pop)

#define SML_GROUP_MAX      6
#define SML_GROUP_KIND_ACT  1

#pragma pack(push, 1)
typedef struct {
    uint32_t kind, n;
    int32_t  member[SML_GROUP_MAX];   // tensor indices, -1 unused
} sml_group;
#pragma pack(pop)

typedef struct {
    const uint8_t * base;    // mapped file
    size_t          size;
    void          * handle;  // platform mapping handle
    // Large-page loads own their memory instead of mapping the file, so close
    // has to free it differently. 0 = mapped, 1 = VirtualAlloc'd.
    int             owns_pages;
    const sml_hdr * hdr;
    const sml_rec * recs;
    sml_rec       * owned;   // non-NULL when records were converted
    const sml_group * groups;
    uint32_t          n_groups;
} sml_model;

int  sml_model_open(sml_model * m, const char * path);
void sml_model_close(sml_model * m);

// Linear lookup: the table is a few hundred entries and this runs at load time.
const sml_rec * sml_find(const sml_model * m, const char * name);
const sml_rec * sml_findf(const sml_model * m, const char * fmt, int layer);

static inline const uint8_t * sml_ptr(const sml_model * m, uint64_t off) {
    return m->base + off;
}

// Weight bytes one forward pass moves at `depth` levels -- the number the whole
// architecture is trying to shrink, so the engine reports it directly rather
// than inferring it from tok/s.
uint64_t sml_bytes_at_depth(const sml_model * m, int depth);

// Physical memory currently available, 0 if the platform will not say.
uint64_t sml_free_ram(void);

// Deepest read whose working set fits `budget` bytes (0 = ask the machine).
// A fixed-precision file answers this at conversion time and cannot revisit it;
// the ladder answers it on arrival, so one file fits every machine it meets.
int sml_fit_depth(const sml_model * m, uint64_t budget);

#ifdef __cplusplus
}
#endif
