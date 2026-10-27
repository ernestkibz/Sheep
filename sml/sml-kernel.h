// SML quantized mat-vec kernel.
//
// The layout and the kernel were designed together. Every ladder level is 2 bits
// wide, which buys a property worth more than the flexibility it gives up: ONE
// kernel serves every level, and the level index never appears in the inner loop.
//
// GROUP-PLANAR PACKING
// --------------------
// A pack-superblock is 128 weights = 4 groups of 32, stored in 32 bytes:
//
//     byte[j] holds  c[j] | c[j+32]<<2 | c[j+64]<<4 | c[j+96]<<6      j = 0..31
//
// so bit-plane g IS group g. Masking the 32 bytes with 0x03 yields group 0's
// codes; shifting right by 2 and masking yields group 1, and so on. One 32-byte
// load produces four groups with no shuffles, no gathers and no cross-lane
// movement.
//
// TWO-LEVEL SCALES
// ----------------
// One fp16 per 256-weight scale-superblock plus one uint8 per 32-weight group:
// 16/256 + 8/32 = 0.3125 bits/weight. Cheaper than an fp16 per group (0.50)
// AND finer-grained. Measured: identical error at 0.19 fewer bits/weight.
//
//     scale[g] = d[g / 8] * q8[g]
//
// THE DOT PRODUCT
// ---------------
// Codes are unsigned (0..3) and activations int8, which is exactly the shape
// vpmaddubsw wants. Reconstruction is (code - offset) * scale, so
//
//     sum_i (c_i - off) * sw * xq_i * sx
//   = sw * sx * ( sum_i c_i * xq_i  -  off * sum_i xq_i )
//                 \_______________/    \______________/
//                  vpmaddubsw           computed once per group per token,
//                                       reused by every row of the matrix
//
// The correction depends only on the activation, so an N-row matrix pays for it
// once rather than N times. That is why codes are unsigned.
//
// Overflow: codes <= 3, |xq| <= 127, so each product <= 381 and vpmaddubsw's
// pairwise sum <= 762 -- comfortably inside int16. No saturation.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SML_GROUP        32                     // weights sharing one uint8 scale
#define SML_SCALE_SUPER  256                    // weights sharing one fp16 scale
#define SML_PACK_SUPER   128                    // weights per packed superblock
#define SML_GROUPS_PER_PACK  (SML_PACK_SUPER / SML_GROUP)    // 4 bit-planes
#define SML_GROUPS_PER_SCALE (SML_SCALE_SUPER / SML_GROUP)   // 8

// Number of fp16 super-scales for a row of `n` weights. CEILING, not truncation:
// a tensor whose group count is not a multiple of 8 still needs a scale for its
// partial tail. Qwen2.5-0.5B has n_embd 896 -> 28 groups -> 4 super-scales, and
// truncating to 3 made it generate gibberish.
#define SML_NSCALE(n) (((n) / SML_GROUP + SML_GROUPS_PER_SCALE - 1) / SML_GROUPS_PER_SCALE)
#define SML_PACK_BYTES   (SML_PACK_SUPER * 2 / 8)            // 32 bytes per level

// ---------------------------------------------------------------- 6-bit scales
//
// SSF v9 stores the per-group scale and min as 6-bit integers rather than 8.
// This is the entire size deficit against the reference format, measured rather
// than guessed: our per-group metadata costs 0.625 bits/weight (uint8 scale +
// uint8 min per 32 weights, plus two fp16 per 8 groups) where Q4_K costs 0.500
// with 6-bit fields. 0.125 b/w is 2.4% of the file, and the file was 2.25%
// larger. Re-quantizing the scales to 6 bits costs 0.55% on the scale itself,
// which is 0.31% on a weight already carrying ~7% -- see RESULTS.md.
//
// Four groups pack into three bytes, and the kernels consume groups four at a
// time, so a pack boundary is never crossed mid-epilogue.
#define SML_SCALE6_BYTES(ng)  (((ng) / 4) * 3)

static inline void sml_unpack6x4(const uint8_t * p, int * v) {
    const uint32_t b0 = p[0], b1 = p[1], b2 = p[2];
    v[0] = (int)( b0        & 63u);
    v[1] = (int)(((b0 >> 6) | (b1 << 2)) & 63u);
    v[2] = (int)(((b1 >> 4) | (b2 << 4)) & 63u);
    v[3] = (int)( (b2 >> 2) & 63u);
}

// One group's 6-bit value, for the scalar paths.
static inline int sml_scale6_at(const uint8_t * base, int g) {
    int v[4];
    sml_unpack6x4(base + (size_t)(g >> 2) * 3, v);
    return v[g & 3];
}

// Reading a per-group field is the ONE place the two layouts differ, so both go
// through here and neither kernel grows a second copy of the indexing.
static inline int sml_gfield(const uint8_t * base, int g, int six) {
    return six ? sml_scale6_at(base, g) : (int) base[g];
}

// Set once when a model is opened; see sml-kernel.c for why this is a file
// property rather than a kernel argument.
void sml_kernel_scale6(int on);
void sml_kernel_minneg(int on);
int  sml_kernel_minneg_get(void);
int  sml_kernel_scale6_get(void);

// The per-group MIN is stored biased so it can live in an unsigned field. The
// bias is half the field, so it moves with the field width: 127 at 8 bits, 31
// at 6. Getting this wrong shifts every min by a constant -- which produces
// fluent, confidently wrong output rather than an error.
#define SML_MIN_BIAS8  127.0f
#define SML_MIN_BIAS6   31.0f

// Software prefetch distance in packs, 0 = off. Set from SML_PF_DIST once at
// init so the hot loop reads a plain int rather than calling getenv.
//
// The hint is NTA, not T0, and that is the point: weights are read once per
// token and never reused, while the KV cache and activations ARE reused. A
// normal prefetch pulls a doomed cache line into L2/L3 and evicts something
// that was going to be needed. NTA asks for the line without promoting it.
extern int sml_pf_dist;
void sml_kernel_init(void);
#define SML_LEVEL_BITS   2
#define SML_MAX_LEVELS   4

// Activations quantized once per token and shared by every matrix that consumes
// them. `sums[g]` is the group's activation sum -- the correction term above.
typedef struct {
    int8_t  * q;      // n values
    float   * scale;  // n / SML_GROUP
    int32_t * sums;   // n / SML_GROUP
    int       n;
} sml_act;

// Quantize `x` (n floats) into `a`. n must be a multiple of SML_GROUP.
void sml_quantize_act(sml_act * a, const float * x, int n);

// Permute-gather: y[i] = x[perm[i]]. Applying the weight matrix's column
// permutation to the activation is what makes the permutation exact --
// (W P)(P^T x) = W x. One pass over a few thousand floats, against millions of
// weight bytes.
void sml_gather(float * y, const float * x, const int32_t * perm, int n);

// Blockwise Hadamard-32, out-of-place. The weights are stored in the rotated
// basis, so the activation is rotated to match; H is orthogonal and symmetric,
// so (W H)(H x) = W x exactly. Blocks of 32 match SML_GROUP so the rotation
// never crosses a scale boundary. Five butterfly passes, not a 32x32 matmul.
void sml_hadamard32(float * out, const float * in, int n);

float sml_f16_to_f32(uint16_t h);

// Reconstruct ONE row to floats, in permuted column order. Used for the
// embedding lookup, which reads a single row rather than streaming the tensor --
// the one place in the forward pass where a laddered tensor is not a matvec.
//
//   codes[lv] : the level's code plane for this tensor
//   row       : which row to pull
//   depth     : levels to read
//   out       : n floats, still in permuted order (caller unpermutes)
void sml_dequant_row(float * out, const uint8_t * const * codes, int nlev, int depth,
                     const uint16_t * sd, const uint8_t * sq,
                     const uint16_t * md, const uint8_t * mq, int row, int n);

// One ladder level's contribution:
//
//     y[r] += sum_g scale_g * xscale_g * ( (dot_g << shift) + bias * sums_g )
//
// Accumulates, so calling it again with the next level's codes refines the same
// output vector -- that is what makes a partial read a valid answer rather than
// a broken one.
//
// `shift` is the number of bits still unread BELOW this level, because the
// levels are bit slices of one integer: with levels [2,2,2] the shifts are
// 4, 2, 0. So level 0's codes carry weight 16, level 1's weight 4, level 2's 1.
//
// `bias` carries the (midpoint - offset) constant and must be applied EXACTLY
// ONCE, on the last level actually read -- pass 0.0f for every earlier level.
// Reading k of 3 levels leaves rem = 6 - 2k bits unknown, so
//
//     bias = (rem ? (1 << (rem - 1)) : 0) - offset          offset = 31.5
//
// Folding it here costs one FMA per group instead of a second pass over the
// scales, and putting it in the signature makes the once-only rule impossible
// to get subtly wrong.
//
// codes:  rows * (n/SML_PACK_SUPER) * SML_PACK_BYTES, group-planar
// sd:     rows * SML_NSCALE(n) fp16
// sq:     rows * (n/SML_GROUP) uint8
// `md`/`mq` carry the per-group MIN of an asymmetric grid (v6). Pass NULL for a
// symmetric file, where the reconstruction is (code - offset) * scale and the
// constant folds into `bias`. With a min, w = min + code * scale, so the
// per-group constant is scale*mid + min and cannot be a single float -- it is
// one extra FMA per group in the epilogue, not a second pass.
void sml_matvec_rows(float * y, const uint8_t * codes, const uint16_t * sd,
                     const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                     const sml_act * a,
                     int row0, int row1, int n, int shift, float bias);

// Dequantize one row of a FLAT tensor. The embedding lookup needs whole rows
// rather than a dot product, and it is the ONE path that reads token_embd
// directly -- so a flat file whose matvec is perfect still produces garbage
// without this, because every token's input vector is misread.
void sml_dequant_row_flat6(float * out, const uint8_t * lo4, const uint8_t * hi2,
                           const uint16_t * sd, const uint8_t * sq,
                           const uint16_t * md, const uint8_t * mq,
                           int row, int n);

// The 1-bit plane of Q3/Q5/Q7: 4 bytes per group per row, bit j = weight j.
// Adds step * 2^shift * sum(bit * x) over a batch of nb slots (y[b*nrows + r]);
// the dequant variant adds the plane's share to a row already dequantized.
void sml_matvec_bits1(float * y, const uint8_t * plane, const uint16_t * sd, const uint8_t * sq,
                      const sml_act * a, int nb, int nrows, int row0, int row1, int n, int shift);
void sml_dequant_bits1_add(float * out, const uint8_t * plane, const uint16_t * sd, const uint8_t * sq,
                           int row, int n, int shift);

// Q8: an 8-bit FLAT tensor is a low and a high nibble plane in the 4-bit layout
// (codes_off[0], codes_off[1]); code = lo + 16*hi. Rows, batches and whole-row
// dequantization, with the same scales and mins as 4-bit.
void sml_matvec_rows_flat8(float * y, const uint8_t * lo, const uint8_t * hi, const uint16_t * sd,
                           const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                           const sml_act * a, int row0, int row1, int n);
void sml_matvec_batch_flat8(float * y, const uint8_t * lo, const uint8_t * hi, const uint16_t * sd,
                            const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                            const sml_act * a, int nb, int nrows, int row0, int row1, int n);
void sml_dequant_row_flat8(float * out, const uint8_t * lo, const uint8_t * hi,
                           const uint16_t * sd, const uint8_t * sq,
                           const uint16_t * md, const uint8_t * mq,
                           int row, int n);

void sml_dequant_row_flat4(float * out, const uint8_t * codes,
                           const uint16_t * sd, const uint8_t * sq,
                           const uint16_t * md, const uint8_t * mq,
                           int row, int n);

// FLAT 4-bit rows (SSF v8, SML_LAYOUT_FLAT with nbits 4). One plane, one pass,
// one epilogue -- no `shift` or `bias` because there is no level to weight and
// no truncation to re-centre. Codes pack two groups per 32-byte load: low
// nibbles are one group, high nibbles the next.
void sml_matvec_rows_flat4(float * y, const uint8_t * codes, const uint16_t * sd,
                           const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                           const sml_act * a,
                           int row0, int row1, int n);

// FLAT 6-bit rows in one pass: lo4 is the nibble plane (flat layout), hi2 the
// 2-bit plane (group-planar packs); code = nibble | extra << 4.
void sml_matvec_rows_flat6(float * y, const uint8_t * lo4, const uint8_t * hi2,
                           const uint16_t * sd, const uint8_t * sq,
                           const uint16_t * md, const uint8_t * mq,
                           const sml_act * a, int row0, int row1, int n);

// FLAT 6-bit batched on per-256 activations: rows-in-lanes on merged 6-bit
// codes where the batch allows, the two-pass form otherwise.
void sml_matvec_batch_flat6_sb(float * y, const uint8_t * lo4, const uint8_t * hi2,
                               const uint16_t * sd, const uint8_t * sq,
                               const uint16_t * md, const uint8_t * mq,
                               const sml_act * a, int nb, int nrows, int row0, int row1, int n);

// Rows-in-lanes packed once (several conversations decoding together; see
// sml-kernel.c): per 8-row tile and superblock, nibble codes in rl's vector
// order plus the fields. Tensors need rows % 8 == 0 and whole superblocks.
int    sml_rlp_ok(int rows, int n, int nbits, int has_min);
size_t sml_rlp_bytes(int rows, int n, int nbits);
void   sml_rlp_pack(uint8_t * dst, const uint8_t * lo4, const uint8_t * hi2,
                    const uint16_t * sd, const uint8_t * sq,
                    const uint16_t * md, const uint8_t * mq,
                    int nbits, int tile0, int tile1, int n);
// y[slot][row] += over rows row0..row1 (multiples of 8); per-256 activations
void   sml_rlp_matvec(float * y, int nrows, const uint8_t * rlp, int nbits,
                      const sml_act * a, int nb, int row0, int row1, int n);

// Interleaved decode layout (see sml-kernel.c): one stream per row, a block per
// 256 weights carrying its codes and all four scale fields. sml_il_ok says
// whether a tensor's shape and fields can be repacked; decode keeps the file's
// arrays for any tensor it cannot.
// Block sizes: 148 / 212 bytes, or 144 / 208 for a v9 file's 6-bit fields.
int    sml_il_ok(int n, int nbits, int has_min);
size_t sml_il_row_bytes(int n, int nbits);
void   sml_il_pack(uint8_t * dst, const uint8_t * lo4, const uint8_t * hi2,
                   const uint16_t * sd, const uint8_t * sq,
                   const uint16_t * md, const uint8_t * mq,
                   int nbits, int row0, int row1, int n);
void   sml_matvec_rows_il4(float * y, const uint8_t * il, const sml_act * a,
                           int row0, int row1, int n);
void   sml_matvec_rows_il6(float * y, const uint8_t * il, const sml_act * a,
                           int row0, int row1, int n);

// Most activation vectors one batched call accepts: the vector arms keep one
// accumulator per batch element on the stack.
#define SML_KERNEL_MAX_BATCH 128

// BATCHED: one weight read, `nb` activation vectors.
//
// This is the whole reason speculative decoding pays on a memory-bound machine.
// A batch-1 decode reads 1.66 GB to produce one token; verifying 4 candidate
// tokens reads the SAME 1.66 GB and produces four. The arithmetic quadruples,
// which is free here -- ~80% of this machine's FLOPs sit idle waiting on the
// bus.
//
// y:  nb * rows, row-major per batch element
// a:  nb activations, each already quantized against this tensor's permutation
// y is [nb][nrows]; row0..row1 is this caller's slice of the rows.
void sml_matvec_batch(float * y, const uint8_t * codes, const uint16_t * sd,
                      const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                      const sml_act * a, int nb,
                      int nrows, int row0, int row1, int n, int shift, float bias);

// BATCHED flat 4-bit. Same contract as sml_matvec_batch, minus `shift`/`bias`:
// a flat plane has no level to weight and no truncation to re-centre.
void sml_matvec_batch_flat4(float * y, const uint8_t * codes, const uint16_t * sd,
                            const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                            const sml_act * a, int nb,
                            int nrows, int row0, int row1, int n);

// Prompt reading: one activation scale per 256 values (the weight superblock),
// written into every group so the per-group kernels stay exact on it. The _sb
// kernel reads it once per superblock and REQUIRES sml_quantize_act_sb input.
// Decode keeps sml_quantize_act.
void sml_quantize_act_sb(sml_act * a, const float * x, int n);
void sml_matvec_batch_flat4_sb(float * y, const uint8_t * codes, const uint16_t * sd,
                               const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                               const sml_act * a, int nb,
                               int nrows, int row0, int row1, int n);

static inline void sml_matvec(float * y, const uint8_t * codes, const uint16_t * sd,
                              const uint8_t * sq, const uint16_t * md,
                              const uint8_t * mq, const sml_act * a,
                              int rows, int n, int shift, float bias) {
    sml_matvec_rows(y, codes, sd, sq, md, mq, a, 0, rows, n, shift, bias);
}

// The asymmetric grid centres the code at 0, not at the midpoint, so only the
// unread-bit midpoint survives -- the -offset term belongs to the symmetric
// reconstruction and must not be applied twice.
static inline float sml_level_mid(int depth, int nlev) {
    const int rem = nlev * SML_LEVEL_BITS - depth * SML_LEVEL_BITS;
    return rem ? (float) (1 << (rem - 1)) : 0.0f;
}

// bias for reading `depth` of `nlev` two-bit levels
static inline float sml_level_bias(int depth, int nlev) {
    const int total = nlev * SML_LEVEL_BITS;
    const int rem   = total - depth * SML_LEVEL_BITS;
    const float offset = (float) ((1 << total) - 1) * 0.5f;
    return (rem ? (float) (1 << (rem - 1)) : 0.0f) - offset;
}

// Bytes a level's read moves, for the traffic counter. We optimize bytes, so we
// measure bytes -- a tok/s number alone cannot tell a real win from a lucky one.
static inline size_t sml_level_bytes(int rows, int n) {
    return (size_t) rows * (size_t) n / SML_PACK_SUPER * SML_PACK_BYTES;
}
static inline size_t sml_scale_bytes_x(int rows, int n, int six) {
    const size_t ng = (size_t) n / SML_GROUP;
    return (size_t) rows * ((size_t) SML_NSCALE(n) * sizeof(uint16_t)
                            + (six ? SML_SCALE6_BYTES(ng) : ng));
}
static inline size_t sml_scale_bytes(int rows, int n) {
    return sml_scale_bytes_x(rows, n, 0);
}

#ifdef __cplusplus
}
#endif
