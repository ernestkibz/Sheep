#include "sml-kernel.h"

#include <stdlib.h>

#include <math.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// fp16 -> fp32. F16C is present on Zen 3; the scalar path is for correctness
// checks and non-AVX2 builds.
static inline float sml_f16(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t s = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) { bits = s; }
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3ff;
            bits = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        bits = s | 0x7f800000u | (m << 13);
    } else {
        bits = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
#endif
}

void sml_quantize_act(sml_act * a, const float * x, int n) {
    const int ng = n / SML_GROUP;
    a->n = n;
#if defined(__AVX2__)
    // Vector form of the loop below, bit for bit: the same amax, the same
    // s and 1/s computed in scalar, cvtps rounds half-to-even as lrintf does
    // under the default MXCSR, and the clamp is the same +-127. Scalar, this was
    // ~1.1 ms of a 32 ms decode token on the 1B (lrintf per element).
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    for (int g = 0; g < ng; g++) {
        const float * xs = x + g * SML_GROUP;
        const __m256 v0 = _mm256_loadu_ps(xs),      v1 = _mm256_loadu_ps(xs + 8);
        const __m256 v2 = _mm256_loadu_ps(xs + 16), v3 = _mm256_loadu_ps(xs + 24);
        __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                 _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);
        const float s  = amax / 127.0f;
        const float is = s > 0.0f ? 1.0f / s : 0.0f;
        const __m256 iv = _mm256_set1_ps(is);
        __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, iv));
        __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, iv));
        __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, iv));
        __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, iv));
        q0 = _mm256_min_epi32(_mm256_max_epi32(q0, lo), hi);
        q1 = _mm256_min_epi32(_mm256_max_epi32(q1, lo), hi);
        q2 = _mm256_min_epi32(_mm256_max_epi32(q2, lo), hi);
        q3 = _mm256_min_epi32(_mm256_max_epi32(q3, lo), hi);
        const __m256i sm = _mm256_add_epi32(_mm256_add_epi32(q0, q1), _mm256_add_epi32(q2, q3));
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sm), _mm256_extracti128_si256(sm, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        // packs interleave 128-bit lanes; the permute puts the 32 bytes back in order
        const __m256i p01 = _mm256_packs_epi32(q0, q1), p23 = _mm256_packs_epi32(q2, q3);
        const __m256i p = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p01, p23),
                                                      _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i *)(a->q + g * SML_GROUP), p);
        a->scale[g] = s;
        a->sums[g]  = _mm_cvtsi128_si32(s4);
    }
    return;
#endif
    for (int g = 0; g < ng; g++) {
        const float * xs = x + g * SML_GROUP;
        float amax = 0.0f;
        for (int i = 0; i < SML_GROUP; i++) {
            const float v = fabsf(xs[i]);
            if (v > amax) amax = v;
        }
        const float s  = amax / 127.0f;
        const float is = s > 0.0f ? 1.0f / s : 0.0f;
        int32_t sum = 0;
        int8_t * q = a->q + g * SML_GROUP;
        for (int i = 0; i < SML_GROUP; i++) {
            int v = (int) lrintf(xs[i] * is);
            if (v >  127) v =  127;
            if (v < -127) v = -127;
            q[i] = (int8_t) v;
            sum += v;
        }
        a->scale[g] = s;
        a->sums[g]  = sum;
    }
}

// Prompt reading only (Ernest's call, 2026-09-18): ONE activation scale per
// 256 values -- the weight superblock -- instead of one per 32, so the kernel
// below can sum a superblock's groups in integers. Graded when it was proposed:
// +1.5-2.4 standard errors of logit error on Qwen2.5-0.5B, inside the noise on
// Llama-3.2-1B, with the engine 12-37% more accurate than the reference overall.
// Decode keeps sml_quantize_act. The scale is written into every group of its
// superblock, so the per-group kernels (6-bit tensors, ladder) stay exact on it.
void sml_quantize_act_sb(sml_act * a, const float * x, int n) {
    const int ng = n / SML_GROUP;
    a->n = n;
#if defined(__AVX2__)
    // Vector form, bit for bit, as in sml_quantize_act: the same amax over the
    // superblock, s and 1/s in scalar, cvtps rounding as lrintf, the same clamp.
    // Scalar, prompt reading called lrintf ~22M times per 128-token pythia prompt.
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int g0 = 0; g0 < ng; g0 += SML_GROUPS_PER_SCALE) {
        const int g1 = g0 + SML_GROUPS_PER_SCALE < ng ? g0 + SML_GROUPS_PER_SCALE : ng;
        __m256 m = _mm256_setzero_ps();
        for (int i = g0 * SML_GROUP; i < g1 * SML_GROUP; i += 8)
            m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + i)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);
        const float s  = amax / 127.0f;
        const float is = s > 0.0f ? 1.0f / s : 0.0f;
        const __m256 iv = _mm256_set1_ps(is);
        for (int g = g0; g < g1; g++) {
            const float * xs = x + g * SML_GROUP;
            __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs), iv));
            __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + 8), iv));
            __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + 16), iv));
            __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + 24), iv));
            q0 = _mm256_min_epi32(_mm256_max_epi32(q0, lo), hi);
            q1 = _mm256_min_epi32(_mm256_max_epi32(q1, lo), hi);
            q2 = _mm256_min_epi32(_mm256_max_epi32(q2, lo), hi);
            q3 = _mm256_min_epi32(_mm256_max_epi32(q3, lo), hi);
            const __m256i sm = _mm256_add_epi32(_mm256_add_epi32(q0, q1), _mm256_add_epi32(q2, q3));
            __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sm), _mm256_extracti128_si256(sm, 1));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
            const __m256i p = _mm256_permutevar8x32_epi32(
                _mm256_packs_epi16(_mm256_packs_epi32(q0, q1), _mm256_packs_epi32(q2, q3)), order);
            _mm256_storeu_si256((__m256i *)(a->q + g * SML_GROUP), p);
            a->scale[g] = s;
            a->sums[g]  = _mm_cvtsi128_si32(s4);
        }
    }
    return;
#endif
    for (int g0 = 0; g0 < ng; g0 += SML_GROUPS_PER_SCALE) {
        const int g1 = g0 + SML_GROUPS_PER_SCALE < ng ? g0 + SML_GROUPS_PER_SCALE : ng;
        float amax = 0.0f;
        for (int i = g0 * SML_GROUP; i < g1 * SML_GROUP; i++) {
            const float v = fabsf(x[i]);
            if (v > amax) amax = v;
        }
        const float s  = amax / 127.0f;
        const float is = s > 0.0f ? 1.0f / s : 0.0f;
        for (int g = g0; g < g1; g++) {
            const float * xs = x + g * SML_GROUP;
            int8_t * q = a->q + g * SML_GROUP;
            int32_t sum = 0;
            for (int i = 0; i < SML_GROUP; i++) {
                int v = (int) lrintf(xs[i] * is);
                if (v >  127) v =  127;
                if (v < -127) v = -127;
                q[i] = (int8_t) v;
                sum += v;
            }
            a->scale[g] = s;
            a->sums[g]  = sum;
        }
    }
}

int sml_pf_dist = 0;

// SML_LANES4=1 keeps the four-lane flat-4 kernels; see matvec_flat4_l8.
static int g_lanes4 = 0;

// SSF v9 stores per-group scales and mins as 6-bit fields. It is a property of
// the FILE, set once when a model is opened, because threading it through seven
// kernel signatures and fifteen call sites is the larger risk in the one place
// where a mistake produces confident wrong numbers rather than a crash.
//
// CONSTRAINT: a process that opens two models (speculative decoding opens a
// draft and a target) requires both to agree. sml_model_open enforces that.
static int g_scale6 = 0;
static int g_minneg = 0;   // SML_MODEL_MINNEG: 6-bit mins are negative-only

void sml_kernel_scale6(int on) { g_scale6 = on; }
void sml_kernel_minneg(int on) { g_minneg = on; }
int  sml_kernel_minneg_get(void) { return g_minneg; }

static inline float sml_minbias(void) {
    return g_scale6 ? (g_minneg ? 63.0f : SML_MIN_BIAS6) : SML_MIN_BIAS8;
}
int  sml_kernel_scale6_get(void) { return g_scale6; }

// Row stride of a per-group byte array: ng bytes at 8 bits, three per four
// groups at 6.
static inline size_t sml_grow(int ng) {
    return g_scale6 ? (size_t) SML_SCALE6_BYTES(ng) : (size_t) ng;
}

void sml_kernel_init(void) {
    const char * e = getenv("SML_PF_DIST");
    sml_pf_dist = e ? atoi(e) : 0;
    if (sml_pf_dist < 0) sml_pf_dist = 0;
    if (sml_pf_dist > 4096) sml_pf_dist = 4096;
    g_lanes4 = getenv("SML_LANES4") != NULL;
}

void sml_gather(float * y, const float * x, const int32_t * perm, int n) {
    for (int i = 0; i < n; i++) y[i] = x[perm[i]];
}

float sml_f16_to_f32(uint16_t h) { return sml_f16(h); }

void sml_dequant_row(float * out, const uint8_t * const * codes, int nlev, int depth,
                     const uint16_t * sd, const uint8_t * sq,
                     const uint16_t * md, const uint8_t * mq, int row, int n) {
    const int npack  = n / SML_PACK_SUPER;
    const int ng     = n / SML_GROUP;
    const int nscale = SML_NSCALE(n);
    const int total  = nlev * SML_LEVEL_BITS;
    const int rem    = total - depth * SML_LEVEL_BITS;
    const float offset = (float)((1 << total) - 1) * 0.5f;
    const float mid    = rem ? (float)(1 << (rem - 1)) : 0.0f;

    for (int col = 0; col < n; col++) {
        const int p = col / SML_PACK_SUPER;              // which packed block
        const int g = (col % SML_PACK_SUPER) / SML_GROUP; // which bit-plane == group
        const int j = col % SML_GROUP;                    // byte within the block
        int code = 0;
        for (int lv = 0; lv < depth; lv++) {
            const uint8_t byte = codes[lv][(size_t) row * npack * SML_PACK_BYTES
                                           + (size_t) p * SML_PACK_BYTES + j];
            code |= (int)((byte >> (2 * g)) & 0x3) << ((nlev - 1 - lv) * SML_LEVEL_BITS);
        }
        const int gi = col / SML_GROUP;
        const float scale = sml_f16(sd[(size_t) row * nscale + gi / SML_GROUPS_PER_SCALE])
                          * (float) sml_gfield(sq + (size_t) row * sml_grow(ng), gi, g_scale6);
        if (mq) {
            const float mn = sml_f16(md[(size_t) row * nscale + gi / SML_GROUPS_PER_SCALE])
                           * ((float) sml_gfield(mq + (size_t) row * sml_grow(ng), gi, g_scale6) - sml_minbias());
            out[col] = ((float) code + mid) * scale + mn;
        } else {
            out[col] = ((float) code + mid - offset) * scale;
        }
    }
}

#if defined(__AVX2__)

// Four consecutive groups' per-group field as floats. At 8 bits this is one
// 4-byte load and a widen; at 6 bits it is three bytes and a shift-unpack. The
// branch is on a value that does not change for the life of a call, so it
// predicts perfectly and the 6-bit path costs a handful of ALU ops per 128
// weights.
static inline __m128 sml_qf4(const uint8_t * base, int gbase) {
    if (g_scale6) {
        int v[4];
        sml_unpack6x4(base + (size_t)(gbase >> 2) * 3, v);
        return _mm_cvtepi32_ps(_mm_setr_epi32(v[0], v[1], v[2], v[3]));
    }
    return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(
        _mm_cvtsi32_si128(*(const int *) (base + gbase))));
}

// Reduce four groups' int32 lanes to one vector of four sums.
//
// Doing this per group instead of per pack cost HALF the kernel's bandwidth
// when the group size dropped from 64 to 32: at group 32 a group is a single
// 256-bit vector, so a full horizontal sum (extract + three shuffle-adds) ran
// for every ~4 cycles of useful multiply. Measured 23.1 -> 11.5 GB/s.
//
// Two hadds fold 4x8 int32 down to lane-wise partial sums, and one cross-lane
// add finishes it: ~5 instructions for four groups instead of ~16.
static inline __m128i sml_hsum4_i32(__m256i a, __m256i b, __m256i c, __m256i d) {
    const __m256i ab = _mm256_hadd_epi32(a, b);   // [a01 a23 b01 b23 | a45 a67 b45 b67]
    const __m256i cd = _mm256_hadd_epi32(c, d);
    const __m256i t  = _mm256_hadd_epi32(ab, cd); // [sumA sumB sumC sumD | upper half]
    return _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
}

void sml_matvec_rows(float * y, const uint8_t * codes, const uint16_t * sd,
                     const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                     const sml_act * a,
                     int row0, int row1, int n, int shift, float bias) {
    const int npack  = n / SML_PACK_SUPER;
    const int ng     = n / SML_GROUP;
    const int nscale = SML_NSCALE(n);
    const float mul  = (float)(1 << shift);
    const __m256i m3  = _mm256_set1_epi8(0x03);
    const __m256i one = _mm256_set1_epi16(1);

    for (int r = row0; r < row1; r++) {
        const uint8_t  * cp = codes + (size_t) r * npack * SML_PACK_BYTES;
        const uint16_t * dp = sd    + (size_t) r * nscale;
        const uint8_t  * qp = sq    + (size_t) r * sml_grow(ng);
        const uint16_t * mdp = md ? md + (size_t) r * nscale : NULL;
        const uint8_t  * mqp = mq ? mq + (size_t) r * sml_grow(ng) : NULL;
        __m128 vsum = _mm_setzero_ps();

        const int pf = sml_pf_dist;
        for (int p = 0; p < npack; p++) {
            if (pf && p + pf < npack)
                _mm_prefetch((const char *)(cp + (size_t) pf * SML_PACK_BYTES),
                             _MM_HINT_NTA);
            const __m256i b = _mm256_loadu_si256((const __m256i *)(cp));
            cp += SML_PACK_BYTES;
            const int gbase = p * SML_GROUPS_PER_PACK;
            const int8_t * xq = a->q + (size_t) gbase * SML_GROUP;

            // bit-plane g IS group g: shift and mask, no shuffles, no gathers.
            // codes unsigned, activations signed -- vpmaddubsw's operand order.
            __m256i acc[SML_GROUPS_PER_PACK];
            for (int g = 0; g < SML_GROUPS_PER_PACK; g++) {
                const __m256i c = _mm256_and_si256(_mm256_srli_epi16(b, 2 * g), m3);
                const __m256i x = _mm256_loadu_si256(
                    (const __m256i *)(xq + (size_t) g * SML_GROUP));
                acc[g] = _mm256_madd_epi16(_mm256_maddubs_epi16(c, x), one);
            }
            const __m128i d4 = sml_hsum4_i32(acc[0], acc[1], acc[2], acc[3]);

            // Vectorised epilogue. At group 32 there are twice as many groups as
            // at group 64, so a scalar epilogue runs twice as often -- that was
            // the rest of the bandwidth lost in the switch. All four groups of a
            // pack fall inside one scale-superblock (a pack spans 4 groups, a
            // scale-superblock 8, and they are aligned), so the fp16 super-scale
            // is loaded once per pack rather than once per group.
            const float dsc = sml_f16(dp[gbase / SML_GROUPS_PER_SCALE]);
            const __m128 qf = sml_qf4(qp, gbase);
            const __m128 af = _mm_loadu_ps(a->scale + gbase);
            const __m128 sf = _mm_cvtepi32_ps(_mm_loadu_si128(
                (const __m128i *)(a->sums + gbase)));
            const __m128 df = _mm_cvtepi32_ps(d4);

            const __m128 val = _mm_add_ps(_mm_mul_ps(df, _mm_set1_ps(mul)),
                                          _mm_mul_ps(sf, _mm_set1_ps(bias)));
            vsum = _mm_add_ps(vsum, _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(dsc), qf),
                                               _mm_mul_ps(af, val)));
            if (mqp) {
                // asymmetric: + min_g * sum(x)_g, scaled by the activation scale
                const float mdsc = sml_f16(mdp[gbase / SML_GROUPS_PER_SCALE]);
                const __m128 mqf = _mm_sub_ps(sml_qf4(mqp, gbase),
                    _mm_set1_ps(sml_minbias()));
                vsum = _mm_add_ps(vsum,
                    _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(mdsc), mqf),
                               _mm_mul_ps(af, sf)));
            }
        }
        // one horizontal sum per row, not per group
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        y[r] += _mm_cvtss_f32(vsum);
    }
}


// FLAT 4-bit (SSF v8). One plane, one pass, one epilogue.
//
// The ladder reads depth 2 as TWO passes over two level arrays, each with its
// own epilogue, because reconstruction has to stay linear in the code for a
// prefix to be a coarser read. A tensor that is never read at more than one
// depth pays for that property and never uses it -- 1.552x of end-to-end
// quality, measured.
//
// Packing: one 32-byte load carries two whole groups. Byte i's LOW nibble is
// group 2p column i, its HIGH nibble is group 2p+1 column i. Two loads
// therefore feed four groups, which keeps the vectorised epilogue at the same
// four-group stride the ladder uses -- the epilogue was 2x more frequent at
// group 32 than at group 64 and that cost real bandwidth once already.
static void matvec_rows_flat4_l4(float * y, const uint8_t * codes, const uint16_t * sd,
                           const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                           const sml_act * a,
                           int row0, int row1, int n) {
    const int ng     = n / SML_GROUP;
    const int nquad  = ng / 4;                 // four groups per iteration
    const int nscale = SML_NSCALE(n);
    const size_t rowbytes = (size_t) n / 2;    // 4 bits per weight
    const __m256i m15 = _mm256_set1_epi8(0x0F);
    const __m256i one = _mm256_set1_epi16(1);

    for (int r = row0; r < row1; r++) {
        const uint8_t  * cp  = codes + (size_t) r * rowbytes;
        const uint16_t * dp  = sd    + (size_t) r * nscale;
        const uint8_t  * qp  = sq    + (size_t) r * sml_grow(ng);
        const uint16_t * mdp = md ? md + (size_t) r * nscale : NULL;
        const uint8_t  * mqp = mq ? mq + (size_t) r * sml_grow(ng) : NULL;
        __m128 vsum = _mm_setzero_ps();

        const int pf = sml_pf_dist;
        for (int p = 0; p < nquad; p++) {
            if (pf && p + pf < nquad)
                _mm_prefetch((const char *)(cp + (size_t) pf * 64), _MM_HINT_NTA);
            const __m256i b0 = _mm256_loadu_si256((const __m256i *)(cp));
            const __m256i b1 = _mm256_loadu_si256((const __m256i *)(cp + 32));
            cp += 64;
            const int gbase = p * 4;
            const int8_t * xq = a->q + (size_t) gbase * SML_GROUP;

            __m256i acc[4];
            const __m256i c0 = _mm256_and_si256(b0, m15);
            const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(b0, 4), m15);
            const __m256i c2 = _mm256_and_si256(b1, m15);
            const __m256i c3 = _mm256_and_si256(_mm256_srli_epi16(b1, 4), m15);
            acc[0] = _mm256_madd_epi16(_mm256_maddubs_epi16(c0,
                        _mm256_loadu_si256((const __m256i *)(xq))), one);
            acc[1] = _mm256_madd_epi16(_mm256_maddubs_epi16(c1,
                        _mm256_loadu_si256((const __m256i *)(xq + SML_GROUP))), one);
            acc[2] = _mm256_madd_epi16(_mm256_maddubs_epi16(c2,
                        _mm256_loadu_si256((const __m256i *)(xq + 2 * SML_GROUP))), one);
            acc[3] = _mm256_madd_epi16(_mm256_maddubs_epi16(c3,
                        _mm256_loadu_si256((const __m256i *)(xq + 3 * SML_GROUP))), one);
            const __m128i d4 = sml_hsum4_i32(acc[0], acc[1], acc[2], acc[3]);

            // Same epilogue as the ladder, run ONCE instead of once per level.
            // No `mul`: there is a single plane, so the code is the value.
            const float dsc = sml_f16(dp[gbase / SML_GROUPS_PER_SCALE]);
            const __m128 qf = sml_qf4(qp, gbase);
            const __m128 af = _mm_loadu_ps(a->scale + gbase);
            const __m128 sf = _mm_cvtepi32_ps(_mm_loadu_si128(
                (const __m128i *)(a->sums + gbase)));
            const __m128 df = _mm_cvtepi32_ps(d4);
            vsum = _mm_add_ps(vsum, _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(dsc), qf),
                                               _mm_mul_ps(af, df)));
            if (mqp) {
                const float mdsc = sml_f16(mdp[gbase / SML_GROUPS_PER_SCALE]);
                const __m128 mqf = _mm_sub_ps(sml_qf4(mqp, gbase),
                    _mm_set1_ps(sml_minbias()));
                vsum = _mm_add_ps(vsum,
                    _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(mdsc), mqf),
                               _mm_mul_ps(af, sf)));
            }
        }
        vsum = _mm_hadd_ps(vsum, vsum);
        vsum = _mm_hadd_ps(vsum, vsum);
        y[r] += _mm_cvtss_f32(vsum);
    }
}


// BATCHED flat 4-bit: one weight read, `nb` activation vectors.
//
// The row kernel above had no batched twin, so `matvec_batch` fell through to
// the LADDER kernel and indexed codes_off[lv] for lv up to depth-1. A flat
// tensor populates only codes_off[0] (and [1] at 6 bits), so verification read
// an unmapped plane and segfaulted -- speculative decoding was dead on every
// v8 model, which is every model carrying the flat win. Third call site of the
// same bug: a flat kernel written for one path while another still assumed the
// ladder.
//
// The unpack (two loads, four masks) happens ONCE per quad and serves every
// batch element. Only the activation loads and the madds repeat, and those are
// arithmetic on a machine whose FLOPs are idle waiting on the bus.
// Tiling this by 4 to cut live accumulators was tried and REVERTED: the
// nb=4 -> nb=5 cost cliff (1.62x -> 2.11x per round) survived it unchanged
// across three runs, so the cliff is not register spill in this kernel.
// Per-token cost floors at ~0.030 s for nb>=4 either way, which is 3x off
// this tensor's own DRAM bound -- the batch path is limited by the
// per-group horizontal sum and float epilogue, not by the weight read.
static void matvec_batch_flat4_l4(float * y, const uint8_t * codes, const uint16_t * sd,
                            const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                            const sml_act * a, int nb,
                            int nrows, int row0, int row1, int n) {
    const int ng     = n / SML_GROUP;
    const int nquad  = ng / 4;
    const int nscale = SML_NSCALE(n);
    const size_t rowbytes = (size_t) n / 2;
    const __m256i m15 = _mm256_set1_epi8(0x0F);
    const __m256i one = _mm256_set1_epi16(1);

    for (int r = row0; r < row1; r++) {
        const uint8_t  * cp  = codes + (size_t) r * rowbytes;
        const uint16_t * dp  = sd    + (size_t) r * nscale;
        const uint8_t  * qp  = sq    + (size_t) r * sml_grow(ng);
        const uint16_t * mdp = md ? md + (size_t) r * nscale : NULL;
        const uint8_t  * mqp = mq ? mq + (size_t) r * sml_grow(ng) : NULL;
        __m128 vsum[SML_KERNEL_MAX_BATCH];
        for (int b = 0; b < nb; b++) vsum[b] = _mm_setzero_ps();

        const int pf = sml_pf_dist;
        for (int p = 0; p < nquad; p++) {
            if (pf && p + pf < nquad)
                _mm_prefetch((const char *)(cp + (size_t) pf * 64), _MM_HINT_NTA);
            const __m256i b0 = _mm256_loadu_si256((const __m256i *)(cp));
            const __m256i b1 = _mm256_loadu_si256((const __m256i *)(cp + 32));
            cp += 64;
            const int gbase = p * 4;

            const __m256i c0 = _mm256_and_si256(b0, m15);
            const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(b0, 4), m15);
            const __m256i c2 = _mm256_and_si256(b1, m15);
            const __m256i c3 = _mm256_and_si256(_mm256_srli_epi16(b1, 4), m15);

            const float dsc = sml_f16(dp[gbase / SML_GROUPS_PER_SCALE]);
            const __m128 qf = sml_qf4(qp, gbase);
            const __m128 wsc = _mm_mul_ps(_mm_set1_ps(dsc), qf);
            __m128 wmin = _mm_setzero_ps();
            if (mqp) {
                const float mdsc = sml_f16(mdp[gbase / SML_GROUPS_PER_SCALE]);
                wmin = _mm_mul_ps(_mm_set1_ps(mdsc),
                    _mm_sub_ps(sml_qf4(mqp, gbase),
                        _mm_set1_ps(sml_minbias())));
            }

            for (int b = 0; b < nb; b++) {
                const sml_act * ab = &a[b];
                const int8_t * xq = ab->q + (size_t) gbase * SML_GROUP;
                __m256i acc[4];
                acc[0] = _mm256_madd_epi16(_mm256_maddubs_epi16(c0,
                            _mm256_loadu_si256((const __m256i *)(xq))), one);
                acc[1] = _mm256_madd_epi16(_mm256_maddubs_epi16(c1,
                            _mm256_loadu_si256((const __m256i *)(xq + SML_GROUP))), one);
                acc[2] = _mm256_madd_epi16(_mm256_maddubs_epi16(c2,
                            _mm256_loadu_si256((const __m256i *)(xq + 2 * SML_GROUP))), one);
                acc[3] = _mm256_madd_epi16(_mm256_maddubs_epi16(c3,
                            _mm256_loadu_si256((const __m256i *)(xq + 3 * SML_GROUP))), one);
                const __m128 df = _mm_cvtepi32_ps(
                    sml_hsum4_i32(acc[0], acc[1], acc[2], acc[3]));
                const __m128 af = _mm_loadu_ps(ab->scale + gbase);
                vsum[b] = _mm_add_ps(vsum[b], _mm_mul_ps(wsc, _mm_mul_ps(af, df)));
                if (mqp) {
                    const __m128 sf = _mm_cvtepi32_ps(_mm_loadu_si128(
                        (const __m128i *)(ab->sums + gbase)));
                    vsum[b] = _mm_add_ps(vsum[b],
                        _mm_mul_ps(wmin, _mm_mul_ps(af, sf)));
                }
            }
        }
        for (int b = 0; b < nb; b++) {
            __m128 v = _mm_hadd_ps(vsum[b], vsum[b]);
            v = _mm_hadd_ps(v, v);
            y[(size_t) b * (size_t) nrows + r] += _mm_cvtss_f32(v);
        }
    }
}

// Eight-lane flat-4: two quads per step and one 256-bit epilogue for eight
// groups, where the four-lane kernels above run two 128-bit ones. The row and
// batch entry points share this one function, so batched prompt reading and
// stepping perform identical arithmetic and stay bit-identical. Measured
// standalone (one thread, same tensor and activations, scratchpad kbench8):
// batch 8 1.08-1.12x on the 2048- and 8192-column shapes; batch 1 0.99-1.12x,
// and 0.86x on 896 columns, whose four-group tail runs the 128-bit path.
// SML_LANES4=1 selects the four-lane kernels.
static inline __m256 sml_qf8(const uint8_t * base, int gbase) {
    if (g_scale6)
        return _mm256_set_m128(sml_qf4(base, gbase + 4), sml_qf4(base, gbase));
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
        _mm_loadl_epi64((const __m128i *)(base + gbase))));
}

// hcodes: an 8-bit tensor's high nibble plane (Q8), else NULL. An 8-bit code is
// lo + 16*hi, LINEAR, so each group's dot is dot(lo) + 16*dot(hi), summed in
// int32: each nibble times an int8 activation stays inside maddubs' int16 range,
// where a whole 8-bit code (255 x 127 x 2) would saturate it. Called with a
// constant NULL, the 4-bit path compiles to exactly what it was.
static inline __attribute__((always_inline)) void matvec_flat4_l8(float * y, const uint8_t * codes,
                            const uint8_t * hcodes, const uint16_t * sd,
                            const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                            const sml_act * a, int nb,
                            int nrows, int row0, int row1, int n) {
    const int ng     = n / SML_GROUP;
    const int nquad  = ng / 4;
    const int nblk   = ng / 8;
    const int tail   = ng - nblk * 8;                  // 0 or 4
    const int nscale = SML_NSCALE(n);
    const size_t rowbytes = (size_t) n / 2;
    const __m256i m15 = _mm256_set1_epi8(0x0F);
    const __m256i one = _mm256_set1_epi16(1);
    const int pf = sml_pf_dist;

    for (int r = row0; r < row1; r++) {
        const uint8_t  * cp  = codes + (size_t) r * rowbytes;
        const uint8_t  * hp  = hcodes ? hcodes + (size_t) r * rowbytes : NULL;
        const uint16_t * dp  = sd    + (size_t) r * nscale;
        const uint8_t  * qp  = sq    + (size_t) r * sml_grow(ng);
        const uint16_t * mdp = md ? md + (size_t) r * nscale : NULL;
        const uint8_t  * mqp = mq ? mq + (size_t) r * sml_grow(ng) : NULL;
        __m256 vsum[SML_KERNEL_MAX_BATCH];
        __m128 vt[SML_KERNEL_MAX_BATCH];
        for (int b = 0; b < nb; b++) { vsum[b] = _mm256_setzero_ps(); vt[b] = _mm_setzero_ps(); }

        for (int k = 0; k < nblk; k++) {
            if (pf && 2 * k + pf < nquad)
                _mm_prefetch((const char *)(cp + (size_t) pf * 64), _MM_HINT_NTA);
            const int gbase = k * 8;
            const __m256i b0 = _mm256_loadu_si256((const __m256i *)(cp));
            const __m256i b1 = _mm256_loadu_si256((const __m256i *)(cp + 32));
            const __m256i b2 = _mm256_loadu_si256((const __m256i *)(cp + 64));
            const __m256i b3 = _mm256_loadu_si256((const __m256i *)(cp + 96));
            cp += 128;
            const __m256i c[8] = {
                _mm256_and_si256(b0, m15), _mm256_and_si256(_mm256_srli_epi16(b0, 4), m15),
                _mm256_and_si256(b1, m15), _mm256_and_si256(_mm256_srli_epi16(b1, 4), m15),
                _mm256_and_si256(b2, m15), _mm256_and_si256(_mm256_srli_epi16(b2, 4), m15),
                _mm256_and_si256(b3, m15), _mm256_and_si256(_mm256_srli_epi16(b3, 4), m15) };
            __m256i h[8];
            if (hcodes) {
                const __m256i e0 = _mm256_loadu_si256((const __m256i *)(hp));
                const __m256i e1 = _mm256_loadu_si256((const __m256i *)(hp + 32));
                const __m256i e2 = _mm256_loadu_si256((const __m256i *)(hp + 64));
                const __m256i e3 = _mm256_loadu_si256((const __m256i *)(hp + 96));
                hp += 128;
                h[0] = _mm256_and_si256(e0, m15); h[1] = _mm256_and_si256(_mm256_srli_epi16(e0, 4), m15);
                h[2] = _mm256_and_si256(e1, m15); h[3] = _mm256_and_si256(_mm256_srli_epi16(e1, 4), m15);
                h[4] = _mm256_and_si256(e2, m15); h[5] = _mm256_and_si256(_mm256_srli_epi16(e2, 4), m15);
                h[6] = _mm256_and_si256(e3, m15); h[7] = _mm256_and_si256(_mm256_srli_epi16(e3, 4), m15);
            }
            const float dsc = sml_f16(dp[gbase / SML_GROUPS_PER_SCALE]);
            const __m256 wsc = _mm256_mul_ps(_mm256_set1_ps(dsc), sml_qf8(qp, gbase));
            __m256 wmin = _mm256_setzero_ps();
            if (mqp) {
                const float mdsc = sml_f16(mdp[gbase / SML_GROUPS_PER_SCALE]);
                wmin = _mm256_mul_ps(_mm256_set1_ps(mdsc),
                    _mm256_sub_ps(sml_qf8(mqp, gbase), _mm256_set1_ps(sml_minbias())));
            }
            for (int b = 0; b < nb; b++) {
                const sml_act * ab = &a[b];
                const int8_t * xq = ab->q + (size_t) gbase * SML_GROUP;
                __m256i acc[8];
                for (int g = 0; g < 8; g++)
                    acc[g] = _mm256_madd_epi16(_mm256_maddubs_epi16(c[g],
                        _mm256_loadu_si256((const __m256i *)(xq + (size_t) g * SML_GROUP))), one);
                if (hcodes)
                    for (int g = 0; g < 8; g++)
                        acc[g] = _mm256_add_epi32(acc[g], _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(h[g],
                            _mm256_loadu_si256((const __m256i *)(xq + (size_t) g * SML_GROUP))), one), 4));
                const __m256 df = _mm256_cvtepi32_ps(_mm256_set_m128i(
                    sml_hsum4_i32(acc[4], acc[5], acc[6], acc[7]),
                    sml_hsum4_i32(acc[0], acc[1], acc[2], acc[3])));
                const __m256 af = _mm256_loadu_ps(ab->scale + gbase);
                vsum[b] = _mm256_add_ps(vsum[b], _mm256_mul_ps(wsc, _mm256_mul_ps(af, df)));
                if (mqp) {
                    const __m256 sf = _mm256_cvtepi32_ps(
                        _mm256_loadu_si256((const __m256i *)(ab->sums + gbase)));
                    vsum[b] = _mm256_add_ps(vsum[b], _mm256_mul_ps(wmin, _mm256_mul_ps(af, sf)));
                }
            }
        }
        if (tail) {
            const int gbase = nblk * 8;
            const __m256i b0 = _mm256_loadu_si256((const __m256i *)(cp));
            const __m256i b1 = _mm256_loadu_si256((const __m256i *)(cp + 32));
            const __m256i c0 = _mm256_and_si256(b0, m15);
            const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(b0, 4), m15);
            const __m256i c2 = _mm256_and_si256(b1, m15);
            const __m256i c3 = _mm256_and_si256(_mm256_srli_epi16(b1, 4), m15);
            const float dsc = sml_f16(dp[gbase / SML_GROUPS_PER_SCALE]);
            const __m128 wsc = _mm_mul_ps(_mm_set1_ps(dsc), sml_qf4(qp, gbase));
            __m128 wmin = _mm_setzero_ps();
            if (mqp) {
                const float mdsc = sml_f16(mdp[gbase / SML_GROUPS_PER_SCALE]);
                wmin = _mm_mul_ps(_mm_set1_ps(mdsc),
                    _mm_sub_ps(sml_qf4(mqp, gbase), _mm_set1_ps(sml_minbias())));
            }
            __m256i hc[4];
            if (hcodes) {
                const __m256i e0 = _mm256_loadu_si256((const __m256i *)(hp));
                const __m256i e1 = _mm256_loadu_si256((const __m256i *)(hp + 32));
                hc[0] = _mm256_and_si256(e0, m15); hc[1] = _mm256_and_si256(_mm256_srli_epi16(e0, 4), m15);
                hc[2] = _mm256_and_si256(e1, m15); hc[3] = _mm256_and_si256(_mm256_srli_epi16(e1, 4), m15);
            }
            for (int b = 0; b < nb; b++) {
                const sml_act * ab = &a[b];
                const int8_t * xq = ab->q + (size_t) gbase * SML_GROUP;
                __m256i a0 = _mm256_madd_epi16(_mm256_maddubs_epi16(c0,
                    _mm256_loadu_si256((const __m256i *)(xq))), one);
                __m256i a1 = _mm256_madd_epi16(_mm256_maddubs_epi16(c1,
                    _mm256_loadu_si256((const __m256i *)(xq + SML_GROUP))), one);
                __m256i a2 = _mm256_madd_epi16(_mm256_maddubs_epi16(c2,
                    _mm256_loadu_si256((const __m256i *)(xq + 2 * SML_GROUP))), one);
                __m256i a3 = _mm256_madd_epi16(_mm256_maddubs_epi16(c3,
                    _mm256_loadu_si256((const __m256i *)(xq + 3 * SML_GROUP))), one);
                if (hcodes) {
                    __m256i * as[4] = { &a0, &a1, &a2, &a3 };
                    for (int g = 0; g < 4; g++)
                        *as[g] = _mm256_add_epi32(*as[g], _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(hc[g],
                            _mm256_loadu_si256((const __m256i *)(xq + (size_t) g * SML_GROUP))), one), 4));
                }
                const __m128 af = _mm_loadu_ps(ab->scale + gbase);
                vt[b] = _mm_add_ps(vt[b], _mm_mul_ps(wsc,
                    _mm_mul_ps(af, _mm_cvtepi32_ps(sml_hsum4_i32(a0, a1, a2, a3)))));
                if (mqp) {
                    const __m128 sf = _mm_cvtepi32_ps(_mm_loadu_si128(
                        (const __m128i *)(ab->sums + gbase)));
                    vt[b] = _mm_add_ps(vt[b], _mm_mul_ps(wmin, _mm_mul_ps(af, sf)));
                }
            }
        }
        for (int b = 0; b < nb; b++) {
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(vsum[b]),
                                  _mm256_extractf128_ps(vsum[b], 1));
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            float tot = _mm_cvtss_f32(s);
            if (tail) {
                __m128 t = _mm_hadd_ps(vt[b], vt[b]);
                t = _mm_hadd_ps(t, t);
                tot += _mm_cvtss_f32(t);
            }
            y[(size_t) b * (size_t) nrows + r] += tot;
        }
    }
}

void sml_matvec_rows_flat4(float * y, const uint8_t * codes, const uint16_t * sd,
                           const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                           const sml_act * a,
                           int row0, int row1, int n) {
    if (g_lanes4) { matvec_rows_flat4_l4(y, codes, sd, sq, md, mq, a, row0, row1, n); return; }
    matvec_flat4_l8(y, codes, NULL, sd, sq, md, mq, a, 1, 0, row0, row1, n);
}

// ---------------------------------------------------------------- interleaved
//
// Decode reads a flat tensor from five arrays per row -- codes, the two fp16
// superblock scales and the two per-group byte fields -- so six threads keep
// thirty streams open. Streaming the same bytes interleaved per 256-weight
// block measured 1.09x the five-array read, and with the engine's kernel on it
// 1.15x (scratchpad kbd3, alternated per token so heat hit both alike). The
// file layout is unchanged; decode repacks a private copy once at start-up.
//
//   4-bit block: [codes 128][sq 8][mq 8][sd 2][md 2]            = 148 bytes
//   6-bit block: [nibbles 128][2-bit packs 64][sq 8][mq 8][sd 2][md 2] = 212
//
// Only for the shape every flat file has in practice: a per-group min, whole
// 256-weight superblocks. Anything else keeps the arrays.
//
// A v9 file (--scale6) packs the group fields at 6 bits, four in three bytes;
// the block then carries them as they are -- 6 bytes each instead of 8, so a
// 4-bit block is 144 bytes, not 148: 2.7% less read per token -- and the kernel
// unpacks eight with one variable shift (il_fields).
int sml_il_ok(int n, int nbits, int has_min) {
    return has_min && (nbits == 4 || nbits == 6) && n % SML_SCALE_SUPER == 0;
}

// fields: [sq][mq] (8 bytes each, or 6 at 6 bits) then [sd 2][md 2]
static inline int il_fbytes(void) { return g_scale6 ? 16 : 20; }
static inline size_t il_blk(int nbits) { return (size_t) (nbits == 6 ? 192 : 128) + (size_t) il_fbytes(); }

size_t sml_il_row_bytes(int n, int nbits) {
    return (size_t) (n / SML_SCALE_SUPER) * il_blk(nbits);
}

void sml_il_pack(uint8_t * dst, const uint8_t * lo4, const uint8_t * hi2,
                 const uint16_t * sd, const uint8_t * sq,
                 const uint16_t * md, const uint8_t * mq,
                 int nbits, int row0, int row1, int n) {
    const int nb = n / SML_SCALE_SUPER, ng = n / SML_GROUP, ns = SML_NSCALE(n);
    const size_t blk = il_blk(nbits);
    const size_t grow = sml_grow(ng);
    const int fw = g_scale6 ? 6 : 8;        // bytes of 8 group fields
    for (int r = row0; r < row1; r++) {
        uint8_t * d = dst + (size_t) r * nb * blk;
        for (int k = 0; k < nb; k++, d += blk) {
            uint8_t * p = d;
            memcpy(p, lo4 + (size_t) r * (n / 2) + (size_t) k * 128, 128); p += 128;
            if (nbits == 6) { memcpy(p, hi2 + (size_t) r * (n / 4) + (size_t) k * 64, 64); p += 64; }
            memcpy(p,          sq + (size_t) r * grow + (size_t) k * fw, (size_t) fw);
            memcpy(p + fw,     mq + (size_t) r * grow + (size_t) k * fw, (size_t) fw);
            memcpy(p + 2 * fw,     sd + (size_t) r * ns + k, 2);
            memcpy(p + 2 * fw + 2, md + (size_t) r * ns + k, 2);
        }
    }
}

// Eight 6-bit fields from six bytes (sml_unpack6x4's layout twice: value i at
// bits 6i of the little-endian 48-bit word), as floats.
static inline __m256 il_f6(const uint8_t * p) {
    uint32_t lo, hi;
    memcpy(&lo, p, 4);
    memcpy(&hi, p + 3, 4);
    lo &= 0xFFFFFFu;
    hi &= 0xFFFFFFu;
    const __m256i v = _mm256_setr_epi32((int) lo, (int) lo, (int) lo, (int) lo,
                                        (int) hi, (int) hi, (int) hi, (int) hi);
    const __m256i sh = _mm256_setr_epi32(0, 6, 12, 18, 0, 6, 12, 18);
    return _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(v, sh), _mm256_set1_epi32(63)));
}

// The block's scale fields -> this superblock's per-group weight scale and min.
static inline void il_fields(const uint8_t * f, __m256 * wsc, __m256 * wmin) {
    __m256 sqf, mqf;
    uint16_t sdh, mdh;
    if (g_scale6) {
        sqf = il_f6(f);
        mqf = il_f6(f + 6);
        memcpy(&sdh, f + 12, 2);
        memcpy(&mdh, f + 14, 2);
    } else {
        sqf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(f))));
        mqf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(f + 8))));
        memcpy(&sdh, f + 16, 2);
        memcpy(&mdh, f + 18, 2);
    }
    *wsc  = _mm256_mul_ps(_mm256_set1_ps(sml_f16(sdh)), sqf);
    *wmin = _mm256_mul_ps(_mm256_set1_ps(sml_f16(mdh)),
                          _mm256_sub_ps(mqf, _mm256_set1_ps(sml_minbias())));
}

// The same arithmetic as matvec_flat4_l8 at nb = 1 and as the fused 6-bit
// kernel, in the same order, so the results are identical to the float.
static inline __attribute__((always_inline)) void il_epilogue(
        __m256 * vsum, const __m256i acc[8], const sml_act * a, int gbase, __m256 wsc, __m256 wmin) {
    const __m256 df = _mm256_cvtepi32_ps(_mm256_set_m128i(
        sml_hsum4_i32(acc[4], acc[5], acc[6], acc[7]),
        sml_hsum4_i32(acc[0], acc[1], acc[2], acc[3])));
    const __m256 af = _mm256_loadu_ps(a->scale + gbase);
    *vsum = _mm256_add_ps(*vsum, _mm256_mul_ps(wsc, _mm256_mul_ps(af, df)));
    const __m256 sf = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(a->sums + gbase)));
    *vsum = _mm256_add_ps(*vsum, _mm256_mul_ps(wmin, _mm256_mul_ps(af, sf)));
}

static inline float il_hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

void sml_matvec_rows_il4(float * y, const uint8_t * il, const sml_act * a,
                         int row0, int row1, int n) {
    const int nb = n / SML_SCALE_SUPER;
    const size_t blk = il_blk(4);
    const __m256i m15 = _mm256_set1_epi8(0x0F);
    const __m256i one = _mm256_set1_epi16(1);
    for (int r = row0; r < row1; r++) {
        const uint8_t * p = il + (size_t) r * nb * blk;
        __m256 vsum = _mm256_setzero_ps();
        for (int k = 0; k < nb; k++, p += blk) {
            const __m256i b0 = _mm256_loadu_si256((const __m256i *)(p));
            const __m256i b1 = _mm256_loadu_si256((const __m256i *)(p + 32));
            const __m256i b2 = _mm256_loadu_si256((const __m256i *)(p + 64));
            const __m256i b3 = _mm256_loadu_si256((const __m256i *)(p + 96));
            const __m256i c[8] = {
                _mm256_and_si256(b0, m15), _mm256_and_si256(_mm256_srli_epi16(b0, 4), m15),
                _mm256_and_si256(b1, m15), _mm256_and_si256(_mm256_srli_epi16(b1, 4), m15),
                _mm256_and_si256(b2, m15), _mm256_and_si256(_mm256_srli_epi16(b2, 4), m15),
                _mm256_and_si256(b3, m15), _mm256_and_si256(_mm256_srli_epi16(b3, 4), m15) };
            const int8_t * xq = a->q + (size_t) k * SML_SCALE_SUPER;
            __m256i acc[8];
            for (int g = 0; g < 8; g++)
                acc[g] = _mm256_madd_epi16(_mm256_maddubs_epi16(c[g],
                    _mm256_loadu_si256((const __m256i *)(xq + (size_t) g * SML_GROUP))), one);
            __m256 wsc, wmin;
            il_fields(p + 128, &wsc, &wmin);
            il_epilogue(&vsum, acc, a, k * 8, wsc, wmin);
        }
        y[r] += il_hsum(vsum);
    }
}

void sml_matvec_rows_il6(float * y, const uint8_t * il, const sml_act * a,
                         int row0, int row1, int n) {
    const int nb = n / SML_SCALE_SUPER;
    const size_t blk = il_blk(6);
    const __m256i m15 = _mm256_set1_epi8(0x0F);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    const __m256i one = _mm256_set1_epi16(1);
    for (int r = row0; r < row1; r++) {
        const uint8_t * p = il + (size_t) r * nb * blk;
        __m256 vsum = _mm256_setzero_ps();
        for (int k = 0; k < nb; k++, p += blk) {
            const __m256i b0 = _mm256_loadu_si256((const __m256i *)(p));
            const __m256i b1 = _mm256_loadu_si256((const __m256i *)(p + 32));
            const __m256i b2 = _mm256_loadu_si256((const __m256i *)(p + 64));
            const __m256i b3 = _mm256_loadu_si256((const __m256i *)(p + 96));
            const __m256i e0 = _mm256_loadu_si256((const __m256i *)(p + 128));
            const __m256i e1 = _mm256_loadu_si256((const __m256i *)(p + 160));
            const __m256i c[8] = {
                _mm256_or_si256(_mm256_and_si256(b0, m15),
                                _mm256_and_si256(_mm256_slli_epi16(e0, 4), m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b0, 4), m15),
                                _mm256_and_si256(_mm256_slli_epi16(e0, 2), m30)),
                _mm256_or_si256(_mm256_and_si256(b1, m15),
                                _mm256_and_si256(e0, m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b1, 4), m15),
                                _mm256_and_si256(_mm256_srli_epi16(e0, 2), m30)),
                _mm256_or_si256(_mm256_and_si256(b2, m15),
                                _mm256_and_si256(_mm256_slli_epi16(e1, 4), m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b2, 4), m15),
                                _mm256_and_si256(_mm256_slli_epi16(e1, 2), m30)),
                _mm256_or_si256(_mm256_and_si256(b3, m15),
                                _mm256_and_si256(e1, m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b3, 4), m15),
                                _mm256_and_si256(_mm256_srli_epi16(e1, 2), m30)) };
            const int8_t * xq = a->q + (size_t) k * SML_SCALE_SUPER;
            __m256i acc[8];
            for (int g = 0; g < 8; g++)
                acc[g] = _mm256_madd_epi16(_mm256_maddubs_epi16(c[g],
                    _mm256_loadu_si256((const __m256i *)(xq + (size_t) g * SML_GROUP))), one);
            __m256 wsc, wmin;
            il_fields(p + 192, &wsc, &wmin);
            il_epilogue(&vsum, acc, a, k * 8, wsc, wmin);
        }
        y[r] += il_hsum(vsum);
    }
}

// FLAT 6-bit decode in ONE pass: code = nibble | extra << 4, built in registers
// and fed to one maddubs per group (63 x 127 x 2 fits int16).
//
// It used to be two kernel calls over the same rows -- the 2-bit plane through
// the ladder kernel with shift 4, then the nibble plane through flat4 -- so every
// group paid two dot products, two horizontal sums and two epilogues. On the
// tied 1B head (128256 x 2048 at 6 bits) that was 30% of a decode token.
//
// Layout as sml_dequant_row_flat6: hi2 is group-planar (pack of 128 weights in
// 32 bytes, bit-plane g of byte j is group g of the pack, column j), lo4 is the
// flat nibble layout. An 8-group block is 128 nibble bytes and two packs.
void sml_matvec_rows_flat6(float * y, const uint8_t * lo4, const uint8_t * hi2,
                           const uint16_t * sd, const uint8_t * sq,
                           const uint16_t * md, const uint8_t * mq,
                           const sml_act * a, int row0, int row1, int n) {
    const int ng     = n / SML_GROUP;
    const int nblk   = ng / 8;
    const int nscale = SML_NSCALE(n);
    if (nblk * 8 != ng) {        // a four-group tail: the two-pass form handles it
        sml_matvec_rows(y, hi2, sd, sq, md, mq, a, row0, row1, n, 4, 0.0f);
        sml_matvec_rows_flat4(y, lo4, sd, sq, NULL, NULL, a, row0, row1, n);
        return;
    }
    const __m256i m15 = _mm256_set1_epi8(0x0F);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    const __m256i one = _mm256_set1_epi16(1);

    for (int r = row0; r < row1; r++) {
        const uint8_t  * cp  = lo4 + (size_t) r * ((size_t) n / 2);
        const uint8_t  * ep  = hi2 + (size_t) r * ((size_t) n / 4);
        const uint16_t * dp  = sd  + (size_t) r * nscale;
        const uint8_t  * qp  = sq  + (size_t) r * sml_grow(ng);
        const uint16_t * mdp = md ? md + (size_t) r * nscale : NULL;
        const uint8_t  * mqp = mq ? mq + (size_t) r * sml_grow(ng) : NULL;
        __m256 vsum = _mm256_setzero_ps();

        for (int k = 0; k < nblk; k++) {
            const int gbase = k * 8;
            const __m256i b0 = _mm256_loadu_si256((const __m256i *)(cp));
            const __m256i b1 = _mm256_loadu_si256((const __m256i *)(cp + 32));
            const __m256i b2 = _mm256_loadu_si256((const __m256i *)(cp + 64));
            const __m256i b3 = _mm256_loadu_si256((const __m256i *)(cp + 96));
            const __m256i e0 = _mm256_loadu_si256((const __m256i *)(ep));
            const __m256i e1 = _mm256_loadu_si256((const __m256i *)(ep + 32));
            cp += 128;
            ep += 64;
            // bits 4-5 of each byte take the pack's bit-plane g; the 16-bit shifts
            // carry across bytes only into bits the mask drops
            const __m256i c[8] = {
                _mm256_or_si256(_mm256_and_si256(b0, m15),
                                _mm256_and_si256(_mm256_slli_epi16(e0, 4), m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b0, 4), m15),
                                _mm256_and_si256(_mm256_slli_epi16(e0, 2), m30)),
                _mm256_or_si256(_mm256_and_si256(b1, m15),
                                _mm256_and_si256(e0, m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b1, 4), m15),
                                _mm256_and_si256(_mm256_srli_epi16(e0, 2), m30)),
                _mm256_or_si256(_mm256_and_si256(b2, m15),
                                _mm256_and_si256(_mm256_slli_epi16(e1, 4), m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b2, 4), m15),
                                _mm256_and_si256(_mm256_slli_epi16(e1, 2), m30)),
                _mm256_or_si256(_mm256_and_si256(b3, m15),
                                _mm256_and_si256(e1, m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b3, 4), m15),
                                _mm256_and_si256(_mm256_srli_epi16(e1, 2), m30)) };
            const int8_t * xq = a->q + (size_t) gbase * SML_GROUP;
            __m256i acc[8];
            for (int g = 0; g < 8; g++)
                acc[g] = _mm256_madd_epi16(_mm256_maddubs_epi16(c[g],
                    _mm256_loadu_si256((const __m256i *)(xq + (size_t) g * SML_GROUP))), one);
            const __m256 df = _mm256_cvtepi32_ps(_mm256_set_m128i(
                sml_hsum4_i32(acc[4], acc[5], acc[6], acc[7]),
                sml_hsum4_i32(acc[0], acc[1], acc[2], acc[3])));
            const __m256 af  = _mm256_loadu_ps(a->scale + gbase);
            const float  dsc = sml_f16(dp[gbase / SML_GROUPS_PER_SCALE]);
            const __m256 wsc = _mm256_mul_ps(_mm256_set1_ps(dsc), sml_qf8(qp, gbase));
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(wsc, _mm256_mul_ps(af, df)));
            if (mqp) {
                const float mdsc = sml_f16(mdp[gbase / SML_GROUPS_PER_SCALE]);
                const __m256 wmin = _mm256_mul_ps(_mm256_set1_ps(mdsc),
                    _mm256_sub_ps(sml_qf8(mqp, gbase), _mm256_set1_ps(sml_minbias())));
                const __m256 sf = _mm256_cvtepi32_ps(
                    _mm256_loadu_si256((const __m256i *)(a->sums + gbase)));
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(wmin, _mm256_mul_ps(af, sf)));
            }
        }
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(vsum), _mm256_extractf128_ps(vsum, 1));
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        y[r] += _mm_cvtss_f32(s);
    }
}

void sml_matvec_batch_flat4(float * y, const uint8_t * codes, const uint16_t * sd,
                            const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                            const sml_act * a, int nb,
                            int nrows, int row0, int row1, int n) {
    if (g_lanes4) {
        matvec_batch_flat4_l4(y, codes, sd, sq, md, mq, a, nb, nrows, row0, row1, n);
        return;
    }
    // A literal 8 lets the compiler keep all slot accumulators in registers;
    // with a runtime nb over SML_KERNEL_MAX_BATCH-sized arrays they live on the
    // stack and every accumulate is a load and a store.
    if (nb == 8) matvec_flat4_l8(y, codes, NULL, sd, sq, md, mq, a, 8, nrows, row0, row1, n);
    else         matvec_flat4_l8(y, codes, NULL, sd, sq, md, mq, a, nb, nrows, row0, row1, n);
}

// The 1-bit plane (Q3, Q5, Q7): per row, per group, 4 bytes, bit j (least
// significant first) is weight j's extra bit. Adds step * 2^shift * sum(bit*x)
// to y; the min rides on the tensor's base plane. Each group's bits become a
// byte mask (broadcast, AND with the bit, compare), the masked int8 activations
// are summed by maddubs against ones: 32 weights per handful of instructions.
static inline __m256i sml_bits_mask(uint32_t w) {
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                          2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x((long long) 0x8040201008040201ULL);
    const __m256i v = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) w), shuf), bit);
    return _mm256_cmpeq_epi8(v, bit);
}

void sml_matvec_bits1(float * y, const uint8_t * plane, const uint16_t * sd, const uint8_t * sq,
                      const sml_act * a, int nb, int nrows, int row0, int row1, int n, int shift) {
    const int ng = n / SML_GROUP, nscale = SML_NSCALE(n);
    const __m256i ones = _mm256_set1_epi8(1), one16 = _mm256_set1_epi16(1);
    const float mul = (float) (1 << shift);
    for (int r = row0; r < row1; r++) {
        const uint8_t  * pp = plane + (size_t) r * ng * 4;
        const uint16_t * dp = sd + (size_t) r * nscale;
        const uint8_t  * qp = sq + (size_t) r * sml_grow(ng);
        for (int b = 0; b < nb; b++) {
            const sml_act * ab = &a[b];
            float tot = 0.0f;
            for (int g = 0; g < ng; g++) {
                uint32_t w;
                memcpy(&w, pp + (size_t) g * 4, 4);
                if (!w) continue;
                const __m256i xm = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(ab->q + (size_t) g * SML_GROUP)),
                                                    sml_bits_mask(w));
                const __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(ones, xm), one16);
                __m128i h = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                h = _mm_hadd_epi32(h, h);
                h = _mm_hadd_epi32(h, h);
                const float step = sml_f16(dp[g / SML_GROUPS_PER_SCALE]) * (float) sml_gfield(qp, g, g_scale6);
                tot += step * ab->scale[g] * (float) _mm_cvtsi128_si32(h);
            }
            y[(size_t) b * (size_t) nrows + r] += tot * mul;
        }
    }
}

// Q8: an 8-bit FLAT tensor is two nibble planes in the 4-bit layout, low
// (codes_off[0]) and high (codes_off[1]); code = lo + 16*hi. The same kernel as
// 4-bit, reading both planes: twice the code bytes, the same scales and mins.
void sml_matvec_rows_flat8(float * y, const uint8_t * lo, const uint8_t * hi, const uint16_t * sd,
                           const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                           const sml_act * a, int row0, int row1, int n) {
    matvec_flat4_l8(y, lo, hi, sd, sq, md, mq, a, 1, 0, row0, row1, n);
}

void sml_matvec_batch_flat8(float * y, const uint8_t * lo, const uint8_t * hi, const uint16_t * sd,
                            const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                            const sml_act * a, int nb, int nrows, int row0, int row1, int n) {
    if (nb == 8) matvec_flat4_l8(y, lo, hi, sd, sq, md, mq, a, 8, nrows, row0, row1, n);
    else         matvec_flat4_l8(y, lo, hi, sd, sq, md, mq, a, nb, nrows, row0, row1, n);
}

// Batched flat 4-bit on per-256 activations (sml_quantize_act_sb ONLY -- it
// reads the superblock scale from the superblock's first group). Each group's
// integer weight scale multiplies inside the pair-add (madd by the scale, not
// by ones), so a superblock's 8 groups sum in int32 and the float epilogue runs
// once per superblock per slot, the horizontal sum once per ROW.
//
// Loop order is what made it pay. Building all 8 groups' codes and scales first
// and then looping slots needs 16 ymm before any work and measured 0.72-1.00x
// the per-group kernel (spills). Groups outside, slots inside keeps one group
// pair's codes and scales live beside the slot accumulators: 1.32-1.54x on one
// thread, exact to 1e-7 against the per-group kernel on the same activations
// (scratchpad/live/kbsb2.c).
// Overflow: |maddubs| <= 2*15*127, x qp <= 255 x 2 per madd, x 8 groups < 2^24.
// Up to 8 slots of one superblock, accumulators in registers. Called with a
// literal m == 8 so the slot loops unroll; a runtime count over larger arrays
// put the accumulators on the stack.
static inline __attribute__((always_inline)) void sb_octet(
        const uint8_t * cp, const uint8_t * qp, int g0, int gn, const sml_act * a, int m,
        __m256 * vs, float dsc, __m256i mn8, float mdsc, int hm, __m256i m15) {
    __m256i acc[8];
    for (int b = 0; b < m; b++) acc[b] = _mm256_setzero_si256();
    for (int h = 0; h < gn / 2; h++) {
        const __m256i bb = _mm256_loadu_si256((const __m256i *)(cp + (size_t)(g0 / 2 + h) * 32));
        const __m256i clo = _mm256_and_si256(bb, m15);
        const __m256i chi = _mm256_and_si256(_mm256_srli_epi16(bb, 4), m15);
        const __m256i slo = _mm256_set1_epi16((short) sml_gfield(qp, g0 + 2 * h, g_scale6));
        const __m256i shi = _mm256_set1_epi16((short) sml_gfield(qp, g0 + 2 * h + 1, g_scale6));
        const size_t olo = (size_t)(g0 + 2 * h) * SML_GROUP, ohi = olo + SML_GROUP;
        for (int b = 0; b < m; b++) {
            const int8_t * xq = a[b].q;
            acc[b] = _mm256_add_epi32(acc[b], _mm256_madd_epi16(_mm256_maddubs_epi16(clo,
                         _mm256_loadu_si256((const __m256i *)(xq + olo))), slo));
            acc[b] = _mm256_add_epi32(acc[b], _mm256_madd_epi16(_mm256_maddubs_epi16(chi,
                         _mm256_loadu_si256((const __m256i *)(xq + ohi))), shi));
        }
    }
    for (int b = 0; b < m; b++)
        vs[b] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[b]),
                                _mm256_set1_ps(dsc * a[b].scale[g0]), vs[b]);
    if (hm) {
        for (int b = 0; b < m; b++) {
            __m256i sm;
            if (gn == SML_GROUPS_PER_SCALE) {
                sm = _mm256_loadu_si256((const __m256i *)(a[b].sums + g0));
            } else {
                int32_t t[SML_GROUPS_PER_SCALE] = { 0 };
                for (int g = 0; g < gn; g++) t[g] = a[b].sums[g0 + g];
                sm = _mm256_loadu_si256((const __m256i *) t);
            }
            vs[b] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_mullo_epi32(mn8, sm)),
                                    _mm256_set1_ps(mdsc * a[b].scale[g0]), vs[b]);
        }
    }
}

// The slots go through each superblock eight at a time, so a batch larger than 8
// still reads each weight superblock from memory ONCE and the later octets find
// it in L1. Prompt reading at 8 slots read the model once per 8 tokens -- ~10 GB
// for a 109-token 1B prompt, ~0.37 s of ~1.3 s at DRAM speed -- where the
// reference engine reads it once per up to 512.
static inline __attribute__((always_inline)) void flat4_sb_body(float * y, const uint8_t * codes,
                               const uint16_t * sd,
                               const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                               const sml_act * a, int nb,
                               int nrows, int row0, int row1, int n) {
    const int ng     = n / SML_GROUP;
    const int nsb    = (ng + SML_GROUPS_PER_SCALE - 1) / SML_GROUPS_PER_SCALE;
    const int nscale = SML_NSCALE(n);
    const size_t rowbytes = (size_t) n / 2;
    const __m256i m15 = _mm256_set1_epi8(0x0F);
    const int mbias = (int) sml_minbias();

    for (int r = row0; r < row1; r++) {
        const uint8_t  * cp  = codes + (size_t) r * rowbytes;
        const uint16_t * dp  = sd    + (size_t) r * nscale;
        const uint8_t  * qp  = sq    + (size_t) r * sml_grow(ng);
        const uint16_t * mdp = md ? md + (size_t) r * nscale : NULL;
        const uint8_t  * mqp = mq ? mq + (size_t) r * sml_grow(ng) : NULL;
        __m256 vs[SML_KERNEL_MAX_BATCH];
        for (int b = 0; b < nb; b++) vs[b] = _mm256_setzero_ps();

        for (int k = 0; k < nsb; k++) {
            const int g0 = k * SML_GROUPS_PER_SCALE;
            const int gn = g0 + SML_GROUPS_PER_SCALE <= ng ? SML_GROUPS_PER_SCALE : ng - g0;
            const float dsc = sml_f16(dp[k]);
            __m256i mn8 = _mm256_setzero_si256();
            float mdsc = 0.0f;
            if (mqp) {
                int32_t m[SML_GROUPS_PER_SCALE] = { 0 };
                for (int g = 0; g < gn; g++) m[g] = sml_gfield(mqp, g0 + g, g_scale6) - mbias;
                mn8 = _mm256_loadu_si256((const __m256i *) m);
                mdsc = sml_f16(mdp[k]);
            }
            for (int o = 0; o < nb; o += 8) {
                const int m = nb - o < 8 ? nb - o : 8;
                if (m == 8) sb_octet(cp, qp, g0, gn, a + o, 8, vs + o, dsc, mn8, mdsc, mqp != NULL, m15);
                else        sb_octet(cp, qp, g0, gn, a + o, m, vs + o, dsc, mn8, mdsc, mqp != NULL, m15);
            }
        }
        for (int b = 0; b < nb; b++) {
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(vs[b]), _mm256_extractf128_ps(vs[b], 1));
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            y[(size_t) b * (size_t) nrows + r] += _mm_cvtss_f32(s);
        }
    }
}

// ---------------------------------------------------------------------------
// Rows in lanes (prompt reading). Eight rows are repacked so one 32-byte vector
// holds four codes of each of the eight, and a token's four activations are
// broadcast against it: the eight int32 lanes of the result ARE the eight rows.
//
// What that buys, per 256 multiply-adds:
//  - within a 32-weight group each row has ONE scale, so the group's eight
//    maddubs steps sum in int16 first (8 * 2 * 15 * 127 = 30480 < 32767) and
//    take the scale in one madd: 9 multiplies instead of 16;
//  - the float epilogue, the mins and the store run once per eight rows, and
//    y is [token][row], so the store is one vector with no horizontal sum.
// Measured 1.9-2.5x per thread against flat4_sb_body on the 1B's shapes, same
// activations, agreeing to ~1e-6 relative (float summation order only; the
// integer superblock sums are identical). scratchpad/cpu/kbench_rl.c.
//
// The repack is per call and per 8 rows: ~n/2 bytes read and transposed, then
// reused by every token in the batch. SML_RL=0 restores flat4_sb_body.
// ---------------------------------------------------------------------------
static int g_rl = -1;

static inline void rl_tr8x8(__m256i r[8]) {
    const __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]), t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    const __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]), t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    const __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]), t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    const __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]), t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    const __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
    const __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
    const __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
    const __m256i u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
    r[0] = _mm256_permute2x128_si256(u0, u4, 0x20); r[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
    r[1] = _mm256_permute2x128_si256(u1, u5, 0x20); r[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    r[2] = _mm256_permute2x128_si256(u2, u6, 0x20); r[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
    r[3] = _mm256_permute2x128_si256(u3, u7, 0x20); r[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}

typedef struct {
    uint8_t * U;      // ng * 256: group g, step j -> 8 rows x 4 codes (0..15)
    __m256i * S;      // ng: the group scale of each row, int16 lanes 2r and 2r+1
    __m256  * D;      // nsb: fp32 superblock scale of each row
    __m256  * MNF;    // ng: mdsc * (min - bias) of each row
    float   * ASC;    // nb * nsb: activation scale per token and superblock
    float   * SF;     // nb * ng: activation scale * group sum, for the mins
} rl_pack;

// Per-thread scratch, grown on demand and kept: the batched body calls this
// from inside its parallel region once per chunk of rows.
static _Thread_local uint8_t * t_rl_buf = NULL;
static _Thread_local size_t    t_rl_cap = 0;

static int rl_setup(rl_pack * p, const sml_act * a, int nb, int n, int mins) {
    const int ng = n / SML_GROUP, nsb = SML_NSCALE(n);
    const size_t need = (size_t) ng * 256 + (size_t) ng * 32 + (size_t) nsb * 32 + (size_t) ng * 32
                      + (size_t) nb * nsb * 4 + (size_t) nb * ng * 4 + 64;
    if (need > t_rl_cap) {
        _mm_free(t_rl_buf);
        t_rl_buf = _mm_malloc(need, 64);
        t_rl_cap = t_rl_buf ? need : 0;
        if (!t_rl_buf) return 0;
    }
    uint8_t * b = t_rl_buf;
    p->U   = b;               b += (size_t) ng * 256;
    p->S   = (__m256i *) b;   b += (size_t) ng * 32;
    p->D   = (__m256 *) b;    b += (size_t) nsb * 32;
    p->MNF = (__m256 *) b;    b += (size_t) ng * 32;
    p->ASC = (float *) b;     b += (size_t) nb * nsb * 4;
    p->SF  = (float *) b;
    for (int t = 0; t < nb; t++) {
        for (int k = 0; k < nsb; k++)
            p->ASC[(size_t) t * nsb + k] = a[t].scale[k * SML_GROUPS_PER_SCALE];
        if (mins)
            for (int g = 0; g < ng; g++)
                p->SF[(size_t) t * ng + g] = a[t].scale[g] * (float) a[t].sums[g];
    }
    return 1;
}

static void rl_repack(rl_pack * p, const uint8_t * codes, const uint16_t * sd, const uint8_t * sq,
                      const uint16_t * md, const uint8_t * mq, int r0, int n) {
    const int ng = n / SML_GROUP, nsb = SML_NSCALE(n);
    const size_t rowbytes = (size_t) n / 2, grow = sml_grow(ng);
    const float bias = sml_minbias();
    const __m256i m15 = _mm256_set1_epi8(0x0F);
    for (int P = 0; P < ng / 2; P++) {
        __m256i lo[8], hi[8];
        for (int r = 0; r < 8; r++) {
            const __m256i v = _mm256_loadu_si256(
                (const __m256i *)(codes + (size_t)(r0 + r) * rowbytes + (size_t) P * 32));
            lo[r] = _mm256_and_si256(v, m15);
            hi[r] = _mm256_and_si256(_mm256_srli_epi16(v, 4), m15);
        }
        rl_tr8x8(lo);
        rl_tr8x8(hi);
        for (int j = 0; j < 8; j++) {
            _mm256_store_si256((__m256i *)(p->U + (size_t)(2 * P)     * 256 + j * 32), lo[j]);
            _mm256_store_si256((__m256i *)(p->U + (size_t)(2 * P + 1) * 256 + j * 32), hi[j]);
        }
    }
    for (int g = 0; g < ng; g++) {
        int16_t s[16];
        for (int r = 0; r < 8; r++)
            s[2 * r] = s[2 * r + 1] = (int16_t) sml_gfield(sq + (size_t)(r0 + r) * grow, g, g_scale6);
        p->S[g] = _mm256_loadu_si256((const __m256i *) s);
    }
    for (int k = 0; k < nsb; k++) {
        float d[8];
        for (int r = 0; r < 8; r++) d[r] = sml_f16(sd[(size_t)(r0 + r) * nsb + k]);
        p->D[k] = _mm256_loadu_ps(d);
    }
    if (mq) for (int g = 0; g < ng; g++) {
        float m[8];
        for (int r = 0; r < 8; r++)
            m[r] = sml_f16(md[(size_t)(r0 + r) * nsb + g / SML_GROUPS_PER_SCALE]) *
                   ((float) sml_gfield(mq + (size_t)(r0 + r) * grow, g, g_scale6) - bias);
        p->MNF[g] = _mm256_loadu_ps(m);
    }
}

static inline __m256i rl_bc4(const int8_t * q) {
    int32_t w;
    memcpy(&w, q, 4);
    return _mm256_set1_epi32(w);
}

// Four tokens in named registers. Accumulator ARRAYS let GCC reorder the loop
// and spill every maddubs product to the stack; see sml_matvec_batch_flat4_rl.
static inline __attribute__((always_inline)) void rl_tile4(
        float * y, int nrows, int r0, const rl_pack * p, const sml_act * a, int t0,
        int ng, int nsb, int mins) {
    const int8_t * q0 = a[t0].q, * q1 = a[t0 + 1].q, * q2 = a[t0 + 2].q, * q3 = a[t0 + 3].q;
    __m256 v0 = _mm256_setzero_ps(), v1 = v0, v2 = v0, v3 = v0;
    for (int k = 0; k < nsb; k++) {
        __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
        for (int g = k * SML_GROUPS_PER_SCALE; g < (k + 1) * SML_GROUPS_PER_SCALE; g++) {
            __m256i h0 = _mm256_setzero_si256(), h1 = h0, h2 = h0, h3 = h0;
            const uint8_t * u = p->U + (size_t) g * 256;
            const size_t o = (size_t) g * SML_GROUP;
#pragma GCC unroll 8
            for (int j = 0; j < 8; j++) {
                const __m256i c = _mm256_load_si256((const __m256i *)(u + j * 32));
                h0 = _mm256_add_epi16(h0, _mm256_maddubs_epi16(c, rl_bc4(q0 + o + j * 4)));
                h1 = _mm256_add_epi16(h1, _mm256_maddubs_epi16(c, rl_bc4(q1 + o + j * 4)));
                h2 = _mm256_add_epi16(h2, _mm256_maddubs_epi16(c, rl_bc4(q2 + o + j * 4)));
                h3 = _mm256_add_epi16(h3, _mm256_maddubs_epi16(c, rl_bc4(q3 + o + j * 4)));
            }
            const __m256i sc = p->S[g];
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(h0, sc));
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(h1, sc));
            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(h2, sc));
            s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(h3, sc));
        }
        const __m256 d = p->D[k];
        const float * as = p->ASC + (size_t) t0 * nsb + k;
        v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s0), _mm256_mul_ps(d, _mm256_set1_ps(as[0])), v0);
        v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s1), _mm256_mul_ps(d, _mm256_set1_ps(as[nsb])), v1);
        v2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s2), _mm256_mul_ps(d, _mm256_set1_ps(as[2 * nsb])), v2);
        v3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s3), _mm256_mul_ps(d, _mm256_set1_ps(as[3 * nsb])), v3);
    }
    if (mins) {
        const float * sf = p->SF + (size_t) t0 * ng;
        for (int g = 0; g < ng; g++) {
            const __m256 m = p->MNF[g];
            v0 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[g]), v0);
            v1 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[ng + g]), v1);
            v2 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[2 * ng + g]), v2);
            v3 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[3 * ng + g]), v3);
        }
    }
    float * yp = y + (size_t) t0 * nrows + r0;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v0)); yp += nrows;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v1)); yp += nrows;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v2)); yp += nrows;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v3));
}

// One token: the tail of a batch that is not a multiple of four.
static inline __attribute__((always_inline)) void rl_tile1(
        float * y, int nrows, int r0, const rl_pack * p, const sml_act * a, int t,
        int ng, int nsb, int mins) {
    const int8_t * q = a[t].q;
    __m256 v = _mm256_setzero_ps();
    for (int k = 0; k < nsb; k++) {
        __m256i s = _mm256_setzero_si256();
        for (int g = k * SML_GROUPS_PER_SCALE; g < (k + 1) * SML_GROUPS_PER_SCALE; g++) {
            __m256i h = _mm256_setzero_si256();
            const uint8_t * u = p->U + (size_t) g * 256;
            const size_t o = (size_t) g * SML_GROUP;
#pragma GCC unroll 8
            for (int j = 0; j < 8; j++)
                h = _mm256_add_epi16(h, _mm256_maddubs_epi16(
                        _mm256_load_si256((const __m256i *)(u + j * 32)), rl_bc4(q + o + j * 4)));
            s = _mm256_add_epi32(s, _mm256_madd_epi16(h, p->S[g]));
        }
        v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s),
                            _mm256_mul_ps(p->D[k], _mm256_set1_ps(p->ASC[(size_t) t * nsb + k])), v);
    }
    if (mins)
        for (int g = 0; g < ng; g++)
            v = _mm256_fmadd_ps(p->MNF[g], _mm256_set1_ps(p->SF[(size_t) t * ng + g]), v);
    float * yp = y + (size_t) t * nrows + r0;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v));
}

// GCC's default x86 pipeline does not schedule before register allocation, and
// without it the four-token loop spilled 20 of its 32 maddubs products to the
// stack. Pressure-aware scheduling, for this function only: zero spills.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("schedule-insns", "sched-pressure")))
#endif
static int sml_matvec_batch_flat4_rl(float * y, const uint8_t * codes, const uint16_t * sd,
                                     const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                                     const sml_act * a, int nb, int nrows, int row0, int row1, int n) {
    const int ng = n / SML_GROUP, nsb = SML_NSCALE(n), mins = mq != NULL;
    rl_pack p;
    if (!rl_setup(&p, a, nb, n, mins)) return row0;
    int r = row0;
    for (; r + 8 <= row1; r += 8) {
        rl_repack(&p, codes, sd, sq, md, mq, r, n);
        int t = 0;
        for (; t + 4 <= nb; t += 4) rl_tile4(y, nrows, r, &p, a, t, ng, nsb, mins);
        for (; t < nb; t++)         rl_tile1(y, nrows, r, &p, a, t, ng, nsb, mins);
    }
    return r;
}

void sml_matvec_batch_flat4_sb(float * y, const uint8_t * codes, const uint16_t * sd,
                               const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                               const sml_act * a, int nb,
                               int nrows, int row0, int row1, int n) {
    if (g_rl < 0) {
        const char * e = getenv("SML_RL");
        g_rl = e ? atoi(e) : 4;   // the smallest batch that takes the rows-in-lanes path
    }
    // whole superblocks only: the tile sums eight groups per superblock
    //
    // Not bit-identical to flat4_sb_body (float summation order), so the batched
    // logits move -- by as much as the per-256 path already sits from stepping,
    // because a 1e-7 change flips a few int8 activation roundings downstream.
    // Graded against full precision on Llama-3.2-1B (sml_hf_grade.py, 20
    // prompts): 0.2373 vs 0.2361 rel-RMS, 12/20 top-1 both; se ~0.018.
    if (g_rl > 0 && nb >= g_rl && n % (SML_GROUP * SML_GROUPS_PER_SCALE) == 0 && row1 - row0 >= 8)
        row0 = sml_matvec_batch_flat4_rl(y, codes, sd, sq, md, mq, a, nb, nrows, row0, row1, n);
    if (row0 >= row1) return;
    // literal 8: the slot accumulators stay in registers (see sml_matvec_batch_flat4)
    if (nb == 8) flat4_sb_body(y, codes, sd, sq, md, mq, a, 8, nrows, row0, row1, n);
    else         flat4_sb_body(y, codes, sd, sq, md, mq, a, nb, nrows, row0, row1, n);
}

// ---------------------------------------------------------------------------
// Rows-in-lanes for FLAT 6-bit prompt reading.
//
// A 6-bit tensor was read as two batched passes -- the 2-bit plane through the
// ladder kernel, then the nibbles through flat4 -- at 57-67 GMAC/s on pythia's
// attn_v and ffn_down, against 270-320 for 4-bit tensors on rows-in-lanes: its
// ffn_down was 8 ms of every layer. Here the repack merges the planes into one
// 6-bit code per byte (code = nibble | extra << 4) and the tile is rl_tile4's.
// One change: a 6-bit product pair reaches 2 x 63 x 127 = 16002, so int16 holds
// two j-steps, not eight -- the partial sums go through madd every two steps.
// Overflow: |h| <= 32004; madd by the group scale (<= 255) <= 16.3M per lane,
// x 4 per group x 8 groups < 2^31.
static void rl_repack6(rl_pack * p, const uint8_t * lo4, const uint8_t * hi2,
                       const uint16_t * sd, const uint8_t * sq,
                       const uint16_t * md, const uint8_t * mq, int r0, int n) {
    const int ng = n / SML_GROUP, nsb = SML_NSCALE(n);
    const size_t rb4 = (size_t) n / 2, rb2 = (size_t) n / 4, grow = sml_grow(ng);
    const float bias = sml_minbias();
    const __m256i m15 = _mm256_set1_epi8(0x0F);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    for (int P = 0; P < ng / 2; P++) {
        __m256i lo[8], hi[8];
        // groups 2P and 2P+1 sit at bit-planes 2(P&1) and 2(P&1)+1 of pack P/2
        const int odd = P & 1;
        for (int r = 0; r < 8; r++) {
            const __m256i v = _mm256_loadu_si256(
                (const __m256i *)(lo4 + (size_t)(r0 + r) * rb4 + (size_t) P * 32));
            const __m256i e = _mm256_loadu_si256(
                (const __m256i *)(hi2 + (size_t)(r0 + r) * rb2 + (size_t)(P >> 1) * 32));
            const __m256i elo = odd ? _mm256_and_si256(e, m30)
                                    : _mm256_and_si256(_mm256_slli_epi16(e, 4), m30);
            const __m256i ehi = odd ? _mm256_and_si256(_mm256_srli_epi16(e, 2), m30)
                                    : _mm256_and_si256(_mm256_slli_epi16(e, 2), m30);
            lo[r] = _mm256_or_si256(_mm256_and_si256(v, m15), elo);
            hi[r] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v, 4), m15), ehi);
        }
        rl_tr8x8(lo);
        rl_tr8x8(hi);
        for (int j = 0; j < 8; j++) {
            _mm256_store_si256((__m256i *)(p->U + (size_t)(2 * P)     * 256 + j * 32), lo[j]);
            _mm256_store_si256((__m256i *)(p->U + (size_t)(2 * P + 1) * 256 + j * 32), hi[j]);
        }
    }
    for (int g = 0; g < ng; g++) {
        int16_t s[16];
        for (int r = 0; r < 8; r++)
            s[2 * r] = s[2 * r + 1] = (int16_t) sml_gfield(sq + (size_t)(r0 + r) * grow, g, g_scale6);
        p->S[g] = _mm256_loadu_si256((const __m256i *) s);
    }
    for (int k = 0; k < nsb; k++) {
        float d[8];
        for (int r = 0; r < 8; r++) d[r] = sml_f16(sd[(size_t)(r0 + r) * nsb + k]);
        p->D[k] = _mm256_loadu_ps(d);
    }
    if (mq) for (int g = 0; g < ng; g++) {
        float m[8];
        for (int r = 0; r < 8; r++)
            m[r] = sml_f16(md[(size_t)(r0 + r) * nsb + g / SML_GROUPS_PER_SCALE]) *
                   ((float) sml_gfield(mq + (size_t)(r0 + r) * grow, g, g_scale6) - bias);
        p->MNF[g] = _mm256_loadu_ps(m);
    }
}

static inline __attribute__((always_inline)) void rl6_tile4(
        float * y, int nrows, int r0, const rl_pack * p, const sml_act * a, int t0,
        int ng, int nsb, int mins) {
    const int8_t * q0 = a[t0].q, * q1 = a[t0 + 1].q, * q2 = a[t0 + 2].q, * q3 = a[t0 + 3].q;
    __m256 v0 = _mm256_setzero_ps(), v1 = v0, v2 = v0, v3 = v0;
    for (int k = 0; k < nsb; k++) {
        __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
        for (int g = k * SML_GROUPS_PER_SCALE; g < (k + 1) * SML_GROUPS_PER_SCALE; g++) {
            const uint8_t * u = p->U + (size_t) g * 256;
            const size_t o = (size_t) g * SML_GROUP;
            const __m256i sc = p->S[g];
#pragma GCC unroll 4
            for (int j = 0; j < 8; j += 2) {
                const __m256i ca = _mm256_load_si256((const __m256i *)(u + j * 32));
                const __m256i cb = _mm256_load_si256((const __m256i *)(u + j * 32 + 32));
                const __m256i h0 = _mm256_add_epi16(_mm256_maddubs_epi16(ca, rl_bc4(q0 + o + j * 4)),
                                                    _mm256_maddubs_epi16(cb, rl_bc4(q0 + o + j * 4 + 4)));
                const __m256i h1 = _mm256_add_epi16(_mm256_maddubs_epi16(ca, rl_bc4(q1 + o + j * 4)),
                                                    _mm256_maddubs_epi16(cb, rl_bc4(q1 + o + j * 4 + 4)));
                const __m256i h2 = _mm256_add_epi16(_mm256_maddubs_epi16(ca, rl_bc4(q2 + o + j * 4)),
                                                    _mm256_maddubs_epi16(cb, rl_bc4(q2 + o + j * 4 + 4)));
                const __m256i h3 = _mm256_add_epi16(_mm256_maddubs_epi16(ca, rl_bc4(q3 + o + j * 4)),
                                                    _mm256_maddubs_epi16(cb, rl_bc4(q3 + o + j * 4 + 4)));
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(h0, sc));
                s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(h1, sc));
                s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(h2, sc));
                s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(h3, sc));
            }
        }
        const __m256 d = p->D[k];
        const float * as = p->ASC + (size_t) t0 * nsb + k;
        v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s0), _mm256_mul_ps(d, _mm256_set1_ps(as[0])), v0);
        v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s1), _mm256_mul_ps(d, _mm256_set1_ps(as[nsb])), v1);
        v2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s2), _mm256_mul_ps(d, _mm256_set1_ps(as[2 * nsb])), v2);
        v3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s3), _mm256_mul_ps(d, _mm256_set1_ps(as[3 * nsb])), v3);
    }
    if (mins) {
        const float * sf = p->SF + (size_t) t0 * ng;
        for (int g = 0; g < ng; g++) {
            const __m256 m = p->MNF[g];
            v0 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[g]), v0);
            v1 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[ng + g]), v1);
            v2 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[2 * ng + g]), v2);
            v3 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[3 * ng + g]), v3);
        }
    }
    float * yp = y + (size_t) t0 * nrows + r0;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v0)); yp += nrows;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v1)); yp += nrows;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v2)); yp += nrows;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v3));
}

static inline __attribute__((always_inline)) void rl6_tile1(
        float * y, int nrows, int r0, const rl_pack * p, const sml_act * a, int t,
        int ng, int nsb, int mins) {
    const int8_t * q = a[t].q;
    __m256 v = _mm256_setzero_ps();
    for (int k = 0; k < nsb; k++) {
        __m256i s = _mm256_setzero_si256();
        for (int g = k * SML_GROUPS_PER_SCALE; g < (k + 1) * SML_GROUPS_PER_SCALE; g++) {
            const uint8_t * u = p->U + (size_t) g * 256;
            const size_t o = (size_t) g * SML_GROUP;
#pragma GCC unroll 4
            for (int j = 0; j < 8; j += 2) {
                const __m256i h = _mm256_add_epi16(
                    _mm256_maddubs_epi16(_mm256_load_si256((const __m256i *)(u + j * 32)), rl_bc4(q + o + j * 4)),
                    _mm256_maddubs_epi16(_mm256_load_si256((const __m256i *)(u + j * 32 + 32)), rl_bc4(q + o + j * 4 + 4)));
                s = _mm256_add_epi32(s, _mm256_madd_epi16(h, p->S[g]));
            }
        }
        v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s),
                            _mm256_mul_ps(p->D[k], _mm256_set1_ps(p->ASC[(size_t) t * nsb + k])), v);
    }
    if (mins)
        for (int g = 0; g < ng; g++)
            v = _mm256_fmadd_ps(p->MNF[g], _mm256_set1_ps(p->SF[(size_t) t * ng + g]), v);
    float * yp = y + (size_t) t * nrows + r0;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v));
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("schedule-insns", "sched-pressure")))
#endif
static int sml_matvec_batch_flat6_rl(float * y, const uint8_t * lo4, const uint8_t * hi2,
                                     const uint16_t * sd, const uint8_t * sq,
                                     const uint16_t * md, const uint8_t * mq,
                                     const sml_act * a, int nb, int nrows, int row0, int row1, int n) {
    const int ng = n / SML_GROUP, nsb = SML_NSCALE(n), mins = mq != NULL;
    rl_pack p;
    if (!rl_setup(&p, a, nb, n, mins)) return row0;
    int r = row0;
    for (; r + 8 <= row1; r += 8) {
        rl_repack6(&p, lo4, hi2, sd, sq, md, mq, r, n);
        int t = 0;
        for (; t + 4 <= nb; t += 4) rl6_tile4(y, nrows, r, &p, a, t, ng, nsb, mins);
        for (; t < nb; t++)         rl6_tile1(y, nrows, r, &p, a, t, ng, nsb, mins);
    }
    return r;
}

// FLAT 6-bit batched, per-256 activations (sml_quantize_act_sb). Rows the
// rows-in-lanes tile cannot take (a tail under 8, a batch under SML_RL) run the
// two-pass form. Returns nothing; y accumulates as every batch kernel does.
void sml_matvec_batch_flat6_sb(float * y, const uint8_t * lo4, const uint8_t * hi2,
                               const uint16_t * sd, const uint8_t * sq,
                               const uint16_t * md, const uint8_t * mq,
                               const sml_act * a, int nb, int nrows, int row0, int row1, int n) {
    if (g_rl < 0) {
        const char * e = getenv("SML_RL");
        g_rl = e ? atoi(e) : 4;
    }
    if (g_rl > 0 && nb >= g_rl && n % (SML_GROUP * SML_GROUPS_PER_SCALE) == 0 && row1 - row0 >= 8)
        row0 = sml_matvec_batch_flat6_rl(y, lo4, hi2, sd, sq, md, mq, a, nb, nrows, row0, row1, n);
    if (row0 >= row1) return;
    sml_matvec_batch(y, hi2, sd, sq, md, mq, a, nb, nrows, row0, row1, n, 4, 0.0f);
    sml_matvec_batch_flat4_sb(y, lo4, sd, sq, NULL, NULL, a, nb, nrows, row0, row1, n);
}

// ---------------------------------------------------------------------------
// Rows-in-lanes, packed ONCE (Exodus 14:21 -- the road made before the crossing).
//
// Several conversations decoding together were read through the decode kernel
// slot by slot (~70 instructions per 256 weights per slot) or through the prompt
// kernel, whose per-call repack only pays for itself past ~64 slots. Here the
// repack is done once, at first use, into rl's own tile layout -- but with the
// codes kept as nibbles, so the copy is the size of the file's -- and the tile
// arithmetic is rl_tile4's / rl6_tile4's, operation for operation, so results
// are those of the prompt kernel bit for bit.
//
// Per 8-row tile and superblock (256 columns), contiguous:
//   codes: per group g, four 32-byte vectors P_t: byte 4r+i = code(r, 4t+i)
//          | code(r, 4(t+4)+i) << 4          -- rl's U vectors j=t and j=t+4
//   6-bit: per group, two 32-byte vectors E_u: byte 4r+i holds the 2-bit extra
//          of vector j=4u+m at bits 2m
//   fields: sd[8] fp16, md[8] fp16, sq[8 groups][8 rows], mq[8][8]
#define RLP_CODES 1024
#define RLP_EXTRA 512
// 6-bit tiles hold each code WHOLE, one byte, vector j of group g at g*256 + j*32
// (2026-09-28). The nibble + 2-bit-plane form cost ~7 instructions per vector to
// reassemble, and at four conversations the kernel is short of instructions, not
// of bandwidth: the 6-bit head ran at 84-87 GMAC/s against 133-152 for 4-bit. The
// copy exists only for batched serving; the codes, and so the sums, are the same.
#define RLP_CODES8 2048
#define RLP_FIELDS 160

int sml_rlp_ok(int rows, int n, int nbits, int has_min) {
    return !g_scale6 && has_min && (nbits == 4 || nbits == 6)
        && rows % 8 == 0 && n % SML_SCALE_SUPER == 0;
}

static size_t rlp_block(int nbits) {
    return (size_t) (nbits == 6 ? RLP_CODES8 : RLP_CODES) + RLP_FIELDS;
}

size_t sml_rlp_bytes(int rows, int n, int nbits) {
    return (size_t) (rows / 8) * (size_t) (n / SML_SCALE_SUPER) * rlp_block(nbits);
}

void sml_rlp_pack(uint8_t * dst, const uint8_t * lo4, const uint8_t * hi2,
                  const uint16_t * sd, const uint8_t * sq,
                  const uint16_t * md, const uint8_t * mq,
                  int nbits, int tile0, int tile1, int n) {
    const int ng = n / SML_GROUP, nsb = SML_NSCALE(n);
    const size_t blk = rlp_block(nbits), rb4 = (size_t) n / 2, rb2 = (size_t) n / 4;
    for (int T = tile0; T < tile1; T++)
        for (int k = 0; k < nsb; k++) {
            uint8_t * b = dst + ((size_t) T * nsb + k) * blk;
            memset(b, 0, blk);
            for (int g = 0; g < 8; g++) {
                const int G = k * 8 + g;
                for (int r = 0; r < 8; r++) {
                    const size_t row = (size_t) T * 8 + r;
                    for (int col = 0; col < SML_GROUP; col++) {
                        const uint8_t nb4 = lo4[row * rb4 + (size_t) (G / 2) * 32 + col];
                        const int code = (G & 1) ? nb4 >> 4 : nb4 & 15;
                        const int j = col / 4, i = col % 4;
                        if (nbits == 6) {
                            const int ex = (hi2[row * rb2 + (size_t) (G / 4) * 32 + col] >> (2 * (G & 3))) & 3;
                            b[(size_t) g * 256 + (size_t) j * 32 + 4 * r + i] = (uint8_t) (code | (ex << 4));
                            continue;
                        }
                        uint8_t * pv = b + (size_t) g * 128 + (size_t) (j & 3) * 32 + 4 * r + i;
                        *pv |= (uint8_t) (j < 4 ? code : code << 4);
                    }
                }
            }
            uint8_t * f = b + (nbits == 6 ? RLP_CODES8 : RLP_CODES);
            for (int r = 0; r < 8; r++) {
                const size_t row = (size_t) T * 8 + r;
                memcpy(f + 2 * r,      &sd[row * nsb + k], 2);
                memcpy(f + 16 + 2 * r, &md[row * nsb + k], 2);
                for (int g = 0; g < 8; g++) {
                    f[32 + g * 8 + r] = sq[row * ng + k * 8 + g];
                    f[96 + g * 8 + r] = mq[row * ng + k * 8 + g];
                }
            }
        }
}

// rl vector j of group g from the packed block (4-bit, or 6-bit with extra)
static inline __m256i rlp_vec(const uint8_t * b, int g, int j, int six, __m256i m15, __m256i m3) {
    (void) m3;
    if (six)   // the whole code, one load
        return _mm256_loadu_si256((const __m256i *)(b + (size_t) g * 256 + (size_t) j * 32));
    const __m256i pv = _mm256_loadu_si256((const __m256i *)(b + (size_t) g * 128 + (size_t) (j & 3) * 32));
    return j < 4 ? _mm256_and_si256(pv, m15) : _mm256_and_si256(_mm256_srli_epi16(pv, 4), m15);
}

// Four slots of one tile, from fields already expanded as rl_repack expands them:
// rl_tile4 (and rl6_tile4 for 6-bit) exactly, each U vector unpacked from the
// nibble copy. Named accumulators, as rl_tile4 found: arrays make GCC spill
// every product to the stack (the first version of this ran at ~12 GB/s).
static inline __attribute__((always_inline)) void rlp_slots(
        float * y, int nrows, int r0, const uint8_t * tb, size_t blk, int six,
        const __m256i * S, const __m256 * D, const float * mnf, const float * sf,
        const sml_act * a, int t0, int ng, int nsb, const int nq) {
    const __m256i m15 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(0x03);
    const int8_t * q0 = a[t0].q;
    const int8_t * q1 = nq > 1 ? a[t0 + 1].q : q0, * q2 = nq > 2 ? a[t0 + 2].q : q0,
                 * q3 = nq > 3 ? a[t0 + 3].q : q0;
    __m256 v0 = _mm256_setzero_ps(), v1 = v0, v2 = v0, v3 = v0;
    for (int k = 0; k < nsb; k++) {
        const uint8_t * b = tb + (size_t) k * blk;
        __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
        for (int g = 0; g < 8; g++) {
            const __m256i sc = S[k * 8 + g];
            const size_t o = (size_t) (k * 8 + g) * SML_GROUP;
            if (!six) {
                __m256i h0 = _mm256_setzero_si256(), h1 = h0, h2 = h0, h3 = h0;
#pragma GCC unroll 8
                for (int j = 0; j < 8; j++) {
                    const __m256i c = rlp_vec(b, g, j, 0, m15, m3);
                    h0 = _mm256_add_epi16(h0, _mm256_maddubs_epi16(c, rl_bc4(q0 + o + j * 4)));
                    if (nq > 1) h1 = _mm256_add_epi16(h1, _mm256_maddubs_epi16(c, rl_bc4(q1 + o + j * 4)));
                    if (nq > 2) h2 = _mm256_add_epi16(h2, _mm256_maddubs_epi16(c, rl_bc4(q2 + o + j * 4)));
                    if (nq > 3) h3 = _mm256_add_epi16(h3, _mm256_maddubs_epi16(c, rl_bc4(q3 + o + j * 4)));
                }
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(h0, sc));
                if (nq > 1) s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(h1, sc));
                if (nq > 2) s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(h2, sc));
                if (nq > 3) s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(h3, sc));
            } else {
#pragma GCC unroll 4
                for (int j = 0; j < 8; j += 2) {
                    const __m256i ca = rlp_vec(b, g, j, 1, m15, m3);
                    const __m256i cb = rlp_vec(b, g, j + 1, 1, m15, m3);
                    const __m256i h0 = _mm256_add_epi16(_mm256_maddubs_epi16(ca, rl_bc4(q0 + o + j * 4)),
                                                        _mm256_maddubs_epi16(cb, rl_bc4(q0 + o + j * 4 + 4)));
                    const __m256i h1 = nq > 1 ? _mm256_add_epi16(_mm256_maddubs_epi16(ca, rl_bc4(q1 + o + j * 4)),
                                                        _mm256_maddubs_epi16(cb, rl_bc4(q1 + o + j * 4 + 4))) : ca;
                    const __m256i h2 = nq > 2 ? _mm256_add_epi16(_mm256_maddubs_epi16(ca, rl_bc4(q2 + o + j * 4)),
                                                        _mm256_maddubs_epi16(cb, rl_bc4(q2 + o + j * 4 + 4))) : ca;
                    const __m256i h3 = nq > 3 ? _mm256_add_epi16(_mm256_maddubs_epi16(ca, rl_bc4(q3 + o + j * 4)),
                                                        _mm256_maddubs_epi16(cb, rl_bc4(q3 + o + j * 4 + 4))) : ca;
                    s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(h0, sc));
                    if (nq > 1) s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(h1, sc));
                    if (nq > 2) s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(h2, sc));
                    if (nq > 3) s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(h3, sc));
                }
            }
        }
        v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s0), _mm256_mul_ps(D[k], _mm256_set1_ps(a[t0].scale[k * SML_GROUPS_PER_SCALE])), v0);
        if (nq > 1) v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s1), _mm256_mul_ps(D[k], _mm256_set1_ps(a[t0 + 1].scale[k * SML_GROUPS_PER_SCALE])), v1);
        if (nq > 2) v2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s2), _mm256_mul_ps(D[k], _mm256_set1_ps(a[t0 + 2].scale[k * SML_GROUPS_PER_SCALE])), v2);
        if (nq > 3) v3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s3), _mm256_mul_ps(D[k], _mm256_set1_ps(a[t0 + 3].scale[k * SML_GROUPS_PER_SCALE])), v3);
    }
    for (int g = 0; g < ng; g++) {
        const __m256 m = _mm256_loadu_ps(mnf + (size_t) g * 8);
        v0 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[(size_t) t0 * ng + g]), v0);
        if (nq > 1) v1 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[(size_t) (t0 + 1) * ng + g]), v1);
        if (nq > 2) v2 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[(size_t) (t0 + 2) * ng + g]), v2);
        if (nq > 3) v3 = _mm256_fmadd_ps(m, _mm256_set1_ps(sf[(size_t) (t0 + 3) * ng + g]), v3);
    }
    float * yp = y + (size_t) t0 * nrows + r0;
    _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v0));
    if (nq > 1) { yp += nrows; _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v1)); }
    if (nq > 2) { yp += nrows; _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v2)); }
    if (nq > 3) { yp += nrows; _mm256_storeu_ps(yp, _mm256_add_ps(_mm256_loadu_ps(yp), v3)); }
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("schedule-insns", "sched-pressure")))
#endif
static void rlp_tile(float * y, int nrows, int T, const uint8_t * rlp, int six,
                     const sml_act * a, int nb, int n, float * mnf, __m256i * S, __m256 * D,
                     const float * sf) {
    const int ng = n / SML_GROUP, nsb = SML_NSCALE(n);
    const size_t blk = rlp_block(six ? 6 : 4), tbase = (size_t) T * nsb * blk;
    const __m256 bias = _mm256_set1_ps(sml_minbias());
    // the fields, expanded as rl_repack expands them, eight rows per instruction
    for (int k = 0; k < nsb; k++) {
        const uint8_t * f = rlp + tbase + (size_t) k * blk + (six ? RLP_CODES8 : RLP_CODES);
        D[k] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) f));
        const __m256 mdv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(f + 16)));
        for (int g = 0; g < 8; g++) {
            const __m128i x16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(f + 32 + g * 8)));
            S[k * 8 + g] = _mm256_set_m128i(_mm_unpackhi_epi16(x16, x16), _mm_unpacklo_epi16(x16, x16));
            const __m256 mq = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(f + 96 + g * 8))));
            _mm256_storeu_ps(mnf + (size_t) (k * 8 + g) * 8, _mm256_mul_ps(mdv, _mm256_sub_ps(mq, bias)));
        }
    }
    int t = 0;
    // quads, then the rest of the slots together: 1, 2 or 3 share one unpack
#define RLP_RUN(SIX) do {                                                                          \
        for (; t + 4 <= nb; t += 4) rlp_slots(y, nrows, T * 8, rlp + tbase, blk, SIX, S, D, mnf, sf, a, t, ng, nsb, 4); \
        switch (nb - t) {                                                                          \
        case 3: rlp_slots(y, nrows, T * 8, rlp + tbase, blk, SIX, S, D, mnf, sf, a, t, ng, nsb, 3); break; \
        case 2: rlp_slots(y, nrows, T * 8, rlp + tbase, blk, SIX, S, D, mnf, sf, a, t, ng, nsb, 2); break; \
        case 1: rlp_slots(y, nrows, T * 8, rlp + tbase, blk, SIX, S, D, mnf, sf, a, t, ng, nsb, 1); break; \
        default: break;                                                                            \
        }                                                                                          \
    } while (0)
    if (six) RLP_RUN(1); else RLP_RUN(0);
#undef RLP_RUN
}

static _Thread_local uint8_t * t_rlp_buf = NULL;
static _Thread_local size_t    t_rlp_cap = 0;

void sml_rlp_matvec(float * y, int nrows, const uint8_t * rlp, int nbits,
                    const sml_act * a, int nb, int row0, int row1, int n) {
    const int ng = n / SML_GROUP, nsb = SML_NSCALE(n);
    const size_t need = (size_t) ng * 32 + (size_t) nsb * 32 + (size_t) ng * 32
                      + (size_t) nb * ng * 4 + 64;
    if (need > t_rlp_cap) {
        _mm_free(t_rlp_buf);
        t_rlp_buf = (uint8_t *) _mm_malloc(need, 64);
        t_rlp_cap = t_rlp_buf ? need : 0;
        if (!t_rlp_buf) return;
    }
    __m256i * S = (__m256i *) t_rlp_buf;
    __m256  * D = (__m256 *) (t_rlp_buf + (size_t) ng * 32);
    float   * mnf = (float *) (t_rlp_buf + (size_t) ng * 32 + (size_t) nsb * 32);
    float   * sf = mnf + (size_t) ng * 8;
    for (int t = 0; t < nb; t++)          // rl_setup's SF, once per call
        for (int g = 0; g < ng; g++)
            sf[(size_t) t * ng + g] = a[t].scale[g] * (float) a[t].sums[g];
    for (int T = row0 / 8; T < row1 / 8; T++)
        rlp_tile(y, nrows, T, rlp, nbits == 6, a, nb, n, mnf, S, D, sf);
}

// `y` is laid out [batch][row] over the FULL row count, not the slice, so
// threads can each take a row range and write into the same buffer.
void sml_matvec_batch(float * y, const uint8_t * codes, const uint16_t * sd,
                      const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                      const sml_act * a, int nb,
                      int nrows, int row0, int row1, int n, int shift, float bias) {
    const int npack  = n / SML_PACK_SUPER;
    const int ng     = n / SML_GROUP;
    const int nscale = SML_NSCALE(n);
    const float mul  = (float)(1 << shift);
    const __m256i m3  = _mm256_set1_epi8(0x03);
    const __m256i one = _mm256_set1_epi16(1);

    for (int r = row0; r < row1; r++) {
        const uint8_t  * cp = codes + (size_t) r * npack * SML_PACK_BYTES;
        const uint16_t * dp = sd    + (size_t) r * nscale;
        const uint8_t  * qp = sq    + (size_t) r * sml_grow(ng);
        const uint16_t * mdp = md ? md + (size_t) r * nscale : NULL;
        const uint8_t  * mqp = mq ? mq + (size_t) r * sml_grow(ng) : NULL;
        __m128 vsum[SML_KERNEL_MAX_BATCH];
        for (int b = 0; b < nb; b++) vsum[b] = _mm_setzero_ps();

        for (int p = 0; p < npack; p++) {
            // the weight load happens ONCE and serves every batch element --
            // that is the entire economics of this function
            const __m256i bcode = _mm256_loadu_si256((const __m256i *)(cp));
            cp += SML_PACK_BYTES;
            const int gbase = p * SML_GROUPS_PER_PACK;

            __m256i cg[SML_GROUPS_PER_PACK];
            for (int g = 0; g < SML_GROUPS_PER_PACK; g++)
                cg[g] = _mm256_and_si256(_mm256_srli_epi16(bcode, 2 * g), m3);

            const float dsc = sml_f16(dp[gbase / SML_GROUPS_PER_SCALE]);
            const __m128 qf = sml_qf4(qp, gbase);
            const __m128 wsc = _mm_mul_ps(_mm_set1_ps(dsc), qf);
            __m128 wmin = _mm_setzero_ps();
            if (mq) {
                const float mdsc = sml_f16(mdp[gbase / SML_GROUPS_PER_SCALE]);
                wmin = _mm_mul_ps(_mm_set1_ps(mdsc),
                    _mm_sub_ps(sml_qf4(mqp, gbase),
                        _mm_set1_ps(sml_minbias())));
            }

            for (int b = 0; b < nb; b++) {
                const sml_act * ab = &a[b];
                const int8_t * xq = ab->q + (size_t) gbase * SML_GROUP;
                __m256i acc[SML_GROUPS_PER_PACK];
                for (int g = 0; g < SML_GROUPS_PER_PACK; g++) {
                    const __m256i x = _mm256_loadu_si256(
                        (const __m256i *)(xq + (size_t) g * SML_GROUP));
                    acc[g] = _mm256_madd_epi16(_mm256_maddubs_epi16(cg[g], x), one);
                }
                const __m128 df = _mm_cvtepi32_ps(
                    sml_hsum4_i32(acc[0], acc[1], acc[2], acc[3]));
                const __m128 af = _mm_loadu_ps(ab->scale + gbase);
                const __m128 sf = _mm_cvtepi32_ps(_mm_loadu_si128(
                    (const __m128i *)(ab->sums + gbase)));
                const __m128 val = _mm_add_ps(_mm_mul_ps(df, _mm_set1_ps(mul)),
                                              _mm_mul_ps(sf, _mm_set1_ps(bias)));
                vsum[b] = _mm_add_ps(vsum[b], _mm_mul_ps(wsc, _mm_mul_ps(af, val)));
                if (mq)
                    vsum[b] = _mm_add_ps(vsum[b],
                        _mm_mul_ps(wmin, _mm_mul_ps(af, sf)));
            }
        }
        for (int b = 0; b < nb; b++) {
            __m128 v = _mm_hadd_ps(vsum[b], vsum[b]);
            v = _mm_hadd_ps(v, v);
            y[(size_t) b * (size_t) nrows + r] += _mm_cvtss_f32(v);
        }
    }
}

#else  // portable reference
//
// This arm had not compiled since the per-group min landed: it called
// sml_matvec_rows with ten arguments where twelve were expected, and
// accumulated the min into an undeclared `acc`. Nothing caught it because the
// only machine this was ever built on has AVX2, so the branch was never
// selected. Anyone without AVX2 -- Apple Silicon included -- got a wall of
// compiler errors rather than a slow engine.

void sml_matvec_batch(float * y, const uint8_t * codes, const uint16_t * sd,
                      const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                      const sml_act * a, int nb,
                      int nrows, int row0, int row1, int n, int shift, float bias) {
    for (int b = 0; b < nb; b++)
        sml_matvec_rows(y + (size_t) b * nrows, codes, sd, sq, md, mq, &a[b],
                        row0, row1, n, shift, bias);
}

void sml_matvec_rows(float * y, const uint8_t * codes, const uint16_t * sd,
                     const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                     const sml_act * a,
                     int row0, int row1, int n, int shift, float bias) {
    const int npack  = n / SML_PACK_SUPER;
    const int ng     = n / SML_GROUP;
    const int nscale = SML_NSCALE(n);
    const float mul  = (float)(1 << shift);

    for (int r = row0; r < row1; r++) {
        const uint8_t  * cp = codes + (size_t) r * npack * SML_PACK_BYTES;
        const uint16_t * dp = sd    + (size_t) r * nscale;
        const uint8_t  * qp = sq    + (size_t) r * sml_grow(ng);
        float sum = 0.0f;

        for (int p = 0; p < npack; p++) {
            const uint8_t * blk = cp + (size_t) p * SML_PACK_BYTES;
            for (int g = 0; g < SML_GROUPS_PER_PACK; g++) {
                const int gi = p * SML_GROUPS_PER_PACK + g;
                const int8_t * xq = a->q + (size_t) gi * SML_GROUP;
                int32_t d = 0;
                for (int i = 0; i < SML_GROUP; i++) {
                    d += (int32_t)((blk[i] >> (2 * g)) & 0x3) * (int32_t) xq[i];
                }
                const float ws = sml_f16(dp[gi / SML_GROUPS_PER_SCALE]) * (float) sml_gfield(qp, gi, g_scale6);
                sum += ws * a->scale[gi]
                     * ((float) d * mul + bias * (float) a->sums[gi]);
                if (mq) {
                    const float mn = sml_f16(md[(size_t) r * nscale
                                                + gi / SML_GROUPS_PER_SCALE])
                                   * ((float) sml_gfield(mq + (size_t) r * sml_grow(ng), gi, g_scale6) - sml_minbias());
                    sum += mn * a->scale[gi] * (float) a->sums[gi];
                }
            }
        }
        y[r] += sum;
    }
}


// Scalar flat 4-bit, mirroring sml_matvec_rows_flat4 exactly.
//
// Nibble layout, which must match the AVX2 arm byte for byte: a quad is four
// groups in 64 bytes. The first 32 bytes hold groups 4p and 4p+1 as low and
// high nibbles; the second 32 hold 4p+2 and 4p+3 the same way. Reading it any
// other way still produces numbers, which is how this class of bug survives.
//
// Per group:  y += scale_w * act_scale * dot(code, xq)
//           + min_w   * act_scale * sum(xq)
// exactly as the vector arm accumulates, and as sml_dequant_row_flat4
// reconstructs a weight.
// The 1-bit plane without AVX2: the same sums, one weight at a time.
void sml_matvec_bits1(float * y, const uint8_t * plane, const uint16_t * sd, const uint8_t * sq,
                      const sml_act * a, int nb, int nrows, int row0, int row1, int n, int shift) {
    const int ng = n / SML_GROUP, nscale = SML_NSCALE(n);
    for (int r = row0; r < row1; r++)
        for (int b = 0; b < nb; b++) {
            float tot = 0.0f;
            for (int g = 0; g < ng; g++) {
                const uint8_t * pb = plane + ((size_t) r * ng + g) * 4;
                int dot = 0;
                for (int j = 0; j < SML_GROUP; j++)
                    if ((pb[j >> 3] >> (j & 7)) & 1) dot += a[b].q[g * SML_GROUP + j];
                const float step = sml_f16(sd[(size_t) r * nscale + g / SML_GROUPS_PER_SCALE])
                                 * (float) sml_gfield(sq + (size_t) r * sml_grow(ng), g, g_scale6);
                tot += step * a[b].scale[g] * (float) dot;
            }
            y[(size_t) b * (size_t) nrows + r] += tot * (float) (1 << shift);
        }
}

// Q8 without AVX2: each row dequantized, then dotted with the activation as the
// engine quantized it -- the same values the vector kernel sums, more slowly.
void sml_matvec_rows_flat8(float * y, const uint8_t * lo, const uint8_t * hi, const uint16_t * sd,
                           const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                           const sml_act * a, int row0, int row1, int n) {
    float * w = (float *) malloc((size_t) n * sizeof(float));
    for (int r = row0; r < row1; r++) {
        sml_dequant_row_flat8(w, lo, hi, sd, sq, md, mq, r, n);
        double s = 0.0;
        for (int i = 0; i < n; i++) s += (double) w[i] * a->q[i] * a->scale[i / SML_GROUP];
        y[r] += (float) s;
    }
    free(w);
}

void sml_matvec_batch_flat8(float * y, const uint8_t * lo, const uint8_t * hi, const uint16_t * sd,
                            const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                            const sml_act * a, int nb, int nrows, int row0, int row1, int n) {
    for (int b = 0; b < nb; b++)
        sml_matvec_rows_flat8(y + (size_t) b * nrows, lo, hi, sd, sq, md, mq, &a[b], row0, row1, n);
}

void sml_matvec_rows_flat4(float * y, const uint8_t * codes, const uint16_t * sd,
                           const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                           const sml_act * a,
                           int row0, int row1, int n) {
    const int ng     = n / SML_GROUP;
    const int nscale = SML_NSCALE(n);
    const size_t rowbytes = (size_t) n / 2;

    for (int r = row0; r < row1; r++) {
        const uint8_t  * cp  = codes + (size_t) r * rowbytes;
        const uint16_t * dp  = sd    + (size_t) r * nscale;
        const uint8_t  * qp  = sq    + (size_t) r * sml_grow(ng);
        const uint16_t * mdp = md ? md + (size_t) r * nscale : NULL;
        const uint8_t  * mqp = mq ? mq + (size_t) r * sml_grow(ng) : NULL;
        float sum = 0.0f;

        for (int g = 0; g < ng; g++) {
            const int quad = g / 4, kk = g % 4;
            const uint8_t * blk = cp + (size_t) quad * 64 + (kk >= 2 ? 32 : 0);
            const int hi = kk & 1;
            const int8_t * xq = a->q + (size_t) g * SML_GROUP;

            int32_t d = 0;
            for (int j = 0; j < SML_GROUP; j++) {
                const uint8_t byte = blk[j];
                const int code = hi ? (byte >> 4) & 0x0F : byte & 0x0F;
                d += (int32_t) code * (int32_t) xq[j];
            }
            const int si = g / SML_GROUPS_PER_SCALE;
            const float ws = sml_f16(dp[si]) * (float) sml_gfield(qp, g, g_scale6);
            sum += ws * a->scale[g] * (float) d;
            if (mqp) {
                const float mn = sml_f16(mdp[si])
                               * ((float) sml_gfield(mqp, g, g_scale6) - sml_minbias());
                sum += mn * a->scale[g] * (float) a->sums[g];
            }
        }
        y[r] += sum;
    }
}

void sml_matvec_rows_flat6(float * y, const uint8_t * lo4, const uint8_t * hi2,
                           const uint16_t * sd, const uint8_t * sq,
                           const uint16_t * md, const uint8_t * mq,
                           const sml_act * a, int row0, int row1, int n) {
    sml_matvec_rows(y, hi2, sd, sq, md, mq, a, row0, row1, n, 4, 0.0f);
    sml_matvec_rows_flat4(y, lo4, sd, sq, NULL, NULL, a, row0, row1, n);
}

// No vector kernels for the interleaved layout here, so decode never repacks.
int sml_il_ok(int n, int nbits, int has_min) { return 0; }
size_t sml_il_row_bytes(int n, int nbits) { return 0; }
void sml_il_pack(uint8_t * dst, const uint8_t * lo4, const uint8_t * hi2,
                 const uint16_t * sd, const uint8_t * sq,
                 const uint16_t * md, const uint8_t * mq,
                 int nbits, int row0, int row1, int n) {}
void sml_matvec_rows_il4(float * y, const uint8_t * il, const sml_act * a,
                         int row0, int row1, int n) {}
void sml_matvec_rows_il6(float * y, const uint8_t * il, const sml_act * a,
                         int row0, int row1, int n) {}

void sml_matvec_batch_flat6_sb(float * y, const uint8_t * lo4, const uint8_t * hi2,
                               const uint16_t * sd, const uint8_t * sq,
                               const uint16_t * md, const uint8_t * mq,
                               const sml_act * a, int nb, int nrows, int row0, int row1, int n) {
    sml_matvec_batch(y, hi2, sd, sq, md, mq, a, nb, nrows, row0, row1, n, 4, 0.0f);
    sml_matvec_batch_flat4_sb(y, lo4, sd, sq, NULL, NULL, a, nb, nrows, row0, row1, n);
}

int sml_rlp_ok(int rows, int n, int nbits, int has_min) { return 0; }
size_t sml_rlp_bytes(int rows, int n, int nbits) { return 0; }
void sml_rlp_pack(uint8_t * dst, const uint8_t * lo4, const uint8_t * hi2,
                  const uint16_t * sd, const uint8_t * sq,
                  const uint16_t * md, const uint8_t * mq,
                  int nbits, int tile0, int tile1, int n) {}
void sml_rlp_matvec(float * y, int nrows, const uint8_t * rlp, int nbits,
                    const sml_act * a, int nb, int row0, int row1, int n) {}

// Without AVX2 the per-group kernel computes the same thing: the per-256
// activation scale is written into every group of its superblock.
void sml_matvec_batch_flat4_sb(float * y, const uint8_t * codes, const uint16_t * sd,
                               const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                               const sml_act * a, int nb,
                               int nrows, int row0, int row1, int n) {
    sml_matvec_batch_flat4(y, codes, sd, sq, md, mq, a, nb, nrows, row0, row1, n);
}

void sml_matvec_batch_flat4(float * y, const uint8_t * codes, const uint16_t * sd,
                            const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                            const sml_act * a, int nb,
                            int nrows, int row0, int row1, int n) {
    for (int b = 0; b < nb; b++)
        sml_matvec_rows_flat4(y + (size_t) b * nrows, codes, sd, sq, md, mq,
                              &a[b], row0, row1, n);
}

#endif

// ---------------------------------------------------------------------------
// Portable on every target: these carry no intrinsics and are defined ONCE,
// outside the AVX2/scalar split. They used to sit inside the AVX2 arm, which
// meant a machine without AVX2 failed to LINK them -- and since flat is the
// shipping layout, the engine did not build at all off x86-64-v3.
// ---------------------------------------------------------------------------

// One row of a FLAT tensor, dequantized whole. Same nibble layout as the matvec
// below; written as a plain loop because it runs once per token, not per weight.
void sml_dequant_row_flat4(float * out, const uint8_t * codes,
                           const uint16_t * sd, const uint8_t * sq,
                           const uint16_t * md, const uint8_t * mq,
                           int row, int n) {
    const int ng     = n / SML_GROUP;
    const int nscale = SML_NSCALE(n);
    const size_t rowbytes = (size_t) n / 2;
    const uint8_t * cp = codes + (size_t) row * rowbytes;
    for (int g = 0; g < ng; g++) {
        const int quad = g / 4, k = g % 4;
        const uint8_t * blk = cp + (size_t) quad * 64 + (k >= 2 ? 32 : 0);
        const int hi = k & 1;
        const int si = (size_t) row * nscale + g / SML_GROUPS_PER_SCALE;
        const float step = sml_f16(sd[si]) * (float) sml_gfield(sq + (size_t) row * sml_grow(ng), g, g_scale6);
        const float mn = mq ? sml_f16(md[si])
                              * ((float) sml_gfield(mq + (size_t) row * sml_grow(ng), g, g_scale6) - sml_minbias())
                            : 0.0f;
        for (int j = 0; j < SML_GROUP; j++) {
            const uint8_t byte = blk[j];
            const int code = hi ? (byte >> 4) & 0x0F : byte & 0x0F;
            out[g * SML_GROUP + j] = (float) code * step + mn;
        }
    }
}

// A 1-bit plane's share of a row (Q3, Q5, Q7): out += step * 2^shift * bit.
void sml_dequant_bits1_add(float * out, const uint8_t * plane, const uint16_t * sd, const uint8_t * sq,
                           int row, int n, int shift) {
    const int ng = n / SML_GROUP, nscale = SML_NSCALE(n);
    for (int g = 0; g < ng; g++) {
        const uint8_t * pb = plane + ((size_t) row * ng + g) * 4;
        const float step = sml_f16(sd[(size_t) row * nscale + g / SML_GROUPS_PER_SCALE])
                         * (float) sml_gfield(sq + (size_t) row * sml_grow(ng), g, g_scale6) * (float) (1 << shift);
        for (int j = 0; j < SML_GROUP; j++)
            if ((pb[j >> 3] >> (j & 7)) & 1) out[g * SML_GROUP + j] += step;
    }
}

// One row of an 8-bit FLAT tensor (Q8): the low and high nibble planes, each in
// the 4-bit layout, give code = lo + 16*hi.
void sml_dequant_row_flat8(float * out, const uint8_t * lo, const uint8_t * hi,
                           const uint16_t * sd, const uint8_t * sq,
                           const uint16_t * md, const uint8_t * mq,
                           int row, int n) {
    const int ng     = n / SML_GROUP;
    const int nscale = SML_NSCALE(n);
    const size_t rowbytes = (size_t) n / 2;
    const uint8_t * lp = lo + (size_t) row * rowbytes;
    const uint8_t * hp = hi + (size_t) row * rowbytes;
    for (int g = 0; g < ng; g++) {
        const int quad = g / 4, k = g % 4;
        const size_t at = (size_t) quad * 64 + (k >= 2 ? 32 : 0);
        const int top = k & 1;
        const int si = (size_t) row * nscale + g / SML_GROUPS_PER_SCALE;
        const float step = sml_f16(sd[si]) * (float) sml_gfield(sq + (size_t) row * sml_grow(ng), g, g_scale6);
        const float mn = mq ? sml_f16(md[si])
                              * ((float) sml_gfield(mq + (size_t) row * sml_grow(ng), g, g_scale6) - sml_minbias())
                            : 0.0f;
        for (int j = 0; j < SML_GROUP; j++) {
            const int l = top ? (lp[at + j] >> 4) & 0x0F : lp[at + j] & 0x0F;
            const int h = top ? (hp[at + j] >> 4) & 0x0F : hp[at + j] & 0x0F;
            out[g * SML_GROUP + j] = (float) (l + 16 * h) * step + mn;
        }
    }
}

// Blockwise Hadamard-32 on the activation, in place of a 32x32 matmul.
//
// The weights are quantized in the rotated basis, so the activation must be
// rotated to match: H is orthogonal and symmetric, so (W H)(H x) = W x exactly.
// Blocks of 32 are chosen to match SML_GROUP, so the rotation never crosses a
// scale boundary.
//
// Five butterfly passes -- 5 add/sub per element rather than the 32 multiplies
// a dense 32x32 would cost. Measured worth 1.103x end to end at ZERO bytes,
// after being rejected on a 1.06x per-tensor number.
void sml_hadamard32(float * out, const float * in, int n) {
    const float inv = 0.17677669529663687f;      // 1/sqrt(32)
    for (int b = 0; b + 32 <= n; b += 32) {
        float t[32];
        for (int i = 0; i < 32; i++) t[i] = in[b + i];
        for (int s = 1; s < 32; s <<= 1) {
            for (int i = 0; i < 32; i++) {
                if (i & s) continue;
                const float a = t[i], c = t[i + s];
                t[i] = a + c;
                t[i + s] = a - c;
            }
        }
        for (int i = 0; i < 32; i++) out[b + i] = t[i] * inv;
    }
    for (int i = (n / 32) * 32; i < n; i++) out[i] = in[i];   // ragged tail
}

// One row of a 6-bit FLAT tensor. code = base(nibble) + 16*extra(2 bits), with
// `extra` in the group-planar layout where bit-plane g IS group g.
//
// The embedding lookup is the only place a tensor is read as a ROW, and pinning
// token_embd to 4 bits to avoid writing this was costing real accuracy: on a
// tied model that tensor IS the output head, so it produces the logits being
// graded. The reference format stores it at 6 bits.
void sml_dequant_row_flat6(float * out, const uint8_t * lo4, const uint8_t * hi2,
                           const uint16_t * sd, const uint8_t * sq,
                           const uint16_t * md, const uint8_t * mq,
                           int row, int n) {
    const int ng     = n / SML_GROUP;
    const int nscale = SML_NSCALE(n);
    const uint8_t * cp4 = lo4 + (size_t) row * ((size_t) n / 2);
    const uint8_t * cp2 = hi2 + (size_t) row * ((size_t) n / 4);
    for (int g = 0; g < ng; g++) {
        const int quad = g / 4, k = g % 4;
        const uint8_t * blk = cp4 + (size_t) quad * 64 + (k >= 2 ? 32 : 0);
        const int hi = k & 1;
        const uint8_t * pk = cp2 + (size_t)(g / SML_GROUPS_PER_PACK) * SML_PACK_BYTES;
        const int gi = g % SML_GROUPS_PER_PACK;
        const int si = (size_t) row * nscale + g / SML_GROUPS_PER_SCALE;
        const float step = sml_f16(sd[si]) * (float) sml_gfield(sq + (size_t) row * sml_grow(ng), g, g_scale6);
        const float mn = mq ? sml_f16(md[si])
                              * ((float) sml_gfield(mq + (size_t) row * sml_grow(ng), g, g_scale6) - sml_minbias())
                            : 0.0f;
        for (int j = 0; j < SML_GROUP; j++) {
            const int base  = hi ? (blk[j] >> 4) & 0x0F : blk[j] & 0x0F;
            const int extra = (pk[j] >> (2 * gi)) & 0x03;
            out[g * SML_GROUP + j] = (float)(base | (extra << 4)) * step + mn;
        }
    }
}
