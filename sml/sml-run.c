#include "sml-run.h"
#include "sml-pool.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// x86 only. On any other target -- Apple Silicon included -- this header does
// not exist, so including it unconditionally made the engine fail to compile
// before it could reach the portable paths below.
#if defined(__AVX2__) || defined(__F16C__)
#include <immintrin.h>
#endif

#if defined(_OPENMP) && defined(_WIN32)
#include <omp.h>
#include <windows.h>
// omp_get_wtime steps in whole milliseconds on this toolchain, and a matvec is
// ~0.1 ms, so every phase timer below was sampling noise. The performance
// counter resolves ~100 ns.
static double sml_now(void) {
    static double inv = 0.0;
    LARGE_INTEGER c;
    if (inv == 0.0) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); inv = 1.0 / (double) f.QuadPart; }
    QueryPerformanceCounter(&c);
    return (double) c.QuadPart * inv;
}
#elif defined(_OPENMP)
#include <omp.h>
static double sml_now(void) { return omp_get_wtime(); }
#else
#include <time.h>
static double sml_now(void) { return (double) clock() / CLOCKS_PER_SEC; }
#endif

// Phase timers. The engine achieves well under the kernel's standalone
// bandwidth, and guessing at why has already cost one wrong hypothesis
// (OpenMP fork/join, worth only 6%). So it gets measured per phase instead.
double sml_t_matvec, sml_t_prep, sml_t_rope, sml_t_attn, sml_t_norm;

// Per-shape matvec accounting. The engine sustains ~17.6 GB/s where the kernel
// alone does 24.4, and the phase timers put 95% of the time inside matvec -- so
// the gap is INSIDE the kernel calls, not around them. The suspicion is small
// matvecs: attn_k and attn_v are 256 rows, which across 6 threads is 42 rows
// each, mostly parallel overhead. This says whether that is true before anything
// is rebuilt to fix it.
#define SML_SHAPE_SLOTS 8
double   sml_shape_time[SML_SHAPE_SLOTS];
uint64_t sml_shape_bytes[SML_SHAPE_SLOTS];
uint64_t sml_shape_calls[SML_SHAPE_SLOTS];
int      sml_shape_rows[SML_SHAPE_SLOTS];

static int shape_slot(int rows) {
    for (int i = 0; i < SML_SHAPE_SLOTS; i++) {
        if (sml_shape_rows[i] == rows) return i;
        if (sml_shape_rows[i] == 0) { sml_shape_rows[i] = rows; return i; }
    }
    return SML_SHAPE_SLOTS - 1;
}

// ---------------------------------------------------------------- small ops

// fp32 <-> fp16 for the KV cache. F16C is on every AVX2 part, so this is a
// single instruction per 8 values in each direction and the halved cache is
// pure win at long context.
#if defined(__F16C__)
static inline void f32_to_f16(uint16_t * dst, const float * src, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i *)(dst + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
    for (; i < n; i++) {
        float v = src[i];
        _mm_storeu_si16(dst + i, _mm_cvtps_ph(_mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT));
    }
}
#else
// Scalar half encode, round-to-nearest-even, with subnormals and Inf/NaN. The
// KV cache is stored as fp16 on every target, so a machine without F16C needs
// this or it cannot keep a cache at all.
static inline uint16_t sml_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawe = (x >> 23) & 0xffu;
    uint32_t man = x & 0x7fffffu;
    if (rawe == 0xff) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
    int32_t exp = (int32_t) rawe - 127 + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        man |= 0x800000u;
        const uint32_t sh = (uint32_t)(14 - exp);
        uint32_t sub = man >> sh;
        const uint32_t rem = man & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (sub & 1u))) sub++;
        return (uint16_t)(sign | sub);
    }
    uint32_t h = ((uint32_t) exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)(sign | h);
}

static inline void f32_to_f16(uint16_t * dst, const float * src, int n) {
    for (int i = 0; i < n; i++) dst[i] = sml_f32_to_f16(src[i]);
}
#endif

#if defined(__AVX2__)
// Horizontal sum of eight floats. One per head per position, not per element.
static inline float sml_hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
#endif

static inline float f16_to_f32(uint16_t h) {
#if defined(__F16C__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
#else
    return sml_f16_to_f32(h);      // scalar decode already lives in the kernel
#endif
}

// LayerNorm: centres as well as scales, and carries a bias. RMSNorm is the
// same thing with the mean assumed zero and no bias, which is why one function
// with a flag covers both rather than two code paths.
static void layer_norm(float * out, const float * x, const float * w,
                       const float * b, int n, float eps) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; i++) { const double d = x[i] - mean; var += d * d; }
    const float s = 1.0f / sqrtf((float)(var / n) + eps);
    for (int i = 0; i < n; i++) {
        out[i] = ((float)(x[i] - mean)) * s * w[i] + (b ? b[i] : 0.0f);
    }
}

static void rms_norm(float * out, const float * x, const float * w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double) x[i] * x[i];
    const float s = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * s * w[i];
}

// One entry point so callers do not each have to know which family the model
// uses -- the file says, and the forward pass just normalises.
static void norm(const sml_ctx * c, float * out, const float * x,
                 const float * w, const float * b, int n) {
    if (c->model->hdr->version >= 4 && c->model->hdr->norm_type == 1)
        layer_norm(out, x, w, b, n, c->model->hdr->rms_eps);
    else
        rms_norm(out, x, w, n, c->model->hdr->rms_eps);
}

static void softmax(float * x, int n);
static void softmax_sink(float * x, int n, float sink);
static void mla_qkv(sml_ctx * c, const sml_layer * ly, const float * x, float * q, float * k, float * v, int depth);
static void mla_pack(const sml_layer * ly, float * y, int nh, int hd);

typedef struct { sml_ctx * c; int L, pos, hd, kvd, rep, swa; float inv, cap; const float * sinks; } attn_job;

// One query row against the cache: scores over [t0, pos], softmax, and the
// weighted sum of V into `out`. kc/vc point at position 0 of this layer's head,
// `stride` elements apart per position.
//
// Each position's dot is the same chain of FMAs as before, in the same order;
// four positions run side by side so the chains overlap instead of waiting on
// one another. The output accumulates in registers across positions instead of
// a load and a store per position -- again the same FMA per element, in the
// same order. So the result is bit-identical to the plain loops.
static void attn_row(const float * qh, const uint16_t * kc, const uint16_t * vc,
                     size_t stride, int t0, int pos, int hd, float inv, float cap,
                     const float * alibi_slope, const float * sink, float * a, float * out) {
    int t = t0;
#if defined(__AVX2__)
    if (hd % 8 == 0) {
        for (; t + 4 <= pos + 1; t += 4) {
            const uint16_t * k0 = kc + (size_t) t * stride;
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
            for (int i = 0; i < hd; i += 8) {
                const __m256 q = _mm256_loadu_ps(qh + i);
                a0 = _mm256_fmadd_ps(q, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k0 + i))), a0);
                a1 = _mm256_fmadd_ps(q, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k0 + stride + i))), a1);
                a2 = _mm256_fmadd_ps(q, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k0 + 2 * stride + i))), a2);
                a3 = _mm256_fmadd_ps(q, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k0 + 3 * stride + i))), a3);
            }
            a[t]     = sml_hsum256(a0) * inv;
            a[t + 1] = sml_hsum256(a1) * inv;
            a[t + 2] = sml_hsum256(a2) * inv;
            a[t + 3] = sml_hsum256(a3) * inv;
            if (alibi_slope)
                for (int j = 0; j < 4; j++) a[t + j] += *alibi_slope * (float) (t + j - pos);
            if (cap > 0.0f)
                for (int j = 0; j < 4; j++) a[t + j] = cap * tanhf(a[t + j] / cap);
        }
    }
#endif
    for (; t <= pos; t++) {
        const uint16_t * kh = kc + (size_t) t * stride;
        float s = 0.0f;
        int i = 0;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= hd; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i),
                                  _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kh + i))), acc);
        s = sml_hsum256(acc);
#endif
        for (; i < hd; i++) s += qh[i] * f16_to_f32(kh[i]);
        a[t] = s * inv;
        // ALiBi: slope * (key - query). HF adds slope * key; the difference is
        // constant per row, so softmax cannot see it.
        if (alibi_slope) a[t] += *alibi_slope * (float) (t - pos);
        // Gemma 2 bounds every attention logit: cap * tanh(s / cap)
        if (cap > 0.0f) a[t] = cap * tanhf(a[t] / cap);
    }
    if (sink) softmax_sink(a + t0, pos - t0 + 1, *sink);
    else softmax(a + t0, pos - t0 + 1);
#if defined(__AVX2__)
    if (hd % 64 == 0) {
        for (int c0 = 0; c0 < hd; c0 += 64) {
            __m256 o0 = _mm256_setzero_ps(), o1 = o0, o2 = o0, o3 = o0, o4 = o0, o5 = o0, o6 = o0, o7 = o0;
            for (int u = t0; u <= pos; u++) {
                const uint16_t * vh = vc + (size_t) u * stride + c0;
                const __m256 wv = _mm256_set1_ps(a[u]);
                o0 = _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vh))), o0);
                o1 = _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vh + 8))), o1);
                o2 = _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vh + 16))), o2);
                o3 = _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vh + 24))), o3);
                o4 = _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vh + 32))), o4);
                o5 = _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vh + 40))), o5);
                o6 = _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vh + 48))), o6);
                o7 = _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vh + 56))), o7);
            }
            float * op = out + c0;
            _mm256_storeu_ps(op,      o0); _mm256_storeu_ps(op + 8,  o1);
            _mm256_storeu_ps(op + 16, o2); _mm256_storeu_ps(op + 24, o3);
            _mm256_storeu_ps(op + 32, o4); _mm256_storeu_ps(op + 40, o5);
            _mm256_storeu_ps(op + 48, o6); _mm256_storeu_ps(op + 56, o7);
        }
        return;
    }
#endif
    memset(out, 0, (size_t) hd * sizeof(float));
    for (int u = t0; u <= pos; u++) {
        const uint16_t * vh = vc + (size_t) u * stride;
        const float w = a[u];
        int i = 0;
#if defined(__AVX2__)
        const __m256 wv = _mm256_set1_ps(w);
        for (; i + 8 <= hd; i += 8)
            _mm256_storeu_ps(out + i, _mm256_fmadd_ps(wv,
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vh + i))), _mm256_loadu_ps(out + i)));
#endif
        for (; i < hd; i++) out[i] += w * f16_to_f32(vh[i]);
    }
}

static void attn_head(void * p, int hh) {
    const attn_job * j = (const attn_job *) p;
    sml_ctx * c = j->c;
    const int hd = j->hd, kvd = j->kvd, pos = j->pos, L = j->L;
    const int kvh = hh / j->rep;
    const int t0 = (j->swa && pos >= j->swa) ? pos - j->swa + 1 : 0;
    const size_t base = (size_t) L * c->n_ctx * kvd + (size_t) kvh * hd;
    attn_row(c->q + (size_t) hh * hd, c->k_cache + base, c->v_cache + base, (size_t) kvd,
             t0, pos, hd, j->inv, j->cap, c->alibi ? &c->alibi[hh] : NULL, j->sinks ? &j->sinks[hh] : NULL,
             c->att + (size_t) hh * c->n_ctx, c->xb + (size_t) hh * hd);
}

static void softmax(float * x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

// gpt-oss: a learned "sink" logit joins the softmax and takes its share, then is dropped
static void softmax_sink(float * x, int n, float sink) {
    float mx = sink;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = expf(sink - mx);
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

// Rotate each head's vector by absolute position. Halves, matching the
// converter's reference implementation.
//
// The rotation angle depends only on (position, index), never on the data, so
// every cos/sin in the model is known before a single token is seen. Computing
// them inline cost 6% of total runtime -- 41k powf+sincos per token, all of them
// recomputed identically for 36 layers. They are tabulated at init instead:
// n_ctx * head_dim/2 * 2 floats, a quarter of a megabyte.
// Q/K norm (Qwen3, Gemma 3, OLMo 2): RMSNorm on the query and the key, BEFORE
// RoPE. The weight's LENGTH says which kind: head_dim long means one norm per
// head sharing the weight (Qwen3, Gemma 3); the whole projection long means one
// norm over the full vector (OLMo 2). Read from the tensor, not declared per
// family -- the same rule the rest of the structure follows.
// With a bias it is a LayerNorm (Persimmon's q_layernorm): mean removed, then
// scaled and shifted.
static void qk_norm(float * x, const float * w0, const float * b0, int wn, int nh, int hd, float eps, int heads) {
    const int per_head = wn == hd || heads;
    const int span = per_head ? hd : wn;
    const int reps = per_head ? nh : 1;
    for (int r = 0; r < reps; r++) {
        float * v = x + (size_t) r * span;
        // per-head rows (Chameleon): head r's own weight and bias
        const float * w = heads ? w0 + (size_t) r * hd : w0;
        const float * b = b0 && heads ? b0 + (size_t) r * hd : b0;
        if (b) {
            double mu = 0.0, var = 0.0;
            for (int i = 0; i < span; i++) mu += v[i];
            mu /= span;
            for (int i = 0; i < span; i++) var += ((double) v[i] - mu) * ((double) v[i] - mu);
            const float s = 1.0f / sqrtf((float) (var / span) + eps);
            for (int i = 0; i < span; i++) v[i] = ((float) (v[i] - mu)) * s * w[i] + b[i];
            continue;
        }
        double ss = 0.0;
        for (int i = 0; i < span; i++) ss += (double) v[i] * v[i];
        const float s = 1.0f / sqrtf((float)(ss / span) + eps);
        for (int i = 0; i < span; i++) v[i] = v[i] * s * w[i];
    }
}

static void rope_tab(const sml_ctx * c, const float * cos_t, const float * sin_t,
                     float * x, int n_head, int head_dim, int pos) {
    // GPT-NeoX/Phi/StableLM rotate only the first n_rot dimensions of each head
    // and leave the rest untouched. Rotating the whole head is silently wrong.
    // Pythia-410m rotates 16 of 64.
    const int half = c->rope_half;   // same value the table was built with
    const float * cs = cos_t + (size_t) pos * half;
    const float * sn = sin_t + (size_t) pos * half;
    const int interleaved = c->model->hdr->version >= 4 &&
                            c->model->hdr->rope_type == 1;
    for (int h = 0; h < n_head; h++) {
        float * v = x + (size_t) h * head_dim;
        if (interleaved) {
            // GPT-J pairs ADJACENT elements (0,1) (2,3) ... Llama/Qwen and
            // ALSO GPT-NeoX/Falcon/Phi pair across the halves, (i, i+half).
            // The reference engine names the halved convention "NEOX rope"
            // after where it first met it, which had this flag inverted for
            // five architectures until it was checked against their own
            // reference implementations. Same rotation, different pairing --
            // getting it wrong produces fluent nonsense, never an error.
            for (int i = 0; i < half; i++) {
                const float x0 = v[2 * i], x1 = v[2 * i + 1];
                v[2 * i]     = x0 * cs[i] - x1 * sn[i];
                v[2 * i + 1] = x0 * sn[i] + x1 * cs[i];
            }
        } else {
            for (int i = 0; i < half; i++) {
                const float x0 = v[i], x1 = v[i + half];
                v[i]        = x0 * cs[i] - x1 * sn[i];
                v[i + half] = x0 * sn[i] + x1 * cs[i];
            }
        }
    }
}

// Layer L's rotary: none on a NoPE layer (Cohere2's and EXAONE 4's global
// layers, every fourth SmolLM3 layer), the local base on a sliding layer that
// has one (Gemma 3), else the model's table.
static void rope_at(const sml_ctx * c, float * x, int n_head, int head_dim, int pos, int L) {
    if (c->layer_rope && !c->layer_rope[L]) return;
    if (c->rope_cos_l && c->layer_win && c->layer_win[L] > 0)
        rope_tab(c, c->rope_cos_l, c->rope_sin_l, x, n_head, head_dim, pos);
    else
        rope_tab(c, c->rope_cos, c->rope_sin, x, n_head, head_dim, pos);
}

// SiLU gate: x * sigmoid(x), over the FFN intermediate.
//
// This runs n_ff times per layer -- 396k calls per token on a 3B with 36 layers
// -- and a scalar expf here was the single largest cost outside the kernel, 21%
// of total runtime. The exp is done with a degree-4 polynomial on the fractional
// part plus exponent-field construction: ~1e-7 relative, far inside the noise of
// a 2-bit weight.
#if defined(__AVX2__)
static inline __m256 sml_exp_ps(__m256 x) {
    const __m256 lo = _mm256_set1_ps(-88.0f), hi = _mm256_set1_ps(88.0f);
    x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);

    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    __m256 f = _mm256_mul_ps(x, log2e);
    __m256 r = _mm256_round_ps(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 g = _mm256_sub_ps(f, r);                  // fractional part, |g| <= 0.5

    // 2^g on [-0.5, 0.5]
    __m256 p = _mm256_set1_ps(1.3697664e-3f);
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(8.3013611e-3f));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(5.5826318e-2f));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(2.4022652e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(6.9314718e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, g), _mm256_set1_ps(1.0f));

    // scale by 2^r via the exponent field
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(r),
                                                   _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

// exp to float rounding (Cephes expf): x = n ln2 + r with ln2 split hi/lo so
// r is exact, e^r by a degree-5 polynomial on [-ln2/2, ln2/2], times 2^n.
static inline __m256 sml_exp_acc_ps(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.693359375f)));
    r = _mm256_sub_ps(r, _mm256_mul_ps(n, _mm256_set1_ps(-2.12194440e-4f)));
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(p, r), r), r), _mm256_set1_ps(1.0f));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n),
                                                         _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

static void silu_mul(float * gate, const float * up, int n) {
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 g = _mm256_loadu_ps(gate + i);
        const __m256 d = _mm256_add_ps(one, sml_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), g)));
        _mm256_storeu_ps(gate + i, _mm256_mul_ps(_mm256_div_ps(g, d),
                                                 _mm256_loadu_ps(up + i)));
    }
    for (; i < n; i++) gate[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
}
#else
// Same function, libm exp. Slower -- the vector version was worth 21% of
// runtime on a 3B -- but correct, and a machine without AVX2 is not going to
// be bound by this anyway.
static void silu_mul(float * gate, const float * up, int n) {
    for (int i = 0; i < n; i++)
        gate[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
}
#endif

// GELU, exact erf form: 0.5 v (1 + erf(v / sqrt 2)). erf by Abramowitz and
// Stegun 7.1.26 (|error| <= 1.5e-7, float rounding), written so 1 + erf never
// cancels: for z < 0 it is P(t) exp(-z^2) directly, for z >= 0 2 - P(t) exp(-z^2).
// The exp is the accurate one gelu_inplace uses.
static void gelu_erf_inplace(float * x, int n) {
    int i = 0;
#if defined(__AVX2__)
    const __m256 rs2 = _mm256_set1_ps(0.70710678118654752f), sign = _mm256_set1_ps(-0.0f);
    const __m256 one = _mm256_set1_ps(1.0f), two = _mm256_set1_ps(2.0f), half = _mm256_set1_ps(0.5f);
    const __m256 p = _mm256_set1_ps(0.3275911f);
    const __m256 a1 = _mm256_set1_ps(0.254829592f), a2 = _mm256_set1_ps(-0.284496736f);
    const __m256 a3 = _mm256_set1_ps(1.421413741f), a4 = _mm256_set1_ps(-1.453152027f);
    const __m256 a5 = _mm256_set1_ps(1.061405429f);
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 z = _mm256_mul_ps(v, rs2);
        const __m256 az = _mm256_andnot_ps(sign, z);
        const __m256 t = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_mul_ps(p, az)));
        __m256 poly = _mm256_add_ps(_mm256_mul_ps(a5, t), a4);
        poly = _mm256_add_ps(_mm256_mul_ps(poly, t), a3);
        poly = _mm256_add_ps(_mm256_mul_ps(poly, t), a2);
        poly = _mm256_add_ps(_mm256_mul_ps(poly, t), a1);
        poly = _mm256_mul_ps(poly, t);
        const __m256 tail = _mm256_mul_ps(poly, sml_exp_acc_ps(_mm256_sub_ps(_mm256_setzero_ps(),
                                                                  _mm256_mul_ps(az, az))));
        // 1 + erf(z): tail for z < 0, 2 - tail for z >= 0
        const __m256 neg = _mm256_cmp_ps(z, _mm256_setzero_ps(), _CMP_LT_OQ);
        const __m256 onep = _mm256_blendv_ps(_mm256_sub_ps(two, tail), tail, neg);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(half, v), onep));
    }
#endif
    for (; i < n; i++) x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118654752f));
}

// GELU (tanh approximation), for FFNs that have no gate projection at all:
// up -> GELU -> down, rather than silu(gate) * up.
static void gelu_inplace(float * x, int n) {
    int i = 0;
#if defined(__AVX2__)
    // 0.5 v (1 + tanh u) == v / (1 + exp(-2u)): the same function through the
    // vector exp silu already uses, instead of a libm tanhf per element --
    // 98k of them per pythia token. Not bit-identical to tanhf; same to float
    // rounding.
    //
    // The exp here is NOT silu's: that polynomial is ~1.5e-4 relative, which
    // through 24 GELU layers moved pythia's logits by 2% -- measured, then
    // replaced by a Cephes-style exp (ln2 split in two, degree-5 on
    // [-ln2/2, ln2/2]) that agrees with libm to float rounding.
    const __m256 k0 = _mm256_set1_ps(-2.0f * 0.7978845608f);
    const __m256 k1 = _mm256_set1_ps(0.044715f);
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 u = _mm256_mul_ps(k0, _mm256_add_ps(v,
                             _mm256_mul_ps(k1, _mm256_mul_ps(v, _mm256_mul_ps(v, v)))));
        _mm256_storeu_ps(x + i, _mm256_div_ps(v, _mm256_add_ps(one, sml_exp_acc_ps(u))));
    }
#endif
    for (; i < n; i++) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

// The file says which GELU the model was trained with (SML_ARCH_GELU_ERF).
static void gelu_act(const sml_ctx * c, float * x, int n) {
    const sml_hdr * h = c->model->hdr;
    if (h->version >= 5 && (h->arch_bits & SML_ARCH_GELU_ERF)) gelu_erf_inplace(x, n);
    else                                                        gelu_inplace(x, n);
}

// A gated FFN multiplies act(gate) by up. The act is SiLU for SwiGLU models;
// Gemma's GeGLU uses GELU (tanh form unless SML_ARCH_GELU_ERF), and running it
// as SiLU was a different model producing plausible numbers (98% logit error
// against transformers on a Gemma-shaped head, 2026-09-28).
static void gate_mul(const sml_ctx * c, float * g, const float * u, int n) {
    const sml_hdr * h = c->model->hdr;
    if (h->version >= 5 && (h->arch_bits & SML_ARCH_SWIGLU_OAI)) {
        // gpt-oss (GptOssExperts._apply_gate): gate capped at 7, up clamped to +-7 and shifted by 1
        for (int i = 0; i < n; i++) {
            const float a = g[i] < 7.0f ? g[i] : 7.0f;
            const float b = u[i] < -7.0f ? -7.0f : u[i] > 7.0f ? 7.0f : u[i];
            g[i] = a / (1.0f + expf(-1.702f * a)) * (b + 1.0f);
        }
    } else if (h->version >= 5 && (h->arch_bits & SML_ARCH_GELU_GATE)) {
        gelu_act(c, g, n);
        for (int i = 0; i < n; i++) g[i] *= u[i];
    } else {
        silu_mul(g, u, n);
    }
}

// An ungated FFN's activation: GELU, or relu(x)^2 (Nemotron, Persimmon, Arcee)
// xIELU (Apertus): alpha_p x^2 + beta x above zero, (expm1(min(x, eps)) - x)
// alpha_n + beta x below -- transformers' XIELUActivation, its numbers per layer.
static void xielu_act(const float * p, float * x, int n) {
    const float ap = p[0], an = p[1], be = p[2], ep = p[3];
    for (int i = 0; i < n; i++) {
        const float v = x[i];
        x[i] = v > 0.0f ? ap * v * v + be * v : (expm1f(v < ep ? v : ep) - v) * an + be * v;
    }
}

static void ffn_act(const sml_ctx * c, float * x, int n) {
    const sml_hdr * h = c->model->hdr;
    if (h->version >= 5 && (h->arch_bits & SML_ARCH_RELU2)) {
        for (int i = 0; i < n; i++) { const float v = x[i] > 0.0f ? x[i] : 0.0f; x[i] = v * v; }
    } else if (h->version >= 5 && (h->arch_bits & SML_ARCH_RELU)) {
        for (int i = 0; i < n; i++) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
    } else {
        gelu_act(c, x, n);
    }
}

// ------------------------------------------------------------------ matvec

// One laddered matvec: gather the activation through this tensor's column
// permutation, quantize it once, then accumulate `depth` levels into y.
//
// The gather is what makes the permutation exact -- the weights were quantized
// in permuted order, so (W P)(P^T x) = W x only holds if the activation is
// permuted to match. It costs one pass over a few thousand floats against
// millions of weight bytes.
// One issue point for every path below (serial, parallel, non-OpenMP), so the
// layout decision cannot drift between them. A FLAT tensor is one plane and one
// pass; `depth` has no meaning for it, because there is no prefix to truncate.
// SML_FLAT6=0 keeps the two-pass 6-bit decode (2-bit plane, then nibbles).
// One row of a Q3/Q5/Q7 tensor from its planes (FLAT_PLANES in sml_quantize.py):
// Q3 = 2-bit + 4*b, Q5 = nibble + 16*b, Q7 = Q6's code (nibble + 16*2-bit) + 64*b.
static void dequant_row_mixed(float * out, const sml_model * m, const sml_rec * r, int row) {
    const uint16_t * sd = (const uint16_t *) sml_ptr(m, r->sd_off);
    const uint8_t  * sq = (const uint8_t  *) sml_ptr(m, r->sq_off);
    const uint16_t * md = r->mq_n ? (const uint16_t *) sml_ptr(m, r->md_off) : NULL;
    const uint8_t  * mq = r->mq_n ? (const uint8_t  *) sml_ptr(m, r->mq_off) : NULL;
    const int n = (int) r->cols;
    const uint8_t * p0 = sml_ptr(m, r->codes_off[0]);
    if (r->nbits == 3) {
        const uint8_t * pl[1] = { p0 };
        sml_dequant_row(out, pl, 1, 1, sd, sq, md, mq, row, n);
        sml_dequant_bits1_add(out, sml_ptr(m, r->codes_off[1]), sd, sq, row, n, 2);
    } else if (r->nbits == 5) {
        sml_dequant_row_flat4(out, p0, sd, sq, md, mq, row, n);
        sml_dequant_bits1_add(out, sml_ptr(m, r->codes_off[1]), sd, sq, row, n, 4);
    } else {
        sml_dequant_row_flat6(out, p0, sml_ptr(m, r->codes_off[1]), sd, sq, md, mq, row, n);
        sml_dequant_bits1_add(out, sml_ptr(m, r->codes_off[2]), sd, sq, row, n, 6);
    }
}

static int flat6_fused(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("SML_FLAT6"); on = e ? atoi(e) != 0 : 1; }
    return on;
}

static inline void issue_rows_act(sml_ctx * c, const sml_act * act, float * y,
                                  const sml_rec * r,
                              const uint16_t * sd, const uint8_t * sq,
                              const uint16_t * md, const uint8_t * mq,
                              int lo, int hi, int n, int depth, int nlev,
                              float cbias) {
    const uint8_t * il = c->il ? c->il[r - c->model->recs] : NULL;
    if (il) {
        if (r->nbits == 6) sml_matvec_rows_il6(y, il, act, lo, hi, n);
        else               sml_matvec_rows_il4(y, il, act, lo, hi, n);
        return;
    }
    if (r->layout == SML_LAYOUT_FLAT) {
        if (r->nbits == 3 || r->nbits == 5 || r->nbits == 7) {
            const uint8_t * p0 = sml_ptr(c->model, r->codes_off[0]), * p1 = sml_ptr(c->model, r->codes_off[1]);
            if (r->nbits == 3) {        // Q3: 2-bit + 4 * 1-bit
                sml_matvec_rows(y, p0, sd, sq, md, mq, act, lo, hi, n, 0, 0.0f);
                sml_matvec_bits1(y, p1, sd, sq, act, 1, 0, lo, hi, n, 2);
            } else if (r->nbits == 5) { // Q5: nibble + 16 * 1-bit
                sml_matvec_rows_flat4(y, p0, sd, sq, md, mq, act, lo, hi, n);
                sml_matvec_bits1(y, p1, sd, sq, act, 1, 0, lo, hi, n, 4);
            } else {                    // Q7: Q6's code (nibble + 16 * 2-bit) + 64 * 1-bit
                sml_matvec_rows_flat6(y, p0, p1, sd, sq, md, mq, act, lo, hi, n);
                sml_matvec_bits1(y, sml_ptr(c->model, r->codes_off[2]), sd, sq, act, 1, 0, lo, hi, n, 6);
            }
            return;
        }
        if (r->nbits == 2) {            // Q2: one 2-bit plane, the min with it
            sml_matvec_rows(y, sml_ptr(c->model, r->codes_off[0]), sd, sq, md, mq, act, lo, hi, n, 0, 0.0f);
            return;
        }
        if (r->nbits == 8) {            // Q8: low + 16*high nibble planes
            sml_matvec_rows_flat8(y, sml_ptr(c->model, r->codes_off[0]), sml_ptr(c->model, r->codes_off[1]),
                                  sd, sq, md, mq, act, lo, hi, n);
            return;
        }
        if (r->nbits == 6) {
            // A 6-bit flat code is base(4) + 16*extra(2), which is LINEAR, so
            //     sum(code*x) = sum(base*x) + 16*sum(extra*x)
            // and both passes already exist. The 2-bit pass takes shift=4 for
            // the factor of 16 and carries the min term; the nibble pass adds
            // the base with the min suppressed, so it is counted once.
            //
            // This is not the ladder. The grid is chosen for the 6 bits that
            // will actually be read, and no prefix of the code is required to
            // mean anything -- which is the whole difference, and worth 1.552x.
            if (flat6_fused())
                sml_matvec_rows_flat6(y, sml_ptr(c->model, r->codes_off[1]),
                                      sml_ptr(c->model, r->codes_off[0]),
                                      sd, sq, md, mq, act, lo, hi, n);
            else {
                sml_matvec_rows(y, sml_ptr(c->model, r->codes_off[0]), sd, sq,
                                md, mq, act, lo, hi, n, 4, 0.0f);
                sml_matvec_rows_flat4(y, sml_ptr(c->model, r->codes_off[1]), sd, sq,
                                      NULL, NULL, act, lo, hi, n);
            }
            return;
        }
        sml_matvec_rows_flat4(y, sml_ptr(c->model, r->codes_off[0]), sd, sq,
                              md, mq, act, lo, hi, n);
        return;
    }
    for (int lv = 0; lv < depth; lv++)
        sml_matvec_rows(y, sml_ptr(c->model, r->codes_off[lv]), sd, sq,
                        lv == depth - 1 ? md : NULL,
                        lv == depth - 1 ? mq : NULL,
                        act, lo, hi, n, (nlev - 1 - lv) * SML_LEVEL_BITS,
                        lv == depth - 1 ? cbias : 0.0f);
}

static inline void issue_rows(sml_ctx * c, float * y, const sml_rec * r,
                              const uint16_t * sd, const uint8_t * sq,
                              const uint16_t * md, const uint8_t * mq,
                              int lo, int hi, int n, int depth, int nlev,
                              float cbias) {
    issue_rows_act(c, &c->act, y, r, sd, sq, md, mq, lo, hi, n, depth, nlev, cbias);
}

typedef struct {
    sml_ctx * c; float * y; const sml_rec * r;
    const uint16_t * sd; const uint8_t * sq; const uint16_t * md; const uint8_t * mq;
    int rows, n, depth, nlev, chunk; float cbias;
} mv_job;

static void mv_chunk(void * p, int k) {
    const mv_job * j = (const mv_job *) p;
    const int lo = k * j->chunk;
    const int hi = lo + j->chunk < j->rows ? lo + j->chunk : j->rows;
    issue_rows(j->c, j->y, j->r, j->sd, j->sq, j->md, j->mq, lo, hi, j->n,
               j->depth, j->nlev, j->cbias);
}

// SML_POOL=0 dispatches decode on OpenMP regions again (see sml-pool.h).
static int use_pool(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("SML_POOL"); on = e ? atoi(e) != 0 : 1; }
    return on;
}

static void matvec(sml_ctx * c, float * y, const sml_rec * r,
                   const float * x, int depth) {
    const int n    = (int) r->cols;
    const int rows = (int) r->rows;
    if (r->kind == SML_KIND_F32) {
        const double tm = sml_now();
        const float * w = (const float *) sml_ptr(c->model, r->off);
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) sum += (double) w[(size_t) i * n + j] * x[j];
            y[i] = (float) sum;
        }
        const double dt = sml_now() - tm;
        const uint64_t bytes = (uint64_t) rows * (uint64_t) n * sizeof(float);
        c->bytes_moved += bytes;
        sml_t_matvec += dt;
        const int sl = shape_slot(rows);
        sml_shape_time[sl] += dt;
        sml_shape_calls[sl] += 1;
        sml_shape_bytes[sl] += bytes;
        return;
    }
    const int nlev = (int) c->model->hdr->n_levels;
    if (depth < 1) depth = 1;
    if (depth > nlev) depth = nlev;

    const double tp = sml_now();
    const int32_t * perm = (const int32_t *) sml_ptr(c->model, r->perm_off);
    if (c->model->hdr->flags & SML_MODEL_HADAMARD) {
        sml_hadamard32(c->xr, x, n);
        sml_gather(c->xg, c->xr, perm, n);
    } else {
        sml_gather(c->xg, x, perm, n);
    }
    sml_quantize_act(&c->act, c->xg, n);
    sml_t_prep += sml_now() - tp;

    const uint16_t * sd = (const uint16_t *) sml_ptr(c->model, r->sd_off);
    const uint8_t  * sq = (const uint8_t  *) sml_ptr(c->model, r->sq_off);
    // v6 carries a per-group min; v5 files have none and stay symmetric.
    const uint16_t * md = r->mq_n ? (const uint16_t *) sml_ptr(c->model, r->md_off) : NULL;
    const uint8_t  * mq = r->mq_n ? (const uint8_t  *) sml_ptr(c->model, r->mq_off) : NULL;
    const float cbias = mq ? sml_level_mid(depth, nlev)
                           : sml_level_bias(depth, nlev);

    const double tm = sml_now();
    memset(y, 0, (size_t) rows * sizeof(float));

    // All levels inside ONE parallel region. Opening a region per level meant
    // 253 * depth fork/joins per token; each thread now walks its own row range
    // through every level, which also keeps that range's scales hot in cache.
#ifdef _OPENMP
    // Scale the team to the work. attn_k and attn_v are 256 rows against 11008
    // for the FFN, and splitting 256 rows six ways leaves each thread ~43 rows
    // -- the fork/join costs more than the arithmetic it is dividing. Measured
    // at 3.8 GB/s against 24.9 for the wide tensors, on 3744 calls per run.
    //
    // 128 rows per thread keeps the wide tensors on the full team and lets the
    // narrow ones run on one or two, where they are bandwidth-bound anyway.
    // Narrow tensors skip the parallel region ENTIRELY.
    //
    // attn_k and attn_v are 256 rows against 11008 for the FFN, and their rate
    // gets WORSE as threads are added -- 6.6 GB/s on one thread, 3.8 on six,
    // while the wide tensors go 6.5 -> 20.7 over the same range. One thread
    // reaches the same rate as every other tensor, so the kernel was never the
    // problem: the barrier costs more than the work it is dividing, and idling
    // the surplus threads does not help because the team is still synchronised.
    //
    // The threshold is work per thread, not row count: below ~128 rows each,
    // dispatch dominates.
    const int par = rows >= 128 * c->n_threads;
    // Fixed slices make every call wait for its slowest thread, and on a
    // machine running anything else one core is always shared. Prompt reading
    // measured +13-22% at six threads from handing rows out in chunks instead
    // (see matvec_batch). SML_DECODE_CHUNK_ROWS sets the chunk; 0 = fixed slices.
    static int chunk_rows = -1;
    if (chunk_rows < 0) {
        const char * e = getenv("SML_DECODE_CHUNK_ROWS");
        chunk_rows = e ? atoi(e) : 32;
        if (chunk_rows < 0) chunk_rows = 0;
    }
    if (!par) {
        issue_rows(c, y, r, sd, sq, md, mq, 0, rows, n, depth, nlev, cbias);
    } else if (chunk_rows > 0 && use_pool()) {
        mv_job j = { c, y, r, sd, sq, md, mq, rows, n, depth, nlev, chunk_rows, cbias };
        sml_pool_for((rows + chunk_rows - 1) / chunk_rows, mv_chunk, &j);
    } else if (chunk_rows > 0) {
        const int nch = (rows + chunk_rows - 1) / chunk_rows;
#pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < nch; k++) {
            const int lo = k * chunk_rows;
            const int hi = lo + chunk_rows < rows ? lo + chunk_rows : rows;
            issue_rows(c, y, r, sd, sq, md, mq, lo, hi, n, depth, nlev, cbias);
        }
    } else {
    int want = rows / 128;
    if (want < 1) want = 1;
#pragma omp parallel
    {
        const int t   = omp_get_thread_num();
        const int all = omp_get_num_threads();
        const int nt  = want < all ? want : all;
        const int lo = t < nt ? (int)((long) rows * t / nt) : 0;
        const int hi = t < nt ? (int)((long) rows * (t + 1) / nt) : 0;
        issue_rows(c, y, r, sd, sq, md, mq, lo, hi, n, depth, nlev, cbias);
    }
    }
#else
    issue_rows(c, y, r, sd, sq, md, mq, 0, rows, n, depth, nlev, cbias);
#endif
    if (r->layout == SML_LAYOUT_FLAT)
        c->bytes_moved += (uint64_t) rows * (uint64_t) n * r->nbits / 8;
    else
        for (int lv = 0; lv < depth; lv++) c->bytes_moved += sml_level_bytes(rows, n);
    // The asymmetric grid (v6+) stores a per-group MIN exactly like the scale,
    // so it is the same size again. sml_scale_bytes counts only sd+sq, which
    // under-reported every byte figure this project has published since v6.
    c->bytes_moved += sml_scale_bytes(rows, n) * (r->mq_n ? 2 : 1);
    const double dt = sml_now() - tm;
    sml_t_matvec += dt;
    {
        const int sl = shape_slot(rows);
        sml_shape_time[sl] += dt;
        sml_shape_calls[sl] += 1;
        sml_shape_bytes[sl] += (uint64_t) depth * sml_level_bytes(rows, n)
                             + sml_scale_bytes(rows, n) * (r->mq_n ? 2 : 1);
    }
}

// Q, K and V in ONE dispatch.
//
// They read the same activation and run back to back, but attn_k and attn_v are
// only 256 rows against attn_q's 2048. Dispatched alone they are too small to
// pay for a parallel region -- measured 6.1 GB/s against 22 for the wide
// tensors -- so they run serially and stay slow. Folded into Q's dispatch they
// ride a team that is already worth forming, and the row space simply spans
// three weight matrices instead of one.
//
// Each tensor keeps its own column permutation, so the activation is prepared
// three times (cheap, ~0.1s of a 4.2s run) and the three quantized copies are
// held in the batch activation buffers, which the single-token path never uses.
// A GROUP is a set of tensors that read the same activation and are issued as
// one unit. The relationship is the thing worth recording: three matrices that
// happen to be adjacent is a storage fact, three matrices that share a flocking
// context is a scheduling fact, and only the second lets the runtime decide
// anything. Q/K/V are one such group, ffn_gate/ffn_up are another.
typedef struct {
    sml_ctx * c; int ng_; float * const * y; const sml_rec * const * r;
    const uint16_t * const * sd; const uint8_t * const * sq;
    const uint16_t * const * md; const uint8_t * const * mq; const float * cb;
    const int * rows; const int * off; const int * src;
    int unit, total, depth, nlev;
} mvn_job;

static void mvn_chunk(void * p, int k) {
    const mvn_job * j = (const mvn_job *) p;
    sml_ctx * c = j->c;
    const int lo = k * j->unit;
    const int hi = lo + j->unit < j->total ? lo + j->unit : j->total;
    for (int i = 0; i < j->ng_; i++) {
        int a = lo - j->off[i], b = hi - j->off[i];
        if (a < 0) a = 0;
        if (b > j->rows[i]) b = j->rows[i];
        if (a >= b) continue;
        const sml_rec * r = j->r[i];
        const sml_act * act = &c->bact[j->src[i]];
        const int n = (int) r->cols;
        const uint8_t * il = c->il ? c->il[r - c->model->recs] : NULL;
        if (il) {
            if (r->nbits == 6) sml_matvec_rows_il6(j->y[i], il, act, a, b, n);
            else               sml_matvec_rows_il4(j->y[i], il, act, a, b, n);
            continue;
        }
        if (r->layout == SML_LAYOUT_FLAT) {
            if (r->nbits == 3 || r->nbits == 5 || r->nbits == 7) {
                issue_rows_act(c, act, j->y[i], r, j->sd[i], j->sq[i], j->md[i], j->mq[i], a, b, n, 0, 0, 0.0f);
                continue;
            }
            if (r->nbits == 2) {
                sml_matvec_rows(j->y[i], sml_ptr(c->model, r->codes_off[0]), j->sd[i], j->sq[i], j->md[i], j->mq[i],
                                act, a, b, n, 0, 0.0f);
                continue;
            }
            if (r->nbits == 8) {
                sml_matvec_rows_flat8(j->y[i], sml_ptr(c->model, r->codes_off[0]), sml_ptr(c->model, r->codes_off[1]),
                                      j->sd[i], j->sq[i], j->md[i], j->mq[i], act, a, b, n);
                continue;
            }
            if (r->nbits == 6 && flat6_fused()) {
                sml_matvec_rows_flat6(j->y[i], sml_ptr(c->model, r->codes_off[1]),
                                      sml_ptr(c->model, r->codes_off[0]),
                                      j->sd[i], j->sq[i], j->md[i], j->mq[i], act, a, b, n);
                continue;
            }
            if (r->nbits == 6)
                sml_matvec_rows(j->y[i], sml_ptr(c->model, r->codes_off[0]),
                                j->sd[i], j->sq[i], j->md[i], j->mq[i], act, a, b, n, 4, 0.0f);
            sml_matvec_rows_flat4(j->y[i], sml_ptr(c->model, r->codes_off[r->nbits == 6 ? 1 : 0]),
                                  j->sd[i], j->sq[i],
                                  r->nbits == 6 ? NULL : j->md[i],
                                  r->nbits == 6 ? NULL : j->mq[i], act, a, b, n);
            continue;
        }
        for (int lv = 0; lv < j->depth; lv++)
            sml_matvec_rows(j->y[i], sml_ptr(c->model, r->codes_off[lv]),
                            j->sd[i], j->sq[i],
                            lv == j->depth - 1 ? j->md[i] : NULL,
                            lv == j->depth - 1 ? j->mq[i] : NULL,
                            act, a, b, n, (j->nlev - 1 - lv) * SML_LEVEL_BITS,
                            lv == j->depth - 1 ? j->cb[i] : 0.0f);
    }
}

static void matvecN(sml_ctx * c, int ng_, float * const y[], const sml_rec * const r[],
                    const float * x, int depth) {
    for (int i = 0; i < ng_; i++) {
        if (r[i]->kind != SML_KIND_F32) continue;
        for (int j = 0; j < ng_; j++) matvec(c, y[j], r[j], x, depth);
        return;
    }
    const int nlev = (int) c->model->hdr->n_levels;
    if (depth < 1) depth = 1;
    if (depth > nlev) depth = nlev;

    // Prepare the activation ONCE per distinct column order. A group whose
    // members share a permutation shares an activation LIFETIME, not merely an
    // input: gather and quantize run once and every member reads the same
    // prepared vector. Builder-side that costs the ~0.6% a per-tensor
    // permutation was worth; here it removes two thirds of the preparation for
    // a three-member group.
    //
    // src[i] is which buffer member i reads, so a file that does NOT share
    // permutations still works -- it just prepares one per member, as before.
    const double tp = sml_now();
    int src[SML_MAX_BATCH];
    int nprep = 0;
    for (int i = 0; i < ng_; i++) {
        int found = -1;
        for (int j = 0; j < i; j++)
            if (r[j]->perm_off == r[i]->perm_off) { found = src[j]; break; }
        if (found >= 0) { src[i] = found; continue; }
        if (c->model->hdr->flags & SML_MODEL_HADAMARD) {
            sml_hadamard32(c->xr, x, (int) r[i]->cols);
            sml_gather(c->xg, c->xr,
                       (const int32_t *) sml_ptr(c->model, r[i]->perm_off),
                       (int) r[i]->cols);
        } else
        sml_gather(c->xg, x, (const int32_t *) sml_ptr(c->model, r[i]->perm_off),
                   (int) r[i]->cols);
        sml_quantize_act(&c->bact[nprep], c->xg, (int) r[i]->cols);
        src[i] = nprep++;
    }
    sml_t_prep += sml_now() - tp;

    const uint16_t * sd[SML_MAX_BATCH], * md[SML_MAX_BATCH];
    const uint8_t  * sq[SML_MAX_BATCH], * mq[SML_MAX_BATCH];
    float cb[SML_MAX_BATCH];
    int rows[SML_MAX_BATCH], off[SML_MAX_BATCH], total = 0;
    for (int i = 0; i < ng_; i++) {
        sd[i] = (const uint16_t *) sml_ptr(c->model, r[i]->sd_off);
        sq[i] = (const uint8_t  *) sml_ptr(c->model, r[i]->sq_off);
        md[i] = r[i]->mq_n ? (const uint16_t *) sml_ptr(c->model, r[i]->md_off) : NULL;
        mq[i] = r[i]->mq_n ? (const uint8_t  *) sml_ptr(c->model, r[i]->mq_off) : NULL;
        cb[i] = mq[i] ? sml_level_mid(depth, nlev) : sml_level_bias(depth, nlev);
        rows[i] = (int) r[i]->rows;
        off[i] = total;
        total += rows[i];
        memset(y[i], 0, (size_t) rows[i] * sizeof(float));
    }

    // Rows go out in chunks any free thread claims, as in matvec: a fixed slice
    // per thread waits on the slowest one. SML_DECODE_CHUNK_ROWS sets the
    // chunk; 0 = one slice per thread.
    static int chunk_rows = -1;
    if (chunk_rows < 0) {
        const char * e = getenv("SML_DECODE_CHUNK_ROWS");
        chunk_rows = e ? atoi(e) : 32;
        if (chunk_rows < 0) chunk_rows = 0;
    }
    const double tm = sml_now();
#ifdef _OPENMP
    const int nth  = omp_get_max_threads();
    const int unit = chunk_rows > 0 ? chunk_rows : (total + nth - 1) / nth;
    const int nch  = (total + unit - 1) / unit;
    if (use_pool()) {
        mvn_job j = { c, ng_, y, r, sd, sq, md, mq, cb, rows, off, src, unit, total, depth, nlev };
        sml_pool_for(nch, mvn_chunk, &j);
    } else
#pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < nch; k++) {
        const int lo = k * unit;
        const int hi = lo + unit < total ? lo + unit : total;
        for (int i = 0; i < ng_; i++) {
            int a = lo - off[i], b = hi - off[i];      // this thread's slice of tensor i
            if (a < 0) a = 0;
            if (b > rows[i]) b = rows[i];
            if (a >= b) continue;
            const uint8_t * il = c->il ? c->il[r[i] - c->model->recs] : NULL;
            if (il) {
                if (r[i]->nbits == 6) sml_matvec_rows_il6(y[i], il, &c->bact[src[i]], a, b, (int) r[i]->cols);
                else                  sml_matvec_rows_il4(y[i], il, &c->bact[src[i]], a, b, (int) r[i]->cols);
                continue;
            }
            if (r[i]->layout == SML_LAYOUT_FLAT) {
                if (r[i]->nbits == 3 || r[i]->nbits == 5 || r[i]->nbits == 7) {
                    issue_rows_act(c, &c->bact[src[i]], y[i], r[i], sd[i], sq[i], md[i], mq[i], a, b,
                                   (int) r[i]->cols, 0, 0, 0.0f);
                    continue;
                }
                if (r[i]->nbits == 2) {
                    sml_matvec_rows(y[i], sml_ptr(c->model, r[i]->codes_off[0]), sd[i], sq[i], md[i], mq[i],
                                    &c->bact[src[i]], a, b, (int) r[i]->cols, 0, 0.0f);
                    continue;
                }
                if (r[i]->nbits == 8) {
                    sml_matvec_rows_flat8(y[i], sml_ptr(c->model, r[i]->codes_off[0]),
                                          sml_ptr(c->model, r[i]->codes_off[1]), sd[i], sq[i], md[i], mq[i],
                                          &c->bact[src[i]], a, b, (int) r[i]->cols);
                    continue;
                }
                if (r[i]->nbits == 6 && flat6_fused()) {
                    sml_matvec_rows_flat6(y[i], sml_ptr(c->model, r[i]->codes_off[1]),
                                          sml_ptr(c->model, r[i]->codes_off[0]),
                                          sd[i], sq[i], md[i], mq[i],
                                          &c->bact[src[i]], a, b, (int) r[i]->cols);
                    continue;
                }
                if (r[i]->nbits == 6)
                    sml_matvec_rows(y[i], sml_ptr(c->model, r[i]->codes_off[0]),
                                    sd[i], sq[i], md[i], mq[i],
                                    &c->bact[src[i]], a, b, (int) r[i]->cols,
                                    4, 0.0f);
                sml_matvec_rows_flat4(y[i],
                    sml_ptr(c->model, r[i]->codes_off[r[i]->nbits == 6 ? 1 : 0]),
                    sd[i], sq[i],
                    r[i]->nbits == 6 ? NULL : md[i],
                    r[i]->nbits == 6 ? NULL : mq[i],
                    &c->bact[src[i]], a, b, (int) r[i]->cols);
                continue;
            }
            for (int lv = 0; lv < depth; lv++)
                sml_matvec_rows(y[i], sml_ptr(c->model, r[i]->codes_off[lv]),
                                sd[i], sq[i],
                                lv == depth - 1 ? md[i] : NULL,
                                lv == depth - 1 ? mq[i] : NULL,
                                &c->bact[src[i]], a, b, (int) r[i]->cols,
                                (nlev - 1 - lv) * SML_LEVEL_BITS,
                                lv == depth - 1 ? cb[i] : 0.0f);
        }
    }
#else
    for (int i = 0; i < ng_; i++) {
        if (r[i]->layout == SML_LAYOUT_FLAT) {
            if (r[i]->nbits == 3 || r[i]->nbits == 5 || r[i]->nbits == 7) {
                issue_rows_act(c, &c->bact[src[i]], y[i], r[i], sd[i], sq[i], md[i], mq[i], 0, rows[i],
                               (int) r[i]->cols, 0, 0, 0.0f);
                continue;
            }
            if (r[i]->nbits == 2) {
                sml_matvec_rows(y[i], sml_ptr(c->model, r[i]->codes_off[0]), sd[i], sq[i], md[i], mq[i],
                                &c->bact[src[i]], 0, rows[i], (int) r[i]->cols, 0, 0.0f);
                continue;
            }
            if (r[i]->nbits == 8) {
                sml_matvec_rows_flat8(y[i], sml_ptr(c->model, r[i]->codes_off[0]),
                                      sml_ptr(c->model, r[i]->codes_off[1]), sd[i], sq[i], md[i], mq[i],
                                      &c->bact[src[i]], 0, rows[i], (int) r[i]->cols);
                continue;
            }
            if (r[i]->nbits == 6)
                sml_matvec_rows(y[i], sml_ptr(c->model, r[i]->codes_off[0]),
                                sd[i], sq[i], md[i], mq[i],
                                &c->bact[src[i]], 0, rows[i], (int) r[i]->cols,
                                4, 0.0f);
            sml_matvec_rows_flat4(y[i],
                sml_ptr(c->model, r[i]->codes_off[r[i]->nbits == 6 ? 1 : 0]),
                sd[i], sq[i],
                r[i]->nbits == 6 ? NULL : md[i],
                r[i]->nbits == 6 ? NULL : mq[i],
                &c->bact[src[i]], 0, rows[i], (int) r[i]->cols);
            continue;
        }
        for (int lv = 0; lv < depth; lv++)
            sml_matvec_rows(y[i], sml_ptr(c->model, r[i]->codes_off[lv]),
                            sd[i], sq[i],
                            lv == depth - 1 ? md[i] : NULL,
                            lv == depth - 1 ? mq[i] : NULL,
                            &c->bact[src[i]], 0, rows[i], (int) r[i]->cols,
                            (nlev - 1 - lv) * SML_LEVEL_BITS,
                            lv == depth - 1 ? cb[i] : 0.0f);
    }
#endif
    const double dt = sml_now() - tm;
    sml_t_matvec += dt;
    for (int i = 0; i < ng_; i++) {
        // Layout-aware: a flat tensor streams nbits per weight in one pass, not
        // `depth` 2-bit planes. The ladder formula under-counted every tensor
        // issued through the group path -- which is Q/K/V and gate/up, most of
        // the model -- and bytes/token is the denominator of every claim.
        c->bytes_moved += (r[i]->layout == SML_LAYOUT_FLAT
                ? (uint64_t) rows[i] * r[i]->cols * r[i]->nbits / 8
                : (uint64_t) depth * sml_level_bytes(rows[i], (int) r[i]->cols))
                        + sml_scale_bytes(rows[i], (int) r[i]->cols)
                          * (r[i]->mq_n ? 2 : 1);
        const int sl = shape_slot(rows[i]);
        sml_shape_time[sl] += dt / ng_;
        sml_shape_calls[sl] += 1;
        sml_shape_bytes[sl] += (uint64_t) depth * sml_level_bytes(rows[i], (int) r[i]->cols)
                             + sml_scale_bytes(rows[i], (int) r[i]->cols);
    }
}


// ------------------------------------------------------------------ context

// Decode's private interleaved copy of every flat tensor the matvecs read
// (sml-kernel.c says why). Built once; a tensor that cannot be repacked, or a
// failed allocation, leaves decode on the file's arrays, which compute the same.
static struct { const sml_model * m; const uint8_t ** il; uint8_t * mem; size_t bytes; int refs; } g_il;

static int g_repack_on = 1;
void sml_set_repack(int on) { g_repack_on = on; }

static void repack_il(sml_ctx * c) {
    const char * e = getenv("SML_REPACK");
    if (e && *e == '0') return;
    if (!g_repack_on || c->il) return;
    const sml_model * m = c->model;
    if (g_il.m == m) {                    // another context of the same model: share
        c->il = g_il.il;
        c->il_bytes = g_il.bytes;
        g_il.refs++;
        return;
    }
    const int nt = (int) m->hdr->n_tensors;
    size_t total = 0;
    size_t * at = (size_t *) calloc((size_t) nt, sizeof(size_t));
    char * want = (char *) calloc((size_t) nt, 1);
    if (!at || !want) { free(at); free(want); return; }
    for (int i = 0; i < nt; i++) {
        const sml_rec * r = &m->recs[i];
        if (r->layout != SML_LAYOUT_FLAT || r->rows == 0) continue;
        if (r == c->emb && c->emb != c->head) continue;     // only ever read as a row
        if (!sml_il_ok((int) r->cols, (int) r->nbits, r->mq_n != 0)) continue;
        want[i] = 1;
        at[i] = total;
        total += (size_t) r->rows * sml_il_row_bytes((int) r->cols, (int) r->nbits);
    }
    uint8_t * mem = total ? (uint8_t *) malloc(total + 64) : NULL;
    const uint8_t ** il = mem ? (const uint8_t **) calloc((size_t) nt, sizeof(*il)) : NULL;
    if (!il) { free(mem); free(at); free(want); return; }
    uint8_t * base = (uint8_t *) (((uintptr_t) mem + 63) & ~(uintptr_t) 63);
    for (int i = 0; i < nt; i++) {
        if (!want[i]) continue;
        const sml_rec * r = &m->recs[i];
        const int rows = (int) r->rows, n = (int) r->cols, nb = (int) r->nbits;
        uint8_t * dst = base + at[i];
        const uint8_t * lo4 = sml_ptr(m, r->codes_off[nb == 6 ? 1 : 0]);
        const uint8_t * hi2 = nb == 6 ? sml_ptr(m, r->codes_off[0]) : NULL;
        const uint16_t * sd = (const uint16_t *) sml_ptr(m, r->sd_off);
        const uint8_t  * sq = (const uint8_t  *) sml_ptr(m, r->sq_off);
        const uint16_t * md = (const uint16_t *) sml_ptr(m, r->md_off);
        const uint8_t  * mq = (const uint8_t  *) sml_ptr(m, r->mq_off);
        const int step = 256;
#pragma omp parallel for schedule(dynamic, 1)
        for (int r0 = 0; r0 < rows; r0 += step)
            sml_il_pack(dst, lo4, hi2, sd, sq, md, mq, nb, r0, r0 + step < rows ? r0 + step : rows, n);
        il[i] = dst;
    }
    free(at); free(want);
    c->il = il;
    c->il_mem = mem;
    c->il_bytes = total;
    if (!g_il.m) {                        // the first model to repack is shared
        g_il.m = m; g_il.il = il; g_il.mem = mem; g_il.bytes = total; g_il.refs = 1;
        c->il_mem = NULL;                 // owned by the share, freed by its last user
    }
}

// The once-packed rows-in-lanes copy for several conversations at a time
// (sml_rlp_pack). Built the first time sml_forward_multi runs, shared by every
// context of the model: a single conversation never pays for it.
static struct { const sml_model * m; const uint8_t ** rlp; uint8_t * mem; int refs; } g_rlp;

static void build_rlp(sml_ctx * c) {
    const sml_model * m = c->model;
    if (c->rlp) return;
    if (g_rlp.m == m) { c->rlp = g_rlp.rlp; g_rlp.refs++; return; }
    if (g_rlp.m) return;                  /* one model at a time */
    const int nt = (int) m->hdr->n_tensors;
    size_t total = 0;
    size_t * at = (size_t *) calloc((size_t) nt, sizeof(size_t));
    if (!at) return;
    for (int i = 0; i < nt; i++) {
        const sml_rec * r = &m->recs[i];
        at[i] = (size_t) -1;
        if (r->layout != SML_LAYOUT_FLAT || r->rows == 0) continue;
        if (r == c->emb && c->emb != c->head) continue;
        if (!sml_rlp_ok((int) r->rows, (int) r->cols, (int) r->nbits, r->mq_n != 0)) continue;
        at[i] = total;
        total += sml_rlp_bytes((int) r->rows, (int) r->cols, (int) r->nbits);
    }
    uint8_t * mem = total ? (uint8_t *) malloc(total + 64) : NULL;
    const uint8_t ** tab = mem ? (const uint8_t **) calloc((size_t) nt, sizeof(*tab)) : NULL;
    if (!tab) { free(mem); free(at); return; }
    uint8_t * base = (uint8_t *) (((uintptr_t) mem + 63) & ~(uintptr_t) 63);
    for (int i = 0; i < nt; i++) {
        if (at[i] == (size_t) -1) continue;
        const sml_rec * r = &m->recs[i];
        const int nb = (int) r->nbits, tiles = (int) r->rows / 8;
        uint8_t * dst = base + at[i];
#pragma omp parallel for schedule(dynamic, 1)
        for (int t0 = 0; t0 < tiles; t0 += 16)
            sml_rlp_pack(dst, sml_ptr(m, r->codes_off[nb == 6 ? 1 : 0]),
                         nb == 6 ? sml_ptr(m, r->codes_off[0]) : NULL,
                         (const uint16_t *) sml_ptr(m, r->sd_off), (const uint8_t *) sml_ptr(m, r->sq_off),
                         (const uint16_t *) sml_ptr(m, r->md_off), (const uint8_t *) sml_ptr(m, r->mq_off),
                         nb, t0, t0 + 16 < tiles ? t0 + 16 : tiles, (int) r->cols);
        tab[i] = dst;
    }
    free(at);
    g_rlp.m = m; g_rlp.rlp = tab; g_rlp.mem = mem; g_rlp.refs = 1;
    c->rlp = tab;
}

int sml_ctx_init(sml_ctx * c, const sml_model * m, int n_ctx, int n_threads) {
    sml_kernel_init();
    memset(c, 0, sizeof *c);
    c->model = m;
    c->n_ctx = n_ctx;
    c->n_threads = n_threads;
    c->depth = (int) m->hdr->n_levels;

    const sml_hdr * h = m->hdr;
    // head_dim from the Q projection's actual shape. Every model before Qwen3
    // had n_head * head_dim == n_embd, which is why dividing never showed;
    // Qwen3-0.6B has a 1024-wide stream and 16 heads of 128, so its attention
    // is twice as wide as the stream and dividing gives the wrong head.
    const sml_rec * q0 = NULL;
    for (uint32_t L0 = 0; L0 < m->hdr->n_layer && !q0; L0++) q0 = sml_findf(m, "blk.%d.attn_q.weight", (int) L0);
    // latent attention with a low-rank Q (DeepSeek V3): its second half gives the heads
    for (uint32_t L0 = 0; L0 < m->hdr->n_layer && !q0; L0++) q0 = sml_findf(m, "blk.%d.attn_q_b.weight", (int) L0);
    const int hd   = (q0 && q0->rows % h->n_head == 0) ? (int)(q0->rows / h->n_head)
                                                      : (int)(h->n_embd / h->n_head);
    // Attention output is n_head * head_dim wide; buffers that receive it are
    // sized for whichever is larger, it or the residual stream.
    const int n_attn = (int) h->n_head * hd;
    const int wide_a = n_attn > (int) h->n_embd ? n_attn : (int) h->n_embd;
    c->head_dim = hd;
    c->step_only = 0;   // set below by the features the batched body lacks
    {
        // Learned absolute positions: read from the tensor table, stored f32 flat.
        const sml_rec * pe = sml_find(m, "position_embd.weight");
        c->pos_embd = pe ? (const float *) sml_ptr(m, pe->off) : NULL;
        c->pos_rows = pe ? (int) (pe->cols / h->n_embd) : 0;
        if (h->arch_bits & SML_ARCH_ALIBI) {
            // BLOOM's series: geometric over the nearest power of two of
            // heads, then every other slope of the next power for the rest.
            const int nh = (int) h->n_head;
            int p2 = 1;
            while (p2 * 2 <= nh) p2 *= 2;
            c->alibi = malloc((size_t) nh * sizeof(float));
            if (c->alibi) {
                const double base = pow(2.0, -pow(2.0, -(log2((double) p2) - 3.0)));
                for (int i = 0; i < p2; i++) c->alibi[i] = (float) pow(base, i + 1);
                const double eb = pow(2.0, -pow(2.0, -(log2(2.0 * p2) - 3.0)));
                for (int i = p2; i < nh; i++) c->alibi[i] = (float) pow(eb, 2 * (i - p2) + 1);
                // Falcon adds the bias to the raw scores and scales the sum
                // (BLOOM and MPT add it after scaling): the same slopes, times the scale
                if (h->version >= 5 && (h->arch_bits & SML_ARCH_ALIBI_SCALED)) {
                    const float sc = h->attn_scale != 0.0f ? h->attn_scale
                                   : 1.0f / sqrtf((float) (h->n_embd / h->n_head));
                    for (int i = 0; i < nh; i++) c->alibi[i] *= sc;
                }
            }
        }
        {
            const sml_rec * en = sml_find(m, "token_embd_norm.weight");
            const sml_rec * eb = sml_find(m, "token_embd_norm.bias");
            c->emb_norm   = en ? (const float *) sml_ptr(m, en->off) : NULL;
            c->emb_norm_b = eb ? (const float *) sml_ptr(m, eb->off) : NULL;
        }
        c->no_rope = pe != NULL || (h->arch_bits & SML_ARCH_ALIBI) != 0;
        if (pe) {
            if (n_ctx > c->pos_rows) {
                // a position past the table has no embedding at all
                fprintf(stderr, "sml: context %d > the %d positions this model learned; "
                        "using %d\n", n_ctx, c->pos_rows, c->pos_rows);
                n_ctx = c->pos_rows;
                c->n_ctx = n_ctx;
            }
        }
    }
    const int nkv  = h->n_head_kv * hd;
    // Every matrix's input is gathered into xg/xr and quantized into act, so
    // these hold the WIDEST input in the file, not just n_ff or n_embd: a
    // Mamba-2 out_proj reads d_inner, twice n_embd, and Nemotron-H's MLP is
    // narrower than that -- the gather wrote past the end (heap corruption,
    // 0xC0000374) on its big-shape family head (2026-10-13).
    int wide = h->n_ff > h->n_embd ? (int) h->n_ff : (int) h->n_embd;
    for (uint64_t i = 0; i < h->n_tensors; i++) {
        const sml_rec * r = &m->recs[i];
        if (r->rows && r->kind != SML_KIND_F32 && (int) r->cols > wide) wide = (int) r->cols;
    }

    c->hist = calloc((size_t) n_ctx, sizeof(int));
    c->n_hist = 0;
    c->k_cache = calloc((size_t) h->n_layer * n_ctx * nkv, sizeof(uint16_t));
    c->v_cache = calloc((size_t) h->n_layer * n_ctx * nkv, sizeof(uint16_t));
    c->x       = calloc(h->n_embd, sizeof(float));
    // Also receives the embedding row, which a padded file stores wider than
    // n_embd (whole zero groups up to a multiple of four).
    const int emb_w = ((int) h->n_embd + 127) / 128 * 128;
    c->xb      = calloc(wide_a > emb_w ? wide_a : emb_w, sizeof(float));      // also receives attention output
    c->xg      = calloc(wide, sizeof(float));
    c->xr      = calloc(wide, sizeof(float));
    c->q       = calloc(h->n_head * hd, sizeof(float));
    c->k       = calloc(nkv, sizeof(float));
    c->v       = calloc(nkv, sizeof(float));
    c->att     = calloc((size_t) h->n_head * n_ctx, sizeof(float));
    // hb2 also receives the attention OUTPUT projection, n_embd wide. Every dense
    // model has n_ff several times n_embd, so sizing these by n_ff always looked
    // right; in a mixture of experts n_ff is ONE expert (512 on Granite 1B-A400M,
    // under its 1024 stream), and each layer wrote 512 floats past the end.
    c->hb      = calloc(wide, sizeof(float));
    c->hb2     = calloc(wide, sizeof(float));
    c->logits  = calloc(h->n_vocab, sizeof(float));

    // RoPE tables: angle depends only on (position, index), so the whole model's
    // cos/sin set is known before the first token.
    // Partial rotary (GPT-NeoX, Phi, StableLM) rotates only n_rot of the head's
    // dimensions, and the frequency ladder is spread across THOSE, not across
    // the whole head.
    const int rot = (h->version >= 5 && h->n_rot) ? (int) h->n_rot : hd;
    const int half = rot / 2;
    c->rope_half = half;
    c->rope_cos = malloc((size_t) n_ctx * half * sizeof(float));
    c->rope_sin = malloc((size_t) n_ctx * half * sizeof(float));
    // The file may carry the frequencies outright. Context-extension schemes
    // (Llama 3.x, linear, dynamic) are exactly a change to these numbers, so a
    // stored table means the engine supports every one of them -- including
    // ones written after this engine was -- without a line of code per scheme.
    // Falling back to theta^(-i/half) keeps older files working.
    const sml_rec * fr = sml_find(m, "rope_freqs");
    const float * freqs = fr ? (const float *) sml_ptr(m, fr->off) : NULL;
    if (fr && (int) fr->cols < half) freqs = NULL;   // wrong size: recompute
    {
        const sml_rec * aw = sml_find(m, "attn_window");
        const sml_rec * ar = sml_find(m, "attn_rope");
        if (aw && (uint64_t) aw->rows * aw->cols >= h->n_layer) {
            const float * w = (const float *) sml_ptr(m, aw->off);
            c->layer_win = malloc(h->n_layer * sizeof(int));
            for (uint32_t L = 0; c->layer_win && L < h->n_layer; L++) c->layer_win[L] = (int) w[L];
        }
        if (ar && (uint64_t) ar->rows * ar->cols >= h->n_layer) {
            const float * w = (const float *) sml_ptr(m, ar->off);
            c->layer_rope = malloc(h->n_layer);
            for (uint32_t L = 0; c->layer_rope && L < h->n_layer; L++) c->layer_rope[L] = w[L] != 0.0f;
        }
        const sml_rec * fl = sml_find(m, "rope_freqs_local");
        if (fl && (int) fl->cols >= half) {
            const float * fq = (const float *) sml_ptr(m, fl->off);
            c->rope_cos_l = malloc((size_t) n_ctx * half * sizeof(float));
            c->rope_sin_l = malloc((size_t) n_ctx * half * sizeof(float));
            for (int p = 0; c->rope_cos_l && c->rope_sin_l && p < n_ctx; p++)
                for (int i = 0; i < half; i++) {
                    const float a = (float) p * fq[i];
                    c->rope_cos_l[(size_t) p * half + i] = cosf(a);
                    c->rope_sin_l[(size_t) p * half + i] = sinf(a);
                }
        }
    }
    if (c->rope_cos && c->rope_sin) {
        for (int p = 0; p < n_ctx; p++) {
            for (int i = 0; i < half; i++) {
                const float freq = freqs ? freqs[i]
                                 : powf(h->rope_theta, -(float) i / (float) half);
                const float a = (float) p * freq;
                c->rope_cos[(size_t) p * half + i] = cosf(a);
                c->rope_sin[(size_t) p * half + i] = sinf(a);
            }
        }
    }

    // Resolve every tensor once. sml_find is a linear scan of the table, and it
    // was running 8 times per layer per token for no reason.
    c->layers = calloc((size_t) h->n_layer, sizeof(sml_layer));
    c->emb = sml_find(m, "token_embd.weight");
    c->out_norm = sml_find(m, "output_norm.weight");
    // Qwen2.5 ties the head to the embedding table; Llama-3 and Mistral do not.
    // Assuming tied on an untied model reads the wrong matrix and produces
    // fluent-looking nonsense rather than an error, so it is resolved once here.
    c->head = sml_find(m, "output.weight");
    if (!c->head) c->head = c->emb;
    {
        const sml_rec * nb = sml_find(m, "output_norm.bias");
        c->out_norm_b = nb ? (const float *) sml_ptr(m, nb->off) : NULL;
        // Phi and GPT-J carry a bias on the output head; it was in the file and
        // never read
        const sml_rec * ob = sml_find(m, "output.bias");
        c->out_b = ob ? (const float *) sml_ptr(m, ob->off) : NULL;
    }
    for (uint32_t L = 0; L < h->n_layer && c->layers; L++) {
        sml_layer * s = &c->layers[L];
        s->attn_norm = sml_findf(m, "blk.%d.attn_norm.weight", (int) L);
        s->ffn_norm  = sml_findf(m, "blk.%d.ffn_norm.weight",  (int) L);
        // Phi, GPT-J, Cohere and Falcon-7B run attention and the FFN in parallel
        // off ONE LayerNorm: the FFN's input is that same norm of the same
        // residual, so it is the attention norm, read again (weights and bias).
        const int par_ln = h->version >= 5 && (h->arch_bits & SML_ARCH_PARALLEL_RESIDUAL);
        const int shared_ln = par_ln && !s->ffn_norm && s->attn_norm;
        if (shared_ln) s->ffn_norm = s->attn_norm;
        {
            const sml_rec * pa = sml_findf(m, "blk.%d.attn_post_norm.weight", (int) L);
            const sml_rec * pf = sml_findf(m, "blk.%d.ffn_post_norm.weight", (int) L);
            s->attn_post_norm = pa ? (const float *) sml_ptr(m, pa->off) : NULL;
            s->ffn_post_norm  = pf ? (const float *) sml_ptr(m, pf->off) : NULL;
            // parallel residual reads the FFN norm off the raw stream: it must exist
            if (par_ln && !s->ffn_norm) c->step_only = 1;
        }
        s->sc_in  = sml_findf(m, "blk.%d.shortconv_in.weight", (int) L);
        s->sc_out = sml_findf(m, "blk.%d.shortconv_out.weight", (int) L);
        if (s->sc_in) {
            const sml_rec * cw = sml_findf(m, "blk.%d.shortconv_conv.weight", (int) L);
            const sml_rec * cb = sml_findf(m, "blk.%d.shortconv_conv.bias", (int) L);
            const sml_rec * ib = sml_findf(m, "blk.%d.shortconv_in.bias", (int) L);
            const sml_rec * ob = sml_findf(m, "blk.%d.shortconv_out.bias", (int) L);
            if (!cw || !s->sc_out) { fprintf(stderr, "layer %d: short convolution incomplete\n", (int) L); return 0; }
            s->sc_w = (const float *) sml_ptr(m, cw->off);
            s->sc_L = (int) (((uint64_t) cw->rows * cw->cols) / h->n_embd);
            s->sc_cb = cb ? (const float *) sml_ptr(m, cb->off) : NULL;
            s->sc_in_b = ib ? (const float *) sml_ptr(m, ib->off) : NULL;
            s->sc_out_b = ob ? (const float *) sml_ptr(m, ob->off) : NULL;
            if (!c->sc_hist) c->sc_hist = calloc(h->n_layer, sizeof(float *));
            if (!c->sc_buf) c->sc_buf = calloc((size_t) 3 * h->n_embd, sizeof(float));
            if (!c->sc_y) c->sc_y = calloc(h->n_embd, sizeof(float));
            if (!c->sc_hist || !c->sc_buf || !c->sc_y) return 0;
            c->sc_hist[L] = calloc((size_t) n_ctx * h->n_embd, sizeof(float));
            if (!c->sc_hist[L]) return 0;
            c->step_only = 1;   // the batched body has no convolution layers yet
        }
        {
            const sml_rec * xr = sml_findf(m, "blk.%d.ffn_xielu.weight", (int) L);
            s->xielu = xr ? (const float *) sml_ptr(m, xr->off) : NULL;
        }
        s->attn_gate = sml_findf(m, "blk.%d.attn_gate.weight", (int) L);
        if (s->attn_gate) {
            if (!c->agate) c->agate = calloc(s->attn_gate->rows, sizeof(float));
            if (!c->agate) return 0;
        }
        s->gdn_qkvz = sml_findf(m, "blk.%d.gdn_qkvz.weight", (int) L);
        if (s->gdn_qkvz) {
            s->gdn_ba = sml_findf(m, "blk.%d.gdn_ba.weight", (int) L);
            s->gdn_out = sml_findf(m, "blk.%d.gdn_out.weight", (int) L);
            const sml_rec * cw = sml_findf(m, "blk.%d.gdn_conv.weight", (int) L);
            const sml_rec * db = sml_findf(m, "blk.%d.gdn_dt_b.weight", (int) L);
            const sml_rec * al = sml_findf(m, "blk.%d.gdn_a_log.weight", (int) L);
            const sml_rec * nw = sml_findf(m, "blk.%d.gdn_norm.weight", (int) L);
            if (!s->gdn_ba || !s->gdn_out || !cw || !db || !al || !nw || !h->gdn_nk || !h->gdn_dk) {
                fprintf(stderr, "layer %d: Gated DeltaNet incomplete\n", (int) L); return 0;
            }
            s->gdn_conv = (const float *) sml_ptr(m, cw->off);
            s->gdn_dt_b = (const float *) sml_ptr(m, db->off);
            s->gdn_a_log = (const float *) sml_ptr(m, al->off);
            s->gdn_nw = (const float *) sml_ptr(m, nw->off);
            s->gdn_nv = (int) ((uint64_t) al->rows * al->cols);
            s->gdn_dv = (int) ((uint64_t) nw->rows * nw->cols);
            const int cd = 2 * (int) (h->gdn_nk * h->gdn_dk) + s->gdn_nv * s->gdn_dv;
            const uint64_t ct = (uint64_t) cw->rows * cw->cols;
            s->gdn_K = ct % (uint64_t) cd == 0 ? (int) (ct / (uint64_t) cd) : 0;
            if (!s->gdn_K || s->gdn_nv % (int) h->gdn_nk
                || (int) s->gdn_out->cols != s->gdn_nv * s->gdn_dv) {
                fprintf(stderr, "layer %d: Gated DeltaNet shapes do not fit\n", (int) L); return 0;
            }
            if (!c->ssm_state) {
                c->ssm_state = calloc(h->n_layer, sizeof(float *));
                c->ssm_cst = calloc(h->n_layer, sizeof(float *));
                if (!c->ssm_state || !c->ssm_cst) return 0;
            }
            c->ssm_state[L] = calloc((size_t) s->gdn_nv * h->gdn_dk * s->gdn_dv, sizeof(float));
            c->ssm_cst[L] = calloc((size_t) (s->gdn_K > 1 ? s->gdn_K - 1 : 1) * cd, sizeof(float));
            float * pr = realloc(c->ssm_proj, ((size_t) s->gdn_qkvz->rows + s->gdn_ba->rows) * sizeof(float));
            float * xb = realloc(c->ssm_xbc, (size_t) cd * sizeof(float));
            float * yy = realloc(c->ssm_y, (size_t) s->gdn_nv * s->gdn_dv * sizeof(float));
            if (!c->ssm_state[L] || !c->ssm_cst[L] || !pr || !xb || !yy) return 0;
            c->ssm_proj = pr; c->ssm_xbc = xb; c->ssm_y = yy;
            c->step_only = 1;
        }
        s->m1_in = sml_findf(m, "blk.%d.ssm1_in.weight", (int) L);
        if (s->m1_in) {
            s->m1_x = sml_findf(m, "blk.%d.ssm1_x.weight", (int) L);
            s->m1_dt = sml_findf(m, "blk.%d.ssm1_dt.weight", (int) L);
            s->m1_out = sml_findf(m, "blk.%d.ssm1_out.weight", (int) L);
            const sml_rec * cw = sml_findf(m, "blk.%d.ssm1_conv.weight", (int) L);
            const sml_rec * cb = sml_findf(m, "blk.%d.ssm1_conv.bias", (int) L);
            const sml_rec * db = sml_findf(m, "blk.%d.ssm1_dt.bias", (int) L);
            const sml_rec * al = sml_findf(m, "blk.%d.ssm1_a_log.weight", (int) L);
            const sml_rec * dd = sml_findf(m, "blk.%d.ssm1_d.weight", (int) L);
            const sml_rec * n1 = sml_findf(m, "blk.%d.ssm1_dt_norm.weight", (int) L);
            const sml_rec * n2 = sml_findf(m, "blk.%d.ssm1_b_norm.weight", (int) L);
            const sml_rec * n3 = sml_findf(m, "blk.%d.ssm1_c_norm.weight", (int) L);
            if (!s->m1_x || !s->m1_dt || !s->m1_out || !cw || !al || !dd) {
                fprintf(stderr, "layer %d: Mamba mixer incomplete\n", (int) L); return 0;
            }
            s->m1_inner = (int) s->m1_out->cols;
            const uint64_t atot = (uint64_t) al->rows * al->cols, ctot = (uint64_t) cw->rows * cw->cols;
            s->m1_N = (int) (atot / (uint64_t) s->m1_inner);
            s->m1_K = (int) (ctot / (uint64_t) s->m1_inner);
            s->m1_R = (int) s->m1_x->rows - 2 * s->m1_N;
            if (s->m1_N < 1 || s->m1_K < 1 || s->m1_R < 1) {
                fprintf(stderr, "layer %d: Mamba shapes do not fit\n", (int) L); return 0;
            }
            s->m1_conv = (const float *) sml_ptr(m, cw->off);
            s->m1_conv_b = cb ? (const float *) sml_ptr(m, cb->off) : NULL;
            s->m1_dt_b = db ? (const float *) sml_ptr(m, db->off) : NULL;
            s->m1_a_log = (const float *) sml_ptr(m, al->off);
            s->m1_d = (const float *) sml_ptr(m, dd->off);
            s->m1_ndt = n1 ? (const float *) sml_ptr(m, n1->off) : NULL;
            s->m1_nb = n2 ? (const float *) sml_ptr(m, n2->off) : NULL;
            s->m1_nc = n3 ? (const float *) sml_ptr(m, n3->off) : NULL;
            if (!c->ssm_state) {
                c->ssm_state = calloc(h->n_layer, sizeof(float *));
                c->ssm_cst = calloc(h->n_layer, sizeof(float *));
                if (!c->ssm_state || !c->ssm_cst) return 0;
            }
            c->ssm_state[L] = calloc((size_t) s->m1_inner * s->m1_N, sizeof(float));
            c->ssm_cst[L] = calloc((size_t) (s->m1_K > 1 ? s->m1_K - 1 : 1) * s->m1_inner, sizeof(float));
            const size_t need = (size_t) 2 * s->m1_inner + s->m1_x->rows + 16;
            float * pr = realloc(c->ssm_proj, need * sizeof(float));
            float * xb = realloc(c->ssm_xbc, (size_t) s->m1_inner * sizeof(float));
            float * yy = realloc(c->ssm_y, (size_t) 2 * s->m1_inner * sizeof(float));
            if (!c->ssm_state[L] || !c->ssm_cst[L] || !pr || !xb || !yy) return 0;
            c->ssm_proj = pr; c->ssm_xbc = xb; c->ssm_y = yy;
            c->step_only = 1;
        }
        s->ssm_in  = sml_findf(m, "blk.%d.ssm_in.weight", (int) L);
        s->ssm_out = sml_findf(m, "blk.%d.ssm_out.weight", (int) L);
        if (s->ssm_in) {
            const sml_rec * cw = sml_findf(m, "blk.%d.ssm_conv.weight", (int) L);
            const sml_rec * cb = sml_findf(m, "blk.%d.ssm_conv.bias", (int) L);
            const sml_rec * db = sml_findf(m, "blk.%d.ssm_dt_b.weight", (int) L);
            const sml_rec * al = sml_findf(m, "blk.%d.ssm_a_log.weight", (int) L);
            const sml_rec * dd = sml_findf(m, "blk.%d.ssm_d.weight", (int) L);
            const sml_rec * nw = sml_findf(m, "blk.%d.ssm_norm.weight", (int) L);
            // Falcon-H1 gates without a norm (mamba_rms_norm false): no ssm_norm then
            if (!cw || !db || !al || !dd || !s->ssm_out) {
                fprintf(stderr, "layer %d: Mamba-2 mixer incomplete\n", (int) L); return 0;
            }
            s->ssm_conv = (const float *) sml_ptr(m, cw->off);
            s->ssm_conv_b = cb ? (const float *) sml_ptr(m, cb->off) : NULL;
            s->ssm_dt_b = (const float *) sml_ptr(m, db->off);
            s->ssm_a_log = (const float *) sml_ptr(m, al->off);
            s->ssm_d = (const float *) sml_ptr(m, dd->off);
            s->ssm_nw = nw ? (const float *) sml_ptr(m, nw->off) : NULL;
            if (!c->hyb && !(c->hyb = calloc(h->n_embd, sizeof(float)))) return 0;   // Falcon-H1
            s->ssm_heads = (int) ((uint64_t) al->rows * al->cols);
            s->ssm_inner = (int) s->ssm_out->cols;
            // a full-precision tensor is stored flat (one row): the conv width is
            // what in_proj carries besides z and dt, the kernel what is left
            {
                const int cd_ = (int) s->ssm_in->rows - s->ssm_inner - s->ssm_heads;
                const uint64_t tot = (uint64_t) cw->rows * cw->cols;
                s->ssm_convdim = cd_;
                s->ssm_K = cd_ > 0 && tot % (uint64_t) cd_ == 0 ? (int) (tot / (uint64_t) cd_) : 0;
            }
            const int P_ = s->ssm_heads ? s->ssm_inner / s->ssm_heads : 0;
            const int N_ = (int) h->ssm_dstate;
            if (!P_ || !N_ || s->ssm_K < 1 || s->ssm_inner % s->ssm_heads) {
                fprintf(stderr, "layer %d: Mamba-2 shapes do not fit\n", (int) L); return 0;
            }
            if (!c->ssm_state) {
                c->ssm_state = calloc(h->n_layer, sizeof(float *));
                c->ssm_cst = calloc(h->n_layer, sizeof(float *));
                if (!c->ssm_state || !c->ssm_cst) return 0;
            }
            c->ssm_state[L] = calloc((size_t) s->ssm_heads * P_ * N_, sizeof(float));
            c->ssm_cst[L] = calloc((size_t) (s->ssm_K > 1 ? s->ssm_K - 1 : 1) * s->ssm_convdim, sizeof(float));
            float * pr = realloc(c->ssm_proj, (size_t) s->ssm_in->rows * sizeof(float));
            float * xb = realloc(c->ssm_xbc, (size_t) s->ssm_convdim * sizeof(float));
            float * yy = realloc(c->ssm_y, (size_t) s->ssm_inner * sizeof(float));
            if (!c->ssm_state[L] || !c->ssm_cst[L] || !pr || !xb || !yy) return 0;
            c->ssm_proj = pr; c->ssm_xbc = xb; c->ssm_y = yy;
            c->step_only = 1;
        }
        s->q = sml_findf(m, "blk.%d.attn_q.weight", (int) L);
        s->mla_kva = sml_findf(m, "blk.%d.attn_kv_a_mqa.weight", (int) L);
        if (s->mla_kva) {
            const sml_rec * kn = sml_findf(m, "blk.%d.attn_kv_a_norm.weight", (int) L);
            s->mla_kvb = sml_findf(m, "blk.%d.attn_kv_b.weight", (int) L);
            s->mla_qa = sml_findf(m, "blk.%d.attn_q_a.weight", (int) L);
            s->mla_qb = sml_findf(m, "blk.%d.attn_q_b.weight", (int) L);
            const sml_rec * qn = sml_findf(m, "blk.%d.attn_q_a_norm.weight", (int) L);
            const int hd_ = hd, nh_ = (int) h->n_head, rot_ = (int) h->n_rot;
            if (!kn || !s->mla_kvb || !(s->q || (s->mla_qa && s->mla_qb && qn)) || !rot_ || rot_ >= hd_ ||
                !(h->version >= 5 && (h->arch_bits & SML_ARCH_MLA))) {
                fprintf(stderr, "layer %d: latent attention is missing a part\n", (int) L);
                return 0;
            }
            s->mla_kva_norm = (const float *) sml_ptr(m, kn->off);
            s->mla_qa_norm = qn ? (const float *) sml_ptr(m, qn->off) : NULL;
            s->mla_rkv = (int) (kn->rows * kn->cols);
            s->mla_rq = s->mla_qa ? (int) s->mla_qa->rows : 0;
            s->mla_vd = (int) s->mla_kvb->rows / nh_ - (hd_ - rot_);
            if ((int) s->mla_kva->rows != s->mla_rkv + rot_ || s->mla_vd <= 0 || s->mla_vd > hd_) {
                fprintf(stderr, "layer %d: latent attention shapes do not agree\n", (int) L);
                return 0;
            }
            const int need = s->mla_rq > s->mla_rkv + rot_ ? s->mla_rq : s->mla_rkv + rot_;
            if (!c->mla_a) {
                c->mla_a = calloc((size_t) need + 64, sizeof(float));
                c->mla_an = calloc((size_t) need + 64, sizeof(float));
                c->mla_kv = calloc((size_t) s->mla_kvb->rows, sizeof(float));
                if (!c->mla_a || !c->mla_an || !c->mla_kv) return 0;
            }
        }
        s->k = sml_findf(m, "blk.%d.attn_k.weight", (int) L);
        s->v = sml_findf(m, "blk.%d.attn_v.weight", (int) L);
        s->o = sml_findf(m, "blk.%d.attn_output.weight", (int) L);
        s->gate = sml_findf(m, "blk.%d.ffn_gate.weight", (int) L);
        s->up   = sml_findf(m, "blk.%d.ffn_up.weight",   (int) L);
        s->down = sml_findf(m, "blk.%d.ffn_down.weight", (int) L);
        const sml_rec * b;
        b = sml_findf(m, "blk.%d.attn_q.bias", (int) L);
        s->q_bias = b ? (const float *) sml_ptr(m, b->off) : NULL;
        b = sml_findf(m, "blk.%d.attn_k.bias", (int) L);
        s->k_bias = b ? (const float *) sml_ptr(m, b->off) : NULL;
        b = sml_findf(m, "blk.%d.attn_v.bias", (int) L);
        s->v_bias = b ? (const float *) sml_ptr(m, b->off) : NULL;
        b = sml_findf(m, "blk.%d.attn_output.bias", (int) L);
        s->o_bias = b ? (const float *) sml_ptr(m, b->off) : NULL;
        {
            const sml_rec * qn = sml_findf(m, "blk.%d.attn_q_norm.weight", (int) L);
            const sml_rec * kn = sml_findf(m, "blk.%d.attn_k_norm.weight", (int) L);
            s->q_norm = qn ? (const float *) sml_ptr(m, qn->off) : NULL;
            s->k_norm = kn ? (const float *) sml_ptr(m, kn->off) : NULL;
            const sml_rec * qb = sml_findf(m, "blk.%d.attn_q_norm.bias", (int) L);
            const sml_rec * kb = sml_findf(m, "blk.%d.attn_k_norm.bias", (int) L);
            s->q_norm_b = qb ? (const float *) sml_ptr(m, qb->off) : NULL;
            s->k_norm_b = kb ? (const float *) sml_ptr(m, kb->off) : NULL;
            s->q_norm_n = qn ? (int) qn->cols : 0;
            s->k_norm_n = kn ? (int) kn->cols : 0;
            const sml_rec * sk = sml_findf(m, "blk.%d.attn_sinks.weight", (int) L);
            s->sinks = sk && (int) ((uint64_t) sk->rows * sk->cols) == (int) h->n_head
                       ? (const float *) sml_ptr(m, sk->off) : NULL;
            if (sk && !s->sinks) { fprintf(stderr, "layer %d: one attention sink per head expected\n", (int) L); return 0; }
            s->qk_heads = h->version >= 5 && (h->arch_bits & SML_ARCH_QKNORM_HEADS);
            if (s->qk_heads) {                    // stored [heads, head_dim]: one row per head
                s->q_norm_n = qn ? (int) (qn->rows * qn->cols) : 0;
                s->k_norm_n = kn ? (int) (kn->rows * kn->cols) : 0;
                if ((qn && s->q_norm_n != (int) h->n_head * hd) || (kn && s->k_norm_n != (int) h->n_head_kv * hd)) {
                    fprintf(stderr, "layer %d: per-head Q/K norms must hold one row per head\n", (int) L);
                    return 0;
                }
            }
            // A norm whose length is neither one head nor the whole projection
            // is a layout this engine does not know. Refuse it: every other
            // architecture mistake here produced fluent nonsense, not an error.
            if ((qn && s->q_norm_n != hd && s->q_norm_n != (int) h->n_head * hd) ||
                (kn && s->k_norm_n != hd && s->k_norm_n != (int) h->n_head_kv * hd)) {
                fprintf(stderr, "layer %d: Q/K norm of length %d/%d fits neither a head "
                                "(%d) nor the projection\n", (int) L, s->q_norm_n, s->k_norm_n, hd);
                return 0;
            }
        }
        {
            const sml_rec * rt = sml_findf(m, "blk.%d.ffn_gate_inp.weight", (int) L);
            s->router = rt ? (const float *) sml_ptr(m, rt->off) : NULL;
            s->n_exp = 0;
            if (rt) {
                // Count experts from the router's SIZE, not its row count: a
                // full-precision tensor is recorded as one flat row, so rows said 1
                // on a 32-expert model (caught on the first smoke build).
                const uint64_t rn = (uint64_t) rt->rows * rt->cols;
                if (rn % h->n_embd) {
                    fprintf(stderr, "layer %d: router of %llu values is not n_embd wide\n",
                            (int) L, (unsigned long long) rn);
                    return 0;
                }
                const int ne = (int) (rn / h->n_embd);
                s->eg = calloc((size_t) ne, sizeof *s->eg);
                s->eu = calloc((size_t) ne, sizeof *s->eu);
                s->ed = calloc((size_t) ne, sizeof *s->ed);
                if (!s->eg || !s->eu || !s->ed) return 0;
                for (int e = 0; e < ne; e++) {
                    char nm[SML_NAME_LEN];
                    snprintf(nm, sizeof nm, "blk.%d.ffn_gate.%d.weight", (int) L, e);
                    s->eg[e] = sml_find(m, nm);
                    snprintf(nm, sizeof nm, "blk.%d.ffn_up.%d.weight", (int) L, e);
                    s->eu[e] = sml_find(m, nm);
                    snprintf(nm, sizeof nm, "blk.%d.ffn_down.%d.weight", (int) L, e);
                    s->ed[e] = sml_find(m, nm);
                    // an expert with a missing tensor is refused, not skipped:
                    // skipping one would renormalize over the wrong set
                    // Nemotron-H's experts are ungated (up, relu^2, down): every expert or none
                    if (!s->eu[e] || !s->ed[e] || (!s->eg[e]) != (!s->eg[0])) {
                        fprintf(stderr, "layer %d: expert %d is incomplete\n", (int) L, e);
                        return 0;
                    }
                }
                s->n_exp = ne;
                {
                    const sml_rec * rb = sml_findf(m, "blk.%d.ffn_gate_inp.bias", (int) L);
                    s->router_b = rb ? (const float *) sml_ptr(m, rb->off) : NULL;
                    char nm[SML_NAME_LEN];
                    snprintf(nm, sizeof nm, "blk.%d.ffn_gate.0.bias", (int) L);
                    if (sml_find(m, nm)) {
                        s->egb = calloc((size_t) ne, sizeof *s->egb);
                        s->eub = calloc((size_t) ne, sizeof *s->eub);
                        s->edb = calloc((size_t) ne, sizeof *s->edb);
                        if (!s->egb || !s->eub || !s->edb) return 0;
                        for (int e = 0; e < ne; e++) {
                            const sml_rec * r3[3];
                            snprintf(nm, sizeof nm, "blk.%d.ffn_gate.%d.bias", (int) L, e); r3[0] = sml_find(m, nm);
                            snprintf(nm, sizeof nm, "blk.%d.ffn_up.%d.bias", (int) L, e);   r3[1] = sml_find(m, nm);
                            snprintf(nm, sizeof nm, "blk.%d.ffn_down.%d.bias", (int) L, e); r3[2] = sml_find(m, nm);
                            if (!r3[0] || !r3[1] || !r3[2]) {
                                fprintf(stderr, "layer %d: expert %d has only some of its biases\n", (int) L, e);
                                return 0;
                            }
                            s->egb[e] = (const float *) sml_ptr(m, r3[0]->off);
                            s->eub[e] = (const float *) sml_ptr(m, r3[1]->off);
                            s->edb[e] = (const float *) sml_ptr(m, r3[2]->off);
                        }
                    }
                }
                s->shg = sml_findf(m, "blk.%d.ffn_gate_shexp.weight", (int) L);
                s->shu = sml_findf(m, "blk.%d.ffn_up_shexp.weight", (int) L);
                s->shd = sml_findf(m, "blk.%d.ffn_down_shexp.weight", (int) L);
                if (!(s->shu && s->shd)) s->shg = s->shu = s->shd = NULL;   // shg NULL: ungated
                {
                    const sml_rec * sgt = sml_findf(m, "blk.%d.ffn_gate_inp_shexp.weight", (int) L);
                    s->shgate = sgt ? (const float *) sml_ptr(m, sgt->off) : NULL;
                    const sml_rec * eb = sml_findf(m, "blk.%d.exp_probs_b.weight", (int) L);
                    s->exp_bias = eb && (int) ((uint64_t) eb->rows * eb->cols) >= ne
                                ? (const float *) sml_ptr(m, eb->off) : NULL;
                }
                // Experts need not be the dense FFN's width (Qwen2-MoE: 1408
                // against 5632), so their scratch is sized by the widest one.
                int wmax = (int) h->n_ff;
                for (int e = 0; e < ne; e++) if ((int) s->eu[e]->rows > wmax) wmax = (int) s->eu[e]->rows;
                if (s->shu && (int) s->shu->rows > wmax) wmax = (int) s->shu->rows;
                if (wmax > c->moe_ff) {
                    free(c->moe_g); free(c->moe_u);
                    c->moe_g = calloc((size_t) wmax, sizeof(float));
                    c->moe_u = calloc((size_t) wmax, sizeof(float));
                    if (!c->moe_g || !c->moe_u) return 0;
                    c->moe_ff = wmax;
                }
                if (!c->moe_logit || ne > c->moe_ne) {
                    free(c->moe_logit); free(c->moe_s); free(c->moe_c);
                    c->moe_logit = calloc((size_t) ne, sizeof(float));
                    c->moe_s = calloc((size_t) ne, sizeof(float));
                    c->moe_c = calloc((size_t) ne, sizeof(float));
                    if (!c->moe_s || !c->moe_c) return 0;
                    c->moe_ne = ne;
                    if (!c->moe_tmp) c->moe_tmp = calloc(h->n_embd, sizeof(float));
                    if (!c->moe_acc) c->moe_acc = calloc(h->n_embd, sizeof(float));
                    if (!c->moe_logit || !c->moe_tmp || !c->moe_acc) return 0;
                }
                // experts run batched now (moe_ffn_batch): no step_only
            }
        }
        b = sml_findf(m, "blk.%d.ffn_up.bias", (int) L);
        s->up_bias = b ? (const float *) sml_ptr(m, b->off) : NULL;
        b = sml_findf(m, "blk.%d.ffn_down.bias", (int) L);
        s->down_bias = b ? (const float *) sml_ptr(m, b->off) : NULL;
        b = sml_findf(m, "blk.%d.ffn_gate.bias", (int) L);
        s->gate_bias = b ? (const float *) sml_ptr(m, b->off) : NULL;
        // Resolve this layer's shared-activation groups. A declared group wins;
        // otherwise fall back to what the tensor names imply.
        s->n_agrp = 0; s->n_fgrp = 0;
        for (uint32_t gi = 0; gi < m->n_groups; gi++) {
            const sml_group * g = &m->groups[gi];
            if (g->kind != SML_GROUP_KIND_ACT) continue;
            const sml_rec * first = &m->recs[g->member[0]];
            const sml_rec ** dst = NULL;
            int * cnt = NULL;
            if (first == s->q)      { dst = s->agrp; cnt = &s->n_agrp; }
            else if (first == s->gate) { dst = s->fgrp; cnt = &s->n_fgrp; }
            if (!dst) continue;
            for (uint32_t k = 0; k < g->n; k++) dst[k] = &m->recs[g->member[k]];
            *cnt = (int) g->n;
            c->grp_from_file++;
        }
        if (!s->n_agrp && s->q && s->k && s->v) {
            s->agrp[0] = s->q; s->agrp[1] = s->k; s->agrp[2] = s->v; s->n_agrp = 3;
            c->grp_inferred++;
        }
        if (!s->n_fgrp && s->gate && s->up) {
            s->fgrp[0] = s->gate; s->fgrp[1] = s->up; s->n_fgrp = 2;
            c->grp_inferred++;
        }
        b = sml_findf(m, "blk.%d.attn_norm.bias", (int) L);
        s->attn_norm_b = b ? (const float *) sml_ptr(m, b->off) : NULL;
        b = sml_findf(m, "blk.%d.ffn_norm.bias", (int) L);
        s->ffn_norm_b = b ? (const float *) sml_ptr(m, b->off) : NULL;
        if (s->ffn_norm == s->attn_norm) s->ffn_norm_b = s->attn_norm_b;
    }

    // One scratch row per OpenMP thread, for the batched attention's scores.
    // Allocating it inside the parallel region cost a heap allocation per
    // thread per layer per chunk.
#ifdef _OPENMP
    c->n_tscr = omp_get_max_threads() > n_threads ? omp_get_max_threads() : n_threads;
#else
    c->n_tscr = 1;
#endif
    c->tscr_len = n_ctx > wide ? n_ctx : wide;
    c->tscr = malloc((size_t) c->n_tscr * c->tscr_len * sizeof(float));

    const int B = SML_MAX_BATCH;
    c->bx   = calloc((size_t) B * h->n_embd, sizeof(float));
    c->bh   = calloc((size_t) B * wide_a, sizeof(float));   // also attention output
    c->bq   = calloc((size_t) B * h->n_head * hd, sizeof(float));
    c->bk   = calloc((size_t) B * nkv, sizeof(float));
    c->bv   = calloc((size_t) B * nkv, sizeof(float));
    c->bff  = calloc((size_t) B * h->n_ff, sizeof(float));
    c->bff2 = calloc((size_t) B * h->n_ff, sizeof(float));
    c->bout = calloc((size_t) B * wide, sizeof(float));
    if (c->agate) c->bgate = calloc((size_t) B * n_attn, sizeof(float));
    for (int b = 0; b < B; b++) {
        c->bact[b].q     = malloc((size_t) wide);
        c->bact[b].scale = malloc((size_t) wide / SML_GROUP * sizeof(float));
        c->bact[b].sums  = malloc((size_t) wide / SML_GROUP * sizeof(int32_t));
    }

    c->act.q     = malloc((size_t) wide);
    c->act.scale = malloc((size_t) wide / SML_GROUP * sizeof(float));
    c->act.sums  = malloc((size_t) wide / SML_GROUP * sizeof(int32_t));

    if (!c->k_cache || !c->v_cache || !c->logits || !c->act.q || !c->tscr
        || !c->rope_cos || !c->rope_sin || !c->layers) {
        fprintf(stderr, "sml: out of memory sizing context (n_ctx=%d)\n", n_ctx);
        sml_ctx_free(c);
        return 0;
    }
#ifdef _OPENMP
    omp_set_num_threads(n_threads);
#else
    // A build without -fopenmp runs every kernel on ONE core whatever --threads
    // says. The Oct 10 binaries were built that way (llvm-mingw, no libomp) and
    // lost 0.28x to llama.cpp before anyone looked at the thread scaling.
    if (n_threads > 1)
        fprintf(stderr, "sml: WARNING built without OpenMP; %d threads requested, 1 used. "
                        "Rebuild with gcc -fopenmp.\n", n_threads);
#endif
    sml_pool_init(n_threads);
    repack_il(c);
    return 1;
}

void sml_ctx_repack(sml_ctx * c) {
    const int was = g_repack_on;
    g_repack_on = 1;
    repack_il(c);
    g_repack_on = was;
}

void sml_ctx_free(sml_ctx * c) {
    if (c->rlp && c->rlp == g_rlp.rlp && --g_rlp.refs == 0) {
        free((void *) g_rlp.rlp); free(g_rlp.mem);
        memset(&g_rlp, 0, sizeof g_rlp);
    }
    if (c->il && c->il == g_il.il) {
        if (--g_il.refs == 0) {
            free((void *) g_il.il); free(g_il.mem);
            memset(&g_il, 0, sizeof g_il);
        }
    } else {
        free((void *) c->il); free(c->il_mem);
    }
    free(c->hist);
    free(c->k_cache); free(c->v_cache);
    free(c->x); free(c->xb); free(c->xg); free(c->xr);
    free(c->q); free(c->k); free(c->v);
    free(c->att); free(c->hb); free(c->hb2); free(c->logits);
    free(c->moe_logit); free(c->moe_tmp); free(c->moe_acc);
    free(c->moe_g); free(c->moe_u); free(c->moe_s); free(c->moe_c);
    free(c->mb_x); free(c->mb_g); free(c->mb_u); free(c->mb_o); free(c->mb_out);
    free(c->mb_w); free(c->mb_sg); free(c->mb_sel); free(c->mb_row);
    if (c->sc_hist) {
        for (uint32_t L = 0; L < c->model->hdr->n_layer; L++) free(c->sc_hist[L]);
        free(c->sc_hist);
    }
    free(c->sc_buf); free(c->sc_y);
    if (c->ssm_state) {
        for (uint32_t L = 0; L < c->model->hdr->n_layer; L++) { free(c->ssm_state[L]); free(c->ssm_cst[L]); }
        free(c->ssm_state); free(c->ssm_cst);
    }
    free(c->ssm_proj); free(c->ssm_xbc); free(c->ssm_y); free(c->agate);
    if (c->layers)
        for (uint32_t L = 0; L < c->model->hdr->n_layer; L++) {
            free((void *) c->layers[L].eg); free((void *) c->layers[L].eu);
            free((void *) c->layers[L].ed);
        }
    free(c->act.q); free(c->act.scale); free(c->act.sums);
    free(c->rope_cos); free(c->rope_sin); free(c->layers);
    free(c->layer_win); free(c->layer_rope); free(c->rope_cos_l); free(c->rope_sin_l);
    free(c->alibi);
    free(c->bx); free(c->bh); free(c->bq); free(c->bk); free(c->bv);
    free(c->bff); free(c->bff2); free(c->bout); free(c->bgate); free(c->tscr);
    for (int b = 0; b < SML_MAX_BATCH; b++) {
        free(c->bact[b].q); free(c->bact[b].scale); free(c->bact[b].sums);
    }
    memset(c, 0, sizeof *c);
}

// ------------------------------------------------------------------ forward

// ---------------------------------------------------------- batched forward

// Same maths as matvec(), but nb activations share one pass over the weights.
// `y` is [nb][rows]; `xs` is [nb][xstride].
// Batched twin of issue_rows, and the same single issue point for the layout
// decision. There was none here: matvec_batch indexed codes_off[lv] straight
// out of the ladder, so a FLAT tensor -- which populates only codes_off[0], and
// [1] at 6 bits -- read an unmapped plane and took the process down. Every
// layout handled in issue_rows must be handled here too.
// SML_PROMPT_SB (default 1): the batched body -- prompt reading and speculative
// verification -- quantizes with one activation scale per 256 values and runs
// sml_matvec_batch_flat4_sb. 0 keeps per-32 activations, bit-exact with stepping.
static int prompt_sb(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("SML_PROMPT_SB");
        on = e ? atoi(e) != 0 : 1;
    }
    return on;
}

static inline void batch_flat4(float * y, const uint8_t * codes, const uint16_t * sd,
                               const uint8_t * sq, const uint16_t * md, const uint8_t * mq,
                               const sml_act * a, int nb, int nrows, int lo, int hi, int n) {
    if (prompt_sb()) sml_matvec_batch_flat4_sb(y, codes, sd, sq, md, mq, a, nb, nrows, lo, hi, n);
    else             sml_matvec_batch_flat4(y, codes, sd, sq, md, mq, a, nb, nrows, lo, hi, n);
}

static inline void issue_rows_batch(sml_ctx * c, float * y, const sml_rec * r,
                                    const uint16_t * sd, const uint8_t * sq,
                                    const uint16_t * md, const uint8_t * mq,
                                    int nb, int nrows, int lo, int hi, int n,
                                    int depth, int nlev, float cbias) {
    if (r->layout == SML_LAYOUT_FLAT) {
        if (r->nbits == 3 || r->nbits == 5 || r->nbits == 7) {
            const uint8_t * p0 = sml_ptr(c->model, r->codes_off[0]), * p1 = sml_ptr(c->model, r->codes_off[1]);
            if (r->nbits == 3) {
                sml_matvec_batch(y, p0, sd, sq, md, mq, c->bact, nb, nrows, lo, hi, n, 0, 0.0f);
                sml_matvec_bits1(y, p1, sd, sq, c->bact, nb, nrows, lo, hi, n, 2);
            } else if (r->nbits == 5) {
                sml_matvec_batch_flat4(y, p0, sd, sq, md, mq, c->bact, nb, nrows, lo, hi, n);
                sml_matvec_bits1(y, p1, sd, sq, c->bact, nb, nrows, lo, hi, n, 4);
            } else {
                sml_matvec_batch_flat4(y, p0, sd, sq, md, mq, c->bact, nb, nrows, lo, hi, n);
                sml_matvec_batch(y, p1, sd, sq, NULL, NULL, c->bact, nb, nrows, lo, hi, n, 4, 0.0f);
                sml_matvec_bits1(y, sml_ptr(c->model, r->codes_off[2]), sd, sq, c->bact, nb, nrows, lo, hi, n, 6);
            }
            return;
        }
        if (r->nbits == 2) {
            sml_matvec_batch(y, sml_ptr(c->model, r->codes_off[0]), sd, sq, md, mq, c->bact, nb, nrows, lo, hi, n, 0, 0.0f);
            return;
        }
        if (r->nbits == 8) {
            sml_matvec_batch_flat8(y, sml_ptr(c->model, r->codes_off[0]), sml_ptr(c->model, r->codes_off[1]),
                                   sd, sq, md, mq, c->bact, nb, nrows, lo, hi, n);
            return;
        }
        if (r->nbits == 6 && prompt_sb() && flat6_fused()) {
            sml_matvec_batch_flat6_sb(y, sml_ptr(c->model, r->codes_off[1]),
                                      sml_ptr(c->model, r->codes_off[0]), sd, sq, md, mq,
                                      c->bact, nb, nrows, lo, hi, n);
            return;
        }
        if (r->nbits == 6) {
            // base(4) + 16*extra(2), linear -- split exactly as the row path
            // splits it, so batch and sequential cannot drift apart.
            sml_matvec_batch(y, sml_ptr(c->model, r->codes_off[0]), sd, sq,
                             md, mq, c->bact, nb, nrows, lo, hi, n, 4, 0.0f);
            batch_flat4(y, sml_ptr(c->model, r->codes_off[1]), sd, sq,
                        NULL, NULL, c->bact, nb, nrows, lo, hi, n);
            return;
        }
        batch_flat4(y, sml_ptr(c->model, r->codes_off[0]), sd, sq,
                    md, mq, c->bact, nb, nrows, lo, hi, n);
        return;
    }
    for (int lv = 0; lv < depth; lv++)
        sml_matvec_batch(y, sml_ptr(c->model, r->codes_off[lv]), sd, sq,
                         lv == depth - 1 ? md : NULL,
                         lv == depth - 1 ? mq : NULL,
                         c->bact, nb, nrows, lo, hi, n,
                         (nlev - 1 - lv) * SML_LEVEL_BITS,
                         lv == depth - 1 ? cbias : 0.0f);
}

static int batch_chunk_rows(void) {
    // Prompt reading measured 80 tok/s at 4 threads and at 6, with the kernel
    // time flat between them, while the reference engine went 134 -> 165. A
    // fixed slice per thread makes every call wait for its slowest thread; rows
    // handed out in chunks let a thread that shares its core with something
    // else simply take fewer. SML_CHUNK_ROWS sets the chunk; 0 = fixed slices.
    static int chunk_rows = -1;
    if (chunk_rows < 0) {
        const char * e = getenv("SML_CHUNK_ROWS");
        chunk_rows = e ? atoi(e) : 32;
        if (chunk_rows < 0) chunk_rows = 0;
    }
    return chunk_rows;
}

// Every tensor in rs[] reads the same nb activations in the same column order
// (q/k/v, gate/up, or one tensor alone). ONE parallel region does it all: the
// slots are gathered and quantized across threads, one barrier, then each
// tensor's rows go out in dynamic chunks with no barrier between tensors.
// The prep used to run serially before the region: 9% of a 109-token 1B prompt,
// which by Amdahl alone capped six threads near 4.1x while the reference engine
// scaled 4.3x. (A separate parallel region for it, with a malloc per thread,
// was a measured regression: the dispatch outweighed a few thousand floats.)
#define SML_BATCH_GROUP_MAX 3
typedef struct {
    sml_ctx * c; const sml_rec * const * rs; float * const * ys; int cnt, nb, n, depth, nlev;
    const uint16_t * const * sd; const uint8_t * const * sq;
    const uint16_t * const * md; const uint8_t * const * mq; const float * cbias;
    const int * rows; const int * chunk; const int * first;
} bg_job;

static void bg_chunk(void * p, int k) {
    const bg_job * j = (const bg_job *) p;
    sml_ctx * c = j->c;
    int i = 0;
    while (i + 1 < j->cnt && k >= j->first[i + 1]) i++;
    const int lo = (k - j->first[i]) * j->chunk[i];
    const int hi = lo + j->chunk[i] < j->rows[i] ? lo + j->chunk[i] : j->rows[i];
    const uint8_t * rp = (c->batch_decode == 1 && c->rlp) ? c->rlp[j->rs[i] - c->model->recs] : NULL;
    if (rp && !(lo & 7) && (hi == j->rows[i] || !(hi & 7))) {
        sml_rlp_matvec(j->ys[i], j->rows[i], rp, (int) j->rs[i]->nbits, c->bact, j->nb, lo, hi, j->n);
        return;
    }
    for (int r0 = lo; r0 < hi; r0 += 8) {
        const int r1 = r0 + 8 < hi ? r0 + 8 : hi;
        for (int b = 0; b < j->nb; b++)
            issue_rows_act(c, &c->bact[b], j->ys[i] + (size_t) b * j->rows[i], j->rs[i],
                           j->sd[i], j->sq[i], j->md[i], j->mq[i], r0, r1, j->n, j->depth, j->nlev,
                           j->cbias[i]);
    }
}

static void batch_group(sml_ctx * c, const sml_rec * const rs[], float * const ys[], int cnt,
                        const float * xs, int xstride, int nb, int depth) {
    for (int i = 0; i < cnt; i++) {
        if (rs[i]->kind != SML_KIND_F32) continue;
        for (int j = 0; j < cnt; j++) {
            const int rows = (int) rs[j]->rows;
            for (int b = 0; b < nb; b++)
                matvec(c, ys[j] + (size_t) b * rows, rs[j],
                       xs + (size_t) b * xstride, depth);
        }
        return;
    }
    const int n    = (int) rs[0]->cols;
    const int nlev = (int) c->model->hdr->n_levels;
    if (depth < 1) depth = 1;
    if (depth > nlev) depth = nlev;
    const int32_t * perm = (const int32_t *) sml_ptr(c->model, rs[0]->perm_off);

    const uint16_t * sd[SML_BATCH_GROUP_MAX], * md[SML_BATCH_GROUP_MAX];
    const uint8_t  * sq[SML_BATCH_GROUP_MAX], * mq[SML_BATCH_GROUP_MAX];
    float cbias[SML_BATCH_GROUP_MAX];
    int rows[SML_BATCH_GROUP_MAX], chunk[SML_BATCH_GROUP_MAX], nch[SML_BATCH_GROUP_MAX];
    const int crows = batch_chunk_rows();
#ifdef _OPENMP
    const int nth = omp_get_max_threads();
#else
    const int nth = 1;
#endif
    for (int i = 0; i < cnt; i++) {
        const sml_rec * r = rs[i];
        rows[i]  = (int) r->rows;
        sd[i]    = (const uint16_t *) sml_ptr(c->model, r->sd_off);
        sq[i]    = (const uint8_t  *) sml_ptr(c->model, r->sq_off);
        md[i]    = r->mq_n ? (const uint16_t *) sml_ptr(c->model, r->md_off) : NULL;
        mq[i]    = r->mq_n ? (const uint8_t  *) sml_ptr(c->model, r->mq_off) : NULL;
        cbias[i] = mq[i] ? sml_level_mid(depth, nlev) : sml_level_bias(depth, nlev);
        chunk[i] = crows > 0 ? crows : (rows[i] + nth - 1) / nth;
        nch[i]   = (rows[i] + chunk[i] - 1) / chunk[i];
        memset(ys[i], 0, (size_t) nb * rows[i] * sizeof(float));
    }

    const double tm = sml_now();
    // Several conversations: a handful of slots, so the prep runs here and the
    // rows go to the resident pool -- an OpenMP region costs ~30 us to fork on
    // this toolchain, and a 2-slot step had ~100 of them.
    if (c->batch_decode && use_pool()) {
        for (int b = 0; b < nb; b++) {
            sml_gather(c->xg, xs + (size_t) b * xstride, perm, n);
            if (c->batch_decode == 1) sml_quantize_act_sb(&c->bact[b], c->xg, n);
            else                      sml_quantize_act(&c->bact[b], c->xg, n);
        }
        int first[SML_BATCH_GROUP_MAX], tot = 0;
        for (int i = 0; i < cnt; i++) { first[i] = tot; tot += nch[i]; }
        bg_job j = { c, rs, ys, cnt, nb, n, depth, nlev, sd, sq, md, mq, cbias, rows, chunk, first };
        sml_pool_for(tot, bg_chunk, &j);
    } else
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        float * xg = c->tscr + (size_t) omp_get_thread_num() * c->tscr_len;
#pragma omp for schedule(static)
#else
        float * xg = c->xg;
#endif
        for (int b = 0; b < nb; b++) {
            sml_gather(xg, xs + (size_t) b * xstride, perm, n);
            if (c->batch_decode == 1 || (prompt_sb() && !c->batch_decode))
                sml_quantize_act_sb(&c->bact[b], xg, n);
            else
                sml_quantize_act(&c->bact[b], xg, n);
        }
        // the implicit barrier above: every slot is ready before any row reads it
        for (int i = 0; i < cnt; i++) {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1) nowait
#endif
            for (int k = 0; k < nch[i]; k++) {
                const int lo = k * chunk[i];
                const int hi = lo + chunk[i] < rows[i] ? lo + chunk[i] : rows[i];
                const uint8_t * rp = (c->batch_decode == 1 && c->rlp) ? c->rlp[rs[i] - c->model->recs] : NULL;
                if (rp && !(lo & 7) && (hi == rows[i] || !(hi & 7))) {
                    sml_rlp_matvec(ys[i], rows[i], rp, (int) rs[i]->nbits, c->bact, nb, lo, hi, n);
                    continue;
                }
                if (c->batch_decode) {
                    // Several conversations: each slot through the decode kernel
                    // itself, 8 rows at a time, so a tile's weights come from
                    // memory once and from L1 for every other slot -- and each
                    // slot computes exactly what it would decoding alone.
                    for (int r0 = lo; r0 < hi; r0 += 8) {
                        const int r1 = r0 + 8 < hi ? r0 + 8 : hi;
                        for (int b = 0; b < nb; b++)
                            issue_rows_act(c, &c->bact[b], ys[i] + (size_t) b * rows[i], rs[i],
                                           sd[i], sq[i], md[i], mq[i], r0, r1, n, depth, nlev,
                                           cbias[i]);
                    }
                    continue;
                }
                issue_rows_batch(c, ys[i], rs[i], sd[i], sq[i], md[i], mq[i], nb, rows[i],
                                 lo, hi, n, depth, nlev, cbias[i]);
            }
        }
    }
    sml_t_matvec += sml_now() - tm;
    // SML_PROF_BATCH=1: one line per batched matvec call, with its GMAC/s --
    // how pythia's 6-bit tensors were found running at a fifth of the 4-bit rate
    static int prof = -1;
    if (prof < 0) prof = getenv("SML_PROF_BATCH") != NULL;
    if (prof) {
        const double tt = sml_now() - tm;
        long long mac = 0;
        for (int i = 0; i < cnt; i++) mac += (long long) rows[i] * n * nb;
        fprintf(stderr, "bg cnt=%d n=%d rows0=%d nb=%d bits=%d %.3f ms %.1f GMAC/s\n",
                cnt, n, rows[0], nb, (int) rs[0]->nbits, tt * 1e3, mac / tt / 1e9);
    }
    for (int i = 0; i < cnt; i++) {
        const sml_rec * r = rs[i];
        if (r->layout == SML_LAYOUT_FLAT)
            c->bytes_moved += (uint64_t) rows[i] * (uint64_t) n * r->nbits / 8;
        else
            for (int lv = 0; lv < depth; lv++) c->bytes_moved += sml_level_bytes(rows[i], n);
        c->bytes_moved += sml_scale_bytes(rows[i], n) * (r->mq_n ? 2 : 1);
    }
}

static void matvec_batch(sml_ctx * c, float * y, const sml_rec * r,
                         const float * xs, int xstride, int nb, int depth) {
    const sml_rec * const rs[1] = { r };
    float * const ys[1] = { y };
    batch_group(c, rs, ys, 1, xs, xstride, nb, depth);
}

// One (slot, head) of several conversations: its own cache, its own score row.
typedef struct {
    sml_ctx * c; sml_ctx * const * kv; const int * pos;
    int L, hd, kvd, rep, swa, n_head, n_attn; float inv, cap; const float * sinks;
} ms_attn_job;

static void ms_attn(void * p, int k) {
    const ms_attn_job * j = (const ms_attn_job *) p;
    const int b = k / j->n_head, hh = k % j->n_head;
    const sml_ctx * o = j->kv[b];
    const int ps = j->pos[b];
    const int t0 = (j->swa && ps >= j->swa) ? ps - j->swa + 1 : 0;
    const size_t base = (size_t) j->L * o->n_ctx * j->kvd + (size_t) (hh / j->rep) * j->hd;
    attn_row(j->c->bq + (size_t) b * j->n_head * j->hd + (size_t) hh * j->hd,
             o->k_cache + base, o->v_cache + base, (size_t) j->kvd,
             t0, ps, j->hd, j->inv, j->cap, j->c->alibi ? &j->c->alibi[hh] : NULL,
             j->sinks ? &j->sinks[hh] : NULL, j->kv[b]->att + (size_t) hh * o->n_ctx,
             j->c->bh + (size_t) b * j->n_attn + (size_t) hh * j->hd);
}

static void moe_ffn_batch(sml_ctx * c, const sml_layer * ly, int nb, int depth);

// The batched body. Slot b is token tokens[b] at position pos[b] of the
// sequence whose KV cache kv[b] holds: for a prompt every slot is c and the
// positions run on; for several conversations (sml_forward_multi) each slot is
// its own context. Weights and batch scratch are c's, read once for all slots.
static void forward_slots(sml_ctx * c, sml_ctx * const * kv, const int32_t * tokens,
                          const int * pos, int nb, int depth, float * logits_out) {
    const sml_model * m = c->model;
    const sml_hdr * h = m->hdr;
    if (depth <= 0) depth = c->depth;
    if (nb > SML_MAX_BATCH) nb = SML_MAX_BATCH;
    if (c->step_only) {
        // experts and Granite's scales live on the step path only
        for (int b = 0; b < nb; b++) {
            const float * l = sml_forward(kv[b], tokens[b], pos[b], depth);
            if (logits_out) memcpy(logits_out + (size_t) b * h->n_vocab, l,
                                   (size_t) h->n_vocab * sizeof(float));
        }
        return;
    }

    const int n_embd = (int) h->n_embd;
    const int n_head = (int) h->n_head;
    const int n_kv   = (int) h->n_head_kv;
    const int hd     = c->head_dim;
    const int n_attn = n_head * hd;   // attention width; n_embd except Qwen3-style
    const int kvd    = n_kv * hd;
    const int rep    = n_head / n_kv;
    const int n_ff   = (int) h->n_ff;

    // embeddings: one row each, always at full depth (see sml_forward) -- or the
    // residual rows a split model's GPU half handed in
    if (c->x_in) memcpy(c->bx, c->x_in, (size_t) nb * n_embd * sizeof(float));
    for (int b = 0; b < nb && !c->x_in; b++) {
        if (c->emb->kind == SML_KIND_F32) {
            const float * w = (const float *) sml_ptr(m, c->emb->off);
            memcpy(c->xb, w + (size_t) tokens[b] * c->emb->cols, (size_t) n_embd * sizeof(float));
        } else if (c->emb->layout == SML_LAYOUT_FLAT && (c->emb->nbits == 3 || c->emb->nbits == 5 || c->emb->nbits == 7)) {
            dequant_row_mixed(c->xb, m, c->emb, tokens[b]);
        } else if (c->emb->layout == SML_LAYOUT_FLAT && c->emb->nbits == 2) {
            const uint8_t * p2[1] = { sml_ptr(m, c->emb->codes_off[0]) };
            sml_dequant_row(c->xb, p2, 1, 1, (const uint16_t *) sml_ptr(m, c->emb->sd_off),
                (const uint8_t *) sml_ptr(m, c->emb->sq_off),
                c->emb->mq_n ? (const uint16_t *) sml_ptr(m, c->emb->md_off) : NULL,
                c->emb->mq_n ? (const uint8_t  *) sml_ptr(m, c->emb->mq_off) : NULL,
                tokens[b], (int) c->emb->cols);
        } else if (c->emb->layout == SML_LAYOUT_FLAT && c->emb->nbits == 8) {
            sml_dequant_row_flat8(c->xb, sml_ptr(m, c->emb->codes_off[0]), sml_ptr(m, c->emb->codes_off[1]),
                (const uint16_t *) sml_ptr(m, c->emb->sd_off), (const uint8_t *) sml_ptr(m, c->emb->sq_off),
                c->emb->mq_n ? (const uint16_t *) sml_ptr(m, c->emb->md_off) : NULL,
                c->emb->mq_n ? (const uint8_t  *) sml_ptr(m, c->emb->mq_off) : NULL,
                tokens[b], (int) c->emb->cols);
        } else if (c->emb->layout == SML_LAYOUT_FLAT && c->emb->nbits == 6) {
            sml_dequant_row_flat6(c->xb,
                sml_ptr(m, c->emb->codes_off[1]), sml_ptr(m, c->emb->codes_off[0]),
                (const uint16_t *) sml_ptr(m, c->emb->sd_off),
                (const uint8_t  *) sml_ptr(m, c->emb->sq_off),
                c->emb->mq_n ? (const uint16_t *) sml_ptr(m, c->emb->md_off) : NULL,
                c->emb->mq_n ? (const uint8_t  *) sml_ptr(m, c->emb->mq_off) : NULL,
                tokens[b], (int) c->emb->cols);
        } else if (c->emb->layout == SML_LAYOUT_FLAT) {
            sml_dequant_row_flat4(c->xb, sml_ptr(m, c->emb->codes_off[0]),
                (const uint16_t *) sml_ptr(m, c->emb->sd_off),
                (const uint8_t  *) sml_ptr(m, c->emb->sq_off),
                c->emb->mq_n ? (const uint16_t *) sml_ptr(m, c->emb->md_off) : NULL,
                c->emb->mq_n ? (const uint8_t  *) sml_ptr(m, c->emb->mq_off) : NULL,
                tokens[b], (int) c->emb->cols);
        } else {
        const uint8_t * planes[SML_MAX_LEVELS];
        for (uint32_t lv = 0; lv < h->n_levels; lv++)
            planes[lv] = sml_ptr(m, c->emb->codes_off[lv]);
        sml_dequant_row(c->xb, planes, (int) h->n_levels, (int) h->n_levels,
                        (const uint16_t *) sml_ptr(m, c->emb->sd_off),
                        (const uint8_t  *) sml_ptr(m, c->emb->sq_off),
                        c->emb->mq_n ? (const uint16_t *) sml_ptr(m, c->emb->md_off) : NULL,
                        c->emb->mq_n ? (const uint8_t  *) sml_ptr(m, c->emb->mq_off) : NULL,
                        tokens[b], (int) c->emb->cols);
        }
        float * dst = c->bx + (size_t) b * n_embd;
        if (c->emb->kind == SML_KIND_F32) memcpy(dst, c->xb, (size_t) n_embd * sizeof(float));
        else {
            const int32_t * perm = (const int32_t *) sml_ptr(m, c->emb->perm_off);
            for (int i = 0; i < n_embd; i++) dst[perm[i]] = c->xb[i];
        }
        // the embedding's epilogue in sml_forward's order: scale (Gemma, Granite),
        // this slot's learned position (GPT-2 lineage), the embedding norm (BLOOM)
        if (h->emb_scale != 0.0f) for (int i = 0; i < n_embd; i++) dst[i] *= h->emb_scale;
        if (c->pos_embd) {
            const float * pr = c->pos_embd + (size_t) pos[b] * n_embd;
            for (int i = 0; i < n_embd; i++) dst[i] += pr[i];
        }
        if (c->emb_norm) norm(c, dst, dst, c->emb_norm, c->emb_norm_b, n_embd);
        // token_embd serves two roles. As the output head the rotation cancels:
        // (E H)(H h) = E h. As the EMBEDDING LOOKUP it does not -- row r of the
        // rotated tensor is E[r]H, and the network needs E[r]. H is its own
        // inverse, so rotating once more recovers it.
        if (m->hdr->flags & SML_MODEL_HADAMARD) {
            sml_hadamard32(c->xr, dst, n_embd);
            memcpy(dst, c->xr, (size_t) n_embd * sizeof(float));
        }
    }

    for (int L = c->layer0; L < (int) h->n_layer; L++) {
        const sml_layer * ly = &c->layers[L];

        for (int b = 0; b < nb; b++) {
            if (ly->attn_norm)
                norm(c, c->bh + (size_t) b * n_embd, c->bx + (size_t) b * n_embd,
                     (const float *) sml_ptr(m, ly->attn_norm->off), ly->attn_norm_b, n_embd);
            else     // OLMo 2: no input norm, the raw stream
                memcpy(c->bh + (size_t) b * n_embd, c->bx + (size_t) b * n_embd, (size_t) n_embd * sizeof(float));
        }
        const float rsc = h->res_scale != 0.0f ? h->res_scale : 1.0f;    // Granite's branch scale
        const int par = (h->version >= 5) &&
                        (h->arch_bits & SML_ARCH_PARALLEL_RESIDUAL);

        if (!ly->q && !ly->mla_kva) {
            // no mixer (Nemotron-H's MLP layers): the stream goes straight to the FFN
            goto ffn_part;
        }
        if (ly->attn_gate)      // Qwen3-Next's output gate reads the normed input
            matvec_batch(c, c->bgate, ly->attn_gate, c->bh, n_embd, nb, depth);

        // q, k and v read the same normed input; when they also share a column
        // order -- every layer of every file built so far -- prepare it once.
        if (ly->mla_kva) {
            for (int b = 0; b < nb; b++)
                mla_qkv(c, ly, c->bh + (size_t) b * n_embd, c->bq + (size_t) b * n_head * hd,
                        c->bk + (size_t) b * kvd, c->bv + (size_t) b * kvd, depth);
        } else if (ly->k->perm_off == ly->q->perm_off && ly->v->perm_off == ly->q->perm_off) {
            const sml_rec * const rs[3] = { ly->q, ly->k, ly->v };
            float * const ys[3] = { c->bq, c->bk, c->bv };
            batch_group(c, rs, ys, 3, c->bh, n_embd, nb, depth);
        } else {
            matvec_batch(c, c->bq, ly->q, c->bh, n_embd, nb, depth);
            matvec_batch(c, c->bk, ly->k, c->bh, n_embd, nb, depth);
            matvec_batch(c, c->bv, ly->v, c->bh, n_embd, nb, depth);
        }

        for (int b = 0; b < nb; b++) {
            float * q = c->bq + (size_t) b * n_head * hd;
            float * k = c->bk + (size_t) b * kvd;
            float * v = c->bv + (size_t) b * kvd;
            if (ly->q_bias) for (int i = 0; i < n_head * hd; i++) q[i] += ly->q_bias[i];
            if (ly->k_bias) for (int i = 0; i < kvd; i++) k[i] += ly->k_bias[i];
            if (ly->v_bias) for (int i = 0; i < kvd; i++) v[i] += ly->v_bias[i];
            if (h->clip_qkv > 0.0f) {
                const float cl = h->clip_qkv;
                for (int i = 0; i < n_head * hd; i++) q[i] = q[i] > cl ? cl : q[i] < -cl ? -cl : q[i];
                for (int i = 0; i < kvd; i++) k[i] = k[i] > cl ? cl : k[i] < -cl ? -cl : k[i];
                for (int i = 0; i < kvd; i++) v[i] = v[i] > cl ? cl : v[i] < -cl ? -cl : v[i];
            }
            const int qk_late = h->version >= 5 && (h->arch_bits & SML_ARCH_QKNORM_LATE);
            if (!qk_late && ly->q_norm) qk_norm(q, ly->q_norm, ly->q_norm_b, ly->q_norm_n, n_head, hd, h->rms_eps, ly->qk_heads);
            if (!qk_late && ly->k_norm) qk_norm(k, ly->k_norm, ly->k_norm_b, ly->k_norm_n, n_kv,   hd, h->rms_eps, ly->qk_heads);
            if (!c->no_rope) {   // learned positions or ALiBi: no rotary at all
                rope_at(kv[b], q, n_head, hd, pos[b], L);
                rope_at(kv[b], k, n_kv,   hd, pos[b], L);
            }
            if (qk_late && ly->q_norm) qk_norm(q, ly->q_norm, ly->q_norm_b, ly->q_norm_n, n_head, hd, h->rms_eps, ly->qk_heads);
            if (qk_late && ly->k_norm) qk_norm(k, ly->k_norm, ly->k_norm_b, ly->k_norm_n, n_kv,   hd, h->rms_eps, ly->qk_heads);
            f32_to_f16(kv[b]->k_cache + ((size_t) L * kv[b]->n_ctx + pos[b]) * kvd, k, kvd);
            f32_to_f16(kv[b]->v_cache + ((size_t) L * kv[b]->n_ctx + pos[b]) * kvd, v, kvd);
        }

        // Attention is per position and reads the cache, so it stays serial.
        // Every batch slot's K/V is already written above, which is what makes
        // a slot able to attend to the slots before it.
        // Gemma scales by a declared constant rather than 1/sqrt(head_dim).
        const float inv = (h->version >= 4 && h->attn_scale != 0.0f)
                          ? h->attn_scale : 1.0f / sqrtf((float) hd);
        // Sliding window: attend only to the last n_swa positions. 0 = full.
        // Per layer when the file says so (Gemma 2/3, Cohere2, GPT-Neo).
        const int swa = c->layer_win ? c->layer_win[L] : (h->version >= 4) ? (int) h->n_swa : 0;
        const float cap = h->attn_softcap;
        // Serial and scalar, this was 26% of a 109-token prompt on the 1B and 18%
        // on the 3B, and it grows with the square of the context. Slots and heads
        // write disjoint outputs and only read cache positions already written,
        // so (slot, head) pairs run in parallel, each thread with its own score
        // row, through the same AVX2 dot and axpy the stepped path uses.
        const double ta = sml_now();
        if (c->batch_decode && use_pool()) {
            ms_attn_job j = { c, kv, pos, L, hd, kvd, rep, swa, n_head, n_attn, inv, cap, ly->sinks };
            sml_pool_for(nb * n_head, ms_attn, &j);
        } else
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            float * a = c->tscr + (size_t) omp_get_thread_num() * c->tscr_len;
#else
            float * a = c->tscr;
#endif
#ifdef _OPENMP
// The causal triangle makes late slots longer; a static split handed the
// longest rows to one thread, so chunks are claimed instead.
#pragma omp for collapse(2) schedule(dynamic, 2)
#endif
            for (int b = 0; b < nb; b++) {
                for (int hh = 0; hh < n_head; hh++) {
                    const int p = pos[b];
                    const int kvh = hh / rep;
                    const int t0 = (swa && p >= swa) ? p - swa + 1 : 0;
                    const sml_ctx * o = kv[b];
                    const size_t base = (size_t) L * o->n_ctx * kvd + (size_t) kvh * hd;
                    attn_row(c->bq + (size_t) b * n_head * hd + (size_t) hh * hd,
                             o->k_cache + base, o->v_cache + base, (size_t) kvd,
                             t0, p, hd, inv, cap, c->alibi ? &c->alibi[hh] : NULL,
                             ly->sinks ? &ly->sinks[hh] : NULL, a,
                             c->bh + (size_t) b * n_attn + (size_t) hh * hd);
                }
            }
        }
        sml_t_attn += sml_now() - ta;

        if (ly->attn_gate)
            for (int b = 0; b < nb; b++) {
                float * y = c->bh + (size_t) b * n_attn;
                const float * g = c->bgate + (size_t) b * n_attn;
                for (int i = 0; i < n_attn; i++) y[i] *= 1.0f / (1.0f + expf(-g[i]));
            }
        if (ly->mla_kva) {
            // rows of n_head * head_dim packed to n_head * vd, one after another
            for (int b = 0; b < nb; b++) {
                float * y = c->bh + (size_t) b * n_attn;
                mla_pack(ly, y, n_head, hd);
                memmove(c->bh + (size_t) b * n_head * ly->mla_vd, y, (size_t) n_head * ly->mla_vd * sizeof(float));
            }
            matvec_batch(c, c->bout, ly->o, c->bh, n_head * ly->mla_vd, nb, depth);
        } else
        matvec_batch(c, c->bout, ly->o, c->bh, n_attn, nb, depth);
        if (ly->o_bias)
            for (int b = 0; b < nb; b++) {
                float * o = c->bout + (size_t) b * n_embd;
                for (int i = 0; i < n_embd; i++) o[i] += ly->o_bias[i];
            }
        if (ly->attn_post_norm)     // Gemma 2/3, OLMo 2: the branch is normed before it is added
            for (int b = 0; b < nb; b++) {
                float * o = c->bout + (size_t) b * n_embd;
                norm(c, c->xb, o, ly->attn_post_norm, NULL, n_embd);
                memcpy(o, c->xb, (size_t) n_embd * sizeof(float));
            }
        if (par) {
            for (int b = 0; b < nb; b++)
                norm(c, c->bh + (size_t) b * n_embd, c->bx + (size_t) b * n_embd,
                     (const float *) sml_ptr(m, ly->ffn_norm->off), ly->ffn_norm_b, n_embd);
        }
        for (int b = 0; b < nb; b++) {
            float * x = c->bx + (size_t) b * n_embd;
            const float * o = c->bout + (size_t) b * n_embd;
            for (int i = 0; i < n_embd; i++) x[i] += rsc * o[i];
        }
      ffn_part:
        if (!par) {
            for (int b = 0; b < nb; b++) {
                if (ly->ffn_norm)
                    norm(c, c->bh + (size_t) b * n_embd, c->bx + (size_t) b * n_embd,
                         (const float *) sml_ptr(m, ly->ffn_norm->off), ly->ffn_norm_b, n_embd);
                else
                    memcpy(c->bh + (size_t) b * n_embd, c->bx + (size_t) b * n_embd,
                           (size_t) n_embd * sizeof(float));
            }
        }
        if (ly->n_exp) {
            moe_ffn_batch(c, ly, nb, depth);
            goto ffn_out;
        }
        if (!ly->down) continue;    // attention-only layer (Nemotron-H): no FFN

        if (ly->gate) {
            if (ly->up->perm_off == ly->gate->perm_off) {
                const sml_rec * const rs[2] = { ly->gate, ly->up };
                float * const ys[2] = { c->bff, c->bff2 };
                batch_group(c, rs, ys, 2, c->bh, n_embd, nb, depth);
            } else {
                matvec_batch(c, c->bff,  ly->gate, c->bh, n_embd, nb, depth);
                matvec_batch(c, c->bff2, ly->up,   c->bh, n_embd, nb, depth);
            }
            for (int b = 0; b < nb; b++) {
                float * g = c->bff + (size_t) b * n_ff;
                float * u = c->bff2 + (size_t) b * n_ff;
                if (ly->gate_bias) for (int i = 0; i < n_ff; i++) g[i] += ly->gate_bias[i];
                if (ly->up_bias)   for (int i = 0; i < n_ff; i++) u[i] += ly->up_bias[i];
                gate_mul(c, g, u, n_ff);
            }
        } else {
            matvec_batch(c, c->bff, ly->up, c->bh, n_embd, nb, depth);
            for (int b = 0; b < nb; b++) {
                float * u = c->bff + (size_t) b * n_ff;
                if (ly->up_bias) for (int i = 0; i < n_ff; i++) u[i] += ly->up_bias[i];
                if (ly->xielu) xielu_act(ly->xielu, u, n_ff); else ffn_act(c, u, n_ff);
            }
        }

        matvec_batch(c, c->bout, ly->down, c->bff, n_ff, nb, depth);
      ffn_out:
        for (int b = 0; b < nb; b++) {
            float * x = c->bx + (size_t) b * n_embd;
            float * o = c->bout + (size_t) b * n_embd;
            if (ly->down_bias) for (int i = 0; i < n_embd; i++) o[i] += ly->down_bias[i];
            if (ly->ffn_post_norm) {
                norm(c, c->xb, o, ly->ffn_post_norm, NULL, n_embd);
                memcpy(o, c->xb, (size_t) n_embd * sizeof(float));
            }
            for (int i = 0; i < n_embd; i++) x[i] += rsc * o[i];
        }
    }

    for (int b = 0; b < nb; b++) kv[b]->tokens++;
    // Prompt reading passes NULL: only the last slot's logits are ever used, and
    // the caller heads that one slot from c->bx.
    if (!logits_out) return;

    for (int b = 0; b < nb; b++)
        norm(c, c->bh + (size_t) b * n_embd, c->bx + (size_t) b * n_embd,
             (const float *) sml_ptr(m, c->out_norm->off), c->out_norm_b, n_embd);

    matvec_batch(c, logits_out, c->head, c->bh, n_embd, nb, depth);
    if (c->out_b)
        for (int b = 0; b < nb; b++)
            for (uint32_t i = 0; i < h->n_vocab; i++)
                logits_out[(size_t) b * h->n_vocab + i] += c->out_b[i];
    if (h->logit_scale != 0.0f)
        for (size_t i = 0; i < (size_t) nb * h->n_vocab; i++) logits_out[i] /= h->logit_scale;
    if (h->version >= 4 && h->logit_softcap != 0.0f) {
        const float cap = h->logit_softcap;
        for (size_t i = 0; i < (size_t) nb * h->n_vocab; i++)
            logits_out[i] = cap * tanhf(logits_out[i] / cap);
    }
}

void sml_forward_batch(sml_ctx * c, const int32_t * tokens, int pos0, int nb,
                       int depth, float * logits_out) {
    if (nb > SML_MAX_BATCH) nb = SML_MAX_BATCH;
    int pos[SML_MAX_BATCH];
    sml_ctx * kv[SML_MAX_BATCH];
    for (int b = 0; b < nb; b++) { pos[b] = pos0 + b; kv[b] = c; }
    forward_slots(c, kv, tokens, pos, nb, depth, logits_out);
}

// Several conversations, one token each, one read of the weights. cs[b] owns
// slot b's cache and position; cs[0]'s batch scratch carries all of them, so
// every context must come from the same model and ns <= SML_MAX_BATCH. Each
// slot's logits land in logits_out[b * n_vocab]. A model the batched body
// cannot run (step_only, Hadamard) steps each context on its own.
void sml_forward_multi(sml_ctx * const * cs, const int32_t * tokens, const int * pos,
                       int ns, int depth, float * logits_out) {
    sml_ctx * c = cs[0];
    const uint32_t nv = c->model->hdr->n_vocab;
    if (depth <= 0) depth = c->depth;
    if (ns > SML_MAX_BATCH) ns = SML_MAX_BATCH;
    if (ns == 1 || c->step_only || (c->model->hdr->flags & SML_MODEL_HADAMARD)) {
        for (int b = 0; b < ns; b++) {
            const float * l = sml_forward(cs[b], tokens[b], pos[b], depth);
            memcpy(logits_out + (size_t) b * nv, l, (size_t) nv * sizeof(float));
        }
        return;
    }
    // Up to SML_MULTI_DECODE_MAX conversations run the decode kernels slot by
    // slot; more than that, the prompt kernels' repack pays for itself.
    static int dmax = -1;
    if (dmax < 0) {
        const char * e = getenv("SML_MULTI_DECODE_MAX");
        dmax = e ? atoi(e) : 16;
    }
    // SML_MULTI=exact: the decode kernels slot by slot (each conversation bit
    // for bit as decoded alone). Default: the once-packed rows-in-lanes copy on
    // per-256 activations -- the prompt kernel's arithmetic, bit for bit.
    static int exact = -1;
    if (exact < 0) { const char * x = getenv("SML_MULTI"); exact = x && !strcmp(x, "exact"); }
    // Two conversations run exact by default: each slot through the decode kernel
    // on 8-row tiles, the second from L1, streamed 58.8-61.2 tok/s of generation
    // against the packed copy's 55.3-57.2 (1B, four alternating pairs, 2026-09-26);
    // from four up the packed copy wins (B=8: ~112 vs ~74). SML_MULTI_EXACT_MAX
    // moves the line.
    static int emax = -1;
    if (emax < 0) {
        const char * e = getenv("SML_MULTI_EXACT_MAX");
        emax = e ? atoi(e) : 2;
    }
    const int ex = exact || ns <= emax;
    if (!ex && !c->rlp) build_rlp(c);
    c->batch_decode = ns > dmax ? 0 : (!ex && c->rlp) ? 1 : 2;
    forward_slots(c, cs, tokens, pos, ns, depth, logits_out);
    c->batch_decode = 0;
}

// Prompt reading stepped one token at a time: one full weight read per prompt
// token, measured 0.25x the reference engine's prompt speed on the 1B and 0.29x
// on the 3B. Every prompt token is already known, so the prompt goes through the
// batched body SML_MAX_BATCH tokens per weight read, and the output head -- the
// largest single matvec in the model -- runs once, for the last position.
const float * sml_prefill(sml_ctx * c, const int32_t * ids, int pos0, int n, int depth) {
    const sml_model * m = c->model;
    const sml_hdr * h = m->hdr;
    if (n <= 0) return NULL;
    if (depth <= 0) depth = c->depth;
    // matvec_batch gathers without the Hadamard rotation matvecN applies, so a
    // rotated file steps rather than reads wrong numbers.
    if ((h->flags & SML_MODEL_HADAMARD) || c->step_only) {
        const float * l = NULL;
        for (int i = 0; i < n; i++) l = sml_forward(c, ids[i], pos0 + i, depth);
        return l;
    }
    // SML_PREFILL_NB sets the slots per weight read (1..SML_MAX_BATCH); unset,
    // SML_PREFILL_NB_DEFAULT. At 1 the batch kernels still run, so a
    // batched-vs-stepped difference splits into what the kernels do and what
    // batching does.
    _Static_assert(SML_MAX_BATCH <= SML_KERNEL_MAX_BATCH,
                   "the batch kernels hold SML_KERNEL_MAX_BATCH accumulators");
    int cap = SML_PREFILL_NB_DEFAULT;
    const char * e = getenv("SML_PREFILL_NB");
    if (e && atoi(e) >= 1 && atoi(e) <= SML_MAX_BATCH) cap = atoi(e);
    int done = 0, nb = 0;
    while (done < n) {
        nb = n - done < cap ? n - done : cap;
        sml_forward_batch(c, ids + done, pos0 + done, nb, depth, NULL);
        done += nb;
    }
    const int n_embd = (int) h->n_embd;
    norm(c, c->xb, c->bx + (size_t) (nb - 1) * n_embd,
         (const float *) sml_ptr(m, c->out_norm->off), c->out_norm_b, n_embd);
    matvec(c, c->logits, c->head, c->xb, depth);
    if (c->out_b) for (uint32_t i = 0; i < h->n_vocab; i++) c->logits[i] += c->out_b[i];
    if (h->logit_scale != 0.0f)
        for (uint32_t i = 0; i < h->n_vocab; i++) c->logits[i] /= h->logit_scale;
    if (h->version >= 4 && h->logit_softcap != 0.0f) {
        const float cap = h->logit_softcap;
        for (uint32_t i = 0; i < h->n_vocab; i++)
            c->logits[i] = cap * tanhf(c->logits[i] / cap);
    }
    return c->logits;
}

const float * sml_forward_from(sml_ctx * c, const float * x, int pos, int L0, int depth) {
    c->x_in = x; c->layer0 = L0;
    const float * l = sml_forward(c, 0, pos, depth);
    c->x_in = NULL; c->layer0 = 0;
    return l;
}

const float * sml_prefill_from(sml_ctx * c, const float * xs, int pos0, int n, int L0, int depth) {
    const sml_model * m = c->model;
    const sml_hdr * h = m->hdr;
    const int n_embd = (int) h->n_embd;
    if (n <= 0) return NULL;
    if (depth <= 0) depth = c->depth;
    const float * l = NULL;
    if ((h->flags & SML_MODEL_HADAMARD) || c->step_only) {
        for (int i = 0; i < n; i++) l = sml_forward_from(c, xs + (size_t) i * n_embd, pos0 + i, L0, depth);
        return l;
    }
    int cap = SML_PREFILL_NB_DEFAULT;
    const char * e = getenv("SML_PREFILL_NB");
    if (e && atoi(e) >= 1 && atoi(e) <= SML_MAX_BATCH) cap = atoi(e);
    int done = 0, nb = 0;
    c->layer0 = L0;
    while (done < n) {
        nb = n - done < cap ? n - done : cap;
        c->x_in = xs + (size_t) done * n_embd;
        sml_forward_batch(c, NULL, pos0 + done, nb, depth, NULL);
        done += nb;
    }
    c->x_in = NULL; c->layer0 = 0;
    norm(c, c->xb, c->bx + (size_t) (nb - 1) * n_embd,
         (const float *) sml_ptr(m, c->out_norm->off), c->out_norm_b, n_embd);
    matvec(c, c->logits, c->head, c->xb, depth);
    if (c->out_b) for (uint32_t i = 0; i < h->n_vocab; i++) c->logits[i] += c->out_b[i];
    if (h->logit_scale != 0.0f)
        for (uint32_t i = 0; i < h->n_vocab; i++) c->logits[i] /= h->logit_scale;
    if (h->version >= 4 && h->logit_softcap != 0.0f) {
        const float cap2 = h->logit_softcap;
        for (uint32_t i = 0; i < h->n_vocab; i++)
            c->logits[i] = cap2 * tanhf(c->logits[i] / cap2);
    }
    return c->logits;
}

// Set SML_DUMP=path to write the residual stream after the embedding and after
// every layer. Comparing that against the numpy reference on the SAME file is
// what turns "the output is wrong" into "layer 3 diverges", which is the
// difference between reading code and finding the bug.
static void dump_x(const float * x, int n) {
    const char * p = getenv("SML_DUMP");
    if (!p) return;
    FILE * f = fopen(p, "ab");
    if (!f) return;
    fwrite(x, sizeof(float), (size_t) n, f);
    fclose(f);
}


// Mixture of experts (Granite MoE, Mixtral, ...). A router scores every expert
// for this token, the n_expert_used best run, and their outputs are summed by the
// router's weights. Only the chosen experts' weights are read: the rest of the
// layer's FFN bytes never move, which is the architecture's whole point and this
// engine's too. Input c->xb (normed), output back into c->xb.
// LFM2's short convolution: in -> [B | C | x]; B*x is kept per position;
// y[i] = C[i] * sum_k w[i][k] * (B*x)[pos - (K-1) + k][i]; out. The result lands
// in c->hb2, where the attention's output would.
static void sc_block(sml_ctx * c, const sml_layer * ly, int L, int pos, int depth) {
    const int d = (int) c->model->hdr->n_embd, K = ly->sc_L;
    float * bcx = c->sc_buf;
    matvec(c, bcx, ly->sc_in, c->xb, depth);
    if (ly->sc_in_b) for (int i = 0; i < 3 * d; i++) bcx[i] += ly->sc_in_b[i];
    const float * B = bcx, * C = bcx + d, * X = bcx + 2 * d;
    float * hist = c->sc_hist[L];
    float * cur = hist + (size_t) pos * d;
    for (int i = 0; i < d; i++) cur[i] = B[i] * X[i];
    float * y = c->sc_y;
    for (int i = 0; i < d; i++) {
        float s = ly->sc_cb ? ly->sc_cb[i] : 0.0f;
        for (int k = 0; k < K; k++) {
            const int t = pos - (K - 1) + k;
            if (t >= 0) s += ly->sc_w[(size_t) i * K + k] * hist[(size_t) t * d + i];
        }
        y[i] = C[i] * s;
    }
    matvec(c, c->hb2, ly->sc_out, y, depth);
    if (ly->sc_out_b) for (int i = 0; i < d; i++) c->hb2[i] += ly->sc_out_b[i];
}

// Mamba-2, one token: the recurrence transformers' Mamba2Mixer runs when it
// decodes (torch_forward, cache path), step for step. Result into c->hb2.
static void ssm_block(sml_ctx * c, const sml_layer * ly, int L, int depth) {
    const sml_hdr * h = c->model->hdr;
    const int di = ly->ssm_inner, cd = ly->ssm_convdim, H = ly->ssm_heads, K = ly->ssm_K;
    const int G = h->ssm_groups ? (int) h->ssm_groups : 1, N = (int) h->ssm_dstate, P = di / H;
    const int rows = (int) ly->ssm_in->rows;
    const int dmlp = (rows - di - cd - H) / 2;
    float * pr = c->ssm_proj;
    matvec(c, pr, ly->ssm_in, c->xb, depth);
    const float * z = pr + 2 * dmlp, * xbc = z + di, * dt = xbc + cd;
    float * cs = c->ssm_cst[L], * cv = c->ssm_xbc;
    for (int j = 0; j < cd; j++) {
        float s = ly->ssm_conv_b ? ly->ssm_conv_b[j] : 0.0f;
        for (int k = 0; k < K - 1; k++) s += ly->ssm_conv[(size_t) j * K + k] * cs[(size_t) k * cd + j];
        s += ly->ssm_conv[(size_t) j * K + K - 1] * xbc[j];
        cv[j] = s / (1.0f + expf(-s));                           // SiLU
    }
    if (K > 1) {
        memmove(cs, cs + cd, (size_t) (K - 2) * cd * sizeof(float));
        memcpy(cs + (size_t) (K - 2) * cd, xbc, (size_t) cd * sizeof(float));
    }
    const float * x = cv, * B = cv + di, * C = cv + di + (size_t) G * N;
    float * S = c->ssm_state[L], * y = c->ssm_y;
    const int per = H / G;
    for (int hh = 0; hh < H; hh++) {
        float d = dt[hh] + ly->ssm_dt_b[hh];
        d = d > 20.0f ? d : log1pf(expf(d));                    // softplus
        if (h->ssm_dt_hi > 0.0f) d = d < h->ssm_dt_lo ? h->ssm_dt_lo : d > h->ssm_dt_hi ? h->ssm_dt_hi : d;
        const float dA = expf(d * -expf(ly->ssm_a_log[hh]));
        const float * Bg = B + (size_t) (hh / per) * N, * Cg = C + (size_t) (hh / per) * N;
        for (int p = 0; p < P; p++) {
            const float xv = x[hh * P + p];
            float * sp = S + ((size_t) hh * P + p) * N;
            float acc = 0.0f;
            for (int n = 0; n < N; n++) {
                sp[n] = sp[n] * dA + d * Bg[n] * xv;
                acc += sp[n] * Cg[n];
            }
            y[hh * P + p] = acc + ly->ssm_d[hh] * xv;
        }
    }
    // gated RMSNorm: y * silu(z), normalized over groups of ssm_norm_group channels
    for (int i = 0; i < di; i++) y[i] *= z[i] / (1.0f + expf(-z[i]));
    const int ng = h->ssm_norm_group ? (int) h->ssm_norm_group : di;
    for (int g0 = 0; g0 < di && ly->ssm_nw; g0 += ng) {
        double ss = 0.0;
        for (int i = g0; i < g0 + ng; i++) ss += (double) y[i] * y[i];
        const float r = 1.0f / sqrtf((float) (ss / ng) + h->rms_eps);
        for (int i = g0; i < g0 + ng; i++) y[i] = y[i] * r * ly->ssm_nw[i];
    }
    matvec(c, c->hb2, ly->ssm_out, y, depth);
}

// y = W x for a tensor that may be full precision (Mamba's dt_proj: its rank
// is rarely a multiple of 32, so it stays f32 in the file)
// (a full-precision tensor is stored flat, so its shape is passed in)
static void mv_any(sml_ctx * c, float * y, const sml_rec * r, const float * x,
                   int rows, int cols, int depth) {
    if (r->kind != SML_KIND_F32) { matvec(c, y, r, x, depth); return; }
    const float * w = (const float *) sml_ptr(c->model, r->off);
    for (int i = 0; i < rows; i++) {
        double s = 0.0;
        for (int j = 0; j < cols; j++) s += (double) w[(size_t) i * cols + j] * x[j];
        y[i] = (float) s;
    }
}

static void rms_inplace(float * v, int n, const float * w, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double) v[i] * v[i];
    const float r = 1.0f / sqrtf((float) (ss / n) + eps);
    for (int i = 0; i < n; i++) v[i] = v[i] * r * (w ? w[i] : 1.0f);
}

// Gated DeltaNet, one token: transformers' torch_recurrent_gated_delta_rule
// with use_qk_l2norm_in_kernel, behind Qwen3NextGatedDeltaNet's projections.
static void gdn_block(sml_ctx * c, const sml_layer * ly, int L, int depth) {
    const sml_hdr * h = c->model->hdr;
    const int nk = (int) h->gdn_nk, dk = (int) h->gdn_dk, nv = ly->gdn_nv, dv = ly->gdn_dv;
    const int kd = nk * dk, vd = nv * dv, cd = 2 * kd + vd, K = ly->gdn_K, rep = nv / nk;
    float * pr = c->ssm_proj;
    matvec(c, pr, ly->gdn_qkvz, c->xb, depth);             // [q | k | v | z]
    float * ba = pr + ly->gdn_qkvz->rows;
    matvec(c, ba, ly->gdn_ba, c->xb, depth);               // [b | a]
    const float * z = pr + cd;
    float * cs = c->ssm_cst[L], * u = c->ssm_xbc;
    for (int j = 0; j < cd; j++) {
        float s = 0.0f;
        for (int k = 0; k < K - 1; k++) s += ly->gdn_conv[(size_t) j * K + k] * cs[(size_t) k * cd + j];
        s += ly->gdn_conv[(size_t) j * K + K - 1] * pr[j];
        u[j] = s / (1.0f + expf(-s));
    }
    if (K > 1) {
        memmove(cs, cs + cd, (size_t) (K - 2) * cd * sizeof(float));
        memcpy(cs + (size_t) (K - 2) * cd, pr, (size_t) cd * sizeof(float));
    }
    float * q = u, * k = u + kd, * v = u + 2 * kd;
    const float qs = 1.0f / sqrtf((float) dk);
    for (int hh = 0; hh < nk; hh++) {                       // L2 norm per key head
        float * qh = q + (size_t) hh * dk, * kh = k + (size_t) hh * dk;
        double sq = 0.0, sk = 0.0;
        for (int i = 0; i < dk; i++) { sq += (double) qh[i] * qh[i]; sk += (double) kh[i] * kh[i]; }
        const float rq = 1.0f / sqrtf((float) sq + 1e-6f), rk = 1.0f / sqrtf((float) sk + 1e-6f);
        for (int i = 0; i < dk; i++) { qh[i] *= rq * qs; kh[i] *= rk; }
    }
    float * y = c->ssm_y;
    for (int hv = 0; hv < nv; hv++) {
        const float * qh = q + (size_t) (hv / rep) * dk, * kh = k + (size_t) (hv / rep) * dk;
        const float * vh = v + (size_t) hv * dv;
        const float beta = 1.0f / (1.0f + expf(-ba[hv]));
        float a = ba[nv + hv] + ly->gdn_dt_b[hv];
        a = a > 20.0f ? a : log1pf(expf(a));
        const float decay = expf(-expf(ly->gdn_a_log[hv]) * a);
        float * S = c->ssm_state[L] + (size_t) hv * dk * dv;    // [dk][dv]
        float * yh = y + (size_t) hv * dv;
        for (int i = 0; i < dk * dv; i++) S[i] *= decay;
        for (int j = 0; j < dv; j++) {
            float mem = 0.0f;
            for (int i = 0; i < dk; i++) mem += S[(size_t) i * dv + j] * kh[i];
            const float delta = (vh[j] - mem) * beta;
            float o = 0.0f;
            for (int i = 0; i < dk; i++) {
                S[(size_t) i * dv + j] += kh[i] * delta;
                o += S[(size_t) i * dv + j] * qh[i];
            }
            yh[j] = o;
        }
        // RMSNorm over the head, weight, then * silu(z)
        double ss = 0.0;
        for (int j = 0; j < dv; j++) ss += (double) yh[j] * yh[j];
        const float r = 1.0f / sqrtf((float) (ss / dv) + h->rms_eps);
        const float * zh = z + (size_t) hv * dv;
        for (int j = 0; j < dv; j++) yh[j] = yh[j] * r * ly->gdn_nw[j] * (zh[j] / (1.0f + expf(-zh[j])));
    }
    matvec(c, c->hb2, ly->gdn_out, y, depth);
}

// Mamba-1, one token: transformers' MambaMixer.slow_forward on its cache path.
static void m1_block(sml_ctx * c, const sml_layer * ly, int L, int depth) {
    const sml_hdr * h = c->model->hdr;
    const int di = ly->m1_inner, N = ly->m1_N, K = ly->m1_K, R = ly->m1_R;
    float * pr = c->ssm_proj;                   // [x | z] then [dt | B | C]
    matvec(c, pr, ly->m1_in, c->xb, depth);
    const float * xin = pr, * z = pr + di;
    float * cs = c->ssm_cst[L], * u = c->ssm_xbc;
    for (int j = 0; j < di; j++) {
        float s = ly->m1_conv_b ? ly->m1_conv_b[j] : 0.0f;
        for (int k = 0; k < K - 1; k++) s += ly->m1_conv[(size_t) j * K + k] * cs[(size_t) k * di + j];
        s += ly->m1_conv[(size_t) j * K + K - 1] * xin[j];
        u[j] = s / (1.0f + expf(-s));
    }
    if (K > 1) {
        memmove(cs, cs + di, (size_t) (K - 2) * di * sizeof(float));
        memcpy(cs + (size_t) (K - 2) * di, xin, (size_t) di * sizeof(float));
    }
    float * xp = pr + 2 * di;                   // [R | N | N]
    matvec(c, xp, ly->m1_x, u, depth);
    float * tsr = xp, * B = xp + R, * C = xp + R + N;
    if (h->ssm_bcdt_eps > 0.0f) {               // Falcon-Mamba: weightless
        rms_inplace(B, N, NULL, h->ssm_bcdt_eps);
        rms_inplace(C, N, NULL, h->ssm_bcdt_eps);
        rms_inplace(tsr, R, NULL, h->ssm_bcdt_eps);
    }
    if (ly->m1_ndt) rms_inplace(tsr, R, ly->m1_ndt, h->rms_eps);   // Jamba: weighted
    if (ly->m1_nb)  rms_inplace(B, N, ly->m1_nb, h->rms_eps);
    if (ly->m1_nc)  rms_inplace(C, N, ly->m1_nc, h->rms_eps);
    float * dtv = c->ssm_y + di;                // per channel
    mv_any(c, dtv, ly->m1_dt, tsr, di, R, depth);
    float * S = c->ssm_state[L], * y = c->ssm_y;
    for (int j = 0; j < di; j++) {
        float d = dtv[j] + (ly->m1_dt_b ? ly->m1_dt_b[j] : 0.0f);
        d = d > 20.0f ? d : log1pf(expf(d));
        float * sp = S + (size_t) j * N;
        const float * al = ly->m1_a_log + (size_t) j * N;
        float acc = 0.0f;
        for (int n = 0; n < N; n++) {
            sp[n] = expf(-expf(al[n]) * d) * sp[n] + d * B[n] * u[j];
            acc += sp[n] * C[n];
        }
        const float g = z[j] / (1.0f + expf(-z[j]));
        y[j] = (acc + ly->m1_d[j] * u[j]) * g;
    }
    matvec(c, c->hb2, ly->m1_out, y, depth);
}

// Which experts a token uses, and their weights, from the router's logits.
// Shared by the CPU (moe_ffn) and the GPU, which computes the logits on the card
// and chooses here, so the two cannot choose differently. Returns k.
int sml_moe_route(const sml_hdr * h, const sml_layer * ly, const float * lg,
                  float * sc, float * ch, int * sel, float * w) {
    const int ne = ly->n_exp;
    int k = (int) h->n_expert_used;
    if (k <= 0 || k > ne) k = ne;
    if (k > SML_MOE_MAX_K) k = SML_MOE_MAX_K;
    if (h->moe_flags & 4u) {
        // PhiMoE's sparsemixer (modeling_phimoe.sparsemixer at inference): each pick is the
        // best expert not yet taken; its weight is its softmax share among the untaken
        // experts within 2 x jitter of it, relative to max(|logit|, best). moe_scale holds
        // the jitter here.
        const float eps2 = 2.0f * h->moe_scale;
        for (int j = 0; j < k; j++) {
            int best = -1;
            for (int e = 0; e < ne; e++) {
                int taken = 0;
                for (int q = 0; q < j; q++) if (sel[q] == e) taken = 1;
                if (!taken && (best < 0 || lg[e] > lg[best])) best = e;
            }
            const float mx = lg[best];
            double sum = 0.0;
            for (int e = 0; e < ne; e++) {
                int taken = 0;
                for (int q = 0; q < j; q++) if (sel[q] == e) taken = 1;
                const float f = fabsf(lg[e]) > mx ? fabsf(lg[e]) : mx;
                if (!taken && (mx - lg[e]) / f <= eps2) sum += exp((double) lg[e] - mx);
            }
            sel[j] = best;
            w[j] = (float) (1.0 / sum);
        }
        return k;
    }
    // top-k, ties to the lower index -- the order the builder's stable sort gives
    const int v3 = (h->moe_flags & 2u) || ly->exp_bias || h->moe_groups > 1;
    if (v3) {
        // Scores (sigmoid, or softmax over every expert); the CHOICE adds the
        // correction bias and, with groups, keeps only the best groups' experts
        // (a group scored by its two best); the WEIGHTS are the plain scores.
        if (h->moe_flags & 2u) {
            for (int e = 0; e < ne; e++) sc[e] = 1.0f / (1.0f + expf(-lg[e]));
        } else {
            float mx = lg[0];
            for (int e = 1; e < ne; e++) if (lg[e] > mx) mx = lg[e];
            double sum = 0.0;
            for (int e = 0; e < ne; e++) sum += expf(lg[e] - mx);
            for (int e = 0; e < ne; e++) sc[e] = (float) (expf(lg[e] - mx) / sum);
        }
        for (int e = 0; e < ne; e++) ch[e] = sc[e] + (ly->exp_bias ? ly->exp_bias[e] : 0.0f);
        const int ng = (int) h->moe_groups, tg = (int) h->moe_topk_group;
        if (ng > 1 && tg < ng && ne % ng == 0) {
            const int per = ne / ng;
            float gsc[64];
            int keep[64] = { 0 };
            for (int g = 0; g < ng && g < 64; g++) {
                float b1 = -1e30f, b2 = -1e30f;
                for (int i = 0; i < per; i++) {
                    const float v = ch[g * per + i];
                    if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) b2 = v;
                }
                // DeepSeek V2's group_limited_greedy scores a group by its best expert;
                // V3's noaux_tc by its best two
                gsc[g] = (h->moe_flags & 8u) ? b1 : b1 + (per > 1 ? b2 : 0.0f);
            }
            for (int t = 0; t < tg; t++) {
                int best = -1;
                for (int g = 0; g < ng; g++) if (!keep[g] && (best < 0 || gsc[g] > gsc[best])) best = g;
                keep[best] = 1;
            }
            for (int e = 0; e < ne; e++) if (!keep[e / per]) ch[e] = 0.0f;   // masked to 0, as they mask
        }
        for (int j = 0; j < k; j++) {
            int best = -1;
            for (int e = 0; e < ne; e++) {
                int taken = 0;
                for (int q = 0; q < j; q++) if (sel[q] == e) taken = 1;
                if (!taken && (best < 0 || ch[e] > ch[best])) best = e;
            }
            sel[j] = best;
            w[j] = sc[best];
        }
        if (h->moe_flags & 1u) {
            float sum = 0.0f;
            for (int j = 0; j < k; j++) sum += w[j];
            for (int j = 0; j < k; j++) w[j] /= sum + 1e-20f;
        }
        if (h->moe_scale != 0.0f) for (int j = 0; j < k; j++) w[j] *= h->moe_scale;
    } else {
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int e = 0; e < ne; e++) {
            int taken = 0;
            for (int q = 0; q < j; q++) if (sel[q] == e) taken = 1;
            if (!taken && (best < 0 || lg[e] > lg[best])) best = e;
        }
        sel[j] = best;
    }
    if (h->moe_flags & 1u) {
        // softmax over the chosen experts only (Granite, Mixtral)
        double sum = 0.0;
        for (int j = 0; j < k; j++) { w[j] = expf(lg[sel[j]] - lg[sel[0]]); sum += w[j]; }
        for (int j = 0; j < k; j++) w[j] = (float)(w[j] / sum);
    } else {
        // softmax over all experts, then the chosen weights as they are
        float mx = lg[0];
        for (int e = 1; e < ne; e++) if (lg[e] > mx) mx = lg[e];
        double sum = 0.0;
        for (int e = 0; e < ne; e++) sum += expf(lg[e] - mx);
        for (int j = 0; j < k; j++) w[j] = (float)(expf(lg[sel[j]] - mx) / sum);
    }
    if (h->moe_scale != 0.0f) for (int j = 0; j < k; j++) w[j] *= h->moe_scale;   // DeepSeek V2 greedy
    }
    return k;
}

// Latent attention's Q, K and V for one normed row x, laid out as ordinary heads:
// q and k [n_head x head_dim] with the rotary dims first in each head, v padded to
// head_dim -- so the rotary, the cache and the attention are the shared ones.
static void mla_qkv(sml_ctx * c, const sml_layer * ly, const float * x, float * q, float * k, float * v, int depth) {
    const sml_hdr * h = c->model->hdr;
    const int nh = (int) h->n_head, hd = c->head_dim, rot = (int) h->n_rot, nope = hd - rot, vd = ly->mla_vd;
    if (ly->mla_qa) {
        matvec(c, c->mla_a, ly->mla_qa, x, depth);
        norm(c, c->mla_an, c->mla_a, ly->mla_qa_norm, NULL, ly->mla_rq);
        matvec(c, q, ly->mla_qb, c->mla_an, depth);
    } else {
        matvec(c, q, ly->q, x, depth);
    }
    matvec(c, c->mla_a, ly->mla_kva, x, depth);            // [latent | shared rotary key]
    norm(c, c->mla_an, c->mla_a, ly->mla_kva_norm, NULL, ly->mla_rkv);
    matvec(c, c->mla_kv, ly->mla_kvb, c->mla_an, depth);   // per head [k_nope | v]
    const float * kr = c->mla_a + ly->mla_rkv;
    for (int hh = 0; hh < nh; hh++) {
        const float * src = c->mla_kv + (size_t) hh * (nope + vd);
        float * kh = k + (size_t) hh * hd, * vh = v + (size_t) hh * hd;
        memcpy(kh, kr, (size_t) rot * sizeof(float));
        memcpy(kh + rot, src, (size_t) nope * sizeof(float));
        memcpy(vh, src + nope, (size_t) vd * sizeof(float));
        if (hd > vd) memset(vh + vd, 0, (size_t) (hd - vd) * sizeof(float));
    }
}

// each head's first mla_vd outputs, packed for o (rows of n_head * head_dim -> n_head * vd)
static void mla_pack(const sml_layer * ly, float * y, int nh, int hd) {
    for (int hh = 1; hh < nh; hh++) memmove(y + (size_t) hh * ly->mla_vd, y + (size_t) hh * hd, (size_t) ly->mla_vd * sizeof(float));
}

static void moe_ffn(sml_ctx * c, const sml_layer * ly, int depth) {
    const sml_hdr * h = c->model->hdr;
    const int n_embd = (int) h->n_embd, ne = ly->n_exp;
    float * lg = c->moe_logit;
    for (int e = 0; e < ne; e++) {
        const float * r = ly->router + (size_t) e * n_embd;
        double s = 0.0;
        for (int i = 0; i < n_embd; i++) s += (double) r[i] * c->xb[i];
        lg[e] = (float) s + (ly->router_b ? ly->router_b[e] : 0.0f);
    }
    int sel[SML_MOE_MAX_K];
    float w[SML_MOE_MAX_K];
    const int k = sml_moe_route(h, ly, lg, c->moe_s, c->moe_c, sel, w);
    float * acc = c->moe_acc;
    memset(acc, 0, (size_t) n_embd * sizeof(float));
    for (int j = 0; j < k; j++) {
        const int e = sel[j];
        const int fe = (int) ly->eu[e]->rows;          // this expert's width
        if (!ly->eg[e]) {                              // ungated (Nemotron-H): up, the FFN activation, down
            matvec(c, c->moe_g, ly->eu[e], c->xb, depth);
            ffn_act(c, c->moe_g, fe);
            matvec(c, c->moe_tmp, ly->ed[e], c->moe_g, depth);
            for (int i = 0; i < n_embd; i++) acc[i] += w[j] * c->moe_tmp[i];
            continue;
        }
        matvec(c, c->moe_g, ly->eg[e], c->xb, depth);
        matvec(c, c->moe_u, ly->eu[e], c->xb, depth);
        if (ly->egb) for (int i = 0; i < fe; i++) { c->moe_g[i] += ly->egb[e][i]; c->moe_u[i] += ly->eub[e][i]; }
        gate_mul(c, c->moe_g, c->moe_u, fe);
        matvec(c, c->moe_tmp, ly->ed[e], c->moe_g, depth);
        if (ly->edb) for (int i = 0; i < n_embd; i++) c->moe_tmp[i] += ly->edb[e][i];
        for (int i = 0; i < n_embd; i++) acc[i] += w[j] * c->moe_tmp[i];
    }
    if (ly->shu) {
        const int fs = (int) ly->shu->rows;
        if (ly->shg) {
            matvec(c, c->moe_g, ly->shg, c->xb, depth);
            matvec(c, c->moe_u, ly->shu, c->xb, depth);
            gate_mul(c, c->moe_g, c->moe_u, fs);
        } else {
            matvec(c, c->moe_g, ly->shu, c->xb, depth);
            ffn_act(c, c->moe_g, fs);
        }
        matvec(c, c->moe_tmp, ly->shd, c->moe_g, depth);
        float sg = 1.0f;
        if (ly->shgate) {
            double d = 0.0;
            for (int i = 0; i < n_embd; i++) d += (double) ly->shgate[i] * c->xb[i];
            sg = 1.0f / (1.0f + expf(-(float) d));
        }
        for (int i = 0; i < n_embd; i++) acc[i] += sg * c->moe_tmp[i];
    }
    memcpy(c->xb, acc, (size_t) n_embd * sizeof(float));
}

// The same experts for nb slots at once: each slot's router picks its experts
// exactly as moe_ffn does; then every expert chosen by anyone reads its weights
// ONCE for all the slots that chose it (their normed rows gathered), instead of
// once per slot. Input c->bh [nb x n_embd], output c->bout [nb x n_embd]. Each
// slot adds its experts' outputs in its own choice order, then the shared
// expert -- the stepped path's order, so the two agree to rounding.
static int moe_batch_scratch(sml_ctx * c, const sml_layer * ly) {
    const int B = SML_MAX_BATCH, n_embd = (int) c->model->hdr->n_embd;
    const int k = SML_MOE_MAX_K;
    (void) ly;
    if (c->mb_x && c->mb_k >= k) return 1;
    free(c->mb_x); free(c->mb_g); free(c->mb_u); free(c->mb_o); free(c->mb_out);
    free(c->mb_w); free(c->mb_sg); free(c->mb_sel); free(c->mb_row);
    c->mb_x   = malloc((size_t) B * n_embd * sizeof(float));
    c->mb_g   = malloc((size_t) B * c->moe_ff * sizeof(float));
    c->mb_u   = malloc((size_t) B * c->moe_ff * sizeof(float));
    c->mb_o   = malloc((size_t) B * n_embd * sizeof(float));
    c->mb_out = malloc((size_t) B * k * n_embd * sizeof(float));
    c->mb_w   = malloc((size_t) B * k * sizeof(float));
    c->mb_sg  = malloc((size_t) B * sizeof(float));
    c->mb_sel = malloc((size_t) B * k * sizeof(int));
    c->mb_row = malloc((size_t) B * k * sizeof(int));
    c->mb_k = k;
    return c->mb_x && c->mb_g && c->mb_u && c->mb_o && c->mb_out && c->mb_w && c->mb_sg
           && c->mb_sel && c->mb_row;
}

static void moe_expert_batch(sml_ctx * c, const sml_rec * g, const sml_rec * u, const sml_rec * d,
                             const float * gb, const float * ub, const float * db,
                             const float * xs, int m, int depth) {
    const int fe = (int) u->rows;
    if (!g) {                                          // ungated (Nemotron-H)
        matvec_batch(c, c->mb_g, u, xs, (int) u->cols, m, depth);
        for (int r = 0; r < m; r++) ffn_act(c, c->mb_g + (size_t) r * fe, fe);
        matvec_batch(c, c->mb_o, d, c->mb_g, fe, m, depth);
        return;
    }
    if (u->perm_off == g->perm_off) {
        const sml_rec * const rs[2] = { g, u };
        float * const ys[2] = { c->mb_g, c->mb_u };
        batch_group(c, rs, ys, 2, xs, (int) g->cols, m, depth);
    } else {
        matvec_batch(c, c->mb_g, g, xs, (int) g->cols, m, depth);
        matvec_batch(c, c->mb_u, u, xs, (int) u->cols, m, depth);
    }
    if (gb)
        for (int r = 0; r < m; r++)
            for (int i = 0; i < fe; i++) { c->mb_g[(size_t) r * fe + i] += gb[i]; c->mb_u[(size_t) r * fe + i] += ub[i]; }
    for (int r = 0; r < m; r++) gate_mul(c, c->mb_g + (size_t) r * fe, c->mb_u + (size_t) r * fe, fe);
    matvec_batch(c, c->mb_o, d, c->mb_g, fe, m, depth);
    if (db) {
        const int no = (int) d->rows;
        for (int r = 0; r < m; r++) for (int i = 0; i < no; i++) c->mb_o[(size_t) r * no + i] += db[i];
    }
}

static void moe_ffn_batch(sml_ctx * c, const sml_layer * ly, int nb, int depth) {
    const sml_hdr * h = c->model->hdr;
    if (!c->mb_x && !moe_batch_scratch(c, ly)) {
        fputs("sml: out of memory for batched experts\n", stderr);
        exit(1);
    }
    const int n_embd = (int) h->n_embd, ne = ly->n_exp, K = c->mb_k;
    int kk[SML_MAX_BATCH];
    for (int b = 0; b < nb; b++) {
        const float * x = c->bh + (size_t) b * n_embd;
        float * lg = c->moe_logit;
        for (int e = 0; e < ne; e++) {
            const float * r = ly->router + (size_t) e * n_embd;
            double s = 0.0;
            for (int i = 0; i < n_embd; i++) s += (double) r[i] * x[i];
            lg[e] = (float) s + (ly->router_b ? ly->router_b[e] : 0.0f);
        }
        kk[b] = sml_moe_route(h, ly, lg, c->moe_s, c->moe_c, c->mb_sel + (size_t) b * K, c->mb_w + (size_t) b * K);
        if (ly->shgate) {
            double s = 0.0;
            for (int i = 0; i < n_embd; i++) s += (double) ly->shgate[i] * x[i];
            c->mb_sg[b] = 1.0f / (1.0f + expf(-(float) s));
        } else {
            c->mb_sg[b] = 1.0f;
        }
    }
    for (int e = 0; e < ne; e++) {
        int m = 0;
        for (int b = 0; b < nb; b++)
            for (int j = 0; j < kk[b]; j++)
                if (c->mb_sel[(size_t) b * K + j] == e) {
                    memcpy(c->mb_x + (size_t) m * n_embd, c->bh + (size_t) b * n_embd, (size_t) n_embd * sizeof(float));
                    c->mb_row[m++] = b * K + j;
                }
        if (!m) continue;
        moe_expert_batch(c, ly->eg[e], ly->eu[e], ly->ed[e], ly->egb ? ly->egb[e] : NULL,
                         ly->eub ? ly->eub[e] : NULL, ly->edb ? ly->edb[e] : NULL, c->mb_x, m, depth);
        for (int r = 0; r < m; r++)
            memcpy(c->mb_out + (size_t) c->mb_row[r] * n_embd, c->mb_o + (size_t) r * n_embd,
                   (size_t) n_embd * sizeof(float));
    }
    for (int b = 0; b < nb; b++) {
        float * acc = c->bout + (size_t) b * n_embd;
        memset(acc, 0, (size_t) n_embd * sizeof(float));
        for (int j = 0; j < kk[b]; j++) {
            const float w = c->mb_w[(size_t) b * K + j];
            const float * o = c->mb_out + ((size_t) b * K + j) * n_embd;
            for (int i = 0; i < n_embd; i++) acc[i] += w * o[i];
        }
    }
    if (ly->shu) {
        moe_expert_batch(c, ly->shg, ly->shu, ly->shd, NULL, NULL, NULL, c->bh, nb, depth);
        for (int b = 0; b < nb; b++) {
            float * acc = c->bout + (size_t) b * n_embd;
            const float * o = c->mb_o + (size_t) b * n_embd;
            const float sg = c->mb_sg[b];
            for (int i = 0; i < n_embd; i++) acc[i] += sg * o[i];
        }
    }
}

const float * sml_forward(sml_ctx * c, int token, int pos, int depth) {
    if (c->ssm_state) {
        // A state-space layer's state is the whole past: position 0 starts it
        // afresh, and any other position must follow the last one run.
        if (pos == 0) {
            const sml_hdr * hs = c->model->hdr;
            for (uint32_t L = 0; L < hs->n_layer; L++) {
                const sml_layer * ly = &c->layers[L];
                if (ly->gdn_qkvz) {
                    const int cd = 2 * (int) (hs->gdn_nk * hs->gdn_dk) + ly->gdn_nv * ly->gdn_dv;
                    memset(c->ssm_state[L], 0, (size_t) ly->gdn_nv * hs->gdn_dk * ly->gdn_dv * sizeof(float));
                    memset(c->ssm_cst[L], 0, (size_t) (ly->gdn_K > 1 ? ly->gdn_K - 1 : 1) * cd * sizeof(float));
                    continue;
                }
                if (ly->m1_in) {
                    memset(c->ssm_state[L], 0, (size_t) ly->m1_inner * ly->m1_N * sizeof(float));
                    memset(c->ssm_cst[L], 0, (size_t) (ly->m1_K > 1 ? ly->m1_K - 1 : 1) * ly->m1_inner * sizeof(float));
                    continue;
                }
                if (!ly->ssm_in) continue;
                memset(c->ssm_state[L], 0, (size_t) ly->ssm_heads * (ly->ssm_inner / ly->ssm_heads)
                       * hs->ssm_dstate * sizeof(float));
                memset(c->ssm_cst[L], 0, (size_t) (ly->ssm_K > 1 ? ly->ssm_K - 1 : 1) * ly->ssm_convdim * sizeof(float));
            }
        } else if (pos != c->ssm_next) {
            fprintf(stderr, "sml: a state-space model must run its positions in order "
                    "(asked for %d after %d)\n", pos, c->ssm_next - 1);
        }
        c->ssm_next = pos + 1;
    }
    const sml_model * m = c->model;
    const sml_hdr * h = m->hdr;
    if (depth <= 0) depth = c->depth;

    const int n_embd = (int) h->n_embd;
    const int n_head = (int) h->n_head;
    const int n_kv   = (int) h->n_head_kv;
    const int hd     = c->head_dim;
    const int kvd    = n_kv * hd;
    const int rep    = n_head / n_kv;

    // Embedding lookup: one row, not a stream -- the only laddered tensor in the
    // forward pass that is not a matvec. Tied weights mean this same tensor is
    // also the output head, where it IS streamed in full.
    //
    // Always read at full depth: it is a single row, so the bytes are free, and
    // the residual stream is the one place a coarse approximation compounds
    // through every layer that follows.
    const sml_rec * emb = c->emb;
    if (c->x_in) {
        memcpy(c->x, c->x_in, (size_t) n_embd * sizeof(float));
    } else {
        const uint8_t * planes[SML_MAX_LEVELS];
        for (uint32_t lv = 0; lv < h->n_levels; lv++)
            planes[lv] = sml_ptr(m, emb->codes_off[lv]);
        if (emb->kind == SML_KIND_F32) {
            const float * w = (const float *) sml_ptr(m, emb->off);
            memcpy(c->xb, w + (size_t) token * emb->cols, (size_t) n_embd * sizeof(float));
        } else if (emb->layout == SML_LAYOUT_FLAT && (emb->nbits == 3 || emb->nbits == 5 || emb->nbits == 7)) {
            dequant_row_mixed(c->xb, m, emb, token);
        } else if (emb->layout == SML_LAYOUT_FLAT && emb->nbits == 2) {
            const uint8_t * p2[1] = { sml_ptr(m, emb->codes_off[0]) };
            sml_dequant_row(c->xb, p2, 1, 1, (const uint16_t *) sml_ptr(m, emb->sd_off),
                (const uint8_t *) sml_ptr(m, emb->sq_off),
                emb->mq_n ? (const uint16_t *) sml_ptr(m, emb->md_off) : NULL,
                emb->mq_n ? (const uint8_t  *) sml_ptr(m, emb->mq_off) : NULL,
                token, (int) emb->cols);
        } else if (emb->layout == SML_LAYOUT_FLAT && emb->nbits == 8) {
            sml_dequant_row_flat8(c->xb, sml_ptr(m, emb->codes_off[0]), sml_ptr(m, emb->codes_off[1]),
                (const uint16_t *) sml_ptr(m, emb->sd_off), (const uint8_t *) sml_ptr(m, emb->sq_off),
                emb->mq_n ? (const uint16_t *) sml_ptr(m, emb->md_off) : NULL,
                emb->mq_n ? (const uint8_t  *) sml_ptr(m, emb->mq_off) : NULL,
                token, (int) emb->cols);
        } else if (emb->layout == SML_LAYOUT_FLAT && emb->nbits == 6) {
            sml_dequant_row_flat6(c->xb,
                sml_ptr(m, emb->codes_off[1]), sml_ptr(m, emb->codes_off[0]),
                (const uint16_t *) sml_ptr(m, emb->sd_off),
                (const uint8_t  *) sml_ptr(m, emb->sq_off),
                emb->mq_n ? (const uint16_t *) sml_ptr(m, emb->md_off) : NULL,
                emb->mq_n ? (const uint8_t  *) sml_ptr(m, emb->mq_off) : NULL,
                token, (int) emb->cols);
        } else if (emb->layout == SML_LAYOUT_FLAT) {
            sml_dequant_row_flat4(c->xb, sml_ptr(m, emb->codes_off[0]),
                (const uint16_t *) sml_ptr(m, emb->sd_off),
                (const uint8_t  *) sml_ptr(m, emb->sq_off),
                emb->mq_n ? (const uint16_t *) sml_ptr(m, emb->md_off) : NULL,
                emb->mq_n ? (const uint8_t  *) sml_ptr(m, emb->mq_off) : NULL,
                token, (int) emb->cols);
        } else {
        sml_dequant_row(c->xb, planes, (int) h->n_levels, (int) h->n_levels,
                        (const uint16_t *) sml_ptr(m, emb->sd_off),
                        (const uint8_t  *) sml_ptr(m, emb->sq_off),
                        emb->mq_n ? (const uint16_t *) sml_ptr(m, emb->md_off) : NULL,
                        emb->mq_n ? (const uint8_t  *) sml_ptr(m, emb->mq_off) : NULL,
                        token, (int) emb->cols);
        }
        // the row came back in permuted order; scatter it back
        if (emb->kind == SML_KIND_F32) memcpy(c->x, c->xb, (size_t) n_embd * sizeof(float));
        else {
            const int32_t * perm = (const int32_t *) sml_ptr(m, emb->perm_off);
            for (int i = 0; i < n_embd; i++) c->x[perm[i]] = c->xb[i];
        }
        // An image row stands in for this token (vision models); the
        // reference replaces the embedding and scales nothing after it.
        if (c->emb_in) memcpy(c->x, c->emb_in, (size_t) n_embd * sizeof(float));
        if (h->emb_scale != 0.0f) for (int i = 0; i < n_embd; i++) c->x[i] *= h->emb_scale;
        if (c->pos_embd) {
            const float * pr = c->pos_embd + (size_t) pos * n_embd;
            for (int i = 0; i < n_embd; i++) c->x[i] += pr[i];
        }
        if (c->emb_norm) norm(c, c->x, c->x, c->emb_norm, c->emb_norm_b, n_embd);
        // See the batch path: the lookup needs E[r], not E[r]H. H is involutive.
        if (m->hdr->flags & SML_MODEL_HADAMARD) {
            sml_hadamard32(c->xr, c->x, n_embd);
            memcpy(c->x, c->xr, (size_t) n_embd * sizeof(float));
        }
        dump_x(c->x, n_embd);
    }

    for (int L = c->layer0; L < (int) h->n_layer; L++) {
        const sml_layer * ly = &c->layers[L];
        if (ly->attn_norm)
            norm(c, c->xb, c->x, (const float *) sml_ptr(m, ly->attn_norm->off),
                 ly->attn_norm_b, n_embd);
        else
            memcpy(c->xb, c->x, (size_t) n_embd * sizeof(float));   // OLMo 2: raw stream

        if (ly->gdn_qkvz) {
            gdn_block(c, ly, L, depth);       // Gated DeltaNet (linear attention)
        } else if (ly->m1_in) {
            m1_block(c, ly, L, depth);        // Mamba-1
        } else if (ly->ssm_in && !ly->q) {
            ssm_block(c, ly, L, depth);       // Mamba-2: a state-space mixer, not attention
        } else if (ly->sc_in) {
            sc_block(c, ly, L, pos, depth);   // LFM2: a short convolution, not attention
        } else if (!ly->q && !ly->mla_kva) {
            // a layer with no mixer at all (Nemotron-H's MLP layers): nothing to add
            memset(c->hb2, 0, (size_t) n_embd * sizeof(float));
        } else {
            if (ly->ssm_in) {                 // Falcon-H1: Mamba beside attention, same input, summed
                ssm_block(c, ly, L, depth);
                memcpy(c->hyb, c->hb2, (size_t) n_embd * sizeof(float));
            }
            if (ly->mla_kva) mla_qkv(c, ly, c->xb, c->q, c->k, c->v, depth);
            else {
                float * const yq[3] = { c->q, c->k, c->v };
                matvecN(c, ly->n_agrp, yq, ly->agrp, c->xb, depth);
            }
            if (ly->attn_gate) matvec(c, c->agate, ly->attn_gate, c->xb, depth);
            if (ly->q_bias) for (int i = 0; i < n_head * hd; i++) c->q[i] += ly->q_bias[i];
            if (ly->k_bias) for (int i = 0; i < kvd; i++) c->k[i] += ly->k_bias[i];
            if (ly->v_bias) for (int i = 0; i < kvd; i++) c->v[i] += ly->v_bias[i];
            if (h->clip_qkv > 0.0f) {   // OLMo, MPT clamp the projections
                const float cl = h->clip_qkv;
                for (int i = 0; i < n_head * hd; i++) c->q[i] = c->q[i] > cl ? cl : c->q[i] < -cl ? -cl : c->q[i];
                for (int i = 0; i < kvd; i++) c->k[i] = c->k[i] > cl ? cl : c->k[i] < -cl ? -cl : c->k[i];
                for (int i = 0; i < kvd; i++) c->v[i] = c->v[i] > cl ? cl : c->v[i] < -cl ? -cl : c->v[i];
            }
            // HunYuan normalizes Q and K AFTER rotating them; everyone else before
            const int qk_late = h->version >= 5 && (h->arch_bits & SML_ARCH_QKNORM_LATE);
            if (!qk_late && ly->q_norm) qk_norm(c->q, ly->q_norm, ly->q_norm_b, ly->q_norm_n, n_head, hd, h->rms_eps, ly->qk_heads);
            if (!qk_late && ly->k_norm) qk_norm(c->k, ly->k_norm, ly->k_norm_b, ly->k_norm_n, n_kv,   hd, h->rms_eps, ly->qk_heads);

            if (!c->no_rope) {   // learned positions or ALiBi: no rotary at all
              const double tr = sml_now();
              rope_at(c, c->q, n_head, hd, pos, L);
              rope_at(c, c->k, n_kv,   hd, pos, L);
              sml_t_rope += sml_now() - tr; }
            if (qk_late && ly->q_norm) qk_norm(c->q, ly->q_norm, ly->q_norm_b, ly->q_norm_n, n_head, hd, h->rms_eps, ly->qk_heads);
            if (qk_late && ly->k_norm) qk_norm(c->k, ly->k_norm, ly->k_norm_b, ly->k_norm_n, n_kv,   hd, h->rms_eps, ly->qk_heads);

            f32_to_f16(c->k_cache + ((size_t) L * c->n_ctx + pos) * kvd, c->k, kvd);
            f32_to_f16(c->v_cache + ((size_t) L * c->n_ctx + pos) * kvd, c->v, kvd);

            const double ta = sml_now();
            // Gemma scales by a declared constant rather than 1/sqrt(head_dim).
            const float inv = (h->version >= 4 && h->attn_scale != 0.0f)
                              ? h->attn_scale : 1.0f / sqrtf((float) hd);
            // Sliding window: attend only to the last n_swa positions. 0 = full.
            const int swa = c->layer_win ? c->layer_win[L] : (h->version >= 4) ? (int) h->n_swa : 0;
            const float cap = h->attn_softcap;
            // Heads are independent (own score row, own output slice), so they go
            // across the team. Serial, this walk grew with the context until it was
            // 6.5 ms of a 44 ms token at 256 generated tokens on the 1B.
            if (use_pool()) {
                attn_job j = { c, L, pos, hd, kvd, rep, swa, inv, cap, ly->sinks };
                sml_pool_for(n_head, attn_head, &j);
            } else {
                attn_job j = { c, L, pos, hd, kvd, rep, swa, inv, cap, ly->sinks };
    #pragma omp parallel for schedule(static) if (pos >= 8 && c->n_threads > 1)
                for (int hh = 0; hh < n_head; hh++) attn_head(&j, hh);
            }

            sml_t_attn += sml_now() - ta;
            if (ly->attn_gate)
                for (int i = 0; i < (int) ly->attn_gate->rows; i++)
                    c->xb[i] *= 1.0f / (1.0f + expf(-c->agate[i]));
            if (ly->mla_kva) mla_pack(ly, c->xb, n_head, hd);
            matvec(c, c->hb2, ly->o, c->xb, depth);
            if (ly->o_bias) for (int i = 0; i < n_embd; i++) c->hb2[i] += ly->o_bias[i];
            if (ly->ssm_in) for (int i = 0; i < n_embd; i++) c->hb2[i] += c->hyb[i];
        }
        if (ly->attn_post_norm) {
            norm(c, c->xb, c->hb2, ly->attn_post_norm, NULL, n_embd);
            memcpy(c->hb2, c->xb, (size_t) n_embd * sizeof(float));
        }

        // Parallel residual (GPT-NeoX): attention and FFN both read the
        // PRE-attention residual and both add into it, instead of the FFN
        // reading the post-attention result. Feeding one into the other is a
        // different model, and it produces fluent output either way.
        const int par = (h->version >= 5) &&
                        (h->arch_bits & SML_ARCH_PARALLEL_RESIDUAL);
        // Granite scales each residual branch; x * 1.0f is exact, so every other
        // model computes bit-for-bit what it did before.
        const float rsc = h->res_scale != 0.0f ? h->res_scale : 1.0f;
        if (par) {
            norm(c, c->xb, c->x, (const float *) sml_ptr(m, ly->ffn_norm->off),
                 ly->ffn_norm_b, n_embd);
            for (int i = 0; i < n_embd; i++) c->x[i] += rsc * c->hb2[i];
        } else {
            for (int i = 0; i < n_embd; i++) c->x[i] += rsc * c->hb2[i];
            if (ly->ffn_norm)
                norm(c, c->xb, c->x, (const float *) sml_ptr(m, ly->ffn_norm->off),
                     ly->ffn_norm_b, n_embd);
            else
                memcpy(c->xb, c->x, (size_t) n_embd * sizeof(float));
        }

        const int n_ff = (int) h->n_ff;
        if (!ly->n_exp && !ly->down) { dump_x(c->x, n_embd); continue; }
        if (ly->n_exp) {
            moe_ffn(c, ly, depth);
            if (ly->ffn_post_norm) {
                norm(c, c->hb2, c->xb, ly->ffn_post_norm, NULL, n_embd);
                memcpy(c->xb, c->hb2, (size_t) n_embd * sizeof(float));
            }
            for (int i = 0; i < n_embd; i++) c->x[i] += rsc * c->xb[i];
            dump_x(c->x, n_embd);
            continue;
        }
        if (ly->gate) {
            // second group: gate and up both read the post-ffn_norm activation
            float * const yf[2] = { c->hb, c->hb2 };
            matvecN(c, ly->n_fgrp, yf, ly->fgrp, c->xb, depth);
            if (ly->gate_bias) for (int i = 0; i < n_ff; i++) c->hb[i]  += ly->gate_bias[i];
            if (ly->up_bias)   for (int i = 0; i < n_ff; i++) c->hb2[i] += ly->up_bias[i];
            gate_mul(c, c->hb, c->hb2, n_ff);   // SwiGLU, or GeGLU
        } else {
            // no gate tensor => GELU FFN. Structure read from the tensor table
            // rather than declared per architecture.
            matvec(c, c->hb, ly->up, c->xb, depth);
            if (ly->up_bias) for (int i = 0; i < n_ff; i++) c->hb[i] += ly->up_bias[i];
            if (ly->xielu) xielu_act(ly->xielu, c->hb, n_ff); else ffn_act(c, c->hb, n_ff);
        }
        matvec(c, c->xb, ly->down, c->hb, depth);
        if (ly->down_bias) for (int i = 0; i < n_embd; i++) c->xb[i] += ly->down_bias[i];
        if (ly->ffn_post_norm) {
            norm(c, c->hb2, c->xb, ly->ffn_post_norm, NULL, n_embd);
            memcpy(c->xb, c->hb2, (size_t) n_embd * sizeof(float));
        }
        for (int i = 0; i < n_embd; i++) c->x[i] += rsc * c->xb[i];
        dump_x(c->x, n_embd);
    }

    norm(c, c->xb, c->x, (const float *) sml_ptr(m, c->out_norm->off),
         c->out_norm_b, n_embd);
    matvec(c, c->logits, c->head, c->xb, depth);
    if (c->out_b) for (uint32_t i = 0; i < h->n_vocab; i++) c->logits[i] += c->out_b[i];
    if (h->logit_scale != 0.0f)
        for (uint32_t i = 0; i < h->n_vocab; i++) c->logits[i] /= h->logit_scale;
    // Gemma bounds its logits with a tanh soft cap.
    if (h->version >= 4 && h->logit_softcap != 0.0f) {
        const float cap = h->logit_softcap;
        for (uint32_t i = 0; i < h->n_vocab; i++)
            c->logits[i] = cap * tanhf(c->logits[i] / cap);
    }

    c->tokens++;
    return c->logits;
}

int sml_cache_prefix(sml_ctx * c, const int * ids, int n) {
    // a state-space model's state is not per position: nothing can be reused
    if (c->ssm_state) return 0;
    int L = 0;
    while (L < n - 1 && L < c->n_hist && c->hist[L] == ids[L]) L++;
    return L;
}

void sml_cache_commit(sml_ctx * c, const int * ids, int n) {
    if (n > c->n_ctx) n = c->n_ctx;
    memcpy(c->hist, ids, (size_t) n * sizeof(int));
    c->n_hist = n;
}

int sml_sample_greedy(const float * logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

float sml_top2_margin(const float * logits, int n) {
    float a = -1e30f, b = -1e30f;
    for (int i = 0; i < n; i++) {
        const float v = logits[i];
        if (v > a) { b = a; a = v; }
        else if (v > b) { b = v; }
    }
    return a - b;
}
