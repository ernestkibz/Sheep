// The vision tower. Written from the model's own definition (patch matmul,
// pre-norm ViT blocks, post-norm, pixel shuffle, projection) and graded against
// the HF reference by tooling/sml_vlm_grade.py -- component by component, since
// a tower that is slightly wrong still hands the text model plausible vectors.
//
// Weights are full-precision f32 for now: the tower is ~86M parameters, read
// once per image rather than once per token, so it is compute, not bandwidth.
#include "sml-vision.h"
#include "sml-json.h"
#include "sml-vocab.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

int sml_vision_cfg(const sml_model * m, sml_vcfg * v) {
    const char * js = (const char *) sml_ptr(m, m->hdr->json_off);
    const size_t jn = (size_t) m->hdr->json_n;
    const sml_json o = sml_json_get(js, jn, "vision");
    if (o.kind != 'o') return 0;
    memset(v, 0, sizeof *v);
    v->n_embd  = (int) sml_json_int(sml_json_get(o.p, o.n, "n_embd"), 0);
    v->n_layer = (int) sml_json_int(sml_json_get(o.p, o.n, "n_layer"), 0);
    v->n_head  = (int) sml_json_int(sml_json_get(o.p, o.n, "n_head"), 0);
    v->n_ff    = (int) sml_json_int(sml_json_get(o.p, o.n, "n_ff"), 0);
    v->image   = (int) sml_json_int(sml_json_get(o.p, o.n, "image"), 0);
    v->patch   = (int) sml_json_int(sml_json_get(o.p, o.n, "patch"), 0);
    v->scale   = (int) sml_json_int(sml_json_get(o.p, o.n, "scale"), 1);
    v->image_token_id = (int) sml_json_int(sml_json_get(o.p, o.n, "image_token_id"), -1);
    v->eps     = (float) sml_json_num(sml_json_get(o.p, o.n, "eps"), 1e-6);
    v->tok_start = v->tok_fake = v->tok_global = v->tok_end = -1;
    const sml_json ch = sml_json_get(o.p, o.n, "chat");
    if (ch.kind == 'o') {
        v->tok_start  = (int) sml_json_int(sml_json_get(ch.p, ch.n, "start"), -1);
        v->tok_fake   = (int) sml_json_int(sml_json_get(ch.p, ch.n, "fake"), -1);
        v->tok_global = (int) sml_json_int(sml_json_get(ch.p, ch.n, "global"), -1);
        v->tok_end    = (int) sml_json_int(sml_json_get(ch.p, ch.n, "end"), -1);
        sml_json_str(sml_json_get(ch.p, ch.n, "user"), v->user, sizeof v->user);
        sml_json_str(sml_json_get(ch.p, ch.n, "assistant"), v->assistant, sizeof v->assistant);
    }
    if (v->n_embd <= 0 || v->n_head <= 0 || v->patch <= 0 || v->image < v->patch || v->scale <= 0)
        return 0;
    const int side = v->image / v->patch;
    v->n_patch = side * side;
    v->n_tok   = v->n_patch / (v->scale * v->scale);
    v->out_dim = (int) m->hdr->n_embd;
    return 1;
}

static const float * vt(const sml_model * m, const char * name) {
    const sml_rec * r = sml_find(m, name);
    return r ? (const float *) sml_ptr(m, r->off) : NULL;
}

static const float * vtl(const sml_model * m, const char * fmt, int layer) {
    const sml_rec * r = sml_findf(m, fmt, layer);
    return r ? (const float *) sml_ptr(m, r->off) : NULL;
}

#if defined(__AVX2__)
static inline float hsum8(__m256 a) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

static inline float dotf(const float * a, const float * b, int k) {
    int i = 0;
    float s = 0.0f;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= k; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    s = hsum8(acc);
#endif
    for (; i < k; i++) s += a[i] * b[i];
    return s;
}

// Y[M x N] = X[M x K] . W[N x K]^T (+ b). W is walked in panels small enough to
// stay in one core's L2 while every row of X passes over it; rows of X go out
// to threads two at a time against four panel rows (eight accumulators).
static void gemm_nt(float * Y, const float * X, const float * W, const float * b,
                    int M, int N, int K, int nt) {
    int pr = (int) ((256 * 1024) / ((size_t) K * sizeof(float)));
    pr = pr < 4 ? 4 : (pr & ~3);
    for (int j0 = 0; j0 < N; j0 += pr) {
        const int j1 = j0 + pr < N ? j0 + pr : N;
        #pragma omp parallel for num_threads(nt) schedule(static)
        for (int i = 0; i < M; i += 2) {
            const int two = i + 1 < M;
            const float * x0 = X + (size_t) i * K;
            const float * x1 = two ? x0 + K : x0;
            int j = j0;
#if defined(__AVX2__)
            if (K % 8 == 0)
                for (; j + 4 <= j1; j += 4) {
                    const float * w0 = W + (size_t) j * K, * w1 = w0 + K, * w2 = w1 + K, * w3 = w2 + K;
                    __m256 a00 = _mm256_setzero_ps(), a01 = a00, a02 = a00, a03 = a00;
                    __m256 a10 = a00, a11 = a00, a12 = a00, a13 = a00;
                    for (int k = 0; k < K; k += 8) {
                        const __m256 u = _mm256_loadu_ps(x0 + k), v = _mm256_loadu_ps(x1 + k);
                        __m256 w = _mm256_loadu_ps(w0 + k);
                        a00 = _mm256_fmadd_ps(u, w, a00); a10 = _mm256_fmadd_ps(v, w, a10);
                        w = _mm256_loadu_ps(w1 + k);
                        a01 = _mm256_fmadd_ps(u, w, a01); a11 = _mm256_fmadd_ps(v, w, a11);
                        w = _mm256_loadu_ps(w2 + k);
                        a02 = _mm256_fmadd_ps(u, w, a02); a12 = _mm256_fmadd_ps(v, w, a12);
                        w = _mm256_loadu_ps(w3 + k);
                        a03 = _mm256_fmadd_ps(u, w, a03); a13 = _mm256_fmadd_ps(v, w, a13);
                    }
                    float * y0 = Y + (size_t) i * N + j;
                    y0[0] = hsum8(a00) + (b ? b[j]     : 0.0f);
                    y0[1] = hsum8(a01) + (b ? b[j + 1] : 0.0f);
                    y0[2] = hsum8(a02) + (b ? b[j + 2] : 0.0f);
                    y0[3] = hsum8(a03) + (b ? b[j + 3] : 0.0f);
                    if (two) {
                        float * y1 = y0 + N;
                        y1[0] = hsum8(a10) + (b ? b[j]     : 0.0f);
                        y1[1] = hsum8(a11) + (b ? b[j + 1] : 0.0f);
                        y1[2] = hsum8(a12) + (b ? b[j + 2] : 0.0f);
                        y1[3] = hsum8(a13) + (b ? b[j + 3] : 0.0f);
                    }
                }
#endif
            for (; j < j1; j++) {
                const float * w = W + (size_t) j * K;
                Y[(size_t) i * N + j] = dotf(x0, w, K) + (b ? b[j] : 0.0f);
                if (two) Y[(size_t)(i + 1) * N + j] = dotf(x1, w, K) + (b ? b[j] : 0.0f);
            }
        }
    }
}

static void layer_norm_rows(float * out, const float * x, const float * w, const float * b,
                            int rows, int n, float eps, int nt) {
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int r = 0; r < rows; r++) {
        const float * xr = x + (size_t) r * n;
        float * o = out + (size_t) r * n;
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < n; i++) mean += xr[i];
        mean /= n;
        for (int i = 0; i < n; i++) { const double d = xr[i] - mean; var += d * d; }
        const float s = 1.0f / sqrtf((float) (var / n) + eps);
        for (int i = 0; i < n; i++) o[i] = (float) (xr[i] - mean) * s * w[i] + b[i];
    }
}

// Bidirectional attention over all patches: every patch sees every patch.
// q, k, v, out are [n_seq][n_embd], heads side by side in each row.
static void attend_all(float * out, const float * q, const float * k, const float * v,
                       int n_seq, int n_embd, int n_head, int nt) {
    const int hd = n_embd / n_head;
    const float scale = 1.0f / sqrtf((float) hd);
    #pragma omp parallel num_threads(nt)
    {
        float * s  = malloc((size_t) n_seq * sizeof(float));
        float * kh = malloc((size_t) n_seq * hd * sizeof(float));
        for (int h = 0; h < n_head; h++) {
            // each thread gathers this head's keys once, then takes its queries
            for (int t = 0; t < n_seq; t++)
                memcpy(kh + (size_t) t * hd, k + (size_t) t * n_embd + (size_t) h * hd, hd * sizeof(float));
            #pragma omp for schedule(static)
            for (int i = 0; i < n_seq; i++) {
                const float * qi = q + (size_t) i * n_embd + (size_t) h * hd;
                float mx = -INFINITY;
                for (int t = 0; t < n_seq; t++) {
                    s[t] = dotf(qi, kh + (size_t) t * hd, hd) * scale;
                    if (s[t] > mx) mx = s[t];
                }
                double sum = 0.0;
                for (int t = 0; t < n_seq; t++) { s[t] = expf(s[t] - mx); sum += s[t]; }
                const float inv = (float) (1.0 / sum);
                float * o = out + (size_t) i * n_embd + (size_t) h * hd;
                memset(o, 0, hd * sizeof(float));
                for (int t = 0; t < n_seq; t++) {
                    const float p = s[t] * inv;
                    const float * vt_ = v + (size_t) t * n_embd + (size_t) h * hd;
                    for (int d = 0; d < hd; d++) o[d] += p * vt_[d];
                }
            }
        }
        free(s);
        free(kh);
    }
}

static void gelu_tanh_rows(float * x, size_t n, int nt) {
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (long i = 0; i < (long) n; i++) {
        const float u = x[i];
        x[i] = 0.5f * u * (1.0f + tanhf(0.7978845608f * (u + 0.044715f * u * u * u)));
    }
}

int sml_vision_prompt(const sml_vcfg * v, const void * vocab, const char * question,
                      int32_t * ids, int max) {
    const sml_vocab * vb = (const sml_vocab *) vocab;
    if (v->tok_start < 0 || v->tok_fake < 0 || v->tok_global < 0 || v->tok_end < 0) return -1;
    int n = 0, k;
#define PUT(t) do { if (n >= max) return -1; ids[n++] = (t); } while (0)
#define ENC(s) do { k = sml_encode(vb, (s), ids + n, max - n); if (k < 0) return -1; n += k; } while (0)
    PUT(v->tok_start);
    ENC(v->user);
    PUT(v->tok_fake);
    PUT(v->tok_global);
    for (int i = 0; i < v->n_tok; i++) PUT(v->image_token_id);
    PUT(v->tok_fake);
    ENC(question);
    PUT(v->tok_end);
    ENC(v->assistant);
#undef PUT
#undef ENC
    return n;
}

int sml_vision_encode(const sml_model * m, const sml_vcfg * v, const float * pixels,
                      float * out, int nt) {
    const int D = v->n_embd, F = v->n_ff, P = v->n_patch, ps = v->patch;
    const int side = v->image / ps, sc = v->scale, pd = 3 * ps * ps;
    const float * wp = vt(m, "v.patch.weight"), * bp = vt(m, "v.patch.bias");
    const float * pos = vt(m, "v.pos.weight");
    const float * lnw = vt(m, "v.post_ln.weight"), * lnb = vt(m, "v.post_ln.bias");
    const float * proj = vt(m, "v.proj.weight");
    if (!wp || !bp || !pos || !lnw || !lnb || !proj) {
        fprintf(stderr, "sml: vision tower incomplete in this file\n");
        return -1;
    }
    const size_t PD = (size_t) P * D;
    float * x  = malloc(PD * sizeof(float));
    float * h  = malloc((size_t) P * (pd > D ? pd : D) * sizeof(float));
    float * q  = malloc(PD * sizeof(float));
    float * k  = malloc(PD * sizeof(float));
    float * vv = malloc(PD * sizeof(float));
    float * a  = malloc(PD * sizeof(float));
    float * f  = malloc((size_t) P * F * sizeof(float));
    int rc = -1;
    if (!x || !h || !q || !k || !vv || !a || !f) goto done;

    // Patches, cut in the (channel, row, column) order the conv kernel is stored
    // in, so the stride-equals-kernel convolution is one matmul.
    for (int py = 0; py < side; py++)
        for (int px = 0; px < side; px++) {
            float * row = h + (size_t) (py * side + px) * pd;
            for (int c = 0; c < 3; c++)
                for (int ky = 0; ky < ps; ky++)
                    memcpy(row + (size_t) c * ps * ps + (size_t) ky * ps,
                           pixels + ((size_t) c * v->image + (size_t) py * ps + ky) * v->image + (size_t) px * ps,
                           ps * sizeof(float));
        }
    gemm_nt(x, h, wp, bp, P, D, pd, nt);
    for (size_t i = 0; i < PD; i++) x[i] += pos[i];   // the file holds the gathered table

    for (int L = 0; L < v->n_layer; L++) {
        const float * w1 = vtl(m, "v.blk.%d.ln1.weight", L), * b1 = vtl(m, "v.blk.%d.ln1.bias", L);
        const float * wq = vtl(m, "v.blk.%d.q.weight", L),   * bq = vtl(m, "v.blk.%d.q.bias", L);
        const float * wk = vtl(m, "v.blk.%d.k.weight", L),   * bk = vtl(m, "v.blk.%d.k.bias", L);
        const float * wv = vtl(m, "v.blk.%d.v.weight", L),   * bv = vtl(m, "v.blk.%d.v.bias", L);
        const float * wo = vtl(m, "v.blk.%d.o.weight", L),   * bo = vtl(m, "v.blk.%d.o.bias", L);
        const float * w2 = vtl(m, "v.blk.%d.ln2.weight", L), * b2 = vtl(m, "v.blk.%d.ln2.bias", L);
        const float * wf1 = vtl(m, "v.blk.%d.fc1.weight", L), * bf1 = vtl(m, "v.blk.%d.fc1.bias", L);
        const float * wf2 = vtl(m, "v.blk.%d.fc2.weight", L), * bf2 = vtl(m, "v.blk.%d.fc2.bias", L);
        if (!w1 || !b1 || !wq || !bq || !wk || !bk || !wv || !bv || !wo || !bo ||
            !w2 || !b2 || !wf1 || !bf1 || !wf2 || !bf2) {
            fprintf(stderr, "sml: vision layer %d incomplete\n", L);
            goto done;
        }
        layer_norm_rows(h, x, w1, b1, P, D, v->eps, nt);
        gemm_nt(q,  h, wq, bq, P, D, D, nt);
        gemm_nt(k,  h, wk, bk, P, D, D, nt);
        gemm_nt(vv, h, wv, bv, P, D, D, nt);
        attend_all(a, q, k, vv, P, D, v->n_head, nt);
        gemm_nt(h, a, wo, bo, P, D, D, nt);
        for (size_t i = 0; i < PD; i++) x[i] += h[i];
        layer_norm_rows(h, x, w2, b2, P, D, v->eps, nt);
        gemm_nt(f, h, wf1, bf1, P, F, D, nt);
        gelu_tanh_rows(f, (size_t) P * F, nt);
        gemm_nt(h, f, wf2, bf2, P, D, F, nt);
        for (size_t i = 0; i < PD; i++) x[i] += h[i];
    }
    layer_norm_rows(h, x, lnw, lnb, P, D, v->eps, nt);

    // Pixel shuffle, from the reference's reshape/permute chain worked through
    // index by index: patch (y, x) lands in token (y/sc)*(side/sc) + x/sc, at
    // channel ((y%sc)*sc + x%sc)*D. Each token carries its sc x sc block.
    {
        const int ts = side / sc, cw = sc * sc * D;
        for (int y = 0; y < side; y++)
            for (int xx = 0; xx < side; xx++) {
                const int tok = (y / sc) * ts + xx / sc;
                const int ch  = ((y % sc) * sc + xx % sc) * D;
                memcpy(f + (size_t) tok * cw + ch, h + (size_t) (y * side + xx) * D, D * sizeof(float));
            }
        gemm_nt(out, f, proj, NULL, v->n_tok, v->out_dim, cw, nt);
    }
    rc = 0;
done:
    free(x); free(h); free(q); free(k); free(vv); free(a); free(f);
    return rc;
}
