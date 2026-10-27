/* The Sheep Quantizer's two searches, fused: each group of weights is read once
 * and every candidate is tried in registers.
 *
 * tooling/sml_quantize.py ran them in numpy: about 80 full passes over each
 * tensor, 0.5M weights a second on six cores, four hours for a 7B (2026-10-06;
 * Ernest: "someone just quantizes their choice, plugs it in, and off the bat
 * can use Sheep"). The arithmetic is the numpy code's, group by group, in
 * float32; sums run in order rather than numpy's pairwise order, so a near-tie
 * between two candidates can fall the other way. tooling/sml_grid_c.py loads
 * this library; tooling/sml_grid_gpu_check.py compares it with numpy.
 *
 *   gcc -O3 -march=native -fopenmp -shared -o sml-quant.dll sml-quant.c
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#define GMAX 256

static float err_of(const float *g, const float *w, int group, float l, float step, float n, int rem, float *q)
{
    float e = 0.0f;
    for (int i = 0; i < group; i++) {
        float c = rintf((g[i] - l) / step);
        if (c < 0.0f) c = 0.0f;
        if (c > n) c = n;
        q[i] = c;
        float qv = rem ? (float)((((int)c >> rem) << rem) + (1 << (rem - 1))) : c;
        float d = qv * step + l - g[i];
        e += d * d * w[i];
    }
    return e;
}

/* Per-group (min, step) of the asymmetric grid: ncand candidate ranges, each
 * refitted by weighted least squares (iters rounds) before the choice.
 * wp: rows x cols (columns already permuted); wt: cols activation weights,
 * already normalised (imp / mean + 1e-6), or NULL for ones; out: rows x ng. */
void sml_asym_grid(const float *wp, int rows, int cols, int group, int nbits, int rem, const float *wt,
                   const float *facs, int ncand, int iters, int refit, float *out_l, float *out_s)
{
    const int ng = cols / group;
    const float n = (float)((1 << nbits) - 1);
    float ones[GMAX];
    for (int i = 0; i < GMAX; i++) ones[i] = 1.0f;
#pragma omp parallel for schedule(dynamic, 4)
    for (int r = 0; r < rows; r++) {
        float q[GMAX], q2[GMAX];
        for (int j = 0; j < ng; j++) {
            const float *g = wp + (size_t)r * cols + (size_t)j * group;
            const float *w = wt ? wt + (size_t)j * group : ones;
            float lo = g[0], hi = g[0], ws = 0.0f;
            for (int i = 0; i < group; i++) {
                if (g[i] < lo) lo = g[i];
                if (g[i] > hi) hi = g[i];
                ws += w[i];
            }
            float c = (lo + hi) * 0.5f;
            float bl = 0, bs = 0, be = 0;
            for (int k = 0; k < ncand; k++) {
                float half = (hi - lo) * 0.5f * facs[k];
                float l = c - half;
                float step = fmaxf(2.0f * half, 1e-12f) / n;
                float e = err_of(g, w, group, l, step, n, rem, q);
                if (refit) {
                    for (int it = 0; it < iters; it++) {
                        float qm = 0, gm = 0;
                        for (int i = 0; i < group; i++) { qm += w[i] * q[i]; gm += w[i] * g[i]; }
                        qm /= ws; gm /= ws;
                        float var = 0, cov = 0;
                        for (int i = 0; i < group; i++) {
                            float dq = q[i] - qm;
                            var += w[i] * dq * dq;
                            cov += w[i] * dq * (g[i] - gm);
                        }
                        float s2 = var > 1e-12f ? cov / fmaxf(var, 1e-12f) : step;
                        s2 = fmaxf(s2, 1e-12f);
                        float l2 = gm - s2 * qm;
                        float e2 = err_of(g, w, group, l2, s2, n, rem, q2);
                        if (e2 < e) {
                            l = l2; step = s2; e = e2;
                            memcpy(q, q2, sizeof(float) * group);
                        }
                    }
                }
                if (k == 0 || e < be) { bl = l; bs = step; be = e; }
            }
            out_l[(size_t)r * ng + j] = bl;
            out_s[(size_t)r * ng + j] = bs;
        }
    }
}

/* Each group's stored step field (q8) and min field (mq8) moved to the best
 * neighbour within +-radius, by weighted error with the codes re-derived.
 * dv, dm: rows x ng, each group's superblock scale for step and min. */
void sml_field_search(const float *w, int rows, int cols, int group, const float *dv, const float *dm,
                      int32_t *q8, int32_t *mq8, int nmax, int hi, int moff, int mmax, const float *wt, int radius)
{
    const int ng = cols / group;
    float ones[GMAX];
    for (int i = 0; i < GMAX; i++) ones[i] = 1.0f;
#pragma omp parallel for schedule(dynamic, 4)
    for (int r = 0; r < rows; r++) {
        for (int j = 0; j < ng; j++) {
            const float *g = w + (size_t)r * cols + (size_t)j * group;
            const float *wg = wt ? wt + (size_t)j * group : ones;
            const size_t o = (size_t)r * ng + j;
            const float d = dv[o], m = dm[o];
            const int qs0 = q8[o], qm0 = mq8[o];
            float be = 0;
            int bs = qs0, bm = qm0, first = 1;
            for (int a = -radius; a <= radius; a++) {
                int qs = qs0 + a;
                if (qs < 1) qs = 1;
                if (qs > hi) qs = hi;
                float S = fmaxf((float)qs * d, 1e-12f);
                for (int b = -radius; b <= radius; b++) {
                    int qm = qm0 + b;
                    if (qm < 0) qm = 0;
                    if (qm > mmax) qm = mmax;
                    float M = ((float)qm - (float)moff) * m;
                    float e = 0;
                    for (int i = 0; i < group; i++) {
                        float c = rintf((g[i] - M) / S);
                        if (c < 0) c = 0;
                        if (c > (float)nmax) c = (float)nmax;
                        float dd = c * S + M - g[i];
                        e += dd * dd * wg[i];
                    }
                    if (first || e < be) { be = e; bs = qs; bm = qm; first = 0; }
                }
            }
            q8[o] = bs;
            mq8[o] = bm;
        }
    }
}
