// What does the SML kernel actually sustain, in isolation?
//
// Gate G2: dense decode must reach >= 0.9x the reference baseline. Measured on
// this machine (reference engine, Qwen2.5-3B 4-bit, r=5): 10.71 +/- 0.22 tok/s at
// -t 3, moving 1.92 GB/token => 20.6 GB/s.
//
// The kernel is bandwidth-bound by construction, so the number that matters is
// achieved GB/s on the weight stream; tok/s follows from it. This must be
// re-run whenever the kernel or the group size changes -- a figure measured
// against a different layout is not evidence about the current one.
//
//   gcc -O3 -march=native -fopenmp sml-kernel-bench.c sml-kernel.c -o bench -lm

#include "sml-kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
static double now_s(void) { return omp_get_wtime(); }
#else
#include <time.h>
static double now_s(void) { return (double) clock() / CLOCKS_PER_SEC; }
#endif

// Qwen2.5-3B-Instruct: what one token actually pulls across the bus.
#define N_LAYER  36
#define N_EMBD   2048
#define N_HEADKV 256
#define N_FF     11008
#define N_VOCAB  151936

static double weights_total(void) {
    double w = 0.0;
    w += (double) N_LAYER * N_EMBD   * N_EMBD;   // q
    w += (double) N_LAYER * N_HEADKV * N_EMBD;   // k
    w += (double) N_LAYER * N_HEADKV * N_EMBD;   // v
    w += (double) N_LAYER * N_EMBD   * N_EMBD;   // o
    w += (double) N_LAYER * N_FF     * N_EMBD;   // gate
    w += (double) N_LAYER * N_FF     * N_EMBD;   // up
    w += (double) N_LAYER * N_EMBD   * N_FF;     // down
    w += (double) N_VOCAB * N_EMBD;              // tied output head
    return w;
}

static double bytes_per_token(int depth) {
    const double per_w = depth * (double) SML_LEVEL_BITS / 8.0
                       + 2.0 / SML_SCALE_SUPER + 1.0 / SML_GROUP;
    return weights_total() * per_w;
}

int main(int argc, char ** argv) {
    int nth = argc > 1 ? atoi(argv[1]) : 4;
    const int reps = argc > 2 ? atoi(argv[2]) : 30;
    const int rows = 11008, n = 2048;   // the FFN shape that dominates

#ifdef _OPENMP
    omp_set_num_threads(nth);
#else
    nth = 1;
#endif

    const int npack  = n / SML_PACK_SUPER;
    const int ng     = n / SML_GROUP;
    const int nscale = n / SML_SCALE_SUPER;
    const size_t cbytes = (size_t) rows * npack * SML_PACK_BYTES;
    const size_t sbytes = (size_t) rows * (nscale * sizeof(uint16_t) + ng);

    // Working set size matters more than "bigger than L3". The engine streams
    // 1.66 GB per token from a 2.44 GB file; a 192 MB bench working set stays
    // far warmer than that, so a number measured there can overstate what the
    // engine can ever reach. Third argument sets it, in MB.
    const size_t budget = (size_t)(argc > 3 ? atoi(argv[3]) : 192) << 20;
    int copies = (int)(budget / (cbytes + sbytes));
    if (copies < 4) copies = 4;
    uint8_t  ** codes = malloc((size_t) copies * sizeof(void *));
    uint16_t ** sd    = malloc((size_t) copies * sizeof(void *));
    uint8_t  ** sq    = malloc((size_t) copies * sizeof(void *));
    for (int c = 0; c < copies; c++) {
        codes[c] = malloc(cbytes);
        sd[c]    = malloc((size_t) rows * nscale * sizeof(uint16_t));
        sq[c]    = malloc((size_t) rows * ng);
        if (!codes[c] || !sd[c] || !sq[c]) { copies = c; break; }
        memset(codes[c], (c * 37) & 0xff, cbytes);
        for (size_t i = 0; i < (size_t) rows * nscale; i++) sd[c][i] = 0x1c00; // 1.0
        memset(sq[c], 8, (size_t) rows * ng);
    }
    if (copies < 1) { fprintf(stderr, "no working set\n"); return 1; }

    float * x = malloc((size_t) n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = (float)((i % 200) - 100) / 100.0f;
    sml_act a;
    a.q = malloc((size_t) n);
    a.scale = malloc((size_t) ng * sizeof(float));
    a.sums = malloc((size_t) ng * sizeof(int32_t));
    sml_quantize_act(&a, x, n);
    float * y = calloc((size_t) rows, sizeof(float));

    printf("matvec %d x %d, group %d, %d threads, working set %.0f MB (L3 is 16 MB)\n",
           rows, n, SML_GROUP, nth, (double)(copies * (cbytes + sbytes)) / 1e6);

    double best = 0.0;
    for (int it = 0; it < 3; it++) {
        const double t0 = now_s();
        for (int r = 0; r < reps; r++) {
            const int c = r % copies;
#ifdef _OPENMP
#pragma omp parallel
            {
                const int t  = omp_get_thread_num();
                const int nt = omp_get_num_threads();
                const int lo = (int)((long) rows * t / nt);
                const int hi = (int)((long) rows * (t + 1) / nt);
                sml_matvec_rows(y, codes[c], sd[c], sq[c], NULL, NULL, &a, lo, hi, n, 0, -31.5f);
            }
#else
            sml_matvec(y, codes[c], sd[c], sq[c], NULL, NULL, &a, rows, n, 0, -31.5f);
#endif
        }
        const double dt = now_s() - t0;
        // dt can round to zero on a coarse timer; an "inf GB/s" reading is a
        // broken measurement, not a fast one
        if (dt <= 0.0) continue;
        const double gbs = (double) reps * (cbytes + sbytes) / dt / 1e9;
        if (gbs > best) best = gbs;
    }

    printf("achieved %.1f GB/s   (the reference achieves 20.6 on the same machine)\n\n", best);

    // Batched: does one weight read really serve k tokens near-free?
    //
    // This is the entire economics of speculative decoding on a memory-bound
    // machine. Scoring one token reads the whole model; scoring k candidates
    // reads the SAME bytes and does k times the arithmetic, which is free when
    // ~80% of the FLOPs are already idle waiting on the bus. If batch-4 costs
    // about what batch-1 costs, verification is nearly free and the only open
    // question is the acceptance rate.
    {
        sml_act ab[8];
        for (int b = 0; b < 8; b++) {
            ab[b].q     = malloc((size_t) n);
            ab[b].scale = malloc((size_t) ng * sizeof(float));
            ab[b].sums  = malloc((size_t) ng * sizeof(int32_t));
            for (int i = 0; i < n; i++)
                x[i] = (float)(((i + b * 7) % 200) - 100) / 100.0f;
            sml_quantize_act(&ab[b], x, n);
        }
        float * yb = calloc((size_t) rows * 8, sizeof(float));
        printf("batched: one weight read, k activation vectors\n");
        printf("  %5s %9s %13s %9s\n", "k", "secs", "scored/s", "vs k=1");
        // Each k=1 pass over one tensor is ~6 MB and finishes in ~10 ms, where
        // timer noise dominates and the k-scaling is unreadable. Run enough
        // repetitions that every measurement is a few hundred ms.
        const int breps = reps * 16;
        double t1 = 0.0;
        for (int k = 1; k <= 8; k *= 2) {
            double bestk = 1e30;
            for (int it = 0; it < 3; it++) {
                const double t0 = now_s();
                for (int r = 0; r < breps; r++) {
                    const int c = r % copies;
#ifdef _OPENMP
#pragma omp parallel
                    {
                        const int t  = omp_get_thread_num();
                        const int nt = omp_get_num_threads();
                        const int lo = (int)((long) rows * t / nt);
                        const int hi = (int)((long) rows * (t + 1) / nt);
                        sml_matvec_batch(yb, codes[c], sd[c], sq[c], NULL, NULL, ab, k,
                                         rows, lo, hi, n, 0, -31.5f);
                    }
#else
                    sml_matvec_batch(yb, codes[c], sd[c], sq[c], NULL, NULL, ab, k,
                                     rows, 0, rows, n, 0, -31.5f);
#endif
                }
                const double dt = now_s() - t0;
                if (dt > 0.0 && dt < bestk) bestk = dt;
            }
            if (k == 1) t1 = bestk;
            printf("  %5d %9.3f %13.1f %8.2fx\n", k, bestk,
                   (double) breps * k / bestk, (t1 * k) / bestk);
        }
        printf("  (scored/s = candidates evaluated per second; 'vs k=1' is the\n");
        printf("   speedup over evaluating them one at a time)\n");
        free(yb);
    }
    printf("implied whole-model throughput at that rate:\n");
    for (int d = 1; d <= 3; d++) {
        const double b = bytes_per_token(d);
        printf("  depth %d: %5.3f GB/token -> %6.1f tok/s   (%.2fx baseline)\n",
               d, b / 1e9, best * 1e9 / b, (best * 1e9 / b) / 10.71);
    }
    printf("  baseline: 1.920 GB/token ->   10.71 tok/s  (measured)\n");
    return 0;
}
