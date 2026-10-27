// Batched serving, measured the way the reference measures it (llama-batched-bench):
// N independent sequences, each with its own KV cache, a P-token prompt each, then
// G tokens generated for all N together -- one read of the weights per step for
// every sequence (sml_forward_multi).
//
//   sml-batched MODEL.ssf [--threads T] [--npp P] [--ntg G] [--npl 1,2,4,8] [--check]
//
// --check decodes every sequence alone as well and reports, per step, the largest
// logit difference between a slot and its sequence decoded alone, and whether the
// greedy tokens agree.
#include "sml-run.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) f.QuadPart;
}
#else
#include <time.h>
static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
#endif

static int argmax(const float * l, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i;
    return b;
}

static void prompt_ids(int32_t * ids, int n, int seq) {
    for (int i = 0; i < n; i++) ids[i] = 1000 + (i * 37 + seq * 101) % 20000;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.ssf [--threads T] [--npp P] [--ntg G] [--npl 1,2,4,8] [--check]\n", argv[0]);
        return 2;
    }
    int threads = 6, npp = 128, ntg = 128, check = 0;
    const char * npl = "1,2,4,8";
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--npp") && i + 1 < argc) npp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ntg") && i + 1 < argc) ntg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--npl") && i + 1 < argc) npl = argv[++i];
        else if (!strcmp(argv[i], "--check")) check = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    sml_model m;
    if (!sml_model_open(&m, argv[1])) return 1;
    const int nv = (int) m.hdr->n_vocab;
    const int n_ctx = npp + ntg + 16;

    printf("|    PP |    TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |\n");
    printf("|-------|-------|------|--------|----------|----------|----------|----------|----------|----------|\n");
    for (const char * p = npl; *p; ) {
        const int ns = atoi(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
        if (ns < 1 || ns > SML_MAX_BATCH) continue;

        sml_ctx * cs = calloc((size_t) ns, sizeof(sml_ctx));
        sml_ctx ** cp = calloc((size_t) ns, sizeof(sml_ctx *));
        for (int s = 0; s < ns; s++) {
            if (!sml_ctx_init(&cs[s], &m, n_ctx, threads)) return 1;
            cp[s] = &cs[s];
        }
        int32_t * ids = malloc((size_t) npp * sizeof(int32_t));
        int32_t * tok = malloc((size_t) ns * sizeof(int32_t));
        int * pos = malloc((size_t) ns * sizeof(int));
        float * lg = malloc((size_t) ns * nv * sizeof(float));

        // prompts: one sequence at a time, each batched through its own context
        const double t0 = now_s();
        for (int s = 0; s < ns; s++) {
            prompt_ids(ids, npp, s);
            const float * l = sml_prefill(cp[s], ids, 0, npp, 0);
            tok[s] = argmax(l, nv);
            pos[s] = npp;
        }
        const double tpp = now_s() - t0;

        // one untimed step warms the path; the timed loop rewrites its positions
        sml_forward_multi(cp, tok, pos, ns, 0, lg);

        // --check: every slot against its sequence decoded alone, in a second set
        // of contexts, for the first steps
        sml_ctx * ref = NULL;
        if (check) {
            ref = calloc((size_t) ns, sizeof(sml_ctx));
            for (int s = 0; s < ns; s++) {
                if (!sml_ctx_init(&ref[s], &m, n_ctx, threads)) return 1;
                prompt_ids(ids, npp, s);
                sml_prefill(&ref[s], ids, 0, npp, 0);
            }
        }
        double worst = 0.0;
        int agree = 0, compared = 0;

        const double t1 = now_s();
        for (int g = 0; g < ntg; g++) {
            sml_forward_multi(cp, tok, pos, ns, 0, lg);
            if (check && g < 16) {
                for (int s = 0; s < ns; s++) {
                    const float * l = sml_forward(&ref[s], tok[s], pos[s], 0);
                    double d = 0.0, mx = 0.0;
                    for (int i = 0; i < nv; i++) {
                        const double e = fabs((double) l[i] - lg[(size_t) s * nv + i]);
                        if (e > d) d = e;
                        if (fabs(l[i]) > mx) mx = fabs(l[i]);
                    }
                    if (d / (mx + 1e-9) > worst) worst = d / (mx + 1e-9);
                    agree += argmax(l, nv) == argmax(lg + (size_t) s * nv, nv);
                    compared++;
                }
            }
            for (int s = 0; s < ns; s++) {
                tok[s] = argmax(lg + (size_t) s * nv, nv);
                pos[s]++;
            }
        }
        const double ttg = now_s() - t1;
        printf("| %5d | %5d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n",
               npp, ntg, ns, ns * (npp + ntg), tpp, ns * npp / tpp, ttg, ns * ntg / ttg,
               tpp + ttg, ns * (npp + ntg) / (tpp + ttg));
        if (check)
            printf("  check: %d slot-steps, greedy tokens agree %d/%d, worst max|dlogit|/max|logit| %.3g\n",
                   compared, agree, compared, worst);
        fflush(stdout);

        if (ref) { for (int s = 0; s < ns; s++) sml_ctx_free(&ref[s]); free(ref); }
        for (int s = 0; s < ns; s++) sml_ctx_free(&cs[s]);
        free(cs); free(cp); free(ids); free(tok); free(pos); free(lg);
    }
    sml_model_close(&m);
    return 0;
}
