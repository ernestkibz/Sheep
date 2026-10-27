// Cross-check the C kernel against the Python reference on a real .ssf file.
//
// The packing, the two-level scales, the per-level shifts and the once-only bias
// are four places where a mistake still produces plausible-looking numbers. A
// bug here would surface later as "the model seems a bit dumb", which is the
// most expensive kind of bug to chase. So the contract gets tested directly:
// this writes y = W.x at every read depth, and tooling/sml_check.py recomputes
// the same thing from the format spec and compares.
//
//   gcc -O3 -march=native sml-check.c sml-kernel.c sml-model.c -o sml-check -lm
//   ./sml-check MODEL.ssf blk.0.ffn_gate.weight out.bin

#include "sml-model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.ssf TENSOR out.bin [max_rows]\n", argv[0]);
        return 2;
    }
    const int max_rows = argc > 4 ? atoi(argv[4]) : 256;

    sml_model m;
    if (!sml_model_open(&m, argv[1])) return 1;

    const sml_rec * r = sml_find(&m, argv[2]);
    if (!r) { fprintf(stderr, "no tensor %s\n", argv[2]); return 1; }
    if (r->kind != SML_KIND_LADDER) { fprintf(stderr, "%s is not laddered\n", argv[2]); return 1; }

    const int n     = (int) r->cols;
    const int rows  = (int) (r->rows < (uint32_t) max_rows ? r->rows : (uint32_t) max_rows);
    // Depth is a ladder concept. A flat tensor stores the code whole, so there
    // is exactly one way to read it and iterating depths would print the same
    // row three times and imply a dial that is not there.
    const int nlev  = (r->layout == SML_LAYOUT_FLAT)
                    ? 1 : (int) m.hdr->n_levels;
    const int ng    = n / SML_GROUP;

    printf("%s  rows=%u cols=%u group=%u levels=%d\n",
           r->name, r->rows, r->cols, r->group, nlev);

    // Deterministic input, matched exactly in the Python side.
    float * x  = malloc((size_t) n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = sinf(0.001f * (float) i) * (1.0f + 0.5f * (float)(i % 7));

    // The activation must be gathered through the tensor's column permutation:
    // the weights were quantized in permuted order, so (W P)(P^T x) = W x only
    // holds if we apply P^T here.
    const int32_t * perm = (const int32_t *) sml_ptr(&m, r->perm_off);
    float * xp = malloc((size_t) n * sizeof(float));
    sml_gather(xp, x, perm, n);

    sml_act a;
    a.q     = malloc((size_t) n);
    a.scale = malloc((size_t) ng * sizeof(float));
    a.sums  = malloc((size_t) ng * sizeof(int32_t));
    sml_quantize_act(&a, xp, n);

    const uint16_t * sd = (const uint16_t *) sml_ptr(&m, r->sd_off);
    const uint8_t  * sq = (const uint8_t  *) sml_ptr(&m, r->sq_off);

    FILE * f = fopen(argv[3], "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    int32_t meta[3] = { rows, nlev, n };
    fwrite(meta, sizeof(int32_t), 3, f);

    float * y = malloc((size_t) rows * sizeof(float));
    const uint16_t * md = r->mq_n ? (const uint16_t *) sml_ptr(&m, r->md_off) : NULL;
    const uint8_t  * mq = r->mq_n ? (const uint8_t  *) sml_ptr(&m, r->mq_off) : NULL;

    for (int depth = 1; depth <= nlev; depth++) {
        memset(y, 0, (size_t) rows * sizeof(float));
        if (r->layout == SML_LAYOUT_FLAT) {
            // This driver called the LADDER kernel unconditionally, so on a
            // flat tensor it produced a number rather than an error and the
            // gate reported FAIL on a model that was correct. Flat has to be
            // issued exactly the way the engine issues it -- see issue_rows()
            // in sml-run.c, which is the single dispatch point this mirrors.
            if (r->nbits == 3) {          // Q3: 2-bit + 4 * 1-bit
                sml_matvec(y, sml_ptr(&m, r->codes_off[0]), sd, sq, md, mq, &a, rows, n, 0, 0.0f);
                sml_matvec_bits1(y, sml_ptr(&m, r->codes_off[1]), sd, sq, &a, 1, 0, 0, rows, n, 2);
            } else if (r->nbits == 5) {   // Q5: nibble + 16 * 1-bit
                sml_matvec_rows_flat4(y, sml_ptr(&m, r->codes_off[0]), sd, sq, md, mq, &a, 0, rows, n);
                sml_matvec_bits1(y, sml_ptr(&m, r->codes_off[1]), sd, sq, &a, 1, 0, 0, rows, n, 4);
            } else if (r->nbits == 7) {   // Q7: Q6's code + 64 * 1-bit
                sml_matvec_rows_flat6(y, sml_ptr(&m, r->codes_off[0]), sml_ptr(&m, r->codes_off[1]), sd, sq, md, mq,
                                      &a, 0, rows, n);
                sml_matvec_bits1(y, sml_ptr(&m, r->codes_off[2]), sd, sq, &a, 1, 0, 0, rows, n, 6);
            } else if (r->nbits == 2) {   // Q2: one 2-bit plane
                sml_matvec(y, sml_ptr(&m, r->codes_off[0]), sd, sq, md, mq, &a, rows, n, 0, 0.0f);
            } else if (r->nbits == 8) {   // Q8: low + 16*high nibble planes
                sml_matvec_rows_flat8(y, sml_ptr(&m, r->codes_off[0]), sml_ptr(&m, r->codes_off[1]),
                                      sd, sq, md, mq, &a, 0, rows, n);
            } else if (r->nbits == 6) {
                sml_matvec(y, sml_ptr(&m, r->codes_off[0]), sd, sq, md, mq,
                           &a, rows, n, 4, 0.0f);
                sml_matvec_rows_flat4(y, sml_ptr(&m, r->codes_off[1]), sd, sq,
                                      NULL, NULL, &a, 0, rows, n);
            } else {
                sml_matvec_rows_flat4(y, sml_ptr(&m, r->codes_off[0]), sd, sq,
                                      md, mq, &a, 0, rows, n);
            }
        } else {
            for (int lv = 0; lv < depth; lv++) {
                // bits still unread below this level; bias on the last one only
                const int shift = (nlev - 1 - lv) * SML_LEVEL_BITS;
                const float cb = mq ? sml_level_mid(depth, nlev)
                                    : sml_level_bias(depth, nlev);
                const float bias = (lv == depth - 1) ? cb : 0.0f;
                sml_matvec(y, sml_ptr(&m, r->codes_off[lv]), sd, sq,
                           lv == depth - 1 ? md : NULL,
                           lv == depth - 1 ? mq : NULL,
                           &a, rows, n, shift, bias);
            }
        }
        fwrite(y, sizeof(float), (size_t) rows, f);

        double nrm = 0.0;
        for (int i = 0; i < rows; i++) nrm += (double) y[i] * y[i];
        printf("  depth %d: |y| = %.6f   y[0..3] = %+.5f %+.5f %+.5f %+.5f\n",
               depth, nrm, y[0], y[1], y[2], y[3]);
    }
    fclose(f);

    for (int d = 1; d <= nlev; d++) {
        printf("  depth %d moves %.3f GB/token over the whole model\n",
               d, (double) sml_bytes_at_depth(&m, d) / 1e9);
    }

    free(x); free(xp); free(y);
    free(a.q); free(a.scale); free(a.sums);
    sml_model_close(&m);
    return 0;
}
