// Race the SML flat4 kernel: AVX2 on the CPU against NVPTX on the 3050.
//
// Same tensor, same activation, same layout. The GPU result is graded against
// the CPU result rather than against itself, because a kernel that reads the
// packing wrong still produces confident numbers -- that is how this project's
// worst bugs have always presented.
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sml-gpu-drv.h"
#include "sml-model.h"
#include "sml-kernel.h"

static double now_s(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double) t.QuadPart / (double) f.QuadPart;
}

static char * slurp(const char * path, size_t * len) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * b = malloc((size_t) n + 1);
    if (fread(b, 1, (size_t) n, f) != (size_t) n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f);
    if (len) *len = (size_t) n;
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 3) { printf("usage: bench MODEL.ssf TENSOR [reps]\n"); return 2; }
    const int reps = argc > 3 ? atoi(argv[3]) : 50;

    sml_model m;
    if (!sml_model_open(&m, argv[1])) { printf("cannot open model\n"); return 1; }
    const sml_rec * r = sml_find(&m, argv[2]);
    if (!r) { printf("no tensor %s\n", argv[2]); return 1; }
    if (r->layout != SML_LAYOUT_FLAT || r->nbits != 4) {
        printf("%s is not flat-4 (layout %u, nbits %u)\n", argv[2], r->layout, r->nbits);
        return 1;
    }

    const int n = (int) r->cols, rows = (int) r->rows;
    const int ng = n / SML_GROUP, nscale = SML_NSCALE(n);
    printf("  tensor      %s  %d x %d, group %d\n", r->name, rows, n, r->group);

    // deterministic activation, quantized exactly as the engine would
    float * x = malloc((size_t) n * sizeof(float));
    for (int i = 0; i < n; i++)
        x[i] = sinf(0.001f * (float) i) * (1.0f + 0.5f * (float)(i % 7));
    const int32_t * perm = (const int32_t *) sml_ptr(&m, r->perm_off);
    float * xp = malloc((size_t) n * sizeof(float));
    sml_gather(xp, x, perm, n);
    sml_act a;
    a.q = malloc((size_t) n);
    a.scale = malloc((size_t) ng * sizeof(float));
    a.sums = malloc((size_t) ng * sizeof(int32_t));
    sml_quantize_act(&a, xp, n);

    const uint8_t  * codes = (const uint8_t  *) sml_ptr(&m, r->codes_off[0]);
    const uint16_t * sd    = (const uint16_t *) sml_ptr(&m, r->sd_off);
    const uint8_t  * sq    = (const uint8_t  *) sml_ptr(&m, r->sq_off);
    const uint16_t * md    = r->mq_n ? (const uint16_t *) sml_ptr(&m, r->md_off) : NULL;
    const uint8_t  * mq    = r->mq_n ? (const uint8_t  *) sml_ptr(&m, r->mq_off) : NULL;

    // bytes the kernel must stream for ONE matvec
    const double bytes = (double) rows * (n / 2)              // codes
                       + (double) rows * ng * (mq ? 2 : 1)    // sq (+mq)
                       + (double) rows * nscale * 2 * (md ? 2 : 1);
    printf("  per matvec  %.2f MB\n", bytes / 1e6);

    // ---------------------------------------------------------------- CPU
    float * ycpu = calloc((size_t) rows, sizeof(float));
    sml_matvec_rows_flat4(ycpu, codes, sd, sq, md, mq, &a, 0, rows, n);
    double tc = now_s();
    for (int i = 0; i < reps; i++) {
        memset(ycpu, 0, (size_t) rows * sizeof(float));
        sml_matvec_rows_flat4(ycpu, codes, sd, sq, md, mq, &a, 0, rows, n);
    }
    tc = now_s() - tc;
    printf("  CPU AVX2    %.3f ms/matvec   %.1f GB/s\n",
           tc * 1e3 / reps, bytes * reps / tc / 1e9);

    // ---------------------------------------------------------------- GPU
    if (!cu_open()) return 1;
    if (!cu_check(CU.Init(0), "cuInit")) return 1;
    CUdevice dev; CUcontext ctx;
    cu_check(CU.DeviceGet(&dev, 0), "cuDeviceGet");
    cu_check(CU.CtxCreate(&ctx, 0, dev), "cuCtxCreate");

    size_t plen = 0;
    char * ptx = slurp(argc > 4 ? argv[4] : "sml-gpu.ptx", &plen);
    if (!ptx) { printf("  cannot read sml_cuda.ptx\n"); return 1; }
    CUmodule mod; CUfunction fn;
    if (!cu_check(CU.ModuleLoadData(&mod, ptx), "cuModuleLoadData")) return 1;
    if (!cu_check(CU.ModuleGetFunction(&fn, mod, "sml_matvec_flat4"), "getFunction")) return 1;

    CUdeviceptr dcodes, dsd, dsq, dmd, dmq, dxq, dxs, dxsum, dy;
    const size_t ncodes = (size_t) rows * (n / 2);
    cu_check(CU.MemAlloc(&dcodes, ncodes), "alloc codes");
    cu_check(CU.MemAlloc(&dsd, (size_t) rows * nscale * 2), "alloc sd");
    cu_check(CU.MemAlloc(&dsq, (size_t) rows * ng), "alloc sq");
    cu_check(CU.MemAlloc(&dmd, (size_t) rows * nscale * 2), "alloc md");
    cu_check(CU.MemAlloc(&dmq, (size_t) rows * ng), "alloc mq");
    cu_check(CU.MemAlloc(&dxq, (size_t) n), "alloc xq");
    cu_check(CU.MemAlloc(&dxs, (size_t) ng * 4), "alloc xscale");
    cu_check(CU.MemAlloc(&dxsum, (size_t) ng * 4), "alloc xsums");
    cu_check(CU.MemAlloc(&dy, (size_t) rows * 4), "alloc y");

    cu_check(CU.MemcpyHtoD(dcodes, codes, ncodes), "HtoD codes");
    cu_check(CU.MemcpyHtoD(dsd, sd, (size_t) rows * nscale * 2), "HtoD sd");
    cu_check(CU.MemcpyHtoD(dsq, sq, (size_t) rows * ng), "HtoD sq");
    if (md) cu_check(CU.MemcpyHtoD(dmd, md, (size_t) rows * nscale * 2), "HtoD md");
    if (mq) cu_check(CU.MemcpyHtoD(dmq, mq, (size_t) rows * ng), "HtoD mq");
    cu_check(CU.MemcpyHtoD(dxq, a.q, (size_t) n), "HtoD xq");
    cu_check(CU.MemcpyHtoD(dxs, a.scale, (size_t) ng * 4), "HtoD xscale");
    cu_check(CU.MemcpyHtoD(dxsum, a.sums, (size_t) ng * 4), "HtoD xsums");

    int rr = rows, nn = n, hm = mq ? 1 : 0;
    void * args[] = { &dy, &dcodes, &dsd, &dsq, &dmd, &dmq,
                      &dxq, &dxs, &dxsum, &rr, &nn, &hm };
    const unsigned block = 256;
    const unsigned grid = (unsigned)((rows + (256/32) - 1) / (256/32));

    if (!cu_check(CU.LaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL),
                  "launch")) return 1;
    if (!cu_check(CU.CtxSynchronize(), "sync")) return 1;

    double tg = now_s();
    for (int i = 0; i < reps; i++)
        CU.LaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL);
    cu_check(CU.CtxSynchronize(), "sync");
    tg = now_s() - tg;

    float * ygpu = malloc((size_t) rows * sizeof(float));
    cu_check(CU.MemcpyDtoH(ygpu, dy, (size_t) rows * 4), "DtoH y");
    printf("  GPU sm_86   %.3f ms/matvec   %.1f GB/s\n",
           tg * 1e3 / reps, bytes * reps / tg / 1e9);

    // grade the answer
    double se = 0.0, sr = 0.0;
    for (int i = 0; i < rows; i++) {
        const double d = (double) ygpu[i] - (double) ycpu[i];
        se += d * d;
        sr += (double) ycpu[i] * (double) ycpu[i];
    }
    const double rel = sqrt(se / (rows ? rows : 1)) / (sqrt(sr / (rows ? rows : 1)) + 1e-30);
    printf("  agreement   rel-RMS %.3e vs the CPU kernel  %s\n",
           rel, rel < 1e-4 ? "PASS" : "MISMATCH");
    printf("  speedup     %.2fx\n", tc / tg);

    CU.CtxDestroy(ctx);
    return rel < 1e-4 ? 0 : 1;
}
