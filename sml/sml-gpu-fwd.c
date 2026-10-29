// GPU forward with SHARED activation quantization, A/B against the fused path.
//
// q/k/v and gate/up read the same input through the same permutation (checked
// per layer at upload, not assumed), so the activation is quantized once per
// shared input and the matvec blocks stage finished words. SML_SHARED=0 runs the
// fused path from the same binary, so the comparison is one variable.
//
// Env: SML_SHARED and SML_DP4A (both default 1: the fastest measured path; 0 falls back), SML_SYNC (0 async, 1 per layer, 2 per launch), SML_GRID blocks per matvec launch (default 48)
// and SML_BLOCK threads (default 128), SML_FUSE2 (default 1: seven launches per layer, graph replay),
// SPEED_ONLY, PROFILE, SML_CTXFLAGS (default 1, spin).
//
//   sml-gpu-fwd MODEL.ssf ids,comma,separated NGEN
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sml-gpu-drv.h"
#include "sml-model.h"
#include "sml-kernel.h"
#include "sml-run.h"
#include "sml-gpu.h"
#include <setjmp.h>

// sml-server runs this code in-process: inside an sml_gpu_* call a failure jumps
// back to that call, which returns NULL, and the server goes on on the CPU.
static jmp_buf g_jb;
static int g_jb_on = 0;
#define GPU_DIE() do { if (g_jb_on) longjmp(g_jb, 1); exit(1); } while (0)

static double now_s(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
    return (double) t.QuadPart / (double) f.QuadPart;
}

enum { KB_QMV4, KB_QMV6, KB_ADDNORM, KB_ROPEKV, KB_ATTN, KB_SILU, KB_GELU, KB_ARGMAX,
       KB_HEAD4, KB_HEAD6, KB_QUANTW, KB_SMV4, KB_SMV6,
       KB_NORMQ, KB_MV3, KB_ROPEATTN, KB_QDMV, KB_FFN, KB_HEAD3, KB_EMBED, KB_N };
static const char * KB_NAME[KB_N] = { "qmv4", "qmv6", "addnorm", "ropekv", "attn", "silu",
    "gelu", "argmax", "head mv4", "head mv6", "quantw", "smv4", "smv6",
    "normq", "mv3 qkv", "ropeattn", "qdmv", "ffn", "head mv3", "embed" };
enum { KF_QMV4, KF_QMV6, KF_ADDNORM, KF_ROPEKV, KF_ATTN, KF_SILU, KF_GELU, KF_ARGMAX,
       KF_QUANTW, KF_SMV4, KF_SMV6, KF_DMV4, KF_DMV6,
       KF_NORMQ, KF_MV3, KF_ROPEATTN, KF_QDMV, KF_FFN,
       KF_NORMQ2, KF_ATTN2, KF_QDMV2, KF_MV3S, KF_FFNS, KF_EMBED, KF_ARGMAX2, KF_QUANTW2,
       KF_EMBEDB, KF_NORMQ2B, KF_QUANTW2B, KF_MVB, KF_FFNB, KF_ROPEKVB, KF_ATTNB,
       KF_TMM, KF_ACTB, KF_TCM, KF_FATTN, KF_TCM2, KF_QUANTSBB, KF_HMM, KF_PERMH, KF_REQUANT8, KF_TCM3, KF_FATC, KF_SPLITK, KF_FATC128, KF_TCM4, KF_TCM5, KF_TCM7, KF_TCM8, KF_TCM9, KF_QKNORM, KF_SCALEB, KF_ADDROW, KF_CAPB, KF_CLAMPB, KF_F32MV, KF_AXPY, KF_ROPEQD,
       KF_MOE_ROUTE, KF_MOE_QUANT, KF_MOE_GU, KF_MOE_DN, KF_MOE_COMB,
       KF_F32MVB, KF_GATHERB, KF_SCATTERB, KF_ZEROB, KF_N };
static CUfunction KF[KF_N];
// Decode defaults are the tuned GPU profile (tooling/sml_tune.py DEFAULT["gpu"]):
// fuse2 + graph replay, 48 blocks of 128 threads per matvec, a spinning host
// context. Until 2026-09-25 fuse2 was opt-in, and racing the bare defaults
// reported GPU decode LOST (0.41-0.91x) where the tuned path wins 1.52x.
static int g_prof = 0, g_shared = 1, g_dp4a = 1, g_sync = 0, g_fuse2 = 1;
// SML_FAST (default 1): the fused path uses sml-gpu-fast.cu -- warp-per-group
// quantize, float norm, block-per-head attention. 0 keeps the old kernels.
static int g_fast = 1;
// SML_GRAPH (default 1): decode replays one captured graph per token. 0 = per-token.
static int g_graph = 1;
// SML_SPLIT (default 1): the o/down projections quantize once (quantw2) and
// then run mv3s, instead of every qdmv block re-quantizing the whole input.
static int g_split = 1;
static CUstream g_stream = NULL;   // NULL = legacy stream; set only while graphing
static double g_acc[KB_N];
static unsigned g_gcap = 48, g_bsize = 128;   // matvec launches: block cap, threads per block
static size_t g_vram = 0;
static CUdevice g_dev = 0;    // the card CUDA gave us (for NVML's view of it)
static long long g_launches = 0;

static CUdeviceptr dalloc(size_t bytes) {
    CUdeviceptr p = 0;
    if (!bytes) bytes = 4;
    if (!cu_check(CU.MemAlloc(&p, bytes), "cuMemAlloc")) GPU_DIE();
    g_vram += bytes;
    return p;
}
static CUdeviceptr dup(const void * host, size_t bytes) {
    CUdeviceptr p = dalloc(bytes);
    if (host && !cu_check(CU.MemcpyHtoD(p, host, bytes), "HtoD")) GPU_DIE();
    return p;
}
static unsigned cdiv(long long a, long long b) { return (unsigned)((a + b - 1) / b); }

static void launch(int kb, int kf, unsigned grid, void ** args) {
    double tp = 0.0;
    if (!grid) grid = 1;
    if (g_prof) { CU.CtxSynchronize(); tp = now_s(); }
    const int mv = kf == KF_QMV4 || kf == KF_QMV6 || kf == KF_SMV4 || kf == KF_SMV6 || kf == KF_DMV4 ||
                   kf == KF_DMV6 || kf == KF_MV3 || kf == KF_QDMV || kf == KF_FFN ||
                   kf == KF_QDMV2 || kf == KF_MV3S || kf == KF_FFNS;
    if (CU.LaunchKernel(KF[kf], grid, 1, 1, mv ? g_bsize : 256, 1, 1, 0, g_stream, args, NULL) != CU_OK) {
        puts("  launch failed"); GPU_DIE();
    }
    g_launches++;
    if (g_sync == 2 && !g_prof) CU.CtxSynchronize();
    if (g_prof) { CU.CtxSynchronize(); g_acc[kb] += now_s() - tp; }
}

typedef struct {
    CUdeviceptr c2, c4, sd, sq, md, mq, perm;
    const int32_t * hperm;
    int rows, n, six, have_min;
} gt;

static gt up_tensor(const sml_model * m, const sml_rec * r) {
    gt t;
    memset(&t, 0, sizeof t);
    if (!r) return t;
    if (r->layout != SML_LAYOUT_FLAT) { puts("  tensor is not flat; the GPU path needs v8 flat"); GPU_DIE(); }
    t.rows = (int) r->rows; t.n = (int) r->cols;
    t.six = r->nbits == 6;
    t.have_min = r->mq_n ? 1 : 0;
    const size_t rows = (size_t) t.rows, n = (size_t) t.n;
    const size_t ng = n / SML_GROUP, nscale = (size_t) SML_NSCALE(t.n);
    if (t.six) {
        t.c2 = dup(sml_ptr(m, r->codes_off[0]), rows * (n / 128) * 32);
        t.c4 = dup(sml_ptr(m, r->codes_off[1]), rows * n / 2);
    } else {
        t.c4 = dup(sml_ptr(m, r->codes_off[0]), rows * n / 2);
    }
    t.sd = dup(sml_ptr(m, r->sd_off), rows * nscale * 2);
    // A v9 file packs the group fields at 6 bits. The card holds its own copy
    // anyway, so they go up as bytes and every kernel reads them unchanged: the
    // scale as it is, the min re-biased from 31 to 127 (mq + 96), so the kernels'
    // (mq - 127) is the file's (mq - 31) exactly -- or from 63 (mq + 64) when the
    // file's mins are negative-only.
    if (sml_kernel_scale6_get()) {
        const uint8_t * s6 = (const uint8_t *) sml_ptr(m, r->sq_off);
        const uint8_t * m6 = t.have_min ? (const uint8_t *) sml_ptr(m, r->mq_off) : NULL;
        const size_t g6 = (size_t) SML_SCALE6_BYTES(ng);
        uint8_t * s8 = malloc(rows * ng), * m8 = t.have_min ? malloc(rows * ng) : NULL;
        for (size_t row = 0; row < rows; row++)
            for (size_t g = 0; g < ng; g++) {
                s8[row * ng + g] = (uint8_t) sml_scale6_at(s6 + row * g6, (int) g);
                if (m8) m8[row * ng + g] = (uint8_t) (sml_scale6_at(m6 + row * g6, (int) g)
                                                      + (sml_kernel_minneg_get() ? 64 : 96));
            }
        t.sq = dup(s8, rows * ng);
        if (m8) {
            t.md = dup(sml_ptr(m, r->md_off), rows * nscale * 2);
            t.mq = dup(m8, rows * ng);
        } else {
            t.md = t.sd; t.mq = t.sq;
        }
        free(s8); free(m8);
        t.hperm = (const int32_t *) sml_ptr(m, r->perm_off);
        t.perm = dup(t.hperm, n * 4);
        return t;
    }
    t.sq = dup(sml_ptr(m, r->sq_off), rows * ng);
    if (t.have_min) {
        t.md = dup(sml_ptr(m, r->md_off), rows * nscale * 2);
        t.mq = dup(sml_ptr(m, r->mq_off), rows * ng);
    } else {
        t.md = t.sd; t.mq = t.sq;     // never read: have_min is 0
    }
    t.hperm = (const int32_t *) sml_ptr(m, r->perm_off);
    t.perm = dup(t.hperm, n * 4);
    return t;
}
static CUdeviceptr up_f(const float * p, size_t count) {
    return p ? dup(p, count * sizeof(float)) : 0;
}
static int same_perm(const gt * a, const gt * b) {
    return a->n && a->n == b->n && memcmp(a->hperm, b->hperm, (size_t) a->n * 4) == 0;
}

// The host's copy of sml_mt in sml-gpu-fast.cu -- one expert tensor's pointers.
typedef struct { uint64_t c2, c4, sd, sq, md, mq; int rows, six, hm, pad; } host_mt;
_Static_assert(sizeof(host_mt) == 64, "host_mt must match sml_mt on the card");

// A layer's experts on the card (Granite MoE, Mixtral, Qwen MoE, DeepSeek...).
typedef struct gmoe {
    int ne;
    gt * eg, * eu, * ed;           // every expert's gate, up and down
    CUdeviceptr router, shgate;    // the router (f32, ne x n_embd); the shared expert's gate row, 0 if none
    gt shg, shu, shd;              // the shared expert, empty if none
    // on-card routing (moe_ffn_dev): each expert's pointers as sml_mt tables for
    // gate, up and down, their column orders, the router's correction bias
    int dev, fe;                   // 1 when the tables are up; the experts' width
    CUdeviceptr tg, tu, td, pg, pd, ebias;
} gmoe;

typedef struct {
    gt q, k, v, o, gate, up, down;
    gmoe * moe;                        // experts in place of a dense FFN, NULL if dense
    CUdeviceptr qb, kb, vb, ob, gb, ub, db, an, anb, fn, fnb;
    CUdeviceptr qn, qnb, kn, knb;      // Q/K norm weights (and LayerNorm biases), 0 if none
    CUdeviceptr pa, pf;                // post-norms on the attention and FFN outputs, 0 if none
    int qn_n, kn_n;                    // their length: one head, or the whole projection
    int kv_same, gu_same;
} gl;

static sml_model M;
static sml_ctx C;
static gl * L;
static gt HEAD, EMB;
static CUdeviceptr ON, ONB, X, XB, Q, Kv, V, ATT, HB, HB2, LOG, KC, VC, RC, RS, IDX, QW, QS, QM;
static CUdeviceptr DPOS;   // the position, read by the fast attention from device memory
static CUdeviceptr DTOK, DLOG;   // next token id, and the log a graph replay writes
static int n_embd, n_head, n_kv, hd, kvd, n_ff, n_vocab, NL, n_ctx, half;
// The widest activation any buffer holds: the FFN, the residual, or the attention
// output (n_head x head_dim, which Qwen3 and Gemma make wider than n_embd).
static int g_wide;
static int layer_norm, interleaved, par, swa;
static int g_qk_late;               // HunYuan: Q/K norm AFTER the rotary
// A model larger than the card: its first NL layers run here and the rest on the
// CPU (CS, the CPU half: sml_forward_from / sml_prefill_from), the stream handed
// across once per token. g_cpu_split = 1 then; NL is the GPU's share, NL_ALL the model.
static int g_cpu_split, NL_ALL;
// A middle stage between the GPU's layers and the CPU's (sml_gpu_set_middle):
// another device runs layers NL..NL+g_mid_n-1 on the hidden rows, in place.
static int g_mid_n = 0;
static int (*g_mid_fn)(float * rows, int pos0, int n) = 0;
static sml_ctx CS;
static const float * g_split_lg;    // the last split forward's logits (host)
static float * g_rows_h;            // a split prompt's residual rows, on the host
// A split model reads its prompts on the card for EVERY layer (2026-10-28). The CPU's
// layers used to read their share of a prompt on the CPU: 57 tok/s on Qwen2.5-7B against
// Ollama's 149, which carries those layers' weights across the bus for a prompt. Here a
// CPU layer is copied into one staging layer on the card (SL), with its cache rows so
// far, runs the same batched kernels as a card layer, and its new cache rows go back to
// the CPU's cache, where the CPU reads them when it writes the answer. Writing does not
// change: each new token still runs the CPU's layers on the CPU. SML_STREAM=0: off.
static int g_pstream = 1, g_stream_ok = 0, g_in_stream = 0;
static gl SL;                       // the staging layer's tensors, on the card
static gl * LS;                     // each CPU layer's own small vectors; tensors are SL's
static CUdeviceptr KS, VS;          // one layer's K and V cache rows (f32), staged
static float * g_kv_h;              // host rows for the cache's fp16 <-> f32
static double g_st_copy, g_st_run, g_st_t, g_st_card;   // SML_STREAM_STATS: where a streamed prompt's time goes
static uint8_t ** g_s8h, ** g_m8h;  // a v9 file's 6-bit group fields as bytes, per CPU tensor
static void host_embed(int tok, int pos, float * out);
// Per-layer attention (Gemma 2/3, Cohere2, SmolLM3, GPT-Neo, EXAONE 4), read per
// layer exactly as sml-run.c does: the layer's window, whether rotary applies at
// all (a half of 0 rotates nothing in every kernel), and -- for the local layers
// of Gemma 3 -- the local rotary table. Kernel arguments, so a captured graph
// holds each layer's own values.
static CUdeviceptr RCL, RSL;
static int l_win(int l)  { return C.layer_win ? C.layer_win[l] : swa; }
// No rotary at all where learned positions (or ALiBi) carry position: GPT-2 on
// the GPU rotated its Q and K anyway and graded 0.876 against the CPU.
static int l_half(int l) { return (C.no_rope || (C.layer_rope && !C.layer_rope[l])) ? 0 : half; }
static CUdeviceptr l_rc(int l) { return (RCL && C.layer_win && C.layer_win[l] > 0) ? RCL : RC; }
static CUdeviceptr l_rs(int l) { return (RCL && C.layer_win && C.layer_win[l] > 0) ? RSL : RS; }
static float inv, eps;
static float * xhost, * rowb;

static unsigned grid_for(int rows) {
    unsigned grid = cdiv(rows, g_bsize / 32);
    return grid > g_gcap ? g_gcap : grid;
}

// ---- fused path (quantize inside every block)
static void mvq(const gt * t, CUdeviceptr y, CUdeviceptr x, CUdeviceptr bias) {
    int rows = t->rows, n = t->n, hm = t->have_min, hb = bias ? 1 : 0;
    CUdeviceptr bb = bias ? bias : x;
    const int head = rows == n_vocab;
    if (t->six) {
        void * a[] = { &y, (void *) &t->c2, (void *) &t->c4, (void *) &t->sd, (void *) &t->sq,
                       (void *) &t->md, (void *) &t->mq, &x, (void *) &t->perm, &bb,
                       &rows, &n, &hm, &hb };
        launch(head ? KB_HEAD6 : KB_QMV6, KF_QMV6, grid_for(rows), a);
    } else {
        void * a[] = { &y, (void *) &t->c4, (void *) &t->sd, (void *) &t->sq,
                       (void *) &t->md, (void *) &t->mq, &x, (void *) &t->perm, &bb,
                       &rows, &n, &hm, &hb };
        launch(head ? KB_HEAD4 : KB_QMV4, KF_QMV4, grid_for(rows), a);
    }
}

// ---- shared path (quantize once, blocks stage the words)
static void quantw(const gt * t, CUdeviceptr x) {
    int n = t->n;
    void * a[] = { &x, (void *) &t->perm, &QW, &QS, &QM, &n };
    launch(KB_QUANTW, KF_QUANTW, cdiv(n / SML_GROUP, 256), a);
}
static void mvs(const gt * t, CUdeviceptr y, CUdeviceptr bias) {
    int rows = t->rows, n = t->n, hm = t->have_min, hb = bias ? 1 : 0;
    CUdeviceptr bb = bias ? bias : QS;
    const int head = rows == n_vocab;
    if (t->six) {
        void * a[] = { &y, (void *) &t->c2, (void *) &t->c4, (void *) &t->sd, (void *) &t->sq,
                       (void *) &t->md, (void *) &t->mq, &QW, &QS, &QM, &bb,
                       &rows, &n, &hm, &hb };
        launch(head ? KB_HEAD6 : KB_SMV6, g_dp4a ? KF_DMV6 : KF_SMV6, grid_for(rows), a);
    } else {
        void * a[] = { &y, (void *) &t->c4, (void *) &t->sd, (void *) &t->sq,
                       (void *) &t->md, (void *) &t->mq, &QW, &QS, &QM, &bb,
                       &rows, &n, &hm, &hb };
        launch(head ? KB_HEAD4 : KB_SMV4, g_dp4a ? KF_DMV4 : KF_SMV4, grid_for(rows), a);
    }
}

static void addnorm(CUdeviceptr y, CUdeviceptr w, CUdeviceptr b, int mode) {
    int nn = n_embd, hb = b ? 1 : 0, ln = w ? layer_norm : 2, md = mode;
    if (!w) w = X;          // no norm (OLMo 2): never read
    float e = eps;
    CUdeviceptr bb = b ? b : w;
    void * a[] = { &X, &y, &XB, &w, &bb, &nn, &e, &ln, &hb, &md };
    launch(KB_ADDNORM, KF_ADDNORM, 1, a);
}

// ---- fuse2 path: seven launches per layer -- norm+quantize, q/k/v, RoPE+KV+attention,
// o, norm+quantize, gate+up+activation, down. Needs q/k/v and gate/up to share a
// permutation, which the upload checks per layer.
static const gt G_NONE;
static CUdeviceptr D_NONE;

static void f2_normq(CUdeviceptr y, CUdeviceptr w, CUdeviceptr b, const gt * t, int mode) {
    int nn = n_embd, hb = b ? 1 : 0, ln = w ? layer_norm : 2, md = mode;
    if (!w) w = X;          // no norm (OLMo 2): never read
    float e = eps;
    CUdeviceptr bb = b ? b : w;
    void * a[] = { &X, &y, &XB, &w, &bb, (void *) &t->perm, &QW, &QS, &QM, &nn, &e, &ln, &hb, &md };
    launch(KB_NORMQ, g_fast ? KF_NORMQ2 : KF_NORMQ, 1, a);
}
static void f2_mv3(int kb, const gt * t0, CUdeviceptr y0, CUdeviceptr b0,
                   const gt * t1, CUdeviceptr y1, CUdeviceptr b1,
                   const gt * t2, CUdeviceptr y2, CUdeviceptr b2) {
    if (!t1) { t1 = &G_NONE; y1 = D_NONE; b1 = 0; }
    if (!t2) { t2 = &G_NONE; y2 = D_NONE; b2 = 0; }
    int n = t0->n;
    int r0 = t0->rows, s0 = t0->six, m0 = t0->have_min, h0 = b0 ? 1 : 0;
    int r1 = t1->rows, s1 = t1->six, m1 = t1->have_min, h1 = b1 ? 1 : 0;
    int r2 = t2->rows, s2 = t2->six, m2 = t2->have_min, h2 = b2 ? 1 : 0;
    CUdeviceptr bb0 = b0 ? b0 : QS, bb1 = b1 ? b1 : QS, bb2 = b2 ? b2 : QS;
    void * a[] = {
        &y0, (void *) &t0->c2, (void *) &t0->c4, (void *) &t0->sd, (void *) &t0->sq,
        (void *) &t0->md, (void *) &t0->mq, &bb0,
        &y1, (void *) &t1->c2, (void *) &t1->c4, (void *) &t1->sd, (void *) &t1->sq,
        (void *) &t1->md, (void *) &t1->mq, &bb1,
        &y2, (void *) &t2->c2, (void *) &t2->c4, (void *) &t2->sd, (void *) &t2->sq,
        (void *) &t2->md, (void *) &t2->mq, &bb2,
        &QW, &QS, &QM,
        &n, &r0, &s0, &m0, &h0, &r1, &s1, &m1, &h1, &r2, &s2, &m2, &h2 };
    const int rmax = r0 > r1 ? (r0 > r2 ? r0 : r2) : (r1 > r2 ? r1 : r2);
    launch(kb, g_fast ? KF_MV3S : KF_MV3, grid_for(rmax), a);
}
static void f2_ropeattn(int l, int pos) {
    if (g_fast) {
        CUdeviceptr kl = KC + (CUdeviceptr)((size_t) l * n_ctx * kvd * 4);
        CUdeviceptr vl = VC + (CUdeviceptr)((size_t) l * n_ctx * kvd * 4);
        int nh = n_head, nk = n_kv, hh = hd, hf = g_qk_late ? 0 : l_half(l), il = interleaved, nc = n_ctx, sw = l_win(l);
        float iv = inv;
        CUdeviceptr rc = l_rc(l), rs = l_rs(l);
        void * a[] = { &XB, &Q, &Kv, &V, &kl, &vl, &ATT, &rc, &rs, &DPOS,
                       &nh, &nk, &hh, &hf, &il, &nc, &sw, &iv };
        launch(KB_ROPEATTN, KF_ATTN2, (unsigned) n_head, a);
        return;
    }
    CUdeviceptr kat = KC + (CUdeviceptr)(((size_t) l * n_ctx + pos) * kvd * 4);
    CUdeviceptr vat = VC + (CUdeviceptr)(((size_t) l * n_ctx + pos) * kvd * 4);
    CUdeviceptr kl  = KC + (CUdeviceptr)((size_t) l * n_ctx * kvd * 4);
    CUdeviceptr vl  = VC + (CUdeviceptr)((size_t) l * n_ctx * kvd * 4);
    CUdeviceptr cs  = l_rc(l) + (CUdeviceptr)((size_t) pos * half * 4);
    CUdeviceptr sn  = l_rs(l) + (CUdeviceptr)((size_t) pos * half * 4);
    int nh = n_head, nk = n_kv, hh = hd, hf = l_half(l), il = interleaved, nc = n_ctx, p = pos;
    const int w = l_win(l);
    int t0 = (w && pos >= w) ? pos - w + 1 : 0;
    float iv = inv;
    void * a[] = { &XB, &Q, &Kv, &V, &kat, &vat, &kl, &vl, &ATT, &cs, &sn,
                   &nh, &nk, &hh, &hf, &il, &nc, &t0, &p, &iv };
    launch(KB_ROPEATTN, KF_ROPEATTN, 1, a);
}
static void f2_qdmv(const gt * t, CUdeviceptr y, CUdeviceptr x, CUdeviceptr bias) {
    if (g_fast && g_split) {
        int nn = t->n;
        CUdeviceptr xx = x;
        void * qa[] = { &xx, (void *) &t->perm, &QW, &QS, &QM, &nn };
        launch(KB_QDMV, KF_QUANTW2, cdiv(t->n / SML_GROUP, 8), qa);
        f2_mv3(KB_QDMV, t, y, bias, NULL, 0, 0, NULL, 0, 0);
        return;
    }
    int rows = t->rows, n = t->n, six = t->six, hm = t->have_min, hb = bias ? 1 : 0;
    CUdeviceptr bb = bias ? bias : x;
    void * a[] = { &y, (void *) &t->c2, (void *) &t->c4, (void *) &t->sd, (void *) &t->sq,
                   (void *) &t->md, (void *) &t->mq, &bb, &x, (void *) &t->perm,
                   &n, &rows, &six, &hm, &hb };
    launch(KB_QDMV, g_fast ? KF_QDMV2 : KF_QDMV, grid_for(rows), a);
}
static void f2_ffn(const gl * y) {
    const int hg = y->gate.n ? 1 : 0;
    const gt * g = hg ? &y->gate : &y->up;
    CUdeviceptr gbb = hg && y->gb ? y->gb : QS, ubb = y->ub ? y->ub : QS;
    int n = y->up.n, rows = y->up.rows, has_gate = hg;
    int gs = g->six, gm = g->have_min, gh = hg && y->gb ? 1 : 0;
    int us = y->up.six, um = y->up.have_min, uh = y->ub ? 1 : 0;
    void * a[] = { &HB,
        (void *) &g->c2, (void *) &g->c4, (void *) &g->sd, (void *) &g->sq,
        (void *) &g->md, (void *) &g->mq, &gbb,
        (void *) &y->up.c2, (void *) &y->up.c4, (void *) &y->up.sd, (void *) &y->up.sq,
        (void *) &y->up.md, (void *) &y->up.mq, &ubb,
        &QW, &QS, &QM, &n, &rows, &has_gate, &gs, &gm, &gh, &us, &um, &uh };
    launch(KB_FFN, g_fast ? KF_FFNS : KF_FFN, grid_for(rows), a);
}
// Q/K norm before rotary (Qwen3, Gemma 3, OLMo 2; with biases, Persimmon's
// LayerNorm): one block per head -- or one for the whole projection when the
// weights are that long -- per token, as sml-run.c's qk_norm does on the CPU.
static void launch2(int kb, int kf, unsigned gx, unsigned gy, unsigned block, void ** args);
static void qknorm(const gl * y, CUdeviceptr q, int qst, CUdeviceptr k, int kst, int nb) {
    for (int w = 0; w < 2; w++) {
        CUdeviceptr x = w ? k : q, nw = w ? y->kn : y->qn, nbias = w ? y->knb : y->qnb;
        if (!nw) continue;
        int hb = nbias ? 1 : 0, wn = w ? y->kn_n : y->qn_n, hh = hd, st = w ? kst : qst;
        float e = eps;
        CUdeviceptr bb = nbias ? nbias : nw;
        const unsigned reps = (unsigned) (wn == hd ? (w ? n_kv : n_head) : 1);
        void * a[] = { &x, &nw, &bb, &hb, &wn, &hh, &st, &e };
        launch2(KB_ROPEATTN, KF_QKNORM, reps, (unsigned) nb, 256, a);
    }
}

// Gemma and Granite scale the stream: the embedding (Gemma by sqrt(n_embd)),
// each residual branch (Granite), and the logits (Granite, Cohere divide them).
// 0 means none, as in the file's header.
static float g_emb_scale, g_res_scale, g_logit_scale;
static void scaleb(CUdeviceptr x, int xst, int n, int nb, float s) {
    if (s == 0.0f || s == 1.0f) return;
    int xs = xst, nn = n, b = nb;
    float ss = s;
    void * a[] = { &x, &xs, &nn, &b, &ss };
    launch2(KB_NORMQ, KF_SCALEB, cdiv(n, 256), (unsigned) nb, 256, a);
}
// Post-norms (Gemma 2/3, EXAONE 4): an RMSNorm on a branch's output before the
// residual add -- sml_g_qknorm with one span the width of the stream (a head
// width of 0 never equals it, so the kernel takes the whole row).
static void postnorm(CUdeviceptr x, int xst, CUdeviceptr w, int nb) {
    if (!w) return;
    int hb = 0, wn = n_embd, hh = 0, st = xst;
    float e = eps;
    void * a[] = { &x, &w, &w, &hb, &wn, &hh, &st, &e };
    launch2(KB_NORMQ, KF_QKNORM, 1, (unsigned) nb, 256, a);
}
static CUdeviceptr MOE_LG, MOE_G, MOE_U, MOE_T;   // router logits; one expert's gate, up, out
static float * moe_lg_h, * moe_sc_h, * moe_ch_h;   // the logits on the host, and the choice's scratch
static CUdeviceptr MOE_LGB;                        // a prompt chunk's router logits, PB x (ne + 1)
static CUdeviceptr MXG, MXO, MIDX, MWT;           // its experts' gathered rows, outputs, token lists
static float * moe_lgb_h;
static void mvq(const gt * t, CUdeviceptr y, CUdeviceptr x, CUdeviceptr bias);

static void f32mv(CUdeviceptr y, CUdeviceptr w, CUdeviceptr x, int rows, int cols) {
    int r = rows, c = cols;
    void * a[] = { &y, &w, &x, &r, &c };
    launch2(KB_FFN, KF_F32MV, (unsigned) rows, 1, 256, a);
}
static void axpy(CUdeviceptr y, CUdeviceptr x, int n, float w, int acc) {
    int nn = n, ac = acc;
    float ww = w;
    void * a[] = { &y, &x, &nn, &ww, &ac };
    launch(KB_FFN, KF_AXPY, cdiv(n, 256), a);
}
// One expert (or the shared one): gate and up from XB, the activation, down,
// weighted into HB2 -- where a dense FFN's output lands.
static void moe_expert(const gt * g, const gt * u, const gt * d, float w, int acc) {
    mvq(g, MOE_G, XB, 0);
    mvq(u, MOE_U, XB, 0);
    int nf = g->rows;
    void * a[] = { &MOE_G, &MOE_U, &nf };
    launch(KB_SILU, KF_SILU, cdiv(nf, 256), a);
    mvq(d, MOE_T, MOE_G, 0);
    axpy(HB2, MOE_T, n_embd, w, acc);
}
// The router runs on the card; its logits come back and the CPU's own
// sml_moe_route chooses, so the GPU and CPU cannot choose differently.
static void moe_ffn_gpu(const gl * y, const sml_layer * s) {
    const gmoe * g = y->moe;
    f32mv(MOE_LG, g->router, XB, g->ne, n_embd);
    if (g->shgate) f32mv(MOE_LG + (CUdeviceptr) g->ne * 4, g->shgate, XB, 1, n_embd);
    CU.MemcpyDtoH(moe_lg_h, MOE_LG, (size_t) (g->ne + 1) * 4);
    int sel[SML_MOE_MAX_K];
    float w[SML_MOE_MAX_K];
    const int k = sml_moe_route(M.hdr, s, moe_lg_h, moe_sc_h, moe_ch_h, sel, w);
    for (int j = 0; j < k; j++) moe_expert(&g->eg[sel[j]], &g->eu[sel[j]], &g->ed[sel[j]], w[j], j > 0);
    if (g->shg.n) {
        const float sg = g->shgate ? 1.0f / (1.0f + expf(-moe_lg_h[g->ne])) : 1.0f;
        moe_expert(&g->shg, &g->shu, &g->shd, sg, k > 0);
    }
}

// On-card experts: route, quantize per chosen expert, gate+up, quantize the
// hidden rows per chosen down, down, combine -- no host in the loop, so decode
// graphs replay MoE layers. XB is the normed input; the result lands in HB2.
static CUdeviceptr MOE_SEL, MOE_W, MOE_SC, MOE_CH, MOE_QW, MOE_QS, MOE_QM, MOE_HK, MOE_TK, MOE_S;
static int moe_wst, moe_sst, moe_hst;
static void moe_ffn_dev(const gl * y, const sml_layer * s) {
    const gmoe * g = y->moe;
    const sml_hdr * h = M.hdr;
    int ne = g->ne, k = (int) h->n_expert_used;
    if (k <= 0 || k > ne) k = ne;
    if (k > SML_MOE_MAX_K) k = SML_MOE_MAX_K;
    f32mv(MOE_LG, g->router, XB, ne, n_embd);
    if (g->shgate) f32mv(MOE_LG + (CUdeviceptr) ne * 4, g->shgate, XB, 1, n_embd);
    {
        int fl = (int) h->moe_flags, hb = g->ebias ? 1 : 0, ng = (int) h->moe_groups, tg = (int) h->moe_topk_group;
        float ms = h->moe_scale;
        CUdeviceptr eb = g->ebias ? g->ebias : MOE_LG;
        void * a[] = { &MOE_LG, &ne, &k, &fl, &eb, &hb, &ng, &tg, &ms, &MOE_SC, &MOE_CH, &MOE_SEL, &MOE_W };
        launch2(KB_FFN, KF_MOE_ROUTE, 1, 1, 32, a);
    }
    (void) s;
    {
        CUdeviceptr x = XB;
        int xst = 0, wst = moe_wst, sst = moe_sst, n = n_embd;
        void * a[] = { &x, &xst, (void *) &g->pg, &MOE_SEL, &MOE_QW, &wst, &MOE_QS, &MOE_QM, &sst, &n };
        launch2(KB_FFN, KF_MOE_QUANT, cdiv(n / SML_GROUP, 8), (unsigned) k, 256, a);
    }
    {
        int hst = moe_hst, wst = moe_wst, sst = moe_sst, n = n_embd;
        void * a[] = { &MOE_HK, &hst, (void *) &g->tg, (void *) &g->tu, &MOE_SEL,
                       &MOE_QW, &wst, &MOE_QS, &MOE_QM, &sst, &n };
        launch2(KB_FFN, KF_MOE_GU, grid_for(g->fe), (unsigned) k, g_bsize, a);
    }
    {
        int xst = moe_hst, wst = moe_wst, sst = moe_sst, n = g->fe;
        void * a[] = { &MOE_HK, &xst, (void *) &g->pd, &MOE_SEL, &MOE_QW, &wst, &MOE_QS, &MOE_QM, &sst, &n };
        launch2(KB_FFN, KF_MOE_QUANT, cdiv(n / SML_GROUP, 8), (unsigned) k, 256, a);
    }
    {
        int tst = n_embd, wst = moe_wst, sst = moe_sst, n = g->fe;
        void * a[] = { &MOE_TK, &tst, (void *) &g->td, &MOE_SEL, &MOE_QW, &wst, &MOE_QS, &MOE_QM, &sst, &n };
        launch2(KB_FFN, KF_MOE_DN, grid_for(n_embd), (unsigned) k, g_bsize, a);
    }
    int hs = g->shg.n ? 1 : 0, hsg = g->shgate ? 1 : 0;
    if (hs) {
        mvq(&g->shg, MOE_G, XB, 0);
        mvq(&g->shu, MOE_U, XB, 0);
        int nf = g->shg.rows;
        void * a[] = { &MOE_G, &MOE_U, &nf };
        launch(KB_SILU, KF_SILU, cdiv(nf, 256), a);
        mvq(&g->shd, MOE_S, MOE_G, 0);
    }
    {
        int tst = n_embd, n = n_embd;
        CUdeviceptr sp = hs ? MOE_S : MOE_TK, sg = MOE_LG + (CUdeviceptr) ne * 4;
        void * a[] = { &HB2, &MOE_TK, &tst, &MOE_W, &k, &sp, &hs, &sg, &hsg, &n };
        launch2(KB_FFN, KF_MOE_COMB, cdiv(n_embd, 256), 1, 256, a);
    }
}

// What a branch's output goes through before it joins the stream.
static void branch_out(CUdeviceptr y, int yst, CUdeviceptr post, int nb) {
    postnorm(y, yst, post, nb);
    scaleb(y, yst, n_embd, nb, g_res_scale);
}

static CUdeviceptr OUTB, POSE;     // output-head bias (Phi, GPT-J); learned positions (GPT-2 lineage)
static CUdeviceptr ALIBI;          // ALiBi slopes per head (BLOOM, MPT, Falcon), read by sml_score
static CUdeviceptr EMBN, EMBNB;    // BLOOM's LayerNorm on the embedding row
static int g_score_adj;            // ALiBi or an attention cap: the tensor-core flash kernel has neither
static float g_softcap, g_clip;    // Gemma 2's final logit cap; OLMo/DBRX's Q/K/V clamp; 0 = none
static void addrow(CUdeviceptr x, int xst, int n, int nb, CUdeviceptr tab, int tst, CUdeviceptr dpos, int pos0) {
    int xs = xst, nn = n, b = nb, ts = tst, p0 = pos0;
    void * a[] = { &x, &xs, &nn, &b, &tab, &ts, &dpos, &p0 };
    launch2(KB_NORMQ, KF_ADDROW, cdiv(n, 256), (unsigned) nb, 256, a);
}
static void capb(CUdeviceptr x, int xst, int n, int nb, float cap) {
    if (cap == 0.0f) return;
    int xs = xst, nn = n, b = nb;
    float c = cap;
    void * a[] = { &x, &xs, &nn, &b, &c };
    launch2(KB_NORMQ, KF_CAPB, cdiv(n, 256), (unsigned) nb, 256, a);
}
static void clampb(CUdeviceptr x, int xst, int n, int nb) {
    if (g_clip == 0.0f) return;
    int xs = xst, nn = n, b = nb;
    float c = g_clip;
    void * a[] = { &x, &xs, &nn, &b, &c };
    launch2(KB_NORMQ, KF_CLAMPB, cdiv(n, 256), (unsigned) nb, 256, a);
}
// Q, K and V straight out of their projections, clamped first as on the CPU.
static void clamp_qkv(CUdeviceptr q, int qst, CUdeviceptr k, int kst, CUdeviceptr v, int vst, int nb) {
    clampb(q, qst, n_head * hd, nb);
    clampb(k, kst, kvd, nb);
    clampb(v, vst, kvd, nb);
}
static void embed_norm(CUdeviceptr x, int nb) {
    if (!EMBN) return;
    int hb = EMBNB ? 1 : 0, wn = n_embd, hh = 0, st = n_embd;
    float e = eps;
    CUdeviceptr bb = EMBNB ? EMBNB : EMBN;
    void * a[] = { &x, &EMBN, &bb, &hb, &wn, &hh, &st, &e };
    launch2(KB_NORMQ, KF_QKNORM, 1, (unsigned) nb, 256, a);
}
// The embedding's epilogue in sml-run.c's order: the scale, the position row, the norm.
static void embed_out(CUdeviceptr x, int nb, CUdeviceptr dpos, int pos0) {
    scaleb(x, n_embd, n_embd, nb, g_emb_scale);
    if (POSE) addrow(x, n_embd, n_embd, nb, POSE, n_embd, dpos, pos0);
    embed_norm(x, nb);
}
// The logits' epilogue: head bias, divided by the logit scale, then the soft cap.
static void logits_out(CUdeviceptr lg) {
    if (OUTB) addrow(lg, n_vocab, n_vocab, 1, OUTB, 0, 0, 0);
    scaleb(lg, n_vocab, n_vocab, 1, g_logit_scale != 0.0f ? 1.0f / g_logit_scale : 0.0f);
    capb(lg, n_vocab, n_vocab, 1, g_softcap);
}

static void f2_layer(const gl * y, int l, int pos) {
    f2_normq(HB2, y->an, y->anb, &y->q, l == 0 ? 0 : 1);
    f2_mv3(KB_MV3, &y->q, Q, y->qb, &y->k, Kv, y->kb, &y->v, V, y->vb);
    clamp_qkv(Q, n_head * hd, Kv, kvd, V, kvd, 1);
    if (g_qk_late && l_half(l)) {
        CUdeviceptr rc = l_rc(l), rs = l_rs(l);
        int nh = n_head, nk = n_kv, hh = hd, hf = l_half(l), il = interleaved;
        void * a[] = { &Q, &Kv, &rc, &rs, &DPOS, &nh, &nk, &hh, &hf, &il };
        launch(KB_ROPEATTN, KF_ROPEQD, cdiv((n_head + n_kv) * l_half(l), 128), a);
    }
    qknorm(y, Q, n_head * hd, Kv, kvd, 1);
    f2_ropeattn(l, pos);
    f2_qdmv(&y->o, HB2, XB, y->ob);
    branch_out(HB2, n_embd, y->pa, 1);
    if (y->moe) {
        // the norm's quantized copy goes unused: each expert quantizes in its own order
        f2_normq(HB2, y->fn, y->fnb, &y->moe->eg[0], par ? 2 : 1);
        // the on-card router knows softmax/sigmoid top-k; PhiMoE's sparsemixer is chosen by
        // the host's sml_moe_route
        if (M.hdr->moe_flags & 4u) moe_ffn_gpu(y, &C.layers[l]);
        else moe_ffn_dev(y, &C.layers[l]);
        branch_out(HB2, n_embd, y->pf, 1);
        return;
    }
    f2_normq(HB2, y->fn, y->fnb, y->gate.n ? &y->gate : &y->up, par ? 2 : 1);
    f2_ffn(y);
    f2_qdmv(&y->down, HB2, HB, y->db);
    branch_out(HB2, n_embd, y->pf, 1);
}

// One token with no host in it: embed from *DTOK, the fused layers (attention
// reads *DPOS), the head, and argmax2 writing *DTOK and *DPOS for the next replay.
static void graph_token(void) {
    int n = n_embd, six = EMB.six, hm = EMB.have_min;
    void * ea[] = { &X, (void *) &EMB.c2, (void *) &EMB.c4, (void *) &EMB.sd, (void *) &EMB.sq,
                    (void *) &EMB.md, (void *) &EMB.mq, (void *) &EMB.perm, &DTOK, &n, &six, &hm };
    launch(KB_EMBED, KF_EMBED, cdiv(n_embd, 256), ea);
    embed_out(X, 1, DPOS, 0);
    for (int l = 0; l < NL; l++) f2_layer(&L[l], l, 0);
    f2_normq(HB2, ON, ONB, &HEAD, 1);
    f2_mv3(KB_HEAD3, &HEAD, LOG, 0, NULL, 0, 0, NULL, 0, 0);
    logits_out(LOG);
    int nv = n_vocab;
    void * aa[] = { &LOG, &nv, &DTOK, &DPOS, &DLOG };
    launch(KB_ARGMAX, KF_ARGMAX2, 1, aa);
}

// ---- batched prompt reading: PB tokens per weight read (sml-gpu-fast.cu).
// Prompt tokens per weight read. 64 with the tensor-core kernel; the dp4a tiled
// kernel only takes 16, so it is not used when PB is larger.
// 512, as the reference reads a prompt. Swept 64/128/256/512 (built with -DPB):
// at equal clock, 512 beat 64 by 10-17% (1B pp512 972 vs 831 at 712 MHz; 3B
// 140.8 vs 125 at 210 MHz) -- fewer launches, each weight tile staged a quarter as
// often, and enough blocks on the small matrices to hide latency. Answers are
// identical at every size.
#ifndef PB
#define PB 512
#endif
static CUdeviceptr XBAT, XBB, HBB, HB2B, QBB, KBB, VBB, QWB, QSB, QMB, ATTB, DTOKS, GBB;
// SML_HMM=1: prompt matmuls on f16 weight tiles (sml_g_hmm) instead of int8 MMAs
// with a per-MMA epilogue. XHB holds each tensor's permuted f16 activations;
// g_xsrc is the f32 input the current matmuls read, g_hperm the permutation
// XHB was last gathered with (0: stale).
static CUdeviceptr XHB, g_xsrc, g_hperm;
static int g_hmm = 0, g_xst = 0;
// SML_TCM3=1: re-quantize each weight tensor to int8 with one scale per 256
// (group scale and min folded in) into W8/S8, then int8 MMAs with no
// per-group epilogue (sml_g_tcm3). The scratch holds one tensor at a time.
static CUdeviceptr W8, S8, PART;

// ---- Joseph's granary (Genesis 41: store in the years of plenty) ----------
// A tensor whose int8 copy cannot be kept is re-made on every call. Its re-making
// moves memory and the matmul before it computes, so they can overlap: while
// tensor i runs on the tensor cores (legacy stream), tensor i+1 is re-made into
// the other of two buffers on a second, non-blocking stream. The first prompt
// learns the order the unkept tensors are used in; later prompts store ahead.
// Events: ready[s] (made, on the side stream) and free[s] (its matmul done).
static int g_joseph = 1;            // SML_JOSEPH=0: re-make in line, as before
// SML_TCM4=1: a tensor that is not kept is never re-made -- its 4-bit codes are
// decoded in passing by sml_g_tcm4 (Acts 5:15)
static int g_tcm4 = 1;
// ...for prompts up to this many tokens; longer ones re-make the tensor once per
// call and run tcm3 (measured on the 3B: in passing 0.889x vs 0.743x at 128
// tokens, 0.815x vs 0.874x at 512 -- median ratios to the reference, 8 rounds)
static int g_tcm4_maxnb = 256;
// SML_TCM5 (default 1): a tensor that is not kept runs tcm3 with its weight tile
// re-made from the 4-bit codes as it is staged (sml_g_tcm5) -- the kept path's
// int8 values, bit for bit, with no pass of its own. Takes precedence over tcm4,
// for prompts up to g_tcm4_maxnb tokens: it re-makes the tile once per 64-token
// block, so past that the one-pass re-make wins. Measured at the thermal floor on
// the 3B: 308 vs tcm4 299 vs re-make 247 tok/s at 128 tokens; 318 vs 349 at 512.
static int g_tcm5 = 1;
static CUstream g_s2;
#define JK_MAX 6
static CUevent j_ready[JK_MAX], j_free[JK_MAX];
static CUdeviceptr JW8[JK_MAX], JS8[JK_MAX];
// SML_GRANARY (default 4): granary slots. With two, a copy could only start one
// tensor ahead; the 3B's unkept tensors came in runs, each copy (3.5 ms on this
// bus) behind a 0.7 ms matmul, and a 128-token prompt waited or re-made instead.
static int JK = 4;
static const void * j_slot_t[JK_MAX];
static const gt * j_order[512];
static int j_n = 0, j_learned = 0, j_pos = 0, j_ok = 0;
// SML_RUTH (default 1): the granary is filled from the HOST. A tensor whose int8
// copy does not fit on the card is re-made once, on the first prompt, and the
// result kept in pinned host memory; every later prompt streams it back into
// the granary on the copy engine -- which is not the tensor cores -- while the
// matmul before it runs. Re-making it on the GPU each call competed with the
// matmuls for the same SMs, and at the thermal floor (210 MHz) that is all the
// SMs have. PCIe here: 6.5 GB/s pinned (SML_PCIE_TEST=1).
static int g_ruth = 1;
static struct { const void * t; void * hw; void * hs; size_t wb, sb; } r_host[512];
static int r_n = 0;
static long r_used = 0, r_fb = 0;   // granary tensors run from the copy / re-made in passing
static int g_nkept = 0;
// How much to stream. A cool GPU reads a 128-token 3B prompt in ~130 ms; its unkept
// copies take ~290 ms over this bus, so streaming them all made the cool prompt
// 0.25x the reference (measured 2026-09-26) while it won at the thermal floor,
// where compute takes ~415 ms. So: the first prompt after learning streams nothing
// and is timed (compute per token); each later prompt streams the fraction
// 0.9 x compute / transfer, spread evenly, and re-makes the rest. SML_RUTH_FRAC
// fixes the fraction instead.
static double g_ruth_f = 1.0, g_ruth_rate = 0.0, g_ruth_bw = 0.0;
static int g_ruth_phase = 0;          // 0 learning, 1 measure (stream none), 2 adaptive
static size_t g_ruth_bytes = 0;       // bytes if every unkept tensor streams
static double g_frac_fixed = -1.0;   // SML_RUTH_FRACS: set per SML_PP entry (a test hook)
static int ruth_stream(int p) {       // position p of the unkept order streams
    const double f = g_ruth_f;
    return (int) ((double) (p + 1) * f) > (int) ((double) p * f);
}
static size_t g_vram_at_keep = 0;

static int ruth_on(void) {
    return g_ruth && g_joseph && j_ok && CU.MemAllocHost && CU.MemcpyHtoDAsync;
}

static int ruth_find(const void * t) {
    for (int i = 0; i < r_n; i++) if (r_host[i].t == t) return i;
    return -1;
}
static unsigned j_seq = 0;
// SML_SPLITK (default 1 = off; measured no gain on the 1B): the most slices a deep tcm3 sum is cut into
static int g_splitk = 1;
// SML_T3SWAP (default 1): tcm3's grid walks token blocks first, so the blocks
// running together share one slab of weight rows: each weight comes from DRAM
// once and the other token blocks find it in L2. Under load this GPU sits at
// 210 MHz (thermal slowdown 0x20) with memory at full clock, where compute hides
// it (measured equal); at 1740 MHz re-reading the weights once per 64 tokens is
// ~10 GB per 512-token prompt, ~55 ms of a reference pp512 that takes 82 ms.
static int g_t3swap = 1;
// SML_RQ_MINNB: below this many tokens a tensor that is not kept skips the
// per-call re-quantization and runs the direct 4-bit kernel (tcm2)
static int g_rq_minnb = 0;   // measured: tcm2 no faster at pp128 on the 3B
static int g_tcm3 = 2;
// device bytes the kept int8 copies may bring the total to (SML_KEEP_MB)
static size_t g_keep_budget = (size_t) 3200 << 20;
// SML_FATC=1: prompt attention on f16 tensor cores (head size 64, no window)
static int g_fatc = 1;
#define HM_KC_HOST 128   /* k per chunk in sml_g_hmm */
// SML_TMM (default 1): prompt matvecs use the tiled kernel; 0 = one warp per row.
static int g_tmm = 1;
// SML_TCM: prompt matvecs on the int8 tensor cores (mma.sync m16n8k32).
static int g_tcm = 1;
// SML_FATTN (default 1): batched attention streams K/V once per kv head; 0 = old kernel.
static int g_fattn = 1;
// SML_GPU_SB (default 1): prompt activations quantized with one scale per 256
// values, and the tensor-core kernel sums a superblock's groups in integers.
static int g_gsb = 1;

static void launch2s(int kb, int kf, unsigned gx, unsigned gy, unsigned block, void ** args,
                     CUstream st);
// SML_TCM7 (default 1): the re-made-in-passing prompt matmul re-makes the next
// superblock behind the current one's MMAs (sml_g_tcm7) instead of between barriers.
static int g_tcm7 = 1;
// SML_TCM8 (default 1): past 64 tokens the re-made tile serves 128 tokens (sml_g_tcm8,
// eight warps), so each weight is re-made half as often.
static int g_tcm8 = 1;
// SML_TCM9 (default 1): kept int8 tensors past 64 tokens (and 1024 rows) take a
// 128 x 128 cp.async tile (sml_g_tcm9) instead of tcm3's 64 x 64.
static int g_tcm9 = 1;
static unsigned g_dsm[KF_N];          // dynamic shared bytes per kernel
static void launch2(int kb, int kf, unsigned gx, unsigned gy, unsigned block, void ** args) {
    double tp = 0.0;
    if (!gx) gx = 1;
    if (g_prof) { CU.CtxSynchronize(); tp = now_s(); }
    if (CU.LaunchKernel(KF[kf], gx, gy, 1, block, 1, 1, g_dsm[kf], g_stream, args, NULL) != CU_OK) {
        puts("  launch failed"); GPU_DIE();
    }
    g_launches++;
    if (g_prof) { CU.CtxSynchronize(); g_acc[kb] += now_s() - tp; }
}

static int prefill_ready(void) {
    // PB is limited by the widest batch kernel: sml_g_tcm takes 64 tokens, the
    // one-warp-per-row fallback any number; the dp4a tiled kernel is skipped above 16.
    // sml_g_tcm2 and every other batch kernel take any batch; the older tcm
    // (SML_GPU_SB=0) holds 64 tokens per block, so it caps the batch there.
    if (!g_fast || (!(g_tcm && g_gsb) && PB > 64)) return 0;
    for (int l = 0; l < NL; l++) if (L[l].moe && !L[l].moe->dev) return 0;   // experts need their tables
    for (int l = 0; l < NL; l++)
        if (!(L[l].kv_same && (!L[l].gate.n || L[l].gu_same))) return 0;
    return 1;
}

static void pp_alloc(void) {
    const int wide = g_wide;
    XBAT = dalloc((size_t) PB * n_embd * 4);
    XBB  = dalloc((size_t) PB * wide * 4);
    HBB  = dalloc((size_t) PB * wide * 4);
    HB2B = dalloc((size_t) PB * wide * 4);
    QBB  = dalloc((size_t) PB * n_head * hd * 4);
    KBB  = dalloc((size_t) PB * kvd * 4);
    VBB  = dalloc((size_t) PB * kvd * 4);
    QWB  = dalloc((size_t) PB * (wide / 4 + 16) * 4);
    XHB  = dalloc((size_t) PB * (wide / 2 + 16) * 4);
    {
        // the largest batched tensor decides the scratch: rows x n bytes
        size_t big = 0;
        for (int l = 0; l < NL; l++) {
            const gt * ts[7] = { &L[l].q, &L[l].k, &L[l].v, &L[l].o, &L[l].gate, &L[l].up, &L[l].down };
            for (int i = 0; i < 7; i++) {
                const size_t b = (size_t) ts[i]->rows * (size_t) ts[i]->n;
                if (b > big) big = b;
            }
            const gmoe * g = L[l].moe;
            for (int e = 0; g && e < g->ne; e++) {
                const gt * es[3] = { &g->eg[e], &g->eu[e], &g->ed[e] };
                for (int i = 0; i < 3; i++) {
                    const size_t b = (size_t) es[i]->rows * (size_t) es[i]->n;
                    if (b > big) big = b;
                }
            }
            if (g) {
                const gt * ss_[3] = { &g->shg, &g->shu, &g->shd };
                for (int i = 0; i < 3; i++) {
                    const size_t b = (size_t) ss_[i]->rows * (size_t) ss_[i]->n;
                    if (b > big) big = b;
                }
                if (!MXG) {
                    MXG = dalloc((size_t) PB * wide * 4); MXO = dalloc((size_t) PB * wide * 4);
                    MIDX = dalloc((size_t) PB * (SML_MOE_MAX_K + 1) * 4);
                    MWT = dalloc((size_t) PB * (SML_MOE_MAX_K + 1) * 4);
                }
            }
        }
        W8 = dalloc(big ? big : 16);
        if (g_joseph && CU.StreamCreate && CU.EventCreate && CU.EventRecord && CU.StreamWaitEvent
            && CU.StreamCreate(&g_s2, 1) == CU_OK) {            // 1: non-blocking
            j_ok = 1;
            if (getenv("SML_GRANARY")) JK = atoi(getenv("SML_GRANARY"));
            if (JK < 2) JK = 2;
            if (JK > JK_MAX) JK = JK_MAX;
            for (int s = 0; s < JK; s++) {
                JW8[s] = dalloc(big ? big : 16);
                JS8[s] = dalloc((big / 256 + 16) * 4);
                if (CU.EventCreate(&j_ready[s], 2) != CU_OK || CU.EventCreate(&j_free[s], 2) != CU_OK)
                    j_ok = 0;
                else
                    CU.EventRecord(j_free[s], NULL);   // both buffers start free
            }
        }
        S8 = dalloc((big / 256 + 16) * 4);
    }
    QSB  = dalloc((size_t) PB * (wide / SML_GROUP + 16) * 4);
    QMB  = dalloc((size_t) PB * (wide / SML_GROUP + 16) * 4);
    ATTB = dalloc((size_t) PB * n_head * n_ctx * 4);
    DTOKS = dalloc((size_t) PB * 4);
    GBB  = dalloc((size_t) PB * wide * 4);
    if (getenv("SML_TMM")) g_tmm = atoi(getenv("SML_TMM"));
    if (getenv("SML_TCM")) g_tcm = atoi(getenv("SML_TCM"));
    if (getenv("SML_FATTN")) g_fattn = atoi(getenv("SML_FATTN"));
    if (getenv("SML_GPU_SB")) g_gsb = atoi(getenv("SML_GPU_SB"));
    if (getenv("SML_HMM")) g_hmm = atoi(getenv("SML_HMM"));
    if (getenv("SML_TCM3")) g_tcm3 = atoi(getenv("SML_TCM3"));
    if (getenv("SML_SPLITK")) g_splitk = atoi(getenv("SML_SPLITK"));
    // partial sums only when a deep sum is split: with split-K off (the default)
    // these 90 MB (the 3B at 512 tokens) are VRAM the kept int8 copies can use
    PART = dalloc(g_splitk > 1 ? (size_t) 4 * PB * wide * 4 : 16);
    if (getenv("SML_T3SWAP")) g_t3swap = atoi(getenv("SML_T3SWAP"));
    if (getenv("SML_RQ_MINNB")) g_rq_minnb = atoi(getenv("SML_RQ_MINNB"));
    if (getenv("SML_JOSEPH")) g_joseph = atoi(getenv("SML_JOSEPH"));
    if (getenv("SML_TCM4")) g_tcm4 = atoi(getenv("SML_TCM4"));
    if (getenv("SML_TCM4_MAXNB")) g_tcm4_maxnb = atoi(getenv("SML_TCM4_MAXNB"));
    if (getenv("SML_TCM5")) g_tcm5 = atoi(getenv("SML_TCM5"));
    if (getenv("SML_RUTH")) g_ruth = atoi(getenv("SML_RUTH"));
    if (g_splitk < 1) g_splitk = 1;
    if (g_splitk > 4) g_splitk = 4;
    // SML_KEEP_AUTO=1: the keep budget from what the driver reports free once the
    // prompt buffers exist, less SML_KEEP_MARGIN_MB (default 256). Opt-in: under
    // Windows (WDDM) the driver reports ~950 MB of this 4 GB card as taken beside
    // our 2161 MB, while nvidia-smi shows ~320 MB, so the reported figure gives a
    // SMALLER budget (2891 MB) than the fixed 3200 every race so far has used
    // without trouble (peak 3524 of 4096 MiB, 2026-09-26).
    // The default budget (2026-09-28): what the card PHYSICALLY has free once
    // the prompt buffers exist (NVML, as nvidia-smi sees it -- other programs'
    // memory included), less SML_KEEP_MARGIN_MB (256). The fixed 3200 MB left
    // ~570 MB of this 4 GB card idle while 1.7 GB of the 3B's int8 copies were
    // re-made or streamed; 3600 measured a 3935 MiB peak and fit. No NVML: the
    // fixed budget stands. SML_KEEP_MB overrides; SML_KEEP_AUTO=0 keeps 3200.
    if (!getenv("SML_KEEP_MB") && !(getenv("SML_KEEP_AUTO") && !atoi(getenv("SML_KEEP_AUTO")))) {
        const size_t fr = nvml_free_bytes(g_dev);
        const size_t margin = (size_t) (getenv("SML_KEEP_MARGIN_MB") ? atoi(getenv("SML_KEEP_MARGIN_MB")) : 256) << 20;
        if (fr > margin) g_keep_budget = g_vram + fr - margin;
        if (getenv("SML_RUTH_STATS"))
            printf("  vram: %zu MB ours, %zu MB physically free -> keep budget %zu MB\n",
                   g_vram >> 20, fr >> 20, g_keep_budget >> 20);
    }
    if (!getenv("SML_KEEP_MB") && getenv("SML_KEEP_AUTO") && atoi(getenv("SML_KEEP_AUTO")) == 2 && CU.MemGetInfo) {
        size_t fr = 0, tot = 0;
        const size_t margin = (size_t) (getenv("SML_KEEP_MARGIN_MB") ? atoi(getenv("SML_KEEP_MARGIN_MB")) : 256) << 20;
        if (CU.MemGetInfo(&fr, &tot) == CU_OK && fr > 0)
            g_keep_budget = fr > margin ? g_vram + fr - margin : g_vram;
        if (getenv("SML_RUTH_STATS"))
            printf("  vram: %zu MB ours, %zu MB free of %zu -> keep budget %zu MB\n",
                   g_vram >> 20, fr >> 20, tot >> 20, g_keep_budget >> 20);
    }
    if (getenv("SML_FATC")) g_fatc = atoi(getenv("SML_FATC"));
    if (getenv("SML_KEEP_MB")) g_keep_budget = (size_t) atoi(getenv("SML_KEEP_MB")) << 20;
    if (hd > 128) g_fattn = 0;   // the streamed kernel holds up to 128 dims per head
}

static void launch2s(int kb, int kf, unsigned gx, unsigned gy, unsigned block, void ** args,
                     CUstream st) {
    if (!gx) gx = 1;
    if (CU.LaunchKernel(KF[kf], gx, gy, 1, block, 1, 1, g_dsm[kf], st, args, NULL) != CU_OK) {
        puts("  launch failed"); GPU_DIE();
    }
    g_launches++;
}

static void requant8(const gt * t, CUdeviceptr w8, CUdeviceptr s8, int nsb, CUstream st) {
    int nn = t->n, rr = t->rows, hmm_ = t->have_min, sx = t->six;
    void * q[] = { (void *) &t->c2, (void *) &t->c4, &sx, (void *) &t->sd, (void *) &t->sq,
                   (void *) &t->md, (void *) &t->mq, &hmm_, &w8, &s8, &nn, &rr };
    launch2s(KB_SMV6, KF_REQUANT8, cdiv(rr * nsb, 8), 1, 256, q, st);
}

static void t5_launch(int kb, unsigned rows, unsigned nb, void ** args) {
    if (g_tcm8 && nb > 64) {
        if (g_t3swap) launch2(kb, KF_TCM8, cdiv(nb, 128), cdiv(rows, 64), 256, args);
        else          launch2(kb, KF_TCM8, cdiv(rows, 64), cdiv(nb, 128), 256, args);
    } else {
        if (g_t3swap) launch2(kb, KF_TCM5, cdiv(nb, 64), cdiv(rows, 64), 128, args);
        else          launch2(kb, KF_TCM5, cdiv(rows, 64), cdiv(nb, 64), 128, args);
    }
}

static void tcm3_launch(int kb, CUdeviceptr y, int ys, CUdeviceptr w8, CUdeviceptr s8,
                        CUdeviceptr bb, int wst, int sst, int n, int rows, int hb, int nb, int nsb) {
    int split = nsb / 8 < g_splitk ? nsb / 8 : g_splitk, b = nb;
    if (split < 1) split = 1;
    if (g_tcm9 && split == 1 && nb > 64 && rows >= 1024) {
        void * a9[] = { &y, &ys, &w8, &s8, &bb, &QWB, &wst, &QSB, &sst, &n, &rows, &hb, &b, &g_t3swap };
        if (g_t3swap) launch2(kb, KF_TCM9, cdiv(nb, 128), cdiv(rows, 128), 256, a9);
        else          launch2(kb, KF_TCM9, cdiv(rows, 128), cdiv(nb, 128), 256, a9);
        return;
    }
    void * c[] = { &y, &ys, &w8, &s8, &bb, &QWB, &wst, &QSB, &sst, &n, &rows, &hb, &b,
                   &PART, &split, &g_t3swap };
    if (g_t3swap) launch2(kb, KF_TCM3, cdiv(nb, 64) * split, cdiv(rows, 64), 128, c);
    else          launch2(kb, KF_TCM3, cdiv(rows, 64), cdiv(nb, 64) * split, 128, c);
    if (split > 1) {
        int rr = rows, bbn = nb, sp = split, hbb = hb;
        void * r[] = { &y, &ys, &PART, &bb, &hbb, &rr, &bbn, &sp };
        launch2(kb, KF_SPLITK, cdiv(rows, 256), (unsigned) nb, 256, r);
    }
}

// Keep int8 copies spread evenly through the model rather than first-come:
// first-come kept layers 0-12 of the 3B and left the rest a run of unkept
// tensors, whose streamed copies then had no kept matmul to hide behind.
// The share is set once, from the budget left and the int8 bytes of every
// layer tensor; a tensor is kept while the kept share of those seen so far
// is below it. SML_KEEP_SPREAD=0: first-come.
static int keep_spread(size_t sz) {
    static double frac = -1.0;
    static size_t seen = 0, kept_b = 0;
    static int on = -1;
    if (on < 0) on = getenv("SML_KEEP_SPREAD") ? atoi(getenv("SML_KEEP_SPREAD")) : 1;
    if (!on) return 1;
    if (frac < 0.0) {
        double total = 0.0;
        for (int l = 0; l < NL; l++) {
            const gt * ts[7] = { &L[l].q, &L[l].k, &L[l].v, &L[l].o, &L[l].gate, &L[l].up, &L[l].down };
            for (int i = 0; i < 7; i++) total += (double) ts[i]->rows * (double) ts[i]->n * (1.0 + 1.0 / 64.0);
        }
        const double avail = g_keep_budget > g_vram ? (double) (g_keep_budget - g_vram) : 0.0;
        frac = total > 0.0 ? avail / total : 1.0;
        if (frac > 1.0) frac = 1.0;
    }
    seen += sz;
    if ((double) (kept_b + sz) <= frac * (double) seen + 1.0) { kept_b += sz; return 1; }
    return 0;
}

static void mvb(int kb, const gt * t, CUdeviceptr y, int yst, CUdeviceptr bias, int nb) {
    const int wide = g_wide;
    int ys = yst, n = t->n, rows = t->rows, six = t->six, hm = t->have_min, hb = bias ? 1 : 0, b = nb;
    int wst = wide / 4 + 16, sst = wide / SML_GROUP + 16;
    CUdeviceptr bb = bias ? bias : QSB;
    void * a[] = { &y, &ys, (void *) &t->c2, (void *) &t->c4, (void *) &t->sd, (void *) &t->sq,
                   (void *) &t->md, (void *) &t->mq, &bb, &QWB, &wst, &QSB, &QMB, &sst,
                   &n, &rows, &six, &hm, &hb, &b };
    if (g_tcm3 && g_tcm && g_gsb) {
        const int nsb = (n / SML_GROUP + 7) / 8;
        // SML_TCM3=2: each tensor re-quantized ONCE and kept (VRAM permitting);
        // 1: into the shared scratch on every call
        static struct { CUdeviceptr c4, w8, s8; } kept[512];
        static int nkept = 0;
        CUdeviceptr w8 = W8, s8 = S8;
        int fresh = 1;
        if (g_tcm3 == 2 && !g_in_stream) {
            int i = 0;
            while (i < nkept && kept[i].c4 != t->c4) i++;
            if (!g_vram_at_keep) g_vram_at_keep = g_vram;
            if (i < nkept) { w8 = kept[i].w8; s8 = kept[i].s8; fresh = 0; }
            else if (nkept < 512 && g_vram + (size_t) rows * n + (size_t) rows * nsb * 4 < g_keep_budget
                     && keep_spread((size_t) rows * n)) {
                // kept only while the card has room (4 GB here): past the budget a
                // tensor is re-quantized into the scratch on every call instead
                kept[nkept].c4 = t->c4;
                kept[nkept].w8 = w8 = dalloc((size_t) rows * n);
                kept[nkept].s8 = s8 = dalloc((size_t) rows * nsb * 4);
                nkept++; g_nkept = nkept;
            }
        }
        // A tensor that is not kept must be re-made on every call; spread over a
        // short prompt that costs more than the 4-bit kernel's per-group epilogue
        // (Qwen2.5-3B pp128). Below SML_RQ_MINNB tokens it takes tcm2 instead.
        if (fresh && w8 == W8 && g_tcm5 && nb <= g_tcm4_maxnb && !ruth_on()) {
            void * c[] = { &y, &ys, (void *) &t->c2, (void *) &t->c4, &six, (void *) &t->sd, (void *) &t->sq,
                           (void *) &t->md, (void *) &t->mq, &hm, &bb, &QWB, &wst, &QSB, &sst,
                           &n, &rows, &hb, &b, &g_t3swap };
            t5_launch(kb, (unsigned) rows, (unsigned) nb, c);
            return;
        }
        if (fresh && w8 == W8 && g_tcm4 && nb <= g_tcm4_maxnb && !ruth_on()) {
            void * c[] = { &y, &ys, (void *) &t->c2, (void *) &t->c4, (void *) &t->sd, (void *) &t->sq,
                           (void *) &t->md, (void *) &t->mq, &bb, &QWB, &wst, &QSB, &QMB, &sst,
                           &n, &rows, &six, &hm, &hb, &b, &g_t3swap };
            if (g_t3swap) launch2(kb, KF_TCM4, cdiv(nb, 64), cdiv(rows, 64), 128, c);
            else          launch2(kb, KF_TCM4, cdiv(rows, 64), cdiv(nb, 64), 128, c);
            return;
        }
        if (fresh && w8 == W8 && g_joseph && j_ok && !g_prof && !g_in_stream) {
            if (!j_learned) {
                if (j_n < 512) j_order[j_n++] = t;      // learning: in line, as before
            } else if (j_order[j_pos] != t) {
                j_learned = 0; j_n = 0; j_pos = 0;         // the order changed: learn again
                j_order[j_n++] = t;
            } else {
                const int slot = (int) (j_seq % (unsigned) JK);
                // fill slot `sl` with tensor `tt`: its host copy if there is one
                // (the copy engine), else re-made on the side stream
#define J_FILL(sl, tt, cp) do {                                                    \
                    const gt * t_ = (tt);                                          \
                    CU.StreamWaitEvent(g_s2, j_free[sl], 0);                       \
                    const int h_ = (ruth_on() && (cp)) ? ruth_find(t_) : -1;       \
                    if (h_ >= 0) {                                                 \
                        CU.MemcpyHtoDAsync(JW8[sl], r_host[h_].hw, r_host[h_].wb, g_s2); \
                        CU.MemcpyHtoDAsync(JS8[sl], r_host[h_].hs, r_host[h_].sb, g_s2); \
                    } else {                                                       \
                        requant8(t_, JW8[sl], JS8[sl], (t_->n / SML_GROUP + 7) / 8, g_s2); \
                    }                                                              \
                    CU.EventRecord(j_ready[sl], g_s2);                             \
                    j_slot_t[sl] = t_;                                             \
                } while (0)
                // the ring holds this tensor and the next JK-1: fill any that is not
                const int short_p = nb <= g_tcm4_maxnb && g_tcm5;
                for (int d = 0; d < JK; d++) {
                    const int sl = (int) ((j_seq + (unsigned) d) % (unsigned) JK);
                    const int p = (j_pos + d) % j_n;
                    const gt * want = j_order[p];
                    const int st = ruth_stream(p);
                    if (!st && short_p) continue;               // re-made in passing when it comes
                    if (j_slot_t[sl] != want) J_FILL(sl, want, st);
                }
#undef J_FILL
                const int cur_stream = ruth_stream(j_pos);
                const int nx = (j_pos + 1) % j_n;
                j_pos = nx; j_seq++;
                if (!cur_stream && short_p) {
                    void * c5[] = { &y, &ys, (void *) &t->c2, (void *) &t->c4, &six, (void *) &t->sd,
                                    (void *) &t->sq, (void *) &t->md, (void *) &t->mq, &hm, &bb, &QWB,
                                    &wst, &QSB, &sst, &n, &rows, &hb, &b, &g_t3swap };
                    t5_launch(kb, (unsigned) rows, (unsigned) nb, c5);
                    r_fb++;
                    return;
                }
                // Esther 4:14 -- "for such a time as this": a short prompt uses the
                // streamed copy only if it has already arrived; otherwise it re-makes
                // the tile in passing (tcm5) rather than wait on the bus. A copy on a
                // 3.5 ms transfer feeding a 0.7 ms matmul is not worth waiting for.
                if (ruth_on() && nb <= g_tcm4_maxnb && g_tcm5 && CU.EventQuery && getenv("SML_RUTH_EARLY")
                    && CU.EventQuery(j_ready[slot]) != CU_OK) {
                    void * c5[] = { &y, &ys, (void *) &t->c2, (void *) &t->c4, &six, (void *) &t->sd,
                                    (void *) &t->sq, (void *) &t->md, (void *) &t->mq, &hm, &bb, &QWB,
                                    &wst, &QSB, &sst, &n, &rows, &hb, &b, &g_t3swap };
                    t5_launch(kb, (unsigned) rows, (unsigned) nb, c5);
                    CU.EventRecord(j_free[slot], NULL);
                    r_fb++;
                    return;
                }
                // this matmul waits only for its own buffer
                r_used++;
                CU.StreamWaitEvent(NULL, j_ready[slot], 0);
                w8 = JW8[slot]; s8 = JS8[slot]; fresh = 0;
                tcm3_launch(kb, y, ys, w8, s8, bb, wst, sst, n, rows, hb, nb, nsb);
                CU.EventRecord(j_free[slot], NULL);
                return;
            }
        }
        if (!(fresh && w8 == W8 && nb < g_rq_minnb)) {
            if (fresh) {
                int nn = n, rr = rows, hmm_ = hm, sx = six;
                void * q[] = { (void *) &t->c2, (void *) &t->c4, &sx, (void *) &t->sd, (void *) &t->sq,
                               (void *) &t->md, (void *) &t->mq, &hmm_, &w8, &s8, &nn, &rr };
                launch2(KB_SMV6, KF_REQUANT8, cdiv(rows * nsb, 8), 1, 256, q);
                // only a tensor that is NOT kept needs a host copy
                if (ruth_on() && w8 == W8 && !j_learned && r_n < 512 && ruth_find(t) < 0) {
                    const size_t wb = (size_t) rows * n, sbz = (size_t) rows * nsb * 4;
                    void * hw = NULL, * hs = NULL;
                    if (CU.MemAllocHost(&hw, wb) == CU_OK && CU.MemAllocHost(&hs, sbz) == CU_OK) {
                        CU.MemcpyDtoH(hw, w8, wb);        // synchronous: the re-make is done
                        CU.MemcpyDtoH(hs, s8, sbz);
                        r_host[r_n].t = t; r_host[r_n].hw = hw; r_host[r_n].hs = hs;
                        r_host[r_n].wb = wb; r_host[r_n].sb = sbz;
                        r_n++;
                    } else {
                        if (hw) CU.MemFreeHost(hw);
                        g_ruth = 0;                        // no pinned memory: re-make as before
                    }
                }
            }
            // Split a deep sum (the down projection: 32 superblocks) into slices of
            // about eight, so the GPU is not left waiting on a last wave of long blocks.
            tcm3_launch(kb, y, ys, w8, s8, bb, wst, sst, n, rows, hb, nb, nsb);
            return;
        }
    }
    if (g_hmm && g_xsrc && !six && n % HM_KC_HOST == 0) {
        int xhst = wide / 2 + 16;
        if (g_hperm != t->perm) {
            CUdeviceptr xsrc = g_xsrc;
            int xst = g_xst, hs = xhst, nn = n;
            void * p[] = { &xsrc, &xst, (void *) &t->perm, &XHB, &hs, &nn };
            launch2(kb, KF_PERMH, cdiv(n / 2, 256), (unsigned) nb, 256, p);
            g_hperm = t->perm;
        }
        void * h[] = { &y, &ys, (void *) &t->c4, (void *) &t->sd, (void *) &t->sq,
                       (void *) &t->md, (void *) &t->mq, &bb, &XHB, &xhst,
                       &n, &rows, &hm, &hb, &b };
        launch2(kb, KF_HMM, cdiv(rows, 64), cdiv(nb, 64), 128, h);
        return;
    }
    // tokens per tcm2 block: must equal the TC2_BN the PTX was built with
    static int bn = 0;
    if (!bn) bn = getenv("SML_TC2_BN") ? atoi(getenv("SML_TC2_BN")) : 32;
    if (g_tcm && g_gsb) launch2(kb, KF_TCM2, cdiv(rows, 64), cdiv(nb, bn), 128, a);
    else if (g_tcm)      launch2(kb, KF_TCM, cdiv(rows, 64), 1, 128, a);
    else if (g_tmm && nb <= 16) launch2(kb, KF_TMM, cdiv(rows, 64), 1, 128, a);
    else       launch2(kb, KF_MVB, cdiv(rows, 4), 1, 128, a);
}

static void normq2b(CUdeviceptr w, CUdeviceptr bias, const gt * t, int mode, int nb) {
    const int wide = g_wide;
    int xs = n_embd, ys = wide, os = wide, wst = wide / 4 + 16, sst = wide / SML_GROUP + 16;
    int nn = n_embd, ln = w ? layer_norm : 2, hb = bias ? 1 : 0, md = mode, sbq = g_tcm && g_gsb;
    float e = eps;
    if (!w) w = XBAT;       // no norm: never read
    CUdeviceptr bb = bias ? bias : w;
    void * a[] = { &XBAT, &xs, &HB2B, &ys, &XBB, &os, &w, &bb, (void *) &t->perm,
                   &QWB, &wst, &QSB, &QMB, &sst, &nn, &e, &ln, &hb, &md, &sbq };
    launch2(KB_NORMQ, KF_NORMQ2B, (unsigned) nb, 1, 256, a);
    g_xsrc = XBB; g_xst = wide; g_hperm = 0;
}

static void quantw2b(CUdeviceptr x, const gt * t, int nb) {
    const int wide = g_wide;
    int xs = wide, wst = wide / 4 + 16, sst = wide / SML_GROUP + 16, n = t->n;
    void * a[] = { &x, &xs, (void *) &t->perm, &QWB, &wst, &QSB, &QMB, &sst, &n };
    g_xsrc = x; g_xst = wide; g_hperm = 0;
    if (g_tcm && g_gsb)
        launch2(KB_QUANTW, KF_QUANTSBB, cdiv((t->n / SML_GROUP + 7) / 8, 8), (unsigned) nb, 256, a);
    else
        launch2(KB_QUANTW, KF_QUANTW2B, cdiv(t->n / SML_GROUP, 8), (unsigned) nb, 256, a);
}

// Experts on the batched prompt: the router's logits for every token (one
// launch), back to the host once per layer, each token routed by the CPU's own
// sml_moe_route, then every chosen expert reads its weights ONCE for all the
// tokens that chose it (rows gathered) through the same tensor-core matmuls as
// a dense FFN, and adds its outputs back into those tokens' rows, weighted.
// XBB holds the normed rows; the result lands in HB2B.
static void moe_prefill(const gl * y, const sml_layer * s, int nb) {
    const gmoe * g = y->moe;
    const int wide = g_wide, ne = g->ne, K = SML_MOE_MAX_K;
    {
        int yst = ne + 1, xst = wide, rows = ne, cols = n_embd;
        void * a[] = { &MOE_LGB, &yst, (void *) &g->router, &XBB, &xst, &rows, &cols };
        launch2(KB_FFN, KF_F32MVB, (unsigned) ne, (unsigned) nb, 256, a);
        if (g->shgate) {
            CUdeviceptr y1 = MOE_LGB + (CUdeviceptr) ne * 4;
            int r1 = 1;
            void * b1[] = { &y1, &yst, (void *) &g->shgate, &XBB, &xst, &r1, &cols };
            launch2(KB_FFN, KF_F32MVB, 1, (unsigned) nb, 256, b1);
        }
    }
    CU.MemcpyDtoH(moe_lgb_h, MOE_LGB, (size_t) nb * (ne + 1) * 4);
    static int sel[PB][SML_MOE_MAX_K], kk[PB];
    static float wsel[PB][SML_MOE_MAX_K];
    for (int b = 0; b < nb; b++)
        kk[b] = sml_moe_route(M.hdr, s, moe_lgb_h + (size_t) b * (ne + 1), moe_sc_h, moe_ch_h, sel[b], wsel[b]);
    // every expert's token list, one upload
    static int idx_h[PB * (SML_MOE_MAX_K + 1)], off[257], cnt[256];
    static float w_h[PB * (SML_MOE_MAX_K + 1)];
    int tot = 0;
    for (int e = 0; e < ne; e++) {
        off[e] = tot;
        for (int b = 0; b < nb; b++)
            for (int j = 0; j < kk[b]; j++)
                if (sel[b][j] == e) { idx_h[tot] = b; w_h[tot] = wsel[b][j]; tot++; }
        cnt[e] = tot - off[e];
    }
    const int sh_off = tot;
    if (g->shg.n) {
        for (int b = 0; b < nb; b++) {
            idx_h[tot] = b;
            w_h[tot] = g->shgate ? 1.0f / (1.0f + expf(-moe_lgb_h[(size_t) b * (ne + 1) + ne])) : 1.0f;
            tot++;
        }
    }
    (void) K;
    CU.MemcpyHtoD(MIDX, idx_h, (size_t) tot * 4);
    CU.MemcpyHtoD(MWT, w_h, (size_t) tot * 4);
    {
        int xs = wide, nn = n_embd, b = nb;
        void * a[] = { &HB2B, &xs, &nn, &b };
        launch2(KB_FFN, KF_ZEROB, cdiv(n_embd, 256), (unsigned) nb, 256, a);
    }
    for (int e = 0; e <= ne; e++) {
        const int shared = e == ne;
        if (shared && !g->shg.n) break;
        const int m = shared ? nb : cnt[e];
        if (!m) continue;
        const gt * tg = shared ? &g->shg : &g->eg[e];
        const gt * tu = shared ? &g->shu : &g->eu[e];
        const gt * td = shared ? &g->shd : &g->ed[e];
        CUdeviceptr ix = MIDX + (CUdeviceptr) (shared ? sh_off : off[e]) * 4;
        CUdeviceptr wx = MWT + (CUdeviceptr) (shared ? sh_off : off[e]) * 4;
        {
            int ds = wide, ss = wide, mm = m, nn = n_embd;
            void * a[] = { &MXG, &ds, &XBB, &ss, &ix, &mm, &nn };
            launch2(KB_FFN, KF_GATHERB, cdiv(n_embd, 256), (unsigned) m, 256, a);
        }
        quantw2b(MXG, tg, m);
        if (!same_perm(tg, tu)) {
            mvb(KB_FFN, tg, GBB, wide, 0, m);
            quantw2b(MXG, tu, m);
            mvb(KB_FFN, tu, HBB, wide, 0, m);
        } else {
            mvb(KB_FFN, tg, GBB, wide, 0, m);
            mvb(KB_FFN, tu, HBB, wide, 0, m);
        }
        {
            int hst2 = wide, rr = tg->rows, bb = m, hgg = 1;
            void * aa[] = { &HBB, &GBB, &hst2, &rr, &bb, &hgg };
            launch2(KB_FFN, KF_ACTB, cdiv(tg->rows, 256), (unsigned) m, 256, aa);
        }
        quantw2b(HBB, td, m);
        mvb(KB_QDMV, td, MXO, wide, 0, m);
        {
            int ds = wide, ss = wide, mm = m, nn = n_embd;
            void * a[] = { &HB2B, &ds, &MXO, &ss, &ix, &wx, &mm, &nn };
            launch2(KB_FFN, KF_SCATTERB, cdiv(n_embd, 256), (unsigned) m, 256, a);
        }
    }
}

// Read a prompt on the GPU: PB tokens per weight read, then the final norm and
// the head for the last token only. Leaves LOG holding the next-token logits
// and the KV cache filled for positions 0..n-1.
// One CPU layer's tensor into the staging tensor's buffers: what up_tensor uploads, copied
// over. Only a tensor of the staging one's shape and kind fits; stream_setup checks.
static int tensor_fits(const gt * d, const sml_rec * r) {
    return r && r->layout == SML_LAYOUT_FLAT && (int) r->rows == d->rows && (int) r->cols == d->n;
}
static void refill(const gt * d, const sml_rec * r, int ti) {
    const size_t rows = (size_t) d->rows, n = (size_t) d->n;
    const size_t ng = n / SML_GROUP, nscale = (size_t) SML_NSCALE(d->n);
    if (d->six) {
        CU.MemcpyHtoD(d->c2, sml_ptr(&M, r->codes_off[0]), rows * (n / 128) * 32);
        CU.MemcpyHtoD(d->c4, sml_ptr(&M, r->codes_off[1]), rows * n / 2);
    } else {
        CU.MemcpyHtoD(d->c4, sml_ptr(&M, r->codes_off[0]), rows * n / 2);
    }
    CU.MemcpyHtoD(d->sd, sml_ptr(&M, r->sd_off), rows * nscale * 2);
    if (sml_kernel_scale6_get()) {
        if (!g_s8h[ti]) {           // re-biased once, as up_tensor does it
            const uint8_t * s6 = (const uint8_t *) sml_ptr(&M, r->sq_off);
            const uint8_t * m6 = d->have_min ? (const uint8_t *) sml_ptr(&M, r->mq_off) : NULL;
            const size_t g6 = (size_t) SML_SCALE6_BYTES(ng);
            g_s8h[ti] = malloc(rows * ng);
            if (m6) g_m8h[ti] = malloc(rows * ng);
            for (size_t row = 0; row < rows; row++)
                for (size_t g = 0; g < ng; g++) {
                    g_s8h[ti][row * ng + g] = (uint8_t) sml_scale6_at(s6 + row * g6, (int) g);
                    if (m6) g_m8h[ti][row * ng + g] = (uint8_t) (sml_scale6_at(m6 + row * g6, (int) g)
                                                               + (sml_kernel_minneg_get() ? 64 : 96));
                }
        }
        CU.MemcpyHtoD(d->sq, g_s8h[ti], rows * ng);
        if (d->have_min) {
            CU.MemcpyHtoD(d->md, sml_ptr(&M, r->md_off), rows * nscale * 2);
            CU.MemcpyHtoD(d->mq, g_m8h[ti], rows * ng);
        }
    } else {
        CU.MemcpyHtoD(d->sq, sml_ptr(&M, r->sq_off), rows * ng);
        if (d->have_min) {
            CU.MemcpyHtoD(d->md, sml_ptr(&M, r->md_off), rows * nscale * 2);
            CU.MemcpyHtoD(d->mq, sml_ptr(&M, r->mq_off), rows * ng);
        }
    }
    CU.MemcpyHtoD(d->perm, sml_ptr(&M, r->perm_off), n * 4);
}

// CPU layer l onto the card for a chunk at position p: its tensors, and its cache rows 0..p-1
static const gl * stream_in(int l, int p) {
    const sml_layer * s = &C.layers[l];
    const sml_rec * rs[7] = { s->q, s->k, s->v, s->o, s->gate, s->up, s->down };
    const gl * g = &LS[l - NL];
    const gt * ds[7] = { &g->q, &g->k, &g->v, &g->o, &g->gate, &g->up, &g->down };
    CU.CtxSynchronize();            // the last layer's kernels are done with the buffers
    const double t0 = now_s();
    if (l == NL) g_st_card += t0 - g_st_t;  // the card layers' share of the chunk
    else g_st_run += t0 - g_st_t;
    for (int i = 0; i < 7; i++) if (rs[i]) refill(ds[i], rs[i], (l - NL) * 7 + i);
    g_st_copy += now_s() - t0;
    g_st_t = now_s();
    g_hperm = 0;
    if (p > 0) {
        const size_t rows = (size_t) p * kvd;
        const uint16_t * kh = CS.k_cache + (size_t) l * CS.n_ctx * kvd;
        const uint16_t * vh = CS.v_cache + (size_t) l * CS.n_ctx * kvd;
        for (size_t i = 0; i < rows; i++) g_kv_h[i] = sml_f16_to_f32(kh[i]);
        CU.MemcpyHtoD(KS, g_kv_h, rows * 4);
        for (size_t i = 0; i < rows; i++) g_kv_h[i] = sml_f16_to_f32(vh[i]);
        CU.MemcpyHtoD(VS, g_kv_h, rows * 4);
    }
    g_in_stream = 1;
    return &LS[l - NL];
}

// fp32 -> fp16, round to nearest even: what the CPU engine stores its cache as (sml-run.c)
static uint16_t kv_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u, rawe = (x >> 23) & 0xffu;
    uint32_t man = x & 0x7fffffu;
    if (rawe == 0xff) return (uint16_t) (sign | 0x7c00u | (man ? 0x200u : 0u));
    const int32_t exp = (int32_t) rawe - 127 + 15;
    if (exp >= 31) return (uint16_t) (sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        man |= 0x800000u;
        const uint32_t sh = (uint32_t) (14 - exp);
        uint32_t sub = man >> sh;
        const uint32_t rem = man & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (sub & 1u))) sub++;
        return (uint16_t) (sign | sub);
    }
    uint32_t h = ((uint32_t) exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t) (sign | h);
}

// ...and its new cache rows p..p+nb-1 back to the CPU's cache, where writing reads them
static void stream_out(int l, int p, int nb) {
    const size_t off = (size_t) p * kvd, rows = (size_t) nb * kvd;
    uint16_t * kh = CS.k_cache + (size_t) l * CS.n_ctx * kvd + off;
    uint16_t * vh = CS.v_cache + (size_t) l * CS.n_ctx * kvd + off;
    CU.MemcpyDtoH(g_kv_h, KS + (CUdeviceptr) (off * 4), rows * 4);
    for (size_t i = 0; i < rows; i++) kh[i] = kv_f16(g_kv_h[i]);
    CU.MemcpyDtoH(g_kv_h, VS + (CUdeviceptr) (off * 4), rows * 4);
    for (size_t i = 0; i < rows; i++) vh[i] = kv_f16(g_kv_h[i]);
}

static void gpu_prefill(const int * ids, int pos0, int n) {
    const int wide = g_wide;
    const double t_pf = now_s();
    g_st_t = t_pf;
    if (ruth_on() && j_learned) {
        const char * fx = getenv("SML_RUTH_FRAC");
        if (g_frac_fixed >= 0.0) g_ruth_f = g_frac_fixed;
        else if (fx) g_ruth_f = atof(fx);
        else if (g_ruth_phase == 1) g_ruth_f = 0.0;
        else if (g_ruth_phase == 2 && g_ruth_bw > 0.0 && g_ruth_bytes > 0) {
            const double f = 0.9 * g_ruth_rate * (double) n / ((double) g_ruth_bytes / g_ruth_bw);
            g_ruth_f = f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f);
        }
    }
    int last_nb = 1;
    const int streaming = g_cpu_split && g_stream_ok && !(g_mid_fn && g_mid_n > 0);
    for (int p0 = 0; p0 < n; p0 += PB) {
        if (streaming && g_st_t > 0.0) { CU.CtxSynchronize(); if (p0 > 0) g_st_run += now_s() - g_st_t; g_st_t = now_s(); }
        const int nb = n - p0 < PB ? n - p0 : PB;
        last_nb = nb;
        CU.MemcpyHtoD(DTOKS, ids + p0, (size_t) nb * 4);
        if (g_cpu_split) {
            // the embedding stays on the host (it is read a row at a time): its
            // rows go up with the chunk
            for (int b = 0; b < nb; b++) host_embed(ids[p0 + b], pos0 + p0 + b, g_rows_h + (size_t) (p0 + b) * n_embd);
            CU.MemcpyHtoD(XBAT, g_rows_h + (size_t) p0 * n_embd, (size_t) nb * n_embd * 4);
            embed_norm(XBAT, nb);
        } else {
            int xs = n_embd, nn = n_embd, six = EMB.six, hm = EMB.have_min;
            void * a[] = { &XBAT, &xs, (void *) &EMB.c2, (void *) &EMB.c4, (void *) &EMB.sd,
                           (void *) &EMB.sq, (void *) &EMB.md, (void *) &EMB.mq, (void *) &EMB.perm,
                           &DTOKS, &nn, &six, &hm };
            launch2(KB_EMBED, KF_EMBEDB, cdiv(n_embd, 256), (unsigned) nb, 256, a);
            embed_out(XBAT, nb, 0, pos0 + p0);
        }
        const int nlp = streaming ? NL_ALL : NL;
        for (int l = 0; l < nlp; l++) {
            const gl * y = l < NL ? &L[l] : stream_in(l, pos0 + p0);
            normq2b(y->an, y->anb, &y->q, l == 0 ? 0 : 1, nb);
            mvb(KB_MV3, &y->q, QBB, n_head * hd, y->qb, nb);
            mvb(KB_MV3, &y->k, KBB, kvd, y->kb, nb);
            mvb(KB_MV3, &y->v, VBB, kvd, y->vb, nb);
            clamp_qkv(QBB, n_head * hd, KBB, kvd, VBB, kvd, nb);
            if (!g_qk_late) qknorm(y, QBB, n_head * hd, KBB, kvd, nb);
            {
                CUdeviceptr kl = l < NL ? KC + (CUdeviceptr)((size_t) l * n_ctx * kvd * 4) : KS;
                CUdeviceptr vl = l < NL ? VC + (CUdeviceptr)((size_t) l * n_ctx * kvd * 4) : VS;
                int qst = n_head * hd, kst = kvd, vst = kvd;
                int nh = n_head, nk = n_kv, hh = hd, hf = l_half(l), il = interleaved, p = pos0 + p0;
                CUdeviceptr rc = l_rc(l), rs = l_rs(l);
                void * a[] = { &QBB, &qst, &KBB, &kst, &VBB, &vst, &kl, &vl, &rc, &rs,
                               &nh, &nk, &hh, &hf, &il, &p };
                launch2(KB_ROPEKV, KF_ROPEKVB, (unsigned) nb, 1, 256, a);
                if (g_qk_late) {
                    // HunYuan: the ROTATED q and k are normed, and k goes into the
                    // cache again (the same kernel with no rotation left to do)
                    qknorm(y, QBB, n_head * hd, KBB, kvd, nb);
                    hf = 0;
                    launch2(KB_ROPEKV, KF_ROPEKVB, (unsigned) nb, 1, 256, a);
                }
                int ost = wide, nc = n_ctx, sw = l_win(l);
                float iv = inv;
                void * b[] = { &XBB, &ost, &QBB, &qst, &kl, &vl, &ATTB,
                               &nh, &nk, &hh, &nc, &sw, &iv, &p };
                if (g_fatc && (hd == 64 || hd == 128) && !sw && !g_score_adj) {
                    const int rep = n_head / n_kv;
                    int nbb = nb;
                    void * f[] = { &XBB, &ost, &QBB, &qst, &kl, &vl, &nh, &nk, &iv, &p, &nbb };
                    launch2(KB_ATTN, hd == 64 ? KF_FATC : KF_FATC128, (unsigned) n_kv,
                            cdiv(nb * rep, 64), 128, f);
                } else if (g_fattn) {
                    const int rep = n_head / n_kv;
                    int nbb = nb;
                    void * f[] = { &XBB, &ost, &QBB, &qst, &kl, &vl,
                                   &nh, &nk, &hh, &nc, &sw, &iv, &p, &nbb };
                    launch2(KB_ATTN, KF_FATTN, (unsigned) n_kv, cdiv(nb * rep, 16), 128, f);
                } else {
                    launch2(KB_ATTN, KF_ATTNB, (unsigned) n_head, (unsigned) nb, 256, b);
                }
            }
            quantw2b(XBB, &y->o, nb);
            mvb(KB_QDMV, &y->o, HB2B, wide, y->ob, nb);
            branch_out(HB2B, wide, y->pa, nb);
            if (y->moe) {
                normq2b(y->fn, y->fnb, &y->moe->eg[0], par ? 2 : 1, nb);
                moe_prefill(y, &C.layers[l], nb);
                branch_out(HB2B, wide, y->pf, nb);
                continue;
            }
            normq2b(y->fn, y->fnb, y->gate.n ? &y->gate : &y->up, par ? 2 : 1, nb);
            {
                const int hg = y->gate.n ? 1 : 0;
                const gt * g = hg ? &y->gate : &y->up;
                CUdeviceptr gbb = hg && y->gb ? y->gb : QSB, ubb = y->ub ? y->ub : QSB;
                int hst = wide, nn = y->up.n, rows = y->up.rows, has_gate = hg, b = nb;
                int wst = wide / 4 + 16, sst = wide / SML_GROUP + 16;
                int gs = g->six, gm = g->have_min, gh = hg && y->gb ? 1 : 0;
                int us = y->up.six, um = y->up.have_min, uh = y->ub ? 1 : 0;
                void * a[] = { &HBB, &hst,
                    (void *) &g->c2, (void *) &g->c4, (void *) &g->sd, (void *) &g->sq,
                    (void *) &g->md, (void *) &g->mq, &gbb,
                    (void *) &y->up.c2, (void *) &y->up.c4, (void *) &y->up.sd, (void *) &y->up.sq,
                    (void *) &y->up.md, (void *) &y->up.mq, &ubb,
                    &QWB, &wst, &QSB, &QMB, &sst, &nn, &rows, &has_gate,
                    &gs, &gm, &gh, &us, &um, &uh, &b };
                if (g_tmm || g_tcm) {
                    // gate and up through the tiled kernel, then the activation
                    if (hg) mvb(KB_FFN, &y->gate, GBB, wide, y->gb, nb);
                    mvb(KB_FFN, &y->up, HBB, wide, y->ub, nb);
                    int hst2 = wide, rr = rows, bb = nb, hgg = hg;
                    void * aa[] = { &HBB, &GBB, &hst2, &rr, &bb, &hgg };
                    launch2(KB_FFN, KF_ACTB, cdiv(rows, 256), (unsigned) nb, 256, aa);
                } else {
                    launch2(KB_FFN, KF_FFNB, cdiv(rows, 4), 1, 128, a);
                }
            }
            quantw2b(HBB, &y->down, nb);
            mvb(KB_QDMV, &y->down, HB2B, wide, y->db, nb);
            branch_out(HB2B, wide, y->pf, nb);
            if (l >= NL) { stream_out(l, pos0 + p0, nb); g_in_stream = 0; }
        }
        if (g_cpu_split) {
            // the chunk's last GPU branch joins its rows, and the rows cross
            addrow(XBAT, n_embd, n_embd, nb, HB2B, wide, 0, 0);
            CU.MemcpyDtoH(g_rows_h + (size_t) p0 * n_embd, XBAT, (size_t) nb * n_embd * 4);
        }
        // one whole chunk has shown the granary the order of the unkept tensors
        if (!j_learned && j_n > 0) {
            j_learned = 1; j_pos = 0;
            if (ruth_on()) {
                // what streaming everything would move, and how fast this bus moves it
                g_ruth_bytes = 0;
                for (int i = 0; i < j_n; i++) {
                    const int h = ruth_find(j_order[i]);
                    if (h >= 0) g_ruth_bytes += r_host[h].wb + r_host[h].sb;
                }
                if (r_n > 0) {
                    CU.CtxSynchronize();
                    const double tb = now_s();
                    CU.MemcpyHtoD(JW8[0], r_host[0].hw, r_host[0].wb);
                    CU.CtxSynchronize();
                    const double dt = now_s() - tb;
                    if (dt > 0.0) g_ruth_bw = (double) r_host[0].wb / dt;
                    j_slot_t[0] = NULL;          // the ring slot no longer holds its tensor
                }
                g_ruth_phase = 1;
            }
        } else if (ruth_on() && j_learned && g_ruth_phase == 1 && !getenv("SML_RUTH_FRAC")) {
            CU.CtxSynchronize();
            g_ruth_rate = (now_s() - t_pf) / (double) n;   // compute per token, nothing streamed
            g_ruth_phase = 2;
        }
    }
    if (g_cpu_split) {
        // a middle stage (another GPU, sml_gpu_set_middle) runs the next layers on
        // the hidden rows; the CPU continues after them
        if (streaming) {
            // every layer ran here: the CPU only norms the last row and reads the head
            g_split_lg = sml_forward_from(&CS, g_rows_h + (size_t) (n - 1) * n_embd, pos0 + n - 1, NL_ALL, 0);
            if (getenv("SML_STREAM_STATS"))
                printf("  stream: %d tokens in %.2f s: bus %.2f s, card layers %.2f s, streamed layers %.2f s\n",
                       n, now_s() - t_pf, g_st_copy, g_st_card, g_st_run), fflush(stdout);
            g_st_copy = g_st_run = g_st_card = 0.0;
            return;
        }
        int L0 = NL;
        if (g_mid_fn && g_mid_n > 0) {
            if (!g_mid_fn(g_rows_h, pos0, n)) GPU_DIE();
            L0 += g_mid_n;
        }
        g_split_lg = sml_prefill_from(&CS, g_rows_h, pos0, n, L0, 0);
        return;
    }
    // the last token alone: its residual add, the final norm, the head
    {
        CUdeviceptr xl = XBAT + (CUdeviceptr)((size_t)(last_nb - 1) * n_embd * 4);
        CUdeviceptr yl = HB2B + (CUdeviceptr)((size_t)(last_nb - 1) * wide * 4);
        int nn = n_embd, ln = layer_norm, hb = ONB ? 1 : 0, md = 1;
        float e = eps;
        CUdeviceptr bb = ONB ? ONB : ON;
        void * a[] = { &xl, &yl, &XB, &ON, &bb, (void *) &HEAD.perm, &QW, &QS, &QM, &nn, &e, &ln, &hb, &md };
        launch(KB_NORMQ, KF_NORMQ2, 1, a);
        f2_mv3(KB_HEAD3, &HEAD, LOG, 0, NULL, 0, 0, NULL, 0, 0);
        logits_out(LOG);
    }
}

// Graphs need every layer on the fused fast path and a driver with the calls.
static int graph_ready(void) {
    if (g_cpu_split) return 0;          // the CPU half runs between tokens
    if (!g_graph || !g_fuse2 || !g_fast || !CU.GraphLaunch || !CU.StreamCreate
        || !CU.StreamBeginCapture || !CU.StreamEndCapture || !CU.GraphInstantiateWithFlags)
        return 0;
    for (int l = 0; l < NL; l++) if (L[l].moe && !L[l].moe->dev) return 0;   // a graph cannot wait on the host's choice
    for (int l = 0; l < NL; l++)
        if (!(L[l].kv_same && (!L[l].gate.n || L[l].gu_same))) return 0;
    return 1;
}

static void gpu_forward(int tok, int pos);
// The prompt through the batched path, or one token at a time through decode's.
static void pp_run(int batched, const int * ids, int n) {
    if (batched) { gpu_prefill(ids, 0, n); return; }
    for (int p = 0; p < n; p++) gpu_forward(ids[p], p);
}

// Token tok's embedding row at position pos, on the host: the lookup, the
// scale, the learned position (the embedding norm runs on the card).
static void host_embed(int tok, int pos, float * out) {
    const sml_rec * e = C.emb;
    const uint16_t * esd = (const uint16_t *) sml_ptr(&M, e->sd_off);
    const uint8_t  * esq = (const uint8_t  *) sml_ptr(&M, e->sq_off);
    const uint16_t * emd = e->mq_n ? (const uint16_t *) sml_ptr(&M, e->md_off) : NULL;
    const uint8_t  * emq = e->mq_n ? (const uint8_t  *) sml_ptr(&M, e->mq_off) : NULL;
    if (e->nbits == 6)
        sml_dequant_row_flat6(rowb, sml_ptr(&M, e->codes_off[1]), sml_ptr(&M, e->codes_off[0]),
                              esd, esq, emd, emq, tok, n_embd);
    else
        sml_dequant_row_flat4(rowb, sml_ptr(&M, e->codes_off[0]), esd, esq, emd, emq, tok, n_embd);
    const int32_t * ep = (const int32_t *) sml_ptr(&M, e->perm_off);
    for (int i = 0; i < n_embd; i++) out[ep[i]] = rowb[i];
    if (g_emb_scale != 0.0f) for (int i = 0; i < n_embd; i++) out[i] *= g_emb_scale;
    if (C.pos_embd) for (int i = 0; i < n_embd; i++) out[i] += C.pos_embd[(size_t) pos * n_embd + i];
}

static void gpu_forward(int tok, int pos) {
    const sml_rec * e = C.emb;
    const uint16_t * esd = (const uint16_t *) sml_ptr(&M, e->sd_off);
    const uint8_t  * esq = (const uint8_t  *) sml_ptr(&M, e->sq_off);
    const uint16_t * emd = e->mq_n ? (const uint16_t *) sml_ptr(&M, e->md_off) : NULL;
    const uint8_t  * emq = e->mq_n ? (const uint8_t  *) sml_ptr(&M, e->mq_off) : NULL;
    if (e->nbits == 6)
        sml_dequant_row_flat6(rowb, sml_ptr(&M, e->codes_off[1]), sml_ptr(&M, e->codes_off[0]),
                              esd, esq, emd, emq, tok, n_embd);
    else
        sml_dequant_row_flat4(rowb, sml_ptr(&M, e->codes_off[0]), esd, esq, emd, emq, tok, n_embd);
    const int32_t * ep = (const int32_t *) sml_ptr(&M, e->perm_off);
    for (int i = 0; i < n_embd; i++) xhost[ep[i]] = rowb[i];
    if (g_emb_scale != 0.0f) for (int i = 0; i < n_embd; i++) xhost[i] *= g_emb_scale;
    if (C.pos_embd) for (int i = 0; i < n_embd; i++) xhost[i] += C.pos_embd[(size_t) pos * n_embd + i];
    CU.MemcpyHtoD(X, xhost, (size_t) n_embd * 4);
    embed_norm(X, 1);
    if (g_fast) CU.MemcpyHtoD(DPOS, &pos, 4);

    for (int l = 0; l < NL; l++) {
        const gl * y = &L[l];
        if (g_fuse2 && (!y->moe || y->moe->dev) && y->kv_same && (!y->gate.n || y->gu_same)) {
            f2_layer(y, l, pos);
            if (g_sync == 1 && !g_prof) CU.CtxSynchronize();
            continue;
        }
        addnorm(HB2, y->an, y->anb, l == 0 ? 0 : 1);

        if (g_shared) {
            quantw(&y->q, XB);
            mvs(&y->q, Q, y->qb);
            if (!y->kv_same) quantw(&y->k, XB);
            mvs(&y->k, Kv, y->kb);
            if (!y->kv_same) quantw(&y->v, XB);
            mvs(&y->v, V, y->vb);
        } else {
            mvq(&y->q, Q, XB, y->qb);
            mvq(&y->k, Kv, XB, y->kb);
            mvq(&y->v, V, XB, y->vb);
        }
        clamp_qkv(Q, n_head * hd, Kv, kvd, V, kvd, 1);
        if (!g_qk_late) qknorm(y, Q, n_head * hd, Kv, kvd, 1);
        {
            CUdeviceptr kat = KC + (CUdeviceptr)(((size_t) l * n_ctx + pos) * kvd * 4);
            CUdeviceptr vat = VC + (CUdeviceptr)(((size_t) l * n_ctx + pos) * kvd * 4);
            CUdeviceptr cs = l_rc(l) + (CUdeviceptr)((size_t) pos * half * 4);
            CUdeviceptr sn = l_rs(l) + (CUdeviceptr)((size_t) pos * half * 4);
            int nh = n_head, nk = n_kv, h = hd, hf = l_half(l), il = interleaved;
            void * a[] = { &Q, &Kv, &V, &kat, &vat, &cs, &sn, &nh, &nk, &h, &hf, &il };
            launch(KB_ROPEKV, KF_ROPEKV, 1, a);
            if (g_qk_late) { qknorm(y, Q, n_head * hd, Kv, kvd, 1); hf = 0; launch(KB_ROPEKV, KF_ROPEKV, 1, a); }
        }
        {
            CUdeviceptr kl = KC + (CUdeviceptr)((size_t) l * n_ctx * kvd * 4);
            CUdeviceptr vl = VC + (CUdeviceptr)((size_t) l * n_ctx * kvd * 4);
            int nh = n_head, nk = n_kv, h = hd, nc = n_ctx, p = pos;
            const int w = l_win(l);
            int t0 = (w && pos >= w) ? pos - w + 1 : 0;
            float iv = inv;
            void * a[] = { &XB, &Q, &kl, &vl, &ATT, &nh, &nk, &h, &nc, &t0, &p, &iv };
            launch(KB_ATTN, KF_ATTN, 1, a);
        }
        if (g_shared) { quantw(&y->o, XB); mvs(&y->o, HB2, y->ob); }
        else            mvq(&y->o, HB2, XB, y->ob);
        branch_out(HB2, n_embd, y->pa, 1);
        addnorm(HB2, y->fn, y->fnb, par ? 2 : 1);

        if (y->moe) {
            moe_ffn_gpu(y, &C.layers[l]);
        } else if (y->gate.n) {
            if (g_shared) {
                quantw(&y->gate, XB);
                mvs(&y->gate, HB, y->gb);
                if (!y->gu_same) quantw(&y->up, XB);
                mvs(&y->up, HB2, y->ub);
            } else {
                mvq(&y->gate, HB, XB, y->gb);
                mvq(&y->up, HB2, XB, y->ub);
            }
            int nf = n_ff;
            void * a[] = { &HB, &HB2, &nf };
            launch(KB_SILU, KF_SILU, cdiv(n_ff, 256), a);
        } else {
            if (g_shared) { quantw(&y->up, XB); mvs(&y->up, HB, y->ub); }
            else            mvq(&y->up, HB, XB, y->ub);
            int nf = n_ff;
            void * a[] = { &HB, &nf };
            launch(KB_GELU, KF_GELU, cdiv(n_ff, 256), a);
        }
        if (!y->moe) {
            if (g_shared) { quantw(&y->down, HB); mvs(&y->down, HB2, y->db); }
            else            mvq(&y->down, HB2, HB, y->db);
        }
        branch_out(HB2, n_embd, y->pf, 1);
        if (g_sync == 1 && !g_prof) CU.CtxSynchronize();
    }
    if (g_cpu_split) {
        // the last GPU layer's branch joins the stream, and the stream crosses
        addrow(X, n_embd, n_embd, 1, HB2, 0, 0, 0);
        CU.MemcpyDtoH(xhost, X, (size_t) n_embd * 4);
        int L0 = NL;
        if (g_mid_fn && g_mid_n > 0) {
            if (!g_mid_fn(xhost, pos, 1)) GPU_DIE();
            L0 += g_mid_n;
        }
        g_split_lg = sml_forward_from(&CS, xhost, pos, L0, 0);
        return;
    }
    if (g_fuse2) {
        f2_normq(HB2, ON, ONB, &HEAD, 1);
        f2_mv3(KB_HEAD3, &HEAD, LOG, 0, NULL, 0, 0, NULL, 0, 0);
        logits_out(LOG);
        return;
    }
    addnorm(HB2, ON, ONB, 1);
    if (g_shared) { quantw(&HEAD, XB); mvs(&HEAD, LOG, 0); }
    else            mvq(&HEAD, LOG, XB, 0);
    logits_out(LOG);
}

static int g_layers_force = -1;      // sml_gpu_open's gpu_layers
static CUcontext g_cuctx;

static void gpu_options(void) {
    if (getenv("SML_GRID")) g_gcap = (unsigned) atoi(getenv("SML_GRID"));
    if (getenv("SML_SHARED")) g_shared = atoi(getenv("SML_SHARED"));
    if (getenv("SML_DP4A")) g_dp4a = atoi(getenv("SML_DP4A"));
    // 0: async. 1: synchronize after every layer. 2: after every launch.
    // Earlier the async decode ran 54 ms/token where the synced profile of the
    // same work ran 39, so how the host waits is under test, not assumed.
    if (getenv("SML_SYNC")) g_sync = atoi(getenv("SML_SYNC"));
    if (getenv("SML_FUSE2")) g_fuse2 = atoi(getenv("SML_FUSE2"));
    if (getenv("SML_FAST")) g_fast = atoi(getenv("SML_FAST"));
    if (getenv("SML_GRAPH")) g_graph = atoi(getenv("SML_GRAPH"));
    if (getenv("SML_SPLIT")) g_split = atoi(getenv("SML_SPLIT"));
    if (getenv("SML_BLOCK")) g_bsize = (unsigned) atoi(getenv("SML_BLOCK"));
}

// Everything the race program did before its tests: open the model, refuse what
// the GPU cannot run (by name), load the module, upload, size the split.
static int gpu_setup(const char * path, int want_ctx, int threads) {
    if (!sml_model_open(&M, path)) return 1;
    const sml_hdr * h = M.hdr;
    if (h->flags & SML_MODEL_HADAMARD) { puts("  Hadamard models not supported on the GPU yet"); return 1; }

    // SML_CTX sets the context; 256 is enough for decode tests, prompt tests need more.
    n_ctx = want_ctx > 0 ? want_ctx : getenv("SML_CTX") ? atoi(getenv("SML_CTX")) : 256;
    if (n_ctx < 64) n_ctx = 64;
#ifdef SML_GPU_LIB
    sml_set_repack(0);       // the server's GPU context: its CPU side only embeds and holds tables
#endif
    if (!sml_ctx_init(&C, &M, n_ctx, threads)) return 1;
#ifdef SML_GPU_LIB
    sml_set_repack(1);       // the split's CPU half (CS) decodes: it keeps its copy
#endif
    n_embd = (int) h->n_embd; n_head = (int) h->n_head; n_kv = (int) h->n_head_kv;
    // The head width is the model's own (Qwen3 128, Gemma 256), not n_embd / n_head:
    // every buffer and kernel reads hd and n_head * hd, and g_wide holds the widest.
    hd = C.head_dim; kvd = n_kv * hd; n_ff = (int) h->n_ff;
    g_wide = n_ff > n_embd ? n_ff : n_embd;
    if (n_head * hd > g_wide) g_wide = n_head * hd;
    // What has no GPU kernel yet is refused here, by name, rather than answered
    // wrongly. Not the CPU's step_only: that flag also covers what the GPU does
    // run (the stream scales, post-norms), and refused Gemma and Granite wholesale.
    {
        const char * why = NULL;
        if (C.emb_norm && (h->norm_type == 1) != (C.emb_norm_b != NULL)) why = "an embedding norm of this kind";
        else if ((int) C.emb->cols != (int) h->n_embd) why = "padded embedding rows";
        for (uint32_t li = 0; !why && li < h->n_layer; li++) {
            const sml_layer * s = &C.layers[li];
            if (!s->n_exp && s->shg) why = "a shared expert without routed ones";
            else if ((s->n_exp && !s->eg[0]) || (s->shu && !s->shg)) why = "ungated experts";
            else if (s->ssm_in || s->m1_in || s->gdn_qkvz) why = "state-space layers";
            else if (s->sc_in) why = "short convolutions";
            else if (s->attn_gate) why = "an attention output gate";
            else if (s->xielu) why = "the xIELU activation";
            else if (!s->q) why = "a layer without attention";
            else if ((!s->attn_norm || !s->ffn_norm) && h->version >= 5
                     && (h->arch_bits & SML_ARCH_PARALLEL_RESIDUAL)) why = "a parallel layer without input norms";
            else if ((s->attn_post_norm || s->ffn_post_norm) && h->version >= 4 && h->norm_type == 1)
                why = "LayerNorm post-norms";
            else {
                // Q8 (8-bit) tensors have no GPU kernel yet: read as 4-bit they would
                // give wrong answers, so the model stays on the CPU, which runs them
                const sml_rec * ts[7] = { s->q, s->k, s->v, s->o, s->gate, s->up, s->down };
                for (int t = 0; t < 7 && !why; t++)
                    if (ts[t] && ts[t]->layout == SML_LAYOUT_FLAT && ts[t]->nbits != 4 && ts[t]->nbits != 6)
                        why = "tensors at a width other than 4 or 6 bits (Q2, Q3, Q5, Q7, Q8)";
            }
        }
        if (!why) {
            const sml_rec * eh[2] = { C.emb, C.head };
            for (int t = 0; t < 2; t++)
                if (eh[t] && eh[t]->layout == SML_LAYOUT_FLAT && eh[t]->nbits != 4 && eh[t]->nbits != 6)
                    why = "an embedding or head at a width other than 4 or 6 bits";
        }
        if (why) { printf("  %s: not on the GPU yet\n", why); return 1; }
    }
    g_emb_scale = h->emb_scale; g_res_scale = h->res_scale; g_logit_scale = h->logit_scale;
    g_softcap = h->version >= 4 ? h->logit_softcap : 0.0f;
    g_clip = h->clip_qkv;
    g_qk_late = h->version >= 5 && (h->arch_bits & SML_ARCH_QKNORM_LATE);
    if (g_qk_late && !g_fast) { puts("  late Q/K norm: needs the fast path (SML_FAST=1)"); return 1; }
    if (C.pos_embd && n_ctx > C.pos_rows) n_ctx = C.pos_rows;   // no position past the learned table
    // The families added on the CPU on 2026-09-28 (GeGLU, relu^2, a late Q/K
    // norm, attention softcap, Q/K/V clamp, per-layer windows or rotary, an
    // output-head bias) have no GPU kernels yet: refused, not approximated.
    {
        const uint32_t known = SML_ARCH_PARALLEL_RESIDUAL | SML_ARCH_GELU_FFN | SML_ARCH_QKNORM_LATE |
                               SML_ARCH_ALIBI | SML_ARCH_GELU_ERF |
                               SML_ARCH_GELU_GATE | SML_ARCH_RELU2 | SML_ARCH_ALIBI_SCALED |
                               SML_ARCH_RELU;
        if (h->version >= 5 && (h->arch_bits & ~known)) {
            puts("  late Q/K norm: not on the GPU yet");
            return 1;
        }
    }
    n_vocab = (int) h->n_vocab; NL = (int) h->n_layer;
    layer_norm = h->version >= 4 && h->norm_type == 1;
    interleaved = h->version >= 4 && h->rope_type == 1;
    par = h->version >= 5 && (h->arch_bits & SML_ARCH_PARALLEL_RESIDUAL);
    inv = (h->version >= 4 && h->attn_scale != 0.0f) ? h->attn_scale : 1.0f / sqrtf((float) hd);
    swa = h->version >= 4 ? (int) h->n_swa : 0;
    eps = h->rms_eps;
    half = C.rope_half;

    if (!cu_open() || !cu_check(CU.Init(0), "cuInit")) return 1;
    CUdevice dev;
    cu_check(CU.DeviceGet(&dev, 0), "cuDeviceGet");
    g_dev = dev;
    const unsigned ctxflags = getenv("SML_CTXFLAGS") ? (unsigned) atoi(getenv("SML_CTXFLAGS")) : 1u;
    if (!cu_check(CU.CtxCreate(&g_cuctx, ctxflags, dev), "cuCtxCreate")) return 1;
    {
        // The standard kernels stage activations n <= 16384 wide. A wider model loads the
        // wide build (n <= 32768); wider still stays off the GPU instead of overrunning the
        // buffers into garbage (Qwen2.5-7B, FFN 18944, wrote ".$_ .$_" until 2026-10-14).
        const char * ptx = getenv("SML_PTX");
        if (!ptx) {
            if (g_wide > 32768) {
                printf("  inputs %d wide: more than the GPU kernels stage (32768): not on the GPU yet\n", g_wide);
                return 1;
            }
            ptx = g_wide > 16384 ? "sml-gpu-fwd-wide.ptx" : "sml-gpu-fwd.ptx";
            if (g_wide > 16384) printf("  inputs %d wide: the wide GPU kernels (%s)\n", g_wide, ptx);
        }
        FILE * f = fopen(ptx, "rb");
        if (!f) { printf("  missing %s\n", ptx); return 1; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char * p = malloc((size_t) n + 1);
        if (fread(p, 1, (size_t) n, f) != (size_t) n) return 1;
        p[n] = 0; fclose(f);
        CUmodule mod;
        if (!cu_check(CU.ModuleLoadData(&mod, p), "load ptx")) return 1;
        // The file says which GELU the model was trained with; every GPU GELU
        // site reads this one flag (sml_gelu in sml-gpu-fwd.cu).
        if (h->version >= 5 && (h->arch_bits & SML_ARCH_GELU_ERF)) {
            CUdeviceptr gp = 0;
            size_t gsz = 0;
            int one = 1;
            if (!CU.ModuleGetGlobal || CU.ModuleGetGlobal(&gp, &gsz, mod, "sml_gelu_erf") != CU_OK
                || gsz != sizeof one) {
                puts("  this model uses the exact erf GELU and the GPU module cannot be told so");
                return 1;
            }
            CU.MemcpyHtoD(gp, &one, sizeof one);
        }
        // The FFN activation likewise: GeGLU (Gemma) or relu^2 (Nemotron, Persimmon)
        // in every FFN site through sml_gated / sml_ungated.
        const int act = h->version >= 5 && (h->arch_bits & SML_ARCH_GELU_GATE) ? 1
                      : h->version >= 5 && (h->arch_bits & SML_ARCH_RELU2) ? 2
                      : h->version >= 5 && (h->arch_bits & SML_ARCH_RELU) ? 3 : 0;
        if (act) {
            CUdeviceptr ap = 0;
            size_t asz = 0;
            if (!CU.ModuleGetGlobal || CU.ModuleGetGlobal(&ap, &asz, mod, "sml_act") != CU_OK
                || asz != sizeof act) {
                puts("  this model's FFN activation cannot be set in the GPU module");
                return 1;
            }
            CU.MemcpyHtoD(ap, &act, sizeof act);
        }
        // ALiBi's slopes (the CPU's own, Falcon's pre-scaled) and the attention cap
        // go to sml_score through two module globals.
        if (C.alibi) {
            ALIBI = up_f(C.alibi, (size_t) h->n_head);
            CUdeviceptr gp = 0;
            size_t gsz = 0;
            if (!CU.ModuleGetGlobal || CU.ModuleGetGlobal(&gp, &gsz, mod, "sml_alibi") != CU_OK
                || gsz != sizeof ALIBI) { puts("  the GPU module cannot be given ALiBi slopes"); return 1; }
            CU.MemcpyHtoD(gp, &ALIBI, sizeof ALIBI);
        }
        if (h->attn_softcap != 0.0f) {
            float cap = h->attn_softcap;
            CUdeviceptr gp = 0;
            size_t gsz = 0;
            if (!CU.ModuleGetGlobal || CU.ModuleGetGlobal(&gp, &gsz, mod, "sml_attn_cap") != CU_OK
                || gsz != sizeof cap) { puts("  the GPU module cannot be given the attention cap"); return 1; }
            CU.MemcpyHtoD(gp, &cap, sizeof cap);
        }
        g_score_adj = C.alibi != NULL || h->attn_softcap != 0.0f;
        const char * names[KF_N] = { "sml_g_qmv4", "sml_g_qmv6", "sml_g_addnorm", "sml_g_ropekv",
            "sml_g_attn", "sml_g_silu", "sml_g_gelu", "sml_g_argmax",
            "sml_g_quantw", "sml_g_smv4", "sml_g_smv6", "sml_g_dmv4", "sml_g_dmv6",
            "sml_g_normq", "sml_g_mv3", "sml_g_ropeattn", "sml_g_qdmv", "sml_g_ffn",
            "sml_g_normq2", "sml_g_attn2", "sml_g_qdmv2", "sml_g_mv3s", "sml_g_ffns",
            "sml_g_embed", "sml_g_argmax2", "sml_g_quantw2",
            "sml_g_embedb", "sml_g_normq2b", "sml_g_quantw2b", "sml_g_mvb", "sml_g_ffnb",
            "sml_g_ropekvb", "sml_g_attnb", "sml_g_tmm", "sml_g_actb", "sml_g_tcm", "sml_g_fattn",
            "sml_g_tcm2", "sml_g_quantsbb", "sml_g_hmm", "sml_g_permh",
            "sml_g_requant8", "sml_g_tcm3", "sml_g_fattn_tc", "sml_g_splitk_sum", "sml_g_fattn_tc128", "sml_g_tcm4", "sml_g_tcm5", "sml_g_tcm7", "sml_g_tcm8", "sml_g_tcm9", "sml_g_qknorm", "sml_g_scaleb", "sml_g_addrow", "sml_g_capb", "sml_g_clampb", "sml_g_f32mv", "sml_g_axpy", "sml_g_ropeqd",
            "sml_g_moe_route", "sml_g_moe_quant", "sml_g_moe_gu", "sml_g_moe_dn", "sml_g_moe_comb",
            "sml_g_f32mvb", "sml_g_gatherb", "sml_g_scatterb", "sml_g_zerob" };
        for (int i = 0; i < KF_N; i++)
            if (!cu_check(CU.ModuleGetFunction(&KF[i], mod, names[i]), names[i])) return 1;
        // Rooms in the ark: the prompt kernels hold ~35 KB of shared memory per
        // block, and unless the SM is told to favour shared memory over L1 only one
        // block fits -- four warps, too few to hide a load. Carveout 100 = the most
        // shared memory the SM can give (CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT).
        {
            // tcm4 was missing here until 2026-09-25: its ~35 KB ran one block per SM
            static const int big[] = { KF_TCM3, KF_TCM2, KF_FATC, KF_FATC128, KF_TCM4, KF_TCM5, KF_TCM7, KF_TCM8, KF_TCM9 };
            const int carve = getenv("SML_CARVEOUT") ? atoi(getenv("SML_CARVEOUT")) : 100;
            for (int i = 0; i < (int) (sizeof big / sizeof big[0]) && CU.FuncSetAttribute && carve >= 0; i++)
                CU.FuncSetAttribute(KF[big[i]], 9, carve);
            // tcm7: two 16 KB weight buffers + a 16 KB activation tile + 64 scales, above
            // the 48 KB a block gets without asking (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
            if (getenv("SML_TCM7")) g_tcm7 = atoi(getenv("SML_TCM7"));
            if (g_tcm7 && CU.FuncSetAttribute && CU.FuncSetAttribute(KF[KF_TCM7], 8, 49408) == CU_OK) {
                KF[KF_TCM5] = KF[KF_TCM7];
                g_dsm[KF_TCM5] = g_dsm[KF_TCM7] = 49408;
            } else g_tcm7 = 0;
            // tcm8: two 16 KB weight buffers, a 32 KB activation tile, 128 + 2 x 64 scales
            if (getenv("SML_TCM8")) g_tcm8 = atoi(getenv("SML_TCM8"));
            if (g_tcm8 && CU.FuncSetAttribute && CU.FuncSetAttribute(KF[KF_TCM8], 8, 66560) == CU_OK)
                g_dsm[KF_TCM8] = 66560;
            else g_tcm8 = 0;
            // tcm9: two stages of 16 KB weights + 16 KB activations, and 2 x 256 scales
            if (getenv("SML_TCM9")) g_tcm9 = atoi(getenv("SML_TCM9"));
            if (g_tcm9 && CU.FuncSetAttribute && CU.FuncSetAttribute(KF[KF_TCM9], 8, 67584) == CU_OK)
                g_dsm[KF_TCM9] = 67584;
            else g_tcm9 = 0;
            if (CU.FuncGetAttribute && getenv("SML_PP")) {
                int regs = 0, smem = 0;
                CU.FuncGetAttribute(&regs, 4, KF[KF_TCM3]);
                CU.FuncGetAttribute(&smem, 1, KF[KF_TCM3]);
                printf("  tcm3: %d registers/thread, %d B shared/block, carveout %d\n", regs, smem, carve);
                CU.FuncGetAttribute(&regs, 4, KF[KF_TCM4]);
                CU.FuncGetAttribute(&smem, 1, KF[KF_TCM4]);
                printf("  tcm4: %d registers/thread, %d B shared/block\n", regs, smem);
            }
        }
    }

    // SML_PCIE_TEST=1: host-to-device bandwidth, pinned and pageable, then exit.
    // Streaming int8 copies from host RAM (instead of re-making them on the GPU)
    // is only worth it if this bus is fast enough to hide behind the matmuls.
    if (getenv("SML_PCIE_TEST")) {
        const size_t sz = (size_t) 256 << 20;
        void * hp = NULL;
        CUdeviceptr dp = dalloc(sz);
        char * pg = malloc(sz);
        memset(pg, 1, sz);
        if (!CU.MemAllocHost || CU.MemAllocHost(&hp, sz) != CU_OK) { puts("  no pinned memory"); return 1; }
        memset(hp, 1, sz);
        for (int k = 0; k < 2; k++) {
            const void * src = k ? pg : hp;
            double best = 1e9;
            for (int r = 0; r < 5; r++) {
                CU.CtxSynchronize();
                const double t0 = now_s();
                CU.MemcpyHtoD(dp, src, sz);
                CU.CtxSynchronize();
                const double dt = now_s() - t0;
                if (dt < best) best = dt;
            }
            printf("  host->device %s: %.2f GB/s\n", k ? "pageable" : "pinned", sz / best / 1e9);
        }
        return 0;
    }
    const double tu = now_s();
    int shared_qkv = 0, shared_gu = 0, moe_ne_max = 0, moe_ff_max = 0;
    {
        // How many layers the card holds. SML_GPU_LAYERS=k sets it; otherwise
        // every layer, unless the weights would not fit in what the card has
        // free less a reserve for scratch, cache and the head's buffers
        // (SML_SPLIT_RESERVE_MB, 700) -- then the leading layers that fit, and
        // the CPU runs the rest. The embedding never goes up in a split: it is
        // read one row per token.
        NL_ALL = NL;
        size_t per[1024], all = 0;
        for (int l = 0; l < NL && l < 1024; l++) {
            const sml_layer * s = &C.layers[l];
            const sml_rec * rs[8] = { s->q, s->k, s->v, s->o, s->gate, s->up, s->down, NULL };
            per[l] = 0;
            for (int i = 0; i < 7; i++) {
                const sml_rec * r = rs[i];
                if (!r) continue;
                per[l] += (size_t) r->perm_n + r->sd_n + r->sq_n + r->md_n + r->mq_n;
                for (int v = 0; v < SML_MAX_LEVELS; v++) per[l] += (size_t) r->codes_n[v];
            }
            for (int e = 0; e < s->n_exp; e++) {
                const sml_rec * er[3] = { s->eg[e], s->eu[e], s->ed[e] };
                for (int i = 0; i < 3; i++) {
                    per[l] += (size_t) er[i]->perm_n + er[i]->sd_n + er[i]->sq_n + er[i]->md_n + er[i]->mq_n;
                    for (int v = 0; v < SML_MAX_LEVELS; v++) per[l] += (size_t) er[i]->codes_n[v];
                }
            }
            per[l] += (size_t) 2 * n_ctx * kvd * 4;          // its K and V cache
            all += per[l];
        }
        size_t head = 0;
        {
            const sml_rec * r = C.head;
            head = (size_t) r->perm_n + r->sd_n + r->sq_n + r->md_n + r->mq_n;
            for (int v = 0; v < SML_MAX_LEVELS; v++) head += (size_t) r->codes_n[v];
            if (C.emb != C.head) head *= 2;                  // and the embedding table
        }
        size_t fr = nvml_free_bytes(g_dev), tot = 0;
        if (!fr && CU.MemGetInfo) CU.MemGetInfo(&fr, &tot);
        const size_t reserve = (size_t) (getenv("SML_SPLIT_RESERVE_MB") ? atoi(getenv("SML_SPLIT_RESERVE_MB")) : 700) << 20;
        int k = NL;
        if (g_layers_force > 0) k = g_layers_force;
        else if (getenv("SML_GPU_LAYERS")) k = atoi(getenv("SML_GPU_LAYERS"));
        else if (fr && all + head + reserve > fr) {
            // room for the staging layer that reads the CPU's layers' prompts on the card:
            // one layer fewer here, prompts 77 -> 133 tok/s on Qwen2.5-7B, writing unchanged
            size_t stage = 0;
            int dense = !(getenv("SML_STREAM") && !atoi(getenv("SML_STREAM")));
            for (int l = 0; l < NL && l < 1024; l++) {
                if (C.layers[l].n_exp) dense = 0;
                if (per[l] > stage) stage = per[l];
            }
            if (!dense) stage = 0;
            size_t used = 0;
            k = 0;
            while (k < NL && used + per[k] + reserve + stage <= fr) used += per[k++];
        }
        if (k < 1) k = 1;
        if (k < NL) {
            size_t on = 0;
            for (int l = 0; l < k; l++) on += per[l];
            g_cpu_split = 1;
            printf("  split: %d of %d layers on the GPU (%.2f of %.2f GB, %.2f GB free), the rest on the CPU\n",
                   k, NL, on / 1e9, (all + head) / 1e9, fr / 1e9);
            if (!sml_ctx_init(&CS, &M, n_ctx, threads)) return 1;
            g_rows_h = malloc((size_t) (n_ctx > 1024 ? n_ctx : 1024) * n_embd * sizeof(float));
            NL = k;
        }
    }
    L = calloc((size_t) NL, sizeof(gl));
    for (int l = 0; l < NL; l++) {
        const sml_layer * s = &C.layers[l];
        L[l].q = up_tensor(&M, s->q); L[l].k = up_tensor(&M, s->k); L[l].v = up_tensor(&M, s->v);
        L[l].o = up_tensor(&M, s->o);
        L[l].gate = up_tensor(&M, s->gate); L[l].up = up_tensor(&M, s->up);
        L[l].down = up_tensor(&M, s->down);
        L[l].kv_same = same_perm(&L[l].q, &L[l].k) && same_perm(&L[l].q, &L[l].v);
        L[l].gu_same = same_perm(&L[l].gate, &L[l].up);
        shared_qkv += L[l].kv_same; shared_gu += L[l].gu_same;
        L[l].qn = up_f(s->q_norm, (size_t) s->q_norm_n); L[l].qnb = up_f(s->q_norm_b, (size_t) s->q_norm_n);
        L[l].kn = up_f(s->k_norm, (size_t) s->k_norm_n); L[l].knb = up_f(s->k_norm_b, (size_t) s->k_norm_n);
        L[l].qn_n = s->q_norm_n; L[l].kn_n = s->k_norm_n;
        L[l].pa = up_f(s->attn_post_norm, n_embd); L[l].pf = up_f(s->ffn_post_norm, n_embd);
        if (s->n_exp) {
            gmoe * g = calloc(1, sizeof *g);
            g->ne = s->n_exp;
            g->eg = calloc((size_t) g->ne, sizeof(gt));
            g->eu = calloc((size_t) g->ne, sizeof(gt));
            g->ed = calloc((size_t) g->ne, sizeof(gt));
            for (int e = 0; e < g->ne; e++) {
                g->eg[e] = up_tensor(&M, s->eg[e]); g->eu[e] = up_tensor(&M, s->eu[e]);
                g->ed[e] = up_tensor(&M, s->ed[e]);
                if (g->eg[e].rows > moe_ff_max) moe_ff_max = g->eg[e].rows;
            }
            {
                // on-card routing needs every expert's gate and up in ONE column
                // order (the activation is quantized per expert with it), and
                // experts of one width
                int ok = 1;
                for (int e = 0; e < g->ne; e++)
                    ok &= same_perm(&g->eg[e], &g->eu[e]) && g->eg[e].rows == g->eg[0].rows;
                if (ok && g->ne <= 256 && s->n_exp) {
                    host_mt * tg = calloc((size_t) g->ne, sizeof *tg), * tu = calloc((size_t) g->ne, sizeof *tu),
                            * td = calloc((size_t) g->ne, sizeof *td);
                    uint64_t * pg = calloc((size_t) g->ne, 8), * pd = calloc((size_t) g->ne, 8);
                    for (int e = 0; e < g->ne; e++) {
                        const gt * src[3] = { &g->eg[e], &g->eu[e], &g->ed[e] };
                        host_mt * dst[3] = { &tg[e], &tu[e], &td[e] };
                        for (int t = 0; t < 3; t++)
                            *dst[t] = (host_mt) { src[t]->c2, src[t]->c4, src[t]->sd, src[t]->sq,
                                                  src[t]->md, src[t]->mq, src[t]->rows, src[t]->six,
                                                  src[t]->have_min, 0 };
                        pg[e] = g->eg[e].perm; pd[e] = g->ed[e].perm;
                    }
                    g->tg = dup(tg, (size_t) g->ne * sizeof *tg); g->tu = dup(tu, (size_t) g->ne * sizeof *tu);
                    g->td = dup(td, (size_t) g->ne * sizeof *td);
                    g->pg = dup(pg, (size_t) g->ne * 8); g->pd = dup(pd, (size_t) g->ne * 8);
                    g->ebias = up_f(s->exp_bias, (size_t) g->ne);
                    g->fe = g->eg[0].rows;
                    g->dev = getenv("SML_MOE_DEV") ? atoi(getenv("SML_MOE_DEV")) : 1;
                    free(tg); free(tu); free(td); free(pg); free(pd);
                }
            }
            g->router = up_f(s->router, (size_t) g->ne * n_embd);
            g->shgate = up_f(s->shgate, (size_t) n_embd);
            g->shg = up_tensor(&M, s->shg); g->shu = up_tensor(&M, s->shu); g->shd = up_tensor(&M, s->shd);
            if (g->shg.rows > moe_ff_max) moe_ff_max = g->shg.rows;
            if (g->ne > moe_ne_max) moe_ne_max = g->ne;
            L[l].moe = g;
        }
        L[l].qb = up_f(s->q_bias, (size_t) n_head * hd); L[l].kb = up_f(s->k_bias, kvd);
        L[l].vb = up_f(s->v_bias, kvd); L[l].ob = up_f(s->o_bias, n_embd);
        L[l].gb = up_f(s->gate_bias, n_ff); L[l].ub = up_f(s->up_bias, n_ff);
        L[l].db = up_f(s->down_bias, n_embd);
        // a missing input norm uploads as 0: the norm kernels then pass the raw stream
        L[l].an  = s->attn_norm ? up_f((const float *) sml_ptr(&M, s->attn_norm->off), n_embd) : 0;
        L[l].anb = up_f(s->attn_norm_b, n_embd);
        L[l].fn  = s->ffn_norm ? up_f((const float *) sml_ptr(&M, s->ffn_norm->off), n_embd) : 0;
        L[l].fnb = up_f(s->ffn_norm_b, n_embd);
    }
    if (g_cpu_split && NL_ALL > NL) {
        if (getenv("SML_STREAM")) g_pstream = atoi(getenv("SML_STREAM"));
        int ok = g_pstream && kvd > 0;
        for (int l = NL; l < NL_ALL && ok; l++) {
            const sml_layer * s = &C.layers[l];
            ok = !s->n_exp && s->q && s->k && s->v && s->o && s->up && s->down
                 && (!s->gate) == (!C.layers[NL].gate);
        }
        // Staging buffers as big as the biggest of each tensor's parts across the CPU's layers:
        // Sheep gives some layers' tensors 6 bits (a second code plane) and others 4.
        const int nc = NL_ALL - NL;
        size_t sz[7][7];                    // [tensor][c2, c4, sd, sq, md, mq, perm]
        int rows_[7], cols_[7];
        memset(sz, 0, sizeof sz);
        for (int i = 0; i < 7 && ok; i++) {
            rows_[i] = cols_[i] = 0;
            for (int l = NL; l < NL_ALL && ok; l++) {
                const sml_layer * s = &C.layers[l];
                const sml_rec * rs[7] = { s->q, s->k, s->v, s->o, s->gate, s->up, s->down };
                const sml_rec * r = rs[i];
                if (!r) continue;
                if (r->layout != SML_LAYOUT_FLAT || (rows_[i] && ((int) r->rows != rows_[i] || (int) r->cols != cols_[i]))) {
                    ok = 0;
                    break;
                }
                rows_[i] = (int) r->rows; cols_[i] = (int) r->cols;
                const size_t rr = r->rows, n = r->cols, ng = n / SML_GROUP, nscale = (size_t) SML_NSCALE(r->cols);
                const size_t part[7] = { r->nbits == 6 ? rr * (n / 128) * 32 : 0, rr * n / 2, rr * nscale * 2, rr * ng,
                                         r->mq_n ? rr * nscale * 2 : 0, r->mq_n ? rr * ng : 0, n * 4 };
                for (int k = 0; k < 7; k++) if (part[k] > sz[i][k]) sz[i][k] = part[k];
            }
        }
        size_t need = (size_t) 2 * n_ctx * kvd * 4;
        for (int i = 0; i < 7; i++) for (int k = 0; k < 7; k++) need += sz[i][k];
        if (ok) {
            size_t fr = nvml_free_bytes(g_dev), tot = 0;     // as the split measured it
            if (!fr && CU.MemGetInfo) CU.MemGetInfo(&fr, &tot);
            const size_t margin = (size_t) (getenv("SML_STREAM_MARGIN_MB") ? atoi(getenv("SML_STREAM_MARGIN_MB")) : 100) << 20;
            if (fr && fr < need + margin) {
                printf("  prompts: no room to read the CPU layers on the card (%.2f GB free, %.2f needed)\n",
                       fr / 1e9, (need + margin) / 1e9);
                ok = 0;
            }
        }
        if (ok) {
            gt * st[7] = { &SL.q, &SL.k, &SL.v, &SL.o, &SL.gate, &SL.up, &SL.down };
            for (int i = 0; i < 7; i++) {
                memset(st[i], 0, sizeof(gt));
                if (!rows_[i]) continue;
                st[i]->rows = rows_[i]; st[i]->n = cols_[i];
                CUdeviceptr * pp[7] = { &st[i]->c2, &st[i]->c4, &st[i]->sd, &st[i]->sq, &st[i]->md, &st[i]->mq, &st[i]->perm };
                for (int k = 0; k < 7; k++) *pp[k] = sz[i][k] ? dalloc(sz[i][k]) : 0;
            }
            LS = calloc((size_t) nc, sizeof(gl));
            g_s8h = calloc((size_t) nc * 7, sizeof(uint8_t *));
            g_m8h = calloc((size_t) nc * 7, sizeof(uint8_t *));
            for (int l = NL; l < NL_ALL && ok; l++) {
                const sml_layer * s = &C.layers[l];
                gl * g = &LS[l - NL];
                const sml_rec * rs[7] = { s->q, s->k, s->v, s->o, s->gate, s->up, s->down };
                gt * ts[7] = { &g->q, &g->k, &g->v, &g->o, &g->gate, &g->up, &g->down };
                for (int i = 0; i < 7; i++) {
                    *ts[i] = *st[i];
                    if (!rs[i]) continue;
                    ts[i]->six = rs[i]->nbits == 6;
                    ts[i]->have_min = rs[i]->mq_n ? 1 : 0;
                    if (!ts[i]->have_min) { ts[i]->md = ts[i]->sd; ts[i]->mq = ts[i]->sq; }
                    ts[i]->hperm = (const int32_t *) sml_ptr(&M, rs[i]->perm_off);
                }
                if (!ok) break;
                g->kv_same = same_perm(&g->q, &g->k) && same_perm(&g->q, &g->v);
                g->gu_same = same_perm(&g->gate, &g->up);
                g->qn = up_f(s->q_norm, (size_t) s->q_norm_n); g->qnb = up_f(s->q_norm_b, (size_t) s->q_norm_n);
                g->kn = up_f(s->k_norm, (size_t) s->k_norm_n); g->knb = up_f(s->k_norm_b, (size_t) s->k_norm_n);
                g->qn_n = s->q_norm_n; g->kn_n = s->k_norm_n;
                g->pa = up_f(s->attn_post_norm, n_embd); g->pf = up_f(s->ffn_post_norm, n_embd);
                g->qb = up_f(s->q_bias, (size_t) n_head * hd); g->kb = up_f(s->k_bias, kvd);
                g->vb = up_f(s->v_bias, kvd); g->ob = up_f(s->o_bias, n_embd);
                g->gb = up_f(s->gate_bias, n_ff); g->ub = up_f(s->up_bias, n_ff);
                g->db = up_f(s->down_bias, n_embd);
                g->an  = s->attn_norm ? up_f((const float *) sml_ptr(&M, s->attn_norm->off), n_embd) : 0;
                g->anb = up_f(s->attn_norm_b, n_embd);
                g->fn  = s->ffn_norm ? up_f((const float *) sml_ptr(&M, s->ffn_norm->off), n_embd) : 0;
                g->fnb = up_f(s->ffn_norm_b, n_embd);
            }
            if (ok) {
                KS = dalloc((size_t) n_ctx * kvd * 4); VS = dalloc((size_t) n_ctx * kvd * 4);
                g_kv_h = malloc((size_t) n_ctx * kvd * sizeof(float));
                g_stream_ok = g_kv_h != NULL;
                // the granary's side copies compete with the staging layer for the card: 97 vs 133 tok/s
                if (g_stream_ok && !getenv("SML_JOSEPH")) g_joseph = 0;
                if (g_stream_ok)
                    printf("  prompts: the CPU's %d layers are read on the card too, one at a time\n", nc);
            }
        }
    }
    // a split model keeps its head and embedding on the CPU half
    if (!g_cpu_split) {
        HEAD = up_tensor(&M, C.head);
        EMB = C.emb == C.head ? HEAD : up_tensor(&M, C.emb);
    }
    ON  = up_f((const float *) sml_ptr(&M, C.out_norm->off), n_embd);
    ONB = up_f(C.out_norm_b, n_embd);
    const int wide = g_wide;
    X = dalloc((size_t) n_embd * 4); XB = dalloc((size_t) wide * 4);
    Q = dalloc((size_t) n_head * hd * 4); Kv = dalloc((size_t) kvd * 4); V = dalloc((size_t) kvd * 4);
    ATT = dalloc((size_t) n_head * n_ctx * 4);
    HB = dalloc((size_t) wide * 4); HB2 = dalloc((size_t) wide * 4);
    LOG = dalloc((size_t) n_vocab * 4);
    KC = dalloc((size_t) NL * n_ctx * kvd * 4); VC = dalloc((size_t) NL * n_ctx * kvd * 4);
    RC = up_f(C.rope_cos, (size_t) n_ctx * half); RS = up_f(C.rope_sin, (size_t) n_ctx * half);
    RCL = up_f(C.rope_cos_l, (size_t) n_ctx * half); RSL = up_f(C.rope_sin_l, (size_t) n_ctx * half);
    OUTB = up_f(C.out_b, (size_t) n_vocab);
    POSE = up_f(C.pos_embd, (size_t) C.pos_rows * n_embd);
    if (moe_ne_max) {
        const int K = SML_MOE_MAX_K;
        MOE_SEL = dalloc((size_t) K * 4); MOE_W = dalloc((size_t) K * 4);
        MOE_SC = dalloc((size_t) (moe_ne_max + 1) * 4); MOE_CH = dalloc((size_t) (moe_ne_max + 1) * 4);
        moe_wst = g_wide / 4 + 16; moe_sst = g_wide / SML_GROUP + 16;
        MOE_QW = dalloc((size_t) K * moe_wst * 4); MOE_QS = dalloc((size_t) K * moe_sst * 4);
        MOE_QM = dalloc((size_t) K * moe_sst * 4);
        moe_hst = moe_ff_max;
        MOE_HK = dalloc((size_t) K * moe_hst * 4); MOE_TK = dalloc((size_t) K * n_embd * 4);
        MOE_S = dalloc((size_t) n_embd * 4);
        MOE_LG = dalloc((size_t) (moe_ne_max + 1) * 4);
        MOE_G = dalloc((size_t) moe_ff_max * 4); MOE_U = dalloc((size_t) moe_ff_max * 4);
        MOE_T = dalloc((size_t) n_embd * 4);
        moe_lg_h = malloc((size_t) (moe_ne_max + 1) * sizeof(float));
        MOE_LGB = dalloc((size_t) PB * (moe_ne_max + 1) * 4);
        moe_lgb_h = malloc((size_t) PB * (moe_ne_max + 1) * sizeof(float));
        moe_sc_h = malloc((size_t) (moe_ne_max + 1) * sizeof(float));
        moe_ch_h = malloc((size_t) (moe_ne_max + 1) * sizeof(float));
    }
    EMBN = up_f(C.emb_norm, (size_t) n_embd); EMBNB = up_f(C.emb_norm_b, (size_t) n_embd);
    IDX = dalloc(4);
    DPOS = dalloc(4);
    DTOK = dalloc(4);
    DLOG = dalloc((size_t) n_ctx * 4);
    if (g_fast && (n_embd > 8192 || hd > 256)) {
        puts("  fast kernels need n_embd <= 8192 and head_dim <= 256; using the old ones");
        g_fast = 0;
    }
    QW = dalloc(((size_t) wide / 4 + 16) * 4);
    QS = dalloc(((size_t) wide / SML_GROUP + 16) * 4);
    QM = dalloc(((size_t) wide / SML_GROUP + 16) * 4);
    xhost = malloc((size_t) n_embd * sizeof(float));
    rowb  = malloc((size_t) n_embd * sizeof(float));
    printf("  path %s, grid cap %u x %u threads, uploaded %.2f GB in %.1f s; q/k/v share a permutation in %d/%d layers, gate/up in %d",
           g_fuse2 ? "FUSE2+dp4a" : (g_shared ? (g_dp4a ? "SHARED+dp4a" : "SHARED") : "fused"), g_gcap, g_bsize, (double) g_vram / 1e9, now_s() - tu, shared_qkv, NL, shared_gu);
    puts("");
    return 0;
}

// ---------------------------------------------------------------- the library face
// sml-server serves through these: one model, one GPU KV cache, absolute positions.
static float * g_lg_host;
static int g_pf_ok;

// The batched prompt path must agree with the stepped one, on THIS model.
//
// The tensor-core prompt kernel (tcm3) re-quantizes each weight tensor to int8
// with one scale per 256. On Qwen2.5-1.5B that drifts far enough that the
// prompt's KV cache is subtly wrong and decode falls into a repeat loop ("Nile
// / Nile / Nile" for "Nile / Congo / Niger"), while every single step still
// graded 0.998 and agreed on its own next token -- so the per-token grade, and
// the first generated token, both looked right (2026-10-07). It is worth 2.9x
// of the 3B's prompt speed, so it is not dropped: it is measured per model and
// kept only where it holds. If the stepped path cannot be matched even without
// it, prompts are read a token at a time.
//
// The probe is the model's own arithmetic on fixed ids: no tokenizer needed, and
// each run writes the cache positions it then reads, so no reset is needed.
// How a prompt is read, and what the model was opened able to do. sml-server
// measures the level at load (see pp_level there): the fault this exists for is
// content-dependent, so it has to be measured on real text, which means where
// the tokenizer is.
static int g_pf_ok0, g_tcm30;

int sml_gpu_open(const char * path, int want_ctx, int threads, int gpu_layers, char * why, int nwhy) {
    // No pinned host copies of the int8 prompt weights in the server: on the 3B
    // they held 2.3 GB of RAM for no measured prompt gain (2026-10-05, hot, 300
    // vs 300 tok/s). SML_RUTH=1 brings them back.
    g_ruth = 0;
    gpu_options();
    g_layers_force = gpu_layers;
    g_jb_on = 1;
    if (setjmp(g_jb)) {
        g_jb_on = 0;
        snprintf(why, (size_t) nwhy, "the GPU failed while loading the model");
        return 0;
    }
    if (gpu_setup(path, want_ctx, threads)) {
        g_jb_on = 0;
        snprintf(why, (size_t) nwhy, "this model or this machine cannot use the GPU path yet");
        return 0;
    }
    // SML_GPU_PP=0 reads prompts a token at a time, as decode does. It is how a
    // fault in the batched prompt path is told apart from one in decode.
    g_pf_ok = prefill_ready() && !(getenv("SML_GPU_PP") && atoi(getenv("SML_GPU_PP")) == 0);
    if (g_pf_ok) pp_alloc();
    g_pf_ok0 = g_pf_ok;
    g_tcm30 = g_tcm3;
    g_lg_host = malloc((size_t) n_vocab * sizeof(float));
    g_jb_on = 0;
    return g_lg_host != NULL;
}

static const float * gpu_logits(void) {
    if (g_cpu_split) return g_split_lg;
    if (!cu_check(CU.MemcpyDtoH(g_lg_host, LOG, (size_t) n_vocab * 4), "logits")) GPU_DIE();
    return g_lg_host;
}

const float * sml_gpu_prefill(const int * ids, int pos0, int n) {
    if (n <= 0 || pos0 + n > n_ctx) return NULL;
    g_jb_on = 1;
    if (setjmp(g_jb)) { g_jb_on = 0; return NULL; }
    if (g_pf_ok) gpu_prefill(ids, pos0, n);
    else for (int i = 0; i < n; i++) gpu_forward(ids[i], pos0 + i);
    const float * l = gpu_logits();
    g_jb_on = 0;
    return l;
}

const float * sml_gpu_step(int tok, int pos) {
    if (pos >= n_ctx) return NULL;
    g_jb_on = 1;
    if (setjmp(g_jb)) { g_jb_on = 0; return NULL; }
    gpu_forward(tok, pos);
    const float * l = gpu_logits();
    g_jb_on = 0;
    return l;
}

void sml_gpu_prompt_mode(int level) {
    g_pf_ok = level >= 1 ? g_pf_ok0 : 0;
    g_tcm3 = level >= 2 ? g_tcm30 : 0;
}

void sml_gpu_set_middle(int layers, int (*fn)(float * rows, int pos0, int n)) {
    g_mid_n = layers;
    g_mid_fn = fn;
}

int sml_gpu_layers(int * of) {
    if (of) *of = NL_ALL ? NL_ALL : NL;
    return NL;
}

#ifndef SML_GPU_LIB
int main(int argc, char ** argv) {
    if (argc < 4) { puts("usage: sml-gpu-fwd MODEL.ssf ids,comma,separated NGEN"); return 2; }
    const int NGEN = atoi(argv[3]);
    gpu_options();

    int ids[512], nid = 0;
    for (char * s = argv[2]; *s && nid < 512; ) {
        ids[nid++] = (int) strtol(s, &s, 10);
        while (*s == ',') s++;
    }

    if (gpu_setup(argv[1], 0, 6)) return 1;

    float * lg = malloc((size_t) n_vocab * sizeof(float));

    // SML_PP=28,128,512: GPU prompt reading. Graded first -- the logits after a
    // 40-token prompt against the CPU engine -- then timed per prompt length.
    // A model the batched path cannot read (experts: the choice is on the host)
    // reads its prompt one token at a time through the decode path -- slower,
    // and said out loud: a skipped test once looked exactly like an empty result.
    const int pp_batched = prefill_ready();
    if (getenv("SML_PP") && !pp_batched)
        puts("  SML_PP: this model cannot use the batched prompt path (see prefill_ready): stepped");
    // SML_LOGITS_OUT=FILE: read exactly the given ids with the batched GPU prompt
    // path and write the next-token logits the way sml-cli --logits does
    // ([n_vocab, depth] int32, then n_vocab floats) -- so the GPU path can be
    // graded against full precision by the same tool as the CPU engine.
    if (getenv("SML_LOGITS_OUT")) {
        if (!prefill_ready()) { puts("  SML_LOGITS_OUT: this model cannot use the GPU prompt path"); return 1; }
        pp_alloc();
        gpu_prefill(ids, 0, nid);
        // SML_LOGITS_REPEAT=N: read the prompt N more times and write the last --
        // the first read learns the order of the tensors that are not kept, and
        // later reads fill the granary ahead (from the host copies, SML_RUTH)
        for (int k = getenv("SML_LOGITS_REPEAT") ? atoi(getenv("SML_LOGITS_REPEAT")) : 0; k > 0; k--)
            gpu_prefill(ids, 0, nid);
        CU.MemcpyDtoH(lg, LOG, (size_t) n_vocab * 4);
        FILE * lf = fopen(getenv("SML_LOGITS_OUT"), "wb");
        if (!lf) { puts("  SML_LOGITS_OUT: cannot open the file"); return 1; }
        const int meta[2] = { n_vocab, 0 };
        fwrite(meta, sizeof(int), 2, lf);
        fwrite(lg, sizeof(float), (size_t) n_vocab, lf);
        fclose(lf);
        return 0;
    }
    if (getenv("SML_PP")) {
        if (pp_batched) pp_alloc();
        const int NT0 = 40 < n_ctx ? 40 : n_ctx;
        int * pid = malloc((size_t) n_ctx * sizeof(int));
        for (int i = 0; i < n_ctx; i++) pid[i] = (i < nid) ? ids[i] : (int)((i * 7919u + 13u) % (unsigned) n_vocab);
        pp_run(pp_batched, pid, NT0);
        if (g_cpu_split) memcpy(lg, g_split_lg, (size_t) n_vocab * 4);
        else CU.MemcpyDtoH(lg, LOG, (size_t) n_vocab * 4);
        // The CPU half of the grade runs AFTER the timed prompts: on this laptop the
        // CPU and GPU share one heatsink, and a 40-token CPU prefill right before the
        // timing heated the GPU into its throttle more often than the reference's
        // run, which does no such check (2026-09-26). Every run is still graded.
        float * lg0 = malloc((size_t) n_vocab * sizeof(float));
        memcpy(lg0, lg, (size_t) n_vocab * sizeof(float));
        // SML_RUTH_FRACS=a,b,..: the i-th prompt size streams fraction list[i % count],
        // so several fractions are timed in one process, minutes apart at most
        double fl[16];
        int nfl = 0, pi = 0;
        if (getenv("SML_RUTH_FRACS"))
            for (char * q = getenv("SML_RUTH_FRACS"); *q && nfl < 16; ) {
                fl[nfl++] = strtod(q, &q);
                while (*q == ',') q++;
            }
        for (char * s = getenv("SML_PP"); *s; ) {
            const int P = (int) strtol(s, &s, 10);
            while (*s == ',') s++;
            if (nfl) { g_frac_fixed = fl[pi % nfl]; printf("  [fraction %.2f] ", g_frac_fixed); }
            pi++;
            if (P <= 0 || P > n_ctx) { printf("  prompt %d skipped (n_ctx %d)", P, n_ctx); puts(""); continue; }
            pp_run(pp_batched, pid, P);
            CU.CtxSynchronize();
            double best = 1e9;
            for (int rep = 0; rep < 3; rep++) {
                const double t0p = now_s();
                pp_run(pp_batched, pid, P);
                CU.CtxSynchronize();
                const double dt = now_s() - t0p;
                if (dt < best) best = dt;
            }
            printf("  GPU prompt %4d tokens: %.1f ms = %.1f tok/s", P, best * 1e3, P / best);
            puts("");
            if (getenv("SML_RUTH_STATS"))
                printf("  granary: %ld from the copy, %ld re-made in passing, %d host copies, %d kept, "
                       "vram %zu MB at the first keep test, budget %zu MB; streamed fraction %.2f "
                       "(%.1f ms/token re-made, %.2f GB/s, %.0f MB)\n",
                       r_used, r_fb, r_n, g_nkept, g_vram_at_keep >> 20, g_keep_budget >> 20,
                       g_ruth_f, g_ruth_rate * 1e3, g_ruth_bw / 1e9, g_ruth_bytes / 1e6);
            if (getenv("PROFILE")) {
                memset(g_acc, 0, sizeof g_acc);
                g_prof = 1;
                pp_run(pp_batched, pid, P);
                g_prof = 0;
                double sum = 0.0;
                for (int i = 0; i < KB_N; i++) sum += g_acc[i];
                for (int i = 0; i < KB_N; i++) if (g_acc[i] > 0.0) {
                    printf("      %-10s %9.2f ms  %5.1f%%", KB_NAME[i], g_acc[i] * 1e3, 100.0 * g_acc[i] / sum);
                    puts("");
                }
            }
        }
        {
            const float * lc = sml_prefill(&C, pid, 0, NT0, 0);
            double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
            int ac = 0, ag = 0;
            for (int i = 0; i < n_vocab; i++) {
                const double x = lc[i], z = lg0[i];
                sa += x; sb += z; sab += x * z; saa += x * x; sbb += z * z;
                if (lc[i] > lc[ac]) ac = i;
                if (lg0[i] > lg0[ag]) ag = i;
            }
            const double nn = n_vocab;
            const double corr = (sab - sa * sb / nn) / (sqrt((saa - sa * sa / nn) * (sbb - sb * sb / nn)) + 1e-30);
            const double sd = sqrt(saa / nn - (sa / nn) * (sa / nn));
            const int tie = ac != ag && (double) lc[ac] - (double) lc[ag] <= 0.01 * sd;
            printf("  GPU prompt vs CPU engine after %d tokens: logit corr %.6f, next token %s (%d / %d)",
                   NT0, corr, ac == ag || tie ? "agrees" : "DIFFERS", ag, ac);
            if (tie) printf(" (a tie within 1%% of a logit s.d.)");
            puts("");
        }
        free(lg0);
        free(pid);
        CU.CtxDestroy(g_cuctx);
        return 0;
    }
    int total = nid + NGEN;
    if (total > n_ctx) total = n_ctx;
    if (getenv("SPEED_ONLY")) total = 1;
    double worst = 1.0, sumc = 0.0;
    int agree = 0, ties = 0, steps = 0, tok = ids[0];
    for (int pos = 0; pos < total; pos++) {
        const float * lc = sml_forward(&C, tok, pos, 0);
        gpu_forward(tok, pos);
        if (g_cpu_split) memcpy(lg, g_split_lg, (size_t) n_vocab * 4);
        else CU.MemcpyDtoH(lg, LOG, (size_t) n_vocab * 4);
        double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
        int ac = 0, ag = 0;
        for (int i = 0; i < n_vocab; i++) {
            const double a = lc[i], b = lg[i];
            sa += a; sb += b; sab += a * b; saa += a * a; sbb += b * b;
            if (lc[i] > lc[ac]) ac = i;
            if (lg[i] > lg[ag]) ag = i;
        }
        const double nn = n_vocab;
        const double corr = (sab - sa * sb / nn) / (sqrt((saa - sa * sa / nn) * (sbb - sb * sb / nn)) + 1e-30);
        if (corr < worst) worst = corr;
        sumc += corr; steps++;
        // A different top-1 counts as agreement only when it is a TIE: the CPU's
        // own logits for the two tokens within 1% of the CPU logits' standard
        // deviation -- two exact engines rounding their activations differently
        // (dp4a on the card, AVX2 on the CPU) may break a tie either way. A
        // random-weight head at hidden 256 produced exactly that: corr 0.99996,
        // top-1 "disagreeing" by a hair.
        const double sd = sqrt(saa / nn - (sa / nn) * (sa / nn));
        if (ac != ag && (double) lc[ac] - (double) lc[ag] <= 0.01 * sd) { ties++; agree++; }
        else agree += (ac == ag);
        tok = pos + 1 < nid ? ids[pos + 1] : ac;
    }
    printf("  GPU vs CPU over %d positions: mean logit corr %.6f, worst %.6f, top-1 agree %d/%d",
           steps, sumc / steps, worst, agree, steps);
    if (ties) printf(" (%d of them ties within 1%% of a logit s.d.)", ties);
    puts("");

    const int NT = 64;
    int seq[64];
    tok = ids[0];
    for (int pos = 0; pos < 8; pos++) gpu_forward(tok, pos);
    CU.CtxSynchronize();
    g_launches = 0;
    const double t0 = now_s();
    for (int pos = 0; pos < NT; pos++) {
        gpu_forward(tok, pos);
        int got = 0;
        if (g_cpu_split) {
            for (int i = 1; i < n_vocab; i++) if (g_split_lg[i] > g_split_lg[got]) got = i;
        } else {
            int nv = n_vocab;
            void * a[] = { &IDX, &LOG, &nv };
            launch(KB_ARGMAX, KF_ARGMAX, 1, a);
            CU.MemcpyDtoH(&got, IDX, 4);
        }
        seq[pos] = got;
        tok = got;
    }
    const double tg = now_s() - t0;
    const long long lpt = g_launches / NT;
    if (graph_ready()) {
        // The same 64 tokens again, each one a single graph replay: embed from
        // *DTOK, layers, head, and argmax2 writing the next *DTOK and *DPOS. The
        // host issues one call per token and reads the whole sequence at the end.
        const int saved_sync = g_sync;
        g_sync = 0;                     // no synchronizing calls inside a capture
        if (!cu_check(CU.StreamCreate(&g_stream, 1), "cuStreamCreate")) exit(1);
        int zero = 0, first = ids[0];
        CU.MemcpyHtoD(DTOK, &first, 4);
        CU.MemcpyHtoD(DPOS, &zero, 4);
        CUgraph graph;
        CUgraphExec gexec;
        if (!cu_check(CU.StreamBeginCapture(g_stream, 2), "begin capture")) exit(1);
        graph_token();
        if (!cu_check(CU.StreamEndCapture(g_stream, &graph), "end capture")) exit(1);
        if (!cu_check(CU.GraphInstantiateWithFlags(&gexec, graph, 0ULL), "instantiate")) exit(1);
        for (int i = 0; i < 8; i++) CU.GraphLaunch(gexec, g_stream);
        CU.StreamSynchronize(g_stream);
        CU.MemcpyHtoD(DTOK, &first, 4);
        CU.MemcpyHtoD(DPOS, &zero, 4);
        const double tq = now_s();
        for (int i = 0; i < NT; i++) CU.GraphLaunch(gexec, g_stream);
        CU.StreamSynchronize(g_stream);
        const double tgg = now_s() - tq;
        int glog[64];
        CU.MemcpyDtoH(glog, DLOG, (size_t) NT * 4);
        const int same = memcmp(glog, seq, sizeof seq) == 0;
        CU.GraphExecDestroy(gexec);
        CU.GraphDestroy(graph);
        g_stream = NULL;                // the rest of the run is per-token again
        g_sync = saved_sync;
        printf("  per-token path %.2f tok/s (%lld launches per token)", NT / tg, lpt);
        puts("");
        printf("  GPU greedy decode: %d tokens in %.2f s = %.2f tok/s, 1 launches/token, graph replay;"
               " tokens %s the per-token path", NT, tgg, NT / tgg, same ? "identical to" : "DIFFER FROM");
        puts("");
    } else {
        printf("  GPU greedy decode: %d tokens in %.2f s = %.2f tok/s, %lld launches/token",
               NT, tg, NT / tg, lpt);
        puts("");
    }

    if (getenv("PROFILE")) {
        memset(g_acc, 0, sizeof g_acc);
        g_prof = 1;
        const int NP = 8;
        const double tw = now_s();
        for (int pos = 0; pos < NP; pos++) {
            gpu_forward(tok, pos);
            int nv = n_vocab;
            void * a[] = { &IDX, &LOG, &nv };
            launch(KB_ARGMAX, KF_ARGMAX, 1, a);
        }
        const double tt = now_s() - tw;
        double sum = 0.0;
        for (int i = 0; i < KB_N; i++) sum += g_acc[i];
        printf("  profile, synced: %.1f ms/token, %.1f ms in kernels", tt * 1e3 / NP, sum * 1e3 / NP);
        puts("");
        for (int i = 0; i < KB_N; i++) if (g_acc[i] > 0.0) {
            printf("    %-10s %8.2f ms  %5.1f%%", KB_NAME[i], g_acc[i] * 1e3 / NP, 100.0 * g_acc[i] / sum);
            puts("");
        }
    }
    CU.CtxDestroy(g_cuctx);
    return 0;
}
#endif
