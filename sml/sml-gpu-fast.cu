// ============================================================ fast single-block work
//
// Included at the end of sml-gpu-fwd.cu; everything it uses is defined there.
//
// PROFILE on the fused path (2026-09-18): normq 10-21% and ropeattn 17-24% of
// kernel time, both one block on 1 of the 16 SMs. The causes, read off the code:
//
//   normq    -- the quantize ran one THREAD per 32-value group, each chasing
//               perm[i] then x[perm[i]] through global memory one load at a time:
//               latency, not arithmetic, with 64-96 of 256 threads busy.
//   ropeattn -- one THREAD per head for the whole attention: 16-32 threads busy.
//   qdmv     -- repeats normq's serial quantize in EVERY block of its grid.
//
// The quantize here is one WARP per group, a lane per value, with the same scale,
// rounding and clamp -- bit-identical words. Attention sums in a different order,
// so it is graded against the CPU engine like every other GPU change.
//
// Attention also reads the position from device memory rather than a launch
// argument, which is what a captured graph needs: no argument of any kernel then
// changes from one token to the next.

__attribute__((device)) static inline float wmax(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        const float u = __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, (unsigned) o, 31);
        v = v > u ? v : u;
    }
    return v;
}

__attribute__((device)) static inline float wsum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, (unsigned) o, 31);
    return v;
}

__attribute__((device)) static inline int wsumi(int v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __nvvm_shfl_sync_bfly_i32(0xffffffffu, v, (unsigned) o, 31);
    return v;
}

// One warp quantizes one group, lane i taking value i. Byte i of the group is
// element i, exactly the little-endian packing quant_group builds a word at a time.
__attribute__((device)) static inline void quant_group_warp(const float * x, const int * pp,
        unsigned int * words, float * scale, int * sum, int lane) {
    const float v = x[pp[lane]];
    const float amax = wmax(v < 0.0f ? -v : v);
    const float s  = amax / 127.0f;
    const float is = s > 0.0f ? 1.0f / s : 0.0f;
    int q = rint_i(v * is);
    if (q >  127) q =  127;
    if (q < -127) q = -127;
    ((unsigned char *) words)[lane] = (unsigned char)(signed char) q;
    const int acc = wsumi(q);
    if (lane == 0) { *scale = s; *sum = acc; }
}

#define FAST_MAX_EMBD 8192
#define FAST_MAX_HD   256

// ---------------------------------------------------------------- normq2
// sml_g_normq with float warp reductions, the vector held in shared memory, and
// the warp quantize gathering from shared rather than global memory.
extern "C" __attribute__((global)) void sml_g_normq2(
        float * x, const float * y, float * out, const float * w, const float * b,
        const int * perm, unsigned int * words, float * scale, int * sums,
        int n, float eps, int layer, int has_bias, int mode) {

    __attribute__((shared)) float sx[FAST_MAX_EMBD];
    __attribute__((shared)) float red[32];

    const int t = TID_X(), nt = NTID_X();
    const int lane = t & (WARP - 1), warp = t / WARP, nwarp = nt / WARP;

    float s = 0.0f;
    for (int i = t; i < n; i += nt) {
        float v = x[i];
        if (mode == 1) { v += y[i]; x[i] = v; }
        sx[i] = v;
        s += v;
    }
    // layer: 0 RMSNorm, 1 LayerNorm, 2 no norm at all (OLMo 2 and EXAONE 4 feed
    // the raw stream to attention) -- the stream is then quantized as it is
    float mean = 0.0f;
    if (layer == 1) {
        s = wsum(s);
        if (lane == 0) red[warp] = s;
        SYNC();
        float tot = 0.0f;
        for (int k = 0; k < nwarp; k++) tot += red[k];
        mean = tot / (float) n;
    }
    SYNC();
    float ss = 0.0f;
    for (int i = t; i < n; i += nt) { const float d = sx[i] - mean; ss += d * d; }
    ss = wsum(ss);
    if (lane == 0) red[warp] = ss;
    SYNC();
    float tot = 0.0f;
    for (int k = 0; k < nwarp; k++) tot += red[k];
    const float scl = 1.0f / __builtin_sqrtf(tot / (float) n + eps);
    for (int i = t; i < n; i += nt) {
        const float o = layer == 2 ? sx[i] : (sx[i] - mean) * scl * w[i] + (has_bias ? b[i] : 0.0f);
        out[i] = o;
        sx[i] = o;
        if (mode == 2) x[i] += y[i];
    }
    SYNC();
    for (int g = warp; g < n / GROUP; g += nwarp)
        quant_group_warp(sx, perm + g * GROUP, words + g * 8, scale + g, sums + g, lane);
}

// ---------------------------------------------------- bank-conflict-free staging
//
// fz_row4 reads word k of groups 2*blk and 2*blk+1 at sxq[16*blk + k]: at one
// instruction the 32 lanes are 16 words apart and land on 2 of the 32 shared
// memory banks, a 16-way conflict on every activation read. fz_row6 reads
// sxq[32*p + 8*g + k]: all 32 lanes on ONE bank, 32-way. Shared memory, not the
// weight stream, bounded the matvec.
//
// Word w of group g is stored at sw_addr(g, w) instead of 8*g + w. Within each
// pack of 4 groups (32 words) the low four bits are XORed with the pack index and
// the 16-word half is flipped by bit 4 of the pack index. Then for both row
// shapes, at any one instruction, lane -> bank is one to one:
//   4-bit (lane blk): bank = 16*((blk & 1) ^ c) + (K ^ ((blk >> 1) & 15))
//   6-bit (lane p):   bank = 16*(half ^ ((p >> 4) & 1)) + (c ^ (p & 15))
// Same values, same arithmetic -- only where they sit in shared memory changes.
__attribute__((device)) static inline int sw_addr(int g, int w) {
    return ((g >> 2) << 5)
         + ((((g >> 1) & 1) ^ ((g >> 6) & 1)) << 4)
         + ((((g & 1) << 3) | w) ^ ((g >> 2) & 15));
}

#define FS_STAGE(n)                                                                    \
    for (int i = TID_X(); i < (n) / 4; i += NTID_X()) sxq[sw_addr(i >> 3, i & 7)] = words[i]; \
    for (int g = TID_X(); g < (n) / GROUP; g += NTID_X()) { sxs[g] = scale[g]; sxm[g] = sums[g]; } \
    SYNC();

// quant_group_warp writing the swizzled layout: byte b of word w of group g.
__attribute__((device)) static inline void quant_group_warp_sw(const float * x, const int * pp,
        unsigned int * sxq, int g, float * scale, int * sum, int lane) {
    const float v = x[pp[lane]];
    const float amax = wmax(v < 0.0f ? -v : v);
    const float s  = amax / 127.0f;
    const float is = s > 0.0f ? 1.0f / s : 0.0f;
    int q = rint_i(v * is);
    if (q >  127) q =  127;
    if (q < -127) q = -127;
    ((unsigned char *)(sxq + sw_addr(g, lane >> 2)))[lane & 3] = (unsigned char)(signed char) q;
    const int acc = wsumi(q);
    if (lane == 0) { *scale = s; *sum = acc; }
}

#define DS4(W, K, X, Y)                                                        \
    X = sml_dp4a_us(X, (W) & LO4, sb[(K) ^ s]);                                \
    Y = sml_dp4a_us(Y, ((W) >> 4) & LO4, sb[(8 + (K)) ^ s]);

__attribute__((device, always_inline)) static inline float fs_row4(
        const unsigned char * cp, const unsigned short * dp, const unsigned char * qp,
        const unsigned short * mdp, const unsigned char * mqp,
        const unsigned int * sxq, const float * sxs, const int * sxm,
        int nblk, int lane, int hm) {
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += WARP) {
        const int gl = 2 * blk, gh = gl + 1;
        const uint4v * v = (const uint4v *)(cp + (long long) blk * 32);
        const uint4v w0 = v[0], w1 = v[1];
        const int s = (blk >> 1) & 15;
        const unsigned int * sb = sxq + ((blk >> 1) << 5) + ((((blk & 1) ^ ((blk >> 5) & 1))) << 4);
        unsigned int dl = 0u, dh = 0u;
        DS4(w0.x, 0, dl, dh) DS4(w0.y, 1, dl, dh) DS4(w0.z, 2, dl, dh) DS4(w0.w, 3, dl, dh)
        DS4(w1.x, 4, dl, dh) DS4(w1.y, 5, dl, dh) DS4(w1.z, 6, dl, dh) DS4(w1.w, 7, dl, dh)
        const int sl = gl / GPS, sh = gh / GPS;
        acc += h2f(dp[sl]) * (float) qp[gl] * sxs[gl] * (float)(int) dl;
        acc += h2f(dp[sh]) * (float) qp[gh] * sxs[gh] * (float)(int) dh;
        if (hm) {
            acc += h2f(mdp[sl]) * ((float) mqp[gl] - 127.0f) * sxs[gl] * (float) sxm[gl];
            acc += h2f(mdp[sh]) * ((float) mqp[gh] - 127.0f) * sxs[gh] * (float) sxm[gh];
        }
    }
    return acc;
}

#define D2S(W, K)                                                              \
    d0 = sml_dp4a_us(d0, (W) & LO2, b01[(K) ^ s]);                             \
    d1 = sml_dp4a_us(d1, ((W) >> 2) & LO2, b01[(8 + (K)) ^ s]);                \
    d2 = sml_dp4a_us(d2, ((W) >> 4) & LO2, b23[(K) ^ s]);                      \
    d3 = sml_dp4a_us(d3, ((W) >> 6) & LO2, b23[(8 + (K)) ^ s]);
#define DN6(W, K, X, Y, B)                                                     \
    X = sml_dp4a_us(X, (W) & LO4, B[(K) ^ s]);                                 \
    Y = sml_dp4a_us(Y, ((W) >> 4) & LO4, B[(8 + (K)) ^ s]);

__attribute__((device, always_inline)) static inline float fs_row6(
        const unsigned char * p2, const unsigned char * p4,
        const unsigned short * dp, const unsigned char * qp,
        const unsigned short * mdp, const unsigned char * mqp,
        const unsigned int * sxq, const float * sxs, const int * sxm,
        int npack, int lane, int hm) {
    float acc = 0.0f;
    for (int p = lane; p < npack; p += WARP) {
        const uint4v * pv = (const uint4v *)(p2 + (long long) p * 32);
        const uint4v * nv = (const uint4v *)(p4 + (long long) p * 64);
        const uint4v e0 = pv[0], e1 = pv[1];
        const uint4v n0 = nv[0], n1 = nv[1], n2 = nv[2], n3 = nv[3];
        const int base = p * 4;
        const int s = p & 15, hp = (p >> 4) & 1;
        const unsigned int * b01 = sxq + (p << 5) + ((0 ^ hp) << 4);
        const unsigned int * b23 = sxq + (p << 5) + ((1 ^ hp) << 4);
        unsigned int d0 = 0u, d1 = 0u, d2 = 0u, d3 = 0u;
        D2S(e0.x, 0) D2S(e0.y, 1) D2S(e0.z, 2) D2S(e0.w, 3)
        D2S(e1.x, 4) D2S(e1.y, 5) D2S(e1.z, 6) D2S(e1.w, 7)
        unsigned int b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
        DN6(n0.x, 0, b0, b1, b01) DN6(n0.y, 1, b0, b1, b01)
        DN6(n0.z, 2, b0, b1, b01) DN6(n0.w, 3, b0, b1, b01)
        DN6(n1.x, 4, b0, b1, b01) DN6(n1.y, 5, b0, b1, b01)
        DN6(n1.z, 6, b0, b1, b01) DN6(n1.w, 7, b0, b1, b01)
        DN6(n2.x, 0, b2, b3, b23) DN6(n2.y, 1, b2, b3, b23)
        DN6(n2.z, 2, b2, b3, b23) DN6(n2.w, 3, b2, b3, b23)
        DN6(n3.x, 4, b2, b3, b23) DN6(n3.y, 5, b2, b3, b23)
        DN6(n3.z, 6, b2, b3, b23) DN6(n3.w, 7, b2, b3, b23)
        const int dsum[4] = { (int) b0 + 16 * (int) d0, (int) b1 + 16 * (int) d1,
                              (int) b2 + 16 * (int) d2, (int) b3 + 16 * (int) d3 };
        for (int g = 0; g < 4; g++) {
            const int gi = base + g;
            acc += h2f(dp[gi / GPS]) * (float) qp[gi] * sxs[gi] * (float) dsum[g];
            if (hm)
                acc += h2f(mdp[gi / GPS]) * ((float) mqp[gi] - 127.0f) * sxs[gi] * (float) sxm[gi];
        }
    }
    return acc;
}

__attribute__((device, always_inline)) static inline float fs_row(
        const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq,
        int r, int n, int six, int hm,
        const unsigned int * sxq, const float * sxs, const int * sxm, int lane) {
    const int ng = n / GROUP;
    const int nscale = (ng + GPS - 1) / GPS;
    const unsigned short * dp  = sd + (long long) r * nscale;
    const unsigned char  * qp  = sq + (long long) r * ng;
    const unsigned short * mdp = md + (long long) r * nscale;
    const unsigned char  * mqp = mq + (long long) r * ng;
    if (six)
        return fs_row6(c2 + (long long) r * FZ_ROWBYTES2(n), c4 + (long long) r * FZ_ROWBYTES4(n),
                       dp, qp, mdp, mqp, sxq, sxs, sxm, n / 128, lane, hm);
    return fs_row4(c4 + (long long) r * FZ_ROWBYTES4(n), dp, qp, mdp, mqp,
                   sxq, sxs, sxm, n / 64, lane, hm);
}

__attribute__((device, always_inline)) static inline void fs_rows(
        float * y, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const float * bias,
        int rows, int n, int six, int hm, int hb,
        const unsigned int * sxq, const float * sxs, const int * sxm,
        int lane, int warp, int nwarp) {
    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        const float acc = fz_reduce(fs_row(c2, c4, sd, sq, md, mq, r, n, six, hm, sxq, sxs, sxm, lane));
        if (lane == 0) y[r] = acc + (hb ? bias[r] : 0.0f);
    }
}

// ---------------------------------------------------------------- qdmv2
// sml_g_qdmv with the warp quantize writing the conflict-free layout.
extern "C" __attribute__((global)) void sml_g_qdmv2(
        float * y, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const float * bias,
        const float * x, const int * perm,
        int n, int rows, int six, int hm, int hb) {

    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];

    const int lane  = TID_X() & (WARP - 1);
    const int warp  = TID_X() / WARP;
    const int nwarp = NTID_X() / WARP;
    for (int g = warp; g < n / GROUP; g += nwarp)
        quant_group_warp_sw(x, perm + g * GROUP, sxq, g, &sxs[g], &sxm[g], lane);
    SYNC();
    fs_rows(y, c2, c4, sd, sq, md, mq, bias, rows, n, six, hm, hb, sxq, sxs, sxm, lane, warp, nwarp);
}

// ---------------------------------------------------------------- mv3s
// sml_g_mv3 on the conflict-free layout.
extern "C" __attribute__((global)) void sml_g_mv3s(
        float * y0, const unsigned char * c20, const unsigned char * c40,
        const unsigned short * sd0, const unsigned char * sq0,
        const unsigned short * md0, const unsigned char * mq0, const float * b0,
        float * y1, const unsigned char * c21, const unsigned char * c41,
        const unsigned short * sd1, const unsigned char * sq1,
        const unsigned short * md1, const unsigned char * mq1, const float * b1,
        float * y2, const unsigned char * c22, const unsigned char * c42,
        const unsigned short * sd2, const unsigned char * sq2,
        const unsigned short * md2, const unsigned char * mq2, const float * b2,
        const unsigned int * words, const float * scale, const int * sums,
        int n, int rows0, int six0, int hm0, int hb0,
        int rows1, int six1, int hm1, int hb1,
        int rows2, int six2, int hm2, int hb2) {

    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];

    const int lane  = TID_X() & (WARP - 1);
    const int warp  = TID_X() / WARP;
    const int nwarp = NTID_X() / WARP;
    FS_STAGE(n)
    fs_rows(y0, c20, c40, sd0, sq0, md0, mq0, b0, rows0, n, six0, hm0, hb0, sxq, sxs, sxm, lane, warp, nwarp);
    fs_rows(y1, c21, c41, sd1, sq1, md1, mq1, b1, rows1, n, six1, hm1, hb1, sxq, sxs, sxm, lane, warp, nwarp);
    fs_rows(y2, c22, c42, sd2, sq2, md2, mq2, b2, rows2, n, six2, hm2, hb2, sxq, sxs, sxm, lane, warp, nwarp);
}

// ---------------------------------------------------------------- ffns
// sml_g_ffn on the conflict-free layout.
extern "C" __attribute__((global)) void sml_g_ffns(
        float * hb,
        const unsigned char * gc2, const unsigned char * gc4,
        const unsigned short * gsd, const unsigned char * gsq,
        const unsigned short * gmd, const unsigned char * gmq, const float * gbias,
        const unsigned char * uc2, const unsigned char * uc4,
        const unsigned short * usd, const unsigned char * usq,
        const unsigned short * umd, const unsigned char * umq, const float * ubias,
        const unsigned int * words, const float * scale, const int * sums,
        int n, int rows, int has_gate,
        int gsix, int ghm, int ghb, int usix, int uhm, int uhb) {

    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];

    const int lane  = TID_X() & (WARP - 1);
    const int warp  = TID_X() / WARP;
    const int nwarp = NTID_X() / WARP;
    FS_STAGE(n)
    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        float g = 0.0f;
        if (has_gate)
            g = fz_reduce(fs_row(gc2, gc4, gsd, gsq, gmd, gmq, r, n, gsix, ghm, sxq, sxs, sxm, lane));
        const float u = fz_reduce(fs_row(uc2, uc4, usd, usq, umd, umq, r, n, usix, uhm, sxq, sxs, sxm, lane));
        if (lane == 0) {
            const float uu = u + (uhb ? ubias[r] : 0.0f);
            if (has_gate) {
                const float gg = g + (ghb ? gbias[r] : 0.0f);
                hb[r] = sml_gated(gg, uu);
            } else {
                hb[r] = sml_ungated(uu);
            }
        }
    }
}

// ---------------------------------------------------------------- attn2
// One block per query head. The block rotates its q head and its kv head, the
// first head of each kv group writes k and v into the cache at pos, and every
// block uses its own rotated copy for position pos itself -- so no block ever
// reads a cache row another block is writing, and no grid-wide barrier is needed.
// Scores: a warp per position, lanes across the head dimension. Output: threads
// across (dimension, slice of positions), summed through shared memory.
extern "C" __attribute__((global)) void sml_g_attn2(
        float * out, const float * q, const float * k, const float * v,
        float * kc, float * vc, float * att,
        const float * rc, const float * rs, const int * dpos,
        int n_head, int n_kv, int hd, int half, int interleaved,
        int n_ctx, int swa, float inv) {

    __attribute__((shared)) float sq[FAST_MAX_HD];
    __attribute__((shared)) float sk[FAST_MAX_HD];
    __attribute__((shared)) float sv[FAST_MAX_HD];
    __attribute__((shared)) float sacc[256];
    __attribute__((shared)) float red[32];

    const int h = CTA_X();
    const int t = TID_X(), nt = NTID_X();
    const int lane = t & (WARP - 1), warp = t / WARP, nwarp = nt / WARP;
    const int pos = *dpos;
    const int t0 = (swa && pos >= swa) ? pos - swa + 1 : 0;
    const int rep = n_head / n_kv, kvh = h / rep;
    const long long kvd = (long long) n_kv * hd;
    const float * cs = rc + (long long) pos * half;
    const float * sn = rs + (long long) pos * half;

    for (int i = t; i < hd; i += nt) {
        sq[i] = q[(long long) h * hd + i];
        sk[i] = k[(long long) kvh * hd + i];
        sv[i] = v[(long long) kvh * hd + i];
    }
    SYNC();
    for (int i = t; i < half; i += nt) {
        const int a0 = interleaved ? 2 * i : i;
        const int a1 = interleaved ? 2 * i + 1 : i + half;
        float x0 = sq[a0], x1 = sq[a1];
        sq[a0] = x0 * cs[i] - x1 * sn[i];
        sq[a1] = x0 * sn[i] + x1 * cs[i];
        x0 = sk[a0]; x1 = sk[a1];
        sk[a0] = x0 * cs[i] - x1 * sn[i];
        sk[a1] = x0 * sn[i] + x1 * cs[i];
    }
    SYNC();
    if (h % rep == 0)
        for (int i = t; i < hd; i += nt) {
            kc[(long long) pos * kvd + (long long) kvh * hd + i] = sk[i];
            vc[(long long) pos * kvd + (long long) kvh * hd + i] = sv[i];
        }

    float * a = att + (long long) h * n_ctx;
    float mx = -3.4e38f;
    for (int tt = t0 + warp; tt <= pos; tt += nwarp) {
        const float * kr = tt == pos ? sk : kc + (long long) tt * kvd + (long long) kvh * hd;
        float s = 0.0f;
        for (int i = lane; i < hd; i += WARP) s += sq[i] * kr[i];
        s = sml_score(wsum(s) * inv, h, tt, pos);
        if (lane == 0) a[tt] = s;
        mx = mx > s ? mx : s;
    }
    if (lane == 0) red[warp] = mx;
    SYNC();
    mx = -3.4e38f;
    for (int w = 0; w < nwarp; w++) mx = mx > red[w] ? mx : red[w];
    SYNC();

    float sum = 0.0f;
    for (int tt = t0 + t; tt <= pos; tt += nt) {
        const float e = sml_exp(a[tt] - mx);
        a[tt] = e;
        sum += e;
    }
    sum = wsum(sum);
    if (lane == 0) red[warp] = sum;
    SYNC();
    sum = 0.0f;
    for (int w = 0; w < nwarp; w++) sum += red[w];
    const float isum = 1.0f / sum;

    const int parts = nt / hd;
    const int i = t % hd, part = t / hd;
    float acc = 0.0f;
    if (part < parts)
        for (int tt = t0 + part; tt <= pos; tt += parts) {
            const float * vr = tt == pos ? sv : vc + (long long) tt * kvd + (long long) kvh * hd;
            acc += a[tt] * vr[i];
        }
    sacc[t] = acc;
    SYNC();
    if (t < hd) {
        float o = 0.0f;
        for (int p = 0; p < parts; p++) o += sacc[p * hd + t];
        out[(long long) h * hd + t] = o * isum;
    }
}


// ---------------------------------------------------------------- embed
// The embedding lookup on the GPU: row *dtok of token_embd, dequantized and
// scattered through its permutation into x. gpu_forward did this on the host and
// uploaded it with a blocking copy every token, which kept the CPU in the loop.
// Same arithmetic as sml_dequant_row_flat4/6 (8-bit per-group fields, v8 flat).
extern "C" __attribute__((global)) void sml_g_embed(
        float * x, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const int * perm,
        const int * dtok, int n, int six, int hm) {
    const long long row = *dtok;
    const int ng = n / GROUP, nscale = (ng + GPS - 1) / GPS;
    const unsigned char * cp4 = c4 + row * (n / 2);
    const unsigned char * cp2 = c2 + row * (n / 4);
    for (int i = TID_X() + CTA_X() * NTID_X(); i < n; i += NTID_X() * NCTA_X()) {
        const int g = i / GROUP, j = i % GROUP;
        const int quad = g / 4, kk = g % 4;
        const unsigned char b = cp4[quad * 64 + (kk >= 2 ? 32 : 0) + j];
        int code = (kk & 1) ? (b >> 4) & 15 : b & 15;
        if (six) code |= ((cp2[(g / 4) * 32 + j] >> (2 * (g % 4))) & 3) << 4;
        const long long si = row * nscale + g / GPS;
        const float step = h2f(sd[si]) * (float) sq[row * ng + g];
        const float mn = hm ? h2f(md[si]) * ((float) mq[row * ng + g] - 127.0f) : 0.0f;
        x[perm[i]] = (float) code * step + mn;
    }
}

// ---------------------------------------------------------------- argmax2
// sml_g_argmax that also feeds the next token: the id goes to *dtok (the embed
// kernel input), into the token log at the position it was produced for, and
// *dpos advances. A captured token graph then replays with no host in the loop.
extern "C" __attribute__((global)) void sml_g_argmax2(
        const float * v, int n, int * dtok, int * dpos, int * tlog) {

    __attribute__((shared)) float bv[256];
    __attribute__((shared)) int   bi[256];

    const int t = TID_X(), nt = NTID_X();
    float best = -3.4e38f;
    int idx = n;
    for (int i = t; i < n; i += nt)
        if (v[i] > best) { best = v[i]; idx = i; }
    bv[t] = best; bi[t] = idx;
    SYNC();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (t < s && (bv[t + s] > bv[t] || (bv[t + s] == bv[t] && bi[t + s] < bi[t]))) {
            bv[t] = bv[t + s]; bi[t] = bi[t + s];
        }
        SYNC();
    }
    if (t == 0) {
        const int p = *dpos;
        tlog[p] = bi[0];
        *dtok = bi[0];
        *dpos = p + 1;
    }
}


// ---------------------------------------------------------------- quantw2
// The o and down projections quantized their input inside EVERY block of the
// matvec: 48 blocks each walking all n/32 groups, 4 warps at a time, with a
// dependent perm -> x load per group. PROFILE on pythia: qdmv 37% of kernel time
// at ~3x its bandwidth cost; the 1B/3B down projection walks ~70 groups deep per
// warp. Quantize once instead, a warp per group across the whole grid, into the
// global words that every mv3s block stages with one coalesced copy.
extern "C" __attribute__((global)) void sml_g_quantw2(
        const float * x, const int * perm, unsigned int * words, float * scale, int * sums,
        int n) {
    const int lane = TID_X() & (WARP - 1);
    const int g = CTA_X() * (NTID_X() / WARP) + TID_X() / WARP;
    if (g < n / GROUP)
        quant_group_warp(x, perm + g * GROUP, words + g * 8, scale + g, sums + g, lane);
}


// ============================================================ batched prompt reading
//
// The decode path reads every weight once per token. A prompt is known in
// advance, so here up to PB_MAX tokens share each weight read: one warp per row
// keeps PB_MAX per-token accumulators in registers, and the activations sit in
// global memory (L1/L2 cached) with one contiguous block of words per token --
// no shared staging, which the wide down-projection inputs would not fit.
// Attention is two launches: RoPE + KV write for every position of the batch,
// then attention, so no block reads a cache row another block is writing.
#define CTA_Y() __nvvm_read_ptx_sreg_ctaid_y()
#define PB_MAX 16

extern "C" __attribute__((global)) void sml_g_embedb(
        float * xall, int xs, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const int * perm,
        const int * toks, int n, int six, int hm) {
    const int slot = CTA_Y();
    const long long row = toks[slot];
    float * x = xall + (long long) slot * xs;
    const int ng = n / GROUP, nscale = (ng + GPS - 1) / GPS;
    const unsigned char * cp4 = c4 + row * (n / 2);
    const unsigned char * cp2 = c2 + row * (n / 4);
    for (int i = TID_X() + CTA_X() * NTID_X(); i < n; i += NTID_X() * NCTA_X()) {
        const int g = i / GROUP, j = i % GROUP;
        const int quad = g / 4, kk = g % 4;
        const unsigned char b = cp4[quad * 64 + (kk >= 2 ? 32 : 0) + j];
        int code = (kk & 1) ? (b >> 4) & 15 : b & 15;
        if (six) code |= ((cp2[(g / 4) * 32 + j] >> (2 * (g % 4))) & 3) << 4;
        const long long si = row * nscale + g / GPS;
        const float step = h2f(sd[si]) * (float) sq[row * ng + g];
        const float mn = hm ? h2f(md[si]) * ((float) mq[row * ng + g] - 127.0f) : 0.0f;
        x[perm[i]] = (float) code * step + mn;
    }
}

// sml_g_normq2, one block per token of the batch.
// ============================================================ integer epilogue (per-256)
//
// sml_g_tcm spent more instructions on its epilogue than on the tensor cores:
// after every mma, each thread scaled four outputs by that group's weight scale
// and that token's activation scale -- ~28 float ops per mma. Profiled at 1.7% of
// the tensor cores' peak.
//
// The format already holds the way out. A group's weight scale is an INTEGER
// (sq) times one fp16 per 256 weights (sd). With ONE activation scale per 256
// values too -- the prompt-only trade Ernest approved on the CPU, the same shape
// as the reference's per-256 activations -- the eight groups of a superblock can
// be summed in integers: iacc += dot_g * sq_g, one IMAD per output. The float
// scale is applied once per superblock, not once per group. The chunk is exactly
// one superblock, so the integer sums are local to a chunk.
// Overflow: |dot_g| <= 32*63*127 = 256,032; * 255 * 8 groups < 2^30.
#ifndef TC2_BN
#define TC2_BN  32          // tokens per block (grid.y covers the batch)
#endif
#define TC2_NT  (TC2_BN / 8)

// One warp quantizes one 256-value superblock with a single scale, written into
// all eight of its group slots (per-group sums stay per group: the min term needs
// them). Lane l takes values l, l+32, ..., l+224 -- one per group.
__attribute__((device)) static inline void quant_sb_warp(const float * x, const int * perm,
        int sb, int ng, unsigned int * words, float * scale, int * sums, int lane) {
    const int g0 = sb * 8;
    const int gn = ng - g0 < 8 ? ng - g0 : 8;
    float v[8];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        v[j] = j < gn ? x[perm[(g0 + j) * GROUP + lane]] : 0.0f;
        const float a = v[j] < 0.0f ? -v[j] : v[j];
        amax = amax > a ? amax : a;
    }
    amax = wmax(amax);
    const float s  = amax / 127.0f;
    const float is = s > 0.0f ? 1.0f / s : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        if (j >= gn) break;
        int q = rint_i(v[j] * is);
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        ((unsigned char *)(words + (g0 + j) * 8))[lane] = (unsigned char)(signed char) q;
        const int acc = wsumi(q);
        if (lane == 0) { scale[g0 + j] = s; sums[g0 + j] = acc; }
    }
}


extern "C" __attribute__((global)) void sml_g_normq2b(
        float * xall, int xs, const float * yall, int ys, float * oall, int os,
        const float * w, const float * b, const int * perm,
        unsigned int * wall, int wst, float * sall, int * mall, int sst,
        int n, float eps, int layer, int has_bias, int mode, int sbq) {

    __attribute__((shared)) float sx[FAST_MAX_EMBD];
    __attribute__((shared)) float red[32];

    const int slot = CTA_X();
    float * x = xall + (long long) slot * xs;
    const float * y = yall + (long long) slot * ys;
    float * out = oall + (long long) slot * os;
    unsigned int * words = wall + (long long) slot * wst;
    float * scale = sall + (long long) slot * sst;
    int * sums = mall + (long long) slot * sst;

    const int t = TID_X(), nt = NTID_X();
    const int lane = t & (WARP - 1), warp = t / WARP, nwarp = nt / WARP;
    float s = 0.0f;
    for (int i = t; i < n; i += nt) {
        float v = x[i];
        if (mode == 1) { v += y[i]; x[i] = v; }
        sx[i] = v;
        s += v;
    }
    // layer: 0 RMSNorm, 1 LayerNorm, 2 no norm at all (OLMo 2 and EXAONE 4 feed
    // the raw stream to attention) -- the stream is then quantized as it is
    float mean = 0.0f;
    if (layer == 1) {
        s = wsum(s);
        if (lane == 0) red[warp] = s;
        SYNC();
        float tot = 0.0f;
        for (int k = 0; k < nwarp; k++) tot += red[k];
        mean = tot / (float) n;
    }
    SYNC();
    float ss = 0.0f;
    for (int i = t; i < n; i += nt) { const float d = sx[i] - mean; ss += d * d; }
    ss = wsum(ss);
    if (lane == 0) red[warp] = ss;
    SYNC();
    float tot = 0.0f;
    for (int k = 0; k < nwarp; k++) tot += red[k];
    const float scl = 1.0f / __builtin_sqrtf(tot / (float) n + eps);
    for (int i = t; i < n; i += nt) {
        const float o = layer == 2 ? sx[i] : (sx[i] - mean) * scl * w[i] + (has_bias ? b[i] : 0.0f);
        out[i] = o;
        sx[i] = o;
        if (mode == 2) x[i] += y[i];
    }
    SYNC();
    if (sbq) {
        const int ngq = n / GROUP;
        for (int sb = warp; sb < (ngq + 7) / 8; sb += nwarp)
            quant_sb_warp(sx, perm, sb, ngq, words, scale, sums, lane);
    } else {
        for (int g = warp; g < n / GROUP; g += nwarp)
            quant_group_warp(sx, perm + g * GROUP, words + g * 8, scale + g, sums + g, lane);
    }
}

extern "C" __attribute__((global)) void sml_g_quantw2b(
        const float * xall, int xs, const int * perm, unsigned int * wall, int wst,
        float * sall, int * mall, int sst, int n) {
    const int slot = CTA_Y();
    const int lane = TID_X() & (WARP - 1);
    const int g = CTA_X() * (NTID_X() / WARP) + TID_X() / WARP;
    if (g < n / GROUP)
        quant_group_warp(xall + (long long) slot * xs, perm + g * GROUP,
                         wall + (long long) slot * wst + g * 8,
                         sall + (long long) slot * sst + g, mall + (long long) slot * sst + g, lane);
}

// Four tokens at a time: unrolling all sixteen made 1280 dp4a and ~7000 virtual
// registers per kernel against 255 physical -- the driver compiler spilled and the
// batched path ran slower than one token at a time. A chunk re-reads its weight
// row from L1/L2; only the first chunk of a batch touches DRAM.
#define PB_CHUNK 4

// One row against tokens b0..b0+PB_CHUNK-1: acc[j] gets this lane's share of row . x_b.
__attribute__((device, always_inline)) static inline void mvb_row(float * acc,
        const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq,
        int r, int n, int six, int hm,
        const unsigned int * words, int wst, const float * scale, const int * sums, int sst,
        int b0, int nb, int lane) {
    const int ng = n / GROUP, nscale = (ng + GPS - 1) / GPS;
    const unsigned short * dp  = sd + (long long) r * nscale;
    const unsigned short * mdp = md + (long long) r * nscale;
    const unsigned char  * qp  = sq + (long long) r * ng;
    const unsigned char  * mqp = mq + (long long) r * ng;
    if (!six) {
        const unsigned char * cp = c4 + (long long) r * (n / 2);
        for (int blk = lane; blk < n / 64; blk += WARP) {
            const uint4v * v = (const uint4v *)(cp + (long long) blk * 32);
            const uint4v w0 = v[0], w1 = v[1];
            const unsigned int cw[8] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w };
            const int gl = 2 * blk, gh = gl + 1;
            const float wl = h2f(dp[gl / GPS]) * (float) qp[gl];
            const float wh = h2f(dp[gh / GPS]) * (float) qp[gh];
            const float ml = hm ? h2f(mdp[gl / GPS]) * ((float) mqp[gl] - 127.0f) : 0.0f;
            const float mh = hm ? h2f(mdp[gh / GPS]) * ((float) mqp[gh] - 127.0f) : 0.0f;
#pragma unroll
            for (int j = 0; j < PB_CHUNK; j++) {
                const int b = b0 + j;
                if (b < nb) {
                    const uint4v * aw = (const uint4v *)(words + (long long) b * wst + gl * 8);
                    const uint4v a0 = aw[0], a1 = aw[1], a2 = aw[2], a3 = aw[3];
                    const unsigned int al[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
                    const unsigned int ah[8] = { a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w };
                    unsigned int dl = 0u, dh = 0u;
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        dl = sml_dp4a_us(dl, cw[k] & LO4, al[k]);
                        dh = sml_dp4a_us(dh, (cw[k] >> 4) & LO4, ah[k]);
                    }
                    const float sl = scale[(long long) b * sst + gl];
                    const float sh = scale[(long long) b * sst + gh];
                    acc[j] += wl * sl * (float)(int) dl + wh * sh * (float)(int) dh;
                    if (hm)
                        acc[j] += ml * sl * (float) sums[(long long) b * sst + gl]
                                + mh * sh * (float) sums[(long long) b * sst + gh];
                }
            }
        }
    } else {
        const unsigned char * p2 = c2 + (long long) r * (n / 4);
        const unsigned char * p4 = c4 + (long long) r * (n / 2);
        for (int p = lane; p < n / 128; p += WARP) {
            const uint4v * pv = (const uint4v *)(p2 + (long long) p * 32);
            const uint4v * nv = (const uint4v *)(p4 + (long long) p * 64);
            const uint4v e0 = pv[0], e1 = pv[1];
            const uint4v n0 = nv[0], n1 = nv[1], n2 = nv[2], n3 = nv[3];
            const unsigned int ew[8]  = { e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w };
            const unsigned int nwa[8] = { n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w };
            const unsigned int nwb[8] = { n2.x, n2.y, n2.z, n2.w, n3.x, n3.y, n3.z, n3.w };
            const int base = p * 4;
            float wsc[4], msc[4];
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const int gi = base + g;
                wsc[g] = h2f(dp[gi / GPS]) * (float) qp[gi];
                msc[g] = hm ? h2f(mdp[gi / GPS]) * ((float) mqp[gi] - 127.0f) : 0.0f;
            }
#pragma unroll
            for (int j = 0; j < PB_CHUNK; j++) {
                const int b = b0 + j;
                if (b < nb) {
                    const uint4v * aw = (const uint4v *)(words + (long long) b * wst + base * 8);
                    unsigned int a[32];
#pragma unroll
                    for (int q = 0; q < 8; q++) {
                        const uint4v t = aw[q];
                        a[4 * q] = t.x; a[4 * q + 1] = t.y; a[4 * q + 2] = t.z; a[4 * q + 3] = t.w;
                    }
                    unsigned int d0 = 0u, d1 = 0u, d2 = 0u, d3 = 0u, b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        d0 = sml_dp4a_us(d0, ew[k] & LO2, a[k]);
                        d1 = sml_dp4a_us(d1, (ew[k] >> 2) & LO2, a[8 + k]);
                        d2 = sml_dp4a_us(d2, (ew[k] >> 4) & LO2, a[16 + k]);
                        d3 = sml_dp4a_us(d3, (ew[k] >> 6) & LO2, a[24 + k]);
                        b0 = sml_dp4a_us(b0, nwa[k] & LO4, a[k]);
                        b1 = sml_dp4a_us(b1, (nwa[k] >> 4) & LO4, a[8 + k]);
                        b2 = sml_dp4a_us(b2, nwb[k] & LO4, a[16 + k]);
                        b3 = sml_dp4a_us(b3, (nwb[k] >> 4) & LO4, a[24 + k]);
                    }
                    const int ds[4] = { (int) b0 + 16 * (int) d0, (int) b1 + 16 * (int) d1,
                                        (int) b2 + 16 * (int) d2, (int) b3 + 16 * (int) d3 };
#pragma unroll
                    for (int g = 0; g < 4; g++) {
                        const long long gi = (long long) b * sst + base + g;
                        acc[j] += wsc[g] * scale[gi] * (float) ds[g];
                        if (hm) acc[j] += msc[g] * scale[gi] * (float) sums[gi];
                    }
                }
            }
        }
    }
}

extern "C" __attribute__((global)) void sml_g_mvb(
        float * y, int yst, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const float * bias,
        const unsigned int * words, int wst, const float * scale, const int * sums, int sst,
        int n, int rows, int six, int hm, int hb, int nb) {
    const int lane = TID_X() & (WARP - 1), warp = TID_X() / WARP, nwarp = NTID_X() / WARP;
    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        for (int b0 = 0; b0 < nb; b0 += PB_CHUNK) {
            float acc[PB_CHUNK];
#pragma unroll
            for (int j = 0; j < PB_CHUNK; j++) acc[j] = 0.0f;
            mvb_row(acc, c2, c4, sd, sq, md, mq, r, n, six, hm, words, wst, scale, sums, sst, b0, nb, lane);
#pragma unroll
            for (int j = 0; j < PB_CHUNK; j++) {
                if (b0 + j < nb) {
                    const float sum = wsum(acc[j]);
                    if (lane == 0) y[(long long)(b0 + j) * yst + r] = sum + (hb ? bias[r] : 0.0f);
                }
            }
        }
    }
}

extern "C" __attribute__((global)) void sml_g_ffnb(
        float * hall, int hst,
        const unsigned char * gc2, const unsigned char * gc4,
        const unsigned short * gsd, const unsigned char * gsq,
        const unsigned short * gmd, const unsigned char * gmq, const float * gbias,
        const unsigned char * uc2, const unsigned char * uc4,
        const unsigned short * usd, const unsigned char * usq,
        const unsigned short * umd, const unsigned char * umq, const float * ubias,
        const unsigned int * words, int wst, const float * scale, const int * sums, int sst,
        int n, int rows, int has_gate,
        int gsix, int ghm, int ghb, int usix, int uhm, int uhb, int nb) {
    const int lane = TID_X() & (WARP - 1), warp = TID_X() / WARP, nwarp = NTID_X() / WARP;
    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        for (int b0 = 0; b0 < nb; b0 += PB_CHUNK) {
            float ag[PB_CHUNK], au[PB_CHUNK];
#pragma unroll
            for (int j = 0; j < PB_CHUNK; j++) { ag[j] = 0.0f; au[j] = 0.0f; }
            if (has_gate)
                mvb_row(ag, gc2, gc4, gsd, gsq, gmd, gmq, r, n, gsix, ghm, words, wst, scale, sums, sst, b0, nb, lane);
            mvb_row(au, uc2, uc4, usd, usq, umd, umq, r, n, usix, uhm, words, wst, scale, sums, sst, b0, nb, lane);
#pragma unroll
            for (int j = 0; j < PB_CHUNK; j++) {
                if (b0 + j < nb) {
                    const float u = wsum(au[j]);
                    const float g = has_gate ? wsum(ag[j]) : 0.0f;
                    if (lane == 0) {
                        const float uu = u + (uhb ? ubias[r] : 0.0f);
                        float o;
                        if (has_gate) {
                            const float gg = g + (ghb ? gbias[r] : 0.0f);
                            o = sml_gated(gg, uu);
                        } else {
                            o = sml_ungated(uu);
                        }
                        hall[(long long)(b0 + j) * hst + r] = o;
                    }
                }
            }
        }
    }
}

// RoPE on q and k of every token in the batch, and k, v into the cache at its position.
extern "C" __attribute__((global)) void sml_g_ropekvb(
        float * qall, int qst, float * kall, int kst, const float * vall, int vst,
        float * kc, float * vc, const float * rc, const float * rs,
        int n_head, int n_kv, int hd, int half, int interleaved, int pos0) {
    const int slot = CTA_X(), t = TID_X(), nt = NTID_X();
    const int pos = pos0 + slot;
    float * q = qall + (long long) slot * qst;
    float * k = kall + (long long) slot * kst;
    const float * v = vall + (long long) slot * vst;
    const float * cs = rc + (long long) pos * half;
    const float * sn = rs + (long long) pos * half;
    for (int idx = t; idx < n_head * half; idx += nt) {
        const int h = idx / half, i = idx % half;
        float * u = q + h * hd;
        const int a0 = interleaved ? 2 * i : i, a1 = interleaved ? 2 * i + 1 : i + half;
        const float x0 = u[a0], x1 = u[a1];
        u[a0] = x0 * cs[i] - x1 * sn[i];
        u[a1] = x0 * sn[i] + x1 * cs[i];
    }
    for (int idx = t; idx < n_kv * half; idx += nt) {
        const int h = idx / half, i = idx % half;
        float * u = k + h * hd;
        const int a0 = interleaved ? 2 * i : i, a1 = interleaved ? 2 * i + 1 : i + half;
        const float x0 = u[a0], x1 = u[a1];
        u[a0] = x0 * cs[i] - x1 * sn[i];
        u[a1] = x0 * sn[i] + x1 * cs[i];
    }
    SYNC();
    const long long kvd = (long long) n_kv * hd;
    for (int i = t; i < kvd; i += nt) {
        kc[(long long) pos * kvd + i] = k[i];
        vc[(long long) pos * kvd + i] = v[i];
    }
}

// Attention for every (head, token) of the batch; every key and value it reads,
// its own position included, is already in the cache.
extern "C" __attribute__((global)) void sml_g_attnb(
        float * oall, int ost, const float * qall, int qst,
        const float * kc, const float * vc, float * att,
        int n_head, int n_kv, int hd, int n_ctx, int swa, float inv, int pos0) {

    __attribute__((shared)) float sq[FAST_MAX_HD];
    __attribute__((shared)) float sacc[256];
    __attribute__((shared)) float red[32];

    const int h = CTA_X(), slot = CTA_Y();
    const int t = TID_X(), nt = NTID_X();
    const int lane = t & (WARP - 1), warp = t / WARP, nwarp = nt / WARP;
    const int pos = pos0 + slot;
    const int t0 = (swa && pos >= swa) ? pos - swa + 1 : 0;
    const int rep = n_head / n_kv, kvh = h / rep;
    const long long kvd = (long long) n_kv * hd;
    const float * q = qall + (long long) slot * qst + (long long) h * hd;
    for (int i = t; i < hd; i += nt) sq[i] = q[i];
    SYNC();

    float * a = att + ((long long) slot * n_head + h) * n_ctx;
    float mx = -3.4e38f;
    for (int tt = t0 + warp; tt <= pos; tt += nwarp) {
        const float * kr = kc + (long long) tt * kvd + (long long) kvh * hd;
        float s = 0.0f;
        for (int i = lane; i < hd; i += WARP) s += sq[i] * kr[i];
        s = sml_score(wsum(s) * inv, h, tt, pos);
        if (lane == 0) a[tt] = s;
        mx = mx > s ? mx : s;
    }
    if (lane == 0) red[warp] = mx;
    SYNC();
    mx = -3.4e38f;
    for (int w = 0; w < nwarp; w++) mx = mx > red[w] ? mx : red[w];
    SYNC();
    float sum = 0.0f;
    for (int tt = t0 + t; tt <= pos; tt += nt) {
        const float e = sml_exp(a[tt] - mx);
        a[tt] = e;
        sum += e;
    }
    sum = wsum(sum);
    if (lane == 0) red[warp] = sum;
    SYNC();
    sum = 0.0f;
    for (int w = 0; w < nwarp; w++) sum += red[w];
    const float isum = 1.0f / sum;
    const int parts = nt / hd;
    const int i = t % hd, part = t / hd;
    float acc = 0.0f;
    if (part < parts)
        for (int tt = t0 + part; tt <= pos; tt += parts)
            acc += a[tt] * vc[(long long) tt * kvd + (long long) kvh * hd + i];
    sacc[t] = acc;
    SYNC();
    if (t < hd) {
        float o = 0.0f;
        for (int p = 0; p < parts; p++) o += sacc[p * hd + t];
        oall[(long long) slot * ost + (long long) h * hd + t] = o * isum;
    }
}


// ============================================================ tiled prompt matmul
//
// sml_g_mvb had one warp per row pull every token's activations from L1/L2 by
// itself. On the 1B down projection that is ~268 MB of activation reads for 8 MB
// of weights: the kernel drowned in its own input. Here a block owns TM_BM rows
// and stages the activations of ALL batch tokens for one chunk of TM_KG groups in
// shared memory once, so every row in the block reads them from there.
//
// Thread (lane, warp) computes rows 2*lane and 2*lane+1 against tokens
// 4*warp .. 4*warp+3: an eight-output micro-tile summed over the whole K by one
// thread, no warp reduction. All lanes of a warp read the same activation words,
// which shared memory broadcasts; weight rows are read with a pitch of one word
// more than their width, so the 32 lanes, 2 rows apart, land on distinct banks.
//
// A chunk is 8 groups, which is exactly one fp16 super-scale: the weight scale for
// every group in it is precomputed once into shared, not per dot product.
#define TM_BM   64          // rows per block
#define TM_BN   16          // tokens per block (== PB, the prompt batch)
#define TM_KG   8           // groups per K chunk (= GPS: one super-scale)
#define TM_W4   32          // 4-bit plane words per row per chunk (8 groups * 16 bytes / 4)
#define TM_W2   16          // 2-bit plane words per row per chunk (6-bit tensors)

extern "C" __attribute__((global)) void sml_g_tmm(
        float * y, int yst, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const float * bias,
        const unsigned int * words, int wst, const float * scale, const int * sums, int sst,
        int n, int rows, int six, int hm, int hb, int nb) {

    __attribute__((shared)) unsigned int w4s[TM_BM][TM_W4 + 1];
    __attribute__((shared)) unsigned int w2s[TM_BM][TM_W2 + 1];
    __attribute__((shared)) unsigned int xs[TM_BN][TM_KG * 8];
    __attribute__((shared)) float        xsc[TM_BN][TM_KG];
    __attribute__((shared)) int          xsm[TM_BN][TM_KG];
    __attribute__((shared)) float        wsc[TM_BM][TM_KG];
    __attribute__((shared)) float        wmn[TM_BM][TM_KG];

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int r0 = CTA_X() * TM_BM;
    const int ng = n / GROUP, nscale = (ng + GPS - 1) / GPS;
    const long long rb4 = n / 2, rb2 = n / 4;          // bytes per row, each plane
    const int nchunk = (ng + TM_KG - 1) / TM_KG;

    float acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int t = 0; t < 4; t++) acc[i][t] = 0.0f;

    for (int c = 0; c < nchunk; c++) {
        const int g0 = c * TM_KG;
        const int gn = ng - g0 < TM_KG ? ng - g0 : TM_KG;   // groups in this chunk
        // ---- stage: weights (both planes), their scales, activations
        for (int i = tid; i < TM_BM * TM_W4; i += 128) {
            const int r = i / TM_W4, w = i % TM_W4;
            const int ok = r0 + r < rows && w < gn * 4;
            w4s[r][w] = ok ? *(const unsigned int *)(c4 + (long long)(r0 + r) * rb4 + (long long) g0 * 16 + w * 4) : 0u;
        }
        if (six)
            for (int i = tid; i < TM_BM * TM_W2; i += 128) {
                const int r = i / TM_W2, w = i % TM_W2;
                const int ok = r0 + r < rows && w < gn * 2;
                w2s[r][w] = ok ? *(const unsigned int *)(c2 + (long long)(r0 + r) * rb2 + (long long) g0 * 8 + w * 4) : 0u;
            }
        for (int i = tid; i < TM_BM * TM_KG; i += 128) {
            const int r = i / TM_KG, g = i % TM_KG;
            float s = 0.0f, m = 0.0f;
            if (r0 + r < rows && g < gn) {
                const long long rr = r0 + r;
                s = h2f(sd[rr * nscale + c]) * (float) sq[rr * ng + g0 + g];
                if (hm) m = h2f(md[rr * nscale + c]) * ((float) mq[rr * ng + g0 + g] - 127.0f);
            }
            wsc[r][g] = s;
            wmn[r][g] = m;
        }
        for (int i = tid; i < TM_BN * TM_KG * 8; i += 128) {
            const int t = i / (TM_KG * 8), w = i % (TM_KG * 8);
            xs[t][w] = (t < nb && w < gn * 8) ? words[(long long) t * wst + (long long) g0 * 8 + w] : 0u;
        }
        for (int i = tid; i < TM_BN * TM_KG; i += 128) {
            const int t = i / TM_KG, g = i % TM_KG;
            const int ok = t < nb && g < gn;
            xsc[t][g] = ok ? scale[(long long) t * sst + g0 + g] : 0.0f;
            xsm[t][g] = ok ? sums[(long long) t * sst + g0 + g] : 0;
        }
        SYNC();

        // ---- compute this thread's 2 rows x 4 tokens over the chunk
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int lr = lane * 2 + i;
            if (!six) {
#pragma unroll
                for (int blk = 0; blk < TM_KG / 2; blk++) {
                    const int gl = 2 * blk, gh = gl + 1;
                    unsigned int dl[4] = { 0u, 0u, 0u, 0u }, dh[4] = { 0u, 0u, 0u, 0u };
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        const unsigned int w = w4s[lr][blk * 8 + k];
                        const unsigned int wl = w & LO4, wh = (w >> 4) & LO4;
#pragma unroll
                        for (int t = 0; t < 4; t++) {
                            const int tt = warp * 4 + t;
                            dl[t] = sml_dp4a_us(dl[t], wl, xs[tt][gl * 8 + k]);
                            dh[t] = sml_dp4a_us(dh[t], wh, xs[tt][gh * 8 + k]);
                        }
                    }
#pragma unroll
                    for (int t = 0; t < 4; t++) {
                        const int tt = warp * 4 + t;
                        acc[i][t] += wsc[lr][gl] * xsc[tt][gl] * (float)(int) dl[t]
                                   + wsc[lr][gh] * xsc[tt][gh] * (float)(int) dh[t];
                        if (hm)
                            acc[i][t] += wmn[lr][gl] * xsc[tt][gl] * (float) xsm[tt][gl]
                                       + wmn[lr][gh] * xsc[tt][gh] * (float) xsm[tt][gh];
                    }
                }
            } else {
#pragma unroll
                for (int pk = 0; pk < TM_KG / 4; pk++) {
                    const int q0 = pk * 4;
                    unsigned int d[4][4], b[4][4];
#pragma unroll
                    for (int t = 0; t < 4; t++)
#pragma unroll
                        for (int g = 0; g < 4; g++) { d[t][g] = 0u; b[t][g] = 0u; }
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        const unsigned int e  = w2s[lr][pk * 8 + k];
                        const unsigned int nl = w4s[lr][pk * 16 + k];
                        const unsigned int nh = w4s[lr][pk * 16 + 8 + k];
                        const unsigned int e0 = e & LO2, e1 = (e >> 2) & LO2;
                        const unsigned int e2 = (e >> 4) & LO2, e3 = (e >> 6) & LO2;
                        const unsigned int n0 = nl & LO4, n1 = (nl >> 4) & LO4;
                        const unsigned int n2 = nh & LO4, n3 = (nh >> 4) & LO4;
#pragma unroll
                        for (int t = 0; t < 4; t++) {
                            const int tt = warp * 4 + t;
                            const unsigned int a0 = xs[tt][(q0 + 0) * 8 + k], a1 = xs[tt][(q0 + 1) * 8 + k];
                            const unsigned int a2 = xs[tt][(q0 + 2) * 8 + k], a3 = xs[tt][(q0 + 3) * 8 + k];
                            d[t][0] = sml_dp4a_us(d[t][0], e0, a0); b[t][0] = sml_dp4a_us(b[t][0], n0, a0);
                            d[t][1] = sml_dp4a_us(d[t][1], e1, a1); b[t][1] = sml_dp4a_us(b[t][1], n1, a1);
                            d[t][2] = sml_dp4a_us(d[t][2], e2, a2); b[t][2] = sml_dp4a_us(b[t][2], n2, a2);
                            d[t][3] = sml_dp4a_us(d[t][3], e3, a3); b[t][3] = sml_dp4a_us(b[t][3], n3, a3);
                        }
                    }
#pragma unroll
                    for (int t = 0; t < 4; t++) {
                        const int tt = warp * 4 + t;
#pragma unroll
                        for (int g = 0; g < 4; g++) {
                            const int gg = q0 + g;
                            const float dsum = (float)((int) b[t][g] + 16 * (int) d[t][g]);
                            acc[i][t] += wsc[lr][gg] * xsc[tt][gg] * dsum;
                            if (hm) acc[i][t] += wmn[lr][gg] * xsc[tt][gg] * (float) xsm[tt][gg];
                        }
                    }
                }
            }
        }
        SYNC();
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = r0 + lane * 2 + i;
        if (r >= rows) continue;
        const float bv = hb ? bias[r] : 0.0f;
#pragma unroll
        for (int t = 0; t < 4; t++) {
            const int tt = warp * 4 + t;
            if (tt < nb) y[(long long) tt * yst + r] = acc[i][t] + bv;
        }
    }
}

// h = silu(g) * u, or gelu(u) without a gate: the FFN activation for the whole
// batch, after the tiled kernel has produced gate and up separately.
extern "C" __attribute__((global)) void sml_g_actb(
        float * h, const float * g, int hst, int rows, int nb, int has_gate) {
    const int r = CTA_X() * NTID_X() + TID_X();
    const int t = CTA_Y();
    if (r >= rows || t >= nb) return;
    const long long i = (long long) t * hst + r;
    const float uu = h[i];
    if (has_gate) {
        const float gg = g[i];
        h[i] = sml_gated(gg, uu);
    } else {
        h[i] = sml_ungated(uu);
    }
}


// ============================================================ tensor-core prompt matmul
//
// One mma.sync.m16n8k32 multiplies a 16-row x 32-deep int8 tile by a 32 x 8 tile:
// and 32 deep is exactly one of our groups, so one instruction computes one
// group's integer dot for 16 rows x 8 tokens, and the per-group float epilogue
// (weight scale x activation scale, plus the min term) stays exactly as before.
// One such instruction replaces 1,024 dp4a.
//
// The format fits the fragments with no shuffling. For the A operand (weights,
// row-major) thread (gid = lane/4, tig = lane%4) holds four consecutive codes of
// row gid at depth 4*tig..4*tig+3 -- and in the flat 4-bit layout those four codes
// are the low (or high) nibbles of ONE 32-bit word, so a mask builds the register.
// Six-bit codes are that nibble plus two bits from the second plane, shifted up.
// The B operand (activations, column-major) is word tig of the token's group,
// already four int8 in a register's worth.
//
// Shared-memory pitches are picked so every fragment load is conflict-free: a warp
// reads rows gid = 0..7 at words tig = 0..3, and with pitch 36 (4-bit), 20 (2-bit)
// and 68 (activations) the bank gid*pitch + tig mod 32 is distinct for all 32 lanes.
#define TC_BM  64          // rows per block: 4 warps x one m16 tile each
// 64 tokens, not 16: every A fragment a warp loads is reused across eight n8 tiles
// instead of two, weights are re-read a quarter as often, and there is four times
// the work between barriers. At 16 the kernel was latency-bound (profiled).
#define TC_BN  64          // tokens: eight n8 tiles
#define TC_NT  (TC_BN / 8)
#define TC_KG  8           // groups per chunk (one fp16 super-scale)
#define TC_P4  36          // 4-bit plane pitch in words (32 + 4)
#define TC_P2  20          // 2-bit plane pitch (16 + 4)
#define TC_PX  68          // activation pitch (64 + 4)

__attribute__((device, always_inline)) static inline void mma_s8(
        int & d0, int & d1, int & d2, int & d3,
        unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
        unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The four int8 codes of `row`, group gg of the chunk, depth word kw (0..7).
__attribute__((device, always_inline)) static inline unsigned int tc_codes(
        const unsigned int (*w4)[TC_P4], const unsigned int (*w2)[TC_P2],
        int row, int gg, int kw, int six) {
    if (!six) {
        const unsigned int w = w4[row][(gg >> 1) * 8 + kw];
        return (gg & 1) ? (w >> 4) & LO4 : w & LO4;
    }
    const int pk = gg >> 2, gq = gg & 3;
    const unsigned int nw = w4[row][pk * 16 + (gq >> 1) * 8 + kw];
    const unsigned int nib = (gq & 1) ? (nw >> 4) & LO4 : nw & LO4;
    const unsigned int two = (w2[row][pk * 8 + kw] >> (2 * gq)) & LO2;
    return nib | (two << 4);
}

extern "C" __attribute__((global)) void sml_g_tcm(
        float * y, int yst, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const float * bias,
        const unsigned int * words, int wst, const float * scale, const int * sums, int sst,
        int n, int rows, int six, int hm, int hb, int nb) {

    __attribute__((shared)) unsigned int w4s[TC_BM][TC_P4];
    __attribute__((shared)) unsigned int w2s[TC_BM][TC_P2];
    __attribute__((shared)) unsigned int xs[TC_BN][TC_PX];
    __attribute__((shared)) float        xsc[TC_BN][TC_KG];
    __attribute__((shared)) int          xsm[TC_BN][TC_KG];
    __attribute__((shared)) float        wsc[TC_BM][TC_KG];
    __attribute__((shared)) float        wmn[TC_BM][TC_KG];

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int r0 = CTA_X() * TC_BM;
    const int wr = warp * 16;                        // this warp's 16 rows within the block
    const int ng = n / GROUP, nscale = (ng + GPS - 1) / GPS;
    const long long rb4 = n / 2, rb2 = n / 4;
    const int nchunk = (ng + TC_KG - 1) / TC_KG;

    // acc[nt][q]: n-tile nt (tokens 0-7 / 8-15), q = C-fragment slot
    //   q0 (row gid,   token 2*tig)   q1 (row gid,   token 2*tig+1)
    //   q2 (row gid+8, token 2*tig)   q3 (row gid+8, token 2*tig+1)
    float acc[TC_NT][4];
#pragma unroll
    for (int a = 0; a < TC_NT; a++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[a][q] = 0.0f;

    for (int c = 0; c < nchunk; c++) {
        const int g0 = c * TC_KG;
        const int gn = ng - g0 < TC_KG ? ng - g0 : TC_KG;
        for (int i = tid; i < TC_BM * 32; i += 128) {
            const int r = i >> 5, w = i & 31;
            const int ok = r0 + r < rows && w < gn * 4;
            w4s[r][w] = ok ? *(const unsigned int *)(c4 + (long long)(r0 + r) * rb4 + (long long) g0 * 16 + w * 4) : 0u;
        }
        if (six)
            for (int i = tid; i < TC_BM * 16; i += 128) {
                const int r = i >> 4, w = i & 15;
                const int ok = r0 + r < rows && w < gn * 2;
                w2s[r][w] = ok ? *(const unsigned int *)(c2 + (long long)(r0 + r) * rb2 + (long long) g0 * 8 + w * 4) : 0u;
            }
        for (int i = tid; i < TC_BM * TC_KG; i += 128) {
            const int r = i / TC_KG, g = i % TC_KG;
            float s = 0.0f, m = 0.0f;
            if (r0 + r < rows && g < gn) {
                const long long rr = r0 + r;
                s = h2f(sd[rr * nscale + c]) * (float) sq[rr * ng + g0 + g];
                if (hm) m = h2f(md[rr * nscale + c]) * ((float) mq[rr * ng + g0 + g] - 127.0f);
            }
            wsc[r][g] = s;
            wmn[r][g] = m;
        }
        for (int i = tid; i < TC_BN * 64; i += 128) {
            const int t = i >> 6, w = i & 63;
            xs[t][w] = (t < nb && w < gn * 8) ? words[(long long) t * wst + (long long) g0 * 8 + w] : 0u;
        }
        for (int i = tid; i < TC_BN * TC_KG; i += 128) {
            const int t = i / TC_KG, g = i % TC_KG;
            const int ok = t < nb && g < gn;
            xsc[t][g] = ok ? scale[(long long) t * sst + g0 + g] : 0.0f;
            xsm[t][g] = ok ? sums[(long long) t * sst + g0 + g] : 0;
        }
        SYNC();

#pragma unroll
        for (int gg = 0; gg < TC_KG; gg++) {
            // A: rows wr+gid and wr+gid+8, depth words tig and 4+tig of group gg
            const unsigned int a0 = tc_codes(w4s, w2s, wr + gid,     gg, tig,     six);
            const unsigned int a1 = tc_codes(w4s, w2s, wr + gid + 8, gg, tig,     six);
            const unsigned int a2 = tc_codes(w4s, w2s, wr + gid,     gg, 4 + tig, six);
            const unsigned int a3 = tc_codes(w4s, w2s, wr + gid + 8, gg, 4 + tig, six);
            const float s_lo = wsc[wr + gid][gg], s_hi = wsc[wr + gid + 8][gg];
            const float m_lo = wmn[wr + gid][gg], m_hi = wmn[wr + gid + 8][gg];
#pragma unroll
            for (int nt = 0; nt < TC_NT; nt++) {
                // B: token nt*8+gid, depth words tig and 4+tig
                const unsigned int b0 = xs[nt * 8 + gid][gg * 8 + tig];
                const unsigned int b1 = xs[nt * 8 + gid][gg * 8 + 4 + tig];
                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                const int t0 = nt * 8 + 2 * tig, t1 = t0 + 1;
                const float x0 = xsc[t0][gg], x1 = xsc[t1][gg];
                acc[nt][0] += s_lo * x0 * (float) d0;
                acc[nt][1] += s_lo * x1 * (float) d1;
                acc[nt][2] += s_hi * x0 * (float) d2;
                acc[nt][3] += s_hi * x1 * (float) d3;
                if (hm) {
                    const float y0 = x0 * (float) xsm[t0][gg], y1 = x1 * (float) xsm[t1][gg];
                    acc[nt][0] += m_lo * y0;
                    acc[nt][1] += m_lo * y1;
                    acc[nt][2] += m_hi * y0;
                    acc[nt][3] += m_hi * y1;
                }
            }
        }
        SYNC();
    }

#pragma unroll
    for (int nt = 0; nt < TC_NT; nt++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int r = r0 + wr + gid + (q >= 2 ? 8 : 0);
            const int t = nt * 8 + 2 * tig + (q & 1);
            if (r < rows && t < nb) y[(long long) t * yst + r] = acc[nt][q] + (hb ? bias[r] : 0.0f);
        }
}


// ============================================================ batched attention, streamed
//
// sml_g_attnb gives every (head, token) its own block, and each block reads the
// whole K and V history by itself: at 512 tokens on the 1B that is ~2 GB of cache
// reads per layer, 27% of prompt time (profiled). Here a block owns ONE kv head
// and FA_QV query vectors that share it (GQA heads, consecutive tokens), streams
// K and V through shared memory once per FA_KT keys, and keeps an online softmax
// per query in registers -- the running max, the running sum and the output are
// rescaled as each tile arrives, so no score row is ever stored.
//
// Mapping: warp w owns query vectors 4w..4w+3. For scores, lane j is key j of the
// tile; K is stored TRANSPOSED in shared (kt[i][j]) so the 32 lanes reading key j
// at dimension i hit 32 consecutive banks, and the query is a broadcast. For the
// output, lane l owns dimensions l, l+32, ... and reads V rows, again consecutive.
#define FA_QV   16          // query vectors per block (4 warps x 4)
#define FA_KT   32          // keys per tile (one per lane)
#define FA_HD   128         // largest head this kernel takes

extern "C" __attribute__((global)) void sml_g_fattn(
        float * oall, int ost, const float * qall, int qst,
        const float * kc, const float * vc,
        int n_head, int n_kv, int hd, int n_ctx, int swa, float inv, int pos0, int nb) {

    __attribute__((shared)) float qs[FA_QV][FA_HD];
    // Pitch FA_KT + 1: staging writes kt[d][j] with consecutive threads on
    // consecutive d, which at pitch 32 is one bank for all 32 lanes -- a 32-way
    // conflict on every store (measured: the kernel was no faster than the one
    // it replaced). At 33 the store is conflict-free and lane-per-key reads stay so.
    __attribute__((shared)) float kt[FA_HD][FA_KT + 1];
    __attribute__((shared)) float vs[FA_KT][FA_HD];

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int kvh = CTA_X();
    const int rep = n_head / n_kv;
    const int qv0 = CTA_Y() * FA_QV;                 // first query vector of this block
    const int nqv = nb * rep;                        // query vectors in the batch for this kv head
    const long long kvd = (long long) n_kv * hd;
    const int dper = (hd + 31) / 32;                 // output dims per lane: 2 (hd 64) or 4 (hd 128)

    // query vector v -> (token, head): consecutive vectors are the rep heads of a token
    for (int i = tid; i < FA_QV * hd; i += NTID_X()) {
        const int v = i / hd, d = i % hd, qv = qv0 + v;
        float x = 0.0f;
        if (qv < nqv) {
            const int tok = qv / rep, head = kvh * rep + qv % rep;
            x = qall[(long long) tok * qst + (long long) head * hd + d];
        }
        qs[v][d] = x;
    }

    float m[4], l[4], o[4][4];
    int qpos[4], qlo[4];
#pragma unroll
    for (int a = 0; a < 4; a++) {
        m[a] = -3.4e38f; l[a] = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; k++) o[a][k] = 0.0f;
        const int qv = qv0 + warp * 4 + a;
        const int tok = qv < nqv ? qv / rep : 0;
        qpos[a] = qv < nqv ? pos0 + tok : -1;        // -1: no query here, every key masked
        qlo[a]  = (swa && qpos[a] >= swa) ? qpos[a] - swa + 1 : 0;
    }
    // keys this block needs: from the earliest window start to its last query
    const int last_tok = (qv0 + FA_QV - 1 < nqv ? qv0 + FA_QV - 1 : nqv - 1) / rep;
    const int kend = pos0 + last_tok;                // inclusive
    int kbeg = 0;
    if (swa) { const int first = pos0 + qv0 / rep; kbeg = first >= swa ? first - swa + 1 : 0; }
    SYNC();

    for (int k0 = kbeg; k0 <= kend; k0 += FA_KT) {
        // stage K (transposed) and V for keys k0 .. k0+31
        for (int i = tid; i < FA_KT * hd; i += NTID_X()) {
            const int j = i / hd, d = i % hd, t = k0 + j;
            const int ok = t <= kend;
            const long long off = (long long) t * kvd + (long long) kvh * hd + d;
            kt[d][j] = ok ? kc[off] : 0.0f;
            vs[j][d] = ok ? vc[off] : 0.0f;
        }
        SYNC();
        const int t = k0 + lane;                     // this lane's key
        // The four queries of this warp are scored TOGETHER: each key element is
        // loaded once and feeds four independent FMA chains, where scoring them one
        // after another left a single dependent chain per query and the warp waited
        // on it (the kernel was latency-bound, profiled). Each query's sum is still
        // taken over d in the same order, so the results are bit-identical.
        float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int d = 0; d < hd; d++) {
            const float kv = kt[d][lane];
#pragma unroll
            for (int a = 0; a < 4; a++) s[a] += qs[warp * 4 + a][d] * kv;
        }
        float p[4];
#pragma unroll
        for (int a = 0; a < 4; a++) {
            const int valid = t <= qpos[a] && t >= qlo[a];
            const float sa = valid ? sml_score(s[a] * inv, kvh * rep + (qv0 + warp * 4 + a) % rep, t, qpos[a])
                                   : -3.4e38f;
            const float tmax = wmax(sa);
            const float mnew = m[a] > tmax ? m[a] : tmax;
            p[a] = 0.0f;
            if (mnew <= -3.0e38f) continue;          // nothing valid yet (warp-uniform)
            const float scl = sml_exp(m[a] - mnew);
            p[a] = valid ? sml_exp(sa - mnew) : 0.0f;
            l[a] = l[a] * scl + wsum(p[a]);
#pragma unroll
            for (int k = 0; k < 4; k++) o[a][k] *= scl;
            m[a] = mnew;
        }
        // V rows likewise: loaded once per key, applied to all four queries
        for (int j = 0; j < FA_KT; j++) {
            float vv[4];
#pragma unroll
            for (int k = 0; k < 4; k++) vv[k] = k < dper ? vs[j][lane + 32 * k] : 0.0f;
#pragma unroll
            for (int a = 0; a < 4; a++) {
                const float pj = __nvvm_shfl_sync_idx_f32(0xffffffffu, p[a], (unsigned) j, 31);
                if (pj != 0.0f) {
#pragma unroll
                    for (int k = 0; k < 4; k++)
                        if (k < dper) o[a][k] += pj * vv[k];
                }
            }
        }
        SYNC();
    }

#pragma unroll
    for (int a = 0; a < 4; a++) {
        const int qv = qv0 + warp * 4 + a;
        if (qv >= nqv) continue;
        const int tok = qv / rep, head = kvh * rep + qv % rep;
        const float il = l[a] > 0.0f ? 1.0f / l[a] : 0.0f;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int d = lane + 32 * k;
            if (k < dper && d < hd) oall[(long long) tok * ost + (long long) head * hd + d] = o[a][k] * il;
        }
    }
}
extern "C" __attribute__((global)) void sml_g_quantsbb(
        const float * xall, int xs, const int * perm, unsigned int * wall, int wst,
        float * sall, int * mall, int sst, int n) {
    const int slot = CTA_Y();
    const int lane = TID_X() & (WARP - 1);
    const int ng = n / GROUP, nsb = (ng + 7) / 8;
    const int sb = CTA_X() * (NTID_X() / WARP) + TID_X() / WARP;
    if (sb < nsb)
        quant_sb_warp(xall + (long long) slot * xs, perm, sb, ng,
                      wall + (long long) slot * wst, sall + (long long) slot * sst,
                      mall + (long long) slot * sst, lane);
}

extern "C" __attribute__((global)) void sml_g_tcm2(
        float * y, int yst, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const float * bias,
        const unsigned int * words, int wst, const float * scale, const int * sums, int sst,
        int n, int rows, int six, int hm, int hb, int nb) {

    // 16-byte aligned: staged with 16-byte vector stores, at pitches of 144, 80 and
    // 272 bytes (all multiples of 16). A shared uint array is only 4-aligned otherwise.
    __attribute__((shared, aligned(16))) unsigned int w4s[TC_BM][TC_P4];
    __attribute__((shared, aligned(16))) unsigned int w2s[TC_BM][TC_P2];
    __attribute__((shared, aligned(16))) unsigned int xs[TC2_BN][TC_PX];
    __attribute__((shared)) float        xsc[TC2_BN];          // one scale per token per chunk
    __attribute__((shared)) int          xsm[TC2_BN][TC_KG];
    __attribute__((shared)) int          sqi[TC_BM][TC_KG];    // integer group scales
    __attribute__((shared)) int          mqi[TC_BM][TC_KG];    // integer group mins (unbiased)
    __attribute__((shared)) float        sdf[TC_BM];           // fp16 super-scales, as float
    __attribute__((shared)) float        mdf[TC_BM];

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int r0 = CTA_X() * TC_BM;
    const int t0b = CTA_Y() * TC2_BN;                 // first token of this block
    const int wr = warp * 16;
    const int ng = n / GROUP, nscale = (ng + GPS - 1) / GPS;
    const long long rb4 = n / 2, rb2 = n / 4;
    const int nchunk = (ng + TC_KG - 1) / TC_KG;

    float acc[TC2_NT][4];
#pragma unroll
    for (int a = 0; a < TC2_NT; a++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[a][q] = 0.0f;

    for (int c = 0; c < nchunk; c++) {
        const int g0 = c * TC_KG;
        const int gn = ng - g0 < TC_KG ? ng - g0 : TC_KG;
        // weights: 16-byte loads (a row's chunk is 128 contiguous bytes), stored
        // at a 16-byte-aligned pitch
        for (int i = tid; i < TC_BM * 8; i += 128) {
            const int r = i >> 3, q4 = i & 7;
            uint4v v = { 0u, 0u, 0u, 0u };
            if (r0 + r < rows && q4 * 4 < gn * 4)
                v = *(const uint4v *)(c4 + (long long)(r0 + r) * rb4 + (long long) g0 * 16 + q4 * 16);
            *(uint4v *) &w4s[r][q4 * 4] = v;
        }
        if (six)
            for (int i = tid; i < TC_BM * 4; i += 128) {
                const int r = i >> 2, q4 = i & 3;
                uint4v v = { 0u, 0u, 0u, 0u };
                if (r0 + r < rows && q4 * 4 < gn * 2)
                    v = *(const uint4v *)(c2 + (long long)(r0 + r) * rb2 + (long long) g0 * 8 + q4 * 16);
                *(uint4v *) &w2s[r][q4 * 4] = v;
            }
        for (int i = tid; i < TC_BM * TC_KG; i += 128) {
            const int r = i / TC_KG, g = i % TC_KG;
            int s = 0, m = 0;
            if (r0 + r < rows && g < gn) {
                const long long rr = r0 + r;
                s = sq[rr * ng + g0 + g];
                if (hm) m = (int) mq[rr * ng + g0 + g] - 127;
            }
            sqi[r][g] = s;
            mqi[r][g] = m;
        }
        for (int r = tid; r < TC_BM; r += 128) {
            const int ok = r0 + r < rows;
            sdf[r] = ok ? h2f(sd[(long long)(r0 + r) * nscale + c]) : 0.0f;
            mdf[r] = ok && hm ? h2f(md[(long long)(r0 + r) * nscale + c]) : 0.0f;
        }
        // activations: 16-byte loads, 16 words per token per chunk... 64 words
        for (int i = tid; i < TC2_BN * 16; i += 128) {
            const int t = i >> 4, q4 = i & 15;
            uint4v v = { 0u, 0u, 0u, 0u };
            if (t0b + t < nb && q4 * 4 < gn * 8)
                v = *(const uint4v *)(words + (long long)(t0b + t) * wst + (long long) g0 * 8 + q4 * 4);
            *(uint4v *) &xs[t][q4 * 4] = v;
        }
        for (int i = tid; i < TC2_BN * TC_KG; i += 128) {
            const int t = i / TC_KG, g = i % TC_KG;
            const int ok = t0b + t < nb && g < gn;
            xsm[t][g] = ok ? sums[(long long)(t0b + t) * sst + g0 + g] : 0;
            if (g == 0) xsc[t] = ok ? scale[(long long)(t0b + t) * sst + g0] : 0.0f;
        }
        SYNC();

        int iacc[TC2_NT][4], imin[TC2_NT][4];
#pragma unroll
        for (int a = 0; a < TC2_NT; a++)
#pragma unroll
            for (int q = 0; q < 4; q++) { iacc[a][q] = 0; imin[a][q] = 0; }

#pragma unroll
        for (int gg = 0; gg < TC_KG; gg++) {
            const unsigned int a0 = tc_codes(w4s, w2s, wr + gid,     gg, tig,     six);
            const unsigned int a1 = tc_codes(w4s, w2s, wr + gid + 8, gg, tig,     six);
            const unsigned int a2 = tc_codes(w4s, w2s, wr + gid,     gg, 4 + tig, six);
            const unsigned int a3 = tc_codes(w4s, w2s, wr + gid + 8, gg, 4 + tig, six);
            const int s_lo = sqi[wr + gid][gg], s_hi = sqi[wr + gid + 8][gg];
            const int m_lo = mqi[wr + gid][gg], m_hi = mqi[wr + gid + 8][gg];
#pragma unroll
            for (int nt = 0; nt < TC2_NT; nt++) {
                const unsigned int b0 = xs[nt * 8 + gid][gg * 8 + tig];
                const unsigned int b1 = xs[nt * 8 + gid][gg * 8 + 4 + tig];
                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                iacc[nt][0] += d0 * s_lo;
                iacc[nt][1] += d1 * s_lo;
                iacc[nt][2] += d2 * s_hi;
                iacc[nt][3] += d3 * s_hi;
                if (hm) {
                    const int t0 = nt * 8 + 2 * tig;
                    const int u0 = xsm[t0][gg], u1 = xsm[t0 + 1][gg];
                    imin[nt][0] += m_lo * u0;
                    imin[nt][1] += m_lo * u1;
                    imin[nt][2] += m_hi * u0;
                    imin[nt][3] += m_hi * u1;
                }
            }
        }
        // once per superblock: the float scales
        const float d_lo = sdf[wr + gid], d_hi = sdf[wr + gid + 8];
        const float n_lo = mdf[wr + gid], n_hi = mdf[wr + gid + 8];
#pragma unroll
        for (int nt = 0; nt < TC2_NT; nt++) {
            const int t0 = nt * 8 + 2 * tig;
            const float x0 = xsc[t0], x1 = xsc[t0 + 1];
            acc[nt][0] += x0 * (d_lo * (float) iacc[nt][0] + n_lo * (float) imin[nt][0]);
            acc[nt][1] += x1 * (d_lo * (float) iacc[nt][1] + n_lo * (float) imin[nt][1]);
            acc[nt][2] += x0 * (d_hi * (float) iacc[nt][2] + n_hi * (float) imin[nt][2]);
            acc[nt][3] += x1 * (d_hi * (float) iacc[nt][3] + n_hi * (float) imin[nt][3]);
        }
        SYNC();
    }

#pragma unroll
    for (int nt = 0; nt < TC2_NT; nt++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int r = r0 + wr + gid + (q >= 2 ? 8 : 0);
            const int t = t0b + nt * 8 + 2 * tig + (q & 1);
            if (r < rows && t < nb) y[(long long) t * yst + r] = acc[nt][q] + (hb ? bias[r] : 0.0f);
        }
}



// ============================================================ f16 weight tiles
//
// sml_g_tcm2 runs int8 MMAs, and every one of them carries its own epilogue --
// the nibble decode, the group scale, the group min. Profiled at pp512 that left
// the tensor cores waiting on ~20 integer instructions per MMA: the matmuls ran
// at about a tenth of the GPU's int8 peak. Here the weight tile is dequantized
// ONCE per block into f16 in shared memory, every scale and min already applied,
// and the block's tokens run f16 MMAs over it with nothing in between. The
// activations are f16 as well, so they are not rounded to int8 at all.
//
// Tile: 64 rows x 64 tokens x 128 k per chunk. Warp w owns rows 16w..16w+15 and
// all eight n8 token tiles. Shared rows are 68 words (64 + 4): the fragment loads
// then fall in 32 distinct banks.
#define HM_BM  64
#define HM_BN  64
#define HM_KC  128
#define HM_P   68

__attribute__((device, always_inline)) static inline unsigned int pack_h2(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__attribute__((device, always_inline)) static inline void mma_f16(
        float & d0, float & d1, float & d2, float & d3,
        unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
        unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// f16 activations in each tensor's column order: h[t][k] = x[t][perm[k]], two per
// word. One launch per distinct permutation (q/k/v and gate/up share one).
extern "C" __attribute__((global)) void sml_g_permh(
        const float * x, int xst, const int * perm, unsigned int * h, int hst, int n) {
    const int t = CTA_Y();
    const int k = 2 * (CTA_X() * NTID_X() + TID_X());
    if (k >= n) return;
    const float * xr = x + (long long) t * xst;
    h[(long long) t * hst + k / 2] = pack_h2(xr[perm[k]], xr[perm[k + 1]]);
}

extern "C" __attribute__((global)) void sml_g_hmm(
        float * y, int yst, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const float * bias,
        const unsigned int * xh, int xhst, int n, int rows, int hm, int hb, int nb) {

    __attribute__((shared, aligned(16))) unsigned int ws[HM_BM][HM_P];
    __attribute__((shared, aligned(16))) unsigned int xs[HM_BN][HM_P];

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int r0 = CTA_X() * HM_BM, t0 = CTA_Y() * HM_BN;
    const int wr = warp * 16;
    const int ng = n / GROUP, nscale = (ng + GPS - 1) / GPS;
    const long long rb4 = n / 2;

    float acc[8][4];
#pragma unroll
    for (int a = 0; a < 8; a++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[a][q] = 0.0f;

    for (int k0 = 0; k0 < n; k0 += HM_KC) {
        // Weights: thread -> (row, block of two groups). A block is 32 bytes: byte j
        // holds column j of the first group (low nibble) and of the second (high).
        {
            const int r = tid >> 1, blk = tid & 1;
            const int g0 = k0 / GROUP + 2 * blk;          // first of the two groups
            unsigned int * dst = &ws[r][blk * 32];
            if (r0 + r < rows) {
                const long long rr = r0 + r;
                const uint4v * src = (const uint4v *)(c4 + rr * rb4 + (long long) k0 / 2 + blk * 32);
                const uint4v v0 = src[0], v1 = src[1];
                const float d = h2f(sd[rr * nscale + g0 / GPS]);
                const float s_lo = d * (float) sq[rr * ng + g0], s_hi = d * (float) sq[rr * ng + g0 + 1];
                float m_lo = 0.0f, m_hi = 0.0f;
                if (hm) {
                    const float dm = h2f(md[rr * nscale + g0 / GPS]);
                    m_lo = dm * (float) ((int) mq[rr * ng + g0] - 127);
                    m_hi = dm * (float) ((int) mq[rr * ng + g0 + 1] - 127);
                }
                const unsigned int wv[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
#pragma unroll
                for (int q = 0; q < 8; q++) {
                    const unsigned int u = wv[q];               // bytes 4q .. 4q+3
                    const float l0 = (float) (u & 15u),         h0 = (float) ((u >> 4) & 15u);
                    const float l1 = (float) ((u >> 8) & 15u),  h1 = (float) ((u >> 12) & 15u);
                    const float l2 = (float) ((u >> 16) & 15u), h2 = (float) ((u >> 20) & 15u);
                    const float l3 = (float) ((u >> 24) & 15u), h3 = (float) (u >> 28);
                    // first group: columns 4q..4q+3 -> words 2q, 2q+1
                    dst[2 * q]          = pack_h2(s_lo * l0 + m_lo, s_lo * l1 + m_lo);
                    dst[2 * q + 1]      = pack_h2(s_lo * l2 + m_lo, s_lo * l3 + m_lo);
                    // second group: 32 columns later -> 16 words later
                    dst[16 + 2 * q]     = pack_h2(s_hi * h0 + m_hi, s_hi * h1 + m_hi);
                    dst[16 + 2 * q + 1] = pack_h2(s_hi * h2 + m_hi, s_hi * h3 + m_hi);
                }
            } else {
#pragma unroll
                for (int q = 0; q < 32; q++) dst[q] = 0u;
            }
        }
        // Activations: 64 tokens x 64 words, 16-byte copies.
        for (int i = tid; i < HM_BN * 16; i += 128) {
            const int t = i >> 4, q4 = i & 15;
            uint4v v = { 0u, 0u, 0u, 0u };
            if (t0 + t < nb)
                v = *(const uint4v *)(xh + (long long) (t0 + t) * xhst + k0 / 2 + q4 * 4);
            *(uint4v *) &xs[t][q4 * 4] = v;
        }
        SYNC();

#pragma unroll
        for (int ks = 0; ks < HM_KC / 16; ks++) {
            const int wb = ks * 8;
            const unsigned int a0 = ws[wr + gid][wb + tig],     a1 = ws[wr + gid + 8][wb + tig];
            const unsigned int a2 = ws[wr + gid][wb + 4 + tig], a3 = ws[wr + gid + 8][wb + 4 + tig];
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const unsigned int b0 = xs[nt * 8 + gid][wb + tig];
                const unsigned int b1 = xs[nt * 8 + gid][wb + 4 + tig];
                mma_f16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a0, a1, a2, a3, b0, b1);
            }
        }
        SYNC();
    }

#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int r = r0 + wr + gid + (q >= 2 ? 8 : 0);
            const int t = t0 + nt * 8 + 2 * tig + (q & 1);
            if (r < rows && t < nb) y[(long long) t * yst + r] = acc[nt][q] + (hb ? bias[r] : 0.0f);
        }
}



// ============================================================ int8 tiles, one scale per 256
//
// sml_g_tcm2 feeds the tensor cores 4-bit codes, so every int8 MMA (one 32-wide
// group) is followed by the group's scale, the group's min and a nibble decode
// -- ~20 integer instructions per MMA, which left the matmuls at ~10% of int8
// peak. f16 tiles removed the epilogue but run at a quarter of the int8 rate on
// this GPU (measured 0.77-0.82x). So here the weights are re-quantized, per
// call, to plain int8 with ONE float scale per row per 256 weights: the group
// scale and the group min are folded into the int8 value. The eight MMAs of a
// superblock then accumulate in int32 with nothing between them, and the float
// epilogue runs once per superblock. The extra rounding is at most half a step
// of 1/127 of the superblock's largest weight -- well under the 4-bit step.
#define T3_BM  64
#define T3_P   68            // shared pitch in words: 256 bytes + 16

// Integer <-> float without the conversion unit, which runs at an eighth of the
// FMA rate on this GPU: a small code c is the float 2^23 + c minus 2^23, and
// adding 1.5 * 2^23 rounds to the nearest integer (ties to even, as rint) and
// leaves it in the low mantissa bits. Exact for |x| < 2^22.
__attribute__((device, always_inline)) static inline float code_f(unsigned int c) {
    return __builtin_bit_cast(float, 0x4B000000u | c) - 8388608.0f;
}
__attribute__((device, always_inline)) static inline int round_i(float x) {
    return __builtin_bit_cast(int, x + 12582912.0f) - 0x4B400000;
}

// Four roundings packed to four int8 bytes. After x + 1.5*2^23 the integer sits in
// the low mantissa bits and 0x4B400000 has a zero low byte, so the byte of the
// float's bits IS round_i(x) & 0xff -- no subtraction, no mask, no shift-or: three
// byte permutes (prmt) replace twelve integer instructions per four weights. The
// bytes are round_i's exactly, so every re-made tile is unchanged bit for bit.
__attribute__((device, always_inline)) static inline unsigned int pack4_rnd(
        float v0, float v1, float v2, float v3) {
    const unsigned int a = __builtin_bit_cast(unsigned int, v0 + 12582912.0f);
    const unsigned int b = __builtin_bit_cast(unsigned int, v1 + 12582912.0f);
    const unsigned int c = __builtin_bit_cast(unsigned int, v2 + 12582912.0f);
    const unsigned int d = __builtin_bit_cast(unsigned int, v3 + 12582912.0f);
    unsigned int ab, cd, r;
    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(ab) : "r"(a), "r"(b));   // a0 b0 . .
    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(cd) : "r"(c), "r"(d));   // c0 d0 . .
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(r) : "r"(ab), "r"(cd));  // a0 b0 c0 d0
    return r;
}

// One warp per (row, superblock), every lane busy: lane l owns columns 8(l%4)..
// 8(l%4)+7 of group l/4. The superblock's scale needs no pass over the weights:
// a group's weights are s*c + m for codes c in 0..cmax, so their extremes are m
// and s*cmax + m, and eight such bounds set the scale. Each weight is then one
// fused c*a + b and a rounding, with no clamp (the bound guarantees |v| <= 127).
// Written in the tensor's own (permuted) column order -- the order the
// activations arrive in. (The first versions decoded every weight twice, the
// second time to find the max: 602 ms of a 3B prompt, re-made per call.)
extern "C" __attribute__((global)) void sml_g_requant8(
        const unsigned char * c2, const unsigned char * c4, int six,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, int hm,
        signed char * w8, float * s8, int n, int rows) {
    const int lane = TID_X() & 31;
    const int ng = n / GROUP, nsb = (ng + GPS - 1) / GPS;
    const long long item = (long long) CTA_X() * (NTID_X() / 32) + TID_X() / 32;
    if (item >= (long long) rows * nsb) return;
    const int r = (int) (item / nsb), sb = (int) (item % nsb);
    const int g = sb * GPS + (lane >> 2), part = lane & 3;
    const int live = g < ng;
    float s = 0.0f, m = 0.0f, bound = 0.0f;
    if (live) {
        const float d = h2f(sd[(long long) r * nsb + sb]);
        s = d * (float) sq[(long long) r * ng + g];
        m = hm ? h2f(md[(long long) r * nsb + sb]) * (float) ((int) mq[(long long) r * ng + g] - 127) : 0.0f;
        const float top = s * (six ? 63.0f : 15.0f) + m;
        const float am = m < 0.0f ? -m : m, at = top < 0.0f ? -top : top;
        bound = am > at ? am : at;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __nvvm_shfl_sync_bfly_f32(0xffffffffu, bound, (unsigned) o, 31);
        bound = other > bound ? other : bound;
    }
    if (lane == 0) s8[(long long) r * nsb + sb] = bound / 127.0f;
    if (!live) return;
    const float inv = bound > 0.0f ? 127.0f / bound : 0.0f;
    const float ka = s * inv, kb = m * inv;
    // a pair of groups shares 32 bytes: byte j low nibble = even group col j, high = odd
    const unsigned int * src = (const unsigned int *)(c4 + (long long) r * (n / 2) + (long long) (g / 2) * 32 + part * 8);
    const unsigned int u0 = src[0], u1 = src[1];
    const int sh = (g & 1) ? 4 : 0;
    // 6-bit: code = nibble + 16 * (2-bit plane), four groups per byte of that plane
    unsigned int e0 = 0, e1 = 0;
    if (six) {
        const unsigned int * s2 = (const unsigned int *)(c2 + (long long) r * (n / 4) + (long long) (g / 4) * 32 + part * 8);
        e0 = s2[0]; e1 = s2[1];
    }
    const int s2h = 2 * (g & 3);
    float fl[4], fh[4];
#pragma unroll
    for (int b = 0; b < 4; b++) {
        const unsigned int x0 = ((u0 >> (8 * b + sh)) & 15u) | (((e0 >> (8 * b + s2h)) & 3u) << 4);
        const unsigned int x1 = ((u1 >> (8 * b + sh)) & 15u) | (((e1 >> (8 * b + s2h)) & 3u) << 4);
        fl[b] = code_f(x0) * ka + kb;
        fh[b] = code_f(x1) * ka + kb;
    }
    const unsigned int lo = pack4_rnd(fl[0], fl[1], fl[2], fl[3]);
    const unsigned int hi = pack4_rnd(fh[0], fh[1], fh[2], fh[3]);
    unsigned int * dst = (unsigned int *)(w8 + (long long) r * n + (long long) g * GROUP + part * 8);
    dst[0] = lo;
    dst[1] = hi;
}

// y[t][r] = sum over superblocks of s8[r][sb] * xscale[t][sb] * (int8 w . int8 x).
// Tile: 64 rows x 64 tokens (square: the most MMAs per byte staged) x one
// superblock (256) per chunk; warp w owns rows 16w..16w+15. Activations are the
// per-256 int8 words sml_g_quantsbb wrote.
//
// Prefetch: the next superblock's tiles are fetched into registers BEFORE this
// one's MMAs run, and stored to shared after them -- the global latency hides
// behind the tensor cores instead of standing between two barriers.
#ifndef T3_LDM
#define T3_LDM 1           // 0: word-by-word fragment loads (the A/B baseline)
#endif
__attribute__((device, always_inline)) static inline unsigned int smem_u32(const void * p) {
    unsigned int r;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p));
    return r;
}
__attribute__((device, always_inline)) static inline void ldsm_x4(unsigned int addr,
        unsigned int & r0, unsigned int & r1, unsigned int & r2, unsigned int & r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}
#define T3_BN  64
#define T3_NT  (T3_BN / 8)
extern "C" __attribute__((global)) void sml_g_tcm3(
        float * y, int yst, const signed char * w8, const float * s8, const float * bias,
        const unsigned int * words, int wst, const float * scale, int sst,
        int n, int rows, int hb, int nb, float * part, int split, int swap) {

    __attribute__((shared, aligned(16))) unsigned int ws[T3_BM][T3_P];
    __attribute__((shared, aligned(16))) unsigned int xs[T3_BN][T3_P];
    __attribute__((shared)) float wsc[T3_BM];
    __attribute__((shared)) float xsc[T3_BN];

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    // grid.y carries (slice, token block): a deep sum is split across blocks
    // swap: grid.x walks token blocks, so blocks running side by side share one
    // slab of weight rows (reused from L2) instead of one slab of activations
    const int tb = (nb + T3_BN - 1) / T3_BN;
    const int bx = swap ? CTA_Y() : CTA_X(), by = swap ? CTA_X() : CTA_Y();
    const int z = by / tb;
    const int r0 = bx * T3_BM, t0b = (by % tb) * T3_BN;
    const int wr = warp * 16;
    const int ng = n / GROUP, nsb = (ng + GPS - 1) / GPS;
    const int per = (nsb + split - 1) / split;
    const int c0 = z * per, c1 = c0 + per < nsb ? c0 + per : nsb;
#if T3_LDM
    // lane l addresses row l%8 of matrix l/8: A matrices are (rows 0-7 | 8-15) x
    // (bytes 0-15 | 16-31) of a group; B matrices are (bytes 0-15 | 16-31) x
    // (token tile nt | nt+1)
    const unsigned int wsa = smem_u32(&ws[0][0]) +
        (unsigned) (((wr + ((lane >> 3) & 1) * 8 + (lane & 7)) * T3_P + (lane >> 4) * 4) * 4);
    const unsigned int xsa = smem_u32(&xs[0][0]) +
        (unsigned) ((((lane >> 4) * 8 + (lane & 7)) * T3_P + ((lane >> 3) & 1) * 4) * 4);
#endif

    // this thread's share of one chunk: 8 of the 1024 16-byte pieces of each tile
    uint4v pw[8], px[8];
    float psw = 0.0f, psx = 0.0f;

    // fetch chunk c into registers (weights, activations, and one scale each)
#define T3_FETCH(c) do {                                                              \
        const int kw_ = (ng - (c) * GPS < GPS ? ng - (c) * GPS : GPS) * 8;             \
        _Pragma("unroll")                                                              \
        for (int k_ = 0; k_ < 8; k_++) {                                               \
            const int i_ = tid + 128 * k_, row_ = i_ >> 4, q4_ = i_ & 15;              \
            const uint4v z_ = { 0u, 0u, 0u, 0u };                                      \
            pw[k_] = (r0 + row_ < rows && q4_ * 4 < kw_)                               \
                ? *(const uint4v *)(w8 + (long long) (r0 + row_) * n + (long long) (c) * 256 + q4_ * 16) : z_; \
            px[k_] = (t0b + row_ < nb && q4_ * 4 < kw_)                                \
                ? *(const uint4v *)(words + (long long) (t0b + row_) * wst + (long long) (c) * 64 + q4_ * 4) : z_; \
        }                                                                              \
        if (tid < T3_BM) psw = r0 + tid < rows ? s8[(long long) (r0 + tid) * nsb + (c)] : 0.0f; \
        else if (tid < T3_BM + T3_BN)                                                  \
            psx = t0b + tid - T3_BM < nb ? scale[(long long) (t0b + tid - T3_BM) * sst + (c) * GPS] : 0.0f; \
    } while (0)
#define T3_STORE() do {                                                               \
        _Pragma("unroll")                                                              \
        for (int k_ = 0; k_ < 8; k_++) {                                               \
            const int i_ = tid + 128 * k_, row_ = i_ >> 4, q4_ = i_ & 15;              \
            *(uint4v *) &ws[row_][q4_ * 4] = pw[k_];                                   \
            *(uint4v *) &xs[row_][q4_ * 4] = px[k_];                                   \
        }                                                                              \
        if (tid < T3_BM) wsc[tid] = psw;                                               \
        else if (tid < T3_BM + T3_BN) xsc[tid - T3_BM] = psx;                          \
    } while (0)

    float acc[T3_NT][4];
#pragma unroll
    for (int a = 0; a < T3_NT; a++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[a][q] = 0.0f;

    if (c0 < c1) {
        T3_FETCH(c0);
        T3_STORE();
    }
    SYNC();

    for (int c = c0; c < c1; c++) {
        if (c + 1 < c1) T3_FETCH(c + 1);            // in flight while the MMAs run

        int iacc[T3_NT][4];
#pragma unroll
        for (int a = 0; a < T3_NT; a++)
#pragma unroll
            for (int q = 0; q < 4; q++) iacc[a][q] = 0;
#if T3_LDM
        // ldmatrix: one instruction loads four 8x8 fragments -- the A tile of a
        // group in one, the B fragments of two token tiles in another: five loads
        // per eight MMAs where word-by-word loads took twenty.
#pragma unroll
        for (int gg = 0; gg < GPS; gg++) {
            unsigned int a0, a1, a2, a3;
            ldsm_x4(wsa + (unsigned) gg * 32u, a0, a1, a2, a3);
#pragma unroll
            for (int pp = 0; pp < T3_NT / 2; pp++) {
                unsigned int b0, b1, c0, c1;
                ldsm_x4(xsa + (unsigned) (pp * 16 * T3_P * 4) + (unsigned) gg * 32u, b0, b1, c0, c1);
                mma_s8(iacc[2 * pp][0], iacc[2 * pp][1], iacc[2 * pp][2], iacc[2 * pp][3],
                       a0, a1, a2, a3, b0, b1);
                mma_s8(iacc[2 * pp + 1][0], iacc[2 * pp + 1][1], iacc[2 * pp + 1][2], iacc[2 * pp + 1][3],
                       a0, a1, a2, a3, c0, c1);
            }
        }
#else
#pragma unroll
        for (int gg = 0; gg < GPS; gg++) {
            const unsigned int a0 = ws[wr + gid][gg * 8 + tig],     a1 = ws[wr + gid + 8][gg * 8 + tig];
            const unsigned int a2 = ws[wr + gid][gg * 8 + 4 + tig], a3 = ws[wr + gid + 8][gg * 8 + 4 + tig];
#pragma unroll
            for (int nt = 0; nt < T3_NT; nt++) {
                const unsigned int b0 = xs[nt * 8 + gid][gg * 8 + tig];
                const unsigned int b1 = xs[nt * 8 + gid][gg * 8 + 4 + tig];
                mma_s8(iacc[nt][0], iacc[nt][1], iacc[nt][2], iacc[nt][3], a0, a1, a2, a3, b0, b1);
            }
        }
#endif
        const float s_lo = wsc[wr + gid], s_hi = wsc[wr + gid + 8];
#pragma unroll
        for (int nt = 0; nt < T3_NT; nt++) {
            const int t = nt * 8 + 2 * tig;
            const float x0 = xsc[t], x1 = xsc[t + 1];
            acc[nt][0] += s_lo * x0 * (float) iacc[nt][0];
            acc[nt][1] += s_lo * x1 * (float) iacc[nt][1];
            acc[nt][2] += s_hi * x0 * (float) iacc[nt][2];
            acc[nt][3] += s_hi * x1 * (float) iacc[nt][3];
        }
        SYNC();
        if (c + 1 < c1) {
            T3_STORE();
            SYNC();
        }
    }
#undef T3_FETCH
#undef T3_STORE

#pragma unroll
    for (int nt = 0; nt < T3_NT; nt++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int r = r0 + wr + gid + (q >= 2 ? 8 : 0);
            const int t = t0b + nt * 8 + 2 * tig + (q & 1);
            if (r >= rows || t >= nb) continue;
            if (split > 1) part[((long long) z * nb + t) * rows + r] = acc[nt][q];
            else           y[(long long) t * yst + r] = acc[nt][q] + (hb ? bias[r] : 0.0f);
        }
}


// Split-K: tcm3 slices a deep sum across blocks and each slice writes its own
// partial; this adds them in a FIXED order (not float atomics, whose order and so
// whose rounding would change from run to run) and adds the bias once.
extern "C" __attribute__((global)) void sml_g_splitk_sum(
        float * y, int yst, const float * part, const float * bias, int hb,
        int rows, int nb, int split) {
    const int r = CTA_X() * NTID_X() + TID_X(), t = CTA_Y();
    if (r >= rows) return;
    float s = 0.0f;
    for (int z = 0; z < split; z++) s += part[((long long) z * nb + t) * rows + r];
    y[(long long) t * yst + r] = s + (hb ? bias[r] : 0.0f);
}


// ============================================================ attention on tensor cores
//
// sml_g_fattn scores keys with scalar FMAs, and once the matmuls moved to int8
// tiles it was 46% of a 512-token prompt (profiled). Here each warp takes 16
// query vectors as one m16 tile: S = Q K^T with f16 MMAs over the head, an
// online softmax on the S fragments (each row lives on one quad of four lanes),
// and P V with the probabilities fed back as the next MMA's A operand straight
// from registers -- the accumulator layout of m16n8 is the A layout of the next
// k16 step, so no score ever touches shared memory. K and V are staged per 64
// keys as f16; V transposed so its B fragments are contiguous. Head size 64.
#define FC_KT  64
// pitches in words: a row of HD halves is HD/2 words, + 4 keeps the fragment
// loads in 32 distinct banks

template <int HD>
__attribute__((device, always_inline)) static inline void fattn_tc_body(
        float * oall, int ost, const float * qall, int qst,
        const float * kc, const float * vc,
        int n_head, int n_kv, float inv, int pos0, int nb) {

    __attribute__((shared, aligned(16))) unsigned int ks[FC_KT][HD / 2 + 4];     // [key][d/2]
    __attribute__((shared, aligned(16))) unsigned int vt[HD][FC_KT / 2 + 4];     // [d][key/2]

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int kvh = CTA_X();
    const int rep = n_head / n_kv;
    const int nqv = nb * rep;
    const long long kvd = (long long) n_kv * HD;
    const int qw = CTA_Y() * 64 + warp * 16;          // this warp's first query vector

    // the two query rows this lane holds: gid and gid + 8 of the warp's tile
    int qpos[2];
    long long qoff[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int qv = qw + gid + 8 * h;
        if (qv < nqv) {
            const int tok = qv / rep, head = kvh * rep + qv % rep;
            qpos[h] = pos0 + tok;
            qoff[h] = (long long) tok * qst + (long long) head * HD;
        } else {
            qpos[h] = -1;
            qoff[h] = -1;
        }
    }
    // Q fragments for the four k16 steps over the head, held for the whole kernel
    unsigned int qa[HD / 16][4];
#pragma unroll
    for (int s = 0; s < HD / 16; s++) {
        const int d0 = s * 16 + 2 * tig;
        float x[2][4];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            if (qoff[h] >= 0) {
                const float * q = qall + qoff[h];
                x[h][0] = q[d0]; x[h][1] = q[d0 + 1]; x[h][2] = q[d0 + 8]; x[h][3] = q[d0 + 9];
            } else {
                x[h][0] = x[h][1] = x[h][2] = x[h][3] = 0.0f;
            }
        }
        qa[s][0] = pack_h2(x[0][0], x[0][1]);
        qa[s][1] = pack_h2(x[1][0], x[1][1]);
        qa[s][2] = pack_h2(x[0][2], x[0][3]);
        qa[s][3] = pack_h2(x[1][2], x[1][3]);
    }

    float o[HD / 8][4];
#pragma unroll
    for (int a = 0; a < HD / 8; a++)
#pragma unroll
        for (int q = 0; q < 4; q++) o[a][q] = 0.0f;
    float mrow[2] = { -3.0e38f, -3.0e38f }, lrow[2] = { 0.0f, 0.0f };

    // keys this block needs: up to the latest query position among its 64 vectors
    const int last = (CTA_Y() * 64 + 63 < nqv ? CTA_Y() * 64 + 63 : nqv - 1) / rep;
    const int kend = pos0 + last;

    for (int k0 = 0; k0 <= kend; k0 += FC_KT) {
        // stage K [key][d] and V transposed [d][key], as f16 pairs
        for (int i = tid; i < FC_KT * (HD / 2); i += 128) {
            const int j = i / (HD / 2), w = i % (HD / 2);  // key, word (d pair)
            const int t = k0 + j;
            float a0 = 0.0f, a1 = 0.0f;
            if (t <= kend) {
                const float * k = kc + (long long) t * kvd + (long long) kvh * HD + 2 * w;
                a0 = k[0]; a1 = k[1];
            }
            ks[j][w] = pack_h2(a0, a1);
        }
        for (int i = tid; i < HD * (FC_KT / 2); i += 128) {
            const int d = i % HD, w = i / HD;              // dim (lanes), word (key pair)
            const int t = k0 + 2 * w;
            const float * v = vc + (long long) kvh * HD + d;
            const float a0 = t <= kend ? v[(long long) t * kvd] : 0.0f;
            const float a1 = t + 1 <= kend ? v[(long long) (t + 1) * kvd] : 0.0f;
            vt[d][w] = pack_h2(a0, a1);
        }
        SYNC();

        // S = Q K^T for 64 keys: eight n8 tiles
        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
            for (int st = 0; st < HD / 16; st++) {
                const unsigned int b0 = ks[nt * 8 + gid][st * 8 + tig];
                const unsigned int b1 = ks[nt * 8 + gid][st * 8 + 4 + tig];
                mma_f16(s[nt][0], s[nt][1], s[nt][2], s[nt][3],
                        qa[st][0], qa[st][1], qa[st][2], qa[st][3], b0, b1);
            }
        }
        // online softmax, per row; a row's 64 scores live on the four lanes of a quad
#pragma unroll
        for (int h = 0; h < 2; h++) {
            float mx = -3.0e38f;
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int key = k0 + nt * 8 + 2 * tig + e;
                    float v = s[nt][2 * h + e] * inv;
                    if (key > qpos[h]) v = -3.0e38f;         // causal (and padded rows)
                    s[nt][2 * h + e] = v;
                    mx = v > mx ? v : mx;
                }
            { const float u = __nvvm_shfl_sync_bfly_f32(0xffffffffu, mx, 1u, 31); mx = u > mx ? u : mx; }
            { const float u = __nvvm_shfl_sync_bfly_f32(0xffffffffu, mx, 2u, 31); mx = u > mx ? u : mx; }
            const float mnew = mx > mrow[h] ? mx : mrow[h];
            const float alpha = mnew <= -2.0e38f ? 1.0f : sml_exp(mrow[h] - mnew);
            float sum = 0.0f;
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const float v = s[nt][2 * h + e];
                    const float p = v <= -2.0e38f ? 0.0f : sml_exp(v - mnew);
                    s[nt][2 * h + e] = p;
                    sum += p;
                }
            lrow[h] = lrow[h] * alpha + sum;                  // per-lane partial, reduced at the end
            mrow[h] = mnew;
#pragma unroll
            for (int dn = 0; dn < HD / 8; dn++) {
                o[dn][2 * h] *= alpha;
                o[dn][2 * h + 1] *= alpha;
            }
        }
        // O += P V: the S accumulators ARE the A fragments of the four k16 steps
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const unsigned int a0 = pack_h2(s[2 * j][0], s[2 * j][1]);
            const unsigned int a1 = pack_h2(s[2 * j][2], s[2 * j][3]);
            const unsigned int a2 = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
            const unsigned int a3 = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
#pragma unroll
            for (int dn = 0; dn < HD / 8; dn++) {
                const unsigned int b0 = vt[dn * 8 + gid][j * 8 + tig];
                const unsigned int b1 = vt[dn * 8 + gid][j * 8 + 4 + tig];
                mma_f16(o[dn][0], o[dn][1], o[dn][2], o[dn][3], a0, a1, a2, a3, b0, b1);
            }
        }
        SYNC();
    }

#pragma unroll
    for (int h = 0; h < 2; h++) {
        float l = lrow[h];
        l += __nvvm_shfl_sync_bfly_f32(0xffffffffu, l, 1u, 31);
        l += __nvvm_shfl_sync_bfly_f32(0xffffffffu, l, 2u, 31);
        if (qoff[h] < 0 || l <= 0.0f) continue;
        const int qv = qw + gid + 8 * h;
        const int tok = qv / rep, head = kvh * rep + qv % rep;
        float * out = oall + (long long) tok * ost + (long long) head * HD;
        const float il = 1.0f / l;
#pragma unroll
        for (int dn = 0; dn < HD / 8; dn++) {
            out[dn * 8 + 2 * tig]     = o[dn][2 * h] * il;
            out[dn * 8 + 2 * tig + 1] = o[dn][2 * h + 1] * il;
        }
    }
}

extern "C" __attribute__((global)) void sml_g_fattn_tc(
        float * oall, int ost, const float * qall, int qst, const float * kc, const float * vc,
        int n_head, int n_kv, float inv, int pos0, int nb) {
    fattn_tc_body<64>(oall, ost, qall, qst, kc, vc, n_head, n_kv, inv, pos0, nb);
}
extern "C" __attribute__((global)) void sml_g_fattn_tc128(
        float * oall, int ost, const float * qall, int qst, const float * kc, const float * vc,
        int n_head, int n_kv, float inv, int pos0, int nb) {
    fattn_tc_body<128>(oall, ost, qall, qst, kc, vc, n_head, n_kv, inv, pos0, nb);
}



// ============================================================ 4-bit, decoded in passing
//
// Acts 5:15 -- the shadow passing by. A tensor whose int8 copy cannot be kept
// (Qwen2.5-3B in 4 GB) was re-made in a pass of its own on every call: 16-35%
// of its prompt time. Here it is never re-made: the 4-bit codes and their
// scales are staged as they are, decoded to int8 as each A fragment is read,
// and the group scale and min applied per group in integers -- tcm2's arithmetic
// exactly -- inside tcm3's frame: 64 x 64 tiles, the next superblock fetched into
// registers while this one's MMAs run, B fragments by ldmatrix, and the grid
// order that shares weight rows across token blocks.
#define T4_P4 36            // 4-bit codes: 32 words per row per superblock + 4
#define T4_P2 20            // 2-bit plane (6-bit tensors): 16 words + 4

extern "C" __attribute__((global)) void sml_g_tcm4(
        float * y, int yst, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const float * bias,
        const unsigned int * words, int wst, const float * scale, const int * sums, int sst,
        int n, int rows, int six, int hm, int hb, int nb, int swap) {

    __attribute__((shared, aligned(16))) unsigned int w4s[T3_BM][T4_P4];
    __attribute__((shared, aligned(16))) unsigned int w2s[T3_BM][T4_P2];
    __attribute__((shared, aligned(16))) unsigned int xs[T3_BN][T3_P];
    __attribute__((shared)) int   sqi[T3_BM][GPS];
    __attribute__((shared)) float sdf[T3_BM], mdf[T3_BM];
    // Mins on the tensor cores: m' = mq - 128 fits int8 exactly, and eight 1s in the
    // next k slots add sum_g s_g, so the MMA gives sum_g (mq_g - 127) s_g -- the min
    // term, exact. Sums (|s| <= 4064) are split s = 64 hi + lo, both int8.
    __attribute__((shared, aligned(8))) unsigned int mA[T3_BM][2];
    __attribute__((shared, aligned(8))) signed char   sHb[T3_BN][GPS];
    __attribute__((shared, aligned(8))) unsigned char sLb[T3_BN][GPS];
    __attribute__((shared)) float xsc[T3_BN];

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int bx = swap ? CTA_Y() : CTA_X(), by = swap ? CTA_X() : CTA_Y();
    const int r0 = bx * T3_BM, t0b = by * T3_BN;
    const int wr = warp * 16;
    const int ng = n / GROUP, nsb = (ng + GPS - 1) / GPS;
    const long long rb4 = n / 2, rb2 = n / 4;
    const unsigned int xsa = smem_u32(&xs[0][0]) +
        (unsigned) ((((lane >> 4) * 8 + (lane & 7)) * T3_P + ((lane >> 3) & 1) * 4) * 4);

    // this thread's share of one superblock
    uint4v p4[4], p2[2], px[8];
    unsigned int psq = 0, pmq = 0;
    float pd = 0.0f;
    int pxm[4];
    float pxs = 0.0f;

#define T4_FETCH(c) do {                                                               \
        const int g0_ = (c) * GPS;                                                      \
        const int gn_ = ng - g0_ < GPS ? ng - g0_ : GPS;                                \
        const uint4v z_ = { 0u, 0u, 0u, 0u };                                           \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 4; k_++) {                                                \
            const int i_ = tid + 128 * k_, r_ = i_ >> 3, q_ = i_ & 7;                   \
            p4[k_] = (r0 + r_ < rows && q_ * 4 < gn_ * 4)                               \
                ? *(const uint4v *)(c4 + (long long) (r0 + r_) * rb4 + (long long) g0_ * 16 + q_ * 16) : z_; \
        }                                                                               \
        if (six) {                                                                      \
            _Pragma("unroll")                                                           \
            for (int k_ = 0; k_ < 2; k_++) {                                            \
                const int i_ = tid + 128 * k_, r_ = i_ >> 2, q_ = i_ & 3;               \
                p2[k_] = (r0 + r_ < rows && q_ * 4 < gn_ * 2)                           \
                    ? *(const uint4v *)(c2 + (long long) (r0 + r_) * rb2 + (long long) g0_ * 8 + q_ * 16) : z_; \
            }                                                                           \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 8; k_++) {                                                \
            const int i_ = tid + 128 * k_, t_ = i_ >> 4, q_ = i_ & 15;                  \
            px[k_] = (t0b + t_ < nb && q_ * 4 < gn_ * 8)                                \
                ? *(const uint4v *)(words + (long long) (t0b + t_) * wst + (long long) g0_ * 8 + q_ * 4) : z_; \
        }                                                                               \
        {   /* group scales and mins: row tid/2, groups 4*(tid&1) .. +3 */              \
            const int r_ = tid >> 1, h_ = (tid & 1) * 4;                                \
            psq = 0u; pmq = 0x7f7f7f7fu;                                                \
            if (r0 + r_ < rows) {                                                       \
                const long long b_ = (long long) (r0 + r_) * ng + g0_ + h_;             \
                _Pragma("unroll")                                                       \
                for (int j_ = 0; j_ < 4; j_++)                                          \
                    if (h_ + j_ < gn_) {                                                \
                        psq |= (unsigned int) sq[b_ + j_] << (8 * j_);                  \
                        if (hm) pmq = (pmq & ~(0xffu << (8 * j_))) | ((unsigned int) mq[b_ + j_] << (8 * j_)); \
                    }                                                                   \
            }                                                                           \
        }                                                                               \
        if (tid < T3_BM) pd = r0 + tid < rows ? h2f(sd[(long long) (r0 + tid) * nsb + (c)]) : 0.0f; \
        else pd = (hm && r0 + tid - T3_BM < rows) ? h2f(md[(long long) (r0 + tid - T3_BM) * nsb + (c)]) : 0.0f; \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 4; k_++) {                                                \
            const int i_ = tid + 128 * k_, t_ = i_ >> 3, g_ = i_ & 7;                   \
            pxm[k_] = (t0b + t_ < nb && g_ < gn_) ? sums[(long long) (t0b + t_) * sst + g0_ + g_] : 0; \
        }                                                                               \
        pxs = (tid < T3_BN && t0b + tid < nb) ? scale[(long long) (t0b + tid) * sst + g0_] : 0.0f; \
    } while (0)

#define T4_STORE() do {                                                                \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 4; k_++) {                                                \
            const int i_ = tid + 128 * k_, r_ = i_ >> 3, q_ = i_ & 7;                   \
            *(uint4v *) &w4s[r_][q_ * 4] = p4[k_];                                      \
        }                                                                               \
        if (six) {                                                                      \
            _Pragma("unroll")                                                           \
            for (int k_ = 0; k_ < 2; k_++) {                                            \
                const int i_ = tid + 128 * k_, r_ = i_ >> 2, q_ = i_ & 3;               \
                *(uint4v *) &w2s[r_][q_ * 4] = p2[k_];                                  \
            }                                                                           \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 8; k_++) {                                                \
            const int i_ = tid + 128 * k_, t_ = i_ >> 4, q_ = i_ & 15;                  \
            *(uint4v *) &xs[t_][q_ * 4] = px[k_];                                       \
        }                                                                               \
        {                                                                               \
            const int r_ = tid >> 1, h_ = (tid & 1) * 4;                                \
            _Pragma("unroll")                                                           \
            for (int j_ = 0; j_ < 4; j_++) {                                            \
                sqi[r_][h_ + j_] = (int) ((psq >> (8 * j_)) & 0xffu);                  \
            }                                                                           \
            mA[r_][tid & 1] = pmq ^ 0x80808080u;   /* mq - 128, as int8 */             \
        }                                                                               \
        if (tid < T3_BM) sdf[tid] = pd; else mdf[tid - T3_BM] = pd;                     \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 4; k_++) {                                                \
            const int i_ = tid + 128 * k_, t_ = i_ >> 3, g_ = i_ & 7;                   \
            sHb[t_][g_] = (signed char) (pxm[k_] >> 6);                                 \
            sLb[t_][g_] = (unsigned char) (pxm[k_] & 63);                               \
        }                                                                               \
        if (tid < T3_BN) xsc[tid] = pxs;                                                \
    } while (0)

    float acc[T3_NT][4];
#pragma unroll
    for (int a = 0; a < T3_NT; a++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[a][q] = 0.0f;

    T4_FETCH(0);
    T4_STORE();
    SYNC();

    for (int c = 0; c < nsb; c++) {
        if (c + 1 < nsb) T4_FETCH(c + 1);           // in flight while the MMAs run

        int iacc[T3_NT][4], imin[T3_NT][4];
#pragma unroll
        for (int a = 0; a < T3_NT; a++)
#pragma unroll
            for (int q = 0; q < 4; q++) { iacc[a][q] = 0; imin[a][q] = 0; }

#pragma unroll
        for (int gg = 0; gg < GPS; gg++) {
            const unsigned int a0 = tc_codes(w4s, w2s, wr + gid,     gg, tig,     six);
            const unsigned int a1 = tc_codes(w4s, w2s, wr + gid + 8, gg, tig,     six);
            const unsigned int a2 = tc_codes(w4s, w2s, wr + gid,     gg, 4 + tig, six);
            const unsigned int a3 = tc_codes(w4s, w2s, wr + gid + 8, gg, 4 + tig, six);
            const int s_lo = sqi[wr + gid][gg], s_hi = sqi[wr + gid + 8][gg];
#pragma unroll
            for (int pp = 0; pp < T3_NT / 2; pp++) {
                unsigned int b0, b1, e0, e1;
                ldsm_x4(xsa + (unsigned) (pp * 16 * T3_P * 4) + (unsigned) gg * 32u, b0, b1, e0, e1);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int nt = 2 * pp + h;
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, h ? e0 : b0, h ? e1 : b1);
                    iacc[nt][0] += d0 * s_lo;
                    iacc[nt][1] += d1 * s_lo;
                    iacc[nt][2] += d2 * s_hi;
                    iacc[nt][3] += d3 * s_hi;
                }
            }
        }
        if (hm) {
            const unsigned int one4 = 0x01010101u;
            const unsigned int ma0 = tig < 2 ? mA[wr + gid][tig] : one4;
            const unsigned int ma1 = tig < 2 ? mA[wr + gid + 8][tig] : one4;
#pragma unroll
            for (int nt = 0; nt < T3_NT; nt++) {
                const int tk = nt * 8 + gid;
                const unsigned int bh = ((const unsigned int *) sHb[tk])[tig & 1];
                const unsigned int bl = ((const unsigned int *) sLb[tk])[tig & 1];
                int h0 = 0, h1 = 0, h2 = 0, h3 = 0, l0 = 0, l1 = 0, l2 = 0, l3 = 0;
                mma_s8(h0, h1, h2, h3, ma0, ma1, 0u, 0u, bh, 0u);
                mma_s8(l0, l1, l2, l3, ma0, ma1, 0u, 0u, bl, 0u);
                imin[nt][0] = 64 * h0 + l0;
                imin[nt][1] = 64 * h1 + l1;
                imin[nt][2] = 64 * h2 + l2;
                imin[nt][3] = 64 * h3 + l3;
            }
        }
        const float d_lo = sdf[wr + gid], d_hi = sdf[wr + gid + 8];
        const float n_lo = mdf[wr + gid], n_hi = mdf[wr + gid + 8];
#pragma unroll
        for (int nt = 0; nt < T3_NT; nt++) {
            const int t = nt * 8 + 2 * tig;
            const float x0 = xsc[t], x1 = xsc[t + 1];
            acc[nt][0] += x0 * (d_lo * (float) iacc[nt][0] + n_lo * (float) imin[nt][0]);
            acc[nt][1] += x1 * (d_lo * (float) iacc[nt][1] + n_lo * (float) imin[nt][1]);
            acc[nt][2] += x0 * (d_hi * (float) iacc[nt][2] + n_hi * (float) imin[nt][2]);
            acc[nt][3] += x1 * (d_hi * (float) iacc[nt][3] + n_hi * (float) imin[nt][3]);
        }
        SYNC();
        if (c + 1 < nsb) {
            T4_STORE();
            SYNC();
        }
    }
#undef T4_FETCH
#undef T4_STORE

#pragma unroll
    for (int nt = 0; nt < T3_NT; nt++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int r = r0 + wr + gid + (q >= 2 ? 8 : 0);
            const int t = t0b + nt * 8 + 2 * tig + (q & 1);
            if (r < rows && t < nb) y[(long long) t * yst + r] = acc[nt][q] + (hb ? bias[r] : 0.0f);
        }
}


// ============================================================ tcm3, re-made in passing
//
// A tensor whose int8 copy is not kept ran either sml_g_requant8 + tcm3 (a pass
// of its own over the weights, every call) or tcm4 (1.67x tcm3's matmul time on
// the 1B at 128 tokens: an integer multiply per MMA result for the group scales,
// and 255 registers). Here tcm3 is unchanged except for where its weight tile
// comes from: each thread fetches its row's 4-bit codes (and 2-bit plane) and
// four groups' scales for the next superblock, and at the store re-quantizes them
// into the shared tile with requant8's arithmetic, expression for expression --
// the same bound, the same ka/kb, the same rounding -- so the int8 tile, and the
// result, are those of the kept path bit for bit. Half the bytes of tcm3 come
// from memory, and nothing is written back.
extern "C" __attribute__((global)) void sml_g_tcm5(
        float * y, int yst, const unsigned char * c2, const unsigned char * c4, int six,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, int hm, const float * bias,
        const unsigned int * words, int wst, const float * scale, int sst,
        int n, int rows, int hb, int nb, int swap) {

    __attribute__((shared, aligned(16))) unsigned int ws[T3_BM][T3_P];
    __attribute__((shared, aligned(16))) unsigned int xs[T3_BN][T3_P];
    __attribute__((shared)) float wsc[T3_BM];
    __attribute__((shared)) float xsc[T3_BN];

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int bx = swap ? CTA_Y() : CTA_X(), by = swap ? CTA_X() : CTA_Y();
    const int r0 = bx * T3_BM, t0b = by * T3_BN;
    const int wr = warp * 16;
    const int ng = n / GROUP, nsb = (ng + GPS - 1) / GPS;
    const long long rb4 = n / 2, rb2 = n / 4;
    const unsigned int wsa = smem_u32(&ws[0][0]) +
        (unsigned) (((wr + ((lane >> 3) & 1) * 8 + (lane & 7)) * T3_P + (lane >> 4) * 4) * 4);
    const unsigned int xsa = smem_u32(&xs[0][0]) +
        (unsigned) ((((lane >> 4) * 8 + (lane & 7)) * T3_P + ((lane >> 3) & 1) * 4) * 4);

    // the weight share of this thread: row rr of the tile, groups hh..hh+3 of the superblock
    const int rr = tid >> 1, hh = (tid & 1) * 4;
    const int rlive = r0 + rr < rows;
    uint4v p4[4], p2[2], px[8];
    unsigned int psq = 0u, pmq = 0u;
    float pd = 0.0f, pdm = 0.0f, psx = 0.0f;

#define T5_FETCH(c) do {                                                               \
        const int g0_ = (c) * GPS;                                                      \
        const int gn_ = ng - g0_ < GPS ? ng - g0_ : GPS;                                \
        const uint4v z_ = { 0u, 0u, 0u, 0u };                                           \
        const int glive_ = rlive && hh < gn_;                                           \
        const unsigned char * cp_ = c4 + (long long) (r0 + rr) * rb4 + (long long) (g0_ + hh) * 16; \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 4; k_++)                                                  \
            p4[k_] = (glive_ && hh + (k_ >> 1) * 2 < gn_) ? *(const uint4v *)(cp_ + k_ * 16) : z_; \
        if (six) {                                                                      \
            const unsigned char * ep_ = c2 + (long long) (r0 + rr) * rb2 + (long long) ((g0_ + hh) / 4) * 32; \
            p2[0] = glive_ ? *(const uint4v *)(ep_) : z_;                               \
            p2[1] = glive_ ? *(const uint4v *)(ep_ + 16) : z_;                          \
        }                                                                               \
        psq = 0u; pmq = 0u; pd = 0.0f; pdm = 0.0f;                                      \
        if (glive_) {                                                                   \
            const long long b_ = (long long) (r0 + rr) * ng + g0_ + hh;                 \
            _Pragma("unroll")                                                           \
            for (int j_ = 0; j_ < 4; j_++)                                              \
                if (hh + j_ < gn_) {                                                    \
                    psq |= (unsigned int) sq[b_ + j_] << (8 * j_);                      \
                    if (hm) pmq |= (unsigned int) mq[b_ + j_] << (8 * j_);              \
                }                                                                       \
            pd = h2f(sd[(long long) (r0 + rr) * nsb + (c)]);                            \
            if (hm) pdm = h2f(md[(long long) (r0 + rr) * nsb + (c)]);                   \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 8; k_++) {                                                \
            const int i_ = tid + 128 * k_, t_ = i_ >> 4, q_ = i_ & 15;                  \
            px[k_] = (t0b + t_ < nb && q_ * 4 < gn_ * 8)                                \
                ? *(const uint4v *)(words + (long long) (t0b + t_) * wst + (long long) g0_ * 8 + q_ * 4) : z_; \
        }                                                                               \
        psx = (tid < T3_BN && t0b + tid < nb) ? scale[(long long) (t0b + tid) * sst + g0_] : 0.0f; \
    } while (0)

    // requant8, per thread: the bounds of its four groups, the row bound from the
    // partner thread (the other four groups), then 128 weights into the tile
#define T5_STORE(c) do {                                                               \
        const int g0_ = (c) * GPS;                                                      \
        const int gn_ = ng - g0_ < GPS ? ng - g0_ : GPS;                                \
        float s_[4], m_[4], bound_ = 0.0f;                                              \
        _Pragma("unroll")                                                               \
        for (int j_ = 0; j_ < 4; j_++) {                                                \
            const int live_ = rlive && hh + j_ < gn_;                                   \
            s_[j_] = 0.0f; m_[j_] = 0.0f;                                               \
            if (live_) {                                                                \
                s_[j_] = pd * (float) ((psq >> (8 * j_)) & 0xffu);                      \
                m_[j_] = hm ? pdm * (float) ((int) ((pmq >> (8 * j_)) & 0xffu) - 127) : 0.0f; \
                const float top_ = s_[j_] * (six ? 63.0f : 15.0f) + m_[j_];              \
                const float am_ = m_[j_] < 0.0f ? -m_[j_] : m_[j_], at_ = top_ < 0.0f ? -top_ : top_; \
                const float b1_ = am_ > at_ ? am_ : at_;                                \
                bound_ = b1_ > bound_ ? b1_ : bound_;                                   \
            }                                                                           \
        }                                                                               \
        {                                                                               \
            const float o_ = __nvvm_shfl_sync_bfly_f32(0xffffffffu, bound_, 1u, 31);    \
            bound_ = o_ > bound_ ? o_ : bound_;                                         \
        }                                                                               \
        if (!(tid & 1)) wsc[rr] = bound_ / 127.0f;                                      \
        const float inv_ = bound_ > 0.0f ? 127.0f / bound_ : 0.0f;                      \
        _Pragma("unroll")                                                               \
        for (int j_ = 0; j_ < 4; j_++) {                                                \
            const float ka_ = s_[j_] * inv_, kb_ = m_[j_] * inv_;                       \
            const int sh_ = (j_ & 1) * 4, s2_ = 2 * j_;                                 \
            _Pragma("unroll")                                                           \
            for (int w_ = 0; w_ < 8; w_++) {                                            \
                const unsigned int u_ = p4[(j_ >> 1) * 2 + (w_ >> 2)][w_ & 3];          \
                const unsigned int e_ = six ? p2[w_ >> 2][w_ & 3] : 0u;                 \
                float f_[4];                                                            \
                _Pragma("unroll")                                                       \
                for (int b_ = 0; b_ < 4; b_++) {                                        \
                    const unsigned int x_ = ((u_ >> (8 * b_ + sh_)) & 15u) | (((e_ >> (8 * b_ + s2_)) & 3u) << 4); \
                    f_[b_] = code_f(x_) * ka_ + kb_;                                    \
                }                                                                       \
                const unsigned int o_ = pack4_rnd(f_[0], f_[1], f_[2], f_[3]);          \
                ws[rr][(hh + j_) * 8 + w_] = (hh + j_ < gn_) ? o_ : 0u;                  \
            }                                                                           \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 8; k_++) {                                                \
            const int i_ = tid + 128 * k_, t_ = i_ >> 4, q_ = i_ & 15;                  \
            *(uint4v *) &xs[t_][q_ * 4] = px[k_];                                       \
        }                                                                               \
        if (tid < T3_BN) xsc[tid] = psx;                                                \
    } while (0)

    float acc[T3_NT][4];
#pragma unroll
    for (int a = 0; a < T3_NT; a++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[a][q] = 0.0f;

    T5_FETCH(0);
    T5_STORE(0);
    SYNC();

    for (int c = 0; c < nsb; c++) {
        if (c + 1 < nsb) T5_FETCH(c + 1);           // in flight while the MMAs run

        int iacc[T3_NT][4];
#pragma unroll
        for (int a = 0; a < T3_NT; a++)
#pragma unroll
            for (int q = 0; q < 4; q++) iacc[a][q] = 0;
#pragma unroll
        for (int gg = 0; gg < GPS; gg++) {
            unsigned int a0, a1, a2, a3;
            ldsm_x4(wsa + (unsigned) gg * 32u, a0, a1, a2, a3);
#pragma unroll
            for (int pp = 0; pp < T3_NT / 2; pp++) {
                unsigned int b0, b1, e0, e1;
                ldsm_x4(xsa + (unsigned) (pp * 16 * T3_P * 4) + (unsigned) gg * 32u, b0, b1, e0, e1);
                mma_s8(iacc[2 * pp][0], iacc[2 * pp][1], iacc[2 * pp][2], iacc[2 * pp][3],
                       a0, a1, a2, a3, b0, b1);
                mma_s8(iacc[2 * pp + 1][0], iacc[2 * pp + 1][1], iacc[2 * pp + 1][2], iacc[2 * pp + 1][3],
                       a0, a1, a2, a3, e0, e1);
            }
        }
        const float s_lo = wsc[wr + gid], s_hi = wsc[wr + gid + 8];
#pragma unroll
        for (int nt = 0; nt < T3_NT; nt++) {
            const int t = nt * 8 + 2 * tig;
            const float x0 = xsc[t], x1 = xsc[t + 1];
            acc[nt][0] += s_lo * x0 * (float) iacc[nt][0];
            acc[nt][1] += s_lo * x1 * (float) iacc[nt][1];
            acc[nt][2] += s_hi * x0 * (float) iacc[nt][2];
            acc[nt][3] += s_hi * x1 * (float) iacc[nt][3];
        }
        SYNC();
        if (c + 1 < nsb) {
            T5_STORE(c + 1);
            SYNC();
        }
    }
#undef T5_FETCH
#undef T5_STORE

#pragma unroll
    for (int nt = 0; nt < T3_NT; nt++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int r = r0 + wr + gid + (q >= 2 ? 8 : 0);
            const int t = t0b + nt * 8 + 2 * tig + (q & 1);
            if (r < rows && t < nb) y[(long long) t * yst + r] = acc[nt][q] + (hb ? bias[r] : 0.0f);
        }
}


// ============================================================ tcm5, re-made behind the MMAs
//
// tcm5 re-makes a superblock's weight tile between two barriers, so while it runs
// the tensor cores wait -- on the 1B at 128 tokens its matmuls took 1.41x tcm3's
// (24.7 vs 17.5 ms, 2026-09-26), and the re-make is ~6 integer and float
// instructions per weight against one MMA per 4096 multiply-adds. Here the weight
// tile is double-buffered: while the MMAs of superblock c run from one buffer, the
// same warps re-make superblock c+1 into the other, a group at a time between the
// MMAs of the second half (the first half gives the fetch time to land). One
// barrier pair per superblock remains, around the activation tile, which is
// cheap to store. The buffers are XOR-swizzled (16-byte chunk q of row r lives at
// q ^ (r & 7)) instead of padded, so two blocks still fit on an SM: 48.25 KB each.
// The arithmetic is requant8's, expression for expression: bit-identical to tcm5
// and to the kept path.
extern "C" __attribute__((global)) void sml_g_tcm7(
        float * y, int yst, const unsigned char * c2, const unsigned char * c4, int six,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, int hm, const float * bias,
        const unsigned int * words, int wst, const float * scale, int sst,
        int n, int rows, int hb, int nb, int swap) {

    extern __attribute__((shared, aligned(16))) unsigned int t7_sm[];
    unsigned int * const ws0 = t7_sm;                    // [2][64 rows][64 words]
    unsigned int * const xs = t7_sm + 2 * 64 * 64;       // [64 tokens][64 words]
    float * const xsc = (float *) (t7_sm + 3 * 64 * 64); // [64]

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3, l7 = lane & 7;
    const int bx = swap ? CTA_Y() : CTA_X(), by = swap ? CTA_X() : CTA_Y();
    const int r0 = bx * T3_BM, t0b = by * T3_BN;
    const int wr = warp * 16;
    const int ng = n / GROUP, nsb = (ng + GPS - 1) / GPS;
    const long long rb4 = n / 2, rb2 = n / 4;
    // ldmatrix: lane l addresses row l%8 of matrix l/8 (as in tcm5); with the
    // swizzle, chunk 2gg+hi of a row whose low three bits are l7 sits at byte
    // ((2gg+hi) ^ l7) * 16 = (gg << 5) ^ ((hi ^ l7) << 4)
    const unsigned int sbase = smem_u32(t7_sm);
    const unsigned int arow = sbase + (unsigned) ((wr + ((lane >> 3) & 1) * 8 + l7) * 256);
    const unsigned int brow = sbase + 32768u + (unsigned) ((((lane >> 4) * 8) + l7) * 256);
    const unsigned int av = (unsigned) (((lane >> 4) ^ l7) << 4);
    const unsigned int bv = (unsigned) ((((lane >> 3) & 1) ^ l7) << 4);

    const int rr = tid >> 1, hh = (tid & 1) * 4;
    const int rlive = r0 + rr < rows;
    const int rsw = rr & 7;
    uint4v p4[4], p2[2], px[8];
    unsigned int psq = 0u, pmq = 0u;
    float pd = 0.0f, pdm = 0.0f, psx = 0.0f;
    float s_[4], m_[4], inv_ = 0.0f, bnd_cur = 0.0f, bnd_nxt = 0.0f;

#define T7_FETCH(c) do {                                                               \
        const int g0_ = (c) * GPS;                                                      \
        const int gn_ = ng - g0_ < GPS ? ng - g0_ : GPS;                                \
        const uint4v z_ = { 0u, 0u, 0u, 0u };                                           \
        const int glive_ = rlive && hh < gn_;                                           \
        const unsigned char * cp_ = c4 + (long long) (r0 + rr) * rb4 + (long long) (g0_ + hh) * 16; \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 4; k_++)                                                  \
            p4[k_] = (glive_ && hh + (k_ >> 1) * 2 < gn_) ? *(const uint4v *)(cp_ + k_ * 16) : z_; \
        if (six) {                                                                      \
            const unsigned char * ep_ = c2 + (long long) (r0 + rr) * rb2 + (long long) ((g0_ + hh) / 4) * 32; \
            p2[0] = glive_ ? *(const uint4v *)(ep_) : z_;                               \
            p2[1] = glive_ ? *(const uint4v *)(ep_ + 16) : z_;                          \
        }                                                                               \
        psq = 0u; pmq = 0u; pd = 0.0f; pdm = 0.0f;                                      \
        if (glive_) {                                                                   \
            const long long b_ = (long long) (r0 + rr) * ng + g0_ + hh;                 \
            _Pragma("unroll")                                                           \
            for (int j_ = 0; j_ < 4; j_++)                                              \
                if (hh + j_ < gn_) {                                                    \
                    psq |= (unsigned int) sq[b_ + j_] << (8 * j_);                      \
                    if (hm) pmq |= (unsigned int) mq[b_ + j_] << (8 * j_);              \
                }                                                                       \
            pd = h2f(sd[(long long) (r0 + rr) * nsb + (c)]);                            \
            if (hm) pdm = h2f(md[(long long) (r0 + rr) * nsb + (c)]);                   \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 8; k_++) {                                                \
            const int i_ = tid + 128 * k_, t_ = i_ >> 4, q_ = i_ & 15;                  \
            px[k_] = (t0b + t_ < nb && q_ * 4 < gn_ * 8)                                \
                ? *(const uint4v *)(words + (long long) (t0b + t_) * wst + (long long) g0_ * 8 + q_ * 4) : z_; \
        }                                                                               \
        psx = (tid < T3_BN && t0b + tid < nb) ? scale[(long long) (t0b + tid) * sst + g0_] : 0.0f; \
    } while (0)

    // the row bound: this thread's four groups, then its partner thread's four
#define T7_BOUND(c) do {                                                               \
        const int g0_ = (c) * GPS;                                                      \
        const int gn_ = ng - g0_ < GPS ? ng - g0_ : GPS;                                \
        float bound_ = 0.0f;                                                            \
        _Pragma("unroll")                                                               \
        for (int j_ = 0; j_ < 4; j_++) {                                                \
            const int live_ = rlive && hh + j_ < gn_;                                   \
            s_[j_] = 0.0f; m_[j_] = 0.0f;                                               \
            if (live_) {                                                                \
                s_[j_] = pd * (float) ((psq >> (8 * j_)) & 0xffu);                      \
                m_[j_] = hm ? pdm * (float) ((int) ((pmq >> (8 * j_)) & 0xffu) - 127) : 0.0f; \
                const float top_ = s_[j_] * (six ? 63.0f : 15.0f) + m_[j_];              \
                const float am_ = m_[j_] < 0.0f ? -m_[j_] : m_[j_], at_ = top_ < 0.0f ? -top_ : top_; \
                const float b1_ = am_ > at_ ? am_ : at_;                                \
                bound_ = b1_ > bound_ ? b1_ : bound_;                                   \
            }                                                                           \
        }                                                                               \
        {                                                                               \
            const float o_ = __nvvm_shfl_sync_bfly_f32(0xffffffffu, bound_, 1u, 31);    \
            bound_ = o_ > bound_ ? o_ : bound_;                                         \
        }                                                                               \
        bnd_nxt = bound_ / 127.0f;                                                      \
        inv_ = bound_ > 0.0f ? 127.0f / bound_ : 0.0f;                                  \
    } while (0)

    // group hh+j of superblock c into weight buffer bf: 32 weights, two 16-byte chunks
#define T7_PART(c, j, bf) do {                                                         \
        const int g0_ = (c) * GPS;                                                      \
        const int gn_ = ng - g0_ < GPS ? ng - g0_ : GPS;                                \
        const float ka_ = s_[j] * inv_, kb_ = m_[j] * inv_;                             \
        const int sh_ = ((j) & 1) * 4, s2_ = 2 * (j);                                   \
        unsigned int o_[8];                                                             \
        _Pragma("unroll")                                                               \
        for (int w_ = 0; w_ < 8; w_++) {                                                \
            const unsigned int u_ = p4[((j) >> 1) * 2 + (w_ >> 2)][w_ & 3];             \
            const unsigned int e_ = six ? p2[w_ >> 2][w_ & 3] : 0u;                     \
            float f_[4];                                                                \
            _Pragma("unroll")                                                           \
            for (int b_ = 0; b_ < 4; b_++) {                                            \
                const unsigned int x_ = ((u_ >> (8 * b_ + sh_)) & 15u) | (((e_ >> (8 * b_ + s2_)) & 3u) << 4); \
                f_[b_] = code_f(x_) * ka_ + kb_;                                        \
            }                                                                           \
            o_[w_] = pack4_rnd(f_[0], f_[1], f_[2], f_[3]);                             \
        }                                                                               \
        const int lv_ = hh + (j) < gn_;                                                 \
        const uint4v v0_ = { lv_ ? o_[0] : 0u, lv_ ? o_[1] : 0u, lv_ ? o_[2] : 0u, lv_ ? o_[3] : 0u }; \
        const uint4v v1_ = { lv_ ? o_[4] : 0u, lv_ ? o_[5] : 0u, lv_ ? o_[6] : 0u, lv_ ? o_[7] : 0u }; \
        unsigned int * row_ = ws0 + (bf) * 4096 + rr * 64;                              \
        *(uint4v *) (row_ + ((((hh + (j)) * 2) ^ rsw) << 2)) = v0_;                     \
        *(uint4v *) (row_ + ((((hh + (j)) * 2 + 1) ^ rsw) << 2)) = v1_;                 \
    } while (0)

#define T7_XS() do {                                                                   \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 8; k_++) {                                                \
            const int i_ = tid + 128 * k_, t_ = i_ >> 4, q_ = i_ & 15;                  \
            *(uint4v *) (xs + t_ * 64 + ((q_ ^ (t_ & 7)) << 2)) = px[k_];               \
        }                                                                               \
        if (tid < T3_BN) xsc[tid] = psx;                                                \
    } while (0)

    float acc[T3_NT][4];
#pragma unroll
    for (int a = 0; a < T3_NT; a++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[a][q] = 0.0f;

    T7_FETCH(0);
    T7_BOUND(0);
    T7_PART(0, 0, 0); T7_PART(0, 1, 0); T7_PART(0, 2, 0); T7_PART(0, 3, 0);
    T7_XS();
    bnd_cur = bnd_nxt;
    SYNC();

    for (int c = 0; c < nsb; c++) {
        const int more = c + 1 < nsb;
        const unsigned int bo = (unsigned) (c & 1) * 16384u;
        if (more) T7_FETCH(c + 1);                  // in flight while the first half runs

        int iacc[T3_NT][4];
#pragma unroll
        for (int a = 0; a < T3_NT; a++)
#pragma unroll
            for (int q = 0; q < 4; q++) iacc[a][q] = 0;
#pragma unroll
        for (int gg = 0; gg < GPS; gg++) {
            unsigned int a0, a1, a2, a3;
            const unsigned int xa = ((unsigned) gg << 5) ^ av, xb = ((unsigned) gg << 5) ^ bv;
            ldsm_x4(arow + bo + xa, a0, a1, a2, a3);
#pragma unroll
            for (int pp = 0; pp < T3_NT / 2; pp++) {
                unsigned int b0, b1, e0, e1;
                ldsm_x4(brow + (unsigned) (pp * 4096) + xb, b0, b1, e0, e1);
                mma_s8(iacc[2 * pp][0], iacc[2 * pp][1], iacc[2 * pp][2], iacc[2 * pp][3],
                       a0, a1, a2, a3, b0, b1);
                mma_s8(iacc[2 * pp + 1][0], iacc[2 * pp + 1][1], iacc[2 * pp + 1][2], iacc[2 * pp + 1][3],
                       a0, a1, a2, a3, e0, e1);
            }
            // the next superblock, a group per step, behind these MMAs
            if (more && gg >= 4) {
                if (gg == 4) T7_BOUND(c + 1);
                T7_PART(c + 1, gg - 4, (c + 1) & 1);
            }
        }
        const float s_lo = __nvvm_shfl_sync_idx_f32(0xffffffffu, bnd_cur, (unsigned) (2 * gid), 31);
        const float s_hi = __nvvm_shfl_sync_idx_f32(0xffffffffu, bnd_cur, (unsigned) (2 * gid + 16), 31);
#pragma unroll
        for (int nt = 0; nt < T3_NT; nt++) {
            const int t = nt * 8 + 2 * tig;
            const float x0 = xsc[t], x1 = xsc[t + 1];
            acc[nt][0] += s_lo * x0 * (float) iacc[nt][0];
            acc[nt][1] += s_lo * x1 * (float) iacc[nt][1];
            acc[nt][2] += s_hi * x0 * (float) iacc[nt][2];
            acc[nt][3] += s_hi * x1 * (float) iacc[nt][3];
        }
        SYNC();
        if (more) {
            T7_XS();
            bnd_cur = bnd_nxt;
            SYNC();
        }
    }
#undef T7_FETCH
#undef T7_BOUND
#undef T7_PART
#undef T7_XS

#pragma unroll
    for (int nt = 0; nt < T3_NT; nt++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int r = r0 + wr + gid + (q >= 2 ? 8 : 0);
            const int t = t0b + nt * 8 + 2 * tig + (q & 1);
            if (r < rows && t < nb) y[(long long) t * yst + r] = acc[nt][q] + (hb ? bias[r] : 0.0f);
        }
}


// ============================================================ tcm7 over 128 tokens
//
// tcm7 measured no faster than tcm5 (1B, 128 tokens): the re-make already ran
// beside the other block's MMAs, so the cost is the instructions themselves --
// ~6 per weight, re-made once per 64-token tile, against one MMA per 4096
// multiply-adds. Here one block holds 128 tokens (eight warps: rows 16(w%4).., tokens
// 64(w/4)..), so each weight is re-made once per 128 tokens -- half the
// instructions per multiply-add -- and 256 threads share the re-make, two groups
// each. The row bound needs four threads (two shuffles); the per-row scale goes
// through shared memory, since the warps that make a row are not the warps that
// use it. Same swizzle, same double-buffered weight tile, same arithmetic:
// bit-identical to tcm5 and to the kept path.
extern "C" __attribute__((global)) void sml_g_tcm8(
        float * y, int yst, const unsigned char * c2, const unsigned char * c4, int six,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, int hm, const float * bias,
        const unsigned int * words, int wst, const float * scale, int sst,
        int n, int rows, int hb, int nb, int swap) {

    extern __attribute__((shared, aligned(16))) unsigned int t8_sm[];
    unsigned int * const ws0 = t8_sm;                     // [2][64 rows][64 words]
    unsigned int * const xs = t8_sm + 2 * 64 * 64;        // [128 tokens][64 words]
    float * const xsc = (float *) (t8_sm + 4 * 64 * 64);  // [128]
    float * const wsc = xsc + 128;                        // [2][64]

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3, l7 = lane & 7;
    const int bx = swap ? CTA_Y() : CTA_X(), by = swap ? CTA_X() : CTA_Y();
    const int r0 = bx * T3_BM, t0b = by * 128;
    const int wr = (warp & 3) * 16, tw = (warp >> 2) * 64;
    const int ng = n / GROUP, nsb = (ng + GPS - 1) / GPS;
    const long long rb4 = n / 2, rb2 = n / 4;
    const unsigned int sbase = smem_u32(t8_sm);
    const unsigned int arow = sbase + (unsigned) ((wr + ((lane >> 3) & 1) * 8 + l7) * 256);
    const unsigned int brow = sbase + 32768u + (unsigned) ((tw + ((lane >> 4) * 8) + l7) * 256);
    const unsigned int av = (unsigned) (((lane >> 4) ^ l7) << 4);
    const unsigned int bv = (unsigned) ((((lane >> 3) & 1) ^ l7) << 4);

    // re-make share: row rr, groups hh and hh+1 (one 32-byte pair of the 4-bit plane)
    const int rr = tid >> 2, hh = (tid & 3) * 2;
    const int rlive = r0 + rr < rows;
    const int rsw = rr & 7;
    uint4v p4[2], p2[2], px[8];
    unsigned int psq = 0u, pmq = 0u;
    float pd = 0.0f, pdm = 0.0f, psx = 0.0f;
    float s_[2], m_[2], inv_ = 0.0f;

#define T8_FETCH(c) do {                                                               \
        const int g0_ = (c) * GPS;                                                      \
        const int gn_ = ng - g0_ < GPS ? ng - g0_ : GPS;                                \
        const uint4v z_ = { 0u, 0u, 0u, 0u };                                           \
        const int glive_ = rlive && hh < gn_;                                           \
        const unsigned char * cp_ = c4 + (long long) (r0 + rr) * rb4 + (long long) (g0_ + hh) * 16; \
        p4[0] = glive_ ? *(const uint4v *)(cp_) : z_;                                   \
        p4[1] = glive_ ? *(const uint4v *)(cp_ + 16) : z_;                              \
        if (six) {                                                                      \
            const unsigned char * ep_ = c2 + (long long) (r0 + rr) * rb2 + (long long) ((g0_ + hh) / 4) * 32; \
            p2[0] = glive_ ? *(const uint4v *)(ep_) : z_;                               \
            p2[1] = glive_ ? *(const uint4v *)(ep_ + 16) : z_;                          \
        }                                                                               \
        psq = 0u; pmq = 0u; pd = 0.0f; pdm = 0.0f;                                      \
        if (glive_) {                                                                   \
            const long long b_ = (long long) (r0 + rr) * ng + g0_ + hh;                 \
            _Pragma("unroll")                                                           \
            for (int j_ = 0; j_ < 2; j_++)                                              \
                if (hh + j_ < gn_) {                                                    \
                    psq |= (unsigned int) sq[b_ + j_] << (8 * j_);                      \
                    if (hm) pmq |= (unsigned int) mq[b_ + j_] << (8 * j_);              \
                }                                                                       \
            pd = h2f(sd[(long long) (r0 + rr) * nsb + (c)]);                            \
            if (hm) pdm = h2f(md[(long long) (r0 + rr) * nsb + (c)]);                   \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 8; k_++) {                                                \
            const int i_ = tid + 256 * k_, t_ = i_ >> 4, q_ = i_ & 15;                  \
            px[k_] = (t0b + t_ < nb && q_ * 4 < gn_ * 8)                                \
                ? *(const uint4v *)(words + (long long) (t0b + t_) * wst + (long long) g0_ * 8 + q_ * 4) : z_; \
        }                                                                               \
        psx = (tid < 128 && t0b + tid < nb) ? scale[(long long) (t0b + tid) * sst + g0_] : 0.0f; \
    } while (0)

    // the row bound: this thread's two groups, then the other three threads' six
#define T8_BOUND(c, bf) do {                                                           \
        const int g0_ = (c) * GPS;                                                      \
        const int gn_ = ng - g0_ < GPS ? ng - g0_ : GPS;                                \
        float bound_ = 0.0f;                                                            \
        _Pragma("unroll")                                                               \
        for (int j_ = 0; j_ < 2; j_++) {                                                \
            const int live_ = rlive && hh + j_ < gn_;                                   \
            s_[j_] = 0.0f; m_[j_] = 0.0f;                                               \
            if (live_) {                                                                \
                s_[j_] = pd * (float) ((psq >> (8 * j_)) & 0xffu);                      \
                m_[j_] = hm ? pdm * (float) ((int) ((pmq >> (8 * j_)) & 0xffu) - 127) : 0.0f; \
                const float top_ = s_[j_] * (six ? 63.0f : 15.0f) + m_[j_];              \
                const float am_ = m_[j_] < 0.0f ? -m_[j_] : m_[j_], at_ = top_ < 0.0f ? -top_ : top_; \
                const float b1_ = am_ > at_ ? am_ : at_;                                \
                bound_ = b1_ > bound_ ? b1_ : bound_;                                   \
            }                                                                           \
        }                                                                               \
        {                                                                               \
            float o_ = __nvvm_shfl_sync_bfly_f32(0xffffffffu, bound_, 1u, 31);          \
            bound_ = o_ > bound_ ? o_ : bound_;                                         \
            o_ = __nvvm_shfl_sync_bfly_f32(0xffffffffu, bound_, 2u, 31);                \
            bound_ = o_ > bound_ ? o_ : bound_;                                         \
        }                                                                               \
        if (!(tid & 3)) wsc[(bf) * 64 + rr] = bound_ / 127.0f;                          \
        inv_ = bound_ > 0.0f ? 127.0f / bound_ : 0.0f;                                  \
    } while (0)

    // group hh+j of superblock c into weight buffer bf
#define T8_PART(c, j, bf) do {                                                         \
        const int g0_ = (c) * GPS;                                                      \
        const int gn_ = ng - g0_ < GPS ? ng - g0_ : GPS;                                \
        const float ka_ = s_[j] * inv_, kb_ = m_[j] * inv_;                             \
        const int sh_ = ((j) & 1) * 4, s2_ = 2 * ((hh + (j)) & 3);                      \
        unsigned int o_[8];                                                             \
        _Pragma("unroll")                                                               \
        for (int w_ = 0; w_ < 8; w_++) {                                                \
            const unsigned int u_ = p4[w_ >> 2][w_ & 3];                                \
            const unsigned int e_ = six ? p2[w_ >> 2][w_ & 3] : 0u;                     \
            float f_[4];                                                                \
            _Pragma("unroll")                                                           \
            for (int b_ = 0; b_ < 4; b_++) {                                            \
                const unsigned int x_ = ((u_ >> (8 * b_ + sh_)) & 15u) | (((e_ >> (8 * b_ + s2_)) & 3u) << 4); \
                f_[b_] = code_f(x_) * ka_ + kb_;                                        \
            }                                                                           \
            o_[w_] = pack4_rnd(f_[0], f_[1], f_[2], f_[3]);                             \
        }                                                                               \
        const int lv_ = hh + (j) < gn_;                                                 \
        const uint4v v0_ = { lv_ ? o_[0] : 0u, lv_ ? o_[1] : 0u, lv_ ? o_[2] : 0u, lv_ ? o_[3] : 0u }; \
        const uint4v v1_ = { lv_ ? o_[4] : 0u, lv_ ? o_[5] : 0u, lv_ ? o_[6] : 0u, lv_ ? o_[7] : 0u }; \
        unsigned int * row_ = ws0 + (bf) * 4096 + rr * 64;                              \
        *(uint4v *) (row_ + ((((hh + (j)) * 2) ^ rsw) << 2)) = v0_;                     \
        *(uint4v *) (row_ + ((((hh + (j)) * 2 + 1) ^ rsw) << 2)) = v1_;                 \
    } while (0)

#define T8_XS() do {                                                                   \
        _Pragma("unroll")                                                               \
        for (int k_ = 0; k_ < 8; k_++) {                                                \
            const int i_ = tid + 256 * k_, t_ = i_ >> 4, q_ = i_ & 15;                  \
            *(uint4v *) (xs + t_ * 64 + ((q_ ^ (t_ & 7)) << 2)) = px[k_];               \
        }                                                                               \
        if (tid < 128) xsc[tid] = psx;                                                  \
    } while (0)

    float acc[T3_NT][4];
#pragma unroll
    for (int a = 0; a < T3_NT; a++)
#pragma unroll
        for (int q = 0; q < 4; q++) acc[a][q] = 0.0f;

    T8_FETCH(0);
    T8_BOUND(0, 0);
    T8_PART(0, 0, 0); T8_PART(0, 1, 0);
    T8_XS();
    SYNC();

    for (int c = 0; c < nsb; c++) {
        const int more = c + 1 < nsb;
        const int cb = c & 1;
        const unsigned int bo = (unsigned) cb * 16384u;
        if (more) T8_FETCH(c + 1);                  // in flight while the first half runs

        int iacc[T3_NT][4];
#pragma unroll
        for (int a = 0; a < T3_NT; a++)
#pragma unroll
            for (int q = 0; q < 4; q++) iacc[a][q] = 0;
#pragma unroll
        for (int gg = 0; gg < GPS; gg++) {
            unsigned int a0, a1, a2, a3;
            const unsigned int xa = ((unsigned) gg << 5) ^ av, xb = ((unsigned) gg << 5) ^ bv;
            ldsm_x4(arow + bo + xa, a0, a1, a2, a3);
#pragma unroll
            for (int pp = 0; pp < T3_NT / 2; pp++) {
                unsigned int b0, b1, e0, e1;
                ldsm_x4(brow + (unsigned) (pp * 4096) + xb, b0, b1, e0, e1);
                mma_s8(iacc[2 * pp][0], iacc[2 * pp][1], iacc[2 * pp][2], iacc[2 * pp][3],
                       a0, a1, a2, a3, b0, b1);
                mma_s8(iacc[2 * pp + 1][0], iacc[2 * pp + 1][1], iacc[2 * pp + 1][2], iacc[2 * pp + 1][3],
                       a0, a1, a2, a3, e0, e1);
            }
            // the next superblock behind these MMAs: the bound and a group at step
            // 4, the other group at step 6
            if (more && gg == 4) { T8_BOUND(c + 1, cb ^ 1); T8_PART(c + 1, 0, cb ^ 1); }
            if (more && gg == 6) T8_PART(c + 1, 1, cb ^ 1);
        }
        const float s_lo = wsc[cb * 64 + wr + gid], s_hi = wsc[cb * 64 + wr + gid + 8];
#pragma unroll
        for (int nt = 0; nt < T3_NT; nt++) {
            const int t = tw + nt * 8 + 2 * tig;
            const float x0 = xsc[t], x1 = xsc[t + 1];
            acc[nt][0] += s_lo * x0 * (float) iacc[nt][0];
            acc[nt][1] += s_lo * x1 * (float) iacc[nt][1];
            acc[nt][2] += s_hi * x0 * (float) iacc[nt][2];
            acc[nt][3] += s_hi * x1 * (float) iacc[nt][3];
        }
        SYNC();
        if (more) {
            T8_XS();
            SYNC();
        }
    }
#undef T8_FETCH
#undef T8_BOUND
#undef T8_PART
#undef T8_XS

#pragma unroll
    for (int nt = 0; nt < T3_NT; nt++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int r = r0 + wr + gid + (q >= 2 ? 8 : 0);
            const int t = t0b + tw + nt * 8 + 2 * tig + (q & 1);
            if (r < rows && t < nb) y[(long long) t * yst + r] = acc[nt][q] + (hb ? bias[r] : 0.0f);
        }
}


// ============================================================ tcm3, as an Ampere GEMM
//
// tcm3 (64 x 64 tile, four warps of 16 rows) runs at about a third of this GPU's
// int8 tensor peak: every warp re-reads all 64 tokens' B fragments for its 16 rows
// (five ldmatrix per eight MMAs, ~440 bytes of shared reads per MMA), and the next
// tile passes through registers on its way to shared memory. Here a block holds
// 128 rows x 128 tokens in eight warps of 32 rows x 64 tokens (six ldmatrix per
// sixteen MMAs, ~190 bytes per MMA), and the tiles go global -> shared directly
// with cp.async, double-buffered, half a superblock (128 bytes of K) per stage.
// Rows are 128 bytes with the 16-byte chunks XOR-swizzled by row, so ldmatrix
// and the copies are conflict-free without padding. The superblock's integer
// sums, its float epilogue and the order of superblocks are tcm3's: the result is
// bit-identical (split-K off).
__attribute__((device, always_inline)) static inline void cpa16(unsigned int dst, const void * src, int ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0) : "memory");
}
__attribute__((device, always_inline)) static inline void cpa4(unsigned int dst, const void * src, int ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(dst), "l"(src), "r"(ok ? 4 : 0) : "memory");
}
__attribute__((device, always_inline)) static inline void cpa_commit(void) {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
__attribute__((device, always_inline)) static inline void cpa_wait1(void) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
}

extern "C" __attribute__((global)) void sml_g_tcm9(
        float * y, int yst, const signed char * w8, const float * s8, const float * bias,
        const unsigned int * words, int wst, const float * scale, int sst,
        int n, int rows, int hb, int nb, int swap) {

    extern __attribute__((shared, aligned(16))) unsigned int t9_sm[];
    // [2][128][128 B] weights, [2][128][128 B] activations, [2][256] scales
    const unsigned int sbase = smem_u32(t9_sm);
    const unsigned int WS = sbase, XS = sbase + 32768u, SC = sbase + 65536u;
    const float * scl = (const float *) (t9_sm + 16384);

    const int tid = TID_X(), lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3, l7 = lane & 7;
    const int bx = swap ? CTA_Y() : CTA_X(), by = swap ? CTA_X() : CTA_Y();
    const int r0 = bx * 128, t0b = by * 128;
    const int wm = (warp & 3) * 32, wn = (warp >> 2) * 64;
    const int ng = n / GROUP, nsb = (ng + GPS - 1) / GPS, nch = nsb * 2;

    // ldmatrix lane addresses (tcm3's fragment mapping, 128-byte swizzled rows)
    const unsigned int arow = (unsigned) ((wm + ((lane >> 3) & 1) * 8 + l7) * 128);
    const unsigned int brow = (unsigned) ((wn + (lane >> 4) * 8 + l7) * 128);
    const unsigned int av = (unsigned) (((lane >> 4) ^ l7) << 4);
    const unsigned int bv = (unsigned) ((((lane >> 3) & 1) ^ l7) << 4);

    // this thread's copies: rows (tokens) tid/8 + 32j, 16-byte chunk tid%8
    const int crow = tid >> 3, cch = tid & 7;
    const unsigned int cdst = (unsigned) (crow * 128 + ((cch ^ (crow & 7)) << 4));

    // issue half-superblock k (h = k & 1) into stage k & 1; at h = 0 also the scales
    auto issue = [&](int k) {
        const int c = k >> 1, h = k & 1, st = k & 1;
        const int g0 = c * GPS, gn = ng - g0 < GPS ? ng - g0 : GPS;
        int kb = gn * 32 - h * 128;                       // valid bytes of K in this half
        kb = kb < 0 ? 0 : (kb > 128 ? 128 : kb);
        const int okc = cch * 16 < kb;
        const long long koff = (long long) c * 256 + h * 128 + cch * 16;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int rr = crow + 32 * j;
            const int okw = okc && r0 + rr < rows;
            const signed char * sw = okw ? w8 + (long long) (r0 + rr) * n + koff : w8;
            cpa16(WS + (unsigned) st * 16384u + cdst + (unsigned) (j * 32 * 128), sw, okw);
            const int okx = okc && t0b + rr < nb;
            const char * sx = okx ? (const char *) words + (long long) (t0b + rr) * wst * 4 + koff
                                  : (const char *) words;
            cpa16(XS + (unsigned) st * 16384u + cdst + (unsigned) (j * 32 * 128), sx, okx);
        }
        if (!h) {
            const unsigned int sd = SC + (unsigned) ((c & 1) * 1024 + tid * 4);
            if (tid < 128) {
                const int ok = r0 + tid < rows;
                cpa4(sd, ok ? s8 + (long long) (r0 + tid) * nsb + c : s8, ok);
            } else {
                const int ok = t0b + tid - 128 < nb;
                cpa4(sd, ok ? scale + (long long) (t0b + tid - 128) * sst + (long long) c * GPS : scale, ok);
            }
        }
    };

    float acc[2][8][4];
    int iacc[2][8][4];
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int a = 0; a < 8; a++)
#pragma unroll
            for (int q = 0; q < 4; q++) { acc[m][a][q] = 0.0f; iacc[m][a][q] = 0; }

    issue(0);
    cpa_commit();
    for (int k = 0; k < nch; k++) {
        if (k + 1 < nch) issue(k + 1);
        cpa_commit();
        cpa_wait1();                                  // stage k has landed (this thread)
        SYNC();                                       // ... and every thread's
        const unsigned int wsb = WS + (unsigned) (k & 1) * 16384u + arow;
        const unsigned int xsb = XS + (unsigned) (k & 1) * 16384u + brow;
#pragma unroll
        for (int gg = 0; gg < 4; gg++) {
            const unsigned int xa = ((unsigned) gg << 5) ^ av, xb = ((unsigned) gg << 5) ^ bv;
            unsigned int a[2][4];
            ldsm_x4(wsb + xa, a[0][0], a[0][1], a[0][2], a[0][3]);
            ldsm_x4(wsb + 2048u + xa, a[1][0], a[1][1], a[1][2], a[1][3]);
#pragma unroll
            for (int pp = 0; pp < 4; pp++) {
                unsigned int b0, b1, e0, e1;
                ldsm_x4(xsb + (unsigned) (pp * 2048) + xb, b0, b1, e0, e1);
#pragma unroll
                for (int m = 0; m < 2; m++) {
                    mma_s8(iacc[m][2 * pp][0], iacc[m][2 * pp][1], iacc[m][2 * pp][2], iacc[m][2 * pp][3],
                           a[m][0], a[m][1], a[m][2], a[m][3], b0, b1);
                    mma_s8(iacc[m][2 * pp + 1][0], iacc[m][2 * pp + 1][1], iacc[m][2 * pp + 1][2], iacc[m][2 * pp + 1][3],
                           a[m][0], a[m][1], a[m][2], a[m][3], e0, e1);
                }
            }
        }
        if (k & 1) {                                  // a superblock is summed: tcm3's epilogue
            const float * sc = scl + ((k >> 1) & 1) * 256;
#pragma unroll
            for (int m = 0; m < 2; m++) {
                const float s_lo = sc[wm + m * 16 + gid], s_hi = sc[wm + m * 16 + gid + 8];
#pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    const int t = wn + nt * 8 + 2 * tig;
                    const float x0 = sc[128 + t], x1 = sc[128 + t + 1];
                    acc[m][nt][0] += s_lo * x0 * (float) iacc[m][nt][0];
                    acc[m][nt][1] += s_lo * x1 * (float) iacc[m][nt][1];
                    acc[m][nt][2] += s_hi * x0 * (float) iacc[m][nt][2];
                    acc[m][nt][3] += s_hi * x1 * (float) iacc[m][nt][3];
                    iacc[m][nt][0] = 0; iacc[m][nt][1] = 0; iacc[m][nt][2] = 0; iacc[m][nt][3] = 0;
                }
            }
        }
        SYNC();                                       // stage k is free for issue(k + 2)
    }

#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int r = r0 + wm + m * 16 + gid + (q >= 2 ? 8 : 0);
                const int t = t0b + wn + nt * 8 + 2 * tig + (q & 1);
                if (r < rows && t < nb) y[(long long) t * yst + r] = acc[m][nt][q] + (hb ? bias[r] : 0.0f);
            }
}


// ============================================================ Q/K norm
//
// Qwen3, Gemma 3 and OLMo 2 normalize Q and K before rotary: RMS (or, with a bias,
// LayerNorm -- Persimmon) over each head when the weights are one head long, or
// over the whole projection when they are that long (OLMo 2). The same arithmetic
// as sml-run.c's qk_norm. Grid: (heads or 1, tokens); x is token-major with
// stride xst. Summed in float with warp shuffles: the double tree this had (to
// match the CPU's double sums) runs at 1/64 rate on a GeForce card and was 39% of
// a Qwen3 head's GPU prompt (4.1 of 9.4 ms at 128 tokens); float moves a 128-wide
// head's sum by ~1e-7 of itself.
extern "C" __attribute__((global)) void sml_g_qknorm(
        float * x, const float * w, const float * b, int has_b, int wn, int hd, int xst, float eps) {
    __attribute__((shared)) float red[32];
    const int span = wn == hd ? hd : wn;
    float * v = x + (long long) CTA_Y() * xst + (long long) CTA_X() * span;
    const int t = TID_X(), nt = NTID_X();
    const int lane = t & (WARP - 1), warp = t / WARP, nwarp = (nt + WARP - 1) / WARP;
    float mean = 0.0f;
    if (has_b) {
        float s = 0.0f;
        for (int i = t; i < span; i += nt) s += v[i];
        s = wsum(s);
        if (lane == 0) red[warp] = s;
        SYNC();
        float tot = 0.0f;
        for (int k = 0; k < nwarp; k++) tot += red[k];
        mean = tot / (float) span;
        SYNC();
    }
    float ss = 0.0f;
    for (int i = t; i < span; i += nt) { const float d = v[i] - mean; ss += d * d; }
    ss = wsum(ss);
    if (lane == 0) red[warp] = ss;
    SYNC();
    float tot = 0.0f;
    for (int k = 0; k < nwarp; k++) tot += red[k];
    const float scl = 1.0f / __builtin_sqrtf(tot / (float) span + eps);
    for (int i = t; i < span; i += nt)
        v[i] = (v[i] - mean) * scl * w[i] + (has_b ? b[i] : 0.0f);
}

// x *= s on nb rows of n: Gemma's embedding scale, Granite's residual and logit
// scales. Grid: (n / 256, rows).
extern "C" __attribute__((global)) void sml_g_scaleb(float * x, int xst, int n, int nb, float s) {
    const int i = CTA_X() * NTID_X() + TID_X();
    const int t = CTA_Y();
    if (i < n && t < nb) x[(long long) t * xst + i] *= s;
}

// x[t] += tab[row]: the output-head bias (tst 0: the same row for every token)
// and learned absolute positions (row = the token's position, read from *dpos
// when a replayed graph must not bake a position in).
extern "C" __attribute__((global)) void sml_g_addrow(float * x, int xst, int n, int nb,
                                                     const float * tab, int tst, const int * dpos, int pos0) {
    const int i = CTA_X() * NTID_X() + TID_X();
    const int t = CTA_Y();
    if (i >= n || t >= nb) return;
    const int row = (dpos ? *dpos : pos0) + t;
    x[(long long) t * xst + i] += tab[(long long) row * tst + i];
}

// cap * tanh(x / cap): Gemma 2's final logit soft cap.
extern "C" __attribute__((global)) void sml_g_capb(float * x, int xst, int n, int nb, float cap) {
    const int i = CTA_X() * NTID_X() + TID_X();
    const int t = CTA_Y();
    if (i >= n || t >= nb) return;
    float * p = x + (long long) t * xst + i;
    const float z = *p / cap;
    float th;
    if (z > 9.0f) th = 1.0f;
    else if (z < -9.0f) th = -1.0f;
    else { const float e = sml_exp(2.0f * z); th = (e - 1.0f) / (e + 1.0f); }
    *p = cap * th;
}

// y = W x in full precision, one block per row, summed in double as the CPU's
// router loop does (sml-run.c moe_ffn): the router's logits decide which
// experts run, and a near-tie read differently would choose differently.
extern "C" __attribute__((global)) void sml_g_f32mv(float * y, const float * w, const float * x,
                                                    int rows, int cols) {
    __attribute__((shared)) double part[256];
    const int r = CTA_X(), t = TID_X(), nt = NTID_X();
    if (r >= rows) return;
    const float * wr = w + (long long) r * cols;
    double s = 0.0;
    for (int i = t; i < cols; i += nt) s += (double) wr[i] * x[i];
    part[t] = s;
    SYNC();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (t < st) part[t] += part[t + st];
        SYNC();
    }
    if (t == 0) y[r] = (float) part[0];
}

// y = a x (acc 0) or y += a x (acc 1): an expert's output, weighted by the router.
extern "C" __attribute__((global)) void sml_g_axpy(float * y, const float * x, int n, float a, int acc) {
    const int i = CTA_X() * NTID_X() + TID_X();
    if (i < n) y[i] = acc ? y[i] + a * x[i] : a * x[i];
}

// Q, K and V clamped to [-c, c] (OLMo, DBRX's clip_qkv).
extern "C" __attribute__((global)) void sml_g_clampb(float * x, int xst, int n, int nb, float c) {
    const int i = CTA_X() * NTID_X() + TID_X();
    const int t = CTA_Y();
    if (i >= n || t >= nb) return;
    float * p = x + (long long) t * xst + i;
    *p = *p > c ? c : (*p < -c ? -c : *p);
}

// RoPE on this token's q and k in place, the position read from device memory so
// a graph replays it. HunYuan normalizes Q and K AFTER rotating them, so the
// rotation cannot wait for the attention kernel (which then rotates nothing).
extern "C" __attribute__((global)) void sml_g_ropeqd(
        float * q, float * k, const float * rc, const float * rs, const int * dpos,
        int n_head, int n_kv, int hd, int half, int interleaved) {
    const int pos = *dpos;
    const float * cs = rc + (long long) pos * half;
    const float * sn = rs + (long long) pos * half;
    for (int idx = CTA_X() * NTID_X() + TID_X(); idx < (n_head + n_kv) * half; idx += NCTA_X() * NTID_X()) {
        const int hh = idx / half, i = idx % half;
        float * u = hh < n_head ? q + hh * hd : k + (hh - n_head) * hd;
        const int a0 = interleaved ? 2 * i : i, a1 = interleaved ? 2 * i + 1 : i + half;
        const float x0 = u[a0], x1 = u[a1];
        u[a0] = x0 * cs[i] - x1 * sn[i];
        u[a1] = x0 * sn[i] + x1 * cs[i];
    }
}

// ============================================================ experts on the card
//
// One token's mixture of experts with no host in the loop, so a graph can replay
// it: the router's logits (sml_g_f32mv) -> sml_g_moe_route picks k experts and
// their weights exactly as sml-run.c's sml_moe_route does -> the activation is
// quantized once per chosen expert with THAT expert's column order -> gate and up
// of every chosen expert (one grid row per choice) -> the hidden rows quantized
// with each down projection's order -> down -> sml_g_moe_comb sums the choices in
// order, then the shared expert. Tables hold each expert's device pointers.
typedef struct {
    const unsigned char * c2; const unsigned char * c4;
    const unsigned short * sd; const unsigned char * sq;
    const unsigned short * md; const unsigned char * mq;
    int rows, six, hm, pad;
} sml_mt;

// A port of sml_moe_route (one thread): sigmoid or softmax scores, the choice on
// score + bias within the best groups (V3), or plain top-k on the logits with
// softmax weights over the chosen (norm) or over all experts. Ties go to the
// lower index, as on the CPU.
extern "C" __attribute__((global)) void sml_g_moe_route(
        const float * lg, int ne, int k, int flags, const float * ebias, int has_bias,
        int ng, int tg, float mscale, float * sc, float * ch, int * sel, float * w) {
    if (CTA_X() != 0 || TID_X() != 0) return;
    const int v3 = (flags & 2) || has_bias || ng > 1;
    if (v3) {
        if (flags & 2) {
            for (int e = 0; e < ne; e++) sc[e] = 1.0f / (1.0f + sml_exp(-lg[e]));
        } else {
            float mx = lg[0];
            for (int e = 1; e < ne; e++) if (lg[e] > mx) mx = lg[e];
            double sum = 0.0;
            for (int e = 0; e < ne; e++) sum += (double) sml_exp(lg[e] - mx);
            for (int e = 0; e < ne; e++) sc[e] = (float) ((double) sml_exp(lg[e] - mx) / sum);
        }
        for (int e = 0; e < ne; e++) ch[e] = sc[e] + (has_bias ? ebias[e] : 0.0f);
        if (ng > 1 && tg < ng && ne % ng == 0) {
            const int per = ne / ng;
            float gsc[64];
            int keep[64];
            for (int g = 0; g < 64; g++) keep[g] = 0;
            for (int g = 0; g < ng && g < 64; g++) {
                float b1 = -1e30f, b2 = -1e30f;
                for (int i = 0; i < per; i++) {
                    const float v = ch[g * per + i];
                    if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) b2 = v;
                }
                gsc[g] = b1 + (per > 1 ? b2 : 0.0f);
            }
            for (int t = 0; t < tg; t++) {
                int best = -1;
                for (int g = 0; g < ng; g++) if (!keep[g] && (best < 0 || gsc[g] > gsc[best])) best = g;
                keep[best] = 1;
            }
            for (int e = 0; e < ne; e++) if (!keep[e / per]) ch[e] = 0.0f;
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
        if (flags & 1) {
            float sum = 0.0f;
            for (int j = 0; j < k; j++) sum += w[j];
            for (int j = 0; j < k; j++) w[j] /= sum + 1e-20f;
        }
        if (mscale != 0.0f) for (int j = 0; j < k; j++) w[j] *= mscale;
        return;
    }
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int e = 0; e < ne; e++) {
            int taken = 0;
            for (int q = 0; q < j; q++) if (sel[q] == e) taken = 1;
            if (!taken && (best < 0 || lg[e] > lg[best])) best = e;
        }
        sel[j] = best;
    }
    if (flags & 1) {
        double sum = 0.0;
        for (int j = 0; j < k; j++) { w[j] = sml_exp(lg[sel[j]] - lg[sel[0]]); sum += w[j]; }
        for (int j = 0; j < k; j++) w[j] = (float) (w[j] / sum);
    } else {
        float mx = lg[0];
        for (int e = 1; e < ne; e++) if (lg[e] > mx) mx = lg[e];
        double sum = 0.0;
        for (int e = 0; e < ne; e++) sum += (double) sml_exp(lg[e] - mx);
        for (int j = 0; j < k; j++) w[j] = (float) ((double) sml_exp(lg[sel[j]] - mx) / sum);
    }
}

// Choice j's activation (x + j * xst; xst 0 = the same row for every choice),
// quantized in the column order of the expert it chose. Grid (groups, k).
extern "C" __attribute__((global)) void sml_g_moe_quant(
        const float * x, int xst, const long long * perms, const int * sel,
        unsigned int * wall, int wst, float * sall, int * mall, int sst, int n) {
    const int j = CTA_Y();
    const int lane = TID_X() & (WARP - 1);
    const int g = CTA_X() * (NTID_X() / WARP) + TID_X() / WARP;
    const int * perm = (const int *) perms[sel[j]];
    if (g < n / GROUP)
        quant_group_warp(x + (long long) j * xst, perm + g * GROUP, wall + (long long) j * wst + g * 8,
                         sall + (long long) j * sst + g, mall + (long long) j * sst + g, lane);
}

// Gate and up of choice j's expert (grid y = j), activated into hb + j * hst.
extern "C" __attribute__((global)) void sml_g_moe_gu(
        float * hb, int hst, const sml_mt * G, const sml_mt * U, const int * sel,
        const unsigned int * wall, int wst, const float * sall, const int * mall, int sst, int n) {
    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];
    const int j = CTA_Y();
    const int lane = TID_X() & (WARP - 1), warp = TID_X() / WARP, nwarp = NTID_X() / WARP;
    const unsigned int * words = wall + (long long) j * wst;
    const float * scale = sall + (long long) j * sst;
    const int * sums = mall + (long long) j * sst;
    FS_STAGE(n)
    const sml_mt g = G[sel[j]], u = U[sel[j]];
    float * out = hb + (long long) j * hst;
    for (int r = CTA_X() * nwarp + warp; r < g.rows; r += NCTA_X() * nwarp) {
        const float gg = fz_reduce(fs_row(g.c2, g.c4, g.sd, g.sq, g.md, g.mq, r, n, g.six, g.hm, sxq, sxs, sxm, lane));
        const float uu = fz_reduce(fs_row(u.c2, u.c4, u.sd, u.sq, u.md, u.mq, r, n, u.six, u.hm, sxq, sxs, sxm, lane));
        if (lane == 0) out[r] = sml_gated(gg, uu);
    }
}

// Down of choice j's expert (grid y = j) into t + j * tst.
extern "C" __attribute__((global)) void sml_g_moe_dn(
        float * t, int tst, const sml_mt * D, const int * sel,
        const unsigned int * wall, int wst, const float * sall, const int * mall, int sst, int n) {
    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];
    const int j = CTA_Y();
    const int lane = TID_X() & (WARP - 1), warp = TID_X() / WARP, nwarp = NTID_X() / WARP;
    const unsigned int * words = wall + (long long) j * wst;
    const float * scale = sall + (long long) j * sst;
    const int * sums = mall + (long long) j * sst;
    FS_STAGE(n)
    const sml_mt d = D[sel[j]];
    float * out = t + (long long) j * tst;
    for (int r = CTA_X() * nwarp + warp; r < d.rows; r += NCTA_X() * nwarp) {
        const float acc = fz_reduce(fs_row(d.c2, d.c4, d.sd, d.sq, d.md, d.mq, r, n, d.six, d.hm, sxq, sxs, sxm, lane));
        if (lane == 0) out[r] = acc;
    }
}

// y = the choices summed in order, w[j] * t[j], then + the shared expert s, gated
// by sigmoid(*sg) when the model has that gate (Qwen2-MoE) and by 1 when not.
extern "C" __attribute__((global)) void sml_g_moe_comb(
        float * y, const float * t, int tst, const float * w, int k,
        const float * s, int has_s, const float * sg, int has_sg, int n) {
    const int r = CTA_X() * NTID_X() + TID_X();
    if (r >= n) return;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) acc += w[j] * t[(long long) j * tst + r];
    if (has_s) {
        const float g = has_sg ? 1.0f / (1.0f + sml_exp(-*sg)) : 1.0f;
        acc += g * s[r];
    }
    y[r] = acc;
}

// ------------------------------------------------ experts on the batched prompt
//
// The prompt's router logits for every token at once (y[b][r] = W[r] . x[b], one
// block per (row, token), summed in double as the CPU router is), the batch's
// rows gathered for one expert, that expert's outputs added back to the rows of
// the tokens that chose it with their weights, and a zeroed accumulator.
extern "C" __attribute__((global)) void sml_g_f32mvb(
        float * y, int yst, const float * w, const float * x, int xst, int rows, int cols) {
    __attribute__((shared)) double part[256];
    const int r = CTA_X(), b = CTA_Y(), t = TID_X(), nt = NTID_X();
    const float * wr = w + (long long) r * cols;
    const float * xb = x + (long long) b * xst;
    double s = 0.0;
    for (int i = t; i < cols; i += nt) s += (double) wr[i] * xb[i];
    part[t] = s;
    SYNC();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (t < st) part[t] += part[t + st];
        SYNC();
    }
    if (t == 0 && r < rows) y[(long long) b * yst + r] = (float) part[0];
}

extern "C" __attribute__((global)) void sml_g_gatherb(
        float * dst, int dst_st, const float * src, int src_st, const int * idx, int m, int n) {
    const int i = CTA_X() * NTID_X() + TID_X(), r = CTA_Y();
    if (i >= n || r >= m) return;
    dst[(long long) r * dst_st + i] = src[(long long) idx[r] * src_st + i];
}

extern "C" __attribute__((global)) void sml_g_scatterb(
        float * dst, int dst_st, const float * src, int src_st, const int * idx, const float * w,
        int m, int n) {
    const int i = CTA_X() * NTID_X() + TID_X(), r = CTA_Y();
    if (i >= n || r >= m) return;
    dst[(long long) idx[r] * dst_st + i] += w[r] * src[(long long) r * src_st + i];
}

extern "C" __attribute__((global)) void sml_g_zerob(float * x, int xst, int n, int nb) {
    const int i = CTA_X() * NTID_X() + TID_X(), t = CTA_Y();
    if (i < n && t < nb) x[(long long) t * xst + i] = 0.0f;
}
