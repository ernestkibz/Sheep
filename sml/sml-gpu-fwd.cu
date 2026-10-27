// SML forward pass on NVIDIA GPUs, as NVPTX kernels built by clang with no CUDA
// toolkit. Every op in sml_forward, on the device, so a token never crosses PCIe
// except as one embedding row in and one token id out.
//
// Each kernel matches the CPU engine's semantics so the two forward passes can be
// graded against each other logit for logit:
//   * activation quantization: per-group amax/127, round-half-even, clamp 127
//   * matvec: int8 activations (the reference engine's GPU decode path also
//     quantizes activations to 8 bits, so this concedes nothing to it)
//   * flat 6-bit = 2-bit group-planar plane at x16 carrying the min, plus the
//     nibble plane without it -- exactly issue_rows()
//   * norms accumulate in double, softmax subtracts the max, exp uses the same
//     polynomial-plus-exponent-field construction as the AVX2 path
// Two deliberate differences, both toward accuracy: the KV cache is float32
// where the CPU stores fp16, and softmax uses that exp rather than libm's.
//
// No libdevice exists on this machine, so there is no libm: sqrt, exp and tanh
// are built from instructions below.

#define TID_X()  __nvvm_read_ptx_sreg_tid_x()
#define CTA_X()  __nvvm_read_ptx_sreg_ctaid_x()
#define NTID_X() __nvvm_read_ptx_sreg_ntid_x()
#define NCTA_X() __nvvm_read_ptx_sreg_nctaid_x()
#define SYNC()   __syncthreads()

#define GROUP            32
#define GPS              8          // groups per fp16 superblock scale
#define WARP             32
// Staged activation words, and their groups. The standard build stages n <= 16384; the
// wide build (-DMAX_W=8192 -DMAX_G=1024, sml-gpu-fwd-wide.ptx) n <= 32768, for models such
// as Qwen2.5-7B (FFN 18944), which overran the standard buffers into garbage. The wide one
// costs occupancy (the 3B: 12.9 -> 8.8 tok/s), so the host loads it only when needed.
#ifndef MAX_W
#define MAX_W            4096
#endif

typedef unsigned int uint4v __attribute__((ext_vector_type(4)));

// ------------------------------------------------------------------ helpers

__attribute__((device)) static inline float h2f(unsigned short h) {
    return (float) __builtin_bit_cast(_Float16, h);
}

// lrintf under the default rounding mode: nearest, ties to even.
__attribute__((device)) static inline int rint_i(float v) {
    int t = (int) v;
    const float fr = v - (float) t;
    if (fr > 0.5f) t += 1;
    else if (fr < -0.5f) t -= 1;
    else if (fr == 0.5f) { if (t & 1) t += 1; }
    else if (fr == -0.5f) { if (t & 1) t -= 1; }
    return t;
}

// exp(x) as 2^g polynomial times 2^r through the exponent field -- the same
// construction and coefficients as sml_exp_ps on the CPU.
__attribute__((device)) static inline float sml_exp(float x) {
    if (x < -88.0f) x = -88.0f;
    if (x >  88.0f) x =  88.0f;
    const float f = x * 1.44269504088896341f;
    const int   r = rint_i(f);
    const float g = f - (float) r;
    float p = 1.3697664e-3f;
    p = p * g + 8.3013611e-3f;
    p = p * g + 5.5826318e-2f;
    p = p * g + 2.4022652e-1f;
    p = p * g + 6.9314718e-1f;
    p = p * g + 1.0f;
    const unsigned int e = (unsigned int)(r + 127) << 23;
    return p * __builtin_bit_cast(float, e);
}

// Which GELU the model was trained with: set once by the host from the file's
// SML_ARCH_GELU_ERF (pythia says "gelu", the exact erf form); 0 = tanh form.
extern "C" { __attribute__((device)) int sml_gelu_erf = 0; }

// GELU for every GPU site. erf by Abramowitz and Stegun 7.1.26, 1 + erf written
// without cancellation (see gelu_erf_inplace on the CPU).
__attribute__((device)) static inline float sml_gelu(float v) {
    if (sml_gelu_erf) {
        const float z = v * 0.70710678118654752f;
        const float az = z < 0.0f ? -z : z;
        const float t = 1.0f / (1.0f + 0.3275911f * az);
        const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
                         + t * (-1.453152027f + t * 1.061405429f))));
        const float tail = poly * sml_exp(-az * az);
        return 0.5f * v * (z < 0.0f ? tail : 2.0f - tail);
    }
    const float z = 0.7978845608f * (v + 0.044715f * v * v * v);
    float th;
    if (z > 9.0f) th = 1.0f;
    else if (z < -9.0f) th = -1.0f;
    else { const float e = sml_exp(2.0f * z); th = (e - 1.0f) / (e + 1.0f); }
    return 0.5f * v * (1.0f + th);
}

// Which FFN activation, set once by the host from the file's bits: 0 = SiLU gate
// (or GELU without a gate), 1 = GELU gate (Gemma's GeGLU, SML_ARCH_GELU_GATE),
// 2 = relu(x)^2 without a gate (SML_ARCH_RELU2). Every FFN site goes through these
// two, so a family's activation is one flag, not a kernel per family.
extern "C" { __attribute__((device)) int sml_act = 0; }

__attribute__((device)) static inline float sml_gated(float g, float u) {
    if (sml_act == 1) return sml_gelu(g) * u;
    return (g / (1.0f + sml_exp(-g))) * u;
}

__attribute__((device)) static inline float sml_ungated(float u) {
    if (sml_act == 2) { const float r = u > 0.0f ? u : 0.0f; return r * r; }
    if (sml_act == 3) return u > 0.0f ? u : 0.0f;
    return sml_gelu(u);
}

// What an attention score goes through after the 1/sqrt(d) scale, as sml-run.c's
// attn_row does: ALiBi's slope x (key - query) per head (BLOOM, MPT, Falcon; the
// host uploads the CPU's own slopes), then Gemma 2's cap x tanh(s / cap). Both
// are set once by the host; 0 / null means none.
extern "C" { __attribute__((device)) float sml_attn_cap = 0.0f; }
extern "C" { __attribute__((device)) const float * sml_alibi = 0; }

__attribute__((device)) static inline float sml_score(float s, int h, int key, int query) {
    if (sml_alibi) s += sml_alibi[h] * (float) (key - query);
    if (sml_attn_cap > 0.0f) {
        const float z = s / sml_attn_cap;
        float th;
        if (z > 9.0f) th = 1.0f;
        else if (z < -9.0f) th = -1.0f;
        else { const float e = sml_exp(2.0f * z); th = (e - 1.0f) / (e + 1.0f); }
        s = sml_attn_cap * th;
    }
    return s;
}

// 1/sqrt in double: a single-precision bit-trick guess, then Newton. Four steps
// from that guess converge past double precision.
__attribute__((device)) static inline double rsqrt_d(double v) {
    // Seed from the hardware single-precision square root, then refine in double.
    // A float seed is good to ~1e-7 and each Newton step squares the error, so two
    // steps land past double precision. The older seed was the 1999 bit-trick
    // constant every graphics codebase carries; it is folklore rather than anyone
    // property, but it is also unnecessary when the GPU has a sqrt instruction.
    double y = (double)(1.0f / __builtin_sqrtf((float) v));
    y = y * (1.5 - 0.5 * v * y * y);
    y = y * (1.5 - 0.5 * v * y * y);
    return y;
}

// --------------------------------------------------------- element-wise ops

extern "C" __attribute__((global)) void sml_g_add(float * x, const float * y, int n) {
    for (int i = CTA_X() * NTID_X() + TID_X(); i < n; i += NCTA_X() * NTID_X())
        x[i] += y[i];
}

extern "C" __attribute__((global)) void sml_g_copy(float * d, const float * s, int n) {
    for (int i = CTA_X() * NTID_X() + TID_X(); i < n; i += NCTA_X() * NTID_X())
        d[i] = s[i];
}

extern "C" __attribute__((global)) void sml_g_silu(float * gate, const float * up, int n) {
    for (int i = CTA_X() * NTID_X() + TID_X(); i < n; i += NCTA_X() * NTID_X()) {
        const float g = gate[i];
        gate[i] = sml_gated(g, up[i]);
    }
}

// GELU, tanh form, matching gelu_inplace: tanh(z) = (e^2z - 1)/(e^2z + 1).
extern "C" __attribute__((global)) void sml_g_gelu(float * x, int n) {
    for (int i = CTA_X() * NTID_X() + TID_X(); i < n; i += NCTA_X() * NTID_X()) {
        x[i] = sml_ungated(x[i]);
    }
}

// ------------------------------------------------- activation quantization

// Gather through the tensor's column permutation and quantize per group, in one
// pass: q (n int8, permuted order), scale (ng), sums (ng). One thread per group.
extern "C" __attribute__((global)) void sml_g_quant(
        const float * x, const int * perm, signed char * q,
        float * scale, int * sums, int n) {
    const int ng = n / GROUP;
    for (int g = CTA_X() * NTID_X() + TID_X(); g < ng; g += NCTA_X() * NTID_X()) {
        const int * pp = perm + g * GROUP;
        float amax = 0.0f;
        for (int i = 0; i < GROUP; i++) {
            float v = x[pp[i]];
            if (v < 0.0f) v = -v;
            if (v > amax) amax = v;
        }
        const float s  = amax / 127.0f;
        const float is = s > 0.0f ? 1.0f / s : 0.0f;
        int sum = 0;
        signed char * qq = q + g * GROUP;
        for (int i = 0; i < GROUP; i++) {
            int v = rint_i(x[pp[i]] * is);
            if (v >  127) v =  127;
            if (v < -127) v = -127;
            qq[i] = (signed char) v;
            sum += v;
        }
        scale[g] = s;
        sums[g]  = sum;
    }
}

// ------------------------------------------------------------------ matvecs

__attribute__((device)) static inline void dot4(unsigned int w, unsigned int al,
                                                unsigned int ah, int * dl, int * dh) {
    int lo = 0, hi = 0;
    for (int b = 0; b < 4; b++) {
        const unsigned int byte = (w >> (8 * b)) & 0xFFu;
        const int xl = (int)(signed char)((al >> (8 * b)) & 0xFFu);
        const int xh = (int)(signed char)((ah >> (8 * b)) & 0xFFu);
        lo += (int)(byte & 0x0Fu)        * xl;
        hi += (int)((byte >> 4) & 0x0Fu) * xh;
    }
    *dl += lo; *dh += hi;
}

// Flat nibble plane. v4's validated layout: warp per row, activation staged as
// words in shared memory once per block. `add` accumulates into y, which the
// 6-bit path needs for its second plane.
extern "C" __attribute__((global)) void sml_g_mv4(
        float * y, const unsigned char * codes,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq,
        const signed char * xq, const float * xscale, const int * xsums,
        int rows, int n, int have_min, int add) {

    __attribute__((shared)) unsigned int sxq[MAX_W];

    const int ng     = n / GROUP;
    const int nscale = (ng + GPS - 1) / GPS;
    const int nblk   = n / 64;
    const int lane   = TID_X() & (WARP - 1);
    const int warp   = TID_X() / WARP;
    const int nwarp  = NTID_X() / WARP;
    const long long rowbytes = (long long) n / 2;

    for (int i = TID_X(); i < n / 4; i += NTID_X())
        sxq[i] = ((const unsigned int *) xq)[i];
    SYNC();

    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        const unsigned char  * cp  = codes + (long long) r * rowbytes;
        const unsigned short * dp  = sd    + (long long) r * nscale;
        const unsigned char  * qp  = sq    + (long long) r * ng;
        const unsigned short * mdp = md    + (long long) r * nscale;
        const unsigned char  * mqp = mq    + (long long) r * ng;

        float acc = 0.0f;
        for (int blk = lane; blk < nblk; blk += WARP) {
            const int gl = 2 * blk, gh = gl + 1;
            const uint4v * v = (const uint4v *)(cp + (long long) blk * 32);
            const uint4v w0 = v[0], w1 = v[1];
            const unsigned int * al = sxq + gl * (GROUP / 4);
            const unsigned int * ah = sxq + gh * (GROUP / 4);

            int dl = 0, dh = 0;
            dot4(w0.x, al[0], ah[0], &dl, &dh);
            dot4(w0.y, al[1], ah[1], &dl, &dh);
            dot4(w0.z, al[2], ah[2], &dl, &dh);
            dot4(w0.w, al[3], ah[3], &dl, &dh);
            dot4(w1.x, al[4], ah[4], &dl, &dh);
            dot4(w1.y, al[5], ah[5], &dl, &dh);
            dot4(w1.z, al[6], ah[6], &dl, &dh);
            dot4(w1.w, al[7], ah[7], &dl, &dh);

            const int sl = gl / GPS, sh = gh / GPS;
            acc += h2f(dp[sl]) * (float) qp[gl] * xscale[gl] * (float) dl;
            acc += h2f(dp[sh]) * (float) qp[gh] * xscale[gh] * (float) dh;
            if (have_min) {
                acc += h2f(mdp[sl]) * ((float) mqp[gl] - 127.0f) * xscale[gl] * (float) xsums[gl];
                acc += h2f(mdp[sh]) * ((float) mqp[gh] - 127.0f) * xscale[gh] * (float) xsums[gh];
            }
        }

        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __nvvm_shfl_sync_down_f32(0xffffffffu, acc, (unsigned) off, 31);
        if (lane == 0) y[r] = add ? y[r] + acc : acc;
    }
}

// One word of the 2-bit group-planar plane: byte b holds weight b of FOUR groups,
// group g in bits 2g..2g+1, each dotted against that group's own activation.
__attribute__((device)) static inline void dot2(unsigned int w,
        unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
        int * d0, int * d1, int * d2, int * d3) {
    for (int b = 0; b < 4; b++) {
        const unsigned int sh = 8u * (unsigned) b;
        const unsigned int byte = (w >> sh) & 0xFFu;
        *d0 += (int)( byte       & 3u) * (int)(signed char)((a0 >> sh) & 0xFFu);
        *d1 += (int)((byte >> 2) & 3u) * (int)(signed char)((a1 >> sh) & 0xFFu);
        *d2 += (int)((byte >> 4) & 3u) * (int)(signed char)((a2 >> sh) & 0xFFu);
        *d3 += (int)((byte >> 6) & 3u) * (int)(signed char)((a3 >> sh) & 0xFFu);
    }
}

// 2-bit group-planar plane: the ladder kernel's layout, used by flat 6-bit
// tensors for their top two bits at mul = 16, carrying the min term.
extern "C" __attribute__((global)) void sml_g_mv2(
        float * y, const unsigned char * codes,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq,
        const signed char * xq, const float * xscale, const int * xsums,
        int rows, int n, int have_min, int add, float mul) {

    __attribute__((shared)) unsigned int sxq[MAX_W];

    const int ng     = n / GROUP;
    const int nscale = (ng + GPS - 1) / GPS;
    const int npack  = n / 128;
    const int lane   = TID_X() & (WARP - 1);
    const int warp   = TID_X() / WARP;
    const int nwarp  = NTID_X() / WARP;
    const long long rowbytes = (long long) npack * 32;

    for (int i = TID_X(); i < n / 4; i += NTID_X())
        sxq[i] = ((const unsigned int *) xq)[i];
    SYNC();

    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        const unsigned char  * cp  = codes + (long long) r * rowbytes;
        const unsigned short * dp  = sd    + (long long) r * nscale;
        const unsigned char  * qp  = sq    + (long long) r * ng;
        const unsigned short * mdp = md    + (long long) r * nscale;
        const unsigned char  * mqp = mq    + (long long) r * ng;

        float acc = 0.0f;
        for (int p = lane; p < npack; p += WARP) {
            const uint4v * cv = (const uint4v *)(cp + (long long) p * 32);
            const uint4v c0 = cv[0], c1 = cv[1];
            const int base = p * 4;
            const unsigned int * a0 = sxq + (base + 0) * (GROUP / 4);
            const unsigned int * a1 = sxq + (base + 1) * (GROUP / 4);
            const unsigned int * a2 = sxq + (base + 2) * (GROUP / 4);
            const unsigned int * a3 = sxq + (base + 3) * (GROUP / 4);

            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            dot2(c0.x, a0[0], a1[0], a2[0], a3[0], &d0, &d1, &d2, &d3);
            dot2(c0.y, a0[1], a1[1], a2[1], a3[1], &d0, &d1, &d2, &d3);
            dot2(c0.z, a0[2], a1[2], a2[2], a3[2], &d0, &d1, &d2, &d3);
            dot2(c0.w, a0[3], a1[3], a2[3], a3[3], &d0, &d1, &d2, &d3);
            dot2(c1.x, a0[4], a1[4], a2[4], a3[4], &d0, &d1, &d2, &d3);
            dot2(c1.y, a0[5], a1[5], a2[5], a3[5], &d0, &d1, &d2, &d3);
            dot2(c1.z, a0[6], a1[6], a2[6], a3[6], &d0, &d1, &d2, &d3);
            dot2(c1.w, a0[7], a1[7], a2[7], a3[7], &d0, &d1, &d2, &d3);

            const int d[4] = { d0, d1, d2, d3 };
            for (int g = 0; g < 4; g++) {
                const int gi = base + g;
                const float ws = h2f(dp[gi / GPS]) * (float) qp[gi];
                acc += ws * xscale[gi] * ((float) d[g] * mul);
                if (have_min)
                    acc += h2f(mdp[gi / GPS]) * ((float) mqp[gi] - 127.0f)
                         * xscale[gi] * (float) xsums[gi];
            }
        }

        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __nvvm_shfl_sync_down_f32(0xffffffffu, acc, (unsigned) off, 31);
        if (lane == 0) y[r] = add ? y[r] + acc : acc;
    }
}

// ------------------------------------------------------------------- norms

// RMSNorm and LayerNorm in one kernel, as the CPU engine has one entry point.
// One block: the sums reduce across its threads in shared memory.
extern "C" __attribute__((global)) void sml_g_norm(
        float * out, const float * x, const float * w, const float * b,
        int n, float eps, int layer, int has_bias) {

    __attribute__((shared)) double part[256];

    const int t = TID_X(), nt = NTID_X();
    double mean = 0.0;
    if (layer == 1) {       // 0 RMSNorm, 1 LayerNorm, 2 none (the raw stream)
        double s = 0.0;
        for (int i = t; i < n; i += nt) s += (double) x[i];
        part[t] = s;
        SYNC();
        for (int st = nt >> 1; st > 0; st >>= 1) {
            if (t < st) part[t] += part[t + st];
            SYNC();
        }
        mean = part[0] / (double) n;
    }
    double ss = 0.0;
    for (int i = t; i < n; i += nt) { const double d = (double) x[i] - mean; ss += d * d; }
    part[t] = ss;
    SYNC();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (t < st) part[t] += part[t + st];
        SYNC();
    }
    const float var = (float)(part[0] / (double) n) + eps;
    const float scl = (float) rsqrt_d((double) var);
    for (int i = t; i < n; i += nt)
        out[i] = ((float)((double) x[i] - mean)) * scl * w[i] + (has_bias ? b[i] : 0.0f);
}

// ------------------------------------------------------- rope and attention

// cs and sn arrive already offset to this position's row of the table.
extern "C" __attribute__((global)) void sml_g_rope(
        float * x, const float * cs, const float * sn,
        int n_head, int hd, int half, int interleaved) {
    for (int h = CTA_X() * NTID_X() + TID_X(); h < n_head; h += NCTA_X() * NTID_X()) {
        float * v = x + h * hd;
        if (interleaved) {
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

// kc and vc arrive offset to this layer's rows: [t * kvd + kvh * hd + i].
extern "C" __attribute__((global)) void sml_g_attn(
        float * out, const float * q, const float * kc, const float * vc,
        float * att, int n_head, int n_kv, int hd, int n_ctx,
        int t0, int pos, float inv) {
    const int rep = n_head / n_kv;
    const long long kvd = (long long) n_kv * hd;
    for (int h = CTA_X() * NTID_X() + TID_X(); h < n_head; h += NCTA_X() * NTID_X()) {
        const float * qh = q + h * hd;
        float * a = att + (long long) h * n_ctx;
        const int kvh = h / rep;
        float mx = -3.4e38f;
        for (int t = t0; t <= pos; t++) {
            const float * kh = kc + (long long) t * kvd + (long long) kvh * hd;
            float s = 0.0f;
            for (int i = 0; i < hd; i++) s += qh[i] * kh[i];
            a[t] = sml_score(s * inv, h, t, pos);
            if (a[t] > mx) mx = a[t];
        }
        float sum = 0.0f;
        for (int t = t0; t <= pos; t++) { a[t] = sml_exp(a[t] - mx); sum += a[t]; }
        const float isum = 1.0f / sum;
        float * o = out + h * hd;
        for (int i = 0; i < hd; i++) o[i] = 0.0f;
        for (int t = t0; t <= pos; t++) {
            const float wt = a[t] * isum;
            const float * vh = vc + (long long) t * kvd + (long long) kvh * hd;
            for (int i = 0; i < hd; i++) o[i] += wt * vh[i];
        }
    }
}

// ----------------------------------------------------------------- argmax

// First index of the maximum, as sml_sample_greedy: ties go to the lower index.
extern "C" __attribute__((global)) void sml_g_argmax(int * out, const float * v, int n) {

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
    if (t == 0) out[0] = bi[0];
}

// ============================================================ fused kernels
//
// Profiling showed the small kernels (quant, add, norm, rope, copy) costing more
// as launches than as arithmetic: a 3B token issued about 22 launches per layer.
// These fold the activation quantize into the matvec (gathered and quantized
// straight into block shared memory, which also removes the staging copy), the
// bias into the matvec, the residual add into the following norm, and RoPE into
// the KV write. 12 launches per layer.

#ifndef MAX_G
#define MAX_G 512
#endif

// Group g of x, gathered through the permutation and quantized exactly as
// sml_g_quant does, written as eight words of int8.
__attribute__((device)) static inline void quant_group(const float * x, const int * pp,
        unsigned int * words, float * scale, int * sum) {
    float amax = 0.0f;
    for (int i = 0; i < GROUP; i++) {
        const float v = x[pp[i]];
        const float a = v < 0.0f ? -v : v;
        if (a > amax) amax = a;
    }
    const float s  = amax / 127.0f;
    const float is = s > 0.0f ? 1.0f / s : 0.0f;
    int acc = 0;
    for (int k = 0; k < 8; k++) {
        unsigned int w = 0;
        for (int b = 0; b < 4; b++) {
            int q = rint_i(x[pp[4 * k + b]] * is);
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            acc += q;
            w |= ((unsigned int)(unsigned char)(signed char) q) << (8 * b);
        }
        words[k] = w;
    }
    *scale = s;
    *sum = acc;
}

// Flat nibble tensor: quantize, matvec and bias in one launch.
extern "C" __attribute__((global)) void sml_g_qmv4(
        float * y, const unsigned char * codes,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq,
        const float * x, const int * perm, const float * bias,
        int rows, int n, int have_min, int has_bias) {

    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];

    const int ng     = n / GROUP;
    const int nscale = (ng + GPS - 1) / GPS;
    const int nblk   = n / 64;
    const int lane   = TID_X() & (WARP - 1);
    const int warp   = TID_X() / WARP;
    const int nwarp  = NTID_X() / WARP;
    const long long rowbytes = (long long) n / 2;

    for (int g = TID_X(); g < ng; g += NTID_X())
        quant_group(x, perm + g * GROUP, sxq + g * 8, &sxs[g], &sxm[g]);
    SYNC();

    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        const unsigned char  * cp  = codes + (long long) r * rowbytes;
        const unsigned short * dp  = sd    + (long long) r * nscale;
        const unsigned char  * qp  = sq    + (long long) r * ng;
        const unsigned short * mdp = md    + (long long) r * nscale;
        const unsigned char  * mqp = mq    + (long long) r * ng;

        float acc = 0.0f;
        for (int blk = lane; blk < nblk; blk += WARP) {
            const int gl = 2 * blk, gh = gl + 1;
            const uint4v * v = (const uint4v *)(cp + (long long) blk * 32);
            const uint4v w0 = v[0], w1 = v[1];
            const unsigned int * al = sxq + gl * 8;
            const unsigned int * ah = sxq + gh * 8;
            int dl = 0, dh = 0;
            dot4(w0.x, al[0], ah[0], &dl, &dh);
            dot4(w0.y, al[1], ah[1], &dl, &dh);
            dot4(w0.z, al[2], ah[2], &dl, &dh);
            dot4(w0.w, al[3], ah[3], &dl, &dh);
            dot4(w1.x, al[4], ah[4], &dl, &dh);
            dot4(w1.y, al[5], ah[5], &dl, &dh);
            dot4(w1.z, al[6], ah[6], &dl, &dh);
            dot4(w1.w, al[7], ah[7], &dl, &dh);
            const int sl = gl / GPS, sh = gh / GPS;
            acc += h2f(dp[sl]) * (float) qp[gl] * sxs[gl] * (float) dl;
            acc += h2f(dp[sh]) * (float) qp[gh] * sxs[gh] * (float) dh;
            if (have_min) {
                acc += h2f(mdp[sl]) * ((float) mqp[gl] - 127.0f) * sxs[gl] * (float) sxm[gl];
                acc += h2f(mdp[sh]) * ((float) mqp[gh] - 127.0f) * sxs[gh] * (float) sxm[gh];
            }
        }
        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __nvvm_shfl_sync_down_f32(0xffffffffu, acc, (unsigned) off, 31);
        if (lane == 0) y[r] = acc + (has_bias ? bias[r] : 0.0f);
    }
}

// Flat 6-bit tensor, both planes in ONE row loop. A pack of four groups owns 32
// bytes of the 2-bit plane and the 64 nibble bytes of blocks 2p and 2p+1, so a
// lane strides whole packs and each group code is base + 16*extra in one pass.
extern "C" __attribute__((global)) void sml_g_qmv6(
        float * y, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq,
        const float * x, const int * perm, const float * bias,
        int rows, int n, int have_min, int has_bias) {

    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];

    const int ng     = n / GROUP;
    const int nscale = (ng + GPS - 1) / GPS;
    const int npack  = n / 128;
    const int lane   = TID_X() & (WARP - 1);
    const int warp   = TID_X() / WARP;
    const int nwarp  = NTID_X() / WARP;
    const long long row2 = (long long) npack * 32;
    const long long row4 = (long long) n / 2;

    for (int g = TID_X(); g < ng; g += NTID_X())
        quant_group(x, perm + g * GROUP, sxq + g * 8, &sxs[g], &sxm[g]);
    SYNC();

    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        const unsigned char  * p2  = c2 + (long long) r * row2;
        const unsigned char  * p4  = c4 + (long long) r * row4;
        const unsigned short * dp  = sd + (long long) r * nscale;
        const unsigned char  * qp  = sq + (long long) r * ng;
        const unsigned short * mdp = md + (long long) r * nscale;
        const unsigned char  * mqp = mq + (long long) r * ng;

        float acc = 0.0f;
        for (int p = lane; p < npack; p += WARP) {
            const uint4v * pv = (const uint4v *)(p2 + (long long) p * 32);
            const uint4v * nv = (const uint4v *)(p4 + (long long) p * 64);
            const uint4v e0 = pv[0], e1 = pv[1];
            const uint4v n0 = nv[0], n1 = nv[1], n2 = nv[2], n3 = nv[3];
            const int base = p * 4;
            const unsigned int * a0 = sxq + (base + 0) * 8;
            const unsigned int * a1 = sxq + (base + 1) * 8;
            const unsigned int * a2 = sxq + (base + 2) * 8;
            const unsigned int * a3 = sxq + (base + 3) * 8;

            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            dot2(e0.x, a0[0], a1[0], a2[0], a3[0], &d0, &d1, &d2, &d3);
            dot2(e0.y, a0[1], a1[1], a2[1], a3[1], &d0, &d1, &d2, &d3);
            dot2(e0.z, a0[2], a1[2], a2[2], a3[2], &d0, &d1, &d2, &d3);
            dot2(e0.w, a0[3], a1[3], a2[3], a3[3], &d0, &d1, &d2, &d3);
            dot2(e1.x, a0[4], a1[4], a2[4], a3[4], &d0, &d1, &d2, &d3);
            dot2(e1.y, a0[5], a1[5], a2[5], a3[5], &d0, &d1, &d2, &d3);
            dot2(e1.z, a0[6], a1[6], a2[6], a3[6], &d0, &d1, &d2, &d3);
            dot2(e1.w, a0[7], a1[7], a2[7], a3[7], &d0, &d1, &d2, &d3);

            int b0 = 0, b1 = 0, b2 = 0, b3 = 0;
            dot4(n0.x, a0[0], a1[0], &b0, &b1); dot4(n0.y, a0[1], a1[1], &b0, &b1);
            dot4(n0.z, a0[2], a1[2], &b0, &b1); dot4(n0.w, a0[3], a1[3], &b0, &b1);
            dot4(n1.x, a0[4], a1[4], &b0, &b1); dot4(n1.y, a0[5], a1[5], &b0, &b1);
            dot4(n1.z, a0[6], a1[6], &b0, &b1); dot4(n1.w, a0[7], a1[7], &b0, &b1);
            dot4(n2.x, a2[0], a3[0], &b2, &b3); dot4(n2.y, a2[1], a3[1], &b2, &b3);
            dot4(n2.z, a2[2], a3[2], &b2, &b3); dot4(n2.w, a2[3], a3[3], &b2, &b3);
            dot4(n3.x, a2[4], a3[4], &b2, &b3); dot4(n3.y, a2[5], a3[5], &b2, &b3);
            dot4(n3.z, a2[6], a3[6], &b2, &b3); dot4(n3.w, a2[7], a3[7], &b2, &b3);

            const int dsum[4] = { b0 + 16 * d0, b1 + 16 * d1, b2 + 16 * d2, b3 + 16 * d3 };
            for (int g = 0; g < 4; g++) {
                const int gi = base + g;
                acc += h2f(dp[gi / GPS]) * (float) qp[gi] * sxs[gi] * (float) dsum[g];
                if (have_min)
                    acc += h2f(mdp[gi / GPS]) * ((float) mqp[gi] - 127.0f) * sxs[gi] * (float) sxm[gi];
            }
        }
        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __nvvm_shfl_sync_down_f32(0xffffffffu, acc, (unsigned) off, 31);
        if (lane == 0) y[r] = acc + (has_bias ? bias[r] : 0.0f);
    }
}

// Residual add folded into the norm that always follows it.
// mode 0: out = norm(x).  1: x += y, then out = norm(x).  2: out = norm(x), then
// x += y (parallel residual: the FFN norm reads the stream before attention adds).
extern "C" __attribute__((global)) void sml_g_addnorm(
        float * x, const float * y, float * out, const float * w, const float * b,
        int n, float eps, int layer, int has_bias, int mode) {

    __attribute__((shared)) double part[256];

    const int t = TID_X(), nt = NTID_X();
    if (mode == 1)
        for (int i = t; i < n; i += nt) x[i] += y[i];
    double mean = 0.0;
    if (layer == 1) {       // 0 RMSNorm, 1 LayerNorm, 2 none (the raw stream)
        double s = 0.0;
        for (int i = t; i < n; i += nt) s += (double) x[i];
        part[t] = s;
        SYNC();
        for (int st = nt >> 1; st > 0; st >>= 1) {
            if (t < st) part[t] += part[t + st];
            SYNC();
        }
        mean = part[0] / (double) n;
    }
    double ss = 0.0;
    for (int i = t; i < n; i += nt) { const double d = (double) x[i] - mean; ss += d * d; }
    part[t] = ss;
    SYNC();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (t < st) part[t] += part[t + st];
        SYNC();
    }
    const float var = (float)(part[0] / (double) n) + eps;
    const float scl = (float) rsqrt_d((double) var);
    for (int i = t; i < n; i += nt) {
        out[i] = layer == 2 ? x[i] : ((float)((double) x[i] - mean)) * scl * w[i] + (has_bias ? b[i] : 0.0f);
        if (mode == 2) x[i] += y[i];
    }
}

// RoPE on q and k, then k and v written into the cache at this position.
extern "C" __attribute__((global)) void sml_g_ropekv(
        float * q, float * k, const float * v, float * kc, float * vc,
        const float * cs, const float * sn,
        int n_head, int n_kv, int hd, int half, int interleaved) {
    for (int h = CTA_X() * NTID_X() + TID_X(); h < n_head; h += NCTA_X() * NTID_X()) {
        const int passes = h < n_kv ? 2 : 1;
        for (int pass = 0; pass < passes; pass++) {
            float * u = (pass == 0 ? q : k) + h * hd;
            if (interleaved) {
                for (int i = 0; i < half; i++) {
                    const float x0 = u[2 * i], x1 = u[2 * i + 1];
                    u[2 * i]     = x0 * cs[i] - x1 * sn[i];
                    u[2 * i + 1] = x0 * sn[i] + x1 * cs[i];
                }
            } else {
                for (int i = 0; i < half; i++) {
                    const float x0 = u[i], x1 = u[i + half];
                    u[i]        = x0 * cs[i] - x1 * sn[i];
                    u[i + half] = x0 * sn[i] + x1 * cs[i];
                }
            }
        }
        if (h < n_kv) {
            const float * kh = k + h * hd;
            const float * vh = v + h * hd;
            float * kd = kc + h * hd;
            float * vd = vc + h * hd;
            for (int i = 0; i < hd; i++) { kd[i] = kh[i]; vd[i] = vh[i]; }
        }
    }
}

// ============================================================ shared quantize
//
// q, k and v read the same input through the SAME permutation, and so do gate
// and up -- confirmed in the files. The fused kernels still gathered and
// quantized that activation once per BLOCK of every matvec: once per block per
// launch, three launches in a row, on identical input. And the block sweep
// showed per-block cost is what limits the forward: 16 blocks 9.9 tok/s on the
// 3B, 64 blocks 5.4, 256 blocks 2.0.
//
// So quantize ONCE per shared input into a global buffer, and let each block
// stage the finished words -- one word copy per four elements where there was a
// gather, compare, divide and round per element.

extern "C" __attribute__((global)) void sml_g_quantw(
        const float * x, const int * perm, unsigned int * words,
        float * scale, int * sums, int n) {
    const int ng = n / GROUP;
    for (int g = CTA_X() * NTID_X() + TID_X(); g < ng; g += NCTA_X() * NTID_X())
        quant_group(x, perm + g * GROUP, words + g * 8, scale + g, sums + g);
}

// Flat nibble tensor against an already-quantized activation.
extern "C" __attribute__((global)) void sml_g_smv4(
        float * y, const unsigned char * codes,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq,
        const unsigned int * words, const float * scale, const int * sums,
        const float * bias, int rows, int n, int have_min, int has_bias) {

    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];

    const int ng     = n / GROUP;
    const int nscale = (ng + GPS - 1) / GPS;
    const int nblk   = n / 64;
    const int lane   = TID_X() & (WARP - 1);
    const int warp   = TID_X() / WARP;
    const int nwarp  = NTID_X() / WARP;
    const long long rowbytes = (long long) n / 2;

    for (int i = TID_X(); i < n / 4; i += NTID_X()) sxq[i] = words[i];
    for (int g = TID_X(); g < ng; g += NTID_X()) { sxs[g] = scale[g]; sxm[g] = sums[g]; }
    SYNC();

    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        const unsigned char  * cp  = codes + (long long) r * rowbytes;
        const unsigned short * dp  = sd    + (long long) r * nscale;
        const unsigned char  * qp  = sq    + (long long) r * ng;
        const unsigned short * mdp = md    + (long long) r * nscale;
        const unsigned char  * mqp = mq    + (long long) r * ng;

        float acc = 0.0f;
        for (int blk = lane; blk < nblk; blk += WARP) {
            const int gl = 2 * blk, gh = gl + 1;
            const uint4v * v = (const uint4v *)(cp + (long long) blk * 32);
            const uint4v w0 = v[0], w1 = v[1];
            const unsigned int * al = sxq + gl * 8;
            const unsigned int * ah = sxq + gh * 8;
            int dl = 0, dh = 0;
            dot4(w0.x, al[0], ah[0], &dl, &dh);
            dot4(w0.y, al[1], ah[1], &dl, &dh);
            dot4(w0.z, al[2], ah[2], &dl, &dh);
            dot4(w0.w, al[3], ah[3], &dl, &dh);
            dot4(w1.x, al[4], ah[4], &dl, &dh);
            dot4(w1.y, al[5], ah[5], &dl, &dh);
            dot4(w1.z, al[6], ah[6], &dl, &dh);
            dot4(w1.w, al[7], ah[7], &dl, &dh);
            const int sl = gl / GPS, sh = gh / GPS;
            acc += h2f(dp[sl]) * (float) qp[gl] * sxs[gl] * (float) dl;
            acc += h2f(dp[sh]) * (float) qp[gh] * sxs[gh] * (float) dh;
            if (have_min) {
                acc += h2f(mdp[sl]) * ((float) mqp[gl] - 127.0f) * sxs[gl] * (float) sxm[gl];
                acc += h2f(mdp[sh]) * ((float) mqp[gh] - 127.0f) * sxs[gh] * (float) sxm[gh];
            }
        }
        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __nvvm_shfl_sync_down_f32(0xffffffffu, acc, (unsigned) off, 31);
        if (lane == 0) y[r] = acc + (has_bias ? bias[r] : 0.0f);
    }
}

// Flat 6-bit tensor against an already-quantized activation: both planes in one
// row loop, as sml_g_qmv6.
extern "C" __attribute__((global)) void sml_g_smv6(
        float * y, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq,
        const unsigned int * words, const float * scale, const int * sums,
        const float * bias, int rows, int n, int have_min, int has_bias) {

    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];

    const int ng     = n / GROUP;
    const int nscale = (ng + GPS - 1) / GPS;
    const int npack  = n / 128;
    const int lane   = TID_X() & (WARP - 1);
    const int warp   = TID_X() / WARP;
    const int nwarp  = NTID_X() / WARP;
    const long long row2 = (long long) npack * 32;
    const long long row4 = (long long) n / 2;

    for (int i = TID_X(); i < n / 4; i += NTID_X()) sxq[i] = words[i];
    for (int g = TID_X(); g < ng; g += NTID_X()) { sxs[g] = scale[g]; sxm[g] = sums[g]; }
    SYNC();

    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        const unsigned char  * p2  = c2 + (long long) r * row2;
        const unsigned char  * p4  = c4 + (long long) r * row4;
        const unsigned short * dp  = sd + (long long) r * nscale;
        const unsigned char  * qp  = sq + (long long) r * ng;
        const unsigned short * mdp = md + (long long) r * nscale;
        const unsigned char  * mqp = mq + (long long) r * ng;

        float acc = 0.0f;
        for (int p = lane; p < npack; p += WARP) {
            const uint4v * pv = (const uint4v *)(p2 + (long long) p * 32);
            const uint4v * nv = (const uint4v *)(p4 + (long long) p * 64);
            const uint4v e0 = pv[0], e1 = pv[1];
            const uint4v n0 = nv[0], n1 = nv[1], n2 = nv[2], n3 = nv[3];
            const int base = p * 4;
            const unsigned int * a0 = sxq + (base + 0) * 8;
            const unsigned int * a1 = sxq + (base + 1) * 8;
            const unsigned int * a2 = sxq + (base + 2) * 8;
            const unsigned int * a3 = sxq + (base + 3) * 8;

            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            dot2(e0.x, a0[0], a1[0], a2[0], a3[0], &d0, &d1, &d2, &d3);
            dot2(e0.y, a0[1], a1[1], a2[1], a3[1], &d0, &d1, &d2, &d3);
            dot2(e0.z, a0[2], a1[2], a2[2], a3[2], &d0, &d1, &d2, &d3);
            dot2(e0.w, a0[3], a1[3], a2[3], a3[3], &d0, &d1, &d2, &d3);
            dot2(e1.x, a0[4], a1[4], a2[4], a3[4], &d0, &d1, &d2, &d3);
            dot2(e1.y, a0[5], a1[5], a2[5], a3[5], &d0, &d1, &d2, &d3);
            dot2(e1.z, a0[6], a1[6], a2[6], a3[6], &d0, &d1, &d2, &d3);
            dot2(e1.w, a0[7], a1[7], a2[7], a3[7], &d0, &d1, &d2, &d3);

            int b0 = 0, b1 = 0, b2 = 0, b3 = 0;
            dot4(n0.x, a0[0], a1[0], &b0, &b1); dot4(n0.y, a0[1], a1[1], &b0, &b1);
            dot4(n0.z, a0[2], a1[2], &b0, &b1); dot4(n0.w, a0[3], a1[3], &b0, &b1);
            dot4(n1.x, a0[4], a1[4], &b0, &b1); dot4(n1.y, a0[5], a1[5], &b0, &b1);
            dot4(n1.z, a0[6], a1[6], &b0, &b1); dot4(n1.w, a0[7], a1[7], &b0, &b1);
            dot4(n2.x, a2[0], a3[0], &b2, &b3); dot4(n2.y, a2[1], a3[1], &b2, &b3);
            dot4(n2.z, a2[2], a3[2], &b2, &b3); dot4(n2.w, a2[3], a3[3], &b2, &b3);
            dot4(n3.x, a2[4], a3[4], &b2, &b3); dot4(n3.y, a2[5], a3[5], &b2, &b3);
            dot4(n3.z, a2[6], a3[6], &b2, &b3); dot4(n3.w, a2[7], a3[7], &b2, &b3);

            const int dsum[4] = { b0 + 16 * d0, b1 + 16 * d1, b2 + 16 * d2, b3 + 16 * d3 };
            for (int g = 0; g < 4; g++) {
                const int gi = base + g;
                acc += h2f(dp[gi / GPS]) * (float) qp[gi] * sxs[gi] * (float) dsum[g];
                if (have_min)
                    acc += h2f(mdp[gi / GPS]) * ((float) mqp[gi] - 127.0f) * sxs[gi] * (float) sxm[gi];
            }
        }
        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __nvvm_shfl_sync_down_f32(0xffffffffu, acc, (unsigned) off, 31);
        if (lane == 0) y[r] = acc + (has_bias ? bias[r] : 0.0f);
    }
}

// ============================================================ dp4a matvecs
//
// The inner product of 4-bit codes and int8 activations is what dp4a exists
// for: one instruction multiplies four unsigned bytes by four signed bytes and
// adds the sum to an accumulator. The shared-path kernels spent eight bit
// extractions and four multiplies per four code bytes to do the same thing.
//
// clang 18 cannot emit dp4a, so the kernels call the placeholder below and
// ptx_dp4a.py swaps every call for the instruction. Semantics were measured on
// this GPU first: dp4a.u32.s32 dst, a, b, acc -> acc + sum u8(a_i) * s8(b_i).
//
// Masking a code word with 0x0F0F0F0F leaves the low nibbles as whole bytes, and
// shifting by 4 first leaves the high nibbles -- so one word of codes becomes two
// dp4a operands, one per group, each against that group own activation word.

extern "C" __attribute__((device)) unsigned int sml_dp4a_us(unsigned int acc, unsigned int a, unsigned int b);

#define LO4 0x0F0F0F0Fu
#define LO2 0x03030303u

#define DN(W, K, X, Y, AL, AH) X = sml_dp4a_us(X, (W) & LO4, AL[K]); Y = sml_dp4a_us(Y, ((W) >> 4) & LO4, AH[K]);
#define D2(W, K) d0 = sml_dp4a_us(d0, (W) & LO2, a0[K]); d1 = sml_dp4a_us(d1, ((W) >> 2) & LO2, a1[K]); d2 = sml_dp4a_us(d2, ((W) >> 4) & LO2, a2[K]); d3 = sml_dp4a_us(d3, ((W) >> 6) & LO2, a3[K]);

extern "C" __attribute__((global)) void sml_g_dmv4(
        float * y, const unsigned char * codes,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq,
        const unsigned int * words, const float * scale, const int * sums,
        const float * bias, int rows, int n, int have_min, int has_bias) {

    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];

    const int ng     = n / GROUP;
    const int nscale = (ng + GPS - 1) / GPS;
    const int nblk   = n / 64;
    const int lane   = TID_X() & (WARP - 1);
    const int warp   = TID_X() / WARP;
    const int nwarp  = NTID_X() / WARP;
    const long long rowbytes = (long long) n / 2;

    for (int i = TID_X(); i < n / 4; i += NTID_X()) sxq[i] = words[i];
    for (int g = TID_X(); g < ng; g += NTID_X()) { sxs[g] = scale[g]; sxm[g] = sums[g]; }
    SYNC();

    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        const unsigned char  * cp  = codes + (long long) r * rowbytes;
        const unsigned short * dp  = sd    + (long long) r * nscale;
        const unsigned char  * qp  = sq    + (long long) r * ng;
        const unsigned short * mdp = md    + (long long) r * nscale;
        const unsigned char  * mqp = mq    + (long long) r * ng;

        float acc = 0.0f;
        for (int blk = lane; blk < nblk; blk += WARP) {
            const int gl = 2 * blk, gh = gl + 1;
            const uint4v * v = (const uint4v *)(cp + (long long) blk * 32);
            const uint4v w0 = v[0], w1 = v[1];
            const unsigned int * al = sxq + gl * 8;
            const unsigned int * ah = sxq + gh * 8;
            unsigned int dl = 0u, dh = 0u;
            DN(w0.x, 0, dl, dh, al, ah)
            DN(w0.y, 1, dl, dh, al, ah)
            DN(w0.z, 2, dl, dh, al, ah)
            DN(w0.w, 3, dl, dh, al, ah)
            DN(w1.x, 4, dl, dh, al, ah)
            DN(w1.y, 5, dl, dh, al, ah)
            DN(w1.z, 6, dl, dh, al, ah)
            DN(w1.w, 7, dl, dh, al, ah)
            const int sl = gl / GPS, sh = gh / GPS;
            acc += h2f(dp[sl]) * (float) qp[gl] * sxs[gl] * (float)(int) dl;
            acc += h2f(dp[sh]) * (float) qp[gh] * sxs[gh] * (float)(int) dh;
            if (have_min) {
                acc += h2f(mdp[sl]) * ((float) mqp[gl] - 127.0f) * sxs[gl] * (float) sxm[gl];
                acc += h2f(mdp[sh]) * ((float) mqp[gh] - 127.0f) * sxs[gh] * (float) sxm[gh];
            }
        }
        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __nvvm_shfl_sync_down_f32(0xffffffffu, acc, (unsigned) off, 31);
        if (lane == 0) y[r] = acc + (has_bias ? bias[r] : 0.0f);
    }
}

extern "C" __attribute__((global)) void sml_g_dmv6(
        float * y, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq,
        const unsigned int * words, const float * scale, const int * sums,
        const float * bias, int rows, int n, int have_min, int has_bias) {

    __attribute__((shared)) unsigned int sxq[MAX_W];
    __attribute__((shared)) float        sxs[MAX_G];
    __attribute__((shared)) int          sxm[MAX_G];

    const int ng     = n / GROUP;
    const int nscale = (ng + GPS - 1) / GPS;
    const int npack  = n / 128;
    const int lane   = TID_X() & (WARP - 1);
    const int warp   = TID_X() / WARP;
    const int nwarp  = NTID_X() / WARP;
    const long long row2 = (long long) npack * 32;
    const long long row4 = (long long) n / 2;

    for (int i = TID_X(); i < n / 4; i += NTID_X()) sxq[i] = words[i];
    for (int g = TID_X(); g < ng; g += NTID_X()) { sxs[g] = scale[g]; sxm[g] = sums[g]; }
    SYNC();

    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        const unsigned char  * p2  = c2 + (long long) r * row2;
        const unsigned char  * p4  = c4 + (long long) r * row4;
        const unsigned short * dp  = sd + (long long) r * nscale;
        const unsigned char  * qp  = sq + (long long) r * ng;
        const unsigned short * mdp = md + (long long) r * nscale;
        const unsigned char  * mqp = mq + (long long) r * ng;

        float acc = 0.0f;
        for (int p = lane; p < npack; p += WARP) {
            const uint4v * pv = (const uint4v *)(p2 + (long long) p * 32);
            const uint4v * nv = (const uint4v *)(p4 + (long long) p * 64);
            const uint4v e0 = pv[0], e1 = pv[1];
            const uint4v n0 = nv[0], n1 = nv[1], n2 = nv[2], n3 = nv[3];
            const int base = p * 4;
            const unsigned int * a0 = sxq + (base + 0) * 8;
            const unsigned int * a1 = sxq + (base + 1) * 8;
            const unsigned int * a2 = sxq + (base + 2) * 8;
            const unsigned int * a3 = sxq + (base + 3) * 8;

            unsigned int d0 = 0u, d1 = 0u, d2 = 0u, d3 = 0u;
            D2(e0.x, 0) D2(e0.y, 1) D2(e0.z, 2) D2(e0.w, 3)
            D2(e1.x, 4) D2(e1.y, 5) D2(e1.z, 6) D2(e1.w, 7)

            unsigned int b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
            DN(n0.x, 0, b0, b1, a0, a1) DN(n0.y, 1, b0, b1, a0, a1)
            DN(n0.z, 2, b0, b1, a0, a1) DN(n0.w, 3, b0, b1, a0, a1)
            DN(n1.x, 4, b0, b1, a0, a1) DN(n1.y, 5, b0, b1, a0, a1)
            DN(n1.z, 6, b0, b1, a0, a1) DN(n1.w, 7, b0, b1, a0, a1)
            DN(n2.x, 0, b2, b3, a2, a3) DN(n2.y, 1, b2, b3, a2, a3)
            DN(n2.z, 2, b2, b3, a2, a3) DN(n2.w, 3, b2, b3, a2, a3)
            DN(n3.x, 4, b2, b3, a2, a3) DN(n3.y, 5, b2, b3, a2, a3)
            DN(n3.z, 6, b2, b3, a2, a3) DN(n3.w, 7, b2, b3, a2, a3)

            const int dsum[4] = { (int) b0 + 16 * (int) d0, (int) b1 + 16 * (int) d1,
                                  (int) b2 + 16 * (int) d2, (int) b3 + 16 * (int) d3 };
            for (int g = 0; g < 4; g++) {
                const int gi = base + g;
                acc += h2f(dp[gi / GPS]) * (float) qp[gi] * sxs[gi] * (float) dsum[g];
                if (have_min)
                    acc += h2f(mdp[gi / GPS]) * ((float) mqp[gi] - 127.0f) * sxs[gi] * (float) sxm[gi];
            }
        }
        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __nvvm_shfl_sync_down_f32(0xffffffffu, acc, (unsigned) off, 31);
        if (lane == 0) y[r] = acc + (has_bias ? bias[r] : 0.0f);
    }
}

// ============================================================ fuse2: 7 launches per layer
//
// Launch submission costs 13.4 us here, and the dp4a forward issued 364 launches
// per pythia token -- 4.9 ms of submission alone, more than the reference engine
// spends on a whole token (4.3 ms). The matvec arithmetic is not what limits the
// forward: replacing it with dp4a moved decode 1-8%. So this file cuts launches.
//
//   norm + quantize                  one single-block launch
//   q, k, v                          one launch, one staged activation
//   RoPE + KV write + attention      one single-block launch
//   o projection, own quantize       one launch
//   norm + quantize                  one single-block launch
//   gate + up + SiLU (or up + GELU)  one launch: row r of gate and of up land in the
//                                    same warp, so the activation runs right there
//   down projection, own quantize    one launch
//
// Every matvec reads its codes with dp4a (placeholder, post-processed).

#define FZ_ROWBYTES4(n) ((long long) (n) / 2)
#define FZ_ROWBYTES2(n) ((long long) ((n) / 128) * 32)

__attribute__((device, always_inline)) static inline float fz_row4(
        const unsigned char * cp, const unsigned short * dp, const unsigned char * qp,
        const unsigned short * mdp, const unsigned char * mqp,
        const unsigned int * sxq, const float * sxs, const int * sxm,
        int nblk, int lane, int hm) {
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += WARP) {
        const int gl = 2 * blk, gh = gl + 1;
        const uint4v * v = (const uint4v *)(cp + (long long) blk * 32);
        const uint4v w0 = v[0], w1 = v[1];
        const unsigned int * al = sxq + gl * 8;
        const unsigned int * ah = sxq + gh * 8;
        unsigned int dl = 0u, dh = 0u;
        DN(w0.x, 0, dl, dh, al, ah)
        DN(w0.y, 1, dl, dh, al, ah)
        DN(w0.z, 2, dl, dh, al, ah)
        DN(w0.w, 3, dl, dh, al, ah)
        DN(w1.x, 4, dl, dh, al, ah)
        DN(w1.y, 5, dl, dh, al, ah)
        DN(w1.z, 6, dl, dh, al, ah)
        DN(w1.w, 7, dl, dh, al, ah)
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

__attribute__((device, always_inline)) static inline float fz_row6(
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
        const unsigned int * a0 = sxq + (base + 0) * 8;
        const unsigned int * a1 = sxq + (base + 1) * 8;
        const unsigned int * a2 = sxq + (base + 2) * 8;
        const unsigned int * a3 = sxq + (base + 3) * 8;
        unsigned int d0 = 0u, d1 = 0u, d2 = 0u, d3 = 0u;
        D2(e0.x, 0) D2(e0.y, 1) D2(e0.z, 2) D2(e0.w, 3)
        D2(e1.x, 4) D2(e1.y, 5) D2(e1.z, 6) D2(e1.w, 7)
        unsigned int b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
        DN(n0.x, 0, b0, b1, a0, a1) DN(n0.y, 1, b0, b1, a0, a1)
        DN(n0.z, 2, b0, b1, a0, a1) DN(n0.w, 3, b0, b1, a0, a1)
        DN(n1.x, 4, b0, b1, a0, a1) DN(n1.y, 5, b0, b1, a0, a1)
        DN(n1.z, 6, b0, b1, a0, a1) DN(n1.w, 7, b0, b1, a0, a1)
        DN(n2.x, 0, b2, b3, a2, a3) DN(n2.y, 1, b2, b3, a2, a3)
        DN(n2.z, 2, b2, b3, a2, a3) DN(n2.w, 3, b2, b3, a2, a3)
        DN(n3.x, 4, b2, b3, a2, a3) DN(n3.y, 5, b2, b3, a2, a3)
        DN(n3.z, 6, b2, b3, a2, a3) DN(n3.w, 7, b2, b3, a2, a3)
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

// One row of one tensor against the staged activation, before reduction.
__attribute__((device, always_inline)) static inline float fz_row(
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
        return fz_row6(c2 + (long long) r * FZ_ROWBYTES2(n), c4 + (long long) r * FZ_ROWBYTES4(n),
                       dp, qp, mdp, mqp, sxq, sxs, sxm, n / 128, lane, hm);
    return fz_row4(c4 + (long long) r * FZ_ROWBYTES4(n), dp, qp, mdp, mqp,
                   sxq, sxs, sxm, n / 64, lane, hm);
}

__attribute__((device, always_inline)) static inline float fz_reduce(float acc) {
    for (int off = WARP / 2; off > 0; off >>= 1)
        acc += __nvvm_shfl_sync_down_f32(0xffffffffu, acc, (unsigned) off, 31);
    return acc;
}

// Every row of one tensor that this warp owns.
__attribute__((device, always_inline)) static inline void fz_rows(
        float * y, const unsigned char * c2, const unsigned char * c4,
        const unsigned short * sd, const unsigned char * sq,
        const unsigned short * md, const unsigned char * mq, const float * bias,
        int rows, int n, int six, int hm, int hb,
        const unsigned int * sxq, const float * sxs, const int * sxm,
        int lane, int warp, int nwarp) {
    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        const float acc = fz_reduce(fz_row(c2, c4, sd, sq, md, mq, r, n, six, hm, sxq, sxs, sxm, lane));
        if (lane == 0) y[r] = acc + (hb ? bias[r] : 0.0f);
    }
}

// Stage the global quantized activation into this block.
#define FZ_STAGE(n)                                                            \
    for (int i = TID_X(); i < (n) / 4; i += NTID_X()) sxq[i] = words[i];       \
    for (int g = TID_X(); g < (n) / GROUP; g += NTID_X()) { sxs[g] = scale[g]; sxm[g] = sums[g]; } \
    SYNC();

// ---------------------------------------------------------------- normq
// addnorm (modes as sml_g_addnorm) followed by quantizing the normed output
// through `perm` into the global words/scale/sums every following matvec stages.
extern "C" __attribute__((global)) void sml_g_normq(
        float * x, const float * y, float * out, const float * w, const float * b,
        const int * perm, unsigned int * words, float * scale, int * sums,
        int n, float eps, int layer, int has_bias, int mode) {

    __attribute__((shared)) double part[256];

    const int t = TID_X(), nt = NTID_X();
    if (mode == 1)
        for (int i = t; i < n; i += nt) x[i] += y[i];
    double mean = 0.0;
    if (layer == 1) {       // 0 RMSNorm, 1 LayerNorm, 2 none (the raw stream)
        double s = 0.0;
        for (int i = t; i < n; i += nt) s += (double) x[i];
        part[t] = s;
        SYNC();
        for (int st = nt >> 1; st > 0; st >>= 1) {
            if (t < st) part[t] += part[t + st];
            SYNC();
        }
        mean = part[0] / (double) n;
    }
    double ss = 0.0;
    for (int i = t; i < n; i += nt) { const double d = (double) x[i] - mean; ss += d * d; }
    part[t] = ss;
    SYNC();
    for (int st = nt >> 1; st > 0; st >>= 1) {
        if (t < st) part[t] += part[t + st];
        SYNC();
    }
    const float var = (float)(part[0] / (double) n) + eps;
    const float scl = (float) rsqrt_d((double) var);
    for (int i = t; i < n; i += nt) {
        out[i] = layer == 2 ? x[i] : ((float)((double) x[i] - mean)) * scl * w[i] + (has_bias ? b[i] : 0.0f);
        if (mode == 2) x[i] += y[i];
    }
    SYNC();
    for (int g = t; g < n / GROUP; g += nt)
        quant_group(out, perm + g * GROUP, words + g * 8, scale + g, sums + g);
}

// ---------------------------------------------------------------- mv3
// Up to three tensors reading the same staged activation (q, k, v; or the head).
// An unused slot has rows 0.
extern "C" __attribute__((global)) void sml_g_mv3(
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
    FZ_STAGE(n)
    fz_rows(y0, c20, c40, sd0, sq0, md0, mq0, b0, rows0, n, six0, hm0, hb0, sxq, sxs, sxm, lane, warp, nwarp);
    fz_rows(y1, c21, c41, sd1, sq1, md1, mq1, b1, rows1, n, six1, hm1, hb1, sxq, sxs, sxm, lane, warp, nwarp);
    fz_rows(y2, c22, c42, sd2, sq2, md2, mq2, b2, rows2, n, six2, hm2, hb2, sxq, sxs, sxm, lane, warp, nwarp);
}

// ---------------------------------------------------------------- ffn
// gate + up + SiLU into hb, or (has_gate 0) up + GELU into hb. Rows of gate and up
// share an index and so a warp, so the activation runs as soon as both reduce.
extern "C" __attribute__((global)) void sml_g_ffn(
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
    FZ_STAGE(n)
    for (int r = CTA_X() * nwarp + warp; r < rows; r += NCTA_X() * nwarp) {
        float g = 0.0f;
        if (has_gate)
            g = fz_reduce(fz_row(gc2, gc4, gsd, gsq, gmd, gmq, r, n, gsix, ghm, sxq, sxs, sxm, lane));
        const float u = fz_reduce(fz_row(uc2, uc4, usd, usq, umd, umq, r, n, usix, uhm, sxq, sxs, sxm, lane));
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

// ---------------------------------------------------------------- qdmv
// One tensor with its own per-block quantize: the o and down projections, whose
// inputs are finished by the previous launch.
extern "C" __attribute__((global)) void sml_g_qdmv(
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
    for (int g = TID_X(); g < n / GROUP; g += NTID_X())
        quant_group(x, perm + g * GROUP, sxq + g * 8, &sxs[g], &sxm[g]);
    SYNC();
    fz_rows(y, c2, c4, sd, sq, md, mq, bias, rows, n, six, hm, hb, sxq, sxs, sxm, lane, warp, nwarp);
}

// ---------------------------------------------------------------- ropeattn
// Phase 1: RoPE on q and k, k and v into the cache. Phase 2: attention, which
// reads the position phase 1 just wrote -- hence the barrier between them.
extern "C" __attribute__((global)) void sml_g_ropeattn(
        float * out, float * q, float * k, const float * v,
        float * kat, float * vat, const float * kc, const float * vc, float * att,
        const float * cs, const float * sn,
        int n_head, int n_kv, int hd, int half, int interleaved,
        int n_ctx, int t0, int pos, float inv) {
    const int t = TID_X(), nt = NTID_X();
    for (int h = t; h < n_head; h += nt) {
        const int passes = h < n_kv ? 2 : 1;
        for (int pass = 0; pass < passes; pass++) {
            float * u = (pass == 0 ? q : k) + h * hd;
            if (interleaved) {
                for (int i = 0; i < half; i++) {
                    const float x0 = u[2 * i], x1 = u[2 * i + 1];
                    u[2 * i]     = x0 * cs[i] - x1 * sn[i];
                    u[2 * i + 1] = x0 * sn[i] + x1 * cs[i];
                }
            } else {
                for (int i = 0; i < half; i++) {
                    const float x0 = u[i], x1 = u[i + half];
                    u[i]        = x0 * cs[i] - x1 * sn[i];
                    u[i + half] = x0 * sn[i] + x1 * cs[i];
                }
            }
        }
        if (h < n_kv) {
            const float * kh = k + h * hd;
            const float * vh = v + h * hd;
            float * kd = kat + h * hd;
            float * vd = vat + h * hd;
            for (int i = 0; i < hd; i++) { kd[i] = kh[i]; vd[i] = vh[i]; }
        }
    }
    SYNC();
    const int rep = n_head / n_kv;
    const long long kvd = (long long) n_kv * hd;
    for (int h = t; h < n_head; h += nt) {
        const float * qh = q + h * hd;
        float * a = att + (long long) h * n_ctx;
        const int kvh = h / rep;
        float mx = -3.4e38f;
        for (int tt = t0; tt <= pos; tt++) {
            const float * kh = kc + (long long) tt * kvd + (long long) kvh * hd;
            float s = 0.0f;
            for (int i = 0; i < hd; i++) s += qh[i] * kh[i];
            a[tt] = sml_score(s * inv, h, tt, pos);
            if (a[tt] > mx) mx = a[tt];
        }
        float sum = 0.0f;
        for (int tt = t0; tt <= pos; tt++) { a[tt] = sml_exp(a[tt] - mx); sum += a[tt]; }
        const float isum = 1.0f / sum;
        float * o = out + h * hd;
        for (int i = 0; i < hd; i++) o[i] = 0.0f;
        for (int tt = t0; tt <= pos; tt++) {
            const float wt = a[tt] * isum;
            const float * vh = vc + (long long) tt * kvd + (long long) kvh * hd;
            for (int i = 0; i < hd; i++) o[i] += wt * vh[i];
        }
    }
}

#include "sml-gpu-fast.cu"
