// SML flat 4-bit matvec for NVIDIA GPUs, via NVPTX.
//
// The measurements that got here, each one naming the next problem:
//
//   v1  thread per group, block per row          22.6 GB/s
//       Groups 4q and 4q+1 are the low and high nibbles of the SAME 32 bytes,
//       so every byte was fetched twice.
//   v2  thread per group-PAIR, byte read once    17.0 GB/s  (WORSE)
//       Halving traffic and losing means traffic was not the constraint.
//   v3  warp per row, shuffle reduction, native cvt.f32.f16   18.3 GB/s
//       Removed 2048 block-wide reductions and a data-dependent `while` loop
//       in the half decode. Barely moved, so neither was the constraint either.
//
// What the three of them were missing: the WEIGHTS are 13 MB but every warp
// re-reads the whole ACTIVATION, one byte at a time. 2048 rows x 11 KB is 22 MB
// of activation traffic against 13 MB of weights -- the thing being streamed
// was never the model. Staging the activation once per block turns 64 scalar
// global loads per iteration into shared-memory word reads.
//
// This is the same lesson as the ladder, in a different place: the arithmetic
// was being tuned on top of the wrong bottleneck.

#define TID_X()  __nvvm_read_ptx_sreg_tid_x()
#define CTA_X()  __nvvm_read_ptx_sreg_ctaid_x()
#define NTID_X() __nvvm_read_ptx_sreg_ntid_x()
#define NCTA_X() __nvvm_read_ptx_sreg_nctaid_x()
#define SYNC()   __syncthreads()

#define GROUP            32
#define GROUPS_PER_SCALE 8
#define WARP             32
#define MAX_N            16384      // activation bytes we can stage

typedef unsigned int uint4v __attribute__((ext_vector_type(4)));

__attribute__((device)) static inline float h2f(unsigned short h) {
    return (float) __builtin_bit_cast(_Float16, h);
}

// Four packed code bytes against four int8 activations, low and high nibbles
// dotted against their OWN groups. Activations arrive as one 32-bit word each
// rather than four byte loads.
__attribute__((device)) static inline void dot4(unsigned int w,
                                                unsigned int al,
                                                unsigned int ah,
                                                int * dl, int * dh) {
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

extern "C" __attribute__((global)) void sml_matvec_flat4(
        float * __restrict__ y,
        const unsigned char  * __restrict__ codes,
        const unsigned short * __restrict__ sd,
        const unsigned char  * __restrict__ sq,
        const unsigned short * __restrict__ md,
        const unsigned char  * __restrict__ mq,
        const signed char    * __restrict__ xq,
        const float          * __restrict__ xscale,
        const int            * __restrict__ xsums,
        int rows, int n, int have_min) {

    __attribute__((shared)) unsigned int sxq[MAX_N / 4];

    const int ng     = n / GROUP;
    const int nscale = (ng + GROUPS_PER_SCALE - 1) / GROUPS_PER_SCALE;
    const int nblk   = n / 64;
    const int lane   = TID_X() & (WARP - 1);
    const int warp   = TID_X() / WARP;
    const int nwarp  = NTID_X() / WARP;
    const long long rowbytes = (long long) n / 2;

    // stage the activation once for the whole block
    const int nw = n / 4;
    for (int i = TID_X(); i < nw; i += NTID_X())
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
            // group g's activation starts at word g*8 of the staged copy
            const unsigned int * al = sxq + (long long) gl * (GROUP / 4);
            const unsigned int * ah = sxq + (long long) gh * (GROUP / 4);

            int dl = 0, dh = 0;
            dot4(w0.x, al[0], ah[0], &dl, &dh);
            dot4(w0.y, al[1], ah[1], &dl, &dh);
            dot4(w0.z, al[2], ah[2], &dl, &dh);
            dot4(w0.w, al[3], ah[3], &dl, &dh);
            dot4(w1.x, al[4], ah[4], &dl, &dh);
            dot4(w1.y, al[5], ah[5], &dl, &dh);
            dot4(w1.z, al[6], ah[6], &dl, &dh);
            dot4(w1.w, al[7], ah[7], &dl, &dh);

            const int sl = gl / GROUPS_PER_SCALE, sh = gh / GROUPS_PER_SCALE;
            acc += h2f(dp[sl]) * (float) qp[gl] * xscale[gl] * (float) dl;
            acc += h2f(dp[sh]) * (float) qp[gh] * xscale[gh] * (float) dh;
            if (have_min) {
                acc += h2f(mdp[sl]) * ((float) mqp[gl] - 127.0f)
                     * xscale[gl] * (float) xsums[gl];
                acc += h2f(mdp[sh]) * ((float) mqp[gh] - 127.0f)
                     * xscale[gh] * (float) xsums[gh];
            }
        }

        for (int off = WARP / 2; off > 0; off >>= 1)
            acc += __nvvm_shfl_sync_down_f32(0xffffffffu, acc, (unsigned) off, 31);
        if (lane == 0) y[r] = acc;
    }
}
