// sml-export: the engine's view of a Sheep file, for a forward pass written
// elsewhere (the DirectML worker that runs Sheep models on AMD and Intel GPUs).
//
// It hands out what the CPU engine itself computes with: each weight
// reconstructed from its codes by the engine's own dequantizers and put back in
// the original column order, the norm and bias vectors, the rotary tables the
// engine built (frequency scaling included), and the CPU engine's logits for
// grading the other device against. Nothing is re-derived from the file by a
// second reader, so the other device runs the same model, not a look-alike.
//
// Built as a DLL (make sml-export.dll). One model per process.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sml-model.h"
#include "sml-run.h"
#include "sml-kernel.h"

#ifdef _WIN32
#define SX __declspec(dllexport)
#else
#define SX __attribute__((visibility("default")))
#endif

static sml_model M;
static sml_ctx C;
static int g_open = 0;

// What the forward pass elsewhere supports: the dense pre-norm families (Llama,
// Mistral, Qwen2 with its QKV bias, Qwen3 with its Q/K norm). Anything else is
// refused by name, as the GPU path refuses, and the server stays on the CPU.
static const char * unsupported(void) {
    const sml_hdr * h = M.hdr;
    if (h->flags & SML_MODEL_HADAMARD) return "a Hadamard-rotated file";
    if (h->version >= 4 && h->norm_type == 1) return "LayerNorm";
    if (h->version >= 5 && h->arch_bits) return "an architecture flag this device has no maths for";
    if (h->version >= 4 && (h->logit_softcap != 0.0f || h->n_swa)) return "softcapped or windowed attention";
    if (h->attn_softcap != 0.0f || h->clip_qkv != 0.0f) return "attention softcap or Q/K/V clamp";
    if (C.pos_embd || C.alibi || C.no_rope) return "positions other than rotary";
    if (C.emb_norm || C.out_b || C.layer_win || C.layer_rope || C.rope_cos_l) return "an embedding norm, head bias or per-layer attention";
    if ((int) C.emb->cols != (int) h->n_embd) return "padded embedding rows";
    for (uint32_t l = 0; l < h->n_layer; l++) {
        const sml_layer * s = &C.layers[l];
        if (s->n_exp || s->shg) return "experts";
        if (s->ssm_in || s->m1_in || s->gdn_qkvz || s->sc_in) return "state-space or convolution layers";
        if (s->attn_gate || s->xielu) return "an attention gate or the xIELU activation";
        if (!s->q || !s->k || !s->v || !s->o || !s->gate || !s->up || !s->down) return "a layer this device cannot run";
        if (!s->attn_norm || !s->ffn_norm || s->attn_post_norm || s->ffn_post_norm) return "post-norms or missing input norms";
        if (s->o_bias || s->up_bias || s->down_bias || s->gate_bias || s->attn_norm_b) return "output or FFN biases";
        if ((s->q_norm && s->q_norm_b) || (s->q_norm && s->q_norm_n != C.head_dim)) return "a Q/K norm of another kind";
    }
    return NULL;
}

SX int sx_open(const char * path, int n_ctx, char * why, int nwhy) {
    if (g_open) return 1;
    if (!sml_model_open(&M, path)) { snprintf(why, (size_t) nwhy, "cannot open the model file"); return 0; }
    sml_set_repack(0);       // the CPU side here only grades; no decode copy
    if (!sml_ctx_init(&C, &M, n_ctx, 6)) { snprintf(why, (size_t) nwhy, "out of memory sizing the context"); return 0; }
    const char * u = unsupported();
    if (u) { snprintf(why, (size_t) nwhy, "%s: not on this device yet", u); return 0; }
    g_open = 1;
    return 1;
}

// iv: n_layer, n_embd, n_head, n_kv, head_dim, n_ff, n_vocab, n_ctx, interleaved,
//     rope_half, tied, qkv_bias, qk_norm, depth; fv: eps, attn scale, emb_scale,
//     res_scale, logit_scale
SX void sx_facts(int * iv, float * fv) {
    const sml_hdr * h = M.hdr;
    const sml_layer * s = &C.layers[0];
    iv[0] = (int) h->n_layer; iv[1] = (int) h->n_embd; iv[2] = (int) h->n_head; iv[3] = (int) h->n_head_kv;
    iv[4] = C.head_dim; iv[5] = (int) h->n_ff; iv[6] = (int) h->n_vocab; iv[7] = C.n_ctx;
    iv[8] = h->version >= 4 && h->rope_type == 1; iv[9] = C.rope_half; iv[10] = C.head == C.emb;
    iv[11] = s->q_bias != NULL; iv[12] = s->q_norm != NULL; iv[13] = C.depth;
    fv[0] = h->rms_eps;
    fv[1] = (h->version >= 4 && h->attn_scale != 0.0f) ? h->attn_scale : 1.0f / sqrtf((float) C.head_dim);
    fv[2] = h->emb_scale; fv[3] = h->res_scale; fv[4] = h->logit_scale;
}

static const sml_rec * rec_of(int layer, int kind) {
    if (kind == 7) return C.head;
    if (kind == 8) return C.emb;
    if (layer < 0 || layer >= (int) M.hdr->n_layer) return NULL;
    const sml_layer * s = &C.layers[layer];
    const sml_rec * t[7] = { s->q, s->k, s->v, s->o, s->gate, s->up, s->down };
    return kind >= 0 && kind < 7 ? t[kind] : NULL;
}

// kind: 0 q, 1 k, 2 v, 3 o, 4 gate, 5 up, 6 down, 7 the output head, 8 the embedding
SX int sx_shape(int layer, int kind, int * rows, int * cols) {
    const sml_rec * r = rec_of(layer, kind);
    if (!r) return 0;
    *rows = (int) r->rows; *cols = (int) r->cols;
    return 1;
}

// The weight as the engine computes with it: rows x cols floats, original column order.
SX int sx_tensor(int layer, int kind, float * out) {
    const sml_rec * r = rec_of(layer, kind);
    if (!r) return 0;
    const int rows = (int) r->rows, n = (int) r->cols;
    if (r->kind == SML_KIND_F32) {
        memcpy(out, sml_ptr(&M, r->off), (size_t) rows * n * sizeof(float));
        return 1;
    }
    const int depth = (kind == 8) ? (int) M.hdr->n_levels : C.depth;
    const int32_t * perm = (const int32_t *) sml_ptr(&M, r->perm_off);
    const uint16_t * sd = (const uint16_t *) sml_ptr(&M, r->sd_off);
    const uint8_t  * sq = (const uint8_t  *) sml_ptr(&M, r->sq_off);
    const uint16_t * md = r->mq_n ? (const uint16_t *) sml_ptr(&M, r->md_off) : NULL;
    const uint8_t  * mq = r->mq_n ? (const uint8_t  *) sml_ptr(&M, r->mq_off) : NULL;
#pragma omp parallel
    {
        float * tmp = malloc((size_t) n * sizeof(float));
#pragma omp for schedule(static)
        for (int row = 0; row < rows; row++) {
            if (r->layout == SML_LAYOUT_FLAT && (r->nbits == 3 || r->nbits == 5 || r->nbits == 7)) {
                // the planes of Q3/Q5/Q7: the base code, then the 1-bit plane on top
                const uint8_t * p0 = sml_ptr(&M, r->codes_off[0]);
                if (r->nbits == 3) { const uint8_t * pl[1] = { p0 }; sml_dequant_row(tmp, pl, 1, 1, sd, sq, md, mq, row, n); }
                else if (r->nbits == 5) sml_dequant_row_flat4(tmp, p0, sd, sq, md, mq, row, n);
                else sml_dequant_row_flat6(tmp, p0, sml_ptr(&M, r->codes_off[1]), sd, sq, md, mq, row, n);
                sml_dequant_bits1_add(tmp, sml_ptr(&M, r->codes_off[r->nbits == 7 ? 2 : 1]), sd, sq, row, n,
                                      r->nbits == 3 ? 2 : r->nbits == 5 ? 4 : 6);
            } else if (r->layout == SML_LAYOUT_FLAT && r->nbits == 2) {
                const uint8_t * p2[1] = { sml_ptr(&M, r->codes_off[0]) };
                sml_dequant_row(tmp, p2, 1, 1, sd, sq, md, mq, row, n);
            } else if (r->layout == SML_LAYOUT_FLAT && r->nbits == 8)
                sml_dequant_row_flat8(tmp, sml_ptr(&M, r->codes_off[0]), sml_ptr(&M, r->codes_off[1]), sd, sq, md, mq, row, n);
            else if (r->layout == SML_LAYOUT_FLAT && r->nbits == 6)
                sml_dequant_row_flat6(tmp, sml_ptr(&M, r->codes_off[1]), sml_ptr(&M, r->codes_off[0]), sd, sq, md, mq, row, n);
            else if (r->layout == SML_LAYOUT_FLAT)
                sml_dequant_row_flat4(tmp, sml_ptr(&M, r->codes_off[0]), sd, sq, md, mq, row, n);
            else {
                const uint8_t * planes[SML_MAX_LEVELS];
                for (uint32_t lv = 0; lv < M.hdr->n_levels; lv++) planes[lv] = sml_ptr(&M, r->codes_off[lv]);
                sml_dequant_row(tmp, planes, (int) M.hdr->n_levels, depth, sd, sq, md, mq, row, n);
            }
            float * dst = out + (size_t) row * n;
            for (int i = 0; i < n; i++) dst[perm[i]] = tmp[i];
        }
        free(tmp);
    }
    return 1;
}

// kind: 0 attn_norm, 1 ffn_norm, 2 q_bias, 3 k_bias, 4 v_bias, 5 q_norm, 6 k_norm, 7 output_norm.
// Returns the length (0 when the model has none); out may be NULL to ask the length.
SX int sx_vec(int layer, int kind, float * out) {
    const float * v = NULL;
    int n = 0;
    const int kvd = (int) M.hdr->n_head_kv * C.head_dim, qd = (int) M.hdr->n_head * C.head_dim;
    if (kind == 7) { v = (const float *) sml_ptr(&M, C.out_norm->off); n = (int) M.hdr->n_embd; }
    else if (layer >= 0 && layer < (int) M.hdr->n_layer) {
        const sml_layer * s = &C.layers[layer];
        switch (kind) {
            case 0: v = (const float *) sml_ptr(&M, s->attn_norm->off); n = (int) M.hdr->n_embd; break;
            case 1: v = (const float *) sml_ptr(&M, s->ffn_norm->off); n = (int) M.hdr->n_embd; break;
            case 2: v = s->q_bias; n = qd; break;
            case 3: v = s->k_bias; n = kvd; break;
            case 4: v = s->v_bias; n = kvd; break;
            case 5: v = s->q_norm; n = s->q_norm_n; break;
            case 6: v = s->k_norm; n = s->k_norm_n; break;
        }
    }
    if (!v) return 0;
    if (out) memcpy(out, v, (size_t) n * sizeof(float));
    return n;
}

// The rotary tables the engine built: n_ctx x rope_half each.
SX void sx_rope(float * cos_t, float * sin_t) {
    const size_t n = (size_t) C.n_ctx * C.rope_half;
    memcpy(cos_t, C.rope_cos, n * sizeof(float));
    memcpy(sin_t, C.rope_sin, n * sizeof(float));
}

// The CPU engine's next-token logits after ids[0..n), from a fresh cache: the grade.
SX int sx_cpu_logits(const int * ids, int n, float * out) {
    if (n <= 0 || n > C.n_ctx) return 0;
    C.n_kv = 0; C.n_hist = 0;
    const float * l = sml_prefill(&C, (const int32_t *) ids, 0, n, C.depth);
    if (!l) return 0;
    memcpy(out, l, (size_t) M.hdr->n_vocab * sizeof(float));
    return 1;
}
