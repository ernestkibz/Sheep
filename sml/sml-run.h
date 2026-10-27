// SML inference context: the smallest engine that produces correct logits.
//
// Deliberately narrow. It runs one architecture (Qwen2-family decoder,
// GQA, SwiGLU, RoPE) and nothing else. Breadth is what made the thing being
// replaced enormous; the point here is to be right about bytes on one model
// before being general about many.
//
// The read depth is a per-token decision, not a build-time constant. That is
// the whole architecture: an easy token is answered from the coarse level, and
// only a close decision pays for refinement.

#pragma once

#include "sml-model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verification scores several drafted tokens against one read of the weights;
// this caps how many.
// Slots the batched body carries per weight read. The batch kernels hold one
// accumulator per slot (SML_KERNEL_MAX_BATCH). Prompt reading uses
// SML_PREFILL_NB_DEFAULT and SML_PREFILL_NB selects anything up to the maximum.
// 128 since 2026-09-25: the 1B read a 128-token prompt at 180-208 tok/s in two
// batches of 64 and 218-221 in one of 128, logits bit-identical; pythia, which
// is bound by arithmetic rather than by weight reads, was unchanged.
#define SML_MAX_BATCH 128
#define SML_PREFILL_NB_DEFAULT 128

// Per-layer tensor pointers, resolved once at init. sml_find is a linear scan
// over the whole table and was being called 8 times per layer per token.
typedef struct {
    const sml_rec * attn_norm, * ffn_norm;
    const sml_rec * q, * k, * v, * o;
    const sml_rec * gate, * up, * down;
    const float * q_bias, * k_bias, * v_bias;
    // GPT-NeoX and Falcon bias the output and FFN projections too. Carrying
    // only the QKV biases drops them silently: the model still speaks, just
    // not as itself.
    const float * o_bias, * up_bias, * down_bias, * gate_bias;
    const float * attn_norm_b, * ffn_norm_b;   // LayerNorm only
    // Post-norms (OLMo 2; also Gemma 2/3): a norm on each branch's OUTPUT before
    // the residual add. OLMo 2 has these and NO input norms, so attn_norm and
    // ffn_norm may be NULL. Read from the tensor table.
    const float * attn_post_norm, * ffn_post_norm;
    const float * q_norm, * k_norm;            // Q/K norm weights, NULL if none
    const float * q_norm_b, * k_norm_b;        // their biases: a LayerNorm (Persimmon), else NULL
    // Qwen2-MoE's shared expert, run for every token beside the routed ones and
    // scaled by sigmoid(shgate . x); NULL when the model has none
    const sml_rec * shg, * shu, * shd;
    const float * shgate;
    const float * exp_bias;      // added to the scores to CHOOSE experts only (V3, ERNIE)
    // LFM2's short convolution in place of attention: in (-> B, C, x), a causal
    // depthwise convolution over the last sc_L positions of B*x, times C, out
    const sml_rec * sc_in, * sc_out;
    const float * sc_w, * sc_cb, * sc_in_b, * sc_out_b;
    int sc_L;
    // Mamba-2 mixer in place of attention: in -> [z | xBC | dt]; xBC through a
    // causal depthwise conv and SiLU -> x, B, C; per head a state updated by
    // exp(dt*A) and dt*B*x, read with C, plus D*x; gated RMSNorm; out
    const sml_rec * ssm_in, * ssm_out;
    const float * ssm_conv, * ssm_conv_b, * ssm_dt_b, * ssm_a_log, * ssm_d, * ssm_nw;
    int ssm_K, ssm_heads, ssm_inner, ssm_convdim;
    // Mamba-1 mixer: in -> [x | z]; x through a causal depthwise conv and SiLU;
    // x_proj -> [dt (rank R) | B | C]; dt_proj -> dt per channel; a [inner x
    // d_state] state; y * silu(z); out. Jamba norms dt, B, C with weights.
    // Gated DeltaNet (Qwen3-Next): [q|k|v|z] and [b|a] projections (the file
    // holds them flat, not per key-head group), a causal conv + SiLU over
    // q|k|v, L2-normed q and k, per value head a [dk x dv] state decayed by
    // exp(g) and corrected by the delta rule, RMSNorm * silu(z) per head, out
    const sml_rec * gdn_qkvz, * gdn_ba, * gdn_out;
    const float * gdn_conv, * gdn_dt_b, * gdn_a_log, * gdn_nw;
    int gdn_K, gdn_nv, gdn_dv;
    // attention output gate (Qwen3-Next): attention out * sigmoid(gate . x)
    const sml_rec * attn_gate;
    const float * xielu;         // Apertus: xIELU's alpha_p, alpha_n, beta, eps; NULL = the file's activation
    const sml_rec * m1_in, * m1_x, * m1_dt, * m1_out;
    const float * m1_conv, * m1_conv_b, * m1_dt_b, * m1_a_log, * m1_d, * m1_ndt, * m1_nb, * m1_nc;
    int m1_K, m1_inner, m1_N, m1_R;
    int q_norm_n, k_norm_n;                    // their lengths: which kind of norm
    int qk_heads;                              // SML_ARCH_QKNORM_HEADS: one weight row per head
    // Mixture of experts: the router (f32, n_exp x n_embd) and each expert's
    // gate, up and down. n_exp 0 means a dense FFN.
    const float * router;
    const float * router_b;                    // the router's bias (gpt-oss): chooses and weights
    const float ** egb, ** eub, ** edb;        // each expert's gate/up/down biases (gpt-oss), or NULL
    const float * sinks;                       // per head: a logit in every softmax's denominator
    // Multi-head latent attention (DeepSeek V2/V3): Q through a low-rank pair or one
    // projection, K/V from a normed latent expanded per head; each head is
    // [rotary part | the rest] (the builder puts the rotary dims first) and V is
    // mla_vd wide, padded to head_dim in the cache and packed again before o.
    const sml_rec * mla_qa, * mla_qb, * mla_kva, * mla_kvb;
    const float * mla_qa_norm, * mla_kva_norm;
    int mla_rq, mla_rkv, mla_vd;
    int n_exp;
    const sml_rec ** eg, ** eu, ** ed;

    // Shared-activation groups, resolved once at init. Taken from the file when
    // it declares them, otherwise inferred from the tensors this engine knows.
    // The engine can work out Llama's grouping because it was taught Llama; a
    // file that states its own grouping does not need the engine to have met
    // that architecture before.
    const sml_rec * agrp[SML_GROUP_MAX]; int n_agrp;   // attention projections
    const sml_rec * fgrp[SML_GROUP_MAX]; int n_fgrp;   // FFN gate/up
} sml_layer;

typedef struct {
    const sml_model * model;
    sml_layer * layers;
    const sml_rec * emb, * out_norm;
    const sml_rec * head;   // output.weight if untied, else emb
    const float * out_norm_b;
    const float * out_b;    // output-head bias (Phi, GPT-J), NULL if none

    // How each layer's schedule was decided. Reported rather than assumed: a
    // file can declare groups and the engine can still be quietly falling back,
    // and the difference is the whole claim.
    int grp_from_file, grp_inferred;

    int n_ctx;
    // Width of one attention head, from the Q projection's real shape. Not
    // n_embd / n_head: Qwen3 runs 16 heads of 128 over a 1024-wide stream.
    int head_dim;
    // Models whose features live only on the step path (experts, Granite's
    // scales) read prompts a token at a time rather than read them wrongly.
    int step_only;
    int batch_decode;   // sml_forward_multi: 1 packed rows-in-lanes, 2 decode kernels per slot
    const uint8_t ** rlp;   // the packed copy per tensor record (shared), NULL until used
    // Learned absolute positions (GPT-2 lineage), NULL for rotary models;
    // pos_rows is how many positions the table holds.
    const float * pos_embd;
    int pos_rows;
    // ALiBi slope per head (BLOOM), NULL otherwise; no_rope when either
    // learned positions or ALiBi carries position instead of rotary.
    float * alibi;
    int no_rope;
    // LayerNorm applied to the embedding row (BLOOM), NULL if none
    const float * emb_norm, * emb_norm_b;
    // When set, sml_forward reads this row instead of the token's embedding:
    // how a vision tower's output enters the text model at an image token.
    const float * emb_in;
    float * moe_logit, * moe_tmp, * moe_acc;   // MoE scratch, NULL if dense
    float * moe_g, * moe_u;                    // expert gate/up scratch, the widest expert long
    float * moe_s, * moe_c;                    // per-expert scores and choice scores
    // short convolutions: B*x per position per conv layer (like a KV cache, so
    // a conversation can be rewound), and scratch
    float ** sc_hist;
    float * sc_buf, * sc_y;
    // Mamba-2: per layer the [heads][head_dim][dstate] state and the last K-1
    // conv inputs. A state cannot be rewound: positions must run in order.
    float ** ssm_state, ** ssm_cst;
    float * ssm_proj, * ssm_xbc, * ssm_y;
    float * agate;      // the attention output gate's values, n_head * head_dim
    float * mla_a, * mla_an, * mla_kv;
    float * hyb;        // Falcon-H1: the Mamba half of a layer that runs Mamba and attention side by side   // MLA scratch: a latent, the normed latent, its expansion
    int ssm_next;
    int moe_ff, moe_ne;                        // their length; experts the logit row holds
    // Batched experts (prompt reading): the batch's rows gathered per expert,
    // that expert's gate/up/down outputs, every (slot, choice) output kept so
    // each slot sums its experts in the stepped path's order. Sized on first use.
    float * mb_x, * mb_g, * mb_u, * mb_o, * mb_out, * mb_w, * mb_sg;
    int   * mb_sel, * mb_row;
    int     mb_k;                               // choices per slot the buffers hold
    // Rotated dims per head / 2. The RoPE table's stride AND its frequency
    // base both derive from this, and so does the loop that reads it -- when
    // the table was built from head_dim/2 while the reader used n_rot/2 the
    // engine read the wrong entries at the wrong frequencies, on a partial-
    // rotary model only. One value, used everywhere, cannot drift.
    int rope_half;
    int n_threads;
    int depth;          // ladder levels to read, 1..n_levels

    // KV cache, laid out [layer][pos][kv_head * head_dim].
    // fp16, not fp32. The cache is read in full for every token, so at long
    // context it is bytes moved -- the one cost this whole architecture exists
    // to cut -- and fp32 was costing double for precision the attention scores
    // cannot use. The reference engine defaults to fp16 here; storing fp32 was
    // us being worse than it for no reason.
    uint16_t * k_cache;
    uint16_t * v_cache;
    int     n_kv;       // positions currently held

    // The tokens the cache currently holds, so a request sharing a prefix with
    // the last one can keep those positions instead of recomputing them. In a
    // chat every turn resends the whole conversation, so without this the
    // prefill cost grows with the transcript and is paid again every turn.
    // Keys are RoPE'd by ABSOLUTE position, so a shared prefix is reusable
    // exactly when the tokens match at the same positions -- which is what a
    // prefix is.
    int   * hist;
    int     n_hist;

    // scratch, all sized once at init
    float * x;          // residual stream          n_embd
    float * xb;         // normalized / temp        n_embd
    float * xg;         // permuted activation      max(n_embd, n_ff)
    float * xr;         // Hadamard-rotated input   max(n_embd, n_ff)
    float * q;          // n_head * head_dim
    float * k;          // n_head_kv * head_dim
    float * v;          // n_head_kv * head_dim
    float * att;        // n_head * n_ctx
    float * hb;         // n_ff
    float * hb2;        // n_ff
    float * logits;     // n_vocab

    float * rope_cos;   // n_ctx * head_dim/2, precomputed at init
    // Per-layer attention, from the file's attn_window / attn_rope /
    // rope_freqs_local tensors (Gemma 2/3, Cohere2, SmolLM3, GPT-Neo, EXAONE 4).
    // NULL = every layer the same (the header's n_swa, rotary on).
    int * layer_win;             // sliding window per layer, 0 = full attention
    unsigned char * layer_rope;  // 1 = rotate this layer's Q/K, 0 = no position (NoPE)
    float * rope_cos_l, * rope_sin_l;   // the local layers' own rotary base (Gemma 3)
    float * rope_sin;

    sml_act act;        // quantized activation, reused every matvec

    // Batch scratch, sized once for SML_MAX_BATCH. Verification scores several
    // known tokens against ONE read of the weights, so every per-token buffer
    // needs a copy per batch slot.
    float * bx;         // nb * n_embd   residual
    float * bh;         // nb * n_embd   normed
    float * bq;         // nb * n_head*head_dim
    float * bk;         // nb * kv_dim
    float * bv;         // nb * kv_dim
    float * bff;        // nb * n_ff
    float * bff2;       // nb * n_ff
    float * bout;       // nb * max(n_embd, n_ff)  matvec destination
    float * bgate;      // nb * n_head*head_dim    attention output gate (Qwen3-Next)
    // A split model (the GPU ran layers 0..layer0-1): the residual after them is
    // handed in, in place of the embedding lookup, and the CPU runs the rest.
    int layer0;
    const float * x_in;  // one row (sml_forward_from) or nb rows (sml_prefill_from)
    sml_act bact[SML_MAX_BATCH];

    // One scratch row per OpenMP thread: n_tscr rows of tscr_len floats.
    float * tscr;
    int     n_tscr;
    int     tscr_len;

    // Decode's private copy of the flat tensors in the interleaved layout
    // (sml-kernel.c), one pointer per tensor record, NULL where the tensor
    // keeps the file's arrays. SML_REPACK=0 turns it off.
    const uint8_t ** il;
    uint8_t        * il_mem;
    size_t           il_bytes;

    // Traffic counter. We optimise bytes, so the engine reports bytes rather
    // than letting a tok/s number stand in for them.
    uint64_t bytes_moved;
    uint64_t tokens;
} sml_ctx;

int  sml_ctx_init(sml_ctx * c, const sml_model * m, int n_ctx, int n_threads);
void sml_ctx_free(sml_ctx * c);

// Decode's private interleaved copy (about the size of the file) for contexts
// made after this call: on (the default) or off. A context that only stands by,
// as the CPU does behind the GPU, skips it; sml_ctx_repack builds it if that
// context is ever called on. SML_REPACK=0 still turns it off everywhere.
void sml_set_repack(int on);
void sml_ctx_repack(sml_ctx * c);

// Score `nb` tokens at consecutive positions in ONE pass over the weights.
//
// This is what makes speculative decoding pay here. A batch-1 step reads the
// whole model to produce one token; verifying a drafted run reads the same
// bytes and scores all of them. Measured: k=2 costs 1.43x a single step rather
// than 2x, k=4 costs 2.17x rather than 4x.
//
// Every token must already be known -- which is true during verification, and
// is why verification batches while generation cannot.
//
// logits_out receives nb * n_vocab floats: row i is the distribution predicted
// AFTER token i, so row i is what decides whether token i+1 was drafted right.
// NULL skips the output head and leaves each slot's residual in c->bx.
void sml_forward_batch(sml_ctx * c, const int32_t * tokens, int pos0, int nb,
                       int depth, float * logits_out);

// Read a prompt: forward n known tokens at positions pos0.. and return the logits
// after the last one (owned by the context). Same numbers as one sml_forward per
// token, with the weights read once per SML_MAX_BATCH tokens and the head once.
const float * sml_prefill(sml_ctx * c, const int32_t * ids, int pos0, int n, int depth);
// Layers L0.. only, from residual rows the caller provides (the GPU ran the
// layers before L0 and holds their cache): the CPU half of a split model.
// Returns the next-token logits after the last row, as sml_forward/prefill do.
const float * sml_forward_from(sml_ctx * c, const float * x, int pos, int L0, int depth);
const float * sml_prefill_from(sml_ctx * c, const float * xs, int pos0, int n, int L0, int depth);

// Several conversations, one token each, one read of the weights: cs[b] holds
// slot b's KV cache and pos[b] its position; logits_out is ns x n_vocab. All
// contexts from one model, ns <= SML_MAX_BATCH.
void sml_forward_multi(sml_ctx * const * cs, const int32_t * tokens, const int * pos,
                       int ns, int depth, float * logits_out);

// Run one token at position `pos`. Returns the logits (owned by the context).
// `depth` levels of the ladder are read; pass 0 to use c->depth.
const float * sml_forward(sml_ctx * c, int token, int pos, int depth);

int sml_sample_greedy(const float * logits, int n);

#define SML_MOE_MAX_K 16     // experts a token may use
// Expert choice from router logits (shared by the CPU and the GPU); returns k.
int sml_moe_route(const sml_hdr * h, const sml_layer * ly, const float * lg,
                  float * sc, float * ch, int * sel, float * w);

// How many leading positions of `ids` the cache already holds. Always leaves at
// least one token to forward, since the caller needs logits back.
int  sml_cache_prefix(sml_ctx * c, const int * ids, int n);
// Record what the cache now holds, generated tokens included.
void sml_cache_commit(sml_ctx * c, const int * ids, int n);

// Margin between the top two logits, in units of the top logit's scale. This is
// the signal the depth gate will use: a wide margin means the coarse level
// already picked the same token the refined one would have, so the refinement
// bytes are not worth moving.
float sml_top2_margin(const float * logits, int n);

#ifdef __cplusplus
}
#endif
