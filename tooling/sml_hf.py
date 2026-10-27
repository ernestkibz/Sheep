#!/usr/bin/env python3
"""Read a Hugging Face checkpoint directly, with no third-party dependency.

This is the intake that matters. Converting from an existing Q4_K_M is DOUBLE
quantization: the ladder converges toward the source's rounded grid rather than
the true weights, so the source becomes a hard quality ceiling and the best SSF
could ever do is tie. From fp16/bf16 it can win.

The safetensors container is parsed here rather than imported:

    u64  header_len
    utf8 header_len bytes of JSON   {name: {dtype, shape, data_offsets}}
    u8   data[]                     offsets relative to the end of the header

Two reasons. It removes a dependency from the only path the project still needs
Python for, and the numpy binding cannot read bf16 at all -- which is the dtype
most current checkpoints ship in. bf16 is just the top 16 bits of fp32, so the
conversion is a shift.

Tensor names map through an explicit table so an unexpected name fails loudly
rather than being silently dropped.
"""
import json
import math
import os
import re
import struct

import numpy as np

_DTYPE = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16,
          "I32": np.int32, "I64": np.int64, "U8": np.uint8, "BOOL": np.bool_}

# HF checkpoints call the same thirteen tensors about 290 different things.
# This table is that knowledge, taken from studying how the reference converter
# handles it -- the knowledge, not the code: theirs is a class per architecture
# (one subclass, sometimes several overrides, for every model family they
# support), ours is one table matched by regex, so a new architecture whose
# names are already in here needs no code at all. Pythia needed only the
# entries; nothing was written for GPT-NeoX specifically.
#
# "{bid}" is the layer index. Names are matched WITHOUT their trailing
# .weight/.bias, so a bias is mapped by the same entry as its weight and the
# two can never disagree.
_NAMES = {
    "token_embd": (
        "backbone.embedding",
        "backbone.embeddings",
        "embed_tokens",
        "embedding.word_embeddings",
        "embeddings.tok_embeddings",
        "embeddings.word_embeddings",
        "encoder",
        "gpt_neox.embed_in",
        "language_model.embedding.word_embeddings",
        "language_model.model.embed_tokens",
        "model.embed_tokens",
        "model.embedding",
        "model.embeddings",
        "model.tok_embeddings",
        "model.transformer.wte",
        "model.word_embeddings",
        "rwkv.embeddings",
        "shared",
        "tok_embeddings",
        "transformer.embd.wte",
        "transformer.in_out_embed",
        "transformer.token_embeddings",
        "transformer.word_embeddings",
        "transformer.wte",
        "word_embeddings",
        "wte",
    ),
    # Learned absolute positions (GPT-2 lineage): one row per position, added
    # to the token embedding. These models have no rotary anywhere.
    # A LayerNorm straight after the embedding lookup (BLOOM).
    "token_embd_norm": (
        "word_embeddings_layernorm",
        "transformer.word_embeddings_layernorm",
    ),
    "position_embd": (
        "transformer.wpe",
        "wpe",
    ),
    "output_norm": (
        "model.embedding_norm",              # LFM2: applied AFTER the layers, despite its name
        "backbone.final_layer_norm",
        "backbone.norm_f",
        "encoder.final_layernorm",
        "final_norm",
        "gpt_neox.final_layer_norm",
        "language_model.encoder.final_layernorm",
        "lm_head.ln",
        "ln_f",
        "model.final_layernorm",
        "model.ln_out",
        "model.norm",
        "model.norm_f",
        "model.transformer.ln_f",
        "norm",
        "rwkv.ln_out",
        "transformer.ln_f",
        "transformer.norm",
        "transformer.norm_f",
        "transformer.rms_norm",
    ),
    "output": (
        "embed_out",
        "head",
        "head.decoder",
        "head.out",
        "lm_head",
        "lm_head.linear",
        "model.transformer.ff_out",
        "output",
        "output_layer",
        "word_embeddings_for_head",
    ),
    "attn_norm": (
        "transformer.h.{bid}.ln_attn",       # Falcon's new decoder: its own pre-attention norm
        "model.layers.{bid}.operator_norm",  # LFM2: before attention OR the convolution
        "backbone.layers.{bid}.norm",
        "encoder.layers.{bid}.input_layernorm",
        "gpt_neox.layers.{bid}.input_layernorm",
        "h.{bid}.input_layernorm",
        "h.{bid}.ln_1",
        "language_model.encoder.layers.{bid}.input_layernorm",
        "layers.{bid}.attention_norm",
        "layers.{bid}.attn_norm",
        "layers.{bid}.input_layernorm",
        "model.layers.layers.{bid}.norm",
        "model.layers.layers.{bid}.pre_mixer_norm",
        "model.layers.{bid}.attention_layernorm",
        "model.layers.{bid}.attention_norm",
        "model.layers.{bid}.input_layernorm",
        "model.layers.{bid}.ln1",
        "model.layers.{bid}.norm",
        "model.layers.{bid}.operator_norm",
        "model.layers.{bid}.pre_attention_layernorm",
        "model.layers.{bid}.pre_attn_norm",
        "model.transformer.blocks.{bid}.attn_norm",
        "rwkv.blocks.{bid}.ln1",
        "transformer.blocks.{bid}.norm_1",
        "transformer.blocks.{bid}.norm_attn_norm.norm_1",
        "transformer.decoder_layer.{bid}.rms_norm",
        "transformer.h.{bid}.input_layernorm",
        "transformer.h.{bid}.ln",
        "transformer.h.{bid}.ln_1",
        "transformer.h.{bid}.ln_mlp",
        "transformer.layers.{bid}.attn_norm",
        "transformer_encoder.{bid}.attention_norm",
    ),
    # Q/K norm weights. Until these entries existed the builder dropped the
    # tensors without a word, which is why the architecture table refused
    # every model that has them.
    # Mixture of experts. The router scores experts per token. Granite stacks
    # every expert into one [n_expert, rows, cols] tensor; those are split per
    # expert below, like the fused Q/K/V, so the pipeline sees 2-D weights.
    "ffn_gate_inp": (
        "model.layers.{bid}.feed_forward.router",         # Jamba
        "model.layers.{bid}.block_sparse_moe.router.layer",
        "model.layers.{bid}.block_sparse_moe.gate",       # Mixtral
        "model.layers.{bid}.mlp.gate",                    # Qwen2-MoE, OLMoE, Qwen3-MoE
        "model.layers.{bid}.mlp.gate.wg",                 # HunYuan-MoE
        "model.layers.{bid}.mlp.router.gate",             # AFMoE
        "model.layers.{bid}.mlp.router",                  # gpt-oss (with a bias)
        "backbone.layers.{bid}.mixer.gate",               # Nemotron-H
        "model.layers.{bid}.feed_forward.gate",           # LFM2-MoE
    ),
    # Qwen2-MoE's shared expert: every token, scaled by a sigmoid of its own gate
    "ffn_gate_shexp": ("model.layers.{bid}.mlp.shared_expert.gate_proj",
                       "model.layers.{bid}.mlp.shared_experts.gate_proj",   # V3 style, ERNIE
                       "model.layers.{bid}.mlp.shared_mlp.gate_proj"),      # HunYuan-MoE
    "ffn_up_shexp": ("model.layers.{bid}.mlp.shared_expert.up_proj",
                     "backbone.layers.{bid}.mixer.shared_experts.up_proj",
                     "model.layers.{bid}.mlp.shared_experts.up_proj",
                     "model.layers.{bid}.mlp.shared_mlp.up_proj"),
    "ffn_down_shexp": ("model.layers.{bid}.mlp.shared_expert.down_proj",
                       "model.layers.{bid}.mlp.shared_experts.down_proj",
                       "model.layers.{bid}.mlp.shared_mlp.down_proj",
                       "backbone.layers.{bid}.mixer.shared_experts.down_proj",
                       "model.layers.{bid}.shared_mlp.output_linear"),     # Granite-MoE-shared
    "ffn_gate_up_shexp": ("model.layers.{bid}.shared_mlp.input_linear",),  # its gate|up, fused
    # GLM-4-0414's sandwich norms on each branch's output
    "attn_post_norm": ("model.layers.{bid}.post_self_attn_layernorm",),
    "ffn_gate_inp_shexp": ("model.layers.{bid}.mlp.shared_expert_gate",),
    # Mamba-2 mixers: backbone.layers.N.mixer.* (Mamba2) or model.layers.N.mamba.*
    # (Bamba, Granite-4-H); dt_bias / A_log / D are bare parameters
    "ssm_in": ("backbone.layers.{bid}.mixer.in_proj", "model.layers.{bid}.mamba.in_proj"),
    "ssm_conv": ("backbone.layers.{bid}.mixer.conv1d", "model.layers.{bid}.mamba.conv1d"),
    "ssm_dt_b": ("backbone.layers.{bid}.mixer.dt_bias", "model.layers.{bid}.mamba.dt_bias"),
    "ssm_a_log": ("backbone.layers.{bid}.mixer.A_log", "model.layers.{bid}.mamba.A_log"),
    "ssm_d": ("backbone.layers.{bid}.mixer.D", "model.layers.{bid}.mamba.D"),
    "ssm_norm": ("backbone.layers.{bid}.mixer.norm", "model.layers.{bid}.mamba.norm"),
    "ssm_out": ("backbone.layers.{bid}.mixer.out_proj", "model.layers.{bid}.mamba.out_proj"),
    # Nemotron-H names every block's module "mixer": attention, MLP and Mamba-2
    "attn_q": ("backbone.layers.{bid}.mixer.q_proj",),
    "attn_k": ("backbone.layers.{bid}.mixer.k_proj",),
    "attn_v": ("backbone.layers.{bid}.mixer.v_proj",),
    "attn_output": ("backbone.layers.{bid}.mixer.o_proj",),
    "ffn_up": ("backbone.layers.{bid}.mixer.up_proj",),
    "ffn_down": ("backbone.layers.{bid}.mixer.down_proj",),
    # Gated DeltaNet (Qwen3-Next): read per key-head group, stored flat
    "gdn_qkvz": ("model.layers.{bid}.linear_attn.in_proj_qkvz",),
    # Qwen 3.5 keeps them apart and flat; the builder joins [qkv; z] and [b; a]
    "gdn_qkv": ("model.layers.{bid}.linear_attn.in_proj_qkv",),
    "gdn_z": ("model.layers.{bid}.linear_attn.in_proj_z",),
    "gdn_b": ("model.layers.{bid}.linear_attn.in_proj_b",),
    "gdn_a": ("model.layers.{bid}.linear_attn.in_proj_a",),
    "gdn_ba": ("model.layers.{bid}.linear_attn.in_proj_ba",),
    "gdn_conv": ("model.layers.{bid}.linear_attn.conv1d",),
    "gdn_dt_b": ("model.layers.{bid}.linear_attn.dt_bias",),
    "gdn_a_log": ("model.layers.{bid}.linear_attn.A_log",),
    "gdn_norm": ("model.layers.{bid}.linear_attn.norm",),
    "gdn_out": ("model.layers.{bid}.linear_attn.out_proj",),
    # Mamba-1 only (renamed ssm1_* for those families below)
    "ssm_x": ("backbone.layers.{bid}.mixer.x_proj", "model.layers.{bid}.mamba.x_proj"),
    "ssm_dt": ("backbone.layers.{bid}.mixer.dt_proj", "model.layers.{bid}.mamba.dt_proj"),
    "ssm_dt_norm": ("model.layers.{bid}.mamba.dt_layernorm",),
    "ssm_b_norm": ("model.layers.{bid}.mamba.b_layernorm",),
    "ssm_c_norm": ("model.layers.{bid}.mamba.c_layernorm",),
    # LFM2's short convolution (the conv weight is [d, 1, K], read as [d, K])
    "shortconv_in": ("model.layers.{bid}.conv.in_proj",),
    "shortconv_conv": ("model.layers.{bid}.conv.conv",),
    "shortconv_out": ("model.layers.{bid}.conv.out_proj",),
    # the routing correction bias: chooses experts, never weights them
    "exp_probs_b": ("model.layers.{bid}.mlp.gate.e_score_correction_bias",
                    "backbone.layers.{bid}.mixer.gate.e_score_correction_bias",
                    "model.layers.{bid}.mlp.moe_statics.e_score_correction_bias",
                    "model.layers.{bid}.mlp.e_score_correction_bias",               # EXAONE-MoE
                    "model.layers.{bid}.block_sparse_moe.e_score_correction_bias",  # MiniMax-M2
                    "model.layers.{bid}.mlp.expert_bias",                           # AFMoE
                    "model.layers.{bid}.feed_forward.expert_bias"),                 # LFM2-MoE
    # AFMoE gates its attention output with a projection of its own
    "attn_gate": ("model.layers.{bid}.self_attn.gate_proj",),
    "ffn_gate_up_exps": (
        "model.layers.{bid}.block_sparse_moe.input_linear",
        # transformers 5's fused experts as stored (Qwen 3.5 MoE): [E, 2*ff, dim], gate then up
        "model.layers.{bid}.mlp.experts.gate_up_proj",
    ),
    # gpt-oss: each expert's biases, stacked [E, 2*ff] (gate, up interleaved) and [E, dim]
    "ffn_gate_up_exps_b": ("model.layers.{bid}.mlp.experts.gate_up_proj_bias",),
    "ffn_down_exps_b": ("model.layers.{bid}.mlp.experts.down_proj_bias",),
    # multi-head latent attention (DeepSeek V2/V3, Kimi K2)
    "attn_q_a": ("model.layers.{bid}.self_attn.q_a_proj",),
    "attn_q_a_norm": ("model.layers.{bid}.self_attn.q_a_layernorm",),
    "attn_q_b": ("model.layers.{bid}.self_attn.q_b_proj",),
    "attn_kv_a_mqa": ("model.layers.{bid}.self_attn.kv_a_proj_with_mqa",),
    "attn_kv_a_norm": ("model.layers.{bid}.self_attn.kv_a_layernorm",),
    "attn_kv_b": ("model.layers.{bid}.self_attn.kv_b_proj",),
    # gpt-oss: one learned logit per head that joins every attention softmax
    "attn_sinks": ("model.layers.{bid}.self_attn.sinks",),
    # Nemotron-H's ungated experts as transformers 5 saves them: [E, ff, dim] and [E, dim, ff]
    "ffn_up_exps": ("backbone.layers.{bid}.mixer.experts.up_proj",),
    "ffn_down_exps": (
        "backbone.layers.{bid}.mixer.experts.down_proj",
        "model.layers.{bid}.block_sparse_moe.output_linear",
        "model.layers.{bid}.mlp.experts.down_proj",          # [E, dim, ff]
    ),
    # Post-norms: on the FFN branch's output. The attention-side post-norm is
    # named post_attention_layernorm in HF -- the same name Llama uses for its
    # PRE-FFN norm -- so it is remapped per architecture in HfWeights.
    "ffn_post_norm": (
        "model.layers.{bid}.post_feedforward_layernorm",
        "model.layers.{bid}.post_mlp_layernorm",          # GLM-4-0414
    ),
    "attn_q_norm": (
        "model.layers.{bid}.self_attn.q_norm",
        "model.layers.{bid}.self_attn.query_layernorm",   # HunYuan
        "model.layers.{bid}.self_attn.q_layernorm",       # Persimmon (a LayerNorm)
    ),
    "attn_k_norm": (
        "model.layers.{bid}.self_attn.k_norm",
        "model.layers.{bid}.self_attn.key_layernorm",     # HunYuan
        "model.layers.{bid}.self_attn.k_layernorm",       # Persimmon
    ),
    "attn_q": (
        "backbone.layers.{bid}.mixer.q_proj",
        "encoder.layer.{bid}.attention.self.query",
        "layers.{bid}.attention.wq",
        "layers.{bid}.self_attn.q_proj",
        "model.layers.layers.{bid}.self_attn.q_proj",
        "model.layers.{bid}.attention.wq",
        "model.layers.{bid}.self_attn.q_proj",
        "model.layers.{bid}.self_attn.q_proj_no_perm",
        "model.transformer.blocks.{bid}.q_proj",
        "transformer.decoder_layer.{bid}.multi_head_attention.query",
        "transformer.h.{bid}.attn.attention.q_proj",
        "transformer.h.{bid}.attn.q_proj",
        "transformer.layer.{bid}.attention.q_lin",
    ),
    "attn_k": (
        "backbone.layers.{bid}.mixer.k_proj",
        "encoder.layer.{bid}.attention.self.key",
        "layers.{bid}.attention.wk",
        "layers.{bid}.self_attn.k_proj",
        "model.layers.layers.{bid}.self_attn.k_proj",
        "model.layers.{bid}.attention.wk",
        "model.layers.{bid}.self_attn.k_proj",
        "model.layers.{bid}.self_attn.k_proj_no_perm",
        "model.transformer.blocks.{bid}.k_proj",
        "transformer.decoder_layer.{bid}.multi_head_attention.key",
        "transformer.h.{bid}.attn.attention.k_proj",
        "transformer.h.{bid}.attn.k",
        "transformer.h.{bid}.attn.k_proj",
        "transformer.layer.{bid}.attention.k_lin",
    ),
    "attn_v": (
        "backbone.layers.{bid}.mixer.v_proj",
        "encoder.layer.{bid}.attention.self.value",
        "layers.{bid}.attention.wv",
        "layers.{bid}.self_attn.v_proj",
        "model.layers.layers.{bid}.self_attn.v_proj",
        "model.layers.{bid}.attention.wv",
        "model.layers.{bid}.self_attn.v_proj",
        "model.transformer.blocks.{bid}.v_proj",
        "transformer.decoder_layer.{bid}.multi_head_attention.value",
        "transformer.h.{bid}.attn.attention.v_proj",
        "transformer.h.{bid}.attn.v",
        "transformer.h.{bid}.attn.v_proj",
        "transformer.layer.{bid}.attention.v_lin",
    ),
    "attn_qkv": (
        "encoder.layers.{bid}.attn.Wqkv",
        "encoder.layers.{bid}.mixer.Wqkv",
        "encoder.layers.{bid}.self_attention.query_key_value",
        "gpt_neox.layers.{bid}.attention.query_key_value",
        "h.{bid}.attn.c_attn",
        "h.{bid}.self_attention.query_key_value",
        "language_model.encoder.layers.{bid}.self_attention.query_key_value",
        "layers.{bid}.attn.Wqkv",
        "model.layers.layers.{bid}.mixer.qkv_proj",
        "model.layers.{bid}.attention.query_key_value",
        "model.layers.{bid}.attention.wqkv",          # InternLM2: per K/V group [q.. k v]
        "model.layers.{bid}.self_attn.W_pack",        # Baichuan: [Q | K | V]
        "model.layers.{bid}.linear_attn.in_proj_qkv",
        "model.layers.{bid}.self_attn.language_expert_query_key_value",
        "model.layers.{bid}.self_attn.qkv_proj",
        "model.layers.{bid}.self_attn.query_key_value",
        "transformer.blocks.{bid}.attn.Wqkv",
        "transformer.blocks.{bid}.norm_attn_norm.attn.Wqkv",
        "transformer.h.{bid}.attn.c_attn",
        "transformer.h.{bid}.mixer.Wqkv",
        "transformer.h.{bid}.self_attention.query_key_value",
        "transformer.layers.{bid}.attn.qkv_proj",
        "transformer_encoder.{bid}.qkv",
    ),
    "attn_output": (
        "backbone.layers.{bid}.mixer.o_proj",
        "encoder.layer.{bid}.attention.output.dense",
        "encoder.layers.{bid}.attn.out_proj",
        "encoder.layers.{bid}.mixer.out_proj",
        "encoder.layers.{bid}.self_attention.dense",
        "gpt_neox.layers.{bid}.attention.dense",
        "h.{bid}.attn.c_proj",
        "h.{bid}.self_attention.dense",
        "language_model.encoder.layers.{bid}.self_attention.dense",
        "layers.{bid}.attention.wo",
        "layers.{bid}.attn.Wo",
        "layers.{bid}.self_attn.o_proj",
        "model.layers.layers.{bid}.mixer.o_proj",
        "model.layers.layers.{bid}.self_attn.o_proj",
        "model.layers.{bid}.attention.dense",
        "model.layers.{bid}.attention.wo",
        "model.layers.{bid}.self_attn.dense",
        "model.layers.{bid}.self_attn.language_expert_dense",
        "model.layers.{bid}.self_attn.linear_attn",
        "model.layers.{bid}.self_attn.o_proj",
        "model.layers.{bid}.self_attn.out_proj",
        "model.transformer.blocks.{bid}.attn_out",
        "transformer.blocks.{bid}.attn.out_proj",
        "transformer.blocks.{bid}.norm_attn_norm.attn.out_proj",
        "transformer.decoder_layer.{bid}.multi_head_attention.linear",
        "transformer.h.{bid}.attn.attention.out_proj",
        "transformer.h.{bid}.attn.c_proj",
        "transformer.h.{bid}.attn.out_proj",
        "transformer.h.{bid}.mixer.out_proj",
        "transformer.h.{bid}.self_attention.dense",
        "transformer.layer.{bid}.attention.out_lin",
        "transformer.layers.{bid}.attn.out_proj",
        "transformer_encoder.{bid}.wo",
    ),
    "ffn_norm": (
        "encoder.layers.{bid}.post_attention_layernorm",
        "gpt_neox.layers.{bid}.post_attention_layernorm",
        "h.{bid}.ln_2",
        "h.{bid}.post_attention_layernorm",
        "transformer.h.{bid}.post_attention_layernorm",   # BLOOM/Falcon saved by transformers
        "model.layers.{bid}.pre_feedforward_layernorm",   # Gemma 2/3
        "model.layers.{bid}.pre_mlp_layernorm",           # AFMoE
        "transformer.h.{bid}.ln_mlp",                      # Falcon's new decoder
        "language_model.encoder.layers.{bid}.post_attention_layernorm",
        "layers.{bid}.ffn_norm",
        "layers.{bid}.mlp_norm",
        "layers.{bid}.post_attention_layernorm",
        "model.layers.layers.{bid}.pre_mlp_norm",
        "model.layers.{bid}.feedforward_layernorm",
        "model.layers.{bid}.ffn_norm",
        "model.layers.{bid}.ln2",
        "model.layers.{bid}.post_attention_layernorm",
        "model.layers.{bid}.pre_ff_layernorm",
        "model.layers.{bid}.pre_mlp_layernorm",
        "model.layers.{bid}.pre_moe_layernorm",
        "model.layers.{bid}.pre_moe_norm",
        "model.transformer.blocks.{bid}.ff_norm",
        "transformer.blocks.{bid}.norm_2",
        "transformer.decoder_layer.{bid}.rms_norm_2",
        "transformer.h.{bid}.ln_2",
        "transformer.layers.{bid}.ffn_norm",
        "transformer_encoder.{bid}.ffn_norm",
    ),
    "ffn_gate": (
        "encoder.layer.{bid}.mlp.gated_layers_w",
        "encoder.layers.{bid}.mlp.fc12",
        "layers.{bid}.feed_forward.w1",
        "layers.{bid}.mlp.gate_proj",
        "model.layers.layers.{bid}.mlp.gate_proj",
        "model.layers.{bid}.feed_forward.gate_proj",
        "model.layers.{bid}.feed_forward.w1",
        "model.layers.{bid}.mlp.gate_proj",
        "model.layers.{bid}.mlp.language_mlp.gate_proj",
        "model.layers.{bid}.residual_mlp.w1",
        "model.transformer.blocks.{bid}.ff_proj",
        "transformer.h.{bid}.mlp.c_fc2",
        "transformer.h.{bid}.mlp.c_fc_0",
        "transformer.h.{bid}.mlp.linear_1",
        "transformer.h.{bid}.mlp.w2",
    ),
    "ffn_up": (
        "backbone.layers.{bid}.mixer.up_proj",
        "encoder.layer.{bid}.intermediate.dense",
        "encoder.layer.{bid}.mlp.gated_layers",
        "encoder.layer.{bid}.mlp.gated_layers_v",
        "encoder.layer.{bid}.mlp.up_gated_layer",
        "encoder.layers.{bid}.mlp.dense_h_to_4h",
        "encoder.layers.{bid}.mlp.fc1",
        "encoder.layers.{bid}.mlp.fc11",
        "gpt_neox.layers.{bid}.mlp.dense_h_to_4h",
        "h.{bid}.mlp.c_fc",
        "h.{bid}.mlp.dense_h_to_4h",
        "language_model.encoder.layers.{bid}.mlp.dense_h_to_4h",
        "layers.{bid}.feed_forward.w3",
        "layers.{bid}.mlp.Wi",
        "layers.{bid}.mlp.up_proj",
        "model.layers.layers.{bid}.mlp.gate_up_proj",
        "model.layers.layers.{bid}.mlp.up_proj",
        "model.layers.{bid}.block_sparse_moe.up",
        "model.layers.{bid}.feed_forward.up_proj",
        "model.layers.{bid}.feed_forward.w3",
        "model.layers.{bid}.mlp.c_fc",
        "model.layers.{bid}.mlp.dense_h_to_4h",
        "model.layers.{bid}.mlp.fc1",
        "model.layers.{bid}.mlp.gate_up_proj",
        "model.layers.{bid}.mlp.language_mlp.up_proj",
        "model.layers.{bid}.mlp.up_proj",
        "model.layers.{bid}.residual_mlp.w3",
        "model.transformer.blocks.{bid}.up_proj",
        "transformer.blocks.{bid}.ffn.up_proj",
        "transformer.h.{bid}.mlp.c_fc",
        "transformer.h.{bid}.mlp.c_fc_1",
        "transformer.h.{bid}.mlp.dense_h_to_4h",
        "transformer.h.{bid}.mlp.fc1",
        "transformer.h.{bid}.mlp.fc_in",
        "transformer.h.{bid}.mlp.linear_3",
        "transformer.h.{bid}.mlp.w1",
        "transformer.layer.{bid}.ffn.lin1",
        "transformer_encoder.{bid}.ffn.w12",
    ),
    "ffn_down": (
        "backbone.layers.{bid}.mixer.down_proj",
        "encoder.layer.{bid}.mlp.down_layer",
        "encoder.layer.{bid}.mlp.wo",
        "encoder.layer.{bid}.output.dense",
        "encoder.layers.{bid}.mlp.dense_4h_to_h",
        "encoder.layers.{bid}.mlp.fc2",
        "gpt_neox.layers.{bid}.mlp.dense_4h_to_h",
        "h.{bid}.mlp.c_proj",
        "h.{bid}.mlp.dense_4h_to_h",
        "language_model.encoder.layers.{bid}.mlp.dense_4h_to_h",
        "layers.{bid}.feed_forward.w2",
        "layers.{bid}.mlp.Wo",
        "layers.{bid}.mlp.down_proj",
        "model.layers.h.{bid}.mlp.c_proj",
        "model.layers.layers.{bid}.mlp.down_proj",
        "model.layers.{bid}.block_sparse_moe.down",
        "model.layers.{bid}.feed_forward.down_proj",
        "model.layers.{bid}.feed_forward.w2",
        "model.layers.{bid}.mlp.c_proj",
        "model.layers.{bid}.mlp.dense_4h_to_h",
        "model.layers.{bid}.mlp.down_proj",
        "model.layers.{bid}.mlp.fc2",
        "model.layers.{bid}.mlp.language_mlp.down_proj",
        "model.layers.{bid}.residual_mlp.w2",
        "model.transformer.blocks.{bid}.ff_out",
        "transformer.blocks.{bid}.ffn.down_proj",
        "transformer.h.{bid}.mlp.c_proj",
        "transformer.h.{bid}.mlp.dense_4h_to_h",
        "transformer.h.{bid}.mlp.fc2",
        "transformer.h.{bid}.mlp.fc_out",
        "transformer.layer.{bid}.ffn.lin2",
        "transformer.layers.{bid}.ffn.proj_2",
        "transformer_encoder.{bid}.ffn.w3",
    ),}

# Compiled once. Longest pattern first so "model.layers.{bid}.self_attn.q_proj"
# is tried before any shorter prefix that could also match it.
# One tensor per expert: {xid} is the expert index. Mixtral names them w1 (gate),
# w3 (up), w2 (down); the Qwen-MoE line and OLMoE name them like a dense MLP.
# OPT (model.decoder.*), BioGPT (biogpt.*), XGLM (model.* with fairseq layer
# names) and CodeGen (GPT-J names, one fused qkv_proj). Read from transformers
# 4.57.6's modelling files for each.
_NAMES_MORE = {
    "token_embd": ("model.decoder.embed_tokens", "biogpt.embed_tokens"),
    "position_embd": ("model.decoder.embed_positions", "biogpt.embed_positions"),
    "output_norm": ("model.decoder.final_layer_norm", "biogpt.layer_norm", "model.layer_norm"),
    "output": ("output_projection",),
    "attn_norm": ("model.decoder.layers.{bid}.self_attn_layer_norm",
                  "biogpt.layers.{bid}.self_attn_layer_norm",
                  "model.layers.{bid}.self_attn_layer_norm"),
    "attn_q": ("model.decoder.layers.{bid}.self_attn.q_proj", "biogpt.layers.{bid}.self_attn.q_proj"),
    "attn_k": ("model.decoder.layers.{bid}.self_attn.k_proj", "biogpt.layers.{bid}.self_attn.k_proj"),
    "attn_v": ("model.decoder.layers.{bid}.self_attn.v_proj", "biogpt.layers.{bid}.self_attn.v_proj"),
    "attn_output": ("model.decoder.layers.{bid}.self_attn.out_proj",
                    "biogpt.layers.{bid}.self_attn.out_proj",
                    "model.layers.{bid}.self_attn.out_proj"),
    "attn_qkv": ("transformer.h.{bid}.attn.qkv_proj",),
    "ffn_norm": ("model.decoder.layers.{bid}.final_layer_norm",
                 "biogpt.layers.{bid}.final_layer_norm",
                 "model.layers.{bid}.final_layer_norm"),
    "ffn_up": ("model.decoder.layers.{bid}.fc1", "biogpt.layers.{bid}.fc1", "model.layers.{bid}.fc1"),
    "ffn_down": ("model.decoder.layers.{bid}.fc2", "biogpt.layers.{bid}.fc2", "model.layers.{bid}.fc2"),
    # Apertus: its own norm names, and xIELU's two learned scalars per layer
    "attn_norm_apertus": (),
    "ffn_xielu_ap": ("model.layers.{bid}.mlp.act_fn.alpha_p",),
    "ffn_xielu_an": ("model.layers.{bid}.mlp.act_fn.alpha_n",),
}
_NAMES_MORE["attn_norm"] = _NAMES_MORE["attn_norm"] + ("model.layers.{bid}.attention_layernorm",
                                                       "transformer.blocks.{bid}.norm_attn_norm.norm_1")
# DBRX: its blocks, its fused Wqkv, and its experts stacked [E x ff, d] (split below)
_NAMES_MORE["ffn_norm"] = _NAMES_MORE["ffn_norm"] + ("transformer.blocks.{bid}.norm_attn_norm.norm_2",)
_NAMES_MORE["attn_qkv"] = _NAMES_MORE["attn_qkv"] + ("transformer.blocks.{bid}.norm_attn_norm.attn.Wqkv",)
_NAMES_MORE["attn_output"] = _NAMES_MORE["attn_output"] + ("transformer.blocks.{bid}.norm_attn_norm.attn.out_proj",)
_NAMES_MORE["ffn_gate_inp"] = ("transformer.blocks.{bid}.ffn.router.layer",)
_NAMES_MORE["dbrx_w1"] = ("transformer.blocks.{bid}.ffn.experts.mlp.w1",)
_NAMES_MORE["dbrx_v1"] = ("transformer.blocks.{bid}.ffn.experts.mlp.v1",)
_NAMES_MORE["dbrx_w2"] = ("transformer.blocks.{bid}.ffn.experts.mlp.w2",)
_NAMES_MORE["ffn_norm"] = _NAMES_MORE["ffn_norm"] + ("model.layers.{bid}.feedforward_layernorm",)
del _NAMES_MORE["attn_norm_apertus"]
for _role, _pats in _NAMES_MORE.items():
    _NAMES[_role] = tuple(_NAMES.get(_role, ())) + tuple(p_ for p_ in _pats if p_ not in _NAMES.get(_role, ()))

_EXPERT_NAMES = {
    "ffn_gate": ("model.layers.{bid}.feed_forward.experts.{xid}.gate_proj",   # Jamba
                 "model.layers.{bid}.feed_forward.experts.{xid}.w1",          # LFM2-MoE
                 "model.layers.{bid}.block_sparse_moe.experts.{xid}.w1",
                 "model.layers.{bid}.mlp.experts.{xid}.gate_proj"),
    "ffn_up":   ("model.layers.{bid}.feed_forward.experts.{xid}.up_proj",
                 "backbone.layers.{bid}.mixer.experts.{xid}.up_proj",       # Nemotron-H (ungated)
                 "model.layers.{bid}.feed_forward.experts.{xid}.w3",
                 "model.layers.{bid}.block_sparse_moe.experts.{xid}.w3",
                 "model.layers.{bid}.mlp.experts.{xid}.up_proj"),
    "ffn_down": ("model.layers.{bid}.feed_forward.experts.{xid}.down_proj",
                 "backbone.layers.{bid}.mixer.experts.{xid}.down_proj",
                 "model.layers.{bid}.feed_forward.experts.{xid}.w2",
                 "model.layers.{bid}.block_sparse_moe.experts.{xid}.w2",
                 "model.layers.{bid}.mlp.experts.{xid}.down_proj"),
}
_EXP = [(re.compile("^" + re.escape(_p).replace(r"\{bid\}", r"(\d+)").replace(r"\{xid\}", r"(\d+)") + "$"), _role)
        for _role, _pats in _EXPERT_NAMES.items() for _p in _pats]

_TOP, _BLK = [], []
for _role, _pats in _NAMES.items():
    for _p in _pats:
        if "{bid}" in _p:
            _BLK.append((re.compile("^" + re.escape(_p).replace(r"\{bid\}", r"(\d+)") + "$"), _role))
        else:
            _TOP.append((re.compile("^" + re.escape(_p) + "$"), _role))
_BLK.sort(key=lambda t: -len(t[0].pattern))
_TOP.sort(key=lambda t: -len(t[0].pattern))


_VISION_NAMES = [
    (r"model\.vision_model\.embeddings\.patch_embedding\.(weight|bias)", r"v.patch.\1"),
    (r"model\.vision_model\.embeddings\.position_embedding\.weight", "v.pos.weight"),
    (r"model\.vision_model\.encoder\.layers\.(\d+)\.layer_norm1\.(weight|bias)", r"v.blk.\1.ln1.\2"),
    (r"model\.vision_model\.encoder\.layers\.(\d+)\.layer_norm2\.(weight|bias)", r"v.blk.\1.ln2.\2"),
    (r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)", r"v.blk.\1.q.\2"),
    (r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)", r"v.blk.\1.k.\2"),
    (r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)", r"v.blk.\1.v.\2"),
    (r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)", r"v.blk.\1.o.\2"),
    (r"model\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.(weight|bias)", r"v.blk.\1.fc1.\2"),
    (r"model\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.(weight|bias)", r"v.blk.\1.fc2.\2"),
    (r"model\.vision_model\.post_layernorm\.(weight|bias)", r"v.post_ln.\1"),
    (r"model\.connector\.modality_projection\.proj\.weight", "v.proj.weight"),
]


def _vlm_chat(path):
    """The Idefics3 chat frame, as the engine needs it to build an image turn
    from the file alone: special tokens by id (read from the tokenizer's own
    added-token table, not assumed) and the plain-text pieces between them,
    which the engine's tokenizer encodes itself. Frame, verified against the
    reference processor: <|im_start|>User:<image block>QUESTION
    <end_of_utterance>\nAssistant:"""
    try:
        with open(os.path.join(path, "tokenizer.json"), encoding="utf-8") as f:
            added = {t["content"]: int(t["id"]) for t in json.load(f).get("added_tokens", [])}
    except (OSError, ValueError, KeyError):
        return None
    need = {"start": "<|im_start|>", "fake": "<fake_token_around_image>",
            "global": "<global-img>", "end": "<end_of_utterance>"}
    if any(v not in added for v in need.values()):
        return None
    out = {k: added[v] for k, v in need.items()}
    out.update(user="User:", assistant="\nAssistant:")
    return out


def vision_name(hf_name):
    """Vision tower / connector tensor -> its v.* name, or None."""
    import re
    for pat, rep in _VISION_NAMES:
        if re.fullmatch(pat, hf_name):
            return re.sub(pat, rep, hf_name)
    return None


def vision_pos_ids(n_pos):
    """Position ids of a full, unpadded image, computed EXACTLY as the HF
    reference does in float32: bucketize(i / side * (1 - 1e-6), arange(1/side,
    1, 1/side), right=True). That maps the 32 patch rows of a 512 image to
    buckets [0, 0, 1, ..., 30] -- rows 0 and 1 share an embedding and bucket 31
    is never read. It looks like an off-by-one; it is what the weights were
    trained against, so the file stores the table gathered this way."""
    side = int(round(n_pos ** 0.5))
    bounds = np.arange(1, side, dtype=np.float32) / np.float32(side)
    frac = np.arange(side, dtype=np.float32) / np.float32(side) * np.float32(1 - 1e-6)
    b = np.searchsorted(bounds, frac, side="right")
    return (b[:, None] * side + b[None, :]).reshape(-1)


def canonical(hf_name):
    """HF tensor name -> the canonical name the rest of the pipeline uses,
    or None for tensors that are not weights (attention masks, inv_freq)."""
    stem, dot, kind = hf_name.rpartition(".")
    if dot and kind in ("weight", "bias"):
        pass
    else:
        stem, kind = hf_name, "weight"
    for rx, role in _TOP:
        if rx.match(stem):
            return f"{role}.{kind}"
    for rx, role in _BLK:
        m = rx.match(stem)
        if m:
            return f"blk.{int(m.group(1))}.{role}.{kind}"
    for rx, role in _EXP:
        m = rx.match(stem)
        if m:
            return f"blk.{int(m.group(1))}.{role}.{int(m.group(2))}.{kind}"
    return None


def _e4m3_table():
    """The 256 float8 e4m3fn values (1 sign, 4 exponent bits biased 7, 3 mantissa; no inf, 0x7f/0xff NaN)."""
    v = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s_, e, m = (b >> 7) & 1, (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            x = np.nan
        elif e == 0:
            x = (m / 8.0) * 2.0 ** -6
        else:
            x = (1 + m / 8.0) * 2.0 ** (e - 7)
        v[b] = -x if s_ else x
    return v


_E4M3 = _e4m3_table()


_FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=np.float32)


def _mxfp4_virtual(header):
    """gpt-oss ships its experts as MXFP4 (X_blocks: [E, out, G, 16] bytes of two 4-bit E2M1
    values, low nibble first; X_scales: [E, out, G] E8M0 exponents biased 127) -- the
    original weights, as published. Each pair becomes one tensor X in transformers' own
    parameter layout, [E, in, out] (integrations/mxfp4.py convert_moe_packed_tensors),
    decoded one expert at a time when read."""
    for b in [k for k in header if k.endswith("_blocks")]:
        base = b[: -len("_blocks")]
        s = base + "_scales"
        if s not in header or header[b]["dtype"] != "U8":
            continue
        E, out, G, B = header[b]["shape"]
        header[base] = {"dtype": "MXFP4", "shape": [E, G * B * 2, out], "blocks": header.pop(b),
                        "scales": header.pop(s)}


class _Shard:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n).decode("utf-8"))
        _mxfp4_virtual(self.header)
        self.data_start = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def get_expert(self, name, e):
        """One expert of a stacked [E, ...] tensor, without reading the others."""
        h = self.header[name]
        if h["dtype"] != "MXFP4":
            return self.get(name)[e]
        bl, sc = h["blocks"], h["scales"]
        E, out, G, B = bl["shape"]
        a0, _ = bl["data_offsets"]
        per = out * G * B
        blk = np.asarray(self.mm[self.data_start + a0 + e * per: self.data_start + a0 + (e + 1) * per]).reshape(out, G, B)
        s0, _ = sc["data_offsets"]
        exp = np.asarray(self.mm[self.data_start + s0 + e * out * G: self.data_start + s0 + (e + 1) * out * G])
        w = np.empty((out, G, B * 2), dtype=np.float32)
        w[..., 0::2] = _FP4[blk & 0x0F]
        w[..., 1::2] = _FP4[blk >> 4]
        w = np.ldexp(w, (exp.astype(np.int32) - 127).reshape(out, G, 1))
        return np.ascontiguousarray(w.reshape(out, G * B * 2).T)

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name):
        e = self.header[name]
        if e["dtype"] == "MXFP4":
            return np.stack([self.get_expert(name, i) for i in range(e["shape"][0])])
        if e["dtype"] == "F8_E4M3":
            # a block-scaled FP8 release: weight x weight_scale_inv per block
            a, b = e["data_offsets"]
            w = _E4M3[np.asarray(self.mm[self.data_start + a: self.data_start + b])].reshape(e["shape"])
            sc = name[: -len("weight")] + "weight_scale_inv" if name.endswith("weight") else None
            if sc and sc in self.header:
                s_ = self.get(sc)
                s_ = s_.reshape(s_.shape[0], -1) if s_.ndim >= 2 else s_.reshape(1, -1)
                br = -(-w.shape[0] // s_.shape[0])
                bc = -(-w.shape[1] // s_.shape[1])
                w = w * np.repeat(np.repeat(s_, br, 0), bc, 1)[: w.shape[0], : w.shape[1]]
            return w.astype(np.float32)
        dt = _DTYPE[e["dtype"]]
        a, b = e["data_offsets"]
        raw = self.mm[self.data_start + a: self.data_start + b]
        arr = np.frombuffer(raw.tobytes(), dtype=dt)
        if e["dtype"] == "BF16":
            # bf16 is the high half of fp32; widen rather than round
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.reshape(e["shape"]).astype(np.float32)


def _past_stack(hf, cfg):
    """A tensor of a layer numbered past the stack: an MTP / next-token layer."""
    m = re.search(r"\blayers\.(\d+)\.", hf)
    n = cfg.get("num_hidden_layers")
    return bool(m and n and int(m.group(1)) >= int(n))


def model_type_of(cfg):
    """config.json's model_type, or -- for a remote-code config that omits it
    (MiniCPM) -- the one its class name gives: MiniCPMForCausalLM -> minicpm."""
    mt = cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type") or ""
    if mt:
        return mt
    for arch in cfg.get("architectures") or []:
        m = re.fullmatch(r"(\w+?)(?:ForCausalLM|LMHeadModel|ForConditionalGeneration)", arch)
        if m:
            return m.group(1).lower()
    return ""


# ---- PyTorch .bin checkpoints, read without torch ---------------------------
# A .bin is a zip: <prefix>/data.pkl (a pickle naming each tensor's storage,
# offset, shape and stride) and <prefix>/data/<key> (each storage's raw bytes).
# The pickle is read with an unpickler that can build ONLY an OrderedDict, a
# tensor descriptor and a storage dtype marker -- anything else in the file is
# refused, so a .bin cannot run code here (what torch's weights_only does).
_BIN_DTYPES = {"FloatStorage": "F32", "HalfStorage": "F16", "BFloat16Storage": "BF16",
               "DoubleStorage": "F64", "LongStorage": "I64", "IntStorage": "I32",
               "ShortStorage": "I16", "CharStorage": "I8", "ByteStorage": "U8", "BoolStorage": "BOOL"}
_BIN_NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64, "I64": np.int64,
           "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


class _BinTensor:
    __slots__ = ("key", "dtype", "offset", "shape", "stride")

    def __init__(self, key, dtype, offset, shape, stride):
        self.key, self.dtype, self.offset = key, dtype, int(offset)
        self.shape, self.stride = [int(x) for x in shape], [int(x) for x in stride]


def bin_tensors(zf):
    """{name: _BinTensor} from an open zipfile.ZipFile over a torch .bin."""
    import collections
    import pickle
    pkl = next((n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"), None)
    if pkl is None:
        raise SystemExit("this .bin is not a torch zip checkpoint (the pre-1.6 format is not read)")

    class _Storage:
        def __init__(self, dtype):
            self.dtype = dtype

    def rebuild_tensor(storage, offset, size, stride, *rest):
        return _BinTensor(storage[1], storage[0], offset, size, stride)

    def rebuild_parameter(data, *rest):
        return data

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("collections", "OrderedDict"):
                return collections.OrderedDict
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild_tensor
            if (module, name) == ("torch._utils", "_rebuild_parameter"):
                return rebuild_parameter
            if module == "torch" and name in _BIN_DTYPES:
                return _Storage(_BIN_DTYPES[name])
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed in a weights file")

        def persistent_load(self, pid):
            # ('storage', storage type, key, location, numel)
            if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
                raise pickle.UnpicklingError("unexpected persistent id in a weights file")
            st = pid[1]
            return (st.dtype if isinstance(st, _Storage) else "F32", str(pid[2]))

    import io
    obj = Restricted(io.BytesIO(zf.read(pkl))).load()
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    return {k: v for k, v in obj.items() if isinstance(v, _BinTensor)}, pkl.rsplit("data.pkl", 1)[0]


class _BinShard:
    """A torch .bin read like a safetensors shard: header (name, dtype, shape)
    up front, each tensor's bytes mapped from the zip on demand."""

    def __init__(self, path):
        import zipfile
        self.path = path
        with zipfile.ZipFile(path) as zf:
            self.tensors, prefix = bin_tensors(zf)
            self.where = {}
            for info in zf.infolist():
                if info.filename.startswith(prefix + "data/"):
                    if info.compress_type != zipfile.ZIP_STORED:
                        raise SystemExit(f"{os.path.basename(path)}: compressed storage is not read")
                    with open(path, "rb") as f:
                        f.seek(info.header_offset)
                        h = f.read(30)
                        n_name, n_extra = struct.unpack("<HH", h[26:30])
                    self.where[info.filename[len(prefix) + 5:]] = (info.header_offset + 30 + n_name + n_extra,
                                                                    info.file_size)
        self.header = {k: {"dtype": t.dtype, "shape": t.shape, "data_offsets": [0, 0]}
                       for k, t in self.tensors.items()}
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def names(self):
        return list(self.tensors)

    def get(self, name):
        t = self.tensors[name]
        start, size = self.where[t.key]
        dt = np.dtype(_BIN_NP[t.dtype])
        flat = np.frombuffer(self.mm[start:start + size].tobytes(), dtype=dt)
        if not t.shape:
            arr = flat[t.offset:t.offset + 1].reshape(())
        else:
            arr = np.lib.stride_tricks.as_strided(flat[t.offset:], shape=t.shape,
                                                  strides=[s * dt.itemsize for s in t.stride])
        arr = np.array(arr)                       # contiguous, owned
        if t.dtype == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)


def pick_weight_files(names):
    """The weight files a transformers load reads, from a repo's or folder's file names:
    root-level safetensors (a LoRA adapter's and Mistral's native consolidated.* aside when
    HF-format shards are there too), else the torch .bin shards, never optimizer or
    training-args .bin files. One rule for the download, the size estimate and the header
    check (2026-10-19: '*.safetensors' also took subfolders and consolidated.safetensors, so
    a Mistral repo shipping both formats downloaded its weights twice)."""
    root = [n for n in names if "/" not in n and "\\" not in n]
    st = sorted(n for n in root if n.endswith(".safetensors") and not n.startswith("adapter_model"))
    hf = [n for n in st if not n.startswith("consolidated")]
    if hf:
        return hf
    bins = sorted(n for n in root if re.fullmatch(r"(pytorch_model|model)(-\d+-of-\d+)?\.bin", n))
    return bins or st


def weight_files(path):
    """The checkpoint's weight files in a folder (pick_weight_files)."""
    return pick_weight_files(os.listdir(path))


class _HeaderShard(_Shard):
    """A safetensors file as its header alone (a stub, or a real file not
    read): every name, dtype and shape; a read returns zeros of the right shape.
    For checking what a build would do without its weights (sml_compat.py)."""
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n).decode("utf-8"))
        _mxfp4_virtual(self.header)
        self.data_start = 8 + n

    def get(self, name):
        return np.zeros(self.header[name]["shape"] or [1], dtype=np.float32)

    def get_expert(self, name, e):
        return np.zeros(self.header[name]["shape"][1:], dtype=np.float32)


class HfWeights:
    """Weights over an HF checkpoint (header_only: shapes and names, no payload)."""

    lossy = False

    def __init__(self, path, header_only=False):
        self.path = path
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        # Vision-language (Idefics3 / SmolVLM): the text side is an ordinary
        # decoder under model.text_model, read with ITS OWN config; the vision
        # tower and connector become v.* tensors that the engine runs apart.
        self.vision = None
        vlm = self.cfg.get("model_type") in _VLM_ARCHS
        if vlm:
            outer, vc = self.cfg, self.cfg["vision_config"]
            self.vision = {
                "arch": outer["model_type"],
                "n_embd": int(vc["hidden_size"]), "n_layer": int(vc["num_hidden_layers"]),
                "n_head": int(vc["num_attention_heads"]), "n_ff": int(vc["intermediate_size"]),
                "image": int(vc["image_size"]), "patch": int(vc["patch_size"]),
                "eps": float(vc.get("layer_norm_eps", 1e-6)),
                "scale": int(outer.get("scale_factor", 1)),
                "image_token_id": int(outer["image_token_id"]),
                "chat": _vlm_chat(path),
            }
            self.cfg = dict(outer["text_config"])
            self.cfg.setdefault("tie_word_embeddings", outer.get("tie_word_embeddings", False))
        self.cfg["model_type"] = model_type_of(self.cfg)
        self.mm_text = _MM_TEXT.get(self.cfg.get("model_type", ""))
        if self.mm_text and "text_config" in self.cfg:
            outer = self.cfg
            self.cfg = dict(outer["text_config"])
            # Gemma 3 ties its head to the embedding inside the wrapper and says
            # so nowhere in the file; the others state it or have an lm_head
            self.cfg.setdefault("tie_word_embeddings",
                                outer.get("tie_word_embeddings", outer.get("model_type") == "gemma3"))
        rp = self.cfg.get("rope_parameters")
        if isinstance(rp, dict) and rp and all(isinstance(v, dict) or v is None for v in rp.values()):
            # {layer type: settings}: the global layers' settings are the model's;
            # a sliding layer's own base becomes the local base (Gemma 3's scheme)
            full = rp.get("full_attention") or next(v for v in rp.values() if v)
            loc = rp.get("sliding_attention") or {}
            if loc.get("rope_theta") and full.get("rope_theta") and loc["rope_theta"] != full["rope_theta"]:
                self.cfg.setdefault("rope_local_base_freq", loc["rope_theta"])
            if loc and loc.get("rope_type", "default") != full.get("rope_type", "default"):
                self.cfg["_rope_scaled_global_only"] = True
            rp = full
        if isinstance(rp, dict):
            # transformers 5 nests the rotary settings; read naively, Qwen 3.5
            # would rotate every dimension at base 1e4 instead of 25% at 1e7
            for k_ in ("rope_theta", "partial_rotary_factor"):
                if k_ in rp and k_ not in self.cfg:
                    self.cfg[k_] = rp[k_]
            if rp.get("rope_type", "default") not in ("default", None) and not self.cfg.get("rope_scaling"):
                self.cfg["rope_scaling"] = dict(rp)
        self.arch = self.cfg.get("model_type", "unknown")
        if self.arch == "opt" and (self.cfg.get("do_layer_norm_before", True) is False
                                   or int(self.cfg.get("word_embed_proj_dim") or self.cfg.get("hidden_size") or 0)
                                   != int(self.cfg.get("hidden_size") or 0)):
            # OPT-350m alone: LayerNorm AFTER each block, and an embedding narrower
            # than the stream with project_in/project_out. Neither runs here.
            raise SystemExit("this OPT checkpoint (OPT-350m) normalizes after each block and projects its "
                             "embedding; the engine runs the pre-norm OPTs (125m, 1.3b and up) only")
        self.norm_head = self.arch == "baichuan" and (int(self.cfg.get("vocab_size") or 0) == 125696
                                                      or "z_loss_weight" in self.cfg)
        _layer_types(self.cfg)
        # HunYuan-A13B states its expert settings once per layer; equal lists are
        # one setting, unequal ones per-layer widths the engine does not run
        for k_ in ("moe_topk", "moe_intermediate_size", "num_shared_expert", "num_experts"):
            v_ = self.cfg.get(k_)
            if isinstance(v_, list) and v_:
                if len(set(v_)) > 1:
                    raise SystemExit(f"{k_} differs from layer to layer ({sorted(set(v_))}): "
                                     "per-layer expert settings are not supported")
                self.cfg[k_] = v_[0]
        rs_ = self.cfg.get("rope_scaling")
        if isinstance(rs_, dict):
            # the trained and the original context, for the scalings that need
            # them (YaRN, LongRoPE), read before LongRoPE caps the context below
            self.cfg["rope_scaling"] = dict(rs_, _max_pos=self.cfg.get("max_position_embeddings"),
                                            _orig_max=self.cfg.get("original_max_position_embeddings"))
        rs_ = self.cfg.get("rope_scaling") or {}
        if str(rs_.get("rope_type", rs_.get("type", ""))).lower() == "longrope":
            orig_ = rs_.get("original_max_position_embeddings") or rs_.get("_orig_max")
            if orig_:
                # the short factors are exact up to the original context: the model
                # is offered that much (Phi-4-mini: 4096 of its 131072)
                self.cfg["max_position_embeddings"] = int(orig_)
        # GPT-2-lineage configs name the same quantities differently, and
        # LayerNorm models state layer_norm_eps(ilon) where this file read only
        # rms_norm_eps -- so every LayerNorm build took the 1e-6 default. The
        # aliases fill ABSENT keys only: a config that states both keeps its own.
        for want, alts in _CFG_ALIASES.items():
            if want not in self.cfg:
                for alt in alts:
                    if self.cfg.get(alt) is not None:
                        self.cfg[want] = self.cfg[alt]
                        break
        c_, a_ = self.cfg, self.cfg.get("model_type", "")
        # Nemotron-H states its blocks, not their count
        if "num_hidden_layers" not in c_:
            if isinstance(c_.get("layers_block_type"), list):
                c_["num_hidden_layers"] = len(c_["layers_block_type"])
            elif c_.get("hybrid_override_pattern"):
                c_["num_hidden_layers"] = len(c_["hybrid_override_pattern"])
        nh_ = int(c_.get("num_attention_heads") or 1)
        # K/V heads under other names: Falcon's multi_query / num_kv_heads,
        # StarCoder-1's multi_query -- one shared K/V head
        # (set, not defaulted: these modelling classes ignore a stray
        # num_key_value_heads, so the file must too)
        if a_ == "falcon":
            c_["num_key_value_heads"] = (int(c_.get("num_kv_heads") or nh_)
                                         if c_.get("new_decoder_architecture")
                                         else 1 if c_.get("multi_query") else nh_)
        if a_ == "gpt_bigcode":
            c_["num_key_value_heads"] = 1 if c_.get("multi_query", True) else nh_
        if a_ == "mpt":
            # transformers' MPT builds 4 x d_model whatever expansion_ratio says,
            # and its GELU is the exact one
            c_["intermediate_size"] = 4 * int(c_["hidden_size"])
            c_.setdefault("hidden_act", "gelu")
        if a_ == "olmo":
            c_.setdefault("rms_norm_eps", 1e-5)     # OlmoLayerNorm's fixed eps
        if a_ == "dbrx":
            ac, fc = c_.get("attn_config") or {}, c_.get("ffn_config") or {}
            c_["num_key_value_heads"] = int(ac.get("kv_n_heads") or c_.get("n_heads") or 1)
            c_.setdefault("rope_theta", float(ac.get("rope_theta", 10000.0)))
            c_["intermediate_size"] = int(fc.get("ffn_hidden_size", 3584))
            c_["num_experts_per_tok"] = int(fc.get("moe_top_k", 1))
            p_norm = fc.get("moe_normalize_expert_weights", 1.0)
            if p_norm not in (None, 1, 1.0):
                raise SystemExit(f"dbrx: expert weights normalized by a {p_norm}-norm; only 1 (their sum) is implemented")
            c_["norm_topk_prob"] = p_norm is not None
            c_.setdefault("rms_norm_eps", 1e-5)     # nn.LayerNorm's default eps
            act = (fc.get("ffn_act_fn") or {}).get("name", "silu")
            if act != "silu":
                raise SystemExit(f"dbrx: expert activation {act} is not implemented")
        if "intermediate_size" not in self.cfg and "hidden_size" in self.cfg:
            self.cfg["intermediate_size"] = 4 * int(self.cfg["hidden_size"])   # GPT-2 n_inner: null
        self.conv1d = self.arch in _CONV1D_ARCHS

        files = weight_files(path)
        if not files:
            raise SystemExit(f"no .safetensors or pytorch_model .bin in {path}")
        self.shards = {}
        self.owner = {}
        for fn in sorted(files):
            sh = (_BinShard if fn.endswith(".bin") else _HeaderShard if header_only else _Shard)(os.path.join(path, fn))
            self.shards[fn] = sh
            for hf in sh.names():
                if self.mm_text:
                    t_ = self._mm_name(hf)
                    if t_ is None:
                        continue
                    c = canonical(t_)
                elif not vlm:
                    # older shards were saved from the base model, without its
                    # "model." prefix (OPT-6.7B's decoder.*): transformers adds it
                    c = canonical(hf) or (canonical("model." + hf) if not hf.startswith("model.") else None)
                elif hf.startswith(("model.vision_model.", "model.connector.")):
                    c = vision_name(hf)
                else:
                    c = canonical(hf.replace("model.text_model.", "model.", 1))
                if c and self.cfg.get("model_type", "") in _MAMBA1 and ".ssm_" in c:
                    c = c.replace(".ssm_", ".ssm1_")    # the same names, Mamba-1 maths
                if c and self.cfg.get("model_type", "") in _POSTNORM_ARCHS \
                        and ".post_attention_layernorm." in hf:
                    c = c.replace(".ffn_norm.", ".attn_post_norm.")
                mb_ = re.match(r"blk\.(\d+)\.", c or "")
                if mb_ and int(mb_.group(1)) >= int(self.cfg.get("num_hidden_layers") or 1 << 30):
                    self.extra_layers = True     # GLM-4.5's next-token-prediction layer, past the stack
                    continue
                if c:
                    self.owner[c] = (fn, hf)

        # A tensor the name table cannot place used to vanish: HunYuan's
        # query_layernorm/key_layernorm were dropped and the model ran without
        # its Q/K norms -- 13% logit error, no complaint (2026-09-28). Buffers
        # that are not weights are the only thing allowed through.
        _not_weights = ("rotary_emb.inv_freq", "attn.masked_bias", "attn.bias",
                        ".inv_freq", "position_ids", "masked_bias",
                        "weight_scale_inv",           # FP8 block scales, applied when the weight is read
                        ".mup_vector")                # Falcon-H1's mixer multipliers: folded (_falcon_h1)
        # multi-token-prediction heads (ERNIE 4.5 mtp_block, Qwen3-Next mtp.*):
        # a training aid and a speculative-decoding draft, never read when the
        # model writes one token at a time
        _mtp = ("mtp.", "model.mtp_block.", "model.mtp.", "model.mtp_")
        lost = []
        for fn, sh in self.shards.items():
            for hf in sh.names():
                e = sh.header[hf]
                # causal-mask buffers (U8/BOOL) and scalars are not weights
                if e["dtype"] in ("U8", "BOOL") or not e["shape"]:
                    continue
                if self.mm_text:
                    t_ = self._mm_name(hf)
                    if t_ is None or canonical(t_) or t_.endswith(_not_weights):
                        continue
                    lost.append(hf)
                    continue
                if vlm and not hf.startswith("model.text_model.") and \
                        not hf.startswith(("model.vision_model.", "model.connector.", "lm_head")):
                    continue
                probe = hf.replace("model.text_model.", "model.", 1) if vlm else hf
                if (vlm and hf.startswith(("model.vision_model.", "model.connector."))) \
                        or canonical(probe) or (not vlm and not hf.startswith("model.") and canonical("model." + hf))                         or hf.endswith(_not_weights) or hf.startswith(_mtp) or _past_stack(hf, self.cfg):
                    continue
                lost.append(hf)
        if lost:
            raise SystemExit(
                f"{len(lost)} tensor(s) in this checkpoint have no place in the name "
                f"table, and running without them is a different model: "
                + ", ".join(lost[:8]) + (" ..." if len(lost) > 8 else ""))

        # Tied embeddings: the head IS the embedding table. Qwen2.5-3B ties;
        # Llama-3 and Mistral do not, and treating an untied model as tied
        # produces fluent-looking nonsense rather than an error.
        self.tied = "output.weight" not in self.owner
        if self.tied:
            if not self.cfg.get("tie_word_embeddings", True):
                raise SystemExit("config says untied but no lm_head tensor found")
            self.owner["output.weight"] = self.owner["token_embd.weight"]

        # GPT-NeoX fuses Q, K and V into one tensor, laid out per HEAD:
        # head0[q|k|v] head1[q|k|v] ... Splitting it here means the rest of the
        # pipeline -- quantizer, format, engine -- sees the same three tensors
        # every other architecture has, with no special case anywhere else.
        self.fused_qkv = {}
        for name in [k for k in self.owner if k.endswith("attn_qkv.weight")]:
            base = name[: -len("attn_qkv.weight")]
            for who in ("q", "k", "v"):
                self.fused_qkv[base + f"attn_{who}.weight"] = (name, who)
            bname = base + "attn_qkv.bias"
            if bname in self.owner:
                for who in ("q", "k", "v"):
                    self.fused_qkv[base + f"attn_{who}.bias"] = (bname, who)

        # OLMo 2 applies post_attention_layernorm to the attention OUTPUT and has
        # no pre-FFN norm; in Llama the same name IS the pre-FFN norm. Read the
        # Llama way, OLMo 2 runs as a different model -- fluently.
        if self.cfg.get("model_type", "") in _POSTNORM_ARCHS:
            for k in [k for k in self.owner if k.endswith(".ffn_norm.weight")]:
                if self.owner[k][1].endswith("post_attention_layernorm.weight"):
                    self.owner[k.replace(".ffn_norm.", ".attn_post_norm.")] = self.owner.pop(k)

        # Stacked experts: input_linear [E, 2*ff, dim] is each expert's gate then
        # up (the first half takes the activation), output_linear [E, dim, ff] its
        # down. A small cache keeps one stacked tensor from being converted once
        # per expert -- 2,300 conversions of 128 MB on the 1B-A400M.
        self.split_exp = {}
        self._exp_cache = {}
        for name in [k for k in self.owner if k.endswith("ffn_gate_up_exps.weight")]:
            base = name[: -len("ffn_gate_up_exps.weight")]
            fn, hf = self.owner[name]
            for e in range(int(self.shards[fn].header[hf]["shape"][0])):
                self.split_exp[f"{base}ffn_gate.{e}.weight"] = (name, e, "gate")
                self.split_exp[f"{base}ffn_up.{e}.weight"] = (name, e, "up")
        for name in [k for k in self.owner if k.endswith("ffn_up_exps.weight")]:
            base = name[: -len("ffn_up_exps.weight")]
            fn, hf = self.owner[name]
            for e in range(int(self.shards[fn].header[hf]["shape"][0])):
                self.split_exp[f"{base}ffn_up.{e}.weight"] = (name, e, "down")   # taken whole
        for name in [k for k in self.owner if k.endswith("ffn_down_exps.weight")]:
            base = name[: -len("ffn_down_exps.weight")]
            fn, hf = self.owner[name]
            for e in range(int(self.shards[fn].header[hf]["shape"][0])):
                self.split_exp[f"{base}ffn_down.{e}.weight"] = (name, e, "down")
        # gpt-oss stores experts as transformers' parameters: gate_up [E, dim, 2*ff] with gate
        # and up INTERLEAVED (even columns gate, odd up), down [E, ff, dim]; biases per expert
        self.oss = self.cfg.get("model_type") == "gpt_oss"
        self.split_expb = {}
        for name in [k for k in self.owner if k.endswith("ffn_gate_up_exps_b.weight")]:
            base = name[: -len("ffn_gate_up_exps_b.weight")]
            fn, hf = self.owner[name]
            for e in range(int(self.shards[fn].header[hf]["shape"][0])):
                self.split_expb[f"{base}ffn_gate.{e}.bias"] = (name, e, 0)
                self.split_expb[f"{base}ffn_up.{e}.bias"] = (name, e, 1)
        for name in [k for k in self.owner if k.endswith("ffn_down_exps_b.weight")]:
            base = name[: -len("ffn_down_exps_b.weight")]
            fn, hf = self.owner[name]
            for e in range(int(self.shards[fn].header[hf]["shape"][0])):
                self.split_expb[f"{base}ffn_down.{e}.bias"] = (name, e, None)

        # DBRX stacks each layer's experts as [E x ff, d]: w1 the gates, v1 the
        # ups, w2 the downs TRANSPOSED (DbrxExpertGLU: x @ w1_e.T, x @ v1_e.T,
        # h @ w2_e) -- read here as E ordinary expert tensors
        self.dbrx_exp = {}
        if self.cfg.get("model_type") == "dbrx":
            ffd = int(self.cfg["intermediate_size"])
            for name in [k for k in self.owner if ".dbrx_w1." in k]:
                base = name[: -len("dbrx_w1.weight")]
                fn, hf = self.owner[name]
                ne = int(self.shards[fn].header[hf]["shape"][0]) // ffd
                for e in range(ne):
                    self.dbrx_exp[f"{base}ffn_gate.{e}.weight"] = (base + "dbrx_w1.weight", e, 0)
                    self.dbrx_exp[f"{base}ffn_up.{e}.weight"] = (base + "dbrx_v1.weight", e, 0)
                    self.dbrx_exp[f"{base}ffn_down.{e}.weight"] = (base + "dbrx_w2.weight", e, 1)

        # A fused gate|up projection (Phi-3, GLM): ffn_up with twice the FFN
        # width and no gate tensor. Every such family's MLP chunks it in two and
        # activates the FIRST half, so the gate is rows [0, ff) and up is [ff, 2ff).
        # Structural, not declared: an ungated up projection is ff rows by
        # definition, so a 2ff-row one with no gate can only be this.
        self.split_gu = {}
        ff = int(self.cfg.get("intermediate_size") or 0)
        if str(self.cfg.get("hidden_act", "silu")).startswith("gelu"):
            ff = 0           # an ungated GELU MLP never fuses a gate
        for name in [k for k in self.owner if k.endswith("ffn_up.weight")]:
            base = name[: -len("ffn_up.weight")]
            if base + "ffn_gate.weight" in self.owner or not ff:
                continue
            fn, hf = self.owner[name]
            sh = self.shards[fn].header[hf]["shape"]
            if len(sh) == 2 and int(sh[0]) == 2 * ff:
                self.split_gu[base + "ffn_gate.weight"] = (name, 0)
                self.split_gu[base + "ffn_up.weight"] = (name, 1)
                bname = base + "ffn_up.bias"
                if bname in self.owner:
                    self.split_gu[base + "ffn_gate.bias"] = (bname, 0)
                    self.split_gu[base + "ffn_up.bias"] = (bname, 1)

        # OLMo (v1) normalizes with no learned weight or bias at all; the file
        # carries ones so the engine's LayerNorm computes exactly that
        self.synth = {}
        # xIELU (Apertus): per layer, the four numbers the activation uses, as
        # transformers computes them -- softplus of the stored bf16 scalars, in
        # bf16 -- stored as one small tensor the engine reads
        for k_ in [k for k in self.owner if k.endswith(".ffn_xielu_ap.weight")]:
            base = k_[: -len("ffn_xielu_ap.weight")]
            fn, hf = self.owner.pop(k_)
            fn2, hf2 = self.owner.pop(base + "ffn_xielu_an.weight")
            sh = self.shards[fn]
            pre = hf[: -len("alpha_p")]

            def raw(n_, s_=sh):
                return float(np.asarray(s_.get(n_), dtype=np.float32).reshape(-1)[0])

            beta = raw(pre + "beta") if (pre + "beta") in sh.header else 0.5
            eps = raw(pre + "eps") if (pre + "eps") in sh.header else _bf16(-1e-6)
            ap = _bf16(_softplus(raw(hf)))
            an = _bf16(beta + _bf16(_softplus(float(np.asarray(self.shards[fn2].get(hf2),
                                                               dtype=np.float32).reshape(-1)[0]))))
            self.synth[base + "ffn_xielu.weight"] = np.array([ap, an, beta, eps], dtype=np.float32)
        # XGLM's positions are sinusoidal and not in the checkpoint: the table
        # transformers computes (tensor2tensor order, sin half then cos half,
        # offset 2), stored as the learned-position rows the engine already adds
        if self.cfg.get("model_type") == "xglm":
            self.synth["position_embd.weight"] = _xglm_positions(
                int(self.cfg.get("max_position_embeddings", 2048)), int(self.cfg["hidden_size"]))
        if self.cfg.get("model_type") == "olmo":
            ones = np.ones(int(self.cfg["hidden_size"]), dtype=np.float32)
            for L in range(int(self.cfg["num_hidden_layers"])):
                self.synth[f"blk.{L}.attn_norm.weight"] = ones
                self.synth[f"blk.{L}.ffn_norm.weight"] = ones
            self.synth["output_norm.weight"] = ones

        # A "shared" MLP in a layer with no router (Granite-4.0-H with no experts)
        # is simply the layer's FFN: filed as a shared expert, nothing ran it.
        for name in [k for k in self.owner if k.endswith("ffn_gate_up_shexp.weight")]:
            base = name[: -len("ffn_gate_up_shexp.weight")]
            dense = (base + "ffn_gate_inp.weight") not in self.owner
            self.split_gu[base + ("ffn_gate.weight" if dense else "ffn_gate_shexp.weight")] = (name, 0)
            self.split_gu[base + ("ffn_up.weight" if dense else "ffn_up_shexp.weight")] = (name, 1)
            if dense and (base + "ffn_down_shexp.weight") in self.owner:
                self.owner[base + "ffn_down.weight"] = self.owner.pop(base + "ffn_down_shexp.weight")

        if self.cfg.get("model_type") == "nemotron_h":
            for k_ in [k for k in self.owner if k.endswith(".ffn_up.weight")]:
                base = k_[: -len("ffn_up.weight")]
                if (base + "attn_q.weight") not in self.owner and (base + "ssm_in.weight") not in self.owner \
                        and (base + "attn_norm.weight") in self.owner:
                    self.owner[base + "ffn_norm.weight"] = self.owner.pop(base + "attn_norm.weight")
            # an expert layer ("E"): its one norm is the FFN's, like an MLP layer's
            for k_ in [k for k in self.owner if k.endswith(".ffn_gate_inp.weight")]:
                base = k_[: -len("ffn_gate_inp.weight")]
                if (base + "attn_q.weight") not in self.owner and (base + "ssm_in.weight") not in self.owner \
                        and (base + "attn_norm.weight") in self.owner:
                    self.owner[base + "ffn_norm.weight"] = self.owner.pop(base + "attn_norm.weight")
        self.joined = {}
        for name in [k for k in self.owner if k.endswith("gdn_qkv.weight")]:
            base = name[: -len("gdn_qkv.weight")]
            self.joined[base + "gdn_qkvz.weight"] = (base + "gdn_qkv.weight", base + "gdn_z.weight")
            self.joined[base + "gdn_ba.weight"] = (base + "gdn_b.weight", base + "gdn_a.weight")
        self.split_qg = {}
        if self.cfg.get("model_type") in _ATTN_GATE or self.cfg.get("attn_output_gate"):
            for name in [k for k in self.owner if k.endswith("attn_q.weight")]:
                base = name[: -len("attn_q.weight")]
                self.split_qg[base + "attn_q.weight"] = (name, 0)
                self.split_qg[base + "attn_gate.weight"] = (name, 1)

        self.t = {k: None for k in self.owner}   # name set, matching Weights.t
        self.t.update({k: None for k in self.synth})
        self.t.update({k: None for k in self.split_qg})
        self.t.update({k: None for k in self.joined})
        for parts in self.joined.values():
            for k_ in parts:
                self.t.pop(k_, None)
        self.t.update({k: None for k in self.split_gu})
        self.t.update({k: None for k in self.fused_qkv})
        self.t.update({k: None for k in self.split_exp})
        self.t.update({k: None for k in self.split_expb})
        self.t.update({k: None for k in self.dbrx_exp})
        for k in list(self.t):
            if (k.endswith("attn_qkv.weight") or k.endswith("attn_qkv.bias")
                    or k.endswith("_exps.weight") or k.endswith("_exps_b.weight")
                    or k.endswith("ffn_gate_up_shexp.weight")
                    or ".dbrx_" in k):
                del self.t[k]

    def flags(self):
        f = arch_flags(self.cfg)
        # Cross-check the name list against the tensors. A norm BIAS can only
        # belong to a LayerNorm -- RMSNorm has no additive term -- so its
        # presence is conclusive and outranks any architecture list. Absence is
        # ambiguous (MPT runs LayerNorm with no bias), so it proves nothing and
        # the list stands. This exists because the RoPE list was silently
        # inverted for five architectures; a flag that contradicts the file it
        # describes should not survive to the header.
        has_norm_bias = any(k.endswith("attn_norm.bias") or
                            k.endswith("output_norm.bias") for k in self.t)
        if has_norm_bias and f["norm_type"] != 1:
            raise SystemExit(
                f"arch '{self.arch}' is listed as RMSNorm but the checkpoint "
                f"carries norm bias tensors, which only LayerNorm has")
        return f

    def _ff_width(self):
        # The checkpoint's own up projection, when layer 0 has one, outranks the
        # config: transformers' MPT builds 4 x d_model whatever expansion_ratio
        # says, and Falcon states ffn_hidden_size. Experts fall back to config.
        for n in ("blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"):
            if n in self.t:
                return int(self.shape(n)[0])
        return int(self.cfg["intermediate_size"])

    def field(self, key, default=None):
        c = self.cfg
        a = self.arch
        table = {
            "general.architecture": a,
            f"{a}.block_count": c["num_hidden_layers"],
            f"{a}.embedding_length": c["hidden_size"],
            # a pure state-space model has no attention heads at all
            f"{a}.attention.head_count": c.get("num_attention_heads", 1),
            f"{a}.attention.head_count_kv": c.get("num_key_value_heads",
                                                  c.get("num_attention_heads", 1)),
            f"{a}.feed_forward_length": self._ff_width(),
            f"{a}.context_length": c.get("max_position_embeddings", 32768),
            f"{a}.rope.freq_base": c.get("rope_theta", 10000.0),
            f"{a}.attention.layer_norm_rms_epsilon": c.get("rms_norm_eps", 1e-6),
        }
        if key in table:
            return table[key]
        if key == "tokenizer.tokens":
            return self._vocab()[0]
        if key == "tokenizer.merges":
            return self._vocab()[1]
        if key == "tokenizer.bos":
            return self._vocab()[2]
        if key == "tokenizer.eos":
            return self._vocab()[3]
        if key == "tokenizer.unigram":
            self._vocab()
            return getattr(self, "_uni", None)
        if key == "tokenizer.pretok":
            self._vocab()
            return getattr(self, "_pret", None)
        return default

    # A fused QKV tensor is packed one of several ways and the file does not
    # say which. GPT-NeoX interleaves per HEAD -- head0[q|k|v] head1[q|k|v] --
    # while GPT-2, MPT and Phi-3 store three contiguous blocks [Q|K|V]. Reading
    # one as the other yields a model that still produces fluent text, so the
    # layout is declared per family and anything unlisted refuses rather than
    # guesses.
    _QKV_LAYOUT = {
        "gpt_neox": "head",
        "persimmon": "head",
        "phi3": "block",
        "gpt2": "block",
        "bloom": "head",      # view(n_head, 3, head_dim): per-head [q|k|v]
        "mpt": "block",
        "dbrx": "block",      # [Q | K | V], K and V kv_n_heads wide
        # CodeGen: qkv reshaped to (4 model-parallel blocks, [q | v | k] each) --
        # value BEFORE key (CodeGenAttention.forward, transformers 4.57.6)
        "codegen": "codegen",
        # InternLM2: wqkv viewed (n_kv, n_rep + 2, head_dim, d) -- each K/V group's
        # query heads, then its key, then its value (InternLM2Attention.forward)
        "internlm2": "group",
        # Baichuan: W_pack is [Q | K | V], all heads (BaichuanAttention.forward)
        "baichuan": "block",
    }

    def _gdn_flat(self, name, a):
        """Qwen3-Next packs in_proj_qkvz per key head: [q k v.. z..] for each of
        nk groups, and in_proj_ba as [b.. a..] per group. The file holds them flat
        -- [q_all | k_all | v_all | z_all] and [b_all | a_all] -- rows only."""
        c = self.cfg
        nk, nv = int(c["linear_num_key_heads"]), int(c["linear_num_value_heads"])
        dk, dv = int(c["linear_key_head_dim"]), int(c["linear_value_head_dim"])
        r = nv // nk
        if name.endswith("gdn_qkvz.weight"):
            g = a.reshape(nk, 2 * dk + 2 * r * dv, -1)
            q, k = g[:, :dk], g[:, dk:2 * dk]
            v, z = g[:, 2 * dk:2 * dk + r * dv], g[:, 2 * dk + r * dv:]
            return np.ascontiguousarray(np.concatenate(
                [q.reshape(nk * dk, -1), k.reshape(nk * dk, -1),
                 v.reshape(nv * dv, -1), z.reshape(nv * dv, -1)], 0))
        g = a.reshape(nk, 2 * r, -1)
        return np.ascontiguousarray(np.concatenate(
            [g[:, :r].reshape(nv, -1), g[:, r:].reshape(nv, -1)], 0))

    def _mm_name(self, hf):
        """A multimodal wrapper's tensor name as its text model's, or None for
        the vision tower and projector (set aside, not lost)."""
        if hf.startswith(self.mm_text["vision"]):
            return None
        for old, new in self.mm_text["rename"]:
            if hf.startswith(old):
                return new + hf[len(old):]
        return hf

    def _qkv_layout(self):
        c = self.cfg
        arch = c.get("model_type", "")
        # Falcon packs three ways, by config (FalconAttention._split_heads):
        # per K/V group [q.. k v] (new_decoder_architecture), [Q | k | v] with
        # one shared head (multi_query), or per head [q k v]
        if arch == "falcon":
            return ("group" if c.get("new_decoder_architecture")
                    else "block" if c.get("multi_query") else "head")
        if arch == "gpt_bigcode":
            return "block" if c.get("multi_query", True) else None
        return self._QKV_LAYOUT.get(arch)

    def _split_qkv(self, name):
        src, who = self.fused_qkv[name]
        fn, hf = self.owner[src]
        a = self._t(name, self.shards[fn].get(hf))
        arch = self.cfg.get("model_type", "")
        layout = self._qkv_layout()
        if layout is None:
            raise SystemExit(
                f"'{arch}' fuses Q/K/V into one tensor and its packing order is "
                f"not recorded. Reading it the wrong way round produces fluent "
                f"output from the wrong model, so add it to _QKV_LAYOUT after "
                f"checking that architecture's reference implementation.")
        n_head = int(self.cfg["num_attention_heads"])
        idx = {"q": 0, "k": 1, "v": 2}[who]
        rows = a.shape[0] // 3
        if layout == "codegen":
            mp = 4
            loc = a.shape[0] // (3 * mp)
            off = {"q": 0, "v": 1, "k": 2}[who]
            g = a.reshape(mp, 3, loc, *a.shape[1:])[:, off]
            return np.ascontiguousarray(g.reshape(mp * loc, *a.shape[1:]))
        if layout == "group":
            n_kv = int(self.cfg["num_key_value_heads"])
            hd = a.shape[0] // (n_head + 2 * n_kv)
            per = n_head // n_kv
            g = a.reshape(n_kv, per + 2, hd, *a.shape[1:])
            if who == "q":
                return np.ascontiguousarray(g[:, :per].reshape(n_head * hd, *a.shape[1:]))
            return np.ascontiguousarray(g[:, per if who == "k" else per + 1].reshape(n_kv * hd, *a.shape[1:]))
        if layout == "block":
            # [Q | K | V] with K and V n_kv heads each: equal thirds only
            # without grouped K/V (Phi-3 has it)
            qn, kn = self._qkv_rows(a.shape[0])
            lo = (0, qn, qn + kn)[idx]
            hi = lo + (qn if who == "q" else kn)
            return np.ascontiguousarray(a[lo:hi])
        if layout == "head":
            hd = a.shape[0] // (3 * n_head)
            v = (a.reshape(n_head, 3, hd, a.shape[1])[:, idx] if a.ndim > 1
                 else a.reshape(n_head, 3, hd)[:, idx])
            return np.ascontiguousarray(v.reshape(n_head * hd, *a.shape[1:]))
        return np.ascontiguousarray(a[idx * rows:(idx + 1) * rows])

    def _qkv_rows(self, total):
        c = self.cfg
        n_head = int(c["num_attention_heads"])
        n_kv = int(c.get("num_key_value_heads") or n_head)
        hd = int(c.get("head_dim") or c["hidden_size"] // n_head)
        if n_head * hd + 2 * n_kv * hd != total:
            hd = total // (n_head + 2 * n_kv)
        return n_head * hd, n_kv * hd

    def _split_exp(self, name):
        src, e, part = self.split_exp[name]
        if getattr(self, "oss", False):
            fn, hf = self.owner[src]
            w = self.shards[fn].get_expert(hf, e).T          # [out, in]
            if part == "gate":
                w = w[0::2]
            elif part == "up":
                w = w[1::2]
            return np.ascontiguousarray(w)
        a = self._exp_cache.get(src)
        if a is None:
            if len(self._exp_cache) >= 2:
                self._exp_cache.pop(next(iter(self._exp_cache)))
            fn, hf = self.owner[src]
            a = self.shards[fn].get(hf)
            self._exp_cache[src] = a
        w = a[e]
        if part == "gate":
            w = w[: w.shape[0] // 2]
        elif part == "up":
            w = w[w.shape[0] // 2:]
        return np.ascontiguousarray(w)

    def _t(self, name, a):
        """GPT-2 stores its projections as Conv1D, [in, out]: the transpose of
        every other checkpoint. Read as-is, a square matrix is silently applied
        backwards and a model still runs; so the flip happens here, once."""
        if self.conv1d and a.ndim == 2 and name.startswith("blk."):
            return np.ascontiguousarray(a.T)
        return a

    def _rope_af(self):
        """{layer: factor} for the layers whose rotary carries an attention factor
        (YaRN, LongRoPE): every rotating layer, or -- OLMo 3 -- the global ones."""
        if hasattr(self, "_af"):
            return self._af
        from sml_common import yarn_attention_factor
        self._af = {}
        c = self.cfg
        af = yarn_attention_factor(c.get("rope_scaling"))
        if af == 1.0 or c.get("model_type") in _QKNORM_AFTER_ROPE:
            return self._af                     # a norm after the rotary cancels it, as in transformers
        n = int(c["num_hidden_layers"])
        _, rope, _ = layer_attn(c, n)
        types = c.get("layer_types") or []
        only_global = c.get("_rope_scaled_global_only") or (c.get("model_type") == "olmo3"
                                                            and "sliding_attention" in types)
        for L in range(n):
            if rope is not None and not rope[L]:
                continue
            if only_global and L < len(types) and types[L] != "full_attention":
                continue
            self._af[L] = af
        return self._af

    def _falcon_h1(self, name, a):
        """Falcon-H1's muP multipliers, folded into the weights they follow (FalconH1DecoderLayer,
        FalconH1Mixer, FalconH1MLP): attention input and key, attention and Mamba outputs, the
        MLP's gate and down, Mamba's input scale and its per-segment zxBCdt vector."""
        m = re.match(r"blk\.\d+\.(\w+)\.(weight|bias)$", name)
        if not m:
            return a
        c, (role, kind) = self.cfg, m.groups()
        if role == "ssm_in":
            di = int(c.get("mamba_d_ssm") or int(c.get("mamba_expand") or 2) * int(c["hidden_size"]))
            gn = int(c.get("mamba_n_groups") or 1) * int(c.get("mamba_d_state") or 16)
            mul = list(c.get("ssm_multipliers") or [1.0] * 5)
            vec = np.concatenate([np.full(di, mul[0]), np.full(di, mul[1]), np.full(gn, mul[2]),
                                  np.full(gn, mul[3]), np.full(a.shape[0] - 2 * di - 2 * gn, mul[4])]).astype(np.float32)
            if kind == "weight":
                vec = vec * float(c.get("ssm_in_multiplier") or 1.0)
            return (a * vec[:, None] if a.ndim == 2 else a * vec).astype(np.float32)
        f = 1.0
        if role in ("attn_q", "attn_k", "attn_v") and kind == "weight":
            f *= float(c.get("attention_in_multiplier") or 1.0)
        if role == "attn_k":
            f *= float(c.get("key_multiplier") or 1.0)
        if role == "attn_output":
            f *= float(c.get("attention_out_multiplier") or 1.0)
        if role == "ssm_out":
            f *= float(c.get("ssm_out_multiplier") or 1.0)
        mm = c.get("mlp_multipliers") or [1.0, 1.0]
        if role == "ffn_gate":
            f *= float(mm[0])
        if role == "ffn_down":
            f *= float(mm[1])
        return a * np.float32(f) if f != 1.0 else a

    def get(self, name):
        a = self._get(name)
        if self.cfg.get("model_type") == "falcon_h1":
            a = self._falcon_h1(name, a)
        if _is_mla(self.cfg) and re.match(r"blk\.\d+\.attn_q(_b)?\.weight$", name):
            # latent attention's query heads are [no rotary | rotary]; the engine rotates
            # the FIRST n_rot dims of a head, so each head's rotary rows go first (the
            # keys are assembled the same way), which leaves every dot product unchanged
            c = self.cfg
            nope, rope = int(c["qk_nope_head_dim"]), int(c["qk_rope_head_dim"])
            nh = int(c["num_attention_heads"])
            a = a.reshape(nh, nope + rope, -1)
            a = np.concatenate([a[:, nope:], a[:, :nope]], 1)
            # YaRN / LongRoPE scale cos and sin, so the rotary dims of q and k each carry the
            # factor: both factors land on q's rotary rows (MiniCPM3's LongRoPE)
            af = self._rope_af().get(int(name.split(".")[1]), 1.0)
            if af != 1.0:
                a[:, :rope] *= af * af
            return np.ascontiguousarray(a.reshape(nh * (nope + rope), -1))
        m_ = re.match(r"blk\.(\d+)\.attn_([qk])(_norm)?\.(weight|bias)$", name)
        if m_ and int(m_.group(1)) in self._rope_af():
            # cos and sin times f is f * rope(q): rope is linear, so f goes on the
            # rotated part of q and k where they are made -- the projection, or
            # the Q/K norm when one follows it (it would undo a scaled projection).
            # Per layer and per dimension, so OLMo 3's global-only YaRN and a
            # partial rotary are as exact as the plain case.
            L, which, is_norm = int(m_.group(1)), m_.group(2), bool(m_.group(3))
            has_norm = f"blk.{L}.attn_{which}_norm.weight" in self.t
            if is_norm == has_norm:
                fl = arch_flags(self.cfg)
                nh = int(self.cfg.get("num_attention_heads") or 1)
                hd = int(self.cfg.get("head_dim") or int(self.cfg["hidden_size"]) // nh)
                rot = int(fl.get("n_rot") or hd)
                f = self._rope_af()[L]
                a = np.array(a, dtype=np.float32, copy=True)
                idx = np.arange(a.shape[0]) % hd < rot
                a[idx] *= f
        if name == "output.weight" and self.norm_head:
            # Baichuan 2's NormHead: the head's rows are L2-normalized before use
            # (modeling_baichuan.py, NormHead.forward in eval)
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        return a

    def _get(self, name):
        if name in self.joined:
            parts = []
            for k_ in self.joined[name]:
                fn, hf = self.owner[k_]
                parts.append(self.shards[fn].get(hf))
            return np.ascontiguousarray(np.concatenate(parts, 0))
        if name in self.split_qg:
            # per head [q (hd) | gate (hd)] -> the q half or the gate half
            src, half = self.split_qg[name]
            fn, hf = self.owner[src]
            a = self.shards[fn].get(hf)
            nh = int(self.cfg["num_attention_heads"])
            hd = a.shape[0] // (2 * nh)
            return np.ascontiguousarray(a.reshape(nh, 2, hd, -1)[:, half].reshape(nh * hd, -1))
        if name in self.synth:
            return self.synth[name]
        if name == "position_embd.weight" and self.arch in _POS_OFFSET2:
            fn, hf = self.owner[name]
            return np.ascontiguousarray(self.shards[fn].get(hf)[2:])
        if name in self.split_gu:
            src, half = self.split_gu[name]
            fn, hf = self.owner[src]
            a = self._t(name, self.shards[fn].get(hf))
            n = a.shape[0] // 2
            return np.ascontiguousarray(a[half * n:(half + 1) * n])
        if name in self.split_exp:
            return self._split_exp(name)
        if name in self.split_expb:
            src, e, half = self.split_expb[name]
            fn, hf = self.owner[src]
            b = np.asarray(self.shards[fn].get(hf), dtype=np.float32)[e]
            return np.ascontiguousarray(b if half is None else b[half::2])
        if name in self.dbrx_exp:
            src, e, tr = self.dbrx_exp[name]
            a = self._exp_cache.get(src)
            if a is None:
                if len(self._exp_cache) >= 3:
                    self._exp_cache.pop(next(iter(self._exp_cache)))
                fn, hf = self.owner[src]
                a = self.shards[fn].get(hf)
                self._exp_cache[src] = a
            ffd = int(self.cfg["intermediate_size"])
            w = a[e * ffd:(e + 1) * ffd]
            return np.ascontiguousarray(w.T if tr else w)
        if name in self.fused_qkv:
            return self._split_qkv(name)
        fn, hf = self.owner[name]
        a = self._t(name, self.shards[fn].get(hf))
        if name.endswith("gdn_qkvz.weight") or name.endswith("gdn_ba.weight"):
            return self._gdn_flat(name, a)
        if self.arch in _NORM_PLUS1 and name.endswith("norm.weight") \
                and not name.endswith("gdn_norm.weight"):   # the gated norm's weight is plain
            a = a + np.float32(1.0)
        if (name.endswith("shortconv_conv.weight") or name.endswith("ssm_conv.weight")
                or name.endswith("ssm1_conv.weight") or name.endswith("gdn_conv.weight")) and a.ndim == 3:
            return np.ascontiguousarray(a.reshape(a.shape[0], a.shape[2]))
        if name == "v.patch.weight":
            # conv [out, c, kh, kw] with stride = kernel IS a matmul over patches
            # flattened in (c, kh, kw) order -- which is how the engine cuts them
            return np.ascontiguousarray(a.reshape(a.shape[0], -1))
        if name == "v.pos.weight":
            return np.ascontiguousarray(a[vision_pos_ids(a.shape[0])])
        return a if a.ndim > 1 else a.reshape(-1)

    def shape(self, name):
        if name in self.joined:
            rows = 0
            for k_ in self.joined[name]:
                fn, hf = self.owner[k_]
                sh = self.shards[fn].header[hf]["shape"]
                rows += int(sh[0])
                cols = int(sh[1])
            return (rows, cols)
        if name in self.split_qg:
            src, _ = self.split_qg[name]
            fn, hf = self.owner[src]
            sh = self.shards[fn].header[hf]["shape"]
            return (int(sh[0]) // 2, int(sh[1]))
        if name in self.synth:
            sh_ = self.synth[name].shape
            return (int(sh_[0]), int(sh_[1])) if len(sh_) > 1 else (1, int(sh_[0]))
        if name == "position_embd.weight" and self.arch in _POS_OFFSET2:
            fn, hf = self.owner[name]
            sh_ = self.shards[fn].header[hf]["shape"]
            return (int(sh_[0]) - 2, int(sh_[1]))
        if name in self.split_gu:
            src, _ = self.split_gu[name]
            fn, hf = self.owner[src]
            sh = self.shards[fn].header[hf]["shape"]
            return (int(sh[0]) // 2, int(sh[1])) if len(sh) > 1 else (1, int(sh[0]) // 2)
        if name in self.dbrx_exp:
            src, _, tr = self.dbrx_exp[name]
            fn, hf = self.owner[src]
            d_ = int(self.shards[fn].header[hf]["shape"][1])
            ffd = int(self.cfg["intermediate_size"])
            return (d_, ffd) if tr else (ffd, d_)
        if name in self.split_exp:
            src, _, part = self.split_exp[name]
            fn, hf = self.owner[src]
            sh = self.shards[fn].header[hf]["shape"]
            if getattr(self, "oss", False):                  # [E, in, out]
                return (int(sh[2]) // 2 if part in ("gate", "up") else int(sh[2]), int(sh[1]))
            rows = int(sh[1]) // 2 if part in ("gate", "up") else int(sh[1])
            return (rows, int(sh[2]))
        if name in getattr(self, "split_expb", {}):
            src, _, half = self.split_expb[name]
            fn, hf = self.owner[src]
            sh = self.shards[fn].header[hf]["shape"]
            return (1, int(sh[1]) // 2 if half is not None else int(sh[1]))
        if name in self.fused_qkv:
            src, _ = self.fused_qkv[name]
            fn, hf = self.owner[src]
            sh = self.shards[fn].header[hf]["shape"]
            if self.conv1d and len(sh) == 2:
                sh = sh[::-1]
            rows = int(sh[0]) // 3
            _, who = self.fused_qkv[name]
            if self._qkv_layout() in ("block", "group"):
                qn, kn = self._qkv_rows(int(sh[0]))
                rows = qn if who == "q" else kn
            return (rows, int(sh[1])) if len(sh) > 1 else (1, rows)
        """Dimensions without touching the payload. Reading 6 GB and converting
        it to fp32 just to learn tensor shapes is not a rounding error: it is
        the whole file, twice."""
        fn, hf = self.owner[name]
        sh = self.shards[fn].header[hf]["shape"]
        if name == "v.patch.weight":
            return (int(sh[0]), int(np.prod(sh[1:])))
        if (name.endswith("shortconv_conv.weight") or name.endswith("ssm_conv.weight")
                or name.endswith("ssm1_conv.weight") or name.endswith("gdn_conv.weight")) and len(sh) == 3:
            return (int(sh[0]), int(sh[2]))
        if self.conv1d and len(sh) == 2 and name.startswith("blk."):
            sh = sh[::-1]
        return (int(sh[0]), int(sh[1])) if len(sh) > 1 else (1, int(sh[0]))

    def _vocab(self):
        if hasattr(self, "_vc"):
            return self._vc
        tjp = os.path.join(self.path, "tokenizer.json")
        if not os.path.exists(tjp) and os.path.exists(os.path.join(self.path, "vocab.json"))                 and os.path.exists(os.path.join(self.path, "merges.txt")):
            # GPT-2-style byte-level BPE in its two original files (OPT, older
            # GPT-2 descendants): the same vocabulary tokenizer.json would hold
            self._vc = gpt2_files_vocab(self.path)
            from sml_pretok import for_gpt2_files
            self._pret = for_gpt2_files()
            return self._vc
        tik = [f for f in os.listdir(self.path) if f.endswith(".tiktoken") or f == "tiktoken.model"] \
            if not os.path.exists(tjp) else []
        if tik:
            # Qwen v1: a tiktoken rank file, its specials and split pattern in its own code
            self._vc, self._pret = tiktoken_vocab(self.path, tik[0])
            return self._vc
        if not os.path.exists(tjp) and os.path.exists(os.path.join(self.path, "tokenizer.model")):
            # Baichuan, InternLM2, Orion, XVERSE ... ship only a SentencePiece
            # model; the engine reads SPM vocabularies (sml-vocab.c), so it is
            # read here directly rather than refused for want of a .json
            *vc, self._uni = spm_vocab(self.path)
            self._vc = tuple(vc)
            if self._uni is None:
                # a SentencePiece BPE (Llama 2 lineage): U+2581 before the text and for
                # every space, then the merges over the whole stretch
                # (Baichuan 2 turns the leading U+2581 off: add_dummy_prefix in the model file)
                from sml_pretok import for_spm_bpe
                norm, trainer = SPM_OPTS.get(os.path.abspath(self.path), ({}, {}))
                self._pret = for_spm_bpe(add_dummy_prefix=norm.get("add_dummy_prefix", True),
                                         byte_fallback=trainer.get("byte_fallback", "<0x00>" in vc[0]),
                                         unk=vc[0].index("<unk>") if "<unk>" in vc[0] else 0)
                if norm.get("remove_extra_whitespaces", True):
                    print("tokenizer: this SentencePiece model folds repeated spaces; the engine "
                          "keeps them (lines with runs of spaces may cut differently)", flush=True)
            return self._vc
        with open(tjp, encoding="utf-8") as f:
            tj = json.load(f)
        model = tj.get("model", {})
        self._uni = None
        if model.get("type") == "BPE" or ("type" not in model and "merges" in model):   # GPT-2's has no type
            # its own text pipeline: normalizer, pre-tokenizer splits, merge options
            # (sml_pretok.py; without it the engine split at spaces and missed
            # Mistral and Phi-3 on almost every line, 2026-10-19)
            from sml_pretok import from_tokenizer_json as pretok
            try:
                self._pret = pretok(tj)
            except SystemExit as exc:
                print(f"tokenizer: {exc}; the engine cuts it the older way", flush=True)
        if model.get("type") == "Unigram" or isinstance(model.get("vocab"), list):
            # Helium, XGLM: a score per piece and no merges; the engine cuts by the
            # best-scoring path (sml_unigram.py, sml-vocab.c), 2026-10-14
            from sml_unigram import from_tokenizer_json
            tokens, self._uni = from_tokenizer_json(tj)
            n = len(tokens)
        else:
            vocab = model.get("vocab", {})
            n = max(vocab.values()) + 1 if vocab else 0
            for a in tj.get("added_tokens", []):
                n = max(n, a["id"] + 1)
            tokens = [""] * n
            for tok, tid in vocab.items():
                tokens[tid] = tok
            for a in tj.get("added_tokens", []):
                tokens[a["id"]] = a["content"]
        rows = self.shape("token_embd.weight")[0] if "token_embd.weight" in self.t else 0
        past = [a for a in tj.get("added_tokens", []) if a["id"] >= rows] if rows and n > rows else []
        if past and len(past) == n - rows and all(a.get("special") for a in past):
            # special placeholders past the embedding that a text model never
            # reads (Gemma 3's <image_soft_token>): dropped, so the vocabulary
            # gate measures the tokens that can occur
            tokens = tokens[:rows]
        merges = [" ".join(m) if isinstance(m, list) else m
                  for m in model.get("merges", [])]

        bos = eos = None
        cfgp = os.path.join(self.path, "tokenizer_config.json")
        if os.path.exists(cfgp):
            with open(cfgp, encoding="utf-8") as f:
                tc = json.load(f)
            lookup = {t: i for i, t in enumerate(tokens)}
            b, e = tc.get("bos_token"), tc.get("eos_token")
            b = b.get("content") if isinstance(b, dict) else b
            e = e.get("content") if isinstance(e, dict) else e
            bos, eos = lookup.get(b), lookup.get(e)

        self._vc = (tokens, merges, bos, eos)
        return self._vc


def _pb_fields(buf):
    """(field number, wire type, value) of one protobuf message, flat."""
    i, n = 0, len(buf)

    def varint():
        nonlocal i
        v = shift = 0
        while True:
            b = buf[i]
            i += 1
            v |= (b & 0x7F) << shift
            if b < 0x80:
                return v
            shift += 7
    while i < n:
        key = varint()
        f, wt = key >> 3, key & 7
        if wt == 0:
            yield f, wt, varint()
        elif wt == 1:
            yield f, wt, buf[i:i + 8]
            i += 8
        elif wt == 2:
            ln = varint()
            yield f, wt, buf[i:i + ln]
            i += ln
        elif wt == 5:
            yield f, wt, buf[i:i + 4]
            i += 4
        else:
            raise SystemExit(f"tokenizer.model: unreadable protobuf (wire type {wt})")


def _gpt2_bytes():
    """GPT-2's byte -> printable code point map (the byte-level alphabet)."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    out, n = {}, 0
    for b in range(256):
        if b in keep:
            out[b] = chr(b)
        else:
            out[b] = chr(256 + n)
            n += 1
    return out


QWEN_PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
            r"\s*[\r\n]+|\s+(?!\S)|\s+")


def tiktoken_vocab(path, fname):
    """((tokens, merges, bos, eos), pipeline) from a tiktoken rank file (Qwen v1's qwen.tiktoken):
    each line is a base64 token and its rank. Tokens become GPT-2 byte-level strings and the
    merges are rebuilt as transformers' TikTokenConverter does (every split of a token into two
    lower-ranked tokens, ordered by the token's rank). tiktoken takes a piece that is itself a
    token whole, which is BPE with ignore_merges. The specials follow the ranks, in the order
    the model's tokenization_*.py lists them; its PAT_STR is the split."""
    import base64
    ranks = {}
    with open(os.path.join(path, fname), "rb") as f:
        for line in f:
            if line.strip():
                tok, r = line.split()
                ranks[base64.b64decode(tok)] = int(r)
    n = max(ranks.values()) + 1
    bmap = _gpt2_bytes()
    show = lambda b: "".join(bmap[x] for x in b)
    tokens = [""] * n
    for b, r in ranks.items():
        tokens[r] = show(b)
    merges = []
    for b, r in ranks.items():
        if len(b) < 2:
            continue
        local = [(ranks[b[:k]], ranks[b[k:]], r, b[:k], b[k:]) for k in range(1, len(b))
                 if b[:k] in ranks and b[k:] in ranks]
        merges += sorted(local)
    merges = [show(a) + " " + show(c) for _, _, _, a, c in sorted(merges, key=lambda m: m[2])]
    # the model's own code, read and never run: its split pattern and its specials
    # (Qwen v1's layout when there is no code beside the ranks)
    pat, specials = QWEN_PAT, ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + [f"<|extra_{i}|>" for i in range(205)]
    code = [f for f in os.listdir(path) if f.startswith("tokenization_") and f.endswith(".py")]
    src = ""
    if code:
        with open(os.path.join(path, code[0]), encoding="utf-8") as f:
            src = f.read()
        pat = _code_pattern(src) or pat
        m = re.search(r"EXTRAS\s*=\s*tuple\(\(f\"<\|extra_\{i\}\|>\" for i in range\((\d+)\)\)\)", src)
        if m:
            specials = specials[:3] + [f"<|extra_{i}|>" for i in range(int(m.group(1)))]
    tc = {}
    if os.path.exists(os.path.join(path, "tokenizer_config.json")):
        with open(os.path.join(path, "tokenizer_config.json"), encoding="utf-8") as f:
            tc = json.load(f)
    m = re.search(r"num_reserved_special_tokens\s*=\s*(\d+)", src)
    if m:
        # Kimi / Moonlight: a block of reserved ids after the ranks, named by
        # tokenizer_config's added_tokens_decoder, <|reserved_token_N|> otherwise
        named = {int(i): (a["content"] if isinstance(a, dict) else str(a))
                 for i, a in (tc.get("added_tokens_decoder") or {}).items()}
        specials = [named.get(i, f"<|reserved_token_{i}|>") for i in range(n, n + int(m.group(1)))]
    start = n
    tokens += specials
    added = list(range(start, start + len(specials)))
    from sml_pretok import for_tiktoken
    pret = for_tiktoken(pat)
    pret["added"] = added
    ids = {t: i for i, t in enumerate(tokens)}

    def tok(k):
        v = tc.get(k)
        return v.get("content") if isinstance(v, dict) else v
    bos = ids.get(tok("bos_token")) if tok("bos_token") else None
    eos = ids.get(tok("eos_token")) if tok("eos_token") else start   # Qwen v1: <|endoftext|>
    return (tokens, merges, bos, eos), pret


def _code_pattern(src):
    """A tokenizer's split pattern from its Python source, by its syntax tree: PAT_STR / pat_str
    as a string, or "|".join([...strings...]). Nothing in the file is executed."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("PAT_STR", "pat_str")
                                                for t in node.targets):
            v = node.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return v.value
            if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "join"
                    and isinstance(v.func.value, ast.Constant) and len(v.args) == 1
                    and isinstance(v.args[0], (ast.List, ast.Tuple))
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in v.args[0].elts)):
                return v.func.value.value.join(e.value for e in v.args[0].elts)
    return None


def gpt2_files_vocab(path):
    """(tokens, merges, bos, eos) from vocab.json + merges.txt, with the added
    tokens tokenizer_config.json declares."""
    with open(os.path.join(path, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    tc = {}
    if os.path.exists(os.path.join(path, "tokenizer_config.json")):
        with open(os.path.join(path, "tokenizer_config.json"), encoding="utf-8") as f:
            tc = json.load(f)
    for i, a in (tc.get("added_tokens_decoder") or {}).items():
        vocab.setdefault(a["content"] if isinstance(a, dict) else str(a), int(i))
    n = max(vocab.values()) + 1
    tokens = [""] * n
    for t, i in vocab.items():
        tokens[i] = t
    with open(os.path.join(path, "merges.txt"), encoding="utf-8") as f:
        merges = [ln.rstrip("\r\n") for ln in f if ln.strip() and not ln.startswith("#version")]

    def tok(key):
        v = tc.get(key)
        v = v.get("content") if isinstance(v, dict) else v
        return vocab.get(v) if isinstance(v, str) else None
    return tokens, merges, tok("bos_token"), tok("eos_token")


def spm_vocab(path):
    """(tokens, merges, bos, eos) from a SentencePiece tokenizer.model, with no
    sentencepiece or transformers installed. Pieces are ModelProto field 1
    (piece = 1, score = 2, type = 3); the trainer's model type is field 2.3.
    Merges are built as transformers' LlamaConverter builds them: every split
    of a piece whose halves are both pieces, ranked by the piece's score (ties
    in id order) -- for a BPE model that IS its merge order. A Unigram model has no merge
    order: it is read with its scores and normalizer for the Viterbi cut (sml_unigram.py).
    Returns (tokens, merges, bos, eos, unigram info or None)."""
    with open(os.path.join(path, "tokenizer.model"), "rb") as f:
        buf = f.read()
    tokens, scores, types, kind = [], [], [], 1     # proto2 default model_type is UNIGRAM
    norm_spec, trainer = {}, {}
    for fno, wt, v in _pb_fields(buf):
        if fno == 1 and wt == 2:
            piece, score, typ = "", 0.0, 1
            for f2, w2, v2 in _pb_fields(v):
                if f2 == 1 and w2 == 2:
                    piece = v2.decode("utf-8", errors="replace")
                elif f2 == 2 and w2 == 5:
                    score = struct.unpack("<f", v2)[0]
                elif f2 == 3 and w2 == 0:
                    typ = v2
            tokens.append(piece)
            scores.append(score)
            types.append(typ)
        elif fno == 2 and wt == 2:              # TrainerSpec
            for f2, w2, v2 in _pb_fields(v):
                if f2 == 3 and w2 == 0:
                    kind = v2
                elif f2 == 35 and w2 == 0:
                    trainer["byte_fallback"] = bool(v2)
        elif fno == 3 and wt == 2:              # NormalizerSpec
            for f2, w2, v2 in _pb_fields(v):
                if f2 == 1 and w2 == 2:
                    norm_spec["name"] = v2.decode("utf-8", errors="replace")
                elif f2 == 2 and w2 == 2:
                    norm_spec["precompiled_charsmap"] = bytes(v2)
                elif f2 == 3 and w2 == 0:
                    norm_spec["add_dummy_prefix"] = bool(v2)
                elif f2 == 4 and w2 == 0:
                    norm_spec["remove_extra_whitespaces"] = bool(v2)
    uni = None
    if kind == 1:
        from sml_unigram import from_spm
        uni = from_spm(list(zip(tokens, scores, types)), norm_spec, trainer)
        merges = []
    elif kind != 2:
        raise SystemExit("tokenizer.model is a SentencePiece "
                         + {3: "word", 4: "char"}.get(kind, f"type-{kind}")
                         + " model; the engine follows BPE and Unigram ones")
    ids = {t: i for i, t in enumerate(tokens)}
    if kind == 2:
        merges = []
        for t, mid in ids.items():
            for k in range(1, len(t)):
                a, b = t[:k], t[k:]
                if a in ids and b in ids:
                    merges.append((-scores[mid], mid, ids[a], ids[b], a + " " + b))
        merges = [m[4] for m in sorted(merges)]
    bos = eos = None
    cfgp = os.path.join(path, "tokenizer_config.json")
    tc = {}
    if os.path.exists(cfgp):
        with open(cfgp, encoding="utf-8") as f:
            tc = json.load(f)

    def tok(key, default):
        v = tc.get(key, default)
        return v.get("content") if isinstance(v, dict) else v
    bos, eos = ids.get(tok("bos_token", "<s>")), ids.get(tok("eos_token", "</s>"))
    SPM_OPTS[os.path.abspath(path)] = (norm_spec, trainer)
    return tokens, merges, bos, eos, uni


SPM_OPTS = {}     # a tokenizer.model's normalizer and trainer options, by folder (spm_vocab)


# Behavioural differences between architectures, read from config.json.
# Structural ones (QKV bias, tied head, Q/K norm, experts) are NOT here: they
# are visible in the tensor table, so deriving them there keeps one source of
# truth instead of a per-architecture declaration to maintain.
_LAYERNORM_ARCHS = {"gpt2", "gptj", "gpt_neox", "falcon", "mpt", "starcoder", "phimoe",
                    "gpt_bigcode", "gpt_neo", "olmo", "jais2",
                    "opt", "biogpt", "xglm", "codegen", "dbrx",
                    "bloom", "phi", "stablelm", "persimmon",
                    # Cohere's LayerNorm has no bias; StarCoder2's has one
                    "cohere", "cohere2", "starcoder2",
                    # LayerNorm1P: stored (w - 1), applied as (w + 1)
                    "nemotron",
                    # Orion and CodeShell: nn.LayerNorm everywhere (found by
                    # sml_compat.py: their checkpoints carry norm biases, which
                    # RMSNorm has not)
                    "orion", "codeshell"}
# ONLY GPT-J interleaves. This set previously held gpt_neox, falcon, stablelm,
# phi and persimmon -- all wrong, and wrong for one specific reason worth
# recording: the reference engine calls the HALVED convention "NEOX rope",
# because that is the checkpoint family it first appeared in. Reading their
# source and carrying the NAME across gives exactly the inverted mapping. Every
# one of these architectures uses rotate_half in its own reference
# implementation; checked against transformers 4.57.6 rather than reasoned about.
# Reverse-engineering a mature codebase imports its history along with its
# knowledge, and its names are historical, not descriptive.
_INTERLEAVED_ROPE = {"gptj", "codegen",
                     # 2026-09-28, read from transformers 4.57.6 itself: these
                     # define rotate_half on x[..., ::2] / x[..., 1::2]. Each was
                     # marked ok and ran the halved convention -- 30-128% logit
                     # error on a random-weight head of each family.
                     "cohere", "cohere2", "ernie4_5", "ernie4_5_moe", "glm", "glm4",
                     "helium"}
# Attention and FFN in parallel off ONE LayerNorm, with no config key saying so
# (GPT-NeoX says use_parallel_residual; these just are).
_PARALLEL_SHARED = {"phi", "gptj", "cohere", "cohere2", "codegen"}
# fairseq-lineage learned positions: row = position + 2 (the file drops the two)
_POS_OFFSET2 = {"opt", "biogpt"}
# Q/K norm applied AFTER the rotary (HunYuan): it divides out any factor on cos/sin
_QKNORM_AFTER_ROPE = {"hunyuan_v1_dense", "hunyuan_v1_moe"}


def _bf16(v):
    """v rounded to bfloat16 (nearest, ties to even), as a Python float"""
    u = np.array([v], dtype=np.float32).view(np.uint32)
    u = ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)).astype(np.uint32)
    return float(u.view(np.float32)[0])


def _softplus(v):
    return v if v > 20.0 else math.log1p(math.exp(v))


def _xglm_positions(n_pos, dim):
    """XGLMSinusoidalPositionalEmbedding.get_embedding(n_pos + 2, dim)[2:], in
    float32 as torch computes it."""
    half = dim // 2
    step = math.log(10000) / (half - 1)
    inv = np.exp(np.arange(half, dtype=np.float32) * np.float32(-step)).astype(np.float32)
    pos = np.arange(n_pos + 2, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([np.sin(pos), np.cos(pos)], 1).astype(np.float32)
    if dim % 2:
        emb = np.concatenate([emb, np.zeros((n_pos + 2, 1), np.float32)], 1)
    return np.ascontiguousarray(emb[2:])
# RMSNorm applied as x * (1 + w): the stored weight is w, so the file gets w + 1
_NORM_PLUS1 = {"gemma", "gemma2", "gemma3", "gemma3_text", "vaultgemma", "nemotron", "qwen3_next",
               "qwen3_5_text", "qwen3_5_moe_text"}
# attention output gated by a second half of q_proj, per head [q | gate]
_ATTN_GATE = {"qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"}
# Embeddings times sqrt(hidden_size)
_EMB_SQRT = {"gemma", "gemma2", "gemma3", "gemma3_text", "vaultgemma"}
# post_attention_layernorm is the attention OUTPUT norm here, not the pre-FFN norm
# (EXAONE-MoE is pre-norm, unlike EXAONE 4: its post_attention_layernorm feeds the FFN)
_POSTNORM_ARCHS = {"olmo2", "olmo3", "flex_olmo", "exaone4", "gemma2", "gemma3", "gemma3_text", "afmoe"}
# Projections stored [in, out] (Conv1D). gpt_bigcode is nn.Linear: NOT here.
_CONV1D_ARCHS = {"gpt2"}
# Vision-language checkpoints whose text side is a plain decoder
_VLM_ARCHS = {"idefics3", "smolvlm"}
# Mamba-1 families: mixer names shared with Mamba-2, different maths
_MAMBA1 = {"mamba", "falcon_mamba", "jamba"}
# Multimodal wrappers read for their TEXT model only (no image path yet): the
# language model's names mapped onto a plain decoder's, the vision tower and
# projector set aside by name -- so the unplaced-tensor check still guards the
# rest. Mistral Small 3.x, Devstral and Magistral ship as mistral3 around a
# plain mistral text model. transformers 4.57 names them model.language_model.*;
# older checkpoints language_model.model.*.
_MM_TEXT = {
    # Qwen 3.5: the multi-token-prediction head (mtp.*) is set aside with the
    # vision tower -- ordinary generation does not run it
    "qwen3_5": {
        "rename": (("model.language_model.", "model."),),
        "vision": ("model.visual.", "mtp."),
    },
    "qwen3_5_moe": {
        "rename": (("model.language_model.", "model."),),
        "vision": ("model.visual.", "mtp."),
    },
    # Chameleon: the image tokenizer (a VQ-GAN) is the image path; the text path is a Llama
    "chameleon": {"rename": (), "vision": ("model.vqmodel.",)},
    "gemma3": {
        "rename": (("model.language_model.", "model."), ("language_model.model.", "model."),
                   ("language_model.lm_head", "lm_head")),
        "vision": ("model.vision_tower.", "model.multi_modal_projector.",
                   "vision_tower.", "multi_modal_projector."),
    },
    "mistral3": {
        "rename": (("model.language_model.", "model."), ("language_model.model.", "model."),
                   ("language_model.lm_head", "lm_head")),
        "vision": ("model.vision_tower.", "model.multi_modal_projector.",
                   "vision_tower.", "multi_modal_projector."),
    },
}
# config key -> the names other families use for it
_CFG_ALIASES = {
    "hidden_size": ("n_embd", "d_model", "n_embed"),
    # XGLM says attention_heads and ffn_dim (fairseq names); read as one head before
    # 2026-10-14, which built and ran -- as a different, broken model
    "num_attention_heads": ("n_head", "n_heads", "num_heads", "attention_heads", "decoder_attention_heads"),
    "num_hidden_layers": ("n_layer", "n_layers", "num_layers", "decoder_layers"),
    "intermediate_size": ("n_inner", "ffn_dim", "decoder_ffn_dim"),
    "max_position_embeddings": ("n_positions", "n_ctx", "max_seq_len"),
    "hidden_act": ("activation_function", "hidden_activation", "activation", "mlp_hidden_act"),
    "rms_norm_eps": ("layer_norm_eps", "layer_norm_epsilon", "norm_epsilon", "norm_eps"),
}


def _num(v):
    """A config number; transformers 5 writes infinity as {"__float__": "Infinity"}."""
    if isinstance(v, dict) and "__float__" in v:
        return float(v["__float__"])
    return v


def arch_flags(cfg):
    """The behaviour flags read from config (_arch_flags). YaRN's and LongRoPE's
    attention factor is not a flag: it is folded into the weights (HfWeights.get).
    A saved auto-fit for this model_type (sml_autofit.py), then SML_FLAGS_OVERRIDE
    (the fit's own search), replace what the name lists say."""
    f = _arch_flags(cfg)
    from sml_archs import fit_for
    fit = fit_for(cfg.get("model_type", ""))
    if fit and fit.get("pass"):
        f.update(fit.get("flags") or {})
    f.update(json.loads(os.environ.get("SML_FLAGS_OVERRIDE") or "{}"))
    return f


def _is_mla(cfg):
    """Multi-head latent attention: DeepSeek V2/V3 and what reuses them (Kimi K2)."""
    return bool(cfg.get("kv_lora_rank")) and cfg.get("qk_rope_head_dim") is not None


def _mla_scale(cfg):
    """DeepSeek's attention scale: 1/sqrt(nope + rope dims), times YaRN's mscale squared when
    the rotary is YaRN with mscale_all_dim (yarn_apply_mscale)."""
    import math
    s = (int(cfg["qk_nope_head_dim"]) + int(cfg["qk_rope_head_dim"])) ** -0.5
    rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    if str(rs.get("rope_type", rs.get("type", ""))) == "yarn" and rs.get("mscale_all_dim"):
        f = float(rs.get("factor") or 1.0)
        m = 0.1 * float(rs["mscale_all_dim"]) * math.log(f) + 1.0 if f > 1 else 1.0
        s *= m * m
    return s


def _arch_flags(cfg):
    a = cfg.get("model_type", "")
    mla = _is_mla(cfg)
    n_head = cfg.get("num_attention_heads", 1)
    # Mistral writes "head_dim": null when it is the default; .get's fallback
    # only covers a missing key
    hd = cfg.get("head_dim") or cfg.get("hidden_size", 0) // max(n_head, 1)
    return {
        "norm_type": 1 if a in _LAYERNORM_ARCHS else 0,
        # DeepSeek V2/V3 rotate interleaved pairs (rope_interleave, true by default)
        "rope_type": 1 if a in _INTERLEAVED_ROPE or (mla and a != "minicpm3" and cfg.get("rope_interleave", True)) else 0,
        # Gemma scales attention by a fixed value rather than 1/sqrt(head_dim)
        # GPT-Neo does not scale attention scores at all
        "attn_scale": _mla_scale(cfg) if mla else 1.0 if a == "gpt_neo" else
                      float(cfg["attention_multiplier"]) if cfg.get("attention_multiplier")
                      else float(cfg.get("query_pre_attn_scalar", 0.0) and
                                 1.0 / (cfg["query_pre_attn_scalar"] ** 0.5) or 0.0),
        # Granite scales embeddings, each residual branch and the logits by
        # config constants. Ignoring them still runs -- as a different model.
        # MiniCPM: Granite's three scales under its own names (modeling_minicpm.py):
        # embeddings x scale_emb, each branch x scale_depth / sqrt(layers), and
        # the final hidden state divided by hidden_size / dim_model_base
        "emb_scale": float(cfg.get("embedding_multiplier") or 0.0)
                     or (float(cfg["scale_emb"]) if a in ("minicpm", "minicpm3") and cfg.get("scale_emb") else 0.0)
                     or (float(cfg.get("hidden_size", 0)) ** 0.5 if a in _EMB_SQRT else 0.0)
                     or (float(cfg.get("hidden_size", 0)) ** 0.5
                         if a in ("biogpt", "xglm") and cfg.get("scale_embedding", True) else 0.0),
        "res_scale": float(cfg.get("residual_multiplier") or 0.0)
                     or (float(cfg["scale_depth"]) / float(cfg["num_hidden_layers"]) ** 0.5
                         if a in ("minicpm", "minicpm3") and cfg.get("scale_depth") else 0.0),
        # the engine DIVIDES by logit_scale (Granite's logits_scaling); Cohere
        # states a multiplier
        "logit_scale": float(cfg.get("logits_scaling") or 0.0)
                       or (float(cfg["hidden_size"]) / float(cfg["dim_model_base"])
                           if a in ("minicpm", "minicpm3") and cfg.get("dim_model_base") else 0.0)
                       or (1.0 / float(cfg["logit_scale"]) if a in ("cohere", "cohere2")
                           and cfg.get("logit_scale") else 0.0)
                       or (1.0 / float(cfg["lm_head_multiplier"]) if a == "falcon_h1"
                           and float(cfg.get("lm_head_multiplier") or 1.0) != 1.0 else 0.0),
        # Mixture of experts. Granite and Mixtral renormalize the chosen
        # experts' weights; OLMoE and Qwen2-MoE say norm_topk_prob false.
        "n_expert_used": int(cfg.get("num_experts_per_tok") or cfg.get("moe_topk")
                             or cfg.get("moe_k") or 0),
        # ERNIE always renormalizes its chosen weights
        # ERNIE, MiniMax-M2 and AFMoE always renormalize their chosen weights
        "moe_norm": True if a in ("ernie4_5_moe", "minimax_m2", "afmoe") else False if a == "jamba"
                    else bool(cfg.get("norm_topk_prob", True)),
        # DeepSeek-V3-style: sigmoid scores, a routing scale, expert groups
        "moe_sigmoid": cfg.get("scoring_func") == "sigmoid"
                       or a in ("glm4_moe", "dots1", "exaone_moe", "minimax_m2", "afmoe", "lfm2_moe")
                       # DeepSeek V3 (Kimi K2) always scores by sigmoid; transformers 5 dropped the key
                       or (a in ("deepseek_v3", "kimi_k2") and cfg.get("scoring_func", "sigmoid") == "sigmoid")
                       or (a == "nemotron_h" and bool(cfg.get("n_routed_experts"))),
        "moe_scale": float(cfg.get("routed_scaling_factor") or cfg.get("route_scale") or 0.0),
        # PhiMoE's sparsemixer: each pick is the best remaining expert, weighted by a softmax
        # over the experts within 2 x jitter (relative) of it (modeling_phimoe.sparsemixer)
        "moe_sparsemixer": a == "phimoe",
        "moe_jitter": float(cfg.get("router_jitter_noise") or 0.01) if a == "phimoe" else 0.0,
        # DeepSeek V2's "greedy" ignores its groups; group_limited_greedy and noaux_tc use them
        "moe_groups": 0 if cfg.get("topk_method") == "greedy" else int(cfg.get("n_group") or 0),
        "moe_group_max": cfg.get("topk_method") == "group_limited_greedy",
        "moe_topk_group": int(cfg.get("topk_group") or 0),
        "logit_softcap": float(cfg.get("final_logit_softcapping") or 0.0),
        # Qwen2.5 declares sliding_window but sets use_sliding_window false --
        # taking the number at face value would apply a window the model was
        # never trained with.
        # Mistral, Mixtral, Phi-3 and StarCoder2 state only sliding_window and
        # apply it (Mistral-7B-v0.1: 4096); treating its absence as "off" ran
        # them with full attention (59% logit error past the window on a
        # Mistral-shaped head). Alternating per-layer windows are not this.
        "n_swa": (int(cfg.get("sliding_window") or 0)
                  if cfg.get("use_sliding_window", "use_sliding_window" not in cfg)
                  and not cfg.get("layer_types") and not cfg.get("sliding_window_pattern")
                  else 0),
        # latent attention's heads: its no-rotary and rotary dims together; only the latter rotate
        "head_dim": int(cfg["qk_nope_head_dim"]) + int(cfg["qk_rope_head_dim"]) if mla else int(hd),
        # GPT-NeoX/Phi/StableLM rotate only part of each head. Rotating all of
        # it instead is silently wrong, not an error.
        "n_rot": int(cfg["qk_rope_head_dim"]) if mla else int(cfg["rotary_dim"]) if cfg.get("rotary_dim")
                 else int(round(hd * float(cfg.get("partial_rotary_factor",
                                                   cfg.get("rotary_pct", 1.0)))))
                 if (cfg.get("partial_rotary_factor") or cfg.get("rotary_pct"))
                 else 0,
        # GPT-NeoX runs attention and FFN on the SAME input and adds both to the
        # residual, rather than feeding one into the other.
        "parallel_residual": bool(cfg.get("use_parallel_residual", False))
                             or a in _PARALLEL_SHARED
                             or (a == "falcon" and bool(cfg.get("parallel_attn", True))),
        "alibi_scaled": a == "falcon" and bool(cfg.get("alibi")),
        # HunYuan's query_layernorm/key_layernorm run on the ROTATED heads
        "qknorm_late": a in ("hunyuan_v1_dense", "hunyuan_v1_moe"),
        # Chameleon: a LayerNorm per head with its own weight and bias row ([heads, head_dim])
        "qknorm_heads": a == "chameleon",
        # gpt-oss's experts: clamp(gate, 7) * sigmoid(1.702 gate) * (clamp(up, -7, 7) + 1)
        "swiglu_oai": a == "gpt_oss",
        "mla": mla,
        "relu2": str(cfg.get("hidden_act", "")) == "relu2",
        "relu": str(cfg.get("hidden_act", "")) == "relu",
        # Mamba-2: groups of B/C, state size, the gated norm's group, dt clamp
        "ssm_groups": int(cfg.get("n_groups") or cfg.get("mamba_n_groups") or 0),
        "ssm_dstate": int(cfg.get("state_size") or cfg.get("mamba_d_state") or cfg.get("ssm_state_size") or 0),
        # Nemotron-H (and Zamba2) normalize the gated output per B/C group
        "ssm_norm_group": (int(cfg["mamba_num_heads"]) * int(cfg["mamba_head_dim"])) // int(cfg.get("n_groups") or 1)
                          if a == "nemotron_h" and cfg.get("mamba_num_heads") else 0,
        "ssm_bcdt_eps": float(cfg.get("mixer_rms_eps") or 0.0) if a == "falcon_mamba" else 0.0,
        "gdn_nk": int(cfg.get("linear_num_key_heads") or 0),
        "gdn_dk": int(cfg.get("linear_key_head_dim") or 0),
        "ssm_dt_lo": _num((cfg.get("time_step_limit") or (0.0, 0.0))[0]) or 0.0,
        "ssm_dt_hi": (lambda v: 0.0 if v in (None, float("inf")) else float(v))(
            _num((cfg.get("time_step_limit") or (0.0, None))[1])),
        # Gemma 2 caps attention logits; OLMo and MPT clamp Q/K/V
        "attn_softcap": float(cfg.get("attn_logit_softcapping") or 0.0),
        "clip_qkv": float(cfg.get("clip_qkv") or (cfg.get("attn_config") or {}).get("clip_qkv") or 0.0),
        # GELU FFN has no gate projection at all -- detectable from the tensor
        # table, so it needs no flag; recorded here only for the summary.
        "gelu_ffn": str(cfg.get("hidden_act", "silu")).startswith("gelu"),
        # Which GELU. HF's "gelu" (and "gelu_python") is the exact erf form;
        # "gelu_new", "gelu_fast" and "gelu_pytorch_tanh" are the tanh
        # approximation, as is BLOOM's, which names none. Pythia (GPT-NeoX) says
        # "gelu": the engine ran it as tanh until 2026-09-25.
        "gelu_erf": str(cfg.get("hidden_act", "")) in ("gelu", "gelu_python"),
        # ALiBi: a per-head linear distance penalty on attention scores in
        # place of any positional encoding. BLOOM always; MPT and Falcon only
        # when their config says so (Falcon-7B is rotary).
        "alibi": a == "bloom"
                 or (a == "mpt" and bool((cfg.get("attn_config") or {}).get("alibi")))
                 or (a == "falcon" and bool(cfg.get("alibi")))
                 # Baichuan 13B (1 and 2): ALiBi, no rotary -- the 7B rotates.
                 # Nothing in the config says which; the 40-layer 13B is the one
                 # (modeling_baichuan.py of each, read 2026-10-13)
                 or (a == "baichuan" and int(cfg.get("num_hidden_layers") or 0) == 40),
    }


def _layer_types(cfg):
    """layer_types for configs written before transformers listed them: the
    same rule each family's own modelling code applied by layer index.
    Without this a Gemma 3 checkpoint saved by transformers 4.5x ran every
    layer as global with the global rotary base (found 2026-10-13)."""
    if cfg.get("layer_types") or not cfg.get("sliding_window"):
        return
    a = cfg.get("model_type", "")
    n = int(cfg.get("num_hidden_layers") or 0)
    if a == "gemma2":                       # Gemma2DecoderLayer: even layers slide
        cfg["layer_types"] = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(n)]
    elif a in ("gemma3", "gemma3_text", "cohere2") and cfg.get("sliding_window_pattern"):
        pat = int(cfg["sliding_window_pattern"])  # every pat-th layer is global
        cfg["layer_types"] = ["full_attention" if (i + 1) % pat == 0 else "sliding_attention" for i in range(n)]


def layer_attn(cfg, n_layer):
    """Per-layer attention: (windows, rotary on/off, local rotary base) or Nones.

    Gemma 2 alternates sliding-window and global layers, Gemma 3 runs five
    local to one global and gives the local ones their own rotary base, Cohere2
    rotates only its sliding layers, EXAONE 4 drops the rotary on its global
    layers when it has sliding ones, SmolLM3 drops it on every fourth layer, and
    GPT-Neo alternates global and local. Read from transformers 4.57.6."""
    a = cfg.get("model_type", "")
    sw = cfg.get("sliding_window")
    win = rope = None
    local = None
    if a == "gpt_neo":
        types = cfg.get("attention_layers")
        if not types:
            types = [t for kinds, n in cfg.get("attention_types", []) for _ in range(n) for t in kinds]
        win = [int(cfg.get("window_size", 256)) if t == "local" else 0 for t in types[:n_layer]]
    elif cfg.get("layer_types") and sw:
        win = [int(sw) if t == "sliding_attention" else 0 for t in cfg["layer_types"][:n_layer]]
    if a in ("cohere2", "afmoe") and win:
        rope = [1 if w else 0 for w in win]     # rotary on the local (sliding) layers only
    if a in ("exaone4", "exaone_moe") and win and any(win):
        rope = [1 if w else 0 for w in win]
    if a in ("jamba", "nemotron_h"):
        rope = [0] * n_layer        # their attention layers carry no positions at all
    if cfg.get("position_embedding_type") == "nope":
        rope = [0] * n_layer        # Granite-4.0-H: no rotary anywhere
    if a == "smollm3" and cfg.get("no_rope_layers"):
        rope = [int(v) for v in cfg["no_rope_layers"][:n_layer]]
    if a in ("gemma3", "gemma3_text") and win and any(win) and cfg.get("rope_local_base_freq"):
        local = float(cfg["rope_local_base_freq"])
    if a == "olmo3" and win and any(win) and cfg.get("rope_scaling"):
        local = float(cfg.get("rope_theta", 10000.0))   # its sliding layers rotate unscaled
    if win is not None and not any(win):
        win = None
    if rope is not None and all(rope):
        rope = None
    return win, rope, local


def open_weights(path, header_only=False):
    """HF checkpoint directory.

    The lossy intake is gone deliberately, not just for the name. Converting from
    an already-quantized source is DOUBLE quantization, and Q4_K_M measures
    5.31% from the true fp16 weights -- above the 5% gate on its own. No amount
    of work downstream can recover that, so the lossy intake was a dead end and
    keeping it would only invite someone to walk down it again."""
    if not os.path.isdir(path):
        raise SystemExit(f"expected an HF checkpoint directory, got: {path}")
    return HfWeights(path, header_only)


if __name__ == "__main__":
    import sys
    W = open_weights(sys.argv[1])
    print(f"arch={W.arch if hasattr(W, 'arch') else '?'} tensors={len(W.t)}")
    a = W.field("general.architecture")
    a = a.decode() if isinstance(a, (bytes, bytearray)) else a
    for k in ("block_count", "embedding_length", "attention.head_count",
              "attention.head_count_kv", "feed_forward_length"):
        print(f"  {k} = {W.field(f'{a}.{k}')}")
    t = W.get("blk.0.ffn_gate.weight")
    print(f"  blk.0.ffn_gate.weight {t.shape} {t.dtype} "
          f"absmax={np.abs(t).max():.4f}")
