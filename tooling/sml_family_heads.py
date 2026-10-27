#!/usr/bin/env python3
"""One head per architecture family, run against transformers' own code for it.

    python tooling/sml_family_heads.py [--only llama,qwen3,...] [--keep DIR]
    python tooling/sml_family_heads.py --all      every head, small and big shape, 3 at a time;
                                                  unchanged results are reused (seconds)

The reference engine lists 100+ architectures. Most of them differ from one
another in a handful of behaviours -- norm type, rotary convention and width,
positional scheme, activation, gating, residual wiring, attention windows, Q/K
norms, expert routing. A family is the set of model types that share every one
of those; a family HEAD is one member run end to end. If the head computes what
transformers computes, every child with the same behaviour does too, and the
child-by-child details (tokenizers, chat templates, a checkpoint's odd tensor)
are left to the checkpoints people actually bring.

Each head here is a small random-weight model built FROM TRANSFORMERS' OWN
MODELLING CODE for that model_type -- not a checkpoint, so no download and no
remote code -- with sizes this engine's 32-wide groups fit (hidden 256, four
heads of 64, two layers). The weights are drawn large enough that every branch
matters (unit-scale activations, norm weights and biases off their defaults), so
a wrong rotary convention, a missing norm, a wrong activation or a wrong window
moves the logits by tens of percent, while 6-bit quantization moves them by a
few. Sheep builds the checkpoint like any other (every tensor at 6 bits), runs
it on the CPU -- stepped one token at a time, and the batched prompt path -- and
the last-position logits are compared with transformers' in fp32.

The families whose real checkpoints were graded earlier (Llama-3.2-1B, Qwen2.5,
Qwen3, pythia, GPT-2, BLOOM, OLMo 2, Granite MoE) set the pass band: a new
family passes when it lands where those do.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CLI = os.path.join(ROOT, "sml", "sml-cli" + (".exe" if os.name == "nt" else ""))
BUILD = os.path.join(HERE, "sml_quantize.py")
TOK_SRC = os.environ.get("SML_TOK_SRC", r"C:\Users\earne\models\gpt2")
VOCAB = 50257

# Every size alias a config class might read; a config ignores the ones it
# does not know (they become plain attributes).
BASE = dict(
    vocab_size=VOCAB, hidden_size=256, n_embd=256, d_model=256,
    num_hidden_layers=2, n_layer=2, n_layers=2,
    num_attention_heads=4, n_head=4, n_heads=4,
    num_key_value_heads=2, head_dim=64,
    intermediate_size=512, n_inner=512, ffn_hidden_size=512,
    max_position_embeddings=256, n_positions=256,
    bos_token_id=50256, eos_token_id=50256, pad_token_id=None,
    tie_word_embeddings=False, use_cache=False,
    attention_dropout=0.0, hidden_dropout=0.0, resid_pdrop=0.0, embd_pdrop=0.0,
    attn_pdrop=0.0,
)

# (label, model_type, overrides, roster key, what the head represents)
HEADS = [
    # ---- verified with real checkpoints: these set the pass band -------------
    ("llama", "llama", {}, "llama", "RMSNorm, halved RoPE, SwiGLU, GQA"),
    ("qwen2", "qwen2", {}, "qwen2", "llama + QKV bias"),
    ("qwen3", "qwen3", {}, "qwen3", "llama + Q/K RMSNorm"),
    ("gpt_neox", "gpt_neox", {"rotary_pct": 0.25, "use_parallel_residual": True,
                              "hidden_act": "gelu"}, "gpt_neox",
     "LayerNorm, partial rotary, parallel residual, GELU"),
    ("gpt2", "gpt2", {"n_inner": 1024}, "gpt2", "learned positions, Conv1D, GELU"),
    ("bloom", "bloom", {"n_head": 4}, "bloom", "ALiBi, LayerNorm, embedding LayerNorm"),
    ("olmo2", "olmo2", {}, "olmo2", "post-norms on each branch, Q/K norm"),
    ("granitemoe", "granitemoe", {"num_local_experts": 4, "num_experts_per_tok": 2,
                                  "embedding_multiplier": 3.0, "residual_multiplier": 0.5,
                                  "logits_scaling": 2.0, "attention_multiplier": 0.1},
     "granitemoe", "top-k experts, Granite scales"),
    # ---- family heads of the rows marked "ok" from config alone ---------------
    ("mistral", "mistral", {}, "llama", "llama naming, full attention"),
    # ---- rotary scalings: frequencies built at build time, attention factor in the weights
    ("llama-yarn", "llama", {"rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                                              "original_max_position_embeddings": 64}}, "llama",
     "YaRN on every layer (Qwen3 long context, DeepSeek-R1 distills)"),
    ("olmo3-yarn", "olmo3", {"sliding_window": 8, "layer_types": ["sliding_attention", "full_attention"],
                             "rope_scaling": {"rope_type": "yarn", "factor": 8.0,
                                              "original_max_position_embeddings": 32,
                                              "attention_factor": 1.2079441541679836}}, "olmo3",
     "YaRN on the global layers only, as OLMo 3 ships, through its Q/K norm"),
    ("phi3-longrope", "phi3", {"partial_rotary_factor": 0.75, "max_position_embeddings": 1024,
                               "original_max_position_embeddings": 64,
                               "rope_scaling": {"rope_type": "longrope",
                                                "short_factor": [1.0 + 0.1 * i for i in range(24)],
                                                "long_factor": [2.0 + 0.5 * i for i in range(24)]}}, "phi3",
     "LongRoPE on a 75% rotary (Phi-4-mini, Phi-3-128k)"),
    ("mistral-swa", "mistral", {"sliding_window": 8}, "llama",
     "llama + uniform sliding window (8 < the 20-token prompt)"),
    ("granite", "granite", {"embedding_multiplier": 3.0, "residual_multiplier": 0.5,
                            "logits_scaling": 2.0, "attention_multiplier": 0.1},
     "granite", "dense Granite scales"),
    ("stablelm", "stablelm", {"partial_rotary_factor": 0.25, "use_qkv_bias": True},
     "stablelm", "LayerNorm, 25% rotary, QKV bias, SwiGLU"),
    ("phi", "phi", {"partial_rotary_factor": 0.5, "hidden_act": "gelu_new"}, "phi2",
     "one shared LayerNorm, parallel attention+MLP, partial rotary"),
    ("phi3", "phi3", {}, "phi3", "fused qkv and gate_up, RMSNorm"),
    ("starcoder2", "starcoder2", {"use_bias": True, "hidden_act": "gelu_pytorch_tanh",
                                  "sliding_window": None}, "starcoder2",
     "LayerNorm with bias, ungated GELU, GQA, RoPE"),
    ("cohere", "cohere", {"logit_scale": 0.5, "use_qk_norm": False}, "command_r",
     "LayerNorm without bias, parallel residual, logit scale"),
    ("gemma", "gemma", {"hidden_activation": "gelu_pytorch_tanh"}, "gemma",
     "(1+w) RMSNorm, GeGLU, sqrt(d) embedding scale"),
    ("olmo", "olmo", {"clip_qkv": None}, "olmo", "non-parametric LayerNorm, SwiGLU"),
    ("ernie4_5", "ernie4_5", {"use_bias": False}, "ernie4_5", "llama-shaped"),
    ("seed_oss", "seed_oss", {}, "seed_oss", "llama-shaped with biases"),
    ("hunyuan_v1_dense", "hunyuan_v1_dense", {}, "hunyuan_dense", "Q/K norm, llama-shaped"),
    ("exaone4", "exaone4", {"sliding_window": None, "layer_types": ["full_attention", "full_attention"]},
     "exaone4", "post-norms and Q/K norm (OLMo-2-like)"),
    ("exaone4-hybrid", "exaone4", {"sliding_window": 8, "layer_types": ["sliding_attention", "full_attention"]},
     "exaone4", "sliding layers rotated, global layers NoPE (EXAONE-4.0-32B style)"),
    ("glm", "glm", {"partial_rotary_factor": 0.5}, "chatglm", "GLM-4 dense"),
    ("helium", "helium", {}, None, "llama-shaped (Kyutai)"),
    ("arcee", "arcee", {}, "arcee", "llama with an UNGATED ReLU^2 MLP"),
    ("smollm3", "smollm3", {"num_hidden_layers": 4}, "smollm3",
     "llama with NoPE on every 4th layer (4 layers: the last has none)"),
    ("persimmon", "persimmon", {"partial_rotary_factor": 0.5, "qk_layernorm": True},
     "persimmon", "LayerNorm Q/K, ReLU^2, partial rotary"),
    ("nemotron", "nemotron", {"partial_rotary_factor": 0.5}, "nemotron",
     "(w+1) LayerNorm, ReLU^2 ungated MLP"),
    ("gptj", "gptj", {"rotary_dim": 32, "n_embd": 256, "n_head": 4}, "gptj",
     "interleaved rotary, parallel attention+MLP"),
    ("falcon", "falcon", {"new_decoder_architecture": False, "multi_query": True,
                          "parallel_attn": True, "alibi": False, "bias": False},
     "falcon", "RoPE falcon-7B style: MQA, parallel attention, shared LayerNorm"),
    ("falcon-alibi", "falcon", {"new_decoder_architecture": False, "multi_query": False,
                                "parallel_attn": False, "alibi": True, "bias": True},
     "falcon", "ALiBi falcon-rw style"),
    ("mpt", "mpt", {"d_model": 256, "n_heads": 4, "n_layers": 2, "expansion_ratio": 2,
                    "attn_config": {"alibi": True}}, "mpt", "ALiBi, LayerNorm without bias"),
    ("gpt_bigcode", "gpt_bigcode", {"multi_query": True, "n_inner": 1024}, "gpt_bigcode",
     "learned positions, fused q|k|v with one shared K/V head"),
    ("gpt_neo", "gpt_neo", {"num_layers": 2, "attention_types": [[["global", "local"], 1]],
                            "window_size": 8, "intermediate_size": 1024}, None,
     "learned positions, alternating local and global attention"),
    ("mixtral", "mixtral", {"num_local_experts": 4, "num_experts_per_tok": 2}, "mixtral",
     "8x7B-style experts, renormalized top-2"),
    ("qwen2_moe", "qwen2_moe", {"num_experts": 4, "num_experts_per_tok": 2,
                                "moe_intermediate_size": 256, "shared_expert_intermediate_size": 512,
                                "norm_topk_prob": False, "decoder_sparse_step": 1}, "qwen2moe",
     "experts + gated shared expert, softmax weights not renormalized"),
    ("olmoe", "olmoe", {"num_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": False},
     "olmoe", "experts, Q/K norm, not renormalized"),
    ("gemma2", "gemma2", {"sliding_window": 8, "attn_logit_softcapping": 50.0,
                          "final_logit_softcapping": 30.0, "query_pre_attn_scalar": 64,
                          "hidden_activation": "gelu_pytorch_tanh"}, "gemma2",
     "alternating window/global, attention softcap"),
    ("gemma3_text", "gemma3_text", {"sliding_window": 8, "query_pre_attn_scalar": 64,
                                    "layer_types": ["sliding_attention", "full_attention"],
                                    "rope_local_base_freq": 10000.0, "rope_theta": 1000000.0,
                                    "hidden_activation": "gelu_pytorch_tanh"}, "gemma3",
     "alternating window/global with two rope bases, Q/K norm"),
    ("cohere2", "cohere2", {"sliding_window": 8, "sliding_window_pattern": 2,
                            "layer_types": ["sliding_attention", "full_attention"]},
     "cohere2", "alternating window/global"),
    ("mistral3", "mistral3", {"text_config": dict(
        model_type="mistral", vocab_size=VOCAB, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        max_position_embeddings=256, tie_word_embeddings=False),
        "vision_config": dict(model_type="pixtral", hidden_size=64, intermediate_size=128,
                              num_hidden_layers=1, num_attention_heads=2, image_size=32,
                              patch_size=16, head_dim=32)},
     "mistral3", "Mistral Small 3.x / Devstral: a mistral text model in a vision wrapper (text path)"),
    # ---- the wider sweep: every other decoder transformers carries ----------
    ("qwen3_moe", "qwen3_moe", {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 256,
                                "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": []},
     "qwen3moe", "experts + Q/K norm"),
    ("glm4", "glm4", {"partial_rotary_factor": 0.5}, "glm4", "GLM-4-0414: post-norms, interleaved partial rotary"),
    ("olmo3", "olmo3", {"sliding_window": 8, "layer_types": ["sliding_attention", "full_attention"]},
     None, "OLMo-2 shape with alternating windows"),
    ("ministral", "ministral", {"sliding_window": 8, "layer_types": ["sliding_attention", "full_attention"]},
     None, "llama with alternating windows"),
    ("vaultgemma", "vaultgemma", {"hidden_activation": "gelu_pytorch_tanh", "query_pre_attn_scalar": 64},
     None, "Gemma shape"),
    ("granitemoeshared", "granitemoeshared", {"num_local_experts": 4, "num_experts_per_tok": 2,
                                              "shared_intermediate_size": 512}, None, "Granite MoE + shared expert"),
    ("flex_olmo", "flex_olmo", {"num_experts": 4, "num_experts_per_tok": 2}, None, "OLMo experts"),
    ("ernie4_5_moe", "ernie4_5_moe", {"moe_num_experts": 4, "moe_k": 2, "moe_intermediate_size": 256,
                                      "moe_layer_start_index": 0, "moe_num_shared_experts": 1}, "ernie4_5_moe",
     "softmax, bias-corrected choice, renormalized, shared expert"),
    ("glm4_moe", "glm4_moe", {"n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 256,
                              "n_shared_experts": 1, "first_k_dense_replace": 1, "routed_scaling_factor": 2.5,
                              "n_group": 2, "topk_group": 1, "partial_rotary_factor": 0.5},
     "glm4_moe", "sigmoid routing, correction bias, 2 groups of 4, scale 2.5, shared expert, a dense first layer"),
    ("dots1", "dots1", {"n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 256,
                        "n_shared_experts": 1, "first_k_dense_replace": 1, "routed_scaling_factor": 2.0,
                        "n_group": 1, "topk_group": 1}, "dots1", "V3-style routing"),
    ("hunyuan_v1_moe", "hunyuan_v1_moe", {"num_experts": 4, "moe_topk": 2}, None, "HunYuan experts"),
    ("phimoe", "phimoe", {"num_local_experts": 4, "num_experts_per_tok": 2}, "phimoe", "sparsemixer routing"),
    ("dbrx", "dbrx", {"d_model": 256, "n_heads": 4, "n_layers": 2,
                      "attn_config": {"kv_n_heads": 2, "clip_qkv": 8.0, "rope_theta": 10000.0},
                      "ffn_config": {"ffn_hidden_size": 256, "moe_num_experts": 4, "moe_top_k": 2}},
     "dbrx", "fused experts, clamp"),
    ("gpt_oss", "gpt_oss", {"num_local_experts": 4, "num_experts_per_tok": 2, "sliding_window": 8,
                            "layer_types": ["sliding_attention", "full_attention"]}, "gpt_oss",
     "attention sinks, clamped SwiGLU experts"),
    ("minicpm3", "minicpm3", {"q_lora_rank": 128, "kv_lora_rank": 64, "qk_nope_head_dim": 32, "qk_rope_head_dim": 32,
                              "v_head_dim": 64, "num_key_value_heads": 4, "head_dim": 32, "scale_emb": 12,
                              "scale_depth": 1.4, "dim_model_base": 64},
     "minicpm3", "MiniCPM3: latent attention with halved rotary, MiniCPM's three scales"),
    ("chameleon", "chameleon", {"vocabulary_map": {}, "vq_config": {}}, "chameleon",
     "Chameleon's text path: per-head LayerNorm Q/K norms"),
    # transformers has no class: its 2023 remote code's expert block on transformers' Llama
    # (sml_ref_deepseek.py)
    ("deepseek_moe_v1", "deepseek", {"n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 2,
                                     "moe_intermediate_size": 128, "first_k_dense_replace": 1},
     "deepseek", "DeepSeek MoE v1: softmax top-k without renormalizing, shared experts, dense first layer"),
    ("deepseek_v2", "deepseek_v2", {"n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 256,
                                    "first_k_dense_replace": 1, "q_lora_rank": None, "kv_lora_rank": 64,
                                    "qk_nope_head_dim": 32, "qk_rope_head_dim": 32, "v_head_dim": 64,
                                    "num_key_value_heads": 4},
     "deepseek2", "multi-head latent attention"),
    ("llama4_text", "llama4_text", {"num_local_experts": 4, "interleave_moe_layer_step": 1,
                                    "intermediate_size_mlp": 512}, "llama4", "NoPE layers, chunked attention, experts"),
    ("opt", "opt", {"ffn_dim": 512, "word_embed_proj_dim": 256}, None,
     "learned positions offset by 2, ReLU"),
    ("codegen", "codegen", {"rotary_dim": 32, "n_ctx": 256}, None, "GPT-J shape, fused qkv in model-parallel order"),
    ("xglm", "xglm", {"ffn_dim": 512, "attention_heads": 4, "num_layers": 2}, None, "sinusoidal positions"),
    ("biogpt", "biogpt", {}, None, "learned positions, scaled embedding"),
    ("apertus", "apertus", {}, "apertus", "xIELU activation, Q/K norm"),
    ("mamba2", "mamba2", {"num_heads": 8, "head_dim": 64, "expand": 2, "state_size": 16, "n_groups": 2,
                          "conv_kernel": 4, "num_hidden_layers": 2, "tie_word_embeddings": False},
     "mamba2", "pure Mamba-2: conv, per-head state in 2 groups, gated norm, no FFN"),
    ("granitemoehybrid", "granitemoehybrid", {"layer_types": ["mamba", "attention"], "mamba_n_heads": 8,
                                              "mamba_d_head": 64, "mamba_n_groups": 2, "mamba_d_state": 16,
                                              "mamba_expand": 2, "mamba_d_conv": 4, "num_local_experts": 4,
                                              "num_experts_per_tok": 2, "shared_intermediate_size": 512,
                                              "position_embedding_type": "rope", "embedding_multiplier": 3.0,
                                              "residual_multiplier": 0.5, "logits_scaling": 2.0,
                                              "attention_multiplier": 0.1},
     "granite_hybrid", "Granite-4.0-H: Mamba-2 + attention + experts + shared MLP + Granite scales"),
    ("granitemoehybrid-nope", "granitemoehybrid", {"layer_types": ["mamba", "attention"], "mamba_n_heads": 8,
                                                   "mamba_d_head": 64, "mamba_n_groups": 1, "mamba_d_state": 16,
                                                   "mamba_expand": 2, "mamba_d_conv": 4, "num_local_experts": 0,
                                                   "shared_intermediate_size": 512,
                                                   "position_embedding_type": "nope"},
     "granite_hybrid", "Granite-4.0-H as shipped: no rotary, dense shared MLP"),
    ("mamba", "mamba", {"state_size": 16, "expand": 2, "conv_kernel": 4, "time_step_rank": 16,
                        "num_hidden_layers": 2, "tie_word_embeddings": False},
     "mamba", "Mamba-1: low-rank dt, per-channel state"),
    ("falcon_mamba", "falcon_mamba", {"state_size": 16, "expand": 2, "conv_kernel": 4, "time_step_rank": 16,
                                      "num_hidden_layers": 2, "tie_word_embeddings": False},
     "mamba", "Falcon-Mamba: Mamba-1 with weightless dt/B/C norms"),
    ("jamba", "jamba", {"attn_layer_period": 2, "attn_layer_offset": 1, "expert_layer_period": 2,
                        "expert_layer_offset": 1, "num_experts": 4, "num_experts_per_tok": 2,
                        "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2, "mamba_dt_rank": 16,
                        "use_mamba_kernels": False},
     "jamba", "Mamba-1 + attention (no positions) + experts, dt/B/C norms"),
    ("bamba", "bamba", {"attn_layer_indices": [1], "mamba_n_heads": 8, "mamba_d_head": 64, "mamba_n_groups": 2,
                        "mamba_d_state": 16, "mamba_expand": 2, "mamba_d_conv": 4},
     None, "Mamba-2 layers with an attention layer and SwiGLU MLPs"),
    ("bitnet", "bitnet", {}, "bitnet", "ternary weights with sub-norms"),
    ("lfm2", "lfm2", {"layer_types": ["conv", "full_attention"], "block_auto_adjust_ff_dim": False,
                      "conv_bias": True}, "lfm2", "short-convolution hybrid"),
    ("qwen3_next", "qwen3_next", {"layer_types": ["linear_attention", "full_attention"],
                                  "partial_rotary_factor": 0.25, "linear_num_value_heads": 4,
                                  "linear_num_key_heads": 2, "linear_key_head_dim": 32,
                                  "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4,
                                  "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 256,
                                  "shared_expert_intermediate_size": 256, "decoder_sparse_step": 1,
                                  "mlp_only_layers": [], "norm_topk_prob": True},
     "qwen3next", "Gated DeltaNet + gated attention + experts, (1+w) norms"),
    ("qwen3_5_text", "qwen3_5_text", {"layer_types": ["linear_attention", "full_attention"],
                                      "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                                      "linear_key_head_dim": 32, "linear_value_head_dim": 64,
                                      "linear_conv_kernel_dim": 4,
                                      "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0,
                                                          "partial_rotary_factor": 0.25}},
     "qwen35", "Qwen 3.5: Gated DeltaNet 1:1 here (3:1 shipped), gated attention, 25% rotary at 1e7 "
               "(needs transformers >= 5: run with the venv-tf5 python)"),
    # ---- transformers >= 5 only (run with the venv-tf5 python) -------------
    ("nemotron_h", "nemotron_h", {"hybrid_override_pattern": "M*-", "num_hidden_layers": 3,
                                  "mamba_num_heads": 8, "mamba_head_dim": 64, "ssm_state_size": 16,
                                  "n_groups": 2, "conv_kernel": 4, "expand": 2, "intermediate_size": 512},
     "nemotron_h", "Mamba-2, attention and MLP as separate layers"),
    ("nemotron_h_moe", "nemotron_h", {"hybrid_override_pattern": "M*E", "num_hidden_layers": 3,
                                      "mamba_num_heads": 8, "mamba_head_dim": 64, "ssm_state_size": 16,
                                      "n_groups": 2, "conv_kernel": 4, "expand": 2, "intermediate_size": 512,
                                      "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 256,
                                      "moe_shared_expert_intermediate_size": 512, "n_group": 1, "topk_group": 1,
                                      "norm_topk_prob": True, "routed_scaling_factor": 2.5},
     "nemotron_h_moe", "Nemotron 3 Nano: Mamba-2, attention and ungated relu^2 experts (sigmoid routing)"),
    ("falcon_h1", "falcon_h1", {"mamba_n_heads": 8, "mamba_d_head": 64, "mamba_n_groups": 2,
                                "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2, "mamba_d_ssm": 512,
                                # every muP multiplier off 1, so each fold is exercised
                                "attention_in_multiplier": 0.8, "key_multiplier": 1.3, "attention_out_multiplier": 0.7,
                                "ssm_out_multiplier": 1.2, "ssm_in_multiplier": 0.9, "mlp_multipliers": [1.1, 0.8],
                                "ssm_multipliers": [1.1, 0.9, 1.2, 0.8, 1.05], "embedding_multiplier": 2.0,
                                "lm_head_multiplier": 0.5},
     "falcon_h1", "attention and Mamba-2 in parallel in every layer, with multipliers"),
    ("lfm2_moe", "lfm2_moe", {"layer_types": ["conv", "full_attention"], "num_experts": 4,
                              "num_experts_per_tok": 2, "moe_intermediate_size": 256, "num_dense_layers": 1},
     "lfm2moe", "LFM2 with experts"),
    ("exaone_moe", "exaone_moe", {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 256},
     "exaone_moe", "EXAONE with experts"),
    ("afmoe", "afmoe", {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 256,
                        "num_dense_layers": 1}, "afmoe", "Arcee MoE"),
    ("minimax_m2", "minimax_m2", {"num_local_experts": 4, "num_experts_per_tok": 2}, "minimax_m2", "MiniMax-M2"),
    ("jais2", "jais2", {}, "jais2", "Jais 2"),
    ("gpt_oss", "gpt_oss", {"num_local_experts": 4, "num_experts_per_tok": 2, "sliding_window": 8,
                            "layer_types": ["sliding_attention", "full_attention"]}, "gpt_oss",
     "attention sinks, clamped SwiGLU experts"),
    ("deepseek_v3", "deepseek_v3", {"n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 256,
                                    "first_k_dense_replace": 1, "q_lora_rank": 128, "kv_lora_rank": 64,
                                    "qk_nope_head_dim": 32, "qk_rope_head_dim": 32, "v_head_dim": 64,
                                    "n_group": 1, "topk_group": 1, "head_dim": 32, "num_key_value_heads": 4},
     "deepseek2",
     "latent attention + V3 routing"),
    ("qwen3_5_moe_text", "qwen3_5_moe_text", {"layer_types": ["linear_attention", "full_attention"],
                                              "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                                              "linear_key_head_dim": 32, "linear_value_head_dim": 64,
                                              "linear_conv_kernel_dim": 4, "num_experts": 4,
                                              "num_experts_per_tok": 2, "moe_intermediate_size": 256,
                                              "shared_expert_intermediate_size": 256,
                                              "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0,
                                                                  "partial_rotary_factor": 0.25}},
     "qwen35moe", "Qwen 3.5 with experts"),
    ("qwen3_next-dense", "qwen3_next", {"layer_types": ["linear_attention", "full_attention"],
                                        "partial_rotary_factor": 0.25, "linear_num_value_heads": 4,
                                        "linear_num_key_heads": 2, "linear_key_head_dim": 32,
                                        "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4,
                                        "num_experts": 4, "mlp_only_layers": [0, 1]},
     "qwen3next", "Qwen3-Next with dense MLPs: the linear attention alone"),
]


# The same family at the shape of a big checkpoint: 128-wide heads, one shared
# K/V head (the 8:1 .. 64:1 GQA of 7B-405B models), more layers, a wider FFN.
# Parameter count is width x depth; what can break between a 1B and a 70B is
# head width, the K/V ratio and the FFN width, and this shape carries all three.
# A head whose own overrides fix a size keeps it.
BIG = dict(BASE, hidden_size=512, n_embd=512, d_model=512, num_hidden_layers=4, n_layer=4, n_layers=4,
           num_key_value_heads=1, head_dim=128, intermediate_size=1408, n_inner=1408, ffn_hidden_size=1408)
SHAPES = {"small": BASE, "big": BIG}


def build_head(label, model_type, over, dst, seed, shape="small"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    ref_type, model_type = model_type, ("llama" if model_type == "deepseek" else model_type)
    # only the size keys this config class declares: a stray num_key_value_heads
    # on an MHA-only class becomes an attribute that other readers then believe
    known = set(AutoConfig.for_model(model_type).to_dict())
    kw = {k: v for k, v in SHAPES[shape].items() if k in known}
    kw.update(over)
    if shape != "small":
        # the head's per-layer pattern, repeated over the deeper stack; state-space
        # heads widened with the model (heads x head width = expand x hidden)
        n_l = int(kw.get("num_hidden_layers") or kw.get("n_layer") or 4)
        if isinstance(kw.get("layer_types"), list) and len(kw["layer_types"]) != n_l:
            lt = kw["layer_types"]
            kw["layer_types"] = [lt[i % len(lt)] for i in range(n_l)]
        rs = kw.get("rope_scaling") or {}
        if rs.get("rope_type") == "longrope":      # one factor per rotary pair at this width
            pairs = int(int(kw.get("head_dim") or 128) * float(kw.get("partial_rotary_factor") or 1.0)) // 2
            kw["rope_scaling"] = dict(rs, **{k: list(np.interp(np.linspace(0, 1, pairs),
                                                               np.linspace(0, 1, len(rs[k])), rs[k]))
                                             for k in ("short_factor", "long_factor")})
        small_l = int(BASE["num_hidden_layers"])
        if kw.get("mlp_only_layers") == list(range(small_l)):      # "every layer dense" stays so
            kw["mlp_only_layers"] = list(range(n_l))
        hid = int(kw.get("hidden_size") or 512)
        if "word_embed_proj_dim" in over:          # OPT: an embedding as wide as the stream
            kw["word_embed_proj_dim"] = hid
        for hk, dk, ek in (("num_heads", "head_dim", "expand"), ("mamba_n_heads", "mamba_d_head", "mamba_expand"),
                           ("mamba_num_heads", "mamba_head_dim", "expand")):
            if hk in over and dk in over:
                kw[hk] = int(kw.get(ek) or 2) * hid // int(over[dk])
                if hk == "mamba_n_heads" and "mamba_d_ssm" in kw:     # Falcon-H1 states the width too
                    kw["mamba_d_ssm"] = kw[hk] * int(over[dk])
    cfg = AutoConfig.for_model(model_type, **kw)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    try:
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    except ValueError:
        # a multimodal wrapper (mistral3): the text path is what is compared
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_config(cfg, torch_dtype=torch.float32)
    ds_cfg = None
    if ref_type == "deepseek":
        import sml_ref_deepseek
        ds_cfg = sml_ref_deepseek.convert(model, cfg.to_dict(), over)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                fan = p.shape[-1] if p.shape[-1] > 1 else p.shape[0]
                if "embed" in name or "wte" in name or "wpe" in name or "word_embeddings" in name:
                    p.copy_(torch.randn(p.shape, generator=g))
                else:
                    p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(fan))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            else:
                # norm weights: off their defaults so a misplaced (w+1) or a
                # dropped norm shows; the (1+w) families store w near 0
                base = 0.0 if model_type in ("gemma", "gemma2", "gemma3_text") else 1.0
                p.copy_(base + 0.2 * torch.randn(p.shape, generator=g))
    model.eval()
    os.makedirs(dst, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    if ds_cfg is not None:                        # the checkpoint's own model_type and keys
        with open(os.path.join(dst, "config.json"), "w") as f:
            json.dump({k: v for k, v in ds_cfg.items() if not k.startswith("_")}, f, indent=1, default=str)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        src = os.path.join(TOK_SRC, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst, f))
    return model


def hf_sensitivity(model, seqs, bits=6):
    """How far transformers ITSELF moves on 6-bit weights: every matrix rounded
    to a min/max grid of 64 levels per 32 inputs. Families differ -- Nemotron's
    (w+1) norms and relu^2 make it twice as sensitive as Llama -- so a head is
    judged against its own family's noise, not Llama's."""
    import copy
    import torch
    q = copy.deepcopy(model)
    with torch.no_grad():
        for _, p in q.named_parameters():
            if p.dim() == 2 and p.shape[-1] % 32 == 0:
                w = p.reshape(-1, 32)
                lo, hi = w.min(1, keepdim=True).values, w.max(1, keepdim=True).values
                s = (hi - lo).clamp_min(1e-12) / (2 ** bits - 1)
                p.copy_((torch.round((w - lo) / s) * s + lo).reshape(p.shape))
    return float(np.median([rel_err(hf_last_logits(q, ids), hf_last_logits(model, ids))
                            for ids in seqs]))


def hf_last_logits(model, ids):
    import torch
    with torch.no_grad():
        out = model(torch.tensor([ids]))
    return out.logits[0, -1].double().numpy()


def sheep_logits(ssf, ids, env_extra):
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    env = dict(os.environ)
    env.update(env_extra)
    try:
        r = subprocess.run([CLI, ssf, "--tokens", ",".join(map(str, ids)), "--gen", "0",
                            "--threads", os.environ.get("SML_HEAD_THREADS", "6"), "--logits", path], env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        if r.returncode != 0:
            return None, (f"exit {r.returncode & 0xFFFFFFFF:#x} " + (r.stdout + r.stderr)[-380:]).strip()
        meta = np.fromfile(path, dtype=np.int32, count=2)
        raw = np.fromfile(path, dtype=np.float32, offset=8)
    finally:
        os.remove(path)
    if raw.size != int(meta[0]):
        return None, "logits size mismatch"
    return raw.astype(np.float64), ""


def rel_err(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-30))


def run_head(label, model_type, over, work, seqs, shape="small"):
    d = os.path.join(work, label + ("" if shape == "small" else "@" + shape))
    ssf = os.path.join(d, "model.ssf")
    res = {"label": label, "model_type": model_type, "shape": shape}
    try:
        model = build_head(label, model_type, over, d, seed=1234, shape=shape)
    except Exception as e:                        # the config itself would not build
        res["error"] = f"transformers: {type(e).__name__}: {e}"[:300]
        return res
    bits = int(os.environ.get("SML_HEAD_BITS", "6"))
    _measure(res, model, d, ssf, seqs, bits)
    if not res.get("pass") and "error" not in res and bits < 8:
        # A random router sits on near-ties: 6-bit rounding flips which experts
        # a token takes, and a deeper stack amplifies the flip (Dots1 and OLMoE at
        # the big shape, 2026-10-13: 0.39 and 0.09 at 6 bits, 0.027 and 0.017 at
        # 8). At 8 bits the maths shows through: a wrong kernel still misses, a
        # flipped router does not. The 6-bit numbers stay in the record.
        six = {k: res.pop(k) for k in ("sens", "stepped", "batched", "pass")}
        os.remove(ssf)
        _measure(res, model, d, ssf, seqs, 8)
        res["at6"] = six
        res["bits"] = 8
    return res


def _measure(res, model, d, ssf, seqs, bits):
    b = subprocess.run([os.environ.get("SML_BUILD_PYTHON", sys.executable), BUILD, d, ssf, "--layout", "flat", "--alloc", f"weight={bits}",
                        "--no-gptq", "--seqs", "4", "--seq-len", "32"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    if b.returncode != 0 or not os.path.exists(ssf):
        tail = (b.stdout + b.stderr).strip().splitlines()
        res["error"] = "build: " + (" | ".join(tail[-3:]) if tail else "failed")[:400]
        return
    res["sens"] = hf_sensitivity(model, seqs, bits)
    for path_name, env in (("stepped", {"SML_SEQ_PROMPT": "1"}), ("batched", {})):
        errs, top = [], 0
        for ids in seqs:
            h = hf_last_logits(model, ids)
            s, why = sheep_logits(ssf, ids, env)
            if s is None:
                res["error"] = f"engine ({path_name}): {why}"
                return
            errs.append(rel_err(s, h))
            top += int(np.argmax(s) == np.argmax(h))
        res[path_name] = {"rel": float(np.median(errs)), "worst": float(np.max(errs)),
                          "top1": f"{top}/{len(seqs)}"}
    # pass: within the 6-bit band the graded families set (<= 0.08), or within
    # 1.6x of what transformers itself loses on the same rounding for this family
    worst_rel = max(res["stepped"]["rel"], res["batched"]["rel"])
    res["pass"] = bool(worst_rel <= max(0.08, 1.6 * res["sens"]))


def _hash_files(*paths):
    import hashlib
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def stamp():
    """What a head's verdict depends on: transformers, the engine, the builder.
    A saved result with the same stamp is still true; anything else re-runs."""
    import transformers
    return {"transformers": transformers.__version__,
            "engine": _hash_files(CLI),
            "builder": _hash_files(BUILD, os.path.join(HERE, "sml_hf.py"), os.path.join(HERE, "sml_format.py"))}


def _one(job):
    """One (head, shape), in a worker process. Its build folder (~100-400 MB) goes
    as soon as it is measured: kept to the end, 170 of them filled the disk."""
    (label, mt, over, key, what), shape, work, seqs, st, keep = job
    try:
        r = run_head(label, mt, over, work, seqs, shape)
    except Exception as e:                            # a head that crashes is a row, not a dead run
        r = {"label": label, "model_type": mt, "shape": shape, "error": f"{type(e).__name__}: {e}"[:300]}
    if not keep:
        shutil.rmtree(os.path.join(work, label + ("" if shape == "small" else "@" + shape)), ignore_errors=True)
    r.update(roster=key, represents=what, stamp=st)
    return r


def _line(r):
    tag = r["label"] + ("" if r.get("shape", "small") == "small" else " @" + r["shape"])
    if "error" in r:
        return f"  {tag:<24} ERROR  {r['error'][:160]}"
    st, bt = r["stepped"], r["batched"]
    return (f"  {tag:<24} {'PASS' if r['pass'] else 'FAIL'}  stepped {st['rel']:.4f} top1 {st['top1']}"
            f" | batched {bt['rel']:.4f} top1 {bt['top1']} | HF on 6-bit {r['sens']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--all", action="store_true",
                    help="every head in both shapes, in parallel; a saved result with the same "
                         "transformers/engine/builder stamp is reused")
    ap.add_argument("--shapes", default="", help="small,big (default: small; with --all: both)")
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (default 1; with --all 3)")
    ap.add_argument("--redo", action="store_true", help="ignore saved results")
    ap.add_argument("--keep", default="")
    ap.add_argument("--json", default=os.path.join(ROOT, "tooling", "family_heads.json"))
    args = ap.parse_args()
    want = set(x for x in args.only.split(",") if x)
    shapes = [x for x in (args.shapes or ("small,big" if args.all else "small")).split(",") if x]
    jobs_n = args.jobs or (3 if args.all else 1)
    work = args.keep or tempfile.mkdtemp(prefix="sml_heads_")
    rng = np.random.default_rng(7)
    seqs = [list(map(int, rng.integers(300, VOCAB - 300, size=20))) for _ in range(4)]
    st = stamp()
    results = []
    if os.path.exists(args.json):
        with open(args.json, encoding="utf-8") as f:
            results = json.load(f)
    for r in results:
        r.setdefault("shape", "small")
    saved = {(r["label"], r["shape"]): r for r in results}
    todo, seen = [], set()
    for h in HEADS:
        if (want and h[0] not in want) or h[0] in seen:
            continue
        seen.add(h[0])                                # a label listed twice runs once
        for shape in shapes:
            old = saved.get((h[0], shape))
            transient = any(x in str((old or {}).get("error", "")) for x in ("space on the disk", "MemoryError", "PermissionError"))
            if old and old.get("stamp") == st and not args.redo and not transient:
                print(_line(old) + "   (saved)", flush=True)
                continue
            todo.append((h, shape, work, seqs, st, bool(args.keep)))

    def save(r):
        nonlocal results
        results = [x for x in results if (x["label"], x["shape"]) != (r["label"], r["shape"])] + [r]
        # write-then-rename: a power cut mid-dump left the record half-written
        # (unreadable, or cut to a subset) twice before this
        with open(args.json + ".tmp", "w", encoding="utf-8") as f:
            json.dump(sorted(results, key=lambda x: (x["label"], x["shape"])), f, indent=1)
        for i in range(10):                       # a virus scanner can hold the file a moment
            try:
                os.replace(args.json + ".tmp", args.json)
                break
            except PermissionError:
                time.sleep(0.5)
        print(_line(r), flush=True)

    os.environ.setdefault("SML_HEAD_THREADS", str(max(1, (os.cpu_count() or 6) // jobs_n)))
    if jobs_n <= 1:
        for j in todo:
            save(_one(j))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(jobs_n) as ex:
            for f in as_completed([ex.submit(_one, j) for j in todo]):
                save(f.result())
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    done = [r for r in results if (not want or r["label"] in want) and r["shape"] in shapes]
    print(f"{sum(bool(r.get('pass')) for r in done)} of {len(done)} head runs pass ({', '.join(shapes)})")


if __name__ == "__main__":
    main()
