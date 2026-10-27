#!/usr/bin/env python3
"""Settings for every architecture the reference engine carries.

UNTESTED BY CONSTRUCTION, and named so. We hold checkpoints for three families
and none of the other hundred-odd. This table does not claim support. Its job is
to make the difference between "runs today", "needs an engine feature we do not
have" and "out of scope" EXPLICIT AT BUILD TIME, so an unsupported model fails
with the reason instead of producing fluent nonsense.

That failure mode is the whole point. Every architecture bug in this project has
produced PLAUSIBLE TEXT rather than an error -- the RoPE convention inverted
across five families, the rotated embedding lookup, a flat plane read as a
ladder. A model we cannot compute correctly has to say so.

WHY A TABLE AND NOT A HUNDRED GRAPH BUILDERS
--------------------------------------------
The reference engine declares each architecture's behaviour in its own graph
construction code, so its breadth is also its maintenance surface. SML derives
every STRUCTURAL difference from the tensor table -- QKV bias, tied head, Q/K
norm tensors, expert tensors are all just "is this tensor present" -- so only
BEHAVIOURAL differences need declaring, and those are a handful of config
fields. `hf` in a note means `arch_flags` already reads everything it needs from
the checkpoint's own config, and the row exists only to record that the
architecture was considered.

Adding an architecture here is a row. For them it is a builder.

SCOPE, WHICH IS THE POINT OF THE `covers` FIELD
-----------------------------------------------
Verifying `llama` is not verifying one model. `model_type: llama` is what
Mistral, Yi, Vicuna, Zephyr, OpenChat, SOLAR, Nous-Hermes, WizardLM, TinyLlama,
CodeLlama and dozens of finetunes all declare. Three verified families cover a
large share of what people actually run, which is a different and more useful
claim than "three models work".

STATUS
  ok      runs today on the axes SML already has
  gap     needs an engine feature listed in NEEDS; the build refuses
  scope   not a transformer decoder; out of scope for this engine
  unknown in the reference roster, behaviour NOT established here. Not a guess
          either way -- these postdate what was written from recollection, and
          the diff against their llama-arch.cpp is what surfaced them.
"""

# =============================================================================
# THEIR SETTINGS TABLE IS NOT TRANSPLANTABLE. Checked against their source
# 2026-09-09, because copying it looked like the obvious way to fill this table.
#
# llama_model_rope_type() in their llama-model.cpp assigns, verbatim:
#   NONE  21 architectures      NORM  31      NEOX  61      MROPE/IMROPE  6
# with NORM commented "operating on pairs of consecutive head values" -- that is
# the INTERLEAVED convention -- and LLM_ARCH_LLAMA is in the NORM list.
#
# Our engine treats llama as HALVED and it demonstrably works (Llama-3.2-1B
# answers "Paris. The capital city of the United States is Washington D.C.").
# Both are correct: their convert_hf_to_gguf.py PERMUTES Q and K at conversion
# so that interleaved RoPE reproduces HF's rotate_half. Their table describes
# their post-conversion representation; we read HF checkpoints directly.
#
# So copying it would invert the convention for 31 architectures -- the same bug
# this project already had once, in the opposite direction, when the name "NEOX
# rope" was carried across for five families.
#
# The settings here must keep coming from the CHECKPOINT'S OWN CONFIG, which is
# what arch_flags does. A mature codebase's tables encode its own pipeline, and
# reverse-engineering it imports its history along with its knowledge.
# =============================================================================

# Engine features SML does not have. A row naming any of these cannot build.
MISSING = {
    "moe":      "expert routing: a top-k router selecting per-token experts",
    "alibi":    "ALiBi positional bias instead of RoPE",
    "qk_norm":  "per-head RMSNorm on Q and K before the dot product",
    "swa_alt":  "alternating sliding-window and full-attention layers",
    "attn_cap": "softcap on attention logits (we cap final logits only)",
    "mla":      "multi-head latent attention with compressed KV",
    "qkv_mqa":  "fused QKV with one shared K/V head: unequal thirds to split",
    "ssm":      "state-space or recurrent mixing rather than attention",
    "hybrid":   "alternating attention and state-space layers in one stack",
    "encdec":   "encoder-decoder or encoder-only stack",
    "vision":   "a vision tower and cross-modal projection",
    "audio":    "an audio codec stack",
    "diffusion":"iterative denoising rather than left-to-right decoding",
    "perlayer": "per-layer head counts or FFN widths",
    "relu2":    "ungated squared-ReLU FFN: up, relu(x)^2, down",
    "ln1p":     "LayerNorm whose stored weight is (w - 1), applied as (w + 1)",
    "postnorm": "norms on each branch's OUTPUT and none on its input (OLMo 2)",
    "xielu":    "xIELU activation with learned per-layer alphas",
    "moe_bias": "expert selection through a learned correction bias",
    "moe_fused":"experts stored fused in one tensor per projection",
    "sinks":    "attention sinks (a learned per-head logit in every softmax)",
    "subnorm":  "norms inside attention and the MLP, before the output projections",
}

# model_type -> (status, needs, note)
ROSTER = {
    # ===================== RMSNorm + halved RoPE + gated FFN =================
    # The mainstream. Everything here is read from config; nothing is declared.
    "llama":      ("ok", (), "RACED, Llama-3.2-1B: 41% better quality, "
                             "1.034x decode vs their Q4_K_M; GQA 32/8"),
    "arcee": ('ok', (), "family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: an UNGATED relu^2 MLP -- the old 'hf' row would have run SwiGLU"),
    "baichuan":   ("ok", (), "hf; W_pack is [Q|K|V]; 7B is RoPE, the 40-layer 13B is ALiBi; Baichuan 2's "
                             "NormHead L2-normalizes the head's rows at inference -- all three read from "
                             "modeling_baichuan.py and applied 2026-10-13 (before, W_pack had no place in the "
                             "name table and no Baichuan checkpoint built)"),
    "codeshell":  ("ok", (), "hf; LayerNorm (its checkpoints carry norm biases), 2026-10-13"),
    "deci":       ("gap", ("perlayer",),
                   "was marked ok until 2026-09-24, but nothing reads its "
                   "block_configs: Llama-Nemotron Super/Ultra vary KV heads and "
                   "FFN width per layer and skip attention in some layers. "
                   "Llama-Nemotron Nano 4B/8B are plain 'llama' and unaffected"),
    "deepseek":   ("ok", (), "hf; MoE v1 family head vs its own expert block on transformers' Llama, 2026-10-19"),
    "exaone":     ("ok", (), "hf"),
    "exaone4": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: post-norms and Q/K norm; hybrid form: windows on the local layers, no rotary on the global ones'),
    "granite": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: embedding/residual/logit/attention scales'),
    "internlm2":  ("ok", (), "hf; wqkv per K/V group [q.. k v] (layout 'group'), mapped 2026-10-13 -- "
                             "before, no InternLM2 checkpoint built"),
    "minicpm":    ("ok", (), "hf; scale_emb, scale_depth/sqrt(layers) and hidden/dim_model_base on Granite's "
                             "three scales, 2026-10-13 -- before, they were ignored and it ran as plain llama; "
                             "its config has no model_type (read from MiniCPMForCausalLM)"),
    "minicpm3": ('ok', (), 'latent attention (arch bit 11) with halved rotary and LongRoPE on the rotary rows, MiniCPM scales; family head vs transformers 5.17, 2026-10-19'),
    "olmo": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: weightless LayerNorm, Q/K/V clamp'),
    "orion":      ("ok", (), "hf; LayerNorm (norm biases in its checkpoints; it ran as RMSNorm), 2026-10-13"),
    "qwen":       ("ok", (), "hf"),
    "qwen2":      ("ok", (), "RACED, Qwen2.5-3B: 0.99405 vs fp16, 23% better "
                             "quality and 1.144x decode vs their Q4_K_M"),
    "smollm3": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: NoPE on every fourth layer, per-layer rotary table'),
    "xverse":     ("ok", (), "hf"),
    "nemotron": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: LayerNorm1P (w+1) and an ungated relu^2 MLP'),
    "seed_oss": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28'),
    "apertus": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-30: xIELU with its learned alpha_p/alpha_n per layer (bf16 softplus, as transformers computes), Q/K norm; CPU only, the GPU refuses xIELU by name'),
    "pangu_embed":("ok", (), "hf"),
    "ernie4_5": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: interleaved rotary'),
    "hunyuan_dense": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28 (hunyuan_v1_dense): Q/K norm AFTER the rotary; its query_layernorm/key_layernorm had been dropped silently'),
    "llama4": ('gap', ('moe_fused',), 'NoPE on every fourth layer (now possible) but also chunked attention, L2 Q/K norm and fused expert tensors'),
    "grok":       ("gap", ("moe", "attn_cap"), ""),

    # ===================== LayerNorm decoders ================================
    "gpt_neox":   ("ok", (), "RACED, pythia-410m: 12.4% better quality, 1.503x "
                             "decode; partial rotary and parallel residual, "
                             "both read from config"),
    "stablelm": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: LayerNorm, 25% rotary, QKV bias'),
    "persimmon": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: LayerNorm Q/K norms with bias, relu^2, partial rotary'),
    "phi2": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28 (phi): one shared LayerNorm, parallel branches, output-head bias'),
    "phi3": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: fused qkv with grouped K/V and fused gate|up; LongRoPE (Phi-3-128k, Phi-4-mini) since 2026-10-13: short factors, context capped at the original window, attention factor in the Q/K weights (family head phi3-longrope)'),
    "starcoder2": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: LayerNorm with bias, ungated GELU; sliding window applied'),
    "refact":     ("unknown", (), "sml_compat.py, 2026-10-13: Refact-1.6B names its tensors attn.q, attn.kv "
                               "(one shared K/V head, fused) and mlp.gate_up_proj -- not in the name table -- and "
                               "transformers has no class for it, so no head can check its maths"),
    "command_r": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28 (cohere): interleaved rotary, one LayerNorm without bias feeding parallel branches, logit_scale multiplies'),
    "cohere2": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: alternating local/global, rotary on the local layers only'),
    "chameleon":  ("ok", (), "text path (the VQ image tokenizer set aside); per-head LayerNorm Q/K norms (arch bit 9, CPU); family head vs transformers 5.17, 2026-10-19"),

    # ===================== interleaved RoPE ==================================
    "gptj": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: interleaved rotary over rotary_dim, parallel branches, head bias'),

    # ===================== ALiBi, no RoPE ====================================
    "bloom":      ("ok", (), "GRADED 2026-09-24, BLOOM-560m: ALiBi slopes equal HF's "
                             "build_alibi_tensor to 3e-8; KL 0.34 to fp32 (GPT-2, 19/20, "
                             "is 0.41), top-1 8/20 on distributions this flat (top two "
                             "often within 0.01). Weights-vs-engine split NOT run"),
    "mpt": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: ALiBi, LayerNorm without bias, fused [Q|K|V]'),
    "jais":       ("gap", ("alibi",), "SwiGLU with a width multiplier"),
    "falcon": ('ok', (), "family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: RoPE (7B: multi-query, parallel, one LayerNorm) and ALiBi (rw: per-head packing, ALiBi before the scale), and the new decoder's K/V groups"),

    # ===================== learned absolute positions ========================
    "gpt2":       ("ok", (), "GRADED 2026-09-24, GPT-2 small: 19/20 top-1 vs HF fp32. "
                             "Conv1D [in,out] weights, learned positions, no rotary; "
                             "the tied head at 6 bits was worth 7/20 -> 19/20"),
    "opt":        ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-30: learned positions offset by 2, pre-LayerNorm, ReLU FFN; llama.cpp has no converter for it'),
    "biogpt":     ('ok', (), 'family head vs transformers 4.57.6, 2026-09-30: learned positions offset by 2, sqrt(d) embedding scale, exact GELU; llama.cpp has no converter for it'),
    "xglm":       ('ok', (), 'family head vs transformers 4.57.6, 2026-09-30: sinusoidal positions stored as a table, sqrt(d) embedding scale; llama.cpp has no converter for it'),
    "codegen":    ('ok', (), 'family head vs transformers 4.57.6, 2026-09-30: GPT-J shape, qkv in four model-parallel [q|v|k] blocks; llama.cpp has no converter for it'),
    "gpt_bigcode": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: StarCoder-1: one shared K/V head in the fused c_attn'),
    "plm":        ("unknown", (), "was listed as abspos; not re-read since"),

    # ===================== Q/K norm ==========================================
    # The cheapest gap on this list: two RMSNorm applications, and it unlocks
    # Qwen3, which is current and widely deployed.
    "qwen3":      ("ok", (), "family head vs transformers 4.57.6 and Qwen3-0.6B built; Q/K norm, and a head (128) wider than n_embd / n_head "
                  "(64) -- the old note 'otherwise identical to qwen2' missed that"),
    "olmo2":      ("ok", (), "GRADED 2026-09-24, OLMo-2-0425-1B: the engine is EXACT -- "
                  "20/20 top-1 and corr 0.9988 against HF running the file's own "
                  "weights. The 4-bit weights cost the rest: 10/20 top-1 vs fp32 "
                  "(Llama-3.2-1B: 12/20 on the same prompts). Post-norms rescale "
                  "every branch, so weight error is not damped: a measured "
                  "allocation (--alloc) is the lever, not the engine"),
    "smallthinker": ("gap", ("qk_norm",), ""),

    # ===================== Gemma =============================================
    "gemma": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: (1+w) norms, GeGLU, sqrt(d) embedding scale -- all three were missing'),
    "gemma2": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: alternating window/global, attention softcap, four norms per layer'),
    "gemma3": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28 (gemma3_text): 5:1 local/global with two rotary bases, Q/K norm; text path -- the image path is vision'),
    "gemma3n":    ("gap", ("swa_alt", "qk_norm", "perlayer"), ""),
    "gemma_embedding": ("scope", ("encdec",), "embedding model, no decoding"),

    # ===================== mixture of experts ================================
    "mixtral": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: renormalized top-2 of per-expert w1/w3/w2'),
    "qwen2moe": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28 (qwen2_moe): softmax top-k (not renormalized) + a shared expert scaled by sigmoid(gate)'),
    "qwen3moe": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: experts + Q/K norm'),
    "phimoe": ('ok', (), 'sparsemixer routing (moe_flags bit 2); family head vs transformers 5.17, 2026-10-19'),
    "dbrx": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-30: experts stored fused as [E*ff, d] (w1 gates, v1 ups, w2 downs transposed), clamped Wqkv, bias-less LayerNorm, softmax top-k renormalized by the sum'),
    "olmoe": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: experts, whole-projection Q/K norm, not renormalized'),
    "granitemoe": ("ok", (), "VERIFIED 2026-09-24 on granite-3.1-1b-a400m: 32 experts, top-8 "
                  "renormalized, Granite scales; vs full precision rel-RMS 0.167, top-1 19/20. "
                  "CPU only, prompts stepped. Other MoE families need their expert names "
                  "mapped (and shared experts, non-renormalized routing) before they count"),
    "arctic":     ("gap", ("moe",), "dense part fits, the router does not"),
    "deepseek2":  ("ok", (), "multi-head latent attention expanded per head in the engine (arch bit 11, CPU), "
                             "V2 and V3 routing; family heads vs transformers 5.17, 2026-10-19"),
    "glm4_moe":   ("ok", (), "GLM-4.5: sigmoid scores, bias-corrected choice in expert groups, routing scale, shared experts; family head 2026-09-28"),
    "bailingmoe": ("gap", ("moe",), ""),
    "bailingmoe2":("gap", ("moe",), ""),
    "dots1":      ("ok", (), "dots.llm1: DeepSeek-V3-style routing; family head 2026-09-28"),
    "ernie4_5_moe": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: softmax scores, experts CHOSEN with a learned correction bias, renormalized; shared experts; interleaved rotary'),
    "hunyuan_moe": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28 (hunyuan_v1_moe): renormalized top-k, ungated shared MLP, Q/K norm after the rotary'),
    "openai_moe": ('ok', (), 'gpt-oss under their name'),
    "grovemoe":   ("gap", ("moe",), ""),
    "lfm2moe":    ("ok", (), "LFM2-MoE: LFM2 layers with sigmoid-routed experts and an expert bias; family head vs transformers 5.17, 2026-09-28"),
    "llada_moe":  ("gap", ("moe", "diffusion"), ""),
    "afmoe":      ("ok", (), "AFMoE (Arcee): gated attention, rotary on local layers, sandwich norms, V3-style routing with route_scale; family head vs transformers 5.17, 2026-09-28"),
    "minimax_m2": ("ok", (), "MiniMax-M2: sigmoid routing with a correction bias, renormalized; family head vs transformers 5.17, 2026-09-28"),
    "nomic_bert_moe": ("scope", ("encdec", "moe"), "encoder"),

    # ===================== hybrid attention + state space ====================
    "jamba":      ("ok", (), "Jamba: Mamba-1 layers with dt/B/C norms, attention with no positions, experts "
                             "(not renormalized); family head 2026-09-28; CPU, prompts stepped"),
    "bamba":      ("ok", (), "IBM Bamba: Mamba-2 and attention layers (Granite-4-H's maths); family head vs transformers 5.17, 2026-10-19"),
    "falcon_h1":  ("ok", (), "Mamba-2 beside attention in every layer, summed; muP multipliers folded into the weights; family head vs transformers 5.17, 2026-10-19"),
    "granite_hybrid": ("ok", (), "Granite-4.0-H (granitemoehybrid): Mamba-2 + attention (RoPE or none) + experts "
                                 "+ shared MLP + Granite scales; family heads 2026-09-28; CPU, prompts stepped"),
    "nemotron_h": ("ok", (), "Nemotron-H / Nemotron Nano 2: Mamba-2 (per-group gated norm), attention with "
                             "no positions, and relu^2 MLP as separate blocks; family head vs transformers 5.17, "
                             "2026-09-28; CPU, prompts stepped"),
    "lfm2":       ("ok", (), "LFM2: short convolutions (B*x through a causal depthwise conv, "
                             "times C) interleaved with Q/K-normed attention; family head 2026-09-28; "
                             "prompts stepped, CPU only"),
    "plamo2":     ("gap", ("hybrid", "ssm"), ""),

    # ===================== pure state space / recurrent ======================
    "mamba":      ("ok", (), "Mamba-1 (and falcon_mamba): family heads 2026-09-28; positions in order, "
                             "prompts stepped, CPU only"),
    "mamba2":     ("ok", (), "Mamba-2: state-space mixer (conv, per-head state, gated norm); "
                             "family head 2026-09-28; positions in order, prompts stepped, CPU only"),
    "rwkv6":      ("scope", ("ssm",), "recurrent"),
    "rwkv7":      ("scope", ("ssm",), "recurrent"),
    "rwkv6qwen2": ("scope", ("ssm",), "recurrent"),
    "arwkv7":     ("scope", ("ssm",), "recurrent"),

    # ===================== encoder / encoder-decoder =========================
    "bert":       ("scope", ("encdec",), "encoder only"),
    "neo_bert":   ("scope", ("encdec",), "encoder only"),
    "nomic_bert": ("scope", ("encdec",), "encoder only"),
    "jina_bert_v2": ("scope", ("encdec",), "encoder only"),
    "t5":         ("scope", ("encdec",), "encoder-decoder"),
    "t5encoder":  ("scope", ("encdec",), "encoder only"),

    # ===================== multimodal ========================================
    "idefics3":   ("ok", (), "VISION, GRADED 2026-09-24 on SmolVLM-256M (splitting off): "
                             "tower rows vs HF image features rel 1.5e-6; 'The circle is "
                             "red.' word for word. CPU, stepped prompt; sml_vlm_grade.py"),
    "qwen2vl":    ("scope", ("vision",), "text stack would run; the vision tower "
                                         "and mrope are not implemented"),
    "qwen3vl":    ("scope", ("vision", "qk_norm"), ""),
    "qwen3vlmoe": ("scope", ("vision", "moe", "qk_norm"), ""),
    "cogvlm":     ("scope", ("vision",), ""),
    "wavtokenizer_dec": ("scope", ("audio",), "audio codec decoder"),

    # ===================== diffusion language models =========================
    "dream":      ("scope", ("diffusion",), "not left-to-right"),
    "llada":      ("scope", ("diffusion",), "not left-to-right"),

    # ===================== other =============================================
    "chatglm":    ("gap", (), "partial rotary with an unusual split; needs a "
                              "checkpoint in hand before anything is claimed"),
    "glm4": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: GLM-4-0414 sandwich norms, interleaved partial rotary'),
    "plamo":      ("gap", (), "parallel layers sharing one norm"),
    "bitnet": ('gap', ('subnorm',), 'sub-norms inside attention and the MLP; ternary weights are a quantizer question on top'),
    "openelm":    ("gap", ("perlayer",), "per-layer head counts and FFN widths"),
    # ===================== found by diffing their llama-arch.cpp ============
    # This roster was first written from recollection; their source lists 121
    # canonical arch strings and 38 were absent here. Most of that gap was
    # naming (they use GGUF strings like "gptneox" and "command-r", we key on HF
    # model_type), but these are genuinely newer than what was recalled.
    #
    # Marked `unknown` rather than guessed. A wrong "ok" here would produce
    # exactly the failure this table exists to prevent.
    "gpt_oss": ('ok', (), 'attention sinks, router and expert biases, the clamped SwiGLU on interleaved gate/up (arch bit 10, CPU), MXFP4 originals decoded as transformers does; family head vs transformers 5.17, 2026-10-19'),
    "exaone_moe":   ("ok", (), "EXAONE-MoE: EXAONE 4 layers with grouped V3-style routing and shared experts; family head vs transformers 5.17, 2026-09-28"),
    "nemotron_h_moe": ("ok", (), "Nemotron 3 Nano: nemotron_h with ungated relu^2 experts, sigmoid routing; family head vs transformers 5.17, 2026-10-19"),
    "qwen3next":    ("ok", (), "Qwen3-Next: Gated DeltaNet linear attention (3 of 4 layers), gated "
                           "full attention, experts with a gated shared expert, (1+w) norms; family head "
                           "2026-09-28; CPU, prompts stepped"),
    "kimi_linear":  ("gap", ("hybrid",), "linear attention"),
    "glm_dsa":      ("unknown", (), "GLM variant; not established here"),
    "jais2":        ("ok", (), "Jais 2: checked against transformers 5.17 (family head), 2026-09-28"),
    "mistral3":     ("ok", (), "text path: a mistral text model inside a Pixtral vision wrapper "
                           "(Mistral Small 3.x, Devstral, Magistral); family head 2026-09-28. "
                           "Images are not run -- the vision tower is set aside by name"),
    "plamo3":       ("unknown", (), "plamo2 is hybrid; v3 not established"),
    "qwen35":       ("ok", (), "Qwen 3.5 (text): Gated DeltaNet linear attention 3:1 with gated full "
                           "attention, (1+w) norms, 25% rotary at 1e7 (rope_parameters); family head vs "
                           "transformers 5.17 2026-09-28; the vision tower and the multi-token-prediction "
                           "head are set aside; CPU, prompts stepped"),
    "qwen35moe":    ("ok", (), "Qwen 3.5-MoE (text): Qwen 3.5 layers with experts and a gated shared expert; family head vs transformers 5.17, 2026-09-28"),
    "step35":       ("unknown", (), "not established"),
    "maincoder":    ("unknown", (), "not established"),
    "rnd1":         ("unknown", (), "not established"),
    "mimo2":        ("unknown", (), "not established"),
    "llama_embed":  ("scope", ("encdec",), "embedding model, no decoding"),
    "modernbert":   ("scope", ("encdec",), "encoder only"),
    "jina_bert_v3": ("scope", ("encdec",), "encoder only"),
    "paddleocr":    ("scope", ("vision",), "OCR"),
    "clip":         ("scope", ("vision",), "vision encoder"),
    "starcoder": ('ok', (), 'same as gpt_bigcode under their GGUF name; family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28'),
    # ===================== found by the family heads (2026-09-28) ============
    "mistral": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: llama naming; sliding_window applied (it had been ignored)'),
    "ministral": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: llama with alternating windows'),
    "helium": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: interleaved rotary'),
    "glm": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: GLM-4 dense (glm-4-9b-chat-hf): interleaved partial rotary, fused gate|up'),
    "olmo3": ('ok', (), 'family head vs transformers 4.57.6 and 5.17 (rotary settings nested per layer type, read since 2026-10-13): OLMo-2 post-norms with alternating windows. The released OLMo 3 checkpoints apply YaRN (attention factor 1.21) to the global layers only: since 2026-10-13 the factor is folded into the Q/K norm weights of those layers and the local layers rotate unscaled (family head olmo3-yarn)'),
    "flex_olmo": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: OLMo-2 post-norms with experts'),
    "vaultgemma": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: Gemma norms, pre-norms only'),
    "granitemoeshared": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: Granite MoE plus a shared MLP (fused gate|up)'),
    "gpt_neo": ('ok', (), 'family head vs transformers 4.57.6 (tooling/sml_family_heads.py), 2026-09-28: learned positions, alternating local/global, unscaled scores'),
}

# What a verified family actually covers. `model_type` is what the checkpoint
# declares, and a great many popular models declare the same three.
COVERS = {
    "llama": ["Mistral", "Mixtral-dense", "Yi", "Vicuna", "Zephyr", "OpenChat",
              "SOLAR", "Nous-Hermes", "WizardLM", "TinyLlama", "CodeLlama",
              "Alpaca", "Orca", "Platypus", "OpenHermes", "Dolphin",
              "Llama-2", "Llama-3", "Llama-3.1", "Llama-3.2"],
    "qwen2": ["Qwen1.5", "Qwen2", "Qwen2.5", "QwQ", "Qwen2.5-Coder",
              "Qwen2.5-Math", "DeepSeek-R1-Distill-Qwen"],
    "gpt_neox": ["Pythia", "GPT-NeoX-20B", "Dolly-v2", "RedPajama-INCITE",
                 "StableLM-base"],
}

# transformers' model_type for a row that is keyed by the reference's name. The
# builder looks rows up by the checkpoint's model_type; without this, qwen2_moe
# and gemma3_text walked past their "gap" rows (found 2026-09-28).
ALIASES = {
    "qwen2_moe": "qwen2moe", "qwen3_moe": "qwen3moe", "gemma3_text": "gemma3",
    "hunyuan_v1_dense": "hunyuan_dense", "hunyuan_v1_moe": "hunyuan_moe",
    "phi": "phi2", "cohere": "command_r", "deepseek_v2": "deepseek2",
    "deepseek_v3": "deepseek2", "kimi_k2": "deepseek2", "llama4_text": "llama4", "qwen3_next": "qwen3next",
    "falcon_mamba": "mamba", "granitemoehybrid": "granite_hybrid",
    # found by sml_compat.py, 2026-10-13: these model_types were refused by the
    # build gate although their rows are 'ok' -- real Qwen 3.5 and LFM2-MoE
    # checkpoints could not be built at all
    "qwen3_5_text": "qwen35", "qwen3_5": "qwen35", "qwen3_5_moe_text": "qwen35moe", "qwen3_5_moe": "qwen35moe",
    "lfm2_moe": "lfm2moe", "gpt_refact": "refact", "PanguEmbedded": "pangu_embed",
}


def summary():
    from collections import Counter
    c = Counter(v[0] for v in ROSTER.values())
    need = Counter()
    for st, needs, _ in ROSTER.values():
        for k in needs:
            need[k] += 1
    return c, need


def fits_path():
    """Where auto-fits are saved (tooling/sml_autofit.py): MLMI_FITS, else the app's home."""
    import os
    return os.environ.get("MLMI_FITS") or os.path.join(
        os.environ.get("MLMI_HOME") or os.path.join(os.path.expanduser("~"), ".mlmi"), "fits.json")


def fit_for(model_type):
    """The saved auto-fit for a model_type ({"flags", "pass", "rel", "t"}), or None."""
    import json
    try:
        with open(fits_path(), encoding="utf-8") as f:
            return (json.load(f) or {}).get(model_type)
    except (OSError, ValueError):
        return None


def check(model_type, cfg=None):
    """(status, needs, note) for a model_type, or None if unlisted."""
    row = ROSTER.get(model_type) or ROSTER.get(ALIASES.get(model_type, ""))
    if row is None or row[0] == "unknown":
        # an architecture nobody listed, fitted to transformers' own code on this machine
        fit = fit_for(model_type)
        if fit and fit.get("pass"):
            return ("ok", (), f"auto-fitted on {str(fit.get('t', ''))[:10]}: rel {float(fit.get('rel', 0)):.3f}")
    if row is None:
        return None
    # falcon is the one row the checkpoint itself decides: ALiBi or RoPE.
    if model_type == "falcon" and cfg is not None and cfg.get("rope_theta"):
        return ("ok", (), "this checkpoint declares RoPE, not ALiBi")
    return row


def build_gate(model_type, cfg=None, override=None):
    """Return the roster row and whether this exact type may be built."""
    row = check(model_type, cfg)
    return row, (row is not None and row[0] == "ok") or override == model_type


def explain(model_type, cfg=None):
    row = check(model_type, cfg)
    if row is None:
        return (f"'{model_type}' is not in the roster. Nothing is known to be "
                f"missing, but nothing has been checked either -- build it and "
                f"grade against transformers before believing the output.")
    st, needs, note = row
    if st == "ok":
        return f"'{model_type}': supported on existing axes. {note}"
    if st == "unknown":
        return (f"'{model_type}': in the reference roster, behaviour NOT "
                f"established here. {note}. Build it and grade against "
                f"transformers before believing the output.")
    if st == "scope":
        return (f"'{model_type}': out of scope -- "
                + "; ".join(MISSING[k] for k in needs) + ".")
    miss = "; ".join(MISSING[k] for k in needs) if needs else note
    return f"'{model_type}': NOT supported. Needs {miss}."


# TRAP, hit while verifying llama and gpt_neox. Without --prompt the CLI
# generates from fixed token ids, and an id means a DIFFERENT token in every
# vocabulary. Llama-3.2 produced "IResolvable, IResolvable, IResolvable" and
# read exactly like a broken build; with a real prompt it answers "Paris. The
# capital city of the United States is Washington D.C." ALWAYS pass --prompt
# when validating a new architecture, or a working model will look broken --
# the inverse of this project's usual failure, where a broken model looks fine.


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            print(" ", explain(a))
    else:
        c, need = summary()
        print(f"roster: {len(ROSTER)} architectures\n")
        for k, lbl in (("ok", "runs today"), ("gap", "needs a feature"),
                       ("unknown", "behaviour not established"),
                       ("scope", "out of scope")):
            print(f"  {k:<6} {c[k]:>3}   {lbl}")
        print("\nmissing engine features, by architectures unlocked:")
        for k, v in need.most_common():
            if k in ("ssm", "encdec", "vision", "audio", "diffusion"):
                continue
            print(f"  {v:>2}  {k:<9} {MISSING[k]}")
        print("\nVERIFIED families and what they cover:")
        for fam, names in COVERS.items():
            print(f"  {fam:<10} {len(names):>2} known model families")
            print(f"             {', '.join(names[:8])}...")
