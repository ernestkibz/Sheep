"""A GPTQ or AWQ copy, or Meta's raw .pth checkpoint, to an original-format folder (float16
safetensors, config, tokenizer), so Sheep's normal build can read it (Part C6, 2026-10-21).

    python tooling/sml_unpack.py <folder> <out folder>

Used only when there is no original checkpoint to build from. GPTQ and AWQ integers are
unpacked here with their own scales and zero points, so neither the GPTQ nor the AWQ library
(CUDA kernels, often no Windows wheel) is needed; the quality is the copy's: a 4-bit copy
stays a 4-bit model, now held in float16. Meta's consolidated.*.pth shards (Llama 1, 2, 3)
are merged and renamed the way transformers' own Llama conversion does. Run by a PyTorch
runtime. Prints "converting: ..." progress lines and one line "RESULT {json}".
"""
import glob
import json
import os
import shutil
import sys

SHARD_BYTES = 2_000_000_000
KEEP = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json", "vocab.json",
        "merges.txt", "vocab.txt", "generation_config.json", "chat_template.json", "chat_template.jinja",
        "added_tokens.json", "LICENSE", "LICENSE.txt", "LICENSE.md", "USE_POLICY.md", "README.md", "NOTICE")


def say(line):
    print("converting: " + line, flush=True)


class Writer:
    """float16 tensors out in ~2 GB safetensors shards, with the index transformers reads."""
    def __init__(self, out):
        self.out, self.cur, self.size, self.files, self.map = out, {}, 0, [], {}

    def add(self, name, t):
        import torch
        t = t.to(torch.float16) if t.is_floating_point() else t
        t = t.contiguous()
        n = t.numel() * t.element_size()
        if self.cur and self.size + n > SHARD_BYTES:
            self.flush()
        self.cur[name] = t
        self.size += n

    def flush(self):
        from safetensors.torch import save_file
        if not self.cur:
            return
        f = f"model-{len(self.files) + 1:05d}.safetensors"
        save_file(self.cur, os.path.join(self.out, f), metadata={"format": "pt"})
        self.files.append(f)
        self.map.update({k: f for k in self.cur})
        self.cur, self.size = {}, 0

    def close(self):
        self.flush()
        total = len(self.files)
        names = {f: f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i, f in enumerate(self.files)}
        if total == 1:
            names = {self.files[0]: "model.safetensors"}
        for old, new in names.items():
            os.replace(os.path.join(self.out, old), os.path.join(self.out, new))
        if total > 1:
            with open(os.path.join(self.out, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
                json.dump({"metadata": {}, "weight_map": {k: names[v] for k, v in sorted(self.map.items())}}, fh, indent=1)


def _unpack_ints(packed, bits, dim):
    """int32 words to their `32 // bits` integers, laid out along `dim` (0: rows, 1: columns)."""
    import torch
    per = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    if dim == 0:      # [rows/per, cols] -> [rows, cols]
        w = (packed.unsqueeze(1) >> shifts.view(1, -1, 1)) & ((1 << bits) - 1)
        return w.reshape(-1, packed.shape[1])
    w = (packed.unsqueeze(2) >> shifts.view(1, 1, -1)) & ((1 << bits) - 1)
    return w.reshape(packed.shape[0], -1)


AWQ_REVERSE = (0, 4, 1, 5, 2, 6, 3, 7)


def dequant_gptq(t, qc):
    """GPTQ (AutoGPTQ / GPTQModel / optimum layout) to a float16 [out, in] weight."""
    import torch
    bits = int(qc.get("bits", 4))
    qweight, qzeros, scales = t["qweight"], t["qzeros"], t["scales"].float()
    w = _unpack_ints(qweight, bits, 0)                          # [in, out]
    z = _unpack_ints(qzeros, bits, 1)[:, :w.shape[1]]           # [groups, out]
    if str(qc.get("checkpoint_format") or "gptq").lower() != "gptq_v2":
        z = z + 1                                                # the v1 files store zero - 1
    g = t.get("g_idx")
    if g is None or not qc.get("desc_act"):
        gs = int(qc.get("group_size") or -1)
        gs = w.shape[0] if gs <= 0 else gs
        g = torch.arange(w.shape[0]) // gs
    g = g.long()
    return ((w - z[g]).float() * scales[g]).t()


def dequant_awq(t, qc):
    """AWQ GEMM layout (AutoAWQ, the hugging-quants and Qwen pages) to a float16 [out, in] weight."""
    import torch
    bits = int(qc.get("bits", qc.get("w_bit", 4)))
    qweight, scales = t["qweight"], t["scales"].float()
    per = 32 // bits
    order = torch.arange(qweight.shape[1] * per).view(-1, per)[:, list(AWQ_REVERSE)].reshape(-1)
    w = _unpack_ints(qweight, bits, 1)[:, order]                 # [in, out]
    if "qzeros" in t:
        z = _unpack_ints(t["qzeros"], bits, 1)[:, order]
    else:
        z = torch.full((scales.shape[0], w.shape[1]), 1 << (bits - 1), dtype=torch.int32)
    gs = int(qc.get("group_size", qc.get("q_group_size", 128)) or -1)
    gs = w.shape[0] if gs <= 0 else gs
    g = torch.arange(w.shape[0]) // gs
    return ((w - z[g]).float() * scales[g]).t()


PARTS = ("qweight", "qzeros", "scales", "g_idx", "wf", "bias")


def unpack_quantized(src, out, cfg):
    from safetensors import safe_open
    qc = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    method = str(qc.get("quant_method") or "").lower()
    bits = int(qc.get("bits", qc.get("w_bit", 4)))
    if bits not in (2, 4, 8):
        raise ValueError(f"{bits}-bit {method.upper()} packing is not one Sheep can unpack (2, 4 and 8 bits are)")
    if method == "awq" and str(qc.get("version") or "gemm").lower() != "gemm":
        raise ValueError(f"AWQ's {qc.get('version')} layout is not one Sheep can unpack (GEMM is)")
    fn = dequant_awq if method == "awq" else dequant_gptq
    files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not files:
        raise ValueError("no .safetensors files in the folder")
    w, n = Writer(out), 0
    for f in files:
        say(f"unpacking {method.upper()} {os.path.basename(f)}")
        with safe_open(f, "pt") as h:
            keys = list(h.keys())
            prefixes = sorted({k[:-len(".qweight")] for k in keys if k.endswith(".qweight")})
            pset = set(prefixes)
            for p in prefixes:
                parts = {s: h.get_tensor(f"{p}.{s}") for s in PARTS if f"{p}.{s}" in keys}
                w.add(p + ".weight", fn(parts, qc))
                if "bias" in parts:
                    w.add(p + ".bias", parts["bias"])
                n += 1
            for k in keys:
                pre, _, last = k.rpartition(".")
                if pre in pset and last in PARTS:
                    continue
                w.add(k, h.get_tensor(k))
    w.close()
    if not n:
        raise ValueError(f"no packed {method.upper()} weights found")
    cfg = dict(cfg)
    cfg.pop("quantization_config", None)
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"] = {k: v for k, v in cfg["text_config"].items() if k != "quantization_config"}
    cfg["torch_dtype"] = "float16"
    return cfg, f"{n} {method.upper()} layers unpacked"


# ---- Meta's raw checkpoints (consolidated.*.pth + params.json), as transformers' convert_llama_weights_to_hf

def _permute(w, heads, rows, cols):
    """Meta's rotary layout (pairs side by side) to transformers' (halves)."""
    return w.view(heads, rows // heads // 2, 2, cols).transpose(1, 2).reshape(rows, cols)


LLAMA31_SPECIAL = (["<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>",
                    "<|reserved_special_token_1|>", "<|finetune_right_pad_id|>", "<|reserved_special_token_2|>",
                    "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>", "<|eot_id|>", "<|python_tag|>"] +
                   [f"<|reserved_special_token_{i}|>" for i in range(3, 248)])
LLAMA3_SPECIAL = (["<|begin_of_text|>", "<|end_of_text|>"] + [f"<|reserved_special_token_{i}|>" for i in range(4)] +
                  ["<|start_header_id|>", "<|end_header_id|>", "<|reserved_special_token_4|>", "<|eot_id|>"] +
                  [f"<|reserved_special_token_{i}|>" for i in range(5, 251)])
LLAMA3_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
                  r"\s*[\r\n]+|\s+(?!\S)|\s+")
LLAMA3_TEMPLATE = ("{{ bos_token }}{% for m in messages %}{{ '<|start_header_id|>' + m['role'] + '<|end_header_id|>\n\n' "
                   "+ m['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}"
                   "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}")
LLAMA2_TEMPLATE = ("{{ bos_token }}{% for m in messages %}{% if m['role'] == 'user' %}{{ '[INST] ' + m['content'] | trim "
                   "+ ' [/INST]' }}{% elif m['role'] == 'system' %}{{ '<<SYS>>\n' + m['content'] | trim + '\n<</SYS>>\n\n' }}"
                   "{% else %}{{ ' ' + m['content'] | trim + ' ' + eos_token }}{% endif %}{% endfor %}")


def _meta_tokenizer(src, out, scaled_rope):
    """Llama 1/2's SentencePiece or Llama 3's tiktoken tokenizer.model, as transformers files.
    Returns (vocab size, bos id, eos ids)."""
    import re
    path = os.path.join(src, "tokenizer.model")
    if not os.path.exists(path):
        raise ValueError("no tokenizer.model beside the .pth files")
    with open(path, "rb") as fh:
        first = fh.readline(200).strip()
    if not re.fullmatch(rb"[A-Za-z0-9+/=]+ \d+", first):       # SentencePiece (Llama 1, 2), not tiktoken lines
        from transformers import AutoTokenizer
        shutil.copy2(path, os.path.join(out, "tokenizer.model"))
        with open(os.path.join(out, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
            json.dump({"tokenizer_class": "LlamaTokenizer", "legacy": False, "add_bos_token": True}, fh)
        tok = AutoTokenizer.from_pretrained(out)
        tok.chat_template = LLAMA2_TEMPLATE
        tok.save_pretrained(out)
        return len(tok), tok.bos_token_id, [tok.eos_token_id]
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import TikTokenConverter
    special = LLAMA31_SPECIAL if scaled_rope else LLAMA3_SPECIAL
    fast = TikTokenConverter(path, pattern=LLAMA3_PATTERN, additional_special_tokens=special).converted()
    if any(fast.token_to_id(t) is None for t in special):   # Transformers 5 leaves them out: added in order
        from tokenizers import AddedToken
        fast.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in special])
    tok = PreTrainedTokenizerFast(tokenizer_object=fast, bos_token="<|begin_of_text|>",
                                  eos_token="<|eot_id|>", model_input_names=["input_ids", "attention_mask"],
                                  clean_up_tokenization_spaces=True)
    tok.chat_template = LLAMA3_TEMPLATE
    tok.save_pretrained(out)
    ids = tok.convert_tokens_to_ids
    eos = [ids("<|end_of_text|>"), ids("<|eot_id|>")] + ([ids("<|eom_id|>")] if scaled_rope else [])
    return len(tok), ids("<|begin_of_text|>"), eos


def unpack_meta(src, out):
    import torch
    with open(os.path.join(src, "params.json"), encoding="utf-8") as fh:
        p = json.load(fh)
    p = p.get("model", p)
    if p.get("moe") or p.get("vision_chunk_size") or p.get("n_experts"):
        raise ValueError("this Meta checkpoint is not a plain Llama text model (mixture or vision); look for its "
                         "Hugging Face-format original")
    dim, layers, heads = int(p["dim"]), int(p["n_layers"]), int(p["n_heads"])
    kv = int(p.get("n_kv_heads") or heads)
    hd = dim // heads
    _n, bos, eos = _meta_tokenizer(src, out, bool(p.get("use_scaled_rope")))
    hidden = int(8 * dim / 3)
    if p.get("ffn_dim_multiplier"):
        hidden = int(float(p["ffn_dim_multiplier"]) * hidden)
    m = int(p.get("multiple_of") or 256)
    hidden = m * ((hidden + m - 1) // m)
    files = sorted(glob.glob(os.path.join(src, "consolidated.*.pth")))
    if not files:
        raise ValueError("no consolidated.*.pth files")
    say(f"reading {len(files)} Meta .pth shard(s)")
    shards = [torch.load(f, map_location="cpu", mmap=True, weights_only=True) for f in files]
    vocab = sum(s["output.weight"].shape[0] for s in shards)        # params.json says -1 on Llama 2
    llama3 = vocab > 100_000

    def cat(name, d):
        ts = [s[name] for s in shards]
        return ts[0] if len(ts) == 1 else torch.cat(ts, dim=d)

    w = Writer(out)
    w.add("model.embed_tokens.weight", cat("tok_embeddings.weight", 0 if llama3 else 1))
    for i in range(layers):
        if i % 8 == 0:
            say(f"layer {i + 1} of {layers}")
        a, hf = f"layers.{i}.", f"model.layers.{i}."
        w.add(hf + "self_attn.q_proj.weight", _permute(cat(a + "attention.wq.weight", 0), heads, dim, dim))
        w.add(hf + "self_attn.k_proj.weight", _permute(cat(a + "attention.wk.weight", 0), kv, kv * hd, dim))
        w.add(hf + "self_attn.v_proj.weight", cat(a + "attention.wv.weight", 0))
        w.add(hf + "self_attn.o_proj.weight", cat(a + "attention.wo.weight", 1))
        w.add(hf + "mlp.gate_proj.weight", cat(a + "feed_forward.w1.weight", 0))
        w.add(hf + "mlp.down_proj.weight", cat(a + "feed_forward.w2.weight", 1))
        w.add(hf + "mlp.up_proj.weight", cat(a + "feed_forward.w3.weight", 0))
        w.add(hf + "input_layernorm.weight", shards[0][a + "attention_norm.weight"])
        w.add(hf + "post_attention_layernorm.weight", shards[0][a + "ffn_norm.weight"])
    w.add("model.norm.weight", shards[0]["norm.weight"])
    w.add("lm_head.weight", cat("output.weight", 0))
    w.close()
    rope = None
    if p.get("use_scaled_rope"):
        rope = {"rope_type": "llama3", "factor": float(p.get("rope_scaling_factor") or 8.0), "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": dim,
           "intermediate_size": hidden, "num_hidden_layers": layers, "num_attention_heads": heads,
           "num_key_value_heads": kv, "head_dim": hd, "vocab_size": vocab, "rms_norm_eps": float(p.get("norm_eps", 1e-5)),
           "rope_theta": float(p.get("rope_theta", 500000.0 if llama3 else 10000.0)), "rope_scaling": rope,
           "max_position_embeddings": 131072 if rope else (8192 if llama3 else 4096), "hidden_act": "silu",
           "tie_word_embeddings": False, "attention_bias": False, "mlp_bias": False, "torch_dtype": "float16",
           "bos_token_id": bos, "eos_token_id": eos[0] if len(eos) == 1 else eos}
    with open(os.path.join(out, "generation_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"bos_token_id": bos, "eos_token_id": eos[0] if len(eos) == 1 else eos}, fh, indent=1)
    return cfg, f"Llama, {layers} layers, from {len(files)} Meta shard(s)"


def main():
    src, out = sys.argv[1], sys.argv[2]
    try:
        tmp = out + ".part"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for f in KEEP:
            if os.path.exists(os.path.join(src, f)):
                shutil.copy2(os.path.join(src, f), os.path.join(tmp, f))
        if os.path.exists(os.path.join(src, "params.json")) and glob.glob(os.path.join(src, "consolidated.*.pth")):
            for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
                if os.path.exists(os.path.join(tmp, f)):
                    os.remove(os.path.join(tmp, f))
            cfg, what = unpack_meta(src, tmp)
            kind = "meta"
        else:
            with open(os.path.join(src, "config.json"), encoding="utf-8") as fh:
                cfg = json.load(fh)
            cfg, what = unpack_quantized(src, tmp, cfg)
            kind = "quantized"
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1)
        shutil.rmtree(out, ignore_errors=True)
        os.replace(tmp, out)
        print("RESULT " + json.dumps({"ok": True, "model_type": cfg.get("model_type"), "kind": kind, "what": what}),
              flush=True)
    except Exception as e:
        print("RESULT " + json.dumps({"ok": False, "why": f"{type(e).__name__}: {str(e)[:300]}"}), flush=True)


if __name__ == "__main__":
    main()
