#!/usr/bin/env python3
"""Family heads against llama.cpp: a family is WON only when its head beats the
reference on quality, CPU speed and GPU speed -- then every child of the family
inherits a config that was earned, not assumed (Ernest, 2026-09-30).

    python tooling/sml_family_race.py [--only llama,qwen3] [--rounds 6] [--keep DIR]

Per family, from ONE set of weights:
  1. the head at race size (hidden 1024, 8 layers, ~150M weights), random but
     scaled like trained weights, built from transformers' own code
  2. Sheep: sml_build, the shipping recipe (flat, 4-bit, attn_v/ffn_down 6-bit)
  3. llama.cpp: THEIR converter (convert_hf_to_gguf.py) and THEIR quantizer, Q4_K_M
  4. quality: llama-perplexity saves its own distributions for 12 windows of held-
     out text; Sheep and full-precision transformers are run on exactly those
     tokens; all three pass through llama.cpp's own 16-bit log-probability
     packing, so no side gains from rounding. Scored by KL divergence from full
     precision and top-1 agreement, paired per position with a bootstrap CI.
  5. speed: sml_cpu_race_ci.py and sml_gpu_race_ci.py, alternating order,
     prompt (pp128) and decode, same threads.

Random weights stand in for trained ones -- the family-head method: the head
proves the family's arithmetic and its speed; children only change sizes.
"""
import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import sml_family_heads as FH  # noqa: E402

CLI = FH.CLI
GPU = os.path.join(ROOT, "sml", "sml-gpu-fwd" + (".exe" if os.name == "nt" else ""))
CONVERT = os.environ.get("REF_CONVERT", r"C:\Users\earne\reference\llama-convert\convert_hf_to_gguf.py")
QUANTIZE = os.environ.get("REF_QUANTIZE", r"C:\Users\earne\llama-cuda\llama-quantize.exe")
PERPLEXITY = os.environ.get("REF_PPL", r"C:\Users\earne\llama-cpu\llama-perplexity.exe")
TEXT = os.path.join(ROOT, "RESULTS.md")
OUT = os.path.join(HERE, "family_race.json")
N_CTX, CHUNKS = 128, 12

RACE_SIZE = dict(hidden_size=1024, n_embd=1024, d_model=1024,
                 num_hidden_layers=8, n_layer=8, n_layers=8,
                 num_attention_heads=16, n_head=16, n_heads=16, num_key_value_heads=4,
                 intermediate_size=2816, n_inner=2816, ffn_hidden_size=2816,
                 max_position_embeddings=2048, n_positions=2048)
# head_dim is NOT set: each family keeps its own (Qwen3 128, Gemma 256, most
# hidden/heads). The children inherit the head's shape, so the head races in it.
# expert and side widths in the head list are sized for a 256-wide model;
# Q4_K needs rows in multiples of 256, so they grow to 768/1536
WIDE_KEYS = ("moe_intermediate_size", "shared_expert_intermediate_size", "shared_intermediate_size",
             "intermediate_size", "intermediate_size_mlp", "ffn_hidden_size")


def scale_over(over):
    out = dict(over)
    for k in WIDE_KEYS:
        if isinstance(out.get(k), int) and out[k] < 768:
            out[k] = out[k] * 3
    return out


def run(cmd, timeout=3600, env=None, cwd=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=e, cwd=cwd,
                          encoding="utf-8", errors="replace")


from sml_recipe import q4km_alloc, sheep_alloc  # noqa: E402,F401  (THE allocation, one source)


# ------------------------------------------------------------ llama.cpp's packing

def pack_like_llama(logits):
    """llama-perplexity's log_softmax packing (tools/perplexity): the range is
    [max-16, max], 65535 steps, anything below the floor at the floor. Applied to
    every side so none gains or loses by the rounding."""
    mx = logits.max()
    mn = max(logits.min(), mx - 16.0)
    lse = mx + np.log(np.exp(logits - mx).sum())
    scale = (mx - mn) / 65535.0
    q = np.where(logits > mn, np.rint((logits - mn) / scale), 0.0) if scale > 0 else np.zeros_like(logits)
    return q * scale + (mn - lse)                      # log-probabilities


def read_llama_logits(path):
    """(n_ctx, tokens[n_chunk, n_ctx], logprobs[n_chunk, n_ctx-1-first, n_vocab])"""
    with open(path, "rb") as f:
        assert f.read(8) == b"_logits_", "not a llama-perplexity logits file"
        n_ctx, = struct.unpack("<i", f.read(4))
        n_vocab, n_chunk = struct.unpack("<ii", f.read(8))
        toks = np.frombuffer(f.read(4 * n_chunk * n_ctx), dtype=np.int32).reshape(n_chunk, n_ctx)
        first = n_ctx // 2
        n_tok = n_ctx - 1 - first
        nv = 2 * ((n_vocab + 1) // 2) + 4
        raw = np.frombuffer(f.read(2 * n_chunk * n_tok * nv), dtype=np.uint16).reshape(n_chunk, n_tok, nv)
    d = raw[..., :4].copy().view(np.float32)             # (scale, min_log_prob) per row
    q = raw[..., 4:4 + n_vocab].astype(np.float64)
    lp = q * d[..., 0:1].astype(np.float64) + d[..., 1:2].astype(np.float64)
    return n_ctx, toks, lp


def gguf_add_bos(path):
    try:
        from gguf import GGUFReader
        r = GGUFReader(path)
        f = r.fields.get("tokenizer.ggml.add_bos_token")
        bos = r.fields.get("tokenizer.ggml.bos_token_id")
        add = bool(f.parts[f.data[0]][0]) if f else False
        bid = int(bos.parts[bos.data[0]][0]) if bos else None
        return add, bid
    except Exception:
        return False, None


def boot_ci(d, n=4000, seed=5):
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def quality(hf_dir, model, ssf, gguf, work):
    lfile = os.path.join(work, "ref_logits.bin")
    r = run([PERPLEXITY, "-m", gguf, "-f", TEXT, "-c", str(N_CTX), "--chunks", str(CHUNKS), "-t", "6",
             "-b", str(N_CTX), "--kl-divergence-base", lfile], timeout=1800)
    if not os.path.exists(lfile):
        return {"error": "llama-perplexity: " + (r.stdout + r.stderr).strip()[-300:]}
    n_ctx, toks, llp = read_llama_logits(lfile)
    add_bos, bos_id = gguf_add_bos(gguf)
    first = n_ctx // 2
    import torch
    kl = {"sheep": [], "llama": []}
    top = {"sheep": 0, "llama": 0}
    n = 0
    for c in range(toks.shape[0]):
        ids = [int(t) for t in toks[c]]
        if add_bos and bos_id is not None:
            ids[0] = bos_id                               # what llama-perplexity fed its model
        with torch.no_grad():
            ref = model(torch.tensor([ids])).logits[0].double().numpy()
        out = os.path.join(work, "sheep_rows.bin")
        s = run([CLI, ssf, "--tokens", ",".join(map(str, ids)), "--gen", "0", "--threads", "6", "--logits", out],
                env={"SML_ALL_LOGITS": str(first)}, timeout=900)
        if s.returncode != 0 or not os.path.exists(out):
            return {"error": "sheep: " + (s.stdout + s.stderr).strip()[-300:]}
        meta = np.fromfile(out, dtype=np.int32, count=2)
        rows = np.fromfile(out, dtype=np.float32, offset=8).reshape(int(meta[1]), int(meta[0])).astype(np.float64)
        for i in range(n_ctx - 1 - first):
            pos = first + i
            pr = pack_like_llama(ref[pos])
            p = np.exp(pr)
            for who, lp in (("sheep", pack_like_llama(rows[pos - first])), ("llama", llp[c, i])):
                kl[who].append(float((p * (pr - lp)).sum()))
                top[who] += int(np.argmax(lp) == np.argmax(pr))
            n += 1
    ks, kl_ = np.array(kl["sheep"]), np.array(kl["llama"])
    lo, hi = boot_ci(ks - kl_)
    return {"kld_sheep": float(ks.mean()), "kld_llama": float(kl_.mean()),
            "top1_sheep": top["sheep"] / n, "top1_llama": top["llama"] / n, "positions": n,
            "kld_diff_ci": [lo, hi], "won": hi < 0, "lost": lo > 0}


def race(tool, ssf, gguf, extra, rounds):
    r = run([sys.executable, os.path.join(HERE, tool), ssf, gguf, "--rounds", str(rounds)] + extra, timeout=5400)
    txt = r.stdout + r.stderr
    m = re.findall(r"median ratio ([\d.]+), 95% CI \[([\d.]+), ([\d.]+)\].*-> (WON|LOST|NOT DECIDED)", txt)
    if not m:
        excl = re.findall(r"EXCLUDED \(sheep (\S+), ref (\S+)\)", txt)
        return {"error": "no result" + (f" (sheep {excl[0][0]}, ref {excl[0][1]})" if excl else ""),
                "tail": txt.strip()[-400:]}
    med, lo, hi, v = m[-1]
    return {"ratio": float(med), "ci": [float(lo), float(hi)], "verdict": v}


def gpu_refusal(ssf):
    """The GPU path's own reason when it will not run a file (it refuses rather
    than computing a different model)."""
    r = run([GPU, os.path.abspath(ssf), "1,2,3", "1"], timeout=300, env={"SPEED_ONLY": "1"},
            cwd=os.path.dirname(GPU))          # it loads its module from its own folder
    txt = (r.stdout + r.stderr)
    m = re.search(r"(refus\w*|not supported|unsupported)[^\n]*", txt, re.I)
    return m.group(0)[:200] if (r.returncode != 0 and m) else ("" if r.returncode == 0 else txt.strip()[-200:])


# Gemma's and Nemotron's converters read ONLY a SentencePiece tokenizer.model.
# The race feeds llama.cpp's own token ids to every side (Sheep, transformers),
# so any valid SentencePiece vocabulary serves; this one was trained once from
# this repository's own documents (4000 pieces, byte fallback) and is kept beside
# the harness so every run tokenizes the same way. The converter pads the
# vocabulary out to the head's 50257 rows.
SPM = os.path.join(HERE, "family_spm.model")


def convert(d, f16):
    """llama.cpp's converter, given the tokenizer file it insists on. The model
    is NAMED neutrally: their vocabulary loader patches tokens by model name
    ("phi3" in the name makes it look up <s>, </s> and <unk>, which a GPT-2
    vocabulary does not have, and the load fails) -- the folder name would
    otherwise become the model's name."""
    cmd = [sys.executable, CONVERT, d, "--outtype", "f16", "--outfile", f16, "--model-name", "family-head"]
    c = run(cmd, timeout=1800)
    if not os.path.exists(f16) and ("_create_vocab_sentencepiece" in (c.stdout + c.stderr)
                                    or "Cannot find Llama BPE tokenizer" in (c.stdout + c.stderr)):
        shutil.copyfile(SPM, os.path.join(d, "tokenizer.model"))
        c = run(cmd, timeout=1800)
    if not os.path.exists(f16) and "opt_num_mtp_layers" in (c.stdout + c.stderr):
        # Qwen3-Next's converter expects multi-token-prediction layers unless
        # told there are none -- a head has none
        c = run(cmd + ["--no-mtp"], timeout=1800)
    return c


# The shape real checkpoints of a family have, where llama.cpp only accepts that
# shape (commandment VII: each side given what it needs). Phi-1/2 are
# multi-head (their loader expects K and V as wide as Q); Command-R ties its
# head to the embedding (their converter has no place for an lm_head).
REFERENCE_SHAPE = {
    # Phi-2 and Falcon: llama.cpp's loader fixes the FFN at 4 x hidden, as the
    # real checkpoints have it
    "phi": {"mha": True, "ff4": True},
    "falcon": {"ff4": True},
    # OLMoE and dots.llm1 are multi-head; llama.cpp's loader sizes K (and OLMoE's
    # K norm) as wide as Q
    "olmoe": {"mha": True},
    # the state-space converters pad the vocabulary to a multiple of 8 or 16 and
    # expect the embedding padded with it, as real checkpoints ship it
    **{k: {"vocab": 50304} for k in ("mamba", "falcon_mamba", "mamba2", "granitemoehybrid",
                                     "granitemoehybrid-nope", "nemotron_h", "bamba", "jamba",
                                     "falcon_h1")},
    "dots1": {"mha": True},
    "cohere": {"tied": True},
    "cohere2": {"tied": True},
    # GLM's converter names <|user|> as the end-of-turn token and needs it in the
    # added vocabulary; real GLM vocabularies are padded past their tokens
    # real LFM2 has no bias on its short convolution (the converter squeezes
    # every conv tensor, a bias included)
    "lfm2": {"over": {"conv_bias": False}},
    "lfm2_moe": {"over": {"conv_bias": False}},
    "glm": {"added": ["<|user|>"], "vocab": 50304},
    "glm4": {"added": ["<|user|>"], "vocab": 50304},
    "glm4_moe": {"added": ["[gMASK]", "<|user|>", "<|observation|>"], "vocab": 50304},
}


TINY = dict(FH.BASE)        # the family sweep's own size (hidden 256, two layers)


def prepare_head(label, mt, over, d, size="race"):
    """The family's head in d, at race size (or the sweep's tiny size, for a
    preflight), with what llama.cpp's converter needs."""
    FH.BASE.clear()
    FH.BASE.update(TINY)
    if size == "race":
        FH.BASE.update(RACE_SIZE)
        FH.BASE.pop("head_dim", None)
        over = scale_over(over)
    rs = REFERENCE_SHAPE.get(label, {})
    if rs:
        over = dict(over)
        if rs.get("mha"):
            over["num_key_value_heads"] = FH.BASE["num_attention_heads"]
        if rs.get("tied"):
            over["tie_word_embeddings"] = True
        if rs.get("vocab"):
            over["vocab_size"] = rs["vocab"]
        if rs.get("ff4"):
            over["intermediate_size"] = over["ffn_hidden_size"] = 4 * FH.BASE["hidden_size"]
        over.update(rs.get("over", {}))
    if size == "race":
        # state-space heads state their mixer width in heads x head size; at race
        # size it must still be expand x hidden
        over = dict(over)
        for nk, dk, ek in (("mamba_n_heads", "mamba_d_head", "mamba_expand"),
                           ("mamba_num_heads", "mamba_head_dim", "expand"),
                           ("num_heads", "head_dim", "expand")):
            if nk in over and dk in over:
                over[nk] = int(over.get(ek, 2)) * FH.BASE["hidden_size"] // int(over[dk])
    model = FH.build_head(label, mt, over, d, seed=1234)
    if rs.get("added"):
        tj = os.path.join(d, "tokenizer.json")
        with open(tj, encoding="utf-8") as f:
            tk = json.load(f)
        nid = max(max(tk["model"]["vocab"].values()), max(a["id"] for a in tk.get("added_tokens", [])))
        for t in rs["added"]:
            nid += 1
            tk.setdefault("added_tokens", []).append(
                {"id": nid, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                 "normalized": False, "special": True})
        with open(tj, "w", encoding="utf-8") as f:
            json.dump(tk, f, ensure_ascii=False)
    # The local GPT-2 tokenizer.json predates the "type" field every newer file
    # carries; llama.cpp's converter reads it and would stop on a KeyError rather
    # than take its own GPT-2 path. Same tokenizer, spelled the current way.
    tj = os.path.join(d, "tokenizer.json")
    if os.path.exists(tj):
        with open(tj, encoding="utf-8") as f:
            tk = json.load(f)
        if "type" not in tk.get("model", {}):
            tk["model"] = {"type": "BPE", **tk["model"]}
            with open(tj, "w", encoding="utf-8") as f:
                json.dump(tk, f, ensure_ascii=False)
    # llama.cpp's GPT-2 converter reads n_ctx, which transformers no longer writes
    cj = os.path.join(d, "config.json")
    with open(cj, encoding="utf-8") as f:
        cfgj = json.load(f)
    if cfgj.get("model_type") == "gpt2" and "n_ctx" not in cfgj:
        cfgj["n_ctx"] = cfgj.get("n_positions", 2048)
    # Falcon-7B-style heads (one shared K/V head, not the new decoder): transformers
    # writes num_kv_heads = n_head and ignores it; llama.cpp's converter reads it.
    # The true value is 1, as real Falcon-7B configs state by leaving it out.
    if cfgj.get("model_type") == "falcon" and cfgj.get("multi_query")             and not cfgj.get("new_decoder_architecture"):
        cfgj["num_kv_heads"] = 1
    # transformers' own MPT class is MptForCausalLM; llama.cpp's converter knows
    # the checkpoint by MosaicML's class name, MPTForCausalLM -- same tensors
    if cfgj.get("model_type") == "mpt":
        cfgj["architectures"] = ["MPTForCausalLM"]
    # HunYuan-A13B's config states its expert settings per layer; llama.cpp's
    # converter reads those lists. transformers builds every expert and the one
    # shared MLP at intermediate_size -- the same numbers, spelled per layer.
    if cfgj.get("model_type") == "hunyuan_v1_moe":
        nl = int(cfgj["num_hidden_layers"])
        k_ = cfgj.get("moe_topk")
        cfgj.setdefault("moe_intermediate_size", [int(cfgj["intermediate_size"])] * nl)
        cfgj["moe_topk"] = k_ if isinstance(k_, list) else [int(k_)] * nl
        cfgj.setdefault("num_shared_expert", [1] * nl)
    # Granite-4.0-H with no experts: nothing is routed, and llama.cpp's loader
    # asserts experts-used <= experts
    if cfgj.get("model_type") == "granitemoehybrid" and not cfgj.get("num_local_experts"):
        cfgj["num_experts_per_tok"] = 0
    with open(cj, "w", encoding="utf-8") as f:
        json.dump(cfgj, f, indent=1)
    # Name the tokenizer class: without it transformers picks one from the model
    # type, and a llama head's LlamaTokenizerFast adds <s> </s> <unk> past the
    # 50257 rows -- which llama.cpp's converter rightly refuses.
    with open(os.path.join(d, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "GPT2Tokenizer", "model_max_length": 2048,
                   "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
                   "unk_token": "<|endoftext|>", "add_bos_token": False}, f)
    return model


def sheep_build(d, ssf, alloc, env=None, budget=0):
    """THE recipe (commandment II): one command for every family. With the
    reference's file size as the budget, the bytes the allocation leaves under
    it widen more tensors to 6 bits (sml_build.promote_order); the stored group
    fields are neighbour-searched (SML_FIELD_SEARCH, radius 2 by default)."""
    return run([sys.executable, FH.BUILD, d, ssf, "--layout", "flat", "--alloc", alloc,
                "--no-gptq", "--scale6", "--seqs", "16", "--seq-len", "64"]
               + (["--budget", str(budget)] if budget else []), timeout=5400, env=env)


def race_family(label, mt, over, work, rounds, do_gpu, reuse=False, build_only=False):
    d = os.path.join(work, label)
    ssf = os.path.join(d, "sheep.ssf")
    f16, q4 = os.path.join(d, "ref-f16.gguf"), os.path.join(d, "ref-q4km.gguf")
    if not reuse:
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    res = {"label": label, "model_type": mt, "t": time.strftime("%Y-%m-%d %H:%M")}
    try:
        model = prepare_head(label, mt, over, d)
    except Exception as e:
        res["error"] = f"transformers: {type(e).__name__}: {e}"[:300]
        return res
    res["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    t0 = time.time()
    n_layer = int(getattr(model.config, "num_hidden_layers", 0) or getattr(model.config, "n_layer", 0) or 8)
    res["alloc"] = q4km_alloc(n_layer)
    if not (reuse and os.path.exists(ssf)):
        # 6-bit group fields with negative-only mins (--scale6, MODEL_MINNEG):
        # Q4_K's metadata size. Llama head: 126.00 MB vs 126.97, KL 0.0279 vs 0.0286.
        # No GPTQ on a random-weight head: it fits the structure of trained
        # activations, which random weights do not have. Measured on the Llama
        # head: GPTQ KL 0.0446, plain rounding 0.0288, llama.cpp Q4_K_M 0.0286.
        b = sheep_build(d, ssf, res["alloc"])
        if b.returncode != 0 or not os.path.exists(ssf):
            res["error"] = "sheep build: " + " | ".join((b.stdout + b.stderr).strip().splitlines()[-3:])[:300]
            return res
        res["sheep_build_s"] = round(time.time() - t0)
    if reuse and os.path.exists(q4):
        return finish(res, model, d, ssf, q4, rounds, do_gpu)
    c = convert(d, f16)
    if not os.path.exists(f16):
        res["error"] = "llama.cpp converter: " + " | ".join((c.stdout + c.stderr).strip().splitlines()[-3:])[:300]
        res["reference_cannot_load"] = True
        return res
    q = run([QUANTIZE, f16, q4, "Q4_K_M"], timeout=1800)
    if not os.path.exists(q4):
        res["error"] = "llama-quantize: " + (q.stdout + q.stderr).strip()[-300:]
        return res
    if build_only:
        # builds run several at a time; the races run later one at a time on a
        # quiet machine (--reuse). The f16 file and the checkpoint are not needed
        # again: the head is rebuilt in memory from its seed for the quality run.
        for big in (f16, os.path.join(d, "model.safetensors")):
            try:
                os.remove(big)
            except OSError:
                pass
        res["built"] = True
        return res
    return finish(res, model, d, ssf, q4, rounds, do_gpu)


def finish(res, model, d, ssf, q4, rounds, do_gpu, extra=None):
    extra = extra or []
    res["bytes"] = {"sheep": os.path.getsize(ssf), "llama": os.path.getsize(q4)}
    res["quality"] = quality(d, model, ssf, q4, d)
    res["cpu_pp"] = race("sml_cpu_race_ci.py", ssf, q4, ["--p", "128"] + extra, rounds)
    res["cpu_tg"] = race("sml_cpu_race_ci.py", ssf, q4, ["--decode", "64"] + extra, rounds)
    if do_gpu:
        why = gpu_refusal(ssf)
        if why:
            res["gpu_refused"] = why
        else:
            res["gpu_pp"] = race("sml_gpu_race_ci.py", ssf, q4, ["--p", "128"] + extra, rounds)
            res["gpu_tg"] = race("sml_gpu_race_ci.py", ssf, q4, ["--decode"] + extra, rounds)
    won = lambda k: res.get(k, {}).get("verdict") == "WON"  # noqa: E731
    res["won"] = {"bytes": res["bytes"]["sheep"] <= res["bytes"]["llama"],
                  "quality": bool(res["quality"].get("won")),
                  "cpu": won("cpu_pp") and won("cpu_tg"),
                  "gpu": do_gpu and won("gpu_pp") and won("gpu_tg")}
    res["family_won"] = all(res["won"].values())
    return res


def show(r):
    if "error" in r:
        return f"{r['label']:<18} ERROR {r['error'][:150]}"
    q = r["quality"]
    t = lambda k: (f"{r[k]['ratio']:.2f}x {r[k]['verdict'][:4]}" if "ratio" in r.get(k, {})  # noqa: E731
                   else ("refused" if k.startswith("gpu") and "gpu_refused" in r else r.get(k, {}).get("error", "-")[:12]))
    ok = lambda b: "WON" if b else "no"  # noqa: E731   (plain: the Windows console is cp1252)
    qs = (f"KLD {q['kld_sheep']:.4f} vs {q['kld_llama']:.4f} top1 {q['top1_sheep']:.3f}/{q['top1_llama']:.3f}"
          if "kld_sheep" in q else q.get("error", "")[:60])
    return (f"{r['label']:<18} {ok(r['family_won'])}  bytes {r['bytes']['sheep'] / 1e6:.0f}/{r['bytes']['llama'] / 1e6:.0f}MB "
            f"{ok(r['won']['bytes'])} | quality {qs} {ok(r['won']['quality'])} | CPU pp {t('cpu_pp')}, tg {t('cpu_tg')} "
            f"{ok(r['won']['cpu'])} | GPU pp {t('gpu_pp')}, tg {t('gpu_tg')} {ok(r['won']['gpu'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default="")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--keep", default="")
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--reuse", action="store_true", help="with --keep: skip builds whose files are there")
    ap.add_argument("--build-only", action="store_true", help="build Sheep and llama.cpp files, race nothing")
    ap.add_argument("--clean", action="store_true", help="with --keep: delete each family's folder after its race")
    a = ap.parse_args()
    want = [x for x in a.only.split(",") if x]
    work = a.keep or tempfile.mkdtemp(prefix="sml_race_")
    results = []
    if os.path.exists(a.json):
        with open(a.json, encoding="utf-8") as f:
            results = [r for r in json.load(f) if not want or r["label"] not in want]
    heads = [h for h in FH.HEADS if not want or h[0] in want]
    for label, mt, over, key, what in heads:
        t0 = time.time()
        try:
            r = race_family(label, mt, over, work, a.rounds, not a.no_gpu, a.reuse, a.build_only)
        except Exception as e:                      # one family's failure must not end the night
            r = {"label": label, "model_type": mt, "error": f"harness: {type(e).__name__}: {e}"[:300]}
        r["roster"], r["represents"], r["minutes"] = key, what, round((time.time() - t0) / 60, 1)
        results = [x for x in results if x["label"] != label] + [r]
        # saved BEFORE printing: a console that could not print a tick once threw
        # away a finished 15-minute race
        with open(a.json + ".tmp", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        os.replace(a.json + ".tmp", a.json)
        try:
            print(show(r), flush=True)
        except Exception as e:                      # a report line must never end the night
            print(f"{label}: (could not format: {e})", flush=True)
        if not a.keep or a.clean:
            shutil.rmtree(os.path.join(work, label), ignore_errors=True)


if __name__ == "__main__":
    main()
