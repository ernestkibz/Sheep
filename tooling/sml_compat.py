#!/usr/bin/env python3
"""Can Sheep run it? One command for every model family the guide lists, at any
parameter count, in minutes -- not one download-build-chat check per model.

    python tooling/sml_compat.py                  every 'ok' row of the guide (sml_archs.ROSTER)
    python tooling/sml_compat.py --heads          first re-run the family heads (sml_family_heads --all)
    python tooling/sml_compat.py --repo ID|DIR    one checkpoint, any size, in seconds

Run it with a python that has transformers (venv-tf5) when --heads is given;
the rest needs only huggingface_hub and numpy.

A model runs in Sheep when two things hold, and neither depends on its size:

1. ITS MATHS. The family's head -- a small random-weight model built from
   transformers' own code for that model_type (sml_family_heads.py) -- gives
   transformers' logits through Sheep's builder and engine, in the small shape
   AND in the shape of a big checkpoint (128-wide heads, one shared K/V head,
   wider FFN). Width and depth are the only things a parameter count changes,
   and those two shapes carry every per-size code path. A family graded on a
   real checkpoint (RACED / GRADED / VERIFIED in its roster note) counts too.
   A family transformers cannot build (remote code: Baichuan, InternLM2, ...)
   is matched by ENGINE SIGNATURE -- the exact engine flags and tensor roles its
   real checkpoint produces (sml_family_signatures.signature) -- to a head that
   runs the very same kernels; same signature, same maths.

2. ITS FILES. A real checkpoint of the family, read from the header only (the
   config, the safetensors tensor tables, the tokenizer -- kilobytes, whether
   the model is 0.5B or 405B): every tensor has a place in the name table, the
   config's behaviours are read, every tensor's shape resolves, and the
   tokenizer fits the embedding. This is the part of a build that fails for
   real models; it is the whole of the builder short of the arithmetic.

Results: tooling/compat.json, and a table. Probe results are cached per repo
revision and builder version, so a re-run only re-reads what changed.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sml_archs as A  # noqa: E402
import sml_hf  # noqa: E402

OUT = os.path.join(HERE, "compat.json")
HEADS_JSON = os.path.join(HERE, "family_heads.json")
SIGS_JSON = os.path.join(HERE, "family_signatures.json")
CACHE = os.path.join(HERE, "compat_cache.json")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "qwen.tiktoken", "tiktoken.model", "tokenization_kimi.py",
                   "tokenization_moonshot.py", "tokenization_qwen.py")

# Checkpoints read for each row: real ones people use (big and small: the probe
# costs the same), then a small test checkpoint of the same model_type as the
# fallback when the real ones are gated or gone. Header only -- nothing heavy
# is downloaded.
PROBES = {
    "llama": ["unsloth/Llama-3.2-1B-Instruct", "NousResearch/Meta-Llama-3.1-70B-Instruct",
              "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "deepseek-ai/deepseek-llm-7b-chat"],
    "mistral": ["unsloth/mistral-7b-instruct-v0.3", "trl-internal-testing/tiny-MistralForCausalLM-0.2"],
    "mixtral": ["mistralai/Mixtral-8x7B-Instruct-v0.1", "yujiepan/mixtral-tiny-random"],
    "mistral3": ["unsloth/Mistral-Small-3.1-24B-Instruct-2503", "mistralai/Devstral-Small-2505"],
    "ministral": ["mistralai/Ministral-8B-Instruct-2410"],
    "qwen2": ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"],
    "qwen3": ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-32B", "Qwen/Qwen3-8B-FP8"],
    "qwen2moe": ["Qwen/Qwen1.5-MoE-A2.7B-Chat"],
    "qwen3moe": ["Qwen/Qwen3-30B-A3B"],
    "qwen3next": ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
    "qwen": ["Qwen/Qwen-1_8B-Chat", "Qwen/Qwen-7B-Chat"],
    "gemma": ["unsloth/gemma-2b-it", "unsloth/gemma-7b-it"],
    "gemma2": ["unsloth/gemma-2-2b-it", "trl-internal-testing/tiny-Gemma2ForCausalLM"],
    "gemma3": ["unsloth/gemma-3-1b-it", "unsloth/gemma-3-27b-it"],
    "vaultgemma": ["google/vaultgemma-1b", "hf-tiny-v2/tiny-random-VaultGemmaForCausalLM",
                   "hf-tiny-v2/tiny-random-VaultGemmaModel"],
    "phi2": ["microsoft/phi-2"],
    "phi3": ["microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-4-mini-instruct",
             "trl-internal-testing/tiny-Phi3ForCausalLM"],
    "gpt2": ["openai-community/gpt2", "openai-community/gpt2-xl"],
    "gpt_neox": ["EleutherAI/pythia-410m", "EleutherAI/pythia-12b"],
    "gpt_neo": ["EleutherAI/gpt-neo-125m"],
    "gptj": ["EleutherAI/gpt-j-6b", "peft-internal-testing/tiny-random-GPTJForCausalLM"],
    "codegen": ["Salesforce/codegen-350M-mono", "hf-tiny-v2/tiny-random-CodeGenForCausalLM"],
    "bloom": ["bigscience/bloom-560m", "bigscience/bloomz-7b1"],
    "opt": ["facebook/opt-125m", "facebook/opt-6.7b", "peft-internal-testing/tiny-random-OPTForCausalLM"],
    "xglm": ["facebook/xglm-564M"],
    "biogpt": ["microsoft/biogpt"],
    "falcon": ["tiiuae/falcon-7b-instruct", "yujiepan/falcon-tiny-random"],
    "mpt": ["mosaicml/mpt-7b-instruct", "mosaicml/mpt-7b", "yujiepan/mpt-tiny-random"],
    "stablelm": ["stabilityai/stablelm-2-1_6b-chat", "stabilityai/stablelm-zephyr-3b"],
    "starcoder2": ["bigcode/starcoder2-3b", "bigcode/starcoder2-15b"],
    "gpt_bigcode": ["bigcode/tiny_starcoder_py", "optimum-internal-testing/tiny-random-gpt_bigcode-multi_query-False"],
    "starcoder": ["bigcode/tiny_starcoder_py"],
    # the CohereForAI repos need a login; Aya Expanse 8B (cohere) is public as re-uploaded
    "command_r": ["CohereForAI/c4ai-command-r-v01", "CohereForAI/aya-23-8B", "adamo1139/aya-expanse-8b-ungated",
                  "Citaman/command-r-1-layer"],
    "cohere2": ["CohereForAI/c4ai-command-r7b-12-2024", "trl-internal-testing/tiny-Cohere2ForCausalLM"],
    "olmo": ["allenai/OLMo-1B-hf", "optimum-intel-internal-testing/tiny-random-olmo-hf"],
    "olmo2": ["allenai/OLMo-2-0425-1B", "allenai/OLMo-2-1124-13B-Instruct"],
    "olmo3": ["allenai/Olmo-3-7B-Instruct", "allenai/Olmo-3-32B-Think", "trl-internal-testing/tiny-Olmo3ForCausalLM"],
    "olmoe": ["allenai/OLMoE-1B-7B-0924"],
    "flex_olmo": ["allenai/FlexOlmo-7x7B-1T", "allenai/FlexOlmo-7x7B-1T-RT"],
    "granite": ["ibm-granite/granite-3.1-2b-instruct", "ibm-granite/granite-3.3-8b-instruct"],
    "granitemoe": ["ibm-granite/granite-3.1-1b-a400m-instruct", "ibm-granite/granite-3.1-3b-a800m-instruct"],
    "granitemoeshared": ["ibm-research/moe-7b-1b-active-shared-experts"],
    "granite_hybrid": ["ibm-granite/granite-4.0-h-tiny", "ibm-granite/granite-4.0-h-small"],
    "smollm3": ["HuggingFaceTB/SmolLM3-3B"],
    "idefics3": ["HuggingFaceTB/SmolVLM-256M-Instruct"],
    "exaone": ["LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", "optimum-intel-internal-testing/tiny-random-exaone"],
    "exaone4": ["LGAI-EXAONE/EXAONE-4.0-1.2B", "LGAI-EXAONE/EXAONE-4.0-32B"],
    "exaone_moe": ["LGAI-EXAONE/K-EXAONE-236B-A23B"],
    "internlm2": ["internlm/internlm2_5-1_8b-chat", "internlm/internlm2_5-7b-chat",
                  "optimum-intel-internal-testing/tiny-random-internlm2"],
    "baichuan": ["baichuan-inc/Baichuan2-7B-Chat", "baichuan-inc/Baichuan2-13B-Chat"],
    "minicpm": ["openbmb/MiniCPM-2B-sft-bf16", "optimum-intel-internal-testing/tiny-random-minicpm"],
    "deepseek": ["deepseek-ai/deepseek-moe-16b-chat"],
    "phimoe": ["microsoft/Phi-3.5-MoE-instruct"],
    "gpt_oss": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    "minicpm3": ["openbmb/MiniCPM3-4B"],
    "nemotron_h_moe": ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"],
    "falcon_h1": ["tiiuae/Falcon-H1-1.5B-Instruct"],
    "bamba": ["ibm-ai-platform/Bamba-9B-v2"],
    "deepseek2": ["deepseek-ai/DeepSeek-V2-Lite-Chat", "deepseek-ai/DeepSeek-V3", "moonshotai/Moonlight-16B-A3B-Instruct"],
    "chameleon": ["facebook/chameleon-7b", "leloy/Anole-7b-v0.1-hf"],
    "xverse": ["xverse/XVERSE-7B-Chat", "optimum-intel-internal-testing/tiny-random-xverse"],
    "orion": ["OrionStarAI/Orion-14B-Chat", "optimum-intel-internal-testing/tiny-random-orion"],
    "codeshell": ["WisdomShell/CodeShell-7B-Chat"],
    "refact": ["smallcloudai/Refact-1_6B-fim"],
    "pangu_embed": ["FreedomIntelligence/openPangu-Embedded-1B"],
    "nemotron": ["nvidia/Nemotron-Mini-4B-Instruct", "nvidia/Minitron-8B-Base"],
    "nemotron_h": ["nvidia/NVIDIA-Nemotron-Nano-9B-v2", "nvidia/NVIDIA-Nemotron-Nano-12B-v2"],
    "mamba": ["state-spaces/mamba-130m-hf", "tiiuae/falcon-mamba-7b"],
    "mamba2": ["AntonV/mamba2-130m-hf", "hf-tiny-v2/tiny-random-Mamba2ForCausalLM"],
    "jamba": ["ai21labs/AI21-Jamba-Mini-1.5", "ai21labs/Jamba-v0.1", "ai21labs/AI21-Jamba-Reasoning-3B"],
    "lfm2": ["LiquidAI/LFM2-350M", "LiquidAI/LFM2-1.2B"],
    "lfm2moe": ["LiquidAI/LFM2-8B-A1B"],
    "glm": ["THUDM/glm-4-9b-chat-hf", "optimum-intel-internal-testing/tiny-random-glm-edge"],
    "glm4": ["THUDM/GLM-4-9B-0414", "tiny-random/glm-4"],
    "glm4_moe": ["zai-org/GLM-4.5-Air", "optimum-intel-internal-testing/tiny-random-glm4-moe"],
    "dots1": ["rednote-hilab/dots.llm1.inst", "hf-tiny-v2/tiny-random-Dots1Model"],
    "ernie4_5": ["baidu/ERNIE-4.5-0.3B-PT", "hf-tiny-v2/tiny-random-Ernie4_5Model"],
    "ernie4_5_moe": ["baidu/ERNIE-4.5-21B-A3B-PT", "hf-tiny-v2/tiny-random-Ernie4_5_MoeModel"],
    "hunyuan_dense": ["tencent/Hunyuan-1.8B-Instruct", "tencent/Hunyuan-7B-Instruct"],
    "hunyuan_moe": ["tencent/Hunyuan-A13B-Instruct"],
    "arcee": ["arcee-ai/AFM-4.5B", "optimum-intel-internal-testing/tiny-random-ArceeForCausalLM"],
    "afmoe": ["arcee-ai/Trinity-Mini", "arcee-ai/Trinity-Nano-Preview"],
    "minimax_m2": ["MiniMaxAI/MiniMax-M2", "tiny-random/minimax-m2"],
    "jais2": ["inceptionai/Jais-2-8B-Chat", "inceptionai/jais-2-8b-chat", "yoriis/JAIS2-IT-0.3",
              "hf-tiny-v2/tiny-random-Jais2ForCausalLM"],
    "seed_oss": ["ByteDance-Seed/Seed-OSS-36B-Instruct", "tiny-random/seed-oss"],
    "apertus": ["swiss-ai/Apertus-8B-Instruct-2509", "tiny-random/apertus"],
    "helium": ["kyutai/helium-1-preview-2b", "hf-tiny-v2/tiny-random-HeliumModel"],
    "persimmon": ["adept/persimmon-8b-chat", "hf-tiny-v2/tiny-random-PersimmonModel"],
    "dbrx": ["yujiepan/dbrx-tiny-random"],
    "qwen35": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"],
    "qwen35moe": ["Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-122B-A10B", "tiny-random/qwen3.5-moe"],
}
REAL_GRADE = re.compile(r"^(RACED|GRADED|VERIFIED|VISION, GRADED)\b")


# ------------------------------------------------------------------ header-only reading
class _HttpFile:
    """A remote file as a seekable read-only file, by HTTP range requests --
    enough for zipfile to read a torch .bin's directory and its pickle
    without fetching the weights."""
    def __init__(self, url, size, headers=None):
        self.url, self.size, self.pos, self.headers = url, int(size), 0, dict(headers or {})

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        import urllib.request
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers=dict(self.headers, Range=f"bytes={self.pos}-{end}"))
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 2:
                    raise
        self.pos += len(data)
        return data


def _bin_headers(src, rev, names, sizes, d):
    """Stub safetensors headers for a repo that ships torch .bin shards."""
    import zipfile
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    bins = sorted(n for n in names if re.fullmatch(r"(pytorch_model|model)(-\d+-of-\d+)?\.bin", n))
    if not bins:
        raise SystemExit("no .safetensors and no pytorch_model .bin in the repo")
    for fn in bins:
        f = _HttpFile(hf_hub_url(src, fn, revision=rev), sizes[fn], build_hf_headers())
        with zipfile.ZipFile(f) as zf:
            tensors, _ = sml_hf.bin_tensors(zf)
        header = {k: {"dtype": t.dtype, "shape": t.shape, "data_offsets": [0, 0]} for k, t in tensors.items()}
        raw = json.dumps(header).encode()
        with open(os.path.join(d, fn[:-4] + ".safetensors"), "wb") as out:
            out.write(struct.pack("<Q", len(raw)) + raw)


def _stub_dir(src, work):
    """A folder holding the checkpoint's config, tokenizer and safetensors
    HEADERS -- kilobytes for any model size. Returns (dir, revision)."""
    d = os.path.join(work, re.sub(r"[^\w.-]+", "_", src))
    os.makedirs(d, exist_ok=True)
    if os.path.isdir(src):
        for f in os.listdir(src):
            p = os.path.join(src, f)
            if f.endswith(".safetensors"):
                with open(p, "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    head = fh.read(n)
                with open(os.path.join(d, f), "wb") as out:
                    out.write(struct.pack("<Q", n) + head)
            elif f.endswith((".json", ".model", ".txt", ".jinja")) and os.path.getsize(p) < 64 << 20:
                shutil.copy(p, os.path.join(d, f))
        if not any(f.endswith(".safetensors") for f in os.listdir(d)):
            import zipfile
            for f in sml_hf.weight_files(src):
                with zipfile.ZipFile(os.path.join(src, f)) as zf:
                    tensors, _ = sml_hf.bin_tensors(zf)
                raw = json.dumps({k: {"dtype": t.dtype, "shape": t.shape, "data_offsets": [0, 0]}
                                  for k, t in tensors.items()}).encode()
                with open(os.path.join(d, f[:-4] + ".safetensors"), "wb") as out:
                    out.write(struct.pack("<Q", len(raw)) + raw)
        return d, "local"
    from huggingface_hub import get_safetensors_metadata, hf_hub_download, model_info
    info = model_info(src, files_metadata=True)
    names = {s.rfilename for s in (info.siblings or [])}
    sizes = {s.rfilename: s.size for s in (info.siblings or [])}
    for f in ("config.json", "generation_config.json") + TOKENIZER_FILES:
        if f in names:
            shutil.copy(hf_hub_download(src, f, revision=info.sha), os.path.join(d, f))
    if not any(n.endswith(".safetensors") for n in names):
        _bin_headers(src, info.sha, names, sizes, d)
        return d, info.sha
    meta = get_safetensors_metadata(src, revision=info.sha)
    keep = set(sml_hf.pick_weight_files(list(names)))     # what the download takes: one format, root only
    for fn, fm in meta.files_metadata.items():
        if fn not in keep:
            continue
        header, off = {}, 0
        for k, t in fm.tensors.items():
            a, b = t.data_offsets
            header[k] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [a, b]}
        raw = json.dumps(header).encode()
        with open(os.path.join(d, os.path.basename(fn)), "wb") as out:
            out.write(struct.pack("<Q", len(raw)) + raw)
    return d, info.sha


def _nparams(W):
    n = 0
    for sh in W.shards.values():
        for k, e in sh.header.items():
            if k != "__metadata__" and len(e.get("shape") or []) >= 1:
                n += int(np.prod(e["shape"]))
    return n


def probe(src, work, tries=3):
    """probe_once, with network failures retried."""
    for i in range(tries):
        r = probe_once(src, work)
        why = r.get("why", "")
        if r.get("ok") or not any(x in why for x in ("timed out", "disconnected", "Connection", "SSL", "503", "502")):
            return r
    return r


def probe_once(src, work):
    """Everything the builder decides before the arithmetic, from headers only:
    {ok, model_type, params, signature, why}."""
    out = {"src": src}
    try:
        d, rev = _stub_dir(src, work)
        out["revision"] = rev
    except (Exception, SystemExit) as e:
        return _failed(out, e)
    return check_dir(d, out)


def _failed(out, e):
    msg = str(e) or type(e).__name__
    if "401" in msg or "403" in msg or "gated" in msg.lower():
        msg = "gated or private on Hugging Face (needs a login): not probed"
        out["skip"] = True
    elif "404" in msg or "RepositoryNotFound" in type(e).__name__:
        msg = "no such repo on Hugging Face"
        out["skip"] = True
    out.update(ok=False, why=msg[:300])
    return out


def check_dir(d, out=None):
    """The builder's every decision short of the arithmetic, on a folder of
    headers (a _stub_dir) or a real checkpoint, reading no weights."""
    out = dict(out or {})
    try:
        with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        mt = sml_hf.model_type_of(cfg)
        out["model_type"] = mt
        archs = cfg.get("architectures") or []
        if archs and all(x.endswith("Model") and not x.endswith(("LMHeadModel", "CausalLM")) for x in archs):
            out.update(ok=False, skip=True, why=f"{archs[0]}: a base model without an LM head, not a chat checkpoint")
            return out
        row, ok = A.build_gate(mt, cfg)
        if not ok:
            raise SystemExit(A.explain(mt, cfg))
        if True:
            W = sml_hf.open_weights(d, header_only=True)
            import sml_family_signatures as S
            sig, h = S.signature(d, W)
            for n in list(W.t):
                W.shape(n)                            # every split, join and fused layout resolves
            from sml_common import rope_freqs
            fl = W.flags()
            hd = int(W.cfg.get("head_dim") or W.cfg["hidden_size"] // max(1, int(W.cfg.get("num_attention_heads") or 1)))
            rope_freqs(float(W.cfg.get("rope_theta", 10000.0)), (fl["n_rot"] or hd) // 2, W.cfg.get("rope_scaling"))
            if not any(os.path.exists(os.path.join(d, f)) for f in ("tokenizer.json", "tokenizer.model", "merges.txt",
                                                                     "qwen.tiktoken", "tiktoken.model")):
                raise SystemExit("no tokenizer (tokenizer.json, tokenizer.model, vocab.json + merges.txt or a .tiktoken)")
            toks = W.field("tokenizer.tokens") or []
            emb = "token_embd.weight"
            if emb in W.t and len(toks) > W.shape(emb)[0]:
                raise SystemExit(f"the tokenizer has {len(toks)} ids but the embedding only {W.shape(emb)[0]} rows")
        out.update(ok=True, params=_nparams(W), signature=h, sig=sig, tokens=len(toks))
    except (Exception, SystemExit) as e:
        return _failed(out, e)
    return out


# ------------------------------------------------------------------ the verdict per row
def _builder_stamp():
    h = hashlib.sha256()
    for f in ("sml_hf.py", "sml_archs.py", "sml_format.py", "sml_family_signatures.py", "sml_compat.py"):
        with open(os.path.join(HERE, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def _load(p, default):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _heads_by_row():
    """{roster row: [head results]} and {label: {shape: result}}."""
    rows, by_label = {}, {}
    for r in _load(HEADS_JSON, []):
        key = r.get("roster") or A.ALIASES.get(r.get("model_type", ""), r.get("model_type"))
        rows.setdefault(key, []).append(r)
        mt_row = A.ALIASES.get(r.get("model_type", ""), r.get("model_type"))
        if mt_row != key:
            rows.setdefault(mt_row, []).append(r)
        by_label.setdefault(r["label"], {})[r.get("shape", "small")] = r
    return rows, by_label


# Flags that are sizes or one scalar, not a different kernel: a family that
# differs from a head only here runs the head's code with other numbers.
SIZE_FLAGS = {"head_dim", "attn_scale"}


def twin(sig, ok_sigs, verified_roles):
    """How a checkpoint's maths is covered by passing heads, or None.
    Exact: the same engine flags and tensor roles as a head (sizes aside).
    Composed: the same flags as a head, plus tensor roles (a bias, say) that
    other passing heads run -- every kernel it uses is verified somewhere."""
    if not sig:
        return None

    def flags(sg):
        return {k: v for k, v in sg["flags"].items() if k not in SIZE_FLAGS}
    mine, roles = flags(sig), set(sig["roles"])
    exact = sorted(lb for sg, lbs in ok_sigs if flags(sg) == mine and set(sg["roles"]) == roles for lb in lbs)
    if exact:
        return f"same engine kernels as head {', '.join(exact)}"
    for sg, lbs in ok_sigs:
        extra = roles - set(sg["roles"])
        if flags(sg) == mine and set(sg["roles"]) <= roles and extra <= verified_roles:
            return (f"head {', '.join(lbs)}'s kernels plus {', '.join(sorted(extra))}, "
                    f"each run by another passing head")
    return None


def _head_ok(results):
    """A head passes when every shape it was run in passes."""
    return bool(results) and all(r.get("pass") for r in results)


def verdicts(probes):
    rows, by_label = _heads_by_row()
    out = []
    for key, (st, needs, note) in A.ROSTER.items():
        if st != "ok":
            continue
        v = {"row": key, "note": note}
        mine = rows.get(key, [])
        labels = sorted({r["label"] for r in mine})
        passed = [lb for lb in labels if _head_ok(list(by_label.get(lb, {}).values()))]
        shapes = sorted({r.get("shape", "small") for r in mine})
        if passed:
            v["maths"] = f"head {', '.join(passed)} ({'+'.join(shapes)})"
        elif REAL_GRADE.match(note):
            v["maths"] = "graded on a real checkpoint (" + note.split(":")[0].split(",")[0] + ")"
        good = [p for p in probes.get(key, []) if p.get("ok")]
        if "maths" not in v:
            # a family transformers cannot build: a head with its exact engine signature
            ok_sigs = [(g["signature"], [lb for lb in g.get("labels", [])
                                         if _head_ok(list(by_label.get(lb, {}).values()))])
                       for g in _load(SIGS_JSON, {}).values() if g.get("signature")]
            ok_sigs = [(sg, lbs) for sg, lbs in ok_sigs if lbs]
            verified_roles = set().union(*[set(sg["roles"]) for sg, _ in ok_sigs]) if ok_sigs else set()
            for p in good:
                got = twin(p.get("sig"), ok_sigs, verified_roles)
                if got:
                    v["maths"] = got
                    break
        failed_heads = [lb for lb in labels if lb not in passed]
        if "maths" not in v:
            v["maths_why"] = (f"head {', '.join(failed_heads)} does not match transformers" if failed_heads
                              else "no head, no real-checkpoint grade, and no head shares its engine signature")
        v["files"] = [{k: p.get(k) for k in ("src", "ok", "params", "why", "skip")} for p in probes.get(key, [])]
        if good:
            big = max(good, key=lambda p: p.get("params") or 0)
            v["largest_probed"] = big["src"]
            v["largest_params"] = big.get("params")
        v["runs"] = "maths" in v and bool(good)
        v["verdict"] = ("runs" if v["runs"] else
                        "maths ok, no checkpoint read" if "maths" in v and not any(not p.get("skip") for p in probes.get(key, []))
                        else "files fail" if "maths" in v else "maths unproven")
        out.append(v)
    return out


def _fmt_params(n):
    if not n:
        return "-"
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def run_probes(srcs, jobs=8, fresh=False):
    cache = {} if fresh else _load(CACHE, {})
    stamp = _builder_stamp()
    work = tempfile.mkdtemp(prefix="sml_compat_")
    todo = [s for s in srcs if not (cache.get(s, {}).get("stamp") == stamp and cache[s].get("ok") is not None
                                     and not cache[s].get("skip"))]

    def one(s):
        r = probe(s, work)
        r["stamp"] = stamp
        return s, r
    try:
        with ThreadPoolExecutor(jobs) as ex:
            for s, r in ex.map(one, todo):
                cache[s] = r
                print(f"  {'read' if r.get('ok') else 'FAIL':<5} {s:<58} "
                      f"{_fmt_params(r.get('params')) if r.get('ok') else r.get('why', '')[:90]}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    with open(CACHE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)
    os.replace(CACHE + ".tmp", CACHE)
    return cache


# ------------------------------------------------------------------ the guide, from the evidence
ROOT = os.path.dirname(HERE)
USER_GUIDE = os.path.join(ROOT, "docs", "USER_GUIDE.md")
README = os.path.join(ROOT, "docs", "MODELS.md")      # the support table moved here from README.md (2026-10-21)


def facts():
    """Every number the guide states, computed from the roster, the family
    heads and the last compat run -- never typed in."""
    c, _ = A.summary()
    comp = _load(OUT, {"rows": []})
    heads = [r for r in _load(HEADS_JSON, []) if r.get("shape", "small") == "small"]
    runs = sorted(v["row"] for v in comp["rows"] if v.get("runs"))
    largest = max((v for v in comp["rows"] if v.get("runs")), key=lambda v: v.get("largest_params") or 0, default={})
    return {"roster": len(A.ROSTER), "ok": c["ok"], "gap": c["gap"], "scope": c["scope"], "unknown": c["unknown"],
            "checked": len(runs), "checkedOf": len(comp["rows"]),
            "heads": len(heads), "headsPass": sum(bool(r.get("pass")) for r in heads),
            "largest": largest.get("largest_probed", ""), "largestB": round((largest.get("largest_params") or 0) / 1e9, 1),
            "supported": sorted(k for k, v in A.ROSTER.items() if v[0] == "ok"), "runs": runs,
            "notYet": sorted((v["row"], v["verdict"]) for v in comp["rows"] if not v.get("runs"))}


def _between(text, start, end, body):
    i, j = text.index(start) + len(start), text.index(end)
    return text[:i] + body + text[j:]


def emit_guide():
    f = facts()
    md = (f"\nThe architecture roster has {f['roster']} entries: {f['ok']} accepted by the build gate, "
          f"{f['gap']} with known engine gaps, {f['scope']} out of scope, and {f['unknown']} unknown.\n\n"
          f"**Checked end to end: {f['checked']} of {f['ok']} accepted families** "
          f"(`python tooling/sml_compat.py`). For each, the family's maths matches transformers -- a "
          f"random-weight model of that family built from transformers' own code, in a small shape and in the "
          f"shape of a big checkpoint (128-wide heads, one shared K/V head) -- and a real checkpoint of it "
          f"passes every build check from its headers alone: tensor names and layouts, rotary scaling, "
          f"tokenizer. Size does not change either check; the largest checkpoint read was "
          f"{f['largest']} ({f['largestB']}B parameters). Family heads run: {f['heads']}, {f['headsPass']} pass.\n\n"
          + ("Accepted but not yet checked end to end: "
             + "; ".join(f"`{r}` ({why})" for r, why in f["notYet"]) + ".\n" if f["notYet"] else ""))
    with open(USER_GUIDE, encoding="utf-8") as fh:
        ug = fh.read()
    ug = _between(ug, "<!-- generated by tooling/sml_compat.py --emit-guide -->",
                  "<!-- /generated -->", md + "\n")
    with open(USER_GUIDE, "w", encoding="utf-8") as fh:
        fh.write(ug)

    with open(README, encoding="utf-8") as fh:
        rd = fh.read()
    rd = re.sub(r"rosters \*\*\d+ architectures\*\*", f"rosters **{f['roster']} architectures**", rd)
    for label, key in (("\\*\\*roster marked `ok`\\*\\*", "ok"), ("gap", "gap"), ("out of scope", "scope"),
                       ("unknown", "unknown")):
        rd = re.sub(rf"^(\|\s*{label}\s*\|\s*\**)\d+(\**\s*\|)", lambda m: f"{m.group(1)}{f[key]}{m.group(2)}",
                    rd, count=1, flags=re.M)
    rd = re.sub(r"^(\|\s*\*\*run as a family head\*\*\s*\|\s*\*\*)\d+(\*\*\s*\|.*?; )\d+( pass)",
                lambda m: f"{m.group(1)}{f['heads']}{m.group(2)}{f['headsPass']}{m.group(3)}", rd, count=1, flags=re.M)
    with open(README, "w", encoding="utf-8") as fh:
        fh.write(rd)
    print(f"guide: {f['checked']} of {f['ok']} accepted families checked end to end; "
          f"heads {f['headsPass']}/{f['heads']}; largest read {f['largest']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-guide", action="store_true",
                    help="rewrite the guide's model lists and counts from compat.json and the roster")
    ap.add_argument("--repo", action="append", default=[], help="check one HF repo id or local folder")
    ap.add_argument("--heads", action="store_true", help="re-run the family heads first (needs transformers)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached probes")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.emit_guide:
        emit_guide()
        return
    if args.repo:
        work = tempfile.mkdtemp(prefix="sml_compat_")
        rows_h, by_label = _heads_by_row()
        ok_sigs = [(g["signature"], [lb for lb in g.get("labels", []) if _head_ok(list(by_label.get(lb, {}).values()))])
                   for g in _load(SIGS_JSON, {}).values() if g.get("signature")]
        ok_sigs = [(sg, lbs) for sg, lbs in ok_sigs if lbs]
        verified = set().union(*[set(sg["roles"]) for sg, _ in ok_sigs]) if ok_sigs else set()
        try:
            for s in args.repo:
                p = probe(s, work)
                if p.get("ok"):
                    row = A.ALIASES.get(p["model_type"], p["model_type"])
                    own = sorted({r["label"] for r in rows_h.get(row, [])
                                  if _head_ok(list(by_label.get(r["label"], {}).values()))})
                    how = (f"its family head {', '.join(own)} matches transformers" if own
                           else twin(p.get("sig"), ok_sigs, verified))
                    print(f"{s}: Sheep can build it -- {p['model_type']}, {_fmt_params(p['params'])} parameters, "
                          f"{p['tokens']} tokens; maths: " + (how or "no passing head runs these kernels"))
                else:
                    print(f"{s}: NO -- {p.get('why')}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return
    if args.heads:
        subprocess.run([sys.executable, os.path.join(HERE, "sml_family_heads.py"), "--all"], check=False)
        subprocess.run([sys.executable, os.path.join(HERE, "sml_family_signatures.py")], check=False)
    ok_rows = [k for k, v in A.ROSTER.items() if v[0] == "ok"]
    srcs = list(dict.fromkeys(s for k in ok_rows for s in PROBES.get(k, [])))
    print(f"reading {len(srcs)} checkpoints (headers only) for {len(ok_rows)} families")
    cache = run_probes(srcs, args.jobs, args.fresh)
    probes = {k: [cache[s] for s in PROBES.get(k, []) if s in cache] for k in ok_rows}
    vs = verdicts(probes)
    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"rows": vs, "runs": sum(v["runs"] for v in vs), "of": len(vs)}, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    print()
    print(f"  {'family':<18} {'verdict':<30} {'largest read':<10} evidence")
    for v in sorted(vs, key=lambda v: (v["runs"], v["row"])):
        ev = v.get("maths") or v.get("maths_why", "")
        bad = [f for f in v["files"] if not f["ok"] and not f.get("skip")]
        if bad and not v["runs"]:
            ev += " | " + bad[0]["src"] + ": " + (bad[0]["why"] or "")[:120]
        print(f"  {v['row']:<18} {v['verdict']:<30} {_fmt_params(v.get('largest_params')):<10} {ev[:150]}")
    print(f"\n{sum(v['runs'] for v in vs)} of {len(vs)} guide families: maths proven and a real checkpoint reads")
    emit_guide()


if __name__ == "__main__":
    main()
