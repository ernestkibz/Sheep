"""Where a Hugging Face ID's original weights are (2026-10-19).

Sheep starts where llama.cpp starts: the original checkpoint -- safetensors or torch .bin,
as published -- is what both convert. A GGUF, GPTQ, AWQ, MLX or bitsandbytes page holds a
copy already compressed by someone else. Such a page names its source in its model card
(base_model), so the original is found from it and built instead, and the person is told
which. Only when no original exists is the copy itself converted (fallback): a GGUF
(Part C1), GPTQ and AWQ copies and Meta's raw .pth checkpoints (Part C6). Other formats
(EXL2, ONNX, Core ML, MLX...) get a plain refusal that names the format.
"""
import json
import re
import sys

from .paths import REPO

sys.path.insert(0, str(REPO / "tooling"))

# quantization_config.quant_method values whose weights are packed integers, not originals
PACKED = {"gptq", "awq", "bitsandbytes", "exl2", "aqlm", "hqq", "eetq", "marlin", "quanto", "torchao",
          "higgs", "vptq", "spqr", "bitnet", "gptqmodel", "autoround", "auto-round"}
WORDS = {"gguf": "a GGUF", "gptq": "a GPTQ copy", "awq": "an AWQ copy", "bitsandbytes": "a bitsandbytes copy",
         "mlx": "an MLX copy", "exl2": "an EXL2 copy", "onnx": "an ONNX export", "compressed": "a compressed copy",
         "packed": "a pre-quantized copy", "adapter": "a LoRA adapter", "gptqmodel": "a GPTQ copy",
         "meta": "Meta's raw .pth checkpoint", "coreml": "a Core ML export"}
# copies Sheep converts itself when there is no original (tooling/sml_unpack.py)
UNPACK = {"gptq", "gptqmodel", "awq", "meta"}


def weight_files(names):
    import sml_hf
    return sml_hf.pick_weight_files(names)


def classify(names, cfg):
    """What a repo holds, from its file names and config.json: "original", or why not --
    gguf, gptq/awq/... (packed integers), mlx, compressed, onnx, adapter, none, no-config."""
    low = [n.lower() for n in names]
    weights = weight_files(names)
    cfg = cfg if isinstance(cfg, dict) else {}
    qc = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    if isinstance(qc, dict):
        m = str(qc.get("quant_method") or "").lower()
        if m in PACKED:
            return m if m in WORDS else "packed"
        if m == "compressed-tensors":
            # FP8 releases are read as originals (sml_hf scales them); integer packs are not
            groups = json.dumps(qc.get("config_groups") or {}).lower()
            if '"type": "int"' in groups or qc.get("format") in ("pack-quantized", "int-quantized", "marlin-24"):
                return "compressed"
    if isinstance(cfg.get("quantization"), dict) and "bits" in cfg["quantization"]:
        return "mlx"
    if any(".mlpackage/" in n or n.endswith((".mlmodel", ".mlpackage")) for n in low) and not weights:
        return "coreml"
    if not weights:
        if "params.json" in low and any(re.fullmatch(r"consolidated\.\d+\.pth", n) for n in low):
            return "meta"
        if any(n.endswith(".gguf") for n in low):
            return "gguf"
        if "adapter_config.json" in low:
            return "adapter"
        if any(n.endswith(".onnx") for n in low):
            return "onnx"
        if any(n.endswith((".exl2", ".pt", ".pth", ".ckpt")) for n in low) and "config.json" not in low:
            return "packed"
        return "none"
    if "config.json" not in low and "params.json" not in low:
        return "no-config"
    return "original"


SUFFIX = re.compile(r"[-_.](gguf|ggml|awq|gptq|exl2|mlx|bnb|nf4|onnx|int[48]|w\d+a\d+|\d+-?bits?|\d+(\.\d+)?bpw|"
                    r"q\d(_k)?(_[msl0-9])*|fp8-dynamic|imatrix|quantized|quant)\b.*$", re.I)


def name_without_suffix(name):
    """Qwen2.5-3B-Instruct-4bit -> Qwen2.5-3B-Instruct; Llama-3.2-1B-Instruct-bnb-4bit -> Llama-3.2-1B-Instruct."""
    out = SUFFIX.sub("", name or "")
    return out if out and out != name else ""


def _bases(card):
    b = getattr(card, "base_model", None) if card is not None and not isinstance(card, dict) else \
        (card or {}).get("base_model")
    if isinstance(b, str):
        return [b]
    return [x for x in (b or []) if isinstance(x, str)]


# Which file of a GGUF page to convert when there is no original: the closest to the original
# weights that is not needlessly large (8-bit first, then 6, then 16, then 5 and 4).
# closest to the original first: an F16/BF16 file is the original itself, offered every level
GGUF_PREF = (r"(?:^|[-_.])(?:f16|bf16)(?:[-_.]|$)", r"q8_0", r"q6_k", r"q5_k_m", r"q4_k_m")


def gguf_file(names):
    """The .gguf file to convert from a page's file names, or None. Multi-part files are skipped."""
    import re
    files = [n for n in names if n.lower().endswith(".gguf") and "/" not in n and not re.search(r"-\d+-of-\d+\.gguf$", n.lower())]
    for pat in GGUF_PREF:
        hit = [n for n in files if re.search(pat, n.lower())]
        if hit:
            return sorted(hit, key=len)[0]
    return sorted(files)[0] if files else None


# Ernest, 2026-10-29: a copy that is already quantized is taken as it is: packed into SSF at its own width,
# never shrunk again. Only an original (safetensors, or an F16/BF16/F32 GGUF) is offered every level.
FLOAT_TAG = re.compile(r"(?:^|[-_.])(f16|bf16|f32|fp16|fp32)(?:[-_.]|$)", re.I)
QUANT_TAG = re.compile(r"(?:^|[-_.])(i?q\d(?:_[a-z0-9]+)*)(?:[-_.]|$)", re.I)
WIDTH_LEVEL = {1: "q2", 2: "q2", 3: "q3", 4: "small", 5: "q5", 6: "high", 7: "q7", 8: "q8"}


def as_is(kind, name="", cfg=None):
    """For a copy that is already quantized: {"id": the Sheep level at its own width, "tag": what it is},
    or None for an original. kind is classify()'s answer; name the GGUF file; cfg its config.json."""
    if kind == "gguf":
        if FLOAT_TAG.search(name):
            return None
        m = QUANT_TAG.search(name)
        tag = m.group(1).upper() if m else "quantized"
        width = int(re.search(r"\d", tag).group()) if m else 4
        # Q4_K_M and Q4_K_L keep 6-bit where it matters, like Sheep's 4/6-bit level
        lv = "balanced" if re.fullmatch(r"Q4_K(_[ML])?", tag) or not m else WIDTH_LEVEL.get(width, "q8")
        return {"id": lv, "tag": f"GGUF {tag}"}
    if kind in ("original", "meta", "none", "no-config", "adapter", None, ""):
        return None
    qc = (cfg or {}).get("quantization_config") or ((cfg or {}).get("text_config") or {}).get("quantization_config") or {}
    bits = qc.get("bits") or qc.get("w_bit") or ((cfg or {}).get("quantization") or {}).get("bits") or 4
    try:
        bits = int(bits)
    except (TypeError, ValueError):
        bits = 4
    return {"id": WIDTH_LEVEL.get(bits, "q8"), "tag": f"{kind.upper().replace('MODEL', '')} {bits}-bit"}


def fallback(repo, model_info=None, read_config=None):
    """What Sheep converts when `repo` has no original anywhere: {"how": "unpack", "kind": "gptq" |
    "awq" | "meta", "bytes": ...}, {"how": "gguf", "file": ..., "bytes": ...}, or None."""
    try:
        if model_info is None:
            from huggingface_hub import HfApi
            model_info = lambda r: HfApi().model_info(r, files_metadata=True)
        info = model_info(repo)
        sib = list(info.siblings or [])
        names = [str(x.rfilename) for x in sib]
        size = {str(x.rfilename): int(getattr(x, "size", 0) or 0) for x in sib}
        cfg = (read_config or _read_config)(repo, info, names)
        kind = classify(names, cfg)
    except Exception:
        return None
    if kind in UNPACK:
        pick = [n for n in names if "/" not in n and (n.endswith(".safetensors") or
                                                      re.fullmatch(r"consolidated\.\d+\.pth", n))]
        return {"how": "unpack", "kind": kind, "bytes": sum(size[n] for n in pick), "as_is": as_is(kind, cfg=cfg)}
    g = gguf_file(names)
    return {"how": "gguf", "file": g, "bytes": size.get(g, 0), "as_is": as_is("gguf", g)} if g else None


def folder_kind(d):
    """classify() for a folder on this computer."""
    from pathlib import Path
    d = Path(d)
    names = [x.name for x in d.iterdir()] if d.is_dir() else []
    try:
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    return classify(names, cfg)


def convert_note(fb, repo, local=False):
    """The person's sentence for a fallback() answer (local: a folder on this computer)."""
    if fb["how"] == "gguf":
        return f"No original weights on Hugging Face. Sheep converts {fb['file']} and builds from it; the quality is that file's."
    if fb["kind"] == "meta":
        return f"{repo} is Meta's raw .pth checkpoint. Sheep converts it to the standard format and builds from it."
    k = fb["kind"].upper().replace("MODEL", "")
    first = f"This folder is a {k} copy." if local else f"No original weights on Hugging Face, only this {k} copy."
    return f"{first} Sheep unpacks it to float16 and builds from it; the quality is the copy's."


def resolve(repo, model_info=None, read_config=None, hops=3, search=None):
    """(the repo to build from, a sentence for the person or "", the kind of the asked repo).
    Follows base_model up to `hops` times from a page that is not an original; a page that
    names no base is matched by its own name, compression suffix taken off, on the Hub."""
    if model_info is None:
        from huggingface_hub import HfApi
        model_info = lambda r: HfApi().model_info(r, files_metadata=True)
    if read_config is None:
        read_config = _read_config
    if search is None:
        search = _search

    def kind_of(r):
        info = model_info(r)
        names = [str(s.rfilename) for s in (info.siblings or [])]
        return info, classify(names, read_config(r, info, names))

    def original(r):
        try:
            return kind_of(r)[1] == "original"
        except Exception:
            return False

    asked, cur, kind0, seen = repo, repo, None, set()
    guess = name_without_suffix(asked.split("/")[-1])
    for _ in range(hops + 1):
        info, kind = kind_of(cur)
        kind0 = kind0 or kind
        if kind == "original":
            if cur == asked:
                return cur, "", kind0
            return cur, (f"{asked} is {WORDS.get(kind0, 'a converted copy')} of {cur}; Sheep builds from {cur}'s "
                         "original weights."), kind0
        if kind == "adapter":
            base = (_bases(getattr(info, "card_data", None)) or ["its base model"])[0]
            raise LookupError(f"{asked} is a LoRA adapter for {base}, not a whole model; Sheep builds whole "
                              "original checkpoints (merging adapters is not in this build)")
        if kind in ("none", "no-config"):
            # an EXL2 page keeps its weights in its own format: its name says what it is, so look for
            # the original as for any other copy (2026-10-21: the refusal read "no config.json with .bin")
            fmt = re.search(r"(gguf|awq|gptq|exl2|mlx|onnx)", cur.split("/")[-1], re.I)
            if not fmt:
                raise LookupError(f"{cur} has no model weights Sheep can read. Paste the ID of the model's own page "
                                  "(the one with its full weights), or a .gguf file")
            kind = kind0 = fmt.group(1).lower()
        seen.add(cur)
        bases = [b for b in _bases(getattr(info, "card_data", None)) if b not in seen]
        if bases:
            cur = bases[0]
            # a card can name the wrong base: mlx-community/Qwen2.5-3B-Instruct-4bit says Qwen/Qwen2.5-3B.
            # The copy's own name at the base's organisation wins when that is an original itself
            # (TheBloke's Llama-2-7B-Chat-GPTQ: meta-llama/Llama-2-7B-Chat is Meta's .pth, not it)
            alt = f"{cur.split('/')[0]}/{guess}" if guess else ""
            if alt and alt.lower() != cur.lower() and alt not in seen and original(alt):
                cur = alt
            continue
        # no base named (hugging-quants' AWQ pages): the same name, suffix off, on the Hub
        found = [r for r in (search(guess) if guess else []) if r not in seen and r != asked]
        hit = next((r for r in found if original(r)), None)
        if hit:
            cur = hit
            continue
        can = "" if kind in UNPACK or kind == "gguf" else ", which Sheep cannot read,"
        raise LookupError(f"{cur} is {WORDS.get(kind, 'a converted copy')}{can} and its page does not name the "
                          "original it was made from. Paste the original model's ID instead (its page has the "
                          "full weights)")
    raise LookupError(f"{asked}: followed its base_model {hops} times without reaching original weights")


def _search(name):
    """Hub repos named exactly `name` (any organisation), most downloaded first."""
    try:
        from huggingface_hub import HfApi
        ms = HfApi().list_models(search=name, sort="downloads", limit=30)
        return [m.id for m in ms if m.id.split("/")[-1].lower() == name.lower()][:5]
    except Exception:
        return []


def _read_config(repo, info, names):
    if "config.json" not in names:
        return {}
    try:
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(repo, "config.json", revision=getattr(info, "sha", None)), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


OFFLINE = ("getaddrinfo", "11001", "name resolution", "nameresolution", "connection", "timed out", "timeout",
           "max retries", "unreachable", "network", "offline")


def hub_error(exc, repo):
    """A person's words for any failure to read a Hugging Face page (2026-10-21: no network
    showed "[Errno 11001] getaddrinfo failed")."""
    m = str(exc).lower()
    gated = gated_reason(exc, repo)
    if gated:
        return gated
    if any(x in m for x in ("404", "not found", "repositorynotfound")) or "notfound" in type(exc).__name__.lower():
        return f"There is no model called {repo} on Hugging Face. Check the spelling (organisation/model)."
    if any(x in m for x in OFFLINE) or isinstance(exc, (ConnectionError, TimeoutError)):
        return "Sheep could not reach Hugging Face. Check the internet connection, then press Check again."
    return f"Hugging Face did not answer as expected ({str(exc)[:120]}). Press Check again in a moment."


def gated_reason(exc, repo):
    """A person's words for a Hugging Face refusal, or None if it is not one."""
    m = str(exc).lower()
    if "repository not found" in m or "repositorynotfound" in type(exc).__name__.lower():
        return None              # the Hub answers a missing repo with 401 too: it is not gated (J3, 2026-10-21)
    if not any(x in m for x in ("gated", "401", "403", "unauthorized", "restricted", "access to model")):
        return None
    if signed_in():
        return (f"{repo} is gated: accept its licence at huggingface.co/{repo} (signed in as {signed_in()}), "
                "then Check again")
    return (f"{repo} is gated: sign in to Hugging Face (just under the model ID), accept its licence at "
            f"huggingface.co/{repo}, then Check again")


_WHO = {"token": None, "name": "", "t": 0.0}


def signed_in():
    """The Hugging Face user name this machine is signed in as, or "" (asked once per token
    and five minutes: the settings page reads it on every draw)."""
    import time
    try:
        from huggingface_hub import get_token
        tok = get_token()
        if not tok:
            return ""
        if _WHO["token"] == tok and time.time() - _WHO["t"] < 300:
            return _WHO["name"]
        from huggingface_hub import HfApi
        name = str(HfApi().whoami(token=tok).get("name") or "")
        _WHO.update(token=tok, name=name, t=time.time())
        return name
    except Exception:
        return ""


def sign_in(token):
    from huggingface_hub import HfApi, login
    token = (token or "").strip()
    if not re.fullmatch(r"hf_[A-Za-z0-9]{20,}", token):
        raise ValueError("a Hugging Face token starts with hf_ (huggingface.co/settings/tokens, read access)")
    who = HfApi().whoami(token=token)
    login(token=token, add_to_git_credential=False)
    _WHO.update(token=None)
    return str(who.get("name") or "")


def sign_out():
    from huggingface_hub import logout
    try:
        logout()
    except Exception:
        pass
    _WHO.update(token=None, name="")
