"""More picture and speech models than the one of each Sheep ships with (2026-10-27).

Pictures: any Stable Diffusion 1.5 model on Hugging Face (the thousands of fine-tunes: realistic,
illustration, anime...). They share SD 1.5's shape, so the 4-step LCM add-on and the video motion
module work on them unchanged, and they fit a 4 GB card. They become the base for pictures AND video.
Bigger families (SDXL, SD 3, Flux) are refused with the reason, not loaded and crashed.

Speech: any faster-whisper model (CTranslate2 Whisper: tiny to large-v3, distil). One file format,
so the same loader reads them; a bigger one is slower and more accurate.
"""
import json
import re
from pathlib import Path

from .adapters import MEDIA_DIR, whisper_cache

STYLES = MEDIA_DIR / "styles"
SD15_CLASS = "StableDiffusionPipeline"
TOO_BIG = {"StableDiffusionXLPipeline": "SDXL", "StableDiffusion3Pipeline": "Stable Diffusion 3",
           "FluxPipeline": "Flux", "PixArtAlphaPipeline": "PixArt", "PixArtSigmaPipeline": "PixArt"}
STYLE_FILES = ("model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json", "text_encoder/*.safetensors",
               "unet/config.json", "unet/*.safetensors", "vae/config.json", "vae/*.safetensors", "feature_extractor/*")
WHISPER_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json")


def slug(repo):
    return re.sub(r"[^A-Za-z0-9_.-]+", "--", repo)


def style_dir(repo):
    return STYLES / slug(repo)


def style_ready(repo):
    d = style_dir(repo)
    return (d / "model_index.json").is_file() and any((d / "unet").glob("*.safetensors"))


def _read_json(repo, name):
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(repo, name), encoding="utf-8") as f:
        return json.load(f)


def check_style(repo, files):
    """(ok, reason, architecture) for a picture model that is not Sheep's own."""
    if "model_index.json" not in files:
        return False, "this repository is not a diffusers picture pipeline (no model_index.json)", ""
    try:
        cls = str(_read_json(repo, "model_index.json").get("_class_name") or "")
    except Exception as exc:
        return False, f"its model_index.json could not be read: {str(exc)[:120]}", ""
    if cls in TOO_BIG:
        return False, (f"this is a {TOO_BIG[cls]} model: it needs about 8 GB of graphics memory and a loader Sheep "
                       "does not have yet. Stable Diffusion 1.5 models work"), cls
    if cls != SD15_CLASS:
        return False, f"its pipeline is {cls or 'unknown'}; Sheep's picture worker loads Stable Diffusion 1.5 models", cls
    if not any(f.startswith("unet/") and f.endswith(".safetensors") for f in files):
        return False, "it has no safetensors weights for its UNet (older .bin files are not loaded)", cls
    try:
        dim = int(_read_json(repo, "unet/config.json").get("cross_attention_dim") or 0)
    except Exception:
        dim = 0
    if dim and dim != 768:
        return False, "its UNet is not SD 1.5's shape (it is likely SD 2), so the 4-step add-on would not fit it", cls
    return True, "a Stable Diffusion 1.5 model: works with Sheep's 4-step pictures and video", "StableDiffusionPipeline (SD 1.5 style)"


def style_bytes(info, prefer_fp16=True):
    """Download size: fp16 weights when the repo has them, else the full ones."""
    import fnmatch
    names = [(str(f.rfilename), int(getattr(f, "size", 0) or 0)) for f in (info.siblings or [])]
    has16 = any(".fp16." in n for n, _ in names)
    total = 0
    for n, size in names:
        if not any(fnmatch.fnmatch(n, p) for p in STYLE_FILES):
            continue
        if n.endswith(".safetensors") and prefer_fp16 and has16 and ".fp16." not in n:
            continue
        total += size
    return total


def install_style(repo):
    """Download the picture model's pipeline files (fp16 when it has them). Returns its folder."""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(repo, files_metadata=True)
    names = [str(f.rfilename) for f in (info.siblings or [])]
    allow = list(STYLE_FILES)
    if any(".fp16." in n for n in names):
        allow = [p for p in allow if not p.endswith("*.safetensors")] + [p.replace("*.safetensors", "*.fp16.safetensors")
                                                                       for p in STYLE_FILES if p.endswith("*.safetensors")]
    d = style_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, local_dir=str(d), allow_patterns=allow)
    if not style_ready(repo):
        raise RuntimeError("the download did not complete; try again")
    return d


def check_whisper(files):
    need = ("config.json", "model.bin", "tokenizer.json")
    if all(n in files for n in need) and any(n.startswith("vocabulary.") for n in files):
        return True, "a faster-whisper (CTranslate2) speech model: Sheep's speech-to-text loads it"
    return False, "this is not a faster-whisper model (it needs model.bin, config.json, tokenizer.json and a vocabulary file)"


def whisper_snapshot_of(repo):
    """The downloaded snapshot of a faster-whisper repo in Sheep's cache, or None."""
    snaps = whisper_cache() / ("models--" + repo.replace("/", "--")) / "snapshots"
    try:
        for rev in sorted(snaps.iterdir()):
            if rev.is_dir() and (rev / "model.bin").is_file() and (rev / "config.json").is_file():
                return rev
    except OSError:
        pass
    return None


def install_whisper(repo):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, cache_dir=str(whisper_cache()), allow_patterns=list(WHISPER_FILES))
    snap = whisper_snapshot_of(repo)
    if not snap:
        raise RuntimeError("the download did not complete; try again")
    return snap


def chosen_base(cfg):
    """The picture/video base: the person's SD 1.5 style when it is here, else Sheep's own SD 1.5."""
    repo = ((cfg or {}).get("media_models") or {}).get("image_style") or ""
    return style_dir(repo) if repo and style_ready(repo) else MEDIA_DIR / "sd15"
