"""Media models: pictures, sound and video, used as tools by the chat model.

A media model never holds the conversation; the chat model (or Nemotron) talks
and plans, and the orchestrator calls these when a job needs a picture, a
transcript, a voice or a video. One list, shown in the same model drop-down as
the chat models, each with what it does and whether it is ready here.
"""
import importlib.util
import os
from pathlib import Path

from ..paths import HOME

MEDIA_DIR = HOME / "media"


def image_ready():
    return ((MEDIA_DIR / "sd15" / "model_index.json").exists()
            and any((MEDIA_DIR / "lcm").glob("*.safetensors")))


def video_ready():
    base = MEDIA_DIR / "sd15"
    required = (
        "model_index.json", "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt",
        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
        "feature_extractor/preprocessor_config.json",
    )
    return (all((base / name).is_file() for name in required)
            and (MEDIA_DIR / "animatediff" / "animatediff_lightning_4step_diffusers.safetensors").is_file())


def video_reference_ready():
    root = MEDIA_DIR / "ip-adapter" / "models"
    encoder = root / "image_encoder"
    return ((root / "ip-adapter-plus_sd15.bin").is_file()
            and (encoder / "config.json").is_file()
            and any((encoder / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")))


def video_installable():
    """The video tool can be offered before its large weights are downloaded."""
    import shutil
    runtime = ("torch", "diffusers", "safetensors", "huggingface_hub", "PIL")
    return bool(shutil.which("ffmpeg")) and all(importlib.util.find_spec(name) for name in runtime)


def image_installable():
    runtime = ("torch", "diffusers", "safetensors", "huggingface_hub", "PIL")
    return all(importlib.util.find_spec(name) for name in runtime)


def whisper_cache():
    """A Sheep-owned HF cache lets the person remove speech weights safely."""
    return MEDIA_DIR / "hf-cache"


def whisper_snapshot():
    snapshots = whisper_cache() / "models--Systran--faster-whisper-small" / "snapshots"
    # CTranslate2's faster-whisper loader reads these five files. The official
    # Systran snapshot does not ship a Transformers preprocessor_config.json.
    required = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")
    try:
        for revision in sorted(snapshots.iterdir()):
            if revision.is_dir() and all((revision / name).is_file() for name in required):
                return revision
    except OSError:
        pass
    return None


def legacy_whisper_snapshot():
    """Find a complete pre-Sheep Whisper cache so an upgrade need not redownload it."""
    hub = os.environ.get("HF_HUB_CACHE")
    if not hub:
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            hf_home = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
                                   "huggingface")
        hub = os.path.join(hf_home, "hub")
    snapshots = Path(hub) / "models--Systran--faster-whisper-small" / "snapshots"
    required = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")
    try:
        for revision in sorted(snapshots.iterdir()):
            if revision.is_dir() and all((revision / name).is_file() for name in required):
                if revision.resolve() != whisper_snapshot():
                    return revision
    except OSError:
        pass
    return None


def _whisper_ready():
    return bool(importlib.util.find_spec("faster_whisper")) and whisper_snapshot() is not None


def catalogue(cfg=None):
    """Media model choices with install/remove metadata and availability details."""
    image = image_ready()
    video = video_ready()
    sapi = os.name == "nt" and bool(importlib.util.find_spec("pyttsx3"))
    whisper = _whisper_ready()
    whisper_shared = not whisper and legacy_whisper_snapshot() is not None
    disabled = set((cfg or {}).get("disabled_media_models") or [])
    return [
        {"id": "sd15-lcm", "kind": "image", "name": "Stable Diffusion 1.5 + LCM", "does": "pictures from words, 4 steps",
         "licence": "CreativeML OpenRAIL-M", "ready": image, "install": "image", "removable": True,
         "installable": image_installable(), "disabled": "sd15-lcm" in disabled,
         "why": "" if image else "not downloaded yet"},
        {"id": "animatediff-lightning", "kind": "video", "name": "AnimateDiff-Lightning (on SD 1.5)",
         "does": "AI video clips from words, 1-6 s, 4 steps", "licence": "CreativeML OpenRAIL-M",
         "ready": video, "install": "video", "removable": True, "installable": video_installable(),
         "disabled": "animatediff-lightning" in disabled, "why": "" if video else "not downloaded yet"},
        {"id": "ip-adapter-plus", "kind": "video", "name": "IP-Adapter Plus photo guidance",
         "does": "animate selected photos as visual references", "licence": "Apache-2.0",
         "ready": video_reference_ready(), "install": "video-reference", "removable": True,
         "installable": video_installable(), "disabled": "ip-adapter-plus" in disabled,
         "why": "" if video_reference_ready() else "not downloaded yet"},
        {"id": "whisper-small", "kind": "audio", "name": "Whisper small", "does": "speech to text",
         "licence": "MIT", "ready": whisper, "reuse_cached": whisper_shared, "install": "audio", "removable": True,
         "installable": bool(importlib.util.find_spec("faster_whisper") and importlib.util.find_spec("huggingface_hub")),
         "disabled": "whisper-small" in disabled,
         "why": "" if whisper else "already in the shared Hugging Face cache" if whisper_shared else "not downloaded yet"},
        {"id": "windows-voices", "kind": "audio", "name": "Windows voices", "does": "text to speech",
         "licence": "part of Windows", "ready": sapi, "install": "", "removable": True, "installable": sapi,
         "disabled": "windows-voices" in disabled,
         "why": "" if sapi else "Windows speech is not available"},
    ]


def defaults(cfg):
    """The media model chosen for each kind: the person's choice, else the first ready one."""
    chosen = dict(cfg.get("media_models") or {})
    out = {}
    disabled = set(cfg.get("disabled_media_models") or [])
    for m in catalogue(cfg):
        key = role(m)
        if m["id"] in disabled:
            continue
        if key not in out and m["ready"]:
            out[key] = m["id"]
        if m["id"] not in disabled and chosen.get(key) == m["id"] and m["ready"]:
            out[key] = m["id"]
    return out


def role(m):
    """The job a media model does: image, video, audio-in (speech to text) or audio-out (voice)."""
    if m["id"] == "ip-adapter-plus":
        return "video-reference"
    if m["kind"] == "audio":
        return "audio-in" if "to text" in m["does"] else "audio-out"
    return m["kind"]


def fit(m, dev, directml=None, cuda_ready=False):
    """Estimate local fit with the devices Sheep actually probes; green/yellow/red."""
    ram = float((dev.get("ram") or {}).get("available") or 0) / 2**30
    cores = int((dev.get("cpu") or {}).get("cores") or 1)
    gpu = dev.get("gpu") or {}
    choices = []
    if gpu and cuda_ready:
        free = max(0.0, (float(gpu.get("vram_total") or 0) - float(gpu.get("vram_used") or 0)) / 2**30)
        choices.append(("NVIDIA CUDA", str(gpu.get("name") or "NVIDIA GPU"), free, False))
    for a in directml if directml is not None else []:
        if not a.get("ok"):
            continue
        choices.append(("DirectML", str(a.get("name") or "DirectML GPU"), float(a.get("vram") or 0) / 2**30,
                        bool(a.get("integrated"))))
    model_id = m.get("id", "")
    if model_id == "windows-voices":
        ready = bool(m.get("ready"))
        return {"level": "green" if ready else "red", "label": "Good match" if ready else "Unavailable",
                "device": "Windows speech", "reason": "Available on this Windows installation" if ready else m.get("why", "Unavailable")}
    if model_id == "whisper-small":
        level = "green" if ram >= 3 and cores >= 4 else "yellow" if ram >= 1.5 else "red"
        return {"level": level, "label": {"green": "Good match", "yellow": "May run slowly", "red": "Tight on memory"}[level],
                "device": "CPU", "reason": f"Whisper small needs about 1.5 GB RAM; {ram:.1f} GB RAM and {cores} CPU cores are available"}
    need = 4.5 if model_id == "ip-adapter-plus" else 3.6 if m.get("kind") == "video" else 3.2
    rated = []
    for backend, name, available, integrated in choices:
        if available >= need + 1.0 and not integrated:
            level = "green"
        elif available >= need or (integrated and available >= need - 0.5):
            level = "yellow"
        else:
            continue
        rated.append((0 if level == "green" else 1, {"green": level, "yellow": level}[level], backend, name, available))
    if rated:
        _, level, backend, name, available = min(rated)
        return {"level": level, "label": "Good match" if level == "green" else "May run slowly",
                "device": name, "reason": f"{backend} has about {available:.1f} GB available; this model uses about {need:.1f} GB"}
    if ram >= 16:
        return {"level": "yellow", "label": "Runs slowly on CPU", "device": "CPU",
                "reason": f"No suitable GPU found; {ram:.1f} GB system RAM is available"}
    return {"level": "red", "label": "Poor match", "device": "Local hardware",
            "reason": f"No checked GPU has about {need:.1f} GB available and system RAM is {ram:.1f} GB"}
