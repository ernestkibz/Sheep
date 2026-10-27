"""Typed, local model slots and Hugging Face preflight checks.

The slot gate deliberately accepts only architectures that the current local
runtime can actually load. A red hardware fit is advisory; a task/architecture
mismatch is a hard rejection. Metadata and small config files may be read during
Check, but model weights are downloaded only by the explicit Use operation.
"""
import fnmatch
import os
import importlib.util
import re
import shutil
from pathlib import Path

from .paths import HOME

SLOTS = (
    {"id": "chat", "label": "Chat", "task": "text generation", "hf": True,
     "description": "The model you chat with: any Hugging Face model, a folder or a .gguf file."},
    {"id": "checker", "label": "Checker", "task": "text generation", "hf": True,
     "description": "A small model that reads each answer before you see it; examined before it starts."},
    {"id": "image", "label": "Image / media", "task": "text-to-image", "hf": True,
     "description": "Makes pictures on this computer."},
    {"id": "video", "label": "Video", "task": "text-to-video", "hf": True,
     "description": "Makes short video clips on this computer."},
    {"id": "photo_guidance", "label": "Photo guidance", "task": "image-conditioned video", "hf": True,
     "description": "Keeps a video close to the photos you give it."},
    {"id": "speech_input", "label": "Speech input", "task": "automatic speech recognition", "hf": True,
     "description": "Turns speech and recordings into text on this computer."},
    {"id": "speech_output", "label": "Speech output", "task": "text-to-speech", "hf": False,
     "description": "Reads text aloud with the voices Windows already has."},
)
SLOT_BY_ID = {row["id"]: row for row in SLOTS}

# These are the only media checkpoints wired to the installed local workers.
# Extending the accepted set requires adding and validating a matching loader.
MEDIA_REPOS = {
    "image": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "sd15-lcm", "image"),
    "video": ("ByteDance/AnimateDiff-Lightning", "animatediff-lightning", "video"),
    "photo_guidance": ("h94/IP-Adapter", "ip-adapter-plus", "video-reference"),
    "speech_input": ("Systran/faster-whisper-small", "whisper-small", "audio"),
}
# what a text checkpoint brings besides its weights (the quantizer downloads the same)
SMALL_FILES = ("*.json", "*.jinja", "tokenizer*", "license*", "use_policy*", "readme.md", "*.model", "*.tiktoken",
               "merges.txt", "vocab.txt")
HF_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _size(info):
    return sum(int(getattr(file, "size", 0) or 0) for file in (info.siblings or []))


def _selected_size(info, patterns):
    return sum(int(getattr(file, "size", 0) or 0) for file in (info.siblings or [])
               if any(fnmatch.fnmatch(str(file.rfilename).lower(), pattern.lower())
                      for pattern in patterns))


def _base(slot, repo_id):
    return {"slot": slot, "repo_id": repo_id, "available": False, "compatible": False,
            "reason": "", "license": "", "download_bytes": 0, "revision": "",
            "pipeline_tag": "", "architecture": "", "recommendation": None,
            "fit": {"level": "red", "label": "Not checked", "reason": "Local fit was not measured."}}


def free_bytes():
    """Free space where Sheep keeps models (MLMI_FAKE_FREE_GB: the journeys' nearly full disk)."""
    if os.environ.get("MLMI_FAKE_FREE_GB"):
        return int(float(os.environ["MLMI_FAKE_FREE_GB"]) * 1e9)
    return shutil.disk_usage(HOME).free


def check(slot, repo_id, *, device=None, directml=None, cuda_ready=False, plan_fn=None, device_pref="auto", plan_kw=None):
    """Inspect an HF repo without fetching weight files and enforce its feature slot."""
    result = _base(slot, repo_id)
    if slot not in SLOT_BY_ID:
        result["reason"] = "unknown model slot"
        return result
    if not SLOT_BY_ID[slot]["hf"]:
        result["reason"] = ("speech output is currently provided by Windows voices; this build has no local Hugging Face "
                             "text-to-speech runtime" if slot == "speech_output" else
                             "this slot has no compatible Hugging Face model runtime")
        return result
    if not isinstance(repo_id, str) or not HF_ID.fullmatch(repo_id.strip()):
        result["reason"] = "enter a Hugging Face repository ID in org/model form"
        return result
    repo_id = repo_id.strip()
    result["repo_id"] = repo_id
    from . import hf_source
    if slot in ("chat", "checker"):
        # a GGUF, GPTQ, AWQ or MLX page: the original it was made from is found and checked
        # instead (Ernest, 2026-10-19: Sheep starts where llama.cpp starts, from the originals)
        # a page that holds a ready Sheep file (published from Sheep, Part C5): downloaded, no build
        try:
            from huggingface_hub import HfApi
            from .publish import ssf_on_page
            sib0 = HfApi().model_info(repo_id, files_metadata=True).siblings or []
            ready = ssf_on_page([str(x.rfilename) for x in sib0])
        except Exception:
            ready = None
        if ready:
            size = next((int(getattr(x, "size", 0) or 0) for x in sib0 if str(x.rfilename) == ready), 0)
            result.update(available=True, compatible=True, ssf=ready, download_bytes=size,
                          reason=f"A ready Sheep file ({ready}): it downloads and runs, with no build.")
            return result
        try:
            original, note, _kind = hf_source.resolve(repo_id)
        except LookupError as exc:
            # no original anywhere (a fine-tune published only as GGUF, GPTQ or AWQ, or Meta's raw
            # .pth): Sheep converts the copy itself (Parts C1, C6)
            fb = hf_source.fallback(repo_id)
            if not fb:
                result["reason"] = str(exc)
                return result
            result.update(available=True, compatible=True, download_bytes=fb["bytes"],
                          recommendation={"quant_level": "balanced", "label": "Q4/6 Balanced"},
                          reason=hf_source.convert_note(fb, repo_id))
            result["gguf" if fb["how"] == "gguf" else "unpack"] = fb.get("file") or fb["kind"]
            return result
        except Exception as exc:
            result["reason"] = hf_source.hub_error(exc, repo_id)
            return result
        if original != repo_id:
            result.update(original_of=repo_id, note=note, repo_id=original)
            repo_id = original
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as exc:
        from . import hf_source
        result["reason"] = hf_source.hub_error(exc, repo_id)
        return result
    result.update(available=True, revision=str(info.sha or ""), pipeline_tag=str(info.pipeline_tag or ""))
    card = getattr(info, "card_data", None)
    if card:
        result["license"] = str(getattr(card, "license", "") or "")
    files = {str(x.rfilename).lower() for x in (info.siblings or [])}
    if slot in ("chat", "checker"):
        # exactly what the download takes: the weight files transformers would load, and the
        # small files beside them (hf_source.weight_files: no subfolders, no second format)
        weights = set(hf_source.weight_files([str(x.rfilename) for x in (info.siblings or [])]))
        result["download_bytes"] = sum(int(getattr(x, "size", 0) or 0) for x in (info.siblings or [])
                                       if str(x.rfilename) in weights or
                                       ("/" not in str(x.rfilename) and any(fnmatch.fnmatch(str(x.rfilename).lower(), p)
                                                                         for p in SMALL_FILES)))
    else:
        result["download_bytes"] = _size(info)
    needed = free_bytes()
    if slot in ("chat", "checker"):
        try:
            from .levels import inspect, cards
            model = inspect(repo_id)
        except Exception as exc:
            result["reason"] = (hf_source.gated_reason(exc, repo_id) or
                                f"repository metadata was found, but Sheep could not inspect its model config: {str(exc)[:180]}")
            return result
        result["architecture"] = model.get("model_type", "")
        result["license"] = model.get("licence") or result["license"]
        tag = result["pipeline_tag"]
        text_tags = {"text-generation", "text2text-generation", "text-generation-inference"}
        if model.get("kind") != "text" or (tag and tag not in text_tags and not model.get("text_path")):
            result["reason"] = f"{repo_id} is made for {tag or model.get('kind')}, not for chat; pick a text model for this job"
            return result
        result["remote_code"] = bool(model.get("remote_code"))
        if not model.get("builds") and not model.get("fallback"):
            result["reason"] = f"Sheep cannot build this kind of model yet: {model.get('note') or 'its settings are not ones Sheep knows'}"
            return result
        original = result["download_bytes"]
        quantized = int(model.get("n_weights", 0) * 0.57)
        need_bytes = original + quantized
        fit_level = "green" if needed >= need_bytes * 1.3 else "yellow" if needed >= need_bytes else "red"
        level_label = {"green": "Good fit", "yellow": "Tight disk space", "red": "Low disk space"}[fit_level]
        result["fit"] = {"level": fit_level, "label": level_label,
                          "reason": f"the download and the built model need about {need_bytes / 1e9:.1f} GB; "
                                    f"{needed / 1e9:.1f} GB is free"}
        if fit_level == "red":
            # never start a download that cannot finish: say how much to free instead
            result["reason"] = (f"Not enough free space: this model needs about {need_bytes / 1e9:.1f} GB and "
                                f"{needed / 1e9:.1f} GB is free. Free up {max(0.1, (need_bytes - needed) / 1e9):.1f} GB, "
                                "then press Check again.")
            return result
        result["recommendation"] = {"quant_level": "balanced", "label": "Q4/6 Balanced"}
        if device is not None and plan_fn is not None:
            cards_result = cards(model, device, plan_fn, device_pref, **(plan_kw or {}))   # as /api/levels
            result["recommendation"] = next(({"quant_level": c["id"], "label": c["name"]}
                                              for c in cards_result if c.get("recommended")),
                                             result["recommendation"])
            best = next((c for c in cards_result if c.get("recommended")), cards_result[0] if cards_result else {})
            level = "green" if best.get("fit") == "green" else "yellow" if best.get("fit") == "amber" else "red"
            result["fit"] = {"level": level, "label": {"green": "Good fit", "yellow": "Constrained fit",
                                                            "red": "Poor hardware fit"}[level],
                              "reason": f"recommended {result['recommendation']['label']}; {best.get('fit_word', 'fit unknown')}"
                                        + (f", about {round(best['est_tok_s'] / 1.3, 1)} words a second here" if best.get("est_tok_s") else "")}
        result["compatible"] = True
        result["reason"] = "Sheep can build this model for this computer"
        if model.get("fallback") == "fit":
            result["fit_needed"] = True
            result["reason"] = ("Sheep will fit this model to its engine first (a few minutes); if nothing fits, "
                                "it runs in compatibility mode, with the maker's own code")
        elif model.get("fallback") == "compat":
            result["compat_needed"] = True
            result["reason"] = ("Sheep's engine does not run this model's maths yet: it runs in compatibility mode, "
                                "with the maker's own code (slower, but correct)")
        return result

    return _check_media(slot, repo_id, result, info, files, device, directml, cuda_ready)


def _check_media(slot, repo_id, result, info, files, device, directml, cuda_ready):
    """Check the repository files and local runtime for an image, video or speech slot."""
    from huggingface_hub import HfApi

    expected_repo, media_id, install_kind = MEDIA_REPOS[slot]
    if repo_id.casefold() != expected_repo.casefold():
        result["reason"] = f"this runtime supports {expected_repo} for the {SLOT_BY_ID[slot]['task']} slot; " \
                            "other architectures are rejected until Sheep has a compatible loader"
        return result
    tag = result["pipeline_tag"]
    if slot == "image":
        required = ("model_index.json", "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json", "tokenizer/merges.txt", "text_encoder/config.json",
                    "text_encoder/model.fp16.safetensors", "unet/config.json",
                    "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
                    "vae/diffusion_pytorch_model.fp16.safetensors", "feature_extractor/preprocessor_config.json")
        if tag not in ("text-to-image", "") or not all(name in files for name in required):
            result["reason"] = "the repo does not expose a Stable Diffusion pipeline supported by Sheep's image worker"
            return result
        result["architecture"] = "StableDiffusionPipeline (SD 1.5)"
        image_patterns = ("model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
                          "text_encoder/model.fp16.safetensors", "unet/config.json",
                          "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
                          "vae/diffusion_pytorch_model.fp16.safetensors", "feature_extractor/*", "readme.md")
        result["download_bytes"] = _selected_size(info, image_patterns)
        # Image use installs the LCM LoRA too; include its selected files in the estimate.
        try:
            lcm = HfApi().model_info("latent-consistency/lcm-lora-sdv1-5", files_metadata=True)
            result["download_bytes"] += _selected_size(lcm, ("*.safetensors", "*.json", "readme.md"))
        except Exception as exc:
            result["reason"] = f"Sheep's image runtime dependency could not be checked: {str(exc)[:160]}"
            return result
    elif slot == "video":
        if not any(name.endswith("animatediff_lightning_4step_diffusers.safetensors") for name in files):
            result["reason"] = "the repository is missing the AnimateDiff-Lightning motion weights required by Sheep"
            return result
        result["architecture"] = "AnimateDiff-Lightning motion adapter"
        result["download_bytes"] = _selected_size(info, ("animatediff_lightning_4step_diffusers.safetensors", "readme.md"))
    elif slot == "photo_guidance":
        if not all(name in files for name in ("models/ip-adapter-plus_sd15.bin", "models/image_encoder/config.json",
                                              "models/image_encoder/model.safetensors")):
            result["reason"] = "the repository does not contain the IP-Adapter Plus and image encoder files Sheep loads"
            return result
        result["architecture"] = "IP-Adapter Plus SD 1.5"
        result["download_bytes"] = _selected_size(info, ("models/ip-adapter-plus_sd15.bin", "models/image_encoder/config.json",
                                                          "models/image_encoder/model.safetensors", "readme.md"))
    elif slot == "speech_input":
        if tag not in ("automatic-speech-recognition", ""):
            result["reason"] = f"Hugging Face tags this repository as {tag}, not speech recognition"
            return result
        if not all(name in files for name in ("model.bin", "config.json", "tokenizer.json")) or not any(
                fnmatch.fnmatch(name, "vocabulary.*") for name in files):
            result["reason"] = "this transcriber needs CTranslate2 Whisper model, config, tokenizer and vocabulary files"
            return result
        result["architecture"] = "CTranslate2 Whisper small"
        result["download_bytes"] = _selected_size(info, ("config.json", "model.bin",
                                                          "tokenizer.json", "vocabulary.*"))
    runtime_ok = {
        "image": lambda: __import__("mlmi.backend.adapters", fromlist=["image_installable"]).image_installable(),
        "video": lambda: __import__("mlmi.backend.adapters", fromlist=["video_installable"]).video_installable(),
        "photo_guidance": lambda: __import__("mlmi.backend.adapters", fromlist=["video_installable"]).video_installable(),
        "speech_input": lambda: bool(importlib.util.find_spec("faster_whisper") and importlib.util.find_spec("huggingface_hub")),
    }[slot]()
    if not runtime_ok:
        result["reason"] = f"the required local {SLOT_BY_ID[slot]['task']} packages are not installed"
        return result
    result["compatible"] = True
    result["reason"] = "matches Sheep's installed local feature runtime"
    if device is not None:
        from . import adapters
        item = next((row for row in adapters.catalogue({}) if row["id"] == media_id), None)
        if item:
            result["fit"] = adapters.fit(item, device, directml, cuda_ready)
            result["fit"]["level"] = {"amber": "yellow"}.get(result["fit"].get("level"), result["fit"].get("level"))
    result["recommendation"] = {"model_id": media_id, "label": "Sheep-supported local model"}
    return result


def current(state):
    """All persistent feature slots and their currently selected model IDs."""
    from . import adapters
    chosen = adapters.defaults(state.cfg)
    config = state.cfg.get("media_models") or {}
    active = {
        "chat": state.model_path or "",
        "checker": (state.cfg.get("checker") or {}).get("path", ""),
        "image": config.get("image", chosen.get("image", "")),
        "video": config.get("video", chosen.get("video", "")),
        "photo_guidance": config.get("video-reference", chosen.get("video-reference", "")),
        "speech_input": config.get("audio-in", chosen.get("audio-in", "")),
        "speech_output": config.get("audio-out", chosen.get("audio-out", "")),
    }
    models = {row["id"]: active.get(row["id"], "") for row in SLOTS}
    catalogue = adapters.catalogue(state.cfg)
    mapped = {"image": "image", "video": "video", "photo_guidance": "video-reference",
              "speech_input": "audio-in", "speech_output": "audio-out"}
    out = []
    for spec in SLOTS:
        value = models[spec["id"]]
        item = next((m for m in catalogue if m["id"] == value), None) if spec["id"] in mapped else None
        out.append({**spec, "active": value, "active_name": item["name"] if item else (Path(value).name if value else "Not selected"),
                    "ready": bool(item.get("ready")) if item else bool(value),
                    "disabled": bool(item.get("disabled")) if item else False})
    return out
