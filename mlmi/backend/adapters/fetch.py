"""Media models into Sheep's own folder (~/.mlmi/media/<name>), as plain files.

The Hugging Face cache links its files with symlinks, which Windows refuses
without Developer Mode (WinError 1314 left Stable Diffusion without its
scheduler and tokenizer configs, 2026-10-05). A plain folder works everywhere,
and the adapters load from it with no network.

    python -m mlmi.backend.adapters.fetch sd15 lcm
"""
import shutil
import sys
from pathlib import Path

from ..paths import HOME

MEDIA = HOME / "media"
MODELS = {
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5",
             ["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
              "text_encoder/model.fp16.safetensors", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
              "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors", "feature_extractor/*", "README.md"]),
    "lcm": ("latent-consistency/lcm-lora-sdv1-5", ["*.safetensors", "*.json", "README.md"]),
    # the motion module that turns SD 1.5 into a video model, distilled to 4 steps
    "animatediff": ("ByteDance/AnimateDiff-Lightning", ["animatediff_lightning_4step_diffusers.safetensors", "README.md"]),
    # SD 1.5 IP-Adapter Plus and its CLIP image encoder, installed only for photo animation
    "ip-adapter": ("h94/IP-Adapter", ["models/ip-adapter-plus_sd15.bin", "models/image_encoder/config.json",
                                     "models/image_encoder/model.safetensors", "README.md"]),
}


def path(name):
    return MEDIA / name


def fetch(name, progress=None):
    """Download (or reuse from the cache) into ~/.mlmi/media/<name>; returns the folder."""
    from huggingface_hub import snapshot_download
    repo, patterns = MODELS[name]
    dst = path(name)
    dst.mkdir(parents=True, exist_ok=True)
    # reuse what the cache already holds, as real files
    try:
        snap = Path(snapshot_download(repo, allow_patterns=patterns, local_files_only=True))
        for f in snap.rglob("*"):
            if f.is_file():
                out = dst / f.relative_to(snap)
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f.resolve(), out)
    except Exception:
        pass
    options = {"allow_patterns": patterns, "local_dir": str(dst)}
    if progress:
        from tqdm.auto import tqdm

        class DownloadProgress(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                progress(self.n, self.total, kwargs.get("desc", ""), kwargs.get("unit", ""))

            def update(self, n=1):
                result = super().update(n)
                progress(self.n, self.total, self.desc, self.unit)
                return result

        options["tqdm_class"] = DownloadProgress
    snapshot_download(repo, **options)
    return dst


if __name__ == "__main__":
    for n in sys.argv[1:] or list(MODELS):
        print(n, fetch(n), flush=True)
