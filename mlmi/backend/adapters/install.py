"""On-demand installation of media models, shared by every media entry point."""
import importlib.util
import logging
import shutil
import threading
import time
import uuid

from ..workspace import ToolError
from . import (MEDIA_DIR, image_ready, video_ready, video_reference_ready, whisper_cache,
               legacy_whisper_snapshot, _whisper_ready)

WHISPER_REPO = "Systran/faster-whisper-small"
_NAMES = {
    "image": "Stable Diffusion 1.5 + LCM",
    "video": "Stable Diffusion 1.5 + AnimateDiff-Lightning",
    "video-reference": "IP-Adapter Plus photo guidance",
    "audio": "Whisper small",
}
_MODEL_INSTALLS = {
    "sd15-lcm": "image", "animatediff-lightning": "video", "ip-adapter-plus": "video-reference",
    "whisper-small": "audio",
}


class MediaInstaller:
    """Serialize installs per process and let concurrent requests share the result."""

    def __init__(self, publish=lambda event: None):
        self.publish = publish
        self._condition = threading.Condition()
        self._active = False
        self._status = None
        self._cancel_event = threading.Event()

    def status(self):
        with self._condition:
            return dict(self._status) if self._status else None

    def _update(self, **fields):
        with self._condition:
            self._status.update(fields)
            current = dict(self._status)
            self._condition.notify_all()
        self.publish({"type": "media_install", **current})

    def _ready(self, kind):
        if kind == "image":
            return image_ready()
        if kind == "video":
            return video_ready()
        if kind == "video-reference":
            return video_reference_ready()
        return _whisper_ready()

    def _begin_or_wait(self, kind):
        with self._condition:
            while True:
                if self._ready(kind):
                    return None
                if not self._active:
                    break
                shared_id = self._status["id"]
                while self._active and self._status["id"] == shared_id:
                    self._condition.wait()
                result = dict(self._status)
                if result["id"] == shared_id and result["kind"] == kind and result["state"] in ("failed", "cancelled"):
                    raise ToolError(result["error"])
            self._active = True
            self._cancel_event.clear()
            self._status = {
                "id": uuid.uuid4().hex, "kind": kind, "model": _NAMES[kind],
                "state": "installing", "phase": "Preparing", "detail": "Checking installed files",
                "progress": None, "error": "",
            }
            current = dict(self._status)
        self.publish({"type": "media_install", **current})
        return current["id"]

    @staticmethod
    def _progress(emit, label, cancel_event=None):
        last_emit = [0.0]

        def update(current, total, desc="", unit=""):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("download cancelled by the user")
            now = time.monotonic()
            if now - last_emit[0] < 1 and not (total and current >= total):
                return
            last_emit[0] = now
            text = (desc or "").strip() or label
            unit = str(unit or "").lower()
            if total and total > 0 and unit in ("file", "files"):
                value = current / total
            else:
                value = None
                if total and total > 0 and unit in ("b", "byte", "bytes"):
                    text = f"{text} · {MediaInstaller._size(current)} of {MediaInstaller._size(total)}"
            emit(phase="Downloading", detail=text, progress=value)
        return update

    @staticmethod
    def _size(value):
        size = float(value)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1000 or unit == "GB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
            size /= 1000
        return f"{size:.1f} GB"

    @staticmethod
    def _user_error(exc):
        raw = str(exc).lower()
        if any(word in raw for word in ("connection", "timeout", "timed out", "network", "dns",
                                        "name resolution", "ssl", "proxy", "http error", "503", "502", "429")):
            return "Download failed. Check your internet connection, then retry the request."
        if "no space" in raw or "disk full" in raw:
            return "Download failed because the disk is full. Free space, then retry."
        if "package is not installed" in raw:
            return str(exc)
        return "Installation failed. Retry the request; expand Details for the cause."

    def ensure(self, kind):
        if kind not in _NAMES:
            raise ValueError(f"unknown media model kind: {kind}")
        job_id = self._begin_or_wait(kind)
        if job_id is None:
            return

        try:
            if kind in ("image", "video"):
                self._update(phase="Downloading", detail="Stable Diffusion 1.5 base model", progress=None)
                from .fetch import fetch
                fetch("sd15", self._progress(self._update, "Stable Diffusion 1.5 base model", self._cancel_event))
                if kind == "image":
                    self._update(phase="Downloading", detail="LCM image-generation adapter", progress=None)
                    fetch("lcm", self._progress(self._update, "LCM image-generation adapter", self._cancel_event))
                else:
                    self._update(phase="Downloading", detail="AnimateDiff-Lightning motion module", progress=None)
                    fetch("animatediff", self._progress(self._update, "AnimateDiff-Lightning motion module", self._cancel_event))
                if not self._ready(kind):
                    component = "LCM image-generation" if kind == "image" else "Stable Diffusion 1.5 or AnimateDiff-Lightning"
                    raise RuntimeError(f"download finished, but the {component} files are incomplete")
            elif kind == "video-reference":
                self._update(phase="Downloading", detail="IP-Adapter Plus photo-guidance model", progress=None)
                from .fetch import fetch
                fetch("ip-adapter", self._progress(self._update, "IP-Adapter Plus photo-guidance model", self._cancel_event))
                if not video_reference_ready():
                    raise RuntimeError("download finished, but the IP-Adapter Plus or image-encoder files are incomplete")
            else:
                self._update(phase="Preparing", detail="Checking for an existing Whisper small model", progress=None)
                if not importlib.util.find_spec("faster_whisper"):
                    raise RuntimeError("the faster-whisper package is not installed")
                legacy = legacy_whisper_snapshot()
                if legacy:
                    target = whisper_cache() / "models--Systran--faster-whisper-small" / "snapshots" / legacy.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    self._update(phase="Preparing", detail="Copying the existing speech model into Sheep's removable cache",
                                 progress=None)
                    shutil.copytree(legacy, target, dirs_exist_ok=True)
                    if _whisper_ready():
                        self._update(state="complete", phase="Ready", detail="Reused the existing local model; no download needed",
                                     progress=1, error="")
                        return
                self._update(phase="Downloading", detail="Whisper small speech-recognition model", progress=None)
                from huggingface_hub import snapshot_download
                snapshot_download(
                    WHISPER_REPO,
                    cache_dir=str(whisper_cache()),
                    allow_patterns=["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"],
                    tqdm_class=self._tqdm_class(self._progress(self._update, "Whisper small", self._cancel_event)),
                )
                if not _whisper_ready():
                    raise RuntimeError("download finished, but the Whisper small model files are incomplete")
            self._update(state="complete", phase="Ready", detail="Installed. Continuing your request.", progress=1, error="")
        except Exception as exc:
            detail = str(exc).strip()[:1000] or exc.__class__.__name__
            logging.getLogger(__name__).exception("media model installation failed (%s)", kind)
            cancelled = self._cancel_event.is_set()
            message = "Download cancelled. The previous model choice is unchanged." if cancelled else self._user_error(exc)
            self._update(state="cancelled" if cancelled else "failed",
                         phase="Download cancelled" if cancelled else "Installation failed",
                         detail="The model download was stopped." if cancelled else "The requested task could not start.",
                         progress=None, error=message, error_detail=detail)
            raise ToolError(f"could not install {_NAMES[kind]}: {message}") from exc
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()

    def cancel(self):
        """Stop an active download at its next file-progress boundary."""
        with self._condition:
            if not self._active or not self._status or self._status.get("state") != "installing":
                return {"cancelled": False, "why": "no media download is active"}
            self._cancel_event.set()
            self._condition.notify_all()
        self.publish({"type": "media_install", **self.status(), "phase": "Stopping download"})
        return {"cancelled": True, "id": self.status().get("id")}

    def remove(self, model_id):
        """Remove only model folders Sheep downloaded and owns; shared base weights stay shared."""
        if model_id not in _MODEL_INSTALLS:
            raise ValueError("that feature is provided by Windows and has no Sheep model download")
        with self._condition:
            if self._active:
                raise RuntimeError("a media model is being downloaded; wait for it to finish before removing files")
            self._active = True
        try:
            from . import video_ready, video_reference_ready, image_ready
            from ..tools import media
            workers = {
                "sd15-lcm": (media.PICTURES, media.VIDEOS, media.REFERENCE_VIDEOS),
                "animatediff-lightning": (media.PICTURES, media.VIDEOS, media.REFERENCE_VIDEOS),
                "ip-adapter-plus": (media.REFERENCE_VIDEOS,),
                "whisper-small": (),
            }[model_id]
            if any(worker.lock.locked() for worker in workers):
                raise RuntimeError("that model is in use right now; wait for the current media job to finish")
            for worker in workers:
                worker.stop()

            targets = []
            if model_id == "sd15-lcm":
                targets.append(MEDIA_DIR / "lcm")
                if not video_ready() and not video_reference_ready():
                    targets.append(MEDIA_DIR / "sd15")
            elif model_id == "animatediff-lightning":
                targets.append(MEDIA_DIR / "animatediff")
                if not image_ready() and not video_reference_ready():
                    targets.append(MEDIA_DIR / "sd15")
            elif model_id == "ip-adapter-plus":
                targets.append(MEDIA_DIR / "ip-adapter")
            else:
                targets.append(whisper_cache() / "models--Systran--faster-whisper-small")
                media._WHISPER.clear()

            root = MEDIA_DIR.resolve()
            for target in targets:
                resolved = target.resolve()
                if resolved == root or root not in resolved.parents:
                    raise RuntimeError("refusing to remove a path outside Sheep's media model folder")
                if target.exists():
                    shutil.rmtree(target)
            return True
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()

    @staticmethod
    def _tqdm_class(callback):
        from tqdm.auto import tqdm

        class DownloadProgress(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                callback(self.n, self.total, kwargs.get("desc", ""), kwargs.get("unit", ""))

            def update(self, n=1):
                result = super().update(n)
                callback(self.n, self.total, self.desc, self.unit)
                return result

        return DownloadProgress


_installer = MediaInstaller()


def set_publisher(publish):
    _installer.publish = publish


def status():
    return _installer.status()


def ensure(kind):
    return _installer.ensure(kind)


def remove(model_id):
    return _installer.remove(model_id)


def cancel():
    return _installer.cancel()
