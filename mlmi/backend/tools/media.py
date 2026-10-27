"""Media tools: pictures, speech to text and voice, for any chat model.

The chat model calls these like any other tool; the media model behind each
is the one chosen in the model drop-down (adapters.defaults). Each result is a
file in the workspace, so it shows in the chat and on the Work Canvas.

Where they run (Settings › Advanced › This computer › pictures, sound, video):
  auto      a checked NVIDIA/CUDA device first, then checked DirectML devices,
            then CPU; a device failure retries the next local option
  cuda / directml / cpu   the person's choice, when this machine has it
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import tool
from ..workspace import ModelSelectionRequired, ToolError

CONFIG = {}          # the app's config, set by the server (media_device, media_models)
LEND = None          # (lend_gpu(what, gb, cancel_event=...), give_back_gpu()) from the server
VIDEO_GB = 3.6       # an AI video at 512 px: 3.5 GB peak measured on the RTX 3050 (2026-10-06)
HERE = Path(__file__).resolve().parent.parent
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _device_candidates(video=False):
    """Local media backends in recommendation order; explicit choices stay pinned."""
    from .. import hardware
    from .. import gpu_runtime
    pref = CONFIG.get("media_device", "auto")
    cpu = (sys.executable, "cpu", "the CPU")
    # NVIDIA: the CUDA runtime (gpu_runtime.py) when it is installed and answered;
    # the app's own Python only if it happens to carry a CUDA PyTorch
    cuda_rt = gpu_runtime.ready()
    cuda = (str(gpu_runtime.PY) if cuda_rt else sys.executable, "cuda", "the NVIDIA GPU")
    if pref == "cpu":
        return [cpu]
    if pref == "cuda":
        return [cuda]

    try:
        ads = hardware.adapters()
    except Exception:
        ads = []
    enough = lambda a: not video or (a.get("vram") or 0) >= VIDEO_GB * 2**30 * 0.95
    cuda_ready = any(a.get("vendor") == "nvidia" and a.get("ok", True) and enough(a) for a in ads)
    dml = []
    if hardware.DML_PY.exists():
        try:
            dml = hardware.directml_adapters()
        except Exception:
            pass
    dml = [a for a in dml if enough(a) and (pref != "directml" or a.get("vendor") in ("amd", "intel"))]
    candidates = []
    if pref == "auto" and cuda_ready and (cuda_rt or hardware.torch_cuda()):
        candidates.append(cuda)
    candidates.extend((str(hardware.DML_PY), f"dml:{a['index']}",
                       f"the {a['name'].replace('(TM)', '')} (DirectML)") for a in dml)
    if pref == "directml":
        # Keep an unavailable manual choice visible as a clear failure; do not
        # silently run it on another device.
        return candidates or [(str(hardware.DML_PY), "dml:0", "the selected AMD / Intel GPU (DirectML)")]
    if pref == "auto":
        candidates.append(cpu)
    return candidates


def _device():
    """Preferred (python, --device value, words) tuple for picture workers."""
    return _device_candidates()[0]


def _video_device():
    """(python, --device value, words) for AI video. It needs about 3.6 GB of one
    GPU: an NVIDIA card of 4 GB or more (lent by the chat model for the job), an
    AMD/Intel card of 4 GB or more, else the CPU (slow). The 2 GB Radeon here ran
    out of memory at 384 px, and below 512 px the 4-step model makes noise."""
    return _device_candidates(video=True)[0]


_DEVICE_FAILURES = ("out of memory", "cuda", "directml", "torch_directml", "hip error", "cudnn", "cublas",
                    "device lost", "device-side", "device error", "allocation", "driver", "no such file",
                    "cannot find", "not found", "not enough gpu")


def _device_error(error):
    text = str(error or "").lower()
    return any(marker in text for marker in _DEVICE_FAILURES)


def _add_media_fallback(text):
    old = CONFIG.get("_media_fallback", "")
    CONFIG["_media_fallback"] = f"{old}; {text}" if old else text


class _Worker:
    """A media worker (pictures, or video): started on first use, kept while used,
    stopped after ten idle minutes so its memory comes back. Only one runs at a
    time: starting one stops the other, so a small GPU holds one model."""

    ALL = []

    def __init__(self, what, args, device=_device, devices=None):
        self.what, self.args, self.device = what, args, device     # args(): the worker's model arguments
        self.devices = devices or (lambda: [self.device()])
        self.proc, self.dev, self.words, self.lock, self.used = None, "", "", threading.Lock(), 0.0
        self.preference, self.candidates, self.candidate_index = None, [], -1
        _Worker.ALL.append(self)
        threading.Thread(target=self._idle, daemon=True).start()

    def _start(self, candidates=None, start_at=0, cancel_event=None):
        from ..adapters import MEDIA_DIR
        candidates = list(candidates if candidates is not None else self.devices())
        if not candidates or start_at >= len(candidates):
            raise ToolError(f"no usable local device remains for the {self.what} model")
        self.candidates = candidates
        pref = CONFIG.get("media_device", "auto")
        failures = []
        for w in _Worker.ALL:
            if w is not self and not w.lock.locked():
                w.stop()
        for index in range(start_at, len(candidates)):
            if cancel_event is not None and cancel_event.is_set():
                raise ToolError("media generation was cancelled by the user")
            py, dev, words = candidates[index]
            if py != sys.executable and not Path(py).exists():
                error = f"{py} is missing"
                failures.append(f"{words}: {error}")
                if pref != "auto":
                    break
                continue
            cmd = [py, str(HERE / "adapters" / "image_worker.py"), "--device", dev,
                   "--base", str(__import__("mlmi.backend.media_styles", fromlist=["chosen_base"]).chosen_base(CONFIG))] + self.args(MEDIA_DIR)
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                             text=True, encoding="utf-8", errors="replace", creationflags=NOWIN)
            except OSError as exc:
                self.proc = None
                error = str(exc)
            else:
                self.dev, self.words, self.preference = dev, words, pref
                hello = self._line(600)
                if cancel_event is not None and cancel_event.is_set():
                    self.stop()
                    raise ToolError("media generation was cancelled by the user")
                if hello.get("ok"):
                    self.candidate_index = index
                    if failures:
                        _add_media_fallback("; ".join(failures) + f"; used {words}")
                    return
                error = str(hello.get("error") or f"the {self.what} worker did not answer")
                self.stop()
            failures.append(f"{words}: {error[:140]}")
            if pref not in ("auto", "directml") or (dev != "cpu" and not _device_error(error)):
                break
        if failures:
            raise ToolError(f"the {self.what} model could not load: " + "; ".join(failures) + gpu_memory_tip("; ".join(failures)))
        raise ToolError(f"no usable local device remains for the {self.what} model")

    def _line(self, timeout, on_progress=None):
        box = {}
        proc = self.proc

        def read():
            try:
                for line in proc.stdout:
                    line = line.strip()
                    if line.startswith("{"):
                        message = json.loads(line)
                        if message.get("type") == "progress":
                            if on_progress:
                                try:
                                    on_progress(message)
                                except Exception:
                                    pass  # progress reporting must not fail media generation
                            continue
                        box["v"] = message
                        return
            except (OSError, ValueError):
                return
        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout)
        return box.get("v") or {"ok": False, "error": f"the {self.what} worker did not answer in time"}

    @staticmethod
    def _clear_output(req):
        import shutil
        out = Path(str(req.get("out") or ""))
        if out.is_dir():
            for child in out.iterdir():
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
        else:
            out.unlink(missing_ok=True)

    def _send(self, req, timeout, on_progress=None):
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return self._line(timeout, on_progress=on_progress)

    def _retry_next(self, req, timeout, error, cancel_event=None, progress=None):
        # Retry is Auto policy. A user's explicit CUDA, DirectML or CPU choice
        # stays pinned; never reinterpret a failed manual selection for them.
        if CONFIG.get("media_device", "auto") != "auto" or not _device_error(error):
            return None
        start_at = self.candidate_index + 1
        if start_at >= len(self.candidates):
            return None
        failed = self.words
        self.stop()
        self._clear_output(req)
        if cancel_event is not None and cancel_event.is_set():
            raise ToolError("media generation was cancelled by the user")
        self._start(self.candidates, start_at, cancel_event=cancel_event)
        _add_media_fallback(f"{failed} failed during {self.what} inference ({str(error)[:120]}); used {self.words}")
        return self._send(req, timeout, on_progress=progress)

    def make(self, req, timeout=600, cancel_event=None, register_cancel=None, progress=None):
        with self.lock:
            if register_cancel:
                register_cancel(self.stop)
            try:
                def check_cancelled():
                    if cancel_event is not None and cancel_event.is_set():
                        raise ToolError("media generation was cancelled by the user")
                check_cancelled()
                pref = CONFIG.get("media_device", "auto")
                if not self.proc or self.proc.poll() is not None or self.preference != pref:
                    if self.proc:
                        self.stop()
                    self._start(cancel_event=cancel_event)
                check_cancelled()
                try:
                    r = self._send(req, timeout, on_progress=progress)
                except (OSError, ValueError) as exc:
                    check_cancelled()
                    raise ToolError(f"the {self.what} worker stopped unexpectedly: {exc}") from exc
                check_cancelled()  # a killed worker must never look like a device failure and trigger fallback
                while not r.get("ok") and self.dev != "cpu":
                    retry = self._retry_next(req, timeout, r.get("error"), cancel_event=cancel_event,
                                             progress=progress)
                    if retry is None:
                        break
                    check_cancelled()
                    r = retry
                self.used = time.time()
                return r
            finally:
                if register_cancel:
                    register_cancel(None)

    def stop(self):
        proc, self.proc = self.proc, None
        if not proc:
            return
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait()
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream and not stream.closed:
                    stream.close()

    def _idle(self):
        while True:
            time.sleep(60)
            if self.proc and self.used and time.time() - self.used > 600 and not self.lock.locked():
                self.stop()


PICTURES = _Worker("picture", lambda d: ["--lora", str(d / "lcm")])
MOTION_FILE = "animatediff_lightning_4step_diffusers.safetensors"
VIDEOS = _Worker("video", lambda d: ["--motion", str(d / "animatediff" / MOTION_FILE)], _video_device)
REFERENCE_VIDEOS = _Worker(
    "photo-guided video",
    lambda d: ["--motion", str(d / "animatediff" / MOTION_FILE), "--ip-adapter", str(d / "ip-adapter")],
    _video_device,
)

VIDEO_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
MAX_REFERENCE_PIXELS = 50_000_000


def _video_sources(ctx, source_images):
    """Validate selected workspace photos before downloading models or starting a worker."""
    if not source_images:
        return []
    if isinstance(source_images, str):
        source_images = [source_images]
    if not isinstance(source_images, (list, tuple)):
        raise ToolError("source_images must be a list of selected picture paths")
    paths = []
    for rel in source_images:
        rel = str(rel or "").strip()
        if not rel:
            continue
        if Path(rel).suffix.lower() not in VIDEO_IMAGE_EXTS:
            raise ToolError(f"{rel} is not a supported picture; use PNG, JPEG, WebP, GIF or BMP")
        src = ctx.ws.path(rel, must_exist=True)
        try:
            from PIL import Image
            with Image.open(src) as image:
                if image.width * image.height > MAX_REFERENCE_PIXELS:
                    raise ToolError(f"{rel} is too large to use for video guidance; resize it below 50 megapixels")
                image.verify()
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"cannot read the selected picture {rel}: {str(exc)[:160]}") from exc
        normalized = ctx.ws.rel(src)
        if normalized not in paths:
            paths.append(normalized)
    if len(paths) > 9:
        raise ToolError("photo-guided video accepts up to 9 selected pictures at once; make a Studio montage for more")
    return [str(ctx.ws.path(rel, must_exist=True)) for rel in paths]


def _video_memory_error(error):
    text = str(error or "").lower()
    return any(marker in text for marker in (
        "out of memory", "cuda error: memory", "not enough memory", "not enough gpu video memory",
        "could not allocate tensor", "failed to allocate memory", "allocation failed",
    ))


def _lend_gpu(ctx, what, need_gb):
    """Reserve the GPU, forwarding Stop while queued for an interactive lease."""
    if not LEND:
        return False
    callback = LEND[0]
    cancel_event = getattr(ctx, "cancel_event", None)
    try:
        return bool(callback(what, need_gb, cancel_event=cancel_event))
    except TypeError as exc:
        # Preserve older/test callbacks that still accept the original pair.
        if "cancel_event" not in str(exc):
            raise
        return bool(callback(what, need_gb))


def _name(ctx, path, prompt, ext):
    if path:
        p = path if path.lower().endswith(ext) else path + ext
    else:
        stem = "-".join(re.findall(r"[a-z0-9]+", prompt.lower())[:5]) or "picture"
        p = f"media/{stem}{ext}"
    k, base = 2, p[: -len(ext)]
    while ctx.ws.path(p).exists():
        p = f"{base}-{k}{ext}"
        k += 1
    return p


def gpu_memory_tip(error):
    """What to do when a picture or video ran out of GPU memory. On an NVIDIA card
    without a CUDA build of PyTorch, media runs through DirectML, which needs far
    more memory: a 4 GB RTX 3050 could make neither (2026-10-13)."""
    if "not enough GPU video memory" not in str(error or ""):
        return ""
    from .. import hardware
    try:
        nvidia = any(a.get("vendor") == "nvidia" for a in hardware.adapters())
    except Exception:
        nvidia = False
    if nvidia and not hardware.media_cuda():
        return (". This NVIDIA card is being used through DirectML, which needs much more memory. "
                "Turning on NVIDIA (CUDA) support for pictures and video lets them fit; "
                "until then, Settings › Advanced › This computer can run them on the CPU (slower).")
    return ". Close other programs that use the GPU, or run pictures and video on the CPU in Settings › Advanced › This computer."


def _require_selected_model(slot, ready, message):
    """Never start a model download from a tool call; the person must confirm it."""
    if ready:
        return
    from ..model_slots import MEDIA_REPOS
    raise ModelSelectionRequired(slot, MEDIA_REPOS[slot][0], message)


@tool("make_image", "media", 2, "Make a picture from a description (Stable Diffusion). Saved as a .png in the workspace.",
      {"prompt": ("string", "what the picture shows, in detail"), "path": ("string", "file name, optional"),
       "width": ("integer", "pixels, 256-768, default 512"), "height": ("integer", "pixels, 256-768, default 512")},
      ("prompt",))
def make_image(ctx, prompt, path="", width=512, height=512):
    from ..adapters import image_ready
    if "sd15-lcm" in set(CONFIG.get("disabled_media_models") or []):
        raise ToolError("Stable Diffusion is disabled. Choose it in the model menu to enable it again.")
    _require_selected_model("image", image_ready(),
                            "The picture model is missing. Check and use a compatible image model to continue.")
    if not image_ready():
        raise ToolError("the picture model is not on this machine yet (Models › Stable Diffusion 1.5 + LCM)")
    rel = _name(ctx, path, prompt, ".png")
    p = ctx.ws.path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    device = _device_candidates()[0][1]
    lent = False
    if LEND and device != "cpu":
        try:
            lent = _lend_gpu(ctx, "an AI image", 2.5)
        except Exception as exc:
            raise ToolError(f"could not reserve the selected GPU for image generation: {str(exc)[:160]}") from exc
        if not lent:
            raise ToolError("the selected GPU is busy or unavailable; Sheep did not start a competing image job")
    try:
        with ctx.ws.changing(p, by=ctx.by, what=f"picture {rel}"):
            r = PICTURES.make({"prompt": prompt, "out": str(p), "width": int(width or 512), "height": int(height or 512)},
                              cancel_event=getattr(ctx, "cancel_event", None),
                              register_cancel=getattr(ctx, "register_cancel", None))
    finally:
        if lent:
            PICTURES.stop()
            LEND[1]()
    if not r.get("ok"):
        raise ToolError(f"the picture was not made: {r.get('error')}" + gpu_memory_tip(r.get("error")))
    ctx.made.add(rel)                       # so the chat and the Work Canvas show it
    note = CONFIG.pop("_media_fallback", "")
    return f"made {rel} ({r.get('width')}x{r.get('height')}) on {PICTURES.words} in {r.get('seconds')} s" + (f". {note}" if note else "")


VIDEO_FPS = 8          # the motion module's own pace; FreeNoise windows carry it past 16 frames
_ENC = {}


def _speeds():
    from ..adapters import MEDIA_DIR
    try:
        return json.loads((MEDIA_DIR / "video_speed.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _speed_saved(dev, per_frame):
    """Seconds per frame on this device, from the last clip made there (for the tip card)."""
    from ..adapters import MEDIA_DIR
    s = _speeds()
    s[dev] = round(per_frame, 2)
    try:
        (MEDIA_DIR / "video_speed.json").write_text(json.dumps(s), encoding="utf-8")
    except OSError:
        pass


def video_estimate_seconds(args):
    """Return a recent-device estimate for this clip, when we have measurements."""
    secs = max(1, min(6, int((args or {}).get("seconds") or 2)))
    _, dev, words = _video_device()
    worker = REFERENCE_VIDEOS if (args or {}).get("source_images") else VIDEOS
    per = _speeds().get(worker.dev or dev)
    if not per:
        return None
    return max(1, round(per * secs * VIDEO_FPS))


def wait_words(args):
    """Describe the video job and give a comparison to recent measured clips."""
    secs = max(1, min(6, int((args or {}).get("seconds") or 2)))
    _, dev, words = _video_device()
    worker = REFERENCE_VIDEOS if (args or {}).get("source_images") else VIDEOS
    mode = "photo-guided " if worker is REFERENCE_VIDEOS else ""
    line = f"Making the {secs}-second {mode}video on {worker.words or words}"
    estimate = video_estimate_seconds(args)
    if estimate:
        estimate_words = f"about {estimate / 60:.0f} min" if estimate >= 90 else f"about {estimate} s"
        line += f" (similar clips took {estimate_words}; this run can vary)"
    if not worker.proc:
        line += " (loading the video model first)"
    return line


def video_tip(chat_device=""):
    """What decides video speed and quality on this machine, what Sheep set, and
    what the person can change (spec 10: the video tip card)."""
    from ..adapters import video_ready
    from .studio import _encoder
    if "enc" not in _ENC:
        _ENC["enc"] = _encoder()[1]
    _, dev, words = _video_device()
    pref = CONFIG.get("media_device", "auto")
    if pref not in ("auto", "cuda") or (pref == "cuda" and dev != "cuda"):
        why = "your choice in Settings › Advanced › This computer"
    elif dev == "cuda":
        why = (f"a clip needs about {VIDEO_GB:.1f} GB of one GPU; " + (
            "the chat model lends the card for the clip and answers from the CPU meanwhile, then gets it back"
            if chat_device in ("gpu", "hybrid", "multi") else "the card is free"))
    elif dev.startswith("dml"):
        why = f"it has the {VIDEO_GB:.1f} GB a clip needs"
    else:
        why = f"no GPU here has the {VIDEO_GB:.1f} GB a clip needs, so the CPU makes it, slowly"
    per = _speeds().get(dev)
    ready = video_ready()
    return {
        "ai": {"model": "AnimateDiff-Lightning on Stable Diffusion 1.5", "ready": ready, "device": words, "why": why,
               "size": "512 × 512", "fps": VIDEO_FPS, "seconds": "1-6", "steps": 4,
               "per_second": round(per * VIDEO_FPS) if per else None},
        "studio": {"encoder": _ENC["enc"], "size": "the project's size (1080p, or 1080 × 1920 for shorts)", "fps": 30},
        "decides": [
            ["GPU", "AI video runs on a GPU many times faster than on the CPU; the GPU's heat and clock set its pace"],
            ["GPU memory", f"a clip needs about {VIDEO_GB:.1f} GB of one card; a 2 GB card is too small"],
            ["Resolution", "this model is made for 512 px (smaller turns to noise); 640 px needs more memory and time"],
            ["Length", "every second is 8 more frames, so time grows with length"],
            ["Model and level", "a larger video model looks better and needs more memory and time"],
            ["Encoder", "NVENC encodes on the NVIDIA video chip, apart from the parts that run AI"],
        ],
        "change": [
            "Where it runs: Settings › Advanced › This computer › pictures, sound and video",
            "Length: say it in the request (\"a 4-second clip\"); 1-6 s, and each second adds 8 frames",
            "Edited videos with titles, logos and music: Sheep Studio, which needs no AI model",
            "A quick look before a full render: ask Studio for a draft render (half size)",
        ],
    }


@tool("make_video", "media", 2, "Make a short AI video clip from a description (AnimateDiff on Stable Diffusion), "
      "1-6 seconds, saved as an .mp4. source_images are selected local photos for visual guidance; the tool never "
      "falls back to text-only when they fail. For exact-photo montages, titles, music and cuts, use Studio tools.",
      {"prompt": ("string", "what happens in the clip, in detail"), "seconds": ("integer", "length, 1-6, default 2"),
       "path": ("string", "file name, optional"), "width": ("integer", "pixels, 512-640, default 512"),
       "height": ("integer", "pixels, 512-640, default 512"),
       "source_images": ("array", "selected source photos in the Sheep workspace; omit for text-only video",
                         {"type": "string"})}, ("prompt",))
def make_video(ctx, prompt, seconds=2, path="", width=512, height=512, source_images=None):
    import shutil
    import tempfile
    from ..adapters import video_ready, video_reference_ready
    from .studio import _encoder
    if not shutil.which("ffmpeg"):
        raise ToolError("making a video needs ffmpeg on this machine")
    image_paths = _video_sources(ctx, source_images)
    disabled = set(CONFIG.get("disabled_media_models") or [])
    video_id = "animatediff-lightning"
    if video_id in disabled:
        raise ToolError("AnimateDiff is disabled. Choose it in the model menu to enable it again.")
    if image_paths and "ip-adapter-plus" in disabled:
        raise ToolError("photo guidance is disabled. Choose IP-Adapter Plus in the model menu to use selected photos.")
    _require_selected_model("video", video_ready(),
                            "The video model is missing. Check and use a compatible video model to continue.")
    if not video_ready():
        raise ToolError("the video model is not on this machine yet (Models › AnimateDiff-Lightning)")
    worker = VIDEOS
    if image_paths:
        _require_selected_model("photo_guidance", video_reference_ready(),
                                "Photo guidance is missing; the selected photos were not used. Check and use a compatible photo-guidance model.")
        if not video_reference_ready():
            raise ToolError("the photo-guidance model is not ready; the selected photos were not used")
        worker = REFERENCE_VIDEOS
    secs = max(1, min(6, int(seconds or 2)))
    rel = _name(ctx, path, prompt, ".mp4")
    p = ctx.ws.path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="sheep-frames-")
    # the NVIDIA card is lent by the chat model for the clip, and returned after
    lent = False
    r = {}
    try:
        if LEND and _video_device()[1] != "cpu":
            try:
                lent = _lend_gpu(ctx, "an AI video", VIDEO_GB)
            except Exception as exc:
                raise ToolError(f"could not reserve the selected GPU for video generation: {str(exc)[:160]}") from exc
            if not lent:
                raise ToolError("the selected GPU is busy or unavailable; Sheep did not start a competing video job")
        req = {"prompt": prompt, "out": tmp, "frames": secs * VIDEO_FPS, "width": int(width or 512),
               "height": int(height or 512)}
        if image_paths:
            req["source_images"] = image_paths
        progress = getattr(ctx, "progress", None)
        if progress:
            report = progress
            progress = lambda event: report({**event, "device": worker.words or worker.dev})
        r = worker.make(req, timeout=3600, cancel_event=getattr(ctx, "cancel_event", None),
                        register_cancel=getattr(ctx, "register_cancel", None), progress=progress)
        if not r.get("ok"):
            error = str(r.get("error") or "the worker returned no details")
            if image_paths and _video_memory_error(error):
                if worker.dev == "cpu":
                    raise ToolError("the photo-guided video did not fit on any available local device, including the CPU. "
                                    "The selected photos were kept; try fewer photos or a shorter clip.")
                raise ToolError("the photo-guided video did not fit in GPU memory. The photos were not ignored; "
                                "try one selected photo, a shorter clip, or choose CPU in Settings › Advanced › This computer "
                                "and retry.")
            if _video_memory_error(error):
                if worker.dev == "cpu":
                    raise ToolError("the video did not fit on any available local device, including the CPU. "
                                    "Try a shorter clip or lower dimensions.")
                raise ToolError("the video did not fit in GPU memory; choose CPU or a GPU with more video memory "
                                "in Settings › Advanced › This computer and retry")
            if image_paths and any(s in error.lower() for s in ("ip-adapter", "image encoder", "ip_adapter_image")):
                raise ToolError(f"photo guidance failed, so no text-only video was substituted: {error[:220]}")
            raise ToolError(f"the video was not made: {error}" + gpu_memory_tip(error))
        enc, enc_words = _encoder()
        with ctx.ws.changing(p, by=ctx.by, what=f"video {rel}"):
            e = subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(VIDEO_FPS), "-i",
                                os.path.join(tmp, "frame_%03d.png")] + enc + ["-pix_fmt", "yuv420p", str(p)],
                               capture_output=True, text=True, timeout=300, creationflags=NOWIN)
        if e.returncode or not p.exists():
            raise ToolError(f"the frames were made but not encoded: {(e.stderr or '').strip()[-200:]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if lent:
            worker.stop()                    # its CUDA memory goes back with the card
            LEND[1]()
    ctx.made.add(rel)
    _speed_saved(worker.dev, r.get("seconds", 0) / max(1, r.get("frames", 1)))
    note = CONFIG.pop("_media_fallback", "")
    photo_note = f" Photo guidance used {len(image_paths)} selected photo(s)." if image_paths else ""
    memory_note = f" Peak allocated GPU memory: {r['peak_gpu_mb']} MB." if r.get("peak_gpu_mb") is not None else ""
    return (f"made {rel}. {r.get('frames') / VIDEO_FPS:.0f} s at {r.get('width')}x{r.get('height')}, {VIDEO_FPS} frames a second, "
            f"made on {worker.words} in {r.get('seconds'):.0f} s and encoded with {enc_words}." + photo_note
            + memory_note + (f" {note}" if note else ""))


_WHISPER = {}


def _whisper():
    if "m" not in _WHISPER:
        from faster_whisper import WhisperModel
        try:
            from ..adapters import whisper_snapshot
            snapshot = whisper_snapshot(CONFIG)
            if not snapshot:
                raise RuntimeError("Whisper small weights are not installed for Sheep")
            _WHISPER["m"] = WhisperModel(str(snapshot), device="cpu", compute_type="int8")
        except Exception as e:
            raise ToolError(f"speech to text did not load: {str(e)[:160]}")
    return _WHISPER["m"]


def audio_16k(path):
    """The sound of a file as 16 kHz mono float32 samples, which Whisper takes as-is.
    faster-whisper's own decoder calls av.open(..., metadata_errors=...), an argument
    PyAV 14+ removed, so every transcript failed with a TypeError on Python 3.14
    (no older PyAV builds for it; 2026-10-13). Decoding here avoids that call."""
    import av
    import numpy as np
    chunks = []
    with av.open(str(path), mode="r") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ToolError(f"{Path(path).name} has no sound to write down")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):            # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ToolError(f"{Path(path).name} has no sound to write down")
    return (np.concatenate(chunks).astype(np.float32) / 32768.0)


def timed_words(ctx, path):
    """Whisper's words with their start and end seconds in the file: [{"word", "start", "end"}]."""
    src = ctx.ws.path(path, must_exist=True)
    segs, _ = _whisper().transcribe(for_whisper(src), vad_filter=True, word_timestamps=True)
    out = []
    for s in segs:
        for w in (s.words or []):
            out.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
    return out


def hear(ctx, path):
    """What a model can know of a sound without ears: speech lines with times (Whisper, when
    installed), silences, the loudest moments and the beat, from the samples themselves.
    Seconds are the file's own."""
    import numpy as np
    src = ctx.ws.path(path, must_exist=True)
    x = audio_16k(src)
    sr, hop = 16000, 800                                  # 50 ms frames
    n = len(x) // hop
    out = {"seconds": round(len(x) / sr, 2)}
    if n < 4:
        return out
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    t = np.arange(n) * hop / sr
    # silences: 0.5 s or more 30 dB under the loud parts
    quiet = db < (np.percentile(db, 95) - 30)
    sil, k = [], 0
    while k < n:
        if quiet[k]:
            j = k
            while j < n and quiet[j]:
                j += 1
            if (j - k) * hop / sr >= 0.5:
                sil.append((round(t[k], 2), round(t[j - 1] + hop / sr, 2)))
            k = j
        else:
            k += 1
    out["silences"] = sil
    # loudest moments, at least a second apart
    order, loud = np.argsort(-rms), []
    for idx in order:
        if all(abs(t[idx] - u) >= 1.0 for u in loud):
            loud.append(round(float(t[idx]), 2))
        if len(loud) >= 8:
            break
    out["loud"] = sorted(loud)
    # beats: energy onsets (rises above the local level), at least 0.25 s apart
    flux = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    thr = flux.mean() + 1.5 * flux.std()
    beats, last = [], -1.0
    for idx in range(1, n - 1):
        if flux[idx] > thr and flux[idx] >= flux[idx - 1] and flux[idx] >= flux[idx + 1] and t[idx] - last >= 0.25:
            beats.append(round(float(t[idx]), 2)); last = t[idx]
    out["beats"] = beats
    if len(beats) >= 4:
        gaps = np.diff(beats)
        out["bpm"] = round(60.0 / float(np.median(gaps)), 1) if np.median(gaps) > 0 else 0
    try:
        segs, _ = _whisper().transcribe(x, vad_filter=True)
        out["speech"] = [(round(float(s.start), 2), round(float(s.end), 2), s.text.strip()) for s in segs if s.text.strip()]
    except Exception as e:                               # no Whisper here: the rest still holds
        out["speech"] = []
        out["speech_note"] = f"speech: not heard ({str(e)[:80] or 'no speech-to-text model'})"
    return out


# Watching a video (2026-10-27): it was only transcribed, its pictures never looked at. Key frames
# are taken where the picture changes (ffmpeg's scene score), evenly when it hardly changes, each
# described by the vision model when one is here; the sound is heard as studio_listen hears it.
LOOK_FRAMES, LOOK_SCENE = 8, 0.3
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")


def key_frame_times(src, duration, cap=LOOK_FRAMES, scene=LOOK_SCENE):
    """Seconds where the picture changes most, at most cap of them; evenly spread when the
    video hardly changes (fewer than 3 cuts)."""
    exe = shutil.which("ffmpeg")
    times = []
    if exe:
        try:
            r = subprocess.run([exe, "-hide_banner", "-i", str(src), "-vf", f"select='gt(scene,{scene})',showinfo",
                                "-an", "-f", "null", "-"], capture_output=True, text=True, timeout=300,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", r.stderr or "")]
        except (OSError, subprocess.TimeoutExpired):
            times = []
    return spread_times(times, duration, cap)


def spread_times(cuts, duration, cap=LOOK_FRAMES):
    """The opening frame, then the cuts (thinned evenly to the cap); even spacing when there are few."""
    d = float(duration or 0)
    if len(cuts) < 3:
        n = max(1, min(cap, int(d // 2) + 1)) if d else 1
        return [round(d * (i + 0.5) / n, 2) for i in range(n)] if d else [0.0]
    # a flickering clip (AI video at 8 fps) "cuts" every few frames: keep cuts apart
    gap = max(0.75, d / (cap * 2)) if d else 0.75
    pts = [0.5]
    for t in sorted(t for t in cuts if 0.5 < t < d):
        if t - pts[-1] >= gap:
            pts.append(t)
    if len(pts) < 3:
        n = max(1, min(cap, int(d // 2) + 1)) if d else 1
        return [round(d * (i + 0.5) / n, 2) for i in range(n)] if d else [0.0]
    if len(pts) > cap:
        step = (len(pts) - 1) / (cap - 1)
        pts = [pts[round(i * step)] for i in range(cap)]
    return [round(t, 2) for t in pts]


def _clock(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


@tool("look_video", "media", 1,
      "Watch a video in the workspace: what is seen at its key moments (when a picture-reading model is "
      "here) and what is heard (speech with times, silences, loud moments, beat). Use it before answering "
      "or editing anything about what happens in a video. Saves <name>.summary.txt.",
      {"path": ("string", "the video file"), "question": ("string", "what to look for, optional")}, ("path",))
def look_video(ctx, path, question=""):
    src = ctx.ws.path(path, must_exist=True)
    if src.suffix.lower() not in VIDEO_EXT:
        raise ToolError(f"{path} is not a video")
    if not shutil.which("ffmpeg"):
        raise ToolError("ffmpeg is needed to look at videos: Settings › Advanced › This computer shows how to get it")
    from .studio import _probe
    from . import vision
    duration, has_audio = _probe(src)
    stem = Path(path).with_suffix("")
    frame_dir = f"{stem.as_posix()}-frames"
    lines, seen_note = [], ""
    for t in key_frame_times(src, duration):
        rel = f"{frame_dir}/{t:07.2f}.jpg".replace(" ", "_")
        out = ctx.ws.path(rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(src),
                        "-frames:v", "1", "-vf", "scale='min(768,iw)':-2", str(out)],
                       capture_output=True, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if not out.exists():
            continue
        what = ""
        if not seen_note:
            try:
                what = str(vision.see_picture(ctx, rel, question or "Describe what is happening in this frame in one sentence."))
            except ToolError as e:
                seen_note = str(e)
        lines.append((t, "seen", what.strip() or f"frame saved as {rel}"))
    heard = {}
    if has_audio:
        try:
            heard = hear(ctx, path)
        except ToolError as e:
            heard = {"speech_note": str(e)}
    for a, b, words in heard.get("speech", []):
        lines.append((a, "said", f"\"{words}\" (to {_clock(b)})"))
    lines.sort(key=lambda r: r[0])
    head = [f"{Path(path).name}: {_clock(duration or 0)} long, {'with' if has_audio else 'no'} sound"]
    if seen_note:
        head.append(f"Pictures: not described ({seen_note}). Add a picture-reading model to see them; key frames are saved in {frame_dir}/.")
    if heard.get("speech_note"):
        head.append(heard["speech_note"])
    if heard.get("silences"):
        head.append("Silences: " + ", ".join(f"{_clock(a)}-{_clock(b)}" for a, b in heard["silences"][:12]))
    if heard.get("loud"):
        head.append("Loudest moments: " + ", ".join(_clock(t) for t in heard["loud"]))
    if heard.get("bpm"):
        head.append(f"Beat: about {heard['bpm']} per minute")
    body = [f"{_clock(t)}  {kind:5}  {text}" for t, kind, text in lines]
    summary = "\n".join(head + [""] + body)
    rel_sum = f"{stem.as_posix()}.summary.txt"
    sp = ctx.ws.path(rel_sum)
    with ctx.ws.changing(sp, by=ctx.by, what=f"video summary {rel_sum}"):
        sp.write_text(summary + "\n", encoding="utf-8")
    ctx.made.add(rel_sum)
    return f"watched {path}; summary saved as {rel_sum}:\n{summary[:4000]}"


def for_whisper(path):
    """Samples Sheep decoded, or the path when that fails (Whisper then tries itself)."""
    try:
        return audio_16k(path)
    except ToolError:
        raise
    except Exception:
        return str(path)


@tool("transcribe_audio", "media", 1, "Write down what is said in a sound or video file (speech to text). Saves a .txt next to it, or at an optional output path.",
      {"path": ("string", "the sound or video file"), "out": ("string", "optional transcript .txt path")}, ("path",))
def transcribe_audio(ctx, path, out=""):
    src = ctx.ws.path(path, must_exist=True)
    if "whisper-small" in set(CONFIG.get("disabled_media_models") or []):
        raise ToolError("Whisper small is disabled. Choose it in the model menu to enable it again.")
    from ..adapters import whisper_snapshot
    _require_selected_model("speech_input", whisper_snapshot(CONFIG) is not None,
                            "Whisper small is missing. Check and use a compatible speech-input model to continue.")
    t0 = time.time()
    segs, info = _whisper().transcribe(for_whisper(src), vad_filter=True)
    text = " ".join(s.text.strip() for s in segs).strip()
    requested = str(out or Path(path).with_suffix(".txt")).replace("\\", "/")
    if Path(requested).suffix.lower() != ".txt":
        raise ToolError("the transcript output must be a .txt file")
    rel = _name(ctx, requested, "transcript", ".txt")
    p = ctx.ws.path(rel)
    with ctx.ws.changing(p, by=ctx.by, what=f"transcript {rel}"):
        p.write_text(text + "\n", encoding="utf-8")
    ctx.made.add(rel)
    return f"transcript ({info.language}, {time.time() - t0:.0f} s) saved as {rel}:\n{text[:3000]}"


PICTURE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def _image_ops(text):
    """[(op, args)] from "grayscale, rotate 90, crop 10,10,80,80, resize 800" -- the words
    a person or a small model uses ("black and white", "greyscale", "mirror") included."""
    t = (text or "").lower().replace(";", ",").replace(" then ", ",").replace(" and then ", ",")
    t = re.sub(r"\bblack[- ]and[- ]white\b|\bb ?& ?w\b|\bgr[ae]y ?scale\b|\bmonochrome\b|\bno colou?r\b", "grayscale", t)
    out = []
    for part in re.split(r",(?![\d.\s]*\d\s*(?:,|$|%))|\n", t):
        part = part.strip()
        if not part:
            continue
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", part)]
        if "gray" in part:
            out.append(("grayscale", []))
        elif "sepia" in part or "vintage" in part:
            out.append(("sepia", []))
        elif "rotate" in part or "turn" in part:
            out.append(("rotate", [nums[0] if nums else (270 if "left" in part else 90)]))
        elif "mirror" in part or ("flip" in part and "vert" not in part and "upside" not in part):
            out.append(("mirror", []))
        elif "flip" in part or "upside" in part:
            out.append(("flip", []))
        elif "crop" in part:
            out.append(("crop", nums[:4]))
        elif "resize" in part or "scale" in part or "smaller" in part or "bigger" in part or "larger" in part:
            out.append(("resize", nums[:2] or [0.5 if "smaller" in part else 2.0]))
        elif "bright" in part or "lighter" in part:
            out.append(("brightness", [nums[0] if nums else 1.3]))
        elif "dark" in part:
            out.append(("brightness", [nums[0] if nums else 0.75]))
        elif "contrast" in part:
            out.append(("contrast", [nums[0] if nums else 1.3]))
        elif "blur" in part or "soft" in part:
            out.append(("blur", [nums[0] if nums else 2.0]))
        elif "sharp" in part:
            out.append(("sharpen", []))
        elif "invert" in part or "negative" in part:
            out.append(("invert", []))
        else:
            raise ToolError(f"'{part}' is not an edit this tool makes: grayscale, sepia, rotate, mirror, flip, crop, "
                            "resize, brightness, contrast, blur, sharpen, invert")
    if not out:
        raise ToolError("say what to change: grayscale, sepia, rotate 90, mirror, flip, crop x,y,w,h, resize W, "
                        "brightness 1.2, contrast 1.2, blur 2, sharpen, invert")
    return out


@tool("edit_image", "media", 2, "Change a picture that is already here: black and white (grayscale), sepia, rotate, "
      "mirror, flip, crop, resize, brightness, contrast, blur, sharpen or invert. The original is kept; the result is "
      "saved as a new file.",
      {"path": ("string", "the picture (.png, .jpg, .webp, .bmp)"),
       "ops": ("string", "the edits in order, comma separated, e.g. 'grayscale' or 'rotate 90, crop 10,10,80,80' "
                         "(crop in percent: left, top, width, height), 'resize 800', 'brightness 1.3'"),
       "out": ("string", "file name for the result, optional (default: the name with -edited)")},
      ("path", "ops"))
def edit_image(ctx, path, ops, out=""):
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    src = ctx.ws.path(path)
    if not src.is_file():
        raise ToolError(f"'{path}' does not exist (use list_files to see what is there)")
    if src.suffix.lower() not in PICTURE_EXT:
        raise ToolError(f"{path} is not a picture Sheep edits ({', '.join(PICTURE_EXT)})")
    steps = _image_ops(ops)
    im = Image.open(src)
    im.load()
    alpha = im.getchannel("A") if im.mode in ("RGBA", "LA") else None
    im = im.convert("RGB")
    done = []
    for op, a in steps:
        if op == "grayscale":
            im = ImageOps.grayscale(im).convert("RGB")
        elif op == "sepia":
            im = ImageOps.colorize(ImageOps.grayscale(im), "#2b1d0e", "#f3e3c3")
        elif op == "rotate":
            im = im.rotate(-a[0], expand=True)        # positive turns clockwise, as people say it
            alpha = alpha.rotate(-a[0], expand=True) if alpha else None
        elif op == "mirror":
            im, alpha = ImageOps.mirror(im), (ImageOps.mirror(alpha) if alpha else None)
        elif op == "flip":
            im, alpha = ImageOps.flip(im), (ImageOps.flip(alpha) if alpha else None)
        elif op == "crop":
            if len(a) != 4:
                raise ToolError("crop needs four numbers in percent: left, top, width, height (e.g. crop 10,10,80,80)")
            w, h = im.size
            box = (int(w * a[0] / 100), int(h * a[1] / 100), int(w * (a[0] + a[2]) / 100), int(h * (a[1] + a[3]) / 100))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ToolError("that crop leaves nothing of the picture")
            im, alpha = im.crop(box), (alpha.crop(box) if alpha else None)
        elif op == "resize":
            w, h = im.size
            if len(a) == 1 and a[0] <= 10:
                nw, nh = int(w * a[0]), int(h * a[0])      # a factor: "resize 0.5"
            elif len(a) == 1:
                nw, nh = int(a[0]), int(h * a[0] / w)       # a width, the height in proportion
            else:
                nw, nh = int(a[0]), int(a[1])
            nw, nh = max(1, min(nw, 8192)), max(1, min(nh, 8192))
            im = im.resize((nw, nh), Image.LANCZOS)
            alpha = alpha.resize((nw, nh), Image.LANCZOS) if alpha else None
        elif op == "brightness":
            im = ImageEnhance.Brightness(im).enhance(max(0.0, min(a[0], 4.0)))
        elif op == "contrast":
            im = ImageEnhance.Contrast(im).enhance(max(0.0, min(a[0], 4.0)))
        elif op == "blur":
            im = im.filter(ImageFilter.GaussianBlur(max(0.1, min(a[0], 50.0))))
        elif op == "sharpen":
            im = im.filter(ImageFilter.SHARPEN)
        elif op == "invert":
            im = ImageOps.invert(im)
        done.append(op + (" " + ",".join(f"{x:g}" for x in a) if a else ""))
    if alpha is not None and alpha.size == im.size:
        im.putalpha(alpha)
    rel = out or (str(Path(path).with_suffix("")) + "-edited" + src.suffix.lower())
    if Path(rel).suffix.lower() not in PICTURE_EXT:
        rel += src.suffix.lower()
    dst = ctx.ws.path(rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fmt = {".jpg": "JPEG", ".jpeg": "JPEG"}.get(dst.suffix.lower())
    with ctx.ws.changing(dst, by=ctx.by, what=f"edited picture {rel}"):
        (im.convert("RGB") if fmt == "JPEG" else im).save(dst, fmt) if fmt else im.save(dst)
    ctx.made.add(rel)
    return f"saved {rel} ({im.size[0]}x{im.size[1]}): {', '.join(done)}; the original {path} is unchanged"


@tool("speak_text", "media", 2, "Read text aloud into a sound file (.wav) with a Windows voice.",
      {"text": ("string", "what to say"), "path": ("string", "file name, optional"),
       "voice": ("string", "female or male, optional")}, ("text",))
def speak_text(ctx, text, path="", voice=""):
    if "windows-voices" in set(CONFIG.get("disabled_media_models") or []):
        raise ToolError("Windows voices are disabled. Open the model menu and choose Windows voices to enable them again.")
    rel = _name(ctx, path, text, ".wav")
    p = ctx.ws.path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    code = ("import sys, pyttsx3\n"
            "e = pyttsx3.init()\n"
            "vs = e.getProperty('voices')\n"
            "want = sys.argv[3]\n"
            "pick = [v for v in vs if (want == 'female' and 'zira' in v.name.lower()) or (want == 'male' and 'david' in v.name.lower())]\n"
            "if pick: e.setProperty('voice', pick[0].id)\n"
            "e.save_to_file(sys.argv[1], sys.argv[2]); e.runAndWait()\n")
    with ctx.ws.changing(p, by=ctx.by, what=f"voice {rel}"):
        r = subprocess.run([sys.executable, "-c", code, text, str(p), (voice or "").lower()], capture_output=True,
                           text=True, timeout=300, creationflags=NOWIN)
    if r.returncode or not p.exists() or p.stat().st_size < 1000:
        raise ToolError(f"the voice was not made: {(r.stderr or '').strip()[-200:]}")
    ctx.made.add(rel)
    return f"saved the voice as {rel}"
