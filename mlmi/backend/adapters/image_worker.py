"""Pictures and video for Sheep: Stable Diffusion 1.5 with the LCM-LoRA (4 steps),
or, with --motion, the same SD 1.5 turned into a video model by the
AnimateDiff-Lightning motion module (4 steps, 8 frames a second).

A resident worker: started once by the image adapter, it loads the pipeline on
its device and then answers one JSON request per line on stdin with one JSON
reply per line on stdout. It runs in whichever Python fits the device: Sheep's
DirectML runtime for AMD and Intel GPUs, the app's Python for CUDA or the CPU.

    python image_worker.py --device dml:1|cuda|cpu
    {"prompt": "...", "out": "C:/.../picture.png", "width": 512, "height": 512, "seed": 7}
    -> {"ok": true, "out": "...", "seconds": 4.2}  or  {"ok": false, "error": "..."}

    python image_worker.py --device ... --motion FILE.safetensors
    {"prompt": "...", "out": "C:/.../frames_dir", "frames": 16, "width": 512, "height": 512}
    -> {"ok": true, "out": "frames_dir", "frames": 16, "seconds": 61.0}   (frame_000.png ...)

Weights: stable-diffusion-v1-5 (CreativeML OpenRAIL-M), lcm-lora-sdv1-5
(OpenRAIL++) and AnimateDiff-Lightning (CreativeML OpenRAIL-M), all from
Sheep's media folder; nothing is fetched at run time.
"""
import argparse
import inspect
import json
import sys
import time

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LCM = "latent-consistency/lcm-lora-sdv1-5"


def device_of(name):
    import torch
    if name.startswith("dml"):
        import torch_directml
        idx = int(name.split(":")[1]) if ":" in name else 0
        return torch_directml.device(idx), torch.float16
    if name == "cuda":
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def load(name, base=BASE, lora=LCM):
    import torch
    from diffusers import LCMScheduler, StableDiffusionPipeline
    dev, dtype = device_of(name)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=dtype, variant="fp16", safety_checker=None,
                                                       requires_safety_checker=False, local_files_only=True)
    except (OSError, ValueError):            # a style model that ships full-size weights only
        pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=dtype, safety_checker=None,
                                                       requires_safety_checker=False, local_files_only=True)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lora, weight_name="pytorch_lora_weights.safetensors", local_files_only=True)
    pipe.fuse_lora()
    if name == "cuda":
        pipe.enable_model_cpu_offload()          # shares a small card with the text model
    else:
        pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    return pipe, dev, torch


def load_motion(name, base, motion, ip_adapter=""):
    """SD 1.5 plus the motion module: a 4-step text-to-video pipeline. Memory is kept
    small for 2-4 GB cards: attention one slice at a time, frames decoded two at a time."""
    import torch
    from diffusers import AnimateDiffPipeline, EulerDiscreteScheduler, MotionAdapter
    from safetensors.torch import load_file
    dev, dtype = device_of(name)
    adapter = MotionAdapter()
    adapter.load_state_dict(load_file(motion, device="cpu"))
    adapter = adapter.to(dtype)
    try:
        pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype, variant="fp16",
                                                   local_files_only=True)
    except (OSError, ValueError):
        pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype, local_files_only=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing",
                                                        beta_schedule="linear")
    # attention a few heads at a time: unsliced, 8 frames at 512 px asked a 2 GB
    # card for one 1.36 GB block (2026-10-06); enable_attention_slicing skips the
    # motion UNet, so the processor is set on it directly
    if ip_adapter:
        pipe.load_ip_adapter(ip_adapter, subfolder="models", weight_name="ip-adapter-plus_sd15.bin",
                             image_encoder_folder="image_encoder", local_files_only=True)
    else:
        # Diffusers' IP-Adapter loader clones existing processors with a no-arg
        # constructor, but SlicedAttnProcessor requires slice_size. Its adapter
        # attention processor already uses torch SDPA on supported PyTorch builds.
        from diffusers.models.attention_processor import SlicedAttnProcessor
        pipe.unet.set_attn_processor(SlicedAttnProcessor(slice_size=4))
    if name == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    return pipe, dev, torch


def video(pipe, torch, req, progress=None):
    """Frames into req["out"] (a folder) as frame_000.png ...; longer than 16 frames
    uses FreeNoise windows so the motion module's 16-frame span still holds."""
    import os
    n = max(8, min(48, int(req.get("frames") or 16)))
    if n > 16 and hasattr(pipe, "enable_free_noise"):
        pipe.enable_free_noise(context_length=16, context_stride=4)
    elif hasattr(pipe, "disable_free_noise"):
        pipe.disable_free_noise()
    # the 4-step motion model was distilled at 512 px: at 256 px it made noise
    # on every device (2026-10-06), so 512 is the floor
    w = max(512, min(640, int(req.get("width") or 512)) // 8 * 8)
    h = max(512, min(640, int(req.get("height") or 512)) // 8 * 8)
    g = torch.Generator("cpu").manual_seed(int(req.get("seed") or 0) or int(time.time()) % 100000)
    kwargs = {"prompt": req["prompt"], "negative_prompt": req.get("negative") or None, "num_frames": n,
              "guidance_scale": 1.0, "num_inference_steps": 4, "width": w, "height": h,
              "generator": g, "decode_chunk_size": 2}
    # These are model denoising steps, not a claim that the MP4 is already saved.
    steps = int(kwargs["num_inference_steps"])

    def report_step(done):
        event = {"type": "progress", "stage": "Denoising video frames", "step": done, "total": steps}
        if progress:
            try:
                progress(event)
            except Exception:
                pass
        try:
            print(json.dumps(event), flush=True)
        except OSError:
            pass  # a closed UI/worker pipe must not break an otherwise valid render

    try:
        call_params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        call_params = {}
    if "callback_on_step_end" in call_params:
        def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
            report_step(step_index + 1)
            return callback_kwargs
        kwargs["callback_on_step_end"] = on_step_end
    elif "callback" in call_params and "callback_steps" in call_params:
        kwargs["callback"] = lambda step_index, _timestep, _latents: report_step(step_index + 1)
        kwargs["callback_steps"] = 1
    opened, reference = [], None
    paths = req.get("source_images") or []
    if paths:
        from PIL import Image, ImageOps
        for path in paths:
            with Image.open(path) as source:
                source.draft("RGB", (768, 768))
                source.thumbnail((768, 768))
                opened.append(source.convert("RGB"))
        if len(opened) == 1:
            reference = opened[0]
        else:
            # One adapter accepts one guidance image. A contact sheet lets every
            # selected photo guide the clip without discarding any of them.
            cols = min(3, len(opened))
            rows = (len(opened) + cols - 1) // cols
            cell = 384
            reference = Image.new("RGB", (cols * cell, rows * cell), (24, 28, 38))
            for i, source in enumerate(opened):
                tile = ImageOps.contain(source, (cell - 16, cell - 16))
                reference.paste(tile, ((i % cols) * cell + (cell - tile.width) // 2,
                                       (i // cols) * cell + (cell - tile.height) // 2))
        kwargs["ip_adapter_image"] = reference
        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(0.65)
    tracks_cuda = (str(getattr(pipe, "_execution_device", "")).startswith("cuda")
                   and getattr(torch, "cuda", None) is not None and torch.cuda.is_available())
    if tracks_cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        frames = pipe(**kwargs).frames[0]
    finally:
        if reference is not None and reference not in opened:
            reference.close()
        for source in opened:
            source.close()
    os.makedirs(req["out"], exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(req["out"], f"frame_{i:03d}.png"))
    result = {"frames": len(frames), "width": w, "height": h}
    if tracks_cuda:
        result["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--base", default=BASE, help="a folder (Sheep's ~/.mlmi/media/sd15) or a Hugging Face id")
    ap.add_argument("--lora", default=LCM)
    ap.add_argument("--motion", default="", help="the AnimateDiff-Lightning file: makes this a video worker")
    ap.add_argument("--ip-adapter", default="", help="local IP-Adapter Plus folder for selected photo guidance")
    a = ap.parse_args()
    t0 = time.time()
    try:
        pipe, dev, torch = (load_motion(a.device, a.base, a.motion, a.ip_adapter) if a.motion
                            else load(a.device, a.base, a.lora))
    except Exception as e:
        what = "video" if a.motion else "image"
        print(json.dumps({"ok": False, "error": f"could not load the {what} model: {str(e)[:300]}"}), flush=True)
        return 1
    print(json.dumps({"ok": True, "ready": True, "device": a.device, "seconds": round(time.time() - t0, 1)}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            t1 = time.time()
            if a.motion:
                r = video(pipe, torch, req)
                print(json.dumps({"ok": True, "out": req["out"], "seconds": round(time.time() - t1, 1), **r}), flush=True)
                continue
            g = torch.Generator("cpu").manual_seed(int(req.get("seed") or 0) or int(time.time()) % 100000)
            w = max(256, min(768, int(req.get("width") or 512)) // 8 * 8)
            h = max(256, min(768, int(req.get("height") or 512)) // 8 * 8)
            img = pipe(req["prompt"], negative_prompt=req.get("negative") or None, num_inference_steps=int(req.get("steps") or 4),
                       guidance_scale=float(req.get("guidance") or 1.2), width=w, height=h, generator=g).images[0]
            img.save(req["out"])
            print(json.dumps({"ok": True, "out": req["out"], "seconds": round(time.time() - t1, 1), "width": w, "height": h}),
                  flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)[:300]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
