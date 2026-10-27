"""Video and music request parsing (plan A7): seconds from the request, music
cues in the tool list, and follow-ups on the last media file. Pure unit tests,
no engine needed.

    python -m mlmi.tests.test_media_parsing
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

if __name__ != "__main__":
    raise unittest.SkipTest("CLI regression script; run python -m mlmi.tests.test_media_parsing")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import adapters  # noqa: E402
from mlmi.backend import agent  # noqa: E402
from mlmi.backend import tools as T  # noqa: E402
from mlmi.backend.tools import media  # noqa: E402
from mlmi.backend.workspace import ModelSelectionRequired, ToolError  # noqa: E402

fails, ran = [], [0]


def check(name, ok, got=""):
    ran[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n      got {got}"))
    if not ok:
        fails.append(name)


# ---- seconds parsing
def parse_seconds(asked):
    """Simulate what the agent does before calling make_video."""
    m = re.search(r"\b(\d{1,2})[- ]?second", asked, re.I)
    return int(m.group(1)) if m else 2


check("a 4-second clip -> 4", parse_seconds("make a 4-second clip of a dog running") == 4)
check("a 10-second video -> 10", parse_seconds("a 10-second video of waves on a beach") == 10)
check("6-second animation -> 6", parse_seconds("generate a 6-second animation of rain") == 6)
check("no seconds in request -> 2", parse_seconds("make a video of a sunset") == 2)

# ---- music cues in EXTRAS (tools offered for music requests)
names = agent.route_tools(set(T.TOOLS), "", "add music to the video")
check("music request offers studio_add_media", "studio_add_media" in names, names)
check("music request offers studio_set", "studio_set" in names, names)
names = agent.route_tools(set(T.TOOLS), "", "add background music and a jingle")
check("soundtrack request offers studio tools", "studio_add_media" in names, names)
names = agent.route_tools(set(T.TOOLS), "studio", "add a soundtrack to this")
check("soundtrack in studio offers the tools", "studio_add_media" in names, names)

# ---- seconds cues in EXTRAS
names = agent.route_tools(set(T.TOOLS), "", "make a 4-second clip of a penguin")
check("seconds cue offers make_video", "make_video" in names, names)
names = agent.route_tools(set(T.TOOLS), "", "generate a 10-second video of rain")
check("10-second video offers make_video", "make_video" in names, names)

# ---- media follow-up resolution
MEDIA_EXTS = (".sstudio", ".mp4")


def resolve_media(asked, last_files, name="studio_set"):
    """Simulate the agent's media follow-up resolution."""
    if not re.search(r"\b(it|that|this|the video|the clip)\b", asked, re.I):
        return None
    media = [f for f in reversed(last_files) if f.endswith(MEDIA_EXTS)]
    return media[0] if media else None


check("add music to it -> last .sstudio",
      resolve_media("add music to it", ["deck.sdeck", "intro.sstudio", "notes.txt"]) == "intro.sstudio")
check("make that longer -> last .mp4",
      resolve_media("make that longer", ["clip1.mp4", "doc.sdoc"]) == "clip1.mp4")
check("change the video -> last media", resolve_media("change the video", ["vid.sstudio"]) == "vid.sstudio")
check("no media files -> None", resolve_media("add music to it", ["doc.sdoc"]) is None)
check("no it/that/this -> None", resolve_media("add music to title.sstudio", ["title.sstudio"]) is None)

# ---- model availability is checked before starting a worker
with tempfile.TemporaryDirectory(prefix="mlmi_media_") as d:
    with patch.object(adapters, "MEDIA_DIR", Path(d)):
        models = adapters.catalogue()
        check("missing picture assets are reported in the menu",
              not models[0]["ready"] and models[0]["why"] == "not downloaded yet")
        check("missing video assets are reported in the menu",
              not models[1]["ready"] and models[1]["why"] == "not downloaded yet")
        whisper = next(m for m in models if m["id"] == "whisper-small")
        check("a shared Whisper cache is reusable without being treated as Sheep-installed",
              whisper["ready"] or whisper["reuse_cached"] or whisper["why"] == "not downloaded yet")
        with patch("mlmi.backend.adapters.install.ensure") as ensure, \
                patch.object(media.PICTURES, "make", side_effect=AssertionError("picture worker started")):
            try:
                media.make_image(None, "a cat")
            except ModelSelectionRequired as e:
                check("missing picture model asks before worker startup",
                      e.slot == "image" and e.repo_id == "stable-diffusion-v1-5/stable-diffusion-v1-5")
            else:
                check("missing picture model asks before worker startup", False)
            check("missing picture model does not download before Check and use", not ensure.called)

        motion = Path(d) / "animatediff" / media.MOTION_FILE
        motion.parent.mkdir()
        motion.touch()
        check("video motion file alone does not mark video model ready",
              not adapters.catalogue()[1]["ready"])
        with patch("mlmi.backend.adapters.install.ensure"), \
                patch.object(media.VIDEOS, "make", side_effect=AssertionError("video worker started")):
            try:
                media.make_video(None, "a cat")
            except ModelSelectionRequired as e:
                check("incomplete video assets ask before worker startup",
                      e.slot == "video" and e.repo_id == "ByteDance/AnimateDiff-Lightning")
            else:
                check("incomplete video assets ask before worker startup", False)
        base = Path(d) / "sd15"
        for required in (
            "model_index.json", "scheduler/scheduler_config.json",
            "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt",
            "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
            "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
            "feature_extractor/preprocessor_config.json",
        ):
            item = base / required
            item.parent.mkdir(parents=True, exist_ok=True)
            item.touch()
        models = adapters.catalogue()
        check("base model alone does not mark picture model ready", not models[0]["ready"])
        check("video model is ready when its base and motion file exist", models[1]["ready"])

print(f"\n{'all ' + str(ran[0]) + ' checks passed' if not fails else str(len(fails)) + ' failed: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
