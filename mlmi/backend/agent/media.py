"""Pictures, video, speech and transcription requests: what was asked for and where it goes.

Split out of agent.py (2026-10-21); every name is re-exported from the agent package."""
import json
import re
from pathlib import Path
from .. import tools as T
from ..workspace import ModelSelectionRequired, ToolError
from .routing import AI_VIDEO, PHOTO_ANIMATION, PHOTO_MONTAGE, STUDIO_EDIT, VIDEO_TEXT_ONLY, explicit_media_request


def explicit_speech_output(text):
    """True when the user explicitly asks Sheep to create a voice/audio file.

    A mention of narration while making a video is handled by
    ``explicit_media_plan``. This detector is for standalone or mixed image and
    speech requests, where a chat model can otherwise claim a WAV exists
    without ever calling ``speak_text``.
    """
    t = str(text or "")
    if re.search(r"\buse\s+(?:sheep['’]s\s+)?speak_text\s+tool\b", t, re.I):
        return True
    action = r"\b(?:create|make|generate|save|record|produce|read|speak|say)\b"
    output = (r"\b(?:spoken description|spoken version|voice[- ]?over|audio(?: file)?|wav|recording|text[- ]to[- ]speech|"
              r"out loud|aloud|in a voice|speech)\b")
    return bool(re.search(action + r"[^.!?\n]{0,120}" + output, t, re.I)
                or re.search(output + r"[^.!?\n]{0,100}" + action, t, re.I))


def words_to_speak(asked):
    """The exact words the person gave to be spoken: quoted, or after a colon ("read this
    out loud: Welcome to the bakery."), else "" (2026-10-14 live run: unquoted words after
    a colon were not taken and the model only pretended to read them)."""
    t = str(asked or "")
    q = re.search(r"\b(?:say|read aloud|read out|speak|containing|the words|out loud|aloud|in a voice)\s*[:：]?\s*"
                  r"[\"“]([^\"”\n]{3,240})[\"”]", t, re.I)
    if q:
        return q.group(1).strip()
    c = re.search(r"\b(?:say|speak|read|out loud|aloud|in a voice|voice|narrat\w*|speech|audio|recording)\b[^:：\n]{0,60}[:：]\s*(\S[^\n]{2,240})$",
                  t.strip(), re.I)
    return c.group(1).strip() if c else ""


def speech_output_path(ws, text):
    """Choose the requested WAV name or a safe unique default in the workspace."""
    match = re.search(
        r"\b(?:as|named|called|file(?:\s+name)?(?:\s+is)?|path(?:\s+is)?)\s*[`\"']?"
        r"(?P<path>[\w./\\ -]+\.wav)\b",
        str(text or ""), re.I)
    requested = match.group("path").strip().strip(" .\\/") if match else "media/prompt-voice.wav"
    return _unique_workspace_path(ws, requested)


def _clean_spoken_line(value):
    """Accept a short plain sentence from a model; reject tool chatter and junk."""
    line = re.sub(r"^\s*(?:spoken sentence|description|narration)\s*:\s*", "", str(value or ""), flags=re.I)
    line = re.sub(r"^\s*[\"'`]|[\"'`]\s*$", "", line.strip())
    line = " ".join(line.split())
    words = re.findall(r"\b[\w'-]+\b", line)
    if not 3 <= len(words) <= 24 or len(line) > 240 or "```" in line or "{" in line:
        return "A sheep astronaut explores the moon beneath the stars."
    return line


def explicit_transcription_request(text):
    """Detect an explicit speech-to-text task so a chat model cannot invent it."""
    return bool(re.search(r"\b(?:transcribe|make a transcript|write down what (?:was|is) said)\b",
                          str(text or ""), re.I))


def transcription_paths(ws, text):
    """Return (existing source, unique destination) for a named transcription job."""
    value = str(text or "")
    src = re.search(r"\b(?:transcribe|transcription|transcript of)\b[^.!?\n]{0,120}?"
                    r"(?P<path>[\w./\\ -]+?\.(?:wav|mp3|m4a|ogg|flac|mp4|webm))\b", value, re.I)
    if not src:
        return "", ""
    source = src.group("path").strip().strip(" .\\/")
    dest = re.search(r"\b(?:save|write|store)\s+(?:the\s+)?transcript\b[^.!?\n]{0,90}"
                     r"\b(?:as|to|named)\s*[`\"']?(?P<path>[\w./\\ -]+?\.txt)\b", value, re.I)
    requested = dest.group("path").strip().strip(" .\\/") if dest else str(Path(source).with_suffix(".txt"))
    return source, _unique_workspace_path(ws, requested)


def _unique_workspace_path(ws, requested):
    """Choose a new artifact path without replacing existing user work."""
    path = Path(str(requested).replace("\\", "/"))
    if not ws.path(path.as_posix()).exists():
        return path.as_posix()
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}").as_posix()
        if not ws.path(candidate).exists():
            return candidate
    raise ToolError("could not find an unused output name for this media job")


# The video model draws what it is told: it was handed the person's own words, trimmed by a
# regex ("a 2-second text-to-video clip of a red kite ... with a short", 2026-10-11). The chat
# model now writes the scene as a picture is described: subject, setting, light, camera, style.
VISUAL_INSTRUCTION = (
    "Rewrite the request as a prompt for an image or video generator. Describe only what is seen: "
    "the subject, what it does, the setting, the light, the camera view and the style, in one or two "
    "sentences, at most 60 words. Leave out words about making a video, seconds, voice-over, music, "
    "files or apps. Do not invent people, brands or text on screen that the request does not mention. "
    "Return only the prompt.")
META = re.compile(r"\b(?:\d{1,2}[- ]?seconds?\b(?:\s+long)?|text[- ]to[- ]video|video clip|a video of|video of|clip of|"
                  r"voice[- ]?over|narrat\w*|mp4|sheep studio)\b", re.I)


def scene_from_request(text):
    """The person's words without the asking: the fallback when no model writes the scene."""
    scene = re.sub(r"^\s*(?:please\s+)?(?:sheep[, :]\s*)?(?:make|create|generate|produce)\s+", "", str(text or ""), flags=re.I)
    scene = re.sub(r"[,;]?\s+(?:and\s+)?(?:with\s+)?(?:brief\s+)?(?:a\s+)?(?:voice[- ]?over|narrat\w*|spoken description)\b.*$",
                   "", scene, flags=re.I)
    scene = re.sub(r"\b(?:a|an)\s+(?:\d{1,2}[- ]?second\s+)?(?:video\s+)?clip\s+of\s+", "", scene, flags=re.I)
    scene = re.sub(r"\s+", " ", META.sub(" ", scene))
    scene = re.sub(r"[,;]?\s+(?:with|and)(?:\s+an?)?(?:\s+(?:short|brief|quick))?\s*$", "", scene, flags=re.I)
    scene = re.sub(r"^(?:(?:an?|the)\s+)+(?=(?:an?|the)\s)", "", scene.strip(), flags=re.I)
    return scene.strip(" .,!;\n")


def clean_visual(written, fallback):
    """The model's scene, kept only when it reads as one: a few sentences of description,
    no code, no meta words left. Otherwise the trimmed request."""
    v = re.sub(r"^\s*(?:prompt\s*:\s*)?[\"'`]+|[\"'`]+\s*$", "", str(written or "").strip(), flags=re.I)
    v = re.sub(r"\s+", " ", META.sub("", v)).strip(" .,;:")
    words = re.findall(r"\b[\w'-]+\b", v)
    if "```" in v or not 5 <= len(words) <= 80 or re.search(r"\b(?:I can|I cannot|sorry|as an AI)\b", v, re.I):
        return fallback
    return v


def explicit_media_plan(ws, text, selected_images, narration="", visual=""):
    """Build a bounded tool sequence for an explicit video/montage request.

    The LLM still writes requested narration. Sheep owns the mechanical routing:
    the 1.7B local model often answers a media prompt in prose instead of making
    tool calls, and must never turn a clear video request into a document.
    """
    t = str(text or "")
    if not explicit_media_request(t):
        return None
    text_only = bool(re.search(VIDEO_TEXT_ONLY, t, re.I))
    montage = bool(re.search(PHOTO_MONTAGE, t, re.I)) and not text_only
    photo_animation = bool(re.search(PHOTO_ANIMATION, t, re.I)) and not text_only
    text_video = bool(re.search(AI_VIDEO, t, re.I) and not montage and not photo_animation
                       and not re.search(STUDIO_EDIT, t, re.I))
    selected = list(dict.fromkeys(str(x) for x in (selected_images or [])))
    if (montage or photo_animation) and not selected:
        raise ToolError("attach or select the photos in Sheep before making this photo video; no photo was used")
    if photo_animation and len(selected) > 9:
        raise ToolError("photo-guided animation supports up to nine selected photos; select fewer, or make a montage")

    seconds_match = re.search(r"\b(\d{1,2})[- ]?second\b", t, re.I)
    seconds = max(1, min(6, int(seconds_match.group(1)))) if seconds_match else 2
    calls = []
    stem = "prompt-media"
    if montage:
        stem = "photo-montage"
    elif photo_animation:
        stem = "photo-animation"
    elif text_video:
        stem = "text-video"

    voice_path = ""
    if narration:
        voice_path = _unique_workspace_path(ws, f"media/{stem}-narration.wav")
        calls.append(("speak_text", {"text": narration, "path": voice_path}))

    if montage:
        project = _unique_workspace_path(ws, f"media/{stem}.sstudio")
        rendered = _unique_workspace_path(ws, f"media/{stem}.mp4")
        duration = max(2, len(selected) * 2)
        calls.append(("studio_create", {"path": project, "format": "16:9", "duration": duration}))
        for index, image_path in enumerate(selected):
            calls.append(("studio_add_media", {"path": project, "file": image_path, "start": index * 2,
                                                  "duration": 2, "width": 100, "height": 100,
                                                  "enter": "fade", "exit": "fade"}))
        if voice_path:
            calls.append(("studio_add_media", {"path": project, "file": voice_path, "start": 0, "volume": 1.0}))
        calls.append(("studio_render", {"path": project, "out": rendered}))

    if photo_animation or text_video:
        raw_video = _unique_workspace_path(ws, f"media/{stem}-source.mp4")
        scene = visual or scene_from_request(t)
        if photo_animation:
            scene = ("A gentle, natural photo-guided animation with subtle camera movement; "
                     "preserve the selected photos' subjects and composition. " + scene).strip()
        calls.append(("make_video", {"prompt": scene or t, "seconds": seconds, "path": raw_video,
                                      "source_images": selected if photo_animation else []}))
        if narration:
            project = _unique_workspace_path(ws, f"media/{stem}-narrated.sstudio")
            rendered = _unique_workspace_path(ws, f"media/{stem}-narrated.mp4")
            duration = max(seconds, 3)
            calls.extend([
                ("studio_create", {"path": project, "format": "16:9", "duration": duration}),
                ("studio_add_media", {"path": project, "file": raw_video, "start": 0, "duration": seconds,
                                       "width": 100, "height": 100}),
                ("studio_add_media", {"path": project, "file": voice_path, "start": 0, "volume": 1.0}),
                ("studio_render", {"path": project, "out": rendered}),
            ])
    return calls


def video_wants_narration(text):
    """Resolve an ambiguous video sound request to speech, not silent output.

    Music and effects are different requests: the installed local tools can add
    speech, but do not synthesize a soundtrack or sound effects.
    """
    t = str(text or "")
    direct = re.search(r"\b(narrat\w*|voice[- ]?over|spoken description)\b", t, re.I)
    if direct:
        return True
    if re.search(r"\b(music|sound effects?|sfx|soundtrack|background track)\b", t, re.I):
        return False
    return bool(re.search(r"\bwith\s+(?:some\s+)?(?:sound|audio)\b|\badd\s+(?:some\s+)?audio\b", t, re.I))


def extract_requested_narration(text):
    """Return wording the user explicitly asked the voice to say, unchanged."""
    t = str(text or "")
    marker = re.search(
        r"\b(?:narration|voice[- ]?over|spoken description)\b[\s\S]{0,64}?"
        r"\b(?:say|says|saying|read|reads|reading)\b\s*[:\-]\s*", t, re.I)
    if not marker:
        marker = re.search(r"\b(?:narration|voice[- ]?over|spoken description)\s*:\s*", t, re.I)
    if not marker:
        marker = re.search(r"\b(?:say|says|saying)\s*:\s*", t, re.I)
    if not marker:
        return ""

    value = t[marker.end():].strip()
    if not value:
        return ""
    quote_pairs = (("\"", "\""), ("'", "'"), ("`", "`"), ("“", "”"), ("‘", "’"))
    for opening, closing in quote_pairs:
        if value.startswith(opening):
            end = value.find(closing, len(opening))
            if end >= 0:
                return value[len(opening):end].strip()
            return value[len(opening):].strip()

    # The common unquoted form is a sentence followed by more instructions:
    # "... says: A kite flies. Use the selected video model ...".
    boundary = re.search(
        r"\s+(?=(?:use the selected\b|use the video\b|combine the narration\b|"
        r"save (?:the|my|it)\b|render (?:the|a)\b|do not\b|don't\b|then\s+(?:render|save|add)\b))",
        value, re.I)
    if boundary:
        value = value[:boundary.start()]
    return value.strip().strip("\"'“”‘’`")


VIDEO_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _selected_video_images(ws, attachments, open_path, asked):
    """Current-turn attachments, images in the open Studio project, and paths named by the user."""
    found = []

    def add(rel):
        rel = str(rel or "").strip().strip("\"'()[]{}.,;!? ")
        if not rel or Path(rel).suffix.lower() not in VIDEO_IMAGE_EXTS:
            return
        try:
            src = ws.path(rel, must_exist=True)
            rel = ws.rel(src)
        except (ToolError, OSError):
            return
        if rel not in found:
            found.append(rel)

    for rel in attachments:
        add(rel)
    if str(open_path or "").lower().endswith(".sstudio"):
        try:
            project = json.loads(ws.path(open_path, must_exist=True).read_text(encoding="utf-8"))
            for layer in project.get("layers", []):
                if layer.get("kind") == "image":
                    add(layer.get("src"))
        except (ToolError, OSError, ValueError, AttributeError):
            pass
    path_re = r"(?<![\w])([\w./\\ -]+\.(?:png|jpe?g|webp|gif|bmp))(?=$|[\s),;.!?])"
    for rel in re.findall(path_re, asked or "", re.I):
        add(rel)
    return found


def _video_call_sources(ws, args, selected, asked):
    """Use only photos attached, selected in the open Studio project, or named by the user."""
    if re.search(VIDEO_TEXT_ONLY, asked or "", re.I):
        args["source_images"] = []
        return
    requested = args.get("source_images")
    if requested:
        if isinstance(requested, str):
            requested = T._coerce(T.TOOLS["make_video"].params["source_images"], requested)
        allowed = {}
        for rel in selected:
            try:
                normalized = ws.rel(ws.path(rel, must_exist=True))
                allowed[normalized.lower()] = normalized
            except ToolError:
                continue
        out = []
        for rel in requested if isinstance(requested, (list, tuple)) else [requested]:
            try:
                normalized = ws.rel(ws.path(str(rel), must_exist=True))
            except ToolError as exc:
                raise ToolError(f"cannot use {rel}: {exc}") from exc
            if normalized.lower() not in allowed:
                raise ToolError(f"{rel} was not attached, selected in the open Studio project, or named by you")
            if normalized not in out:
                out.append(normalized)
        args["source_images"] = out
    else:
        args["source_images"] = list(selected)
    if not args["source_images"] and re.search(PHOTO_ANIMATION, asked or "", re.I):
        raise ToolError("this request needs a photo for image-guided video; attach a picture or select one in Studio, "
                        "or ask for a text-only video")
