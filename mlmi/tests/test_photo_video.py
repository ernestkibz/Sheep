"""Photo-aware video routing and worker tests; model inference is an opt-in smoke test."""
import json
import os
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image  # noqa: E402
from mlmi.backend import adapters, agent, orchestrator, orient  # noqa: E402
from mlmi.backend.adapters import image_worker  # noqa: E402
from mlmi.backend import tools as T  # noqa: E402
from mlmi.backend.tools import media, studio  # noqa: E402
from mlmi.backend.workspace import ModelSelectionRequired, ToolError, Workspace  # noqa: E402


class PhotoVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root / "workspace")

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, color=(120, 80, 40)):
        path = self.ws.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), color).save(path)
        return path

    def test_routes_text_photo_animation_and_exact_montage(self):
        allowed = set(T.offered(3))
        animate = agent.route_tools(allowed, "", "animate this attached photo into a video")
        self.assertIn("make_video", animate)
        self.assertNotIn("studio_render", animate)
        text_only = agent.route_tools(allowed, "", "make a text-only video without using the attached photo")
        self.assertIn("make_video", text_only)
        text_only = agent.route_tools(allowed, "", "make a video of a dog running")
        self.assertIn("make_video", text_only)
        montage = agent.route_tools(allowed, "", "make a photo montage from these pictures")
        self.assertNotIn("make_video", montage)
        self.assertIn("studio_add_media", montage)
        self.assertIn("studio_render", montage)

    def test_video_sound_defaults_to_narration_but_music_and_effects_are_not_substituted(self):
        self.assertTrue(agent.video_wants_narration("make a short video with sound"))
        self.assertTrue(agent.video_wants_narration("make a clip with audio"))
        self.assertTrue(agent.video_wants_narration("make a video with narration and music"))
        self.assertFalse(agent.video_wants_narration("make a video with music"))
        self.assertFalse(agent.video_wants_narration("make a clip with sound effects"))
        self.assertFalse(agent.video_wants_narration("make a silent video"))

    def test_background_canaries_yield_without_waiting_under_scheduler_lock(self):
        instance = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
        instance.st = types.SimpleNamespace(model_path="model.ssf")
        instance.plan = {"device": "cpu"}
        instance.validation_signature = lambda *args: "signature"
        instance.entry = lambda *args: {"deep_validation": {"signature": "signature", "rows": []}}
        instance._background_should_yield = lambda: True
        instance._background_yield_reason = lambda: ("waiting for chat", "interactive request active")
        instance.background_checks = {"canaries": {}}
        outcome = instance.canary_suite("background", [("probe", [], [], lambda text, calls: True)],
                                        signature="signature")
        self.assertIsNone(outcome)
        self.assertEqual(instance.background_checks["canaries"]["state"], "waiting for chat")
        self.assertEqual(instance.background_checks["canaries"]["done"], 0)

    def test_orientation_yields_at_task_boundary_instead_of_waiting(self):
        instance = types.SimpleNamespace(chat_kwargs=lambda: {}, write_mode=lambda: "auto")
        state = types.SimpleNamespace(orch=instance)
        with tempfile.TemporaryDirectory() as directory:
            rows = orient.run_inprocess(state, [object()], Path(directory), should_yield=lambda: False)
        self.assertEqual(rows, [])

    def test_background_audits_resume_after_interactive_turn(self):
        instance = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
        instance.st = types.SimpleNamespace(model_path="model.ssf", engine=types.SimpleNamespace(alive=lambda: True))
        instance.plan = {"device": "cpu"}
        instance.validation_signature = lambda *args: "signature"
        instance._background_thread = None
        instance._resume_thread = None
        instance.loading = threading.Event()
        instance._background_should_yield = lambda: False
        queued = []
        instance._queue_background_checks = lambda path, signature: queued.append((path, signature))
        self.assertTrue(instance.resume_background_checks())
        self.assertEqual(queued, [("model.ssf", "signature")])

    def test_gpu_lease_forwards_stop_event_while_waiting(self):
        cancel_event = types.SimpleNamespace(is_set=lambda: False)
        ctx = types.SimpleNamespace(cancel_event=cancel_event)
        previous = media.LEND
        received = []
        media.LEND = (lambda what, gb, cancel_event=None: received.append((what, gb, cancel_event)) or True,
                      lambda: None)
        try:
            self.assertTrue(media._lend_gpu(ctx, "video", 3.6))
        finally:
            media.LEND = previous
        self.assertIs(received[0][2], cancel_event)

    def test_words_context_routes_explicit_video_and_narration_to_media_and_studio(self):
        allowed = set(T.offered(3))
        routed = agent.route_tools(allowed, "studio",
                                   "Create a 2-second text-to-video clip with narration")
        self.assertIn("make_video", routed)
        self.assertIn("studio_create", routed)
        self.assertIn("studio_add_media", routed)
        self.assertIn("studio_render", routed)
        self.assertIn("speak_text", routed)

        abilities = {name: True for name in ("makes video", "makes pictures", "reads pictures", "hears", "speaks")}
        state = types.SimpleNamespace(permission="auto", model_card=lambda: {"tier": 3},
                                      orch=types.SimpleNamespace(abilities=lambda: abilities, NEEDS={}))
        _, names, _ = agent.setup(state, "words", "Create a 2-second text-to-video clip with narration")
        self.assertIn("make_video", names)
        self.assertIn("studio_add_media", names)
        self.assertIn("studio_render", names)
        self.assertIn("speak_text", names)
        self.assertTrue(agent.explicit_media_request(
            "Create a text-to-video clip with narration and render the result through Sheep Studio"))
        self.assertTrue(agent.is_ai_video(
            "Create a text-to-video clip with narration and render the result through Sheep Studio"))

    def test_cloud_tool_tier_does_not_raise_the_local_checker_tier(self):
        state = types.SimpleNamespace(
            permission="auto", model_card=lambda: {"tier": 0},
            orch=types.SimpleNamespace(abilities=lambda: {}, NEEDS={}))
        _, local_names, _ = agent.setup(state, "", "Create a document about tea")
        _, cloud_names, _ = agent.setup(state, "", "Create a document about tea", tier_override=3)
        self.assertNotIn("words_create", local_names)
        self.assertIn("words_create", cloud_names)
        self.assertEqual(state.model_card()["tier"], 0)

    def test_explicit_media_plan_creates_montage_and_guided_animation_with_voice(self):
        first = self.image("attachments/first.png")
        second = self.image("attachments/second.jpg")
        selected = ["attachments/first.png", "attachments/second.jpg"]
        prompt = "Create an exact-photo montage and photo-guided animation from the selected pictures, with narration."
        plan = agent.explicit_media_plan(self.ws, prompt, selected, "Your selected photographs unfold gently.")
        names = [name for name, _ in plan]
        self.assertEqual(names.count("studio_render"), 2)
        self.assertEqual(names.count("make_video"), 1)
        self.assertEqual(names.count("speak_text"), 1)
        image_layers = [args for name, args in plan if name == "studio_add_media" and args.get("file", "").endswith((".png", ".jpg"))]
        self.assertEqual([args["file"] for args in image_layers], selected)
        video = next(args for name, args in plan if name == "make_video")
        self.assertEqual(video["source_images"], selected)
        self.assertNotEqual(first.resolve(), second.resolve())

    def test_explicit_photo_creation_never_silently_ignores_missing_selection(self):
        prompt = "Create an exact-photo montage and photo-guided animation with narration."
        with self.assertRaisesRegex(ToolError, "attach or select the photos"):
            agent.explicit_media_plan(self.ws, prompt, [], "A gentle visual story.")

    def test_text_video_plan_keeps_text_only_override(self):
        prompt = "Create a 2-second text-to-video clip, without using any photos."
        plan = agent.explicit_media_plan(self.ws, prompt, ["attachments/ignored.png"], "A scene moves gently.")
        video = next(args for name, args in plan if name == "make_video")
        self.assertEqual(video["source_images"], [])
        self.assertFalse(any(name == "studio_add_media" and args.get("file", "").endswith(".png")
                             for name, args in plan))

    def test_explicit_speech_file_requests_are_routed_and_named_safely(self):
        prompt = ("Use Sheep's speak_text tool with the Windows voice to create a one-sentence description. "
                  "Save it in this workspace as sheep_astronaut_description.wav.")
        self.assertTrue(agent.explicit_speech_output(prompt))
        self.assertIn("speak_text", agent.route_tools(set(T.offered(3)), "", prompt))
        self.assertEqual(agent.speech_output_path(self.ws, prompt), "sheep_astronaut_description.wav")

        existing = self.ws.path("sheep_astronaut_description.wav")
        existing.write_bytes(b"previous audio placeholder")
        self.assertEqual(agent.speech_output_path(self.ws, prompt), "sheep_astronaut_description-2.wav")

    def test_spoken_description_request_is_detected_but_video_narration_stays_in_video_plan(self):
        self.assertTrue(agent.explicit_speech_output(
            "Create an illustration and a short spoken description using a Windows voice."))
        self.assertFalse(agent.explicit_speech_output(
            "Create a text-to-video clip with narration and render it through Studio."))
        self.assertEqual(agent._clean_spoken_line("A sheep astronaut explores the moon beneath the stars."),
                         "A sheep astronaut explores the moon beneath the stars.")
        self.assertEqual(agent._clean_spoken_line("{tool call"),
                         "A sheep astronaut explores the moon beneath the stars.")

    def test_transcription_paths_keep_the_requested_source_and_avoid_overwrite(self):
        prompt = ("Transcribe sheep_astronaut_description.wav with local Whisper and save the transcript "
                  "as sheep_astronaut_description_transcript.txt.")
        self.assertTrue(agent.explicit_transcription_request(prompt))
        source = self.ws.path("sheep_astronaut_description.wav")
        source.write_bytes(b"fake test audio")
        audio, transcript = agent.transcription_paths(self.ws, prompt)
        self.assertEqual(audio, "sheep_astronaut_description.wav")
        self.assertEqual(transcript, "sheep_astronaut_description_transcript.txt")
        self.ws.path(transcript).write_text("preserve existing work", encoding="utf-8")
        self.assertEqual(agent.transcription_paths(self.ws, prompt)[1],
                         "sheep_astronaut_description_transcript-2.txt")

    def test_transcribe_tool_saves_local_result_to_requested_txt_path(self):
        source = self.ws.path("spoken.wav")
        source.write_bytes(b"test audio")
        fake_model = types.SimpleNamespace(transcribe=lambda *_args, **_kwargs: (
            [types.SimpleNamespace(text=" A sheep explores the moon.")], types.SimpleNamespace(language="en")))
        with patch.object(media, "_require_selected_model"), patch.object(media, "_whisper", return_value=fake_model):
            ctx = T.Ctx(self.ws, "test")
            result = T.run(self.ws, "transcribe_audio", {"path": "spoken.wav", "out": "named-transcript.txt"},
                           by="test", ctx=ctx)
        self.assertIn("saved as named-transcript.txt", result)
        self.assertEqual(self.ws.path("named-transcript.txt").read_text(encoding="utf-8"),
                         "A sheep explores the moon.\n")

    def test_agent_executes_requested_voice_file_if_chat_model_only_claims_a_path(self):
        import threading

        class FakeLocal:
            retry_device = None

            def chat(self, messages, _tools=None, **_kwargs):
                system = messages[0]["content"] if messages else ""
                if system.startswith("Write exactly one short"):
                    return {"content": "A small sheep explores the moon."}, {"route": "local", "ms": 1}
                return {"content": "Output path: spoken-description.wav"}, {"route": "local", "ms": 1}

        class FakeSession:
            def __init__(self):
                self.stop = threading.Event()
                self.pending = {}

            def reset_stop(self):
                self.stop.clear()

            def begin_action(self, _name):
                return True

            def end_action(self, _name):
                pass

            def set_cancel_hook(self, _hook):
                pass

        state = types.SimpleNamespace(
            session=FakeSession(), ws=self.ws, mode=2, permission="auto",
            model_card=lambda: {"tier": 3}, local=FakeLocal(), cloud=types.SimpleNamespace(), orch=None)
        events = []

        def fake_run(_ws, name, args, by="agent", ctx=None):
            self.assertEqual(name, "speak_text")
            ctx.made.add(args["path"])
            return f"saved the voice as {args['path']}"

        prompt = "Create a short spoken description and save it as spoken-description.wav."
        with patch.object(agent, "setup", return_value=(['media'], ['speak_text'], "test system")), \
                patch.object(agent.shepherd, "intent", return_value=None), \
                patch.object(T, "run", side_effect=fake_run):
            agent.run(state, [{"role": "user", "content": prompt}], app="words", emit=events.append, cache=False)

        calls = [event for event in events if event.get("type") == "tool_call"]
        text = [event.get("text", "") for event in events if event.get("type") == "text"]
        self.assertEqual([event["name"] for event in calls], ["speak_text"])
        self.assertTrue(any("Created the spoken audio: spoken-description.wav" in line for line in text))
        self.assertFalse(any(line.startswith("Output path:") for line in text))

    def test_cloud_media_plan_completes_explicit_voice_output_it_omits(self):
        import threading

        class FakeLocal:
            retry_device = None

            def chat(self, messages, _tools=None, **_kwargs):
                return {"content": "A sheep astronaut explores the moon."}, {"route": "local", "ms": 1}

        class FakeCloud:
            def ready(self):
                return True

            def chat(self, *_args, **_kwargs):
                return {"content": "planner response"}, {"route": "cloud", "ms": 1}

        class FakeSession:
            def __init__(self):
                self.stop = threading.Event()
                self.pending = {}

            def reset_stop(self):
                self.stop.clear()

            def begin_action(self, _name):
                return True

            def end_action(self, _name):
                pass

            def set_cancel_hook(self, _hook):
                pass

        state = types.SimpleNamespace(
            session=FakeSession(), ws=self.ws, mode=1, permission="auto",
            model_card=lambda: {"tier": 3}, local=FakeLocal(), cloud=FakeCloud(), orch=None)
        events = []
        planned = [{"tool": "make_image", "args": {"prompt": "a sheep astronaut"}}]

        def fake_run(_ws, name, args, by="agent", ctx=None):
            output = "image.png" if name == "make_image" else args["path"]
            ctx.made.add(output)
            return f"saved {output}"

        prompt = "Create an illustration and a short spoken description using a Windows voice."
        with patch.object(agent, "setup", return_value=([], ["make_image", "speak_text"], "test system")), \
                patch.object(agent, "_plan_calls", return_value=planned), \
                patch.object(T, "run", side_effect=fake_run):
            agent.run(state, [{"role": "user", "content": prompt}], app="chat",
                      emit=events.append, auto_confirm="yes", cache=False)

        calls = [event for event in events if event.get("type") == "tool_call"]
        self.assertEqual([event["name"] for event in calls], ["make_image", "speak_text"])
        self.assertEqual(calls[1]["route"], "local")
        self.assertTrue(any(event.get("type") == "file_changed" and event.get("path", "").endswith(".wav")
                            for event in events))
        self.assertTrue(any(event.get("type") == "done" and "image.png" in event.get("changed", [])
                            for event in events))

    def test_video_tool_is_available_for_first_use_before_weights_are_downloaded(self):
        orch = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
        orch.st = types.SimpleNamespace(model_path="", mode=0)
        orch.canaries = []
        with patch.object(adapters, "catalogue", return_value=[]), \
                patch.object(adapters, "video_installable", return_value=True), \
                patch("mlmi.backend.tools.vision.has_tower", return_value=False), \
                patch("mlmi.backend.tools.vision.model", return_value=None):
            self.assertTrue(orch.abilities()["makes video"])
        with patch.object(adapters, "catalogue", return_value=[]), \
                patch.object(adapters, "video_installable", return_value=False), \
                patch("mlmi.backend.tools.vision.has_tower", return_value=False), \
                patch("mlmi.backend.tools.vision.model", return_value=None):
            self.assertFalse(orch.abilities()["makes video"])

    def test_attached_and_open_studio_images_are_selected(self):
        attached = self.image("attachments/attached.jpg")
        imported = self.image("media/imported.png")
        project = self.ws.path("selected.sstudio")
        project.write_text(json.dumps({"layers": [{"kind": "image", "src": "media/imported.png"}]}),
                           encoding="utf-8")
        selected = agent._selected_video_images(self.ws, ["attachments/attached.jpg"], "selected.sstudio",
                                                "animate selected.png in this video")
        self.assertEqual(selected, ["attachments/attached.jpg", "media/imported.png"])
        args = {}
        agent._video_call_sources(self.ws, args, selected, "animate these photos")
        self.assertEqual(args["source_images"], selected)
        self.assertTrue(attached.is_file() and imported.is_file())

    def test_explicit_text_only_overrides_attached_photos(self):
        self.image("attachments/attached.jpg")
        args = {}
        agent._video_call_sources(self.ws, args, ["attachments/attached.jpg"],
                                  "make a text-only video without using the attached photo")
        self.assertEqual(args["source_images"], [])

    def test_photo_animation_without_a_selected_photo_does_not_make_text_video(self):
        with self.assertRaisesRegex(ToolError, "needs a photo for image-guided video"):
            agent._video_call_sources(self.ws, {}, [], "animate this attached photo into a video")
        args = {}
        agent._video_call_sources(self.ws, args, [],
                                  "animate this attached photo into a text-only video without using it")
        self.assertEqual(args["source_images"], [])

    def test_missing_feature_models_wait_for_check_and_use_without_downloading(self):
        image_ctx = T.Ctx(self.ws)
        photo_ctx = T.Ctx(self.ws)
        self.image("reference.png")
        with patch.dict(media.CONFIG, {"disabled_media_models": []}), \
                patch.object(adapters, "image_ready", return_value=False), \
                patch.object(adapters, "video_ready", return_value=False), \
                patch.object(adapters, "video_reference_ready", return_value=False), \
                patch("mlmi.backend.adapters.install.ensure") as ensure, \
                patch("shutil.which", return_value="ffmpeg"):
            with self.assertRaises(ModelSelectionRequired) as image_error:
                media.make_image(image_ctx, "a cat")
            self.assertEqual((image_error.exception.slot, image_error.exception.repo_id),
                             ("image", "stable-diffusion-v1-5/stable-diffusion-v1-5"))
            with self.assertRaises(ModelSelectionRequired) as video_error:
                media.make_video(photo_ctx, "a cat")
            self.assertEqual((video_error.exception.slot, video_error.exception.repo_id),
                             ("video", "ByteDance/AnimateDiff-Lightning"))
            ensure.assert_not_called()

    def test_photo_guidance_model_is_required_before_selected_photos_are_used(self):
        self.image("reference.png")
        with patch("shutil.which", return_value="ffmpeg"), \
                patch.object(adapters, "video_ready", return_value=True), \
                patch.object(adapters, "video_reference_ready", return_value=False), \
                patch("mlmi.backend.adapters.install.ensure") as ensure:
            with self.assertRaises(ModelSelectionRequired) as caught:
                media.make_video(T.Ctx(self.ws), "animate the selected photo", source_images=["reference.png"])
        self.assertEqual((caught.exception.slot, caught.exception.repo_id),
                         ("photo_guidance", "h94/IP-Adapter"))
        ensure.assert_not_called()

    def test_missing_speech_input_model_waits_for_explicit_use(self):
        audio = self.ws.path("voice.wav")
        audio.write_bytes(b"local audio placeholder")
        with patch.dict(media.CONFIG, {"disabled_media_models": []}), \
                patch.object(adapters, "whisper_snapshot", return_value=None), \
                patch("mlmi.backend.adapters.install.ensure") as ensure:
            with self.assertRaises(ModelSelectionRequired) as caught:
                media.transcribe_audio(T.Ctx(self.ws), "voice.wav")
        self.assertEqual((caught.exception.slot, caught.exception.repo_id),
                         ("speech_input", "Systran/faster-whisper-small"))
        ensure.assert_not_called()

    def test_unselected_workspace_photo_cannot_be_used_by_agent(self):
        self.image("private/photo.png")
        with self.assertRaisesRegex(ToolError, "was not attached, selected"):
            agent._video_call_sources(self.ws, {"source_images": ["private/photo.png"]}, [], "animate a photo")

    def test_invalid_or_inaccessible_source_images_fail_before_install(self):
        text = self.ws.path("notes.txt")
        text.write_text("not an image", encoding="utf-8")
        with self.assertRaisesRegex(ToolError, "not a supported picture"):
            media._video_sources(T.Ctx(self.ws), ["notes.txt"])
        with self.assertRaises(ToolError):
            media._video_sources(T.Ctx(self.ws), ["../outside.png"])
        broken = self.ws.path("attachments/broken.png")
        broken.parent.mkdir(parents=True)
        broken.write_bytes(b"not a real png")
        with self.assertRaisesRegex(ToolError, "cannot read the selected picture"):
            media._video_sources(T.Ctx(self.ws), ["attachments/broken.png"])

    def test_tool_selects_photo_worker_and_keeps_text_worker_unchanged(self):
        photo = self.image("attachments/reference.png")
        ctx = T.Ctx(self.ws)
        requests = {"text": [], "photo": []}

        def finish_mp4(command, **kwargs):
            Path(command[-1]).write_bytes(b"mp4")
            return types.SimpleNamespace(returncode=0, stderr="")

        def run_case(images):
            target = media.REFERENCE_VIDEOS if images else media.VIDEOS
            key = "photo" if images else "text"
            def make(req, timeout, **kwargs):
                requests[key].append(req)
                return {"ok": True, "frames": 8, "width": 512, "height": 512, "seconds": 1.0}
            with patch("shutil.which", return_value="ffmpeg"), \
                    patch("mlmi.backend.adapters.install.ensure"), \
                    patch.object(adapters, "video_ready", return_value=True), \
                    patch.object(adapters, "video_reference_ready", return_value=True), \
                    patch.object(media.VIDEOS, "make", side_effect=make if target is media.VIDEOS else AssertionError("text worker used")), \
                    patch.object(media.REFERENCE_VIDEOS, "make", side_effect=make if target is media.REFERENCE_VIDEOS else AssertionError("photo worker used")), \
                    patch("mlmi.backend.tools.studio._encoder", return_value=([], "libx264")), \
                    patch("mlmi.backend.tools.media.subprocess.run", side_effect=finish_mp4), \
                    patch.object(media, "_speed_saved"):
                return media.make_video(ctx, "a person waving", path=f"result-{key}.mp4", source_images=images)

        run_case(["attachments/reference.png"])
        run_case([])
        self.assertEqual(requests["photo"][0]["source_images"], [str(photo)])
        self.assertNotIn("source_images", requests["text"][0])

    def test_photo_gpu_memory_failure_does_not_retry_as_text_only(self):
        self.image("reference.png")
        errors = ("CUDA out of memory", "Could not allocate tensor with 2147483648 bytes. "
                  "There is not enough GPU video memory available!")
        for error in errors:
            with self.subTest(error=error), \
                    patch("shutil.which", return_value="ffmpeg"), \
                    patch("mlmi.backend.adapters.install.ensure"), \
                    patch.object(adapters, "video_ready", return_value=True), \
                    patch.object(adapters, "video_reference_ready", return_value=True), \
                    patch.object(media.REFERENCE_VIDEOS, "make", return_value={"ok": False, "error": error}), \
                    patch.object(media.VIDEOS, "make", side_effect=AssertionError("must not retry without photos")):
                with self.assertRaisesRegex(ToolError, "photos were not ignored"):
                    media.make_video(T.Ctx(self.ws), "animate this", path="oom.mp4", source_images=["reference.png"])

    def test_cuda_and_directml_memory_errors_are_classified(self):
        self.assertTrue(media._video_memory_error("CUDA out of memory"))
        self.assertTrue(media._video_memory_error(
            "Could not allocate tensor with 2147483648 bytes. There is not enough GPU video memory available!"))
        self.assertFalse(media._video_memory_error("the image encoder model is missing"))

    def test_text_video_gpu_memory_error_names_a_workable_device_choice(self):
        with patch("shutil.which", return_value="ffmpeg"), \
                patch("mlmi.backend.adapters.install.ensure"), \
                patch.object(adapters, "video_ready", return_value=True), \
                patch.object(media.VIDEOS, "make", return_value={"ok": False,
                                                                   "error": "Could not allocate tensor; not enough GPU video memory"}):
            with self.assertRaisesRegex(ToolError, "choose CPU or a GPU with more video memory"):
                media.make_video(T.Ctx(self.ws), "make a video", path="text-oom.mp4")

    def test_worker_stop_reaps_child_and_closes_pipes(self):
        class Stream:
            closed = False
            def close(self): self.closed = True

        worker = media._Worker("test", lambda _d: [])
        process = MagicMock()
        process.poll.return_value = None
        process.stdin, process.stdout, process.stderr = Stream(), Stream(), Stream()
        worker.proc = process
        worker.stop()
        self.assertIsNone(worker.proc)
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)
        self.assertTrue(all(stream.closed for stream in (process.stdin, process.stdout, process.stderr)))

    def test_worker_sends_selected_photo_as_ip_adapter_image(self):
        photo = self.image("reference.png")
        out = self.root / "frames"

        class Generator:
            def __init__(self, device):
                self.device = device
            def manual_seed(self, seed):
                return self

        Torch = types.SimpleNamespace(Generator=Generator)

        class Pipe:
            def __init__(self):
                self.kwargs = None
                self.scale = None
            def set_ip_adapter_scale(self, scale):
                self.scale = scale
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return types.SimpleNamespace(frames=[[Image.new("RGB", (8, 8))]])

        pipe = Pipe()
        result = image_worker.video(pipe, Torch, {"prompt": "wave", "out": str(out), "frames": 8,
                                                  "source_images": [str(photo)]})
        self.assertTrue((out / "frame_000.png").is_file())
        self.assertEqual(result["frames"], 1)
        self.assertEqual(pipe.kwargs["ip_adapter_image"].size, (32, 24))
        self.assertEqual(pipe.scale, 0.65)

    def test_worker_no_photo_call_does_not_add_adapter_input(self):
        out = self.root / "frames-text"

        class Generator:
            def __init__(self, device): pass
            def manual_seed(self, seed): return self
        Torch = types.SimpleNamespace(Generator=Generator)
        class Pipe:
            def __init__(self): self.kwargs = None
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return types.SimpleNamespace(frames=[[Image.new("RGB", (8, 8))]])

        pipe = Pipe()
        image_worker.video(pipe, Torch, {"prompt": "wave", "out": str(out), "frames": 8})
        self.assertNotIn("ip_adapter_image", pipe.kwargs)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are not installed")
    def test_studio_montage_keeps_source_layers_and_renders_probeable_mp4(self):
        self.image("media/first.png", (220, 30, 30))
        self.image("media/second.png", (30, 220, 30))
        ctx = T.Ctx(self.ws)
        studio.studio_create(ctx, "photo-montage.sstudio", duration=2)
        studio.studio_add_media(ctx, "photo-montage.sstudio", "media/first.png", start=0, duration=1)
        studio.studio_add_media(ctx, "photo-montage.sstudio", "media/second.png", start=1, duration=1)
        project = json.loads(self.ws.path("photo-montage.sstudio").read_text(encoding="utf-8"))
        self.assertEqual([layer["src"] for layer in project["layers"]], ["media/first.png", "media/second.png"])
        studio.studio_render(ctx, "photo-montage.sstudio", "photo-montage.mp4", preview=True)
        probe = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_entries",
                                "format=duration:stream=codec_type,width,height", "-of", "json",
                                str(self.ws.path("photo-montage.mp4"))], capture_output=True, text=True, check=True)
        details = json.loads(probe.stdout)
        self.assertTrue(any(stream.get("codec_type") == "video" for stream in details["streams"]))
        self.assertGreaterEqual(float(details["format"]["duration"]), 1.9)

    @unittest.skipUnless(os.environ.get("SHEEP_RUN_REAL_VIDEO") == "1",
                         "set SHEEP_RUN_REAL_VIDEO=1 to run local model inference")
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are not installed")
    def test_real_photo_guided_video_smoke(self):
        if not adapters.video_ready() or not adapters.video_reference_ready():
            self.skipTest("local AnimateDiff and IP-Adapter weights are not installed")
        previous = media.CONFIG.get("media_device")
        media.CONFIG["media_device"] = os.environ.get("SHEEP_VIDEO_DEVICE", "auto")
        photo = self.image("attachments/local-reference.png", (40, 120, 210))
        started = time.monotonic()
        try:
            result = media.make_video(T.Ctx(self.ws), "A gentle camera move around this blue object",
                                      seconds=1, path="local-photo-video.mp4",
                                      source_images=["attachments/local-reference.png"])
        finally:
            media.REFERENCE_VIDEOS.stop()
            media.VIDEOS.stop()
            if previous is None:
                media.CONFIG.pop("media_device", None)
            else:
                media.CONFIG["media_device"] = previous
        output = self.ws.path("local-photo-video.mp4", must_exist=True)
        probe = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_entries",
                                "format=duration:stream=codec_type,width,height", "-of", "json", str(output)],
                               capture_output=True, text=True, check=True)
        details = json.loads(probe.stdout)
        self.assertTrue(any(stream.get("codec_type") == "video" for stream in details["streams"]))
        self.assertGreaterEqual(float(details["format"]["duration"]), 0.9)
        self.assertIn("Photo guidance used 1 selected photo", result)
        print(f"\nLocal photo video smoke: {time.monotonic() - started:.1f}s; {result}")

    @unittest.skipUnless(os.environ.get("SHEEP_RUN_TEXT_VIDEO") == "1",
                         "set SHEEP_RUN_TEXT_VIDEO=1 to run text-only local inference")
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are not installed")
    def test_real_text_video_smoke(self):
        if not adapters.video_ready():
            self.skipTest("local AnimateDiff weights are not installed")
        previous = media.CONFIG.get("media_device")
        media.CONFIG["media_device"] = os.environ.get("SHEEP_VIDEO_DEVICE", "auto")
        started = time.monotonic()
        try:
            result = media.make_video(T.Ctx(self.ws), "A small red ball rolls across a clean white table",
                                      seconds=1, path="local-text-video.mp4")
        finally:
            media.VIDEOS.stop()
            if previous is None:
                media.CONFIG.pop("media_device", None)
            else:
                media.CONFIG["media_device"] = previous
        output = self.ws.path("local-text-video.mp4", must_exist=True)
        probe = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_entries",
                                "format=duration:stream=codec_type,width,height", "-of", "json", str(output)],
                               capture_output=True, text=True, check=True)
        details = json.loads(probe.stdout)
        self.assertTrue(any(stream.get("codec_type") == "video" for stream in details["streams"]))
        self.assertGreaterEqual(float(details["format"]["duration"]), 0.9)
        self.assertNotIn("Photo guidance used", result)
        print(f"\nLocal text video smoke: {time.monotonic() - started:.1f}s; {result}")

    @unittest.skipUnless(os.environ.get("SHEEP_RUN_DIRECTML_VIDEO") == "1",
                         "set SHEEP_RUN_DIRECTML_VIDEO=1 to attempt DirectML inference")
    def test_real_directml_photo_guided_video_smoke(self):
        if not adapters.video_ready() or not adapters.video_reference_ready():
            self.skipTest("local AnimateDiff and IP-Adapter weights are not installed")
        from mlmi.backend import hardware
        amd = hardware.amd_adapter()
        if not amd or amd.get("missing") or not hardware.DML_PY.exists():
            self.skipTest("no working local DirectML runtime or adapter was detected")
        from mlmi.backend.adapters import MEDIA_DIR
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            self.skipTest("ffmpeg and ffprobe are required")
        photo = self.image("attachments/directml-reference.png", (40, 120, 210))
        frames = self.root / "directml-frames"
        command = [str(hardware.DML_PY), str(media.HERE / "adapters" / "image_worker.py"),
                   "--device", f"dml:{amd['index']}", "--base", str(MEDIA_DIR / "sd15"),
                   "--motion", str(MEDIA_DIR / "animatediff" / media.MOTION_FILE),
                   "--ip-adapter", str(MEDIA_DIR / "ip-adapter")]
        request = {"prompt": "A gentle camera move around this blue object", "out": str(frames), "frames": 8,
                   "width": 512, "height": 512, "source_images": [str(photo)]}
        started = time.monotonic()
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", creationflags=media.NOWIN)
        try:
            stdout, stderr = process.communicate(json.dumps(request) + "\n", timeout=900)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            self.fail("DirectML photo-guided video inference exceeded 15 minutes")
        lines = [line for line in stdout.splitlines() if line.strip()]
        self.assertTrue(lines, f"DirectML worker returned no status: {stderr[-500:]}")
        hello = json.loads(lines[0])
        if not hello.get("ok"):
            reason = str(hello.get("error") or "DirectML model load failed")
            if any(word in reason.lower() for word in ("out of memory", "not enough memory", "allocation")):
                self.skipTest(f"DirectML GPU could not fit the video model: {reason[:220]}")
            self.fail(f"DirectML video worker failed to load: {reason}")
        self.assertGreaterEqual(len(lines), 2, f"DirectML worker returned no generation result: {stderr[-500:]}")
        result = json.loads(lines[1])
        if not result.get("ok") and any(word in str(result.get("error", "")).lower()
                                         for word in ("out of memory", "not enough memory", "not enough gpu video memory",
                                                      "could not allocate tensor", "allocation")):
            self.skipTest(f"DirectML GPU could not fit photo-guided inference: {result['error'][:220]}")
        self.assertTrue(result.get("ok"), f"DirectML inference failed: {result.get('error', stderr[-500:])}")
        output = self.ws.path("local-directml-photo-video.mp4")
        encoded = subprocess.run([ffmpeg, "-y", "-v", "error", "-framerate", str(media.VIDEO_FPS), "-i",
                                  str(frames / "frame_%03d.png"), "-pix_fmt", "yuv420p", str(output)],
                                 capture_output=True, text=True, timeout=300)
        self.assertEqual(encoded.returncode, 0, encoded.stderr[-500:])
        self.assertTrue(output.is_file())
        probe = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                                "format=duration:stream=codec_type", "-of", "json", str(output)],
                               capture_output=True, text=True, check=True)
        self.assertTrue(any(stream.get("codec_type") == "video" for stream in json.loads(probe.stdout)["streams"]))
        print(f"\nDirectML photo video smoke: {time.monotonic() - started:.1f}s; {result}")


if __name__ == "__main__":
    unittest.main()
