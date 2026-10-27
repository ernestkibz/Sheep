"""Orientation for media (2026-10-14): every picture, video and speech task must route to
its tool, and each grader must pass only on a real file the job made."""
import pytest

from mlmi.backend import orient as O
from test_cues import media_tool

WANT = {"picture": "make_image", "video": "make_video", "speech": "speak_text"}


@pytest.mark.parametrize("area", list(WANT))
def test_every_media_prompt_routes_to_its_tool(area):
    for t in O.tasks(area, 24):
        text = t.turns[0][0]
        got = media_tool(text)
        assert got == WANT[area], (text, got)


def test_picture_grader_needs_a_real_picture(tmp_path):
    t = O.tasks("picture", 1)[0]
    res = {"files": []}
    assert t.grade(tmp_path, res)[0] is False
    (tmp_path / "pic.png").write_bytes(b"\x89PNG" + b"0" * 5000)
    assert t.grade(tmp_path, {"files": ["pic.png"]})[0] is True


def test_video_grader_rejects_an_empty_file(tmp_path):
    t = O.tasks("video", 1)[0]
    (tmp_path / "clip.mp4").write_bytes(b"0" * 100)            # too small to be a video
    assert t.grade(tmp_path, {"files": ["clip.mp4"]})[0] is False
    (tmp_path / "clip.mp4").write_bytes(b"0" * 20_000)
    assert t.grade(tmp_path, {"files": ["clip.mp4"]})[0] is True


def test_missing_media_model_counts_when_sheep_shows_the_checklist(tmp_path):
    # no picture model on this computer: the right answer is the setup checklist, not a failure
    t = O.tasks("picture", 1)[0]
    res = {"files": [], "texts": ["Sheep is not set up for this yet. For picture work it needs: ..."]}
    ok, why = t.grade(tmp_path, res)
    assert ok and "checklist" in why
    assert t.grade(tmp_path, {"files": [], "texts": ["I can't make images."]})[0] is False


def test_speech_grader_needs_audio(tmp_path):
    t = O.tasks("speech", 1)[0]
    assert t.grade(tmp_path, {"files": []})[0] is False
    (tmp_path / "voice.wav").write_bytes(b"RIFF" + b"0" * 2000)
    assert t.grade(tmp_path, {"files": ["voice.wav"]})[0] is True


def test_quick_orientation_covers_media():
    areas = {t.area for t in O.quick()}
    assert {"picture", "video", "speech"} <= areas
