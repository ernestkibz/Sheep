"""Typo-tolerant routing (2026-10-14): a misspelt or split request must reach the same
route as the correct spelling. Each case is (typo text, the route it must reach)."""
import re

import pytest

from mlmi.backend import agent, cues, direct


def media_tool(text):
    """The tool the MEDIA_NEEDS cues send this request to, the way agent.run reads them."""
    t = cues.fix(text)
    return next((tool for pat, tool, _ in agent.MEDIA_NEEDS if re.search(pat, t, re.I)), None)


PICTURE = [
    "generat e an image of a a dog",
    "generte a picutre of a cat",
    "gnerate an imge of a tree",
    "mak a picture of a boat",
    "draw me a pitcure of a horse",
    "creat a logoo for Sheep",
    "produc an illustraton of a farm",
]
VIDEO = [
    "mak a vidio of a sheep dancing",
    "generat e a vidoe of a river",
    "anmate this photo",
    "animte the picutre into a clip",
]
SPEECH = [
    "transcibe this recrding",
    "trancribe the meeting",
    "narate it with a voise",
    "make a recrding of this text",
]


@pytest.mark.parametrize("text", PICTURE)
def test_picture_typos_reach_make_image(text):
    assert media_tool(text) == "make_image", text


@pytest.mark.parametrize("text", VIDEO)
def test_video_typos_reach_video(text):
    t = cues.fix(text)
    assert media_tool(text) == "make_video" or agent.explicit_media_request(t), text


@pytest.mark.parametrize("text", SPEECH)
def test_speech_typos_reach_speech(text):
    t = cues.fix(text)
    assert media_tool(text) in ("transcribe_audio", "speak_text") or agent.explicit_transcription_request(t), text


def test_correct_spelling_still_routes():
    assert media_tool("generate an image of a dog") == "make_image"
    assert media_tool("transcribe this recording") == "transcribe_audio"


def test_split_word_joined_back():
    assert cues.fix("generat e an image") == "generate an image"


def test_file_names_are_not_changed():
    # the direct command reads its file name exactly; routing text may correct words, never a name
    assert cues.fix("export budget_2026.xlsx to pdf").endswith("budget_2026.xlsx to pdf")
    assert cues.fix("open Ruth.docx") == "open Ruth.docx"


@pytest.mark.parametrize("text", [
    "tips for remote work",                     # remote -> remove would read as a delete
    "how does the heart pump blood",            # heart -> chart
    "malaria prevention tips",                  # prevention -> presentation
    "what is a good test for this",             # test -> latest would trigger a web search
    "a red barn at the market",
    "waves rolling onto a beach in the rain",
    "make a slideshow of my photos",
])
def test_real_words_are_never_corrected(text):
    assert cues.fix(text) == text


@pytest.mark.parametrize("text", [
    "open note.txt", "edit budget.ssheet", "show plan.sdeck", "rename my_sheet to v2-draft", "open Ruth.docx",
])
def test_file_names_stay_exact(text):
    assert cues.fix(text) == text


def test_correct_orientation_prompts_are_unchanged():
    # 1,200 correctly spelt requests: the first version changed 408 words in them
    from mlmi.backend import orient as O
    changed = [t.turns[k][0] for a in O.AREAS for t in O.tasks(a, 100) for k in range(len(t.turns))
               if cues.fix(t.turns[k][0]) != t.turns[k][0]]
    assert not changed, changed[:5]


def test_common_short_words_are_left_alone():
    assert cues.fix("the cat is on the mat and it was with you") == "the cat is on the mat and it was with you"


@pytest.mark.parametrize("text,area_pattern", [
    ("lsit the fiels in the workspace", agent.LIST_ASK),
    ("chrt the sheet data", r"\b(chart|graph|plot)\b"),
    ("creat a presentaton about sheep", r"\b(presentation|deck|slides?)\b"),
    ("sumarize the docment", r"\b(summari[sz]e|summary)\b"),
])
def test_other_areas_typos_reach_their_cue(text, area_pattern):
    assert re.search(area_pattern, cues.fix(text), re.I), text


def test_vocabulary_comes_from_the_patterns():
    v = cues.vocab()
    for word in ("picture", "transcribe", "animate", "presentation"):
        assert word in v, word


@pytest.mark.parametrize("text,cant", [
    ("I don't have the capability to generate images. However, you can search online.", True),
    ("I can't create a video for you.", True),
    ("As an AI language model, I cannot draw.", True),
    ("I'm only a text-based model.", True),
    ("I made the picture: dog.png", False),
    ("I can help with that.", False),
    ("I am able to make that.", False),
])
def test_a_reply_that_says_it_cannot_is_recognised(text, cant):
    # the dog picture, 2026-10-14: the model said it could not; Sheep could, so it is asked again
    assert bool(re.search(agent.CANT, text, re.I)) is cant
