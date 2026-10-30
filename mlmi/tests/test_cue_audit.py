"""The cue audit (2026-10-14): every kind of request through Sheep's own routing, no model (2026-10-14).

For each prompt: the lane run() would take before the model speaks (a direct command, an
ability question, a document/deck/sheet job, a plain answer, a Studio plan, a transcription,
an AI-video/montage plan, a photo edit, or the model with tools) and the tools it is offered.
A misfire is a lane or a tool set that differs from what a person meant."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mlmi.backend import agent as A, cues, direct, shepherd, tools as T  # noqa: E402

FILES = {"budget.ssheet", "trip-plan.sdoc", "save-water.sdeck", "intro.sstudio", "logo.png", "photo.jpg",
         "meeting.wav", "notes.txt", "report.docx", "sales.xlsx", "deck.pptx", "clip.mp4", "song.mp3", "main.py"}
ALLOWED = set(T.offered(3, ["files", "web", "media", "mail", "words", "sheets", "slides", "studio"]))


def lane(asked, app=""):
    cue = cues.fix(asked)
    exists = lambda p: p in FILES
    tool_app = "studio" if A.explicit_media_request(cue) else app
    names = A.route_tools(ALLOWED, tool_app if tool_app != "files" else "", cue)
    job = shepherd.intent(asked, app, "", [], "", exists) if not A.explicit_media_request(cue) else None
    if not job and cue != asked and not A.explicit_media_request(cue):
        job = shepherd.intent(cue, app, "", [], "", exists)
    if not app and A.ability_question(cue):
        return "answer(ability)", []
    if not A.wants_tools(cue, app) and not job:
        return "answer", []
    if job:
        return f"job:{job['kind']}:{job['action']}", names
    if shepherd.studio_plan(asked) and app in ("", "studio") and not A.is_ai_video(cue) \
            and not re.search(A.PHOTO_MONTAGE, cue, re.I):
        return "studio", names
    if A.explicit_transcription_request(cue):
        return "transcribe", names
    if A.explicit_media_request(cue):
        return "media", names
    pic = re.search(r"(?<![\w./\\-])([\w./\\-]+\.(?:png|jpe?g|webp|bmp|gif))\b", asked, re.I)
    if pic and A.image_ops_in(cue) and exists(pic.group(1)) and "edit_image" in names \
            and not re.search(r"\b(slide|deck|document|doc|report|video|montage)\b", asked, re.I):
        return "edit_image", names
    if direct.parse(asked):
        return "direct:" + direct.parse(asked)[0], names
    return "tools", names


# (prompt, lane it must take (prefix, or a tuple of allowed prefixes), tools that must be offered, tools that must not)
CASES = [
    # ---- plain questions and chat: words, no tools
    ("What is the capital of Uganda?", "answer", [], []),
    ("Why do leaves change colour in autumn?", "answer", [], []),
    ("How do I make a good cup of tea?", "answer", [], []),
    ("How do I draw a dog?", "answer", [], []),
    ("Explain how photosynthesis works", "answer", [], []),
    ("Tell me a joke", "answer", [], []),
    ("Hello", "answer", [], []),
    ("Thanks, that was great!", "answer", [], []),
    ("I need someone to talk to like a friend", "answer", [], []),
    ("I need someone to talk to like a friend, and make a budget sheet", "job:sheets:create", [], []),
    ("Who wrote Things Fall Apart?", "answer", [], []),
    ("What should I write in a birthday card for my mum?", "answer", [], []),
    ("Is it safe to eat raw eggs?", "answer", [], []),
    ("Translate 'good morning' into Luganda", ("answer", "tools"), [], ["words_create", "slides_create"]),
    ("Write a short poem about the rain", ("answer", "tools"), [], ["slides_create", "sheets_create"]),
    ("Give me five ideas for a bakery name", ("answer", "tools"), [], ["slides_create", "sheets_create"]),
    ("Summarize the history of Kenya in three sentences", ("answer", "tools"), [], ["slides_create", "sheets_create"]),
    ("What does photosynthesis mean?", "answer", [], []),
    ("Can you make spreadsheets?", "answer(ability)", [], []),
    ("Can you edit videos?", "answer(ability)", [], []),
    ("What can you do?", "answer(ability)", [], []),
    ("Are you able to make pictures?", "answer(ability)", [], []),
    ("Write a python function that reverses a string", "answer", [], []),
    ("How do I sort a list in javascript?", "answer", [], []),
    # ---- documents
    ("Write a report on climate change in Uganda", "job:words", [], []),
    ("Make a Word document about our trip to Mombasa", "job:words", [], []),
    ("Create a document with my meeting notes", "job:words", [], []),
    ("Draft a letter to my landlord about the broken tap", ("job:words", "tools", "answer"), [], ["make_image"]),
    ("Add a section about costs to trip-plan.sdoc", "job:words:extend", [], []),
    ("In trip-plan.sdoc replace the word Nairobi with Mombasa", ("direct", "tools"), [], []),
    ("Export trip-plan.sdoc as a PDF", ("direct", "tools", "job"), [], []),
    ("Summarize trip-plan.sdoc", ("tools", "answer"), [], ["make_image", "make_video"]),
    ("Turn report.docx into a presentation", ("job:slides", "tools"), [], []),
    # ---- slides
    ("Make a PowerPoint about saving water", "job:slides:create", [], []),
    ("Create a presentation on road safety with 5 slides", "job:slides:create", [], []),
    ("Make slides about the solar system with pictures", "job:slides:create", [], []),
    ("Add two slides about rainwater tanks to save-water.sdeck", "job:slides:extend", [], []),
    ("Give save-water.sdeck a dark theme", ("direct", "tools"), [], []),
    ("Rename slide 2 of save-water.sdeck to Why it matters", ("tools", "direct"), ["slides_set"], []),
    ("Add speaker notes to slide 3 of save-water.sdeck", "tools", ["slides_set"], []),
    ("Export save-water.sdeck to PowerPoint", ("direct", "tools"), [], []),
    # ---- sheets
    ("Make a budget spreadsheet for a wedding", "job:sheets:create", [], []),
    ("Create an Excel sheet of my monthly expenses", "job:sheets:create", [], []),
    ("Add a row for transport, 20000 to budget.ssheet", ("job:sheets", "tools"), [], []),
    ("Sort budget.ssheet by cost", ("tools", "direct"), [], ["make_image"]),
    ("Add a bar chart of costs to budget.ssheet", ("tools", "direct"), [], ["make_image"]),
    ("Export budget.ssheet to Excel", ("direct", "tools"), [], []),
    ("What is the total in budget.ssheet?", ("tools", "answer"), [], ["make_image"]),
    # ---- pictures
    ("Draw me a picture of a dog playing in the snow", "tools", ["make_image"], ["slides_add_image", "make_video"]),
    ("Make a logo for my bakery", ("tools", "studio"), ["make_image"], []),
    ("Generate an image of a sunset over Lake Victoria", "tools", ["make_image"], ["make_video"]),
    ("Create a poster for a church fundraiser", "tools", ["make_image"], []),
    ("Paint a sheep in a green field", "tools", ["make_image"], []),
    ("What is in this picture? photo.jpg", "tools", ["see_picture"], ["make_image"]),
    ("Describe photo.jpg", "tools", ["see_picture"], ["make_image"]),
    # ---- photo edits
    ("Make logo.png black and white", "edit_image", ["edit_image"], ["slides_add_image"]),
    ("Edit logo.png to make it black and white", "edit_image", ["edit_image"], ["slides_add_image"]),
    ("Rotate photo.jpg 90 degrees", "edit_image", ["edit_image"], []),
    ("Crop photo.jpg to 10,10,80,80", "edit_image", ["edit_image"], []),
    ("Make photo.jpg brighter", "edit_image", ["edit_image"], []),
    ("Resize logo.png to 400", "edit_image", ["edit_image"], []),
    ("Flip photo.jpg upside down", "edit_image", ["edit_image"], []),
    ("Put logo.png on slide 2 of save-water.sdeck", "tools", ["slides_add_image"], ["edit_image"]),
    ("Add photo.jpg to trip-plan.sdoc", "tools", ["words_add_image"], ["edit_image"]),
    # ---- voice and sound
    ("Read this sentence out loud in a voice: Welcome to the bakery.", "tools", ["speak_text"], []),
    ("Make a voice over saying welcome to our shop", "tools", ["speak_text"], []),
    ("Say hello in a voice", "tools", ["speak_text"], []),
    ("Transcribe meeting.wav", "transcribe", ["transcribe_audio"], []),
    ("What was said in meeting.wav?", ("tools", "transcribe"), ["transcribe_audio"], []),
    ("Write down what is said in song.mp3", ("transcribe", "tools"), ["transcribe_audio"], []),
    # ---- video: AI scenes vs Studio
    ("Make a video of a dog running on a beach", "media", ["make_video"], []),
    ("Generate a 4-second clip of waves rolling onto a beach", "media", ["make_video"], []),
    ("Animate photo.jpg", "media", ["make_video"], []),
    ("Make a short video ad for my bakery", "studio", [], []),
    ("Make an intro video with the title Sheep", "studio", [], ["make_video"]),
    ("Create a 9:16 short that says Grand Opening", "studio", [], []),
    ("Make a logo reveal for Sheep", "studio", [], []),
    ("Make a slideshow of my photos", ("media", "tools"), [], []),
    ("Render intro.sstudio", "direct", [], []),
    ("Add a crossfade between the clips in intro.sstudio", "tools", ["studio_transition"], []),
    ("Slow down the first clip in intro.sstudio", "tools", ["studio_set"], []),
    ("Add captions to intro.sstudio", "tools", ["studio_transcribe"], []),
    ("Put the words on the beat in intro.sstudio", "tools", ["studio_listen"], []),
    ("Add background music song.mp3 to intro.sstudio", "tools", ["studio_add_media"], []),
    # ---- web
    ("What is the weather in Kampala today?", "tools", ["web_search"], []),
    ("Search the web for the latest news on Uganda", "tools", ["web_search"], []),
    ("What is the exchange rate of the dollar to shilling right now?", "tools", ["web_search"], []),
    ("Who won the match last night?", ("tools", "answer"), [], []),
    # ---- mail and calendar
    ("Check my email", "tools", ["mail_list"], []),
    ("Do I have unread emails?", "tools", ["mail_list"], []),
    ("Send an email to John about the meeting", "tools", ["mail_draft"], []),
    ("Add a calendar event called Dentist on 2026-11-03 at 10:00", ("direct", "tools"), [], []),
    ("Remind me to pay rent on Friday", "tools", ["calendar_add"], []),
    ("What is on my calendar this week?", "tools", ["calendar_list"], []),
    # ---- files
    ("What files do I have?", "tools", ["list_files"], []),
    ("Show me my documents", "tools", ["list_files"], []),
    ("Delete notes.txt", ("direct", "tools"), [], []),
    ("Copy notes.txt to backup.txt", ("direct", "tools"), [], []),
    ("Create a folder called Invoices", "tools", [], []),
    ("Open notes.txt", "tools", ["read_file"], []),
    ("Run main.py", ("tools", "answer"), [], []),
    # ---- typos
    ("mak a powerpont about water", "job:slides", [], []),
    ("draw me a pitcure of a horse", "tools", ["make_image"], []),
    ("trancribe meeting.wav", "transcribe", [], []),
    ("make logo.png blak and white", "edit_image", ["edit_image"], []),
]


CASES += [
    # how-to and what-is questions that name a kind of file: still words
    ("How do I make a chart in Excel?", "answer", [], []),
    ("How can I add a slide in PowerPoint?", "answer", [], []),
    ("What is a spreadsheet?", "answer", [], []),
    ("What is the best way to write a CV?", "answer", [], []),
    ("How to convert a Word document to PDF?", "answer", [], []),
    ("Remind me what photosynthesis is", "answer", [], []),
    ("Read me a story about a lion", ("answer", "tools"), [], ["speak_text", "make_image"]),
    ("Write me a story about a lion", ("answer", "job:words"), [], []),
    ("Give me a summary of the French revolution", "answer", [], []),
    ("Habari yako?", "answer", [], []),
    ("I have an event tomorrow, what should I wear?", "answer", [], []),
    # asks phrased as wishes
    ("I want a picture of a lion", "tools", ["make_image"], []),
    ("Show me a picture of a lion in the savanna", "tools", ["make_image"], []),
    ("I need a spreadsheet for my shop's sales", "job:sheets", [], []),
    ("Can you make me a budget sheet?", "job:sheets", [], []),
    ("Could you write a report on malaria in Uganda?", "job:words", [], []),
    ("Turn this text into speech: hello and welcome", "tools", ["speak_text"], []),
    ("Remind me to call Sarah tomorrow at 9", "tools", ["calendar_add"], []),
    ("Set a reminder for my dentist appointment", "tools", ["calendar_add"], []),
    ("Write an email to my boss asking for leave", "tools", ["mail_draft"], []),
    # precise edits: the tool loop, not new writing
    ("Fix the typos in trip-plan.sdoc", "tools", [], []),
    ("Make the title bigger in save-water.sdeck", ("direct", "tools"), [], []),
    ("Delete slide 3 from save-water.sdeck", "tools", ["slides_delete"], []),
    ("Change the theme of save-water.sdeck to forest", ("tools", "direct"), [], []),
    ("Rename budget.ssheet to wedding.ssheet", ("tools", "direct"), [], []),
    ("Convert report.docx to PDF", ("tools", "direct"), [], []),
    ("Tell me about photo.jpg", "tools", ["see_picture"], []),
    ("Make a video about my business", ("studio", "media"), [], []),
    ("Make a video from my photos", ("media", "studio", "tools"), [], []),
    ("Add a title that says Welcome to intro.sstudio", "tools", ["studio_add_text"], []),
    ("Make the second clip in intro.sstudio twice as fast", "tools", ["studio_set"], []),
    ("Add a wipe transition to intro.sstudio", "tools", ["studio_transition"], []),
]
CASES += [
    ("What are my files?", "tools", ["list_files"], []),
    ("What is in my inbox?", "tools", ["mail_list"], []),
    ("What's on my calendar tomorrow?", "tools", ["calendar_list"], []),
    ("Why is budget.ssheet showing an error?", "tools", [], []),
    ("How do I add a row to budget.ssheet?", ("tools", "answer"), [], []),
    ("What is a good name for my bakery?", "answer", [], []),
    ("What are the steps to bake bread?", "answer", [], []),
    ("Make a presentation about how to bake bread", "job:slides:create", [], []),
    ("Write a guide on how to bake bread", "job:words:create", [], []),
    ("Make a short video about our new menu", "studio", [], []),
    ("Write a short story about a goat", ("answer", "job:words"), [], []),
    ("Write a short summary of photosynthesis", "answer", [], []),
    ("Make a short list of fruits", ("answer", "tools", "job"), [], ["make_video", "studio_create"]),
    ("Can you draw?", "answer", [], []),
    ("Can you read PDFs?", "answer(ability)", [], []),
    ("Send it", ("tools", "answer"), [], []),
    ("yes", ("answer", "tools"), [], []),
    ("Make it longer", ("tools", "answer", "job"), [], ["make_video"]),
    ("Remind me about the meeting tomorrow", "tools", ["calendar_add"], []),
    ("Remind me how to tie a tie", "answer", [], []),
    ("Show me a photo of my cat photo.jpg", "tools", [], ["make_image"]),
]


import pytest  # noqa: E402


@pytest.mark.parametrize("prompt,want,has,hasnt", CASES, ids=[c[0][:50] for c in CASES])
def test_routes_as_meant(prompt, want, has, hasnt):
    got, names = lane(prompt)
    wants = want if isinstance(want, tuple) else (want,)
    assert any(got.startswith(w) for w in wants), f"lane {got}, meant {wants}"
    assert not [t for t in has if t not in names], f"missing {[t for t in has if t not in names]}"
    assert not [t for t in hasnt if t in names], f"unwanted {[t for t in hasnt if t in names]}"

