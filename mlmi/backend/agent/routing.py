"""Which tools a request is offered, from its words, the open app and the model's ability.

Split out of agent.py (2026-10-21); every name is re-exported from the agent package."""
import re
import time
from .. import tools as T
from .. import checker, cues, shepherd, verify
from .prompts import PLAN_ONLY, SYSTEM, brief, taught


# Tool routing (the orchestrator's choice, not the person's). Offered all 24 tools
# of an app, Llama 3.2 3B wandered: it browsed herders, re-formatted, exported
# unasked (first live runs, 2026-09-30). Each app has a small CORE, always
# offered in this order so the warm-up's prefix is reused; the EXTRAS join after
# it only when the request names what they do.
CORE = {
    "words": ["words_create", "words_read", "words_add", "words_add_list", "words_add_table", "words_set"],
    "sheets": ["sheets_create", "sheets_read", "sheets_table", "sheets_write"],
    "slides": ["slides_create", "slides_read", "slides_add", "slides_set"],
    "studio": ["studio_create", "studio_read", "studio_template", "studio_add_text", "studio_add_media", "studio_render"],
    "": ["list_files", "read_file", "words_create", "words_add", "words_add_list", "sheets_create", "sheets_write",
         "slides_create", "slides_add"],
}


# Current or outside facts: ABIS web search joins the tools (when web access is on).
WEB_CUES = (r"\b(search|look (it )?up|google|latest|current(ly)?|today|tonight|this (week|month|year)|news|"
            r"price|cost of|exchange rate|weather|forecast|score|who won|released?|launch(ed)?|"
            r"on the (web|internet)|online|website|202[4-9]|right now|recent(ly)?)\b")


# A video made by an AI model (a scene: "a video of a dog running on a beach"),
# as against an edited one in Studio (titles, logo, music, cuts). An AI video
# request drops the Studio tools unless it also asks for Studio's kind of work.
AI_VIDEO = (r"\b(ai[- ](generated )?(video|clip)|text[- ]to[- ]video|veo|sora|kling|"
            r"text[- ](generated|prompted) video|"
            r"(video|clip|footage|animation) (of|showing|where|with) (a|an|the|some|two|three)|"
            # "a video of waves rolling onto a beach": no article after "of" (2026-10-14)
            r"(video|clip|footage) (of|showing) (?!(this|that|these|those|my|the attached|the selected)\b)[a-z]{3,}|"
            r"generat\w* (a |an )?(short )?(\d[- ]second )?(text[- ]to[- ]video )?(video|clip)|"
            r"animat\w* (this |that |the |my |these |those |an? )?(attached |selected )?(photo|picture|image)|"
            r"(photo|picture|image).{0,40}\b(animate|animation|bring .* to life)\b|"
            r"(video|clip) (of|from|using|with) ((the )?(attached|selected)|this|that|these|those|my) (photo|picture|image))")


PHOTO_MONTAGE = (r"\b(montage|slideshow|photo reel|photo compilation|photo sequence|"
                 r"exact photos|original photos|preserve the photos)\b")


# An ad made from the person's photos (studio_ad): photos in sequence with pan and zoom, words over them,
# an end card, music and a voice-over -- not an AI clip guided by the photos (2026-10-27)
PHOTO_AD = (r"\b(?:ads?|adverts?|advertisements?|commercials?|promos?|promotional video|marketing video)\b.{0,80}"
            r"\b(?:photos?|pictures?|images?|shots?)\b|"
            r"\b(?:photos?|pictures?|images?)\b.{0,80}\b(?:into|as|for) (?:an? )?(?:video )?(?:ad|advert|commercial|promo)\b")


VIDEO_TEXT_ONLY = (r"\b(text[- ]only|without(?:\s+\w+){0,5}\s+(photo|picture|image)|"
                   r"(?:do not|don't|never) use(?:\s+\w+){0,5}\s+(photo|picture|image)|"
                   r"ignore (the )?(attached|selected) (photo|picture|image)|no (photo|picture|image) reference)\b")


PHOTO_ANIMATION = (r"\b(?:photo|picture|image)[- ]guided\b|"
                   r"\b(?:animat\w*|bring.{0,20}to life)\b.{0,60}\b(?:photo|picture|image)\b|"
                   r"\b(?:photo|picture|image)\b.{0,60}\b(?:animat\w*|bring.{0,20}to life|guid\w*|reference)\b|"
                   r"\b(?:video|clip)\b.{0,60}\b(?:from|using|with|guided by|based on)\b.{0,30}\b(?:photo|picture|image)\b")


STUDIO_WORK = (r"\b(intro|outro|titles?|logo reveal|lower third|music|jingle|captions?|subtitles?|edit|template|"
               r"studio|\.sstudio|project|text on|words on|end card|title card|montage|slideshow|photo reel|"
               r"photo compilation|photo sequence|exact photos|original photos|preserve the photos)\b")


STUDIO_EDIT = (r"\b(intro|outro|titles?|logo reveal|lower third|music|jingle|captions?|subtitles?|edit|template|"
               r"\.sstudio|project|text on|words on|end card|title card)\b")


# a question about a picture that is already here (or attached)
SEE_ASK = (r"\b(what('?s| is) in|describe|look at|see|tell me about|caption)\b.{0,30}\b(this|the|that|my|attached)?\s*"
           r"(photo|picture|image|screenshot|[\w./\\-]+\.(png|jpe?g|webp|bmp|gif))\b")


# "I want a picture of a lion", "show me a picture of ...": a picture to make (2026-10-14 cue audit)
WANT_PICTURE = (r"^(?!.*\.(?:png|jpe?g|webp|bmp|gif)\b).*?"      # a named picture is shown, not made
                r"\b(i want|i need|i'd like|i would like|give me|show me|get me|can i (have|get|see)) (a|an|some|one)"
                r"( \w+)? (pictures?|images?|photos?|drawings?|illustrations?|posters?|logos?|paintings?) of\b")


# requests that need a model of another kind: (cue, the tool it needs, what to add)
MEDIA_NEEDS = [
    (SEE_ASK, "see_picture", "vision (picture-reading)"),
    (AI_VIDEO, "make_video", "video"),
    (r"\b(draw|paint|illustrat\w*)\b|\b(make|create|design|generat\w*|generat\s+e|produc\w*)\b.{0,30}\b(picture|image|drawing|illustration|logo|icon|poster)\b|"
     r"\b(i want|i need|i'd like|give me|show me|get me) (a|an|some|one)( \w+)? (picture|image|photo|drawing|illustration|poster|logo|painting)s? of\b",
     "make_image", "picture"),
    (r"\b(transcribe|transcript|speech to text|what (is|was) said in)\b", "transcribe_audio", "speech-to-text"),
    (r"\b(read (it |this |that )?aloud|narrat\w*|voice ?-?over|voice it|say (it|this) out loud|text to speech|"
     r"speech (file|audio|recording)|make (a |an )?(recording|audio( file)?|wav))\b",
     "speak_text", "voice"),
]


# a deck asked for with pictures in it
DECK_PICTURES = r"\b(with|and|add|include|including|plus)\b.{0,25}\b(images?|pictures?|photos?|illustrations?|visuals?)\b"


MEDIA_KIND = {"see_picture": "vision", "make_video": "video", "make_image": "picture",
              "transcribe_audio": "speech", "speak_text": "voice"}


EXTRAS = [
    # a slide, block or cell named by number or place: the tools that edit one (the 3B, not offered
    # slides_set, added slides to another deck when asked for speaker notes; 2026-10-06 orientation)
    (r"\b(slides? (\d+|one|two|three|four|five|six|seven|eight|nine|ten)|(first|last|second|third|next|final) slide|"
     r"speaker notes?|notes)\b", ["slides_set", "slides_delete", "slides_move"]),
    (r"\b((block|paragraph|heading|line) \d+|(first|last|second) (paragraph|heading|line|bullet))\b", ["words_set", "words_delete"]),
    (r"\b(cell [a-z]{1,2}\d+|[a-z]{1,2}\d+ (in|of|to))\b", ["sheets_set_cell"]),
    (r"\b(bold|italic|align|centre|center|colou?r|font|size|format)", ["words_format", "sheets_format"]),
    (r"\b(image|picture|photo|logo)", ["words_add_image", "slides_add_image"]),
    (r"\b(page break|new page)", ["words_add_page_break"]),
    (r"\b(cell|table)", ["words_set_cell", "words_add_table", "sheets_set_cell"]),
    (r"\b(move|reorder|order|before|after)", ["words_move", "slides_move"]),
    (r"\b(delete|remove|drop)", ["words_delete", "slides_delete"]),
    (r"\b(replace|change|fix|correct|rename)", ["words_replace", "words_set", "slides_set"]),
    (r"\b(chart|graph|plot)", ["sheets_add_chart"]),
    (r"\b(sort|rank|order)", ["sheets_sort"]),
    (r"\b(tab|another sheet|new sheet|sheets)\b", ["sheets_add_sheet"]),
    (r"\b(theme|dark|light|style|look)", ["slides_set_theme"]),
    (r"\b(text box|caption|footer|label)", ["slides_add_text"]),
    # "Word" the program, not "replace the word Nairobi" (that exported a .docx unasked, 2026-10-12)
    (r"\b(export|docx|xlsx|pptx|excel|power ?point|csv|markdown|\.md|pdf|microsoft word|word (file|document|doc)|"
     r"(as|to|in|into) word)\b", ["export_file"]),
    (r"\b(import|open)\b.*\.(docx|xlsx|pptx)", ["import_file"]),
    (r"\b(list (the |my |all )?files|which files|what files|folders?\b|what.s in|workspace)", ["list_files", "read_file", "create_folder"]),
    (r"\b(read|look at|summari[sz]e|check)", ["read_file"]),
    # "Copy packing.txt to backup/packing.txt" names no "file": a path is enough
    # (the 3B rewrote packing.txt instead, 2026-10-13 simulation)
    (r"\b(rename|move|copy|delete|remove)\b.*(\b(file|folder)\b|\b[\w./\\-]+\.[a-z0-9]{1,6}\b)",
     ["move_file", "copy_file", "delete_file"]),
    (r"\b(text file|\.txt|notes file)", ["write_text_file"]),
    (r"\b(number(ed)? (the )?headings?)", ["h_number_headings"]),
    (r"\b(add (a )?totals? (row )?(to|under) (the|an?) (existing|table))", ["h_add_totals"]),
    # an agenda SLIDE, not "an agenda for Monday" (that is the calendar's)
    (r"\bagenda (slide|page)\b|\b(add|insert|put) an? agenda\b(?!.{0,30}\b(for|meeting|calendar|event)\b)",
     ["h_agenda_slide"]),
    (r"\b(herder|custom tool|your own tool)", ["herder_list", "herder_help", "herder_make", "herder_run"]),
    # "short" alone is not a video ("a short summary"); YouTube Shorts are
    (r"\b(video|intro|outro|animat\w*|clip|reel|shorts|youtube short|logo reveal|motion|render|mp4|lower third|title card)",
     ["studio_create", "studio_template", "studio_add_text", "studio_add_media", "studio_render"]),
    (r"\b(shape|bar|circle|box)\b", ["studio_add_shape"]),
    (WEB_CUES, ["web_search", "web_fetch"]),
    # photo montages preserve their inputs by using Studio image layers and its renderer
    (PHOTO_MONTAGE, ["studio_create", "studio_read", "studio_add_media", "studio_render"]),
    # photo animation is a video-model request, with the images passed as guidance
    (r"\b(animat\w*|bring .* to life)\b.*\b(photo|picture|image)\b|"
     r"\b(photo|picture|image)\b.{0,40}\b(animate|animation|bring .* to life)\b", ["make_video"]),
    (r"\b(?:text[- ]only|without(?:\s+\w+){0,5}\s+(?:photo|picture|image))\b.{0,40}\b(?:video|clip|footage)\b|"
     r"\b(?:video|clip|footage)\b.{0,40}\b(?:text[- ]only|without(?:\s+\w+){0,5}\s+(?:photo|picture|image))\b",
     ["make_video"]),
    # media models as tools of any chat model (the model drop-down picks which one)
    (AI_VIDEO, ["make_video"]),
    # changing a picture that is here (2026-10-14 live run: "make logo.png black and white" had
    # no tool and the 3B tried slides and video tools)
    (r"\b(black[- ]and[- ]white|gr[ae]y ?scale|monochrome|sepia|crop|rotate|mirror|flip|resize|brighten|darken|"
     r"brightness|contrast|blur|sharpen|invert)\b.{0,60}\.(png|jpe?g|webp|bmp|gif)\b|"
     r"\.(png|jpe?g|webp|bmp|gif)\b.{0,60}\b(black[- ]and[- ]white|gr[ae]y ?scale|monochrome|sepia|crop|rotate|mirror|"
     r"flip|resize|brighten|darken|brightness|contrast|blur|sharpen|invert|smaller|bigger|larger|brighter|darker|"
     r"lighter|sharper|blurr?y)\b|"
     r"\b(edit|retouch|touch up)\b.{0,30}\b(photo|picture|image|\w+\.(png|jpe?g|webp|bmp))\b", ["edit_image"]),
    # making a picture, not mentioning one ("add the picture to slide 2" is slides_add_image)
    (r"\b(draw|paint|illustrat\w*|drawing|artwork|poster|wallpaper)\b|"
     r"\b(make|create|generate|design|produce)\b.{0,30}\b(picture|image|photo|logo|icon)\b|" + WANT_PICTURE, ["make_image"]),
    # looking at a picture that is already here
    (SEE_ASK, ["see_picture"]),
    # Studio's own voice and captions on a project's timeline
    (r"\b(voice ?over|narrat\w*)\b.{0,60}\b(video|studio|project|clip|\.sstudio)\b|"
     r"\b(video|studio|project|clip|\.sstudio)\b.{0,60}\b(voice ?over|narrat\w*)\b", ["studio_voice"]),
    (r"\b(captions?|subtitles?)\b", ["studio_transcribe"]),
    (PHOTO_AD, ["studio_ad", "studio_render"]),
    # Studio: transitions, clip speed, and listening to a sound to place things on its moments
    (r"\b(transitions?|cross-?fade|fade (between|into)|wipe (between|into)|slide (between|into))\b", ["studio_transition"]),
    (r"\b(slow[- ]?mo(tion)?|slow (it |this )?down|speed (it |this )?up|faster|slower|speed|"
     r"(twice|half|three times) as (fast|slow|quick)|double speed|half speed|\d(\.\d)?x)\b", ["studio_set"]),
    (r"\b(beats?|rhythm|in time with|on the beat|sync\w*|when (he|she|they|it) says?|pauses?|silences?|listen|hear)\b",
     ["studio_listen", "studio_set", "studio_add_text"]),
    (r"\b(transcribe|transcript|what (is|was) said|speech to text|recording|voice note|audio file|\.(mp3|wav|m4a|ogg|webm))\b",
     ["transcribe_audio"]),
    # watching a video: what happens in it, finding a moment, cutting by what is seen or said
    (r"\b(what (happens|is happening|goes on)|watch (it|this|the video)|describe (the|this) video|in (the|this) video|"
     r"(find|keep|cut|remove) the (part|bit|moment|scene)s?|where (he|she|they|it) (says?|appears?|laughs?)|\.(mp4|mov|mkv|m4v))\b",
     ["look_video", "studio_set", "studio_split"]),
    (r"\b(?:speak_text|read (it |this )?aloud|voice ?over|narrat\w*|spoken description|text to speech|say it|speak|audio version|"
     r"out loud|aloud|in a voice|(into|to) (speech|audio|a voice|a recording))\b", ["speak_text"]),
    # Studio's own change words, only when a Studio project, clip or layer is meant ("make the title
    # bigger in save-water.sdeck" was offered studio_set, 2026-10-14 live run)
    (r"^(?=.*(\.sstudio\b|\bstudio\b|\bvideo|\bclips?\b|\blayers?\b|\btimeline\b|\bcaptions?\b))"
     r"(?=.*\b(change|move|longer|shorter|later|earlier|bigger|smaller|colou?r|remove|delete)\b)", ["studio_set", "studio_delete"]),
    # a named deck: the tools that change its slides
    (r"\.sdeck\b", ["slides_read", "slides_set"]),
    # music and audio on a video: Studio tools for adding a soundtrack (plan A7)
    (r"\b(music|soundtrack|audio track|background music|song|jingle|beat|add sound|add music)\b",
     ["studio_add_media", "studio_set"]),
    # "4-second clip", "a 10-second video" — make_video's seconds in the request (plan A7)
    (r"\b(\d{1,2})[- ]?second[- ]?(?:clip|video|scene|animation|footage|short)\b", ["make_video"]),
    # mail and calendar over IMAP/SMTP and .ics (plan A10); sending always asks first
    (r"\b(e-?mails?|inbox|mail|unread)\b", ["mail_list", "mail_read", "mail_draft"]),
    (r"\b(send|reply|forward)\b.*\b(e-?mail|mail|message|draft)\b|\b(e-?mail|mail) (him|her|them|it)\b", ["mail_draft", "mail_send"]),
    (r"\b(calendar|schedule|appointment|meeting|event|remind me|agenda for)\b", ["calendar_add", "calendar_list"]),
]


# a bare ask for "a file" with no kind or subject: the chat's own files are named (run)
FILE_ASK = (r"^\s*(?:please\s+)?(?:i\s+)?(?:want|need|would like|'d like|give me|send me|where(?:'s| is))\s+"
            r"(?:a|the|my|that|this)\s+(?:file|files|copy|download)\s*(?:please)?[.!?]*\s*$")


ACTION = (r"\b(make|create|write|draft|add|insert|put|edit|change|update|fix|delete|remove|rename|move|copy|export|"
          r"import|save|build|prepare|generate|fill|format|sort|chart|graph|list (the |my |all )?files|open|turn .* into|"
          r"convert|summari[sz]e|read|translate|plan (a|my|the)|number (the |my |its )?(headings|pages|slides|rows|lines|items)|"
          r"render|animate|edit|draw|paint|illustrate|transcribe|narrate)\b")


# Code asked for in the chat is answered with a code block (it lands on the Work
# Canvas, where it can be run or saved); only a named file to write is a file job.
CODE = (r"\b(python|javascript|typescript|java|c\+\+|rust|golang|sql|regex|html|css|bash|powershell|code|function|"
        r"script|snippet|algorithm)\b")


# "Can you make spreadsheets?" asks what Sheep can do; "Can you make me a budget
# sheet?" asks for one. The first was sent down the job path and the 3B started
# making a document nobody asked for (live, 2026-10-13). An ability question names
# no particular thing -- no "me", "a", "my", "about ...", "that ..." -- and gets
# words, from the model that now knows Sheep's reach (brief).
ABILITY_Q = re.compile(r"(?:are you (?:able|capable)(?: of| to)?|is it possible for you to|do you have the ability to|"
                       r"(?:can|could) (?:you|sheep)(?: also| even| really| actually)?)\s+"
                       r"(?:do|make|making|create|creating|write|writing|edit|editing|handle|read|reading|open|"
                       r"work with|build|generate|produce)\s+(.*)", re.I | re.S)


SPECIFIC = re.compile(r"^(?:me|us|a|an|the|my|our|this|that|these|those|one|some|it)\b|"
                      r"\b(?:about|on the|titled|called|named|for (?:my|our|me|us)|with (?:my|our)|from (?:my|our)|"
                      r"that|which)\b|\.\w{2,5}\b", re.I)


def ability_question(text):
    """Is this turn asking WHETHER Sheep can do something, not asking it to?"""
    t = (text or "").strip()
    if re.search(r"\b(what (else )?can (you|sheep) do|what are you able to do|what can you help (me )?with|"
                 r"are you (just|only) a)\b", t, re.I):
        return True
    m = ABILITY_Q.search(t)
    return bool(m) and not SPECIFIC.search(m.group(1))


# the kinds of thing Sheep makes or works on; a question or a chat request naming none of them
# is answered in words. "my" alone is not one: "What should I write in a card for my mum?"
# was sent to the tools (2026-10-14 cue audit)
ARTIFACT_NOUNS = (r"\b(documents?|docs?|files?|folders?|sheets?|workbooks?|spreadsheets?|decks?|slides?|"
                  r"presentations?|powerpoint|excel|word document|pdf|tables?|charts?|graphs?|videos?|clips?|"
                  r"pictures?|images?|photos?|drawings?|logos?|posters?|icons?|recordings?|audio|voice|"
                  r"e-?mails?|inbox|calendar|events?|reminders?|studio|projects?|workspace)\b|"
                  r"\b(this|these|that|those) (one|ones)\b")


def wants_tools(text, app=""):
    """Does this turn ask for work on files, or just for an answer? Asked "What is
    the capital of Uganda?" with tools on offer, the 3B called tools nine times
    and failed (2026-09-30); a question gets a plain answer, no tools."""
    import re
    t = (text or "").lower()
    # a named file of any kind Sheep works on: pictures, sound, video and Studio projects too
    # ("Describe photo.jpg", "Slow down the first clip in intro.sstudio" were answered in
    # words with no tool, 2026-10-14 cue audit)
    if re.search(r"\b[\w-]+\.(sdoc|ssheet|sdeck|sstudio|docx|xlsx|pptx|csv|txt|md|py|js|html|pdf|eml|ics|"
                 r"png|jpe?g|webp|bmp|gif|wav|mp3|m4a|ogg|flac|mp4|mov|webm|mkv)\b", t):
        return True
    if re.search(CODE, t) and not app:
        return False                 # code in the chat: a code block, not a file job
    # mail, the calendar, the person's files and a spoken voice are work, asked as questions
    # or not ("Do I have unread emails?", "What files do I have?", "Say hello in a voice")
    if re.search(r"\b(e-?mails?|inbox|unread|mailbox|calendar|appointments?|reminders?|agenda)\b|"
                 r"\bremind me (to|at|on|in|tomorrow|tonight|next|every|about (the|my|our))\b", t):
        return True
    if re.search(LIST_ASK, t):
        return True                  # "What are my files?" is a listing, asked as a question
    if re.search(shepherd.HOWTO, t):
        return False                 # "How do I make a chart in Excel?": the how, in words
    if re.search(r"\b(in a voice|out loud|aloud|voice ?-?over|text to speech|speak (it|this|that)|"
                 r"(into|to) (speech|audio|a voice|a recording))\b", t):
        return True
    if re.search(WANT_PICTURE, t):
        return True
    # writing in the chat -- a poem, a translation, a summary, ideas -- that names no file and
    # no kind of file is answered in words; a document job is still caught by shepherd.intent
    if (not app and re.match(r"\s*(?:please\s+|can you\s+|could you\s+|now\s+)?(?:translate|summari[sz]e|"
                             r"write|compose|give me|suggest|brainstorm|list|tell me|explain|describe)\b", t)
            and not re.search(ARTIFACT_NOUNS, t)):
        return False
    # "Can you make an equivalent of this for the tech team?" is a request in polite words: read as a
    # question it named no file kind and got no tools (Ernest's chat, 2026-10-30); "Can you make
    # spreadsheets?" is still an ability question, answered in run
    if re.match(r"\s*(?:please\s+)?(?:can|could|would|will) (?:you|sheep)\s+(?:please\s+)?(?:just\s+)?"
                r"(?:make|create|build|prepare|draft|generate|produce|design|put together|turn|convert)\b", t):
        return True
    # A question about the world is answered, not worked on: "Why do leaves change
    # colour?" matched the action word "change" and went down the file path
    # (2026-10-06). A question that names no file, document or app gets words.
    question = bool(re.match(r"\s*(why|what|how|when|where|who|whom|which|whose|is|are|was|were|do|does|did|can|could|"
                             r"should|would|will|tell me|explain|describe)\b", t)) or t.rstrip().endswith("?")
    if question and not app and not re.search(ARTIFACT_NOUNS, t):
        from ..tools import web
        return bool(web.STATE["on"] and re.search(WEB_CUES, t))   # a question about now still searches
    if re.search(ACTION, t):
        return True
    from ..tools import web
    if web.STATE["on"] and re.search(WEB_CUES, t):
        return True                  # a question about now: answered from the web, with sources
    return bool(app) and bool(re.search(r"\b(this|the|my) (doc|document|sheet|workbook|deck|slides?|file|table|page)\b", t))


def route_tools(allowed, app, text):
    """The tools for this request: the app's core, then the extras it names."""
    import re
    core = CORE.get(app if app in CORE else "", CORE[""])
    out = [n for n in core if n in allowed]
    t = (text or "").lower()
    for pat, names in EXTRAS:
        if t and re.search(pat, t):
            out += [n for n in names if n in allowed and n not in out
                    and (T.TOOLS[n].app in ("files", "web", "media", "mail", app) or not app)]
    # at most one proven custom tool whose phrase (or, with none, all of its name's words) is in the request
    from ..tools.herders import library as _custom_tools
    for h in _custom_tools().values():
        n = "h_" + h["name"]
        if n in out or n not in allowed or not h.get("proven") or (app and T.TOOLS[n].app != app):
            continue
        words = [w for w in h["name"].split("_") if len(w) > 2]
        if any(a in t for a in h.get("asks") or []) or (words and all(re.search(r"\b" + re.escape(w[:5]), t) for w in words)):
            out.append(n)
            break
    if "edit_image" in out and not re.search(r"\b(slide|deck|document|doc|report|page|video|montage|studio)\b", t):
        # a photo edit: the picture-placing and video tools only distract a small model
        out = [n for n in out if n not in ("words_add_image", "slides_add_image", "make_video", "make_image")]
    if "make_image" in out and not re.search(r"\b(slides?|deck|presentation|document|doc|report|page|sheet)\b|"
                                             r"\.(sdoc|sdeck|ssheet)\b", t):
        # a picture to make: placing one in a slide or a document is another request
        out = [n for n in out if n not in ("words_add_image", "slides_add_image")]
    # "an intro video with the title Sheep" is Studio's, not a scene (is_ai_video says why)
    explicit_video = bool((re.search(AI_VIDEO, t) and not re.search(STUDIO_EDIT, t)) or re.search(PHOTO_ANIMATION, t))
    narration = bool(re.search(r"\b(narrat\w*|voice[- ]?over|spoken description)\b", t))
    if "make_video" in out and app != "studio" and not re.search(STUDIO_WORK, t) and not narration:
        out = [n for n in out if not n.startswith("studio_")]
    elif "make_video" in out and re.search(STUDIO_WORK, t) and not explicit_video:
        out = [n for n in out if n != "make_video"]      # titles, intros, logos: Studio's work
    return out


IMAGE_OP_WORDS = (r"black[- ]and[- ]white|gr[ae]y ?scale|monochrome|sepia|vintage|invert(?:ed)?|negative|"
                  r"sharpen|sharper|blur(?:ry)?|soften|brighter|brighten|lighter|darker|darken|"
                  r"(?:more |less |higher |lower )?contrast(?: \d+(?:\.\d+)?)?|"
                  r"rotate(?:d)?(?: it| this)?(?: by)?(?: -?\d+(?: degrees)?| left| right)?|turn(?: it)? (?:left|right)|"
                  r"(?:flip(?:ped)?(?: it)? )?upside down|mirror(?:ed)?|flip(?:ped)?(?: it)?(?: horizontally| vertically)?|"
                  r"crop(?: it)?(?: to)? \d+(?:\.\d+)?, ?\d+(?:\.\d+)?, ?\d+(?:\.\d+)?, ?\d+(?:\.\d+)?|"
                  r"resize(?: it)?(?: to)? \d+(?:x\d+)?|smaller|bigger|larger")


def image_ops_in(text):
    """The picture edits a message asks for, in the words edit_image reads ("" if none)."""
    found = []
    # the picture's name out first: "crop photo.jpg to 10,10,80,80" reads as "crop to 10,10,80,80"
    text = re.sub(r"\s*[\w./\\-]+\.(?:png|jpe?g|webp|bmp|gif)\b", "", str(text or ""), flags=re.I)
    for m in re.finditer(r"\b(" + IMAGE_OP_WORDS + r")\b", text, re.I):
        w = m.group(1).lower()
        w = re.sub(r"\b(it|this|by|to|degrees)\b", "", w)
        w = re.sub(r"\s+", " ", w).strip()
        if w in ("more contrast", "higher contrast"):
            w = "contrast 1.4"
        elif w in ("less contrast", "lower contrast"):
            w = "contrast 0.7"
        elif w.startswith("turn"):
            w = "rotate " + ("270" if "left" in w else "90")
        elif w.startswith("rotate") and "left" in w:
            w = "rotate 270"
        elif w.startswith("rotate") and "right" in w:
            w = "rotate 90"
        elif w.endswith("upside down"):
            w = "flip" if w.startswith("flip") else "rotate 180"
        elif "flip" in w and "vertical" in w:
            w = "flip"
        elif "flip" in w:
            w = "mirror"
        elif w in ("soften", "blurry"):
            w = "blur"
        elif w == "sharper":
            w = "sharpen"
        found.append(w)
    return ", ".join(dict.fromkeys(found))


def is_ai_video(text):
    """A scene for the video model, not a Studio edit: "an intro video with the title
    Sheep" matched "video with the" and skipped the Studio planner (2026-10-07)."""
    t = (text or "").lower()
    return bool(re.search(AI_VIDEO, t)) and not re.search(STUDIO_EDIT, t)


def explicit_media_request(text):
    """Whether the person clearly asked Sheep to create video or a photo montage.

    This lets media requests bypass the document writer and gives the local
    tool router enough context to include Studio when narration or a montage is
    part of the request. It does not classify ordinary Studio edits as AI video.
    """
    t = str(text or "")
    montage = bool(re.search(PHOTO_MONTAGE, t, re.I)) or bool(re.search(PHOTO_AD, t, re.I))
    photo_animation = bool(re.search(PHOTO_ANIMATION, t, re.I))
    text_video = bool(re.search(AI_VIDEO, t, re.I) and not re.search(STUDIO_EDIT, t, re.I))
    return montage or photo_animation or text_video


# Tool → ability required on this machine. Tools without an entry are always offered.
# The ability is missing when the model or machine cannot do it (plan A5).
TOOL_ABILITY = {
    "make_video": "makes video",
    "make_image": "makes pictures",
    "see_picture": "reads pictures",
    "transcribe_audio": "hears",
    "speak_text": "speaks",
    "studio_voice": "speaks",
    "studio_transcribe": "hears",
}


def able(state, names):
    """(the tools this machine can actually run, tips for the missing abilities).
    A tool whose required ability is missing is not offered (plan A5)."""
    orch = getattr(state, "orch", None)
    abl = orch.abilities() if orch and hasattr(orch, "abilities") else {}
    needs = getattr(orch, "NEEDS", {}) if orch else {}
    kept, tips = [], []
    for n in names:
        req = TOOL_ABILITY.get(n)
        if req and not abl.get(req):
            tip = needs.get(req, f"a model that can {req}")
            if tip not in tips:
                tips.append(tip)
            continue
        kept.append(n)
    return kept, tips


def setup(state, app, text=""):
    """(apps, tool names, system text) for a job in `app`. With no text (the
    warm-up) it is the app's core; a request adds the extras it names after it."""
    tier = state.model_card()["tier"]
    # Media intent is app-independent: a video request typed while Words is open
    # still needs Sheep's video and Studio tools, rather than the document writer.
    tool_app = "studio" if explicit_media_request(text) else app
    apps = ["files", "web", "media", "mail"] + ([tool_app] if tool_app and tool_app != "files" else ["words", "sheets", "slides", "studio"])
    allowed = T.offered(1 if state.permission == "plan" else tier, apps) if tier > 0 else []
    # tools by ability, and the model is told what to add (plan A5)
    if allowed:
        allowed, tips = able(state, allowed)
        if tips:
            tips_text = "Some tools are not available because this machine has no model that can: " \
                        + "; ".join(tips) + ". Add one in Settings > Models if needed."
            SYSTEM_TIP = tips_text
        else:
            SYSTEM_TIP = ""
    else:
        SYSTEM_TIP = ""
    names = route_tools(set(allowed), tool_app if tool_app != "files" else "", text)
    return apps, names, system_prompt(state, SYSTEM_TIP)


def system_prompt(state, tip=""):
    """The system turn for a job: the rules, what Sheep can do here, what this
    model was taught by its orientation, and what is missing on this machine."""
    lessons = taught(state)
    # the date: "remind me tomorrow at 9" went into the calendar on 15 Nov 2023 (2026-10-14 live run)
    today = time.strftime("Today is %A, %d %B %Y.")
    return (SYSTEM + "\n\n" + today + "\n\n" + brief(state)
            + ("\n\nLessons from your first jobs here:\n" + "\n".join("- " + h for h in lessons) if lessons else "")
            + ("\n\n" + tip if tip else "")
            + (PLAN_ONLY if state.permission == "plan" else ""))


def warm(state, app):
    """Read an app's system turn and tool list into an engine slot before anyone
    asks: on a CPU a 3.5K-token tool prompt is two minutes, paid once here
    instead of on the person's first request."""
    _, names, system = setup(state, app)
    t0 = time.perf_counter()
    state.local.chat([{"role": "system", "content": system}, {"role": "user", "content": "ready"}],
                     T.schemas(names) or None, max_tokens=1)
    return (time.perf_counter() - t0) * 1000


# Media tools whose success completes a media-only request, and the words that
# say the request wants more than the media (then the model carries on).
MEDIA_DONE = {
    "make_image": lambda r: "Here is your picture." + (" " + r.split(". ", 1)[1] if ". " in r else ""),
    "make_video": lambda r: "Here is your video clip." + (" " + r.split(". ", 1)[1] if ". " in r else ""),
    "speak_text": lambda r: "Here is the recording.",
    "transcribe_audio": lambda r: "Here is what was said:\n\n" + (r.split(":\n", 1)[1] if ":\n" in r else r),
}


# "what files do I have?" -- asked of the workspace, not of one named file. Sheep
# reads the folder and the model answers from that listing, because models fail
# this two ways: one invented a folder, hit the error and gave up; another said it
# had no access to any files and called nothing (2026-10-07).
LIST_ASK = (r"\b(list|show|which|what|tell me)\b[^.?!]{0,40}\b(files?|documents?|folders?|workspace|"
            r"contents?|saved|stored)\b|\bwhat (have i got|did i save|is in (here|this folder))\b|"
            r"\bwhat did i save\b|"  # noqa: E501 -- "save" is a listing word here, so this path has its own veto
            r"\bshow me my (work|files|documents)\b")


OTHER_WORK = (r"\b(document|doc|sheet|workbook|deck|slides?|presentation|report|video|studio|then|and (put|add|insert|use)|"
              r"add (it|them|this) to|insert|into (a|the|my))\b")
