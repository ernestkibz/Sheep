"""Typo-tolerant cues (2026-10-14). Sheep decides some routes by reading the person's
request with patterns (agent.py MEDIA_NEEDS, the explicit_* checks). People misspell
words and split them ("generat e an image"), and a pattern then misses.

fix(text) returns the request with each misspelt word corrected to the cue word it
was meant to be, and split words joined ("generat e" -> "generate"). Routing reads
fix(text); the model still gets the person's own words.

A word is corrected only when it is not a real English word (english_words.txt): the
first version corrected toward the ~600 pattern words alone and changed 408 words in
1,200 correctly spelt requests, among them remote -> remove (a delete), heart -> chart
and test -> latest (a web search). File names, capitalised names and short words are
left alone."""
import difflib
import re
from functools import lru_cache
from pathlib import Path

# the words routing listens for, beyond the ones the patterns happen to spell out
EXTRA = """
generate create make produce design draw paint illustrate picture image images photo photos video
clip animate animation voice narration narrate recording audio speech transcribe transcript
spoken speak read aloud logo poster icon drawing illustration sound music write document documents
sheet sheets spreadsheet presentation slides slide deck studio calendar event email mail draft
search summarize summary translate chart table formula budget project delete remove move copy rename
export save open show list find replace update edit insert add text code python website page files
powerpoint excel black white grayscale sepia rotate crop resize brighter darker blur sharpen flip mirror
inbox unread remind reminder schedule appointment weather aloud loud
""".split()

# a word touching one of these is part of a file name or path
PATHY = frozenset("./\\_-")

# short misspellings that happen to be word fragments in the English list
MISSPELT = {"mak": "make", "creat": "create"}

# common words that are never corrected, however close a cue word is
STOP = set("""
a an and are as at be but by can did do does for from had has have he her his i if in into is it its me
my no not of off on or our out she so that the their them then there these they this to too up us was we
were what when where which who why will with would you your about after again all also any been before
being both each few just more most much must now only other over same some such than through under very
""".split())


@lru_cache(maxsize=1)
def english():
    """Real English words: a word in this set is never corrected."""
    try:
        lines = Path(__file__).with_name("english_words.txt").read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    return frozenset(w for w in lines if w and not w.startswith("#"))


@lru_cache(maxsize=1)
def pattern_words():
    """Every word the routing patterns spell out (agent.py, direct.py): never corrected,
    even when not in the English list ("slideshow", "sdeck")."""
    from . import agent, direct
    words = set()
    for mod in (agent, direct):
        for val in vars(mod).values():
            if isinstance(val, str):
                items = [val]
            elif isinstance(val, (list, tuple)):
                items = list(val)
            else:
                continue
            for it in items:
                if isinstance(it, (list, tuple)):
                    it = next((x for x in it if isinstance(x, str)), "")
                if isinstance(it, str) and ("\\" in it or "|" in it or "(" in it):
                    words.update(re.findall(r"[a-z]{3,}", it.lower()))
    return frozenset(words)


@lru_cache(maxsize=1)
def vocab():
    """The cue words a misspelling may be corrected to: the real words in the patterns
    (fragments such as "generat" or "produc" are not targets) and the ways of asking above."""
    eng = english()
    return frozenset(set(EXTRA) | {w for w in pattern_words() if len(w) >= 4 and w in eng})


@lru_cache(maxsize=1)
def _anagrams(v):
    """Cue words by their letters in sorted order: "lsit" and "fiels" are letter swaps of list and files."""
    by = {}
    for x in v:
        by.setdefault("".join(sorted(x)), []).append(x)
    return by


def _fix_word(w, v, eng):
    """One plain lowercase word: a known misspelling, a letter swap of a cue word, or the
    nearest cue word when close enough. Real words and names stay as they are."""
    if w != w.lower() or len(w) < 3 or w in STOP:
        return w
    if w in MISSPELT:
        return MISSPELT[w]
    if w in v or w in eng or w in pattern_words():
        return w
    swap = [x for x in _anagrams(v).get("".join(sorted(w)), []) if x != w]
    if len(swap) == 1:
        return swap[0]
    near = difflib.get_close_matches(w, list(v), n=1, cutoff=0.8)
    return near[0] if near else w


def fix(text):
    """The request with its misspelt words corrected for routing (see the module doc)."""
    text = str(text or "")
    if not text:
        return text
    v, eng = vocab(), english()
    parts = re.split(r"([A-Za-z]+)", text)          # odd entries are words, even entries are the gaps
    idx = list(range(1, len(parts), 2))
    i = 0
    while i < len(idx):
        k = idx[i]
        w = parts[k]
        if i + 1 < len(idx):
            j = idx[i + 1]
            gap, nxt = parts[j - 1], parts[j]
            joined = (w + nxt).lower()
            # a word split by a space: "generat e" -> "generate", "e mail" -> "email"
            if (gap.isspace() and len(gap) <= 2 and joined in v and w.islower() and nxt.islower()
                    and (w not in eng or len(nxt) <= 2) and nxt not in STOP):
                parts[k], parts[j], parts[j - 1] = joined, None, None
                i += 2
                continue
        # part of a file name or path ("note.txt", "my_sheet", "v2-draft"): left exactly as typed
        before, after = parts[k - 1], parts[k + 1] if k + 1 < len(parts) else ""
        if not (before[-1:] in PATHY or after[:1] in PATHY or before[-1:].isdigit() or after[:1].isdigit()):
            parts[k] = _fix_word(w, v, eng)
        i += 1
    return "".join(p for p in parts if p is not None)
