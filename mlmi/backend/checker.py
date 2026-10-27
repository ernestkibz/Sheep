"""The Checker: is this model, with this configuration, answering properly?

Two parts:

  rules      deterministic faults that need no second opinion: an empty reply,
             leaked role markers, a repeating loop, JSON that was asked for and
             does not parse, a code block that does not parse, tool-call
             arguments that are not JSON, and web answers that cite none of the
             sources that were opened
  a judge    optional: a small Sheep model the person plugs in, asked whether an
             answer is right. It is the one advanced step the spec leaves to the
             person (train or obtain a smaller Checker and plug it in).

A new judge sits an exam before it becomes active: known-good and known-bad
answers it must tell apart. A judge that fails is not installed and the one
before it stays (the rollback). The orchestrator's canaries run through both.
"""
import ast
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROLE_MARK = re.compile(r"<\|(?:assistant|user|system|tool|im_start|im_end|start_header_id|eot_id)\|>"
                       r"|^(?:assistant|user|system)\s*:\s*$", re.I | re.M)
FENCE = re.compile(r"```([\w+-]*)[ \t]*\n([\s\S]*?)```")
PASS_MARK = 0.83          # a judge must sort at least 10 of the 12 exam answers
EXAM_REVISION = 1         # increment when held-out questions or the pass rule change

# The exam: (question, answer, should pass). Plain facts any checker must know,
# wrong answers that look like answers, and the faults the rules also catch.
EXAM = [
    ("What is the capital of France?", "Paris.", True),
    ("What is the capital of France?", "Berlin is the capital of France.", False),
    ("What is 12 times 12?", "144", True),
    ("What is 12 times 12?", "124", False),
    ("How many legs does a spider have?", "Eight.", True),
    ("How many legs does a spider have?", "A spider has six legs.", False),
    ("Translate 'thank you' into French.", "Merci.", True),
    ("Translate 'thank you' into French.", "Gracias.", False),
    ("Which planet is closest to the Sun?", "Mercury.", True),
    ("Which planet is closest to the Sun?", "Jupiter is the closest planet to the Sun.", False),
    ("Write one sentence about the sea.", "The sea covers most of the Earth's surface.", True),
    ("Write one sentence about the sea.", "Bananas are yellow and grow on trees.", False),
]
# More held-out questions, never in the training lessons (teach.py leaves them
# out): a 12-answer exam is too small to tell a trained Checker from its base.
EXAM_MORE = [
    ("What is the capital of Nepal?", "Kathmandu.", True), ("What is the capital of Nepal?", "Dhaka.", False),
    ("What is the capital of Hungary?", "Budapest is the capital.", True), ("What is the capital of Hungary?", "Prague.", False),
    ("What is 9 times 7?", "63", True), ("What is 9 times 7?", "56", False),
    ("What is 100 minus 37?", "63.", True), ("What is 100 minus 37?", "73.", False),
    ("How many wheels does a tricycle have?", "Three.", True), ("How many wheels does a tricycle have?", "Two.", False),
    ("Which animal says moo?", "A cow.", True), ("Which animal says moo?", "A cat says moo.", False),
    ("Translate 'water' into French.", "Eau.", True), ("Translate 'water' into French.", "Agua.", False),
    ("Who was the first person to walk on the moon?", "Neil Armstrong.", True),
    ("Who was the first person to walk on the moon?", "Yuri Gagarin.", False),
    ("How many hours are in two days?", "48", True), ("How many hours are in two days?", "24", False),
    ("Write one sentence about the stars.", "Stars shine in the night sky.", True),
    ("Write one sentence about the stars.", "Cars need fuel to run.", False),
    ("What is the opposite of up?", "Down.", True), ("What is the opposite of up?", "Left.", False),
    ("What do cows drink?", "Water.", True), ("What do cows drink?", "the the the the the the", False),
]


def rules(text, request="", calls=None, sources=None):
    """Hard faults only. Returns a list of reasons (empty when clean)."""
    text = (text or "").strip()
    calls = calls or []
    faults = []
    if not text and not calls:
        return ["empty reply"]
    if ROLE_MARK.search(text):
        faults.append("role markers in the reply")
    words = re.findall(r"[\w'-]+", text.lower())
    if len(words) >= 18:
        for w in (1, 2, 3, 4):
            if any(words[i:i + w] == words[i + w:i + 2 * w] == words[i + 2 * w:i + 3 * w] == words[i + 3 * w:i + 4 * w]
                   for i in range(len(words) - 4 * w + 1)):
                faults.append("a repeating loop")
                break
    if re.search(r"\b(?:valid )?json\b", request or "", re.I) and re.search(
            r"\b(?:only|return|reply|output|format|valid)\b", request or "", re.I):
        if _json_of(text) is None:
            faults.append("JSON was asked for and the reply is not valid JSON")
    for c in calls:
        args = (c.get("function") or {}).get("arguments")
        if isinstance(args, str):
            try:
                json.loads(args or "{}")
            except ValueError:
                faults.append(f"tool call {(c.get('function') or {}).get('name', '?')}: arguments are not JSON")
    for lang, body in FENCE.findall(text):
        lang = lang.lower()
        if lang in ("python", "py"):
            try:
                ast.parse(body)
            except SyntaxError as e:
                faults.append(f"Python does not parse (line {e.lineno}: {e.msg})")
        elif lang in ("javascript", "js") and shutil.which("node"):
            msg = _node_check(body)
            if msg:
                faults.append(f"JavaScript does not parse ({msg})")
        elif lang == "json":
            try:
                json.loads(body)
            except ValueError as e:
                faults.append(f"JSON block does not parse ({str(e)[:60]})")
    if sources:
        cited = re.findall(r"\[(\d+)\]", text) or re.findall(r"https?://\S+", text)
        if not cited and not re.search(r"\b(?:could ?n[o']t|unable to|cannot) (?:verify|find|check)\b", text, re.I):
            faults.append("web answer cites none of the sources it opened")
    return faults


REPEAT_ASKED = re.compile(r"\brepeat|\btimes\b", re.I)      # the engine's own exemption (sml-server.c)


def tidy(text):
    """A reply without chat-template leaks: markers at the start are dropped, and
    a marker after the answer ends it (what follows is the model writing the
    next turn itself). Nothing else is touched."""
    t = (text or "").strip()
    m = ROLE_MARK.search(t)
    while m and m.start() == 0:
        t = t[m.end():].lstrip()
        m = ROLE_MARK.search(t)
    return t[:m.start()].rstrip() if m else t


def live_faults(text, request=""):
    """The hard faults a live local reply is sent back for once: empty, a loop,
    JSON or code that does not parse. Structure only -- never what it says."""
    return [f for f in rules(text, request)
            if f != "role markers in the reply"
            and not (f == "a repeating loop" and REPEAT_ASKED.search(request or ""))]


# A STAND-IN's refusal: the cloud model only answers when the person's own local
# model was cut for gibberish or is being fixed. If the stand-in then refuses,
# it would undo the model the person chose, so its refusal is not shown as the
# answer (agent.run). Never applied to the person's own model's replies.
REFUSAL = re.compile(r"^\s*(?:i'?m sorry,? (?:but )?|sorry,? (?:but )?|unfortunately,? )?"
                     r"(?:i (?:can(?:no|')t|cannot|won'?t|am not able to|'m not able to|am unable to|'m unable to)"
                     r" (?:help|assist|provide|comply|do that|fulfil|fulfill|create|write|answer|engage))", re.I)


def refusal(text):
    return bool(REFUSAL.search(text or ""))


def _json_of(text):
    t = text.strip()
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", t)
    if m:
        t = m.group(1).strip()
    for cand in (t, t[t.find("{"):t.rfind("}") + 1] if "{" in t else "", t[t.find("["):t.rfind("]") + 1] if "[" in t else ""):
        if not cand:
            continue
        try:
            return json.loads(cand)
        except ValueError:
            continue
    return None


def _node_check(body):
    f = tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False)
    try:
        f.write(body)
        f.close()
        r = subprocess.run([shutil.which("node"), "--check", f.name], capture_output=True, text=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode:
            lines = (r.stderr or r.stdout or "").strip().splitlines()
            err = next((ln for ln in lines if "Error" in ln), "")      # not node's closing version banner
            return (err or "invalid syntax")[:120]
    except (OSError, subprocess.TimeoutExpired):
        pass
    finally:
        Path(f.name).unlink(missing_ok=True)
    return ""


# The judge answers the question itself BEFORE it sees the answer under test, then
# compares the two. Asked for one word, a 1.5B judge passed "Nairobi is the
# capital of Uganda"; shown the answer while recalling the fact, it copied the
# answer as the truth (2026-10-05). Answering blind removes the anchor.
JUDGE_COMPARE = ("You compare two answers to the same question. Reply YES if the given answer agrees with "
                 "the reference answer (or, for an open request, fits the request as well as the reference does). "
                 "Reply NO if it contradicts the reference, is off topic or is nonsense. Reply with one word.")


class Judge:
    """A Checker model: its own small engine on its own port, started when needed."""

    def __init__(self, engine_cls, local_cls, path, port=8097, frame="own"):
        self.path, self.frame = path, frame          # own: the model's template; builtin: the engine's frame
        self.engine = engine_cls(port=port, ctx=1024, slots=1, threads=4)
        self.local = local_cls(self.engine.url)

    def up(self):
        if not self.engine.alive() or self.engine.model != self.path:
            if not self.engine.start(self.path):
                raise RuntimeError("the Checker model's engine did not start: " + " | ".join(self.engine.log[-2:]))
            self.local._chat = None

    def judge(self, question, answer):
        """(ok, word): the judge's verdict on one answer."""
        self.up()
        own = self.frame == "own"
        ref, _ = self.local.chat([{"role": "user", "content": f"{question}\nAnswer in a few words."}],
                                 max_tokens=32, cache=False, temperature=0, render=own)
        ref = (ref.get("content") or "").strip()
        msgs = [{"role": "system", "content": JUDGE_COMPARE},
                {"role": "user", "content": f"Question: {question}\nReference answer: {ref}\nGiven answer: {answer}\n"
                                            f"Do they agree? YES or NO."}]
        msg, _ = self.local.chat(msgs, max_tokens=4, cache=False, temperature=0, render=own)
        word = (msg.get("content") or "").strip().upper()
        if word.startswith("YES"):
            return True, "PASS"
        if word.startswith("NO"):
            return False, "FAIL"
        return None, word[:20] or "(nothing)"

    def stop(self):
        self.engine.stop()


def configure(checker, path):
    """Set a Checker model up for the job, as the built-in Checker was: the judge
    answers blind, then compares (a judge shown the answer copied it), and it is
    asked in whichever frame it reads best -- its own template or the engine's.
    Examined on the 12-answer exam (the pass mark) and the held-out extra set.
    Returns {frame, exam_ok, exam_right, right, of, accuracy, rows}."""
    best = None
    for frame in ("own", "builtin"):
        cand = Judge(checker.engine_cls, checker.local_cls, path, port=8096, frame=frame)
        try:
            rep = exam(cand, EXAM)
            more = exam(cand, EXAM_MORE)
        finally:
            cand.stop()
        r = {"frame": frame, "exam_ok": rep["ok"], "exam_right": rep.get("right", 0),
             "right": rep.get("right", 0) + more.get("right", 0), "of": len(EXAM) + len(EXAM_MORE),
             "rows": rep["rows"] + more["rows"], "why": rep.get("why", "")}
        r["accuracy"] = round(r["right"] / r["of"], 3)
        if best is None or (r["exam_ok"], r["right"]) > (best["exam_ok"], best["right"]):
            best = r
    return best


def model_revision(path):
    """Content identity for safe reuse of a held-out exam result."""
    digest = hashlib.sha256()
    with open(path, "rb") as model:
        for block in iter(lambda: model.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def review_reply(checker, question, draft, sources=None):
    """Return objective integrity faults only; learned Checker feedback is advisory.

    A Checker opinion must not rewrite or withhold a valid model answer. The
    deterministic rules still identify empty replies, role markers, loops,
    malformed requested JSON/code, and missing citations for sourced answers.
    """
    faults = []
    # rules
    r = list(rules(draft or ""))
    if r:
        faults += r
    # web sources: if sources were given but the draft names none
    if sources and draft:
        cited = bool(re.findall(r"\\[?\\d+\\]?", draft))
        if not cited:
            faults.append("the answer does not cite any sources")
    return faults


def exam(judge, items=None):
    """The judge on known-good and known-bad answers. Returns a report."""
    items = items or EXAM
    rows, right = [], 0
    t0 = time.time()
    for q, a, want in items:
        try:
            got, word = judge.judge(q, a)
        except Exception as e:
            return {"ok": False, "accuracy": 0.0, "rows": rows, "why": f"the Checker model failed: {str(e)[:160]}"}
        hit = got is want
        right += hit
        rows.append({"q": q, "a": a, "want": "PASS" if want else "FAIL", "got": word, "ok": hit})
    acc = right / len(items)
    ok = acc >= PASS_MARK
    return {"ok": ok, "accuracy": round(acc, 3), "right": right, "of": len(items), "rows": rows,
            "seconds": round(time.time() - t0, 1),
            "why": "" if ok else f"sorted {right} of {len(items)} exam answers; it needs {round(PASS_MARK * len(items))}"}


class Checker:
    """What the app holds: the rules always, a judge when one passed its exam."""

    def __init__(self, cfg, engine_cls, local_cls, save):
        self.cfg, self.engine_cls, self.local_cls, self.save = cfg, engine_cls, local_cls, save
        self.judge = None
        self.trial = {"state": "idle"}
        self.trainer = None
        c = cfg.get("checker") or {}
        if c.get("path") and Path(c["path"]).is_file():
            self.judge = Judge(engine_cls, local_cls, c["path"], frame=c.get("frame", "own"))

    def status(self):
        c = self.cfg.get("checker") or {}
        return {"judge": c if self.judge else None,
                "trial": self.trainer.state if self.trainer and self.trial.get("state") == "training" else self.trial,
                "rules": ["empty reply", "role markers", "repeating loops", "JSON asked for", "code that does not parse",
                          "tool-call arguments", "web answers without sources"]}

    @staticmethod
    def source(path):
        """The original weights a Checker model was quantized from (needed to train it), or None."""
        from .paths import HOME
        try:
            reg = json.loads((HOME / "models.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            reg = {}
        src = (reg.get(path) or reg.get(str(Path(path))) or {}).get("source_dir")
        for cand in ([src] if src else []) + [str(HOME / "originals" / Path(path).stem)]:
            if cand and (Path(cand) / "config.json").exists():
                return cand
        return None

    def _use(self, path, rep):
        if self.judge:
            self.judge.stop()
        self.judge = Judge(self.engine_cls, self.local_cls, path, frame=rep.get("frame", "own"))
        self.cfg["checker"] = {"path": path, "name": Path(path).stem, "accuracy": rep["accuracy"],
                               "right": rep["right"], "of": rep["of"], "frame": rep.get("frame", "own"),
                               "exam_revision": EXAM_REVISION, "exam_ok": bool(rep.get("exam_ok")),
                               "model_revision": rep.get("model_revision") or model_revision(path),
                               "t": time.time()}
        self.save()

    def install(self, path, train=False, orch=None, training_mode="cpu"):
        """Set up a new judge in the background and install it only if it passes
        its exam (else the one before stays). train=True first teaches it the
        Checker's job (teach.py): the trained and the original are both examined
        and the better one that passes is kept."""
        if self.trial.get("state") in ("running", "training", "paused", "stopping"):
            raise RuntimeError("a Checker is already being set up")
        if not Path(path).is_file() or not path.lower().endswith(".ssf"):
            raise RuntimeError("choose a Sheep model file (.ssf) to use as the Checker")
        if train:
            src = self.source(path)
            if not src:
                raise RuntimeError("training needs the model's original weights, and this file was not made by the "
                                   "Sheep Quantizer here. Quantize it from its original first, or check it as it is.")
            from .teach import Trainer

            def done(best, rep):
                if rep["exam_ok"]:
                    self._use(best, rep)
            self.trainer = Trainer(self, orch, path, src, done, training_mode=training_mode)
            self.trial = {"state": "training", "path": path, "t": time.time()}
            self.trainer.start()
            return self.trainer.state
        self.trial = {"state": "running", "path": path, "t": time.time()}

        def go():
            lent = False
            try:
                if orch:
                    lent = orch.lend_gpu("examining a Checker", 2.0, wait=True, priority=8)
                rep = configure(self, path)
            except Exception as e:
                self.trial = {"state": "rolled back", "path": path, "why": str(e)[:200]}
                return
            finally:
                if lent:
                    orch.give_back_gpu()
            if rep["exam_ok"]:
                self._use(path, rep)
                self.trial = {"state": "installed", **rep, "path": path}
            else:
                self.trial = {"state": "rolled back", **rep, "path": path}
        threading.Thread(target=go, daemon=True).start()
        return self.trial

    def remove(self):
        if self.judge:
            self.judge.stop()
        self.judge = None
        self.cfg.pop("checker", None)
        self.save()

    def check(self, question, text, request="", calls=None, sources=None):
        """Run hard integrity rules and return learned Checker feedback separately."""
        faults = rules(text, request or question, calls, sources)
        verdict = None
        if not faults and self.judge and text and not calls:
            try:
                _ok, word = self.judge.judge(question, text)
                verdict = word
            except Exception as e:
                verdict = f"judge unavailable: {str(e)[:80]}"
        return faults, verdict

    def rest(self):
        """Free the judge's memory between checks."""
        if self.judge:
            self.judge.stop()
