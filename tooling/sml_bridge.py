#!/usr/bin/env python3
"""Sheep bridge: answer on this machine, and go to a large model only when the turn needs it.

    python tooling/sml_bridge.py --local http://127.0.0.1:8099 --port 8080
    open http://127.0.0.1:8080

Cloud settings come from the environment, never from the command line or a file in
the repository, so a key cannot end up in a commit or a screen recording:

    NEBIUS_API_KEY    required for cloud turns; without it every turn stays local
                      and the interface says so
    NEBIUS_BASE_URL   default https://api.tokenfactory.nebius.com/v1
    NEBIUS_MODEL      default nvidia/nemotron-3-super-120b-a12b
    NEBIUS_VISION_MODEL  the vision model's id, copied from the Token Factory
                      Models tab (e.g. NVIDIA Nemotron Nano 2 VL). Not guessed:
                      without it, a turn with an image says what to set.

IMAGES. This engine has no vision tower yet, so a turn carrying an image goes
to the vision model on Nebius and says so. Earlier images are replaced by
"[image]" before a conversation reaches the local engine, which reads text.

HOW A TURN IS ROUTED, AND WHY

Measured on Llama-3.2-1B through this engine (2026-09-23), the obvious signal does
not work. The engine reports, per token, the gap between its top two logits. Mean
confidence did NOT separate right answers from wrong ones: the model was more sure
of a hallucinated Tour de France winner (mean gap 5.67) than of a correct "Hola"
(2.01). A small model is fluent and confidently wrong about facts, so asking it how
sure it is cannot catch the error that matters most.

So routing is two layers, each doing the job it can actually do:

  1. BEFORE the turn, by what is being asked. A 1B model is good at language --
     rewording, summarising what you gave it, conversation, short writing -- and
     bad at knowledge and long reasoning. Questions of fact (who, when, how many,
     measurements), multi-step reasoning, code, and very long prompts go to the
     large model. Everything else stays here.
  2. AFTER a local turn, by the model's own evidence -- OFF by default, and here is
     why. The idea: if the WORST token of the answer was a coin toss (top-two gap
     under --min-margin), re-ask the large model. On eight short factual answers it
     looked right: the one wrong boiling point had a 0.00 gap, correct answers
     0.18 and up. Then it met open-ended writing, and a correct polite rewrite
     scored 0.02 and "say something kind" 0.01 -- because in free text a near tie
     is usually two good words ("Could you" / "Would you"), not a wrong answer.
     The signal cannot tell a synonym from an error, so turning it on by default
     would send good answers to the cloud. It stays as an opt-in flag for
     short factual use, where it was measured to help.

  "/cloud ..." and "/local ..." at the start of a message force the route.
"""
import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "..", "sml",
                   "sml-cli" + (".exe" if os.name == "nt" else ""))
UI = os.path.join(HERE, "..", "sml", "ui", "index.html")

KNOWLEDGE = re.compile(
    r"\bwho (won|was|is|invented|discovered|wrote|founded|painted|directed)\b"
    r"|\b(what|which) year\b|\bwhen (did|was|were|is)\b"
    r"|\bhow (many|much|tall|far|old|long|big|heavy|fast)\b"
    r"|\b(population|boiling point|melting point|distance|founded|born|died|capital of"
    r"|price of|stock|latest|current|today|news)\b")
REASONING = re.compile(
    r"\bstep[- ]by[- ]step\b|\bprove\b|\bderive\b|\bexplain why\b|\bcompare\b|\banaly[sz]e\b"
    r"|\bplan\b|\bpros and cons\b|\btrade-?offs?\b|\bwhy does\b|\bwhat would happen\b")
CODE = re.compile(
    r"```|\b(write|implement|fix|debug|refactor)\b.*\b(code|function|script|class|program|query)\b"
    r"|\b(python|javascript|typescript|rust|golang|java|c\+\+|sql|regex)\b")
LONG_CHARS = 1200

# OpenRouter stands in while Token Factory is not open to Uganda. Nebius Support
# (2026-10-13): a call counts as Nebius only when it is pinned to the Nebius
# provider with fallbacks off, so a model Nebius does not serve on OpenRouter
# fails loudly instead of quietly answering from another provider.
OPENROUTER_PIN = {"only": ["nebius"], "allow_fallbacks": False}
# one line per cloud call -- where it was sent, how it was pinned, and which
# provider OpenRouter says answered; never the messages themselves
EVIDENCE = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                        "Sheep", "routing_evidence.jsonl")


def pin_route(base, body):
    """Add the Nebius pin to a chat body bound for OpenRouter."""
    if "openrouter" in base.lower():
        body["provider"] = dict(OPENROUTER_PIN)
        # Nemotron thinks before it answers (362 of 396 tokens on one plan): a short think is most of the
        # wait saved, and the plans stay right. NEBIUS_REASONING=medium or high gives it more room.
        body.setdefault("reasoning", {"effort": os.environ.get("NEBIUS_REASONING", "low")})
    return body


def note_route(base, body, answer=None, error=""):
    """Append one call to the routing evidence; returns the provider that answered."""
    answer = answer or {}
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "endpoint": base,
           "model_asked": body.get("model"), "provider_sent": body.get("provider"),
           "id": answer.get("id"), "provider_answered": answer.get("provider"),
           "model_answered": answer.get("model"), "error": error[:300]}
    try:
        os.makedirs(os.path.dirname(EVIDENCE), exist_ok=True)
        with open(EVIDENCE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass
    return row["provider_answered"]


def route_before(text):
    """(route, reason) from what is being asked, before any model runs."""
    t = text.strip().lower()
    if t.startswith("/cloud"):
        return "cloud", "forced with /cloud"
    if t.startswith("/local"):
        return "local", "forced with /local"
    if len(text) > LONG_CHARS:
        return "cloud", f"long prompt ({len(text)} characters)"
    if CODE.search(t):
        return "cloud", "code"
    if REASONING.search(t):
        return "cloud", "multi-step reasoning"
    if KNOWLEDGE.search(t):
        return "cloud", "a question of fact -- small models invent these"
    return "local", "language task -- a small model does these well"


def text_of(content):
    """The text of a message whose content is a string or an OpenAI parts list."""
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def image_of(content):
    """The first image in an OpenAI-style message, as PIL, or None. Accepts a
    data: URL (what a browser upload sends) or an http(s) link."""
    if not isinstance(content, list):
        return None
    for p in content:
        if isinstance(p, dict) and p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            try:
                from PIL import Image
                if url.startswith("data:"):
                    raw = base64.b64decode(url.split(",", 1)[1])
                else:
                    with urllib.request.urlopen(url, timeout=20) as r:
                        raw = r.read()
                return Image.open(io.BytesIO(raw))
            except (ValueError, OSError, IndexError, urllib.error.URLError):
                return None
    return None


def local_vision(ssf, image, question, max_tokens):
    """Answer an image turn with the local vision model: our own preprocessing
    (tooling/sml_image.py), then the engine frames the turn and stops at the
    model's end-of-utterance. Returns (answer, seconds)."""
    import sml_image
    fd, px = tempfile.mkstemp(suffix=".f32")
    os.close(fd)
    try:
        sml_image.prepare(image).tofile(px)
        t0 = time.perf_counter()
        r = subprocess.run([CLI, ssf, "--image", px, "--ask", question or "Describe this image.",
                            "--gen", str(max(1, min(int(max_tokens), 200))),
                            "--threads", str(os.cpu_count() or 6)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=600)
        dt = time.perf_counter() - t0
    finally:
        os.remove(px)
    out = r.stdout
    i = out.find('\n  "')
    j = out.find('"\n\ngenerate', i)
    if r.returncode != 0 or i < 0 or j < 0:
        raise OSError((r.stderr or out)[-160:].strip() or "the local vision model gave no answer")
    return out[i + 4:j].strip(), dt


def has_image(content):
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in content)


def text_only(messages):
    """The conversation as the local engine can read it: images become a marker."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = (text_of(c) + (" [image]" if has_image(c) else "")).strip()
        out.append({"role": m.get("role", "user"), "content": c or ""})
    return out


def strip_force(messages):
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            c = m.get("content") or ""
            if isinstance(c, list):
                break
            for tag in ("/cloud", "/local"):
                if c.strip().lower().startswith(tag):
                    m["content"] = c.strip()[len(tag):].lstrip()
            break
    return out


def sse(obj):
    return ("data: " + json.dumps(obj) + "\n\n").encode("utf-8")


def piece(text):
    return sse({"choices": [{"index": 0, "delta": {"content": text}}]})


def relay(resp, write, meta=None):
    """Forward OpenAI-style SSE content pieces; return (characters, pieces).
    A dict passed as meta collects the stream's id, provider and model."""
    chars = pieces = 0
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        try:
            chunk = json.loads(body)
            if meta is not None:
                for k in ("id", "provider", "model"):
                    if chunk.get(k) and not meta.get(k):
                        meta[k] = chunk[k]
            delta = chunk["choices"][0].get("delta", {})
        except (ValueError, KeyError, IndexError, AttributeError):
            continue
        text = delta.get("content") or ""
        if text:
            write(piece(text))
            chars += len(text)
            pieces += 1
    return chars, pieces


class Bridge:
    def __init__(self, a):
        self.local = a.local.rstrip("/")
        self.min_margin = a.min_margin
        self.key = os.environ.get("NEBIUS_API_KEY", "")
        self.base = os.environ.get("NEBIUS_BASE_URL",
                                   "https://api.tokenfactory.nebius.com/v1").rstrip("/")
        self.model = os.environ.get("NEBIUS_MODEL", "nvidia/nemotron-3-super-120b-a12b")
        self.vision = os.environ.get("NEBIUS_VISION_MODEL", "")
        # a Sheep vision model on THIS machine; when set, images never leave it
        self.vision_local = a.vision or os.environ.get("SHEEP_VISION_MODEL", "")
        self.stats = {"route": "", "reason": "", "cloud_ms": 0.0, "cloud_chars": 0,
                      "local_turns": 0, "cloud_turns": 0, "escalations": 0,
                      "cloud_errors": 0, "last_error": ""}

    def cloud_ready(self):
        return bool(self.key)

    def ask_local(self, messages, max_tokens, write):
        body = json.dumps({"model": "sml", "stream": True, "max_tokens": max_tokens,
                           "messages": text_only(messages)}).encode()
        req = urllib.request.Request(self.local + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            relay(r, write)
        self.stats["local_turns"] += 1
        try:
            with urllib.request.urlopen(self.local + "/stats", timeout=10) as r:
                return json.load(r)
        except (urllib.error.URLError, ValueError):
            return {}

    def ask_cloud(self, messages, max_tokens, write, model=None):
        body = pin_route(self.base, {"model": model or self.model, "stream": True,
                                     "max_tokens": max_tokens, "messages": messages})
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.key,
            "Accept": "text/event-stream"})
        t0 = time.perf_counter()
        meta = {}
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                chars, _ = relay(r, write, meta)
        except (urllib.error.URLError, OSError) as e:
            detail = e.read().decode("utf-8", "replace") if isinstance(e, urllib.error.HTTPError) else ""
            note_route(self.base, body, error=f"{e} {detail}".strip())
            raise
        self.stats["cloud_provider"] = note_route(self.base, body, meta)
        self.stats["cloud_ms"] = (time.perf_counter() - t0) * 1000.0
        self.stats["cloud_chars"] = chars
        self.stats["cloud_turns"] += 1

    def turn(self, req, write):
        messages = req.get("messages") or []
        max_tokens = int(req.get("max_tokens") or 512)
        last_raw = next((m.get("content") or "" for m in reversed(messages)
                         if m.get("role") == "user"), "")
        last = text_of(last_raw)
        if has_image(last_raw) and self.vision_local:
            reason = "an image -- answered on this machine by the local vision model"
            self.stats.update(route="local", reason=reason)
            write(sse({"sheep_route": {"route": "local", "reason": reason,
                                       "model": os.path.basename(self.vision_local)}}))
            img = image_of(last_raw)
            if img is None:
                write(piece("[the image could not be read]"))
                return
            try:
                ans, dt = local_vision(self.vision_local, img, last, max_tokens)
                self.stats["local_turns"] += 1
                self.stats["vision_s"] = round(dt, 2)
                write(piece(ans))
            except (OSError, subprocess.SubprocessError) as e:
                write(piece(f"[local vision failed: {str(e)[:100]}]"))
            return
        if has_image(last_raw):
            if self.key and self.vision:
                reason = "an image -- this machine has no vision model yet"
                self.stats.update(route="cloud", reason=reason)
                write(sse({"sheep_route": {"route": "cloud", "reason": reason,
                                           "model": self.vision}}))
                try:
                    self.ask_cloud(messages, max_tokens, write, model=self.vision)
                except (urllib.error.URLError, OSError, ValueError) as e:
                    self.stats["cloud_errors"] += 1
                    self.stats["last_error"] = str(e)[:160]
                    write(piece(f"[the vision model could not be reached: {str(e)[:80]}]"))
                return
            need = "NEBIUS_API_KEY" if not self.key else "NEBIUS_VISION_MODEL"
            reason = f"an image, and no vision model is configured ({need} is not set)"
            self.stats.update(route="local", reason=reason)
            write(sse({"sheep_route": {"route": "local", "reason": reason, "model": "sheep"}}))
            write(piece("I can't see images on this machine yet. Set " + need +
                        " to send pictures to a vision model on Nebius."))
            return
        route, reason = route_before(last)
        messages = strip_force(messages)

        if route == "cloud" and not self.cloud_ready():
            route, reason = "local", reason + "; cloud not configured (no NEBIUS_API_KEY)"
        self.stats.update(route=route, reason=reason)
        write(sse({"sheep_route": {"route": route, "reason": reason,
                                   "model": self.model if route == "cloud" else "sheep"}}))

        if route == "cloud":
            try:
                self.ask_cloud(messages, max_tokens, write)
                return
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.stats["cloud_errors"] += 1
                self.stats["last_error"] = str(e)[:160]
                reason = f"cloud failed ({str(e)[:60]}); answered here"
                self.stats.update(route="local", reason=reason)
                write(sse({"sheep_route": {"route": "local", "reason": reason, "model": "sheep"}}))

        local = self.ask_local(messages, max_tokens, write)
        worst = local.get("last_margin_min")
        # a negative gap means "not measured" (a cached turn decodes nothing)
        if (worst is not None and 0 <= worst < self.min_margin and self.cloud_ready()
                and not last.strip().lower().startswith("/local")):
            reason = f"local answer had a coin-toss token (gap {worst:.2f}); re-asked"
            self.stats["escalations"] += 1
            self.stats.update(route="cloud", reason=reason)
            write(sse({"sheep_route": {"route": "cloud", "reason": reason,
                                       "model": self.model, "escalated": True}}))
            try:
                self.ask_cloud(messages, max_tokens, write)
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.stats["cloud_errors"] += 1
                self.stats["last_error"] = str(e)[:160]
                write(piece(f"\n[cloud unreachable: {str(e)[:80]} -- the local answer stands]"))


def make_handler(bridge):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send_body(self, code, ctype, data):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                try:
                    self.send_body(200, "text/html; charset=utf-8", open(UI, "rb").read())
                except OSError:
                    self.send_body(404, "text/plain", b"sml/ui/index.html not found")
            elif self.path.startswith("/stats"):
                try:
                    with urllib.request.urlopen(bridge.local + "/stats", timeout=10) as r:
                        s = json.load(r)
                except (urllib.error.URLError, ValueError):
                    s = {"model": "local engine not reachable"}
                s.update(bridge.stats)
                s["cloud_configured"] = bridge.cloud_ready()
                s["cloud_model"] = bridge.model
                s["vision_model"] = (os.path.basename(bridge.vision_local) + " (local)"
                                     if bridge.vision_local else
                                     bridge.vision if bridge.cloud_ready() else "")
                self.send_body(200, "application/json", json.dumps(s).encode())
            elif self.path.startswith("/health"):
                self.send_body(200, "application/json", b'{"status":"ok","engine":"sheep-bridge"}')
            else:
                self.send_body(404, "application/json", b'{"error":"not found"}')

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                self.send_body(404, "application/json", b'{"error":"not found"}')
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                self.send_body(400, "application/json", b'{"error":"bad json"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()

            def write(b):
                self.wfile.write(b)
                self.wfile.flush()

            try:
                bridge.turn(req, write)
            except (urllib.error.URLError, OSError) as e:
                write(piece(f"[engine unreachable: {str(e)[:100]}]"))
            write(b"data: [DONE]\n\n")
            self.close_connection = True

    return H


def check_models(bridge):
    """Ask Token Factory which models exist and say whether the configured ones
    are among them. The docs publish no Nemotron ids, so a guessed id would only
    fail at the first cloud turn -- on camera. Lists the NVIDIA models on offer."""
    try:
        req = urllib.request.Request(bridge.base + "/models",
                                     headers={"Authorization": "Bearer " + bridge.key})
        with urllib.request.urlopen(req, timeout=20) as r:
            ids = sorted(m.get("id", "") for m in json.load(r).get("data", []))
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"  could not list Token Factory models: {str(e)[:100]}")
        return
    for name, mid in (("NEBIUS_MODEL", bridge.model), ("NEBIUS_VISION_MODEL", bridge.vision)):
        if mid:
            print(f"  {name} {mid}: {'available' if mid in ids else 'NOT in the model list'}")
    nv = [i for i in ids if "nvidia" in i.lower() or "nemotron" in i.lower()]
    print("  NVIDIA models on Token Factory: " + (", ".join(nv) if nv else "none listed"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--local", default="http://127.0.0.1:8099", help="sml-server address")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--vision", default="",
                    help="a Sheep vision model (.ssf): image turns are answered on this "
                         "machine instead of the cloud (or set SHEEP_VISION_MODEL)")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="opt-in: re-ask the cloud when the local answer's worst top-two gap is "
                         "below this (0 = off; see the docstring for why it is off)")
    a = ap.parse_args()
    bridge = Bridge(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(bridge))
    print(f"sheep bridge on http://127.0.0.1:{a.port}  local {bridge.local}  "
          f"cloud {'ON: ' + bridge.model if bridge.cloud_ready() else 'OFF (no NEBIUS_API_KEY)'}")
    if bridge.cloud_ready():
        check_models(bridge)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
