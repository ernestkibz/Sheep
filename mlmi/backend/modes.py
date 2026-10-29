"""The two modes: who answers each step.

  Mode 2 -- SHEEP: the local model through sml-server, and nothing leaves the machine.
  Mode 1 -- SHEEP + Nebius: the blend. Nebius (Token Factory, NVIDIA
            models) plans a job and takes the steps the local model cannot;
            SHEEP runs the routine tool calls, so file contents stay here
            unless a step is sent out -- and every byte sent out is logged.

Routing rules and the Nebius settings come from tooling/sml_bridge.py, the
bridge that already decides local-versus-cloud for chat; they are imported,
not copied, so the two cannot drift apart.
"""
import json
import os
import sys
import threading
import time
import types
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tooling"))
from sml_bridge import Bridge, note_route, pin_route, route_before  # noqa: E402


class LLMError(Exception):
    pass


def _post(url, body, headers=None, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise LLMError(f"{e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    except (urllib.error.URLError, OSError) as e:
        raise LLMError(str(e)[:300])


def _one_call_each(messages):
    """An assistant turn with several tool calls, followed by their results, as
    one call and its result at a time. Llama 3.2's own template refuses a turn
    with more than one call ("only supports single tool-calls"); the render then
    fell back to the engine's built-in frame, which changed the whole prompt and
    threw away the cached conversation (a 4-call step on Sheep + Nemotron,
    2026-09-30: step two re-read 2334 tokens, 60 s)."""
    out, i = [], 0
    while i < len(messages):
        m = messages[i]
        calls = m.get("tool_calls") or []
        if m.get("role") == "assistant" and len(calls) > 1:
            results = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                results.append(messages[j])
                j += 1
            for k, c in enumerate(calls):
                out.append({"role": "assistant", "content": m.get("content") if k == 0 else None, "tool_calls": [c]})
                if k < len(results):
                    out.append(results[k])
            out.extend(results[len(calls):])
            i = j
            continue
        out.append(m)
        i += 1
    return out


class Local:
    """The SHEEP engine. Greedy, thinking off: an agent step wants the call, not an essay."""
    name = "sheep"

    def __init__(self, url):
        self.url = url.rstrip("/")
        self._chat = None          # the model's own chat block, fetched once
        self.ctx = 8192            # the engine's context, set by the orchestrator's plan
        self.inflight = 0          # requests waiting on the engine: a silent engine with none is hung, not busy
        self._count = threading.Lock()
        self.retry_device = None   # installed by State: retry only local device failures while Auto is selected
        self.pace = lambda: 0      # installed by the orchestrator: the heat guard's rest per written token, in ms

    def chat_block(self):
        """The model's OWN template and stop tokens, as the builder stored them in
        the file (STANDARDS.md section 7). {} when the file has none."""
        if self._chat is None:
            try:
                with urllib.request.urlopen(self.url + "/v1/template", timeout=10) as r:
                    self._chat = json.load(r)
            except (urllib.error.URLError, OSError, ValueError):
                return {}
        return self._chat

    def render(self, messages, tools=None, max_tokens=768):
        """The prompt exactly as the model was trained to see it: its own Jinja
        template, rendered in a sandbox the way transformers renders it. None when
        there is no template or it cannot render these messages -- the server's
        built-in frames are used then, and the reason is logged.

        Some templates write the tool list as indented JSON (Llama 3.2: indent 4);
        24 tools came to ~21K characters, past an 8K context with room for the
        reply. When the prompt would not fit, the same template renders the same
        JSON without the indentation: the wording is the model's own, only the
        whitespace changes."""
        c = self.chat_block()
        if not c.get("template"):
            return None
        try:
            import datetime
            import json as _json
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            def _raise(m):
                raise ValueError(m)

            def build(compact):
                env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
                env.globals["raise_exception"] = _raise
                env.globals["strftime_now"] = lambda f: datetime.datetime.now().strftime(f)
                if compact:
                    env.filters["tojson"] = lambda v, indent=None, **kw: _json.dumps(v, ensure_ascii=False)
                return env.from_string(c["template"]).render(
                    messages=messages, tools=tools or None, add_generation_prompt=True,
                    bos_token=c.get("bos_token", ""), eos_token=c.get("eos_token", ""),
                    enable_thinking=False)
            # Decided from the tool list alone, so the same app always renders the
            # same prefix and the engine's cache reuses it (the warm-up and the
            # first step must match): compact when the indented list would take
            # more than a quarter of the context (~3 characters a token for JSON).
            compact = bool(tools) and len(_json.dumps(tools, indent=4)) / 3 > self.ctx / 4
            messages = _one_call_each(messages)
            # A template with a thinking switch in the system turn (NVIDIA Nemotron: "detailed
            # thinking off") writes it only when there is no system message; Sheep always sends
            # one, so the model reasoned until it ran out of tokens (2026-10-21)
            if "detailed thinking" in c["template"] and messages and messages[0].get("role") == "system" \
                    and "detailed thinking" not in str(messages[0].get("content") or ""):
                messages = [dict(messages[0], content="detailed thinking off\n\n" + str(messages[0].get("content") or ""))] \
                    + list(messages[1:])
            return build(compact)
        except Exception as e:          # a template this message shape breaks
            print(f"template: falling back to built-in frames ({type(e).__name__}: {str(e)[:120]})")
            return None

    def _stream(self, body, on_delta):
        """POST with "stream": true; returns the same shape as a whole reply."""
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(dict(body, stream=True)).encode(),
                                     headers={"Content-Type": "application/json"})
        text, calls, finish, usage, sml = [], [], "", {}, {}
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    ev = json.loads(line[6:])
                except ValueError:
                    continue
                if "usage" in ev:
                    usage, sml = ev["usage"], ev.get("sml", {})
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content"):
                        text.append(d["content"])
                        on_delta(d["content"])
                    calls += d.get("tool_calls") or []
                    finish = ch.get("finish_reason") or finish
        msg = {"role": "assistant", "content": "".join(text) or None}
        if calls:
            msg["tool_calls"] = calls
        return {"choices": [{"message": msg, "finish_reason": finish}], "usage": usage, "sml": sml}

    def chat(self, messages, tools=None, max_tokens=768, render=True, temperature=0, top_p=1.0, cache=True,
             verify=True, on_delta=None, on_retry=None):
        """render=False uses the engine's built-in frames, and temperature/top_p the
        model's own sampling: both are the orchestrator's choices (the Egypt doctor)."""
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                "enable_thinking": False}
        pace = self.pace()
        if pace:
            body["pace_ms"] = pace         # hot: the engine rests a little after each token instead of stopping
        if tools:
            body["tools"] = tools
        if not cache:
            body["cache"] = False          # a fresh answer, never a remembered one
        if not verify:
            body["verify"] = False         # the turn quotes sources: no arithmetic check on their numbers
        prompt = self.render(messages, tools, max_tokens) if render else None
        if prompt:
            body["prompt"] = prompt
        t0 = time.perf_counter()
        fallbacks = []
        while True:
            with self._count:
                self.inflight += 1
            try:
                if on_delta and not tools:
                    # The UI clears a partial draft when the local device is retried.
                    r = self._stream(body, on_delta)
                else:
                    r = _post(self.url + "/v1/chat/completions", body)
                if "choices" not in r:
                    raise LLMError(str(r.get("error", r))[:300])
            except (LLMError, urllib.error.URLError, OSError) as exc:
                reason = str(exc)[:300]
                fallback = self.retry_device(reason) if self.retry_device and len(fallbacks) < 2 else None
                if not fallback:
                    if isinstance(exc, LLMError):
                        raise
                    raise LLMError(reason) from exc
                fallbacks.append(str(fallback))
                if on_delta:
                    on_delta(None)
                if on_retry:
                    on_retry(str(fallback))
                continue
            finally:
                with self._count:
                    self.inflight -= 1
            break
        msg = r["choices"][0]["message"]
        sml = r.get("sml", {})
        return msg, {"route": "local", "ms": (time.perf_counter() - t0) * 1000,
                     "tokens": r.get("usage", {}), "sml": sml, "finish": r["choices"][0].get("finish_reason", ""),
                     "template": "model's own" if prompt else "built-in",
                     "degenerate": sml.get("degenerate", False),
                     "device_fallback": fallbacks}

    def stats(self):
        try:
            with urllib.request.urlopen(self.url + "/stats", timeout=5) as r:
                return json.load(r)
        except (urllib.error.URLError, OSError, ValueError):
            return {}


class Cloud:
    """NVIDIA models on Nebius Token Factory. The key lives in the environment only."""
    name = "nebius"

    def __init__(self):
        self.cfg = Bridge(types.SimpleNamespace(local="", min_margin=0.0, vision=""))
        self.sent = []                # the "sent to cloud" log the UI shows
        self.lock = threading.Lock()
        self.models = []

    def ready(self):
        return self.cfg.cloud_ready()

    def reload(self):
        """Read the key, endpoint and model again (set in Settings › Accounts)."""
        self.cfg = Bridge(types.SimpleNamespace(local="", min_margin=0.0, vision=""))

    @property
    def model(self):
        return self.cfg.model

    @property
    def provider(self):
        """Where the calls really go, said plainly (OpenRouter stands in while
        Nebius Token Factory is not open to Uganda, 2026-09-30; pinned to
        Nebius with fallbacks off, 2026-10-13)."""
        b = self.cfg.base.lower()
        return ("Nebius Token Factory" if "nebius" in b else
                "Nebius via OpenRouter (pinned)" if "openrouter" in b else b)

    def list_models(self):
        if not self.ready():
            return []
        try:
            req = urllib.request.Request(self.cfg.base + "/models", headers={"Authorization": "Bearer " + self.cfg.key})
            with urllib.request.urlopen(req, timeout=20) as r:
                self.models = sorted(m.get("id", "") for m in json.load(r).get("data", []))
        except (urllib.error.URLError, OSError, ValueError):
            pass
        return self.models

    def chat(self, messages, tools=None, max_tokens=4096, why=""):
        # Nemotron 3 reasons before it answers (362 of 396 tokens on a plan):
        # a small budget cuts the answer off, so the budget is generous
        if not self.ready():
            raise LLMError("Mode 1 needs NEBIUS_API_KEY in the environment")
        body = pin_route(self.cfg.base, {"model": self.cfg.model, "messages": messages,
                                         "max_tokens": max_tokens, "temperature": 0})
        if tools:
            body["tools"] = tools
        raw = json.dumps(messages)
        with self.lock:
            entry = {"t": time.time(), "why": why, "model": self.cfg.model, "chars": len(raw),
                     "tools": len(tools or []),
                     "preview": [(m.get("role"), str(m.get("content") or m.get("tool_calls") or "")[:160])
                                 for m in messages[-4:]]}
            self.sent.append(entry)
            self.sent = self.sent[-200:]
        t0 = time.perf_counter()
        try:
            r = _post(self.cfg.base + "/chat/completions", body, {"Authorization": "Bearer " + self.cfg.key}, timeout=60)   # a stalled cloud gives way to the laptop within a minute
        except LLMError as e:
            note_route(self.cfg.base, body, error=str(e))
            raise
        if "choices" not in r:
            note_route(self.cfg.base, body, r, error=str(r.get("error", r)))
            raise LLMError(str(r.get("error", r))[:300])
        answered = note_route(self.cfg.base, body, r)
        entry["provider"] = answered
        entry["gen"] = str(r.get("id") or "")          # the provider's own id for this call (routing evidence)
        return r["choices"][0]["message"], {"route": "cloud", "ms": (time.perf_counter() - t0) * 1000,
                                            "tokens": r.get("usage", {}), "model": self.cfg.model,
                                            "provider": answered}


def needs_plan(text, apps):
    """Mode 1 plans a job in the cloud when it is multi-step or beyond a small model."""
    route, why = route_before(text)
    if route == "cloud":
        return True, why
    words = len(text.split())
    if words > 60 or len(apps) > 2 or sum(text.lower().count(k) for k in (" then ", " and ", ";", "\n")) >= 3:
        return True, "a multi-step job"
    return False, why
