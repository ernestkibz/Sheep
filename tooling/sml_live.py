"""Live voice loop: microphone -> Sheep -> spoken reply, on this laptop.

The pieces were measured before this existed (scratchpad/live/latency.py):
speech to text 338 ms for a 5 s utterance with tiny.en, the Windows voice
synthesises at 0.002x real time, voice activity costs 0.06 ms per 32 ms block.
What was missing was the loop and its end-to-end number, which this prints for
every turn: from the moment you stop speaking to the moment the reply starts.

    sml/sml-server.exe MODEL.ssf --port 8080 --threads 6 --ctx 2048   (one shell)
    python tooling/sml_live.py                                        (another)
    python tooling/sml_live.py --wav question.wav    # no microphone needed
    python tooling/sml_live.py --say "what is the capital of France"

The server keeps the conversation in its cache, so each turn only reads the new
words rather than the whole transcript again.
"""
import argparse
import json
import queue
import time
import urllib.request

import numpy as np

SR = 16000
BLOCK = 512          # 32 ms, the block size the detector expects


def now():
    return time.perf_counter()


class Voice:
    """Windows SAPI, spoken asynchronously so a reply can be cut off."""

    def __init__(self, enabled=True):
        self.v = None
        if not enabled:
            return
        import win32com.client
        self.v = win32com.client.Dispatch("SAPI.SpVoice")

    def say(self, text):
        if self.v is not None and text:
            self.v.Speak(text, 1)            # async

    def stop(self):
        if self.v is not None:
            self.v.Speak("", 3)              # async + purge what is queued

    def wait(self, ms=60000):
        if self.v is not None:
            self.v.WaitUntilDone(ms)


class Ears:
    """Speech to text small enough to keep up with a CPU."""

    def __init__(self, size="tiny.en", threads=4):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(size, device="cpu", compute_type="int8",
                              cpu_threads=threads)

    def hear(self, audio):
        segs, _ = self.m.transcribe(audio, beam_size=1, language="en",
                                    without_timestamps=True, vad_filter=False,
                                    condition_on_previous_text=False)
        return "".join(s.text for s in segs).strip()


class Detector:
    """Silero voice activity, one block at a time."""

    def __init__(self):
        from faster_whisper.vad import get_vad_model
        self.m = get_vad_model()

    def prob(self, block):
        return float(np.ravel(self.m(block.astype(np.float32)))[-1])


# how much of text past start ends on a finished sentence
def sentence_end(text, start):
    cut = start
    for i in range(start, len(text)):
        if text[i] in ".!?":
            cut = i + 1
    return cut


# Returns text, usage, and seconds until the first words were spoken. Given a
# voice, the reply is spoken sentence by sentence as it arrives, so the talking
# starts while the rest of the answer is still being written.
def ask(url, messages, max_tokens, voice=None, timeout=180):
    payload = {"messages": messages, "max_tokens": max_tokens}
    if voice is not None:
        payload["stream"] = True
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = now()
    first = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if "text/event-stream" not in (r.headers.get("Content-Type") or ""):
            out = json.loads(r.read().decode())
            ch = (out.get("choices") or [{}])[0]
            text = (ch.get("message") or {}).get("content") or ch.get("text") or ""
            return text.strip(), out.get("usage") or {}, now() - t0
        text, spoken, ntok = "", 0, 0
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                piece = json.loads(data)["choices"][0]["delta"].get("content") or ""
            except Exception:
                continue
            text += piece
            ntok += 1
            cut = sentence_end(text, spoken)
            if cut > spoken:
                say = text[spoken:cut].strip()
                spoken = cut
                if say and voice is not None:
                    if first is None:
                        first = now() - t0
                    voice.say(say)
        tail = text[spoken:].strip()
        if tail and voice is not None:
            if first is None:
                first = now() - t0
            voice.say(tail)
    took = first if first is not None else now() - t0
    return text.strip(), {"completion_tokens": ntok}, took


def trim_reply(text, max_sentences=2):
    """Keep whole sentences. The token limit otherwise cuts mid-word, and the cut
    text goes back into the transcript as if the model had said exactly that."""
    head = text.split("\n\n")[0].strip()
    kept, count = [], 0
    for ch in head:
        kept.append(ch)
        if ch in ".!?":
            count += 1
            if count >= max_sentences:
                break
    out = "".join(kept).strip()
    return out or head


def reply_turn(cfg, state, voice, text, t_end, t_asr):
    """One exchange, timed from the end of speech to the first spoken word."""
    state["messages"].append({"role": "user", "content": text})
    t0 = now()
    speaker = voice if cfg.stream else None
    answer, usage, t_first = ask(cfg.url, state["messages"], cfg.max_tokens, speaker)
    t_think = now() - t0
    if cfg.stream:
        # everything that arrived was spoken, so the transcript keeps all of it
        state["messages"].append({"role": "assistant", "content": answer})
        t_total = t0 + t_first - t_end
    else:
        answer = trim_reply(answer)
        state["messages"].append({"role": "assistant", "content": answer})
        voice.say(answer)
        t_total = now() - t_end
    ntok = usage.get("completion_tokens", 0)
    nprompt = usage.get("prompt_tokens", 0)
    print(f"  you  : {text}")
    print(f"  sheep: {answer}")
    print(f"  heard in {t_asr * 1000:.0f} ms, thought in {t_think * 1000:.0f} ms, "
          f"speaking after {t_total * 1000:.0f} ms "
          f"[{ntok} tokens out, {nprompt} in prompt]")
    return t_total


def load_wav(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)
    return audio


def listen(cfg, ears, det, voice, state):
    import sounddevice as sd

    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    on_blocks = max(1, int(round(cfg.speech_ms / 1000 * SR / BLOCK)))
    off_blocks = max(1, int(round(cfg.silence_ms / 1000 * SR / BLOCK)))
    pre_blocks = max(1, int(round(cfg.preroll_ms / 1000 * SR / BLOCK)))

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK, callback=cb):
        print("listening. speak when ready, ctrl+c to stop.")
        pre, voiced = [], []
        speaking = False
        run = quiet = turns = 0
        while turns < cfg.turns:
            block = q.get()
            p = det.prob(block)
            if not speaking:
                pre.append(block)
                if len(pre) > pre_blocks:
                    pre.pop(0)
                run = run + 1 if p > cfg.on else 0
                if run >= on_blocks:
                    speaking = True
                    voiced = list(pre)
                    if cfg.barge_in:
                        voice.stop()
            else:
                voiced.append(block)
                quiet = quiet + 1 if p < cfg.off else 0
                if quiet >= off_blocks:
                    t_end = now()
                    audio = np.concatenate(voiced)
                    speaking = False
                    run = quiet = 0
                    pre, voiced = [], []
                    if len(audio) / SR < cfg.min_speech_s:
                        continue
                    t0 = now()
                    said = ears.hear(audio)
                    t_asr = now() - t0
                    if not said:
                        continue
                    reply_turn(cfg, state, voice, said, t_end, t_asr)
                    turns += 1
                    if not cfg.barge_in:
                        voice.wait()
                        while not q.empty():      # do not hear ourselves
                            q.get()


def main():
    ap = argparse.ArgumentParser(description="talk to Sheep")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--asr", default="tiny.en", help="tiny.en, base.en or small")
    ap.add_argument("--asr-threads", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--turns", type=int, default=100)
    ap.add_argument("--system", default="You are Sheep, a live voice assistant. "
                                        "Reply in one or two short sentences.")
    ap.add_argument("--wav", help="turns from wav files, comma separated, no microphone")
    ap.add_argument("--say", help="one turn of typed text, no microphone")
    ap.add_argument("--no-speak", action="store_true", help="print, do not speak")
    ap.add_argument("--no-stream", dest="stream", action="store_false",
                    help="wait for the whole reply instead of speaking as it arrives")
    ap.add_argument("--barge-in", action="store_true",
                    help="let speech cut the reply off (needs headphones)")
    ap.add_argument("--on", type=float, default=0.5, help="speech starts above this")
    ap.add_argument("--off", type=float, default=0.35, help="silence below this")
    ap.add_argument("--speech-ms", type=int, default=96)
    ap.add_argument("--silence-ms", type=int, default=500)
    ap.add_argument("--preroll-ms", type=int, default=320)
    ap.add_argument("--min-speech-s", type=float, default=0.4)
    cfg = ap.parse_args()

    state = {"messages": [{"role": "system", "content": cfg.system}]}
    voice = Voice(not cfg.no_speak)

    if cfg.say:
        t0 = now()
        reply_turn(cfg, state, voice, cfg.say, t0, 0.0)
        voice.wait()
        return

    t0 = now()
    ears = Ears(cfg.asr, cfg.asr_threads)
    print(f"speech model {cfg.asr} ready in {now() - t0:.1f} s")

    if cfg.wav:
        for path in cfg.wav.split(","):
            audio = load_wav(path)
            t_end = now()
            t0 = now()
            said = ears.hear(audio)
            t_asr = now() - t0
            if not said:
                print(f"nothing heard in {path}")
                continue
            reply_turn(cfg, state, voice, said, t_end, t_asr)
            voice.wait()
        return

    det = Detector()
    try:
        listen(cfg, ears, det, voice, state)
    except KeyboardInterrupt:
        voice.stop()
        print("\nstopped.")


if __name__ == "__main__":
    main()
