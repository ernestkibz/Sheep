// Voice for every chat model: the model never needs to be a voice model.
//
//   Mic (dictation)  press, speak, press again: Whisper writes your words into the
//                    message box, ready to send (as dictation in other chat apps does).
//   Talk (live)      a spoken conversation: Sheep listens, notices when you stop
//                    talking, sends your words to the chat model, speaks the answer
//                    with this computer's voices, and listens again.
//
// Speech to text runs on this computer (Whisper, /api/transcribe); the answer is
// spoken by the system voices, so nothing leaves the machine.
"use strict";

(() => {
  const { $, toast, S, icon } = MLMI;
  const V = MLMI.voice = {};
  let rec = null, chunks = [], stream = null;

  async function mic() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("no microphone access here");
    return navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  }
  async function transcribe(blob) {
    const r = await fetch("/api/transcribe", { method: "POST", headers: { "Content-Type": blob.type || "audio/webm" }, body: blob });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || r.status);
    return (j.text || "").trim();
  }
  function record(s) {
    chunks = [];
    const r = new MediaRecorder(s, { mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "" });
    r.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    r.start(250);
    return r;
  }
  const stopped = (r) => new Promise((res) => { r.onstop = () => res(new Blob(chunks, { type: r.mimeType || "audio/webm" })); r.stop(); });

  // ------------------------------------------------------------ dictation
  V.dictate = async () => {
    const b = $("#micBtn");
    if (rec) {                                   // second press: stop and write it down
      const r = rec; rec = null;
      b.classList.remove("live"); b.classList.add("busy");
      const blob = await stopped(r);
      stream.getTracks().forEach((t) => t.stop());
      try {
        const text = await transcribe(blob);
        const box = $("#askText");
        box.value = (box.value ? box.value.trimEnd() + " " : "") + text;
        box.dispatchEvent(new Event("input"));
        box.focus();
        if (!text) toast("I did not catch any words", "bad");
      } catch (e) { toast(`speech to text: ${e.message}`, "bad"); }
      b.classList.remove("busy");
      return;
    }
    try { stream = await mic(); } catch (e) { return toast(`microphone: ${e.message}`, "bad"); }
    rec = record(stream);
    b.classList.add("live");
  };

  // ------------------------------------------------------------ speaking
  V.say = (text) => new Promise((res) => {
    const t = String(text || "").replace(/```[\s\S]*?```/g, " (code on screen) ").replace(/\[(\d+)\]/g, "").replace(/[*#`]/g, "").trim();
    if (!t || !window.speechSynthesis) return res();
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(t.slice(0, 2500));
    const vs = speechSynthesis.getVoices();
    const pick = vs.find((v) => /zira|aria|jenny|female/i.test(v.name)) || vs.find((v) => /^en/i.test(v.lang)) || vs[0];
    if (pick) u.voice = pick;
    u.rate = 1.05;
    u.onend = u.onerror = () => res();
    speechSynthesis.speak(u);
  });

  // ------------------------------------------------------------ live talk
  // Listening: an analyser watches the microphone level; speech starts a
  // recording, and 1.1 s of quiet after it ends the turn.
  const T = { on: false };
  V.talk = async () => {
    if (T.on) return stopTalk();
    try { T.stream = await mic(); } catch (e) { return toast(`microphone: ${e.message}`, "bad"); }
    T.on = true;
    T.ctx = new AudioContext();
    const src = T.ctx.createMediaStreamSource(T.stream);
    T.an = T.ctx.createAnalyser(); T.an.fftSize = 1024;
    src.connect(T.an);
    T.buf = new Float32Array(T.an.fftSize);
    overlay(true);
    listen();
  };
  function level() { T.an.getFloatTimeDomainData(T.buf); let s = 0; for (const x of T.buf) s += x * x; return Math.sqrt(s / T.buf.length); }
  function listen() {
    if (!T.on) return;
    state("Listening…", "listen");
    let talking = false, quietSince = 0, r = null, floor = 0.01;
    const t0 = Date.now();
    const tick = async () => {
      if (!T.on) { if (r) r.stop(); return; }
      const lv = level();
      if (Date.now() - t0 < 600) floor = Math.max(floor, lv * 1.8);          // learn the room's noise first
      const loud = lv > Math.max(0.02, floor);
      $("#vtMeter") && ($("#vtMeter").style.transform = `scale(${1 + Math.min(lv * 8, 0.8)})`);
      if (loud && !talking) { talking = true; r = record(T.stream); state("Hearing you…", "hear"); }
      if (talking) {
        if (loud) quietSince = 0; else if (!quietSince) quietSince = Date.now();
        if (quietSince && Date.now() - quietSince > 1100) {
          const blob = await stopped(r);
          return turn(blob);
        }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  async function turn(blob) {
    state("Writing down what you said…", "think");
    let text = "";
    try { text = await transcribe(blob); } catch (e) { toast(`speech to text: ${e.message}`, "bad"); }
    if (!T.on) return;
    if (!text || text.length < 2) return listen();
    $("#vtYou").textContent = text;
    state("Thinking…", "think");
    const before = S.log.length;
    await MLMI.send(text);                          // the chat model answers like any message
    if (!T.on) return;
    const bot = S.log.slice(before).reverse().find((e) => e.role !== "you");
    if (!bot) {
      // send() declined (no Chat model, or a job still running): say so, never re-read an old answer
      const noModel = !(S.state && S.state.model && S.state.model.path);
      const why = noModel ? "There is no Chat model loaded. Add one from the model menu, then talk again."
                          : "Sheep is still busy with another job. Try again when it finishes.";
      $("#vtSheep").textContent = why;
      state("Could not answer", "think");
      await V.say(why);
      if (noModel) return stopTalk();
      return listen();
    }
    const answer = bot.text || "";
    $("#vtSheep").textContent = answer || "";
    state("Speaking…", "speak");
    await V.say(answer);
    listen();
  }
  function stopTalk() {
    const wasOn = T.on;
    T.on = false;
    // ending the talk ends its answer too, not only the listening
    if (wasOn && S.busy) MLMI.api("/api/stop", {}).catch(() => {});
    window.speechSynthesis && speechSynthesis.cancel();
    if (T.stream) T.stream.getTracks().forEach((t) => t.stop());
    if (T.ctx) T.ctx.close().catch(() => {});
    overlay(false);
  }
  V.stopTalk = stopTalk;
  V.talking = () => T.on;
  // Read aloud is a toggle: pressing it while speaking stops the voice
  V.toggleSay = (text) => (window.speechSynthesis && speechSynthesis.speaking ? speechSynthesis.cancel() : V.say(text));
  function overlay(on) {
    let o = $("#voiceTalk");
    if (!on) { if (o) o.hidden = true; $("#talkBtn").classList.remove("live"); return; }
    if (!o) {
      o = document.createElement("div");
      o.id = "voiceTalk";
      document.body.appendChild(o);
    }
    o.innerHTML = `<div class="vt"><div class="vt-orb"><i id="vtMeter"></i>${icon("sheep")}</div>
      <div class="vt-state" id="vtState">Listening…</div>
      <div class="vt-lines"><p class="you" id="vtYou"></p><p class="sheep" id="vtSheep"></p></div>
      <div class="vt-model dim">Talking with ${MLMI.$("#modelName").textContent}. Speech is written down on this computer; answers are spoken by its voices.</div>
      <button class="danger" id="vtEnd">${icon("stop")}End talk</button></div>`;
    o.hidden = false;
    $("#vtEnd").onclick = stopTalk;
    $("#talkBtn").classList.add("live");
  }
  function state(text, cls) {
    const s = $("#vtState");
    if (s) { s.textContent = text; s.className = "vt-state " + cls; }
  }
  if (window.speechSynthesis) speechSynthesis.getVoices();     // voices load lazily
  const wire = () => { const m = $("#micBtn"), t = $("#talkBtn"); if (m) m.onclick = V.dictate; if (t) t.onclick = V.talk; };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire); else wire();
})();
