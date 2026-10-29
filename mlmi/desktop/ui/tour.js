// The tour: a teacher sheep walks to each part of Sheep, points at it, and says what it is, how to use it,
// and its limits -- on THIS computer (its own graphics cards, processor and memory, read from /api/hardware)
// and for the model plugged in now (where it runs, how much it remembers, what is still missing). Opened
// from the Guide; never by itself. Arrow keys move, Esc leaves.
"use strict";

(() => {
  const { $, esc, api, S } = MLMI;

  // ---------------------------------------------------------------- the teacher (vector, animated by CSS)
  const SHEEP = `<svg class="ts-svg" viewBox="0 0 140 130" aria-hidden="true">
    <g class="ts-legs">
      <rect class="ts-leg l1" x="48" y="86" width="7" height="26" rx="3.5"/>
      <rect class="ts-leg l2" x="62" y="88" width="7" height="24" rx="3.5"/>
      <rect class="ts-leg l1" x="80" y="88" width="7" height="24" rx="3.5"/>
      <rect class="ts-leg l2" x="94" y="86" width="7" height="26" rx="3.5"/>
    </g>
    <g class="ts-body">
      <g class="ts-wool">
        <circle cx="74" cy="66" r="30"/><circle cx="50" cy="60" r="17"/><circle cx="62" cy="44" r="17"/>
        <circle cx="84" cy="41" r="18"/><circle cx="102" cy="55" r="17"/><circle cx="104" cy="76" r="15"/>
        <circle cx="86" cy="90" r="15"/><circle cx="62" cy="88" r="15"/><circle cx="47" cy="76" r="14"/>
      </g>
      <g class="ts-arm"><path d="M58 80 L18 86" /><circle cx="15" cy="86" r="4"/></g>
      <g class="ts-head">
        <ellipse class="ts-ear" cx="16" cy="52" rx="9" ry="5" transform="rotate(-25 16 52)"/>
        <ellipse class="ts-ear" cx="46" cy="47" rx="9" ry="5" transform="rotate(25 46 47)"/>
        <ellipse class="ts-face" cx="31" cy="62" rx="16" ry="19"/>
        <g class="ts-eyes"><circle cx="25" cy="58" r="4.2" class="ts-eye"/><circle cx="37" cy="58" r="4.2" class="ts-eye"/>
          <circle cx="26" cy="59" r="1.9" class="ts-pupil"/><circle cx="38" cy="59" r="1.9" class="ts-pupil"/></g>
        <path class="ts-mouth" d="M26 71 Q31 75 36 71"/>
        <g class="ts-cap"><path d="M8 38 L31 29 L54 38 L31 47 Z"/><rect x="24" y="40" width="14" height="7" rx="2"/>
          <path class="ts-tassel" d="M50 39 L52 52"/><circle class="ts-tassel-end" cx="52" cy="54" r="2.4"/></g>
      </g>
    </g>
  </svg>`;

  // ---------------------------------------------------------------- what this computer and this model are
  const gb = (b) => (b ? (b > 1e6 ? b / 2 ** 30 : b) : 0);
  const fmtGb = (x) => (x >= 10 ? Math.round(x) : Math.round(x * 10) / 10) + " GB";

  async function context() {
    const c = { cards: [], cpuName: "", ramGb: 0, st: {}, stats: {}, missing: [], model: "" };
    try { c.st = await api("/api/state"); } catch (_) { c.st = S.state || {}; }
    c.stats = (c.st.engine && c.st.engine.stats) || {};
    try {
      const hw = await api("/api/hardware");
      c.cards = (hw.adapters || []).map((a) => ({ name: a.name, vendor: a.vendor, vram: gb(a.vram), runs: a.runs || [] }));
      c.setup = hw.setup || "";
    } catch (_) { /* the tour still runs on what it knows */ }
    try {
      const sc = await api("/api/setup_check");
      c.missing = (sc.items || []).filter((i) => !i.ok && !i.optional).map((i) => i.label);
    } catch (_) {}
    c.model = ($("#modelName") && $("#modelName").textContent.trim()) || (c.st.model && c.st.model.name) || "your model";
    c.engineCard = c.cards.find((a) => a.runs.some((r) => /CUDA/.test(r)));
    c.otherCards = c.cards.filter((a) => a !== c.engineCard);
    c.heat = c.st.orch && c.st.orch.heat;
    return c;
  }

  const cardName = (a) => `${a.name}${a.vram ? ` (${fmtGb(a.vram)})` : ""}`;
  function whereItRuns(c) {
    const s = c.stats, dev = s.device || "";
    const card = c.engineCard ? cardName(c.engineCard) : "the graphics card";
    if (dev === "gpu") return `${c.model} runs entirely on ${card}.`;
    if (dev === "hybrid") return `${c.model} is bigger than ${card}: ${s.gpu_layers} of its ${s.gpu_of} layers sit on the card, the rest on the processor, and prompts are still read on the card.`;
    if (dev === "multi") return `${c.model} is spread across several devices: ${s.device_name || "your graphics cards"} and the processor.`;
    if (dev && dev !== "cpu") return `${c.model} runs on ${s.device_name || dev}.`;
    if (dev === "cpu") return `${c.model} runs on the processor${s.threads ? ` with ${s.threads} threads` : ""}${c.engineCard ? ` (the speed trial found it faster than ${cardName(c.engineCard)} here)` : ""}.`;
    return `${c.model} is starting; Sheep is choosing where it runs.`;
  }
  function hardwareLine(c) {
    if (!c.cards.length) return "Sheep found no graphics card it can use, so every model runs on the processor. That works; it is slower.";
    const parts = [];
    if (c.engineCard) parts.push(`${cardName(c.engineCard)} runs Sheep's own engine`);
    for (const a of c.otherCards) parts.push(`${cardName(a)} helps with pictures and video${a.runs.length ? "" : " where it can"}`);
    if (!c.engineCard) parts.push("chat models run on the processor (Sheep's engine needs an NVIDIA card for the fast path; others help through DirectML)");
    return parts.join("; ") + ".";
  }
  function memoryLine(c) {
    const ctx = c.stats.ctx;
    return ctx ? `It remembers about ${ctx.toLocaleString()} tokens of a conversation, roughly ${Math.round(ctx * 0.75).toLocaleString()} words; older turns fall away first.` : "";
  }
  function sizeLimit(c) {
    const m = /(\d+(?:\.\d+)?)\s*B\b/i.exec(c.model);
    const b = m ? parseFloat(m[1]) : 0;
    if (!b) return "A small model makes more mistakes than a big one; check important work.";
    if (b < 4) return `A ${b}B model is quick but makes mistakes on hard reasoning. Switch on Sheep + Nebius and Nemotron checks its drafts.`;
    if (b < 10) return `A ${b}B model is a good all-rounder; on this computer it is slower than a small one.`;
    return `A ${b}B model is strong but heavy; expect slow replies unless the graphics card holds most of it.`;
  }
  const heatLine = (c) => c.heat && c.heat.gpu ? `Your graphics card is at ${Math.round(c.heat.gpu)} °C now; Sheep pauses work at 95 °C and carries on when it cools.` : "Sheep pauses work if the graphics card reaches 95 °C, so the laptop never shuts down mid-job.";

  // ---------------------------------------------------------------- the lesson, one stop per part
  const settings = (tab) => () => { MLMI.closeCanvas && MLMI.closeCanvas(); MLMI.openSettings(tab); };
  const home = () => { MLMI.go("home"); };
  const STOPS = [
    { title: "Hello! I'm your guide", at: null, prep: home,
      say: (c) => `Sheep runs AI on this computer and makes real files. Here is what I found: ${hardwareLine(c)}`,
      how: () => "Use the arrows or Next. Esc leaves; you can come back from the Guide.",
      limit: () => "Everything I tell you is about this computer and the model plugged in now." },
    { title: "Ask in plain words", at: "#composer .box", prep: home,
      say: () => "Type what you need: a question, a document, a budget, a deck, a picture or a video. Sheep does the work and the file opens beside the chat.",
      how: () => "Press Enter to send. Shift+Enter starts a new line. Try “powerpoint about the history of coffee”, then “pptx”.",
      limit: (c) => sizeLimit(c) },
    { title: "Sheep or Sheep + Nebius", at: "#modeSeg", prep: home,
      say: (c) => `In Sheep mode nothing leaves this computer. Sheep + Nebius lets NVIDIA Nemotron plan long jobs, check drafts and stand in while a model loads.${c.st.cloud && c.st.cloud.ready ? ` Connected: ${c.st.cloud.model} (${c.st.cloud.provider}).` : " It is not connected yet: add a key in Settings › Accounts."}`,
      how: () => "Click either half. Your choice is kept.",
      limit: () => "The cloud needs internet and a key; every request that leaves is listed in Settings › Advanced › Sent out." },
    { title: "Attach anything", at: "#attachBtn", prep: home,
      say: () => "Documents, sheets, slides, PDFs, pictures, recordings and videos. Sheep reads them first: a video is watched, frame by frame, with what is said and when.",
      how: () => "Click the clip, or drop files on the chat. Then ask about them: “what happens in this video?”",
      limit: (c) => c.missing.length ? `Some readers are not installed yet (${c.missing.slice(0, 3).join(", ")}); the chat will show a button to get them.` : "Pictures are read by a small local vision model: it names things and colours, not fine detail." },
    { title: "Speak instead of typing", at: "#micBtn", prep: home,
      say: () => "The microphone turns your speech into text in the box. The wave button next to it starts a spoken conversation: Sheep listens, answers aloud, and listens again.",
      how: () => "Press, speak, press again.",
      limit: () => "Speech is recognised on this computer by Whisper; the first use downloads it. Voices are the ones Windows has." },
    { title: "The model in charge", at: "#modelPick", prep: home,
      say: (c) => `${whereItRuns(c)} ${memoryLine(c)}`,
      how: () => "Click to switch to another model you have added.",
      limit: (c) => sizeLimit(c) },
    { title: "The status lamp", at: "#status", prep: home,
      say: () => "Green: ready. Amber: starting, checking or repairing. Red: something needs you, with the reason.",
      how: () => "Click it for what Sheep is doing right now.",
      limit: (c) => heatLine(c) },
    { title: "Sheep's apps", at: "#side .tiles", prep: home,
      say: () => "Words for documents, Sheets for workbooks, Slides for decks, Studio for video, Text for notes and code. Each saves Sheep's own file and exports to Word, Excel, PowerPoint, PDF or MP4.",
      how: () => "Open one here, or just ask in the chat: the model uses the same tools a click does, with one shared undo.",
      limit: () => "Exports keep text, tables, charts and pictures; exotic Office features (macros, tracked changes) are not made." },
    { title: "Studio: videos and ads", at: '.tile[data-app="studio"]', prep: home,
      say: (c) => `Titles, logo intros, slideshows and ads are drawn exactly, frame by frame. Crop, split, fade, pan and zoom on photos, music under a voice-over.${c.engineCard ? ` AI video clips run on ${cardName(c.engineCard)}.` : " AI video clips need a graphics card; drawn videos work on any computer."}`,
      how: () => "“Make a 15-second ad from these photos with the headline Fresh Coffee”, or open Studio and edit the timeline.",
      limit: () => "AI clips are short (a few seconds) and 512 px on a 4 GB card; their quality follows the video model you plug in." },
    { title: "Your work, in one place", at: "#workBtn", prep: home,
      say: () => "The Canvas holds everything this chat made: open, play, run code, save, pin and reorder.",
      how: () => "Click Canvas; drag blocks to reorder.",
      limit: () => "Each chat has its own Canvas." },
    { title: "Undo, redo", at: "#undoBtn", prep: home,
      say: () => "Every change, yours or the model's, can be undone.",
      how: () => "Ctrl+Z and Ctrl+Y work too.",
      limit: () => "Undo follows the open file; closing Sheep keeps the files, not the undo history." },
    { title: "Chats", at: "#chats", prep: home,
      say: () => "Every conversation is kept on this computer, searchable.",
      how: () => "New chat starts fresh; search finds old ones.",
      limit: () => "Chats stay on this computer; nothing is synced." },
    { title: "Settings › Basics", at: "#setBody", prep: settings("basics"),
      say: () => "How Sheep runs, as three plain choices: fastest, balanced, or quiet and cool. Theme and web answers live here too.",
      how: () => "Pick one; Sheep sets the details.",
      limit: () => "Quiet keeps memory free for other apps and runs slower." },
    { title: "Settings › Models", at: "#setBody", prep: settings("models"),
      say: (c) => `One model for each job: chat, checker, pictures, video, photo guidance and speech.${c.missing.length ? ` Still missing here: ${c.missing.join(", ")}.` : " Every job has a model."}`,
      how: () => "Add any model by its Hugging Face name, a folder or a .gguf file. Sheep fits it to this computer, builds it and checks it. You set nothing.",
      limit: (c) => `What fits depends on memory: ${hardwareLine(c)} A model bigger than the card is split with the processor and writes more slowly.` },
    { title: "Settings › Accounts", at: "#setBody", prep: settings("accounts"),
      say: () => "Keys for Nebius (NVIDIA Nemotron) and Tavily (web search), and Hugging Face for gated models.",
      how: () => "Paste a key and Save. Keys stay on this computer.",
      limit: () => "Without a Tavily key, web answers use a keyless public search." },
    { title: "Settings › Activity", at: "#setBody", prep: settings("activity"),
      say: () => "Everything Sheep did and why: where each model runs, every check it ran on a model, every repair.",
      how: () => "Open a row for its full report: what was asked, the tools used, the reply, the files made.",
      limit: () => "Checks run in the background and give way to you." },
    { title: "Settings › Advanced", at: "#setBody", prep: settings("advanced"),
      say: (c) => `The engineer's room: this computer's devices, the checker, files and what was sent out. ${hardwareLine(c)}`,
      how: () => "Most people never need it; Auto chooses for you.",
      limit: (c) => heatLine(c) },
    { title: "The Guide", at: "#openGuide", prep: () => { MLMI.go("home"); },
      say: () => "Examples to copy, what each model does, and this tour again.",
      how: () => "Click Guide any time.",
      limit: () => "" },
    { title: "You're ready", at: null, prep: home,
      say: (c) => `That's all of Sheep. ${whereItRuns(c)} Ask for something real and watch it work.`,
      how: () => "Close me with Done.",
      limit: () => "" },
  ];

  // ---------------------------------------------------------------- the stage
  let root, spot, dock, bubble, i = 0, ctx = null, typing = 0, moving = 0;
  const reduce = () => window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  function build() {
    root = document.createElement("div");
    root.id = "tour";
    root.innerHTML = `<div class="tour-block"></div><div class="tour-spot"></div>
      <div class="tour-dock"><div class="tour-sheep">${SHEEP}</div>
        <div class="tour-bubble" role="dialog" aria-live="polite">
          <div class="tb-top"><span class="tb-step"></span><button class="ghost tb-x" aria-label="Leave the tour">×</button></div>
          <h3 class="tb-title"></h3><p class="tb-say"></p>
          <div class="tb-row tb-how"><b>How</b><span></span></div>
          <div class="tb-row tb-limit"><b>Limits</b><span></span></div>
          <div class="tb-bar"><i></i></div>
          <div class="tb-nav"><button class="ghost tb-back">Back</button><span class="grow"></span><button class="tb-next">Next</button></div>
        </div></div>`;
    document.body.appendChild(root);
    spot = root.querySelector(".tour-spot");
    dock = root.querySelector(".tour-dock");
    bubble = root.querySelector(".tour-bubble");
    root.querySelector(".tb-next").onclick = () => (i < STOPS.length - 1 ? show(i + 1) : close());
    root.querySelector(".tb-back").onclick = () => i > 0 && show(i - 1);
    root.querySelector(".tb-x").onclick = close;
    root.querySelector(".tour-block").onclick = () => finishTyping();
    window.addEventListener("resize", place);
  }

  function key(e) {
    if (!root || root.hidden) return;
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight" || e.key === "Enter") { e.preventDefault(); root.querySelector(".tb-next").click(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); root.querySelector(".tb-back").click(); }
  }

  function target() {
    const s = STOPS[i];
    const el = s.at && $(s.at);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return r.width && r.height ? r : null;
  }

  // the spotlight wraps the part; the sheep walks beside it, faces it and points
  function place() {
    if (!root) return;
    const r = target(), W = innerWidth, H = innerHeight, pad = 8;
    if (r) Object.assign(spot.style, { left: r.left - pad + "px", top: r.top - pad + "px", width: r.width + pad * 2 + "px", height: r.height + pad * 2 + "px", opacity: 1 });
    else Object.assign(spot.style, { left: W / 2 + "px", top: H / 2 + "px", width: "0px", height: "0px", opacity: 1 });
    const dw = dock.offsetWidth || 520, dh = dock.offsetHeight || 260, gap = 22;
    let x, y;
    if (!r) { x = (W - dw) / 2; y = (H - dh) / 2; }
    else {
      const cy = r.top + r.height / 2;
      const room = { right: W - r.right, left: r.left, below: H - r.bottom, above: r.top };
      if (room.right >= dw + gap) { x = r.right + gap; y = cy - dh / 2; }
      else if (room.left >= dw + gap) { x = r.left - gap - dw; y = cy - dh / 2; }
      else if (room.below >= dh + gap) { x = r.left + r.width / 2 - dw / 2; y = r.bottom + gap; }
      else if (room.above >= dh + gap) { x = r.left + r.width / 2 - dw / 2; y = r.top - gap - dh; }
      else { x = W - dw - 24; y = H - dh - 24; }        // a big part (a settings page): the corner
    }
    x = Math.max(12, Math.min(W - dw - 12, x));
    y = Math.max(12, Math.min(H - dh - 12, y));
    const was = dock.getBoundingClientRect();
    dock.style.left = x + "px";
    dock.style.top = y + "px";
    // walk while moving; then face the part and point at it. The drawing faces left: f = 1 (the part is
    // to the right) mirrors it, and puts the sheep on the bubble's right, the side nearest the part.
    const f = r && r.left + r.width / 2 > x + dw / 2 ? 1 : -1;
    dock.classList.toggle("rev", f > 0);
    dock.style.setProperty("--flip", f > 0 ? -1 : 1);
    if (r) {
      const sx = f > 0 ? x + dw - 80 : x + 50, sy = y + dh - 50;
      const ahead = Math.max(20, Math.abs(r.left + r.width / 2 - sx)), dy = r.top + r.height / 2 - sy;
      const ang = Math.max(-60, Math.min(60, Math.atan2(dy, ahead) * 180 / Math.PI));
      dock.style.setProperty("--point", -ang + "deg");
      dock.classList.add("pointing");
    } else dock.classList.remove("pointing");
    if (Math.abs(was.left - x) + Math.abs(was.top - y) > 8 && !reduce()) {
      dock.classList.add("walking");
      clearTimeout(moving);
      moving = setTimeout(() => dock.classList.remove("walking"), 950);
    }
  }

  function typeInto(el, text) {
    clearInterval(typing);
    if (reduce()) { el.textContent = text; return; }
    el.textContent = "";
    dock.classList.add("talking");
    let k = 0;
    typing = setInterval(() => {
      k = Math.min(text.length, k + 3);
      el.textContent = text.slice(0, k);
      if (k >= text.length) { clearInterval(typing); dock.classList.remove("talking"); }
    }, 16);
    el.dataset.full = text;
  }
  function finishTyping() {
    const el = root && root.querySelector(".tb-say");
    if (el && el.dataset.full) { clearInterval(typing); el.textContent = el.dataset.full; dock.classList.remove("talking"); }
  }

  async function show(n) {
    i = n;
    const s = STOPS[i];
    try { s.prep && s.prep(); } catch (_) {}
    await new Promise((r) => setTimeout(r, 260));          // let a page open before it is measured
    root.querySelector(".tb-step").textContent = `${i + 1} of ${STOPS.length}`;
    root.querySelector(".tb-title").textContent = s.title;
    const how = s.how ? s.how(ctx) : "", lim = s.limit ? s.limit(ctx) : "";
    root.querySelector(".tb-how span").textContent = how;
    root.querySelector(".tb-limit span").textContent = lim;
    root.querySelector(".tb-how").hidden = !how;
    root.querySelector(".tb-limit").hidden = !lim;
    root.querySelector(".tb-bar i").style.width = ((i + 1) / STOPS.length) * 100 + "%";
    root.querySelector(".tb-back").disabled = i === 0;
    root.querySelector(".tb-next").textContent = i === STOPS.length - 1 ? "Done" : "Next";
    bubble.classList.remove("pop"); void bubble.offsetWidth; bubble.classList.add("pop");
    place();
    typeInto(root.querySelector(".tb-say"), s.say(ctx));
  }

  async function open() {
    MLMI.closeGuide && MLMI.closeGuide();
    if (!root) build();
    root.hidden = false;
    document.addEventListener("keydown", key);
    root.classList.add("in");
    ctx = await context();
    show(0);
  }
  function close() {
    if (!root) return;
    clearInterval(typing);
    root.hidden = true;
    document.removeEventListener("keydown", key);
    MLMI.go("home");
  }
  MLMI.tour = open;
})();
