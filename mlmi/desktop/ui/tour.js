// The tour: Sheep's own logo, stood up on its two legs with a shepherd's crook, walks to each part of the
// app, turns to it and touches it with the crook, and says what it is, how to use it and its limits -- on
// THIS computer (its own graphics cards, processor and heat, from /api/hardware and /api/state) and for the
// model plugged in now. Every frame is computed here: critically damped springs move the sheep, the
// spotlight and the bubble; the walk cycle (legs, bob, lean) follows the distance actually travelled; the
// crook's angle is solved from the shoulder to the point it touches; the eyes follow the cursor; a click
// makes it hop. Opened from the Guide; never by itself. Arrow keys move, Esc leaves.
"use strict";

(() => {
  const { $, esc, api, S } = MLMI;

  // ---------------------------------------------------------------- the teacher: the logo, stood up
  // The logo's own shapes (brand/sheep-mark.svg): the six-circle cloud of wool, the head with its eye cut
  // out, the ear, the two rounded legs. Local frame: facing right, feet at (115, 250).
  const S_ = 0.42;                                     // the logo's 512 grid down to this body
  const CLOUD = "M122,196 a74,74 0 1,0 148,0 a74,74 0 1,0 -148,0 Z M196,164 a82,82 0 1,0 164,0 a82,82 0 1,0 -164,0 Z " +
                "M284,206 a72,72 0 1,0 144,0 a72,72 0 1,0 -144,0 Z M108,278 a78,78 0 1,0 156,0 a78,78 0 1,0 -156,0 Z " +
                "M184,290 a88,88 0 1,0 176,0 a88,88 0 1,0 -176,0 Z M278,282 a74,74 0 1,0 148,0 a74,74 0 1,0 -148,0 Z";
  // The app's own logo (icons.js): a fleece of seven circles, a black face with two ears and two white eyes,
  // two black legs, every black part edged in white. Mirrored to face right and stood up: the head on top of
  // the fleece, a black arm holding a black crook edged in white. Drawn on the logo's 64 grid, x4.
  const L = "translate(-5 18) scale(4)";
  const FLEECE = "translate(31 48) scale(0.92 1.12) translate(-31 -48)";     // stood up: taller, a little narrower
  const HOOK = "M0 0 C6 0 8.5 -6.2 5.5 -9.8 C3 -12.6 -1.6 -11 -0.6 -7.2";
  const SHEEP = `<svg class="ts-svg" viewBox="0 0 260 270" aria-hidden="true">
    <ellipse class="ts-shadow" cx="116" cy="251" rx="46" ry="6.5"/>
    <g class="ts-legL"><g transform="${L}"><rect class="ts-ink" x="21.5" y="40" width="4.5" height="18" rx="2"/></g></g>
    <g class="ts-legR"><g transform="${L}"><rect class="ts-ink" x="34.5" y="40" width="4.5" height="18" rx="2"/></g></g>
    <g class="ts-upper">
      <g transform="${L} ${FLEECE}" class="ts-fleece"><circle cx="42" cy="28" r="10"/><circle cx="30" cy="22" r="11"/><circle cx="19" cy="29" r="10"/>
        <circle cx="36" cy="38" r="10"/><circle cx="23" cy="39" r="10"/><circle cx="46" cy="37" r="7"/><circle cx="14" cy="38" r="7"/></g>
      <g class="ts-board" transform="rotate(-9 82 168)">
        <line class="ts-limb-e" x1="96" y1="128" x2="88" y2="166"/><line class="ts-limb-i" x1="96" y1="128" x2="88" y2="166"/>
        <rect class="ts-board-r" x="60" y="140" width="40" height="52" rx="5"/><rect class="ts-ink" x="72" y="134" width="16" height="10" rx="3"/>
        <line class="ts-board-l" x1="68" y1="158" x2="92" y2="158"/><line class="ts-board-l" x1="68" y1="167" x2="92" y2="167"/>
        <line class="ts-board-l" x1="68" y1="176" x2="84" y2="176"/><circle class="ts-ink" cx="88" cy="168" r="7"/></g>
      <g class="ts-crook"><g transform="scale(4)">
        <line class="ts-crook-edge" x1="-8" y1="0" x2="34" y2="0"/><line class="ts-crook-ink" x1="-8" y1="0" x2="34" y2="0"/>
        <g transform="translate(34 0)"><path class="ts-crook-edge" d="${HOOK}"/><path class="ts-crook-ink" d="${HOOK}"/></g></g></g>
      <g class="ts-arm">
        <line class="ts-limb-e ts-ua"/><line class="ts-limb-i ts-ua"/><line class="ts-limb-e ts-fa"/><line class="ts-limb-i ts-fa"/>
        <circle class="ts-ink ts-hand" r="8.5"/></g>
      <g class="ts-ear"><g transform="${L}"><ellipse class="ts-ink" cx="40.5" cy="0" rx="4" ry="2.4" transform="rotate(25 40.5 0)"/>
        <ellipse class="ts-ink" cx="28.5" cy="-2" rx="4" ry="2.4" transform="rotate(-25 28.5 -2)"/></g></g>
      <g transform="${L}"><ellipse class="ts-ink" cx="34" cy="6" rx="8" ry="9.5"/></g>
      <g class="ts-eye"><g transform="${L}"><circle cx="31" cy="5" r="1.6"/><circle cx="37" cy="5" r="1.6"/></g></g>
    </g>
  </svg>`;
  // where things are in the local frame (px of the 260 x 270 drawing): logo grid x4, shifted
  const FEET = { x: 116, y: 250 }, SHOULDER = { x: 168, y: 126 }, HIPS = [{ x: 90, y: 178 }, { x: 142, y: 178 }];
  const HEAD = { x: 131, y: 42 }, EYE = { x: 131, y: 38 };
  // the arm: upper arm and forearm (px), and the crook from the hand to its hook's front
  const UPPER = 36, FORE = 34, CROOK = 200, REACH = UPPER + FORE + CROOK;

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
    { title: "Sheep or Sheep + Nebius", gain: () => "Quality: Nemotron plans each step and checks the local model's draft, so files come out right more often. Speed: a long plan comes back in seconds. Privacy: Sheep mode keeps everything here.", at: "#modeSeg", prep: home,
      say: (c) => `In Sheep mode nothing leaves this computer. Sheep + Nebius lets NVIDIA Nemotron plan long jobs, check drafts and stand in while a model loads.${c.st.cloud && c.st.cloud.ready ? ` Connected: ${c.st.cloud.model} (${c.st.cloud.provider}).` : " It is not connected yet: add a key in Settings › Accounts."}`,
      how: () => "Click either half. Your choice is kept.",
      limit: () => "The cloud needs internet and a key; every request that leaves is listed in Settings › Advanced › Sent out." },
    { title: "Attach anything", gain: () => "Quality: answers come from your own files, not the model's memory.", at: "#attachBtn", prep: home,
      say: () => "Documents, sheets, slides, PDFs, pictures, recordings and videos. Sheep reads them first: a video is watched, frame by frame, with what is said and when.",
      how: () => "Click the clip, or drop files on the chat. Then ask about them: “what happens in this video?”",
      limit: (c) => c.missing.length ? `Some readers are not installed yet (${c.missing.slice(0, 3).join(", ")}); the chat will show a button to get them.` : "Pictures are read by a small local vision model: it names things and colours, not fine detail." },
    { title: "Speak instead of typing", at: "#micBtn", prep: home,
      say: () => "The microphone turns your speech into text in the box. The wave button next to it starts a spoken conversation: Sheep listens, answers aloud, and listens again.",
      how: () => "Press, speak, press again.",
      limit: () => "Speech is recognised on this computer by Whisper; the first use downloads it. Voices are the ones Windows has." },
    { title: "The model in charge", gain: () => 'Speed: a model that fits the graphics card writes several times faster than on the processor; Sheep times both and keeps the faster. Quality: a bigger model answers better and writes slower.', at: "#modelPick", prep: home,
      say: (c) => `${whereItRuns(c)} ${memoryLine(c)}`,
      how: () => "Click to switch to another model you have added.",
      limit: (c) => sizeLimit(c) },
    { title: "The status lamp", at: "#status", prep: home,
      say: () => "Green: ready. Amber: starting, checking or repairing. Red: something needs you, with the reason.",
      how: () => "Click it for what Sheep is doing right now.",
      limit: (c) => heatLine(c) },
    { title: "Sheep's apps", gain: () => "Speed and exactness: the model writes the words and Sheep's own tools make the file, so a deck or a budget takes seconds and every total adds up.", at: "#side .tiles", prep: home,
      say: () => "Words for documents, Sheets for workbooks, Slides for decks, Studio for video, Text for notes and code. Each saves Sheep's own file and exports to Word, Excel, PowerPoint, PDF or MP4.",
      how: () => "Open one here, or just ask in the chat: the model uses the same tools a click does, with one shared undo.",
      limit: () => "Exports keep text, tables, charts and pictures; exotic Office features (macros, tracked changes) are not made." },
    { title: "Studio: videos and ads", gain: () => 'Quality: titles, logos and ads are drawn exactly, frame by frame, with no AI guessing. Speed: drawn videos need no AI model and render on any computer.', at: '.tile[data-app="studio"]', prep: home,
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
    { title: "Settings › Basics", gain: () => "Speed: Fastest fills the card's spare memory with ready-made weights. Quality is the same in every choice; Quiet runs cooler and leaves memory for other apps.", at: "#setBody", point: "#setBody .phead h1, #setBody h1", prep: settings("basics"),
      say: () => "How Sheep runs, as three plain choices: fastest, balanced, or quiet and cool. Theme and web answers live here too.",
      how: () => "Pick one; Sheep sets the details.",
      limit: () => "Quiet keeps memory free for other apps and runs slower." },
    { title: "Settings › Models", at: "#setBody", point: "#setBody .phead h1, #setBody h1", prep: settings("models"),
      say: (c) => `One model for each job.${c.missing.length ? ` Still missing here: ${c.missing.join(", ")}.` : ""} Chat: any Hugging Face model, a folder, or a .gguf file; a GGUF, GPTQ, AWQ, EXL2 or MLX copy is traced back to its original and rebuilt from that; a family Sheep's engine has not met is fitted to it, or runs in compatibility mode. Pictures: any Stable Diffusion 1.5 model. Video: AnimateDiff on Stable Diffusion 1.5. Speech: any faster-whisper size.`,
      gain: () => "Quality: built from the original weights, a Sheep file stays closer to the original model than a standard 4-bit GGUF (12-41% less error in our races). Speed: Sheep sizes it for this computer, places it on the card, the processor or both, and times which is faster.",
      how: () => "Add any model by its Hugging Face name, a folder or a .gguf file. Sheep fits it to this computer, builds it and checks it. You set nothing.",
      limit: (c) => `What fits depends on memory: ${hardwareLine(c)} A model bigger than the card is split with the processor and writes more slowly.` },
    { title: "Settings › Accounts", at: "#setBody", point: "#setBody .phead h1, #setBody h1", prep: settings("accounts"),
      say: () => "Keys for Nebius (NVIDIA Nemotron) and Tavily (web search), and Hugging Face for gated models.",
      how: () => "Paste a key and Save. Keys stay on this computer.",
      limit: () => "Without a Tavily key, web answers use a keyless public search." },
    { title: "Settings › Activity", gain: () => 'Quality: the checks catch a model that starts giving wrong answers, and Sheep repairs it by itself.', at: "#setBody", point: "#setBody .phead h1, #setBody h1", prep: settings("activity"),
      say: () => "Everything Sheep did and why: where each model runs, every check it ran on a model, every repair.",
      how: () => "Open a row for its full report: what was asked, the tools used, the reply, the files made.",
      limit: () => "Checks run in the background and give way to you." },
    { title: "Settings › Advanced", gain: () => 'Speed: it shows which device helps with what, so you can see why Sheep chose it.', at: "#setBody", point: "#setBody .phead h1, #setBody h1", prep: settings("advanced"),
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

  // ---------------------------------------------------------------- the stage, frame by frame
  const SCALE = 0.66;                                    // the drawing on screen
  const reduce = () => window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  // a critically damped spring: no overshoot, no wobble, arrives in about 4 / w seconds
  function spring(o, key, target, w, dt) {
    const v = o[key + "V"] || 0, x = o[key];
    const a = w * w * (target - x) - 2 * w * v;
    o[key + "V"] = v + a * dt;
    o[key] = x + o[key + "V"] * dt;
  }

  let typedN = 0, sayLen = 0, said = [];
  let root, spot, ring, actor, bubble, svg, parts = {}, subs = [], i = 0, ctx = null, typing = 0, raf = 0, last = 0;
  const st = { tx: 230, ty: -40, sub: -1, reach: 0, rx: 0, ry: 0, rw: 0, rh: 0, ringT: null, ear: 0, earT: 0, x: 0, y: 0, face: 1, phase: 0, arm: -78, tap: 0, hop: 0, hopV: 0, blink: 0, nextBlink: 2,
               ex: 0, ey: 0, sx: 0, sy: 0, sw: 0, sh: 0, bx: 0, by: 0, t: 0, arrived: 0, said: true };
  const goal = { x: 0, y: 0, face: 1, point: null, spot: null, bubble: null };
  const mouse = { x: -1, y: -1 };

  function build() {
    root = document.createElement("div");
    root.id = "tour";
    root.innerHTML = `<div class="tour-block"></div><div class="tour-spot"></div><div class="tour-ring"></div>
      <div class="tour-actor" title="Click me">${SHEEP}</div>
      <div class="tour-bubble" role="dialog" aria-live="polite">
        <div class="tb-top"><span class="tb-step"></span><button class="ghost tb-x" aria-label="Leave the tour">×</button></div>
        <h3 class="tb-title"></h3><p class="tb-say"></p>
        <div class="tb-row tb-how"><b>How</b><span></span></div>
        <div class="tb-row tb-gain"><b>Gains</b><span></span></div>
        <div class="tb-row tb-limit"><b>Limits</b><span></span></div>
        <div class="tb-bar"><i></i></div>
        <div class="tb-nav"><button class="ghost tb-back">Back</button><span class="grow"></span><button class="tb-next">Next</button></div>
      </div>`;
    document.body.appendChild(root);
    spot = root.querySelector(".tour-spot");
    actor = root.querySelector(".tour-actor");
    bubble = root.querySelector(".tour-bubble");
    svg = actor.querySelector("svg");
    for (const k of ["legL", "legR", "upper", "crook", "eye", "ear", "shadow", "hand"]) parts[k] = svg.querySelector(".ts-" + k);
    parts.ua = svg.querySelectorAll(".ts-ua");
    parts.fa = svg.querySelectorAll(".ts-fa");
    ring = root.querySelector(".tour-ring");
    root.querySelector(".tb-next").onclick = () => (i < STOPS.length - 1 ? show(i + 1) : close());
    root.querySelector(".tb-back").onclick = () => i > 0 && show(i - 1);
    root.querySelector(".tb-x").onclick = close;
    root.querySelector(".tour-block").onclick = () => finishTyping();
    actor.onclick = () => { if (!st.hop) st.hopV = -520; finishTyping(); };
    window.addEventListener("resize", () => aim(true));
    window.addEventListener("mousemove", (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
  }

  function key(e) {
    if (!root || root.hidden) return;
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight" || e.key === "Enter") { e.preventDefault(); root.querySelector(".tb-next").click(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); root.querySelector(".tb-back").click(); }
  }

  function rectOf(sel) {
    const el = sel && $(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return r.width && r.height ? r : null;
  }

  // Where to stand: beside the part, the shoulder about 0.85 of the crook away from the point it touches.
  // A part too big to stand beside (a settings page) is touched near its title from below-right.
  function aim(snap) {
    const W = innerWidth, H = innerHeight, spotR = rectOf(STOPS[i].at), r = rectOf(STOPS[i].point) || spotR, k = SCALE;
    const aw = 260 * k, ah = 270 * k, reach = REACH * k, sh = { x: (SHOULDER.x - FEET.x) * k, y: (SHOULDER.y - FEET.y) * k };
    let fx, fy, face = 1, P = null;
    if (!r) { fx = W * 0.5 - 170; fy = H * 0.5 + 150; }
    else {
      const big = r.width > W * 0.45 || r.height > H * 0.55;
      const cy = r.top + r.height / 2;
      if (big) { P = { x: r.left + 48, y: r.top + 56 }; face = -1; fx = r.left + 330; fy = r.top + 330; }
      else if (r.right + aw + 30 < W) { P = { x: r.right, y: cy }; face = -1; }
      else if (r.left - aw - 30 > 0) { P = { x: r.left, y: cy }; face = 1; }
      else { P = { x: r.left + r.width / 2, y: r.top }; face = r.left + r.width / 2 > W / 2 ? 1 : -1; }
      if (!big) {
        const s = { x: P.x - face * reach * 0.8, y: P.y + reach * 0.38 };     // the shoulder's spot
        fx = s.x - face * sh.x; fy = s.y - sh.y;
      }
    }
    fx = clamp(fx, aw * 0.45 + 8, W - aw * 0.45 - 8);
    fy = clamp(fy, ah - 30, H - 6);
    Object.assign(goal, { x: fx, y: fy, face, point: P, spot: spotR });
    if (snap || reduce()) { st.x = fx; st.y = fy; st.face = face; }
  }

  // The parts inside a section, in reading order: the crook visits each in turn, round and round, until
  // the person moves on. A section with nothing inside is touched as a whole.
  function partsOf(stop) {
    const box = stop.at && $(stop.at);
    if (!box) return [];
    const sel = stop.parts || ".tile, [data-tab], .sect > .sh, .field, button, input, select, .g-card, .row";
    const W = innerWidth, H = innerHeight;
    const seen = [];
    for (const el of box.querySelectorAll(sel)) {
      const r = el.getBoundingClientRect();
      if (r.width < 8 || r.height < 8 || r.bottom < 0 || r.top > H || r.right < 0 || r.left > W) continue;
      if (seen.some((o) => o.contains(el) || el.contains(o))) continue;
      seen.push(el);
      if (seen.length >= 8) break;
    }
    seen.sort((a, b) => { const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect(); return (ra.top - rb.top) || (ra.left - rb.left); });
    return seen.length ? seen : [box];
  }

  // the bubble sits behind the sheep (the side away from the part), at its head, kept on screen
  function bubbleGoal() {
    const W = innerWidth, H = innerHeight, bw = bubble.offsetWidth || 400, bh = bubble.offsetHeight || 240;
    const k = SCALE, headY = st.y + (HEAD.y - FEET.y) * k;
    let bx = goal.face > 0 ? st.x - 120 * k - bw - 6 : st.x + 120 * k + 6;
    if (bx < 10 || bx + bw > W - 10) bx = goal.face > 0 ? st.x + 120 * k + 6 : st.x - 120 * k - bw - 6;
    bx = clamp(bx, 10, W - bw - 10);
    const by = clamp(headY - 30, 10, H - bh - 10);
    return { x: bx, y: by };
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    const dt = Math.min(0.033, (now - (last || now)) / 1000) || 0.016;
    last = now;
    st.t += dt;
    const k = SCALE;
    // move: a spring to the standing spot; the walk cycle runs on the distance covered
    const ox = st.x, oy = st.y;
    spring(st, "x", goal.x, 4.2, dt); spring(st, "y", goal.y, 4.2, dt);
    const dist = Math.hypot(st.x - ox, st.y - oy), speed = dist / dt;
    const walk = clamp(speed / 260, 0, 1);
    st.phase += dist / (46 * k) * Math.PI;
    const far = Math.hypot(goal.x - st.x, goal.y - st.y);
    // turn to the way it walks, then to the part; scaleX passes through 0, a turn, not a flip
    const faceT = far > 40 ? Math.sign(goal.x - st.x) || goal.face : goal.face;
    st.face += (faceT - st.face) * Math.min(1, dt * 9);
    // What the crook's hook should touch, in the drawing's own frame. Walking, it is carried upright
    // ahead; once there, it visits each part of the section in turn and taps it twice.
    const local = (P) => ({ x: (P.x - st.x) / k * goal.face + FEET.x, y: (P.y - st.y - st.hop) / k + FEET.y });
    let T = { x: SHOULDER.x + 70 + Math.sin(st.phase) * 6 * walk, y: SHOULDER.y - 170 }, looking = null;
    if (far < 6) {
      if (!st.arrived) { st.arrived = st.t; st.sub = -1; if (!st.said) say(); }
      let P = goal.point, pr = null;
      const since = st.t - st.arrived;
      const order = said.length ? said : subs.map((el) => ({ el }));
      let cur = null;
      if (said.length && typedN < sayLen) {                 // reading: the last part the words have reached
        for (const x of said) if (x.at <= typedN) cur = x.el;
      } else if (order.length) {                            // read: each part in reading order, round and round
        cur = order[Math.floor(since / 1.7) % order.length].el;
      }
      if (cur) {
        if (cur !== st.cur) { st.cur = cur; st.curAt = st.t; }
        pr = cur.getBoundingClientRect();
        P = { x: pr.left + pr.width / 2, y: pr.top + pr.height / 2 };
      }
      if (P) {
        const L = local(P), each = st.t - (st.curAt || st.arrived);
        const d = Math.hypot(L.x - SHOULDER.x, L.y - SHOULDER.y) || 1;
        const push = each < 0.9 ? Math.sin(each * 17) * 7 * Math.exp(-each * 3.5) : 0;   // the double tap
        T = { x: L.x + (L.x - SHOULDER.x) / d * push, y: L.y + (L.y - SHOULDER.y) / d * push };
        looking = P;
      } else T = { x: SHOULDER.x + 60, y: SHOULDER.y - 175 + Math.sin(st.t * 3) * 10 };    // no part: a wave
      st.ringT = pr ? [pr.left - 6, pr.top - 6, pr.width + 12, pr.height + 12] : null;
    } else { st.arrived = 0; st.ringT = null; }
    spring(st, "tx", T.x, 10, dt); spring(st, "ty", T.y, 10, dt);
    // two-bone IK: the hand stands one crook-length short of the target on the line from the shoulder;
    // the elbow bends below that line; the crook runs from the hand to the target
    {
      const dx = st.tx - SHOULDER.x, dy = st.ty - SHOULDER.y, D = Math.hypot(dx, dy) || 1, ux = dx / D, uy = dy / D;
      const c = clamp(D - CROOK, 14, UPPER + FORE - 2);
      const hx = SHOULDER.x + ux * c, hy = SHOULDER.y + uy * c;
      const bend = Math.acos(clamp((UPPER * UPPER + c * c - FORE * FORE) / (2 * UPPER * c), -1, 1));
      const base = Math.atan2(uy, ux), side = ux >= 0 ? 1 : -1;
      const ex = SHOULDER.x + UPPER * Math.cos(base + bend * side), ey = SHOULDER.y + UPPER * Math.sin(base + bend * side);
      for (const l of parts.ua) { l.setAttribute("x1", SHOULDER.x); l.setAttribute("y1", SHOULDER.y); l.setAttribute("x2", ex.toFixed(1)); l.setAttribute("y2", ey.toFixed(1)); }
      for (const l of parts.fa) { l.setAttribute("x1", ex.toFixed(1)); l.setAttribute("y1", ey.toFixed(1)); l.setAttribute("x2", hx.toFixed(1)); l.setAttribute("y2", hy.toFixed(1)); }
      parts.hand.setAttribute("cx", hx.toFixed(1)); parts.hand.setAttribute("cy", hy.toFixed(1));
      const ca = Math.atan2(st.ty - hy, st.tx - hx) * 180 / Math.PI;
      parts.crook.setAttribute("transform", `translate(${hx.toFixed(1)} ${hy.toFixed(1)}) rotate(${ca.toFixed(2)})`);
      st.reach = D;
    }
    // a ring follows the crook from part to part
    if (st.ringT) {
      const R = st.ringT;
      spring(st, "rx", R[0], 9, dt); spring(st, "ry", R[1], 9, dt); spring(st, "rw", R[2], 9, dt); spring(st, "rh", R[3], 9, dt);
      ring.style.opacity = "1";
      ring.style.transform = `translate(${st.rx}px, ${st.ry}px)`;
      ring.style.width = Math.max(0, st.rw) + "px";
      ring.style.height = Math.max(0, st.rh) + "px";
    } else ring.style.opacity = "0";
    // a hop when clicked
    if (st.hopV || st.hop) { st.hopV += 2100 * dt; st.hop += st.hopV * dt; if (st.hop > 0) { st.hop = 0; st.hopV = 0; } }
    // eyes: blink now and then, look at the cursor
    if (st.t > st.nextBlink) { st.blink = 0.14; st.nextBlink = st.t + 2.5 + Math.random() * 3; }
    st.blink = Math.max(0, st.blink - dt);
    const hx = st.x + (HEAD.x - FEET.x) * k * st.face, hy = st.y + (HEAD.y - FEET.y) * k;
    const seen = looking || (mouse.x < 0 ? null : mouse);
    const lx = seen ? clamp((seen.x - hx) / 120, -1, 1) : 0, ly = seen ? clamp((seen.y - hy) / 120, -1, 1) : 0;
    st.ex += (lx * 2.2 * Math.sign(st.face || 1) - st.ex) * Math.min(1, dt * 8);
    st.ey += (ly * 2.2 - st.ey) * Math.min(1, dt * 8);
    // pose
    // A gait, not a swing: each leg is planted for 55% of the cycle and sweeps back under the body while
    // the foot stays put; then it lifts, swings forward on an eased arc and lands. The legs are half a
    // cycle apart; the body is lowest as a foot lands and rolls a little over the planted leg.
    const leg = (q) => {
      q = ((q % 1) + 1) % 1;
      const A = 24 * walk, Lift = 15 * walk;
      if (q < 0.55) return { a: A * (1 - 2 * q / 0.55), lift: 0 };
      const u = (q - 0.55) / 0.45, e = 0.5 - 0.5 * Math.cos(Math.PI * u);
      return { a: -A + 2 * A * e, lift: Math.sin(Math.PI * u) * Lift };
    };
    const gl = { a: 0, lift: 0 }, gr = { a: 0, lift: 0 };    // the legs stay still (Ernest, 2026-10-29)
    const bob = -Math.abs(Math.cos(st.phase)) * 6 * walk * k + (1 - walk) * Math.sin(st.t * 2.1) * 1.1;
    const lean = clamp((st.xV || 0) / 80, -7, 7) * Math.sign(st.face || 1) + Math.sin(st.phase) * 2.2 * walk
      + clamp(((st.reach || 0) - REACH * 0.7) / 25, 0, 5) * Math.sign(st.face || 1);
    parts.legL.setAttribute("transform", `translate(0 ${-gl.lift}) rotate(${gl.a} ${HIPS[0].x} ${HIPS[0].y})`);
    parts.legR.setAttribute("transform", `translate(0 ${-gr.lift}) rotate(${gr.a} ${HIPS[1].x} ${HIPS[1].y})`);
    // squash on each footfall, stretch between: the wool settles onto the legs
    const sq = 0.04 * walk * (1 - Math.abs(Math.cos(st.phase))) - (st.hopV < 0 ? 0.05 : 0);
    parts.upper.setAttribute("transform", `translate(0 ${bob / k}) rotate(${lean} ${FEET.x} ${FEET.y}) ` +
      `translate(${FEET.x} 178) scale(${1 + sq * 0.6} ${1 - sq}) translate(${-FEET.x} -178)`);
    // the ear trails the head: it flops with each step and settles
    st.earT = -Math.sin(st.phase * 2) * 12 * walk + Math.sin(st.t * 1.7) * 2;
    spring(st, "ear", st.earT, 14, dt);
    parts.ear.setAttribute("transform", `rotate(${st.ear} 131 14)`);
    parts.eye.setAttribute("transform", `translate(${st.ex} ${st.ey}) translate(${EYE.x} ${EYE.y}) scale(1 ${st.blink > 0 ? 0.1 : 1}) translate(${-EYE.x} ${-EYE.y})`);
    // the shadow stays on the ground and shrinks as the sheep leaves it
    const lift = clamp(-st.hop / 90, 0, 1);
    parts.shadow.setAttribute("transform", `translate(0 ${-st.hop / k}) translate(116 251) scale(${1 - lift * 0.45}) translate(-116 -251)`);
    parts.shadow.setAttribute("opacity", String(0.55 - lift * 0.35));
    const f = Math.abs(st.face) < 0.06 ? 0.06 * Math.sign(st.face || 1) : st.face;
    actor.style.transform = `translate(${st.x - FEET.x * k}px, ${st.y - FEET.y * k + st.hop}px) scale(${f * k}, ${k})`;
    // the spotlight and the bubble trail on their own springs
    const r = goal.spot, pad = 9;
    const tgt = r ? [r.left - pad, r.top - pad, r.width + pad * 2, r.height + pad * 2] : [innerWidth / 2, innerHeight / 2, 0, 0];
    spring(st, "sx", tgt[0], 6, dt); spring(st, "sy", tgt[1], 6, dt); spring(st, "sw", tgt[2], 6, dt); spring(st, "sh", tgt[3], 6, dt);
    spot.style.transform = `translate(${st.sx}px, ${st.sy}px)`;
    spot.style.width = Math.max(0, st.sw) + "px";
    spot.style.height = Math.max(0, st.sh) + "px";
    const b = bubbleGoal();
    spring(st, "bx", b.x, 5, dt); spring(st, "by", b.y, 5, dt);
    bubble.style.transform = `translate(${st.bx}px, ${st.by}px)`;
  }

  function typeInto(el, text) {
    clearInterval(typing);
    el.dataset.full = text;
    if (reduce()) { el.textContent = text; return; }
    el.textContent = "";
    let n = 0;
    typedN = 0; sayLen = text.length;
    typing = setInterval(() => {
      n = Math.min(text.length, n + 2);
      typedN = n;
      el.textContent = text.slice(0, n);
      if (n >= text.length) clearInterval(typing);
    }, 14);
  }
  function finishTyping() {
    const el = root && root.querySelector(".tb-say");
    if (!st.said) say();
    if (el && el.dataset.full) { clearInterval(typing); el.textContent = el.dataset.full; typedN = sayLen; }
  }
  function say() {
    st.said = true;
    typeInto(root.querySelector(".tb-say"), STOPS[i].say(ctx));
  }

  async function show(n) {
    i = n;
    const s = STOPS[i];
    try { s.prep && s.prep(); } catch (_) {}
    await new Promise((r) => setTimeout(r, 240));          // let a page open before it is measured
    root.querySelector(".tb-step").textContent = `${i + 1} of ${STOPS.length}`;
    root.querySelector(".tb-title").textContent = s.title;
    const how = s.how ? s.how(ctx) : "", lim = s.limit ? s.limit(ctx) : "";
    root.querySelector(".tb-how span").textContent = how;
    root.querySelector(".tb-limit span").textContent = lim;
    root.querySelector(".tb-how").hidden = !how;
    const gain = s.gain ? s.gain(ctx) : "";
    root.querySelector(".tb-gain span").textContent = gain;
    root.querySelector(".tb-gain").hidden = !gain;
    root.querySelector(".tb-limit").hidden = !lim;
    root.querySelector(".tb-bar i").style.width = ((i + 1) / STOPS.length) * 100 + "%";
    root.querySelector(".tb-back").disabled = i === 0;
    root.querySelector(".tb-next").textContent = i === STOPS.length - 1 ? "Done" : "Next";
    root.querySelector(".tb-say").textContent = "";
    st.said = false; st.arrived = 0;
    aim(false);
    subs = partsOf(s);
    // the parts the words name, in the order they are named: "Words for documents, Sheets for ..."
    const text = (s.say(ctx) || "").toLowerCase();
    said = subs.map((el) => {
      const label = ((el.getAttribute("aria-label") || el.textContent || "").trim().split("\n")[0]).trim().toLowerCase();
      return { el, at: label && label.length < 40 ? text.indexOf(label) : -1 };
    }).filter((x) => x.at >= 0).sort((a, b) => a.at - b.at);
    typedN = 0; sayLen = 1e9; st.cur = null;
    setTimeout(() => { if (!st.said && STOPS[i] === s) say(); }, 1600);   // a long walk still talks
  }

  async function open() {
    MLMI.closeGuide && MLMI.closeGuide();
    if (!root) build();
    root.hidden = false;
    document.addEventListener("keydown", key);
    ctx = await context();
    i = 0;
    aim(true);
    st.y = innerHeight + 200;                               // it walks in from below the window
    st.sx = innerWidth / 2; st.sy = innerHeight / 2; st.sw = st.sh = 0;
    const b = bubbleGoal(); st.bx = b.x; st.by = b.y + 40;
    cancelAnimationFrame(raf); last = 0; raf = requestAnimationFrame(frame);
    show(0);
  }
  function close() {
    if (!root) return;
    clearInterval(typing);
    cancelAnimationFrame(raf);
    root.hidden = true;
    document.removeEventListener("keydown", key);
    MLMI.go("home");
  }
  MLMI.tour = open;
})();
// the sidebar's own way in, beside New chat
(() => {
  const wire = () => { const b = document.getElementById("openTour"); if (b) b.onclick = () => MLMI.tour(); };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire); else wire();
})();
