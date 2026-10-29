// The quick guide: what Sheep does and how to start, in a few lines per part,
// with examples that drop straight into the message box. Opens only from Guide
// in the sidebar: it never opens by itself (Ernest, 2026-10-06).
"use strict";

(() => {
  const { $, $$, esc, icon, S } = MLMI;
  const PARTS = [
    ["sheep", "Just ask", "Type what you need. Sheep answers on this computer; nothing leaves it in Sheep mode."],
    ["doc", "It makes real files", "Ask for a document, workbook, deck, video or code. It is made with the tools and lands on the Work Canvas beside the chat."],
    ["canvas", "Work Canvas", "Everything a chat made, as blocks: open, play, run code, save, pin, drag to reorder. The Canvas button at the top shows it."],
    ["route", "Web answers", "Questions about now (news, prices, scores) are searched on the web and answered with numbered sources. Switch it off in Settings › Basics."],
    ["cloud", "Sheep or Sheep + Nebius", "Sheep keeps everything local. Sheep + Nebius lets NVIDIA Nemotron plan long jobs and stand in while a model is being fixed."],
    ["sheep", "Models in one place", "Settings › Models picks the model for chat, pictures, video and speech. Add any model: Sheep fits it to this computer and checks it before use."],
    ["check", "Fix buttons in chat", "If something is missing, the chat says what and shows a button to get it, then Try again. Choose another… picks a different model."],
    ["play", "Videos and ads", "Ask for a clip, or attach a video and ask what happens in it. Studio crops, splits, fades and turns photos into an ad."],
    ["gauge", "Heat", "Sheep runs at full speed and pauses only if the graphics card reaches 95 °C, then carries on."],
    ["mic", "Speak instead of typing", "Mic: press, speak, press again; your words appear in the box. Talk: a spoken conversation, where Sheep listens, answers aloud, and listens again. Works with any chat model."],
  ];
  const QUICK_PARTS = [
    ["sheep", "Ask", "Type a question or ask for a file, picture or video."],
    ["canvas", "Find your work", "Open Canvas to reopen files made in this chat."],
    ["cloud", "Choose where it runs", "Sheep mode stays on this computer. Sheep + Nebius uses your configured cloud service."],
    ["mic", "Speak", "Mic turns speech into text. Talk listens and replies aloud."],
    ["check", "Missing something?", "The chat shows a button to get it, then Try again."],
    ["play", "Videos and ads", "Attach a video to ask what happens in it, or photos to make an ad."],
    ["sliders", "Change models", "Settings › Models: one place for every model."],
  ];
  const SLOTS = [
    ["Chat", "Answers questions and follows instructions."],
    ["Checker", "Reviews new chat models before use."],
    ["Image/media", "Makes still images; any Stable Diffusion 1.5 model works."],
    ["Video", "Makes short clips and reads videos you attach."],
    ["Photo guidance", "Uses photos you attach or select to guide a video."],
    ["Speech input", "Turns a recording or microphone speech into text."],
    ["Speech output", "Reads answers aloud with voices on this computer."],
  ];
  const EXAMPLES = [
    "Make a one-page project brief in Sheep Words called brief.sdoc with a title, a short summary and three goals.",
    "Make a workbook called budget.ssheet with Item and Cost: Rent 800, Food 250, Transport 90, and a Total row.",
    "Make a 15-second logo intro video with the title Sheep.",
    "Make a 2-second AI video of waves rolling onto a beach at sunset.",
    "What is the latest news about the Uganda Cranes?",
    "Show me a short Python function that checks if a number is prime.",
  ];
  function open() {
    let m = $("#guide");
    if (!m) {
      m = document.createElement("div");
      m.className = "modal";
      m.id = "guide";
      document.body.appendChild(m);
      m.addEventListener("click", (e) => {
        if (e.target === m || e.target.closest("#closeGuide")) return close();
        if (e.target.closest("#startTour")) return MLMI.tour && MLMI.tour();
        const ex = e.target.closest("[data-ex]");
        if (ex) { close(); $("#askText").value = ex.dataset.ex; $("#askText").focus(); $("#askText").dispatchEvent(new Event("input")); }
      });
    }
    m.innerHTML = `<div class="sheet guide"><button class="ghost ico close" id="closeGuide" title="Close">${icon("x")}</button>
      <div class="g-head"><span class="logo">${icon("sheep")}</span><div><h1>Quick guide</h1><span class="sub">Sheep runs AI on this computer. Ask, and it does the work.</span></div>
        <button class="g-tour" id="startTour">${icon("play")}<span>Take the tour</span></button></div>
      <div class="g-grid">${PARTS.map(([ic, t, d]) => `<div class="g-card"><span class="g-ic">${icon(ic)}</span><div><b>${esc(t)}</b><p>${esc(d)}</p></div></div>`).join("")}</div>
      <div class="g-ex"><div class="caps">Try one</div>${EXAMPLES.map((x) => `<button data-ex="${esc(x)}">${icon("up")}<span>${esc(x)}</span></button>`).join("")}</div>
      <div class="g-foot dim">The full user guide is in the Sheep folder: docs/USER_GUIDE.md</div></div>`;
    // Put copy-and-try examples first and keep implementation detail folded
    // away from people who only want to know what Sheep can do.
    const exampleSection = m.querySelector(".g-ex");
    const cards = m.querySelector(".g-grid");
    if (exampleSection && cards) {
      exampleSection.querySelector(".caps").textContent = "Try a request";
      cards.before(exampleSection);
      cards.innerHTML = QUICK_PARTS.map(([ic, title, body]) => `<div class="g-card"><span class="g-ic">${icon(ic)}</span><div><b>${esc(title)}</b><p>${esc(body)}</p></div></div>`).join("");
    }
    const slotSection = document.createElement("section");
    slotSection.className = "g-slots";
    slotSection.innerHTML = `<div class="caps">What each model does</div><div class="g-slot-grid">${SLOTS.map(([name, body]) => `<div><b>${esc(name)}</b><span>${esc(body)}</span></div>`).join("")}</div>`;
    m.querySelector(".g-foot").before(slotSection);
    m.hidden = false;
  }
  function close() { const m = $("#guide"); if (m) m.hidden = true; }
  MLMI.openGuide = open;
  MLMI.closeGuide = close;

  // The video tip card (spec 10): what decides video speed and quality on this
  // machine, what Sheep set and why, and what the person can change. Opened from
  // Studio's toolbar and from an AI video in the chat.
  async function videoTip() {
    let m = $("#vtip");
    if (!m) {
      m = document.createElement("div");
      m.className = "modal";
      m.id = "vtip";
      document.body.appendChild(m);
      m.addEventListener("click", (e) => { if (e.target === m || e.target.closest("#closeTip")) m.hidden = true; });
    }
    let t;
    try { t = await MLMI.api("/api/video_tip"); } catch (e) { MLMI.toast(e.message, "bad"); return; }
    const ai = t.ai;
    const speed = ai.per_second ? `about ${ai.per_second} s of work per second of video here` : "measured after the first clip";
    m.innerHTML = `<div class="sheet guide vtip"><button class="ghost ico close" id="closeTip" title="Close">${icon("x")}</button>
      <div class="g-head"><span class="logo">${icon("play")}</span><div><h1>Video on this machine</h1><span class="sub">What Sheep set, why, and what you can change</span></div></div>
      <div class="vt-two">
        <div class="g-card"><span class="g-ic">${icon("spark")}</span><div><b>AI video from a prompt</b>
          <p>${esc(ai.model)}${ai.ready ? "" : " · not downloaded yet (model menu)"}</p>
          <p><i>runs on</i> ${esc(ai.device)}: ${esc(ai.why)}.</p>
          <p>Attach or import a photo and ask Sheep to animate it. Photos stay on this computer and guide local generation; photo-guidance weights download on first use.</p>
          <p><i>set to</i> ${esc(ai.size)}, ${ai.fps} frames a second, ${esc(ai.seconds)} s, ${ai.steps} steps · ${esc(speed)}</p></div></div>
        <div class="g-card"><span class="g-ic">${icon("play")}</span><div><b>Studio: titles, logos, intros, edits</b>
          <p>No AI model: every frame is drawn exactly, then encoded. Use a photo montage or slideshow to preserve the original photos as Studio layers.</p>
          <p><i>encoder</i> ${esc(t.studio.encoder)}</p>
          <p><i>set to</i> ${esc(t.studio.size)}, ${t.studio.fps} frames a second</p></div></div>
      </div>
      <div class="g-who"><div class="caps">What decides speed and quality</div>
        <div class="g-table">${t.decides.map(([k, v]) => `<div class="g-row vt-row"><b>${esc(k)}</b><span class="dim">${esc(v)}</span></div>`).join("")}</div></div>
      <div class="g-who"><div class="caps">What you can change</div>
        <ul class="vt-list">${t.change.map((c) => `<li>${esc(c)}</li>`).join("")}</ul></div></div>`;
    m.hidden = false;
  }
  function closeVideoTip() { const m = $("#vtip"); if (m) m.hidden = true; }
  MLMI.videoTip = videoTip;
  MLMI.closeVideoTip = closeVideoTip;

  const wire = () => {
    const b = $("#openGuide");
    if (b) b.onclick = open;
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire); else wire();
})();
