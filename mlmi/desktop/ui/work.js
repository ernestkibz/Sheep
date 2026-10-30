// The Work Canvas: everything this chat's jobs made, as blocks beside the chat.
// Blocks come from the conversation itself (the files each job changed, code in
// its answers, the sources it read, long written answers), so nothing is stored
// twice; the chat keeps only the block order and pins. A document, sheet or deck
// opens its editor, a video, picture or sound plays here, code can be copied,
// saved as a file or run (sandboxed, output below), and blocks drag to reorder.
"use strict";

(() => {
  const { $, $$, esc, api, op, toast, S, icon } = MLMI;
  const KIND = {
    sdoc: "document", docx: "document", md: "document", txt: "document",
    ssheet: "sheet", xlsx: "sheet", csv: "sheet",
    sdeck: "deck", pptx: "deck",
    sstudio: "video", mp4: "video", webm: "video", mov: "video", mkv: "video",
    png: "image", jpg: "image", jpeg: "image", gif: "image", webp: "image", bmp: "image",
    mp3: "audio", wav: "audio", ogg: "audio", m4a: "audio", flac: "audio",
    py: "source", js: "source", ts: "source", html: "source", css: "source", json: "source", sql: "source", sh: "source",
  };
  const KIND_ICON = { job: "spark", document: "doc", sheet: "grid", deck: "slides", video: "play", image: "image", audio: "audio",
    code: "code", source: "code", note: "note", sources: "route", file: "folder" };
  const KIND_NAME = { job: "Work", document: "Document", sheet: "Sheet", deck: "Deck", video: "Video", image: "Picture", audio: "Sound",
    code: "Code", source: "Code file", note: "Note", sources: "Sources", file: "File" };
  const FENCE = /```([\w+-]*)[ \t]*\n([\s\S]*?)```/g;
  const RUNS = { python: "python", py: "python", javascript: "javascript", js: "javascript", node: "javascript" };
  const raw = (p) => "/api/raw?path=" + encodeURIComponent(p);

  // ------------------------------------------------------------ blocks from the conversation
  function blocks() {
    const out = [], seen = new Set();
    S.log.forEach((e, i) => {
      if (e.role === "you") return;
      if (e.role === "bot" && e.jobId && ["active", "stopping", "failed", "cancelled"].includes(e.workState)) {
        const id = "j:" + e.jobId;
        seen.add(id);
        out.push({ id, kind: "job", title: e.workTask || (S.log[i - 1] && S.log[i - 1].text) || "Sheep is working",
          stage: e.workStage || (e.workState === "stopping" ? "Stopping safely" : "Starting"),
          state: e.workState, progress: Number.isFinite(e.workProgress) ? e.workProgress : null,
          files: e.files || [], ask: (S.log[i - 1] && S.log[i - 1].text) || "", cloud: e.route === "cloud" });
      }
      // a finished answer that made no file is just an answer, not a work card (Ernest, 2026-10-13)
      const ask = (S.log[i - 1] && S.log[i - 1].role === "you" ? S.log[i - 1].text : "") || "";
      for (const p of e.files || []) {
        const id = "f:" + p;
        if (seen.has(id)) continue;        // a file a later job changed again: one block
        seen.add(id);
        out.push({ id, kind: KIND[p.split(".").pop().toLowerCase()] || "file", path: p, title: p.split("/").pop(), ask,
          cloud: e.route === "cloud" });
      }
      const text = e.text || "";
      let m, k = 0;
      FENCE.lastIndex = 0;
      while ((m = FENCE.exec(text))) {
        out.push({ id: `c:${i}:${k++}`, kind: "code", lang: (m[1] || "text").toLowerCase(), code: m[2].replace(/\n$/, ""),
          title: (m[1] || "code").toLowerCase(), ask });
      }
      const prose = text.replace(FENCE, "").trim();
      if (k === 0 && prose.length > 280 && !(e.files || []).length) out.push({ id: `n:${i}`, kind: "note", text: prose, title: ask.slice(0, 60) || "Answer", ask });
      if ((e.sources || []).length) out.push({ id: `s:${i}`, kind: "sources", items: e.sources, title: ask.slice(0, 60) || "Sources", ask });
    });
    // A job keeps running if the person opens another conversation. Keep its
    // card reachable in Canvas until they return to the conversation that owns it.
    const elsewhere = S.activeJob && S.activeJob.chatId !== S.chatId ? S.activeJob
      : S.lastJob && S.lastJob.chatId !== S.chatId ? S.lastJob : null;
    if (elsewhere && elsewhere.bot && !seen.has("j:" + elsewhere.id)) {
      const b = elsewhere.bot;
      const state = b.workState || (b.done ? "complete" : "active");
      out.unshift({ id: "j:" + elsewhere.id, kind: "job", title: b.workTask || elsewhere.prompt || "Sheep is working",
        stage: b.workStage || (state === "stopping" ? "Stopping safely" : state === "complete" ? "Completed" : "Starting"),
        state, progress: Number.isFinite(b.workProgress) ? b.workProgress : null, files: b.files || [], ask: elsewhere.prompt || "" });
    }
    const meta = S.canvas || (S.canvas = { order: [], pinned: {} });
    const rank = (b) => { const j = meta.order.indexOf(b.id); return j < 0 ? 1e6 + out.indexOf(b) : j; };
    return out.sort((a, b) => (!!meta.pinned[b.id] - !!meta.pinned[a.id]) || rank(a) - rank(b));
  }
  MLMI.workCount = () => blocks().filter((b) => b.kind !== "job").length;
  MLMI.workActiveCount = () => blocks().filter((b) => b.kind === "job" && ["active", "stopping"].includes(b.state)).length;
  MLMI.highlight = (code) => highlight(code);
  // Chat text with fenced code and Markdown tables rendered as contained blocks.
  MLMI.md = (text) => {
    const parts = [];
    let last = 0, m;
    const re = /```([\w+-]*)[ \t]*\n([\s\S]*?)```/g;
    const inline = (t) => esc(t).replace(/`([^`\n]+)`/g, "<code>$1</code>").replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>")
      .replace(/\[(\d{1,2})\]/g, '<sup class="cite">$1</sup>');
    const cellsOf = (line) => {
      let s = String(line || "").trim();
      if (!s.includes("|")) return null;
      if (s.startsWith("|")) s = s.slice(1);
      if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
      const cells = [];
      let cell = "";
      for (let i = 0; i < s.length; i++) {
        if (s[i] === "\\" && s[i + 1] === "|") { cell += "|"; i++; }
        else if (s[i] === "|") { cells.push(cell.trim()); cell = ""; }
        else cell += s[i];
      }
      cells.push(cell.trim());
      return cells.length > 1 ? cells : null;
    };
    const prose = (t) => {
      const lines = String(t || "").split(/\r?\n/), out = [];
      for (let i = 0; i < lines.length;) {
        const header = cellsOf(lines[i]), divider = cellsOf(lines[i + 1]);
        const align = divider && divider.map((c) => {
          const d = c.replace(/\s/g, "");
          if (!/^:?-{3,}:?$/.test(d)) return "";
          return d.startsWith(":") ? (d.endsWith(":") ? "center" : "left") : d.endsWith(":") ? "right" : "left";
        });
        if (header && divider && header.length === divider.length && align.every(Boolean)) {
          const rows = [];
          let j = i + 2, row;
          while (j < lines.length && (row = cellsOf(lines[j])) && row.length === header.length) {
            rows.push(row); j++;
          }
          const th = header.map((c, n) => `<th scope="col" style="text-align:${align[n]}">${inline(c)}</th>`).join("");
          const body = rows.map((r) => `<tr>${header.map((_, n) => `<td style="text-align:${align[n]}">${inline(r[n])}</td>`).join("")}</tr>`).join("");
          out.push(`<div class="chat-table-wrap" role="region" aria-label="Table in response" tabindex="0"><table class="chat-table"><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`);
          i = j;
        } else {
          out.push(inline(lines[i])); i++;
        }
      }
      return out.join("\n");
    };
    while ((m = re.exec(text))) {
      parts.push(prose(text.slice(last, m.index)));
      parts.push(`<pre class="code"><span class="code-head">${esc(m[1] || "code")}<button class="ghost" data-copycode title="Copy">${icon("copy")}Copy</button></span><code>${highlight(m[2].replace(/\n$/, ""))}</code></pre>`);
      last = m.index + m[0].length;
    }
    parts.push(prose(text.slice(last)));
    return parts.join("");
  };

  // ------------------------------------------------------------ a small highlighter for code blocks
  const KW = /\b(def|class|return|if|elif|else|for|while|in|import|from|as|with|try|except|finally|raise|lambda|yield|pass|break|continue|and|or|not|is|None|True|False|const|let|var|function|new|this|async|await|of|null|undefined|true|false|export|default)\b/g;
  function highlight(code) {
    const out = [];
    const re = /(#[^\n]*|\/\/[^\n]*)|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`)|(\b\d+(?:\.\d+)?\b)/g;
    let last = 0, m;
    while ((m = re.exec(code))) {
      out.push(esc(code.slice(last, m.index)).replace(KW, '<b class="kw">$1</b>'));
      out.push(`<i class="${m[1] ? "cm" : m[2] ? "st" : "nu"}">${esc(m[0])}</i>`);
      last = m.index + m[0].length;
    }
    out.push(esc(code.slice(last)).replace(KW, '<b class="kw">$1</b>'));
    return out.join("");
  }

  function bodyHtml(b) {
    switch (b.kind) {
      case "job": {
        const pct = Number.isFinite(b.progress) ? Math.max(0, Math.min(100, Math.round(b.progress))) : null;
        const outputs = (b.files || []).map((p) => `<button class="wb-job-file" data-job-open="${esc(p)}">${icon("folder")}<span>${esc(p.split("/").pop())}</span></button>`).join("");
        const previewPath = (b.files || []).find((p) => KIND[p.split(".").pop().toLowerCase()] === "image" || KIND[p.split(".").pop().toLowerCase()] === "video");
        const previewKind = previewPath && KIND[previewPath.split(".").pop().toLowerCase()];
        const preview = previewKind === "image" ? `<img loading="lazy" src="${raw(previewPath)}" alt="${esc(previewPath.split("/").pop())}">`
          : previewKind === "video" ? `<video controls preload="metadata" src="${raw(previewPath)}"></video>` : "";
        return `<div class="wb-job-stage"><span class="wb-state ${esc(b.state)}">${esc(({ active: "In progress", stopping: "Stopping", complete: "Complete", failed: "Failed", cancelled: "Cancelled" })[b.state] || b.state)}</span>
          ${/^(complete|completed|done)$/i.test(b.stage || "") ? "" : `<span>${esc(b.stage || "Working")}</span>`}</div>${pct === null ? `<div class="wb-progress indeterminate"><i></i></div>` : `<div class="wb-progress"><i style="width:${pct}%"></i></div>`}
          ${preview ? `<div class="wb-job-preview">${preview}</div>` : ""}${outputs ? `<div class="wb-job-files">${outputs}</div>` : ""}`;
      }
      case "video": return b.path.endsWith(".sstudio") ? `<div class="wb-open">${icon("play")}<span>Studio project · open to edit and render</span></div>`
        : `<video controls preload="metadata" src="${raw(b.path)}"></video>`;
      case "image": return `<img loading="lazy" src="${raw(b.path)}" alt="${esc(b.title)}">`;
      case "audio": return `<audio controls preload="metadata" src="${raw(b.path)}"></audio>`;
      case "code": return `<pre class="wb-code"><code>${highlight(b.code)}</code></pre><pre class="wb-out" hidden></pre>`;
      case "note": return `<div class="wb-note">${esc(b.text.length > 900 ? b.text.slice(0, 900) + "…" : b.text)}</div>`;
      case "sources": return `<div class="wb-src">${b.items.map((s, i) => { let h = ""; try { h = new URL(s.url).hostname.replace(/^www\./, ""); } catch (_) {}
        return `<a href="${esc(s.url)}" target="_blank" rel="noopener"><span class="n">${i + 1}</span><span><b>${esc(s.title || h)}</b><small>${esc(h)}</small></span></a>`; }).join("")}</div>`;
      default: return `<div class="wb-preview" data-preview="${esc(b.path)}"><span class="dim">reading…</span></div>`;
    }
  }
  function actions(b) {
    const a = [];
    if (b.path && ["document", "sheet", "deck", "video", "source"].includes(b.kind)) a.push(["open", "open", "Open"]);
    if (b.kind === "code") {
      a.push(["copy", "copy", "Copy"], ["save", "save", "Save as file"]);
      if (RUNS[b.lang]) a.push(["run", "play", "Run"]);
    }
    if (b.kind === "note") a.push(["copy", "copy", "Copy"], ["save", "save", "Save as file"]);
    if (b.path) a.push(["reveal", "folder", "Show in folder"]);
    return a.map(([k, ic, t]) => `<button class="ghost ico" data-act="${k}" title="${t}">${icon(ic)}</button>`).join("");
  }

  S.views.work = {
    mount(el) {
      this.el = el;
      el.innerHTML = `<div class="work"><div class="work-head"><span class="note" id="wkSub"></span></div><div class="work-grid" id="wkGrid"></div></div>`;
      el.addEventListener("click", (e) => this.click(e));
      el.addEventListener("dragstart", (e) => { const c = e.target.closest(".wb"); if (c) { this.drag = c.dataset.id; c.classList.add("dragging"); } });
      el.addEventListener("dragend", (e) => { const c = e.target.closest(".wb"); if (c) c.classList.remove("dragging"); });
      el.addEventListener("dragover", (e) => { if (this.drag) e.preventDefault(); });
      el.addEventListener("drop", (e) => {
        const c = e.target.closest(".wb");
        if (!c || !this.drag || c.dataset.id === this.drag) return;
        e.preventDefault();
        const ids = this.list.map((b) => b.id).filter((x) => x !== this.drag);
        ids.splice(ids.indexOf(c.dataset.id), 0, this.drag);
        S.canvas.order = ids;
        this.drag = null;
        this.draw(); MLMI.saveChat();
      });
      this.draw();
    },
    unmount() { this.el = null; },
    refresh() { if (this.el) this.draw(); },
    draw() {
      const list = this.list = blocks();
      const active = list.filter((b) => b.kind === "job" && ["active", "stopping"].includes(b.state)).length;
      const results = list.filter((b) => b.kind !== "job").length;
      $("#wkSub", this.el).textContent = active || results ? `${active ? `${active} active · ` : ""}${results} result${results === 1 ? "" : "s"} · drag to reorder · pin to keep`
        : "";
      $("#wkGrid", this.el).innerHTML = list.length ? list.map((b) => `<div class="wb ${b.kind} ${b.cloud ? "cloud" : ""} ${S.canvas.pinned[b.id] ? "pinned" : ""}" data-id="${esc(b.id)}" ${b.kind === "job" ? "" : 'draggable="true"'}>
          <div class="wb-head"><span class="wb-ic">${icon(KIND_ICON[b.kind] || "folder")}</span>
            <span class="wb-title"><b>${esc(b.title)}</b><small>${KIND_NAME[b.kind] || "File"}${b.kind === "code" ? "" : b.ask ? " · " + esc(b.ask.slice(0, 48)) : ""}</small></span>
            <span class="wb-acts">${actions(b)}<button class="ghost ico pin" data-act="pin" title="${S.canvas.pinned[b.id] ? "Unpin" : "Pin to the top"}">${icon("pin")}</button></span></div>
          <div class="wb-body">${bodyHtml(b)}</div></div>`).join("")
        : `<div class="work-empty">${icon("canvas")}<b>Your work appears here</b><span>Ask Sheep to make something: documents, sheets, decks, videos and code land here as blocks.</span></div>`;
      $$("[data-preview]", this.el).forEach(async (d) => {
        const r = await api("/api/preview?path=" + encodeURIComponent(d.dataset.preview)).catch((e) => ({ text: e.message }));
        if (d.isConnected) d.innerHTML = `<pre>${esc(r.text || "")}</pre>`;
      });
    },
    async click(e) {
      const btn = e.target.closest("[data-act]"), card = e.target.closest(".wb");
      if (!card) return;
      const b = this.list.find((x) => x.id === card.dataset.id);
      if (!b) return;
      const jobFile = e.target.closest("[data-job-open]");
       if (jobFile) return MLMI.openFile(jobFile.dataset.jobOpen);
      const act = btn ? btn.dataset.act : (b.path && ["document", "sheet", "deck", "source"].includes(b.kind) && e.target.closest(".wb-body") ? "open" : "");
      if (act === "open") return MLMI.openFile(b.path);
      if (act === "pin") { S.canvas.pinned[b.id] = !S.canvas.pinned[b.id]; if (!S.canvas.pinned[b.id]) delete S.canvas.pinned[b.id]; this.draw(); MLMI.saveChat(); return; }
      // the block's path is workspace-relative; the system file browser needs it absolute
      if (act === "reveal") return MLMI.reveal(b.path);
      const text = b.kind === "code" ? b.code : b.text;
      if (act === "copy") { await navigator.clipboard.writeText(text); return toast("Copied"); }
      if (act === "save") {
        const ext = b.kind === "code" ? ({ python: "py", py: "py", javascript: "js", js: "js", html: "html", css: "css", json: "json" }[b.lang] || "txt") : "md";
        const name = await MLMI.ask("Save as", { ok: "Save", input: `${b.kind === "code" ? "code" : "note"}-${Date.now().toString(36).slice(-4)}.${ext}` });
        if (!name) return;
        try { await op("write_text_file", { path: name, text }); toast(`Saved ${name}`); MLMI.loadTree(); } catch (err) { toast(err.message, "bad"); }
        return;
      }
      if (act === "run") {
        const out = $(".wb-out", card);
        out.hidden = false; out.className = "wb-out"; out.textContent = "running…";
        try {
          const r = await api("/api/run_code", { lang: RUNS[b.lang], code: b.code });
          out.textContent = (r.output || "(no output)") + `\n— exit ${r.code} · ${r.ms} ms`;
          out.classList.toggle("bad", r.code !== 0);
        } catch (err) { out.textContent = err.message; out.classList.add("bad"); }
      }
    },
  };
})();
