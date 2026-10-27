// Sheep Slides: the .sdeck editor. A slide sorter, the slide itself
// edited in place, speaker notes, our own themes, and full-screen presenting.
"use strict";

(() => {
  const { $, $$, esc, api, op, toast, S } = MLMI;
  const LAYOUTS = [["bullets", "title and points"], ["title", "title slide"], ["section", "section"], ["title_only", "title only"], ["blank", "blank"]];

  function slideHtml(s, th, edit, small) {
    const [bg, tc, bc, ac] = th;
    const ce = edit ? 'contenteditable="true"' : "";
    const lay = s.layout || "bullets";
    let h = `<div class="slide ${lay === "title" || lay === "section" ? "title" : ""} ${small ? "small" : ""}" style="background:#${bg};color:#${bc}">`;
    h += `<div style="position:absolute;left:0;top:0;bottom:0;width:1.2%;background:#${ac}"></div>`;
    if (lay !== "blank") h += `<div class="t" data-f="title" ${ce} style="color:#${tc}${s.title_scale ? `;--ts:${+s.title_scale}` : ""}">${esc(s.title || "")}</div>`;
    if (lay === "title" || lay === "section") h += `<div class="st" data-f="subtitle" ${ce}>${esc(s.subtitle || "")}</div>`;
    if (lay === "bullets") {
      const bl = s.bullets && s.bullets.length ? s.bullets : edit ? [""] : [];
      h += `<div class="b" data-f="bullets" ${ce}>${bl.map((b) => `<div style="${/^\s{2}/.test(b) ? "margin-left:1.6em;opacity:.85" : ""}">${esc(b.trim()) || "<br>"}</div>`).join("")}</div>`;
    }
    (s.items || []).forEach((it, n) => {
      const pos = `left:${it.x}%;top:${it.y}%;width:${it.w}%`;
      // on the stage an item can be typed in (text), dragged by its handle, and removed
      const grip = edit ? `<span class="it-grip" data-grip="${n}" title="Drag to move">⠿</span><span class="it-x" data-rm="${n}" title="Remove">×</span>` : "";
      const tag = edit ? `data-item="${n}" tabindex="0"` : "";
      if (it.kind === "image") h += `<div class="it ${edit ? "edit" : ""}" ${tag} style="${pos}">${grip}<img src="/api/raw?path=${encodeURIComponent(it.src)}"></div>`;
      else if (it.kind === "text") h += `<div class="it ${edit ? "edit" : ""}" ${tag} style="${pos};font-size:${small ? 6 : (it.size || 18) / 10}${small ? "px" : "vw"}">${grip}<span class="it-t" ${ce}>${esc(it.text)}</span></div>`;
    });
    return h + "</div>";
  }

  S.views.slides = {
    mount(el, path) {
      this.el = el;
      this.file = path || "";
      if (!this.file) {
        return MLMI.picker(el, "slides", { ext: "sdeck", office: "pptx", create: "slides_create",
          noun: { title: "Sheep Slides", sub: "decks with our own themes; .pptx in and out", one: "deck", many: "decks", example: "launch plan" } });
      }
      el.innerHTML = `<div class="editor">
        <div class="tools"><span class="file">▭ <b>${esc(this.file)}</b></span>
          <button id="pAdd">+ Slide</button><select id="pLayout" title="Layout">${LAYOUTS.map(([v, n]) => `<option value="${v}">${n}</option>`).join("")}</select>
          <button id="pUp" title="Move up">↑</button><button id="pDown" title="Move down">↓</button><button id="pDup" title="Duplicate this slide">Duplicate</button><button id="pDel" class="danger">Delete</button><span class="sep"></span>
          <button id="pText" title="Add a text box">+ Text box</button><button id="pPic" title="Add a picture">+ Picture</button><span class="sep"></span>
          <select id="pTheme" title="Theme"></select><span style="flex:1"></span>
          <button id="pShow">▶ Present</button><button id="pPdf" title="Save as PDF">PDF</button><button id="pExport" class="primary">Export .pptx</button><button id="pClose" title="All decks">Files</button></div>
        <div class="deck"><div class="sorter" id="pSorter"></div><div class="stage" id="pStage"></div></div></div>`;
      this.i = 0;
      this.wire();
      this.load();
    },
    path() { return this.file; },
    unmount() { this.el = null; },
    reload() { if (!MLMI.isQuiet(this)) this.load(); },
    async load() {
      try { this.d = await api("/api/doc?path=" + encodeURIComponent(this.file)); } catch (e) { toast(e.message, "bad"); return; }
      if (!this.el) return;
      this.i = Math.max(0, Math.min(this.i, this.d.slides.length - 1));
      this.render();
    },
    th() { return this.d.themes[this.d.theme] || this.d.themes.light; },
    render() {
      const th = this.th();
      $("#pTheme", this.el).innerHTML = Object.keys(this.d.themes).map((t) => `<option ${t === this.d.theme ? "selected" : ""}>${t}</option>`).join("");
      $("#pSorter", this.el).innerHTML = this.d.slides.map((s, i) => `<div class="thumbwrap ${i === this.i ? "on" : ""}" data-i="${i}"><span class="n">${i + 1}</span>${slideHtml(s, th, false, true)}</div>`).join("")
        || `<div class="note">No slides yet.</div>`;
      const s = this.d.slides[this.i];
      $("#pStage", this.el).innerHTML = s ? `${slideHtml(s, th, true, false)}<div class="notes"><div class="caps" style="margin-bottom:4px">speaker notes</div>
        <textarea id="pNotes" placeholder="what to say on this slide">${esc(s.notes || "")}</textarea></div>`
        : `<div class="empty"><h2>An empty deck</h2><p>Add the first slide with + slide, or ask the agent for a whole deck.</p></div>`;
      if (s) $("#pLayout", this.el).value = s.layout || "bullets";
    },
    // read the slide back from the stage and save it if it changed
    async save() {
      const s = this.d.slides[this.i];
      const st = $("#pStage .slide", this.el);
      if (!s || !st) return;
      const next = { ...s };
      const t = st.querySelector('[data-f="title"]'), sub = st.querySelector('[data-f="subtitle"]'), b = st.querySelector('[data-f="bullets"]');
      if (t) next.title = t.innerText.trim();
      if (sub) next.subtitle = sub.innerText.trim();
      if (b) next.bullets = [...b.childNodes].map((n) => (n.nodeType === 1 && n.style && n.style.marginLeft ? "  " : "") + (n.textContent || "").trim()).filter((x) => x.trim());
      if (s.items) next.items = s.items.map((it, n) => {
        const t = it.kind === "text" && st.querySelector(`[data-item="${n}"] .it-t`);
        return t ? { ...it, text: t.innerText.trim() } : it;
      });
      const notes = $("#pNotes", this.el);
      if (notes) next.notes = notes.value;
      if (JSON.stringify(next) === JSON.stringify(s)) return;
      this.d.slides[this.i] = next;
      MLMI.quiet(this);
      await op("slides_put", { path: this.file, slide: this.i + 1, data: next });
      MLMI.quiet(this);
      const th = this.th();
      const thumb = $(`#pSorter .thumbwrap[data-i="${this.i}"]`, this.el);
      if (thumb) thumb.innerHTML = `<span class="n">${this.i + 1}</span>${slideHtml(next, th, false, true)}`;
    },
    present() {
      let k = this.i;
      const box = document.createElement("div");
      box.className = "present";
      document.body.appendChild(box);
      const show = () => (box.innerHTML = slideHtml(this.d.slides[k], this.th(), false, false));
      show();
      if (box.requestFullscreen) box.requestFullscreen().catch(() => {});
      const key = (e) => {
        if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) { k = Math.min(this.d.slides.length - 1, k + 1); show(); }
        else if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) { k = Math.max(0, k - 1); show(); }
        else if (e.key === "Escape") end();
        e.preventDefault(); e.stopPropagation();
      };
      // Escape in full screen is taken by the window to leave full screen, so leaving it ends the show too
      const left = () => { if (!document.fullscreenElement) end(); };
      const end = () => { MLMI.showEnded = Date.now(); document.removeEventListener("keydown", key, true); document.removeEventListener("fullscreenchange", left); box.remove(); if (document.fullscreenElement) document.exitFullscreen().catch(() => {}); };
      document.addEventListener("fullscreenchange", left);
      box.onclick = () => { if (k >= this.d.slides.length - 1) end(); else { k++; show(); } };
      document.addEventListener("keydown", key, true);
    },
    wire() {
      $("#pSorter", this.el).onclick = async (e) => {
        const t = e.target.closest(".thumbwrap");
        if (!t) return;
        await this.save();
        this.i = +t.dataset.i;
        this.render();
      };
      $("#pStage", this.el).addEventListener("focusout", (e) => { if (e.target.closest("[data-f], .it-t") || e.target.id === "pNotes") this.save(); });
      // the slide with one change, saved whole (one undo)
      const putSlide = async (s) => {
        MLMI.quiet(this);
        try { await op("slides_put", { path: this.file, slide: this.i + 1, data: s }); } catch (e) { return; }
        this.render();
      };
      const removeItem = async (n) => {
        await this.save();
        const s = { ...this.d.slides[this.i], items: (this.d.slides[this.i].items || []).filter((_, k) => k !== n) };
        this.d.slides[this.i] = s;
        putSlide(s);
      };
      $("#pStage", this.el).addEventListener("click", (e) => {
        const rm = e.target.closest("[data-rm]");
        if (rm) { e.preventDefault(); removeItem(+rm.dataset.rm); }
      });
      $("#pStage", this.el).addEventListener("keydown", (e) => {
        const it = e.target.closest("[data-item]");
        // a selected picture goes with Delete; a text box only when the key is pressed on the box, not while typing
        if (it && e.target === it && (e.key === "Delete" || e.key === "Backspace")) { e.preventDefault(); removeItem(+it.dataset.item); }
      });
      $("#pStage", this.el).addEventListener("pointerdown", (e) => {
        const g = e.target.closest("[data-grip]");
        if (!g) return;
        e.preventDefault();
        const n = +g.dataset.grip, box = g.closest(".it"), slide = box.closest(".slide").getBoundingClientRect();
        const s = this.d.slides[this.i], it = { ...s.items[n] };
        const sx = e.clientX, sy = e.clientY, x0 = +it.x || 0, y0 = +it.y || 0;
        const moveTo = (ev) => {
          it.x = Math.max(0, Math.min(95, x0 + ((ev.clientX - sx) / slide.width) * 100));
          it.y = Math.max(0, Math.min(95, y0 + ((ev.clientY - sy) / slide.height) * 100));
          box.style.left = it.x + "%"; box.style.top = it.y + "%";
        };
        const up = async () => {
          document.removeEventListener("pointermove", moveTo); document.removeEventListener("pointerup", up);
          await this.save();
          const cur = this.d.slides[this.i], items = (cur.items || []).slice();
          items[n] = { ...items[n], x: Math.round(it.x * 10) / 10, y: Math.round(it.y * 10) / 10 };
          this.d.slides[this.i] = { ...cur, items };
          putSlide(this.d.slides[this.i]);
        };
        document.addEventListener("pointermove", moveTo); document.addEventListener("pointerup", up);
      });
      $("#pText", this.el).onclick = async () => {
        if (!this.d.slides.length) return toast("add a slide first", "bad");
        await this.save();
        const s = this.d.slides[this.i];
        const next = { ...s, items: [...(s.items || []), { kind: "text", text: "Text", x: 10, y: 72, w: 40, size: 18 }] };
        this.d.slides[this.i] = next;
        await putSlide(next);
        const t = $(`#pStage [data-item="${next.items.length - 1}"] .it-t`, this.el);
        if (t) { t.focus(); document.execCommand("selectAll"); }
      };
      $("#pPic", this.el).onclick = async () => {
        if (!this.d.slides.length) return toast("add a slide first", "bad");
        if (!window.mlmiDesktop) return toast("Add a picture from the desktop app, or ask Sheep to place one", "bad");
        const src = await window.mlmiDesktop.pickFile({ filters: [{ name: "Pictures", extensions: ["png", "jpg", "jpeg"] }] });
        if (!src) return;
        await this.save();
        let path;
        try { ({ path } = await api("/api/import_media", { path: src })); } catch (e) { return toast(e.message, "bad"); }
        const s = this.d.slides[this.i];
        const next = { ...s, items: [...(s.items || []), { kind: "image", src: path, x: 55, y: 25, w: 35 }] };
        this.d.slides[this.i] = next;
        putSlide(next);
      };
      $("#pDup", this.el).onclick = async () => {
        if (!this.d.slides.length) return;
        await this.save();
        const copy = JSON.parse(JSON.stringify(this.d.slides[this.i]));
        MLMI.quiet(this);
        try {
          await op("slides_add", { path: this.file, title: copy.title || "Copy", after: this.i + 1 });
          await op("slides_put", { path: this.file, slide: this.i + 2, data: copy });
        } catch (e) { return; }
        this.i += 1;
        this.load();
      };
      $("#pStage", this.el).addEventListener("paste", (e) => {
        if (!e.target.closest("[data-f]")) return;
        e.preventDefault();
        document.execCommand("insertText", false, e.clipboardData.getData("text/plain"));
      });
      $("#pAdd", this.el).onclick = async () => {
        await this.save();
        MLMI.quiet(this);
        await op("slides_add", { path: this.file, title: "New slide", bullets: ["First point"], after: this.d.slides.length ? this.i + 1 : undefined });
        this.i = this.d.slides.length ? this.i + 1 : 0;
        await this.load();
        const t = $("#pStage [data-f=title]", this.el);
        if (t) { t.focus(); document.execCommand("selectAll"); }
      };
      $("#pLayout", this.el).onchange = async (e) => {
        await this.save();
        const s = this.d.slides[this.i];
        if (!s) return;
        s.layout = e.target.value;
        MLMI.quiet(this);
        await op("slides_put", { path: this.file, slide: this.i + 1, data: s });
        this.render();
      };
      const mv = async (d) => {
        const to = this.i + d;
        if (to < 0 || to >= this.d.slides.length) return;
        await this.save();
        MLMI.quiet(this);
        await op("slides_move", { path: this.file, slide: this.i + 1, to: to + 1 });
        this.i = to;
        this.load();
      };
      $("#pUp", this.el).onclick = () => mv(-1);
      $("#pDown", this.el).onclick = () => mv(1);
      $("#pDel", this.el).onclick = async () => {
        if (!this.d.slides.length) return;
        MLMI.quiet(this);
        await op("slides_delete", { path: this.file, slide: this.i + 1 });
        this.i = Math.max(0, this.i - 1);
        this.load();
        toast("slide deleted; undo brings it back");
      };
      $("#pTheme", this.el).onchange = async (e) => {
        MLMI.quiet(this);
        await op("slides_set_theme", { path: this.file, theme: e.target.value });
        this.load();
      };
      $("#pShow", this.el).onclick = async () => { await this.save(); if (this.d.slides.length) this.present(); };
      $("#pExport", this.el).onclick = async () => { await this.save(); MLMI.exportBtn(this.file); };
      $("#pPdf", this.el).onclick = async () => { await this.save(); MLMI.exportBtn(this.file, "pdf"); };
      $("#pClose", this.el).onclick = () => { S.open.slides = ""; MLMI.go("slides"); };
    },
  };
})();
