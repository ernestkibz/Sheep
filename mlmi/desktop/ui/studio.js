// Sheep Studio: video and motion as one app. One frame, one timeline; text,
// pictures, clips, shapes and music are all layers. The live preview below
// draws the project with the same motion maths as the renderer
// (backend/tools/studio.py), so what plays here is what the MP4 shows.
"use strict";

(() => {
  const { $, $$, esc, api, op, toast, S } = MLMI;
  const MOVES = ["none", "fade", "pop", "slide-up", "slide-left", "zoom", "wipe", "typewriter"];
  const TEMPLATES = [["title-intro", "Title intro"], ["logo-reveal", "Logo reveal"], ["outro", "Outro end card"], ["short", "Short (9:16 style)"], ["lower-third", "Lower third"]];
  const KIND_ICON = { clip: "play", image: "doc", text: "pencil", shape: "spark", audio: "cloud" };
  const COLORS = { clip: "#bdbdbd", image: "#8f8f8f", text: "#e6e6e6", shape: "#666666", audio: "#a5a5a5" };
  const MEDIA = /\.(mp4|mov|webm|mkv|avi|png|jpe?g|webp|gif|bmp|mp3|wav|m4a|aac|ogg|flac)$/i;

  // ------------------------------------------------------------ the motion maths (studio.py mirrors it)
  const clamp = (u) => (u < 0 ? 0 : u > 1 ? 1 : u);
  const easeOut = (u) => 1 - (1 - u) ** 3;
  const easeBack = (u) => { const c1 = 1.70158; return 1 + (c1 + 1) * (u - 1) ** 3 + c1 * (u - 1) ** 2; };
  const easeInOut = (u) => (u < 0.5 ? 4 * u ** 3 : 1 - (-2 * u + 2) ** 3 / 2);
  // the part of the source shown: [x, y, w, h] in % (studio.py crop_box mirrors this)
  function cropBox(L, sw, sh) {
    const c = Array.isArray(L.crop) ? L.crop.map(Number) : null;
    if (!c || c.length !== 4 || c.some((v) => !isFinite(v))) return [0, 0, sw, sh];
    const x = Math.max(0, Math.min(95, c[0])), y = Math.max(0, Math.min(95, c[1]));
    const w = Math.max(5, Math.min(100 - x, c[2])), h = Math.max(5, Math.min(100 - y, c[3]));
    return [sw * x / 100, sh * y / 100, sw * w / 100, sh * h / 100];
  }
  // draw a source part into a box: cover fills and is clipped, contain shows all of it
  function fitDraw(g, src, sx, sy, sw, sh, bw, bh, cover) {
    const r = cover ? Math.max(bw / sw, bh / sh) : Math.min(bw / sw, bh / sh), w = sw * r, h = sh * r;
    if (cover) { g.beginPath(); g.rect(-bw / 2, -bh / 2, bw, bh); g.clip(); }
    g.drawImage(src, sx, sy, sw, sh, -w / 2, -h / 2, w, h);
  }

  function state(L, t) {
    const start = +L.start || 0, dur = +L.dur || 3, local = t - start;
    if (local < 0 || local >= dur) return null;
    const s = { x: +(L.x ?? 50), y: +(L.y ?? 50), w: +(L.w ?? 60), h: +(L.h ?? 20), scale: 1, rot: +(L.rotate || 0), op: +(L.opacity ?? 1), wipe: 1, chars: null };
    const keys = (L.keys || []).slice().sort((a, b) => (a.t || 0) - (b.t || 0));
    if (keys.length) {
      const map = [["x", "x"], ["y", "y"], ["scale", "scale"], ["rotate", "rot"], ["opacity", "op"]];
      let prev = { t: 0 }; map.forEach(([a, b]) => (prev[a] = s[b]));
      let done = false;
      for (const k of keys) {
        if (local <= (k.t || 0)) {
          const u = easeInOut(clamp((local - prev.t) / Math.max(1e-6, (k.t || 0) - prev.t)));
          map.forEach(([a, b]) => { if (a in k) s[b] = (prev[a] ?? s[b]) + (k[a] - (prev[a] ?? s[b])) * u; });
          done = true; break;
        }
        prev = { ...prev, ...k };
      }
      if (!done) map.forEach(([a, b]) => { if (a in prev) s[b] = prev[a]; });
    }
    for (const [side, u0] of [["in", local / Math.max(1e-6, +((L.in || {}).dur ?? 0.6))], ["out", (dur - local) / Math.max(1e-6, +((L.out || {}).dur ?? 0.5))]]) {
      const mv = (L[side] || {}).move || "none";
      if (mv === "none" || u0 >= 1) continue;
      const u = clamp(u0), e = easeOut(u);
      if (mv === "fade") s.op *= e;
      else if (mv === "pop") { s.scale *= side === "in" ? Math.max(0, easeBack(u)) : e; s.op *= Math.min(1, u * 3); }
      else if (mv === "slide-up") { s.y += (1 - e) * 12; s.op *= e; }
      else if (mv === "slide-left") { s.x += (1 - e) * 15; s.op *= e; }
      else if (mv === "zoom") { s.scale *= 0.6 + 0.4 * e; s.op *= e; }
      else if (mv === "wipe") s.wipe = Math.min(s.wipe, e);
      else if (mv === "typewriter") s.chars = u;
    }
    return s;
  }
  const lengthOf = (d) => Math.max(+d.duration || 10, ...d.layers.map((l) => (+l.start || 0) + (+l.dur || 0)));
  const fmtT = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
  const raw = (p) => "/api/raw?path=" + encodeURIComponent(p);

  S.views.studio = {
    mount(el, path) {
      this.el = el;
      this.file = path || "";
      if (!this.file) {
        return MLMI.picker(el, "studio", { ext: "sstudio", office: "mp4", create: "studio_create",
          noun: { title: "Sheep Studio", sub: "videos and motion graphics: text, pictures, clips, shapes and music on one timeline", one: "video", many: "videos", example: "launch intro" } });
      }
      el.innerHTML = `<div class="editor studio">
        <div class="tools"><span class="file">${MLMI.icon("play")} <b>${esc(this.file)}</b></span>
          <button id="stPlay" title="Play or pause (Space)">▶</button><span class="st-time" id="stTime">0:00.0</span><span class="sep"></span>
          <button id="stText">+ Text</button>
          <span class="st-menu"><button id="stMedia">+ Media ▾</button><div class="st-pop" id="stMediaPop"></div></span>
          <span class="st-menu"><button id="stShape">+ Shape ▾</button><div class="st-pop" id="stShapePop">
            <button data-shape="rect">Bar or box</button><button data-shape="circle">Circle</button><button data-shape="line">Line</button></div></span>
          <span class="st-menu"><button id="stTpl">Templates ▾</button><div class="st-pop" id="stTplPop">${TEMPLATES.map(([k, n]) => `<button data-tpl="${k}">${n}</button>`).join("")}</div></span>
          <span style="flex:1"></span>
          <button id="stTip" title="What decides video speed and quality here">Tips</button>
          <button id="stWatch" title="Watch the last render">Watch</button>
          <button id="stRender" class="primary">Render MP4</button><button id="stClose" title="All videos">Files</button></div>
        <div class="st-main">
          <div class="st-stage" id="stStage"><canvas id="stCanvas"></canvas><video id="stResult" controls hidden></video><div class="st-busy" id="stBusy" hidden></div></div>
          <aside class="st-insp" id="stInsp"></aside>
        </div>
        <div class="st-tl" id="stTl"></div>
      </div>`;
      this.t = 0; this.sel = -1; this.playing = false;
      this.imgs = {}; this.vids = {}; this.auds = {};
      this.wire();
      this.load();
    },
    path() { return this.file; },
    unmount() {
      this.pause(); cancelAnimationFrame(this.raf); window.removeEventListener("keydown", this._keys);
      // each opening adds these; without removal they piled up for the whole session
      if (this._closePops) document.removeEventListener("click", this._closePops);
      if (this._ro) this._ro.disconnect();
      this._closePops = this._ro = null;
      this.el = null;
    },
    reload() { if (!MLMI.isQuiet(this)) this.load(); },
    async load() {
      try { this.d = await api("/api/doc?path=" + encodeURIComponent(this.file)); } catch (e) { toast(e.message, "bad"); return; }
      if (!this.el) return;
      ["path", "length", "moves", "formats"].forEach((k) => delete this.d[k]);
      if (this.sel >= this.d.layers.length) this.sel = -1;
      await document.fonts.load("800 40px Nunito").catch(() => {});
      this.media();
      this.fit();
      this.drawAll();
    },
    // ---------------------------------------------------------- saving: the whole project through the editor's tool
    save() {
      clearTimeout(this._st);
      this._st = setTimeout(async () => {
        MLMI.quiet(this);
        try { await op("studio_put", { path: this.file, data: this.d }); } catch (e) { /* shown */ }
        MLMI.quiet(this);
      }, 350);
    },
    // ---------------------------------------------------------- media elements for the preview
    media() {
      for (const L of this.d.layers) {
        if (L.kind === "image" && !this.imgs[L.src]) { const im = new Image(); im.onload = () => this.draw(); im.src = raw(L.src); this.imgs[L.src] = im; }
        if (L.kind === "clip" && !this.vids[L.src]) { const v = document.createElement("video"); v.muted = true; v.preload = "auto"; v.src = raw(L.src); v.onseeked = () => this.draw(); this.vids[L.src] = v; }
        if (L.kind === "audio" && !this.auds[L.src]) { const a = new Audio(raw(L.src)); a.preload = "auto"; this.auds[L.src] = a; }
      }
    },
    fit() {
      const st = $("#stStage", this.el), c = $("#stCanvas", this.el);
      const W = st.clientWidth - 24, H = st.clientHeight - 24, r = this.d.w / this.d.h;
      const w = Math.max(160, Math.min(W, H * r)), h = w / r;
      c.style.width = w + "px"; c.style.height = h + "px";
      const dpr = window.devicePixelRatio || 1;
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
      this.k = c.width / this.d.w;          // project pixels to canvas pixels
    },
    // ---------------------------------------------------------- drawing one frame
    draw() {
      if (!this.el || !this.d) return;
      const c = $("#stCanvas", this.el), g = c.getContext("2d"), d = this.d, k = this.k, t = this.t;
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.fillStyle = "#" + (d.bg || "0B0B0B"); g.fillRect(0, 0, c.width, c.height);
      d.layers.forEach((L, i) => {
        if (L.kind === "audio") return;
        const s = state(L, t);
        if (!s) return;
        const bw = d.w * s.w / 100 * k, bh = d.h * s.h / 100 * k, cx = d.w * s.x / 100 * k, cy = d.h * s.y / 100 * k;
        g.save();
        g.globalAlpha = Math.max(0, Math.min(1, s.op));
        g.translate(cx, cy); g.rotate((s.rot || 0) * Math.PI / 180); g.scale(s.scale, s.scale);
        if (s.wipe < 1) { g.beginPath(); g.rect(-bw / 2 - 2000, -bh / 2 - 2000, (bw + 4000) * 0 + bw * s.wipe + 2000, bh + 4000); g.clip(); }
        if (L.kind === "text") {
          let text = String(L.text || "");
          if (s.chars !== null) text = text.slice(0, Math.max(0, Math.ceil(text.length * s.chars)));
          const size = (+L.size || 72) * d.h / 1080 * k;
          g.font = `${+L.weight || 800} ${size}px Nunito, sans-serif`;
          g.fillStyle = "#" + (L.color || "FFFFFF"); g.textBaseline = "top";
          const lines = []; text.split("\n").forEach((para) => { let line = ""; para.split(" ").forEach((w) => {
            const test = (line + " " + w).trim(); if (line && g.measureText(test).width > bw) { lines.push(line); line = w; } else line = test; }); lines.push(line); });
          const lh = size * 1.18, total = lh * lines.length + size * 0.35;
          const tw = Math.max(bw, ...lines.map((ln) => g.measureText(ln).width));
          lines.forEach((ln, j) => {
            const lw = g.measureText(ln).width, al = L.align || "center";
            const x = al === "left" ? -tw / 2 : al === "right" ? tw / 2 - lw : -lw / 2;
            g.fillText(ln, x, -total / 2 + j * lh);
          });
        } else if (L.kind === "shape") {
          g.fillStyle = "#" + (L.fill || "A3A3A3");
          const h2 = Math.max(bh, 2);
          if (L.shape === "circle") { g.beginPath(); g.ellipse(0, 0, bw / 2, h2 / 2, 0, 0, Math.PI * 2); g.fill(); }
          else if (L.shape === "line") { const th = Math.max(2, h2 / 4); g.fillRect(-bw / 2, -th / 2, bw, th); }
          else { const r = Math.min((+L.radius || 12) * d.h / 1080 * k, h2 / 2, bw / 2); g.beginPath(); g.roundRect(-bw / 2, -h2 / 2, bw, h2, r); g.fill(); }
        } else if (L.kind === "image") {
          const im = this.imgs[L.src];
          if (im && im.complete && im.naturalWidth) {
            const [sx, sy, sw, sh] = cropBox(L, im.naturalWidth, im.naturalHeight);
            fitDraw(g, im, sx, sy, sw, sh, bw, bh, L.fit === "cover");
          }
        } else if (L.kind === "clip") {
          const v = this.vids[L.src];
          if (v && v.readyState >= 2) {
            const [sx, sy, sw, sh] = cropBox(L, v.videoWidth, v.videoHeight);
            fitDraw(g, v, sx, sy, sw, sh, bw, bh, true);
          } else { g.fillStyle = "#292929"; g.fillRect(-bw / 2, -bh / 2, bw, bh); }
        }
        g.restore();
        if (i === this.sel) {
          g.save(); g.strokeStyle = "#f1f1f1"; g.lineWidth = 2; g.setLineDash([6, 4]);
          g.strokeRect(cx - bw * s.scale / 2, cy - bh * s.scale / 2, bw * s.scale, bh * s.scale); g.restore();
        }
      });
      $("#stTime", this.el).textContent = `${fmtT(t)} / ${fmtT(lengthOf(d))}`;
      const ph = $("#stTl .st-head", this.el);
      if (ph) ph.style.left = `${this.x0 + t * this.pps}px`;
    },
    // keep clips at the right moment while scrubbing (paused)
    seekClips() {
      for (const L of this.d.layers) {
        if (L.kind !== "clip") continue;
        const v = this.vids[L.src], local = this.t - (+L.start || 0);
        if (v && local >= 0 && local < +L.dur) { const want = (+L.trim || 0) + local * (+L.speed || 1); if (Math.abs(v.currentTime - want) > 0.04) v.currentTime = want; }
      }
    },
    drawAll() { this.timeline(); this.inspector(); this.seekClips(); this.draw(); },
    // ---------------------------------------------------------- play and pause
    play() {
      if (this.playing) return;
      if (this.t >= lengthOf(this.d) - 0.05) this.t = 0;
      this.playing = true; $("#stPlay", this.el).textContent = "❚❚";
      const t0 = performance.now() - this.t * 1000;
      const step = () => {
        if (!this.playing || !this.el) return;
        this.t = (performance.now() - t0) / 1000;
        if (this.t >= lengthOf(this.d)) { this.t = lengthOf(this.d); this.pause(); this.draw(); return; }
        for (const L of this.d.layers) {
          const local = this.t - (+L.start || 0), on = local >= 0 && local < +L.dur;
          const m = L.kind === "clip" ? this.vids[L.src] : L.kind === "audio" ? this.auds[L.src] : null;
          if (!m) continue;
          if (on) {
            // a layer's speed: slow motion or fast, as the renderer plays it (studio.py _Clip, atempo)
            const sp = Math.min(4, Math.max(0.25, +L.speed || 1)), want = (+L.trim || 0) + local * sp;
            if (m.playbackRate !== sp) m.playbackRate = sp;
            if (m.paused) { m.currentTime = want; m.play().catch(() => {}); }
            else if (Math.abs(m.currentTime - want) > 0.3) m.currentTime = want;
            if (L.kind === "audio") m.volume = Math.max(0, Math.min(1, +(L.volume ?? 1)));
          } else if (!m.paused) m.pause();
        }
        this.draw();
        this.raf = requestAnimationFrame(step);
      };
      this.raf = requestAnimationFrame(step);
    },
    pause() {
      this.playing = false;
      cancelAnimationFrame(this.raf);
      // `unmount()` can run after a partially initialized Studio load; keep
      // navigation/close actions safe even when one media map is unavailable.
      Object.values(this.vids || {}).concat(Object.values(this.auds || {})).forEach((m) => {
        if (m && typeof m.pause === "function") m.pause();
      });
      const playButton = this.el && $("#stPlay", this.el);
      if (playButton) playButton.textContent = "▶";
    },
    // ---------------------------------------------------------- the timeline
    timeline() {
      const tl = $("#stTl", this.el), d = this.d, len = Math.max(lengthOf(d), 1);
      const width = tl.clientWidth || 800;
      this.x0 = 168; this.pps = Math.max(20, (width - this.x0 - 16) / len);
      const ticks = []; for (let s = 0; s <= len + 0.001; s += len > 30 ? 5 : 1) ticks.push(s);
      tl.innerHTML = `<div class="st-ruler">${ticks.map((s) => `<span style="left:${this.x0 + s * this.pps}px">${s}s</span>`).join("")}</div>
        ${d.layers.map((L, i) => `<div class="st-row ${i === this.sel ? "on" : ""}" data-i="${i}">
          <span class="st-lab">${MLMI.icon(KIND_ICON[L.kind] || "doc")}<span>${esc(L.kind === "text" ? L.text : L.src ? L.src.split("/").pop() : L.shape || L.kind)}</span></span>
          <span class="st-bar" style="left:${this.x0 + (+L.start || 0) * this.pps}px;width:${Math.max(6, (+L.dur || 0) * this.pps)}px;--c:${COLORS[L.kind]}"><i class="st-grip"></i></span></div>`).join("")
        || `<div class="st-empty">Add text, media or a shape above, or pick a template.</div>`}
        <div class="st-head" style="left:${this.x0 + this.t * this.pps}px"></div>`;
    },
    // ---------------------------------------------------------- the selected layer's settings
    inspector() {
      const box = $("#stInsp", this.el), d = this.d;
      if (this.sel < 0) {
        box.innerHTML = `<h3>Video</h3>
          <label>Shape<select data-p="format">${["16:9", "9:16", "1:1"].map((f) => `<option ${d.format === f ? "selected" : ""}>${f}</option>`).join("")}</select></label>
          <label>Length (s)<input type="number" step="0.5" min="1" data-p="duration" value="${d.duration}"></label>
          <label>Background<input type="color" data-p="bg" value="#${d.bg}"></label>
          <p class="note">Pick a layer on the timeline or in the frame to change it.</p>`;
        return;
      }
      const L = d.layers[this.sel];
      const num = (k, lab, step = 1, val = L[k]) => `<label>${lab}<input type="number" step="${step}" data-k="${k}" value="${(+(val ?? 0)).toFixed(step < 1 ? 1 : 0)}"></label>`;
      const mv = (side, lab) => `<label>${lab}<select data-move="${side}">${MOVES.map((m) => `<option ${((L[side] || {}).move || "none") === m ? "selected" : ""}>${m}</option>`).join("")}</select></label>`;
      box.innerHTML = `<h3>${esc(L.kind[0].toUpperCase() + L.kind.slice(1))} <span class="dim">· layer ${this.sel + 1}</span></h3>
        ${L.kind === "text" ? `<label class="wide">Words<textarea data-k="text" rows="3">${esc(L.text || "")}</textarea></label>` : ""}
        ${L.src ? `<p class="note">${esc(L.src)}</p>` : ""}
        <div class="grid2">${num("start", "Starts (s)", 0.1)}${num("dur", "Length (s)", 0.1)}</div>
        ${L.kind !== "audio" ? `<div class="grid2">${num("x", "Across %")}${num("y", "Down %")}${num("w", "Width %")}${num("h", "Height %")}</div>` : ""}
        ${L.kind === "text" ? `<div class="grid2">${num("size", "Text size")}<label>Colour<input type="color" data-k="color" value="#${L.color || "FFFFFF"}"></label></div>
          <div class="grid2"><label>Weight<select data-k="weight">${[["400", "regular"], ["600", "semibold"], ["800", "bold"], ["900", "black"]].map(([v, n]) =>
            `<option value="${v}" ${String(L.weight || 800) === v ? "selected" : ""}>${n}</option>`).join("")}</select></label>
          <label>Align<select data-k="align">${["left", "center", "right"].map((a) => `<option ${(L.align || "center") === a ? "selected" : ""}>${a}</option>`).join("")}</select></label></div>` : ""}
        ${L.kind !== "audio" ? num("rotate", "Turn (degrees)", 1, L.rotate || 0) : ""}
        ${L.kind === "shape" ? `<label>Colour<input type="color" data-k="fill" value="#${L.fill || "A3A3A3"}"></label>` : ""}
        ${L.kind !== "audio" ? `<div class="grid2">${mv("in", "Comes in")}${mv("out", "Goes out")}</div>${num("opacity", "Opacity (0-1)", 0.1, L.opacity ?? 1)}` : ""}
        ${L.kind === "clip" || L.kind === "audio" ? `<div class="grid2">${num("volume", "Volume (0-1)", 0.1, L.volume ?? 1)}${num("trim", "Skip start (s)", 0.1, L.trim || 0)}</div>
          <div class="grid2">${num("speed", "Speed (0.5 slow, 2 fast)", 0.25, L.speed ?? 1)}</div>` : ""}
        ${L.kind === "image" || L.kind === "clip" ? `<details class="st-more"${L.crop ? " open" : ""}><summary>Crop the ${L.kind === "clip" ? "clip" : "picture"}</summary>
          <div class="grid2">${["Left %", "Top %", "Width %", "Height %"].map((lab, j) =>
            `<label>${lab}<input type="number" step="1" min="0" max="100" data-crop="${j}" value="${(Array.isArray(L.crop) ? L.crop : [0, 0, 100, 100])[j]}"></label>`).join("")}</div>
          ${L.kind === "image" ? `<label>Fill<select data-k="fit"><option value="contain" ${L.fit !== "cover" ? "selected" : ""}>Show all of it</option>
            <option value="cover" ${L.fit === "cover" ? "selected" : ""}>Fill the box</option></select></label>` : ""}</details>` : ""}
        <div class="grid2"><label>Fade in (s)<input type="number" step="0.1" min="0" data-fade="in" value="${(L.in || {}).move === "fade" ? (+(L.in.dur || 0.6)).toFixed(1) : "0.0"}"></label>
          <label>Fade out (s)<input type="number" step="0.1" min="0" data-fade="out" value="${(L.out || {}).move === "fade" ? (+(L.out.dur || 0.5)).toFixed(1) : "0.0"}"></label></div>
        ${L.kind === "text" || L.kind === "shape" ? "" : `<button data-act="split" title="Cut this layer in two where the playhead is">Split at playhead</button>`}
        <div class="grid2"><button data-act="up">Bring forward</button><button data-act="down">Send back</button></div>
        <button class="danger" data-act="del">Remove layer</button>`;
    },
    // ---------------------------------------------------------- adding things
    async add(layer) {
      this.d.layers.push(layer);
      this.sel = this.d.layers.length - 1;
      this.media(); this.drawAll(); this.save();
    },
    async addMedia(rel) {
      MLMI.quiet(this);
      try { await op("studio_add_media", { path: this.file, file: rel, start: Math.round(this.t * 10) / 10 }); } catch (e) { return; }
      await this.load();
      this.sel = this.d.layers.findIndex((l) => l.src === rel && Math.abs((+l.start || 0) - Math.round(this.t * 10) / 10) < 0.05);
      this.drawAll();
    },
    mediaMenu() {
      const pop = $("#stMediaPop", this.el);
      const files = S.files.filter((f) => MEDIA.test(f.path));
      pop.innerHTML = `<button data-pick="1">Import from this computer…</button><div class="mg">Workspace media</div>`
        + (files.length ? files.map((f) => `<button data-media="${esc(f.path)}">${esc(f.path)}</button>`).join("")
          : `<small class="dim">AI clips, rendered MP4s, pictures and sound appear here.</small>`);
      pop.classList.toggle("open");
    },
    // ---------------------------------------------------------- events
    wire() {
      const el = this.el, closePops = () => $$(".st-pop.open", el).forEach((p) => p.classList.remove("open"));
      $("#stPlay", el).onclick = () => (this.playing ? this.pause() : this.play());
      $("#stText", el).onclick = () => this.add({ kind: "text", text: "Your text", start: Math.round(this.t * 10) / 10, dur: 3, x: 50, y: 50, w: 80, h: 16,
        size: 90, color: "FFFFFF", weight: 800, in: { move: "pop", dur: 0.6 }, out: { move: "fade", dur: 0.5 } });
      $("#stMedia", el).onclick = (e) => { e.stopPropagation(); closePops(); this.mediaMenu(); };
      $("#stMediaPop", el).onclick = async (e) => {
        const b = e.target.closest("button"); if (!b) return; closePops();
        if (b.dataset.media) return this.addMedia(b.dataset.media);
        if (!window.mlmiDesktop) return toast("Pick a file in the desktop app, or put it in the workspace's media folder", "bad");
        const p = await window.mlmiDesktop.pickFile({ filters: [{ name: "Video, pictures, music", extensions: ["mp4", "mov", "webm", "png", "jpg", "jpeg", "webp", "gif", "mp3", "wav", "m4a"] }] });
        if (!p) return;
        try { const r = await api("/api/import_media", { path: p }); await MLMI.loadTree(); this.addMedia(r.path); } catch (err) { toast(err.message, "bad"); }
      };
      $("#stShape", el).onclick = (e) => { e.stopPropagation(); closePops(); $("#stShapePop", el).classList.toggle("open"); };
      $("#stShapePop", el).onclick = (e) => {
        const b = e.target.closest("[data-shape]"); if (!b) return; closePops();
        const sh = b.dataset.shape;
        this.add({ kind: "shape", shape: sh, start: Math.round(this.t * 10) / 10, dur: 3, x: 50, y: 50, w: sh === "circle" ? 12 : 30, h: sh === "circle" ? 21 : sh === "line" ? 1 : 10,
          fill: "A3A3A3", radius: 12, opacity: 1, in: { move: "fade", dur: 0.6 }, out: { move: "fade", dur: 0.5 } });
      };
      $("#stTpl", el).onclick = (e) => { e.stopPropagation(); closePops(); $("#stTplPop", el).classList.toggle("open"); };
      $("#stTplPop", el).onclick = async (e) => {
        const b = e.target.closest("[data-tpl]"); if (!b) return; closePops();
        MLMI.quiet(this);
        try { await op("studio_template", { path: this.file, template: b.dataset.tpl, start: Math.round(this.t * 10) / 10 }); } catch (err) { return; }
        await this.load(); this.sel = -1; this.drawAll();
        toast("Template added: click its text to change the words");
      };
      this._closePops = closePops;
      document.addEventListener("click", closePops);
      $("#stRender", el).onclick = async () => {
        this.pause();
        const busy = $("#stBusy", el); busy.hidden = false;
        $("#stRender", el).disabled = true;
        // a render can take minutes: show it is alive and how long it has run
        const t0 = Date.now(), secs = Math.max(1, Math.round(lengthOf(this.d)));
        const tick = setInterval(() => {
          const s = Math.round((Date.now() - t0) / 1000);
          busy.textContent = `Rendering ${secs} s of video on this machine… ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
        }, 500);
        busy.textContent = `Rendering ${secs} s of video on this machine…`;
        try {
          await op("studio_render", { path: this.file });
          this.lastOut = this.file.replace(/\.sstudio$/i, ".mp4");
          toast(`Rendered ${this.lastOut}. Watch it here or add it with + Media.`);
          this.watch(true);
          MLMI.loadTree();
        } catch (err) { /* shown */ }
        clearInterval(tick);
        if (!this.el) return;
        busy.hidden = true; $("#stRender", el).disabled = false;
      };
      $("#stWatch", el).onclick = () => this.watch(!this.watching);
      $("#stTip", el).onclick = () => MLMI.videoTip();
      $("#stClose", el).onclick = () => { S.open.studio = ""; MLMI.go("studio"); };
      // the timeline: click to seek or select, drag a bar to move it, its end to change its length
      const tl = $("#stTl", el);
      tl.addEventListener("pointerdown", (e) => {
        const row = e.target.closest(".st-row"), bar = e.target.closest(".st-bar");
        const rect = tl.getBoundingClientRect(), x = e.clientX - rect.left + tl.scrollLeft;
        if (row) { this.sel = +row.dataset.i; this.inspector(); this.draw(); $$(".st-row", tl).forEach((r) => r.classList.toggle("on", r === row)); }
        if (bar && row) {
          const L = this.d.layers[+row.dataset.i], grip = e.target.classList.contains("st-grip");
          const sx = e.clientX, s0 = +L.start || 0, d0 = +L.dur || 1;
          const move = (ev) => {
            const dt = Math.round(((ev.clientX - sx) / this.pps) * 10) / 10;
            if (grip) L.dur = Math.max(0.2, d0 + dt); else L.start = Math.max(0, s0 + dt);
            bar.style.left = `${this.x0 + L.start * this.pps}px`; bar.style.width = `${Math.max(6, L.dur * this.pps)}px`;
            this.draw();
          };
          const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); this.drawAll(); this.save(); };
          window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
          return;
        }
        if (x > this.x0 - 4) {
          const seek = (ev) => { this.t = Math.max(0, Math.min(lengthOf(this.d), (ev.clientX - rect.left + tl.scrollLeft - this.x0) / this.pps)); this.seekClips(); this.draw(); };
          seek(e);
          const up = () => { window.removeEventListener("pointermove", seek); window.removeEventListener("pointerup", up); };
          window.addEventListener("pointermove", seek); window.addEventListener("pointerup", up);
        }
      });
      // the frame: click a layer to pick it, drag to place it
      const cv = $("#stCanvas", el);
      cv.addEventListener("pointerdown", (e) => {
        const r = cv.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * 100, py = (e.clientY - r.top) / r.height * 100;
        let hit = -1;
        this.d.layers.forEach((L, i) => {
          if (L.kind === "audio") return;
          const s = state(L, this.t); if (!s) return;
          if (Math.abs(px - s.x) <= s.w / 2 && Math.abs(py - s.y) <= s.h / 2) hit = i;
        });
        this.sel = hit; this.drawAll();
        if (hit < 0) return;
        const L = this.d.layers[hit], x0 = +L.x, y0 = +L.y;
        const move = (ev) => {
          L.x = Math.round((x0 + ((ev.clientX - e.clientX) / r.width) * 100) * 10) / 10;
          L.y = Math.round((y0 + ((ev.clientY - e.clientY) / r.height) * 100) * 10) / 10;
          this.draw();
        };
        const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); this.inspector(); this.save(); };
        window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
      });
      // the settings panel
      const insp = $("#stInsp", el);
      insp.addEventListener("input", (e) => {
        const f = e.target;
        if (this.sel < 0 && f.dataset.p) {
          const k = f.dataset.p;
          if (k === "bg") this.d.bg = f.value.slice(1).toUpperCase();
          else if (k === "duration") this.d.duration = Math.max(1, +f.value || 10);
          this.draw(); this.save(); return;
        }
        const L = this.d.layers[this.sel]; if (!L) return;
        if (f.dataset.move) L[f.dataset.move] = { ...(L[f.dataset.move] || { dur: 0.6 }), move: f.value };
        else if (f.dataset.crop) {
          const c = Array.isArray(L.crop) ? L.crop.slice() : [0, 0, 100, 100];
          c[+f.dataset.crop] = Math.max(0, Math.min(100, +f.value || 0));
          L.crop = c.join() === "0,0,100,100" ? undefined : c;
        } else if (f.dataset.fade) {
          const v = Math.max(0, +f.value || 0);
          L[f.dataset.fade] = { ...(L[f.dataset.fade] || {}), move: v > 0 ? "fade" : "none", dur: v || 0.5 };
        }
        else if (f.dataset.k) {
          const k = f.dataset.k;
          L[k] = f.type === "color" ? f.value.slice(1).toUpperCase() : f.type === "number" ? +f.value : f.value;
        }
        if (f.dataset.k === "start" || f.dataset.k === "dur") this.timeline();
        this.draw(); this.save();
      });
      insp.addEventListener("change", (e) => {
        if (e.target.dataset.p === "format") {
          const fm = e.target.value, size = { "16:9": [1920, 1080], "9:16": [1080, 1920], "1:1": [1080, 1080] }[fm];
          Object.assign(this.d, { format: fm, w: size[0], h: size[1] });
          this.fit(); this.draw(); this.save();
        }
      });
      insp.addEventListener("click", (e) => {
        const b = e.target.closest("[data-act]"); if (!b || this.sel < 0) return;
        const a = b.dataset.act, L = this.d.layers;
        if (a === "del") { L.splice(this.sel, 1); this.sel = -1; }
        else if (a === "split") {
          // the same cut as studio_split: the second part plays on from where the first stopped
          const one = L[this.sel], at = this.t, st = +one.start || 0, du = +one.dur || 3;
          if (!(at > st + 0.1 && at < st + du - 0.1)) { MLMI.toast("Move the playhead inside this layer to split it", "bad"); return; }
          const two = JSON.parse(JSON.stringify(one));
          one.dur = +(at - st).toFixed(3);
          two.start = +at.toFixed(3); two.dur = +(st + du - at).toFixed(3);
          if (one.kind === "clip" || one.kind === "audio") two.trim = +((+one.trim || 0) + (at - st) * (+one.speed || 1)).toFixed(3);
          one.out = { move: "none", dur: 0.5 }; two.in = { move: "none", dur: 0.6 };
          L.splice(this.sel + 1, 0, two);
        }
        else if (a === "up" && this.sel < L.length - 1) { [L[this.sel], L[this.sel + 1]] = [L[this.sel + 1], L[this.sel]]; this.sel++; }
        else if (a === "down" && this.sel > 0) { [L[this.sel], L[this.sel - 1]] = [L[this.sel - 1], L[this.sel]]; this.sel--; }
        this.drawAll(); this.save();
      });
      this._keys = (e) => {
        if (!this.el || e.target.closest("input, textarea, select, [contenteditable=true]")) return;
        if (e.key === "Escape" && this._closePops) this._closePops();
        if (e.code === "Space") { e.preventDefault(); this.playing ? this.pause() : this.play(); }
        if ((e.key === "Delete" || e.key === "Backspace") && this.sel >= 0) { this.d.layers.splice(this.sel, 1); this.sel = -1; this.drawAll(); this.save(); }
      };
      window.addEventListener("keydown", this._keys);
      this._ro = new ResizeObserver(() => { if (this.el && this.d) { this.fit(); this.timeline(); this.draw(); } });
      this._ro.observe($("#stStage", el));
    },
    watch(on) {
      const v = $("#stResult", this.el), c = $("#stCanvas", this.el);
      const out = this.lastOut || this.file.replace(/\.sstudio$/i, ".mp4");
      if (on && !S.files.some((f) => f.path === out) && !this.lastOut) return toast("Render it first", "bad");
      this.watching = on;
      v.hidden = !on; c.hidden = on;
      $("#stWatch", this.el).classList.toggle("on", on);
      if (on) { this.pause(); v.src = raw(out) + "&v=" + Date.now(); v.play().catch(() => {}); } else v.pause();
    },
  };
})();
