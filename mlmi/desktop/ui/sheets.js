// Sheep Sheets: the .ssheet editor. Formulas stay formulas and are evaluated by
// the backend (tools/formula.py), so the grid, the agent and an exported .xlsx agree.
"use strict";

(() => {
  const { $, $$, esc, api, op, toast, S } = MLMI;
  const col = (n) => { let s = ""; for (n++; n > 0; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + ((n - 1) % 26)) + s; return s; };
  const colIdx = (a) => [...a.toUpperCase()].reduce((v, ch) => v * 26 + ch.charCodeAt(0) - 64, 0) - 1;
  const parse = (ref) => { const m = /^([A-Z]+)(\d+)$/i.exec(ref || ""); return m ? [colIdx(m[1]), +m[2] - 1] : null; };
  const ref = (c, r) => `${col(c)}${r + 1}`;

  function numFmt(v, nf) {
    if (typeof v !== "number" || !nf) return typeof v === "number" ? String(Math.round(v * 1e6) / 1e6) : v;
    const dec = (nf.split(".")[1] || "").replace(/[^0]/g, "").length;
    if (nf.includes("%")) return (v * 100).toFixed(dec) + "%";
    let s = nf.includes(",") ? v.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec }) : v.toFixed(dec);
    if (nf.includes("$")) s = "$" + s;
    return s;
  }

  S.views.sheets = {
    mount(el, path) {
      this.el = el;
      this.file = path || "";
      if (!this.file) {
        return MLMI.picker(el, "sheets", { ext: "ssheet", office: "xlsx", create: "sheets_create",
          noun: { title: "Sheep Sheets", sub: "workbooks with live formulas; .xlsx and .csv in and out", one: "workbook", many: "workbooks", example: "budget" } });
      }
      el.innerHTML = `<div class="editor">
        <div class="tools"><span class="file">▦ <b>${esc(this.file)}</b></span>
          <button id="sBold" title="Bold"><b>B</b></button>
          <label class="dim" title="Fill colour">Fill <input type="color" id="sFill" value="#e5e5e5" style="width:28px;height:22px;padding:0;border:0;background:none"></label>
          <select id="sNf" title="Number format"><option value="">Number format</option><option>0</option><option>0.00</option><option>#,##0</option>
            <option>#,##0.00</option><option>$#,##0.00</option><option>0%</option><option>0.0%</option></select>
          <button data-al="left">⟸</button><button data-al="center">≡</button><button data-al="right">⟹</button><span class="sep"></span>
          <button id="sSortA" title="Sort the selection by its first column, A to Z">Sort ↑</button><button id="sSortD" title="Sort, Z to A">Sort ↓</button>
          <select id="sChart" title="Chart the selection"><option value="">+ Chart</option><option value="bar">bar</option><option value="line">line</option><option value="pie">pie</option></select>
          <span style="flex:1"></span><button id="sCsv" title="Save the sheet you are on as .csv">Export .csv</button><button id="sPdf" title="Save as PDF">PDF</button><button id="sExport" class="primary">Export .xlsx</button><button id="sClose" title="All workbooks">Files</button></div>
        <div class="fbar"><input class="ref" id="sRef" value="A1"><span class="dim">ƒ</span><input type="text" id="sF" placeholder="a value, or a formula like =SUM(B2:B9)"></div>
        <div class="gridwrap" id="sWrap" style="flex:1;min-height:0"></div>
        <div class="charts" id="sCharts" style="display:none"></div>
        <div class="tabs" id="sTabs"></div></div>`;
      this.sel = { c: 0, r: 0, c2: 0, r2: 0 };
      this.sheet = 0;
      this.wire();
      this.load();
    },
    path() { return this.file; },
    unmount() { if (this._up) document.removeEventListener("mouseup", this._up); this._up = null; this.el = null; },
    reload() { if (!MLMI.isQuiet(this)) this.load().then(() => $$("td", this.el).forEach((t) => t.classList.add("flash"))); },
    async load() {
      try { this.d = await api("/api/doc?path=" + encodeURIComponent(this.file)); } catch (e) { toast(e.message, "bad"); return; }
      if (!this.el) return;
      this.sheet = Math.min(this.sheet, this.d.sheets.length - 1);
      this.render();
    },
    sh() { return this.d.sheets[this.sheet]; },
    shown(k) {
      const s = this.sh(), raw = s.cells[k];
      let v = typeof raw === "string" && raw.startsWith("=") ? (this.d.values[s.name] || {})[k] : raw;
      if (typeof v === "string" && /^-?\d+(\.\d+)?(e[-+]?\d+)?$/i.test(v)) v = +v;     // computed values arrive as text
      const f = s.fmt[k] || {};
      return { raw, v, f, text: v === null || v === undefined ? "" : typeof v === "boolean" ? (v ? "TRUE" : "FALSE") : String(numFmt(v, f.nf)) };
    },
    render() {
      const s = this.sh();
      let mc = 0, mr = 0;
      for (const k of Object.keys(s.cells)) { const p = parse(k); if (p) { mc = Math.max(mc, p[0]); mr = Math.max(mr, p[1]); } }
      const nc = Math.max(12, mc + 4), nr = Math.max(40, mr + 12);
      let h = `<table class="grid"><thead><tr><th></th>${Array.from({ length: nc }, (_, c) =>
        `<th style="${s.widths[col(c)] ? `min-width:${Math.round(s.widths[col(c)] * 7.5)}px` : ""}">${col(c)}</th>`).join("")}</tr></thead><tbody>`;
      for (let r = 0; r < nr; r++) {
        h += `<tr><th>${r + 1}</th>`;
        for (let c = 0; c < nc; c++) {
          const k = ref(c, r), x = this.shown(k);
          const isErr = typeof x.v === "string" && x.v.startsWith("#");
          const st = `${x.f.b ? "font-weight:700;" : ""}${x.f.bg ? `background:#${x.f.bg};color:#171717;` : ""}${x.f.align ? `text-align:${x.f.align};` : ""}`;
          h += `<td data-k="${k}" class="${typeof x.v === "number" ? "num" : ""} ${isErr ? "err" : ""}" style="${st}" title="${esc(typeof x.raw === "string" && x.raw.startsWith("=") ? x.raw : "")}">${esc(x.text)}</td>`;
        }
        h += "</tr>";
      }
      $("#sWrap", this.el).innerHTML = h + "</tbody></table>";
      $("#sTabs", this.el).innerHTML = this.d.sheets.map((x, i) => `<button class="${i === this.sheet ? "on" : ""}" data-t="${i}">${esc(x.name)}</button>`).join("") + `<button id="sAddSheet" title="Add a sheet">+</button>`;
      this.charts();
      this.paintSel();
    },
    charts() {
      const s = this.sh(), box = $("#sCharts", this.el);
      box.style.display = s.charts.length ? "flex" : "none";
      box.innerHTML = s.charts.map((ch) => this.chartSvg(ch)).join("");
    },
    chartSvg(ch) {
      const [a, b] = ch.range.split(":");
      const p0 = parse(a), p1 = parse(b);
      if (!p0 || !p1) return "";
      const names = [], labels = [], series = [];
      for (let c = p0[0] + 1; c <= p1[0]; c++) { names.push(this.shown(ref(c, p0[1])).text); series.push([]); }
      for (let r = p0[1] + 1; r <= p1[1]; r++) {
        labels.push(this.shown(ref(p0[0], r)).text);
        for (let c = p0[0] + 1; c <= p1[0]; c++) { const v = this.shown(ref(c, r)).v; series[c - p0[0] - 1].push(typeof v === "number" ? v : 0); }
      }
      const W = 380, H = 220, pad = 30, pal = ["#dedede", "#a8a8a8", "#ffffff", "#737373", "#c2c2c2", "#555555"];
      const max = Math.max(1e-9, ...series.flat().map(Math.abs));
      let g = "";
      if (ch.kind === "pie") {
        const vals = series[0] || [], tot = vals.reduce((x, y) => x + Math.max(0, y), 0) || 1;
        let a0 = -Math.PI / 2;
        vals.forEach((v, i) => {
          const a1 = a0 + (2 * Math.PI * Math.max(0, v)) / tot, cx = 110, cy = 120, R = 80;
          const large = a1 - a0 > Math.PI ? 1 : 0;
          g += `<path d="M${cx},${cy} L${cx + R * Math.cos(a0)},${cy + R * Math.sin(a0)} A${R},${R} 0 ${large} 1 ${cx + R * Math.cos(a1)},${cy + R * Math.sin(a1)} Z" fill="${pal[i % pal.length]}"/>`;
          g += `<rect x="220" y="${40 + i * 18}" width="10" height="10" fill="${pal[i % pal.length]}"/><text x="236" y="${49 + i * 18}" font-size="11" fill="currentColor">${esc(labels[i])}</text>`;
          a0 = a1;
        });
      } else {
        const n = labels.length || 1, bw = (W - pad * 2) / n;
        series.forEach((sv, si) => {
          if (ch.kind === "line") {
            g += `<polyline fill="none" stroke="${pal[si % pal.length]}" stroke-width="2" points="${sv.map((v, i) => `${pad + bw * (i + 0.5)},${H - pad - ((H - pad * 2) * v) / max}`).join(" ")}"/>`;
          } else {
            const w = (bw * 0.8) / series.length;
            sv.forEach((v, i) => { const hh = ((H - pad * 2) * Math.max(0, v)) / max; g += `<rect x="${pad + bw * i + bw * 0.1 + w * si}" y="${H - pad - hh}" width="${w}" height="${hh}" fill="${pal[si % pal.length]}" rx="2"/>`; });
          }
        });
        labels.forEach((l, i) => (g += `<text x="${pad + bw * (i + 0.5)}" y="${H - 12}" font-size="10" text-anchor="middle" fill="currentColor">${esc(String(l).slice(0, 10))}</text>`));
        g += `<line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="currentColor" opacity=".3"/>`;
        names.forEach((nm, i) => (g += `<rect x="${pad + i * 90}" y="4" width="10" height="10" fill="${pal[i % pal.length]}"/><text x="${pad + 14 + i * 90}" y="13" font-size="11" fill="currentColor">${esc(nm)}</text>`));
      }
      return `<div class="chart"><div class="caps" style="margin-bottom:4px">${esc(ch.title || ch.range)}</div><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="color:var(--ink)">${g}</svg></div>`;
    },
    range() {
      const { c, r, c2, r2 } = this.sel;
      const a = ref(Math.min(c, c2), Math.min(r, r2)), b = ref(Math.max(c, c2), Math.max(r, r2));
      return a === b ? a : `${a}:${b}`;
    },
    paintSel() {
      const { c, r, c2, r2 } = this.sel;
      const [x0, x1, y0, y1] = [Math.min(c, c2), Math.max(c, c2), Math.min(r, r2), Math.max(r, r2)];
      $$("td.cur, td.inrange", this.el).forEach((t) => t.classList.remove("cur", "inrange"));
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const t = this.el.querySelector(`td[data-k="${ref(x, y)}"]`);
        if (t) t.classList.add(x === c && y === r ? "cur" : "inrange");
      }
      const k = ref(c, r), raw = this.sh().cells[k];
      $("#sRef", this.el).value = this.range();
      if (document.activeElement !== $("#sF", this.el)) $("#sF", this.el).value = raw === undefined || raw === null ? "" : String(raw);
    },
    async commit(k, value) {
      const s = this.sh();
      const old = s.cells[k];
      if (String(old ?? "") === String(value)) return;
      MLMI.quiet(this);
      await op("sheets_set_cell", { path: this.file, cell: k, value, sheet: s.name });
      await this.load();
    },
    edit(initial) {
      const k = ref(this.sel.c, this.sel.r);
      const td = this.el.querySelector(`td[data-k="${k}"]`);
      if (!td) return;
      const raw = this.sh().cells[k];
      td.innerHTML = `<input value="${esc(initial !== undefined ? initial : raw ?? "")}">`;
      const inp = td.querySelector("input");
      inp.focus();
      if (initial === undefined) inp.select();
      let done = false;
      const finish = async (save, move) => {
        if (done) return;
        done = true;
        if (save) await this.commit(k, inp.value); else this.render();
        if (move) this.move(move[0], move[1], false);
      };
      inp.onkeydown = (e) => {
        if (e.key === "Enter") { e.preventDefault(); finish(true, [0, 1]); }
        else if (e.key === "Tab") { e.preventDefault(); finish(true, [1, 0]); }
        else if (e.key === "Escape") finish(false);
        e.stopPropagation();
      };
      inp.onblur = () => finish(true);
    },
    move(dc, dr, extend) {
      const s = this.sel;
      if (extend) { s.c2 = Math.max(0, s.c2 + dc); s.r2 = Math.max(0, s.r2 + dr); }
      else { s.c = s.c2 = Math.max(0, s.c + dc); s.r = s.r2 = Math.max(0, s.r + dr); }
      this.paintSel();
      const t = this.el.querySelector(`td[data-k="${ref(s.c2, s.r2)}"]`);
      if (t) t.scrollIntoView({ block: "nearest", inline: "nearest" });
    },
    box() {
      const { c, r, c2, r2 } = this.sel;
      return [Math.min(c, c2), Math.max(c, c2), Math.min(r, r2), Math.max(r, r2)];
    },
    selTsv() {
      const [x0, x1, y0, y1] = this.box(), s = this.sh(), rows = [];
      for (let y = y0; y <= y1; y++) {
        const row = [];
        for (let x = x0; x <= x1; x++) { const raw = s.cells[ref(x, y)]; row.push(raw === undefined || raw === null ? "" : String(raw)); }
        rows.push(row.join("\t"));
      }
      return rows.join("\n");
    },
    async clearSel() {
      const [x0, x1, y0, y1] = this.box();
      const blank = Array.from({ length: y1 - y0 + 1 }, () => Array(x1 - x0 + 1).fill(""));
      MLMI.quiet(this);
      try { await op("sheets_write", { path: this.file, cell: ref(x0, y0), values: blank, sheet: this.sh().name }); } catch (e) { return; }
      this.load();
    },
    async paste(text) {
      const lines = String(text || "").replace(/\r\n?/g, "\n").replace(/\n$/, "").split("\n");
      if (!lines.length || (lines.length === 1 && !lines[0])) return;
      const values = lines.map((l) => l.split("\t"));
      const [x0, , y0] = this.box();
      MLMI.quiet(this);
      try { await op("sheets_write", { path: this.file, cell: ref(x0, y0), values, sheet: this.sh().name }); } catch (e) { return; }
      const w = Math.max(...values.map((r) => r.length));
      this.sel = { c: x0, r: y0, c2: x0 + w - 1, r2: y0 + values.length - 1 };
      await this.load();
    },
    async fmtSel(args) {
      MLMI.quiet(this);
      await op("sheets_format", { path: this.file, range: this.range(), sheet: this.sh().name, ...args });
      this.load();
    },
    wire() {
      const wrap = $("#sWrap", this.el);
      let dragging = false;
      wrap.onmousedown = (e) => {
        const td = e.target.closest("td[data-k]");
        if (!td || e.target.tagName === "INPUT") return;
        const [c, r] = parse(td.dataset.k);
        if (e.shiftKey) { this.sel.c2 = c; this.sel.r2 = r; } else this.sel = { c, r, c2: c, r2: r };
        dragging = true;
        this.paintSel();
        wrap.focus();
      };
      wrap.onmouseover = (e) => {
        if (!dragging) return;
        const td = e.target.closest("td[data-k]");
        if (!td) return;
        const [c, r] = parse(td.dataset.k);
        this.sel.c2 = c; this.sel.r2 = r;
        this.paintSel();
      };
      this._up = () => (dragging = false);
      document.addEventListener("mouseup", this._up);
      wrap.ondblclick = (e) => { if (e.target.closest("td[data-k]")) this.edit(); };
      wrap.tabIndex = 0;
      wrap.onkeydown = (e) => {
        if (e.target.tagName === "INPUT") return;
        const mv = { ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0], Enter: [0, 1], Tab: [1, 0] }[e.key];
        if (mv) { e.preventDefault(); return this.move(mv[0], mv[1], e.shiftKey && e.key.startsWith("Arrow")); }
        if (e.key === "Delete" || e.key === "Backspace") {
          e.preventDefault();
          return this.clearSel();          // one write, one undo (it was one call per cell)
        }
        // copy, cut and paste as tab-separated rows: what Excel and Google Sheets put on the clipboard
        const k = (e.ctrlKey || e.metaKey) && e.key.toLowerCase();
        if (k === "c" || k === "x") {
          e.preventDefault();
          navigator.clipboard.writeText(this.selTsv()).then(() => (k === "x" ? this.clearSel() : toast("Copied")))
            .catch(() => toast("the clipboard is not available here", "bad"));
          return;
        }
        if (k === "v") {
          e.preventDefault();
          navigator.clipboard.readText().then((t) => this.paste(t)).catch(() => toast("the clipboard is not available here", "bad"));
          return;
        }
        if (e.key === "F2") { e.preventDefault(); return this.edit(); }
        if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) { e.preventDefault(); this.edit(e.key); }
      };
      $("#sF", this.el).onkeydown = (e) => {
        if (e.key === "Enter") { e.preventDefault(); this.commit(ref(this.sel.c, this.sel.r), e.target.value).then(() => { this.move(0, 1); wrap.focus(); }); }
        if (e.key === "Escape") { this.paintSel(); wrap.focus(); }
      };
      $("#sRef", this.el).onkeydown = (e) => {
        if (e.key !== "Enter") return;
        const [a, b] = e.target.value.toUpperCase().split(":");
        const p = parse(a), q = parse(b || a);
        if (p && q) { this.sel = { c: p[0], r: p[1], c2: q[0], r2: q[1] }; this.paintSel(); wrap.focus(); }
      };
      $("#sBold", this.el).onclick = () => {
        const f = this.sh().fmt[ref(this.sel.c, this.sel.r)] || {};
        this.fmtSel({ bold: !f.b });
      };
      $("#sFill", this.el).onchange = (e) => this.fmtSel({ fill: e.target.value.slice(1) });
      $("#sNf", this.el).onchange = (e) => { if (e.target.value) this.fmtSel({ number_format: e.target.value }); e.target.value = ""; };
      $$("[data-al]", this.el).forEach((b) => (b.onclick = () => this.fmtSel({ align: b.dataset.al })));
      const sort = async (desc) => {
        if (!this.range().includes(":")) return toast("select the table first, header row included", "bad");
        MLMI.quiet(this);
        await op("sheets_sort", { path: this.file, range: this.range(), column: col(Math.min(this.sel.c, this.sel.c2)), descending: desc, sheet: this.sh().name });
        this.load();
      };
      $("#sSortA", this.el).onclick = () => sort(false);
      $("#sSortD", this.el).onclick = () => sort(true);
      $("#sChart", this.el).onchange = async (e) => {
        const kind = e.target.value;
        e.target.value = "";
        if (!kind) return;
        if (!this.range().includes(":")) return toast("select the data first: labels in the first column, names in the first row", "bad");
        MLMI.quiet(this);
        await op("sheets_add_chart", { path: this.file, range: this.range(), kind, title: "", sheet: this.sh().name });
        this.load();
      };
      $("#sTabs", this.el).onclick = async (e) => {
        if (e.target.id === "sAddSheet") {
          MLMI.quiet(this);
          await op("sheets_add_sheet", { path: this.file, name: `Sheet${this.d.sheets.length + 1}` });
          this.sheet = this.d.sheets.length;
          return this.load();
        }
        const b = e.target.closest("[data-t]");
        if (b) { this.sheet = +b.dataset.t; this.sel = { c: 0, r: 0, c2: 0, r2: 0 }; this.render(); }
      };
      $("#sExport", this.el).onclick = () => MLMI.exportBtn(this.file);
      $("#sCsv", this.el).onclick = () => MLMI.exportBtn(this.file, "csv", { sheet: this.sh().name });
      $("#sPdf", this.el).onclick = () => MLMI.exportBtn(this.file, "pdf");
      $("#sClose", this.el).onclick = () => { S.open.sheets = ""; MLMI.go("sheets"); };
    },
  };
})();
