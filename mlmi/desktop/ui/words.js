// Sheep Words: the .sdoc editor. Every edit is one of the agent's own tools run
// through /api/op, so a keystroke and a model's call land in the same file and
// share one undo. Text keeps its marks inline: **bold**, *italic*, __underline__.
"use strict";

(() => {
  const { $, $$, esc, api, op, toast, S } = MLMI;

  // ------------------------------------------------------------ shared by the three editors
  // The page shown when an app has no file open: its files, and a new one.
  MLMI.picker = function picker(el, app, { ext, office, noun, create }) {
    const files = S.files.filter((f) => f.path.toLowerCase().endsWith("." + ext) || f.path.toLowerCase().endsWith("." + office));
    el.innerHTML = `<div class="page"><div class="phead rise"><h1>${esc(noun.title)}</h1><span class="sub">${esc(noun.sub)}</span></div>
      <div class="card rise"><h2>New ${esc(noun.one)}</h2><div class="inrow"><input type="text" id="pkName" placeholder="name, e.g. ${esc(noun.example)}">
        <button class="primary" id="pkNew">Create</button></div></div>
      <div class="card rise"><h2>Open</h2><div class="filelist">${files.map((f) => `<div class="f" data-p="${esc(f.path)}" title="${esc(f.path)}"><span>${esc(f.path.replace(new RegExp("\\." + ext + "$", "i"), ""))}</span>
        <span class="dim">${f.path.toLowerCase().endsWith("." + office) ? "imports as ." + ext : ""}</span></div>`).join("") || `<div class="note">No ${esc(noun.many)} yet.</div>`}</div></div></div>`;
    $(".filelist", el).onclick = async (e) => {
      const f = e.target.closest("[data-p]");
      if (!f) return;
      const p = f.dataset.p;
      // a clip has no importer: it becomes a new Studio project with the clip on its timeline
      if (app === "studio" && p.toLowerCase().endsWith(".mp4")) {
        const proj = p.slice(0, -4) + ".sstudio";
        try {
          if (!S.files.some((x) => x.path === proj)) {
            await op("studio_create", { path: proj });
            await op("studio_add_media", { path: proj, file: p });
            await MLMI.loadTree();
          }
          return MLMI.openFile(proj);
        } catch (err) { return; /* op() showed it */ }
      }
      MLMI.openFile(p);
    };
    const make = async () => {
      let name = $("#pkName", el).value.trim();
      if (!name) return toast("give it a name first", "bad");
      const path = name.toLowerCase().endsWith("." + ext) ? name : name + "." + ext;
      try {
        const title = name.replace(/\.[^.]+$/, "");
        await op(create, { path, ...(app === "slides" || app === "words" ? { title } : app === "studio" ? { template: "title-intro", title } : {}) });
        await MLMI.loadTree();
        MLMI.openFile(path);
      } catch (e) { /* op() showed it */ }
    };
    $("#pkNew", el).onclick = make;
    $("#pkName", el).onkeydown = (e) => { if (e.key === "Enter") make(); };
  };

  // Our own writes come back as file_changed events; those must not re-render
  // under the person's cursor. Reloads within this window after our op are ours.
  MLMI.quiet = function quiet(v) { v._quietUntil = Date.now() + 1200; };
  MLMI.isQuiet = (v) => Date.now() < (v._quietUntil || 0);

  MLMI.exportBtn = async function exportBtn(path, fmt, extra = {}) {
    try {
      const r = await op("export_file", { path, ...(fmt ? { format: fmt } : {}), ...extra });
      toast(r);
      MLMI.loadTree();
    } catch (e) { /* shown */ }
  };

  // ------------------------------------------------------------ marks <-> HTML
  const toHtml = (t) => esc(t || "")
    .replace(/\\\*/g, "&#42;").replace(/\\_/g, "&#95;")
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/__(.+?)__/g, "<u>$1</u>").replace(/\*(.+?)\*/g, "<i>$1</i>")
    .replace(/\n/g, "<br>");
  function fromHtml(node) {
    let out = "";
    for (const n of node.childNodes) {
      if (n.nodeType === 3) out += n.textContent.replace(/\\/g, "\\\\").replace(/\*/g, "\\*").replace(/_/g, "\\_");
      else if (n.nodeType === 1) {
        const tag = n.tagName;
        const inner = fromHtml(n);
        if (tag === "BR") out += "\n";
        else if (!inner.trim()) out += inner;
        else if (tag === "B" || tag === "STRONG" || (tag === "SPAN" && /bold|700/.test(n.style.fontWeight))) out += `**${inner}**`;
        else if (tag === "I" || tag === "EM") out += `*${inner}*`;
        else if (tag === "U") out += `__${inner}__`;
        else if (tag === "DIV" || tag === "P") out += (out && !out.endsWith("\n") ? "\n" : "") + inner;
        else out += inner;
      }
    }
    return out;
  }
  const KIND = [["paragraph", 0, "Paragraph"], ["heading", 0, "Title"], ["heading", 1, "Heading 1"], ["heading", 2, "Heading 2"],
    ["heading", 3, "Heading 3"], ["bullet", 0, "Bullet"], ["numbered", 0, "Numbered"], ["quote", 0, "Quote"]];
  const kindOf = (b) => (b.style === "quote" ? "quote:0" : b.type === "heading" ? `heading:${b.level ?? 1}` : `${b.type}:0`);

  S.views.words = {
    mount(el, path) {
      this.el = el;
      this.file = path || "";
      if (!this.file) {
        return MLMI.picker(el, "words", { ext: "sdoc", office: "docx", create: "words_create",
          noun: { title: "Sheep Words", sub: "documents; .docx comes in through import and goes out through export", one: "document", many: "documents", example: "project brief" } });
      }
      el.innerHTML = `<div class="editor">
        <div class="tools"><span class="file">¶ <b>${esc(this.file)}</b></span>
          <select id="wKind" title="Block style">${KIND.map(([t, l, n]) => `<option value="${t}:${l}">${n}</option>`).join("")}</select>
          <button data-cmd="bold" title="Bold (Ctrl+B)"><b>B</b></button><button data-cmd="italic" title="Italic (Ctrl+I)"><i>I</i></button>
          <button data-cmd="underline" title="Underline (Ctrl+U)"><u>U</u></button><span class="sep"></span>
          <button data-align="left" title="Align left">⟸</button><button data-align="center" title="Centre">≡</button><button data-align="right" title="Align right">⟹</button>
          <span class="sep"></span><button id="wTable">+ Table</button><button id="wPic" title="Insert a picture">+ Picture</button><button id="wBreak">+ Page break</button><button id="wDel" title="Delete this block">Delete block</button>
          <span class="wtab" id="wTabTools" hidden><span class="sep"></span><button data-tbl="row+" title="Add a row below">+ Row</button><button data-tbl="col+" title="Add a column to the right">+ Column</button>
            <button data-tbl="row-" title="Remove this row">− Row</button><button data-tbl="col-" title="Remove this column">− Column</button></span>
          <span class="spacer" style="flex:1"></span><button id="wFind">Find & replace</button><button id="wMd" title="Save as Markdown">.md</button><button id="wPdf" title="Save as PDF">PDF</button><button id="wExport" class="primary">Export .docx</button><button id="wClose" title="All documents">Files</button></div>
        <div class="fbar" id="wFindBar" style="display:none"><span class="caps">Find</span><input type="text" id="wF1" aria-label="Find">
          <span class="caps">Replace with</span><input type="text" id="wF2" aria-label="Replace with"><button id="wF3" class="primary">Replace all</button></div>
        <div class="desk"><div class="pagesheet" id="wPage"></div></div></div>`;
      this.page = $("#wPage", el);
      this.cur = -1;
      this.wire();
      this.load();
    },
    path() { return this.file; },
    unmount() { this.el = null; },
    async load(keepFocus) {
      try { this.d = await api("/api/doc?path=" + encodeURIComponent(this.file)); } catch (e) { toast(e.message, "bad"); return; }
      if (!this.el) return;
      this.render(keepFocus);
    },
    reload(soft) {
      if (MLMI.isQuiet(this)) return;
      this.load(true).then(() => $$(".blk", this.page).forEach((b) => b.classList.add("agent-touched")));
    },
    render(keepFocus) {
      const blocks = this.d.blocks || [];
      let n = 0;
      this.page.innerHTML = blocks.map((b, i) => {
        n = b.type === "numbered" ? n + 1 : 0;
        const st = `${b.align ? `text-align:${b.align};` : ""}${b.size ? `font-size:${b.size}px;` : ""}${b.color ? `color:#${b.color};` : ""}`;
        if (b.type === "table") {
          return `<div class="blk" data-i="${i}"><table>${(b.rows || []).map((r, ri) => `<tr>${r.map((c, ci) =>
            `<td contenteditable="true" data-r="${ri}" data-c="${ci}">${toHtml(c)}</td>`).join("")}</tr>`).join("")}</table></div>`;
        }
        // focusable, so a click selects them and Delete block (or the Delete key) removes them
        if (b.type === "image") return `<div class="blk pick" tabindex="0" data-i="${i}" title="Picture: Delete removes it"><img src="/api/raw?path=${encodeURIComponent(b.src)}" style="width:${Math.min(100, (b.width || 5) * 16)}%"></div>`;
        if (b.type === "pagebreak") return `<div class="blk pagebreak pick" tabindex="0" data-i="${i}" title="Page break: Delete removes it"></div>`;
        const cls = b.style === "quote" ? "quote" : b.type === "heading" ? `h${b.level ?? 1}` : b.type === "bullet" || b.type === "numbered" ? `${b.type} lv${b.level || 0}` : "";
        return `<div class="blk ${cls}" data-i="${i}" ${b.type === "numbered" ? `data-n="${n}"` : ""} contenteditable="true" style="${st}">${toHtml(b.text)}</div>`;
      }).join("") || "";
      if (!blocks.length) this.page.innerHTML = `<div class="blk" data-i="0" data-new="1" contenteditable="true"></div>`;
      // Until a text block receives focus, there is no selection and the
      // toolbar's native first option (Paragraph) can mislabel a new document
      // whose first block is its Title. Reflect the visible current/first text
      // block so the style control is accurate immediately on open.
      const styleBlock = blocks[this.cur] || blocks[0];
      if (styleBlock && !["table", "image", "pagebreak"].includes(styleBlock.type)) {
        $("#wKind", this.el).value = kindOf(styleBlock);
      }
      if (keepFocus && this.cur >= 0) {
        const b = this.page.querySelector(`.blk[data-i="${this.cur}"][contenteditable]`);
        if (b) this.caretEnd(b);
      }
    },
    caretEnd(b) {
      b.focus();
      const r = document.createRange();
      r.selectNodeContents(b);
      r.collapse(false);
      const s = getSelection();
      s.removeAllRanges();
      s.addRange(r);
    },
    async put(i, block, insert) {
      MLMI.quiet(this);
      await op("words_put", { path: this.file, index: i, block, ...(insert ? { insert: true } : {}) });
      MLMI.quiet(this);
    },
    // save one text block if its text changed
    async save(el) {
      const i = +el.dataset.i;
      const text = fromHtml(el).replace(/\n+$/, "");
      if (el.dataset.new) {
        if (!text.trim()) return;
        delete el.dataset.new;
        this.d.blocks.push({ type: "paragraph", text });
        try { return await this.put(this.d.blocks.length - 1, { type: "paragraph", text }); }
        catch (e) { el.dataset.new = "1"; this.d.blocks.pop(); throw e; }
      }
      const b = this.d.blocks[i];
      if (!b || b.text === text) return;
      const previous = b.text;
      b.text = text;
      try { await this.put(i, b); }
      catch (e) { b.text = previous; throw e; }
    },
    queueSave(save) {
      const prior = (this._pendingSave || Promise.resolve()).catch(() => {});
      const pending = prior.then(save);
      this._pendingSave = pending;
      pending.finally(() => { if (this._pendingSave === pending) this._pendingSave = null; }).catch(() => {});
      pending.catch(() => {}); // focusout cannot await; the operation reports its own error.
      return pending;
    },
    async saveCell(td) {
      const i = +td.closest(".blk").dataset.i, r = +td.dataset.r, c = +td.dataset.c;
      const text = fromHtml(td), b = this.d.blocks[i];
      if (!b || b.rows[r][c] === text) return;
      const previous = b.rows[r][c];
      b.rows[r][c] = text;
      MLMI.quiet(this);
      try { await op("words_set_cell", { path: this.file, index: i, row: r, col: c, text }); }
      catch (e) { b.rows[r][c] = previous; throw e; }
    },
    async flushCurrent() {
      if (this._pendingSave) await this._pendingSave;
      const active = document.activeElement;
      if (active && this.page.contains(active) && active.matches("td[data-r]")) {
        await this.saveCell(active);
        return;
      }
      if (this.cur < 0) return;
      const el = this.page.querySelector(`.blk[data-i="${this.cur}"][contenteditable="true"]`);
      if (el) await this.save(el);
    },
    wire() {
      const page = this.page;
      page.addEventListener("focusin", (e) => {
        const b = e.target.closest(".blk");
        if (!b) return;
        this.cur = +b.dataset.i;
        $$(".blk.sel", page).forEach((x) => x.classList.remove("sel"));
        const blk = this.d && this.d.blocks[this.cur];
        if (blk && !["table", "image", "pagebreak"].includes(blk.type)) $("#wKind", this.el).value = kindOf(blk);
        $("#wTabTools", this.el).hidden = !(blk && blk.type === "table");
      });
      // a selected picture or page break: Delete / Backspace removes it
      page.addEventListener("keydown", async (e) => {
        const pick = e.target.closest(".blk.pick");
        if (!pick || (e.key !== "Delete" && e.key !== "Backspace")) return;
        e.preventDefault();
        MLMI.quiet(this);
        try { await op("words_delete", { path: this.file, index: +pick.dataset.i }); } catch (err) { return; }
        this.cur = Math.max(0, +pick.dataset.i - 1);
        this.load(true);
      });
      page.addEventListener("focusout", (e) => {
        const td = e.target.closest("td[data-r]");
        if (td) {
          this.queueSave(() => this.saveCell(td));
          return;
        }
        const b = e.target.closest(".blk[contenteditable]");
        if (b) this.queueSave(() => this.save(b));
      });
      page.addEventListener("paste", (e) => {
        e.preventDefault();
        document.execCommand("insertText", false, e.clipboardData.getData("text/plain"));
      });
      page.addEventListener("keydown", async (e) => {
        const el = e.target.closest(".blk[contenteditable]");
        if (!el) return;
        const i = +el.dataset.i;
        const b = this.d.blocks[i];
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          // split at the caret: what follows it becomes the next block
          const sel = getSelection();
          const r = sel.getRangeAt(0);
          const tail = document.createRange();
          tail.setStart(r.endContainer, r.endOffset);
          tail.setEnd(el, el.childNodes.length);
          const box = document.createElement("div");
          box.appendChild(tail.extractContents());
          const after = fromHtml(box);
          await this.save(el);
          const same = b && (b.type === "bullet" || b.type === "numbered") ? { type: b.type, level: b.level || 0 } : { type: "paragraph" };
          if (b && same.type !== "paragraph" && !fromHtml(el).trim() && !after.trim()) {
            // Enter on an empty list item ends the list
            b.type = "paragraph"; delete b.level;
            await this.put(i, b);
            this.cur = i;
            return this.load(true);
          }
          const nb = { ...same, text: after };
          this.d.blocks.splice(i + 1, 0, nb);
          await this.put(i + 1, nb, true);
          this.cur = i + 1;
          await this.load();
          const nx = this.page.querySelector(`.blk[data-i="${i + 1}"]`);
          if (nx) { nx.focus(); const rr = document.createRange(); rr.setStart(nx, 0); rr.collapse(true); getSelection().removeAllRanges(); getSelection().addRange(rr); }
        } else if (e.key === "Backspace" && b && !el.textContent.length && this.d.blocks.length > 1) {
          e.preventDefault();
          MLMI.quiet(this);
          await op("words_delete", { path: this.file, index: i });
          this.cur = Math.max(0, i - 1);
          await this.load(true);
        } else if ((e.ctrlKey || e.metaKey) && ["b", "i", "u"].includes(e.key.toLowerCase())) {
          e.preventDefault();
          document.execCommand({ b: "bold", i: "italic", u: "underline" }[e.key.toLowerCase()]);
        }
      });
      $("#wKind", this.el).onchange = async (e) => {
        const [t, l] = e.target.value.split(":");
        const i = this.cur;
        if (i < 0) return;
        await this.flushCurrent();
        const b = this.d.blocks[i];
        if (!b || ["table", "image", "pagebreak"].includes(b.type)) return;
        delete b.style;
        if (t === "quote") { b.type = "paragraph"; b.style = "quote"; } else { b.type = t; b.level = +l; }
        if (t === "paragraph") delete b.level;
        await this.put(i, b);
        this.load(true);
      };
      $$("[data-cmd]", this.el).forEach((btn) => (btn.onmousedown = (e) => { e.preventDefault(); document.execCommand(btn.dataset.cmd); }));
      $$("[data-align]", this.el).forEach((btn) => (btn.onmousedown = async (e) => {
        e.preventDefault();
        if (this.cur < 0) return;
        await this.flushCurrent();
        MLMI.quiet(this);
        await op("words_format", { path: this.file, index: this.cur, align: btn.dataset.align });
        this.load(true);
      }));
      $("#wTable", this.el).onclick = async () => {
        await this.flushCurrent();
        MLMI.quiet(this);
        await op("words_add_table", { path: this.file, rows: [["Item", "Detail"], ["", ""], ["", ""]], ...(this.cur >= 0 ? { after: this.cur } : {}) });
        this.load();
      };
      $("#wPic", this.el).onclick = async () => {
        if (!window.mlmiDesktop) return toast("Insert a picture from the desktop app, or ask Sheep to add one", "bad");
        const src = await window.mlmiDesktop.pickFile({ filters: [{ name: "Pictures", extensions: ["png", "jpg", "jpeg"] }] });
        if (!src) return;
        await this.flushCurrent();
        try {
          const { path } = await api("/api/import_media", { path: src });          // copied into media/
          MLMI.quiet(this);
          await op("words_add_image", { path: this.file, image: path, ...(this.cur >= 0 ? { after: this.cur } : {}) });
        } catch (e) { return toast(e.message, "bad"); }
        this.load();
      };
      // rows and columns of the table the caret is in, as one block replace (one undo)
      $$("[data-tbl]", this.el).forEach((btn) => (btn.onmousedown = async (e) => {
        e.preventDefault();
        const td = document.activeElement && document.activeElement.closest && document.activeElement.closest("td[data-r]");
        const i = this.cur, b = this.d.blocks[i];
        if (!b || b.type !== "table") return;
        await this.flushCurrent();
        const r = td ? +td.dataset.r : b.rows.length - 1, c = td ? +td.dataset.c : (b.rows[0] || []).length - 1;
        const rows = b.rows.map((x) => x.slice()), w = Math.max(1, ...rows.map((x) => x.length));
        const act = btn.dataset.tbl;
        if (act === "row+") rows.splice(r + 1, 0, Array(w).fill(""));
        if (act === "col+") rows.forEach((x) => x.splice(c + 1, 0, ""));
        if (act === "row-") { if (rows.length < 2) return toast("a table keeps at least one row", "bad"); rows.splice(r, 1); }
        if (act === "col-") { if (w < 2) return toast("a table keeps at least one column", "bad"); rows.forEach((x) => x.splice(c, 1)); }
        b.rows = rows;
        try { await this.put(i, b); } catch (err) { return; }
        this.load();
      }));
      $("#wBreak", this.el).onclick = async () => {
        await this.flushCurrent();
        MLMI.quiet(this);
        await this.put(this.cur >= 0 ? this.cur + 1 : this.d.blocks.length, { type: "pagebreak" }, true);
        this.load();
      };
      $("#wDel", this.el).onclick = async () => {
        if (this.cur < 0 || !this.d.blocks.length) return;
        await this.flushCurrent();
        MLMI.quiet(this);
        await op("words_delete", { path: this.file, index: this.cur });
        this.cur = Math.max(0, this.cur - 1);
        this.load(true);
      };
      $("#wFind", this.el).onclick = () => {
        const bar = $("#wFindBar", this.el);
        bar.style.display = bar.style.display === "none" ? "flex" : "none";
        if (bar.style.display === "flex") $("#wF1", this.el).focus();
      };
      $("#wF3", this.el).onclick = async () => {
        const find = $("#wF1", this.el).value;
        if (!find) return toast("type what to find", "bad");
        await this.flushCurrent();
        MLMI.quiet(this);
        toast(await op("words_replace", { path: this.file, find, replace: $("#wF2", this.el).value }));
        this.load();
      };
      $("#wExport", this.el).onclick = async () => { await this.flushCurrent(); await MLMI.exportBtn(this.file); };
      $("#wPdf", this.el).onclick = async () => { await this.flushCurrent(); await MLMI.exportBtn(this.file, "pdf"); };
      $("#wMd", this.el).onclick = async () => { await this.flushCurrent(); await MLMI.exportBtn(this.file, "md"); };
      $("#wClose", this.el).onclick = () => { S.open.words = ""; MLMI.go("words"); };
    },
  };
})();
