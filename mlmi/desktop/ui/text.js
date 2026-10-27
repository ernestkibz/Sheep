// Sheep Text & Code: plain text, notes and code files in the workspace. Edit,
// save (Ctrl+S, with undo through the workspace history), preview Markdown, run
// Python or JavaScript in a separate process, and Check: the Checker's rules
// (does the code parse, is the JSON valid) on the file as it stands.
"use strict";

(() => {
  const { $, esc, api, op, toast, S, icon } = MLMI;
  const EXT = ["txt", "md", "markdown", "py", "js", "mjs", "ts", "json", "html", "htm", "css", "csv", "tsv", "yaml", "yml",
    "toml", "ini", "cfg", "log", "xml", "sql", "sh", "ps1", "bat", "c", "h", "cpp", "hpp", "java", "go", "rs", "rb", "php"];
  MLMI.TEXT_EXT = EXT;
  const LANG = { py: "python", js: "javascript", mjs: "javascript", ts: "typescript", json: "json", md: "markdown", markdown: "markdown",
    html: "html", htm: "html", css: "css", sql: "sql", sh: "bash", ps1: "powershell", c: "c", h: "c", cpp: "cpp", hpp: "cpp",
    java: "java", go: "go", rs: "rust", rb: "ruby", php: "php", yaml: "yaml", yml: "yaml", toml: "toml", xml: "xml" };
  const RUNS = { python: "python", javascript: "javascript" };

  // an answer as an unsaved draft in Text & Code (Markdown, named from its first line)
  MLMI.openDraft = (text) => {
    const first = (text.split(/\r?\n/).find((l) => l.trim()) || "draft").replace(/^#+\s*/, "");
    const name = (first.toLowerCase().match(/[a-z0-9]+/g) || ["draft"]).slice(0, 5).join("-") + ".md";
    S.textDraft = { text, name };
    MLMI.openApp("text", "");
  };
  // a reply shaped like a document: headings and length, not a short answer
  MLMI.isDocument = (text) => (text.match(/^#{1,3} /gm) || []).length >= 2 && text.split(/\s+/).length >= 250;

  S.views.text = {
    mount(el, path) {
      this.el = el;
      this.path = path || "";
      this.dirty = false;
      el.innerHTML = `<div class="tx">
        <div class="tx-bar"><input class="tx-path" id="txPath" placeholder="file name, e.g. notes.md or script.py" value="${esc(this.path)}">
          <span class="grow"></span>
          <button class="ghost" id="txPrev" title="Preview Markdown">${icon("doc")}Preview</button>
          <button class="ghost" id="txCheck" title="Check: does it parse? (the Checker's rules)">${icon("check")}Check</button>
          <button class="ghost" id="txRun" title="Run in a separate process">${icon("play")}Run</button>
          <button class="primary" id="txSave" title="Save (Ctrl+S)">${icon("save")}Save</button></div>
        <div class="tx-main"><textarea id="txArea" spellcheck="false" placeholder="Start typing, or ask Sheep to write it."></textarea>
          <div class="tx-prev text" id="txView" hidden></div></div>
        <div class="tx-foot"><span id="txInfo" class="dim"></span></div>
        <pre class="tx-out" id="txOut" hidden></pre></div>`;
      const area = $("#txArea", el);
      area.addEventListener("input", () => { this.dirty = true; this.info(); });
      area.addEventListener("keydown", (e) => {
        if (e.key === "Tab") { e.preventDefault(); document.execCommand("insertText", false, "    "); }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); this.save(); }
      });
      $("#txSave", el).onclick = () => this.save();
      $("#txRun", el).onclick = () => this.run();
      $("#txCheck", el).onclick = () => this.check();
      $("#txPrev", el).onclick = () => this.preview();
      $("#txPath", el).onchange = () => { this.path = $("#txPath", el).value.trim(); this.info(); };
      if (S.textDraft) {
        // an answer opened as an unsaved draft: the text, a suggested name, nothing written yet
        const d = S.textDraft;
        S.textDraft = null;
        area.value = d.text;
        $("#txPath", el).value = this.path = d.name;
        this.dirty = true;
        this.info();
      } else if (this.path) this.load(); else this.info();
    },
    unmount() { this.el = null; },
    path_() { return this.path; },
    lang() { return LANG[(this.path.split(".").pop() || "").toLowerCase()] || "text"; },
    info() {
      if (!this.el) return;
      const t = $("#txArea", this.el).value;
      const lines = t ? t.split("\n").length : 0;
      $("#txInfo", this.el).textContent = `${this.lang()} · ${lines} line${lines === 1 ? "" : "s"} · ${t.length} characters${this.dirty ? " · not saved (Ctrl+S saves it)" : ""}`;
      $("#txRun", this.el).hidden = !RUNS[this.lang()];
      $("#txPrev", this.el).hidden = this.lang() !== "markdown";
      if (this.lang() !== "markdown") { $("#txView", this.el).hidden = true; $("#txArea", this.el).hidden = false; }  // no preview to stay in
    },
    async load() {
      try {
        const r = await fetch("/api/raw?path=" + encodeURIComponent(this.path));
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
        $("#txArea", this.el).value = await r.text();
        this.dirty = false;
      } catch (e) { toast(`could not open ${this.path}: ${e.message}`, "bad"); }
      this.info();
    },
    async reload() { if (!this.dirty) await this.load(); },
    async save() {
      const path = ($("#txPath", this.el).value || "").trim();
      if (!path) return toast("give the file a name first", "bad");
      this.path = path;
      try {
        await op("write_text_file", { path, text: $("#txArea", this.el).value });
        this.dirty = false; S.open.text = path;
        toast(`Saved ${path}`); MLMI.loadTree();
      } catch (e) { toast(e.message, "bad"); }
      this.info();
    },
    async run() {
      const out = $("#txOut", this.el);
      out.hidden = false; out.className = "tx-out"; out.textContent = "running…";
      try {
        const r = await api("/api/run_code", { lang: RUNS[this.lang()], code: $("#txArea", this.el).value });
        out.textContent = (r.output || "(no output)") + `\n— exit ${r.code} · ${r.ms} ms`;
        out.classList.toggle("bad", r.code !== 0);
      } catch (e) { out.textContent = e.message; out.classList.add("bad"); }
    },
    async check() {
      const out = $("#txOut", this.el);
      out.hidden = false; out.className = "tx-out"; out.textContent = "checking…";
      try {
        const r = await api("/api/check_text", { lang: this.lang(), text: $("#txArea", this.el).value });
        out.textContent = r.faults.length ? r.faults.map((f) => "✕ " + f).join("\n") : "✓ No faults: it parses.";
        out.classList.toggle("bad", r.faults.length > 0);
      } catch (e) { out.textContent = e.message; out.classList.add("bad"); }
    },
    preview() {
      const v = $("#txView", this.el), a = $("#txArea", this.el);
      const on = v.hidden;
      v.hidden = !on; a.hidden = on;
      if (on) v.innerHTML = (MLMI.md ? MLMI.md(a.value) : esc(a.value)).replace(/^#{1,3} (.*)$/gm, "<b class=\"h\">$1</b>");
      $("#txPrev", this.el).classList.toggle("on", on);
    },
  };
})();
