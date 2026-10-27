// Sheep settings pages: Basics, Models, Accounts, Activity and Advanced (This computer, Checker,
// Custom tools, Saved files, Sent out).
// Each registers on MLMI.S.views as { mount(el, open), update?(state), unmount?() }.
// Every figure comes from the backend (STANDARDS.md section 8): no state only the page knows.
"use strict";

(() => {
  const { $, $$, esc, api, toast, S, go } = MLMI;
  const V = S.views;
  const C = 251.3;                                   // ring circumference, r = 40

  // ------------------------------------------------------------ small helpers
  const when = (t) => {
    if (!t) return "–";
    const d = new Date(t * 1000), now = new Date();
    const hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return d.toDateString() === now.toDateString() ? hm : `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${hm}`;
  };
  const ago = (t) => {
    if (!t) return "never";
    const s = Date.now() / 1000 - t;
    return s < 60 ? "just now" : s < 3600 ? `${Math.round(s / 60)} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`;
  };
  const gb = (b) => (b / 1e9).toFixed(1) + " GB";
  const isLlama = (lic) => /^llama/i.test(lic || "");
  const builtWith = (m) => (isLlama(m && m.licence) ? `<span class="built">Built with <b>Llama</b></span>` : "");

  // ------------------------------------------------------------ components
  function ringHtml(k, fill, num, unit, grow) {
    const target = C * (1 - Math.max(0, Math.min(1, fill || 0)));
    return `<div class="ring"><svg viewBox="0 0 100 100"><circle class="track" cx="50" cy="50" r="40"/>` +
      `<circle class="val" data-k="${k}" data-fill="${fill || 0}" cx="50" cy="50" r="40" style="stroke-dashoffset:${grow ? C : target}"/></svg>` +
      `<div class="num" data-k="${k}-n"><span>${num}<small>${esc(unit || "")}</small></span></div></div>`;
  }
  // rings fill and bars rise: after a full render, on the next frames
  function grow(root) {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $$(".ring .val", root).forEach((c) => (c.style.strokeDashoffset = C * (1 - Math.max(0, Math.min(1, +c.dataset.fill || 0)))));
      $$(".bars", root).forEach((b) => b.classList.add("grown"));
    }));
  }
  // A light morph: when only values changed, patch the keyed leaves in place so
  // rings glide from the old value to the new instead of replaying from zero.
  function patch(root, html) {
    const t = document.createElement("div");
    t.innerHTML = html;
    const nk = $$("[data-k]", t);
    if (!root.firstChild || nk.length !== $$("[data-k]", root).length) {
      root.innerHTML = html;
      return false;
    }
    for (const n of nk) {
      const o = root.querySelector(`[data-k="${CSS.escape(n.dataset.k)}"]`);
      if (!o) { root.innerHTML = html; return false; }
      if (n.dataset.fill !== undefined) {
        o.dataset.fill = n.dataset.fill;
        o.style.strokeDashoffset = C * (1 - Math.max(0, Math.min(1, +n.dataset.fill || 0)));
        continue;
      }
      const cls = n.getAttribute("class") || "";
      if ((o.getAttribute("class") || "") !== cls && !o.classList.contains("sect") && !o.classList.contains("kpi")) o.setAttribute("class", cls);
      if (o.classList.contains("kpi")) {
        ["green", "amber", "red", "none"].forEach((c) => o.classList.toggle(c, n.classList.contains(c)));
      }
      if (!n.querySelector("[data-k]") && o.innerHTML !== n.innerHTML) o.innerHTML = n.innerHTML;
    }
    return true;
  }
  // open by default: the device and web controls were folded away, so people missed them (2026-10-13)
  const openSects = new Set(["plan", "canaries", "egypt", "orch", "quant", "ws", "keys", "look", "about", "mode1", "sent", "current", "found", "hw", "web", "checker"]);
  function sect(id, title, right, body) {
    return `<div class="sect rise ${openSects.has(id) ? "open" : ""}" data-sect="${id}"><div class="sh"><span class="chev">▸</span><h2>${title}</h2>` +
      `<span class="spacer"></span><span data-k="${id}-r">${right || ""}</span></div><div class="x"><div><div class="sb" data-k="${id}-b">${body}</div></div></div></div>`;
  }
  function wireSects(root) {
    root.addEventListener("click", (e) => {
      const h = e.target.closest(".sect > .sh");
      if (!h || e.target.closest("button, input, select, a")) return;
      const s = h.parentElement;
      s.classList.toggle("open");
      if (s.classList.contains("open")) openSects.add(s.dataset.sect); else openSects.delete(s.dataset.sect);
    });
  }
  function miniTabs(el, items, on, onPick, cls = "") {
    el.className = "minitabs " + cls;
    el.innerHTML = items.map(([id, label, off, tip]) => `<button data-t="${id}" class="${id === on ? "on" : ""}${off ? " off" : ""}"` +
      ` title="${esc(off || tip || "")}"${off ? " aria-disabled=\"true\"" : ""}>${esc(label)}</button>`).join("") + `<span class="mark"></span>`;
    const place = () => {
      const b = el.querySelector("button.on"), m = el.querySelector(".mark");
      if (b && m) { m.style.left = b.offsetLeft + "px"; m.style.width = b.offsetWidth + "px"; }
    };
    requestAnimationFrame(place);
    el.onclick = (e) => {
      const b = e.target.closest("button[data-t]");
      if (!b || b.classList.contains("on")) return;
      if (b.classList.contains("off")) return toast(b.title, "bad");
      $$("button", el).forEach((x) => x.classList.toggle("on", x === b));
      place();
      onPick(b.dataset.t);
    };
    return place;
  }
  const pill = (rag, text) => `<span class="pill ${rag}">${esc(text)}</span>`;
  // a status tag in place of a tick or a cross: green when present, red when missing
  const stag = (ok, yes = "Ready", no = "Missing") => `<span class="stag ${ok ? "ok" : "bad"}">${esc(ok ? yes : no)}</span>`;
  const lampWord = { green: "in sync", amber: "working", red: "needs attention", off: "off" };

  // ============================================================ DASHBOARD
  V.dashboard = {
    mount(el) {
      this.el = el;
      el.innerHTML = `<div class="page">
        <div class="phead rise"><h1>Activity</h1><span class="sub" id="dModel"></span><span class="spacer"></span>
          <button type="button" class="chip orch-chip" id="dOrch" aria-label="Orchestrator status; open troubleshooting"><span class="lamp off"></span><span>orchestrator</span></button>
          <button id="dHist" title="Every event for this model">History</button>
          <button id="dRefresh" title="Measure again">Refresh</button></div>
        <div id="dBody" style="display:grid;gap:18px"></div></div>
        <div class="scrim" id="hScrim"></div>
        <aside class="drawer" id="hDrawer"><div class="dh"><b id="hTitle">MODEL HISTORY</b><span style="flex:1"></span>
          <button id="hToggle" style="display:none"></button><button class="iconbtn" id="hClose">✕</button></div><div class="db" id="hBody"></div></aside>`;
      wireSects($("#dBody", el));
      $("#dRefresh", el).onclick = () => this.load(true);
      $("#dHist", el).onclick = () => this.history();
      $("#dOrch", el).onclick = () => MLMI.openSettings("orchestrator");
      $("#hClose", el).onclick = $("#hScrim", el).onclick = () => this.closeHistory();
      $("#dBody", el).addEventListener("click", async (e) => {
        const k = e.target.closest(".kpi");
        if (k) { k.classList.toggle("open"); return; }
        const ca = e.target.closest("[data-check-again]");
        if (ca) {
          try { await api("/api/orchestrator", { action: "recheck", check: ca.dataset.checkAgain }); toast("Checking again in the background; chat stays ready"); }
          catch (err) { toast(err.message, "bad"); }
          setTimeout(() => this.load(false), 600);
          return;
        }
        const b = e.target.closest("button[data-act]");
        if (!b) return;
        try {
          await api("/api/orchestrator", { action: b.dataset.act });
          toast(b.dataset.act === "replan" ? "Planning again for this computer" : "Checking the answers again");
          setTimeout(() => this.load(false), 600);
        } catch (err) { toast(err.message, "bad"); }
      });
      this.load(true);
      this.timer = setInterval(() => this.load(false), 3000);
    },
    unmount() { clearInterval(this.timer); this.el = null; },
    async load(full) {
      let d, o;
      try { [d, o] = await Promise.all([api("/api/dashboard"), api("/api/orchestrator")]); } catch (e) {
        // said, not swallowed: the page polls every 3 s and fills in once Sheep answers again
        const box = this.el && $("#dOrch", this.el);
        if (box) box.innerHTML = `<span class="lamp red"></span><span>Sheep is not answering; retrying…</span>`;
        return;
      }
      if (!this.el) return;
      this.d = d; this.o = o;
      $("#dModel", this.el).textContent = d.model.name ? `Model in use: ${MLMI.modelName(d.model.name)}` : "No model in use yet";
      const orchText = d.on ? d.stage || lampWord[d.lamp] : "orchestrator off";
      $("#dOrch", this.el).innerHTML = `<span class="lamp ${d.on ? d.lamp : "off"}"></span><span>${esc(orchText)}</span>`;
      $("#dOrch", this.el).title = `${orchText}. Open settings for the cause and repair actions.`;
      $("#dOrch", this.el).setAttribute("aria-label", `Orchestrator status: ${orchText}. Open settings for details.`);
      const html = this.sheep(d, o, full);
      const body = $("#dBody", this.el);
      if (full) { body.innerHTML = html; grow(body); }
      else if (!patch(body, html)) grow(body);
      if (full && S.focus) {          // a row of the checks card opened this page at its check
        const sec = $(`[data-sect="${S.focus}"]`, this.el);
        if (sec) { sec.classList.add("open"); openSects.add(S.focus); sec.scrollIntoView({ block: "start" }); }
        S.focus = "";
      }
    },
    kpiCard(k, g, nv) {
      const num = k.value != null ? k.value : k.rag === "green" ? "✓" : "–";
      return `<div class="kpi rise ${k.rag} ${nv ? "nv" : ""}" data-k="kpi-${k.id}" title="${esc(k.tip || "")}"><div class="top">${ringHtml("r-" + k.id, k.fill, num, k.unit, g)}
        <div><div class="label">${esc(k.label)}</div><div class="word" data-k="w-${k.id}">${esc(k.word)}</div>
        <div class="more">Details ▾</div></div></div>
        <div class="x"><div><div class="kvx" data-k="d-${k.id}">${k.detail.map(([a, b]) => `<span class="k">${esc(a)}</span><span>${esc(b)}</span>`).join("")}</div></div></div></div>`;
    },
    sheep(d, o, g) {
      const p = d.plan || {}, f = d.facts || {}, dev = d.device, fit = d.kpis.find((k) => k.id === "fit") || {};
      const gpu = dev.gpu;
      const glyph = { green: "✓", amber: "!", red: "✕", none: "·" }[fit.rag] || "·";
      const big = !d.model.path ? "No model plugged in: choose one on the Models page"
        : fit.rag === "green" ? `${esc(MLMI.modelName(d.model.name))} runs well on this computer` : fit.rag === "red" ? `${esc(MLMI.modelName(d.model.name))} is too big for the free RAM` : "Planning…";
      const layerCount = Number(f.n_layer) || 0, gpuLayers = Number(p.gpu_layers) || 0, midLayers = Number(p.mid_layers) || 0;
      const where = { gpu: "GPU", hybrid: `CPU+GPU · GPU ${gpuLayers} of ${layerCount} layers`, cpu: "CPU",
        amd: `${p.device_name || "AMD GPU"} · ${(dev.amd || {}).backend === "vk" ? "Vulkan" : "DirectML"}`,
        multi: `NVIDIA ${gpuLayers} + ${p.device_name || "AMD"} ${midLayers} + CPU ${Math.max(0, layerCount - gpuLayers - midLayers)} layers` }[p.device] || "";
      // the mode chosen in Settings, and what it resolved to on this machine now (it shifts
      // as the hardware does: charger, heat, free memory, a card added)
      const pref = (d.preferences || {}).device || "auto";
      const MODE = { auto: "Auto", gpu: "NVIDIA GPU", hybrid: "NVIDIA + CPU", multi: "Multi: NVIDIA + other GPU",
                     amd: "Other GPU", cpu: "CPU" };
      const picked = (p.why || []).some((w) => /measured fastest/.test(w));
      const modeLine = !p.device ? "" : pref === "auto"
        ? `Mode: Auto → ${where.split(" · ")[0] || p.device}${picked ? " (measured fastest here)" : ""}`
        : `Mode: ${MODE[pref] || pref} (your choice)` + (p.device !== pref && !(pref === "gpu" && p.device === "hybrid")
          ? ` → running as ${where.split(" · ")[0] || p.device}` : "");
      const sub = p.device ? `${modeLine} · ${where} · ${p.threads} threads · context ${p.ctx} × ${p.slots}` : "";
      const compat = `<div class="compat rise ${fit.rag || "none"}" data-k="compat"><div class="glyph">${glyph}</div>
        <div><div class="big">${big}</div><div class="note" style="margin-top:4px">${esc(sub)}</div>
        <div class="chips" style="margin-top:10px"><span class="chip">${esc(dev.cpu.name.replace(/\s+with Radeon.*$/i, "").replace(/\s+\d+-Core Processor/i, ""))}</span>
          <span class="chip">RAM <b>${gb(dev.ram.total)}</b></span>
          ${gpu ? `<span class="chip">${esc(gpu.name.replace(/^NVIDIA GeForce /, ""))} <b>${Math.round(gpu.vram_total / 2 ** 30)} GB</b></span>` : `<span class="chip">NVIDIA <b>–</b></span>`}
          ${(dev.others || []).length ? dev.others.map((a) => `<span class="chip">${esc(a.name.replace(/\(TM\)/g, ""))} <b>${(a.vram / 2 ** 30).toFixed(0)} GB</b></span>`).join("") : `<span class="chip">AMD / Intel <b>–</b></span>`}
          <span class="chip">${dev.power.plugged === false ? `battery ${dev.power.percent}%` : "AC power"}</span></div></div></div>`;
      const kpis = `<div class="kpis">${d.kpis.map((k) => this.kpiCard(k, g)).join("")}</div>`;
      const lamp = pill(d.on ? d.lamp : "off", d.on ? lampWord[d.lamp] : "off");
      const planBody = p.why ? `<ul class="why">${p.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>
        <div style="display:flex;gap:8px;margin-top:12px"><button data-act="replan">Re-plan now</button></div>`
        : `<div class="note">No plan yet.</div>`;
      const log = (o.log || []).slice().reverse();
      const logBody = `<div class="log">${log.map((e) => `<div class="e ${e.kind}"><span class="t">${when(e.t)}</span><span>${esc(e.what)}${e.why ? `<div class="w">${esc(e.why)}</div>` : ""}</span></div>`).join("") || `<div class="note">Nothing decided yet.</div>`}</div>`;
      return compat + kpis +
        this.checksHtml(d) + this.repairHtml(d) +
        sect("plan", "Why it runs this way", lamp, planBody) +
        sect("log", "Decision log", `<span class="note">${log.length}</span>`, logBody);
    },
    checksHtml(d) {
      const a = d.activity || {}, recs = { startup: a.quick, canaries: a.deep, orientation: a.orientation };
      const names = { startup: "Model ready", canaries: "Answers checked", orientation: "Apps checked" };
      return (d.checks || []).map((c) => {
        const items = ((recs[c.id] || {}).rows) || [];
        const n = items.filter((r) => r.ok).length;
        const body = (items.length ? items.map((r) => this.itemHtml(c.id, r)).join("")
          : `<div class="note">${esc(c.why || "Not run yet for this model.")}</div>`)
          + `<div style="margin-top:10px"><button data-check-again="${esc(c.id)}">Check again</button></div>`;
        return sect("chk-" + c.id, names[c.id] || esc(c.label),
          pill(c.state === "passed" ? "green" : c.state === "failed" ? "red" : "amber",
            c.state === "running" ? `checking · ${n}/${items.length || c.total || "?"}` : items.length && ["passed", "failed"].includes(c.state) ? `${n}/${items.length}`
              : c.state === "waiting for temperature" ? "paused: computer is hot" : /^(queued|waiting)/.test(c.state || "") ? "waiting"
              : c.state === "blocked" ? "no model yet" : c.state === "not started" ? "not run yet" : c.state || ""), body);
      }).join("");
    },
    itemHtml(kind, r) {
      const title = kind === "orientation" ? (r.turns || []).join(" → ") : r.name;
      const steps = (r.steps || []).map(([n, ok, t]) => `<li>${stag(ok, "Done", "Failed")} ${esc(String(n).replace(/_/g, " "))} <span class="dim">${esc(t || "")}</span></li>`).join("");
      const replies = (r.replies || (r.reply ? [r.reply] : [])).map((x) => `<div class="note">${esc(x)}</div>`).join("");
      const files = (r.files || []).map((f) => `<span class="chip">${esc(f)}</span>`).join(" ");
      const secs = r.s != null ? `${r.s} s` : r.ms != null ? `${(r.ms / 1000).toFixed(1)} s` : "";
      return `<details class="chk"><summary><span class="lamp ${r.ok ? "green" : "red"}"></span> <b>${esc(title || r.id || "")}</b>
        <span class="dim"> ${esc(r.ok ? (r.note || "") : (r.why || ""))}${r.tries > 1 ? " · second try" : ""}</span> <span class="dim">${secs}</span></summary>
        ${steps ? `<ul class="why">${steps}</ul>` : ""}${replies}${files ? `<div class="chips">${files}</div>` : ""}</details>`;
    },
    repairHtml(d) {
      const eg = d.egypt, reps = (d.activity || {}).repairs || [];
      if (!eg && !reps.length) return "";
      const now = eg ? `<div class="kvx">${eg.fails ? `<span class="k">went wrong</span><span>${esc(eg.fails.join("; "))}</span>` : ""}
          ${(eg.tried || []).map((t) => `<span class="k">tried</span><span>${esc(t.fix)}: ${t.ok ? "fixed it" : esc(t.why || "not the cause")}</span>`).join("")}
          ${eg.cause ? `<span class="k">fixed by</span><span>${esc(eg.cause)}</span>` : ""}${eg.offer ? `<span class="k">next</span><span>${esc(eg.offer)}</span>` : ""}</div>` : "";
      const past = reps.map((h) => `<div class="e"><span class="t">${when(h.t)}</span><span>${esc(String(h.event).replace(/^egypt: /, ""))}${h.cause ? ` · ${esc(h.cause)}` : ""}${h.fix ? ` · ${esc(h.fix)}` : ""}</span></div>`).join("");
      const rag = !eg || eg.stage === "returned" ? "green" : eg.stage === "stuck" ? "red" : "amber";
      return sect("repair", "Automatic repair", pill(rag, !eg ? "repaired" : eg.stage === "stuck" ? "needs you" : eg.stage === "returned" ? "repaired" : "repairing"), now + (past ? `<div class="log">${past}</div>` : ""));
    },
    // -------------------------------------------------------- the model history (plan G)
    async history(path) {
      $("#hScrim", this.el).classList.add("open");
      $("#hDrawer", this.el).classList.add("open");
      const body = $("#hBody", this.el), tog = $("#hToggle", this.el);
      if (!path) {
        tog.style.display = "none";
        $("#hTitle", this.el).textContent = "MODEL HISTORY";
        const r = await api("/api/registry").catch(() => ({ models: [] }));
        body.innerHTML = r.models.length ? r.models.sort((a, b) => (b.last || 0) - (a.last || 0)).map((m) =>
          `<div class="mrow rise" data-p="${esc(m.path)}"><span class="lamp ${m.path === r.current ? "green" : "off"}"></span>
            <div class="grow"><b>${esc(m.name)}</b><div class="dim">${m.mb ? m.mb + " MB · " : ""}${m.events} events · last ${ago(m.last)}${m.exists ? "" : " · file missing"}</div></div>
            ${m.canary && m.canary.of ? pill(m.canary.passed === m.canary.of ? "green" : "amber", `${m.canary.passed}/${m.canary.of}`) : ""}<span class="dim">›</span></div>`).join("")
          : `<div class="note">No model has run on this machine yet.</div>`;
        body.onclick = (e) => { const r2 = e.target.closest(".mrow"); if (r2) this.history(r2.dataset.p); };
        return;
      }
      const e = await api("/api/registry?path=" + encodeURIComponent(path)).catch(() => null);
      if (!e) return;
      $("#hTitle", this.el).innerHTML = `<span class="iconbtn" style="display:inline-grid;cursor:pointer" id="hBack">‹</span> ${esc(e.display || path.split(/[\\/]/).pop())}`;
      $("#hBack", this.el).onclick = () => this.history();
      let asKpi = false;
      const draw = () => {
        tog.textContent = asKpi ? "Show as log" : "Show as KPIs";
        body.innerHTML = asKpi ? this.histKpis(e) : this.histLog(e);
        grow(body);
      };
      tog.style.display = "";
      tog.onclick = () => { asKpi = !asKpi; draw(); };
      body.onclick = null;
      draw();
    },
    closeHistory() {
      $("#hScrim", this.el).classList.remove("open");
      $("#hDrawer", this.el).classList.remove("open");
    },
    histLog(e) {
      const h = (e.history || []).slice().reverse();
      const detail = (x) => Object.entries(x).filter(([k]) => !["t", "event"].includes(k))
        .map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`).join(" · ");
      const kind = (ev) => ev.startsWith("egypt") ? "egypt" : ev === "guard trip" || ev === "fault" ? "fault" : ev.split(" ")[0];
      return `<div class="log" style="max-height:none">${h.map((x) => `<div class="e ${kind(x.event)}"><span class="t">${when(x.t)}</span><span><b>${esc(x.event)}</b>
        <div class="w">${esc(detail(x))}</div></span></div>`).join("") || `<div class="note">No events yet.</div>`}</div>`;
    },
    histKpis(e) {
      const h = e.history || [];
      const speeds = h.filter((x) => x.tok_s).slice(-24);
      const cans = h.filter((x) => x.event === "canaries").slice(-24);
      const loads = h.filter((x) => x.event === "loaded").length;
      const trips = h.filter((x) => x.event === "guard trip").length;
      const egypt = h.filter((x) => x.event === "egypt: warned").length;
      const back = h.filter((x) => x.event === "egypt: returned").length;
      const avg = speeds.length ? (speeds.reduce((a, x) => a + x.tok_s, 0) / speeds.length).toFixed(1) : "–";
      const maxS = Math.max(1, ...speeds.map((x) => x.tok_s));
      const last = cans[cans.length - 1];
      const g = e.grade || {};
      return `<div class="minikpis">
          <div class="mk rise"><div class="caps">average speed</div><div class="v">${avg}</div><div class="note">tokens/s</div></div>
          <div class="mk rise"><div class="caps">last answer check</div><div class="v ${last && last.passed === last.of ? "hot" : "warn"}">${last ? `${last.passed}/${last.of}` : "–"}</div><div class="note">${last ? ago(last.t) : "not run"}</div></div>
          <div class="mk rise"><div class="caps">loads</div><div class="v">${loads}</div><div class="note">engine starts</div></div>
          <div class="mk rise"><div class="caps">guard trips</div><div class="v ${trips ? "warn" : "hot"}">${trips}</div><div class="note">replies cut off</div></div>
          <div class="mk rise"><div class="caps">Repairs</div><div class="v ${egypt > back ? "bad" : "hot"}">${egypt ? `${back}/${egypt}` : "0"}</div><div class="note">faults fixed</div></div>
        </div>
        <div class="card rise"><h2>Speed over time</h2><div class="bars">${speeds.map((x) => `<i class="${x.tok_s < 3.5 ? "red" : x.tok_s < 7 ? "amber" : ""}" style="height:${Math.round((100 * x.tok_s) / maxS)}%" title="${x.tok_s} tok/s · ${when(x.t)}"></i>`).join("") || ""}</div>
          <div class="note" style="margin-top:6px">${speeds.length ? `${speeds.length} measurements, tokens per second` : "No speed measured yet."}</div></div>
        <div class="card rise"><h2>Answer checks</h2><div class="bars">${cans.map((x) => `<i class="${x.passed === x.of ? "" : x.passed >= x.of * 0.75 ? "amber" : "red"}" style="height:${Math.round((100 * x.passed) / (x.of || 8))}%" title="${x.passed}/${x.of} · ${when(x.t)}"></i>`).join("")}</div></div>
        ${Object.keys(g).length ? `<div class="card rise"><h2>Graded against the original</h2><div class="kvx">${Object.entries(g).map(([a, b]) => `<span class="k">${esc(a)}</span><span>${esc(b)}</span>`).join("")}</div></div>` : ""}`;
    },
  };

  // ============================================================ MODELS
  V.models = {
    async mount(el) {
      this.el = el;
      el.innerHTML = `<div class="page"><div class="phead rise"><h1>Models</h1><span class="sub">What runs on this computer, and adding new ones</span></div>
        <div id="mBody" class="stack"></div><div id="mQuant" class="add-any"></div>
        <div id="mSlots" class="model-slots"></div></div>`;
      wireSects($("#mBody", el));
      quantizer.mount($("#mQuant", el));
      el.addEventListener("click", async (e) => {
        const pub = e.target.closest("[data-publish-model]");
        if (pub) {
          if (!await MLMI.ask("Publish this Sheep file to your Hugging Face account? It starts private; you can make it public on Hugging Face. Only models whose licence allows sharing are published.", { ok: "Publish" })) return;
          pub.disabled = true; pub.textContent = "Publishing…";
          try {
            const r = await api("/api/model_publish", { path: pub.dataset.publishModel });
            toast(`Published (private) at ${r.url}`);
          } catch (err) { toast(err.message, "bad"); }
          pub.disabled = false; pub.textContent = "Publish";
          return;
        }
        const remove = e.target.closest("[data-remove-model]");
        if (remove) {
          if (!await MLMI.ask("Remove this model, and the original files Sheep downloaded for it if no other model uses them? Hugging Face's shared download cache is kept.")) return;
          remove.disabled = true;
          try {
            const result = await api("/api/model_remove", { path: remove.dataset.removeModel });
            toast(result.original_removed ? "Sheep model and unused original removed; shared cache kept." : "Sheep model removed; shared original and Hugging Face cache kept.");
            this.draw();
          } catch (err) { toast(err.message, "bad"); remove.disabled = false; }
          return;
        }
        const slot = e.target.closest("[data-slot-check]");
        if (slot) return MLMI.modelSlotChoice(slot.dataset.slotCheck);
        const feature = e.target.closest("[data-slot-use]");
        if (feature && feature.dataset.slotUse) {
          feature.disabled = true;
          try {
            await api("/api/media_model", { id: feature.dataset.slotUse, action: "use" });
            toast("Feature model selected. Its files stay in Sheep storage.");
            this.draw();
          } catch (err) { toast(err.message, "bad"); feature.disabled = false; }
        }
      });
      el.addEventListener("change", (e) => {
        const select = e.target.closest("[data-slot-model]");
        if (!select) return;
        const button = select.parentElement.querySelector("[data-slot-use]");
        if (button) button.dataset.slotUse = select.value;
      });
      $("#mBody", el).addEventListener("click", async (e) => {
        const b = e.target.closest("button[data-use]");
        if (!b) return;
        b.disabled = true;
        try {
          await api("/api/settings", { model: b.dataset.use });
          toast("Switching: Sheep plans it for this computer and checks its answers");
          setTimeout(() => this.draw(), 800);
        } catch (err) { toast(err.message, "bad"); b.disabled = false; }
      });
      this.draw();
    },
    async draw() {
      let m, st;
      try { [m, st] = await Promise.all([api("/api/models"), api("/api/state")]); } catch (e) {
        if (!this.el) return;
        $("#mBody", this.el).innerHTML = `<div class="card rise">
          <div class="note">Sheep is not answering (${esc(e.message)}). Your models are safe on disk.</div>
          <button class="primary" id="mRetry">Try again</button></div>`;
        $("#mRetry", this.el).onclick = (ev) => { ev.stopPropagation(); this.draw(); };
        return;
      }
      if (!this.el) return;
      const roleFor = { image: "image", video: "video", photo_guidance: "video-reference",
        speech_input: "audio-in", speech_output: "audio-out" };
      const slotRows = (m.slots || []).map((slot) => {
        const role = roleFor[slot.id];
        const choices = role ? (m.media || []).filter((x) => x.role === role) : [];
        const control = role ? `<div class="model-slot-actions"><select data-slot-model="${esc(slot.id)}" aria-label="${esc(slot.label)} model">
            ${choices.map((x) => `<option value="${esc(x.id)}" ${x.id === slot.active ? "selected" : ""}>${esc(x.name)}${x.disabled ? " · disabled" : ""}</option>`).join("")}</select>
            <button data-slot-use="${esc(slot.active || (choices[0] || {}).id || "")}" ${choices.length ? "" : "disabled"}>Use</button>
            <button data-slot-check="${esc(slot.id)}">Check Hugging Face ID…</button></div>`
          : `<button data-slot-check="${esc(slot.id)}">Add from Hugging Face…</button>`;
        return `<div class="card model-slot-row"><div><div class="caps">${esc(slot.label)}</div><b>${esc(slot.active_name ? (slot.id === "chat" || slot.id === "checker" ? MLMI.modelName(slot.active_name) : slot.active_name) : "Not selected")}</b>
          <div class="dim">${esc(slot.description || "")}</div></div>${control}</div>`;
      }).join("");
      $("#mSlots", this.el).innerHTML = `<div class="caps">A model for each job: chat, checking, pictures, video, speech</div>${slotRows}`;
      const cur = st.model, reg = m.grades || {};
      const e = reg[cur.path] || {};
      const g = e.grade || {};
      const duties = e.duties || [];
      const curBody = cur.path ? `<div class="kvx">
          <span class="k">file</span><span>${esc(cur.path)}</span>
          <span class="k">status</span><span data-model-live-status>${pill(st.orch.lamp, st.orch.stage || st.orch.lamp)}</span>
          ${e.canary ? `<span class="k">answers checked</span><span>${e.canary.passed}/${e.canary.of} · ${ago(e.canary.t)}</span>` : ""}
          ${Object.entries(g).map(([a, b]) => `<span class="k">${esc(a)}</span><span>${esc(b)}</span>`).join("")}
          ${e.licence ? `<span class="k">licence</span><span>${esc(e.licence)}</span>` : ""}
          ${duties.map((d) => `<span class="k">duty</span><span>${esc(d)}</span>`).join("")}
        </div>${builtWith(cur) ? `<div style="margin-top:10px">${builtWith(cur)}</div>` : ""}` : `<div class="note">No model plugged in. Choose one below.</div>`;
      const WHERE = { gpu: "graphics card", cpu: "processor", hybrid: "graphics card + processor",
        multi: "two graphics cards + processor", amd: "graphics card", compat: "maker's own code" };
      const size = (mb) => mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`;
      const cards = m.found.map((f) => {
        const inUse = f.path === cur.path, plan = (reg[f.path] || {}).plan || {};
        const where = f.compat ? WHERE.compat : WHERE[plan.device] || "";
        return `<div class="mcard rise ${inUse ? "on" : ""}"><div class="mcard-top"><span class="lamp ${inUse ? st.orch.lamp : "off"}" aria-hidden="true"></span>
            <b title="${esc(f.path)}">${esc(MLMI.modelName(f.name))}</b>${inUse ? pill("green", "in use") : ""}</div>
          <div class="dim">${[f.mb ? size(f.mb) : "", where ? `runs on the ${where}` : "", f.canary != null ? `${f.canary} of 8 answers right` : ""].filter(Boolean).join(" · ")}</div>
          <div class="mcard-acts">${inUse ? "" : `<button data-use="${esc(f.path)}">Use</button>`}
          ${!f.compat ? `<button data-publish-model="${esc(f.path)}" title="Share this model on your Hugging Face account">Publish</button>` : ""}
          ${f.managed ? `<button data-remove-model="${esc(f.path)}" title="Remove this model and the original files only it uses">Remove</button>` : ""}</div></div>`;
      }).join("");
      const yours = m.found.length ? `<div class="mcards">${cards}</div>`
        : `<div class="note">No models yet. Add one below: a Hugging Face ID, a folder or a .gguf file.</div>`;
      $("#mBody", this.el).innerHTML = `<div class="caps">Your models</div>${yours}` + (cur.path ? sect("current", `In use: ${esc(MLMI.modelName(cur.name || ""))}`, "", curBody) : "");
    },
    refreshState(st) {
      const status = this.el && this.el.querySelector("[data-model-live-status]");
      if (!status || !st || !st.model || !st.model.path) return;
      const orch = st.orch || {};
      status.innerHTML = pill(orch.lamp, orch.stage || orch.lamp);
    },
    unmount() { quantizer.unmount(); this.el = null; },
  };

  // What Sheep did for you: the model's setup as steps; the engineer's line sits behind each step
  const storyHtml = (steps) => (steps || []).length
    ? steps.map((x) => `<li><span class="lamp green" aria-hidden="true"></span><span><b>${esc(x.what)}</b>${x.t ? ` <span class="dim">${ago(x.t)}</span>` : ""}
        ${x.tech ? `<details class="tech"><summary>Details</summary><div class="note">${esc(x.tech)}</div></details>` : ""}</span></li>`).join("")
    : `<li class="note">Add a model and its setup shows here, step by step.</li>`;

  // The add-model job as the same steps, live: done green, now amber, to come grey
  const addSteps = (q) => {
    if (!q || q.state === "idle") return "";
    const order = ["fetching", "converting", "fitting", "building", "done"];
    const at = q.state === "failed" ? -1 : order.indexOf(q.state);
    const rows = [["fetching", "Found it and downloaded the files"],
      ...(q.converted || q.state === "converting" ? [["converting", q.converted === "GGUF" ? "Converted it from a GGUF file" : "Converted it into a form Sheep can build from"]] : []),
      ...(q.state === "fitting" || (q.log || []).some((l) => /fit/i.test(l)) ? [["fitting", "Fitted Sheep's engine to this kind of model"]] : []),
      ["building", q.level === "compat" || (q.result && q.result.compat) ? "Set it up with the maker's own code" : "Built it for this computer"],
      ["done", "Checking its answers and app tasks next (the Orchestrator page shows each step)"]];
    return rows.map(([k, w]) => {
      const i = order.indexOf(k), tone = q.state === "failed" ? (i <= 0 ? "red" : "off") : i < at || q.state === "done" ? "green" : i === at ? "amber" : "off";
      return `<li><span class="lamp ${tone}" aria-hidden="true"></span><span>${esc(w)}</span></li>`;
    }).join("");
  };

  // C:\Users\…\workspace: the start and the end of a long path, which are the parts a person knows
  const keyOn = (st, k) => !!(st.cloud && st.cloud.keys && st.cloud.keys[k]);
  const keyLamp = (st, k) => `<i class="lamp ${keyOn(st, k) ? "green" : "off"}" title="${keyOn(st, k) ? "saved" : "not set"}"></i>`;
  const midEllipsis = (s, n) => s.length <= n ? s : s.slice(0, Math.ceil(n * 0.45)) + "…" + s.slice(-Math.floor(n * 0.55));

  MLMI.addSteps = addSteps;          // the add-model dialog shows the same steps

  // ============================================================ SETTINGS (General, Orchestrator, the quantizer)
  const settingsPage = (title, sub, parts) => ({
    mount(el) {
      this.el = el;
      const st = S.state || {};
      const has = (x) => parts.includes(x);
      el.innerHTML = `<div class="${title ? "page" : ""}">${title ? `<div class="phead rise"><h1>${title}</h1><span class="sub">${sub}</span></div>` : ""}
        <div id="sBody" class="stack">
        ${!has("orch") ? "" : sect("orch", "Orchestrator", `<span id="sLampPill"></span>`, `<div class="orchbox">
            <span class="lamp" id="sLamp"></span><span class="state" id="sState"></span></div>
          <div class="note" style="margin-top:8px" title="It plans each model for this computer (where it runs, how much it remembers, how many cores it uses), checks it, watches every reply and repairs a model that misbehaves. It is always on.">Plans, checks and repairs each model for you; always on. <span class="dim">ⓘ</span></div>
          <div style="margin-top:12px"><div class="caps">What Sheep did for you</div><ol class="story" id="sStory"></ol></div>
          <details class="adv" style="margin-top:10px"><summary>What it decided, in detail</summary><ul class="why tech" id="sWhy"></ul></details>`)}
        ${!has("checker") ? "" : sect("checker", "Checker", `<span id="cPill"></span>`, `<div class="fields">
            <div class="note" title="Every model is checked when it is added and at each load: 8 questions, each answer passed through these rules, and through the Checker model if one is plugged in.">Validates every model before you use it. <span class="dim">ⓘ</span></div>
            <div class="kvx" id="cNow"></div>
            <div class="field"><span class="k">use another Checker model</span><span class="inrow"><input type="text" id="cPath" placeholder="a small Sheep model file">
              <button id="cBrowse">Browse…</button><button id="cGo">Examine and use</button></span></div>
            <div class="note" id="cTrial"></div>
            <div class="kvx" id="cRows"></div></div>`)}
        ${!has("hw") ? "" : sect("hw", "Where models run", `<span class="note" id="hSetup"></span>`, `<div class="fields">
            <div class="field"><span class="k">text and code run on</span><span id="hText"></span></div>
            <div class="note" id="hTextWhy"></div>
            <div class="field" style="margin-top:6px"><span class="k">pictures, sound, video</span><span id="hMedia"></span></div>
            <div class="note" id="hMediaWhy"></div>
            <div class="field" style="margin-top:10px"><span class="k">heat guard</span><span class="inrow"><button class="switch" id="hHeat" title="Heat guard on or off"></button><span class="note" id="hHeatNow"></span></span></div>
            <details class="adv"><summary>On: Sheep pauses only at 99 °C, so this computer does not shut down mid-work</summary>
              <div class="note">Sheep watches this computer's temperature and runs everything at full speed. Only when the graphics card reaches 99 °C (or the processor sensor 103 °C) does it pause briefly to let it cool, then carry on. On battery, background checks wait for the charger so the charge goes to your own requests, and when the battery is low new GPU work waits too, so a power cut does not run the laptop flat while Sheep works.</div></details>
            <div style="margin-top:10px"><div class="caps">What Sheep is set up for</div>
              <div class="note">Only a chat model is needed. Everything else is an optional extra: add it when you want that feature.</div>
              <ul id="hSetupList"></ul></div>
            <details class="adv" id="hAdv"><summary>GPU pooling, memory, GPU support, Checker training</summary>
            <div class="note">"How Sheep runs" in Basics sets these for you. Changing one here makes that choice "Custom".</div>
            <div class="note" id="hPoolWhy" hidden></div>
            <div class="field" data-pool="pool" style="margin-top:10px"><span class="k">Pool GPUs</span>
              <span class="inrow"><select id="hPool" aria-label="Pool GPUs">
                <option value="smart">When it is faster (recommended)</option><option value="always">Always, slower GPUs too</option>
                <option value="off">Off: the main GPU and the CPU</option></select>
                <label class="dim" title="How many times slower another graphics card may be than the main one and still share the work">at most
                <input type="number" id="hGap" min="1" max="20" step="0.5" style="width:60px"> x slower</label></span></div>
            <div class="note" data-pool="pool">A model that fits the main GPU always runs there alone. For a larger one, Sheep measures each GPU's speed and uses another GPU's memory only where that beats the CPU and keeps up with the main GPU.</div>
            <div class="field" data-pool="gpus" style="margin-top:8px"><span class="k">GPUs Sheep may use</span><span class="inrow" id="hGpusOn"></span></div>
            <div class="field" data-pool="reserve"><span class="k">VRAM to keep free</span><span class="inrow"><input type="number" id="hReserve" min="0" max="16" step="0.5" style="width:70px"> GB on the main GPU, for games and other programs</span></div>
            <div class="field" data-pool="offload"><span class="k">GPU and CPU together</span><span class="inrow"><button class="switch" id="hOffload" title="Split one model between GPU and CPU when it does not fit the GPU"></button>
              <span class="note">On: a model too large for the GPU runs the part that does not fit on the CPU. Off: it runs on the GPUs alone, or on the CPU alone.</span></span></div>
            <div style="margin-top:10px" id="hCudaRow"><div class="caps">GPU support for pictures and video</div>
              <div id="hGpuRt" class="kvx"></div>
              <div class="note">Each graphics card needs its own picture-and-video runtime: NVIDIA has one, AMD and Intel share another. Without it, pictures and video fall back to a slower path (on a 4 GB NVIDIA card, the shared one cannot make them at all). Each installs once, beside Sheep, and does not change your other programs.</div></div>
            <div class="field" style="margin-top:10px"><span class="k">optional Checker training</span><select id="hCheckerTraining"><option value="auto">Auto · recommended for this device</option><option value="cpu">CPU</option><option value="gpu_bounded">GPU · checkpointed training</option></select></div>
            <div class="note">Auto is the recommended starting point. Checker training is an optional 8-step trial followed by the same full exam; it will not train in the background at startup.</div>
            <div class="kvx" id="hList"></div>
            <div><button id="hRescan" title="Look for graphics cards again and check them">Scan again</button></div>
            </details></div>`)}
        ${!has("quant") ? "" : sect("quant", "Add any model", `<span id="sQPill"></span>`, `<div class="fields">
            <div class="note">A Hugging Face ID, a folder, or a .gguf file. Sheep reads it, converts it if it must, fits it to this computer and checks it. Nothing to set.</div>
            <div class="field"><span class="k">model</span><span class="inrow"><input type="text" id="qSrc" placeholder="a Hugging Face ID, a model folder, or a .gguf file">
              <button id="qBrowse">Folder…</button><button id="qFile">File…</button><button id="qLook">Check</button></span></div>
            <details class="hf-signin" id="qHf">${MLMI.hfSignInHtml()}</details>
            <div class="note" id="qInfo"></div>
            <div class="levels" id="qLevels"></div>
            <div class="field" id="qNameRow" style="display:none"><span class="k">name</span><input type="text" id="qName" placeholder="optional: the folder's name is used (a Llama model's name must begin with Llama)"></div>
            <div style="display:flex;gap:8px;align-items:center"><button class="primary" id="qGo" disabled>Build</button><button class="danger" id="qStop" style="display:none">Stop</button>
              <span class="note" title="Sheep builds from a model's original weights whenever they exist; a GGUF is converted only when there is no original, and keeps that file's quality.">Original weights first; GGUF when there is no original. <span class="dim">ⓘ</span></span></div>
            <div id="qProg" style="display:none;gap:8px"><div class="progress" id="qBar"><i></i></div>
              <ol class="story" id="qSteps"></ol>
              <div class="note" id="qText"></div><details class="tech"><summary>Details</summary><pre class="tail" id="qLog"></pre></details></div></div>`)}
        ${!has("ws") ? "" : sect("ws", "Your files and the agent", "", `<div class="fields">
            <div class="field"><span class="k">workspace folder</span><span class="inrow"><span class="path-mid" id="wShow" title="${esc(st.workspace || "")}">${esc(midEllipsis(st.workspace || "", 56))}</span>
              <button id="wOpen">Open folder</button><button id="wBrowse">Change…</button></span></div>
            <div class="field" id="wTypeRow" hidden><span class="k">type a folder</span><span class="inrow"><input type="text" id="wPath" value="${esc(st.workspace || "")}"><button id="wApply">Apply</button></span></div>
            <div class="field"><span class="k">the agent may</span><span id="wPerm"></span></div></div>`)}
        ${!has("keys") ? "" : sect("keys", "Nemotron on Nebius and web search", "", `<div class="fields">
            <div class="note">Sheep + Nebius uses NVIDIA Nemotron on Nebius to plan, check and research big jobs; Tavily finds web pages for questions about now. Keys are saved on this computer only and never shown again.</div>
            <div class="field"><span class="k">${keyLamp(st, "nebius")}Nebius key</span><span class="inrow"><input type="password" id="kNebius" autocomplete="off" placeholder="${keyOn(st, "nebius") ? "saved · type to replace" : "paste the key"}"></span></div>
            <div class="field"><span class="k">${keyLamp(st, "tavily")}Tavily key</span><span class="inrow"><input type="password" id="kTavily" autocomplete="off" placeholder="${keyOn(st, "tavily") ? "saved · type to replace" : "optional · paste the key"}"></span></div>
            <details class="adv"><summary>Change the Nebius endpoint or model</summary>
            <div class="field"><span class="k">endpoint</span><span class="inrow"><input type="text" id="kBase" value="${esc((st.cloud && st.cloud.keys && st.cloud.keys.nebius_base) || "")}" placeholder="https://api.tokenfactory.nebius.com/v1"></span></div>
            <div class="field"><span class="k">model</span><span class="inrow"><input type="text" id="kModel" value="${esc((st.cloud && st.cloud.keys && st.cloud.keys.nebius_model) || "")}" placeholder="nvidia/nemotron-3-nano-30b-a3b"></span></div></details>
            <div><button class="primary" id="kSave">Save keys</button></div></div>`)}
        ${!has("hf") ? "" : sect("hf", "Hugging Face", "", `<div class="fields"><div class="note">Needed only for gated models (Llama, Gemma, Mistral) and for publishing your own.</div>
            <details class="hf-signin" id="aHf" open>${MLMI.hfSignInHtml()}</details></div>`)}
        ${!has("look") ? "" : sect("look", "Look", "", `<div class="fields"><div class="field"><span class="k">theme</span><span id="lTheme"></span></div></div>`)}
        ${!has("about") ? "" : `<div class="about-line dim">Sheep · local AI on this computer · model <span id="aModel"></span></div>`}
        </div></div>`;
      wireSects($("#sBody", el));
      const bind = (id, fn) => { const x = $(id, el); if (x) x.onclick = fn; };
      if ($("#hPool", el)) {
        $("#hPool", el).value = st.pool_mode || "smart";
        $("#hGap", el).value = st.pool_gap || 4;
        $("#hGap", el).disabled = (st.pool_mode || "smart") !== "smart";
        const savePool = async (body) => {
          try { await api("/api/settings", body); toast("Planning the model again with this choice"); }
          catch (e) { toast(e.message, "bad"); }
          MLMI.refresh();
        };
        $("#hPool", el).onchange = (e) => { $("#hGap", el).disabled = e.target.value !== "smart"; savePool({ pool_mode: e.target.value }); };
        $("#hGap", el).onchange = (e) => savePool({ pool_gap: Number(e.target.value) });
        // each GPU on or off, VRAM kept free, and GPU+CPU splitting: the parts of pooling
        $("#hReserve", el).value = st.vram_reserve_gb || 0;
        $("#hReserve", el).onchange = (e) => savePool({ vram_reserve_gb: Number(e.target.value) });
        $("#hOffload", el).classList.toggle("on", st.cpu_offload !== false);
        $("#hOffload", el).onclick = () => {
          const on = !$("#hOffload", el).classList.contains("on");
          $("#hOffload", el).classList.toggle("on", on);
          savePool({ cpu_offload: on });
        };
        api("/api/gpu_runtime").then((g) => {
          if (!this.el || !$("#hGpusOn", el)) return;
          const off = new Set((st.gpus_off || []).map((n) => n.toLowerCase()));
          $("#hGpusOn", el).innerHTML = (g.devices || []).map((d) => `<label style="margin-right:12px">
            <input type="checkbox" data-gpu="${esc(d.name)}" ${off.has(d.name.toLowerCase()) ? "" : "checked"}> ${esc(d.name)}</label>`).join("")
            || `<span class="dim">no GPU found</span>`;
          $("#hGpusOn", el).onchange = () => savePool({ gpus_off: $$("[data-gpu]", el).filter((c) => !c.checked).map((c) => c.dataset.gpu) });
        }).catch(() => {});
      }
      if ($("#hCudaRow", el)) {
        // NVIDIA support: status, a manual Install, and its progress while it runs
        // one row per GPU runtime this computer needs: Ready, or Missing with Download
        const drawCuda = async () => {
          const g = await api("/api/gpu_runtime").catch(() => null);
          if (!g || !this.el || !$("#hGpuRt", el)) return;
          const rows = (g.runtimes || []).filter((r) => r.needed);
          const running = rows.some((r) => (r.job || {}).state === "running");
          $("#hGpuRt", el).innerHTML = rows.map((r) => {
            const job = r.job || {};
            const busy = job.state === "running";
            const now = r.ready ? `${stag(true)} PyTorch ${esc(String(r.torch).replace(/\+cpu$/, ""))}${r.kind === "directml" ? " with DirectML" : ""}` : busy ? `${esc(job.step)}…`
              : `${stag(false)}${job.state === "failed" ? ` ${esc(job.error)}` : ""}`;
            return `<span class="k">${esc(r.label)}<small class="dim"> · ${esc(r.gpus.join(", "))}</small></span>
              <span class="inrow">${now}${!r.ready && !busy ? ` <button data-rt="${esc(r.kind)}">Download (${esc(r.size)})</button>` : ""}
              ${busy ? `<span class="progress" style="flex:1"><i style="width:${Math.round(100 * (job.progress || 0))}%"></i></span>` : ""}</span>`;
          }).join("") || `<span class="k">No GPU</span><span>pictures and video run on the CPU</span>`;
          if (running) setTimeout(drawCuda, 2000);
          else drawSetup();
        };
        $("#hGpuRt", el).onclick = async (e) => {
          const b = e.target.closest("[data-rt]");
          if (!b) return;
          const row = b.closest(".inrow");
          if (!await MLMI.ask(`Download GPU support (${b.textContent.replace(/^Download /, "")}) for pictures and video? It installs once.`)) return;
          b.disabled = true;
          await api("/api/gpu_runtime", { action: "install", kind: b.dataset.rt }).catch((err) => toast(err.message, "bad"));
          if (row) row.firstChild.textContent = "starting… ";
          drawCuda();
        };
        const drawSetup = async () => {
          const r = await api("/api/setup_check").catch(() => ({ items: [] }));
          if (!this.el || !$("#hSetupList", el)) return;
          // what is missing first (what is needed before what is optional), then what is ready
          const order = (i) => (i.ok ? 2 : i.optional ? 1 : 0);
          const items = r.items.slice().sort((a, b) => order(a) - order(b));
          $("#hSetupList", el).innerHTML = MLMI.setupRows(items.map((i) => ({ ...i, main: !i.optional })));
        };
        this.drawSetup = drawSetup;
        $("#hSetupList", el).onclick = async (e) => {
          const b = e.target.closest("[data-setup]");
          if (!b) return;
          const a = JSON.parse(b.dataset.setup);
          if (await MLMI.setupAction(a, b) && a.do === "runtime") drawCuda();
          drawSetup();
        };
        drawCuda();
      }
      if ($("#hHeat", el)) {
        const heat = (st.orch && st.orch.heat) || {};
        $("#hHeat", el).classList.toggle("on", heat.on !== false);
        $("#hHeatNow", el).textContent = heat.on === false ? `off${heat.why ? ` · ${heat.why}` : ""}`
          : heat.level ? `now ${heat.level}${heat.why ? ` · ${heat.why}` : ""}` : "";
        $("#hHeat", el).onclick = async () => {
          const on = !$("#hHeat", el).classList.contains("on");
          if (!on && !await MLMI.ask("Turn the heat guard off? Sheep will run at full speed however hot this computer gets.")) return;
          $("#hHeat", el).classList.toggle("on", on);
          await api("/api/settings", { thermal_guard: on }).catch((e) => toast(e.message, "bad"));
          MLMI.refresh();
        };
      }
      if ($("#wPerm", el)) miniTabs($("#wPerm", el), [["ask", "Ask first"], ["auto", "Edit freely"], ["plan", "Plan only"]], st.permission || "auto", async (p) => {
        await api("/api/settings", { permission: p }); MLMI.refresh();
      });
      if ($("#lTheme", el)) miniTabs($("#lTheme", el), [["system", "Same as Windows"], ["dark", "Dark"], ["light", "Light"]], st.theme || "dark", async (t) => {
        document.documentElement.dataset.theme = MLMI.themeOf(t);
        await api("/api/settings", { theme: t }); MLMI.refresh();
      });
      const pick = async (id, folder) => {
        if (!window.mlmiDesktop) return toast("type the path: the picker is in the desktop app", "bad");
        const p = folder ? await window.mlmiDesktop.pickFolder() : await window.mlmiDesktop.pickFile();
        if (p) $(id, el).value = p;
      };
      bind("#qBrowse", () => pick("#qSrc", true));
      bind("#qFile", () => pick("#qSrc", false));
      // the desktop app picks and applies in one go; in a browser the folder is typed
      bind("#wBrowse", async () => {
        if (!window.mlmiDesktop) { $("#wTypeRow", el).hidden = false; return; }
        const p = await window.mlmiDesktop.pickFolder();
        if (!p) return;
        $("#wPath", el).value = p;
        $("#wApply", el).click();
      });
      bind("#wOpen", () => window.mlmiDesktop ? window.mlmiDesktop.openFolder(st.workspace || "")
        : toast("Open it from your file manager: " + (st.workspace || ""), "bad"));
      bind("#kSave", async () => {
        const keys = {};
        if ($("#kNebius", el).value.trim()) keys.nebius_key = $("#kNebius", el).value.trim();
        if ($("#kTavily", el).value.trim()) keys.tavily_key = $("#kTavily", el).value.trim();
        keys.nebius_base = $("#kBase", el).value.trim();
        keys.nebius_model = $("#kModel", el).value.trim();
        try {
          await api("/api/settings", { keys });
          $("#kNebius", el).value = ""; $("#kTavily", el).value = "";
          toast("Saved on this computer"); await MLMI.refresh(); this.mount(el);
        } catch (e) { toast(e.message, "bad"); }
      });
      bind("#wApply", async () => {
        try {
          const w = $("#wPath", el).value;
          await api("/api/settings", { workspace: w }); toast("Workspace changed"); MLMI.loadTree();
          st.workspace = w; $("#wShow", el).textContent = midEllipsis(w, 56); $("#wShow", el).title = w;
        } catch (e) { toast(e.message, "bad"); }
      });
      // the checkpoint identified, then one card per level for THIS machine; the
      // orchestrator's pick is pre-selected and the person may choose another
      const look = async () => {
        const src = $("#qSrc", el).value.trim();
        this.srcFor = src; this.src = "";
        this.relook = () => { if (this.el && $("#qLevels", el) && $("#qLevels", el).children.length) look(); };
        if (!src) return toast("enter a model first", "bad");
        $("#qInfo", el).textContent = "Checking…";
        $("#qLevels", el).innerHTML = "";
        $("#qGo", el).disabled = true;
        const r = await api("/api/levels?src=" + encodeURIComponent(src)).catch((e) => ({ error: e.message }));
        if (!this.el) return;
        if (r.error) {
          $("#qInfo", el).innerHTML = `<span class="bad">${esc(r.error)}</span>`;
          if (/gated|sign in/i.test(r.error) && $("#qHf", el)) $("#qHf", el).open = true;
          return;
        }
        if (r.convert) {                              // no original: the GGUF, GPTQ/AWQ copy or Meta .pth is converted, then built
          this.src = src;
          this.level = this.level || "balanced";
          $("#qInfo", el).innerHTML = `<b>${esc(r.gguf)}</b> · ${esc(r.convert)}`;
          const drawG = () => {
            $("#qLevels", el).innerHTML = r.levels.map((c) => `<button class="lvl ${c.id === this.level ? "on" : ""}" data-lv="${c.id}" title="${esc(c.what)}">
              <span class="lname">${esc(c.name)}</span><span class="kv2"><span>${esc(c.what)}</span></span></button>`).join("");
            $("#qGo", el).textContent = this.level === "exact" ? "Convert and build as-is" : "Convert and build";
          };
          drawG();
          $("#qLevels", el).onclick = (e) => { const b = e.target.closest("[data-lv]"); if (b) { this.level = b.dataset.lv; drawG(); } };
          $("#qGo", el).disabled = false;
          $("#qNameRow", el).style.display = "";
          return;
        }
        const i = r.info;
        this.src = i.src || src;                      // the original, when a GGUF/GPTQ/MLX page was given
        $("#qInfo", el).innerHTML = (i.original_of ? `<span class="q-orig">Building from the original <b>${esc(i.src)}</b> · ${esc(i.original_of)} is a copy of it</span><br>` : "") + `A <b>${esc(i.model_type || "new")}</b> model · ${(i.n_weights / 1e9).toFixed(1)} billion weights · licence ${esc(i.licence || "unknown")}` +
          (i.builds ? "" : `<br><span class="bad">${esc(i.note)}</span>`);
        const currentLevel = r.cards.find((c) => c.id === this.level);
        this.level = currentLevel ? currentLevel.id : (r.cards.find((c) => c.recommended) || r.cards[1] || {}).id;
        const draw = () => {
          $("#qLevels", el).innerHTML = r.cards.map((c) => `<button class="lvl ${c.id === this.level ? "on" : ""} ${c.fit}" data-lv="${c.id}" title="${esc(c.what)}">
            <span class="lname">${esc(c.name)}${c.recommended ? `<span class="rec" title="${esc(c.why_recommended || "")}">recommended</span>` : ""}</span>
            <span class="lbits">${esc(c.bits)}</span>
            <span class="kv2"><span>size</span><b>${(c.bytes / 1e9).toFixed(2)} GB</b></span>
            <span class="kv2"><span>quality</span><b>${esc(c.closeness)}</b></span>
            <span class="kv2"><span>speed here</span><b>about ${Math.round(c.est_tok_s / 1.3)} words/s · ${esc({ gpu: "graphics card", hybrid: "card + processor", cpu: "processor", amd: "graphics card" }[c.device] || c.device)}</b></span>
            <span class="kv2"><span>this computer</span><b class="fitw"><i class="lamp ${c.fit}"></i>${esc(c.fit_word)}</b></span>
            <span class="kv2"><span>build</span><b>${c.id === "exact" ? "as published, unchanged" : `about ${c.build_min} min`}</b></span></button>`).join("");
          const selected = r.cards.find((c) => c.id === this.level);
          $("#qGo", el).textContent = selected
            ? (selected.id === "exact" ? "Build as published" : `Build at ${selected.name}`)
            : "Build model";
        };
        draw();
        $("#qLevels", el).onclick = (e) => { const b = e.target.closest("[data-lv]"); if (b) { this.level = b.dataset.lv; draw(); } };
        $("#qGo", el).disabled = !r.cards.length;
        $("#qNameRow", el).style.display = r.cards.length ? "" : "none";
      };
      bind("#qLook", look);
      if ($("#qHf", el)) MLMI.hfSignIn($("#qHf", el), (t) => t && toast(t, "bad"));
      if ($("#aHf", el)) MLMI.hfSignIn($("#aHf", el), (t) => t && toast(t, "bad"));
      if ($("#qSrc", el)) $("#qSrc", el).onkeydown = (e) => { if (e.key === "Enter") look(); };
      bind("#qGo", async () => {
        const typed = $("#qSrc", el).value.trim();
        if (!typed) return toast("enter a model first", "bad");
        const src = this.srcFor === typed && this.src ? this.src : typed;
        try {
          await api("/api/quantize", { src, name: $("#qName", el).value.trim(), level: this.level || "balanced" });
          this.poll();
        } catch (e) { toast(e.message, "bad"); }
      });
      bind("#qStop", () => api("/api/quantize", { stop: true }).then(() => this.poll()));
      bind("#hRescan", async () => { $("#hSetup", el).textContent = "scanning…"; await api("/api/hardware?refresh=1").catch(() => null); setTimeout(() => this.hw(), 1500); });
      if ($("#hText", el)) this.hw();
      if ($("#cNow", el)) this.chk();
      bind("#cBrowse", async () => {
        if (!window.mlmiDesktop) return toast("type the path: the picker is in the desktop app", "bad");
        const p = await window.mlmiDesktop.pickFile({ filters: [{ name: "Sheep models", extensions: ["ssf"] }] });
        if (p) $("#cPath", el).value = p;
      });
      const trainingChoice = async (path, resume = false) => {
        const hw = await api("/api/hardware").catch(() => null);
        if (!hw) return toast("could not read this machine’s training fit", "bad");
        const advice = hw.checker_training || {};
        const thermal = advice.thermal || hw.thermal || {};
        const savedMode = hw.prefs?.checker_training_mode || "auto";
        const mode = savedMode === "auto" ? (advice.recommended === "gpu_bounded" && advice.gpu_available ? "gpu_bounded" : "cpu") : savedMode;
        if (mode === "gpu_bounded" && !advice.gpu_available) {
          return toast("GPU training is no longer available for this setup. Change the saved choice in Settings › Advanced › This computer.", "bad");
        }
        const where = mode === "gpu_bounded" ? "GPU" : "CPU";
        const tempNote = thermal.available ? ` GPU temperature is ${thermal.current_c} °C.` : " GPU temperature sensor is unavailable.";
        try {
          await api("/api/checker", resume
            ? { resume_training: true, training_mode: mode }
            : { path, train: true, training_mode: mode });
          toast(`Short Checker trial started on ${where}: up to 8 steps, then the full held-out exam.${tempNote} Chat and media remain available.`);
          this.chk();
        } catch (err) { toast(err.message, "bad"); }
      };
      this.trainingChoice = trainingChoice;
      bind("#cGo", async () => {
        const path = $("#cPath", el).value.trim();
        if (!path) return toast("Choose a Checker model first", "bad");
        try { await api("/api/checker", { path, train: false }); this.chk(); }
        catch (err) { toast(err.message, "bad"); }
      });
      this.update(st);
      this.poll();
      this.timer = setInterval(() => this.update(S.state), 1500);
    },
    unmount() { clearInterval(this.timer); clearTimeout(this.qt); clearTimeout(this.ht); clearTimeout(this.ct); this.el = null; },
    // the Checker: its rules, the judge model if one passed its exam, and the last exam
    async chk() {
      clearTimeout(this.ct);
      const r = await api("/api/checker").catch(() => null);
      if (!r || !this.el || !$("#cNow", this.el)) return;
      const j = r.judge, t = r.trial || {};
      const busy = ["running", "training", "stopping"].includes(t.state);
      $("#cPill", this.el).innerHTML = pill(busy ? "amber" : "green", busy ? (t.stage ? "training" : "examining") : j ? "model + rules" : "rules");
      $("#cNow", this.el).innerHTML = `<span class="k">rules</span><span>${esc(r.rules.join(" · "))}</span>
        <span class="k">Checker model</span><span>${j ? `${esc(j.name)} · passed its exam ${j.right}/${j.of} <button id="cTrain">Train…</button><button id="cRemove" class="danger" style="margin-left:8px">Back to rules only</button>` : "–"}</span>`;
      const rm = $("#cRemove", this.el);
      if (rm) rm.onclick = async () => {
        if (!await MLMI.ask("Go back to rules only? The Checker model stays on disk; you can choose it again later.")) return;
        try { await api("/api/checker", { remove: true }); } catch (e) { toast(e.message, "bad"); }
        this.chk();
      };
      const train = $("#cTrain", this.el);
      if (train) train.onclick = () => this.trainingChoice(j.path);
      const sc = (x) => x ? `${x.right}/${x.of}` : "–";
      const trainingProgress = t.stage && t.stage.toLowerCase().includes("quantiz") ? t.quantize_progress : t.progress;
      const stepProgress = Number.isFinite(Number(t.steps)) ? ` · step ${Number(t.step || 0)}/${Number(t.steps)}` : "";
      const fitCheck = t.preflight && Number.isFinite(Number(t.preflight.actual_free_vram_gb))
        ? ` · GPU fit ${Number(t.preflight.actual_free_vram_gb).toFixed(1)}/${Number(t.preflight.gpu_need_gb).toFixed(1)} GB after Chat yields` : "";
      const stageNames = { load_seconds: "load", prepare_seconds: "prepare", train_seconds: "train",
        merge_seconds: "merge", quantize_seconds: "convert", baseline_exam_seconds: "baseline exam",
        candidate_exam_seconds: "candidate exam" };
      const timings = Object.entries(t.stage_times || {}).filter(([, seconds]) => Number.isFinite(Number(seconds)))
        .map(([name, seconds]) => `${stageNames[name] || name} ${Math.round(Number(seconds))}s`).join(" · ");
      $("#cTrial", this.el).innerHTML = t.stage && busy
        ? `Sheep is training the Checker: <b>${esc(t.stage)}</b>${stepProgress}${trainingProgress != null ? ` · ${Math.round(100 * trainingProgress)}%` : ""}${t.eta_seconds ? ` · about ${Math.max(1, Math.round(t.eta_seconds / 60))} min remaining at this pace` : ""}${t.temperature_c != null ? ` · GPU ${t.temperature_c} °C` : ""}${fitCheck}${t.before ? ` · before ${sc(t.before)}` : ""}<br><span class="dim">${esc((t.log || []).slice(-1)[0] || "")}</span>${timings ? `<br><span class="dim">Measured stages: ${esc(timings)}</span>` : ""} <button id="cStop" class="danger">Stop safely</button>`
        : ["paused", "stopped"].includes(t.state) ? `<span class="dim">Training ${esc(t.state)} at a safe checkpoint.</span> ${esc(t.why || "")}${timings ? `<br>Measured stages: ${esc(timings)}.` : ""} <button id="cResume">Continue…</button>`
        : t.stage && t.before ? `${t.state === "installed" ? '<span class="good">Installed</span>' : `<span class="bad">${esc(t.state)}</span>`}: before training ${sc(t.before)}, after ${sc(t.after)}${t.trained_better ? " (the trained one is in use)" : t.state === "installed" ? " (the original did better and is in use)" : ""}. ${timings ? `<br>Measured stages: ${esc(timings)}.` : ""} ${esc(t.why || "")}`
        : t.state === "running" ? "Examining the new Checker on 36 known-good and known-bad answers, in each chat frame…"
        : t.state === "installed" ? `<span class="good">Installed:</span> it sorted ${t.right} of ${t.of} exam answers in ${t.seconds} s.`
        : t.state === "rolled back" ? `<span class="bad">Not installed:</span> ${esc(t.why)}. The Checker before it stays.` : "";
      $("#cRows", this.el).innerHTML = (t.rows || []).map((x) => `<span class="k">${stag(x.ok, "Right", "Wrong")} ${esc(x.want)} expected</span><span>${esc(x.q)} → “${esc(x.a)}”: said ${esc(x.got)}</span>`).join("");
      const st = $("#cStop", this.el);
      if (st) st.onclick = async () => {
        try { await api("/api/checker", { stop_training: true }); } catch (e) { toast(e.message, "bad"); }
        this.chk();
      };
      const resume = $("#cResume", this.el);
      if (resume) resume.onclick = () => this.trainingChoice(t.path, true);
      if (busy) this.ct = setTimeout(() => this.chk(), 2000);
    },
    // the pool controls only matter for some choices of where text runs: the rest are greyed, with why
    poolFor(dev) {
      const el = this.el, why = el && $("#hPoolWhy", el);
      if (!why) return;
      const single = dev === "gpu" || dev === "amd";
      const off = { cpu: ["pool", "gpus", "reserve", "offload"], hybrid: ["pool", "gpus", "offload"],
                    multi: ["pool"] }[dev] || (single ? ["pool", "gpus"] : []);
      const say = { cpu: "Text runs on the CPU alone, so the GPU choices below are not used. Pick Auto or a GPU to use them.",
                    hybrid: "One GPU with the CPU: the layers that do not fit go to the CPU, so pooling and the GPU list are not used.",
                    multi: "Both GPUs are pooled for every model, so the pooling rule is not used; you can still leave a GPU out." }[dev]
        || (single ? "One GPU chosen: pooling and the GPU list are not used. VRAM to keep free and GPU and CPU together still apply." : "");
      $$("[data-pool]", el).forEach((f) => {
        const no = off.includes(f.dataset.pool);
        f.classList.toggle("unused", no);
        $$("input, select, button", f).forEach((c) => (c.disabled = no));
      });
      if (!off.includes("pool")) $("#hGap", el).disabled = $("#hPool", el).value !== "smart";
      why.textContent = say; why.hidden = !say;
    },
    // what this machine has, and the choices it allows (unavailable ones greyed, with the reason)
    async hw() {
      clearTimeout(this.ht);
      const r = await api("/api/hardware").catch(() => null);
      if (!r || !this.el || !$("#hText", this.el)) return;
      $("#hSetup", this.el).textContent = r.setup;
      const show = (id, whyId, list, cur, key) => {
        const pick = list.find((m) => m.id === cur) || list[0];
        // short names on the buttons ("RTX 3050"), the full name and the reason in the tooltip
        const short = (s) => String(s).replace(/^NVIDIA (GeForce )?/, "").replace(/ Laptop GPU| \(integrated\)|\(TM\)|\(R\)/g, "")
          .replace(/^AMD Radeon\b.*Graphics$/, "AMD Radeon");
        miniTabs($(id, this.el), list.map((m) => [m.id, short(m.label), m.ok ? "" : m.why, [m.label, m.why].filter(Boolean).join(" · ")]), pick.id, async (v) => {
          try {
            await api("/api/settings", { [key]: v });
            toast(key === "device" ? "the orchestrator is re-planning the model" : "saved");
          } catch (e) { toast(e.message, "bad"); }
          // the quantize cards were judged for the old choice: judge them again for this one
          if (key === "device" && this.relook) this.relook();
          this.hw();
        });
        $(whyId, this.el).textContent = pick.why;
        return pick.id;
      };
      const textPick = show("#hText", "#hTextWhy", r.modes.text, r.prefs.device, "device");
      this.poolFor(textPick);
      // each choice with how it would carry the loaded model: green, amber or red, and the speed
      const fits = r.fits || {};
      $$("#hText button[data-t]", this.el).forEach((b) => {
        const f = fits[b.dataset.t];
        if (!f || b.classList.contains("off")) return;
        b.insertAdjacentHTML("afterbegin", `<i class="lamp ${esc(f.fit)}" aria-hidden="true"></i> `);
        b.title = f.words + (f.est_tok_s ? ` · about ${f.est_tok_s} tokens/s` : "");
      });
      const cur = fits[textPick];
      // the speed is already in the reason line; said twice it read as a stutter (2026-10-21)
      if (cur) $("#hTextWhy", this.el).textContent += ` · this model here: ${cur.words}`;
      show("#hMedia", "#hMediaWhy", r.modes.media, r.prefs.media_device, "media_device");
      const training = $("#hCheckerTraining", this.el);
      if (training) {
        training.value = r.prefs.checker_training_mode || "auto";
        const gpuOption = training.querySelector('option[value="gpu_bounded"]');
        if (gpuOption) gpuOption.disabled = !r.checker_training?.gpu_available;
        training.onchange = async () => {
          try {
            await api("/api/settings", { checker_training_mode: training.value });
            toast("Checker training preference saved. It will be used when you choose Train.");
          } catch (e) { toast(e.message, "bad"); this.hw(); }
        };
      }
      const gb = (b) => b ? `${(b / 2 ** 30).toFixed(b < 2 ** 31 ? 1 : 0)} GB` : "–";
      const nv = r.adapters.filter((a) => a.vendor === "nvidia"), ot = r.adapters.filter((a) => a.vendor !== "nvidia");
      const row = (a) => `${esc(a.name)} · ${gb(a.vram)}${a.integrated ? " shared" : ""}${a.runs.length ? " · runs " + esc(a.runs.join(", ")) : ""}`;
      const thermal = r.thermal || {};
      const fit = r.checker_training || {};
      const trainingFit = `GPU ${fit.gpu_available ? "fits" : "does not fit yet"} · ${fit.gpu_free_gb == null ? "free memory unknown" : `${fit.gpu_free_gb.toFixed(1)} GB free`} ${esc(fit.gpu_fit_basis || "")}; trial needs about ${fit.gpu_need_gb ?? "–"} GB. CPU needs about ${fit.cpu_need_gb ?? "–"} GB available RAM and ${fit.cpu_available ? "fits now" : "does not fit now"}.`;
      const S0 = S.state || {};
      $("#hList", this.el).innerHTML = [
        ["CPU", `AVX2 ${stag(!!r.avx2, "Available", "Missing")}`],
        ["NVIDIA GPU", nv.length ? nv.map(row).join("<br>") : "–"],
        ["AMD / Intel GPU", ot.length ? ot.map(row).join("<br>") : "–"],
        ["DirectML", { ok: "passed its check", probing: "checking…", missing: "–", failed: esc(r.directml.why || "failed") }[r.directml.state] || "–"],
        ["Checker training fit", trainingFit],
        ["GPU temperature", thermal.available
          ? `${thermal.current_c} °C · driver target ${thermal.target_c ?? "unknown"} °C · training pauses at ${thermal.pause_at_c ?? "no limit reported"} °C`
          : esc(thermal.reason || "Temperature sensor unavailable")],
      ].map(([k, v]) => `<span class="k">${k}</span><span>${v}</span>`).join("");
      if (r.directml.state === "probing") this.ht = setTimeout(() => this.hw(), 2500);
    },
    async update(st) {
      if (!st || !this.el || !$("#sLamp", this.el)) { if (st && this.el && $("#aModel", this.el)) $("#aModel", this.el).innerHTML = `${esc(st.model.name ? MLMI.modelName(st.model.name) : "none yet")} ${builtWith(st.model)}`; return; }
      const o = st.orch || {};
      $("#sLamp", this.el).className = "lamp " + (o.lamp || "off");
      $("#sState", this.el).textContent = o.stage && o.stage !== lampWord[o.lamp] ? o.stage : lampWord[o.lamp] || "";
      $("#sLampPill", this.el).innerHTML = pill(o.lamp || "off", lampWord[o.lamp] || "starting");
      if (!this._why || Date.now() - this._why > 5000) {
        this._why = Date.now();
        const r = await api("/api/orchestrator").catch(() => null);
        if (r && this.el) {
          $("#sWhy", this.el).innerHTML = (r.plan && r.plan.why || ["no plan yet"]).map((w) => `<li>${esc(w)}</li>`).join("");
          $("#sStory", this.el).innerHTML = storyHtml(r.story);
        }
      }
    },
    async poll() {
      clearTimeout(this.qt);
      const q = await api("/api/quantize").catch(() => null);
      if (!q || !this.el || !$("#qProg", this.el)) return;
      const run = ["fetching", "converting", "fitting", "building"].includes(q.state);
      $("#qProg", this.el).style.display = "grid";
      $("#qBar", this.el).style.display = q.state === "idle" ? "none" : "";
      $("#qLog", this.el).style.display = q.state === "idle" && !q.log.length ? "none" : "";
      $("#qGo", this.el).disabled = run;
      $("#qStop", this.el).style.display = run ? "" : "none";
      $("#qBar", this.el).classList.toggle("indet", q.state === "fetching" || (run && !q.progress));
      $("#qBar i", this.el).style.width = `${Math.round(100 * (q.state === "done" ? 1 : q.progress))}%`;
      const eta = q.eta_s ? ` · about ${Math.max(1, Math.round(q.eta_s / 60))} min left` : "";
      $("#qText", this.el).innerHTML = q.state === "idle" ? `Enter a model and press <b>Check</b>: Sheep says how it will run here before anything downloads.`
        : q.state === "done" ? `Built <b>${esc(q.result.path)}</b>: ${q.result.mb} MB in ${q.result.minutes} min. ${q.level === "exact" ? "The original weights were stored as-is without quantization." : "The selected quantized level was built."} The orchestrator is preparing it and the Checker is validating it; it is yours to use when the lamp turns green.`
        : q.state === "failed" ? `<span class="bad">Building stopped.</span> See the lines below.`
        : q.state === "fetching" ? "Downloading the original checkpoint…"
          : q.level === "exact" ? "Building as-is: copying the original numeric weights into an exact-weight SSF…"
            : q.state === "converting" ? "Converting it into a form Sheep can build from…"
              : q.state === "fitting" ? "Fitting Sheep's engine to this kind of model…"
                : `Building: part ${q.layer} of ${q.n_layer || "?"}${eta}`;
      $("#qSteps", this.el).innerHTML = addSteps(q);
      $("#qSrc", this.el).value ||= q.src || "";
      $("#qLog", this.el).textContent = q.log.join("\n");
      $("#qLog", this.el).scrollTop = 1e9;
      $("#sQPill", this.el).innerHTML = q.state === "idle" ? pill("none", "ready") : pill(run ? "amber" : q.state === "done" ? "green" : "red", q.state);
      if (run) this.qt = setTimeout(() => this.poll(), 2000);
    },
  });
  const basicsRest = settingsPage("", "", ["ws", "look", "about"]);
  V.accounts = settingsPage("Accounts", "Keys for Nemotron on Nebius, web search and Hugging Face", ["keys", "hf"]);
  V.orchestrator = settingsPage("Orchestrator", "How Sheep plans, checks and repairs each model", ["orch"]);
  V.machine = settingsPage("This computer", "Where models run, heat, and GPU support", ["hw"]);
  const checkerPart = settingsPage("", "", ["checker"]);
  const quantizer = settingsPage("", "", ["quant"]);

  // ============================================================ HERDERS
  V.herders = {
    async mount(el) {
      el.innerHTML = `<div class="page"><div class="phead rise"><h1>Custom tools</h1><span class="sub">Small tools that change a document the same way every time. Ask for "a custom tool that…" and the model writes another.</span></div><div id="hdBody" class="stack"></div></div>`;
      wireSects($("#hdBody", el));
      this.el = el;
      $("#hdBody", el).addEventListener("click", (e) => this.act(e));
      await this.draw();
    },
    async draw() {
      const r = await api("/api/herders").catch(() => ({ herders: [] }));
      if (!this.el) return;
      const EXT = { words: "sdoc", sheets: "ssheet", slides: "sdeck" };
      const KIND = { words: "document", sheets: "spreadsheet", slides: "deck" };
      // each herder runs on a file of its app, and one the AI made can be deleted (asks first)
      const runRow = (h) => {
        const files = S.files.filter((f) => f.path.toLowerCase().endsWith("." + EXT[h.app]));
        return `<div class="inrow" style="margin-top:10px">${files.length
          ? `<select data-hfile="${esc(h.name)}" aria-label="File to run ${esc(h.name)} on">${files.map((f) => `<option value="${esc(f.path)}">${esc(f.path.split(/[\\/]/).pop().replace(/\.[^.]+$/, ""))}</option>`).join("")}</select>
             <button data-hrun="${esc(h.name)}">Run on this file</button>`
          : `<span class="dim">No ${KIND[h.app] || "document"} in your files yet.</span>`}
          ${h.builtin ? "" : `<button class="danger" data-hdel="${esc(h.name)}">Delete</button>`}</div>`;
      };
      const APP = { words: "Documents", sheets: "Spreadsheets", slides: "Slides" };
      // each input gets a label and its hint under it, not instructions as a placeholder
      const inputs = (h) => Object.entries(h.params || {}).map(([k, spec]) => {
        const hint = String(Array.isArray(spec) ? spec[1] : spec), optional = /^optional\b/i.test(hint);
        const name = k.replace(/_/g, " "), label = name.charAt(0).toUpperCase() + name.slice(1);
        return `<label class="field" style="margin-top:8px"><span class="k">${esc(label)}</span><input type="text" data-harg="${esc(k)}" placeholder="${optional ? "optional" : ""}" aria-label="${esc(label)}"></label>
          <div class="note tool-hint">${esc(hint.replace(/^optional:\s*/i, ""))}</div>`;
      }).join("");
      $("#hdBody", this.el).innerHTML = r.herders.map((h) => sect("hd-" + h.name, esc((h.name.charAt(0).toUpperCase() + h.name.slice(1)).replace(/_/g, " ")),
        `${h.builtin ? "" : pill(h.proven ? "green" : "amber", h.proven ? "made by the model" : "made by the model · not run yet")} <span class="chip">${esc(APP[h.app] || h.app)}</span>`,
        `<div class="note" style="margin:6px 0 4px">${esc(h.description || "")}</div>${inputs(h)}${runRow(h)}
         <details class="adv" style="margin-top:4px"><summary>Show code</summary><pre class="code">${esc(h.code || "")}</pre></details>`)).join("") ||
        `<div class="note">No custom tools yet.</div>`;
    },
    async act(e) {
      const run = e.target.closest("[data-hrun]"), del = e.target.closest("[data-hdel]");
      if (run) {
        const sel = $(`[data-hfile="${CSS.escape(run.dataset.hrun)}"]`, this.el);
        const args = Object.fromEntries([...run.closest(".sect").querySelectorAll("[data-harg]")].filter((i) => i.value.trim()).map((i) => [i.dataset.harg, i.value.trim()]));
        try { toast(await MLMI.op("herder_run", { name: run.dataset.hrun, path: sel.value, args })); MLMI.loadTree(); this.draw(); } catch (err) { /* shown */ }
      } else if (del) {
        try { toast(await MLMI.op("herder_delete", { name: del.dataset.hdel })); } catch (err) { return; }
        this.draw();
      }
    },
    unmount() { this.el = null; },
  };

  // ============================================================ LOCAL SAVING
  V.files = {
    async mount(el, open) {
      this.el = el;
      el.innerHTML = `<div class="page"><div class="phead rise"><h1>Saved files</h1><span class="sub" id="fRoot"></span>
          <span class="spacer" style="flex:1"></span><button class="ghost" id="fRefresh" title="Read the folder again">Refresh</button></div>
        <div class="cols2"><div class="card rise"><h2>Files</h2><div class="filelist" id="fList"></div></div>
        <div style="display:grid;gap:18px"><div class="card rise" id="fPrev"><h2>Preview</h2><div class="note">Choose a file.</div></div>
        <div class="card rise"><h2>Every change, kept</h2><div class="log" id="fHist"></div></div></div></div></div>`;
      $("#fList", el).onclick = (e) => { const f = e.target.closest("[data-p]"); if (f) this.show(f.dataset.p); };
      $("#fRefresh", el).onclick = () => this.list();
      $("#fHist", el).onclick = async (e) => {
        const b = e.target.closest("[data-restore]");
        if (!b) return;
        if (!await MLMI.ask(`Put ${b.dataset.path} back as it was before this change? Undo takes the restore back.`)) return;
        try { toast((await api("/api/restore", { t: +b.dataset.restore, path: b.dataset.path })).result); }
        catch (err) { return toast(err.message, "bad"); }
        MLMI.loadTree(); this.list(); this.show(b.dataset.path);
      };
      await this.list();
      if (open) this.show(open);
    },
    async list() {
      const r = await api("/api/files").catch(() => ({ files: [], root: "" }));
      if (!this.el) return;
      $("#fRoot", this.el).textContent = r.root;
      const fs = r.files.filter((f) => !f.path.endsWith("/"));
      $("#fList", this.el).innerHTML = fs.map((f) => `<div class="f" data-p="${esc(f.path)}"><span>${esc(f.path)}</span><span class="dim">${(f.size / 1024).toFixed(1)} KB · ${ago(f.mtime)}</span></div>`).join("")
        || `<div class="note">The workspace is empty.</div>`;
      this.hist(this.cur);
    },
    async hist(path) {
      const h = await api("/api/history" + (path ? "?path=" + encodeURIComponent(path) : "")).catch(() => ({ entries: [] }));
      if (!this.el) return;
      // with one file chosen, each change can put that file back as it was before it
      $("#fHist", this.el).innerHTML = h.entries.slice().reverse().map((e) => `<div class="e"><span class="t">${when(e.t)}</span><span>${esc(e.what || "change")}
        <div class="w">${esc(e.by || "")} · ${e.files.map((f) => esc(f.path)).join(", ")}</div>
        ${path ? `<button class="ghost" data-restore="${e.t}" data-path="${esc(path)}" title="Put this file back as it was before this change">Restore before this</button>` : ""}</span></div>`).join("")
        || `<div class="note">No changes yet.</div>`;
    },
    async show(p) {
      const box = $("#fPrev", this.el);
      this.cur = p;
      const ext = p.split(".").pop().toLowerCase();
      const url = "/api/raw?path=" + encodeURIComponent(p);
      let inner;
      if (/^(png|jpe?g|gif|webp|svg|bmp|ico)$/.test(ext)) inner = `<img src="${url}" style="max-width:100%;border-radius:6px">`;
      else if (/^(mp4|webm|mov|mkv|m4v)$/.test(ext)) inner = `<video src="${url}" controls style="max-width:100%"></video>`;
      else if (/^(mp3|wav|ogg|oga|m4a|flac|aac|opus)$/.test(ext)) inner = `<audio src="${url}" controls></audio>`;
      else if (/^(sdoc|ssheet|sdeck|sstudio|docx|xlsx|pptx)$/.test(ext) || (MLMI.TEXT_EXT || []).includes(ext)) inner = `<button class="primary" id="fOpen">Open in its Sheep app</button>`;
      else if (ext === "pdf") inner = `<iframe src="${url}" style="width:100%;height:420px;border:0;border-radius:6px"></iframe>`;
      else {
        // anything else is shown only when it is text; a binary file is described, not dumped
        const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
        const text = bytes.slice(0, 4096).some((c) => c === 0) ? null : new TextDecoder().decode(bytes.slice(0, 20000));
        inner = text === null ? `<div class="note">${(bytes.length / 1024).toFixed(1)} KB of binary data: no preview for .${esc(ext)} files.</div>`
          : `<pre class="code">${esc(text)}</pre>`;
      }
      box.innerHTML = `<h2>${esc(p)}</h2>${inner}<div class="inrow" style="margin-top:10px">
        <input type="text" id="fName" value="${esc(p)}" aria-label="New name"><button id="fRename">Rename</button>
        <button class="danger" id="fDelete">Delete</button></div>`;
      const b = $("#fOpen", box);
      if (b) b.onclick = () => MLMI.openFile(p);
      // the same tools the agent uses; op() asks the person's yes for the delete
      $("#fRename", box).onclick = async () => {
        const to = $("#fName", box).value.trim();
        if (!to || to === p) return;
        try { await MLMI.op("move_file", { src: p, dst: to }); } catch (e) { return; }
        MLMI.loadTree(); await this.list(); this.show(to);
      };
      $("#fDelete", box).onclick = async () => {
        try { toast(await MLMI.op("delete_file", { path: p })); } catch (e) { return; }
        this.cur = null; box.innerHTML = `<h2>Preview</h2><div class="note">Deleted. Undo (Ctrl+Z) brings it back.</div>`;
        MLMI.loadTree(); this.list();
      };
      this.hist(p);
    },
    unmount() { this.el = null; },
  };

  // ============================================================ SENT OUT: the web (ABIS) and Nebius
  V.cloud = {
    async mount(el) {
      this.el = el;
      el.innerHTML = `<div class="page"><div class="phead rise"><h1>Sent out</h1><span class="sub">Exactly what left this machine, and why</span></div>
        <div id="cBody" style="display:grid;gap:14px"></div></div>`;
      const [r, w] = await Promise.all([api("/api/cloudlog").catch(() => ({ sent: [] })), api("/api/weblog").catch(() => ({ on: false, sent: [], via: "" }))]);
      if (!this.el) return;
      const webLog = w.sent.length ? `<div class="log" style="max-height:none">${w.sent.slice().reverse().map((s) => `<div class="e"><span class="t">${when(s.t)}</span>
          <span><b>${s.kind === "search" ? "searched" : "read a page"}</b> · ${esc(s.via)}<div class="w">${esc(s.what)}</div></span></div>`).join("")}</div>`
        : `<div class="note">Nothing yet. A question about now (news, prices, scores) is searched on the web; each search and page appears here.</div>`;
      const nebLog = r.sent.length ? `<div class="log" style="max-height:none">${r.sent.slice().reverse().map((s) => `<div class="e egypt"><span class="t">${when(s.t)}</span>
          <span><b>${esc(s.why || "step")}</b> · ${esc(s.model)} · ${s.chars} chars · ${s.tools} tools${s.provider ? ` · answered by ${esc(s.provider)}` : ""}${s.gen ? ` · <span class="dim">${esc(s.gen)}</span>` : ""}
          ${(s.preview || []).map(([role, text]) => `<div class="w">${esc(role)}: ${esc(text)}</div>`).join("")}</span></div>`).join("")}</div>`
        : `<div class="note">In SHEEP mode nothing goes to Nebius. In SHEEP + NEBIUS mode, each request is listed here with its reason.</div>`;
      $("#cBody", el).innerHTML = sect("web", "Sent to the web", `<span class="note">${esc(w.via)}</span>`, `<div class="fields">
          <div class="orchbox"><button class="switch ${w.on ? "on" : ""}" id="wSwitch" title="Web access on or off"></button>
            <span class="state">${w.on ? "Web access on: questions about now are searched, with sources" : "Web access off: nothing is searched"}</span></div>${webLog}</div>`)
        + sect("neb", "Sent to Nebius", `<span class="note">${r.sent.length} request${r.sent.length === 1 ? "" : "s"}</span>`, nebLog);
      wireSects($("#cBody", el));
      $("#wSwitch", el).onclick = async () => { await api("/api/settings", { web: !w.on }).catch((e) => toast(e.message, "bad")); this.mount(el); };
    },
    unmount() { this.el = null; },
  };

  // ============================================================ BASICS: the everyday choices, in plain words
  const PRESETS = [
    ["fast", "bolt", "Fastest", "Uses every graphics card that helps. Fans may spin up."],
    ["balanced", "balance", "Balanced", "Fast, and leaves the computer free for other work."],
    ["quiet", "leaf", "Cool and quiet", "Keeps 2 GB of graphics memory free and one card only. Cooler and quieter."],
  ];
  V.basics = {
    mount(el) {
      this.el = el;
      el.innerHTML = `<div class="page"><div class="phead rise"><h1>Basics</h1><span class="sub">How Sheep runs, where answers come from, and your files</span></div>
        <div class="stack">
        <section class="block rise"><h2>How Sheep runs</h2><div class="choice-cards" id="bPreset" role="radiogroup" aria-label="How Sheep runs"></div>
          <div class="note" id="bPresetNote"></div></section>
        <section class="block rise"><h2>Where answers come from</h2><div class="choice-cards two" id="bMode" role="radiogroup" aria-label="Where answers come from"></div>
          <div class="field"><span class="k">search the web</span><span class="inrow"><button class="switch" id="bWeb" title="Web search on or off"></button>
            <span class="note" id="bWebNote"></span></span></div></section>
        <div id="bRest"></div></div></div>`;
      $("#bPreset", el).onclick = async (e) => {
        const b = e.target.closest("[data-preset]");
        if (!b || b.classList.contains("on")) return;
        try { await api("/api/settings", { preset: b.dataset.preset }); toast(`${b.querySelector("b").textContent}: Sheep plans the model again`); }
        catch (err) { toast(err.message, "bad"); }
        await MLMI.refresh();
      };
      $("#bMode", el).onclick = async (e) => {
        const b = e.target.closest("[data-mode]");
        if (!b || b.classList.contains("on")) return;
        const st = S.state || {};
        if (b.dataset.mode === "1" && !(st.cloud && st.cloud.ready)) {
          toast("Paste a Nebius key first: Settings › Accounts", "bad");
          return MLMI.openSettings("accounts");
        }
        try { await api("/api/settings", { mode: Number(b.dataset.mode) }); } catch (err) { toast(err.message, "bad"); }
        await MLMI.refresh();
      };
      $("#bWeb", el).onclick = async () => {
        const on = !$("#bWeb", el).classList.contains("on");
        try { await api("/api/settings", { web: on }); } catch (err) { toast(err.message, "bad"); }
        await MLMI.refresh();
      };
      basicsRest.mount($("#bRest", el));
      this.update(S.state);
    },
    update(st) {
      if (!st || !this.el) return;
      basicsRest.update(st);
      const preset = st.preset || "balanced";
      // redrawn only when a choice changed, so a card under the pointer does not flicker every refresh
      const sig = [preset, st.mode, st.cloud && st.cloud.ready, st.web].join();
      if (sig === this.sig) return;
      this.sig = sig;
      $("#bPreset", this.el).innerHTML = PRESETS.map(([id, ic, name, what]) => `<button class="choice ${id === preset ? "on" : ""}" data-preset="${id}" role="radio" aria-checked="${id === preset}">
          <span class="choice-ic">${MLMI.icon(ic)}</span><b>${name}</b><span class="dim">${what}</span></button>`).join("");
      $("#bPresetNote", this.el).innerHTML = preset === "custom"
        ? `Custom: you changed a setting in Advanced. Choose one above to go back to a plain choice.`
        : `The heat guard stays on in every choice. Advanced holds the details.`;
      const mode = Number(st.mode) === 1 ? "1" : "2", ready = !!(st.cloud && st.cloud.ready);
      $("#bMode", this.el).innerHTML = [["2", "chip", "Only this computer", "Nothing you type leaves this computer."],
        ["1", "route", "This computer + Nemotron on Nebius", ready ? "NVIDIA Nemotron plans, checks and researches the big jobs." : "Needs a Nebius key in Accounts."]]
        .map(([id, ic, name, what]) => `<button class="choice ${id === mode ? "on" : ""}" data-mode="${id}" role="radio" aria-checked="${id === mode}">
          <span class="choice-ic">${MLMI.icon(ic)}</span><b>${name}</b><span class="dim">${what}</span></button>`).join("");
      $("#bWeb", this.el).classList.toggle("on", st.web !== false);
      $("#bWebNote", this.el).textContent = st.web !== false ? "On: questions about now (news, prices, scores) are searched, with sources" : "Off: nothing is searched";
    },
    unmount() { basicsRest.unmount(); this.el = null; this.sig = ""; },
  };

  // A page made of other pages, each in its own fold, mounted the first time it opens.
  // Settings › Activity and Settings › Advanced; S.fold names a fold to open (the old page names).
  const foldPage = (title, sub, parts) => ({
    mount(el) {
      this.el = el; this.kids = {};
      el.innerHTML = `<div class="page"><div class="phead rise"><h1>${title}</h1><span class="sub">${sub}</span></div>
        <div class="stack folds">${parts.map(([id, name, ic, hint, plain]) => plain ? `<div class="fold-plain" data-fold="${id}"></div>`
          : `<div class="sect fold rise" data-fold="${id}"><div class="sh"><span class="chev">▸</span><span class="fold-ic">${MLMI.icon(ic)}</span><h2>${name}</h2>
            <span class="spacer"></span><span class="note">${hint}</span></div><div class="x"><div><div class="sb fold-body"></div></div></div></div>`).join("")}</div></div>`;
      $(".folds", el).addEventListener("click", (e) => {
        const h = e.target.closest(".sect > .sh");
        if (!h || !h.parentElement.classList.contains("fold") || e.target.closest("button, input, select, a")) return;
        this.toggle(h.parentElement.dataset.fold);
      });
      for (const [id, , , , plain] of parts) if (plain) this.open(id);
      if (S.fold) { const f = S.fold; S.fold = ""; this.open(f, true); }
    },
    open(id, scroll) {
      const box = this.el && this.el.querySelector(`[data-fold="${id}"]`);
      if (!box) return null;
      box.classList.add("open");
      if (!this.kids[id]) {
        const v = V[id] || FOLD_VIEWS[id];
        const body = box.querySelector(".fold-body") || box;
        this.kids[id] = v;
        v.mount(body);
        window.MLMI_ICON.paint(body);
      }
      if (scroll) setTimeout(() => box.scrollIntoView({ block: "start", behavior: "smooth" }), 60);
      return this.kids[id];
    },
    toggle(id) {
      const box = this.el.querySelector(`[data-fold="${id}"]`);
      if (box.classList.contains("open")) box.classList.remove("open"); else this.open(id);
    },
    update(st) { for (const v of Object.values(this.kids || {})) if (v.update) v.update(st); },
    refreshState(st) { for (const v of Object.values(this.kids || {})) if (v.refreshState) v.refreshState(st); },
    // Saved files: a file opened from elsewhere is shown in its fold
    show(path) { const v = this.open("files", true); if (v && v.show) v.show(path); },
    unmount() { for (const v of Object.values(this.kids || {})) if (v.unmount) v.unmount(); this.kids = {}; this.el = null; },
  });
  const FOLD_VIEWS = { checker: checkerPart };
  V.activity = foldPage("Activity", "What Sheep did for you, and how the model runs", [
    ["dashboard", "", "", "", true], ["orchestrator", "", "", "", true]]);
  V.advanced = foldPage("Advanced", "Every setting behind Basics. Sheep chooses these for you; change them only if you need to.", [
    ["machine", "This computer", "chip", "where models run, GPUs, heat"],
    ["checker", "Checker", "check", "the model that checks every answer"],
    ["herders", "Custom tools", "tool", "small tools for your documents"],
    ["files", "Saved files", "folder", "every file and every change, kept"],
    ["cloud", "Sent out", "cloud", "exactly what left this computer"]]);

})();
