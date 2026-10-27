// Sheep interface core: the chat, the canvas the apps open in, chat history,
// the model picker and the settings sheet. Apps and settings pages register on
// MLMI.views; this file only knows which one is open and where.
"use strict";

const MLMI = (() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const icon = (n) => window.MLMI_ICON.icon(n);
  // "Qwen--Qwen2.5-3B-Instruct.ssf" reads "Qwen2.5 3B Instruct"
  const prettyModel = (n) => String(n || "").replace(/\.(ssf|compat\.json)$/i, "").replace(/^[^-]+--/, "").replace(/[-_]+/g, " ").trim();

  const APPS = ["words", "sheets", "slides", "studio", "text", "work"];
  // Settings: plain choices first (Basics), the engineer's controls folded into Advanced.
  // The old page names still open the page and the fold that now holds them.
  const SET_TABS = ["basics", "models", "accounts", "activity", "advanced"];
  const SET_ROUTE = { general: ["basics"], keys: ["accounts"], dashboard: ["activity"], orchestrator: ["activity", "orchestrator"],
    machine: ["advanced", "machine"], checker: ["advanced", "checker"], herders: ["advanced", "herders"],
    files: ["advanced", "files"], cloud: ["advanced", "cloud"] };
  const SETTINGS = [...SET_TABS, ...Object.keys(SET_ROUTE)];
  const APP_OF = { sdoc: "words", ssheet: "sheets", sdeck: "slides", sstudio: "studio", docx: "words", xlsx: "sheets", pptx: "slides" };
  // plain text, notes and code open in Sheep Text & Code (csv stays a sheet when imported)
  for (const x of ["txt", "md", "markdown", "py", "js", "mjs", "ts", "json", "html", "htm", "css", "csv", "tsv", "yaml", "yml", "toml", "ini", "cfg", "log", "xml", "sql", "sh", "ps1", "bat", "c", "h", "cpp", "hpp", "java", "go", "rs", "rb", "php"]) APP_OF[x] = APP_OF[x] || "text";
  const NATIVE = { words: "sdoc", sheets: "ssheet", slides: "sdeck", studio: "sstudio" };
  const OFFICE = { docx: "sdoc", xlsx: "ssheet", pptx: "sdeck" };
  const APP_NAME = { words: "Sheep Words", sheets: "Sheep Sheets", slides: "Sheep Slides", studio: "Sheep Studio", text: "Sheep Text", work: "Work Canvas" };

  const S = {
    state: null, views: {}, open: {}, files: [], busy: false,
    app: "", mounted: null, tab: "", setMounted: null,
    chatId: null, chatVersion: 0, convo: [], log: [], canvas: { order: [], pinned: {} },
    activeJob: null, lastJob: null,
  };

  async function api(path, body) {
    const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(path, opt);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) {
      if (r.status === 404 && path === "/api/orient/retry") {
        throw new Error("The running Sheep service does not recognize this retry yet. Restart Sheep to update its service, then try again.");
      }
      throw Object.assign(new Error(j.error || `${r.status}`), { confirm: j.confirm });
    }
    return j;
  }
  async function op(tool, args, confirmed = false) {
    try { return (await api("/api/op", { tool, args, confirmed })).result; }
    catch (e) {
      // a delete, send or overwrite: the same yes the agent asks for
      if (e.confirm && !confirmed && await ask(`Sheep needs your yes:\n\n${e.confirm}`)) return op(tool, args, true);
      toast(e.confirm ? "Cancelled." : e.message, e.confirm ? "" : "bad"); throw e;
    }
  }
  function toast(text, cls = "") {
    const t = document.createElement("div");
    t.className = "toast" + (cls === "bad" ? " bad" : "");
    t.textContent = text;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), cls === "bad" ? 5000 : 2600);
  }

  function mediaInstall(job) {
    const card = $("#mediaInstall");
    if (!card || !job) return;
    const priorState = card.dataset.job === job.id ? card.dataset.state : "";
    let dismissed = "";
    try { dismissed = localStorage.getItem("sheep.mediaInstall.dismissed") || ""; } catch (_) { /* optional */ }
    if (dismissed === job.id) { card.hidden = true; return; }
    const terminal = ["complete", "failed", "cancelled"].includes(job.state);
    const percent = Number.isFinite(job.progress) ? Math.max(0, Math.min(100, Math.round(job.progress * 100))) : null;
    $("#mediaInstallTitle").textContent = ({ image: "Installing picture models", video: "Installing video models",
      "video-reference": "Installing photo-guidance model", audio: "Preparing speech recognition" })[job.kind] || "Installing media model";
    $("#mediaInstallPhase").textContent = `${job.model} · ${job.phase}`;
    $("#mediaInstallDetail").textContent = job.detail || "";
    $("#mediaInstallHint").textContent = terminal
      ? (job.state === "complete" ? "Your request is continuing automatically." : "The requested task stopped. You can try again.")
      : "Your request will continue when installation finishes.";
    if (job.state === "cancelled") $("#mediaInstallHint").textContent = "Download cancelled. Your previous model choice is unchanged.";
    $("#mediaInstallError").hidden = job.state !== "failed";
    $("#mediaInstallError").textContent = job.state === "failed" ? job.error : "";
    $("#mediaInstallErrorDetails").hidden = job.state !== "failed" || !job.error_detail;
    $("#mediaInstallErrorCause").textContent = job.error_detail || "";
    $("#mediaInstallPct").textContent = percent === null ? "" : `${percent}%`;
    $("#mediaInstallTrack").classList.toggle("indeterminate", percent === null && !terminal);
    if (percent === null) $("#mediaInstallTrack").removeAttribute("aria-valuenow");
    else $("#mediaInstallTrack").setAttribute("aria-valuenow", String(percent));
    $("#mediaInstallTrack").setAttribute("aria-valuetext", percent === null ? job.phase : `${percent}%`);
    $("#mediaInstallFill").style.width = percent === null ? "" : `${percent}%`;
    card.classList.toggle("failed", job.state === "failed");
    card.classList.toggle("complete", job.state === "complete");
    card.classList.toggle("cancelled", job.state === "cancelled");
    $("#mediaInstallDismiss").disabled = !terminal;
    $("#mediaInstallCancel").hidden = terminal;
    card.hidden = false;
    card.dataset.job = job.id;
    card.dataset.state = job.state;
    if (terminal && priorState !== job.state && $("#modelMenu").classList.contains("open")) {
      $("#modelMenu").classList.remove("open");
      modelMenu();
    }
  }
  async function loadMediaInstall() {
    try {
      const r = await api("/api/media_install");
      if (r.job) mediaInstall(r.job);
    } catch (_) { /* the backend may still be starting */ }
  }

  // ------------------------------------------------------------ state
  const WORD = { green: "Ready", amber: "Checking", red: "Needs attention", off: "Orchestrator off" };
  const READY_PREF = "sheep.readiness.expanded";
  function readinessExpanded(hasModel) {
    try {
      const saved = localStorage.getItem(READY_PREF);
      return saved === null ? !!hasModel : saved === "1";
    } catch (_) { return !!hasModel; }
  }
  function setReadinessExpanded(expanded) {
    const details = $("#rbDetails"), toggle = $("#rbToggle");
    details.hidden = !expanded;
    toggle.setAttribute("aria-expanded", String(expanded));
    toggle.textContent = expanded ? "Hide details" : "Details";
    try { localStorage.setItem(READY_PREF, expanded ? "1" : "0"); } catch (_) { /* storage can be disabled */ }
  }
  const readinessNames = { startup: "Model ready", canaries: "Answers checked", orientation: "Apps checked" };
  const readinessStates = { "not started": "Not started", queued: "Waiting", running: "In progress", passed: "Done", failed: "Needs a retry", blocked: "Unavailable", deferred: "Waiting for a quiet moment", "waiting for chat": "Waiting for chat", "waiting for quick exam": "Waiting for the first check", "blocked by canary failures": "Waiting for an earlier check", "waiting for GPU": "Waiting for the GPU", "waiting for temperature": "Paused: this computer is hot; it continues when it cools", "waiting for model load": "Waiting for model load" };
  const readinessStages = { "probing this machine": "Checking this computer", planning: "Choosing a device", "starting the engine": "Starting the model", "retrying one app task": "Retrying one app task" };
  async function refresh() {
    try { S.state = await api("/api/state"); } catch (e) {
      $("#status").className = "status red"; $("#statusText").textContent = "Sheep unavailable";
      $("#status").title = "The local Sheep service did not answer. Click to retry the connection.";
      $("#status").setAttribute("aria-label", "Sheep unavailable. Click to retry the connection.");
      return;
    }
    const st = S.state, o = st.orch || {};
    if (S.tab === "models" && S.setMounted && S.setMounted.refreshState) S.setMounted.refreshState(st);
    const hasModel = !!st.model.path;
    document.documentElement.dataset.mode = Number(st.mode) === 1 ? "nebius" : "sheep";
    $("#modelName").textContent = st.model.name ? prettyModel(st.model.name) : "No model";
    const lamp = o.on ? o.lamp : "off";
    if (!hasModel) {
      $("#status").className = "status add-model";
      $("#statusText").textContent = "Add model";
      $("#status").title = "Add a Chat model to start";
      $("#status").setAttribute("aria-label", "Add a Chat model to start");
    } else {
      $("#status").className = "status " + lamp;
      $("#statusText").textContent = !st.engine.up && o.on ? "Starting" : lamp === "amber" ? (/^(starting|probing|planning)/.test(o.stage || "") ? "Getting ready" : "Checking the model") : WORD[lamp] || "";
      $("#status").title = o.on ? `Orchestrator: ${o.stage || lamp}. Click for the dashboard.` : "The orchestrator is off";
      $("#status").setAttribute("aria-label", `System status: ${$("#statusText").textContent || "ready"}`);
    }
    $$("#modeSeg button").forEach((b) => b.classList.toggle("on", Number(b.dataset.mode) === st.mode));
    // Startup smoke and longer checks share one visible readiness panel. The
    // active model becomes usable after its quick exam; background rows continue
    // below the chat priority line.
    // every task passed is a pass, whatever an older record's state says: a retried
    // task left "Needs a retry" beside 10/10 passed (2026-10-13)
    const checks = (Array.isArray(o.checks) ? o.checks : []).map((c) =>
      c.state === "failed" && Number(c.passed) > 0 && Number(c.passed) >= Number(c.of || c.total)
        ? { ...c, state: "passed", why: "" } : c);
    const startupCheck = checks.find((c) => c.id === "startup");
    const waiting = hasModel && startupCheck && startupCheck.state === "running";
    const showChecks = hasModel && checks.some((c) => c.state && c.state !== "not started");
    const backgroundChecks = checks.filter((c) => c.id !== "startup");
    const backgroundStarted = backgroundChecks.some((c) => c.state && c.state !== "not started");
    // closed by the person, it stays closed until a check changes; all passed, it goes by itself
    const fp = `${st.model.path}|${checks.map((c) => `${c.id}:${c.state}`).join(",")}`;
    S.readinessFp = fp;
    let closedFp = ""; try { closedFp = localStorage.getItem("sheep.readiness.closed") || ""; } catch (_) { /* storage off */ }
    const allPassed = checks.length > 0 && checks.every((c) => c.state === "passed");
    $("#readyBar").hidden = (hasModel && !waiting && !showChecks) || (!waiting && (allPassed || closedFp === fp));
    $("#rbCancel").hidden = !waiting;
    $("#rbChoose").hidden = hasModel;
    $("#rbProgress").hidden = !hasModel || (!waiting && startupCheck?.state === "failed" && !backgroundStarted);
    // open while the person waits for chat, or when a check needs them; once chat is
    // ready the background checks fold to one line (open only if the person opened it)
    const needsThem = checks.some((c) => c.state === "failed");
    let openSaved = false;
    try { openSaved = localStorage.getItem(READY_PREF) === "1"; } catch (_) { /* storage can be disabled */ }
    const expanded = waiting || needsThem ? readinessExpanded(hasModel) : openSaved;
    $("#rbDetails").hidden = !expanded;
    $("#rbToggle").setAttribute("aria-expanded", String(expanded));
    $("#rbToggle").textContent = expanded ? "Hide details" : "Details";
    $("#rbEmpty").hidden = hasModel;
    if (!hasModel) {
      $("#rbText").textContent = "No model selected. Add a model to start; checks haven’t begun.";
      $("#rbPct").textContent = "";
      $("#rbFill").style.width = "0%";
      $("#rbEmpty").textContent = "Choose a chat model first. Sheep will run its checks after setup.";
    }
    if (waiting) {
      const cover = st.mode === 1 && st.cloud.ready;
      const stage = !st.engine.up || /^(probing|planning|starting)/.test(o.stage || "") ? "Starting your model" : "Checking your model";
      $("#rbText").textContent = cover ? `${stage} · Nemotron answers meanwhile` : `${stage} · you can type; I'll answer when it's ready`;
      $("#rbPct").textContent = "";
      $("#rbFill").style.width = "35%";
    } else if (showChecks) {
      const background = backgroundChecks;
      const completed = background.filter((c) => ["passed", "failed", "blocked by canary failures"].includes(c.state)).length;
      const active = background.find((c) => ["queued", "running", "waiting for chat"].includes(c.state));
      $("#rbText").textContent = startupCheck?.state === "failed" && !backgroundStarted
        ? "Model didn’t start" : active ? `Model ready · ${active.label.toLowerCase()} ${active.state}` : "System readiness";
      $("#rbPct").textContent = backgroundStarted ? `${completed}/${background.length} checks` : "";
      const overall = background.length ? background.reduce((sum, c) => sum + (Number(c.progress) || 0), 0) / background.length : 1;
      $("#rbFill").style.width = backgroundStarted ? `${Math.round(100 * overall)}%` : "0%";
    }
    if (waiting) {
      $("#rbText").textContent = readinessNames.startup;
      const known = startupCheck.progress !== null && startupCheck.progress !== undefined && Number.isFinite(Number(startupCheck.progress));
      const pct = known ? Math.max(0, Math.min(100, Math.round(100 * Number(startupCheck.progress)))) : null;
      $("#rbPct").textContent = pct === null ? "" : `${pct}%`;
      $("#rbFill").style.width = pct === null ? "35%" : `${pct}%`;
      $("#rbProgress").classList.toggle("indeterminate", pct === null);
    } else if (showChecks) {
      const active = checks.find((c) => ["running", "waiting for chat", "waiting for quick exam"].includes(c.state))
        || checks.find((c) => c.state === "queued");
      $("#rbText").textContent = active ? (readinessNames[active.id] || active.label) : "Model checks";
      const background = backgroundChecks;
      const completed = background.filter((c) => ["passed", "failed", "blocked by canary failures"].includes(c.state)).length;
      const activeProgress = active && active.state === "running" && active.progress !== null && active.progress !== undefined && Number.isFinite(Number(active.progress))
        ? Math.max(0, Math.min(100, Math.round(Number(active.progress) * 100))) : null;
      $("#rbPct").textContent = !backgroundStarted ? "" : background.length ? `${activeProgress === null ? "" : `${activeProgress}% · `}${completed}/${background.length} done` : "Done";
      const overall = background.length ? (completed + (activeProgress === null ? 0 : activeProgress / 100)) / background.length : 1;
      $("#rbFill").style.width = backgroundStarted ? `${Math.round(100 * overall)}%` : "0%";
      $("#rbProgress").classList.remove("indeterminate");
    }
    const appCheckIssue = checks.find((c) => c.id === "orientation" && c.state === "failed");
    if (appCheckIssue && startupCheck?.state === "passed" && !checks.some((c) => ["queued", "running", "waiting for chat"].includes(c.state))) {
      const passed = Number(appCheckIssue.passed ?? appCheckIssue.done);
      const total = Number(appCheckIssue.of || appCheckIssue.total);
      $("#rbText").textContent = "Chat is ready";
      $("#rbPct").textContent = Number.isFinite(passed) && Number.isFinite(total) && total > 0
        ? `${passed} of ${total} app tasks work` : "One app task needs a retry";
      // The audit has finished. Its result is advisory and does not make Chat unavailable.
      $("#rbFill").style.width = "100%";
    }
    const appCheckActive = checks.find((c) => c.id === "orientation" &&
      ["running", "waiting for chat", "waiting for GPU", "waiting for temperature"].includes(c.state));
    if (appCheckActive && startupCheck?.state === "passed") {
      const passed = Number(appCheckActive.passed ?? appCheckActive.done);
      const total = Number(appCheckActive.of || appCheckActive.total);
      const ran = Number(appCheckActive.done);
      const ratio = Number.isFinite(ran) && Number.isFinite(total) && total > 0 ? `Checking app tasks · ${ran} of ${total}` : "Checking app tasks";
      $("#rbText").textContent = appCheckActive.state === "waiting for chat"
        ? "Chat is ready · app check waits while you chat"
        : /^retrying/.test(appCheckActive.stage || "") ? `Chat is ready · ${appCheckActive.stage}` : "Chat is ready";
      $("#rbPct").textContent = ratio;
    }
    if (showChecks) {
      $("#rbChecks").innerHTML = checks.map((c) => {
        const tone = c.state === "passed" ? "green" : c.state === "failed" && c.id === "startup" ? "red"
          : c.state === "not started" ? "off" : "amber";
        const validNumber = (n) => n !== null && n !== undefined && n !== "" && Number.isFinite(Number(n));
        // "done" is tasks run, "passed" is tasks that worked: 10/10 beside "one fell short" was the
        // run count (2026-10-13); the count and the words now say the same thing
        const shortBy = validNumber(c.passed) && validNumber(c.total) ? Number(c.total) - Number(c.passed) : 1;
        const stage = c.id === "orientation" && c.state === "failed"
          ? `Chat is ready; ${shortBy === 1 ? "one app task" : `${shortBy} app tasks`} fell short`
            + (c.why ? ` (${String(c.why).split(";")[0].slice(0, 70)})` : "")
          : c.why || (c.state === "running" ? (readinessStages[c.stage] || c.stage) : "");
        const detail = [readinessStates[c.state] || c.state || "Waiting", stage, c.device,
          validNumber(c.elapsed_seconds) && Number(c.elapsed_seconds) > 0 ? `${Math.round(Number(c.elapsed_seconds))}s elapsed` : "",
          validNumber(c.eta_seconds) && Number(c.eta_seconds) > 0 ? `about ${Math.ceil(Number(c.eta_seconds))}s left` : ""]
          .filter(Boolean).join(" · ");
        const finished = ["passed", "failed"].includes(c.state) && validNumber(c.passed);
        const count = validNumber(c.done) && validNumber(c.total) && Number(c.total) > 0
          ? `${finished ? c.passed : c.done}/${c.total}` : "";
        const hasCompletedWork = validNumber(c.done) && Number(c.done) > 0;
        const terminal = ["passed", "failed", "blocked"].includes(c.state);
        const known = c.progress !== null && c.progress !== undefined && Number.isFinite(Number(c.progress))
          && (terminal || hasCompletedWork || (c.state === "running" && Number(c.progress) > 0));
        const pct = known ? Math.max(0, Math.min(100, Math.round(Number(c.progress) * 100))) : null;
        const moving = c.state === "running" || String(c.state || "").startsWith("waiting for ");
        const label = readinessNames[c.id] || c.label;
        const appPassedForLabel = Number(c.passed ?? c.done), appTotalForLabel = Number(c.of || c.total);
        const appFailures = Number.isFinite(appPassedForLabel) && Number.isFinite(appTotalForLabel)
          ? Math.max(1, appTotalForLabel - appPassedForLabel) : 1;
        const stateText = c.id === "orientation" && c.state === "failed"
          ? `${appFailures} app task${appFailures === 1 ? " needs" : "s need"} a retry`
          : readinessStates[c.state] || c.state || "Waiting";
        const appPassed = validNumber(c.passed) ? c.passed : c.done;
        // "0/14 passed" while the check was still running read as fourteen failures (2026-10-21)
        const appOf = c.of || c.total;
        const resultCount = c.id !== "orientation" || !validNumber(appOf) ? count
          : c.state === "passed" ? `All ${appOf} work`
          : c.state === "failed" && validNumber(appPassed) ? `${appPassed} of ${appOf} work`
          : validNumber(c.done) ? `${c.done} of ${appOf}` : "";
        return `<button type="button" class="readiness-row" data-readiness-check="${esc(c.id)}" aria-label="${esc(`${label}: ${stateText}. Open it in Settings, Activity.`)}">
          <span class="readiness-label"><span class="lamp ${tone}"></span><span><b>${esc(label)}</b><small>${esc(detail || stateText)}</small></span><span class="readiness-count">${esc(resultCount || (pct === null ? "" : `${pct}%`))}</span></span>
          <span class="progress ${pct === null && moving ? "indeterminate" : ""}"><i style="width:${pct === null ? (moving ? 35 : 0) : pct}%"></i></span></button>`;
      }).join("");
    } else {
      $("#rbChecks").innerHTML = "";
    }
    // The empty chat shows the first-run card or the suggestions, depending on
    // whether a model is plugged in; state arrives after the first paint, so the
    // empty thread is redrawn the once that answer changes.
    const none = !st.model.path && !cloudChat(st);
    if (!S.log.length && S.noModel !== none) { S.noModel = none; renderThread(); }
    document.documentElement.dataset.theme = themeOf(st.theme);
    if (S.mounted && S.mounted.update) S.mounted.update(st);
    if (S.setMounted && S.setMounted.update) S.setMounted.update(st);
    if (S.pendingChat && st.model.path && ((o.on && o.ready) || (!o.on && st.engine && st.engine.up))) resumePendingChat();
  }
  async function loadTree() {
    try { S.files = (await api("/api/files")).files; } catch (e) { /* the backend is starting */ }
  }

  // ------------------------------------------------------------ the canvas (apps beside the chat)
  // Before an editor goes: blur the focused field so its save-on-leave (Words'
  // focusout) runs while the element still exists, then ask before dropping
  // changes an editor holds unsaved (Text & Code saves only on Ctrl+S).
  function leaveOk() {
    const a = document.activeElement;
    if (a && $("#canvasBody").contains(a) && a.blur) a.blur();
    const v = S.mounted;
    // nothing unsaved: answer now, so opening an app stays immediate; otherwise ask in Sheep's own box
    if (!(v && v.dirty)) return true;
    return ask(`${v.path || "This file"} has changes that are not saved. Close without saving?`, { ok: "Close without saving", danger: true });
  }
  // Sheep's own yes/no box, in place of the Windows one: same theme, Enter is yes, Escape is no (2026-10-13)
  // with input (a string, prefilled): a text box, and the answer is its trimmed value or null
  function ask(text, { ok = "OK", cancel = "Cancel", danger = false, input = null } = {}) {
    return new Promise((done0) => {
      const done = (yes) => done0(input === null ? yes : yes ? (box.querySelector(".askin").value.trim() || null) : null);
      const box = document.createElement("div");
      box.className = "modal ask"; box.setAttribute("role", "alertdialog"); box.setAttribute("aria-modal", "true");
      box.innerHTML = `<div class="askbox"><p></p>${input === null ? "" : `<input class="askin" type="text" value="${esc(input)}" aria-label="${esc(text)}">`}<div class="askrow"><button type="button" id="askNo">${esc(cancel)}</button>
        <button type="button" id="askOk" class="${danger ? "danger" : "primary"}">${esc(ok)}</button></div></div>`;
      box.querySelector("p").textContent = text;
      const was = document.activeElement;
      const end = (yes) => { document.removeEventListener("keydown", key, true); box.remove(); if (was && was.focus) was.focus(); done(yes); };
      const key = (e) => {
        if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); end(false); }
        else if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); end(document.activeElement && document.activeElement.id === "askNo" ? false : true); }
      };
      document.addEventListener("keydown", key, true);
      box.querySelector("#askOk").onclick = () => end(true);
      box.querySelector("#askNo").onclick = () => end(false);
      box.addEventListener("click", (e) => { if (e.target === box) end(false); });
      document.body.appendChild(box);
      if (input === null) box.querySelector("#askOk").focus(); else { const f = box.querySelector(".askin"); f.focus(); f.select(); }
    });
  }
  async function openApp(app, path) {
    if (S.mounted) { const ok = leaveOk(); if (ok !== true && !(await ok)) return; }
    closeSettings();
    if (path !== undefined) S.open[app] = path;
    if (S.mounted && S.mounted.unmount) S.mounted.unmount();
    S.app = app;
    $("#canvas").hidden = false;
    $("#splitter").hidden = false;
    $("#split").classList.add("canvas");
    $("#split").classList.toggle("wide", app === "studio");      // a video needs the room
    $$("#side [data-app]").forEach((b) => b.classList.toggle("on", b.dataset.app === app));
    const body = $("#canvasBody");
    body.innerHTML = "";
    const v = S.views[app];
    S.mounted = v || null;
    if (v) v.mount(body, S.open[app]);
    window.MLMI_ICON.paint(body);
    if (NATIVE[app]) api("/api/warm", { app }).catch(() => {});
    $("#askText").placeholder = app === "work" ? "Message Sheep" : `Message Sheep about ${APP_NAME[app] || "this"}`;
    $("#canvasTitle").textContent = APP_NAME[app] || "";
    $("#toWork").hidden = app === "work" || !(MLMI.workCount && MLMI.workCount()) && !(MLMI.workActiveCount && MLMI.workActiveCount());
    workBadge();
  }
  async function closeCanvas() {
    if (S.mounted) { const ok = leaveOk(); if (ok !== true && !(await ok)) return; }
    if (S.mounted && S.mounted.unmount) S.mounted.unmount();
    S.mounted = null; S.app = "";
    $("#canvas").hidden = true;
    $("#splitter").hidden = true;
    $("#split").classList.remove("canvas", "wide", "focus");
    $("#focusCanvas").classList.remove("on");
    $$("#side [data-app]").forEach((b) => b.classList.remove("on"));
    $("#askText").placeholder = "Message Sheep";
    workBadge();
  }
  // The Canvas badge separates finished artifacts from work still in progress.
  function workBadge() {
    const n = MLMI.workCount ? MLMI.workCount() : 0;
    const active = MLMI.workActiveCount ? MLMI.workActiveCount() : 0;
    const b = $("#workBtn");
    if (!b) return;
    b.classList.toggle("on", S.app === "work");
    b.classList.toggle("has-active", active > 0);
    b.title = `Work Canvas: ${n} completed, ${active} active`;
    b.setAttribute("aria-label", `Work Canvas, ${n} completed, ${active} active`);
    $("#workN").textContent = active ? `${n} · ${active} active` : n ? String(n) : "";
    $("#toWork").hidden = S.app === "work" || (!n && !active);
  }
  function refreshWork() {
    workBadge();
    if (S.app === "work" && S.mounted && S.mounted.refresh) S.mounted.refresh();
  }
  async function openFile(path) {
    const ext = path.split(".").pop().toLowerCase();
    if (OFFICE[ext]) {
      const native = path.slice(0, -ext.length) + OFFICE[ext];
      if (!S.files.some((f) => f.path === native)) { await op("import_file", { path }); await loadTree(); }
      return openFile(native);
    }
    const app = APP_OF[ext];
    if (app) return openApp(app, path);
    openSettings("files"); setTimeout(() => S.setMounted && S.setMounted.show && S.setMounted.show(path), 300);
  }
  // one door for every page: apps open in the canvas, the rest in Settings
  function go(view) {
    if (view === "video" || view === "motion") view = "studio";       // the old names
    if (APPS.includes(view)) return openApp(view);
    if (SETTINGS.includes(view)) return openSettings(view);
    if (view === "settings") return openSettings("basics");
    closeSettings(); closeCanvas();
  }

  // ------------------------------------------------------------ settings sheet
  function openSettings(tab = "basics", focus = "") {
    $("#settings").hidden = false;
    S.focus = focus;                 // a section to open and scroll to (Activity's checks)
    S.fold = "";                     // a fold to open on a page made of folds (Advanced)
    if (SET_ROUTE[tab]) { S.fold = SET_ROUTE[tab][1] || ""; tab = SET_ROUTE[tab][0]; }
    if (S.setMounted && S.setMounted.unmount) S.setMounted.unmount();
    S.tab = tab;
    $$("#setTabs [data-tab]").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
    const body = $("#setBody");
    body.innerHTML = "";
    const v = S.views[tab];
    S.setMounted = v || null;
    if (v) v.mount(body);
    window.MLMI_ICON.paint(body);
  }
  function closeSettings() {
    if (S.setMounted && S.setMounted.unmount) S.setMounted.unmount();
    S.setMounted = null;
    $("#settings").hidden = true;
  }

  // ------------------------------------------------------------ chat history
  async function loadChats() {
    const words = ($("#chatSearch") && $("#chatSearch").value.trim()) || "";
    const r = await api("/api/chats" + (words ? "?q=" + encodeURIComponent(words) : "")).catch(() => ({ chats: [] }));
    $("#chats").innerHTML = r.chats.map((c) => `<button class="side-item ${c.id === S.chatId ? "on" : ""}" data-chat="${c.id}" title="${esc(c.title)}">
      <span class="ctitle" style="overflow:hidden;text-overflow:ellipsis">${esc(c.title)}</span>
      <span class="del" data-ren="${c.id}" title="Rename">${icon("pencil")}</span><span class="del" data-del="${c.id}" title="Delete">${icon("trash")}</span></button>`).join("")
      || (words ? `<div class="note" style="padding:6px 10px">No chat mentions “${esc(words)}”.</div>` : "");
  }
  function renameChat(row, id) {
    const span = row.querySelector(".ctitle");
    const was = span.textContent;
    const box = document.createElement("input");
    box.className = "chat-rename"; box.value = was; box.setAttribute("aria-label", "Chat name");
    span.replaceWith(box); box.focus(); box.select();
    let done = false;
    const finish = async (save) => {
      if (done) return; done = true;
      const name = box.value.trim();
      if (save && name && name !== was) {
        try { await api("/api/chats", { id, rename: name }); } catch (e) { toast(e.message, "bad"); }
      }
      loadChats();
    };
    box.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); };
    box.onclick = (e) => e.stopPropagation();
    box.onblur = () => finish(true);
  }
  async function deleteChat(id) {
    const row = $(`#chats [data-chat="${id}"] .ctitle`);
    if (!await ask(`Delete the chat “${row ? row.textContent : "this chat"}”? Files it made stay in the workspace.`)) return;
    // a job still working for this chat would keep writing to a chat that is gone
    if (S.activeJob && S.activeJob.chatId === id) { try { await api("/api/stop", {}); } catch (_) { /* already finished */ } }
    try { await api("/api/chats", { id, delete: true }); } catch (e) { return toast(e.message, "bad"); }
    if (S.chatId === id) newChat(); else loadChats();
  }
  // the open chat survives a reload of this window (a backend restart), not a new launch
  function rememberChat(id) {
    try { if (id) sessionStorage.setItem("sheep.openChat", id); else sessionStorage.removeItem("sheep.openChat"); } catch (_) { /* optional */ }
  }
  async function openChat(id) {
    const version = ++S.chatVersion;
    const c = await api("/api/chats?id=" + id).catch(() => null);
    if (!c && version === S.chatVersion) { rememberChat(null); loadChats(); return; }    // gone: show the list
    if (!c || version !== S.chatVersion) return;
    S.chatId = id; S.convo = c.messages || []; S.log = c.view || [];
    rememberChat(id);
    S.canvas = Object.assign({ order: [], pinned: {} }, c.canvas || {});
    const liveJob = [S.activeJob, S.lastJob].find((job) => job && job.chatId === id);
    if (liveJob) {
      const at = S.log.findIndex((entry) => entry.jobId === liveJob.id);
      if (at >= 0) S.log[at] = liveJob.bot;
      liveJob.log = S.log; liveJob.convo = S.convo; liveJob.canvas = S.canvas; liveJob.chatVersion = version;
    } else {
      let recovered = false;
      for (const entry of S.log) {
        if (entry.role === "bot" && ["active", "stopping"].includes(entry.workState)) {
          entry.done = true; entry.workState = "failed"; entry.workStage = "Interrupted before Sheep finished";
          entry.thinking = null;
          entry.text = entry.text || "This run was interrupted. Any completed files remain available in the workspace.";
          recovered = true;
        }
      }
      if (recovered) saveChat();
    }
    renderThread();
    refreshWork();
    loadChats();
  }
  function newChat() {
    S.chatVersion++;
    S.chatId = null; S.convo = []; S.log = []; S.canvas = { order: [], pinned: {} };
    rememberChat(null);
    renderThread();
    refreshWork();
    loadChats();
    $("#askText").focus();
  }
  async function saveChat(job = null) {
    const owner = job || { chatId: S.chatId, convo: S.convo, log: S.log, canvas: S.canvas, chatVersion: S.chatVersion };
    const current = () => S.chatVersion === owner.chatVersion && S.log === owner.log;
    try {
      const r = await api("/api/chats", { id: owner.chatId, messages: owner.convo, view: owner.log, canvas: owner.canvas });
      owner.chatId = r.id;
      if (current()) { S.chatId = r.id; rememberChat(r.id); }
      loadChats();
      return r.id;
    } catch (e) { /* history is a convenience */ }
  }

  // ------------------------------------------------------------ the thread
  // what the job card says while a tool runs: plain words, never a tool name
  const DOING = { words_create: "Creating the document", sheets_create: "Creating the workbook", slides_create: "Creating the deck",
    words_add: "Writing", words_add_list: "Adding a list", sheets_write: "Filling the cells", slides_add: "Adding a slide",
    export_file: "Exporting", make_image: "Making the picture", make_video: "Setting up your video", speak_text: "Recording the voice",
    web_search: "Searching the web", web_fetch: "Reading a page" };
  const VERB = {
    words_create: "Created", sheets_create: "Created", slides_create: "Created", words_add: "Wrote in", words_add_list: "Added a list to",
    words_add_table: "Added a table to", words_set: "Edited", words_set_cell: "Edited a table in", words_delete: "Removed a block from",
    words_move: "Moved a block in", words_replace: "Replaced text in", words_format: "Formatted", words_add_image: "Added a picture to",
    words_add_page_break: "Added a page break to", words_read: "Read", sheets_read: "Read", slides_read: "Read", read_file: "Read",
    list_files: "Looked through the workspace", sheets_table: "Filled a table in", sheets_write: "Wrote cells in", sheets_set_cell: "Edited a cell in",
    sheets_format: "Formatted", sheets_add_chart: "Added a chart to", sheets_sort: "Sorted", sheets_add_sheet: "Added a sheet to",
    slides_add: "Added a slide to", slides_set: "Edited a slide in", slides_delete: "Removed a slide from", slides_move: "Reordered",
    slides_set_theme: "Restyled", slides_add_image: "Added a picture to", slides_add_text: "Added text to", export_file: "Exported",
    import_file: "Imported", write_text_file: "Wrote", copy_file: "Copied", move_file: "Moved", delete_file: "Deleted", create_folder: "Made a folder",
    herder_run: "Ran a custom tool on", herder_make: "Wrote a custom tool", herder_list: "Checked the custom tools", herder_help: "Read how to write a custom tool",
  };
  // A step that went wrong and was then put right is part of the work, not an
  // error: it shows a quiet "retried" mark. The warning icon is kept for a job
  // that ended without doing what was asked (Ernest, 2026-10-13).
  function stepHtml(s, failedJob) {
    const f = s.args && (s.args.path || s.args.dst || s.args.src);
    const bad = s.ok === false && failedJob;
    const cls = s.ok === undefined ? "run" : s.ok ? "ok" : bad ? "fail" : "retry";
    const ic = s.ok === undefined ? "spark" : s.ok ? "check" : bad ? "alert" : "redo";
    const label = VERB[s.name] || s.name.replace(/_/g, " ");
    const why = s.ok === false ? ` <span class="dim">· ${bad ? esc(String(s.text || "").split(/[.;\n]/)[0].slice(0, 90)) : "retried"}</span>` : "";
    return `<div class="step ${cls}">${icon(ic)}<span>${esc(label)}</span>${f ? `<span class="file" data-open="${esc(f)}">${esc(f)}</span>` : ""}${why}</div>`;
  }
  // File names in an answer open the file on the canvas: white and underlined
  // when Sheep answered, green and italic when Nebius did (Ernest, 2026-10-13).
  const FILE_RE = /(^|[\s(“"'`])([\w.\/-]+\.(?:sdoc|ssheet|sdeck|sstudio|docx|xlsx|pptx|pdf|md|txt|csv|json|py|js|html|mp4|mov|webm|png|jpe?g|gif|webp|wav|mp3|eml|ics))(?=$|[\s).,;:!?”"'`])/gi;
  function linkFiles(root, cloud) {
    const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (n.parentElement.closest("a, code, pre, .file, .flink") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT) });
    const nodes = [];
    while (walk.nextNode()) nodes.push(walk.currentNode);
    for (const n of nodes) {
      if (!FILE_RE.test(n.nodeValue)) continue;
      FILE_RE.lastIndex = 0;
      const span = document.createElement("span");
      span.innerHTML = esc(n.nodeValue).replace(FILE_RE, (m, lead, p) =>
        `${lead}<a href="#" class="flink ${cloud ? "cloud" : "sheep"}" data-open="${esc(p)}" title="Open ${esc(p)}">${esc(p)}</a>`);
      n.replaceWith(...span.childNodes);
    }
  }
  // A document-shaped answer is shown as a file first: its title and first lines,
  // "Show text" to read it here, "Edit & save" to type in it (Text & Code, .md).
  function docReply(e) {
    const first = (e.text.split(/\r?\n/).find((l) => l.trim()) || "Document").replace(/^#+\s*/, "").slice(0, 80);
    const lines = e.text.split(/\r?\n/).filter((l) => l.trim()).slice(1, 4).map((l) => l.replace(/^[#>*\-\d.\s]+/, "")).join(" · ");
    return `<div class="chatblock doc reply-doc"><div class="filecard" data-docreply><span class="fi">${icon("doc")}</span>
        <span><b>${esc(first)}</b><small>Draft · ${e.text.split(/\s+/).length} words · not saved yet</small></span></div>
      <div class="cb-prev">${esc(lines.slice(0, 220))}</div>
      <div class="cb-foot"><span></span><span class="row"><button class="ghost" data-showtext>${icon("doc")}Show text</button>
        <button class="ghost" data-draft>${icon("pencil")}Edit &amp; save</button></span></div>
      <div class="text reply-full" hidden>${MLMI.md ? MLMI.md(e.text) : esc(e.text)}</div></div>`;
  }
  // What a job made, shown in the chat as the block it is on the Work Canvas: a
  // picture as a picture, a video or sound as a player, a document with its
  // first lines. (Reported 2026-10-06: the canvas blocks did not show in chat.)
  const MEDIA_OF = { png: "image", jpg: "image", jpeg: "image", gif: "image", webp: "image", bmp: "image",
    mp4: "video", webm: "video", mov: "video", mkv: "video", mp3: "audio", wav: "audio", ogg: "audio", m4a: "audio", flac: "audio" };
  const rawUrl = (p) => "/api/raw?path=" + encodeURIComponent(p);
  function fileCard(p) {
    const ext = p.split(".").pop().toLowerCase();
    const name = p.split("/").pop();
    const media = MEDIA_OF[ext];
    const foot = (what) => `<div class="cb-foot"><span><b>${esc(name)}</b><small>${what}</small></span>
      <button class="ghost" data-work title="Show on the Work Canvas">${icon("canvas")}Canvas</button></div>`;
    if (media === "image") return `<div class="chatblock"><img loading="lazy" src="${rawUrl(p)}" alt="${esc(name)}">${foot("Picture")}</div>`;
    if (media === "video") return `<div class="chatblock"><video controls preload="metadata" src="${rawUrl(p)}"></video>${foot(p.startsWith("media/")
      ? `AI video · <a href="#" data-vtip>what decides its speed and quality</a>` : "Video")}</div>`;
    if (media === "audio") return `<div class="chatblock audio"><audio controls preload="metadata" src="${rawUrl(p)}"></audio>${foot("Sound")}</div>`;
    const app = APP_OF[ext];
    const ic = { words: "doc", sheets: "grid", slides: "slides", studio: "play" }[app] || "doc";
    // the name a person gave, with its app's icon; Open and the Office copy one click away
    const shown = app ? name.replace(/\.[^.]+$/, "") : name;
    const office = { words: "docx", sheets: "xlsx", slides: "pptx" }[app];
    return `<div class="chatblock doc"><div class="filecard" data-open="${esc(p)}"><span class="fi">${icon(ic)}</span><span><b>${esc(shown)}</b><small>${esc(APP_NAME[app] || "file")}</small></span></div>
      ${app && app !== "studio" ? `<div class="cb-prev" data-prev="${esc(p)}"></div>` : ""}<div class="cb-foot"><span><small>${esc(APP_NAME[app] || "File")}</small></span>
      <span class="row"><button class="ghost" data-open="${esc(p)}">${icon(ic)}Open</button>${office ? `<button class="ghost" data-export="${esc(p)}" title="Save a copy as .${office}">.${office}</button>` : ""}
      <button class="ghost" data-work title="Show on the Work Canvas">${icon("canvas")}Canvas</button></span></div></div>`;
  }
  // the first lines of each document block, read once and kept
  const PREV = {};
  function fillPreviews(changedPath = null) {
    $$("#thread [data-prev]").forEach(async (d) => {
      const p = d.dataset.prev;
      if (changedPath && p !== changedPath) return;
      if (d.dataset.done && !changedPath) return;
      if (changedPath) { delete PREV[p]; delete d.dataset.done; }
      d.dataset.done = "1";
      if (!(p in PREV)) PREV[p] = (await api("/api/preview?path=" + encodeURIComponent(p)).catch(() => ({ text: "" }))).text || "";
      d.textContent = PREV[p];
    });
  }
  function durationWords(seconds) {
    const s = Math.max(0, Math.floor(Number(seconds) || 0));
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60), rest = s % 60;
    if (m < 60) return rest ? `${m}m ${rest}s` : `${m}m`;
    const h = Math.floor(m / 60), mins = m % 60;
    return mins ? `${h}h ${mins}m` : `${h}h`;
  }
  function elapsedText(startedAt, estimateSeconds) {
    const elapsed = Math.max(0, Math.floor((Date.now() - Number(startedAt || Date.now())) / 1000));
    const estimate = Number(estimateSeconds) || 0;
    let text = `${durationWords(elapsed)} elapsed`;
    if (estimate > 0 && elapsed > estimate + Math.max(60, estimate * 0.25)) {
      text += " · taking longer than similar recent clips";
    } else if (estimate > 0) {
      text += ` · similar clips took about ${durationWords(estimate)}`;
    }
    return text;
  }
  // What is missing, each with its fix, the same rows as Settings › Advanced › This computer.
  // Speed-ups (GPU support) are shown as optional; they never block the request.
  const setupTag = (ok) => `<span class="stag ${ok ? "ok" : "bad"}">${ok ? "Ready" : "Missing"}</span>`;
  function setupButton(a, label, main) {
    if (!a) return "";
    const data = esc(JSON.stringify({ ...a, what: label, main }));
    // the suggested model, or any model of the person's choice for the same job
    const slot = { "sd15-lcm": "image", "animatediff-lightning": "video", "whisper-small": "speech_input" }[a.id];
    const other = a.do === "media" && slot
      ? `<button class="ghost" data-setup='${esc(JSON.stringify({ do: "slot", slot, label: "Choose another" }))}' title="Pick any Hugging Face model for this job instead">Choose another…</button>` : "";
    return a.do === "link" ? `<a class="btn" href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.label)}</a>`
      : `<button class="${main ? "primary" : ""}" data-setup='${data}'>${esc(a.label)}</button>${other}`;
  }
  function setupRows(items) {
    return items.map((i) => `<li class="setup-row ${i.ok ? "ready" : ""}">${setupTag(i.ok)}
      <div class="setup-what"><div><b>${esc(i.label)}</b>${!i.ok && (i.optional || i.speedup) ? ` <span class="setup-opt">${i.speedup ? "speeds it up" : "optional"}</span>` : ""}</div>
        ${i.ok ? "" : `<div class="dim">${esc(i.need || i.fix || "")}</div>`}</div>
      <div class="setup-act">${i.ok ? "" : setupButton(i.action, i.label, i.main !== undefined ? i.main : !i.speedup)}</div></li>`).join("");
  }
  function setupHtml(items) {
    return `<div class="chat-setup"><ul class="setup-list">${setupRows(items)}</ul>
      <div class="row"><button data-retry>Try again</button><span class="dim">after adding what is missing</span></div></div>`;
  }
  // One place that starts a fix, for the chat and for Settings: add a model, download a
  // feature model or GPU support (after a yes), or open a page outside Sheep.
  async function setupAction(a, btn) {
    if (a.do === "slot") { modelSlotChoice(a.slot); return true; }
    if (a.do === "link") return true;
    const size = (String(a.label).match(/\(([^)]*)\)/) || [])[1];
    if (!await ask(`Download ${a.what || "it"}${size ? ` (${size})` : ""}? It installs once, beside Sheep, and does not change your other programs.`, { ok: "Download" })) return false;
    if (btn) { btn.disabled = true; btn.textContent = "Downloading…"; }
    try {
      if (a.do === "media") await api("/api/media_model", { id: a.id, action: "use" });
      if (a.do === "runtime") await api("/api/gpu_runtime", { action: "install", kind: a.kind });
      toast(a.do === "runtime" ? "Downloading GPU support: the progress shows in Settings" : "Ready to use");
      if (btn) { btn.textContent = a.do === "runtime" ? "Downloading" : "Done"; }
      return true;
    } catch (err) {
      toast(err.message, "bad");
      if (btn) { btn.disabled = false; btn.textContent = a.label; }
      return false;
    }
  }

  function botHtml(e) {
    const cloud = e.route === "cloud";
    const timerStart = (e.thinking && e.thinking.t0) || e.t0 || Date.now();
    const timerEstimate = (e.thinking && e.thinking.estimateSeconds) || 0;
    return `<div class="msg bot"><span class="av ${cloud ? "cloud" : ""}">${cloud ? icon("cloud") : icon("sheep")}</span><div class="body">
      ${e.note ? `<div class="note-line">${icon("cloud")}<span>${esc(e.note)}</span></div>` : ""}
      ${e.waiting ? `<div class="wait-line"><span>${esc(/canary/.test(e.waiting.stage) ? "Checking your model" : "Starting your model")}… ${Math.round(100 * e.waiting.progress)}%</span>
        <div class="progress"><i style="width:${Math.round(100 * e.waiting.progress)}%"></i></div></div>` : ""}
      ${e.plan ? `<details class="plan"><summary>${icon("route")}<span>Planned by ${esc(cloudName())} on Nebius · ${e.plan.steps || ""} step${e.plan.steps === 1 ? "" : "s"}</span></summary>
        <pre>${esc(e.plan.text)}</pre></details>` : ""}
      ${e.review ? `<details class="plan"><summary>${icon("check")}<span>Checked by ${esc(cloudName(e.review.by))}${e.review.ok ? ": no changes needed"
        : e.review.fixed ? ` · ${e.review.fixes.length} fix${e.review.fixes.length === 1 ? "" : "es"} made` : " · its notes"}</span></summary>
        ${e.review.fixes.length ? `<ul class="why">${e.review.fixes.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : ""}</details>` : ""}
      ${e.steps && e.steps.length ? `<div class="steps">${e.steps.map((s) => stepHtml(s, e.workState === "failed")).join("")}</div>` : ""}
      ${e.confirm ? `<div class="confirm"><b>Allow this?</b> ${esc(e.confirm.reason)}<div class="row">${e.confirm.done ? `<span class="dim">${esc(e.confirm.done)}</span>`
        : `<button class="primary" data-allow="1">Allow</button><button data-allow="0">Decline</button>`}</div></div>` : ""}
      ${e.text && e.done && MLMI.isDocument && MLMI.isDocument(e.text) ? docReply(e)
        : e.text ? `<div class="text">${MLMI.md ? MLMI.md(e.text) : esc(e.text)}</div>` : e.draft && !e.done ? `<div class="text live">${MLMI.md ? MLMI.md(e.draft) : esc(e.draft)}<span class="caret"></span></div>`
        : e.done ? "" : `<div class="think-line"><span class="thinking"><i></i><i></i><i></i></span>
        <span class="shimmer">${esc(e.thinking ? e.thinking.what : "Thinking")}</span><span class="tsec" data-t0="${timerStart}" data-estimate="${timerEstimate}">${elapsedText(timerStart, timerEstimate)}</span></div>`}
      ${e.setup && e.setup.length ? setupHtml(e.setup) : ""}
      ${e.files && e.files.length ? e.files.map(fileCard).join("") : ""}
      ${e.done && e.text ? `<div class="msg-tools"><button class="copy-message" data-copymsg title="Copy the answer">${icon("copy")}Copy</button>
        <button class="copy-message" data-speak title="Read it aloud">${icon("audio")}Read aloud</button>${e.text.length > 400 && !(MLMI.isDocument && MLMI.isDocument(e.text))
          ? `<button class="copy-message" data-draft title="Open it in Sheep Text as an unsaved draft to edit and save">${icon("pencil")}Edit as draft</button>` : ""}</div>` : ""}
      ${e.sources && e.sources.length ? `<div class="sources">${e.sources.map((s, i) => { let host = ""; try { host = new URL(s.url).hostname.replace(/^www\./, ""); } catch (_) {}
        return `<a class="src" href="${esc(s.url)}" target="_blank" rel="noopener" title="${esc(s.snippet || "")}"><span class="n">${i + 1}</span><span><b>${esc(s.title || host)}</b><small>${esc(host)}</small></span></a>`; }).join("")}</div>` : ""}
    </div></div>`;
  }
  const SUGGEST = [
    ["doc", "Write a document", "a one-page project brief", "Make a one-page project brief in Sheep Words called brief.sdoc with a title, a short summary and a list of three goals."],
    ["grid", "Make a budget", "with a total row", "Make a workbook called budget.ssheet with columns Item and Cost: Rent 800, Food 250, Transport 90, and a Total row."],
    ["slides", "Build a deck", "three slides to pitch an idea", "Make a deck called pitch.sdeck titled My idea with two slides: The problem and The solution, each with two points."],
    ["doc", "Plan my week", "a to-do list, day by day", "Make a document called week.sdoc with the heading This week and a bullet list: Monday gym, Tuesday report, Wednesday call mum."],
  ];
  function renderThread() {
    const t = $("#thread");
    if (!S.log.length) {
      // No model yet: those suggestions would have nothing to answer them, so the
      // first run offers the two ways to get one instead (plan section 1).
      if (MLMI.welcome && S.state && S.state.model && !S.state.model.path && !cloudChat(S.state)) {
        t.innerHTML = `<div class="col">${MLMI.welcome.html(S.state)}</div>`;
        window.MLMI_ICON.paint(t);
        return;
      }
      t.innerHTML = `<div class="col"><div class="hello"><span class="logo">${icon("sheep")}</span><h1>What can I help with?</h1>
        <div class="suggest">${SUGGEST.map(([ic, a, b, q]) => `<button data-q="${esc(q)}">${icon(ic)}<span><b>${a}</b>${b}</span></button>`).join("")}</div></div></div>`;
      return;
    }
    t.innerHTML = `<div class="col">${S.log.map((e) => (e.role === "you" ? youHtml(e) : botHtml(e))).join("")}</div>`;
    linkAll(t);
    t.scrollTop = 1e9;
    fillPreviews();
  }
  function linkAll(root) {
    if (!root) return;
    const msgs = root.matches && root.matches(".msg.bot") ? [root] : $$(".msg.bot", root);
    msgs.forEach((m) => $$(".text", m).forEach((t) => linkFiles(t, !!m.querySelector(".av.cloud"))));
  }
  // streamed pieces repaint once a frame, not once a piece
  let paintQueued = false;
  function paintSoon(jobCtx = null) { if (!paintQueued) { paintQueued = true; requestAnimationFrame(() => { paintQueued = false; paintLast(jobCtx); }); } }
  function paintLast(jobCtx = null) {
    if (jobCtx && (S.chatVersion !== jobCtx.chatVersion || S.log !== jobCtx.log)) { refreshWork(); return; }
    const col = $("#thread .col");
    const log = jobCtx ? jobCtx.log : S.log;
    const last = log[log.length - 1];
    if (!col || !last) return renderThread();
    const node = col.lastElementChild;
    const html = last.role === "you" ? youHtml(last) : botHtml(last);
    const tmp = document.createElement("div");
    tmp.innerHTML = html;
    if (node && node.dataset.live) { tmp.firstElementChild.dataset.live = "1"; tmp.firstElementChild.style.animation = "none"; node.replaceWith(tmp.firstElementChild); }
    else { tmp.firstElementChild.dataset.live = "1"; col.appendChild(tmp.firstElementChild); }
    $("#thread").scrollTop = 1e9;
    if (last.done) { linkAll(col.lastElementChild); fillPreviews(); }
  }

  // ------------------------------------------------------------ the agent
  // A job the person walked away from (New chat, Stop) is let go at once: its
  // events are ignored and the composer is free. Reported 2026-10-01: after a
  // long job the chat kept "thinking" and refused new messages, even in a new chat.
  function release(job) {
    if (job) {
      S.lastJob = job;
      if (S.activeJob === job) S.activeJob = null;
    }
    S.busy = false;
    $("#sendBtn").hidden = false; $("#stopBtn").hidden = true;
    $("#stopBtn").disabled = false; $("#stopBtn").title = "Stop";
    workBadge();
  }
  function youHtml(e) {
    const chips = (e.attached || []).map((p) => `<span class="achip" data-open="${esc(p)}">${icon("clip")}${esc(p.split("/").pop())}</span>`).join("");
    return `<div class="msg you">${chips ? `<div class="achips">${chips}</div>` : ""}<div class="text">${esc(e.text)}</div></div>`;
  }
  // ------------------------------------------------------------ attachments: saved in the workspace
  // under attachments/; the orchestrator imports, reads or transcribes them before the model sees the request
  S.attach = [];
  function paintAttach() {
    const row = $("#attachRow");
    row.hidden = !S.attach.length;
    row.innerHTML = S.attach.map((a, i) => `<span class="achip ${a.path ? "" : "busy"}" title="${esc(a.path || "uploading")}">${icon("clip")}${esc(a.name)}<button type="button" class="ghost" data-unattach="${i}" title="Remove">×</button></span>`).join("");
  }
  async function attachFiles(files) {
    for (const f of files) {
      const a = { name: f.name, path: "" };
      S.attach.push(a);
      paintAttach();
      try {
        const r = await fetch("/api/upload?name=" + encodeURIComponent(f.name), { method: "POST", body: f });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || r.status);
        a.path = j.path;
      } catch (e) {
        S.attach.splice(S.attach.indexOf(a), 1);
        toast(`${f.name}: ${e.message}`, "bad");
      }
      paintAttach();
    }
  }

  // no local model yet, but Sheep + Nebius is on with a key: the chat is open (Nemotron answers)
  // "nvidia/nemotron-3-nano-30b-a3b" reads "Nemotron 3 Nano 30B"
  function cloudName(m) {
    const id = String(m || (S.state && S.state.cloud && S.state.cloud.model) || "Nemotron").replace(/^.*\//, "").replace(/:free$/, "");
    return id.replace(/-a\d+b$/i, "").split("-").map((w) => /^\d+b$/i.test(w) ? w.toUpperCase() : w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
  }
  // "system" follows Windows' light or dark choice
  const darkQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function themeOf(t) { return t === "light" || t === "dark" ? t : darkQuery && !darkQuery.matches ? "light" : "dark"; }
  if (darkQuery) darkQuery.addEventListener("change", () => { if (S.state) document.documentElement.dataset.theme = themeOf(S.state.theme); });
  function cloudChat(st) { return !!(st && st.mode === 1 && st.cloud && st.cloud.ready); }
  async function send(text) {
    text = (text || "").trim();
    if (S.busy) return toast("Sheep is finishing the current request. Stop it first if you want to cancel.");
    if (S.attach.some((a) => !a.path)) return toast("still uploading the attachment", "bad");
    const cloudReady = !!(S.state && S.state.mode === 1 && S.state.cloud && S.state.cloud.ready);
    if (!(S.state && S.state.model && S.state.model.path) && !cloudReady) {
      if (!text && !S.attach.length) return;
      S.pendingChat = { text: text || "Here is an attachment." };
      const composer = $("#askText");
      composer.value = S.pendingChat.text;
      composer.style.height = "auto";
      composer.style.height = Math.min(220, composer.scrollHeight) + "px";
      toast("Choose a local Chat model. This message will resume after it passes setup.");
      modelSlotChoice("chat");
      return;
    }
    const attached = S.attach.map((a) => a.path);
    if (!text && attached.length) text = "Here " + (attached.length > 1 ? "are some files." : "is a file.");
    if (!text) return;
    S.attach = [];
    paintAttach();
    if (!S.log.length) $("#thread").innerHTML = `<div class="col"></div>`;
    const job = S.job = (S.job || 0) + 1;
    S.busy = true;
    $("#sendBtn").hidden = true; $("#stopBtn").hidden = false;
    $$("#thread .col > [data-live]").forEach((n) => delete n.dataset.live);
    S.log.push({ role: "you", text, attached });
    paintLast();
    $$("#thread .col > [data-live]").forEach((n) => delete n.dataset.live);
    S.convo.push({ role: "user", content: text, ...(attached.length ? { attachments: attached } : {}) });
    const bot = { role: "bot", route: "local", steps: [], text: "", files: [], t0: Date.now(),
      jobId: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`, workState: "active",
      workTask: text.slice(0, 100), workStage: "Starting", workBefore: MLMI.workCount ? MLMI.workCount() : 0 };
    S.log.push(bot);
    paintLast();
    const calls = {};
    const jobCtx = { id: bot.jobId, chatId: S.chatId, chatVersion: S.chatVersion, convo: S.convo, log: S.log,
      canvas: S.canvas, bot, prompt: text, app: NATIVE[S.app] ? S.app : "", path: S.open[S.app] || "",
      started: false, cancelRequested: false, saveQueue: Promise.resolve() };
    S.activeJob = jobCtx;
    const queueSave = () => { jobCtx.saveQueue = jobCtx.saveQueue.then(() => saveChat(jobCtx)); return jobCtx.saveQueue; };
    await queueSave();
    workBadge(); refreshWork();
    if (jobCtx.cancelRequested) {
      bot.done = true; bot.stopped = true; bot.workState = "cancelled"; bot.workStage = "Stopped before the request started";
      bot.thinking = null; bot.text = "Stopped before the request started.";
      jobCtx.convo.push({ role: "assistant", content: bot.text });
      paintLast(jobCtx); release(jobCtx); await queueSave(); refreshWork();
      return;
    }
    jobCtx.started = true;
    try {
      let r;
      for (let tries = 0; ; tries++) {
        r = await fetch("/api/agent", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: jobCtx.convo, app: jobCtx.app, path: jobCtx.path }) });
        if (r.status !== 409 || tries > 240 || job !== S.job) break;
        // the last job is still finishing its current step after a Stop: wait for it
        bot.thinking = { what: "Finishing the last job first", t0: bot.thinking ? bot.thinking.t0 : Date.now() };
        bot.workStage = bot.thinking.what;
        paintLast(jobCtx); refreshWork();
        await new Promise((res) => setTimeout(res, 1000));
      }
      if (job !== S.job) return;
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        if (job !== S.job) { reader.cancel().catch(() => {}); return; }
        buf += dec.decode(value, { stream: true });
        let k;
        while ((k = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, k);
          buf = buf.slice(k + 2);
          if (chunk.startsWith("data: ")) {
            const ev = JSON.parse(chunk.slice(6));
            onEvent(ev, bot, calls, jobCtx);
            if (["tool_call", "tool_result", "file_changed", "done", "error"].includes(ev.type)) queueSave();
          }
        }
      }
    } catch (e) {
      bot.text = bot.text || `Something went wrong: ${e.message}`;
      bot.workState = "failed"; bot.workStage = "Request failed";
    }
    if (job !== S.job) return;
    bot.done = true;
    bot.thinking = null;
    if (bot.workState === "active" || bot.workState === "stopping") {
      bot.workState = bot.stopped ? "cancelled" : bot.text.startsWith("Something went wrong:") || !bot.text && !bot.files.length ? "failed" : "complete";
      bot.workStage = bot.workState === "cancelled" ? "Stopped safely" : bot.workState === "failed" ? "Failed" : "Completed";
    }
    if (!bot.text) bot.text = bot.files.length
      ? `Created ${bot.files.join(", ")}. Open the result from Work Canvas.`
      : "I could not finish that.";
    paintLast(jobCtx);
    jobCtx.convo.push({ role: "assistant", content: bot.text, files: bot.files && bot.files.length ? bot.files.slice() : undefined });
    await jobCtx.saveQueue;
    release(jobCtx);
    loadTree(); refresh(); saveChat(jobCtx);
    // made something: it lands on the Work Canvas, opened unless an editor is in use
    refreshWork();
  }

  function resumePendingChat() {
    if (!S.pendingChat || S.busy) return;
    const pending = S.pendingChat;
    S.pendingChat = null;
    const composer = $("#askText");
    composer.value = "";
    composer.style.height = "auto";
    send(pending.text);
  }
  function onEvent(ev, bot, calls, jobCtx) {
    switch (ev.type) {
      case "step": if (ev.route === "cloud") bot.route = "cloud"; bot.waiting = null; bot.thinking = null; bot.workStage = "Preparing response"; break;
      case "thinking": {
        const sameTool = ev.tool_id && ev.tool_id === bot.activeToolId;
        bot.thinking = { what: ev.what + (ev.step > 1 ? ` · step ${ev.step}` : ""),
          t0: sameTool ? (bot.toolStartedAt || Date.now()) : Date.now(),
          estimateSeconds: Number.isFinite(Number(ev.estimate_seconds)) ? Number(ev.estimate_seconds) : 0 };
        bot.workStage = bot.thinking.what; break;
      }
      case "waiting": bot.waiting = ev.stage === "ready" ? null : ev; break;
      case "route": if (ev.route === "cloud") { bot.route = "cloud"; bot.note = ev.why; } break;
      case "plan": bot.plan = { text: ev.text, steps: ev.steps }; bot.thinking = { what: "Sheep is making it", t0: Date.now() }; break;
      case "tool_call": { const s = { name: ev.name, args: ev.args || {} }; calls[ev.id] = s; bot.steps.push(s);
        bot.activeToolId = ev.id; bot.toolStartedAt = Date.now();
        bot.workTask = DOING[ev.name] || "Working"; bot.workStage = bot.workTask; bot.workProgress = null;
        bot.thinking = { what: bot.workTask,
          t0: bot.toolStartedAt, estimateSeconds: 0 };
        if (Number.isFinite(ev.progress)) bot.workProgress = ev.progress <= 1 ? ev.progress * 100 : ev.progress; break; }
      case "tool_result": { const s = calls[ev.id]; if (s) { s.ok = ev.ok; s.text = ev.text; }
        if (bot.activeToolId === ev.id) { bot.activeToolId = null; bot.toolStartedAt = null; bot.thinking = null; }
        bot.workProgress = null;
        bot.workStage = ev.ok ? `Done: ${(DOING[ev.name] || "the step").toLowerCase()}` : `Could not finish: ${(DOING[ev.name] || "a step").toLowerCase()}`; break; }
      case "tool_progress": {
        const step = Number(ev.step), total = Number(ev.total);
        const stepText = Number.isFinite(step) && Number.isFinite(total) && total > 0 ? ` · step ${step}/${total}` : "";
        const device = ev.device ? ` on ${ev.device}` : "";
        bot.workStage = `${ev.stage || "Working"}${stepText}${device}`;
        // Step counts are real progress inside the model stage. Keep the overall
        // job bar indeterminate until the tool has actually saved its output.
        bot.workProgress = null;
        bot.thinking = { what: bot.workStage, t0: bot.toolStartedAt || (bot.thinking && bot.thinking.t0) || Date.now(),
          estimateSeconds: bot.thinking ? bot.thinking.estimateSeconds : 0 };
        break;
      }
      case "confirm":
        if (ev.auto) return;
        bot.confirm = { id: ev.id, reason: ev.reason };
        break;
      case "model_required":
        bot.thinking = { what: `Waiting for Check and use (${ev.slot})`, t0: Date.now() };
        modelSlotChoice(ev.slot, { repo_id: ev.repo_id, resumeId: ev.id });
        break;
      case "file_changed": {
        const fx = ev.path.split(".").pop().toLowerCase();
        if (!bot.files.includes(ev.path) && (APP_OF[fx] || MEDIA_OF[fx])) bot.files.push(ev.path);   // pictures, video, sound too
        bot.workStage = `Output ready: ${ev.path.split("/").pop()}`;
        const ext = ev.path.split(".").pop().toLowerCase();
        if (S.app && APP_OF[ext] === S.app && !S.open[S.app] && NATIVE[S.app] === ext) setTimeout(() => openFile(ev.path), 300);
        break;
      }
      // the answer as it is written; a local device retry clears partial output before replaying it
      case "draft": if (ev.delta === null) bot.draft = ""; else bot.draft = (bot.draft || "") + (ev.delta || "");
        bot.thinking = null; bot.waiting = null; return paintSoon(jobCtx);
      case "device_retry": bot.draft = ""; bot.thinking = { what: ev.why, t0: Date.now() }; bot.workStage = ev.why; return paintSoon(jobCtx);
      case "text": if (ev.text && !ev.partial) { bot.text = ev.text; bot.draft = ""; } if (ev.route === "cloud") bot.route = "cloud"; break;
      case "error": bot.text = `Something went wrong: ${ev.text}`; bot.workState = "failed"; bot.workStage = "Failed"; break;
      case "sources": bot.sources = ev.items || []; break;
      case "setup_needed": bot.setup = ev.items || []; break;
      case "review": bot.review = { by: ev.by, ok: !!ev.ok, fixes: ev.fixes || [], fixed: !!ev.fixed }; break;
      case "done":
        if (ev.stopped) {
          bot.stopped = true; bot.workState = "cancelled"; bot.workStage = "Stopped safely";
          bot.text = ev.changed && ev.changed.length ? `Stopped after the current step. Completed: ${ev.changed.join(", ")}.` : "Stopped.";
        } else if (ev.failed) {
          // the job ended without doing what was asked: its failed steps show as errors
          bot.workState = "failed"; bot.workStage = "Not done";
        }
        break;
      default: return;
    }
    paintLast(jobCtx); refreshWork();
  }

  // ------------------------------------------------------------ live changes from the agent
  function listen() {
    const es = new EventSource("/api/events");
    let t = null;
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === "media_install") { mediaInstall(ev); return; }
      if (ev.type !== "file_changed") return;
      clearTimeout(t);
      t = setTimeout(loadTree, 300);
      const v = S.mounted;
      // File editors expose their current path with different shapes: document,
      // sheet, slide and Studio use path(), while Text & Code stores a path string.
      // Treating that string as a method throws from the SSE handler and prevents
      // previews/Canvas from updating after a Text file save.
      const mountedPath = v && (typeof v.path === "function" ? v.path()
        : typeof v.path === "string" ? v.path
          : typeof v.path_ === "function" ? v.path_() : "");
      if (mountedPath === ev.path && v && typeof v.reload === "function") v.reload(true);
      fillPreviews(ev.path);
      refreshWork();
    };
    es.onerror = () => { es.close(); setTimeout(listen, 3000); };
  }
  async function undo(redo) {
    try {
      const r = await api(redo ? "/api/redo" : "/api/undo", {});
      const e = r.undone || r.redone;
      toast(e ? `${redo ? "Redid" : "Undid"}: ${e.what || "change"}` : `Nothing to ${redo ? "redo" : "undo"}`);
      if (S.mounted && S.mounted.reload) S.mounted.reload(true);
      loadTree();
    } catch (e) { toast(e.message, "bad"); }
  }

  // ------------------------------------------------------------ model picker
  async function openSsfModel() {
    let path = "";
    if (window.mlmiDesktop) {
      path = await window.mlmiDesktop.pickFile({ filters: [{ name: "Sheep model (.ssf)", extensions: ["ssf"] }] });
    } else {
      path = await ask("Path to a Sheep .ssf model file", { ok: "Open", input: "" });
    }
    if (!path) return;
    try {
      const choice = await api(`/api/model_fit?path=${encodeURIComponent(path)}`);
      ssfChoice(choice);
    } catch (err) { toast(err.message, "bad"); }
  }

  function ssfChoice(model) {
    const fit = model.match;
    const modal = document.createElement("div");
    modal.className = "modal media-choice-overlay";
    modal.innerHTML = `<section class="sheet ask media-choice" role="dialog" aria-modal="true" aria-labelledby="ssfChoiceTitle" tabindex="-1">
      <button class="media-choice-close ghost" data-close aria-label="Close">×</button>
      <div class="caps">Sheep .ssf model · ${esc(model.arch || "language model")}</div>
      <h2 id="ssfChoiceTitle">${esc(model.name)}</h2>
      <p>Native Sheep model format. The weights stay on this computer.</p>
      <div class="model-match"><i class="match-dot ${esc(fit.level)}" aria-hidden="true"></i><div><b>${esc(fit.label)}</b>
        <small>${esc(fit.device)} · ${esc(fit.reason)}</small></div></div>
      <div class="media-choice-error" hidden role="alert"></div>
      <div class="row"><button class="primary" data-use>Check and use</button><button data-close>Not now</button></div>
    </section>`;
    document.body.appendChild(modal);
    const close = () => modal.remove();
    modal.querySelectorAll("[data-close]").forEach((b) => (b.onclick = close));
    modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
    modal.querySelector("[data-use]").onclick = async (e) => {
      if (fit.level === "red" && !await ask(`${fit.label}: ${fit.reason}\n\nTry this model anyway?`)) return;
      const button = e.currentTarget; button.disabled = true; button.textContent = "Checking and starting…";
      try {
        await api("/api/settings", { model: model.path });
        close(); toast("Sheep is starting the model and running its validation checks"); refresh();
      } catch (err) {
        const error = modal.querySelector(".media-choice-error"); error.textContent = err.message; error.hidden = false;
        button.disabled = false; button.textContent = "Check and use"; toast(err.message, "bad");
      }
    };
    modal.querySelector(".sheet").focus();
  }

  function mediaChoice(x) {
    const fit = x.match || { level: "red", label: "Not checked", device: "Local hardware", reason: "No hardware check is available." };
    const modal = document.createElement("div");
    modal.className = "modal media-choice-overlay";
    modal.innerHTML = `<section class="sheet ask media-choice" role="dialog" aria-modal="true" aria-labelledby="mediaChoiceTitle" tabindex="-1">
      <button class="media-choice-close ghost" data-close aria-label="Close">×</button>
      <div class="caps">${esc(x.kind)} model</div>
      <h2 id="mediaChoiceTitle">${esc(x.name)}</h2>
      <p>${esc(x.does)}${x.licence ? ` · ${esc(x.licence)}` : ""}</p>
      <div class="model-match"><i class="match-dot ${esc(fit.level)}" aria-hidden="true"></i><div><b>${esc(fit.label)}</b>
        <small>${esc(fit.device)} · ${esc(fit.reason)}</small></div></div>
      <p class="media-choice-state">${x.disabled ? "Disabled in Sheep. Choosing it again enables it." : x.ready ? "Installed on this computer." : x.reuse_cached ? "Already in the shared Hugging Face cache. Sheep will copy it into its own removable model folder." : x.installable ? "Not installed yet. Check and use will download it to this computer." : esc(x.why || "Required local installer is unavailable.")}</p>
      <div class="media-choice-error" hidden role="alert"></div>
      <div class="row">
        <button class="primary" data-use ${!x.ready && !x.installable ? "disabled" : ""}>${x.ready && !x.disabled ? "Check and use" : x.reuse_cached ? "Check, reuse and use" : x.install ? "Check, download and use" : "Check and enable"}</button>
        <button data-hf-slot>Enter a Hugging Face ID for this job…</button>
        ${x.removable && x.ready && !x.disabled ? `<button class="danger" data-remove>${x.install ? "Remove download" : "Disable in Sheep"}</button>` : ""}
        <button data-close>Not now</button>
      </div>
    </section>`;
    document.body.appendChild(modal);
    const close = () => modal.remove();
    modal.querySelectorAll("[data-close]").forEach((b) => (b.onclick = close));
    modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
    modal.querySelector("[data-hf-slot]").onclick = () => {
      const slot = x.id === "sd15-lcm" ? "image" : x.id === "animatediff-lightning" ? "video" :
        x.id === "ip-adapter-plus" ? "photo_guidance" : x.id === "whisper-small" ? "speech_input" : "speech_output";
      close(); modelSlotChoice(slot);
    };
    modal.querySelector("[data-use]").onclick = async (e) => {
      if (fit.level === "red" && !await ask(`${fit.label}: ${fit.reason}\n\nTry this model anyway?`)) return;
      const button = e.currentTarget, old = button.textContent;
      button.disabled = true; button.textContent = x.ready ? "Checking…" : "Downloading…";
      const error = modal.querySelector(".media-choice-error"); error.hidden = true;
      try {
        await api("/api/media_model", { id: x.id, action: "use" });
        close();
        toast(`${x.name} is ready for ${x.kind} jobs. The chat model stays unchanged.`);
        modelMenu();
      } catch (err) {
        error.textContent = err.message; error.hidden = false;
        button.disabled = !x.installable && !x.ready; button.textContent = old;
        toast(err.message, "bad");
      }
    };
    const remove = modal.querySelector("[data-remove]");
    if (remove) remove.onclick = async (e) => {
      const note = x.install ? "Remove Sheep's downloaded model files? Other installed features that share the base weights will be kept." : "Disable Windows voices inside Sheep? Windows itself will not be changed.";
      if (!await ask(note)) return;
      e.currentTarget.disabled = true;
      try {
        await api("/api/media_model", { id: x.id, action: "remove" });
        close(); modelMenu(); toast(x.install ? `${x.name} download removed` : `${x.name} disabled in Sheep`);
      } catch (err) { e.currentTarget.disabled = false; toast(err.message, "bad"); }
    };
    modal.querySelector(".sheet").focus();
  }

  // Hugging Face sign-in, folded: gated originals (Llama, Gemma, Mistral) need a read token
  // once; it stays on this machine (huggingface_hub's own token file)
  function hfSignInHtml() {
    return `<summary data-hf-sum>Gated model (Llama, Gemma, Mistral), or publishing? Sign in to Hugging Face</summary>
      <div class="inrow" data-hf-row><span class="dim">Checking…</span></div>`;
  }
  function hfSignIn(box, err) {
    const row = box.querySelector("[data-hf-row]");
    const draw = async () => {
      let who = "";
      try { who = (await api("/api/hf_login")).user || ""; } catch (e) { who = ""; }
      box.querySelector("[data-hf-sum]").textContent = who ? `Signed in to Hugging Face as ${who}`
        : "Gated model (Llama, Gemma, Mistral), or publishing? Sign in to Hugging Face";
      row.innerHTML = who
        ? `<span class="dim">Gated models open once you accept their licence on huggingface.co. Publishing needs a token of type Write.</span><button data-hf-out>Sign out</button>`
        : `<input data-hf-token type="password" autocomplete="off" spellcheck="false" aria-label="Hugging Face token" placeholder="hf_… from huggingface.co/settings/tokens (Write, to publish)">
           <button data-hf-in>Sign in</button>`;
      const tok = row.querySelector("[data-hf-token]"), inB = row.querySelector("[data-hf-in]"), outB = row.querySelector("[data-hf-out]");
      if (tok) tok.onkeydown = (e) => { if (e.key === "Enter") inB.click(); };
      if (inB) inB.onclick = async () => {
        inB.disabled = true;
        try { await api("/api/hf_login", { token: tok.value }); err(""); toast("Signed in to Hugging Face"); }
        catch (e) { err(e.message); }
        draw();
      };
      if (outB) outB.onclick = async () => { try { await api("/api/hf_login", { logout: true }); } catch (e) {} draw(); };
    };
    draw();
  }

  function modelSlotChoice(initialSlot = "chat", options = {}) {
    const slotLabels = { chat: "Chat", checker: "Checker", image: "Image / media", video: "Video",
      photo_guidance: "Photo guidance", speech_input: "Speech input", speech_output: "Speech output" };
    const modal = document.createElement("div");
    modal.className = "modal media-choice-overlay";
    modal.innerHTML = `<section class="sheet ask media-choice model-slot-choice" role="dialog" aria-modal="true" aria-labelledby="slotChoiceTitle" tabindex="-1">
      <button class="media-choice-close ghost" data-close aria-label="Close">×</button>
      <div class="caps">Models</div><h2 id="slotChoiceTitle">${initialSlot === "chat" ? "Add any model" : "Choose a model"}</h2>
      <p data-slot-intro></p>
      <label class="field" data-slot-row hidden><span class="k">Job</span><select data-slot>${Object.entries(slotLabels).map(([id, label]) => `<option value="${id}" ${id === initialSlot ? "selected" : ""}>${label}</option>`).join("")}</select></label>
      <label class="field"><span class="k">Hugging Face ID</span><input data-repo type="text" autocomplete="off" spellcheck="false" placeholder="org/model"></label>
      <div class="hf-examples" data-examples hidden><span class="dim">Try</span>${[["Qwen/Qwen3-0.6B", "Qwen3 0.6B"], ["Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B"], ["Qwen/Qwen3-8B", "Qwen3 8B"], ["bartowski/Qwen2.5-7B-Instruct-GGUF", "a GGUF page"]].map(([id, label]) => `<button type="button" class="chip" data-ex="${id}" title="${id}">${label}</button>`).join("")}</div>
      <details class="hf-signin" data-hf-signin>${hfSignInHtml()}</details>
      <label class="field" data-level-row hidden><span class="k">Size and quality</span><select data-level>
        <option value="balanced">Q4/6 Balanced</option><option value="small">Q4 Small</option><option value="q2">Q2 Tiny</option>
        <option value="q3">Q3 Compact</option><option value="q5">Q5 Fine</option><option value="high">Q6 High</option>
        <option value="q7">Q7 Very high</option><option value="q8">Q8 Near-lossless</option><option value="exact">Exact weights</option>
      </select></label>
      <div class="media-choice-error" data-slot-error hidden role="alert"></div>
      <div class="model-slot-result" data-slot-result hidden></div>
      <div class="model-slot-progress" data-slot-progress hidden></div>
      <progress class="slot-install-progress" data-progress max="100" hidden aria-label="Model download and build progress"></progress>
      <div class="row"><button data-check>Check</button><button class="primary" data-use disabled>Use</button><button data-cancel hidden>Cancel install</button><button data-stop-pending hidden>Stop request</button><button data-close>Close</button></div>
    </section>`;
    document.body.appendChild(modal);
    const sheet = modal.querySelector(".sheet"), slotInput = modal.querySelector("[data-slot]"), repoInput = modal.querySelector("[data-repo]");
    const checkButton = modal.querySelector("[data-check]"), useButton = modal.querySelector("[data-use]"), cancelButton = modal.querySelector("[data-cancel]");
    const errorBox = modal.querySelector("[data-slot-error]"), resultBox = modal.querySelector("[data-slot-result]"), progressBox = modal.querySelector("[data-slot-progress]"), installProgress = modal.querySelector("[data-progress]");
    if (options.resumeId) slotInput.disabled = true;
    repoInput.value = options.repo_id || "";
    let checked = null, pollTimer = 0, inProgress = false, resumeSent = false, pendingNotice = null;
    const stopPending = async (button) => {
      button.disabled = true; button.textContent = "Stopping…";
      try { await api("/api/stop", {}); }
      catch (e) { button.disabled = false; button.textContent = "Stop request"; err(e.message); return; }
      close(true);
    };
    const showPendingNotice = () => {
      if (pendingNotice && pendingNotice.isConnected) return;
      pendingNotice = document.createElement("div");
      pendingNotice.className = "pending-model-notice";
      pendingNotice.setAttribute("role", "status");
      pendingNotice.innerHTML = `<span>This request is waiting for a compatible model.</span><button data-reopen>Choose model</button><button data-stop-pending>Stop request</button>`;
      document.body.appendChild(pendingNotice);
      pendingNotice.querySelector("[data-reopen]").onclick = () => { pendingNotice.remove(); pendingNotice = null; modal.hidden = false; sheet.focus(); };
      pendingNotice.querySelector("[data-stop-pending]").onclick = (event) => stopPending(event.currentTarget);
    };
    const close = (force = false) => {
      if (inProgress && !force) return;
      if (options.resumeId && !force) {
        modal.hidden = true;
        showPendingNotice();
        return;
      }
      clearTimeout(pollTimer);
      if (options.resumeId && !resumeSent) {
        resumeSent = true;
        api("/api/confirm", { id: options.resumeId, allow: false }).catch(() => {});
      }
      if (pendingNotice) { pendingNotice.remove(); pendingNotice = null; }
      modal.remove();
    };
    const stopChoice = modal.querySelector("[data-stop-pending]");
    stopChoice.hidden = !options.resumeId;
    stopChoice.onclick = () => stopPending(stopChoice);
    modal.querySelectorAll("[data-close]").forEach((button) => (button.onclick = () => close(false)));
    modal.addEventListener("click", (event) => { if (event.target === modal && !inProgress) close(); });
    const err = (text) => { errorBox.textContent = text; errorBox.hidden = !text; };
    const resetCheck = () => {
      checked = null; useButton.disabled = true; useButton.textContent = "Use"; resultBox.hidden = true; err("");
      const levelRow = modal.querySelector("[data-level-row]"); if (levelRow) levelRow.hidden = true;
    };
    // gated originals (Llama, Gemma, Mistral) need a read token once; it stays on this machine
    const hfBox = modal.querySelector("[data-hf-signin]");
    hfSignIn(hfBox, err);
    const setInstallProgress = (value, label = "") => {
      installProgress.hidden = false;
      if (Number.isFinite(value)) {
        const percent = Math.max(0, Math.min(100, value));
        installProgress.value = percent;
        installProgress.setAttribute("aria-valuetext", label || `${Math.round(percent)}%`);
      } else {
        installProgress.removeAttribute("value");
        installProgress.setAttribute("aria-valuetext", label || "Progress not reported");
      }
    };
    const slotHint = () => {
      const text = { chat: "Text generation · Sheep quantizes compatible architectures to .ssf.",
        checker: "Text generation · Sheep quantizes, examines, and activates only a passing Checker.",
        image: "Text-to-image · only a pipeline supported by Sheep's local image worker is accepted.",
        video: "Text-to-video · only a motion model supported by Sheep's local video worker is accepted.",
        photo_guidance: "Image-conditioned video · only adapters compatible with Sheep's video worker are accepted.",
        speech_input: "Speech recognition · only a model supported by Sheep's local transcriber is accepted.",
        speech_output: "Text-to-speech · this build uses Windows voices and has no Hugging Face TTS runtime." }[slotInput.value];
      modal.querySelector("#slotChoiceTitle").textContent = `${slotLabels[slotInput.value]} model`;
      const textSlot = ["chat", "checker"].includes(slotInput.value);
      modal.querySelector("[data-level-row]").hidden = true;          // shown once a check passes
      modal.querySelector("[data-examples]").hidden = !textSlot || !!repoInput.value;
      modal.querySelector("[data-hf-signin]").hidden = !textSlot;
      modal.querySelector("[data-slot-intro]").textContent = textSlot
        ? "Paste any Hugging Face model ID: the original weights, or a GGUF, GPTQ, AWQ or MLX copy (Sheep finds the original it was made from). Check only reads the page; nothing downloads until you choose."
        : "Enter a Hugging Face repository ID. Check reads repository metadata and runtime fit only; weights download after you choose.";
      modal.querySelector("[data-repo]").disabled = slotInput.value === "speech_output";
      checkButton.disabled = slotInput.value === "speech_output";
      modal.querySelector("[data-repo]").placeholder = slotInput.value === "speech_output" ? "No HF runtime in this build"
        : textSlot ? "org/model, e.g. Qwen/Qwen2.5-3B-Instruct" : "org/model";
      modal.querySelector("[data-slot-result]").hidden = true;
      modal.querySelector("[data-slot-progress]").textContent = text;
      modal.querySelector("[data-slot-progress]").hidden = textSlot;    // the intro says it for text models
      resetCheck();
    };
    slotInput.onchange = slotHint;
    repoInput.oninput = () => {
      resetCheck();
      modal.querySelector("[data-examples]").hidden = !!repoInput.value || !["chat", "checker"].includes(slotInput.value);
    };
    repoInput.onkeydown = (e) => { if (e.key === "Enter" && !checkButton.disabled) checkButton.click(); };
    modal.querySelector("[data-examples]").onclick = (e) => {
      const b = e.target.closest("[data-ex]");
      if (!b) return;
      repoInput.value = b.dataset.ex; repoInput.oninput(); checkButton.click();
    };
    slotHint();
    checkButton.onclick = async () => {
      const slot = slotInput.value, repo_id = repoInput.value.trim();
      if (!repo_id) return err("Enter a Hugging Face ID such as org/model.");
      checkButton.disabled = true; checkButton.textContent = "Checking…"; useButton.disabled = true;
      resultBox.hidden = true; err("");
      try {
        checked = await api("/api/model_check", { slot, repo_id });
        const fit = checked.fit || { level: "red", label: "Not checked", reason: "Fit was not available." };
        const gb = (Number(checked.download_bytes || 0) / 1e9).toFixed(1);
        resultBox.innerHTML = `${checked.original_of ? `<div class="slot-original"><span><span class="dim">${esc(checked.original_of)}</span> is a copy · building from the original
            <b>${esc(checked.repo_id)}</b></span></div>` : ""}
          <div class="model-match"><i class="match-dot ${esc(fit.level)}" aria-hidden="true"></i><div><b>${esc(fit.label || "Fit")}</b><small>${esc(fit.reason || "")}</small></div></div>
          <div class="kvx"><span class="k">kind</span><span>${esc(checked.architecture || "not said on its page")}</span>
          <span class="k">licence</span><span>${esc(checked.license || "unknown; review the model card")}</span>
          <span class="k">download</span><span>${gb} GB</span>
          <span class="k">check</span><span>${esc(checked.reason || "")}</span></div>
          <details class="slot-more"><summary>Details</summary><div class="kvx"><span class="k">task</span><span>${esc(checked.pipeline_tag || slotLabels[slot])}</span>
          <span class="k">revision</span><span>${esc((checked.revision || "").slice(0, 12))}</span></div></details>`;
        // a check that never reached the model (gated, missing, not a checkpoint) is its error alone
        resultBox.hidden = !checked.compatible && !checked.architecture && !checked.original_of;
        if (!inProgress) progressBox.hidden = true;               // the slot hint repeated the result
        if (checked.compatible) {
          if (["chat", "checker"].includes(slot)) {
            const recommended = checked.recommendation?.quant_level;
            const sel = modal.querySelector("[data-level]");
            if (recommended) sel.value = recommended;
            [...sel.options].forEach((o) => { o.textContent = o.textContent.replace(/ · recommended$/, "") + (o.value === recommended ? " · recommended" : ""); });
            modal.querySelector("[data-level-row]").hidden = !!checked.compat_needed;   // the maker's code: no Sheep build
            useButton.textContent = checked.compat_needed ? (Number(gb) > 0 ? `Download ${gb} GB and use (compatibility mode)` : "Use (compatibility mode)")
              : checked.fit_needed ? (Number(gb) > 0 ? `Download ${gb} GB, fit and build` : "Fit, build and use")
              : Number(gb) > 0 ? `Download ${gb} GB and build` : "Build and use";
          } else useButton.textContent = Number(gb) > 0 ? `Download ${gb} GB and use` : "Use";
          useButton.disabled = false;
        } else {
          err(checked.reason || "This repository is incompatible with the selected feature slot.");
          if (/gated|sign in/i.test(checked.reason || "")) hfBox.open = true;
        }
      } catch (e) { err(e.message); }
      finally { checkButton.disabled = false; checkButton.textContent = "Check"; }
    };
    const poll = async () => {
      if (!modal.isConnected) return;
      try {
        const job = await api("/api/model_slot_job", {});
        const q = job.quantize || {}, checker = job.checker || {}, trial = checker.trial || {};
        const active = q.state === "fetching" || q.state === "converting" || q.state === "fitting" || q.state === "building" || trial.state === "running";
        const percent = q.progress == null ? "" : ` · ${Math.round(q.progress * 100)}%`;
        const transfer = q.state === "fetching" && (q.log || []).slice().reverse().find(line => /downloaded\s+[\d.]+\s+of\s+[\d.]+\s+GB/i.test(line));
        const transferMatch = transfer && transfer.match(/downloaded\s+([\d.]+)\s+of\s+([\d.]+)\s+GB/i);
        const transferDone = transferMatch ? Number(transferMatch[1]) : NaN;
        const transferTotal = transferMatch ? Number(transferMatch[2]) : NaN;
        const transferPct = transferTotal > 0 ? transferDone / transferTotal * 100 : null;
        progressBox.textContent = q.state === "fetching" ? `Downloading the model${transfer ? ` · ${transferDone.toFixed(2)} of ${transferTotal.toFixed(2)} GB (${Math.round(transferPct)}%)` : percent}`
          : q.state === "converting" ? "Converting it into a form Sheep can build from"
          : q.state === "fitting" ? "Fitting Sheep's engine to this kind of model"
          : q.state === "building" ? `Building it for this computer · part ${q.layer || 0} of ${q.n_layer || "?"}${percent}`
            : trial.state === "running" ? "Running the Checker exam; the previous Checker stays active until the candidate passes."
              : trial.state === "rolled back" ? `Checker candidate did not pass and was not activated: ${trial.why || "exam failed"}`
                : trial.state === "installed" ? "Checker exam passed. The new Checker is active."
                  : q.state === "failed" ? `Build failed. ${q.log?.slice(-2).join(" · ") || "Retry after resolving the cause."}`
                    : q.state === "done" ? "Built. Sheep is checking its answers and app tasks." : "Working…";
        // the same steps as the Models page, live: the moment a person sees what Sheep does for them
        const steps = slotInput.value === "chat" && MLMI.addSteps ? MLMI.addSteps(q) : "";
        if (steps) progressBox.insertAdjacentHTML("afterbegin", `<ol class="story">${steps}</ol>`);
        cancelButton.hidden = !active; cancelButton.disabled = !active;
        checkButton.hidden = inProgress; useButton.hidden = inProgress;
        if (slotInput.value === "chat" && q.state === "done") {
          const status = job.orchestrator || {};
          const expected = q.result && q.result.path;
          const selected = job.model && job.model.path === expected;
          const ready = status.lamp === "green" && (status.stage === "in sync" || /\bready\b/i.test(status.stage || ""));
          const fixedDefaultsReady = !status.on && job.engine_up && selected;
          if (selected && (ready || fixedDefaultsReady)) {
            progressBox.textContent = `Model is active. ${status.plan?.device ? `Running on ${status.plan.device}.` : ""}`;
            inProgress = false; cancelButton.hidden = true; checkButton.hidden = false; useButton.hidden = false;
          } else if (status.lamp === "red") {
            progressBox.textContent = `${status.stage || "Model validation failed"}. The previous selection is being restored.`;
          }
        }
        if (trial.state === "installed" || trial.state === "rolled back" || q.state === "failed") inProgress = false;
        if (active) {
          if (q.state === "fetching") setInstallProgress(transferPct, transfer ? `${transferDone.toFixed(2)} of ${transferTotal.toFixed(2)} GB downloaded` : "Downloading checkpoint");
          else if (q.state === "building" && Number.isFinite(Number(q.progress))) setInstallProgress(Number(q.progress) * 100, `Building part ${q.layer || 0} of ${q.n_layer || "?"}`);
          else if (q.state === "converting") setInstallProgress(null, "Converting it into a form Sheep can build from");
          else if (q.state === "fitting") setInstallProgress(null, "Fitting Sheep's engine to this kind of model");
          else setInstallProgress(null, "Checking the candidate");
        } else if (inProgress) setInstallProgress(null, "Loading and checking the installed model");
        else installProgress.hidden = true;
        if (!inProgress) { cancelButton.hidden = true; checkButton.hidden = false; useButton.hidden = false; }
        if (active || inProgress) pollTimer = setTimeout(poll, 1500);
      } catch (_) { if (inProgress) pollTimer = setTimeout(poll, 2000); }
    };
    useButton.onclick = async () => {
      if (!checked?.compatible) return;
      const slot = slotInput.value, fit = checked.fit || {};
      if (fit.level === "red" && !await ask(`${fit.label || "Poor fit"}: ${fit.reason || "This may run slowly or fail."}\n\nTry it anyway?`)) return;
      // a model Sheep may have to run with the maker's own code: that code needs a yes
      const remote = !!(checked.remote_code && (checked.fit_needed || checked.compat_needed));
      if (remote && !await ask("This model runs its own code from Hugging Face. Allow?", { ok: "Allow" })) return;
      useButton.disabled = true; checkButton.disabled = true; err("");
      if (!["chat", "checker"].includes(slot)) {
        inProgress = true; progressBox.hidden = false; progressBox.textContent = "Starting the confirmed local install…";
        cancelButton.hidden = false; cancelButton.disabled = false;
        checkButton.hidden = true; useButton.hidden = true;
      }
      try {
        const result = await api("/api/model_use", { slot, repo_id: checked.repo_id,
          quant_level: modal.querySelector("[data-level]").value, allow_remote_code: remote });
        if (["chat", "checker"].includes(slot)) {
          inProgress = true; progressBox.hidden = false; progressBox.textContent = result.message || "Downloading and building locally…";
          setInstallProgress(null, "Preparing the Sheep model");
          cancelButton.hidden = false; checkButton.hidden = true; useButton.hidden = true; poll();
        } else {
          inProgress = true;
          progressBox.hidden = false; progressBox.textContent = "Checking and installing this model locally…";
          cancelButton.hidden = false; cancelButton.disabled = false;
          checkButton.hidden = true; useButton.hidden = true;
          if (options.resumeId) {
            const resumed = await api("/api/confirm", { id: options.resumeId, allow: true });
            if (!resumed.ok) throw new Error("The original request has already ended; the model is ready in Settings.");
            resumeSent = true;
          }
          inProgress = false;
          close(true); toast(`${slotLabels[slot]} model is ready on this computer.`); modelMenu();
        }
      } catch (e) {
        inProgress = false; cancelButton.hidden = true;
        checkButton.hidden = false; useButton.hidden = false;
        err(e.message); useButton.disabled = false; checkButton.disabled = false;
      }
    };
    cancelButton.onclick = async () => {
      cancelButton.disabled = true;
      try { await api("/api/model_slot_cancel", { slot: slotInput.value }); progressBox.textContent = "Stopping at the next safe download boundary…"; }
      catch (e) { err(e.message); cancelButton.disabled = false; }
    };
    sheet.focus();
  }

  async function modelMenu() {
    const m = $("#modelMenu");
    if (m.classList.contains("open")) return m.classList.remove("open");
    const r = await api("/api/models").catch(() => ({ found: [], media: [] }));
    const cur = S.state && S.state.model.path;
    // ONE drop-down (Ernest, 2026-10-06): the chat model talks and plans; picture,
    // sound and video models are its tools, chosen here as the default for their job
    const badge = (k) => `<span class="kb ${k}">${{ text: "Text", code: "Code", image: "Image", video: "Video", audio: "Audio" }[k] || k}</span>`;
    const media = r.media || [];
    const group = (title, items) => items.length ? `<div class="mg">${title}</div>` + items.join("") : "";
    // Only models that are here are listed by name; a kind with none says "No … model"
    // and leads to Settings › Models › Feature model slots, where models are added
    // (one door for adding, not two; Ernest, 2026-10-13)
    const mediaRow = (x) => `<button class="mi ${x.disabled ? "off" : ""}" data-media="${esc(x.id)}">
        <span style="width:18px">${x.default ? icon("check") : ""}</span>
        <span><b>${esc(x.name)} ${badge(x.kind)}</b><small>${esc(x.disabled ? "turned off" : x.does)}</small></span></button>`;
    const KINDS = [["image", "picture"], ["video", "video"], ["audio", "sound"]];
    const mediaRows = KINDS.map(([k, word]) => {
      const here = media.filter((x) => x.kind === k && x.ready);
      return here.length ? here.map(mediaRow).join("")
        : `<button class="mi off" data-noslot="${k}"><span style="width:18px"></span><span><b>No ${word} model ${badge(k)}</b><small>add one in Settings › Models</small></span></button>`;
    }).join("");
    m.innerHTML = group("Chat", r.found.map((f) => `<button class="mi" data-model="${esc(f.path)}" data-name="${esc(prettyModel(f.name))}"><span style="width:18px">${f.path === cur ? icon("check") : ""}</span>
      <span><b>${esc(prettyModel(f.name))} ${badge(f.kind || "text")}</b><small>${f.mb} MB${f.licence ? " · " + esc(f.licence) : ""}</small></span></button>`))
      + (r.found.length ? "" : `<button class="mi off" data-noslot="chat"><span style="width:18px"></span><span><b>No chat model ${badge("text")}</b><small>add one to start</small></span></button>`)
      + `<div class="mg">Used as tools by the chat model</div>`
      + mediaRows
      + `<div class="sep"></div><button class="mi" data-slot-choice="chat"><span class="mi-ic">${icon("plus")}</span><span><b>Add a model…</b><small>any Hugging Face model: Sheep checks it fits before it downloads</small></span></button>
       <button class="mi" data-open-ssf><span class="mi-ic">${icon("folder")}</span><span><b>Open a Sheep model file…</b><small>a model already on this computer</small></span></button>
       <button class="mi" data-go="models"><span class="mi-ic">${icon("sheep")}</span><span><b>Manage models</b><small>Settings › Models</small></span></button>
       <button class="mi" data-go="activity"><span class="mi-ic">${icon("gauge")}</span><span><b>How this model runs</b><small>Settings › Activity</small></span></button>`;
    m.classList.add("open");
  }

  function start() {
    window.MLMI_ICON.paint();
    $("#newChat").onclick = newChat;
    $("#closeCanvas").onclick = closeCanvas;
    // the splitter: drag to set the chat's width (kept for next time), double-click to reset
    const split = $("#split"), bar = $("#splitter");
    try { const w = localStorage.getItem("sheep.chatw"); if (w) split.style.setProperty("--chatw", w); } catch (_) { /* a convenience only */ }
    bar.onpointerdown = (e) => {
      e.preventDefault();
      bar.setPointerCapture(e.pointerId);
      split.classList.add("dragging");
      const left = split.getBoundingClientRect().left, total = split.getBoundingClientRect().width;
      bar.onpointermove = (m) => {
        const w = Math.max(300, Math.min(total - 420, m.clientX - left));
        split.style.setProperty("--chatw", w + "px");
      };
      bar.onpointerup = () => {
        bar.onpointermove = bar.onpointerup = null;
        split.classList.remove("dragging");
        try { localStorage.setItem("sheep.chatw", split.style.getPropertyValue("--chatw")); } catch (_) { /* fine */ }
      };
    };
    bar.ondblclick = () => {
      split.style.removeProperty("--chatw");
      try { localStorage.removeItem("sheep.chatw"); } catch (_) { /* fine */ }
    };
    // focus: the app takes the whole width; the same button brings the chat back
    $("#focusCanvas").onclick = () => $("#focusCanvas").classList.toggle("on", split.classList.toggle("focus"));
    $("#workBtn").onclick = () => (S.app === "work" ? closeCanvas() : openApp("work"));
    $("#toWork").onclick = () => openApp("work");
    $$("#side [data-app]").forEach((b) => (b.onclick = () => (S.app === b.dataset.app ? closeCanvas() : openApp(b.dataset.app))));
    $("#collapse").onclick = () => $("#shell").classList.add("noside");
    $("#expand").onclick = () => $("#shell").classList.remove("noside");
    $("#openSettings").onclick = () => openSettings("basics");
    $("#closeSettings").onclick = closeSettings;
    $("#settings").onclick = (e) => { if (e.target.id === "settings") closeSettings(); };
    $$("#setTabs [data-tab]").forEach((b) => (b.onclick = () => openSettings(b.dataset.tab)));
    $("#status").onclick = async () => {
      if (!S.state) {
        await refresh();
        if (!S.state) { toast("Sheep’s local service is still unavailable. Close and reopen Sheep, then try again.", "bad"); return; }
      }
      openSettings(!S.state.model.path ? "models" : "dashboard");
    };
    $("#modelBtn").onclick = (e) => { e.stopPropagation(); modelMenu(); };
    $("#mediaInstallDismiss").onclick = () => {
      const job = $("#mediaInstall").dataset.job;
      try { localStorage.setItem("sheep.mediaInstall.dismissed", job || ""); } catch (_) { /* optional */ }
      $("#mediaInstall").hidden = true;
    };
    $("#mediaInstallCancel").onclick = async () => {
      $("#mediaInstallCancel").disabled = true;
      try { await api("/api/model_slot_cancel", { slot: "media" }); }
      catch (e) { toast(e.message, "bad"); }
      finally { $("#mediaInstallCancel").disabled = false; }
    };
    // a model that is here: Use or Remove, nothing more to read through
    const modelActions = (title, isCurrent, onUse, onRemove) => {
      const m = $("#modelMenu");
      m.innerHTML = `<div class="mg">${esc(title)}</div>
        ${isCurrent ? `<div class="mi off"><span style="width:18px">${icon("check")}</span><span><b>In use</b></span></div>`
          : `<button class="mi" data-act="use"><span style="width:18px">${icon("check")}</span><span><b>Use this model</b></span></button>`}
        <button class="mi" data-act="remove"><span style="width:18px">${icon("trash")}</span><span><b>Remove</b><small>from Sheep and this computer</small></span></button>
        <button class="mi" data-act="back"><span style="width:18px">${icon("chev")}</span><span><b>Back</b></span></button>`;
      m.classList.add("open");
      m.onpick = async (act) => {
        m.classList.remove("open"); m.onpick = null;
        if (act === "back") return modelMenu();
        try {
          if (act === "use") await onUse();
          if (act === "remove") {
            if (!await ask(`Remove ${title}? Its files are deleted from Sheep's model folder.`)) return;
            await onRemove();
            toast(`${title} removed`);
          }
        } catch (err) { toast(err.message, "bad"); }
        refresh();
      };
    };
    // no model of a kind: straight to where models are added
    MLMI.openSlots = () => {
      openSettings("models");
      setTimeout(() => { const s = $("#mSlots"); if (s) { s.scrollIntoView({ block: "start" }); s.classList.add("flash"); } }, 350);
    };
    $("#modelMenu").onclick = async (e) => {
      const act = e.target.closest("[data-act]");
      if (act && $("#modelMenu").onpick) { e.stopPropagation(); return $("#modelMenu").onpick(act.dataset.act); }
      const b = e.target.closest("[data-model]"), g = e.target.closest("[data-go]"), md = e.target.closest("[data-media]"), pick = e.target.closest("[data-open-ssf]"), slot = e.target.closest("[data-slot-choice]");
      const none = e.target.closest("[data-noslot]");
      if (b || md) e.stopPropagation();                  // the menu turns into the model's actions
      $("#modelMenu").classList.remove("open");
      if (none) return none.dataset.noslot === "chat" ? modelSlotChoice("chat") : MLMI.openSlots();
      if (g) return openSettings(g.dataset.go);
      if (slot) return modelSlotChoice(slot.dataset.slotChoice || "chat");
      if (pick) return openSsfModel();
      if (md) {
        const catalog = await api("/api/models").catch(() => ({ media: [] }));
        const x = (catalog.media || []).find((y) => y.id === md.dataset.media);
        if (!x) return;
        return modelActions(x.name, x.default && !x.disabled,
          () => api("/api/media_model", { id: x.id, action: "use" }),
          () => api("/api/media_model", { id: x.id, action: "remove" }));
      }
      if (b) {
        const cur = S.state && S.state.model.path;
        return modelActions(b.dataset.name || "Model", b.dataset.model === cur,
          async () => {
            // the fit check stays, but only when the model may not suit this computer
            const fit = await api(`/api/model_fit?path=${encodeURIComponent(b.dataset.model)}`).catch(() => null);
            if (fit && fit.level === "red" && !await ask(`${fit.label}: ${fit.reason}\n\nUse it anyway?`)) return;
            await api("/api/settings", { model: b.dataset.model });
            toast("Switching: Sheep plans it for this computer and checks it");
          },
          () => api("/api/model_remove", { path: b.dataset.model }));
      }
    };
    document.addEventListener("click", (e) => { if (!e.target.closest("#modelPick")) $("#modelMenu").classList.remove("open"); });
    $("#chats").onclick = async (e) => {
      const d = e.target.closest("[data-del]");
      if (d) { e.stopPropagation(); return deleteChat(d.dataset.del); }
      const rn = e.target.closest("[data-ren]");
      if (rn) { e.stopPropagation(); return renameChat(rn.closest("[data-chat]"), rn.dataset.ren); }
      const c = e.target.closest("[data-chat]");
      if (c) openChat(c.dataset.chat);
    };
    let searchTimer = 0;
    $("#chatSearch").oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadChats, 200); };
    $("#chatSearch").onkeydown = (e) => { if (e.key === "Escape" && e.target.value) { e.stopPropagation(); e.target.value = ""; loadChats(); } };
    $("#thread").onclick = (e) => {
      const cm = e.target.closest("[data-copymsg]");
      if (cm) { navigator.clipboard.writeText(cm.closest(".body").querySelector(".text").innerText); toast("Copied"); return; }
      const dr = e.target.closest("[data-draft]");
      if (dr) {
        // each log entry is one .msg in the thread, in order
        const i = [...$$("#thread .col > .msg")].indexOf(dr.closest(".msg"));
        const entry = S.log[i];
        if (entry && entry.text) MLMI.openDraft(entry.text);
        return;
      }
      const st = e.target.closest("[data-showtext]");
      if (st) {
        const full = st.closest(".reply-doc").querySelector(".reply-full");
        full.hidden = !full.hidden;
        st.lastChild.textContent = full.hidden ? "Show text" : "Hide text";
        return;
      }
      if (e.target.closest("[data-docreply]")) {
        const i = [...$$("#thread .col > .msg")].indexOf(e.target.closest(".msg"));
        if (S.log[i] && S.log[i].text) MLMI.openDraft(S.log[i].text);
        return;
      }
      const sp = e.target.closest("[data-speak]");
      if (sp) { MLMI.voice && MLMI.voice.toggleSay(sp.closest(".body").querySelector(".text").innerText); return; }
      const cp = e.target.closest("[data-copycode]");
      if (cp) { navigator.clipboard.writeText(cp.closest("pre").querySelector("code").innerText); toast("Copied"); return; }
      const q = e.target.closest("[data-q]");
      if (q) { $("#askText").value = q.dataset.q; grow(); $("#askText").focus(); return; }
      if (e.target.closest("[data-work]")) return openApp("work");
      const ex = e.target.closest("[data-export]");
      if (ex) return MLMI.exportBtn && MLMI.exportBtn(ex.dataset.export);
      if (e.target.closest("[data-vtip]")) { e.preventDefault(); return MLMI.videoTip && MLMI.videoTip(); }
      const f = e.target.closest("[data-open]");
      if (f) { e.preventDefault(); return openFile(f.dataset.open); }
      const fix = e.target.closest("[data-setup]");
      if (fix) return setupAction(JSON.parse(fix.dataset.setup), fix);
      if (e.target.closest("[data-retry]")) {
        const last = [...S.log].reverse().find((m) => m.role === "you");
        if (last && !S.busy) send(last.text);
        return;
      }
      const a = e.target.closest("[data-allow]");
      if (a) {
        const bot = S.log[S.log.length - 1];
        if (bot && bot.confirm) {
          api("/api/confirm", { id: bot.confirm.id, allow: a.dataset.allow === "1" });
          bot.confirm.done = a.dataset.allow === "1" ? "Allowed" : "Declined";
          paintLast();
        }
      }
    };
    $$("#modeSeg button").forEach((b) => (b.onclick = async () => {
      // it warned and then switched anyway, leaving a mode lit that could not answer
      if (Number(b.dataset.mode) === 1 && S.state && !S.state.cloud.ready) {
        toast("Sheep + Nebius needs a Nebius key. Paste it in Settings › Accounts; until then everything stays on this computer.", "bad");
        return openSettings("accounts");
      }
      await api("/api/settings", { mode: Number(b.dataset.mode) }).catch((e) => toast(e.message, "bad"));
      refresh();
    }));
    const ta = $("#askText");
    const grow = () => { ta.style.height = "auto"; ta.style.height = Math.min(220, ta.scrollHeight) + "px"; };
    ta.oninput = grow;
    ta.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } };
    $("#composer").onsubmit = (e) => { e.preventDefault(); const v = ta.value; ta.value = ""; grow(); send(v); };
    $("#attachBtn").onclick = () => $("#attachInput").click();
    $("#attachInput").onchange = (e) => { attachFiles([...e.target.files]); e.target.value = ""; };
    $("#attachRow").onclick = (e) => {
      const b = e.target.closest("[data-unattach]");
      if (b) { S.attach.splice(+b.dataset.unattach, 1); paintAttach(); }
    };
    // drop files anywhere on the message box
    const box = $("#composer");
    box.addEventListener("dragover", (e) => { if (e.dataTransfer && [...e.dataTransfer.types].includes("Files")) { e.preventDefault(); box.classList.add("drop"); } });
    box.addEventListener("dragleave", () => box.classList.remove("drop"));
    box.addEventListener("drop", (e) => { e.preventDefault(); box.classList.remove("drop"); if (e.dataTransfer.files.length) attachFiles([...e.dataTransfer.files]); });
    // a pasted screenshot or copied file becomes an attachment; pasted text stays text
    ta.addEventListener("paste", (e) => {
      const files = [...((e.clipboardData && e.clipboardData.files) || [])];
      if (!files.length) return;
      e.preventDefault();
      attachFiles(files.map((f, i) => (f.name && f.name !== "image.png") ? f
        : new File([f], `pasted-${Date.now()}-${i}.${(f.type.split("/")[1] || "png")}`, { type: f.type })));
    });
    $("#stopBtn").onclick = async () => {
      const job = S.activeJob;
      if (!job || job.bot.done || job.bot.workState === "stopping") return;
      job.bot.workState = "stopping";
      job.bot.workStage = "Waiting for the current safe boundary";
      job.bot.thinking = { what: "Stopping safely", t0: Date.now() };
      $("#stopBtn").disabled = true;
      $("#stopBtn").title = "Stopping safely";
      paintLast(job); refreshWork();
      if (!job.started) { job.cancelRequested = true; return; }
      try {
        const result = await api("/api/stop", {});
        if (result.active_action) toast(`Stopping after ${result.active_action.replaceAll("_", " ")} reaches a safe boundary.`);
      } catch (err) {
        job.bot.workState = "active"; job.bot.workStage = "Stop request failed";
        job.bot.thinking = null;
        $("#stopBtn").disabled = false; $("#stopBtn").title = "Stop";
        toast(`Could not stop Sheep: ${err.message}`, "bad");
        paintLast(job); refreshWork();
      }
    };
    // the seconds on a thinking line count up
    setInterval(() => $$(".tsec[data-t0]").forEach((n) => {
      n.textContent = elapsedText(n.dataset.t0, n.dataset.estimate);
    }), 1000);
    // Stop a load in progress: a big model on a slow disk is minutes, and the
    // person who picked the wrong one had no way out but to quit Sheep.
    $("#rbCancel").onclick = async () => {
      $("#rbCancel").disabled = true;
      try { await api("/api/orchestrator", { action: "cancel" }); } finally { $("#rbCancel").disabled = false; }
      refresh();
    };
    $("#rbToggle").onclick = () => setReadinessExpanded($("#rbDetails").hidden);
    $("#rbChoose").onclick = () => openSettings("models");
    $("#rbClose").onclick = () => { try { localStorage.setItem("sheep.readiness.closed", S.readinessFp || ""); } catch (_) { /* storage off */ } $("#readyBar").hidden = true; };
    $("#rbChecks").onclick = (e) => {
      const row = e.target.closest("[data-readiness-check]");
      if (row) openSettings("dashboard", "chk-" + row.dataset.readinessCheck);
    };
    $("#undoBtn").onclick = () => undo(false);
    $("#redoBtn").onclick = () => undo(true);
    // a hidden or minimised window does not animate
    document.addEventListener("visibilitychange", () => document.body.classList.toggle("window-hidden", document.hidden));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (MLMI.voice && MLMI.voice.talking && MLMI.voice.talking()) { MLMI.voice.stopTalk(); return; }
        if (window.speechSynthesis && speechSynthesis.speaking) { speechSynthesis.cancel(); return; }
        const overlay = [...document.querySelectorAll(".media-choice-overlay")].find((item) => !item.hidden);
        if (overlay) { overlay.querySelector("[data-close]")?.click(); return; }
        const tip = $("#vtip"), guide = $("#guide"), menu = $("#modelMenu");
        if (tip && !tip.hidden) { MLMI.closeVideoTip && MLMI.closeVideoTip(); return; }
        if (guide && !guide.hidden) { MLMI.closeGuide && MLMI.closeGuide(); return; }
        if (!$("#settings").hidden) { closeSettings(); return; }
        if (menu.classList.contains("open")) { menu.classList.remove("open"); return; }
        // a popup inside an app, or a slide show that just ended, takes this Escape: not the whole canvas
        const pops = document.querySelectorAll("#canvas .st-pop.open, #canvas .menu.open");
        if (pops.length) { pops.forEach((p) => p.classList.remove("open")); return; }
        if (document.querySelector(".present") || Date.now() - (MLMI.showEnded || 0) < 800) return;
        if (S.app) closeCanvas();
      }
      const typing = e.target.closest("input, textarea, [contenteditable=true]");
      if (!typing && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(e.shiftKey); }
      if (!typing && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); undo(true); }
    });
    renderThread();
    refresh(); loadTree(); loadMediaInstall(); listen();
    // after a backend restart the window reloads: reopen the chat the person was in
    let open = null;
    try { open = sessionStorage.getItem("sheep.openChat"); } catch (_) { /* storage may be off */ }
    if (open) openChat(open); else loadChats();
    setInterval(() => { refresh(); loadMediaInstall(); }, 3000);
    api("/api/warm", { app: "" }).catch(() => {});
    // #words, #dashboard ... open a page directly (the screenshot check uses it)
    const fromHash = () => { const h = location.hash.slice(1); if (h) go(h); };
    window.addEventListener("hashchange", fromHash);
    fromHash();
    ta.focus();
  }

  return { $, $$, esc, api, op, toast, ask, S, go, themeOf, setupAction, setupRows, modelName: prettyModel, openFile, openApp, closeCanvas, openSettings, loadTree, refresh, start, NATIVE, send, icon, saveChat, attachFiles, modelSlotChoice, hfSignIn, hfSignInHtml };
})();
