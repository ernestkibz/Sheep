// Sheep desktop: one window over the local backend.
//
// Electron is only the frame. It starts the backend (python -m mlmi.backend.server),
// which starts the SHEEP engine; the window then loads the interface the backend
// serves on 127.0.0.1. Closing the window hides it: Sheep sleeps in the tray with
// the model loaded, so opening it again is instant (Ernest, 2026-10-06). Quit in
// the tray menu stops both. Nothing listens beyond this machine.

const { app, BrowserWindow, Menu, Tray, dialog, ipcMain, nativeImage, screen, shell } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

const REPO = path.resolve(__dirname, "..", "..");            // the sheep repo
const PORT = Number(process.env.MLMI_PORT || 8080);
const PYTHON = process.env.MLMI_PYTHON || (process.platform === "win32" ? "python" : "python3");
const APP_ENTRY = path.resolve(__dirname);

// The app was called ML.M.I until 2026-10-21; an existing install keeps its saved settings there.
{
  const fs = require("fs");
  const old = path.join(app.getPath("appData"), "ML.M.I");
  if (fs.existsSync(old)) app.setPath("userData", old);
  // a run on its own home (tests, the photo audit) is its own Sheep: its own window data and
  // instance lock, so it opens beside the person's Sheep instead of handing over to it
  if (process.env.MLMI_HOME) app.setPath("userData", path.join(process.env.MLMI_HOME, "window"));
}

let backend = null;
let win = null;
let tray = null;

function healthy() {
  return new Promise((resolve) => {
    const req = http.get({ host: "127.0.0.1", port: PORT, path: "/health", timeout: 1500 }, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.on("timeout", () => { req.destroy(); resolve(false); });
  });
}

let lastLines = [];                                 // the backend's last words, for a plain failure message
let restarts = [];

async function startBackend() {
  if (await healthy()) return true;               // one already running (development)
  const args = ["-m", "mlmi.backend.server", "--port", String(PORT)];
  backend = spawn(PYTHON, args, { cwd: REPO, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
  const keep = (d) => { lastLines = lastLines.concat(String(d).split(/\r?\n/).filter((l) => l.trim())).slice(-6); };
  backend.stdout.on("data", (d) => { keep(d); process.stdout.write(`[backend] ${d}`); });
  backend.stderr.on("data", (d) => { keep(d); process.stderr.write(`[backend] ${d}`); });
  backend.on("exit", (code) => { backend = null; if (win && !app.isQuitting) backendDied(code); });
  for (let i = 0; i < 240; i++) {                   // the engine may be loading a model
    if (await healthy()) return true;
    if (!backend) return false;
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

// The backend died while Sheep is open: start it again (up to 3 times in 5 minutes),
// with the boot screen saying so, then reload the app. Beyond that, a plain message
// and a Retry button instead of a dead window.
async function backendDied(code) {
  restarts = restarts.filter((t) => Date.now() - t < 5 * 60 * 1000);
  if (restarts.length >= 3) return bootFailed(`Sheep's backend stopped ${restarts.length + 1} times (exit ${code}).`);
  restarts.push(Date.now());
  await win.loadFile(path.join(__dirname, "ui", "boot.html"));
  win.webContents.send("boot-status", "Sheep's backend stopped; starting it again…");
  if (await startBackend()) win.loadURL(`http://127.0.0.1:${PORT}/`);
  else bootFailed("Sheep's backend did not start again.");
}

function bootFailed(why) {
  if (!win) return;
  const detail = lastLines.length ? "\n\nLast lines:\n" + lastLines.join("\n") : "";
  win.webContents.send("boot-failed", why + (/python/i.test(lastLines.join(" ")) || !lastLines.length
    ? " Is Python installed and on PATH?" : "") + detail);
}

ipcMain.handle("retry-backend", async () => {
  restarts = [];
  win.webContents.send("boot-status", "Starting Sheep…");
  if (await startBackend()) win.loadURL(`http://127.0.0.1:${PORT}/`);
  else bootFailed("Sheep's backend did not start.");
});

function stopBackend() {
  if (!backend) return;
  if (process.platform === "win32") spawn("taskkill", ["/pid", String(backend.pid), "/T", "/F"], { windowsHide: true });
  else backend.kill("SIGTERM");
  backend = null;
}

async function createWindow() {
  // the window fits the screen it opens on (a 1080p laptop at 150% is 1280 x 720)
  const wa = screen.getPrimaryDisplay().workAreaSize;
  win = new BrowserWindow({
    width: Math.min(1440, wa.width), height: Math.min(900, wa.height), minWidth: 980, minHeight: 600,
    backgroundColor: "#0b0b0b", title: "Sheep", autoHideMenuBar: true, icon: path.join(__dirname, "sheep.png"),
    show: !process.argv.includes("--hidden"),
    webPreferences: { preload: path.join(__dirname, "preload.js"), contextIsolation: true, nodeIntegration: false,
                      backgroundThrottling: false },
  });
  win.loadFile(path.join(__dirname, "ui", "boot.html"));
  // close = sleep in the tray; only Quit (tray menu) stops Sheep
  win.on("close", (e) => {
    if (app.isQuitting || process.env.MLMI_SHOTS) return;
    e.preventDefault();
    win.hide();
    if (tray && !app.toldTray && process.platform === "win32") {
      app.toldTray = true;
      tray.displayBalloon({ title: "Sheep is still running", content: "Your model stays loaded. Open Sheep from this icon; Quit is in its menu.",
                            iconType: "info" });
    }
  });
  const ok = await startBackend();
  if (ok) win.loadURL(`http://127.0.0.1:${PORT}/`);
  else bootFailed("Sheep's backend did not start.");
  if (ok && process.env.MLMI_SHOTS) shots(process.env.MLMI_SHOTS);
  // links out of the app open in the system browser, never inside it
  win.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: "deny" }; });
}

// Done means screenshotted (STANDARDS.md section 8): MLMI_SHOTS=DIR opens each
// view named in MLMI_VIEWS (default: the main ones), waits for its motion to
// settle, saves DIR/<view>.png, and quits. MLMI_JS runs a script in the page first.
async function shots(dir) {
  const fs = require("fs");
  fs.mkdirSync(dir, { recursive: true });
  win.webContents.on("console-message", (e) => {
    const m = e.message !== undefined ? e : { message: arguments[2], level: arguments[1] };
    process.stdout.write(`[page] ${m.level}: ${m.message}\n`);
  });
  const views = (process.env.MLMI_VIEWS || "home,dashboard,models,settings").split(",");
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  await new Promise((r) => win.webContents.once("did-finish-load", r));
  await wait(1500);
  for (const v of views) {
    const [name, js] = v.split(":");
    await win.webContents.executeJavaScript(`location.hash = ${JSON.stringify(name)}`);
    await wait(Number(process.env.MLMI_WAIT || 2600));
    // MLMI_REC_MS=N records each view as frames (DIR/rec/<n>.png, as fast as the page paints) for N ms
    // after its script runs: motion is checked by eye, frame by frame, or joined into an MP4
    if (process.env.MLMI_REC_MS) {
      if (js) await win.webContents.executeJavaScript(js).catch(() => {});
      const rec = path.join(dir, "rec");
      fs.mkdirSync(rec, { recursive: true });
      const t0 = Date.now();
      shots.n = shots.n || 0;
      while (Date.now() - t0 < Number(process.env.MLMI_REC_MS)) {
        const f = await win.webContents.capturePage();
        fs.writeFileSync(path.join(rec, `${String(shots.n++).padStart(5, "0")}.png`), f.toPNG());
      }
      fs.appendFileSync(path.join(rec, "times.txt"), `${name} ${shots.n} ${Date.now() - t0}\n`);
      continue;
    }
    if (js) { await win.webContents.executeJavaScript(js).catch(() => {}); await wait(1400); }
    if (process.env.MLMI_JS) { await win.webContents.executeJavaScript(process.env.MLMI_JS).catch(() => {}); await wait(Number(process.env.MLMI_JS_WAIT || 1200)); }
    const img = await win.webContents.capturePage();
    fs.writeFileSync(path.join(dir, `${name}${js ? "-" + views.indexOf(v) : ""}.png`), img.toPNG());
  }
  app.isQuitting = true;
  if (process.env.MLMI_KEEP_BACKEND) backend = null;
  app.quit();
}

ipcMain.handle("pick-file", async (_e, opts) => {
  const r = await dialog.showOpenDialog(win, { properties: ["openFile"], ...(opts || {}) });
  return r.canceled ? null : r.filePaths[0];
});
ipcMain.handle("pick-folder", async () => {
  const r = await dialog.showOpenDialog(win, { properties: ["openDirectory", "createDirectory"] });
  return r.canceled ? null : r.filePaths[0];
});
ipcMain.handle("reveal", async (_e, p) => shell.showItemInFolder(path.normalize(p)));
// folders only: a file path here would run the file
ipcMain.handle("open-folder", async (_e, p) => {
  const dir = path.normalize(String(p || ""));
  return require("fs").statSync(dir, { throwIfNoEntry: false })?.isDirectory() ? shell.openPath(dir) : "not a folder";
});

function show() {
  if (!win) return createWindow();
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
}

// Windows launches electron.exe directly. It must receive Sheep's app folder as
// its argument (otherwise Electron shows its own welcome page), and --hidden so
// a login start sleeps in the tray while the model loads and its checks run
// before anyone asks for it (Ernest, 2026-10-13).
const LOGIN_ARGS = [APP_ENTRY, "--hidden"];
function loginItemOptions(openAtLogin) {
  return { openAtLogin, path: process.execPath, args: LOGIN_ARGS };
}

function migrateLoginItem() {
  if (process.platform !== "win32") return;
  try {
    const current = app.getLoginItemSettings({ path: process.execPath, args: LOGIN_ARGS });
    const older = [["--hidden"], [APP_ENTRY]].some((args) => app.getLoginItemSettings({ path: process.execPath, args }).openAtLogin);
    // Repair an enabled older Sheep entry (no app folder, or not hidden), and
    // remove it so Windows starts Sheep once.
    if (older && !current.openAtLogin) {
      for (const args of [["--hidden"], [APP_ENTRY]]) app.setLoginItemSettings({ openAtLogin: false, path: process.execPath, args });
      app.setLoginItemSettings(loginItemOptions(true));
    }
    // On by default: the first launch turns it on once. A person who turns it
    // off later is never overruled (the marker remembers it was decided).
    // Test runs (their own MLMI_HOME) leave this machine's login entry alone.
    const marker = path.join(app.getPath("userData"), "login-default-set");
    if (!process.env.MLMI_HOME && !require("fs").existsSync(marker)) {
      if (!current.openAtLogin && !older) app.setLoginItemSettings(loginItemOptions(true));
      require("fs").writeFileSync(marker, new Date().toISOString());
    }
  } catch (e) {
    process.stderr.write(`[startup] could not update the Windows login entry: ${e.message}\n`);
  }
}

function makeTray() {
  const icon = nativeImage.createFromPath(path.join(__dirname, process.platform === "win32" ? "sheep-mono-20261010.ico" : "sheep.png"));
  tray = new Tray(icon.isEmpty() ? nativeImage.createFromPath(path.join(__dirname, "sheep.png")) : icon);
  tray.setToolTip("Sheep: running, model loaded");
  const menu = () => Menu.buildFromTemplate([
    { label: "Open Sheep", click: show },
    { type: "separator" },
    { label: "Start Sheep with Windows", type: "checkbox",
      checked: app.getLoginItemSettings({ path: process.execPath, args: LOGIN_ARGS }).openAtLogin,
      click: (it) => app.setLoginItemSettings(loginItemOptions(it.checked)) },
    { type: "separator" },
    { label: "Quit Sheep", click: () => { app.isQuitting = true; app.quit(); } },
  ]);
  tray.setContextMenu(menu());
  tray.on("right-click", () => tray.setContextMenu(menu()));
  tray.on("click", show);
  tray.on("double-click", show);
}

// one Sheep at a time: opening it again shows the one sleeping in the tray
if (process.env.MLMI_SHOTS) {
  // screenshots (STANDARDS.md section 8) run beside a Sheep that may be sleeping in the tray
  app.setPath("userData", path.join(app.getPath("temp"), "sheep-shots"));
  app.whenReady().then(createWindow);
} else if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", show);
  app.whenReady().then(async () => {
    migrateLoginItem();
    if (!process.env.MLMI_SHOTS) makeTray();
    await createWindow();
  });
}
app.on("before-quit", () => { app.isQuitting = true; stopBackend(); });
app.on("window-all-closed", () => { if (app.isQuitting) { stopBackend(); app.quit(); } });
