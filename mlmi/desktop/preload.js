// The only doors from the page to the machine: native file and folder pickers.
// Everything else goes through the backend's API, as it does in a browser.
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("mlmiDesktop", {
  pickFile: (opts) => ipcRenderer.invoke("pick-file", opts),
  pickFolder: () => ipcRenderer.invoke("pick-folder"),
  reveal: (p) => ipcRenderer.invoke("reveal", p),
  openFolder: (p) => ipcRenderer.invoke("open-folder", p),
  onBootFailed: (fn) => ipcRenderer.on("boot-failed", (_e, msg) => fn(msg)),
  onBackendExit: (fn) => ipcRenderer.on("backend-exit", (_e, code) => fn(code)),
  onBootStatus: (fn) => ipcRenderer.on("boot-status", (_e, msg) => fn(msg)),
  retryBackend: () => ipcRenderer.invoke("retry-backend"),
});
