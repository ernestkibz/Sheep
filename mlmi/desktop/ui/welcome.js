// The first run, with no model plugged in. Keep model choice open: the user can
// enter a Hugging Face id, choose an original checkpoint folder, or use an SSF.
(function () {
  const { $, icon, openSettings, api, toast } = MLMI;

  function html(state) {
    return `<div class="welcome">
      <span class="logo big">${icon("sheep")}</span>
      <h1>Sheep runs AI on this computer</h1>
      <span class="sub">Nothing leaves this computer. Choose any model: Sheep reads it, fits it to this computer
        and checks it, and tells you how it will run before anything downloads. Nothing to set.</span>
      <div class="w-cards">
        <button class="w-card go" id="wChoose">
          <span class="w-ic">${icon("cloud")}</span>
          <b>Add a model from Hugging Face</b>
          <span class="dim">Paste any model ID, even a GGUF page: Sheep finds the original weights,
            checks them, sizes them for this computer and builds them.</span>
        </button>
        <button class="w-card" id="wLocal">
          <span class="w-ic">${icon("folder")}</span>
          <b>Use files on this computer</b>
          <span class="dim">A folder of a model's original files, or a Sheep model file you already have.</span>
        </button>
        ${state.cloud && state.cloud.ready
          ? `<button class="w-card" id="wCloud">
          <span class="w-ic">${icon("route")}</span>
          <b>Start now with Nemotron on Nebius</b>
          <span class="dim">NVIDIA Nemotron answers at once, with no download. Your messages go to Nebius until you add a local model.</span>
        </button>`
          : `<button class="w-card" id="wKeys">
          <span class="w-ic">${icon("route")}</span>
          <b>Start with Nemotron on Nebius</b>
          <span class="dim">Paste a Nebius key in Settings › Accounts and ask straight away; add a local model whenever you like.</span>
        </button>`}
      </div>
      <p class="w-foot dim">Nothing downloads until you have seen the size and how well it fits this computer.</p>
    </div>`;
  }

  // One listener for the life of the page: the card is redrawn with the thread.
  document.addEventListener("click", async (e) => {
    if (e.target.closest("#wCloud")) {
      try { await api("/api/settings", { mode: 1 }); await MLMI.refresh(); $("#askText").focus(); }
      catch (err) { toast(err.message, "bad"); }
      return;
    }
    if (e.target.closest("#wKeys")) { openSettings("accounts"); return; }
    if (e.target.closest("#wChoose")) {
      MLMI.modelSlotChoice("chat");
      setTimeout(() => { const r = document.querySelector(".model-slot-choice [data-repo]"); if (r) r.focus(); }, 100);
      return;
    }
    if (!e.target.closest("#wLocal")) return;
    openSettings("models");
    setTimeout(() => { const src = $("#qSrc"); if (src) { src.scrollIntoView({ block: "center" }); src.focus(); } }, 350);
  });

  MLMI.welcome = { html };
})();
