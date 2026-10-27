// Sheep icons: drawn for this app (24px grid, 1.8 stroke), plus the Sheep logo.
// Any element with data-icon="name" gets the icon prepended; MLMI.icon(name) returns the SVG.
"use strict";

(() => {
  const P = {
    pencil: '<path d="M4 20h4L19 9l-4-4L4 16v4z"/><path d="M14 6l4 4"/>',
    search: '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5"/>',
    canvas: '<rect x="3" y="4" width="8" height="7" rx="1.5"/><rect x="13" y="4" width="8" height="4" rx="1.5"/><rect x="13" y="10" width="8" height="10" rx="1.5"/><rect x="3" y="13" width="8" height="7" rx="1.5"/>',
    code: '<path d="M8 8l-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14"/>',
    pin: '<path d="M9 4h6l-1 6 3 3H7l3-3z"/><path d="M12 13v7"/>',
    image: '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="10" r="1.8"/><path d="M21 16l-5-5-8 8"/>',
    audio: '<path d="M9 18V6l10-2v12"/><circle cx="6.5" cy="18" r="2.5"/><circle cx="16.5" cy="16" r="2.5"/>',
    copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
    save: '<path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.3a2.5 2.5 0 0 1 4.8.9c0 1.7-2.4 2.2-2.4 3.8"/><circle cx="12" cy="17" r=".6"/>',
    mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>',
    wave: '<path d="M4 10v4M8 7v10M12 4v16M16 7v10M20 10v4"/>',
    note: '<path d="M5 4h14v16H5z"/><path d="M8 9h8M8 13h8M8 17h5"/>',
    doc: '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h6"/>',
    grid: '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 10h16M4 15h16M10 4v16"/>',
    slides: '<rect x="3" y="5" width="18" height="12" rx="2"/><path d="M12 17v3M8 20h8"/>',
    play: '<circle cx="12" cy="12" r="9"/><path d="M10 8.5v7l6-3.5z"/>',
    spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M18 16l.8 2.2L21 19l-2.2.8L18 22l-.8-2.2L15 19l2.2-.8z"/>',
    gear: '<circle cx="12" cy="12" r="3"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/>',
    panel: '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/>',
    chev: '<path d="M7 10l5 5 5-5"/>',
    up: '<path d="M12 19V6M6 11l6-6 6 6"/>',
    clip: '<path d="M20 12.5l-7.6 7.6a5 5 0 0 1-7.1-7.1l8.5-8.5a3.3 3.3 0 0 1 4.7 4.7l-8.5 8.5a1.7 1.7 0 0 1-2.4-2.4l7.8-7.8"/>',
    stop: '<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/>',
    undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 010 11H11"/>',
    redo: '<path d="M15 14l5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 000 11H13"/>',
    x: '<path d="M6 6l12 12M18 6L6 18"/>',
    gauge: '<path d="M4 16a8 8 0 1116 0"/><path d="M12 16l4-5"/>',
    route: '<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M8.5 6H14a4 4 0 010 8h-4a4 4 0 000 8"/>',
    tool: '<path d="M14.5 5.5a4 4 0 00-5 5L4 16l4 4 5.5-5.5a4 4 0 005-5l-2.5 2.5-2.5-.5-.5-2.5z"/>',
    folder: '<path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>',
    cloud: '<path d="M7 18a4 4 0 01-.5-8A5.5 5.5 0 0117 8.5 4.5 4.5 0 0117 18z"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    alert: '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.5"/>',
    open: '<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v5a1 1 0 01-1 1H5a1 1 0 01-1-1V7a1 1 0 011-1h5"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    chip: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/>',
    key: '<circle cx="8" cy="15" r="4"/><path d="M11 12l8-8M16 7l3 3M14 9l2 2"/>',
    sliders: '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
    bolt: '<path d="M13 3L5 13.5h6L10 21l8-10.5h-6z"/>',
    balance: '<path d="M12 4v16M7 20h10M5 8h14"/><path d="M5 8l-2.5 6a2.5 2.5 0 005 0zM19 8l-2.5 6a2.5 2.5 0 005 0z"/>',
    leaf: '<path d="M5 19c0-8 5-14 15-14 0 10-6 15-14 15"/><path d="M5 19l7-7"/>',
    trash: '<path d="M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13"/>',
  };
  // The Sheep logo: a fleece of circles, a dark face, two legs. Our own drawing.
  const SHEEP = `<svg viewBox="0 0 64 64" class="sheep-logo" aria-hidden="true">
    <g fill="var(--fleece, #f4f4f4)"><circle cx="22" cy="28" r="10"/><circle cx="34" cy="22" r="11"/><circle cx="45" cy="29" r="10"/>
    <circle cx="28" cy="38" r="10"/><circle cx="41" cy="39" r="10"/><circle cx="18" cy="37" r="7"/><circle cx="50" cy="38" r="7"/></g>
    <g fill="var(--face, #050505)" stroke="var(--face-edge, #f4f4f4)" stroke-width="2" stroke-linejoin="round"><rect x="25" y="46" width="4.5" height="10" rx="2"/><rect x="38" y="46" width="4.5" height="10" rx="2"/>
    <ellipse cx="14" cy="30" rx="8" ry="9.5"/><ellipse cx="7.5" cy="24" rx="4" ry="2.4" transform="rotate(-25 7.5 24)"/>
    <ellipse cx="19.5" cy="22" rx="4" ry="2.4" transform="rotate(25 19.5 22)"/></g>
    <circle cx="11" cy="29" r="1.6" fill="var(--logo-eye, #fff)"/><circle cx="17" cy="29" r="1.6" fill="var(--logo-eye, #fff)"/></svg>`;

  function icon(name) {
    if (name === "sheep") return SHEEP;
    const p = P[name];
    return p ? `<svg viewBox="0 0 24 24" class="i" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>` : "";
  }
  function paint(root = document) {
    root.querySelectorAll("[data-icon]").forEach((el) => {
      if (el.dataset.painted) return;
      el.dataset.painted = "1";
      el.insertAdjacentHTML("afterbegin", icon(el.dataset.icon));
    });
  }
  window.MLMI_ICON = { icon, paint };
})();
