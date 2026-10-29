"""The benchmark pictures as scorecards: one question per picture, one row per test, read in a second.

    python tooling/bench_cards.py [OUT_DIR]

Each row is one test. The bar says how far ahead Sheep is (blue, to the right) or behind (orange, to the
left), in per cent of the other engine's result; the numbers at the end of the row are the measurements
themselves. Within 2% is a tie (grey): runs on this laptop vary that much. The headline counts the rows.
The data are the race logs, kept in tooling/benchmark_charts.py and its .jsonl files.

Colours: blue #2a78d6 and orange #eb6834 are the reference palette's slots 1 and 2, a warm/cool pair with a
neutral grey midpoint, validated for colour-blind separation (dataviz validate_palette.js: all checks pass).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_charts as B  # noqa: E402

AHEAD, BEHIND, EVEN = "#2a78d6", "#eb6834", "#b9b7b1"
INK, SUB, MUTED, RULE, BG = "#1b1b1a", "#4a4945", "#77756f", "#e3e1dc", "#fcfcfb"
TIE = 2.0                    # per cent: a difference this small is run-to-run noise here
CAP = 60.0                   # the bar scale stops here; a bigger lead is written, not drawn longer
MACHINE = "Ryzen 5 5600H · RTX 3050 Laptop 4 GB · on AC · Qwen2.5-3B at 4 bits on every engine"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK})


def lead(sheep, other, higher_better=True):
    """Sheep's lead over the other engine, in per cent of the other's result (negative: behind)."""
    if not sheep or not other:
        return 0.0
    return ((sheep / other) - 1) * 100 if higher_better else ((other / sheep) - 1) * 100


def card(out, name, title, sub, rows, rival, foot):
    """rows: (label, group, lead %, 'Sheep value', 'other value'), drawn top to bottom, groups divided."""
    n = len(rows)
    H = 2.75 + 0.5 * n                                   # inches: a fixed header, a fixed footer
    at = lambda inch: 1 - inch / H
    fig = plt.figure(figsize=(13, H), facecolor=BG)
    ax = fig.add_axes([0.30, 0.55 / H, 0.42, (H - 2.2) / H], facecolor=BG)
    ahead = sum(r[2] > TIE for r in rows)
    behind = sum(r[2] < -TIE for r in rows)
    even = n - ahead - behind
    fig.text(0.03, at(0.3), title, fontsize=19, fontweight="bold", va="top")
    words = [f"ahead in {ahead}", f"behind in {behind}"] + ([f"tied in {even}"] if even else [])
    fig.text(0.03, at(0.78), f"Of {n} tests, Sheep is " + ", ".join(words) + ".", fontsize=13, color=SUB, va="top")
    fig.text(0.03, at(1.15), sub, fontsize=9.5, color=MUTED, va="top", wrap=True)
    ys = list(range(n))[::-1]
    last = None
    for y, (label, group, pct, sv, ov) in zip(ys, rows):
        if group != last:
            if last is not None:
                ax.axhline(y + 0.5, color=RULE, lw=1)
            ax.text(-CAP - 4, y + 0.42, group.upper(), fontsize=8.5, color=MUTED, ha="right", va="bottom",
                    fontweight="bold", transform=ax.transData)
            last = group
        col = AHEAD if pct > TIE else BEHIND if pct < -TIE else EVEN
        w = max(-CAP, min(CAP, pct))
        if abs(pct) <= TIE:
            ax.plot([0], [y], "o", ms=8, color=EVEN, mec=BG, mew=2, zorder=3)
        else:
            ax.barh(y, w, height=0.5, color=col, edgecolor=BG, linewidth=2, zorder=2)
            if abs(pct) > CAP:                                   # a lead bigger than the scale
                ax.text(w + (1.5 if w > 0 else -1.5), y, "»" if w > 0 else "«", color=col, va="center",
                        ha="left" if w > 0 else "right", fontsize=13, fontweight="bold")
        txt = "tie" if abs(pct) <= TIE else f"{pct:+.0f}%"
        ax.text(CAP + 4, y, txt, va="center", ha="left", fontsize=12, fontweight="bold", color=INK)
        ax.text(CAP + 20, y, f"{sv}  vs  {ov}", va="center", ha="left", fontsize=10, color=SUB)
        ax.text(-CAP - 4, y, label, va="center", ha="right", fontsize=11.5, color=INK)
    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.set_xlim(-CAP, CAP)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks([-50, -25, 0, 25, 50], ["50%", "25%", "0", "25%", "50%"], fontsize=9, color=MUTED)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="x", color=RULE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    top = n - 0.3
    ax.text(-CAP, top + 0.15, f"◀  {rival} ahead", color=BEHIND, fontsize=10.5, fontweight="bold", va="bottom")
    ax.text(CAP, top + 0.15, "Sheep ahead  ▶", color=AHEAD, fontsize=10.5, fontweight="bold", va="bottom", ha="right")
    ax.text(CAP + 20, top + 0.15, f"Sheep  vs  {rival}", color=MUTED, fontsize=9.5, va="bottom")
    fig.text(0.03, 0.12 / H, foot, fontsize=8.5, color=MUTED)
    p = Path(out) / name
    fig.savefig(p, dpi=150, facecolor=BG)
    plt.close(fig)
    print(p)


def versus(out, rival, name):
    M = B.MARKET
    rows = []
    for dev, word in (("gpu", "graphics card"), ("cpu", "processor")):
        s, r = M["Sheep"][dev], M[rival][dev]
        rows += [(f"Reading a prompt", word, lead(s[0], r[0]), f"{s[0]:.0f}", f"{r[0]:.0f} tok/s"),
                 (f"Writing the answer", word, lead(s[1], r[1]), f"{s[1]:.1f}", f"{r[1]:.1f} tok/s"),
                 (f"Start-up", word, lead(s[3], r[3], False), f"{s[3]:.1f}", f"{r[3]:.1f} s"),
                 (f"Hard questions right", word, lead(s[2], r[2]), f"{s[2]}", f"{r[2]} of 40")]
    card(out, name, f"Sheep against {rival}", MACHINE + ". Speeds: median of the runs; questions: 40, at temperature 0.",
         rows, rival, "tooling/market_race.py · bar = Sheep's lead in % of the other's result · within 2% = tie · "
         "» = the lead runs past the scale")


def engines(out):
    rows = [(n.split(" · ")[1].capitalize(), n.split(" · ")[0].replace("CPU", "processor").replace("GPU", "graphics card"),
             (r - 1) * 100 if not (lo <= 1 <= hi) else 0.0, f"{r:.2f}×", f"95% range {lo:.2f}–{hi:.2f}")
            for n, r, lo, hi in B.ENGINE_RACES]
    card(out, "sml_vs_ggml.png", "Engine against engine: Sheep's SML and GGML",
         "GGML is the engine inside llama.cpp and Ollama. Same laptop, same model, same 4-bit budget; 8 rounds in "
         "ABBA order each, with a bootstrap 95% range.",
         rows, "GGML", "tooling/sml_cpu_race_ci.py, sml_gpu_race_ci.py · a range that crosses 1.00× is a tie")


def formats(out):
    rows = []
    for m, (gc, sc), (gt, st_), size in B.FORMAT:
        rows += [("Error against the original", m, lead(1 - sc, 1 - gc, False), f"{1 - sc:.4f}", f"{1 - gc:.4f}"),
                 ("Same next word as the original", m, lead(st_, gt), f"{st_:.1f}%", f"{gt:.1f}%"),
                 ("File size", m, lead(size, 1.0, False), f"{size:.3f}×", "1.000× GGUF")]
    card(out, "ssf_vs_gguf.png", "File against file: Sheep's SSF and GGUF Q4_K_M",
         "Same model, same 4-bit budget, scored against the full-precision model. Ollama loads the same GGUF, so this "
         "is SSF against both.", rows, "GGUF", "RESULTS.md, \"Three specs raced\" · every position of 256-512 "
         "held-out tokens scored")


def hardware(out):
    """How much of the machine each engine puts to work, and what it gets for the power."""
    rows = {}
    for line in B.HW_LOG.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[({"sheep": "Sheep", "llamacpp": "llama.cpp", "ollama": "Ollama"}[r["kind"]], r["device"])] = r
    card_mb = 4096
    engines = ["Sheep", "llama.cpp", "Ollama"]
    g = {e: rows[(e, "gpu")] for e in engines}
    panels = [("Card memory put to work", "% of the 4 GB card", lambda r: r["peak_vram_mb"] / card_mb * 100, "{:.0f}%"),
              ("Graphics card busy", "% of the time while answering", lambda r: r["avg_gpu_util"], "{:.0f}%"),
              ("Words for the power", "tokens written per watt-second ×100", lambda r: r["gen_tok_s"] / r["avg_gpu_w"] * 100, "{:.1f}")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
    fig.text(0.03, 0.94, "How much of the machine each engine puts to work", fontsize=19, fontweight="bold", va="top")
    fig.text(0.03, 0.84, "On the graphics card, Sheep fills the card's spare memory with ready-made copies of the weights,\n"
             "keeps the card the busiest, and writes the most words per watt.", fontsize=12, color=SUB, va="top")
    for ax, (t, unit, f, fmt) in zip(axes, panels):
        vals = [f(g[e]) for e in engines]
        best = max(vals)
        for i, (e, v) in enumerate(zip(engines, vals)):
            col = AHEAD if e == "Sheep" else EVEN
            ax.barh(i, v, height=0.56, color=col, edgecolor=BG, linewidth=2)
            ax.text(v + best * 0.03, i, fmt.format(v), va="center", fontsize=12, fontweight="bold" if e == "Sheep" else "normal")
        ax.set_yticks(range(3), engines, fontsize=11.5)
        ax.invert_yaxis()
        ax.set_xlim(0, best * 1.35)
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{t}\n", loc="left", fontsize=12.5, fontweight="bold")
        ax.text(0, 1.04, unit, transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ram = ", ".join(f"{e} {rows[(e, 'gpu')]['peak_ram_mb'] / 1024:.1f} GB" for e in engines)
    fig.text(0.03, 0.03, f"tooling/hw_race.jsonl (market_race.py: card memory counted from before the engine started; busy "
             f"and power averaged over the runs).\nMemory (RAM) on the card runs: {ram}.", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.58, bottom=0.17, wspace=0.45)
    p = Path(out) / "hardware.png"
    fig.savefig(p, dpi=150, facecolor=BG)
    plt.close(fig)
    print(p)


def big_model(out):
    first = {}
    for line in B.BIG_LOG.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        e = {"sheep": "Sheep", "llamacpp": "llama.cpp", "ollama": "Ollama"}[r["kind"]]
        first.setdefault(e, r)                      # the first round: before the card hit its thermal floor
    s = first["Sheep"]
    rows = []
    for rival in ("llama.cpp", "Ollama"):
        r = first[rival]
        rows += [("Reading a prompt", f"against {rival}", lead(s["prompt_tok_s"], r["prompt_tok_s"]),
                  f"{s['prompt_tok_s']:.0f}", f"{r['prompt_tok_s']:.0f} tok/s"),
                 ("Writing the answer", f"against {rival}", lead(s["gen_tok_s"], r["gen_tok_s"]),
                  f"{s['gen_tok_s']:.2f}", f"{r['gen_tok_s']:.2f} tok/s"),
                 ("Start-up", f"against {rival}", lead(s["load_s"], r["load_s"], False),
                  f"{s['load_s']:.1f}", f"{r['load_s']:.1f} s")]
    card(out, "big_model.png", "A model too big for the card: Qwen2.5-7B on a 4 GB GPU",
         "4.7 GB at 4 bits; each engine splits it between the card and the processor by itself. Sheep reads every "
         "layer's prompt on the card and writes its answer on the processor (writing 5.5 → 6.9 tok/s since 29 Oct).", rows, "the other",
         "tooling/race7b.jsonl · first round of each, back to back at 94-95 °C (the second round hit the card's thermal floor)")


def main(out):
    style()
    Path(out).mkdir(parents=True, exist_ok=True)
    versus(out, "llama.cpp", "sheep_vs_llamacpp.png")
    versus(out, "Ollama", "sheep_vs_ollama.png")
    engines(out)
    formats(out)
    hardware(out)
    big_model(out)
    B.style()
    B.features(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "mlmi" / "media" / "bench"))
