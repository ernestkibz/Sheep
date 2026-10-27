"""The benchmark pictures, one question each, for the README and Devpost.

    python tooling/benchmark_charts.py [OUT_DIR]

1 sheep_vs_llamacpp.png   Sheep against llama.cpp: speed, answers right, start-up
2 sheep_vs_ollama.png     Sheep against Ollama: the same
3 sml_vs_ggml.png         the engines: Sheep's SML against GGML (inside llama.cpp and Ollama)
4 ssf_vs_gguf.png         the file formats: Sheep's SSF against GGUF Q4_K_M (Ollama runs the same GGUF)
5 features.png            what each one does besides running a model

Every number is a measured result copied from the race logs (2026-09-29 to 2026-10-27; Ryzen 5 5600H,
RTX 3050 Laptop 4 GB, on AC): tooling/market_race.py, sml_cpu_race_ci.py, sml_gpu_race_ci.py and the
three-spec format race in RESULTS.md.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, BG = "#1f2328", "#6e7781", "#d8dee4", "#ffffff"
WIN, TIE, LOSS = "#1a7f37", "#8c959f", "#cf222e"
SHEEP, LLAMA, OLLAMA = "#0b5cad", "#c2410c", "#57606a"
MACHINE = "Qwen2.5-3B, 4-bit · Ryzen 5 5600H, RTX 3050 Laptop 4 GB, on AC · GPU runs on a hot card (95-98 °C)"

# tooling/market_race.py (each engine as a server, ~710-token prompt, 128 new tokens, median of 5 runs;
# 40 hard questions at temperature 0). Sheep's GPU row is the run after the 2026-10-27 prompt-path fix.
MARKET = {  # engine: {device: (read tok/s, write tok/s, right of 40, start-up s)}
    # speeds and start-up: the 2026-10-27 night re-race, two rows each in mirrored order (A B C C B A),
    # averaged; Sheep's GPU start-up is the cached prompt check (2.03 s); answers right: the earlier 40-question run
    "Sheep": {"gpu": (365.5, 11.57, 33, 2.03), "cpu": (83.6, 15.36, 32, 0.78)},
    "llama.cpp": {"gpu": (380.8, 11.70, 31, 2.25), "cpu": (71.7, 15.39, 32, 2.73)},
    "Ollama": {"gpu": (383.6, 11.99, 32, 6.36), "cpu": (80.7, 15.21, 33, 6.25)},
}
BY_CLASS = {  # right of 10 per class, GPU then CPU
    "Sheep": {"gpu": (10, 8, 9, 6), "cpu": (10, 8, 9, 5)},
    "llama.cpp": {"gpu": (10, 4, 10, 7), "cpu": (10, 5, 10, 7)},
    "Ollama": {"gpu": (10, 6, 9, 7), "cpu": (9, 7, 10, 7)},
}
CLASSES = ("Maths", "Code", "Facts", "Format")
# sml_cpu_race_ci.py / sml_gpu_race_ci.py: SML against llama.cpp's GGML, same machine, ABBA order,
# median of 8 rounds with a 95% bootstrap range
ENGINE_RACES = [
    ("CPU · read 128 tokens", 1.395, 1.318, 1.462),
    ("CPU · read 512 tokens", 1.257, 1.210, 1.315),
    ("CPU · write 128 tokens", 1.001, 0.986, 1.005),
    ("GPU · read 128 tokens", 0.991, 0.988, 1.006),
    ("GPU · read 512 tokens", 1.083, 1.082, 1.087),
    ("GPU · write 64 tokens, cool card", 0.986, 0.986, 0.988),
    ("GPU · write 64 tokens, hot card", 1.391, 1.390, 1.399),
]
# RESULTS.md "Three specs raced": SSF against llama.cpp's own Q4_K_M GGUF, decoded and scored by the same
# forward pass; correlation and top-1 against the full-precision model, file size as a ratio
FORMAT = [  # model, (GGUF corr, SSF corr), (GGUF top-1 %, SSF top-1 %), SSF size / GGUF size
    ("Qwen2.5-3B", (0.98992, 0.99405), (88.9, 91.6), 1.0225),
    ("Llama-3.2-1B", (0.97902, 0.99270), (85.0, 89.6), 1.0168),
    ("pythia-410m", (0.97656, 0.98205), (51.6, 59.8), 0.9981),
]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})


def clean(ax, keep_bottom=False):
    for s in ("top", "right", "left") + (() if keep_bottom else ("bottom",)):
        ax.spines[s].set_visible(False)


def head(fig, title, sub):
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, sub, fontsize=10.5, color=MUTED)


def score(fig, parts, n):
    """The verdict in one line under the title: how many of the n tests each side won."""
    x = 0.02
    fig.text(x, 0.845, f"Of {n} tests:", fontsize=13, color=INK, fontweight="bold")
    x += 0.075
    for word, k, col in parts:
        t = f"{k} {word}{'s' if word == 'tie' and k != 1 else ''}"
        fig.text(x, 0.845, t, fontsize=13, color=col, fontweight="bold")
        x += 0.0085 * len(t) + 0.025


def save(fig, out, name):
    p = Path(out) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, facecolor=BG)
    plt.close(fig)
    print(p)


def pair_bars(ax, title, rival, rival_col, metric, unit, higher_better=True, fmt="{:.0f}"):
    """Two devices, two bars each: Sheep and the rival, with who is better written under them.
    Speeds within 2% are a tie (the runs vary that much); answers right tie only when equal."""
    wins = []
    for i, dev in enumerate(("gpu", "cpu")):
        s, r = MARKET["Sheep"][dev][metric], MARKET[rival][dev][metric]
        top = max(s, r)
        edge = (s / r - 1) if higher_better else (r / s - 1)
        close = s == r if metric == 2 else abs(edge) < 0.02
        who = "tie" if close else "Sheep" if edge > 0 else rival
        wins.append(who)
        for j, (v, col) in enumerate(((s, SHEEP), (r, rival_col))):
            y = i * 2.6 + j
            ax.barh(y, v, height=0.8, color=col)
            ax.text(v + top * 0.02, y, f"{fmt.format(v)} {unit}", va="center", fontsize=10.5, color=INK)
        word, c = (("Tie", TIE) if who == "tie" else (f"✓ Sheep better", WIN) if who == "Sheep"
                   else (f"{rival} better", LOSS))
        ax.text(0, i * 2.6 + 1.75, word, va="center", fontsize=11, color=c, fontweight="bold")
        ax.text(-top * 0.02, i * 2.6 + 0.5, dev.upper(), va="center", ha="right", fontsize=11, color=INK,
                fontweight="bold")
    ax.set_xlim(0, max(max(MARKET[e][d][metric] for d in ("gpu", "cpu")) for e in ("Sheep", rival)) * 1.45)
    ax.set_ylim(4.9, -0.8)
    ax.set_yticks([])
    ax.set_xticks([])
    clean(ax)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    return wins


def versus(out, rival, col, name):
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.6), facecolor=BG)
    head(fig, f"Sheep vs {rival}",
         f"{MACHINE}. {rival} runs the GGUF Q4_K_M file; Sheep runs its own SSF file.")
    wins = pair_bars(axes[0], "Reading the prompt\n(tokens a second, higher is better)", rival, col, 0, "tok/s")
    wins += pair_bars(axes[1], "Writing the answer\n(tokens a second, higher is better)", rival, col, 1, "tok/s",
                      fmt="{:.1f}")
    wins += pair_bars(axes[2], "Hard questions right\n(of 40: maths, code, facts, format)", rival, col, 2, "/ 40")
    wins += pair_bars(axes[3], "Start-up, ready to answer\n(seconds, lower is better)", rival, col, 3, "s",
                      higher_better=False, fmt="{:.1f}")
    score(fig, [("Sheep better", wins.count("Sheep"), WIN), (f"{rival} better", wins.count(rival), LOSS),
                ("tie", wins.count("tie"), TIE)], len(wins))
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SHEEP), plt.Rectangle((0, 0), 1, 1, color=col)],
               labels=["Sheep", rival], loc="lower left", ncol=2, frameon=False, bbox_to_anchor=(0.015, 0.0))
    fig.text(0.985, 0.02, "tooling/market_race.py · median of 5 runs · 40 questions at temperature 0",
             ha="right", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.72, bottom=0.1, wspace=0.28)
    save(fig, out, name)


def engines(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 6.2), facecolor=BG, gridspec_kw={"width_ratios": [1.35, 1]})
    head(fig, "Engine vs engine: Sheep's SML against GGML",
         "GGML is the engine inside llama.cpp and Ollama. Same machine, same model, same 4-bit budget. "
         "Right of the line = SML faster.")
    ys = list(range(len(ENGINE_RACES)))[::-1]
    tally = []
    for y, (n, r, lo, hi) in zip(ys, ENGINE_RACES):
        c = WIN if lo > 1 else LOSS if hi < 1 else TIE
        a.plot([lo, hi], [y, y], color=c, lw=8, alpha=0.35, solid_capstyle="round")
        a.plot([r], [y], "o", color=c, ms=10, mec=BG, mew=1.5)
        word = "✓ SML better" if c == WIN else "GGML better" if c == LOSS else "tie"
        a.text(1.49, y, f"{r:.2f}×  {word}", va="center", fontsize=10.5, color=c, fontweight="bold")
        tally.append(c)
    a.axvline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    a.set_yticks(ys, [n for n, *_ in ENGINE_RACES])
    a.set_xlim(0.9, 1.75)
    a.set_xticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4], ["0.9×", "1.0×", "1.1×", "1.2×", "1.3×", "1.4×"])
    a.grid(axis="x", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    clean(a, keep_bottom=True)
    a.set_title("SML vs GGML in llama.cpp, head to head\n(8 rounds in ABBA order; bar = 95% range)",
                loc="left", fontsize=12, color=INK, fontweight="bold")

    rows = [("GPU read", 0), ("GPU write", 1), ("CPU read", 0), ("CPU write", 1)]
    for i, (n, k) in enumerate(rows):
        dev = n.split()[0].lower()
        s = MARKET["Sheep"][dev][k]
        for j, (eng, col) in enumerate((("llama.cpp", LLAMA), ("Ollama", OLLAMA))):
            r = s / MARKET[eng][dev][k]
            y = i * 2.8 + j
            c = WIN if r > 1.02 else LOSS if r < 0.98 else TIE
            b.barh(y, r - 1, left=1, height=0.8, color=col, alpha=0.85)
            word = "✓ SML better" if c == WIN else "GGML better" if c == LOSS else "tie"
            b.text(max(r, 1) + 0.012, y, f"{r:.2f}× vs {eng}: {word}", va="center", fontsize=10, color=c,
                   fontweight="bold")
            tally.append(c)
        b.text(0.795, i * 2.8 + 0.5, n, va="center", ha="right", fontsize=11, color=INK)
    b.axvline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    b.set_xlim(0.8, 1.62)
    b.set_ylim(11, -0.8)
    b.set_yticks([])
    b.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2], ["0.8×", "0.9×", "1.0×", "1.1×", "1.2×"])
    clean(b, keep_bottom=True)
    b.set_title("As whole servers\n(SML speed ÷ GGML's; 710-token prompt)", loc="left", fontsize=12, color=INK, fontweight="bold")
    score(fig, [("SML better", tally.count(WIN), WIN), ("GGML better", tally.count(LOSS), LOSS),
                ("tie", tally.count(TIE), TIE)], len(tally))
    fig.text(0.02, 0.02, "Left: tooling/sml_cpu_race_ci.py, sml_gpu_race_ci.py. Right: tooling/market_race.py. "
             "Grey = within 2%, a tie.", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.71, bottom=0.1, wspace=0.42)
    save(fig, out, "sml_vs_ggml.png")


def formats(out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), facecolor=BG)
    head(fig, "File vs file: Sheep's SSF against GGUF Q4_K_M",
         "Same model, same 4-bit budget. GGUF made by llama.cpp's own llama-quantize; Ollama loads that same "
         "GGUF file, so its numbers are the GGUF bars. Scored against the full-precision model.")
    names = [m for m, *_ in FORMAT]
    panels = [
        ("How close to the original\n(error, 1 − correlation; lower is better)", lambda f: 1 - f[1][0],
         lambda f: 1 - f[1][1], "{:.4f}"),
        ("Same next word as the original\n(top-1 %, higher is better)", lambda f: f[2][0], lambda f: f[2][1],
         "{:.1f}%"),
    ]
    for ax, (t, g, s, fmt) in zip(axes[:2], panels):
        top = max(max(g(f), s(f)) for f in FORMAT)
        for i, f in enumerate(FORMAT):
            for j, (v, col, lab) in enumerate(((s(f), SHEEP, "SSF"), (g(f), LLAMA, "GGUF"))):
                y = i * 2.8 + j
                ax.barh(y, v, height=0.8, color=col)
                ax.text(v + top * 0.02, y, f"{lab} {fmt.format(v)}", va="center", fontsize=10, color=INK)
            ax.text(-top * 0.02, i * 2.8 + 0.5, f[0], va="center", ha="right", fontsize=11, color=INK)
            ax.text(0, i * 2.8 + 1.75, "✓ SSF better", va="center", fontsize=10.5, color=WIN, fontweight="bold")
        ax.set_xlim(0, top * 1.45)
        ax.set_ylim(8.9, -0.8)
        ax.set_yticks([])
        ax.set_xticks([])
        clean(ax)
        ax.set_title(t, loc="left", fontsize=12, color=INK, fontweight="bold")
    c = axes[2]
    for i, f in enumerate(FORMAT):
        r = f[3]
        col = LOSS if r > 1.002 else TIE
        c.barh(i, (r - 1) * 100, height=0.6, color=col)
        c.text((r - 1) * 100 + (0.08 if r >= 1 else -0.08), i, f"{(r - 1) * 100:+.2f}%", va="center",
               ha="left" if r >= 1 else "right", fontsize=10.5, color=col, fontweight="bold")
        c.text(0.05, i + (0.42 if col == LOSS else 0), "GGUF smaller" if col == LOSS else "tie", va="center", ha="left", fontsize=10.5,
               color=col, fontweight="bold")
    c.axvline(0, color=MUTED, lw=1)
    c.set_yticks(range(len(FORMAT)), names)
    c.invert_yaxis()
    c.set_xlim(-0.8, 3.2)
    c.set_xticks([0, 1, 2, 3], ["0%", "+1%", "+2%", "+3%"])
    clean(c, keep_bottom=True)
    c.set_title("File size, SSF against GGUF\n(right of 0 = SSF bigger)", loc="left", fontsize=12, color=INK,
                fontweight="bold")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SHEEP), plt.Rectangle((0, 0), 1, 1, color=LLAMA)],
               labels=["SSF (Sheep)", "GGUF Q4_K_M (llama.cpp and Ollama)"], loc="lower left", ncol=2,
               frameon=False, bbox_to_anchor=(0.015, 0.0))
    size_l = sum(f[3] > 1.002 for f in FORMAT)
    score(fig, [("SSF better", 2 * len(FORMAT), WIN), ("GGUF better", size_l, LOSS),
                ("tie", len(FORMAT) - size_l, TIE)], 3 * len(FORMAT))
    fig.text(0.985, 0.02, "RESULTS.md, \"Three specs raced\": every position of 256-512 held-out tokens scored",
             ha="right", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.7, bottom=0.12, wspace=0.42)
    save(fig, out, "ssf_vs_gguf.png")


# What each does besides running a model. Yes / part / no; "part" says what is missing in the note.
FEATURES = [
    ("Desktop app: chat window, no terminal", "yes", "part", "yes"),
    ("Makes documents, workbooks and slide decks", "yes", "no", "no"),
    ("Makes pictures and video clips", "yes", "no", "no"),
    ("Video editor (Studio: crop, split, fades, ads)", "yes", "no", "no"),
    ("Reads attached videos, pictures, documents", "yes", "part", "part"),
    ("Speech in (mic) and spoken replies", "yes", "no", "no"),
    ("Web answers with sources", "yes", "no", "part"),
    ("Its own 4-bit format and quantizer", "yes", "yes", "no"),
    ("Fits any model to this computer by itself", "yes", "part", "part"),
    ("Checks a model's answers before you use it", "yes", "no", "no"),
    ("Pauses at 99 °C instead of letting the laptop shut down", "yes", "no", "no"),
    ("Optional cloud help (NVIDIA Nemotron on Nebius)", "yes", "no", "part"),
    ("Big ready-made model library to pull from", "part", "part", "yes"),
    ("Runs on Mac and Linux too", "no", "yes", "yes"),
]
MARK = {"yes": ("✓", WIN), "part": ("◐", "#9a6700"), "no": ("–", "#afb8c1")}


def features(out):
    fig, ax = plt.subplots(figsize=(13, 8.6), facecolor=BG)
    head(fig, "What else each one does", "Beyond running a model. ✓ yes   ◐ partly or with add-ons   – no. "
         "Rows where the others are ahead are listed too.")
    cols = [("Sheep", SHEEP), ("llama.cpp", LLAMA), ("Ollama", OLLAMA)]
    n = len(FEATURES)
    for i, (feat, *marks) in enumerate(FEATURES):
        y = n - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f6f8fa", zorder=0)
        ax.text(0.0, y, feat, va="center", fontsize=11.5, color=INK)
        for j, m in enumerate(marks):
            sym, col = MARK[m]
            ax.text(6.4 + j * 1.6, y, sym, va="center", ha="center", fontsize=17, color=col, fontweight="bold")
    for j, (name, col) in enumerate(cols):
        ax.text(6.4 + j * 1.6, n - 0.1, name, ha="center", va="bottom", fontsize=12.5, color=col, fontweight="bold")
    lead = sum(m[0] == "yes" and "yes" not in m[1:] for _, *m in FEATURES)
    behind = sum(m[0] != "yes" and "yes" in m[1:] for _, *m in FEATURES)
    fig.text(0.02, 0.86, f"Sheep alone has {lead} of these {len(FEATURES)}. The others are ahead on {behind}.",
             fontsize=13, color=WIN, fontweight="bold")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.7, n + 0.6)
    ax.axis("off")
    fig.text(0.02, 0.02, "llama.cpp: llama-server has a basic web page; reads pictures with multimodal models. "
             "Ollama: desktop app and web search (account needed); cloud models through ollama.com.",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.84, bottom=0.06)
    save(fig, out, "features.png")


HW_LOG = Path(__file__).resolve().parent / "hw_race.jsonl"   # tooling/market_race.py rows, copied here


def hardware(out):
    """How much of the machine each engine takes to do the same work: less memory, CPU and power
    leaves room for a bigger model, other apps, and battery."""
    import json
    rows = {}
    for line in HW_LOG.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        name = {"sheep": "Sheep", "llamacpp": "llama.cpp", "ollama": "Ollama"}[r["kind"]]
        rows[(name, r["device"])] = r
    panels = [
        ("Memory (RAM) at its peak\n(MB, lower is better)", "peak_ram_mb", "MB", "{:.0f}"),
        ("Graphics memory at its peak\n(MB, lower is better)", "peak_vram_mb", "MB", "{:.0f}"),
        ("CPU busy while answering\n(% of the whole CPU, lower is better)", "avg_cpu_pct", "%", "{:.0f}"),
        ("Graphics card power\n(watts, lower is better)", "avg_gpu_w", "W", "{:.1f}"),
    ]
    engines = [("Sheep", SHEEP), ("llama.cpp", LLAMA), ("Ollama", OLLAMA)]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 6.0), facecolor=BG)
    head(fig, "Hardware used for the same work",
         f"{MACHINE}. Same prompt and answer length for each. Less used = more room for a bigger model or other apps.")
    wins = []
    for ax, (t, k, unit, fmt) in zip(axes, panels):
        top = max(rows[(e, d)].get(k) or 0 for e, _ in engines for d in ("gpu", "cpu")) or 1
        for i, dev in enumerate(("gpu", "cpu")):
            vals = [(e, c, rows[(e, dev)].get(k) or 0) for e, c in engines]
            for j, (e, c, v) in enumerate(vals):
                y = i * 4.2 + j
                ax.barh(y, v, height=0.8, color=c)
                ax.text(v + top * 0.02, y, f"{fmt.format(v)} {unit}", va="center", fontsize=10, color=INK)
            best = min(v for _, _, v in vals)
            slack = {"%": 0.5, "MB": 50, "W": 0.3}[unit]     # a few MB or a tenth of a watt is noise
            lead = [e for e, _, v in vals if v <= best * 1.02 + slack]
            if len(lead) == len(vals):
                who, word, c = "tie", "Tie", TIE
            elif "Sheep" in lead:
                who, word, c = "Sheep", "✓ Sheep uses least" + (" (tie)" if len(lead) > 1 else ""), WIN
            else:
                who, word, c = lead[0], f"{' and '.join(lead)} use{'s' if len(lead) == 1 else ''} least", LOSS
            wins.append(who)
            ax.text(0, i * 4.2 + 2.85, word, va="center", fontsize=10.5, color=c, fontweight="bold")
            ax.text(-top * 0.03, i * 4.2 + 1, dev.upper(), va="center", ha="right", fontsize=11, color=INK,
                    fontweight="bold")
        ax.set_xlim(0, top * 1.5)
        ax.set_ylim(7.6, -0.8)
        ax.set_yticks([])
        ax.set_xticks([])
        clean(ax)
        ax.set_title(t, loc="left", fontsize=12, color=INK, fontweight="bold")
    score(fig, [("Sheep uses least", wins.count("Sheep"), WIN),
                ("another uses less", sum(w not in ("Sheep", "tie") for w in wins), LOSS),
                ("tie", wins.count("tie"), TIE)], len(wins))
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c in engines], labels=[e for e, _ in engines],
               loc="lower left", ncol=3, frameon=False, bbox_to_anchor=(0.015, 0.0))
    fig.text(0.985, 0.02, "tooling/market_race.py: memory of the engine's whole process tree; card memory counted "
             "from before the engine started; CPU and power averaged over the runs", ha="right", fontsize=9,
             color=MUTED)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.7, bottom=0.1, wspace=0.3)
    save(fig, out, "hardware.png")


def main(out):
    style()
    if HW_LOG.exists():
        hardware(out)
    versus(out, "llama.cpp", LLAMA, "sheep_vs_llamacpp.png")
    versus(out, "Ollama", OLLAMA, "sheep_vs_ollama.png")
    engines(out)
    formats(out)
    features(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "mlmi" / "media" / "bench"))
