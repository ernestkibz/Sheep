"""The benchmark picture for the README and Devpost: Sheep against llama.cpp and Ollama on one laptop.

    python tooling/benchmark_chart.py [OUT.png]

Left: how many times as fast Sheep is than llama.cpp b11901 per test (median of 8 ABBA rounds, bar = 95%
bootstrap range; right of 1 = Sheep faster). Middle: prompt-reading speed of the three engines on the GPU
and the CPU. Right: answers right out of 40 hard questions and Sheep's 8 checks. Every number is a
measured result, copied from the race logs (2026-10-26/27, Ryzen 5 5600H, RTX 3050 Laptop 4 GB, on AC).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# tooling/sml_cpu_race_ci.py and sml_gpu_race_ci.py: (test, ratio, ci_low, ci_high)
RACES = [
    ("CPU · read 128-token prompt", 1.395, 1.318, 1.462),
    ("CPU · read 512-token prompt", 1.257, 1.210, 1.315),
    ("CPU · write 128 tokens", 1.001, 0.986, 1.005),
    ("GPU · read 128-token prompt", 0.991, 0.988, 1.006),
    ("GPU · read 512-token prompt", 1.083, 1.082, 1.087),
    ("GPU · write 64 tokens (cool card)", 0.986, 0.986, 0.988),
    ("GPU · write 64 tokens (hot card, 86-94 °C)", 1.391, 1.390, 1.399),
]
# tooling/market_race.py: each engine as a server, ~710-token prompt, 128 new tokens, median of 5 runs;
# accuracy_eval.py's 40 questions at temperature 0; Sheep's 8 checks (all engines passed 8/8)
ENGINES = [  # name, GPU read, CPU read, GPU write, CPU write, right on GPU, right on CPU
    ("Sheep", 361.0, 72.4, 11.4, 15.1, 33, 32),
    ("llama.cpp b11901", 380.1, 62.7, 11.7, 15.2, 31, 32),
    ("Ollama 0.40.11", 383.7, 81.0, 12.0, 15.7, 32, 33),
]

INK, MUTED, GRID, BG = "#1f2328", "#6e7781", "#d8dee4", "#ffffff"
WIN, TIE, LOSS = "#1a7f37", "#8c959f", "#cf222e"
SHEEP, OTHER = "#0b5cad", "#a8b1bb"


def verdict(lo, hi):
    return WIN if lo > 1 else LOSS if hi < 1 else TIE


def clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main(out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1.5, 1, 0.8]}, facecolor=BG)
    fig.suptitle("Sheep against llama.cpp and Ollama on one laptop (Qwen2.5-3B, 4-bit)", x=0.015, ha="left",
                 fontsize=16, fontweight="bold", color=INK)
    fig.text(0.015, 0.905, "Ryzen 5 5600H, RTX 3050 Laptop 4 GB, on AC. Sheep's SSF 1.84 GiB; llama.cpp and Ollama "
             "run the same Q4_K_M GGUF, 1.80 GiB.", fontsize=10.5, color=MUTED)

    ys = list(range(len(RACES)))[::-1]
    for y, (name, r, lo, hi) in zip(ys, RACES):
        col = verdict(lo, hi)
        a.plot([lo, hi], [y, y], color=col, lw=7, alpha=0.35, solid_capstyle="round")
        a.plot([r], [y], "o", color=col, ms=9, mec=BG, mew=1.5)
        word = "Sheep faster" if col == WIN else "llama.cpp faster" if col == LOSS else "tie"
        num = f"{r:.3f}" if abs(r - 1) < 0.02 else f"{r:.2f}"
        a.text(1.47, y, f"{num}×  {word}", va="center", fontsize=10, color=col, fontweight="bold")
    a.axvline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    a.set_yticks(ys, [n for n, *_ in RACES])
    a.set_xlim(0.85, 1.8)
    a.set_xticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    a.set_xticklabels(["0.9×", "1.0×", "1.1×", "1.2×", "1.3×", "1.4×"])
    a.grid(axis="x", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    clean(a)
    a.set_title("Sheep vs llama.cpp, head to head\n(median of 8 rounds, bar = 95% range; crossing 1 = tie)",
                loc="left", fontsize=11, color=INK)

    names = [n for n, *_ in ENGINES]
    for i, (n, gr, cr, gw, cw, rg, rc) in enumerate(ENGINES):
        col = SHEEP if n == "Sheep" else OTHER
        b.barh(i - 0.19, gr, height=0.36, color=col)
        b.barh(i + 0.19, cr * 4, height=0.36, color=col, alpha=0.5)
        b.text(gr + 6, i - 0.19, f"GPU {gr:.0f} tok/s", va="center", fontsize=9.5, color=INK)
        b.text(cr * 4 + 6, i + 0.19, f"CPU {cr:.1f} tok/s", va="center", fontsize=9.5, color=INK)
    b.set_yticks(range(len(ENGINES)), names)
    b.invert_yaxis()
    b.set_xticks([])
    b.set_xlim(0, 560)
    clean(b)
    b.spines["bottom"].set_visible(False)
    b.set_title("Reading a 710-token prompt\n(CPU bars at 4× scale; writing: all within 0.6 tok/s,\n"
                "GPU 11.4-12.0, CPU 15.1-15.7)", loc="left", fontsize=10.5, color=INK)

    for i, (n, gr, cr, gw, cw, rg, rc) in enumerate(ENGINES):
        col = SHEEP if n == "Sheep" else OTHER
        c.barh(i - 0.19, rg, height=0.36, color=col)
        c.barh(i + 0.19, rc, height=0.36, color=col, alpha=0.5)
        c.text(rg + 0.5, i - 0.19, f"{rg}/40 GPU", va="center", fontsize=9.5, color=INK)
        c.text(rc + 0.5, i + 0.19, f"{rc}/40 CPU", va="center", fontsize=9.5, color=INK)
    c.set_yticks(range(len(ENGINES)), [""] * len(ENGINES))
    c.invert_yaxis()
    c.set_xticks([])
    c.set_xlim(0, 52)
    clean(c)
    c.spines["bottom"].set_visible(False)
    c.set_title("Hard questions right, of 40\n(math, code, facts, format; temp 0)\nAll passed Sheep's 8 checks",
                loc="left", fontsize=10.5, color=INK)

    fig.text(0.015, 0.015, "Races: tooling/sml_cpu_race_ci.py, sml_gpu_race_ci.py (ABBA order, bootstrap CI). Engines: "
             "tooling/market_race.py (each engine's own counters). Sheep GPU reading measured after the 2026-10-27 "
             "engine fix (was 13.8 tok/s).", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.74, bottom=0.09, wspace=0.32)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=BG)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "mlmi" / "media" / "benchmark.png"))
