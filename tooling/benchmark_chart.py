"""The benchmark picture for the README and Devpost: Sheep against llama.cpp b11901 on one laptop.

    python tooling/benchmark_chart.py [OUT.png]

Left: how many times as fast Sheep is per test (median of 8 ABBA rounds, bar = 95% bootstrap range;
right of 1 = Sheep faster). Right: tokens per second on the CPU for Sheep, llama.cpp and Ollama
(one timed run each, same prompt). Every number is a measured result, copied from the race logs.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (test, ratio, ci_low, ci_high) from tooling/sml_cpu_race_ci.py and sml_gpu_race_ci.py, Qwen2.5-3B-Instruct,
# Sheep SSF (1.84 GiB) against llama.cpp b11901 Q4_K_M (1.80 GiB), Ryzen 5 5600H / RTX 3050 Laptop 4 GB
RACES = [
    ("CPU · read 128-token prompt", 1.395, 1.318, 1.462),
    ("CPU · read 512-token prompt", 1.257, 1.210, 1.315),
    ("CPU · write 128 tokens", 1.001, 0.986, 1.005),
    ("GPU · read 128-token prompt", 0.991, 0.988, 1.006),
    ("GPU · read 512-token prompt", 1.083, 1.082, 1.087),
    ("GPU · write 64 tokens", 0.986, 0.986, 0.988),
]
# tooling/market_race.py, CPU, 6 threads, ~710-token prompt, 128 new tokens (tokens per second)
ENGINES = [("Sheep", 97.9, 15.65), ("llama.cpp b11901", 80.8, 15.28), ("Ollama 0.40.11", 87.4, 15.13)]

INK, MUTED, GRID, BG = "#1f2328", "#6e7781", "#d8dee4", "#ffffff"
WIN, TIE, LOSS = "#1a7f37", "#8c959f", "#cf222e"
SHEEP, OTHER = "#0b5cad", "#a8b1bb"


def verdict(lo, hi):
    return WIN if lo > 1 else LOSS if hi < 1 else TIE


def main(out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.35, 1]}, facecolor=BG)
    fig.suptitle("Sheep against llama.cpp on one laptop (Qwen2.5-3B, 4-bit)", x=0.02, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, "Ryzen 5 5600H, RTX 3050 Laptop 4 GB. Sheep's SSF 1.84 GiB vs llama.cpp b11901 Q4_K_M 1.80 GiB.",
             fontsize=10, color=MUTED)

    ys = list(range(len(RACES)))[::-1]
    for y, (name, r, lo, hi) in zip(ys, RACES):
        c = verdict(lo, hi)
        a.plot([lo, hi], [y, y], color=c, lw=7, alpha=0.35, solid_capstyle="round")
        a.plot([r], [y], "o", color=c, ms=9, mec=BG, mew=1.5)
        word = "Sheep faster" if c == WIN else "llama.cpp faster" if c == LOSS else "tie"
        num = f"{r:.3f}" if abs(r - 1) < 0.02 else f"{r:.2f}"
        a.text(1.48, y, f"{num}×  {word}", va="center", fontsize=10, color=c, fontweight="bold")
    a.axvline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    a.set_yticks(ys, [n for n, *_ in RACES])
    a.set_xlim(0.85, 1.75)
    a.set_xticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    a.set_xticklabels(["0.9×", "1.0×", "1.1×", "1.2×", "1.3×", "1.4×"])
    a.grid(axis="x", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.set_title("How many times as fast as llama.cpp\n(median of 8 rounds, bar = 95% range; crossing 1 = tie)",
                loc="left", fontsize=11, color=INK)

    names = [n for n, *_ in ENGINES]
    for i, (n, pp, tg) in enumerate(ENGINES):
        b.barh(i, pp, height=0.55, color=SHEEP if n == "Sheep" else OTHER)
        b.text(pp + 1.5, i, f"{pp:.1f} tok/s", va="center", fontsize=10, color=INK)
    b.set_yticks(range(len(ENGINES)), names)
    b.invert_yaxis()
    b.set_xticks([])
    b.set_xlim(0, 125)
    for sp in ("top", "right", "bottom", "left"):
        b.spines[sp].set_visible(False)
    writing = ", ".join(f"{n.split()[0]} {tg:.1f}" for n, _, tg in ENGINES)
    b.set_title("Reading a 710-token prompt on the CPU\n(one timed run each, same model)\n"
                f"Writing was a tie:\n{writing} tok/s", loc="left", fontsize=10.5, color=INK)

    fig.text(0.02, 0.015, "Races: tooling/sml_cpu_race_ci.py, tooling/sml_gpu_race_ci.py (ABBA order, bootstrap CI).  "
             "Engines: tooling/market_race.py.  All runs 2026-10-26/27, on AC power.", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.74, bottom=0.11, wspace=0.42)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=BG)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "mlmi" / "media" / "benchmark.png"))
