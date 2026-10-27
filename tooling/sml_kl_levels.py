"""Measure Sheep quantization quality against the matching full-precision model.

For every level, run the independent numpy forward pass on the same token IDs
against the original checkpoint and the SSF file, then report KL(p_fp || p_ssf).
Missing levels can be built from local original weights with Sheep's quantizer.

    python tooling/sml_kl_levels.py MODEL.ssf --originals ORIGINALS_DIR
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

DEFAULT_TOKENS = (1, 11, 358, 1079)
LEVEL_ALIASES = {
    "Q2": "q2", "Q3": "q3", "Q4": "small", "Q4/6": "balanced",
    "Q5": "q5", "Q6": "high", "Q7": "q7", "Q8": "q8", "EXACT": "exact",
}


def _log_softmax(values):
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    if x.ndim not in (1, 2) or not x.size or not np.isfinite(x).all():
        raise ValueError("the logits must be a non-empty finite vector or matrix")
    top = np.max(x, axis=-1, keepdims=True)
    return x - top - np.log(np.exp(x - top).sum(axis=-1, keepdims=True))


def kl_from_logits(reference, quantized):
    """Exact categorical KL in nats; average rows for sequence logits.

    For matrices, each row is the next-token distribution after one prefix in
    the same input sequence. Averaging keeps the result one interpretable score
    while measuring every prefix instead of just its final token.
    """
    import numpy as np
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(quantized, dtype=np.float64)
    if ref.shape != got.shape:
        if ref.ndim == got.ndim and ref.shape[:-1] == got.shape[:-1]:
            raise ValueError(f"reference and quantized vocabularies differ: {ref.shape[-1]} != {got.shape[-1]}")
        raise ValueError(f"reference and quantized logits differ: {ref.shape} != {got.shape}")
    if ref.ndim not in (1, 2) or not ref.size or not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError("the logits must have matching finite vector or matrix shapes")
    p_log = _log_softmax(ref)
    q_log = _log_softmax(got)
    p = np.exp(p_log)
    row_kl = np.sum(p * (p_log - q_log), axis=-1, dtype=np.float64)
    value = float(np.mean(row_kl, dtype=np.float64))
    # Small negative values can arise from floating-point summation only.
    return max(0.0, value)


def kl_measure(ssf, originals, tokens=DEFAULT_TOKENS):
    """Mean next-token KL across prefixes of the same full-precision input."""
    from sml_logits_check import reference_logits
    started = time.perf_counter()
    fp_logits = reference_logits(str(originals), list(tokens), all_positions=True)
    quant_logits = reference_logits(str(ssf), list(tokens), depth=0, all_positions=True)
    elapsed = time.perf_counter() - started
    return kl_from_logits(fp_logits, quant_logits), elapsed


def speed_measure(ssf, gen=128, threads=0, tokens=DEFAULT_TOKENS):
    """Measure prompt and decode throughput with the project's native CLI."""
    cli = REPO / "sml" / ("sml-cli.exe" if os.name == "nt" else "sml-cli")
    cmd = [str(cli), str(ssf), "--tokens", ",".join(map(str, tokens)), "--gen", str(gen),
           "--threads", str(threads or os.cpu_count() or 6), "--ctx", str(max(512, len(tokens) + gen + 8))]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode:
        raise RuntimeError(output[-500:].strip() or f"sml-cli exited {proc.returncode}")
    prompt = re.search(r"prompt\s+(\d+)\s+tokens\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+tok/s\)", output)
    decode = re.search(r"generate\s+(\d+)\s+tokens\s+in\s+([\d.]+)s", output)
    if not prompt or not decode:
        raise RuntimeError("sml-cli did not report prompt and decode timing")
    prompt_count, prompt_seconds = int(prompt.group(1)), float(prompt.group(2))
    decode_count, decode_seconds = int(decode.group(1)), float(decode.group(2))
    return {
        "prompt_tok_s": round(prompt_count / prompt_seconds, 2) if prompt_seconds else None,
        "throughput_tok_s": round(decode_count / decode_seconds, 2) if decode_seconds else None,
        "prompt_tokens": prompt_count,
        "decode_tokens": decode_count,
    }


def _level_id(value):
    raw = value.strip()
    return LEVEL_ALIASES.get(raw.upper(), raw.lower())


def _build_level(originals, level, output):
    """Build one missing level through the exact quantizer recipe used by Settings."""
    from mlmi.backend.levels import LEVELS, inspect
    from mlmi.backend.quantizer import recipe
    info = inspect(str(originals))
    if not info.get("builds"):
        raise RuntimeError(f"the source architecture cannot be built: {info.get('note', 'unsupported')}")
    if level not in {item["id"] for item in LEVELS}:
        raise ValueError(f"unsupported quantization level: {level}")
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    part.unlink(missing_ok=True)
    cmd = recipe(str(originals), str(part), info["n_layer"], 0, info.get("licence", ""), level)
    completed = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=24 * 3600,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if completed.returncode or not part.is_file():
        part.unlink(missing_ok=True)
        raise RuntimeError((completed.stdout + "\n" + completed.stderr)[-1500:].strip()
                           or f"quantizer exited with status {completed.returncode}")
    os.replace(part, output)
    return output


def measure_levels(ssf, originals=None, levels=None, tokens=DEFAULT_TOKENS, build_missing=True,
                   gen=128, threads=0):
    """Return measured KL, on-disk bytes, prompt speed and decode speed per level."""
    source = Path(originals).expanduser().resolve() if originals else None
    if source and not source.is_dir():
        raise ValueError("--originals must be a local folder containing the full-precision safetensors")
    requested = levels or ["q2", "q3", "small", "balanced", "q5", "high", "q7", "q8", "exact"]
    results = []
    base = Path(ssf).expanduser().resolve()
    for value in requested:
        level = _level_id(value)
        if level == "balanced" and value.lower() == "q4":
            level = "small"
        level_file = base.with_name(f"{base.stem}-{level}{base.suffix}")
        row = {"level": level, "tokens": list(tokens)}
        try:
            if not level_file.is_file() and level_file.resolve() != base.resolve():
                if source is None or not build_missing:
                    raise FileNotFoundError("level file not found; provide --originals to build it")
                _build_level(source, level, level_file)
            if not level_file.is_file():
                raise FileNotFoundError(f"no SSF file for level {level}: {level_file}")
            kl, compare_s = kl_measure(level_file, source, tokens) if source else (None, None)
            speed = speed_measure(level_file, gen=gen, threads=threads, tokens=tokens)
            row.update({"kl": kl, "kl_measured": kl is not None,
                        "kl_seconds": round(compare_s, 2) if compare_s is not None else None,
                        "bytes": level_file.stat().st_size, **speed})
            if kl is None:
                row["kl_note"] = "not measured: pass --originals for the matching full-precision checkpoint"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        results.append(row)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ssf", help="reference SSF path; sibling level files use its stem")
    ap.add_argument("--originals", default="", help="local full-precision checkpoint folder (required for KL)")
    ap.add_argument("--levels", default="q2,q3,small,balanced,q5,high,q7,q8,exact",
                    help="comma-separated levels: q2,q3,small,balanced,q5,high,q7,q8,exact")
    ap.add_argument("--tokens", default=",".join(map(str, DEFAULT_TOKENS)),
                    help="same comma-separated token IDs are run against both checkpoints")
    ap.add_argument("--gen", type=int, default=128, help="decode tokens used for throughput measurement")
    ap.add_argument("--threads", type=int, default=0, help="CLI threads; 0 uses available CPUs")
    ap.add_argument("--no-build", action="store_true", help="report missing levels instead of building them")
    ap.add_argument("--out", default="kl_levels.json", help="where to write the result JSON")
    args = ap.parse_args()
    tokens = [int(token) for token in args.tokens.split(",") if token.strip()]
    if not tokens:
        ap.error("--tokens must contain at least one integer")
    results = measure_levels(args.ssf, args.originals or None, args.levels.split(","), tokens,
                             not args.no_build, args.gen, args.threads)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 1 if any("error" in row for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
