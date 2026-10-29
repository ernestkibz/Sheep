"""Auto-fit across Hugging Face: the most-downloaded text-generation models, each checked the way Sheep checks a
model someone adds (its config and file tables read, every tensor placed, the tokenizer fitted, its family's
maths matched against transformers). Writes tooling/model_sweep.jsonl and prints the tally.

    python tooling/model_sweep.py [N]        (N most-downloaded, default 300)
"""
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
ids = [m.id for m in HfApi().list_models(pipeline_tag="text-generation", library="safetensors",
                                          sort="downloads", limit=N)]


def original_of(repo):
    """What the app builds from: a compressed copy (GPTQ, AWQ, bitsandbytes, FP8, GGUF, MLX) is traced back to
    its original by mlmi.backend.hf_source.resolve, the same call the app makes when someone adds it."""
    try:
        from mlmi.backend.hf_source import resolve
        src, note, kind = resolve(repo)
        return src if src != repo else None
    except Exception:
        return None


def check(repo, traced=False):
    try:
        r = subprocess.run([sys.executable, str(HERE / "sml_compat.py"), "--repo", repo], capture_output=True,
                           text=True, timeout=240, encoding="utf-8", errors="replace")
        line = next((l for l in r.stdout.splitlines() if l.startswith(repo + ":")), "") or (r.stdout + r.stderr).strip()[-300:]
    except subprocess.TimeoutExpired:
        line = f"{repo}: timed out reading its headers"
    verdict = line.split(":", 1)[1].strip() if ":" in line else line
    if "can build it" in verdict:
        kind = "runs natively"
    elif re.search(r"compatib", verdict, re.I):
        kind = "runs in compatibility mode"
    elif re.search(r"gated|401|403|access|token", verdict, re.I):
        kind = "gated (needs the maker's approval)"
    else:
        kind = "not yet"
    if kind == "not yet" and not traced and "no place in the name table" in verdict:
        src = original_of(repo)
        if src:
            r2 = check(src, traced=True)
            if r2["kind"] == "runs natively":
                return {"repo": repo, "kind": "runs natively (built from its original)", "from": src, "verdict": r2["verdict"]}
    return {"repo": repo, "kind": kind, "verdict": verdict[:400]}


prev = {}
if "--retry" in sys.argv and (HERE / "model_sweep.jsonl").exists():
    prev = {r["repo"]: r for r in map(json.loads, (HERE / "model_sweep.jsonl").read_text(encoding="utf-8").splitlines()) if r}
    ids = [r for r in prev]
todo = [r for r in ids if not prev or prev[r]["kind"] == "not yet"]
with ThreadPoolExecutor(6) as ex:
    done = dict(zip(todo, ex.map(check, todo)))
rows = [done.get(r) or prev[r] for r in ids]
(HERE / "model_sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
c = Counter(r["kind"] for r in rows)
print(f"{len(rows)} models checked:", dict(c))
for r in rows:
    if r["kind"] == "not yet":
        print(" ", r["repo"], "-", r["verdict"][:140])
