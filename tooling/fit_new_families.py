"""Auto-fit every family the model sweep found unknown: one real checkpoint's config and tokenizer per family
(kilobytes), then sml_autofit fit. Run with a python whose transformers knows the families (venv-tf5).

    python tooling/fit_new_families.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (HERE / "model_sweep.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
fams = {}
for r in rows:
    m = re.search(r"'([\w.]+)' is not in the roster", r["verdict"])
    if m and r["kind"] == "not yet":
        fams.setdefault(m.group(1), r["repo"])
env = dict(os.environ, SML_BUILD_PYTHON=os.environ.get("SML_BUILD_PYTHON", sys.executable))
for fam, repo in fams.items():
    d = Path(tempfile.mkdtemp(prefix="fit_")) / fam
    d.mkdir(parents=True)
    try:
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            try:
                shutil.copy(hf_hub_download(repo, f), d / f)
            except Exception:
                if f in ("config.json", "tokenizer.json"):
                    raise
    except Exception as e:
        print(f"{fam}: could not read {repo} ({str(e)[:80]})", flush=True)
        continue
    r = subprocess.run([sys.executable, str(HERE / "sml_autofit.py"), "fit", str(d)], capture_output=True, text=True,
                       env=env, timeout=1800, encoding="utf-8", errors="replace")
    res = next((l[7:] for l in r.stdout.splitlines() if l.startswith("RESULT ")), "")
    try:
        j = json.loads(res)
        print(f"{fam}: {'FITTED ' + json.dumps(j.get('flags')) if j.get('pass') else 'not fitted: ' + str(j.get('why'))[:160]}", flush=True)
    except ValueError:
        print(f"{fam}: no result ({(r.stdout + r.stderr)[-160:]})", flush=True)
