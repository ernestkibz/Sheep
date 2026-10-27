"""Append reproducible benchmark summaries to the user's local run history."""
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def default_path():
    return Path(os.environ.get("MLMI_HOME", Path.home() / ".mlmi")) / "benchmark-runs.jsonl"


def file_facts(path):
    p = Path(path).expanduser().resolve()
    try:
        st = p.stat()
    except OSError:
        return {"path": str(p), "available": False}
    return {"path": str(p), "available": True, "bytes": st.st_size,
            "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}


def revision():
    root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=3)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def record(path, details):
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": 1,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "processor": platform.processor(),
                 "machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "sheep_revision": revision(),
        **details,
    }
    with target.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return target
