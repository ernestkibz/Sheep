"""The Sheep workspace: where everything is saved, and how every change is undone.

Every path a tool or an editor names is resolved INSIDE the workspace root; a
path that climbs out of it is refused, whoever asked. Before any file is
written, moved or deleted its previous version is copied into
.mlmi/history, and the change goes on one journal -- so the model's edits and
the person's clicks share one undo, and nothing the agent does is final.
"""
import json
import os
import shutil
import threading
import time
from pathlib import Path


class ToolError(Exception):
    """A refusal or failure the model is told about, in words it can act on."""


class ModelSelectionRequired(ToolError):
    """Pause a feature action until its local model is explicitly checked and used."""
    def __init__(self, slot, repo_id, message=""):
        self.slot = str(slot)
        self.repo_id = str(repo_id)
        super().__init__(message or f"Select and use a compatible {self.slot} model ({self.repo_id}) to continue.")


HIDDEN = ".mlmi"


class Workspace:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = self.root / HIDDEN
        self.hist = self.meta / "history"
        self.hist.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.meta / "journal.jsonl"
        self.lock = threading.RLock()
        self.undone = []          # redo stack of journal entries
        self.listeners = []       # fn(rel_path) on every change, for live editors

    # ------------------------------------------------------------ paths
    def path(self, rel, must_exist=False):
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("a file path is required")
        p = (self.root / rel.strip().lstrip("/\\")).resolve()
        try:
            inside = p.relative_to(self.root)
        except ValueError:
            raise ToolError(f"'{rel}' is outside the workspace; only files under {self.root.name} can be used")
        if inside.parts and inside.parts[0] == HIDDEN:
            raise ToolError("the history folder cannot be edited directly; use undo")
        if must_exist and not p.exists():
            raise ToolError(f"'{rel}' does not exist (use list_files to see what is there)")
        return p

    def rel(self, p):
        return Path(p).resolve().relative_to(self.root).as_posix()

    # ------------------------------------------------------------ history
    def _snap(self, p):
        if not p.exists() or p.is_dir():
            return None
        d = self.hist / self.rel(p)
        d.mkdir(parents=True, exist_ok=True)
        snap = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{time.perf_counter_ns() % 1000000:06d}{p.suffix}"
        shutil.copy2(p, snap)
        return snap.relative_to(self.meta).as_posix()

    def _journal(self, entry):
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _entries(self):
        if not self.journal_path.exists():
            return []
        with open(self.journal_path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def _rewrite(self, entries):
        tmp = self.journal_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        os.replace(tmp, self.journal_path)

    def changing(self, *paths, by="user", what=""):
        """Context manager: snapshot `paths` before the change, journal it after."""
        ws = self

        class _C:
            def __enter__(self_):
                ws.lock.acquire()
                self_.snaps = [(ws.rel(p), ws._snap(p), p.exists()) for p in paths]
                return self_

            def __exit__(self_, et, ev, tb):
                try:
                    if et is None:
                        ws._journal({"t": time.time(), "by": by, "what": what,
                                     "files": [{"path": r, "snap": s, "existed": e} for r, s, e in self_.snaps]})
                        ws.undone.clear()
                        for r, _, _ in self_.snaps:
                            ws._notify(r)
                finally:
                    ws.lock.release()
                return False
        return _C()

    def _notify(self, rel):
        for fn in list(self.listeners):
            try:
                fn(rel)
            except Exception:
                pass

    def _restore(self, entry):
        """Put the files of one journal entry back; return the entry that reverses it."""
        back = {"t": time.time(), "by": entry.get("by"), "what": entry.get("what"), "files": []}
        for f in entry["files"]:
            p = self.root / f["path"]
            cur = self._snap(p)
            back["files"].append({"path": f["path"], "snap": cur, "existed": p.exists()})
            if f["snap"]:
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.meta / f["snap"], p)
            elif not f["existed"] and p.exists():
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
            self._notify(f["path"])
        return back

    def undo(self):
        with self.lock:
            entries = self._entries()
            if not entries:
                return None
            last = entries.pop()
            self._rewrite(entries)
            self.undone.append(self._restore(last))
            return last

    def redo(self):
        with self.lock:
            if not self.undone:
                return None
            e = self.undone.pop()
            back = self._restore(e)
            self._journal(back)
            return e

    def restore_before(self, t, rel, by="user"):
        """Put one file back as it was just before the change journalled at time t.
        The restore is itself a change, so undo takes it back."""
        e = next((e for e in reversed(self._entries()) if abs(float(e.get("t", 0)) - float(t)) < 1e-3
                  and any(f["path"] == rel for f in e["files"])), None)
        if not e:
            raise ToolError(f"no change to {rel} at that time")
        f = next(f for f in e["files"] if f["path"] == rel)
        p = self.path(rel)
        if not f["snap"] and f["existed"]:
            raise ToolError(f"that version of {rel} was not kept")
        with self.changing(p, by=by, what=f"restore {rel} to before: {e.get('what') or 'a change'}"):
            if f["snap"]:
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.meta / f["snap"], p)
            elif p.exists():                 # the change created the file: before it, there was none
                p.unlink()
        return f"restored {rel}"

    def history(self, rel=None, limit=50):
        out = [e for e in self._entries() if rel is None or any(f["path"] == rel for f in e["files"])]
        return out[-limit:]

    # ------------------------------------------------------------ listing
    def tree(self, folder="."):
        base = self.path(folder) if folder not in (".", "") else self.root
        if not base.exists():
            raise ToolError(f"'{folder}' does not exist")
        out = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != HIDDEN and not d.startswith("."))
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                out.append({"path": self.rel(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime})
            for d in dirnames:
                out.append({"path": self.rel(Path(dirpath) / d) + "/", "size": 0, "mtime": 0})
        return sorted(out, key=lambda x: x["path"])
