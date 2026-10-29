"""The quantizer behind Settings: an original checkpoint in, a Sheep file out.

Originals first -- a safetensors folder or a Hugging Face id: the quality comes from building
off the original weights (Ernest, 2026-09-30). A GGUF is converted back to an original-format
folder only when there is no original to build from: a .gguf file on this computer, or a
fine-tune published only as GGUF (Ernest, 2026-10-21); GPTQ and AWQ copies and Meta's raw
.pth checkpoints the same way (Part C6).
One recipe for every model, the one that won the pilot races (STANDARDS.md
section 3). One build at a time; progress is read from the builder's own lines.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .paths import HOME, REPO

TOOLING = REPO / "tooling"
OUT_DIR = HOME / "models"


def gpu_searches():
    """Can the quantizer's searches run on an NVIDIA card here (torch with CUDA, a card seen)?"""
    import importlib.util
    if not importlib.util.find_spec("torch"):
        return False
    try:
        from . import hardware
        return any(a.get("vendor") == "nvidia" for a in hardware.adapters())
    except Exception:
        return False


def recipe(src, dst, n_layer, budget=0, licence="", level="balanced"):
    from .levels import alloc
    if level == "exact":
        cmd = [sys.executable, str(TOOLING / "sml_quantize.py"), str(src), str(dst), "--exact"]
        if licence:
            cmd += ["--license", licence]
        return cmd
    cmd = [sys.executable, str(TOOLING / "sml_quantize.py"), str(src), str(dst), "--layout", "flat",
           "--alloc", alloc(level, n_layer), "--no-gptq", "--scale6", "--seqs", "16", "--seq-len", "64"]
    if budget and level == "balanced":
        cmd += ["--budget", str(int(budget))]
    if licence:
        cmd += ["--license", licence]
    return cmd


class Job:
    def __init__(self):
        self.state = "idle"        # idle, fetching, fitting, building, done, failed
        self.fallback = ""         # "fit" an unknown architecture first, or run "compat" mode
        self.allow_remote_code = False
        self.lines = []
        self.src = self.dst = ""
        self.layer = self.n_layer = 0
        self.t0 = 0.0
        self.result = None
        self.proc = None
        self.level = "balanced"
        self.converted = ""        # "GGUF" or "copy" when the weights were converted first (the timeline)
        self.lend = None           # (lend(), give_back()) from the orchestrator: the GPU for the searches

    def status(self):
        el = time.time() - self.t0 if self.t0 else 0
        frac = self.layer / self.n_layer if self.n_layer else 0.0
        eta = el / frac - el if 0.02 < frac < 1 else None
        return {"state": self.state, "src": self.src, "dst": self.dst, "layer": self.layer, "n_layer": self.n_layer,
                "level": self.level, "converted": self.converted,
                "progress": round(frac, 3), "elapsed_s": round(el), "eta_s": round(eta) if eta else None,
                "log": self.lines[-40:], "result": self.result}

    def start(self, src, name="", budget_mb=0, licence="", on_done=None, level="balanced", fallback="",
              allow_remote_code=False):
        if self.state in ("fetching", "converting", "fitting", "building"):
            raise RuntimeError("a build is already running")
        from .levels import LEVELS
        if level not in [x["id"] for x in LEVELS]:
            raise RuntimeError(f"no such level: {level}")
        self.__init__()
        self.src, self.t0, self.level = src, time.time(), level
        self.fallback, self.allow_remote_code = fallback, bool(allow_remote_code)
        threading.Thread(target=self._run, args=(src, name, budget_mb, licence, on_done), daemon=True).start()

    def _keep(self, keep):
        """An already-quantized copy is built only at its own width (Ernest, 2026-10-29)."""
        if keep and self.level != keep["id"]:
            self._say(f"already quantized ({keep['tag']}): packed into SSF as it is, not shrunk again")
            self.level = keep["id"]

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.state = "failed"
            self.lines.append("stopped by the person")

    WANT = ("*.json", "*.jinja", "tokenizer*", "LICENSE*", "USE_POLICY*", "README.md", "*.model", "*.tiktoken",
            "merges.txt", "vocab.txt")

    def _download(self, repo, dst, stall_s=60, tries=200, extra=()):
        """The checkpoint into dst, resumable byte by byte. Hugging Face's own
        downloader started every retry from zero in a new partial file (1.33.0):
        after twelve hours on a flaky line the 7B had 16 GB of fragments and not
        one finished shard (2026-10-06). Here each file grows in one .part file
        with HTTP range requests; a read that stalls for stall_s seconds is
        retried from where it stopped; each shard is checked by its sha256."""
        import fnmatch
        import hashlib
        import requests
        from huggingface_hub import HfApi
        from huggingface_hub.utils import build_hf_headers
        dst.mkdir(parents=True, exist_ok=True)
        parts = dst / ".parts"
        parts.mkdir(exist_ok=True)
        info = HfApi().model_info(repo, files_metadata=True)
        from .hf_source import weight_files
        # the weights a transformers load reads (safetensors, else torch .bin shards read without
        # torch), root level only and one format: and the small files beside them
        weights = set(weight_files([s.rfilename for s in info.siblings]))
        files = [s for s in info.siblings if s.rfilename in weights or
                 ("/" not in s.rfilename and any(fnmatch.fnmatch(s.rfilename, w) for w in self.WANT + tuple(extra)))]
        self._salvage(dst, parts, files)
        total = sum(s.size or 0 for s in files)
        headers = build_hf_headers()
        done_bytes = lambda: sum((dst / s.rfilename).stat().st_size if (dst / s.rfilename).exists() else
                                 ((parts / (s.rfilename.replace("/", "__") + ".part")).stat().st_size
                                  if (parts / (s.rfilename.replace("/", "__") + ".part")).exists() else 0) for s in files)
        for s in files:
            final = dst / s.rfilename
            if final.exists() and (s.size is None or final.stat().st_size == s.size):
                continue
            part = parts / (s.rfilename.replace("/", "__") + ".part")
            url = f"https://huggingface.co/{repo}/resolve/{info.sha}/{s.rfilename}"
            fails, shown = 0, 0.0
            while True:
                if self.state == "failed":
                    raise RuntimeError("stopped by the person")
                have = part.stat().st_size if part.exists() else 0
                if s.size is not None and have >= s.size:
                    break
                try:
                    h = dict(headers, Range=f"bytes={have}-") if have else headers
                    with requests.get(url, headers=h, stream=True, timeout=(30, stall_s), allow_redirects=True) as r:
                        if have and r.status_code == 200:      # the server ignored the range: start over
                            have = 0
                            part.unlink(missing_ok=True)
                        elif r.status_code not in (200, 206):
                            raise IOError(f"HTTP {r.status_code}")
                        with open(part, "ab") as f:
                            for chunk in r.iter_content(1 << 20):
                                if self.state == "failed":
                                    raise RuntimeError("stopped by the person")
                                f.write(chunk)
                                fails = 0
                                if time.time() - shown > 2:
                                    shown = time.time()
                                    self.layer, self.n_layer = 0, 0
                                    self._progress_line(f"downloaded {done_bytes() / 1e9:.2f} of {total / 1e9:.2f} GB")
                    if s.size is None:
                        break
                except RuntimeError:
                    raise
                except Exception as e:
                    fails += 1
                    if fails > tries:
                        raise RuntimeError(f"could not download {s.rfilename}: {e}")
                    self._say(f"the download paused ({str(e)[:80]}); resuming {s.rfilename} from "
                              f"{(part.stat().st_size if part.exists() else 0) / 1e9:.2f} GB")
                    time.sleep(min(60, 5 * fails))
            if s.lfs and s.lfs.sha256:
                hsh = hashlib.sha256()
                with open(part, "rb") as f:
                    for block in iter(lambda: f.read(1 << 24), b""):
                        hsh.update(block)
                if hsh.hexdigest() != s.lfs.sha256:
                    part.unlink(missing_ok=True)
                    raise RuntimeError(f"{s.rfilename} arrived damaged (sha256 differs); it will download again")
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part, final)
        shutil.rmtree(parts, ignore_errors=True)
        shutil.rmtree(dst / ".cache", ignore_errors=True)          # the old downloader's fragments
        return dst

    def _salvage(self, dst, parts, files):
        """The old downloader's fragments are prefixes of the shards, named by the
        shard's sha256: the longest per shard becomes its .part, the rest go."""
        frag = dst / ".cache" / "huggingface" / "download"
        if not frag.is_dir():
            return
        by_sha = {s.lfs.sha256: s for s in files if s.lfs}
        best = {}
        for f in frag.glob("*.incomplete"):
            bits = f.name.split(".")
            sha = bits[1] if len(bits) > 2 else ""
            if sha in by_sha and f.stat().st_size > best.get(sha, (0, None))[0]:
                best[sha] = (f.stat().st_size, f)
        for sha, (n, f) in best.items():
            s = by_sha[sha]
            part = parts / (s.rfilename.replace("/", "__") + ".part")
            if n <= (s.size or 0) and n > (part.stat().st_size if part.exists() else 0):
                os.replace(f, part)
                self._say(f"kept {n / 1e9:.2f} GB already downloaded of {s.rfilename}")
        shutil.rmtree(dst / ".cache", ignore_errors=True)

    def _progress_line(self, text):
        """One updating line for the download, not one per check."""
        if self.lines and self.lines[-1].startswith("downloaded "):
            self.lines[-1] = text
        else:
            self._say(text)

    def _fit(self, d):
        """Fit an unlisted architecture to the engine's settings (tooling/sml_autofit.py, run
        by the PyTorch runtime); True when it passed and the fit is saved for the build."""
        from . import gpu_runtime
        try:
            cfg = json.loads((Path(d) / "config.json").read_text(encoding="utf-8"))
            mt = str((cfg.get("text_config") or {}).get("model_type") if isinstance(cfg.get("text_config"), dict)
                     and not cfg.get("hidden_size") else cfg.get("model_type") or "")
        except (OSError, ValueError):
            mt = ""
        py, _kind = gpu_runtime.torch_python_for(mt, say=self._say)
        if not py:
            self._say("fitting needs PyTorch, which is not installed")
            return False
        self.state = "fitting"
        self._say("Sheep is fitting this model to its engine (a few minutes)")
        env = dict(os.environ, SML_BUILD_PYTHON=sys.executable, MLMI_HOME=str(HOME), PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen([str(py), str(REPO / "tooling" / "sml_autofit.py"), "fit", str(d)], cwd=str(REPO),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
                                     encoding="utf-8", errors="replace",
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result = {}
        for line in self.proc.stdout:
            line = line.rstrip()
            if line.startswith("RESULT "):
                try:
                    result = json.loads(line[7:])
                except ValueError:
                    pass
            elif line.startswith("fitting"):
                self._say(line)
        self.proc.wait()
        if self.state == "failed":                # stopped by the person
            return False
        if result.get("pass"):
            self._say(f"fitted: {result.get('model_type')} matches the maker's code (rel {result.get('rel', 0):.3f}); "
                      "building at native speed")
            return True
        self._say("no fit: " + str(result.get("why") or "the fit did not finish") + "; using compatibility mode")
        return False

    @staticmethod
    def _ssf_on_page(repo):
        try:
            from huggingface_hub import HfApi
            from .publish import ssf_on_page
            return ssf_on_page([str(x.rfilename) for x in (HfApi().model_info(repo).siblings or [])])
        except Exception:
            return None

    def _download_one(self, repo, filename, dst):
        """One file of a page (resumable), with the page's licence and README beside it."""
        from huggingface_hub import hf_hub_download
        dst.mkdir(parents=True, exist_ok=True)
        self._say(f"downloading {repo}/{filename}")
        path = hf_hub_download(repo, filename, local_dir=str(dst))
        for extra in ("LICENSE", "README.md"):
            try:
                hf_hub_download(repo, extra, local_dir=str(dst))
            except Exception:
                pass
        return Path(path)

    def _from_gguf(self, path):
        """A .gguf file to an original-format folder (tooling/sml_gguf.py, run by the PyTorch
        runtime); the folder is kept and reused."""
        return self._convert("sml_gguf.py", path, HOME / "originals" / (path.stem + "-from-gguf"), "GGUF",
                             {"gguf": "gguf>=0.10"})

    def _unpack(self, path):
        """A GPTQ/AWQ folder or Meta's raw .pth checkpoint to a float16 original-format folder
        (tooling/sml_unpack.py, run by the PyTorch runtime); kept and reused."""
        return self._convert("sml_unpack.py", path, HOME / "originals" / (path.name + "-float16"), "copy",
                             {"sentencepiece": "sentencepiece", "tiktoken": "tiktoken"})

    def _convert(self, script, path, out, what, needs):
        """Runs tooling/<script> on path into out with the PyTorch runtime, installing the
        small readers it needs (import name: pip name) once."""
        self.converted = what
        from . import gpu_runtime
        if (out / "config.json").exists() and list(out.glob("*.safetensors")):
            self._say(f"using the folder already converted from {path.name}")
            return out
        py, _kind = gpu_runtime.torch_python(say=self._say)
        if not py:
            raise ValueError(f"converting a {what} needs PyTorch, which is not installed")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.state = "converting"
        for mod, pip in needs.items():
            if subprocess.run([str(py), "-c", f"import {mod}"], capture_output=True, creationflags=flags).returncode:
                self._say(f"installing {mod} for PyTorch (small, once)")
                r = subprocess.run([str(py), "-m", "pip", "install", "--disable-pip-version-check", pip],
                                   capture_output=True, text=True, creationflags=flags)
                if r.returncode:
                    raise ValueError(f"could not install {mod}: " + (r.stderr or r.stdout)[-200:])
        self.state = "converting"
        self._say(f"converting {path.name} to an original-format folder")
        proc = subprocess.Popen([str(py), str(REPO / "tooling" / script), str(path), str(out)],
                                cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", creationflags=flags)
        self.proc = proc
        result = {}
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("RESULT "):
                try:
                    result = json.loads(line[7:])
                except ValueError:
                    pass
            elif line.startswith("converting"):
                self._say(line)
        proc.wait()
        if self.state == "failed":
            raise ValueError("stopped")
        if not result.get("ok"):
            raise ValueError(f"the {what} could not be converted: " + str(result.get("why") or "the converter stopped"))
        self._say(f"converted: {result.get('model_type')} from {path.name}" +
                  (f" ({result['what']})" if result.get("what") else ""))
        return out

    def _compat(self, d, repo, cfg, on_done):
        """Compatibility mode: the maker's own code (transformers) on the original weights,
        as a small .compat.json the engine starter reads (compat_server.py)."""
        remote = bool(cfg.get("auto_map"))
        if remote and not self.allow_remote_code:
            raise RuntimeError("This model runs its own code from Hugging Face; it was not allowed")
        dst = OUT_DIR / (d.name + ".compat.json")
        dst.write_text(json.dumps({"src": str(d), "repo": repo, "trust_remote_code": remote}, indent=1), encoding="utf-8")
        self.dst = str(dst)
        self.state = "done"
        self.result = {"path": str(dst), "mb": 0, "source_dir": str(d), "level": "compat", "compat": True,
                       "minutes": round((time.time() - self.t0) / 60, 1)}
        self._say(f"{repo} runs in compatibility mode (the maker's own code)")
        if on_done:
            on_done(self.result)

    def _say(self, line):
        self.lines.append(line)
        self.lines = self.lines[-400:]

    def _run(self, src, name, budget_mb, licence, on_done):
        try:
            d = Path(src)
            if d.is_file() and d.suffix.lower() == ".gguf":
                from .hf_source import as_is
                self._keep(as_is("gguf", d.name))
                d = self._from_gguf(d)
                name = name or d.name
            elif d.is_dir():
                from .hf_source import UNPACK, WORDS, as_is, folder_kind
                k = folder_kind(d)
                if k in UNPACK:
                    try:
                        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        cfg = {}
                    self._keep(as_is(k, cfg=cfg))
                    d = self._unpack(d)
                elif k not in ("original", "none", "no-config"):
                    raise ValueError(f"{d.name} is {WORDS.get(k, 'a converted copy')}, which Sheep cannot read; "
                                     "use the original model it was made from")
            else:
                if not re.fullmatch(r"[\w.-]+/[\w.-]+", src):
                    raise ValueError("give a folder with the original safetensors, a .gguf file, or a Hugging Face "
                                     "id like org/model")
                ready = self._ssf_on_page(src)                # a page that holds a Sheep file: no build
                if ready:
                    self.state = "fetching"
                    self._say(f"{src} holds a ready Sheep file ({ready}); downloading it, no build needed")
                    got = self._download_one(src, ready, HOME / "originals" / (src.replace("/", "--") + "-ssf"))
                    OUT_DIR.mkdir(parents=True, exist_ok=True)
                    dst = OUT_DIR / (src.replace("/", "--") + ".ssf")
                    shutil.copyfile(got, dst)
                    self.dst = str(dst)
                    self.state = "done"
                    self.result = {"path": str(dst), "mb": round(dst.stat().st_size / 1e6), "source_dir": str(got.parent),
                                   "level": "ready", "minutes": 0}
                    self._say(f"ready: {dst.name}")
                    if on_done:
                        on_done(self.result)
                    return
                from .hf_source import convert_note, fallback, resolve
                try:
                    original, note, _ = resolve(src)          # a GGUF/GPTQ/MLX page: its original
                    fb = None
                    if note:                                 # a copy is kept as it is when Sheep can read it
                        fb = fallback(src)
                        if not fb:
                            raise ValueError(f"{note.split(';')[0]}, which Sheep cannot pack as it is. Add its "
                                             f"original, {original}, to choose a level.")
                        if not fb.get("as_is"):
                            fb = None
                except LookupError:
                    fb = fallback(src)                       # no original anywhere: the copy itself
                    if not fb:
                        raise
                if fb:
                    self._keep(fb.get("as_is"))
                if fb and fb["how"] == "gguf":
                    self._say(f"{src} has no original weights on Hugging Face; Sheep converts its GGUF ({fb['file']}). "
                              "The quality is that file's.")
                    self.state = "fetching"
                    d = self._from_gguf(self._download_one(src, fb["file"], HOME / "originals" / (src.replace("/", "--") + "-gguf")))
                    name = name or src.replace("/", "--")
                elif fb:
                    self._say(convert_note(fb, src))
                    self.state = "fetching"
                    self._say(f"downloading {src}")
                    d = self._unpack(self._download(src, HOME / "originals" / (src.replace("/", "--") + "-packed"),
                                                    extra=("consolidated.*.pth",)))
                    name = name or src.replace("/", "--")
                else:
                    if note:
                        self._say(note)
                        src = self.src = original
                    self.state = "fetching"
                    self._say(f"downloading {src} (weights, configs, tokenizer, licence)")
                    d = self._download(src, HOME / "originals" / src.replace("/", "--"))
            if not (list(d.glob("*.safetensors")) or list(d.glob("pytorch_model*.bin")) or list(d.glob("model*.bin"))):
                raise ValueError(f"no .safetensors or pytorch_model .bin in {d}: "
                                 "Sheep builds from the original checkpoint only")
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            tc = cfg.get("text_config") or {}
            self.n_layer = int(cfg.get("num_hidden_layers") or tc.get("num_hidden_layers") or cfg.get("n_layer") or 0)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            compat = self.fallback == "compat" or os.environ.get("SML_FORCE_COMPAT") == "1"
            if self.fallback == "fit" and not compat:
                compat = not self._fit(d)
                if self.state == "failed":
                    return
            if compat:
                self._compat(d, src, cfg, on_done)
                return
            dst = OUT_DIR / ((name or d.name).removesuffix(".ssf") + ".ssf")
            self.dst = str(dst)
            # written under a working name, renamed when whole: a quantize cut off when the session
            # closed left a 4.7 GB 7B under its final name with all-zero logits (2026-10-06)
            part = dst.with_name(dst.name + ".part")
            part.unlink(missing_ok=True)
            cmd = recipe(d, part, self.n_layer, int(float(budget_mb or 0) * 1e6), licence, self.level)
            self.state = "building"
            # The searches run on the NVIDIA card (sml_grid_gpu.py); the chat model lends it for
            # the quantize. Squeezed in beside a chat engine on the GPU, a search took its memory
            # and the engine hung (2026-10-06). Without the card they run on the CPU, slower.
            lent = False
            if self.lend and gpu_searches():
                lent = self.lend[0]()
                if lent:
                    self._say("the chat model lent the GPU to the quantizer; it answers from the CPU meanwhile")
            env = dict(os.environ, SML_GRID_GPU="1" if (lent or not self.lend) else "0")
            from .levels import LEVELS
            lv = next(x for x in LEVELS if x["id"] == self.level)
            self._say(f"{lv['name']}: {lv['what']}" + (f", budget {budget_mb} MB" if budget_mb else ""))
            self.proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                         text=True, encoding="utf-8", errors="replace",
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in self.proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                self._say(line)
                m = re.search(r"layer (\d+)/(\d+)", line)
                if m:
                    self.layer, self.n_layer = int(m.group(1)), int(m.group(2))
            rc = self.proc.wait()
            if self.state == "failed":
                return
            if rc != 0 or not part.exists():
                raise RuntimeError("the builder refused or failed: " + " | ".join(self.lines[-3:]))
            os.replace(part, dst)
            self.state = "done"
            self.result = {"path": str(dst), "mb": round(dst.stat().st_size / 1e6), "source_dir": str(d), "level": self.level,
                           "minutes": round((time.time() - self.t0) / 60, 1)}
            self._say(f"built {dst.name}: {self.result['mb']} MB in {self.result['minutes']} min")
            if on_done:
                on_done(self.result)
        except Exception as e:
            self.state = "failed"
            self._say(f"failed: {e}")
        finally:
            if locals().get("lent"):
                self.lend[1]()
