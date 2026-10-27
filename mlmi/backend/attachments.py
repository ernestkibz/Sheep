"""Files attached to a message: saved in the workspace, then made readable before
the model sees the request.

    save(ws, name, data)       -> the workspace path (attachments/<name>, never overwriting)
    intake(ws, paths, ctx)     -> [(path the job works on, what it is)]

By kind:
  .docx .xlsx .pptx   imported into Sheep's formats (import_file): the job edits the Sheep file
  .pdf                its text, as <name>.txt beside it
  sound and video     what is said, as <name>.txt (Whisper, transcribe_audio)
  pictures            kept as they are: the tools place them in documents, slides and Studio
  .txt .md .csv code, Sheep files   used as they are
"""
import re
from pathlib import PurePosixPath

from . import tools as T
from .workspace import ToolError

OFFICE = {".docx", ".xlsx", ".pptx"}
SOUND = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".wma"}
VIDEO = {".mp4", ".mov", ".mkv", ".avi"}
PICTURE = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_BYTES = 500 * 2**20


def save(ws, name, data):
    """Write an uploaded file under attachments/, with a new name if one is taken."""
    if len(data) > MAX_BYTES:
        raise ToolError(f"{name} is larger than {MAX_BYTES // 2**20} MB")
    base = re.sub(r"[^\w.\- ]+", "_", PurePosixPath(str(name).replace("\\", "/")).name).strip() or "file"
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    rel, k = f"attachments/{base}", 2
    while ws.path(rel).exists():
        rel = f"attachments/{stem}-{k}{'.' + ext if ext else ''}"
        k += 1
    p = ws.path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    with ws.changing(p, by="user", what=f"attach {rel}"):
        p.write_bytes(data)
    return rel


def _pdf_text(src):
    from pypdf import PdfReader
    pages = PdfReader(str(src)).pages
    return "\n\n".join((pg.extract_text() or "").strip() for pg in pages[:200])


def intake(ws, paths, ctx):
    """[(path, what it is)] for the job, after importing, extracting or transcribing."""
    out = []
    for rel in paths:
        ext = PurePosixPath(rel).suffix.lower()
        try:
            if ext in OFFICE:
                native = rel[: -len(ext)] + {".docx": ".sdoc", ".xlsx": ".ssheet", ".pptx": ".sdeck"}[ext]
                if not ws.path(native).exists():
                    T.run(ws, "import_file", {"path": rel, "to": native}, by="agent", ctx=ctx)
                out.append((native, f"the attached {ext[1:]} file, imported into Sheep as {native}"))
            elif ext == ".pdf":
                txt = rel[:-4] + ".txt"
                if not ws.path(txt).exists():
                    ws.path(txt).write_text(_pdf_text(ws.path(rel, must_exist=True)), encoding="utf-8")
                out.append((txt, f"the text of the attached PDF {rel}"))
            elif ext in SOUND or ext in VIDEO:
                txt = rel[: -len(ext)] + ".txt"
                if not ws.path(txt).exists():
                    T.run(ws, "transcribe_audio", {"path": rel}, by="agent", ctx=ctx)
                kind = "video" if ext in VIDEO else "recording"
                out.append((txt, f"what is said in the attached {kind} {rel}"))
                if ext in VIDEO:
                    out.append((rel, "the attached video itself, for Studio"))
            elif ext in PICTURE:
                out.append((rel, "an attached picture. Call see_picture to look at it before answering "
                                 "anything about it; it can also be placed in a document, a slide or a Studio video"))
            else:
                out.append((rel, "an attached file"))
        except (ToolError, OSError, Exception) as e:      # an attachment that cannot be read is said, not hidden
            out.append((rel, f"an attached file that could not be read: {str(e)[:120]}"))
    return out
