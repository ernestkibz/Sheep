"""Files: see, read, write, move and delete inside the workspace."""
import shutil

from . import tool, overwrite_check
from ..workspace import ToolError

TEXT_EXT = {".txt", ".md", ".csv", ".json", ".html", ".css", ".js", ".py", ".xml", ".yaml", ".yml", ".log"}
READ_LIMIT = 6000
LIST_MAX = 40          # lines a listing shows before it summarizes


def _size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@tool("list_files", "files", 1, "List the files and folders in the workspace (or in one folder of it).",
      {"folder": ("string", "folder to list; omit for the whole workspace")})
def list_files(ctx, folder="."):
    # A folder the model invented ("reports") used to be a hard error, and a model
    # that hit it spent its remaining steps guessing and gave up (2026-10-07). Say
    # it is not there and show what is, which is what was being asked for anyway.
    asked, missing = folder or ".", ""
    try:
        if asked not in (".", "") and ctx.ws.path(asked).is_file():
            # a file given as the folder read "the workspace is empty", and the model
            # rebuilt a document that was there all along (2026-10-13 simulation)
            asked, missing = ".", f"'{folder}' is a file, not a folder (read_file reads it). The workspace holds:\n"
        items = ctx.ws.tree(asked)
    except ToolError:
        if asked in (".", ""):
            raise
        items, missing = ctx.ws.tree("."), f"there is no folder called '{asked}'. The workspace holds:\n"
    if not items:
        return missing + "the workspace is empty" if missing else "the workspace is empty"
    line = lambda i: f"{i['path']}" + ("" if i["path"].endswith("/") else f"  ({_size(i['size'])})")
    if len(items) <= LIST_MAX:
        return missing + "\n".join(line(i) for i in items)
    # A full folder would not fit a small model's context (400 files ran a 4096-token
    # prompt over, 2026-10-07): the folders, the newest files, and a count by type.
    folders = [i for i in items if i["path"].endswith("/")]
    files = sorted((i for i in items if not i["path"].endswith("/")), key=lambda i: -i["mtime"])
    kinds = {}
    for i in files:
        name = i["path"].rsplit("/", 1)[-1]
        ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else "no type"
        kinds[ext] = kinds.get(ext, 0) + 1
    shown = files[:LIST_MAX - min(len(folders), 10)]
    out = [f"{len(files)} files" + (f" in {len(folders)} folders" if folders else "") + ": "
           + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]))]
    out += [line(i) for i in folders[:10]]
    out.append(f"the {len(shown)} newest files:")
    out += [line(i) for i in shown]
    out.append(f"and {len(files) - len(shown)} more; list one folder, or name a file, to see others")
    return missing + "\n".join(out)


@tool("read_file", "files", 1, "Read a file. Documents, sheets and slides come back as an outline; "
      "text files as their text.", {"path": ("string", "file to read")}, ("path",))
def read_file(ctx, path):
    p = ctx.ws.path(path, must_exist=True)
    ext = p.suffix.lower()
    from . import TOOLS
    by_ext = {".sdoc": "words_read", ".ssheet": "sheets_read", ".sdeck": "slides_read",
              ".mp4": "video_probe", ".mov": "video_probe", ".mkv": "video_probe", ".webm": "video_probe"}
    if ext in by_ext and by_ext[ext] in TOOLS:
        return TOOLS[by_ext[ext]].fn(ctx, path=path)
    if ext in (".docx", ".xlsx", ".pptx"):
        return f"{path} is a Microsoft file. Bring it into Sheep with import_file to read and edit it."
    if ext in TEXT_EXT or p.stat().st_size < 200_000:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"{path} is a binary file ({_size(p.stat().st_size)})"
        if len(text) > READ_LIMIT:
            return text[:READ_LIMIT] + f"\n... [{len(text) - READ_LIMIT} more characters]"
        return text or "(empty file)"
    return f"{path} is a binary file ({_size(p.stat().st_size)})"


@tool("write_text_file", "files", 2, "Create or replace a plain text file (.txt, .md, .csv).",
      {"path": ("string", "file to write"), "text": ("string", "the full contents")}, ("path", "text"),
      confirm=overwrite_check("path"))
def write_text_file(ctx, path, text):
    p = ctx.ws.path(path)
    if p.suffix.lower() in (".docx", ".xlsx", ".pptx", ".sdoc", ".ssheet", ".sdeck"):
        raise ToolError(f"{p.suffix} files are made with the words/sheets/slides tools (and export_file), not as text")
    with ctx.ws.changing(p, by=ctx.by, what=f"write {path}"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return f"wrote {path} ({len(text)} characters)"


@tool("create_folder", "files", 2, "Create a folder.", {"path": ("string", "folder to create")}, ("path",))
def create_folder(ctx, path):
    p = ctx.ws.path(path)
    if p.exists():
        return f"{path} already exists"
    with ctx.ws.changing(p, by=ctx.by, what=f"create folder {path}"):
        p.mkdir(parents=True)
    return f"created folder {path}"


@tool("move_file", "files", 3, "Move or rename a file or folder.",
      {"src": ("string", "current path"), "dst": ("string", "new path")}, ("src", "dst"),
      confirm=overwrite_check("dst"))
def move_file(ctx, src, dst):
    a = ctx.ws.path(src, must_exist=True)
    b = ctx.ws.path(dst)
    if a.is_dir():
        # the journal keeps files, so a folder moves as its files: undo puts each one back
        if a == ctx.ws.root:
            raise ToolError("the workspace itself cannot be moved")
        if b.exists():
            raise ToolError(f"{dst} already exists; choose a new folder name")
        if b == a or a in b.parents:
            raise ToolError("a folder cannot be moved inside itself")
        inside = [f for f in a.rglob("*") if f.is_file()]
        with ctx.ws.changing(*inside, *[b / f.relative_to(a) for f in inside], by=ctx.by,
                             what=f"move folder {src} -> {dst}"):
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(a), str(b))
        return f"moved folder {src} to {dst} ({len(inside)} file{'s' if len(inside) != 1 else ''})"
    with ctx.ws.changing(a, b, by=ctx.by, what=f"move {src} -> {dst}"):
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(a), str(b))
    return f"moved {src} to {dst}"


@tool("copy_file", "files", 2, "Copy a file.", {"src": ("string", "file to copy"), "dst": ("string", "new copy")},
      ("src", "dst"), confirm=overwrite_check("dst"))
def copy_file(ctx, src, dst):
    a = ctx.ws.path(src, must_exist=True)
    b = ctx.ws.path(dst)
    with ctx.ws.changing(b, by=ctx.by, what=f"copy {src} -> {dst}"):
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a, b)
    return f"copied {src} to {dst}"


@tool("delete_file", "files", 3, "Delete a file or a folder. It stays in the history and can be restored with undo.",
      {"path": ("string", "file or folder to delete")}, ("path",), confirm="always")
def delete_file(ctx, path):
    p = ctx.ws.path(path, must_exist=True)
    if p.is_dir():
        if p == ctx.ws.root:
            raise ToolError("the workspace itself cannot be deleted")
        inside = [f for f in p.rglob("*") if f.is_file()]
        with ctx.ws.changing(*inside, by=ctx.by, what=f"delete folder {path}"):
            shutil.rmtree(p)
        return f"deleted folder {path} with {len(inside)} file{'s' if len(inside) != 1 else ''} (undo restores them)"
    with ctx.ws.changing(p, by=ctx.by, what=f"delete {path}"):
        p.unlink()
    return f"deleted {path} (undo restores it)"
