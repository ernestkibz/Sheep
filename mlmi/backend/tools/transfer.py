"""Import and export: the only doors between Sheep's formats and Microsoft's.

import_file  .docx -> .sdoc, .xlsx -> .ssheet, .pptx -> .sdeck (pictures into <name>_assets/)
export_file  .sdoc -> .docx, .md or .pdf; .ssheet -> .xlsx, .csv or .pdf; .sdeck -> .pptx or .pdf
"""
import csv
import io
from pathlib import Path

from . import tool
from .. import formats as F
from .words import open_doc
from ..workspace import ToolError

EXPORTS = {".sdoc": ("docx", "md", "pdf"), ".ssheet": ("xlsx", "csv", "pdf"), ".sdeck": ("pptx", "pdf")}
FORMAT_ALIASES = {"powerpoint": "pptx", "power point": "pptx", "ppt": "pptx", "pptx": "pptx",
                  "word": "docx", "word document": "docx", "word file": "docx", "docx": "docx",
                  "excel": "xlsx", "spreadsheet": "xlsx", "xlsx": "xlsx", "pdf": "pdf",
                  "markdown": "md", "md": "md", "csv": "csv"}


def _native_of(path):
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in F.OFFICE:
        raise ToolError(f"only .docx, .xlsx and .pptx files are imported; {path} is {ext or 'not an office file'}")
    return str(p.with_suffix(F.OFFICE[ext]).as_posix())


def _confirm_import(ws, args):
    ws.path(args.get("path", ""), must_exist=True)     # no question about a file that is not there
    dst = args.get("to") or _native_of(args.get("path", ""))
    return f"replace existing {dst}" if ws.path(dst).exists() else None


def _export_target(args):
    p = Path(args.get("path", ""))
    ext = p.suffix.lower()
    if ext not in EXPORTS:
        raise ToolError(f"{args.get('path')} is not a Sheep document (.sdoc, .ssheet or .sdeck)")
    raw_fmt = str(args.get("format") or "").strip().lower().lstrip(".")
    fmt = FORMAT_ALIASES.get(raw_fmt, raw_fmt) if raw_fmt else EXPORTS[ext][0]
    target = args.get("to")
    if not isinstance(target, str) or target.strip().lower() in {"", "none", "null", "undefined"}:
        target = ""
    target = target.strip()
    # Models sometimes put the requested format in `to` ("powerpoint") rather
    # than `format`. Treat that as a format request, not an extensionless file.
    target_fmt = FORMAT_ALIASES.get(target.lower())
    if target_fmt:
        if not raw_fmt:
            fmt = target_fmt
        target = ""
    if fmt not in EXPORTS[ext]:
        raise ToolError(f"a {ext} file exports as: {', '.join(EXPORTS[ext])}")
    if target and not Path(target).suffix:
        target += "." + fmt
    return target or str(p.with_suffix("." + fmt).as_posix()), fmt


def _confirm_export(ws, args):
    dst, _ = _export_target(args)
    return f"replace existing {dst}" if ws.path(dst).exists() else None


@tool("import_file", "files", 2, "Bring a Microsoft file into Sheep: .docx becomes .sdoc (Sheep Words), "
      ".xlsx becomes .ssheet (Sheep Sheets), .pptx becomes .sdeck (Sheep Presentations).",
      {"path": ("string", "the .docx, .xlsx or .pptx file"), "to": ("string", "where to save it (optional)")},
      ("path",), confirm=_confirm_import)
def import_file(ctx, path, to=""):
    src = ctx.ws.path(path, must_exist=True)
    native = _native_of(path)
    dst_rel = to or native
    want = Path(native).suffix
    if Path(dst_rel).suffix.lower() != want:
        if Path(dst_rel).suffix:       # "notes.txt" would hold Sheep JSON under a text name
            raise ToolError(f"{path} imports as a {want} file; give `to` a {want} name or leave it out")
        dst_rel += want
    dst = ctx.ws.path(dst_rel)
    ext = src.suffix.lower()
    assets = dst.parent / (dst.stem + "_assets")
    rel_assets = ctx.ws.rel(assets)
    try:
        if ext == ".docx":
            d = F.docx_import(str(src), str(assets), rel_assets)
        elif ext == ".xlsx":
            d = F.xlsx_import(str(src))
        else:
            d = F.pptx_import(str(src), str(assets), rel_assets)
    except ToolError:
        raise
    except Exception as e:      # a damaged or unusual file: say so, do not crash the agent
        raise ToolError(f"{path} could not be imported: {type(e).__name__}: {e}")
    with ctx.ws.changing(dst, by=ctx.by, what=f"import {path}"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        F.save(dst, d)
    ctx.made.add(ctx.ws.rel(dst))
    size = {".docx": lambda: f"{len(d['blocks'])} blocks", ".xlsx": lambda: f"{len(d['sheets'])} sheet(s)",
            ".pptx": lambda: f"{len(d['slides'])} slides"}[ext]()
    note = " (charts are not carried over from .xlsx)" if ext == ".xlsx" else ""
    return f"imported {path} as {ctx.ws.rel(dst)} ({size}){note}"


@tool("export_file", "files", 2, "Save a Sheep document as a Microsoft file or a PDF: .sdoc to .docx (or .md, .pdf), "
      ".ssheet to .xlsx (or .csv, .pdf), .sdeck to .pptx (or .pdf).",
      {"path": ("string", "the .sdoc, .ssheet or .sdeck file"),
       "format": ("string", "docx, md, xlsx, csv, pptx or pdf; omit for the Microsoft one"),
       "to": ("string", "where to save it (optional)"),
       "sheet": ("string", "csv only: which sheet (tab) to save; default the first")},
      ("path",), confirm=_confirm_export)
def export_file(ctx, path, format="", to="", sheet=""):
    dst_rel, fmt = _export_target({"path": path, "format": format, "to": to})
    kind = F.EXT[Path(path).suffix.lower()]
    p, d = open_doc(ctx, path, kind)
    dst = ctx.ws.path(dst_rel)
    with ctx.ws.changing(dst, by=ctx.by, what=f"export {path} -> {dst_rel}"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        root = str(ctx.ws.root)
        if fmt == "docx":
            F.docx_export(d, str(dst), root)
        elif fmt == "xlsx":
            F.xlsx_export(d, str(dst))
        elif fmt == "pptx":
            F.pptx_export(d, str(dst), root)
        elif fmt == "md":
            dst.write_text(_markdown(d), encoding="utf-8")
        elif fmt == "csv":
            dst.write_text(_csv(d, sheet), encoding="utf-8", newline="")
        elif fmt == "pdf":
            from .. import pdf_export
            try:
                pages = pdf_export.write(kind, d, dst, root)
            except Exception as e:
                raise ToolError(f"{path} could not be made into a PDF: {e}")
            ctx.made.add(ctx.ws.rel(dst))
            return f"exported {path} to {ctx.ws.rel(dst)} ({pages} page{'s' if pages != 1 else ''}, checked)"
    ctx.made.add(ctx.ws.rel(dst))
    return f"exported {path} to {ctx.ws.rel(dst)}"


def _markdown(d):
    out = []
    num = 0
    for b in d.get("blocks", []):
        t = b.get("type")
        num = num + 1 if t == "numbered" else 0
        if t == "heading":
            out.append("#" * max(1, int(b.get("level", 1))) + " " + b.get("text", ""))
        elif t == "bullet":
            out.append("  " * int(b.get("level", 0)) + "- " + b.get("text", ""))
        elif t == "numbered":
            out.append("  " * int(b.get("level", 0)) + f"{num}. " + b.get("text", ""))
        elif t == "table":
            # cells may be numbers, and a "|" inside one would split the column
            rows = [["" if c is None else str(c).replace("|", "\\|").replace("\n", " ") for c in r]
                    for r in (b.get("rows") or [])]
            if rows:
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                head, body = (rows[0], rows[1:]) if b.get("header", True) else ([""] * width, rows)
                out.append("| " + " | ".join(head) + " |")
                out.append("|" + "---|" * width)
                out.extend("| " + " | ".join(r) + " |" for r in body)
        elif t == "image":
            out.append(f"![]({b.get('src')})")
        elif t == "pagebreak":
            out.append("---")
        else:
            out.append(("> " if b.get("style") == "quote" else "") + b.get("text", ""))
        out.append("")
    return "\n".join(out)


def _csv(d, sheet=""):
    from .sheets import _sheet, extent, fmt, values
    from openpyxl.utils import get_column_letter
    s = _sheet(d, sheet)
    vals = values(d).get(s["name"], {})
    cols, rows = extent(s)
    buf = io.StringIO()
    w = csv.writer(buf)
    for r in range(1, rows + 1):
        row = []
        for c in range(1, cols + 1):
            k = f"{get_column_letter(c)}{r}"
            v = vals.get(k, s["cells"].get(k))
            row.append(fmt(v))
        w.writerow(row)
    return buf.getvalue()
