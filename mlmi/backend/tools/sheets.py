"""Sheep Sheets: .ssheet workbooks. Formulas are kept as formulas and evaluated
here (formula.py), so the grid, the model and an exported .xlsx all agree."""
import builtins
import re

from openpyxl.utils import column_index_from_string, get_column_letter, range_boundaries

from . import tool, overwrite_check
from .. import formats as F
from .formula import Evaluator
from .words import confirm_create, open_doc, save_doc, start_new
from ..workspace import ToolError

CELL = re.compile(r"^\$?([A-Za-z]{1,3})\$?(\d+)$")
_range = builtins.range      # the tools' own parameter is called "range", as the model sees it


def _sheet(d, name):
    sh = d["sheets"]
    if not name:
        return sh[min(int(d.get("active", 0)), len(sh) - 1)]
    for s in sh:
        if s["name"] == name:
            return s
    raise ToolError(f"there is no sheet '{name}'; sheets: {', '.join(s['name'] for s in sh)}")


def _cell(ref):
    m = CELL.match(str(ref).strip())
    if not m:
        raise ToolError(f"'{ref}' is not a cell like B2")
    return column_index_from_string(m.group(1).upper()), int(m.group(2))


def _box(rng):
    try:
        return range_boundaries(str(rng).upper().replace("$", ""))
    except (ValueError, TypeError):
        raise ToolError(f"'{rng}' is not a range like A1:D4")


def coerce(v):
    """Numbers typed as text become numbers; '=...' stays a formula; '' clears."""
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("="):
            return s
        if re.fullmatch(r"-?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?", s):
            f = float(s.replace(",", ""))
            return int(f) if f.is_integer() and "." not in s else f
        if s.upper() in ("TRUE", "FALSE"):
            return s.upper() == "TRUE"
    return v


def fmt(v):
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.6g}"
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return str(v)


def extent(s):
    cols = rows = 0
    for k in s["cells"]:
        c, r = _cell(k)
        cols, rows = max(cols, c), max(rows, r)
    return cols, rows


def values(d):
    """Every formula cell's computed value: {sheet: {coord: value}}."""
    ev = Evaluator(F.SheetView(d))
    return {s["name"]: {k: ev.value(s["name"], k) for k, v in s["cells"].items()
                        if isinstance(v, str) and v.startswith("=")} for s in d["sheets"]}


def grid(d, s, rng=None, max_rows=60, max_cols=15):
    ev = Evaluator(F.SheetView(d))
    if rng:
        c0, r0, c1, r1 = _box(rng)
    else:
        cols, rows = extent(s)
        c0, r0, c1, r1 = 1, 1, max(1, min(cols, max_cols)), max(1, min(rows, max_rows))
    lines = ["\t".join([""] + [get_column_letter(c) for c in _range(c0, c1 + 1)])]
    for r in _range(r0, r1 + 1):
        row = []
        for c in _range(c0, c1 + 1):
            k = f"{get_column_letter(c)}{r}"
            raw = s["cells"].get(k)
            if isinstance(raw, str) and raw.startswith("="):
                row.append(f"{fmt(ev.value(s['name'], k))} [{raw}]")
            else:
                row.append(fmt(raw))
        lines.append("\t".join([str(r)] + row))
    return "\n".join(lines)


def model(path, ws):
    class C:
        pass
    c = C()
    c.ws = ws
    _, d = open_doc(c, path, "sheets")
    vals = values(d)
    return {"path": path, **d, "values": {k: {c2: fmt(v) if isinstance(v, float) else v for c2, v in m.items()}
                                          for k, m in vals.items()}}


# ------------------------------------------------------------------ tools

def _table(s, headers, rows, total):
    """headers + rows at A1, the header bold on a fill, and a Total row summing
    every column that holds numbers. Returns (n_rows, n_cols, total_text)."""
    headers = [str(h) for h in (headers or [])]
    rows = [r if isinstance(r, list) else [r] for r in (rows or [])]
    w = max([len(headers)] + [len(r) for r in rows] + [1])
    grid_ = ([headers + [""] * (w - len(headers))] if headers else []) + [list(r) + [""] * (w - len(r)) for r in rows]
    first = 2 if headers else 1
    last = first + len(rows) - 1
    tot = ""
    if total and rows:
        line = ["Total"] + [""] * (w - 1)
        for c in _range(1, w):
            if any(isinstance(coerce(r[c]), (int, float)) and not isinstance(coerce(r[c]), bool) for r in rows):
                L = get_column_letter(c + 1)
                line[c] = f"=SUM({L}{first}:{L}{last})"
        grid_.append(line)
    for dr, row in enumerate(grid_):
        for dc, v in enumerate(row):
            v = coerce(v)
            if v not in (None, ""):
                s["cells"][f"{get_column_letter(dc + 1)}{dr + 1}"] = v
    if headers:
        for c in _range(1, w + 1):
            s["fmt"][f"{get_column_letter(c)}1"] = {"b": True, "bg": "DDEBF7"}
    if total and rows:
        for c in _range(1, w + 1):
            s["fmt"].setdefault(f"{get_column_letter(c)}{len(grid_)}", {})["b"] = True
    return len(grid_), w


@tool("sheets_create", "sheets", 2, "Create a new Sheep Sheets workbook (.ssheet). Give headers and rows to fill "
      "it in the same call; total=true adds a Total row.",
      {"path": ("string", "new file, e.g. budget.ssheet"), "sheet": ("string", "name of the first sheet"),
       "headers": ("array", "column names, e.g. [\"Item\", \"Cost\"]"),
       "rows": ("array", "rows of values, e.g. [[\"Fuel\", 40], [\"Food\", 25]]", {"type": "array", "items": {}}),
       "total": ("boolean", "true to add a Total row that sums the number columns"),
       "overwrite": ("boolean", "true only to replace an existing workbook")},
      ("path",), confirm=confirm_create("sheets"))
def sheets_create(ctx, path, sheet="Sheet1", headers=None, rows=None, total=False, overwrite=False):
    p, msg = start_new(ctx, path, "sheets", overwrite)
    if msg:
        return msg
    d = F.new("sheets", sheet=(sheet or "Sheet1")[:31])
    extra = ""
    if headers or rows:
        n, w = _table(d["sheets"][0], headers, rows, total)
        extra = f" with a {n}x{w} table" + (f"; {_totals(d)}" if total else "")
    save_doc(ctx, p, d, f"create {ctx.ws.rel(p)}")
    return f"created {ctx.ws.rel(p)}" + extra


def _totals(d):
    ev = Evaluator(F.SheetView(d))
    s = d["sheets"][0]
    return ", ".join(f"{k} = {fmt(ev.value(s['name'], k))}" for k, v in s["cells"].items()
                     if isinstance(v, str) and v.startswith("=SUM("))


@tool("sheets_table", "sheets", 2, "Write a whole table into a sheet at A1: headers, rows, and total=true for a "
      "Total row.",
      {"path": ("string", "the .ssheet file"), "headers": ("array", "column names"),
       "rows": ("array", "rows of values", {"type": "array", "items": {}}),
       "total": ("boolean", "true to add a Total row that sums the number columns"),
       "sheet": ("string", "sheet name; omit for the first")}, ("path", "rows"))
def sheets_table(ctx, path, rows, headers=None, total=False, sheet=""):
    p, d = open_doc(ctx, path, "sheets")
    s = _sheet(d, sheet)
    n, w = _table(s, headers, rows, total)
    save_doc(ctx, p, d, f"table in {path}")
    return f"wrote a {n}x{w} table at A1" + (f"; {_totals(d)}" if total else "")


@tool("sheets_read", "sheets", 1, "Read a sheet as a grid. Formula cells show their value and [the formula].",
      {"path": ("string", "the .ssheet file"), "sheet": ("string", "sheet name; omit for the first"),
       "range": ("string", "optional range like A1:D20")}, ("path",))
def sheets_read(ctx, path, sheet="", range=""):
    _, d = open_doc(ctx, path, "sheets")
    s = _sheet(d, sheet)
    cols, rows = extent(s)
    head = f"sheets: {', '.join(x['name'] for x in d['sheets'])}; showing '{s['name']}' ({rows} rows x {cols} columns)"
    return head + "\n" + grid(d, s, range or None)


@tool("sheets_write", "sheets", 2, "Write values into cells, starting at a cell. values is a list of rows. "
      "Text starting with = is a formula, e.g. \"=SUM(B2:B5)\".",
      {"path": ("string", "the .ssheet file"), "cell": ("string", "top-left cell, e.g. A1"),
       "values": ("array", "rows of values, e.g. [[\"Item\",\"Cost\"],[\"Rent\",1200]]",
                  {"type": "array", "items": {}}),
       "sheet": ("string", "sheet name; omit for the first")}, ("path", "cell", "values"))
def sheets_write(ctx, path, cell, values, sheet=""):
    p, d = open_doc(ctx, path, "sheets")
    s = _sheet(d, sheet)
    c0, r0 = _cell(cell)
    rows = [r if isinstance(r, list) else [r] for r in (values if isinstance(values, list) else [[values]])]
    n = 0
    written = []
    for dr, row in enumerate(rows):
        for dc, v in enumerate(row):
            k = f"{get_column_letter(c0 + dc)}{r0 + dr}"
            v = coerce(v)
            if v is None or v == "":
                s["cells"].pop(k, None)
            else:
                s["cells"][k] = v
            written.append((k, v))
            n += 1
    save_doc(ctx, p, d, f"write {n} cells in {path}")
    end = f"{get_column_letter(c0 + max(len(r) for r in rows) - 1)}{r0 + len(rows) - 1}"
    ev = Evaluator(F.SheetView(d))
    tail = "; ".join(f"{k} = {fmt(ev.value(s['name'], k))}" for k, v in written
                     if isinstance(v, str) and v.startswith("="))
    return f"wrote {n} cells {cell.upper()}:{end}" + (f"; {tail[:600]}" if tail else "")


@tool("sheets_set_cell", "sheets", 2, "Set one cell to a value or a formula (empty clears it).",
      {"path": ("string", "the .ssheet file"), "cell": ("string", "cell, e.g. C7"),
       "value": ("string", "the value, or a formula starting with ="),
       "sheet": ("string", "sheet name; omit for the first")}, ("path", "cell", "value"))
def sheets_set_cell(ctx, path, cell, value, sheet=""):
    p, d = open_doc(ctx, path, "sheets")
    s = _sheet(d, sheet)
    c, r = _cell(cell)
    k = f"{get_column_letter(c)}{r}"
    v = coerce(value)
    if v is None or v == "":
        s["cells"].pop(k, None)
    else:
        s["cells"][k] = v
    save_doc(ctx, p, d, f"set {k} in {path}")
    if isinstance(v, str) and v.startswith("="):
        return f"{k} = {v} -> {fmt(Evaluator(F.SheetView(d)).value(s['name'], k))}"
    return f"{k} = {fmt(v)}"


@tool("sheets_add_sheet", "sheets", 2, "Add a sheet (tab).",
      {"path": ("string", "the .ssheet file"), "name": ("string", "new sheet name")}, ("path", "name"))
def sheets_add_sheet(ctx, path, name):
    p, d = open_doc(ctx, path, "sheets")
    if any(s["name"] == name for s in d["sheets"]):
        return f"sheet '{name}' already exists"
    d["sheets"].append({"name": name[:31], "cells": {}, "fmt": {}, "widths": {}, "charts": []})
    save_doc(ctx, p, d, f"add sheet {name} to {path}")
    return f"added sheet '{name}'"


@tool("sheets_format", "sheets", 2, "Format a range: bold, fill color, number format, column width, alignment.",
      {"path": ("string", "the .ssheet file"), "range": ("string", "cells, e.g. A1:D1"),
       "bold": ("boolean", "bold text"), "fill": ("string", "background hex color, e.g. DDEBF7"),
       "number_format": ("string", "e.g. 0.00, #,##0, 0%, $#,##0.00"),
       "width": ("number", "column width in characters"),
       "align": ("string", "left, center or right", ("left", "center", "right")),
       "sheet": ("string", "sheet name; omit for the first")}, ("path", "range"))
def sheets_format(ctx, path, range, bold=None, fill=None, number_format=None, width=None, align=None, sheet=""):
    p, d = open_doc(ctx, path, "sheets")
    s = _sheet(d, sheet)
    c0, r0, c1, r1 = _box(range)
    for r in _range(r0, r1 + 1):
        for c in _range(c0, c1 + 1):
            f = s["fmt"].setdefault(f"{get_column_letter(c)}{r}", {})
            if bold is not None:
                if bold:
                    f["b"] = True
                else:
                    f.pop("b", None)
            if fill:
                f["bg"] = fill.strip("#").upper()[:6]
            if number_format:
                f["nf"] = number_format
            if align:
                f["align"] = align
    if width:
        for c in _range(c0, c1 + 1):
            s["widths"][get_column_letter(c)] = float(width)
    s["fmt"] = {k: v for k, v in s["fmt"].items() if v}
    save_doc(ctx, p, d, f"format {range} in {path}")
    return f"formatted {str(range).upper()}"


def _chart_box(s, c0, r0, c1, r1):
    """The range a chart means: a small model charted B2:D5 of an Item/Cost table
    (no labels, no header, two empty columns; 2026-10-13 simulation). Empty edge
    columns and rows go, a Total row at the bottom goes, and the label column and
    header row just outside the range come in."""
    cell = lambda c, r: s["cells"].get(f"{get_column_letter(c)}{r}")
    empty = lambda v: v is None or v == ""
    num = lambda v: isinstance(v, (int, float)) or (isinstance(v, str) and v.startswith("="))
    text = lambda v: isinstance(v, str) and v.strip() and not v.startswith("=")
    while c1 > c0 and all(empty(cell(c1, r)) for r in builtins.range(r0, r1 + 1)):
        c1 -= 1
    while r1 > r0 and all(empty(cell(c, r1)) for c in builtins.range(c0, c1 + 1)):
        r1 -= 1
    rows = builtins.range(r0, r1 + 1)
    if c0 > 1 and all(num(cell(c0, r)) or empty(cell(c0, r)) for r in rows) \
            and any(text(cell(c0 - 1, r)) for r in rows):
        c0 -= 1
    if r0 > 1 and all(num(cell(c, r0)) or empty(cell(c, r0)) for c in builtins.range(c0 + 1, c1 + 1)) \
            and any(text(cell(c, r0 - 1)) for c in builtins.range(c0 + 1, c1 + 1)):
        r0 -= 1
    while r1 > r0 + 1 and (empty(cell(c0, r1)) or re.match(r"(grand )?total\b", str(cell(c0, r1)), re.I)):
        r1 -= 1
    return c0, r0, c1, r1


@tool("sheets_add_chart", "sheets", 2, "Add a chart made from a range whose first column is labels "
      "and first row is series names.",
      {"path": ("string", "the .ssheet file"), "range": ("string", "data range including labels, e.g. A1:C6"),
       "kind": ("string", "bar, line or pie", ("bar", "line", "pie")), "title": ("string", "chart title"),
       "anchor": ("string", "top-left cell of the chart, e.g. F2"),
       "sheet": ("string", "sheet name; omit for the first")}, ("path", "range"))
def sheets_add_chart(ctx, path, range, kind="bar", title="", anchor="", sheet=""):
    p, d = open_doc(ctx, path, "sheets")
    s = _sheet(d, sheet)
    asked = str(range).upper()
    c0, r0, c1, r1 = _chart_box(s, *_box(range))
    if c1 <= c0 or r1 <= r0:
        raise ToolError("the range needs a label column plus at least one data column, and a header row")
    rng = f"{get_column_letter(c0)}{r0}:{get_column_letter(c1)}{r1}"
    s["charts"].append({"kind": kind if kind in ("bar", "line", "pie") else "bar", "range": rng,
                        "title": title, "anchor": (anchor if anchor and anchor.upper() != f"{get_column_letter(c0)}{r0}"
                                                   else f"{get_column_letter(c1 + 2)}{r0}").upper()})
    save_doc(ctx, p, d, f"chart in {path}")
    return f"added a {kind} chart of {rng}" + (f" (asked {asked}: labels and header included, empty and total rows left out)"
                                                if rng != asked else "")


@tool("sheets_sort", "sheets", 2, "Sort the rows of a range by one column (the header row stays on top).",
      {"path": ("string", "the .ssheet file"), "range": ("string", "range including its header row, e.g. A1:D20"),
       "column": ("string", "column letter to sort by, e.g. B"), "descending": ("boolean", "largest first"),
       "sheet": ("string", "sheet name; omit for the first")}, ("path", "range", "column"))
def sheets_sort(ctx, path, range, column, descending=False, sheet=""):
    p, d = open_doc(ctx, path, "sheets")
    s = _sheet(d, sheet)
    c0, r0, c1, r1 = _box(range)
    key = column_index_from_string(str(column).upper()) - c0
    if not 0 <= key <= c1 - c0:
        raise ToolError(f"column {column} is outside {range}")
    cols = [get_column_letter(c) for c in _range(c0, c1 + 1)]
    rows = [[s["cells"].get(f"{cl}{r}") for cl in cols] for r in _range(r0 + 1, r1 + 1)]
    if any(isinstance(v, str) and v.startswith("=") for row in rows for v in row):
        raise ToolError("the range holds formulas; sorting would break their references")

    def k(row):
        v = row[key]
        return (0, v) if isinstance(v, (int, float)) else (1, str(v or "").lower())
    rows.sort(key=k, reverse=bool(descending))
    for i, row in enumerate(rows):
        for cl, v in zip(cols, row):
            if v is None:
                s["cells"].pop(f"{cl}{r0 + 1 + i}", None)
            else:
                s["cells"][f"{cl}{r0 + 1 + i}"] = v
    save_doc(ctx, p, d, f"sort {range} in {path}")
    return f"sorted {len(rows)} rows by column {str(column).upper()}"
