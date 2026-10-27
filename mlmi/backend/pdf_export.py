"""Sheep documents, decks and sheets as PDF, written by Sheep itself (reportlab):
no Office, no LibreOffice. Each PDF is read back after writing (pypdf) and must
have pages and the document's own first words, or the export fails loudly.

    write(kind, d, dst, root) -> pages
"""
import re
from pathlib import Path
from xml.sax.saxutils import escape


def _rich(text):
    """Sheep's inline marks (**bold**, *italic*) as reportlab's tags, the rest escaped."""
    t = escape(str(text or ""))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    return t


def _styles():
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    ss = getSampleStyleSheet()
    return {"title": ss["Title"], "h1": ss["Heading1"], "h2": ss["Heading2"], "h3": ss["Heading3"],
            "body": ss["BodyText"], "quote": ParagraphStyle("q", parent=ss["BodyText"], leftIndent=18, textColor="#555555"),
            "cell": ParagraphStyle("c", parent=ss["BodyText"], fontSize=9, leading=11)}


def _table(rows, st, header=True):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle
    data = [[Paragraph(_rich(c), st["cell"]) for c in r] for r in rows]
    t = Table(data, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf7")))
    t.setStyle(TableStyle(style))
    return t


def _image(root, src, width):
    from reportlab.platypus import Image
    p = Path(root) / src
    if not p.exists():
        return None
    img = Image(str(p))
    scale = min(1.0, width / max(img.drawWidth, 1))
    img.drawWidth *= scale
    img.drawHeight *= scale
    return img


def doc_pdf(d, dst, root):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    st = _styles()
    flow, items, kind = [], [], None

    def flush():
        nonlocal items, kind
        if items:
            flow.append(ListFlowable(items, bulletType="1" if kind == "numbered" else "bullet", leftIndent=14))
        items, kind = [], None
    for b in d.get("blocks", []):
        t = b.get("type")
        if t in ("bullet", "numbered"):
            if kind and kind != t:
                flush()
            kind = t
            items.append(ListItem(Paragraph(_rich(b.get("text")), st["body"]), leftIndent=14 + 12 * int(b.get("level", 0))))
            continue
        flush()
        if t == "heading":
            lv = int(b.get("level", 1))
            flow.append(Paragraph(_rich(b.get("text")), st["title"] if lv == 0 else st[f"h{min(3, max(1, lv))}"]))
        elif t == "table" and b.get("rows"):
            flow += [_table(b["rows"], st, b.get("header", True)), Spacer(1, 8)]
        elif t == "image":
            im = _image(root, b.get("src", ""), 16 * cm)
            if im:
                flow.append(im)
        elif t == "pagebreak":
            flow.append(PageBreak())
        else:
            flow.append(Paragraph(_rich(b.get("text")), st["quote"] if b.get("style") == "quote" else st["body"]))
    flush()
    SimpleDocTemplate(str(dst), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                      bottomMargin=2 * cm).build(flow or [Paragraph("(empty document)", st["body"])])


def deck_pdf(d, dst, root):
    """One landscape page per slide: the title slide centred, the others a title and points."""
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    st = _styles()
    big = ParagraphStyle("big", parent=st["title"], fontSize=34, leading=40, spaceBefore=4 * cm)
    sub = ParagraphStyle("sub", parent=st["body"], fontSize=16, leading=20, alignment=1, textColor="#555555")
    head = ParagraphStyle("head", parent=st["h1"], fontSize=24, leading=28)
    pt = ParagraphStyle("pt", parent=st["body"], fontSize=15, leading=20)
    flow = []
    for i, s in enumerate(d.get("slides", [])):
        if i:
            flow.append(PageBreak())
        k = float(s.get("title_scale") or 1.0)
        if s.get("layout") == "title" and not s.get("bullets"):
            b2 = ParagraphStyle("big", parent=big, fontSize=34 * k, leading=40 * k) if k != 1.0 else big
            flow += [Paragraph(_rich(s.get("title")), b2), Paragraph(_rich(s.get("subtitle")), sub)]
            continue
        h2 = ParagraphStyle("head", parent=head, fontSize=24 * k, leading=28 * k) if k != 1.0 else head
        flow += [Paragraph(_rich(s.get("title")), h2), Spacer(1, 10)]
        pts = [ListItem(Paragraph(_rich(b.strip()), pt), leftIndent=18 + (16 if b.startswith("  ") else 0))
               for b in s.get("bullets") or []]
        if pts:
            flow.append(ListFlowable(pts, bulletType="bullet", leftIndent=18))
        for it in s.get("items") or []:
            if it.get("kind") == "image":
                im = _image(root, it.get("src", ""), 12 * cm)
                if im:
                    flow.append(im)
    SimpleDocTemplate(str(dst), pagesize=landscape(A4), leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm,
                      bottomMargin=1.5 * cm).build(flow or [Paragraph("(empty deck)", st["body"])])


def sheet_pdf(d, dst):
    """Each sheet as a table of what it shows: a formula prints its computed value,
    as on screen (it printed "=SUM(B2:B4)" until 2026-10-12), and empty columns
    between used ones stay, so the layout matches the sheet."""
    from openpyxl.utils import column_index_from_string, get_column_letter
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer
    from .tools import sheets as _sheets
    try:
        computed = _sheets.values(d)
    except Exception:          # a broken formula must not stop the export; it shows as written
        computed = {}
    st = _styles()
    flow = []
    for i, sh in enumerate(d.get("sheets", [])):
        cells = sh.get("cells") or {}
        pos = [(m.group(1), int(m.group(2))) for k in cells for m in [re.match(r"([A-Z]+)(\d+)$", k)] if m]
        if not pos:
            continue
        cols = [get_column_letter(n) for n in range(1, max(column_index_from_string(c) for c, _ in pos) + 1)]
        last = max(r for _, r in pos)
        shown = computed.get(sh.get("name"), {})

        def val(k):
            v = shown[k] if k in shown else cells.get(k)
            v = v.get("v", v.get("f", "")) if isinstance(v, dict) else v
            return _sheets.fmt(v) if v is not None else ""
        rows = [[val(f"{c}{r}") for c in cols] for r in range(1, last + 1)]
        if i:
            flow.append(PageBreak())
        flow += [Paragraph(_rich(sh.get("name") or f"Sheet {i + 1}"), st["h2"]), Spacer(1, 6), _table(rows, st)]
    SimpleDocTemplate(str(dst), pagesize=landscape(A4), leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm,
                      bottomMargin=1.5 * cm).build(flow or [Paragraph("(empty workbook)", st["body"])])


def first_words(kind, d):
    """Words the PDF must contain: the document's first heading, slide title or cell."""
    if kind == "words":
        b = next((b for b in d.get("blocks", []) if b.get("text")), None)
        return str(b.get("text")) if b else ""
    if kind == "presentations":
        s = next((s for s in d.get("slides", []) if s.get("title")), None)
        return str(s.get("title")) if s else ""
    sh = (d.get("sheets") or [{}])[0].get("cells") or {}
    v = sh.get("A1")
    return str(v.get("v") if isinstance(v, dict) else v or "")


def write(kind, d, dst, root):
    """Write the PDF, then read it back: it must have pages and the document's first words."""
    if kind == "words":
        doc_pdf(d, dst, root)
    elif kind == "presentations":
        deck_pdf(d, dst, root)
    else:
        sheet_pdf(d, dst)
    from pypdf import PdfReader
    r = PdfReader(str(dst))
    pages = len(r.pages)
    want = re.sub(r"[*_`]", "", first_words(kind, d)).split()[:3]
    text = " ".join((p.extract_text() or "") for p in r.pages[:3])
    if not pages or (want and not all(w in text for w in want)):
        raise RuntimeError(f"the PDF did not come out right ({pages} pages; '{' '.join(want)}' not found in it)")
    return pages
