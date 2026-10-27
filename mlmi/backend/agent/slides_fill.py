"""Topping up and repairing a deck the model wrote, and keeping rows a person named.

Split out of agent.py (2026-10-21); every name is re-exported from the agent package."""
import re
from .. import checker, cues, shepherd, verify
from ..workspace import ModelSelectionRequired, ToolError


def fill_slide_shortfall(job, parsed, have, deck_title, outline, writer):
    """Ask for and add missing unique slides even when an earlier draft existed."""
    want = int(job.get("count") or 0)
    # Up to three rounds: "add three slides" added one when the 3B's next round repeated titles the deck
    # already had (proof pass, 2026-10-21). A round that only repeats just uses up one of the three.
    for _ in range(3):
        slides = parsed[2]
        if want <= len(slides):
            return parsed
        missing = want - len(slides)
        existing = list(have) + [slide["title"].strip().lower() for slide in slides]
        more = dict(job, count=missing, action="extend", taken=[str(title).title() for title in existing])
        written_titles = "\n".join(f"## {slide['title']}" for slide in slides)
        context = "\n".join(part for part in (outline, written_titles) if part).strip()
        extra = shepherd.parse("slides", writer(more, context, vary=True))
        extra = shepherd.drop_thin(shepherd.drop_repeats(extra, existing, deck_title))
        if not extra[2]:
            continue
        parsed = shepherd.merge_repeats((parsed[0], parsed[1], slides + extra[2][:missing]))
    return parsed


def repair_slide_placeholders(job, parsed, have, deck_title, outline, writer):
    """Rewrite a draft once when it contains template text; never save placeholders."""
    from ..orient import no_placeholders
    texts = [parsed[0], parsed[1]] + [
        text for slide in parsed[2] for text in [slide.get("title", ""), *(slide.get("bullets") or [])]
    ]
    ok, _ = no_placeholders(texts)
    if ok:
        return parsed
    draft = "\n".join(
        f"## {slide.get('title', '')}\n" + "\n".join(f"- {bullet}" for bullet in slide.get("bullets") or [])
        for slide in parsed[2]
    )
    repair_context = (
        (outline or "")[:2500]
        + "\n\nThe previous draft contained template text. Rewrite the requested new slides with finished, specific "
          "content. Keep the useful slide titles, replace every placeholder with real points, and do not use angle-bracket "
          "examples or instructions. The incomplete draft was:\n" + draft[:2500]
    )
    repaired = shepherd.parse("slides", writer(dict(job), repair_context, again=True, vary=True))
    repaired = shepherd.drop_thin(shepherd.drop_repeats(repaired, have, deck_title))
    repaired_text = [repaired[0], repaired[1]] + [
        text for slide in repaired[2] for text in [slide.get("title", ""), *(slide.get("bullets") or [])]
    ]
    valid, why = no_placeholders(repaired_text)
    expected = max(1, len(parsed[2]))
    if not valid or len(repaired[2]) < expected:
        detail = why or f"the rewrite returned {len(repaired[2])} slides instead of {expected}"
        raise ToolError(f"Sheep could not finish those slides ({detail}); no incomplete slide was saved. Try again.")
    return repaired


def preserve_explicit_sheet_row(asked, columns, parsed):
    """Keep a plainly specified row label and number even if the draft omitted it."""
    if len(columns or []) < 2:
        return parsed
    match = re.search(
        r"\b(?:add|insert|include)\s+(?:(?:a|one|another)\s+)?(?:new\s+)?row\s+for\s+(.+?)\s+"
        r"(?:cost(?:ing)?|priced\s+at|valued\s+at|at)\s+[$£€]?\s*([\d,]+(?:\.\d+)?)",
        asked or "", re.I,
    )
    if not match:
        return parsed
    label = match.group(1).strip(" \t.,:;\"'")
    if not label or len(label) > 80:
        return parsed
    normalized_columns = [str(column).strip().lower() for column in columns]
    label_index = next((i for i, name in enumerate(normalized_columns)
                        if name in {"item", "name", "description", "expense", "category", "product"}), 0)
    value_index = next((i for i, name in enumerate(normalized_columns)
                        if any(word in name for word in ("cost", "price", "amount", "value", "rate"))),
                       1 if len(columns) > 1 else 0)
    if value_index == label_index:
        value_index = 1 if label_index == 0 else 0
    value = shepherd._num(match.group(2).replace(",", ""))
    headers, rows = parsed
    rows = [list(row) for row in rows]
    while any(len(row) < len(columns) for row in rows):
        rows = [row + [""] * (len(columns) - len(row)) for row in rows]
    for row in rows:
        if str(row[label_index]).strip().casefold() == label.casefold():
            if row[value_index] != value:
                row[value_index] = value
            return headers, rows
    row = [""] * len(columns)
    row[label_index], row[value_index] = label, value
    return headers, rows + [row]
