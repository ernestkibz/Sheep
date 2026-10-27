import re


def herd(doc, args):
    counts = [0, 0, 0]
    n = 0
    for b in doc["blocks"]:
        if b.get("type") != "heading" or not 1 <= b.get("level", 1) <= 3:
            continue
        lvl = b.get("level", 1)
        counts[lvl - 1] += 1
        for k in range(lvl, 3):
            counts[k] = 0
        num = ".".join(str(c) for c in counts[:lvl]) + ("." if lvl == 1 else "")
        text = re.sub(r"^\d+(\.\d+)*\.?\s+", "", b.get("text", ""))
        b["text"] = num + " " + text
        n += 1
    return f"numbered {n} headings"
