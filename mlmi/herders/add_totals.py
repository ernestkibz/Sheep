def herd(doc, args):
    sheets = doc["sheets"]
    name = args.get("sheet") or sheets[doc.get("active", 0)]["name"]
    sh = next(s for s in sheets if s["name"] == name)
    rng = args.get("range")
    if not rng:   # run from Settings with no inputs: the sheet's filled block
        cells = [split_cell(k) for k in sh["cells"]]
        cols = sorted({c for c, _ in cells}, key=col_index)
        rng = f"{cols[0]}{min(r for _, r in cells)}:{cols[-1]}{max(r for _, r in cells)}"
    a, b = rng.upper().split(":")
    c0, r0 = split_cell(a)
    c1, r1 = split_cell(b)
    total_row = r1 + 1
    sh["cells"][f"{c0}{total_row}"] = "Total"
    sh["fmt"].setdefault(f"{c0}{total_row}", {})["b"] = True
    summed = []
    for ci in range(col_index(c0) + 1, col_index(c1) + 1):
        col = col_letter(ci)
        nums = [sh["cells"].get(f"{col}{r}") for r in range(r0 + 1, r1 + 1)]
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nums):
            sh["cells"][f"{col}{total_row}"] = f"=SUM({col}{r0 + 1}:{col}{r1})"
            sh["fmt"].setdefault(f"{col}{total_row}", {})["b"] = True
            summed.append(col)
    return f"row {total_row}: totals for columns {', '.join(summed) or 'none (no numbers found)'}"
