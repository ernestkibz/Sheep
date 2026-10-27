def herd(doc, args):
    slides = doc["slides"]
    slides[:] = [s for s in slides if s.get("title") != "Agenda"]
    start = 1 if slides and slides[0].get("layout") == "title" else 0
    titles = [s.get("title", "") for s in slides[start:] if s.get("title") and s.get("layout") != "section"]
    slides.insert(start, {"layout": "bullets", "title": "Agenda", "bullets": titles, "notes": "", "items": []})
    return f"agenda slide {start + 1} lists {len(titles)} slides"
