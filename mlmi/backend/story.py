"""What Sheep did for you: one model's setup as a short timeline in plain words.

Read from what Sheep already keeps (the registry entry's history and build record, the
model's facts and its plan); nothing new is recorded for it. Each step is
{"t", "what", "tech"}: `what` is for anyone, `tech` is the engineer's line behind Details."""

PLACES = {"gpu": "the graphics card", "cpu": "the processor", "hybrid": "the graphics card and the processor together",
          "multi": "two graphics cards and the processor together", "compat": "the maker's own code"}
CONVERTED = {"GGUF": "Converted it from a GGUF file into a form Sheep can build from",
             "copy": "Unpacked a compressed copy (GPTQ, AWQ or a raw checkpoint) into a form Sheep can build from"}


def display_name(file_name):
    """org--model.ssf -> org/model: the name a person gave, not the file's."""
    for end in (".compat.json", ".ssf"):
        if file_name.lower().endswith(end):
            file_name = file_name[:-len(end)]
    return file_name.replace("--", "/")


def last(history, event):
    hits = [h for h in history if h.get("event") == event]
    return hits[-1] if hits else None


def story(entry, facts=None, plan=None):
    entry = entry or {}
    facts, plan = facts or entry.get("facts") or {}, plan or entry.get("plan") or {}
    history = entry.get("history") or []
    steps = []

    def add(t, what, tech=""):
        steps.append({"t": t, "what": what, "tech": tech})

    added = last(history, "added")
    if added or entry.get("added"):
        name = entry.get("display") or display_name(entry.get("name") or "") or "the model"
        add((added or {}).get("t") or entry.get("added"), f"Added {name}",
            (entry.get("built") or {}).get("origin") or (added or {}).get("note", ""))
    if facts:
        gb = (facts.get("bytes") or 0) / 1e9
        add(None, f"Read it: a {facts.get('arch', 'new')} model" + (f", {gb:.1f} GB" if gb else ""),
            f"{facts.get('n_layer', '?')} layers, {facts.get('n_ctx_train') or '?'} tokens trained context")
    built = entry.get("built") or {}
    if built.get("from") in CONVERTED:
        add(built.get("t"), CONVERTED[built["from"]], built.get("origin", ""))
    level = built.get("level")
    if level == "compat" or facts.get("compat"):
        add(built.get("t"), "Ran it with the maker's own code, since Sheep cannot build this kind yet",
            "compatibility mode (transformers)")
    elif level == "ready":
        add(built.get("t"), "Downloaded a ready Sheep file: nothing to build", "")
    elif built:
        add(built.get("t"), f"Built it at the {level or 'balanced'} level"
            + (f" in {built['minutes']} min" if built.get("minutes") else ""), built.get("recipe", ""))
    if plan.get("device"):
        where = PLACES.get(plan["device"]) or plan.get("device_name") or plan["device"]
        loaded = last(history, "loaded")
        add((loaded or {}).get("t"), f"Placed it on {where}, sized for this computer",
            "; ".join(plan.get("why") or []))
    can = last(history, "canaries")
    if can:
        add(can.get("t"), f"Checked its answers: {can.get('passed')} of {can.get('of')} right")
    ori = last(history, "oriented")
    if ori:
        add(ori.get("t"), f"Checked app tasks: {ori.get('passed')} of {ori.get('of')} work")
    for h in history:
        if h.get("event") == "egypt: returned":
            add(h.get("t"), "Repaired it" + (f": {h['cause']}" if h.get("cause") else ""))
        elif h.get("event") == "upgrade: started":
            add(h.get("t"), "Moved it from the maker's code to Sheep's own engine, which is faster", h.get("source", ""))
    return steps
