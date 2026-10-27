"""Sheep's diagrams as one draw.io file (docs/diagrams/sheep.drawio): three pages, drawn from the code.

    python tooling/diagrams.py
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams" / "sheep.drawio"
FILL = {"ui": "#dae8fc;strokeColor=#6c8ebf", "core": "#fff2cc;strokeColor=#d6b656", "engine": "#d5e8d4;strokeColor=#82b366",
        "cloud": "#e1d5e7;strokeColor=#9673a6", "check": "#f8cecc;strokeColor=#b85450", "plain": "#f5f5f5;strokeColor=#666666"}


class Page:
    def __init__(self, name):
        self.name, self.cells, self.n = name, [], 2
        self.shapes, self.links, self.notes = {}, [], []          # the same drawing, for the PNG

    def box(self, x, y, w, h, text, kind="plain"):
        self.n += 1
        style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={FILL[kind]};fontSize=12;arcSize=10;"
        self.cells.append(f'<mxCell id="c{self.n}" value="{escape(text, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        self.shapes[f"c{self.n}"] = (x, y, w, h, text, kind)
        return f"c{self.n}"

    def note(self, x, y, w, h, text):
        self.notes.append((x, y, w, h, text))
        self.n += 1
        self.cells.append(f'<mxCell id="c{self.n}" value="{escape(text, {chr(34): "&quot;"})}" style="text;html=1;whiteSpace=wrap;'
                          f'fontSize=11;fontColor=#555555;align=left;verticalAlign=top;" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def arrow(self, a, b, label=""):
        self.links.append((a, b, label))
        self.n += 1
        self.cells.append(f'<mxCell id="c{self.n}" value="{escape(label, {chr(34): "&quot;"})}" style="endArrow=block;html=1;'
                          f'rounded=1;fontSize=10;edgeStyle=orthogonalEdgeStyle;" edge="1" parent="1" source="{a}" target="{b}">'
                          f'<mxGeometry relative="1" as="geometry"/></mxCell>')

    def xml(self):
        return (f'<diagram name="{escape(self.name)}" id="{self.name.lower().replace(" ", "-")}"><mxGraphModel dx="1200" dy="800" '
                f'grid="1" gridSize="10" page="1" pageWidth="1400" pageHeight="900"><root><mxCell id="0"/>'
                f'<mxCell id="1" parent="0"/>{"".join(self.cells)}</root></mxGraphModel></diagram>')


def architecture():
    p = Page("How Sheep is built")
    ui = p.box(40, 40, 1300, 60, "Sheep desktop app (Electron): chat · Words · Sheets · Slides · Studio · Text · Work Canvas · Settings", "ui")
    srv = p.box(40, 150, 300, 70, "Local server (server.py)\nlocalhost only, HTTP + live events", "core")
    agent = p.box(390, 150, 260, 70, "Agent: request → tool calls\n(typo fix, routing, shepherd)", "core")
    tools = p.box(700, 150, 300, 70, "Tools shared with the editors\nfiles · Words · Sheets · Slides · Studio · web · media", "core")
    ws = p.box(1050, 150, 290, 70, "Workspace: every file Sheep makes,\nwith undo for every change", "core")
    orch = p.box(40, 280, 300, 90, "Orchestrator: plans the device, starts the engine, checks answers, engine tune, heat (pause at 99 °C), repair", "check")
    checker = p.box(390, 280, 260, 90, "Checker: rules (and an optional small model) read every answer", "check")
    modes = p.box(700, 280, 300, 90, "Who answers each step:\nthe local model, or Nemotron on Nebius", "core")
    eng = p.box(40, 430, 300, 80, "Sheep engine (sml-server, C + OpenMP + CUDA)\n.ssf model on CPU, GPU or both", "engine")
    compat = p.box(390, 430, 260, 80, "Compatibility mode\n(transformers on the original weights)", "engine")
    dml = p.box(700, 430, 300, 80, "AMD / Intel graphics (DirectML)", "engine")
    media = p.box(1050, 280, 290, 90, "Media workers: pictures + video (SD 1.5 + LCM, AnimateDiff), Whisper speech, Windows voices", "engine")
    neb = p.box(1050, 430, 290, 80, "NVIDIA Nemotron on Nebius\nplans big jobs, checks the local draft", "cloud")
    web = p.box(1050, 570, 290, 80, "Web search (Tavily), one of the tools:\nanswers about now, with sources", "cloud")
    for a, b, t in ((ui, srv, "requests"), (srv, agent, ""), (agent, tools, "tool calls"), (tools, ws, "files"),
                    (srv, orch, ""), (agent, checker, "drafts"), (agent, modes, ""), (orch, eng, "starts, plans"),
                    (orch, compat, "unfamiliar models"), (orch, dml, ""), (tools, media, "pictures, video, speech"),
                    (modes, neb, "Sheep + Nebius mode")):
        p.arrow(a, b, t)
    p.note(40, 570, 560, 120, "Everything runs on this computer except the two purple boxes, which the person can switch off "
           "(Settings › Basics). Every request that leaves the computer is listed in Settings › Advanced › Sent out.")
    return p


def request():
    p = Page("One request end to end")
    steps = [("You type: 'powrpoint about the history of coffee'", "ui"),
             ("Typo fix: 'powrpoint' → 'powerpoint' (cues.fix)", "core"),
             ("Routing: a deck request gets the Slides tools; a video request gets the video tools", "core"),
             ("Missing a model for it? The reply shows what is missing with a Download / Choose another button", "check"),
             ("Sheep mode: the local model writes, the shepherd turns it into tool calls.\nSheep + Nebius: Nemotron plans exact tool calls", "cloud"),
             ("Tools make the real file in the workspace (undo works)", "core"),
             ("Checks: the file is opened and compared with what was asked; Nemotron reviews the draft when on", "check"),
             ("Reply with the file card: Open · .pptx · Canvas", "ui")]
    prev = None
    for i, (t, k) in enumerate(steps):
        c = p.box(200, 30 + i * 100, 760, 64, t, k)
        if prev:
            p.arrow(prev, c)
        prev = c
    return p


def onboarding():
    p = Page("A model from a name to Ready")
    row = [("Any Hugging Face ID, a folder or a .gguf", "ui"), ("Fit check before download: size, licence, will it run here", "check"),
           ("Download the original weights (or convert GGUF / GPTQ / AWQ)", "core"),
           ("Known family → build .ssf at the chosen level\nUnknown family → auto-fit, else compatibility mode", "engine")]
    row2 = [("Orchestrator plans the device (GPU, CPU or both) and starts the engine", "check"),
            ("Model ready → answers checked → apps checked", "check"),
            ("Speed trial: the fastest device for this model here", "engine"),
            ("Engine tune: faster GPU settings, kept only if the answer is identical", "engine")]
    prev = None
    for i, (t, k) in enumerate(row):
        c = p.box(40 + i * 330, 60, 300, 90, t, k)
        if prev:
            p.arrow(prev, c)
        prev = c
    for i, (t, k) in enumerate(row2):
        c = p.box(40 + (3 - i) * 330, 240, 300, 90, t, k)
        p.arrow(prev, c)
        prev = c
    done = p.box(40, 420, 300, 70, "Ready: the lamp turns green", "ui")
    p.arrow(prev, done)
    media = p.box(420, 420, 600, 90, "Picture, video and speech models: one row per job in Settings › Models. "
                  "Any SD 1.5 model or faster-whisper model is checked, downloaded and given a test run before use.", "plain")
    p.note(40, 530, 980, 60, "Each step is shown to the person in Settings › Activity › What Sheep did for you. "
           "Checks pause only when the graphics card reaches 99 °C.")
    return p


PNG_FILL = {"ui": "#dae8fc", "core": "#fff2cc", "engine": "#d5e8d4", "cloud": "#e1d5e7", "check": "#f8cecc", "plain": "#f5f5f5"}


def png(page, out):
    """The page drawn with matplotlib (draw.io itself is not on every machine)."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    xs = [x + w for x, y, w, h, *_ in page.shapes.values()] + [x + w for x, y, w, h, _ in page.notes]
    ys = [y + h for x, y, w, h, *_ in page.shapes.values()] + [y + h for x, y, w, h, _ in page.notes]
    W, H = max(xs) + 40, max(ys) + 40
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=130)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_facecolor("white")
    for a, b, label in page.links:
        ax0, ay0, aw, ah = page.shapes[a][:4]
        bx0, by0, bw, bh = page.shapes[b][:4]
        # from the side facing the other box: below = bottom to top, beside = edge to edge
        if by0 >= ay0 + ah - 1:
            start, end = (ax0 + aw / 2, ay0 + ah), (bx0 + bw / 2, by0)
        elif bx0 >= ax0 + aw - 1:
            start, end = (ax0 + aw, ay0 + ah / 2), (bx0, by0 + bh / 2)
        elif bx0 + bw <= ax0 + 1:
            start, end = (ax0, ay0 + ah / 2), (bx0 + bw, by0 + bh / 2)
        else:
            start, end = (ax0 + aw / 2, ay0), (bx0 + bw / 2, by0 + bh)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2, shrinkA=0, shrinkB=0))
    for x, y, w, h, text, kind in page.shapes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fc=PNG_FILL[kind],
                                    ec="#666666", lw=1))
        lines = []
        for para in text.split("\n"):
            lines += textwrap.wrap(para, max(18, int(w / 8.6))) or [""]
        ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=8.5, color="#1f2328")
    for x, y, w, h, text in page.notes:
        ax.text(x, y, "\n".join(textwrap.wrap(text, int(w / 7.6))), ha="left", va="top", fontsize=8.5, color="#555555")
    ax.set_title(page.name, loc="left", fontsize=13, fontweight="bold", color="#1f2328")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pages = [architecture(), request(), onboarding()]
    OUT.write_text('<mxfile host="app.diagrams.net">' + "".join(p.xml() for p in pages) + "</mxfile>\n", encoding="utf-8")
    print(OUT)
    try:
        for k, page in enumerate(pages, 1):
            out = OUT.with_name(f"sheep-{k}-{page.name.lower().replace(' ', '-')}.png")
            png(page, out)
            print(out)
    except ImportError:
        print("matplotlib is not here: the .drawio file is written; PNGs skipped")


if __name__ == "__main__":
    main()
