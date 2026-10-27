"""The docs a judge or a new developer reads: every relative link resolves, and the names the window no
longer uses are gone (polish plan, step 6).

    python -m pytest mlmi/tests/test_docs.py -q
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "README.md", ROOT / "CHANGELOG.md", *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "docs" / "evidence" / "README.md", *sorted((ROOT / "submission").glob("*.md"))]
# where an old name is the point: the internal-names note, the history, and the test's own description
OLD_NAMES_ALLOWED = {"ARCHITECTURE.md", "DEVELOPMENT_HISTORY.md", "TESTING.md", "CHANGELOG.md"}
OLD = [r"ML\.M\.I", r"\bDashboard\b", r"\bHerders?\b", r"\bThis machine\b", r"Sheep Presentations"]
LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)]*)?\)")


def test_every_relative_link_resolves():
    broken = []
    for doc in DOCS:
        for target in LINK.findall(doc.read_text(encoding="utf-8")):
            if re.match(r"[a-z]+:", target):
                continue
            if not (doc.parent / target).exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not broken, broken


def test_old_names_are_gone():
    found = []
    for doc in DOCS:
        if doc.name in OLD_NAMES_ALLOWED:
            continue
        text = doc.read_text(encoding="utf-8")
        found += [f"{doc.relative_to(ROOT)}: {pat}" for pat in OLD if re.search(pat, text)]
    assert not found, found
