"""Every package the backend imports is declared in pyproject.toml (or is one of
the optional media/training extras installed on demand). reportlab and jinja2
were imported but undeclared until 2026-10-12: PDF export failed for every
installed user, and every chat silently fell back from the model's own chat
template to built-in frames."""
import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# import name -> distribution name, where they differ
DIST = {"PIL": "pillow", "docx": "python-docx", "pptx": "python-pptx", "huggingface_hub": "huggingface-hub",
        "faster_whisper": "faster-whisper"}
# installed only when the person turns the feature on (media models, Checker training, speech)
OPTIONAL = {"torch", "torch_directml", "diffusers", "transformers", "peft", "safetensors"}
# imported only inside a subprocess's code string, so the scan below cannot see them
SUBPROCESS_ONLY = {"pyttsx3"}


def declared():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M).group(1)
    return {re.split(r"[<>=!~ ]", s.strip().strip('"'))[0].lower() for s in re.findall(r'"([^"]+)"', block)}


def imported():
    std, ours = set(sys.stdlib_module_names), {p.stem for p in (ROOT / "tooling").glob("*.py")} | {"mlmi"}
    out = {}
    for p in (ROOT / "mlmi" / "backend").rglob("*.py"):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else []
            for m in names:
                top = m.split(".")[0]
                if top not in std and top not in ours:
                    out.setdefault(top, p.relative_to(ROOT).as_posix())
    return out


class DependencyTests(unittest.TestCase):
    def test_every_backend_import_is_declared(self):
        have = declared()
        missing = {m: f for m, f in imported().items()
                   if m not in OPTIONAL and DIST.get(m, m).lower().replace("_", "-") not in have}
        self.assertEqual(missing, {}, "imported by the backend but not in pyproject.toml dependencies")

    def test_voice_packages_are_installed_with_sheep(self):
        # Read aloud, voice-overs, dictation and Talk did not work for installed users (2026-10-13)
        self.assertTrue(SUBPROCESS_ONLY <= declared())
        self.assertIn("faster-whisper", declared())


if __name__ == "__main__":
    unittest.main()
