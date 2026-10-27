import colorsys
import re
import unittest
from pathlib import Path

from PIL import Image


UI = Path(__file__).resolve().parents[1] / "desktop" / "ui"
DESKTOP = UI.parent
HEX = re.compile(r"#([0-9a-fA-F]{3,8})\b")
RGB = re.compile(r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})", re.I)
BLUE_WORD = re.compile(r"\b(?:cobalt|blue)\b", re.I)


def _rgb(hex_value):
    if len(hex_value) in (3, 4):
        values = [int(char * 2, 16) for char in hex_value[:3]]
    elif len(hex_value) in (6, 8):
        values = [int(hex_value[index:index + 2], 16) for index in (0, 2, 4)]
    else:
        return None
    return tuple(value / 255 for value in values)


def _is_blue(rgb):
    hue, saturation, _ = colorsys.rgb_to_hsv(*rgb)
    degrees = hue * 360
    return 190 <= degrees <= 260 and saturation >= 0.18


class SheepBrandingTests(unittest.TestCase):
    def test_ui_assets_have_no_cobalt_or_blue_brand_treatments(self):
        violations = []
        for path in UI.rglob("*"):
            if path.suffix.lower() not in (".css", ".js", ".html", ".svg"):
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            violations.extend(f"{path.relative_to(UI)}:{match.start()}: {match.group(0)}"
                              for match in BLUE_WORD.finditer(source))
            for match in HEX.finditer(source):
                rgb = _rgb(match.group(1))
                if rgb and _is_blue(rgb):
                    violations.append(f"{path.relative_to(UI)}:{match.start()}: {match.group(0)}")
            for match in RGB.finditer(source):
                rgb = tuple(int(channel) / 255 for channel in match.groups())
                if _is_blue(rgb):
                    violations.append(f"{path.relative_to(UI)}:{match.start()}: {match.group(0)}")
        self.assertEqual(violations, [])

    def test_shortcut_and_tray_art_are_neutral_grayscale(self):
        for name in ("sheep.png", "icon.png", "sheep.ico", "icon.ico", "sheep-contrast.ico", "sheep-mono-20261010.ico"):
            with self.subTest(name=name):
                with Image.open(DESKTOP / name) as image:
                    frames = ([image.ico.getimage(size).convert("RGBA").copy()
                               for size in image.ico.sizes()] if name.endswith(".ico")
                              else [image.convert("RGBA")])
                for frame in frames:
                    for red, green, blue, alpha in frame.getdata():
                        if alpha:
                            self.assertEqual(red, green)
                            self.assertEqual(green, blue)

    def test_desktop_mark_has_black_legs_with_light_contrast_edges(self):
        with Image.open(DESKTOP / "sheep.png") as image:
            image = image.convert("RGBA")
            for point in ((483, 760), (648, 760)):
                with self.subTest(point=point):
                    self.assertLessEqual(image.getpixel(point)[0], 10)
            self.assertGreater(image.getpixel((449, 760))[0], 200)

    def test_logo_has_no_stray_wave_and_controls_keep_contrast(self):
        css = (UI / "app.css").read_text(encoding="utf-8")
        dash_css = (UI / "dash.css").read_text(encoding="utf-8")
        icons = (UI / "icons.js").read_text(encoding="utf-8")
        boot = (UI / "boot.html").read_text(encoding="utf-8")
        self.assertNotIn("logo-wave", css + dash_css + icons + boot)
        self.assertIn(".minitabs button.on { color:var(--accent-ink)", dash_css)
        self.assertIn(".seg button.on { background:var(--accent); color:var(--accent-ink)", css)
        self.assertIn("background:var(--accent-ink)", dash_css)
        self.assertIn("--fleece:#f4f4f4; --face:#050505", css)
        self.assertIn("--fleece:#111111; --face:#ffffff", css)
        self.assertIn("--face-edge:#f4f4f4", css)
        self.assertIn("--face-edge:#111111", css)
        self.assertIn('stroke="var(--face-edge, #f4f4f4)"', icons)

    def test_chat_sheep_avatar_has_contrasting_plate_and_legible_size(self):
        app_css = (UI / "app.css").read_text(encoding="utf-8")
        sheep_css = (UI / "sheep.css").read_text(encoding="utf-8")
        self.assertIn("background:var(--icon-plate)", app_css)
        self.assertIn(".msg.bot .av:not(.cloud) {\n  width:34px; height:34px; background:var(--panel2)", sheep_css)
        self.assertIn(".msg.bot .av:not(.cloud) svg { width:32px; height:32px; }", sheep_css)
        self.assertNotIn(".filecard .fi, .msg.bot .av:not(.cloud)", sheep_css)

    def test_studio_defaults_are_neutral(self):
        source = (DESKTOP.parent / "backend" / "tools" / "studio.py").read_text(encoding="utf-8")
        studio_ui = (UI / "studio.js").read_text(encoding="utf-8")
        self.assertNotIn("0B1226", source)
        self.assertNotIn("2F6BFF", source)
        self.assertNotIn("2F6BFF", studio_ui)
        self.assertIn('bg="0B0B0B"', source)
        self.assertIn('color="A3A3A3"', source)
        self.assertIn('fill: "A3A3A3"', studio_ui)

    def test_nebius_accent_is_mode_scoped_and_uses_existing_brand_token(self):
        css = (UI / "app.css").read_text(encoding="utf-8")
        app = (UI / "app.js").read_text(encoding="utf-8")
        html = (UI / "index.html").read_text(encoding="utf-8")
        self.assertIn(':root[data-mode="nebius"] { --accent:var(--nv);', css)
        self.assertIn(':root[data-mode="nebius"][data-theme="light"]', css)
        self.assertIn('document.documentElement.dataset.mode = Number(st.mode) === 1 ? "nebius" : "sheep";', app)
        self.assertIn('data-mode="1">Sheep + Nebius', html)

    def test_windows_shortcuts_and_tray_use_versioned_monochrome_icon(self):
        main = (DESKTOP / "main.js").read_text(encoding="utf-8")
        installer = (DESKTOP.parents[1] / "submission" / "sheep.iss").read_text(encoding="utf-8")
        self.assertIn("sheep-mono-20261010.ico", main)
        self.assertIn('IconFilename:', installer)
        self.assertIn('UninstallDisplayIcon=', installer)
        self.assertEqual(installer.count("sheep-mono-20261010.ico"), 3)


if __name__ == "__main__":
    unittest.main()
