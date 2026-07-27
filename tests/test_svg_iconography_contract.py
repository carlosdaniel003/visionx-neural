import unittest
from pathlib import Path
from xml.etree import ElementTree

from src.ui.iconography_model import ALL_ICON_NAMES


ROOT = Path(__file__).resolve().parents[1]


class SvgIconographyContractTests(unittest.TestCase):
    def test_all_declared_icons_exist_and_are_valid_svg(self):
        icon_dir = ROOT / "src" / "ui" / "icons"
        for icon_name in sorted(ALL_ICON_NAMES):
            path = icon_dir / f"{icon_name}.svg"
            self.assertTrue(path.is_file(), path)
            root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
            self.assertTrue(root.tag.endswith("svg"), path)
            self.assertEqual(root.attrib.get("viewBox"), "0 0 24 24")

    def test_hooks_are_installed_before_controller_construction(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_iconography_hooks"),
            source.index("panel = ControlPanel()"),
        )
        self.assertGreater(
            source.index("install_svg_iconography(panel)"),
            source.index("install_operational_controls(panel)"),
        )

    def test_visual_layer_uses_real_qicons_and_separate_status_icons(self):
        source = (ROOT / "src" / "ui" / "iconography.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("button.setIcon(svg_icon(icon_name))", source)
        self.assertIn("label.setPixmap(svg_icon(icon_name).pixmap", source)
        self.assertIn("sanitize_visual_text", source)
        self.assertIn("_rebuild_status_bar", source)

    def test_light_button_text_no_longer_depends_on_unicode_arrows(self):
        source = (ROOT / "src" / "ui" / "iconography.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('self._set_text(self.panel.btn_light_mid, "Luz MID")', source)
        self.assertIn('self._set_text(self.panel.btn_light_side, "Luz SIDE")', source)
        self.assertIn('self._set_text(self.panel.btn_light_top, "Luz TOP")', source)


if __name__ == "__main__":
    unittest.main()
