import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QFrame, QGridLayout, QLabel, QWidget

from src.ui.control_panel_ui import ControlPanelUI
from src.ui.iconography import SvgIconographyPresenter


class ResponsiveIconographyStatusBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _build_panel(profile_name="compact"):
        panel = QWidget()
        panel.status_frame = QFrame(panel)

        builder = ControlPanelUI()
        builder._active_profile_name = profile_name
        builder.status_layout = QGridLayout(panel.status_frame)

        panel.lbl_status_network = QLabel("Ouvindo AOI", panel.status_frame)
        panel.lbl_status_brain = QLabel("Sistema Ocioso", panel.status_frame)
        panel.lbl_status_history = QLabel("Última Peça: Nenhuma", panel.status_frame)

        builder.status_widgets = [
            panel.lbl_status_network,
            panel.lbl_status_brain,
            panel.lbl_status_history,
        ]
        panel.status_layout_items = list(builder.status_widgets)
        panel.ui_builder = builder

        for column, label in enumerate(builder.status_widgets):
            builder.status_layout.addWidget(label, 0, column)

        return panel, builder

    def test_svg_iconography_accepts_grid_status_layout(self):
        panel, builder = self._build_panel("compact")

        SvgIconographyPresenter(panel)

        self.assertEqual(builder.status_layout.count(), 3)
        self.assertEqual(len(panel.status_layout_items), 3)
        self.assertTrue(
            all(
                widget.objectName() == "statusGroup"
                for widget in panel.status_layout_items
            )
        )
        self.assertTrue(
            all(
                label.parent().objectName() == "statusGroup"
                for label in builder.status_widgets
            )
        )

    def test_status_icon_groups_survive_responsive_reflow(self):
        panel, builder = self._build_panel("compact")
        SvgIconographyPresenter(panel)

        builder._layout_status_bar(panel, compact=False)
        self.assertEqual(builder.status_layout.count(), 3)

        builder._layout_status_bar(panel, compact=True)
        self.assertEqual(builder.status_layout.count(), 3)
        self.assertEqual(len(panel.status_layout_items), 3)
        self.assertTrue(
            all(
                label.parent().objectName() == "statusGroup"
                for label in builder.status_widgets
            )
        )


if __name__ == "__main__":
    unittest.main()
