import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.services.ng_image_archive import (
    NGImageArchiveQueue,
    build_ng_archive_filename,
    install_ng_image_archive,
    safe_archive_category,
)


class FakePanel:
    _ng_image_archive_installed = False

    def __init__(self):
        self.current_ng = np.full((12, 16, 3), 90, dtype=np.uint8)
        self.current_aoi_info = {"category": "Muito Adesivo"}
        self.saved = []
        self.archived = []
        self.status = []
        self._ng_archive_submitter = self._capture_archive

    def _capture_archive(self, image, category):
        self.archived.append((image.copy(), category))

    def update_network_status(self, message):
        self.status.append(str(message))

    def save_label(self, decision, source="button"):
        self.saved.append((str(decision), str(source)))
        # Simula o ciclo real limpando a imagem antes de retornar.
        self.current_ng = None
        self.current_aoi_info = {}
        return "saved"


install_ng_image_archive(FakePanel)


class NGArchiveNamingTests(unittest.TestCase):
    def test_category_is_safe_for_windows_filename(self):
        self.assertEqual(
            safe_archive_category("Muito Adesivo / Peça Nº 3"),
            "MUITO_ADESIVO_PECA_N_3",
        )

    def test_filename_contains_date_time_and_category(self):
        stamp = datetime(2026, 9, 30, 13, 53, 27, 245000)
        self.assertEqual(
            build_ng_archive_filename("Deslocado", stamp),
            "2026-09-30_13-53-27-245_DESLOCADO.png",
        )


class NGArchiveDecisionTests(unittest.TestCase):
    def test_archive_is_disabled_by_default(self):
        panel = FakePanel()
        panel.save_label("NG", source="button")
        self.assertEqual(panel.saved, [("NG", "button")])
        self.assertEqual(panel.archived, [])

    def test_ok_is_never_archived_even_when_enabled(self):
        panel = FakePanel()
        panel.set_ng_archive_enabled(True)
        panel.save_label("OK", source="button")
        self.assertEqual(panel.archived, [])

    def test_final_ng_archives_snapshot_before_inner_cycle_clears_image(self):
        panel = FakePanel()
        original = panel.current_ng.copy()
        panel.set_ng_archive_enabled(True)

        panel.save_label("NG", source="button")

        self.assertEqual(panel.saved, [("NG", "button")])
        self.assertIsNone(panel.current_ng)
        self.assertEqual(len(panel.archived), 1)
        image, category = panel.archived[0]
        self.assertTrue(np.array_equal(image, original))
        self.assertEqual(category, "Muito Adesivo")

    def test_automatic_ng_is_archived_when_toggle_is_enabled(self):
        panel = FakePanel()
        panel.set_ng_archive_enabled(True)

        panel.save_label("NG", source="auto")

        self.assertEqual(panel.saved, [("NG", "auto")])
        self.assertEqual(len(panel.archived), 1)


class NGArchiveQueueTests(unittest.TestCase):
    def test_queue_writes_png_without_blocking_decision_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = NGImageArchiveQueue(Path(temp_dir))
            image = np.full((10, 14, 3), (10, 80, 220), dtype=np.uint8)
            stamp = datetime(2026, 9, 30, 13, 54, 1, 12000)

            self.assertTrue(queue.submit(image, "Deslocado", stamp))
            queue.wait_until_idle()

            files = list(Path(temp_dir).glob("*.png"))
            self.assertEqual(len(files), 1)
            self.assertEqual(
                files[0].name,
                "2026-09-30_13-54-01-012_DESLOCADO.png",
            )
            loaded = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
            self.assertIsNotNone(loaded)
            self.assertTrue(np.array_equal(loaded, image))


class NGArchiveSourceContractTests(unittest.TestCase):
    def test_archive_wrapper_is_inside_production_confidence_gate(self):
        source = Path("main.py").read_text(encoding="utf-8")
        learning = source.index("install_anomaly_learning(ControlPanel)")
        archive = source.index("install_ng_image_archive(ControlPanel)")
        production = source.index(
            "install_production_confidence_gate(ControlPanel, OperationalControlsPresenter)"
        )
        self.assertLess(learning, archive)
        self.assertLess(archive, production)

    def test_ui_exposes_checkable_archive_toggle(self):
        source = Path("src/ui/control_panel_ui.py").read_text(encoding="utf-8")
        self.assertIn("btn_toggle_ng_archive.setCheckable(True)", source)
        self.assertIn("Salvar imagens NG • DESATIVADO", source)
        self.assertIn("_layout_ng_archive(window, compact=compact)", source)


if __name__ == "__main__":
    unittest.main()
