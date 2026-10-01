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
        self.current_analysis = {"is_defect": True, "detail": {}}
        self.current_aoi_info = {"category": "Muito Adesivo"}
        self.capture_cycle_source = "network"
        self.network_intake_last_validation = {"event_id": "evt-001"}
        self.network_intake_last_image_event_id = "evt-001"
        self.network_intake_last_image = np.full(
            (20, 30, 3),
            (10, 20, 230),
            dtype=np.uint8,
        )
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
            "MUITO_ADESIVO_PECA_NO_3",
        )

    def test_filename_contains_date_time_and_category(self):
        stamp = datetime(2026, 9, 30, 13, 53, 27, 245000)
        self.assertEqual(
            build_ng_archive_filename("Deslocado", stamp),
            "30d09m2026_13h53min27s245ms_DESLOCADO.png",
        )


class NGArchiveDecisionTests(unittest.TestCase):
    def test_archive_is_enabled_by_default(self):
        panel = FakePanel()
        panel.save_label("NG", source="button")
        self.assertEqual(panel.saved, [("NG", "button")])
        self.assertEqual(len(panel.archived), 1)

    def test_operator_can_disable_archive_for_current_session(self):
        panel = FakePanel()
        panel.set_ng_archive_enabled(False)
        panel.save_label("NG", source="button")
        self.assertEqual(panel.archived, [])

    def test_ok_is_never_archived_even_when_enabled(self):
        panel = FakePanel()
        panel.set_ng_archive_enabled(True)
        panel.save_label("OK", source="button")
        self.assertEqual(panel.archived, [])

    def test_final_ng_archives_exact_xp_frame_not_current_ng_crop(self):
        panel = FakePanel()
        exact_xp_frame = panel.network_intake_last_image.copy()
        crop = panel.current_ng.copy()
        panel.set_ng_archive_enabled(True)

        panel.save_label("NG", source="button")

        self.assertEqual(panel.saved, [("NG", "button")])
        self.assertIsNone(panel.current_ng)
        self.assertEqual(len(panel.archived), 1)
        image, category = panel.archived[0]
        self.assertTrue(np.array_equal(image, exact_xp_frame))
        self.assertEqual(image.shape, (20, 30, 3))
        self.assertNotEqual(image.shape, crop.shape)
        self.assertEqual(category, "Muito Adesivo")

    def test_automatic_ng_is_archived_when_toggle_is_enabled(self):
        panel = FakePanel()
        panel.set_ng_archive_enabled(True)

        panel.save_label("NG", source="auto")

        self.assertEqual(panel.saved, [("NG", "auto")])
        self.assertEqual(len(panel.archived), 1)


    def test_same_xp_event_is_archived_only_once(self):
        panel = FakePanel()
        panel.set_ng_archive_enabled(True)
        exact_xp_frame = panel.network_intake_last_image.copy()

        panel.save_label("NG", source="button")
        # Simula o eco do PRESS_1 retornando do hook do XP como CMD_NG depois
        # que a primeira decisão já limpou a interface.
        panel.save_label("NG", source="xp_keyboard")

        self.assertEqual(len(panel.archived), 1)
        image, category = panel.archived[0]
        self.assertTrue(np.array_equal(image, exact_xp_frame))
        self.assertEqual(category, "Muito Adesivo")

    def test_missing_category_never_creates_sem_categoria_archive(self):
        panel = FakePanel()
        panel.set_ng_archive_enabled(True)
        panel.current_aoi_info = {}

        panel.save_label("NG", source="button")

        self.assertEqual(panel.archived, [])
        self.assertTrue(
            any(
                "Nenhum arquivo SEM_CATEGORIA foi criado" in message
                for message in panel.status
            )
        )

    def test_new_xp_event_can_be_archived_after_previous_event(self):
        panel = FakePanel()
        panel.set_ng_archive_enabled(True)

        panel.save_label("NG", source="button")
        self.assertEqual(len(panel.archived), 1)

        panel.current_ng = np.full((12, 16, 3), 120, dtype=np.uint8)
        panel.current_analysis = {"is_defect": True, "detail": {}}
        panel.current_aoi_info = {"category": "Invertido"}
        panel.capture_cycle_source = "network"
        panel.network_intake_last_validation = {"event_id": "evt-002"}
        panel.network_intake_last_image_event_id = "evt-002"
        panel.network_intake_last_image = np.full(
            (20, 30, 3),
            (30, 40, 210),
            dtype=np.uint8,
        )

        panel.save_label("NG", source="button")

        self.assertEqual(len(panel.archived), 2)
        self.assertEqual(panel.archived[1][1], "Invertido")

    def test_local_capture_never_archives_previous_xp_frame(self):
        panel = FakePanel()
        panel.set_ng_archive_enabled(True)
        panel.capture_cycle_source = "local"

        panel.save_label("NG", source="button")

        self.assertEqual(panel.archived, [])

    def test_mismatched_event_never_falls_back_to_current_ng(self):
        panel = FakePanel()
        panel.set_ng_archive_enabled(True)
        panel.network_intake_last_image_event_id = "evt-other"

        panel.save_label("NG", source="button")

        self.assertEqual(panel.archived, [])
        self.assertTrue(
            any(
                "Nenhum recorte alternativo foi usado" in message
                for message in panel.status
            )
        )


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
                "30d09m2026_13h54min01s012ms_DESLOCADO.png",
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

    def test_archive_and_copy_image_share_the_same_xp_frame_source(self):
        archive_source = Path(
            "src/services/ng_image_archive.py"
        ).read_text(encoding="utf-8")
        debug_source = Path(
            "src/ui/network_xp_debug.py"
        ).read_text(encoding="utf-8")

        self.assertIn("network_xp_frame_snapshot", archive_source)
        self.assertIn("network_xp_frame_snapshot", debug_source)
        self.assertNotIn("archive_image = self.current_ng.copy()", archive_source)

    def test_ui_exposes_checkable_archive_toggle(self):
        source = Path("src/ui/control_panel_ui.py").read_text(encoding="utf-8")
        self.assertIn("btn_toggle_ng_archive.setCheckable(True)", source)
        self.assertIn("Salvar imagens NG • ATIVADO", source)
        self.assertIn("btn_toggle_ng_archive.setChecked(True)", source)
        self.assertIn("_layout_ng_archive(window, compact=compact)", source)


if __name__ == "__main__":
    unittest.main()
