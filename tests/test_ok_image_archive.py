import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.services.ok_image_archive import (
    OKImageArchiveQueue,
    build_ok_archive_filename,
    install_ok_image_archive,
)


class FakeOKPanel:
    _ok_image_archive_installed = False

    def __init__(self):
        self.current_ng = np.full((12, 16, 3), 90, dtype=np.uint8)
        self.current_analysis = {"is_defect": False, "detail": {}}
        self.current_aoi_info = {"category": "Muito Adesivo"}
        self.capture_cycle_source = "network"

        self.capture_debug_last_record = {
            "event_id": "evt-ok-001",
            "source": "windows_xp",
        }
        self.capture_debug_last_image_event_id = "evt-ok-001"
        self.capture_debug_last_image = np.full(
            (20, 30, 3),
            (10, 20, 230),
            dtype=np.uint8,
        )

        # Fallback legado propositalmente diferente para comprovar que o
        # arquivo OK usa a mesma evidência genérica de Copiar imagem.
        self.network_intake_last_validation = {"event_id": "evt-ok-001"}
        self.network_intake_last_image_event_id = "evt-ok-001"
        self.network_intake_last_image = np.full(
            (18, 28, 3),
            (100, 110, 120),
            dtype=np.uint8,
        )

        self.saved = []
        self.archived = []
        self.status = []
        self._ok_archive_submitter = self._capture_archive

    def _capture_archive(self, image, category):
        self.archived.append((image.copy(), category))

    def update_network_status(self, message):
        self.status.append(str(message))

    def save_label(self, decision, source="button"):
        self.saved.append((str(decision), str(source)))
        self.current_ng = None
        self.current_analysis = None
        self.current_aoi_info = {}
        return "saved"


install_ok_image_archive(FakeOKPanel)


class OKArchiveNamingTests(unittest.TestCase):
    def test_filename_uses_same_format_as_ng_archive(self):
        stamp = datetime(2026, 10, 2, 8, 11, 12, 345000)
        self.assertEqual(
            build_ok_archive_filename("Muito Adesivo", stamp),
            "2026-10-02_0811_MUITO_ADESIVO.png",
        )


class OKArchiveDecisionTests(unittest.TestCase):
    def test_archive_is_enabled_by_default(self):
        panel = FakeOKPanel()
        panel.save_label("OK", source="button")

        self.assertEqual(panel.saved, [("OK", "button")])
        self.assertEqual(len(panel.archived), 1)

    def test_operator_can_disable_archive_for_current_session(self):
        panel = FakeOKPanel()
        panel.set_ok_archive_enabled(False)
        panel.save_label("OK", source="button")

        self.assertEqual(panel.archived, [])

    def test_ng_is_never_archived_by_ok_archive(self):
        panel = FakeOKPanel()
        panel.save_label("NG", source="button")

        self.assertEqual(panel.archived, [])

    def test_automatic_ok_is_not_archived(self):
        panel = FakeOKPanel()
        panel.save_label("OK", source="auto")

        self.assertEqual(panel.saved, [("OK", "auto")])
        self.assertEqual(panel.archived, [])

    def test_xp_keyboard_ok_is_operator_decision_and_is_archived(self):
        panel = FakeOKPanel()
        panel.save_label("OK", source="xp_keyboard")

        self.assertEqual(panel.saved, [("OK", "xp_keyboard")])
        self.assertEqual(len(panel.archived), 1)

    def test_ok_archives_exact_same_generic_frame_used_by_copy_image(self):
        panel = FakeOKPanel()
        exact_copy_frame = panel.capture_debug_last_image.copy()
        legacy_xp_frame = panel.network_intake_last_image.copy()

        panel.save_label("OK", source="button")

        self.assertEqual(len(panel.archived), 1)
        image, category = panel.archived[0]
        self.assertTrue(np.array_equal(image, exact_copy_frame))
        self.assertFalse(np.array_equal(image, legacy_xp_frame))
        self.assertEqual(image.shape, (20, 30, 3))
        self.assertEqual(category, "Muito Adesivo")

    def test_local_mss_ok_archives_exact_copy_image_evidence(self):
        panel = FakeOKPanel()
        panel.capture_cycle_source = "local"
        panel.capture_debug_last_record = {
            "event_id": "evt-local-ok",
            "source": "local_mss",
        }
        panel.capture_debug_last_image_event_id = "evt-local-ok"
        panel.capture_debug_last_image = np.full(
            (24, 32, 3),
            (15, 70, 190),
            dtype=np.uint8,
        )

        panel.save_label("OK", source="button")

        self.assertEqual(len(panel.archived), 1)
        image, category = panel.archived[0]
        self.assertTrue(
            np.array_equal(image, np.full(
                (24, 32, 3),
                (15, 70, 190),
                dtype=np.uint8,
            ))
        )
        self.assertEqual(category, "Muito Adesivo")

    def test_local_mss_never_falls_back_to_previous_xp_frame(self):
        panel = FakeOKPanel()
        panel.capture_cycle_source = "local"
        panel.capture_debug_last_record = {
            "event_id": "evt-local-current",
            "source": "local_mss",
        }
        panel.capture_debug_last_image_event_id = "evt-local-stale"

        panel.save_label("OK", source="button")

        self.assertEqual(panel.archived, [])
        self.assertTrue(
            any(
                "evidência de Copiar imagem" in message
                for message in panel.status
            )
        )

    def test_same_event_is_archived_only_once(self):
        panel = FakeOKPanel()
        panel.save_label("OK", source="button")

        # Restaura estado visual suficiente para simular um eco do mesmo evento.
        panel.current_ng = np.full((12, 16, 3), 90, dtype=np.uint8)
        panel.current_analysis = {"is_defect": False, "detail": {}}
        panel.current_aoi_info = {"category": "Muito Adesivo"}
        panel.save_label("OK", source="xp_keyboard")

        self.assertEqual(len(panel.archived), 1)

    def test_missing_category_never_creates_sem_categoria_archive(self):
        panel = FakeOKPanel()
        panel.current_aoi_info = {}

        panel.save_label("OK", source="button")

        self.assertEqual(panel.archived, [])
        self.assertTrue(
            any(
                "Nenhum arquivo SEM_CATEGORIA foi criado" in message
                for message in panel.status
            )
        )

    def test_new_event_can_be_archived_after_previous_event(self):
        panel = FakeOKPanel()
        panel.save_label("OK", source="button")
        self.assertEqual(len(panel.archived), 1)

        panel.current_ng = np.full((12, 16, 3), 90, dtype=np.uint8)
        panel.current_analysis = {"is_defect": False, "detail": {}}
        panel.current_aoi_info = {"category": "Invertido"}
        panel.capture_cycle_source = "network"
        panel.capture_debug_last_record = {
            "event_id": "evt-ok-002",
            "source": "windows_xp",
        }
        panel.capture_debug_last_image_event_id = "evt-ok-002"
        panel.capture_debug_last_image = np.full(
            (20, 30, 3),
            (30, 40, 210),
            dtype=np.uint8,
        )

        panel.save_label("OK", source="button")

        self.assertEqual(len(panel.archived), 2)
        self.assertEqual(panel.archived[1][1], "Invertido")


class OKArchiveQueueTests(unittest.TestCase):
    def test_queue_writes_png_with_shared_filename_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = OKImageArchiveQueue(Path(temp_dir))
            image = np.full((10, 14, 3), (10, 80, 220), dtype=np.uint8)
            stamp = datetime(2026, 10, 2, 8, 15, 1, 12000)

            self.assertTrue(queue.submit(image, "Deslocado", stamp))
            queue.wait_until_idle()

            files = list(Path(temp_dir).glob("*.png"))
            self.assertEqual(len(files), 1)
            self.assertEqual(
                files[0].name,
                "2026-10-02_0815_DESLOCADO.png",
            )
            loaded = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
            self.assertIsNotNone(loaded)
            self.assertTrue(np.array_equal(loaded, image))


class OKArchiveSourceContractTests(unittest.TestCase):
    def test_archive_wrapper_is_inside_production_confidence_gate(self):
        source = Path("main.py").read_text(encoding="utf-8")
        learning = source.index("install_anomaly_learning(ControlPanel)")
        ng_archive = source.index("install_ng_image_archive(ControlPanel)")
        ok_archive = source.index("install_ok_image_archive(ControlPanel)")
        production = source.index(
            "install_production_confidence_gate(ControlPanel, OperationalControlsPresenter)"
        )

        self.assertLess(learning, ng_archive)
        self.assertLess(ng_archive, ok_archive)
        self.assertLess(ok_archive, production)

    def test_ok_archive_and_copy_image_share_generic_resolver(self):
        archive_source = Path(
            "src/services/ok_image_archive.py"
        ).read_text(encoding="utf-8")
        debug_source = Path(
            "src/ui/network_xp_debug.py"
        ).read_text(encoding="utf-8")

        self.assertIn("current_copy_image_snapshot", archive_source)
        self.assertIn("current_copy_image_snapshot", debug_source)
        self.assertNotIn("archive_image = self.current_ng.copy()", archive_source)

    def test_ui_places_ok_toggle_after_ng_toggle_and_enabled_by_default(self):
        source = Path("src/ui/control_panel_ui.py").read_text(encoding="utf-8")

        ng_build = source.index(
            "self._build_ng_archive_bar(window, window.content_layout)"
        )
        ok_build = source.index(
            "self._build_ok_archive_bar(window, window.content_layout)"
        )

        self.assertLess(ng_build, ok_build)
        self.assertIn("btn_toggle_ok_archive.setCheckable(True)", source)
        self.assertIn("Salvar imagens OK • ATIVADO", source)
        self.assertIn("btn_toggle_ok_archive.setChecked(True)", source)
        self.assertIn("_layout_ok_archive(window, compact=compact)", source)
        self.assertIn("public/ok_archive", source)


if __name__ == "__main__":
    unittest.main()
