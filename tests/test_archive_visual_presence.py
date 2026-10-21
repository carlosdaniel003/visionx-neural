"""Regressões de reconhecimento real da foto SIDE no acervo OK/NG."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication, QWidget

from src.services.archive_visual_presence import ArchiveVisualPresenceIndex
from src.ui.archive_visual_presence_feedback import (
    SideArchivePresenceFeedback, submit_side_archive_presence,
)
from src.ui.neural_telemetry_model import memory_seen_state, memory_panel_text
from src.ui.inspection_memory_feedback import memory_feedback_state


def screen(seed=1):
    """Imagem sintética AOI: contornos, regiões gabarito/teste e ruído leve."""
    img = np.full((840, 1165, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (20, 250), (584, 835), (19, 20, 18), -1)
    cv2.rectangle(img, (600, 250), (1150, 835), (22, 18, 20), -1)
    for dx in (0, 580):
        for x in (55, 360):
            cv2.rectangle(img, (x + dx, 310), (x + dx + 100, 520),
                          (15, 25, 190), -1)
        cv2.rectangle(img, (120 + dx, 550), (380 + dx, 700),
                      (38, 38, 38), -1)
    if seed:
        noise = np.random.default_rng(seed).integers(
            -7, 8, size=img.shape, dtype=np.int16
        )
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def save(folder, name, image):
    folder.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(folder / name), image)


def analysis():
    return {
        "verdict": "DEFEITO REAL", "is_defect": True,
        "detail": {"recognition_route": "NEW_CNN", "side_early_exit": {
            "analyzed_lights": ["SIDE"], "skipped_lights": ["TOP", "MID"]}},
    }


class ArchivePresenceBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ok = self.root / "ok_archive"
        self.ng = self.root / "ng_archive"
        self.ok.mkdir()
        self.ng.mkdir()

    def index(self):
        return ArchiveVisualPresenceIndex(ok_dir=self.ok, ng_dir=self.ng)

    def test_ng_archived_even_when_pixel_hash_changes_and_filename_has_suffix(self):
        save(self.ng, "2026-10-21_1559_FALTANDO_3.png", screen(1))
        result = self.index().inspect(screen(2), "FALTANDO", "SIDE")
        self.assertEqual(result["status"], "SIMILAR")
        self.assertEqual(result["label"], "NG")
        self.assertEqual(result["matched_labels"], ["NG"])
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["decision_authorized"])

    def test_ok_and_ng_similar_is_ambiguous_not_a_voted_ng(self):
        image = screen(1)
        save(self.ng, "2026-10-21_1559_FALTANDO.png", image)
        save(self.ok, "2026-10-21_1601_FALTANDO.png", image)
        outcome = self.index().inspect(screen(2), "FALTANDO", "SIDE")
        self.assertEqual(outcome["status"], "AMBIGUOUS")
        self.assertEqual(outcome["label"], "")
        self.assertEqual(outcome["matched_labels"], ["OK", "NG"])

    def test_other_category_and_light_do_not_count_as_same_archive_event(self):
        save(self.ng, "2026-10-21_1559_INVERTIDO_SIDE.png", screen(1))
        save(self.ng, "2026-10-21_1600_FALTANDO_TOP.png", screen(1))
        result = self.index().inspect(screen(2), "FALTANDO", "SIDE")
        self.assertEqual(result["status"], "NO_SIMILAR")
        self.assertEqual(result["indexed_ng"], 2)

    def test_focal_physical_change_remains_unseen(self):
        original = screen(1)
        changed = original.copy()
        cv2.rectangle(changed, (650, 460), (1020, 780),
                      (245, 240, 250), -1)
        save(self.ng, "2026-10-21_1600_FALTANDO.png", original)
        result = self.index().inspect(changed, "FALTANDO", "SIDE")
        self.assertEqual(result["status"], "NO_SIMILAR")

    def test_new_file_after_first_query_is_visible_without_resetting_history(self):
        index = self.index()
        self.assertEqual(index.inspect(screen(2), "FALTANDO", "SIDE")["status"],
                         "NO_SIMILAR")
        save(self.ng, "2026-10-21_1601_FALTANDO.png", screen(1))
        self.assertEqual(index.inspect(screen(2), "FALTANDO", "SIDE")["status"],
                         "SIMILAR")

    def test_empty_missing_or_invalid_sources_cannot_assert_seen(self):
        self.assertEqual(self.index().inspect(None, "FALTANDO", "SIDE")["status"],
                         "UNAVAILABLE")
        for folder in (self.ok, self.ng):
            folder.rmdir()
        self.assertEqual(self.index().inspect(screen(1), "FALTANDO", "SIDE")["status"],
                         "UNAVAILABLE")


class ArchivePresenceFeedbackTests(unittest.TestCase):
    def test_pending_then_similarity_changes_only_visual_state_not_verdict(self):
        a = analysis()
        a["detail"]["side_archive_visual_presence"] = {
            "status": "PENDING", "diagnostic_only": True,
            "decision_authorized": False,
        }
        self.assertEqual(memory_feedback_state(a)[0], "CONSULTANDO ACERVO")
        a["detail"]["side_archive_visual_presence"] = {
            "status": "SIMILAR", "label": "NG",
            "diagnostic_only": True, "decision_authorized": False,
        }
        state = memory_seen_state(a)
        self.assertEqual(state["status"], "VI_SEMELHANTE")
        self.assertTrue(state["archive_visual_seen"])
        self.assertFalse(state["approximate_ng_seen"])
        self.assertFalse(state["exact_pair_seen"])
        self.assertIn("ARQUIVO NG", memory_panel_text(a)[0])
        self.assertEqual(memory_feedback_state(a)[0], "VI SEMELHANTE")
        self.assertEqual(a["verdict"], "DEFEITO REAL")
        self.assertTrue(a["is_defect"])

    def test_no_match_is_sem_par_exato_and_ambiguous_is_neutral(self):
        a = analysis()
        a["detail"]["side_archive_visual_presence"] = {
            "status": "NO_SIMILAR", "label": "",
            "diagnostic_only": True, "decision_authorized": False,
        }
        self.assertEqual(memory_feedback_state(a)[0], "SEM PAR EXATO")
        a["detail"]["side_archive_visual_presence"]["status"] = "AMBIGUOUS"
        self.assertEqual(memory_feedback_state(a)[0], "ACERVO AMBÍGUO")

    def test_forged_similarity_not_accepted_and_exact_remains_priority(self):
        a = analysis()
        a["detail"]["side_archive_visual_presence"] = {
            "status": "SIMILAR", "label": "NG",
            "diagnostic_only": False, "decision_authorized": True,
        }
        self.assertEqual(memory_feedback_state(a)[0], "SEM PAR EXATO")
        a["detail"].update(recognition_route="KNOWN_KNN",
                           recognition_known_label="OK")
        self.assertEqual(memory_feedback_state(a)[0], "JÁ VI")

    def test_controller_rejects_stale_or_decided_events_without_repainting(self):
        app = QApplication.instance() or QApplication([])
        class Panel(QWidget):
            def __init__(self):
                super().__init__()
                self.current_analysis = None
                self.adhesive_multilight_primary_event_id = "one"
                self.refreshed = 0
            def _update_reference_panel(self, analysis):
                self.refreshed += 1

        panel = Panel()
        self.addCleanup(panel.deleteLater)
        with tempfile.TemporaryDirectory() as folder:
            index = ArchiveVisualPresenceIndex(
                ok_dir=Path(folder)/"ok", ng_dir=Path(folder)/"ng"
            )
            controller = SideArchivePresenceFeedback(panel, index=index)
            current = analysis()
            pending = {"status": "PENDING"}
            current["detail"]["side_archive_visual_presence"] = pending
            panel.current_analysis = current
            controller._deliver("other", current, {"status": "SIMILAR"})
            self.assertIs(current["detail"]["side_archive_visual_presence"], pending)
            panel.current_analysis = None  # operador já julgou
            controller._deliver("one", current, {"status": "SIMILAR"})
            self.assertEqual(panel.refreshed, 0)
            panel.current_analysis = current
            controller._deliver("one", current, {"status": "NO_SIMILAR"})
            self.assertEqual(panel.refreshed, 1)
            self.assertEqual(current["detail"]["side_archive_visual_presence"]["status"],
                             "NO_SIMILAR")

    def test_side_source_requires_matching_event_and_never_uses_old_frame(self):
        panel = SimpleNamespace(
            adhesive_multilight_primary_event_id="piece-A",
            adhesive_multilight_last_event_id="piece-B",
            adhesive_multilight_last_source_frames={"SIDE": screen()},
        )
        self.assertFalse(submit_side_archive_presence(panel, analysis(), "FALTANDO"))


class SourceWiringTests(unittest.TestCase):
    def test_side_only_queues_archive_check_without_changing_production_policy(self):
        path = Path("src/ui/adhesive_multilight_inspection.py")
        src = path.read_text(encoding="utf-8")
        self.assertIn("if side_only:\n        # A consulta ao acervo", src)
        self.assertIn("submit_side_archive_presence(panel, fused, category)", src)
        self.assertIn("panel.current_analysis = fused", src)
        worker = Path("src/ui/archive_visual_presence_feedback.py").read_text(
            encoding="utf-8")
        self.assertIn("daemon=True", worker)
        self.assertNotIn("production_decision_policy(", worker)
        self.assertNotIn("send_xp", worker)


if __name__ == "__main__":
    unittest.main()
