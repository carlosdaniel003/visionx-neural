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

    def test_same_archived_png_is_exact_frame_not_98pct_similarity(self):
        image = screen(17)
        save(self.ok, "2026-10-21_1441_FALTANDO_SIDE.png", image)
        index = self.index()
        exact = index.inspect(image.copy(), "FALTANDO", "SIDE")
        self.assertEqual(exact["status"], "EXACT_FRAME")
        self.assertEqual(exact["label"], "OK")
        self.assertEqual(exact["exact_frame_labels"], ["OK"])
        self.assertEqual(len(exact["exact_frame_sha256"]), 64)
        self.assertFalse(exact["decision_authorized"])
        # Uma diferença real de pixels continua apenas semelhante.
        changed = image.copy()
        changed[400, 700, 1] ^= 1
        self.assertNotEqual(
            index.inspect(changed, "FALTANDO", "SIDE")["status"],
            "EXACT_FRAME",
        )

    def test_full_frame_exact_conflict_is_neutral_and_deletion_is_checked(self):
        image = screen(18)
        save(self.ok, "2026-10-21_1441_FALTANDO_SIDE.png", image)
        save(self.ng, "2026-10-21_1443_FALTANDO_SIDE.png", image)
        index = self.index()
        self.assertEqual(
            index.inspect(image, "FALTANDO", "SIDE")["status"],
            "EXACT_FRAME_CONFLICT",
        )
        (self.ng / "2026-10-21_1443_FALTANDO_SIDE.png").unlink()
        result = index.inspect(image, "FALTANDO", "SIDE")
        self.assertEqual(result["status"], "EXACT_FRAME")
        self.assertEqual(result["label"], "OK")

    def test_exact_frame_scope_ignores_category_but_respects_light(self):
        image = screen(19)
        save(self.ok, "2026-10-21_1441_FALTANDO_TOP.png", image)
        index = self.index()
        self.assertNotEqual(
            index.inspect(image, "FALTANDO", "SIDE")["status"],
            "EXACT_FRAME",
        )
        self.assertEqual(
            index.inspect(image, "INVERTIDO", "TOP")["status"],
            "EXACT_FRAME",
        )
        self.assertEqual(
            index.inspect(image, "UNKNOWN", "TOP")["status"],
            "EXACT_FRAME",
        )
        self.assertEqual(
            index.inspect(image, "", "TOP")["status"],
            "EXACT_FRAME",
        )
        self.assertEqual(
            index.inspect(image, "FALTANDO", "TOP")["status"],
            "EXACT_FRAME",
        )

    def test_ok_and_ng_similar_is_ambiguous_not_a_voted_ng(self):
        image = screen(1)
        save(self.ng, "2026-10-21_1559_FALTANDO.png", image)
        save(self.ok, "2026-10-21_1601_FALTANDO.png", image)
        outcome = self.index().inspect(screen(2), "FALTANDO", "SIDE")
        self.assertEqual(outcome["status"], "AMBIGUOUS")
        self.assertEqual(outcome["label"], "")
        self.assertEqual(outcome["matched_labels"], ["OK", "NG"])

    def test_other_category_matches_in_same_light_only(self):
        save(self.ng, "2026-10-21_1559_INVERTIDO_SIDE.png", screen(1))
        save(self.ng, "2026-10-21_1600_FALTANDO_TOP.png", screen(1))
        result = self.index().inspect(screen(2), "FALTANDO", "SIDE")
        self.assertEqual(result["status"], "SIMILAR")
        self.assertEqual(result["label"], "NG")
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

    def test_top_mid_similar_archive_ok_is_not_lost_when_side_is_new(self):
        a = analysis()
        row = lambda light: {
            "schema": "visionx.side_archive_visual_presence.v1",
            "status": "SIMILAR", "label": "OK", "lighting": light,
            "source": "PUBLIC_ARCHIVE_PNG",
            "diagnostic_only": True, "decision_authorized": False,
        }
        a["detail"]["side_archive_visual_presence"] = {
            "status": "NO_SIMILAR", "label": "",
            "diagnostic_only": True, "decision_authorized": False,
            "by_light": {
                "SIDE": {"status": "NO_SIMILAR"},
                "TOP": row("TOP"), "MID": row("MID"),
            },
        }
        memory = memory_seen_state(a)
        self.assertEqual(memory["status"], "VI_SEMELHANTE")
        self.assertEqual(memory["archive_label"], "OK")
        self.assertEqual(memory["archive_visual_lights"], ["TOP", "MID"])
        self.assertIn("TOP, MID", memory_panel_text(a)[0])
        self.assertIn("rótulo de arquivo", memory_panel_text(a)[1])
        self.assertEqual(a["verdict"], "DEFEITO REAL")
        self.assertNotIn("proposed_decision", a)

    def test_top_mid_similar_archive_ng_and_ok_conflict_is_ambiguous(self):
        a = analysis()
        def row(label):
            return {
                "schema": "visionx.side_archive_visual_presence.v1",
                "source": "PUBLIC_ARCHIVE_PNG",
                "status": "SIMILAR", "label": label,
                "diagnostic_only": True, "decision_authorized": False,
            }
        a["detail"]["side_archive_visual_presence"] = {
            "status": "NO_SIMILAR", "label": "",
            "diagnostic_only": True, "decision_authorized": False,
            "by_light": {"TOP": row("NG"), "MID": row("OK")},
        }
        memory = memory_seen_state(a)
        self.assertEqual(memory["status"], "ACERVO_AMBIGUO")
        self.assertEqual(memory["archive_label"], "")
        self.assertFalse(memory["archive_visual_seen"])
        self.assertEqual(memory_feedback_state(a)[0], "ACERVO AMBÍGUO")
        self.assertEqual(a["verdict"], "DEFEITO REAL")

        # Mesmo uma tela EXATA da SIDE não deve esconder um vizinho
        # contrário de TOP/MID (a memória humana exata é separada).
        a["detail"]["side_archive_visual_presence"]["by_light"]["SIDE"] = {
            **row("OK"), "status": "EXACT_FRAME",
            "exact_frame_sha256": "a" * 64,
            "exact_frame_labels": ["OK"],
        }
        self.assertEqual(memory_seen_state(a)["status"], "ACERVO_AMBIGUO")
        a["detail"].update(
            recognition_route="KNOWN_KNN", recognition_known_label="OK",
        )
        self.assertEqual(memory_seen_state(a)["status"], "JA_VI")

    def test_unverified_multilight_archive_cannot_invent_seen(self):
        a = analysis()
        a["detail"]["side_archive_visual_presence"] = {
            "status": "NO_SIMILAR",
            "by_light": {
                "TOP": {
                    "status": "SIMILAR", "label": "OK",
                    "diagnostic_only": True, "decision_authorized": False,
                    "source": "UNVERIFIED",
                },
            },
        }
        self.assertEqual(memory_seen_state(a)["status"], "NUNCA_VI")

    def test_exact_full_screen_shows_seen_without_forging_exact_aoi_pair(self):
        a = analysis()
        a["detail"]["side_archive_visual_presence"] = {
            "schema": "visionx.side_archive_visual_presence.v1",
            "status": "EXACT_FRAME", "label": "OK",
            "source": "PUBLIC_ARCHIVE_PNG",
            "diagnostic_only": True, "decision_authorized": False,
            "exact_frame_sha256": "a" * 64,
            "exact_frame_labels": ["OK"],
        }
        memory = memory_seen_state(a)
        self.assertEqual(memory["status"], "JA_VI")
        self.assertTrue(memory["archive_exact_frame_seen"])
        self.assertFalse(memory["exact_pair_seen"])
        self.assertIn("FRAME EXATO", memory_panel_text(a)[0])
        self.assertEqual(a["verdict"], "DEFEITO REAL")
        self.assertEqual(memory_feedback_state(a)[0], "JÁ VI")

    def test_exact_frame_in_top_only_is_seen_and_conflict_is_ambiguous(self):
        a = analysis()
        a["detail"]["side_archive_visual_presence"] = {
            "status": "NO_SIMILAR",
            "by_light": {
                "TOP": {
                    "schema": "visionx.side_archive_visual_presence.v1",
                    "status": "EXACT_FRAME", "label": "OK",
                    "source": "PUBLIC_ARCHIVE_PNG",
                    "diagnostic_only": True, "decision_authorized": False,
                    "exact_frame_sha256": "a" * 64,
                    "exact_frame_labels": ["OK"],
                },
                "SIDE": {"status": "NO_SIMILAR"},
            },
        }
        self.assertEqual(memory_seen_state(a)["status"], "JA_VI")
        a["detail"]["side_archive_visual_presence"]["by_light"]["SIDE"] = {
            "status": "EXACT_FRAME_CONFLICT",
        }
        self.assertEqual(memory_seen_state(a)["status"], "ACERVO_AMBIGUO")

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
        self._qt_app = QApplication.instance() or QApplication([])
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

    def test_local_mss_async_archive_result_updates_copied_debug(self):
        from unittest.mock import patch
        from src.services.capture_evidence import capture_debug_record

        self._qt_app = QApplication.instance() or QApplication([])
        class Panel(QWidget):
            def __init__(self):
                super().__init__()
                self.current_analysis = None
                self.local_capture_debug_event_id = "mss-event"
                self.capture_debug_last_image_event_id = "mss-event"
                self.capture_debug_last_record = {
                    "event_id": "mss-event", "source": "local_mss",
                    "decision": {"verdict": "DEFEITO REAL"},
                }
            def _update_reference_panel(self, analysis):
                pass

        panel = Panel()
        self.addCleanup(panel.deleteLater)
        with tempfile.TemporaryDirectory() as folder:
            index = ArchiveVisualPresenceIndex(
                ok_dir=Path(folder)/"ok", ng_dir=Path(folder)/"ng",
            )
            controller = SideArchivePresenceFeedback(panel, index=index)
            item = analysis()
            item["detail"]["side_archive_visual_presence"] = {
                "status": "PENDING",
            }
            panel.current_analysis = item
            result = {
                "schema": "visionx.side_archive_visual_presence.v1",
                "status": "EXACT_FRAME",
                "label": "OK",
                "diagnostic_only": True, "decision_authorized": False,
                "by_light": {
                    "SIDE": {"status": "EXACT_FRAME", "label": "OK"},
                },
            }
            with patch(
                "src.ui.network_xp_debug.sync_network_debug_controls",
            ) as sync:
                controller._deliver("mss-event", item, result)
                sync.assert_called_once_with(panel)
            recorded = capture_debug_record(panel)["decision"]
            self.assertEqual(
                recorded["archive_visual_presence"]["status"], "EXACT_FRAME",
            )
            self.assertEqual(recorded["verdict"], "DEFEITO REAL")
            self.assertEqual(item["verdict"], "DEFEITO REAL")
            controller._deliver("outdated", item, {"status": "NO_SIMILAR"})
            self.assertEqual(
                capture_debug_record(panel)["decision"]
                ["archive_visual_presence"]["status"], "EXACT_FRAME",
            )

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
