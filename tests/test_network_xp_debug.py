import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.services.capture_debug_payload import decision_record
from src.ui.decision_background import (
    apply_decision_background,
    decision_background_state,
    install_decision_background,
)
from src.ui.theme import APP_STYLESHEET
from src.ui.network_xp_debug import (
    DEBUG_SCHEMA,
    copy_network_image_to_clipboard,
    format_network_debug_report,
    network_debug_image_available,
    network_debug_image_snapshot,
    sync_network_debug_controls,
)


class NetworkXPDebugFormatTests(unittest.TestCase):
    def test_report_contains_rejection_context_and_full_json(self):
        record = {
            "schema": DEBUG_SCHEMA,
            "event_id": "evt-001",
            "timestamp": "2026-09-30T07:55:00.000",
            "source_ip": "169.254.95.200",
            "stage": "aoi_intake_validation",
            "mode": "Modo Produção",
            "transport": {
                "image": {
                    "valid": True,
                    "shape": [840, 1165, 3],
                    "dtype": "uint8",
                },
                "stable_required_frames": 2,
            },
            "cycle": {
                "accepting_images": False,
                "generation": 7,
                "ignored_images": 0,
            },
            "validation_message": "tela sem epicentro de anomalia",
            "validation": {
                "valid": False,
                "reason": "missing_epicenter",
                "raw_anomaly_count": 2,
                "old_epicenter_count": 0,
                "real_epicenter_count": 0,
                "global_box_info": {"w": 410, "h": 250},
                "diagnostic_hints": [
                    "A marcação verde aparece no TESTE, mas o Radar não encontrou candidato."
                ],
            },
        }

        report = format_network_debug_report(record)

        self.assertIn(
            "ODIN - Observador Digital Inteligente - DEBUG DE ENTRADA WINDOWS XP",
            report,
        )
        self.assertIn("Evento: evt-001", report)
        self.assertIn("169.254.95.200", report)
        self.assertIn("tela sem epicentro de anomalia", report)
        self.assertIn("missing_epicenter", report)
        self.assertIn("Anomalias brutas: 2", report)
        self.assertIn("INDÍCIOS DIAGNÓSTICOS", report)
        self.assertIn("REGISTRO COMPLETO (JSON)", report)

    def test_report_exposes_final_missing_decision_and_memory_suppression(self):
        record = {
            "schema": DEBUG_SCHEMA,
            "event_id": "evt-missing",
            "timestamp": "2026-10-01T07:36:26.226",
            "source_ip": "169.254.95.200",
            "stage": "aoi_intake_validation",
            "mode": "Modo Teste",
            "transport": {"image": {"valid": True}},
            "cycle": {},
            "validation_message": "epicentro válido",
            "validation": {
                "valid": True,
                "reason": "valid_epicenter",
                "diagnostic_hints": [],
            },
            "decision": {
                "category": "FALTANDO",
                "verdict": "DEFEITO REAL",
                "is_defect": True,
                "confidence": 0.99,
                "final_score": 1.0,
                "physical_score": 0.93,
                "fusion_rule": "missing_hard_absence",
                "dominant_engine": "missing",
                "operator_review_required": False,
                "hard_missing_evidence": True,
                "reason": "AUSÊNCIA FÍSICA FORTE",
                "missing": {
                    "missing_score": 0.93,
                    "missing_classification": "COMPONENTE FISICAMENTE AUSENTE",
                    "missing_changed_coverage": 0.50,
                    "missing_background_exposure": 0.54,
                    "missing_hard_absence_reason": (
                        "conteúdo do gabarito foi substituído pelo fundo da região"
                    ),
                },
                "memory": {
                    "best_match_label": "OK",
                    "best_similarity": 0.98,
                    "memory_conflict": False,
                    "suppressed_by_hard_missing": True,
                },
            },
        }

        report = format_network_debug_report(record)

        self.assertIn(
            "DECISÃO ODIN - Observador Digital Inteligente",
            report,
        )
        self.assertIn("Categoria: FALTANDO", report)
        self.assertIn("Regra de fusão: missing_hard_absence", report)
        self.assertIn("Ausência física forte efetiva: True", report)
        self.assertIn("KNN melhor rótulo: OK", report)
        self.assertIn("KNN suprimido por ausência física: True", report)

    def test_report_exposes_component_body_presence_witness(self):
        analysis = {
            "is_defect": False,
            "verdict": "FALHA FALSA",
            "confidence": 0.99,
            "reason": "CORPO DO COMPONENTE PRESENTE",
            "detail": {
                "missing_component_body_present": True,
                "missing_body_presence_veto": True,
                "missing_body_presence_source": "aoi_epicenter",
                "missing_body_presence_box": [60, 21, 401, 239],
                "missing_body_coarse_similarity": 0.76,
                "missing_body_silhouette_dice": 0.95,
                "missing_body_area_ratio": 0.97,
                "missing_body_centroid_shift": 0.01,
                "missing_body_presence_reason": (
                    "corpo geométrico preservado apesar da divergência de aparência"
                ),
                "missing_global_envelope_active": True,
                "missing_global_envelope_support": True,
                "missing_global_envelope_veto": True,
                "missing_global_envelope_box": [26, 26, 307, 514],
                "missing_global_envelope_row_profile": 0.89,
                "missing_global_envelope_col_profile": 0.88,
                "missing_global_envelope_coarse_similarity": 0.70,
                "missing_global_envelope_background_exposure": 0.0,
                "missing_global_envelope_reason": (
                    "envelope global preserva perfis horizontal/vertical do componente"
                ),
                "missing_hard_absence": False,
                "decision_trace": {
                    "hard_missing_evidence": False,
                    "raw_hard_missing_evidence": False,
                    "operator_review_required": False,
                    "fusion_rule": "best_match_strong",
                    "memory": {},
                },
            },
        }

        decision = decision_record(
            analysis,
            {"category": "FALTANDO"},
        )
        record = {
            "schema": DEBUG_SCHEMA,
            "event_id": "evt-body-present",
            "timestamp": "2026-10-02T09:49:04.447",
            "source_ip": "169.254.95.200",
            "stage": "aoi_intake_validation",
            "mode": "Modo Teste",
            "transport": {"image": {"valid": True}},
            "cycle": {},
            "validation_message": "epicentro válido",
            "validation": {"valid": True, "reason": "valid_epicenter"},
            "decision": decision,
        }

        report = format_network_debug_report(record)
        self.assertIn("Corpo presente: True", report)
        self.assertIn("Veto por corpo presente: True", report)
        self.assertIn("Fonte da presença: aoi_epicenter", report)
        self.assertIn("Presença coarse similarity: 0.76", report)
        self.assertIn("Presença silhouette dice: 0.95", report)
        self.assertIn("Presença area ratio: 0.97", report)
        self.assertIn("Presença centroid shift: 0.01", report)
        self.assertIn("Envelope global ativo: True", report)
        self.assertIn("Envelope global suporta presença: True", report)
        self.assertIn("Envelope global vetou hard missing: True", report)
        self.assertIn("Envelope global caixa: [26, 26, 307, 514]", report)
        self.assertIn("Envelope global perfil horizontal: 0.89", report)
        self.assertIn("Envelope global perfil vertical: 0.88", report)
        self.assertIn("Envelope global coarse similarity: 0.7", report)

    def test_report_exposes_invariant_presence_plus_strong_ok_witness(self):
        analysis = {
            "is_defect": False,
            "verdict": "FALHA FALSA",
            "confidence": 0.99,
            "reason": "PRESENÇA GLOBAL INVARIÁVEL + OK FORTE",
            "detail": {
                "missing_hard_absence": True,
                "missing_global_envelope_active": True,
                "missing_global_envelope_support": False,
                "missing_global_envelope_invariant_support": True,
                "missing_global_envelope_veto": False,
                "missing_global_envelope_box": [26, 26, 308, 514],
                "missing_global_envelope_background_exposure": 0.0,
                "missing_global_envelope_reference_dark_fraction": 0.39,
                "missing_global_envelope_test_dark_fraction": 0.42,
                "missing_global_envelope_dark_retention": 1.08,
                "missing_global_envelope_invariant_row_profile": 0.94,
                "missing_global_envelope_invariant_col_profile": 0.93,
                "final_score": 0.0,
                "physical_score": 1.0,
                "fusion_rule": (
                    "hard_missing_invariant_presence_ok_witness"
                ),
                "dominant_engine": "knn",
                "decision_trace": {
                    "hard_missing_evidence": False,
                    "raw_hard_missing_evidence": True,
                    "hard_missing_contradicted_by_exact_ok": False,
                    "hard_missing_contradicted_by_invariant_ok": True,
                    "operator_review_required": False,
                    "fusion_rule": (
                        "hard_missing_invariant_presence_ok_witness"
                    ),
                    "memory": {
                        "has_memory": True,
                        "memory_available": True,
                        "best_match_label": "OK",
                        "best_similarity": 0.9230359348654749,
                        "best_ok_similarity": 0.9230359348654749,
                        "best_ng_similarity": 0.9100668640434741,
                        "memory_conflict": False,
                        "operator_review_required": False,
                        "suppressed_by_hard_missing": False,
                        "hard_missing_contradicted_by_invariant_ok": True,
                        "role": (
                            "TESTEMUNHA OK FORTE + PRESENÇA GLOBAL INVARIÁVEL"
                        ),
                    },
                },
            },
        }

        decision = decision_record(
            analysis,
            {"category": "FALTANDO"},
        )
        record = {
            "schema": DEBUG_SCHEMA,
            "event_id": "fb7de76ab04843c3b2ab4ad3e16da3f6",
            "timestamp": "2026-10-02T14:52:50.388",
            "source_ip": "169.254.95.200",
            "stage": "aoi_intake_validation",
            "mode": "Modo Teste",
            "transport": {"image": {"valid": True}},
            "cycle": {},
            "validation_message": "epicentro válido",
            "validation": {"valid": True, "reason": "valid_epicenter"},
            "decision": decision,
        }

        self.assertFalse(decision["hard_missing_evidence"])
        self.assertTrue(decision["raw_hard_missing_evidence"])
        self.assertTrue(
            decision["hard_missing_contradicted_by_invariant_ok"]
        )
        self.assertFalse(
            decision["memory"]["suppressed_by_hard_missing"]
        )

        report = format_network_debug_report(record)
        self.assertIn("Veredito: FALHA FALSA", report)
        self.assertIn("Ausência física forte efetiva: False", report)
        self.assertIn(
            "Hard missing contradito por presença invariável + OK forte: True",
            report,
        )
        self.assertIn("Envelope global suporte invariável: True", report)
        self.assertIn("KNN melhor rótulo: OK", report)

    def test_report_distinguishes_raw_hard_missing_from_exact_ok_witness(self):
        analysis = {
            "is_defect": False,
            "verdict": "FALHA FALSA",
            "confidence": 0.99,
            "reason": "TESTEMUNHA OK QUASE EXATA",
            "detail": {
                "missing_hard_absence": True,
                "missing_hard_absence_reason": (
                    "estrutura esperada colapsou sem correspondência próxima válida"
                ),
                "final_score": 0.0,
                "physical_score": 0.979522189040151,
                "fusion_rule": "hard_missing_exact_ok_witness",
                "dominant_engine": "knn",
                "decision_trace": {
                    "hard_missing_evidence": False,
                    "raw_hard_missing_evidence": True,
                    "hard_missing_contradicted_by_exact_ok": True,
                    "operator_review_required": False,
                    "fusion_rule": "hard_missing_exact_ok_witness",
                    "memory": {
                        "has_memory": True,
                        "memory_available": True,
                        "best_match_label": "OK",
                        "best_similarity": 0.9999999946355821,
                        "best_ok_similarity": 0.9999999946355821,
                        "best_ng_similarity": 0.8724773603677749,
                        "operator_review_required": False,
                        "suppressed_by_hard_missing": False,
                        "hard_missing_contradicted_by_exact_ok": True,
                        "role": "TESTEMUNHA OK QUASE EXATA",
                    },
                },
            },
        }

        decision = decision_record(
            analysis,
            {"category": "FALTANDO"},
        )
        record = {
            "schema": DEBUG_SCHEMA,
            "event_id": "evt-exact-ok",
            "timestamp": "2026-10-02T08:44:28.829",
            "source_ip": "169.254.95.200",
            "stage": "aoi_intake_validation",
            "mode": "Modo Teste",
            "transport": {"image": {"valid": True}},
            "cycle": {},
            "validation_message": "epicentro válido",
            "validation": {"valid": True, "reason": "valid_epicenter"},
            "decision": decision,
        }

        self.assertFalse(decision["hard_missing_evidence"])
        self.assertTrue(decision["raw_hard_missing_evidence"])
        self.assertTrue(
            decision["hard_missing_contradicted_by_exact_ok"]
        )
        self.assertFalse(
            decision["memory"]["suppressed_by_hard_missing"]
        )

        report = format_network_debug_report(record)
        self.assertIn("Veredito: FALHA FALSA", report)
        self.assertIn("Ausência física forte efetiva: False", report)
        self.assertIn("Ausência física forte bruta: True", report)
        self.assertIn(
            "Hard missing contradito por OK quase exato: True",
            report,
        )
        self.assertIn("KNN melhor rótulo: OK", report)

    def test_report_exposes_cross_category_absence_guard(self):
        record = {
            "schema": DEBUG_SCHEMA,
            "event_id": "evt-emborcado-missing",
            "timestamp": "2026-10-01T08:18:19.174",
            "source_ip": "169.254.95.200",
            "stage": "aoi_intake_validation",
            "mode": "Modo Teste",
            "transport": {"image": {"valid": True}},
            "cycle": {},
            "validation_message": "epicentro válido",
            "validation": {"valid": True, "reason": "valid_epicenter"},
            "decision": {
                "category": "EMBORCADO",
                "verdict": "DEFEITO REAL",
                "is_defect": True,
                "confidence": 0.99,
                "final_score": 1.0,
                "physical_score": 1.0,
                "fusion_rule": "missing_hard_absence",
                "dominant_engine": "missing",
                "operator_review_required": False,
                "hard_missing_evidence": True,
                "reason": "ausência física transversal confirmada",
                "missing": {
                    "missing_hard_absence": True,
                    "missing_cross_category_guard": True,
                    "missing_guard_policy": (
                        "cross_category_physical_absence_guard_v1"
                    ),
                    "missing_guard_source_category": "EMBORCADO",
                    "missing_guard_physical_support": {
                        "supported": True,
                        "structural": 0.54,
                        "semantic": 0.71,
                        "physical_score": 1.0,
                    },
                },
                "memory": {
                    "best_match_label": "OK",
                    "best_similarity": 0.906,
                    "memory_conflict": False,
                    "suppressed_by_hard_missing": True,
                },
            },
        }

        report = format_network_debug_report(record)

        self.assertIn("Categoria: EMBORCADO", report)
        self.assertIn("Ausência física forte efetiva: True", report)
        self.assertIn("Guarda transversal: True", report)
        self.assertIn(
            "Guarda categoria origem: EMBORCADO",
            report,
        )
        self.assertIn(
            "KNN suprimido por ausência física: True",
            report,
        )

    def test_report_exposes_dual_scale_presence_context(self):
        record = {
            "schema": DEBUG_SCHEMA,
            "event_id": "evt-dual-scale",
            "timestamp": "2026-10-01T09:53:28.744",
            "source_ip": "169.254.95.200",
            "stage": "aoi_intake_validation",
            "mode": "Modo Teste",
            "transport": {"image": {"valid": True}},
            "cycle": {},
            "validation_message": "epicentro válido",
            "validation": {"valid": True, "reason": "valid_epicenter"},
            "decision": {
                "category": "FALTANDO",
                "verdict": "DEFEITO REAL",
                "is_defect": True,
                "confidence": 0.99,
                "final_score": 1.0,
                "physical_score": 1.0,
                "fusion_rule": "missing_hard_absence",
                "dominant_engine": "missing",
                "operator_review_required": False,
                "hard_missing_evidence": True,
                "reason": "contexto maior confirmou ausência",
                "missing": {
                    "missing_hard_absence": True,
                    "missing_dual_scale_policy": "dual_scale_presence_v1",
                    "missing_dual_scale_active": True,
                    "missing_dual_scale_triggered": True,
                    "missing_scale_disagreement": True,
                    "missing_local_global_area_ratio": 0.106,
                    "missing_context_box": [110, 9, 350, 270],
                    "missing_context_score": 0.795,
                    "missing_context_coverage": 0.388,
                    "missing_context_residual_mean": 0.688,
                    "missing_context_structure_loss": 0.327,
                    "missing_context_direct_similarity": 0.562,
                    "missing_context_appearance_loss": 0.438,
                    "missing_context_best_similarity": 0.198,
                    "missing_context_hard_absence": True,
                    "missing_context_hard_reason": (
                        "contexto maior confirma desaparecimento físico"
                    ),
                    "missing_context_physical_support": {
                        "supported": True,
                        "structural": 0.56,
                        "semantic": 0.56,
                    },
                },
                "memory": {
                    "best_match_label": "OK",
                    "best_similarity": 0.894,
                    "memory_conflict": False,
                    "suppressed_by_hard_missing": True,
                },
            },
        }

        report = format_network_debug_report(record)

        self.assertIn("Dual-scale ativo: True", report)
        self.assertIn("Dual-scale executado: True", report)
        self.assertIn("Dual-scale desacordo: True", report)
        self.assertIn("Dual-scale razão local/global: 0.106", report)
        self.assertIn(
            "Dual-scale caixa contexto: [110, 9, 350, 270]",
            report,
        )
        self.assertIn("Dual-scale hard absence: True", report)
        self.assertIn(
            "KNN suprimido por ausência física: True",
            report,
        )

    def test_local_mss_report_uses_local_capture_identity(self):
        record = {
            "schema": "visionx.capture_debug.v1",
            "event_id": "local-001",
            "timestamp": "2026-10-01T16:00:00.000",
            "source": "local_mss",
            "source_ip": "",
            "stage": "local_capture_analysis",
            "mode": "Modo Teste",
            "transport": {
                "image": {
                    "valid": True,
                    "shape": [1080, 1920, 3],
                    "dtype": "uint8",
                },
            },
            "cycle": {"source": "local"},
            "validation_message": "Captura local MSS analisada.",
            "validation": {
                "valid": True,
                "reason": "local_capture_processed",
            },
            "decision": {
                "category": "DESLOCADO",
                "verdict": "FALHA FALSA",
                "is_defect": False,
                "confidence": 0.99,
                "memory": {},
                "missing": {},
            },
        }

        report = format_network_debug_report(record)

        self.assertIn(
            "ODIN - Observador Digital Inteligente - DEBUG DA CAPTURA LOCAL MSS",
            report,
        )
        self.assertIn("Origem: Captura local MSS", report)
        self.assertIn("Categoria: DESLOCADO", report)

    def test_empty_record_has_safe_message(self):
        report = format_network_debug_report({})
        self.assertIn("Nenhuma captura", report)


class NetworkXPImageClipboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _panel(event_id="evt-001"):
        class Panel:
            pass

        panel = Panel()
        panel.network_intake_last_validation = {
            "schema": DEBUG_SCHEMA,
            "event_id": event_id,
        }
        panel.network_intake_last_image_event_id = event_id
        panel.network_intake_last_image = np.zeros((12, 18, 3), dtype=np.uint8)
        panel.network_intake_last_image[:, :] = (10, 20, 230)  # BGR
        panel.btn_copy_network_debug = QPushButton("Copiar debug")
        panel.btn_copy_network_image = QPushButton("Copiar imagem")
        return panel

    def test_same_event_enables_both_copy_actions(self):
        panel = self._panel()
        sync_network_debug_controls(panel)

        self.assertTrue(panel.btn_copy_network_debug.isEnabled())
        self.assertTrue(panel.btn_copy_network_image.isEnabled())
        self.assertTrue(network_debug_image_available(panel))

    def test_local_mss_capture_enables_debug_and_exact_image_copy(self):
        panel = self._panel("legacy-network")
        panel.capture_debug_last_record = {
            "schema": "visionx.capture_debug.v1",
            "event_id": "local-002",
            "source": "local_mss",
            "validation": {
                "valid": True,
                "reason": "local_capture_processed",
            },
        }
        panel.capture_debug_last_image_event_id = "local-002"
        panel.capture_debug_last_image = np.full(
            (20, 30, 3),
            (15, 60, 210),
            dtype=np.uint8,
        )

        class Preview:
            def __init__(self):
                self.image = None
                self.placeholder = None

            def set_source_image(self, image):
                self.image = image.copy()

            def clear_source_image(self, placeholder):
                self.placeholder = placeholder
                self.image = None

        panel.lbl_capture_evidence_preview = Preview()

        sync_network_debug_controls(panel)

        self.assertIsNotNone(panel.lbl_capture_evidence_preview.image)
        self.assertEqual(panel.lbl_capture_evidence_preview.image.width(), 30)
        self.assertEqual(panel.lbl_capture_evidence_preview.image.height(), 20)
        preview_pixel = panel.lbl_capture_evidence_preview.image.pixelColor(0, 0)
        self.assertEqual(preview_pixel.red(), 210)
        self.assertEqual(preview_pixel.green(), 60)
        self.assertEqual(preview_pixel.blue(), 15)

        snapshot = network_debug_image_snapshot(panel)
        self.assertIsNotNone(snapshot)
        self.assertTrue(np.array_equal(snapshot, panel.capture_debug_last_image))

        self.assertTrue(panel.btn_copy_network_debug.isEnabled())
        self.assertTrue(panel.btn_copy_network_image.isEnabled())
        self.assertTrue(network_debug_image_available(panel))
        self.assertTrue(copy_network_image_to_clipboard(panel))

        copied = QApplication.clipboard().image()
        self.assertEqual(copied.width(), 30)
        self.assertEqual(copied.height(), 20)
        pixel = copied.pixelColor(0, 0)
        self.assertEqual(pixel.red(), 210)
        self.assertEqual(pixel.green(), 60)
        self.assertEqual(pixel.blue(), 15)

    def test_local_mss_never_falls_back_to_previous_xp_frame(self):
        panel = self._panel("legacy-network")
        panel.capture_debug_last_record = {
            "schema": "visionx.capture_debug.v1",
            "event_id": "local-003",
            "source": "local_mss",
            "validation": {
                "valid": True,
                "reason": "local_capture_processed",
            },
        }
        panel.capture_debug_last_image_event_id = "local-other"
        panel.capture_debug_last_image = np.full(
            (20, 30, 3),
            (15, 60, 210),
            dtype=np.uint8,
        )

        class Preview:
            def __init__(self):
                self.image = "stale"
                self.placeholder = None

            def set_source_image(self, image):
                self.image = image.copy()

            def clear_source_image(self, placeholder):
                self.placeholder = placeholder
                self.image = None

        panel.lbl_capture_evidence_preview = Preview()

        sync_network_debug_controls(panel)

        self.assertTrue(panel.btn_copy_network_debug.isEnabled())
        self.assertFalse(panel.btn_copy_network_image.isEnabled())
        self.assertFalse(network_debug_image_available(panel))
        self.assertIsNone(network_debug_image_snapshot(panel))
        self.assertIsNone(panel.lbl_capture_evidence_preview.image)
        self.assertEqual(
            panel.lbl_capture_evidence_preview.placeholder,
            "Aguardando captura",
        )
        self.assertFalse(copy_network_image_to_clipboard(panel))

    def test_waiting_state_hides_preview_but_preserves_copy_evidence(self):
        panel = self._panel()
        panel.capture_debug_last_record = {
            "schema": DEBUG_SCHEMA,
            "event_id": "evt-waiting",
            "source": "windows_xp",
        }
        panel.capture_debug_last_image_event_id = "evt-waiting"
        panel.capture_debug_last_image = np.full(
            (16, 24, 3),
            (20, 80, 180),
            dtype=np.uint8,
        )
        panel._inspection_images_visible = False

        class Preview:
            def __init__(self):
                self.image = "previous"
                self.placeholder = None

            def set_source_image(self, image):
                self.image = image.copy()

            def clear_source_image(self, placeholder):
                self.image = None
                self.placeholder = placeholder

        panel.lbl_capture_evidence_preview = Preview()

        sync_network_debug_controls(panel)

        self.assertIsNone(panel.lbl_capture_evidence_preview.image)
        self.assertEqual(
            panel.lbl_capture_evidence_preview.placeholder,
            "Aguardando captura",
        )
        self.assertTrue(panel.btn_copy_network_image.isEnabled())
        self.assertTrue(network_debug_image_available(panel))
        snapshot = network_debug_image_snapshot(panel)
        self.assertIsNotNone(snapshot)
        self.assertTrue(
            np.array_equal(snapshot, panel.capture_debug_last_image)
        )

    def test_different_event_blocks_image_copy(self):
        panel = self._panel()
        panel.network_intake_last_image_event_id = "evt-other"
        sync_network_debug_controls(panel)

        self.assertTrue(panel.btn_copy_network_debug.isEnabled())
        self.assertFalse(panel.btn_copy_network_image.isEnabled())
        self.assertFalse(network_debug_image_available(panel))
        self.assertFalse(copy_network_image_to_clipboard(panel))

    def test_copy_image_places_exact_frame_on_clipboard(self):
        panel = self._panel()

        self.assertTrue(copy_network_image_to_clipboard(panel))
        copied = QApplication.clipboard().image()
        self.assertFalse(copied.isNull())
        self.assertEqual(copied.width(), 18)
        self.assertEqual(copied.height(), 12)

        pixel = copied.pixelColor(0, 0)
        self.assertEqual(pixel.red(), 230)
        self.assertEqual(pixel.green(), 20)
        self.assertEqual(pixel.blue(), 10)


class NeutralDecisionBackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_every_analysis_maps_to_neutral_background(self):
        self.assertEqual(decision_background_state(None), "neutral")
        self.assertEqual(decision_background_state({}), "neutral")
        self.assertEqual(
            decision_background_state({"is_defect": False}),
            "neutral",
        )
        self.assertEqual(
            decision_background_state({"is_defect": True}),
            "neutral",
        )

    def test_background_compatibility_hook_never_applies_ok_or_ng_state(self):
        class Panel(QWidget):
            def __init__(self):
                super().__init__()
                self.setObjectName("rootWindow")
                self.root_content = QWidget()
                self.root_content.setObjectName("rootContent")
                self.root_scroll = QScrollArea()
                self.root_scroll.viewport().setObjectName("rootViewport")
                self.section = QFrame(self)
                self.section.setObjectName("sectionPanel")
                self.controls = QFrame(self)
                self.controls.setObjectName("controlsSection")
                self.current_analysis = None
                self.is_locked = False

            def _update_reference_panel(self, analysis):
                self.current_analysis = analysis
                self.is_locked = True

            def _reset_confidence_panel(self):
                self.current_analysis = None
                self.is_locked = False

            def save_label(self, _decision, source="button"):
                self.current_analysis = None
                self.is_locked = False
                return source

        install_decision_background(Panel)
        panel = Panel()

        original_stylesheet = (
            "QPushButton { background: #181818; } "
            "QPushButton:hover { background: #242424; }"
        )
        panel.setStyleSheet(original_stylesheet)

        apply_decision_background(panel, "ng")
        self.assertEqual(panel.property("decisionState"), "neutral")
        self.assertEqual(panel.root_content.property("decisionState"), "neutral")
        self.assertEqual(
            panel.root_scroll.viewport().property("decisionState"),
            "neutral",
        )
        self.assertEqual(panel.section.property("decisionState"), "neutral")
        self.assertEqual(panel.controls.property("decisionState"), "neutral")
        self.assertEqual(panel.styleSheet(), original_stylesheet)

        panel._update_reference_panel({"is_defect": True})
        self.assertEqual(panel.property("decisionState"), "neutral")

        panel._update_reference_panel({"is_defect": False})
        self.assertEqual(panel.property("decisionState"), "neutral")

        panel._reset_confidence_panel()
        self.assertEqual(panel.property("decisionState"), "neutral")

    def test_forced_ng_request_still_renders_neutral_surface(self):
        panel = QWidget()
        panel.setObjectName("rootWindow")
        panel.setStyleSheet(APP_STYLESHEET)
        panel.resize(240, 140)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        section = QFrame(panel)
        section.setObjectName("sectionPanel")
        layout.addWidget(section)

        panel.show()
        self.app.processEvents()

        theme_before = panel.styleSheet()
        apply_decision_background(panel, "ng")
        self.app.processEvents()

        rendered = section.grab().toImage()
        pixel = rendered.pixelColor(
            max(1, rendered.width() // 2),
            max(1, rendered.height() // 2),
        )

        self.assertEqual(pixel.name().lower(), "#0d0d0d")
        self.assertEqual(section.property("decisionState"), "neutral")
        self.assertEqual(panel.styleSheet(), theme_before)
        panel.close()

    def test_main_does_not_install_dynamic_background(self):
        source = open("main.py", encoding="utf-8").read()
        self.assertNotIn("install_decision_background", source)


class NetworkXPDebugUILayoutTests(unittest.TestCase):
    def test_waiting_piece_contract_clears_previous_inspection_images(self):
        source = open(
            "src/ui/control_panel.py",
            encoding="utf-8",
        ).read()
        debug_source = open(
            "src/ui/network_xp_debug.py",
            encoding="utf-8",
        ).read()

        self.assertIn("def _clear_inspection_images(self):", source)
        self.assertIn(
            "self._inspection_images_visible = False",
            source,
        )
        self.assertIn(
            '("lbl_sample", "Aguardando peça")',
            source,
        )
        self.assertIn(
            '("lbl_ng", "Aguardando peça")',
            source,
        )
        self.assertIn(
            'preview.clear_source_image("Aguardando captura")',
            source,
        )
        self.assertIn(
            "self._clear_inspection_images()\n"
            '        self.lbl_verdict.setText("AGUARDANDO PEÇA")',
            source,
        )
        self.assertIn(
            'if not bool(getattr(self, "_inspection_images_visible", False)):\n'
            "            return",
            source,
        )
        self.assertIn(
            "self._inspection_images_visible = True\n"
            '        self.update_brain_status("🧠 Processando Tensores Matemáticos...", True)',
            source,
        )
        self.assertIn(
            'if not bool(getattr(panel, "_inspection_images_visible", True)):',
            debug_source,
        )

    def test_inspection_section_has_responsive_full_capture_preview(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()
        debug_source = open(
            "src/ui/network_xp_debug.py",
            encoding="utf-8",
        ).read()

        self.assertIn("class _ResponsiveCapturePreview(QLabel):", source)
        self.assertIn(
            '"CAPTURA RECEBIDA • EVIDÊNCIA COMPLETA"',
            source,
        )
        self.assertIn(
            "window.lbl_capture_evidence_preview = _ResponsiveCapturePreview(",
            source,
        )
        self.assertLess(
            source.index('"CAPTURA RECEBIDA • EVIDÊNCIA COMPLETA"'),
            source.index('"GABARITO • VISÃO COMPLETA"'),
        )
        self.assertIn(
            "preview_min_height = 140 if compact else 180",
            source,
        )
        self.assertIn(
            "image = network_debug_image_snapshot(panel)",
            debug_source,
        )
        self.assertGreaterEqual(
            debug_source.count("image = network_debug_image_snapshot(panel)"),
            2,
        )

    def test_debug_actions_live_outside_global_status_bar(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()

        self.assertIn("def _build_network_debug_bar", source)
        self.assertIn('QFrame#networkDebugFrame', source)
        self.assertIn('btn_copy_network_image', source)
        self.assertIn('copy_network_image_to_clipboard', source)
        self.assertNotIn(
            "status_layout.addWidget(window.btn_copy_network_debug)",
            source,
        )
        self.assertNotIn(
            "status_layout.addWidget(window.btn_copy_network_image)",
            source,
        )

    def test_copy_actions_share_one_responsive_action_group(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()

        self.assertIn(
            "self.network_debug_actions_layout.addWidget(\n"
            "            window.btn_copy_network_debug,",
            source,
        )
        self.assertIn(
            "self.network_debug_actions_layout.addWidget(\n"
            "            window.btn_copy_network_image,",
            source,
        )
        self.assertIn(
            'window.btn_copy_network_image = QPushButton("Copiar imagem")',
            source,
        )

        layout_start = source.index("def _layout_network_debug")
        layout_end = source.index("def _build_status_bar", layout_start)
        layout_source = source[layout_start:layout_end]
        compact_start = layout_source.index("if compact:")
        compact = layout_source[
            compact_start:
            layout_source.index("else:", compact_start)
        ]
        self.assertIn(
            "grid.addWidget(window.network_debug_actions, 2, 0)",
            compact,
        )
        self.assertNotIn("btn_copy_network_debug", compact)
        self.assertNotIn("btn_copy_network_image", compact)

    def test_network_debug_buttons_have_clickable_responsive_policy(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()

        self.assertIn("min-height: 40px", source)
        self.assertIn(
            "window.btn_copy_network_debug.setFocusPolicy(Qt.FocusPolicy.StrongFocus)",
            source,
        )
        self.assertIn(
            "window.btn_copy_network_image.setFocusPolicy(Qt.FocusPolicy.StrongFocus)",
            source,
        )
        self.assertIn(
            "window.network_debug_actions.setMinimumWidth(0 if compact else 300)",
            source,
        )


    def test_full_page_uses_viewport_width_and_compact_reflow(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()

        self.assertIn(
            "window.root_scroll.viewport().installEventFilter",
            source,
        )
        self.assertIn(
            "self._layout_header(window, compact=compact)",
            source,
        )
        self.assertIn(
            "self._layout_status_bar(window, compact=compact)",
            source,
        )
        self.assertIn(
            "window.main_splitter.setMinimumWidth(0)",
            source,
        )
        self.assertIn(
            "window.telemetry_section.setMinimumWidth(0)",
            source,
        )
        self.assertNotIn(
            "max(700, width - image_width)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
