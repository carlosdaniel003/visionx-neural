import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QPushButton

from src.ui.network_xp_debug import (
    DEBUG_SCHEMA,
    copy_network_image_to_clipboard,
    format_network_debug_report,
    network_debug_image_available,
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

        self.assertIn("DECISÃO ODIN", report)
        self.assertIn("Categoria: FALTANDO", report)
        self.assertIn("Regra de fusão: missing_hard_absence", report)
        self.assertIn("Ausência física forte: True", report)
        self.assertIn("KNN melhor rótulo: OK", report)
        self.assertIn("KNN suprimido por ausência física: True", report)

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
        self.assertIn("Ausência física forte: True", report)
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

    def test_empty_record_has_safe_message(self):
        report = format_network_debug_report({})
        self.assertIn("Nenhuma imagem recebida", report)


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


class NetworkXPDebugUILayoutTests(unittest.TestCase):
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
            'window.btn_copy_network_image = QPushButton("Copiar imagem XP")',
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
