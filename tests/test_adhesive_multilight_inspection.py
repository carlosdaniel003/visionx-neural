from copy import deepcopy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

import src.ui.adhesive_multilight_inspection as multilight_module
from src.services.network_receiver import NetworkReceiver
from src.ui.adhesive_multilight_inspection import (
    AdhesiveMultiLightView,
    build_adhesive_view_payload,
    category_from_aoi_info,
    is_adhesive_category,
    is_multilight_category,
)
from src.utils.text_normalizer import normalize_aoi_text


class AdhesiveCategoryTests(unittest.TestCase):
    def test_all_requested_adhesive_aliases_are_canonical(self):
        for text in (
            "ADESIVO",
            "MUITO ADESIVO",
            "MUCH ADHESIVE",
            "ADHESIVE",
        ):
            with self.subTest(text=text):
                category, _value = normalize_aoi_text(text)
                self.assertEqual(category, "MUITO ADESIVO")

    def test_view_activation_accepts_category_or_raw_value(self):
        self.assertTrue(
            is_adhesive_category({"category": "MUITO ADESIVO"})
        )
        self.assertTrue(
            is_adhesive_category({"value": "MUCH ADHESIVE"})
        )
        self.assertTrue(
            is_adhesive_category({"value": "ADESIVO"})
        )
        self.assertFalse(
            is_adhesive_category({"value": "MISSING"})
        )
        self.assertEqual(
            category_from_aoi_info({"value": "ADESIVO"}),
            "MUITO ADESIVO",
        )

    def test_every_known_aoi_category_is_multilight_eligible(self):
        self.assertTrue(is_multilight_category({"value": "MISSING"}))
        self.assertTrue(is_multilight_category({"category": "INVERTIDO"}))
        self.assertTrue(is_multilight_category({"category": "DESLOCADO"}))
        self.assertTrue(is_multilight_category({"value": "ADESIVO"}))
        self.assertFalse(is_multilight_category({"category": "Unknown"}))


class AdhesivePayloadTests(unittest.TestCase):
    def test_payload_contains_main_large_and_small_views(self):
        sample = np.full((300, 300, 3), (30, 30, 30), dtype=np.uint8)
        test = sample.copy()

        large_box = {"x": 20, "y": 30, "w": 240, "h": 210, "detected": True}
        small_box = (100, 90, 70, 60)
        small_crop = test[90:150, 100:170].copy()

        context = {
            "valid": True,
            "raw_anomalies": [],
            "old_epicenters": [small_box],
            "global_box_info": large_box,
            "real_epicenters": [small_box],
            "focus_gab": sample[90:150, 100:170].copy(),
            "focus_ng": small_crop,
        }
        payload = build_adhesive_view_payload(
            sample,
            test,
            context=context,
        )

        self.assertTrue(np.array_equal(payload["main"], test))
        self.assertEqual(payload["large"].shape[:2], (210, 240))
        self.assertEqual(payload["small"].shape[:2], (60, 70))
        self.assertEqual(payload["small_box"], small_box)
        self.assertEqual(payload["large_box"]["w"], 240)

    def test_visual_payload_failure_never_breaks_main_image(self):
        sample = np.full((80, 80, 3), 40, dtype=np.uint8)
        test = sample.copy()

        with patch.object(
            multilight_module,
            "build_lighting_context",
            side_effect=RuntimeError("visual-only failure"),
        ):
            payload = build_adhesive_view_payload(sample, test)

        self.assertTrue(np.array_equal(payload["main"], test))
        self.assertEqual(payload["large"].size, 0)
        self.assertEqual(payload["small"].size, 0)


class AdhesiveResponsiveViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_notebook_and_monitor_breakpoints(self):
        self.assertEqual(AdhesiveMultiLightView.columns_for_width(900), 1)
        self.assertEqual(AdhesiveMultiLightView.columns_for_width(999), 1)
        self.assertEqual(AdhesiveMultiLightView.columns_for_width(1000), 3)
        self.assertEqual(AdhesiveMultiLightView.columns_for_width(1600), 3)

    def test_three_lighting_cards_exist_with_twelve_viewports(self):
        view = AdhesiveMultiLightView()

        self.assertEqual(set(view.cards), {"SIDE", "TOP", "MID"})
        for card in view.cards.values():
            self.assertIsNotNone(card.original_view)
            self.assertIsNotNone(card.main_view)
            self.assertIsNotNone(card.large_view)
            self.assertIsNotNone(card.small_view)

    def test_viewport_layout_hints_do_not_depend_on_scaled_pixmap(self):
        view = AdhesiveMultiLightView()
        viewport = view.cards["SIDE"].main_view
        original_hint = viewport.sizeHint()
        original_minimum = viewport.minimumSizeHint()
        viewport.resize(240, 300)
        viewport.show()
        pixels = np.full((540, 300, 3), 117, dtype=np.uint8)
        self.assertTrue(viewport.set_bgr_image(pixels))
        self.app.processEvents()
        self.assertEqual(viewport.sizeHint(), original_hint)
        self.assertEqual(viewport.minimumSizeHint(), original_minimum)
        rendered = viewport._rendered_target_size
        viewport._render()
        self.assertEqual(viewport._rendered_target_size, rendered)
        viewport.resize(200, 260)
        self.app.processEvents()
        self.assertEqual(viewport.sizeHint(), original_hint)
        self.assertEqual(viewport.minimumSizeHint(), original_minimum)

    def test_each_card_shows_its_own_full_xp_frame_and_original_test_crop(self):
        view = AdhesiveMultiLightView()
        for mode, value in (("SIDE", 35), ("TOP", 95), ("MID", 155)):
            with self.subTest(mode=mode):
                original = np.full((84, 116, 3), (value, 30, 10), dtype=np.uint8)
                test = np.full((43, 31, 3), (10, 70, value), dtype=np.uint8)
                self.assertTrue(view.set_lighting_payload(mode, {
                    "original_frame": original,
                    "main": test,
                }))
                card = view.cards[mode]
                self.assertEqual(
                    (card.original_view._source_pixmap.width(),
                     card.original_view._source_pixmap.height()),
                    (116, 84),
                )
                self.assertEqual(
                    (card.main_view._source_pixmap.width(),
                     card.main_view._source_pixmap.height()),
                    (31, 43),
                )
                original_pixel = card.original_view._source_pixmap.toImage().pixelColor(0, 0)
                test_pixel = card.main_view._source_pixmap.toImage().pixelColor(0, 0)
                self.assertEqual(
                    (original_pixel.red(), original_pixel.green(), original_pixel.blue()),
                    (10, 30, value),
                )
                self.assertEqual(
                    (test_pixel.red(), test_pixel.green(), test_pixel.blue()),
                    (value, 70, 10),
                )

    def test_missing_original_does_not_display_test_as_original_and_reset_clears_all(self):
        view = AdhesiveMultiLightView()
        test = np.full((40, 50, 3), 100, dtype=np.uint8)
        view.set_lighting_payload("SIDE", {"main": test})
        card = view.cards["SIDE"]
        self.assertTrue(card.original_view._source_pixmap.isNull())
        self.assertFalse(card.main_view._source_pixmap.isNull())
        self.assertIn("indisponível", card.original_view.text())

        view.set_lighting_payload(
            "MID", {"original_frame": np.full((80, 100, 3), 33, dtype=np.uint8)}
        )
        view.clear_all()
        for card in view.cards.values():
            self.assertTrue(card.original_view._source_pixmap.isNull())
            self.assertTrue(card.main_view._source_pixmap.isNull())

    def test_store_view_routes_raw_xp_frame_separately_from_inspection_crop(self):
        from types import SimpleNamespace

        view = AdhesiveMultiLightView()
        panel = SimpleNamespace(
            combo_mode=SimpleNamespace(currentText=lambda: "Modo Teste"),
            adhesive_multilight_views={},
            adhesive_multilight_view=view,
            adhesive_multilight_analysis_view=None,
            current_aoi_info={"category": "FALTANDO"},
        )
        sample = np.full((40, 40, 3), 12, dtype=np.uint8)
        test = np.full((40, 40, 3), 45, dtype=np.uint8)
        original = np.full((80, 120, 3), 90, dtype=np.uint8)
        self.assertTrue(multilight_module._store_view(
            panel, "TOP", sample, test, context={"valid": False},
            source_frame=original,
        ))
        payload = panel.adhesive_multilight_views["TOP"]
        self.assertTrue(np.array_equal(payload["original_frame"], original))
        self.assertTrue(np.array_equal(payload["main"], test))
        self.assertFalse(view.cards["TOP"].original_view._source_pixmap.isNull())
        self.assertTrue(view.cards["SIDE"].original_view._source_pixmap.isNull())


class AuxiliaryNetworkModeTests(unittest.TestCase):
    def test_auxiliary_mode_is_explicit_and_does_not_open_main_gate(self):
        receiver = NetworkReceiver(port=0)
        self.assertFalse(receiver.is_auxiliary_image_mode())
        self.assertTrue(receiver.is_image_gate_open())

        receiver.lock_image_gate()
        self.assertFalse(receiver.is_image_gate_open())

        receiver.set_auxiliary_image_mode(True)
        self.assertTrue(receiver.is_auxiliary_image_mode())
        self.assertFalse(receiver.is_image_gate_open())

        receiver.set_auxiliary_image_mode(False)
        self.assertFalse(receiver.is_auxiliary_image_mode())
        self.assertFalse(receiver.is_image_gate_open())


class AdhesiveSourceContractTests(unittest.TestCase):
    def test_main_installs_multilight_outside_network_and_local_guards(self):
        source = open("main.py", encoding="utf-8").read()

        intake = source.index("install_network_aoi_intake_filter(ControlPanel)")
        local = source.index("install_local_capture_safety(ControlPanel)")
        multilight = source.index(
            "install_adhesive_multilight_inspection(ControlPanel)"
        )

        self.assertLess(intake, local)
        self.assertLess(local, multilight)

    def test_non_adhesive_default_view_remains_available(self):
        source = open("src/ui/control_panel_ui.py", encoding="utf-8").read()

        self.assertIn("window.normal_inspection_view", source)
        self.assertIn("window.adhesive_multilight_view", source)
        self.assertNotIn("window.normal_telemetry_view", source)
        self.assertNotIn("window.adhesive_multilight_analysis_view", source)
        self.assertNotIn("ANÁLISE DOS ESPECIALISTAS", source)
        self.assertIn("_CurrentPageStack", source)
        self.assertIn(
            "window.inspection_view_stack.setCurrentWidget(\n"
            "            window.normal_inspection_view",
            source,
        )

    def test_auxiliary_frames_run_visual_experts_before_advancing_automation(self):
        source = open(
            "src/ui/adhesive_multilight_inspection.py",
            encoding="utf-8",
        ).read()

        store_index = source.index("stored = _store_view(")
        prefetch_index = source.index("early_advanced = False")
        analyze_index = source.index("lighting_analysis = analyze_lighting(")
        render_index = source.index(
            "analysis_view.set_analysis(\n"
            "                        aux_mode,"
        )
        advance_index = source.index("frame_stored(aux_mode)")

        # TOP válida precede o pedido antecipado de MID, sem dispensar a
        # análise e a renderização antes do fechamento do ciclo.
        self.assertLess(store_index, prefetch_index)
        self.assertLess(prefetch_index, analyze_index)
        self.assertLess(analyze_index, render_index)
        self.assertLess(render_index, advance_index)
        self.assertIn('if aux_mode == "TOP":', source)
        self.assertIn('getattr(automation, "top_captured", None)', source)
        self.assertIn("adhesive_multilight_analyses", source)
        self.assertIn(
            "getattr(self, \"current_aoi_info\", None)",
            source,
        )

    def test_side_display_rejects_debug_fallback_from_another_event(self):
        source = open(
            "src/ui/adhesive_multilight_inspection.py",
            encoding="utf-8",
        ).read()
        self.assertIn("side_original_for_view = (", source)
        self.assertIn('"network_intake_last_image_event_id"', source)
        self.assertIn("source_frame=side_original_for_view", source)
        self.assertIn("capture_debug_last_image", source)

    def test_auxiliary_frames_do_not_reopen_main_cycle(self):
        source = open("src/services/network_receiver.py", encoding="utf-8").read()

        self.assertIn("auxiliary_delivery", source)
        self.assertIn("self.image_received.emit(img, ip_origem)", source)
        self.assertIn("set_auxiliary_image_mode", source)



class SideEarlyExitCaptureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _run_capture(self, *, confidence, verdict, mode, side_analysis=None, category="FALTANDO"):
        import time
        from types import SimpleNamespace
        from unittest.mock import Mock

        active_analysis = {
            "verdict": verdict,
            "is_defect": verdict == "DEFEITO REAL",
            "confidence": confidence,
            "production_review_required": False,
            "detail": {
                "recognition_route": "NEW_EXPERTS",
                "decision_trace": {"operator_review_required": False},
            },
        }

        if side_analysis is not None:
            active_analysis = side_analysis

        class FakePanel:
            def __init__(self):
                self.combo_mode = SimpleNamespace(currentText=lambda: mode)
                self.capture_cycle_source = "network"
                self.capture_start_time = time.perf_counter()
                self.capture_start_source = "network_payload_received"
                self.current_aoi_info = {"category": category}
                self.current_ng = np.zeros((64, 64, 3), dtype=np.uint8)
                self.current_analysis = None
                self.ui_builder = None
                self.network_receiver = SimpleNamespace(
                    set_auxiliary_image_mode=Mock()
                )
                self.adhesive_multilight_automation = SimpleNamespace(start=Mock())
                self.adhesive_multilight_view = None
                self.adhesive_multilight_analysis_view = None
                self.lbl_verdict = SimpleNamespace(
                    setText=Mock(), setStyleSheet=Mock()
                )
                self.lbl_reason = SimpleNamespace(setText=Mock())
                self.lbl_timer = SimpleNamespace(setText=Mock(), setToolTip=Mock())
                self.btn_start = SimpleNamespace(setEnabled=Mock())
                self.btn_skip = SimpleNamespace(setEnabled=Mock())
                self.btn_save_ok = SimpleNamespace(setEnabled=Mock())
                self.btn_save_ng = SimpleNamespace(setEnabled=Mock())
                self.update_network_status = Mock()
                self.update_brain_status = Mock()
                self.notify_production_analysis_ready = Mock()
                self.send_command_to_xp = Mock()
                self._update_confidence_panel = Mock()
                self._update_reference_panel = Mock()

            def handle_network_image(self, img, ip):
                return True

            def process_aoi_images(self, sample, test, info):
                self.current_ng = test
                self.current_analysis = active_analysis
                return True

            def save_label(self, *args, **kwargs):
                return True

            def skip_image(self, *args, **kwargs):
                return True

        multilight_module.install_adhesive_multilight_inspection(FakePanel)
        panel = FakePanel()
        pixels = np.full((64, 64, 3), 110, dtype=np.uint8)
        with (
            patch.object(multilight_module, "_store_view", return_value=True),
            patch.object(multilight_module, "_sync_multilight_debug_controls"),
            patch.object(multilight_module, "capture_debug_record", return_value=None),
        ):
            result = panel.process_aoi_images(pixels, pixels.copy(), {"category": category})
            QApplication.processEvents()
        self.assertTrue(result)
        return panel

    def test_97_percent_side_skips_top_mid_in_test_shadow_and_production(self):
        from src.ui.production_confidence_gate import production_decision_policy

        for mode in ("Modo Teste", "Modo Sombra", "Modo Produção"):
            for verdict in ("DEFEITO REAL", "FALHA FALSA"):
                with self.subTest(mode=mode, verdict=verdict):
                    panel = self._run_capture(
                        confidence=0.97, verdict=verdict, mode=mode
                    )
                    panel.adhesive_multilight_automation.start.assert_not_called()
                    panel.send_command_to_xp.assert_not_called()
                    self.assertFalse(panel.adhesive_multilight_active)
                    self.assertTrue(panel.adhesive_multilight_final_analysis["multilight_final"])
                    self.assertEqual(panel.current_analysis["verdict"], verdict)
                    self.assertEqual(set(panel.adhesive_multilight_analyses), {"SIDE"})
                    self.assertEqual(
                        panel.current_analysis["detail"]["side_early_exit"]["skipped_lights"],
                        ["TOP", "MID"],
                    )
                    self.assertTrue(production_decision_policy(panel.current_analysis)["auto_allowed"])
                    if mode == "Modo Produção":
                        panel.notify_production_analysis_ready.assert_called_once()
                    else:
                        panel.notify_production_analysis_ready.assert_not_called()

    def test_trial_ng_ends_after_side_in_all_modes_without_sending_xp(self):
        from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
        from src.ui.production_confidence_gate import production_decision_policy

        for mode in ("Modo Teste", "Modo Sombra", "Modo Produção"):
            side = {
                "verdict": "DEFEITO REAL", "is_defect": True,
                "confidence": .9999722242355347,
                "lighting_mode": "SIDE",
                "production_review_required": True,
                "detail": {
                    "recognition_route": "NEW_CNN",
                    "operator_review_required": True,
                    "cnn_v2_active": True,
                    "cnn_v2_experimental": True,
                    "cnn_v2_status": "INFERENCE_OK",
                    "cnn_v2_aoi_category": "FALTANDO",
                    "cnn_v2_lighting_mode": "SIDE",
                    "cnn_v2_checkpoint_verified": True,
                    "cnn_v2_checkpoint_sha256": TRIAL_97P70_SHA256,
                    "cnn_v2_trial_manual_only": True,
                    "cnn_v2_ng_score_uncalibrated": .9999722242355347,
                    "decision_trace": {
                        "operator_review_required": True,
                        "fusion_rule": "cnn_v2_dualscale_direct",
                    },
                },
            }
            with self.subTest(mode=mode):
                panel = self._run_capture(
                    confidence=side["confidence"], verdict="DEFEITO REAL",
                    mode=mode, side_analysis=side,
                )
                panel.adhesive_multilight_automation.start.assert_not_called()
                panel.network_receiver.set_auxiliary_image_mode.assert_called_with(False)
                panel.send_command_to_xp.assert_not_called()
                self.assertFalse(panel.adhesive_multilight_active)
                self.assertEqual(set(panel.adhesive_multilight_analyses), {"SIDE"})
                self.assertTrue(panel.current_analysis["production_review_required"])
                self.assertEqual(
                    panel.current_analysis["detail"]["side_early_exit"]["skipped_lights"],
                    ["TOP", "MID"],
                )
                policy = production_decision_policy(panel.current_analysis)
                # Somente o controlador de Produção pode enviar a decisão;
                # a captura não transmite tecla nem em Teste/Sombra.
                self.assertTrue(policy["auto_allowed"])
                self.assertFalse(policy["operator_review_required"])
                self.assertEqual(policy["proposed_decision"], "NG")
                self.assertEqual(policy["reason"], "side_trial_verified_auto_ng")
                if mode == "Modo Produção":
                    panel.notify_production_analysis_ready.assert_called_once()
                else:
                    panel.notify_production_analysis_ready.assert_not_called()

    def test_trial_invertido_ng_finishes_after_side_in_all_modes(self):
        from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
        from src.ui.production_confidence_gate import production_decision_policy

        for mode in ("Modo Teste", "Modo Sombra", "Modo Produção"):
            side = {
                "verdict": "DEFEITO REAL",
                "is_defect": True,
                "confidence": .9999839067459106,
                "lighting_mode": "SIDE",
                "production_review_required": True,
                "detail": {
                    "recognition_route": "NEW_CNN",
                    "operator_review_required": True,
                    "cnn_v2_active": True,
                    "cnn_v2_experimental": True,
                    "cnn_v2_status": "INFERENCE_OK",
                    "cnn_v2_aoi_category": "INVERTIDO",
                    "cnn_v2_lighting_mode": "SIDE",
                    "cnn_v2_checkpoint_verified": True,
                    "cnn_v2_checkpoint_sha256": TRIAL_97P70_SHA256,
                    "cnn_v2_trial_manual_only": True,
                    "cnn_v2_ng_score_uncalibrated": .9999839067459106,
                    "decision_trace": {
                        "operator_review_required": True,
                        "fusion_rule": "cnn_v2_dualscale_direct",
                    },
                    "knn_c": {"status": "REVISAO", "bank_status": "LOADING"},
                },
            }
            with self.subTest(mode=mode):
                panel = self._run_capture(
                    confidence=side["confidence"], verdict="DEFEITO REAL",
                    mode=mode, side_analysis=side, category="INVERTIDO",
                )
                panel.adhesive_multilight_automation.start.assert_not_called()
                panel.network_receiver.set_auxiliary_image_mode.assert_called_with(False)
                panel.send_command_to_xp.assert_not_called()
                self.assertFalse(panel.adhesive_multilight_active)
                self.assertEqual(set(panel.adhesive_multilight_analyses), {"SIDE"})
                self.assertEqual(panel.current_analysis["multilight_category"], "INVERTIDO")
                self.assertEqual(
                    panel.current_analysis["detail"]["side_early_exit"]["skipped_lights"],
                    ["TOP", "MID"],
                )
                policy = production_decision_policy(panel.current_analysis)
                self.assertEqual(policy["proposed_decision"], "NG")
                self.assertTrue(policy["auto_allowed"])
                if mode == "Modo Produção":
                    panel.notify_production_analysis_ready.assert_called_once()
                else:
                    panel.notify_production_analysis_ready.assert_not_called()

    def test_test_mode_cnn_ng_knn_both_ok_reviews_continue_top_mid_only_here(self):
        from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
        from src.ui.production_confidence_gate import production_decision_policy
        base = {
            "verdict": "DEFEITO REAL", "is_defect": True,
            "confidence": .9998650550842285,
            "lighting_mode": "SIDE", "production_review_required": True,
            "detail": {
                "recognition_route": "NEW_CNN",
                "operator_review_required": True,
                "cnn_v2_active": True,
                "cnn_v2_experimental": True,
                "cnn_v2_status": "INFERENCE_OK",
                "cnn_v2_aoi_category": "FALTANDO",
                "cnn_v2_lighting_mode": "SIDE",
                "cnn_v2_checkpoint_verified": True,
                "cnn_v2_checkpoint_sha256": TRIAL_97P70_SHA256,
                "cnn_v2_trial_manual_only": True,
                "cnn_v2_ng_score_uncalibrated": .9998650550842285,
                "decision_trace": {
                    "operator_review_required": True,
                    "fusion_rule": "cnn_v2_dualscale_direct",
                },
                "knn_c": {
                    "status": "REVISAO", "reason": "SEM_SUGESTAO_CONFIAVEL",
                    "bank_status": "READY", "diagnostic_only": True,
                    "auto_decision_authorized": False,
                    "vote_A": "OK", "vote_B": "OK",
                    "suggestion_A": "REVISAO", "suggestion_B": "REVISAO",
                },
            },
        }
        for mode in ("Modo Teste", "Modo Sombra", "Modo Produção"):
            with self.subTest(mode=mode):
                panel = self._run_capture(
                    confidence=base["confidence"], verdict="DEFEITO REAL",
                    mode=mode, side_analysis=deepcopy(base),
                )
                panel.send_command_to_xp.assert_not_called()
                if mode == "Modo Teste":
                    self.assertTrue(panel.adhesive_multilight_active)
                    self.assertIsNone(panel.adhesive_multilight_final_analysis)
                    self.assertEqual(set(panel.adhesive_multilight_analyses), {"SIDE"})
                    panel.adhesive_multilight_automation.start.assert_called_once()
                    panel.network_receiver.set_auxiliary_image_mode.assert_called_with(True)
                    panel.notify_production_analysis_ready.assert_not_called()
                    self.assertEqual(panel.current_analysis["verdict"], "DEFEITO REAL")
                    self.assertTrue(panel.current_analysis["is_defect"])
                    self.assertEqual(
                        panel.current_analysis["detail"][
                            "test_mode_side_knn_disagreement"]["status"],
                        "DEFER_TO_TOP_MID",
                    )
                    self.assertNotIn("side_early_exit", panel.current_analysis["detail"])
                    panel.btn_save_ok.setEnabled.assert_called_with(False)
                    panel.btn_save_ng.setEnabled.assert_called_with(False)
                else:
                    self.assertFalse(panel.adhesive_multilight_active)
                    panel.adhesive_multilight_automation.start.assert_not_called()
                    self.assertIsNotNone(panel.adhesive_multilight_final_analysis)
                    self.assertNotIn(
                        "test_mode_side_knn_disagreement",
                        panel.current_analysis["detail"],
                    )
                    self.assertEqual(
                        production_decision_policy(panel.current_analysis)["proposed_decision"],
                        "NG",
                    )
                    if mode == "Modo Produção":
                        panel.notify_production_analysis_ready.assert_called_once()
                    else:
                        panel.notify_production_analysis_ready.assert_not_called()

    def test_real_loading_case_test_only_collects_top_mid_no_auto_xp(self):
        from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
        from src.ui.production_confidence_gate import production_decision_policy
        side = {
            "verdict": "DEFEITO REAL", "is_defect": True,
            "confidence": 0.9998650550842285, "lighting_mode": "SIDE",
            "production_review_required": True,
            "detail": {
                "recognition_route": "NEW_CNN",
                "operator_review_required": True,
                "cnn_v2_active": True,
                "cnn_v2_experimental": True,
                "cnn_v2_status": "INFERENCE_OK",
                "cnn_v2_aoi_category": "FALTANDO",
                "cnn_v2_lighting_mode": "SIDE",
                "cnn_v2_checkpoint_verified": True,
                "cnn_v2_checkpoint_sha256": TRIAL_97P70_SHA256,
                "cnn_v2_trial_manual_only": True,
                "cnn_v2_ng_score_uncalibrated": 0.9998650550842285,
                "decision_trace": {
                    "operator_review_required": True,
                    "fusion_rule": "cnn_v2_dualscale_direct",
                },
                "knn_c": {
                    "schema": "visionx.knn_c_live_review_gated.v1",
                    "status": "REVISAO", "reason": "MEMORIA_VISUAL_EM_PREPARACAO",
                    "bank_status": "LOADING", "diagnostic_only": True,
                    "auto_decision_authorized": False,
                    "reference_count": 0, "xp_command": None,
                },
            },
        }
        for mode in ("Modo Teste", "Modo Sombra", "Modo Produção"):
            with self.subTest(mode=mode):
                panel = self._run_capture(
                    confidence=side["confidence"], verdict="DEFEITO REAL",
                    mode=mode, side_analysis=deepcopy(side),
                )
                panel.send_command_to_xp.assert_not_called()
                self.assertEqual(panel.current_analysis["verdict"], "DEFEITO REAL")
                if mode == "Modo Teste":
                    self.assertTrue(panel.adhesive_multilight_active)
                    panel.network_receiver.set_auxiliary_image_mode.assert_called_with(True)
                    panel.adhesive_multilight_automation.start.assert_called_once()
                    panel.notify_production_analysis_ready.assert_not_called()
                    self.assertIsNone(panel.adhesive_multilight_final_analysis)
                    self.assertEqual(
                        panel.current_analysis["detail"][
                            "test_mode_side_knn_disagreement"]["reason"],
                        "CNN_NG_KNN_C_LOADING_SEM_VOTO",
                    )
                    self.assertEqual(set(panel.adhesive_multilight_analyses), {"SIDE"})
                    panel.btn_save_ok.setEnabled.assert_called_with(False)
                    panel.btn_save_ng.setEnabled.assert_called_with(False)
                else:
                    self.assertFalse(panel.adhesive_multilight_active)
                    panel.adhesive_multilight_automation.start.assert_not_called()
                    self.assertIsNotNone(panel.adhesive_multilight_final_analysis)
                    self.assertNotIn(
                        "test_mode_side_knn_disagreement",
                        panel.current_analysis["detail"],
                    )
                    self.assertEqual(
                        production_decision_policy(panel.current_analysis)["proposed_decision"],
                        "NG",
                    )

    def test_test_mode_without_knn_both_ok_maintains_side_early(self):
        from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
        side = {
            "verdict": "DEFEITO REAL", "is_defect": True,
            "confidence": .9998650550842285, "lighting_mode": "SIDE",
            "production_review_required": True,
            "detail": {
                "recognition_route": "NEW_CNN", "operator_review_required": True,
                "cnn_v2_active": True, "cnn_v2_experimental": True,
                "cnn_v2_status": "INFERENCE_OK", "cnn_v2_aoi_category": "FALTANDO",
                "cnn_v2_lighting_mode": "SIDE", "cnn_v2_checkpoint_verified": True,
                "cnn_v2_checkpoint_sha256": TRIAL_97P70_SHA256,
                "cnn_v2_trial_manual_only": True,
                "cnn_v2_ng_score_uncalibrated": .9998650550842285,
                "decision_trace": {
                    "operator_review_required": True,
                    "fusion_rule": "cnn_v2_dualscale_direct",
                },
                "knn_c": {
                    "status": "REVISAO", "bank_status": "READY",
                    "diagnostic_only": True, "auto_decision_authorized": False,
                    "vote_A": "NG", "vote_B": "OK",
                    "suggestion_A": "REVISAO", "suggestion_B": "REVISAO",
                },
            },
        }
        panel = self._run_capture(
            confidence=side["confidence"], verdict="DEFEITO REAL",
            mode="Modo Teste", side_analysis=side,
        )
        self.assertFalse(panel.adhesive_multilight_active)
        panel.adhesive_multilight_automation.start.assert_not_called()
        self.assertIn("side_early_exit", panel.current_analysis["detail"])

    def test_just_below_threshold_keeps_three_light_capture(self):
        panel = self._run_capture(
            confidence=0.9699, verdict="FALHA FALSA", mode="Modo Teste"
        )
        panel.adhesive_multilight_automation.start.assert_called_once()
        self.assertTrue(panel.adhesive_multilight_active)
        self.assertIsNone(panel.adhesive_multilight_final_analysis)
        self.assertEqual(set(panel.adhesive_multilight_analyses), {"SIDE"})
        panel.network_receiver.set_auxiliary_image_mode.assert_called_with(True)


if __name__ == "__main__":
    unittest.main()
