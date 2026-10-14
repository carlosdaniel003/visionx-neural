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
        self.assertIn("window.normal_telemetry_view", source)
        self.assertIn("window.adhesive_multilight_analysis_view", source)
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


if __name__ == "__main__":
    unittest.main()
