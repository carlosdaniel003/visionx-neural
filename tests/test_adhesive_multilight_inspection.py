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

    def test_three_lighting_cards_exist_with_nine_viewports(self):
        view = AdhesiveMultiLightView()

        self.assertEqual(set(view.cards), {"SIDE", "TOP", "MID"})
        for card in view.cards.values():
            self.assertIsNotNone(card.main_view)
            self.assertIsNotNone(card.large_view)
            self.assertIsNotNone(card.small_view)


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

        analyze_index = source.index("lighting_analysis = analyze_lighting(")
        render_index = source.index(
            "analysis_view.set_analysis(\n"
            "                        aux_mode,"
        )
        advance_index = source.index("frame_stored(aux_mode)")

        self.assertLess(analyze_index, render_index)
        self.assertLess(render_index, advance_index)
        self.assertIn("adhesive_multilight_analyses", source)
        self.assertIn(
            "getattr(self, \"current_aoi_info\", None)",
            source,
        )

    def test_auxiliary_frames_do_not_reopen_main_cycle(self):
        source = open("src/services/network_receiver.py", encoding="utf-8").read()

        self.assertIn("auxiliary_delivery", source)
        self.assertIn("self.image_received.emit(img, ip_origem)", source)
        self.assertIn("set_auxiliary_image_mode", source)


if __name__ == "__main__":
    unittest.main()
