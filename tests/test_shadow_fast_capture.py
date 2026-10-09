"""Modo Sombra rápido: isolamento de performance, rótulo humano e 0/1 seguro."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication, QWidget

from src.services.screen_monitor import ScreenMonitor
from src.services.anomaly_learning import _decision_task
from src.ui.control_panel import ControlPanel
from src.ui.adhesive_multilight_inspection import (
    _store_view, install_adhesive_multilight_inspection,
)
from src.ui.adhesive_multilight_automation import AdhesiveMultiLightAutomation


class Mode:
    def __init__(self,text="Modo Sombra"):
        self.value=text
    def currentText(self):
        return self.value


class TextWidget:
    def __init__(self):
        self.value=""
    def setText(self,value):
        self.value=str(value)
    def setStyleSheet(self,_value):
        pass
    def setToolTip(self,_value):
        pass
    def setEnabled(self,_value):
        pass
    def width(self):
        return 500
    def height(self):
        return 200


class FakeOrchestrator:
    def __init__(self):
        self.calls=[]
    def inspect(self,gab,test,raw,info,box,epicenters):
        self.calls.append((raw,box,epicenters,info.copy()))
        return {
            "is_defect":False,"verdict":"FALHA FALSA",
            "confidence":.99,"reason":"CNN local OK",
            "active_engines":["faltando_cnn_v2.py"],
            "all_boxes":{},"detail":{"cnn_v2_active":True},
        }


class ShadowFake:
    def __init__(self,mode="Modo Sombra"):
        self.combo_mode=Mode(mode)
        self.orchestrator=FakeOrchestrator()
        self.capture_cycle_source="network"
        self.capture_start_time=0.0
        self.capture_start_source="network_payload_received"
        self.shown=0
        self._inspection_images_visible=False
        self.metric_labels={}
        self.lbl_sample=TextWidget()
        self.lbl_ng=TextWidget()
        self.lbl_sample_focus=TextWidget()
        self.lbl_ng_focus=TextWidget()
        self.lbl_verdict=TextWidget()
        self.lbl_reason=TextWidget()
        self.lbl_timer=TextWidget()
        self.btn_start=TextWidget()
        self.btn_save_ok=TextWidget()
        self.btn_save_ng=TextWidget()
        self.btn_skip=TextWidget()

    def _safe_maximize(self):
        self.shown+=1
    def update_brain_status(self,*args):
        pass
    def _update_aoi_info(self,*args):
        pass
    def _update_reference_panel(self,*args):
        raise AssertionError("Custo do debugger neural na rota sombra")
    def _update_confidence_panel(self,*args):
        raise AssertionError("Painéis de especialistas na rota sombra")
    def numpy_to_pixmap(self,*args):
        raise AssertionError("QPixmap não pertence ao caminho sombra")


class ShadowFastInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def test_cnn_shadow_skips_unnecessary_geometry_and_debug_render(self):
        import time
        p=ShadowFake()
        p.capture_start_time=time.perf_counter()
        g=np.ones((48,64,3),dtype=np.uint8)
        t=np.ones((48,64,3),dtype=np.uint8)*3
        with patch("src.ui.control_panel.normalize_aoi_text",return_value=("FALTANDO","X")), \
             patch("src.ui.control_panel.detect_anomalies",
                   side_effect=AssertionError("Geometria desnecessária")):
            ControlPanel.process_aoi_images(p,g,t,{"value":"FALTANDO"})
        self.assertEqual(p.shown,0)
        self.assertEqual(len(p.orchestrator.calls),1)
        raw,box,epicenters,info=p.orchestrator.calls[0]
        self.assertEqual((raw,box,epicenters),([] , {}, []))
        self.assertEqual(info["category"],"FALTANDO")
        self.assertEqual(p.lbl_verdict.value,"FALHA FALSA")
        self.assertTrue(p.current_analysis["detail"]["shadow_fast_path"])
        self.assertFalse(p._inspection_images_visible)

    def test_non_shadow_must_not_skip_detector(self):
        import time
        p=ShadowFake("Modo Teste")
        p.capture_start_time=time.perf_counter()
        g=np.ones((48,64,3),dtype=np.uint8)
        with patch("src.ui.control_panel.normalize_aoi_text",return_value=("FALTANDO","X")), \
             patch("src.ui.control_panel.detect_anomalies",
                   side_effect=RuntimeError("detector called")):
            with self.assertRaisesRegex(RuntimeError,"detector called"):
                ControlPanel.process_aoi_images(p,g,g,{"value":"FALTANDO"})
        self.assertEqual(p.shown,1)

    def test_shadow_auxiliary_ocr_reuses_same_part_without_tesseract(self):
        monitor=ScreenMonitor()
        monitor._shadow_fast=True
        info={"board":"BOARD","parts":"U2~5","value":"FALTANDO"}
        monitor._shadow_auxiliary_info=info
        with patch.object(monitor,"_ocr_fast",side_effect=AssertionError("OCR extra")):
            actual=monitor._extract_text_info(
                np.zeros((70,90,3),np.uint8),(0,0,30,8),(35,0,30,8)
            )
        self.assertEqual(actual,info)
        self.assertIsNot(actual,info)
        with patch("src.services.screen_monitor.cv2.imwrite") as write:
            monitor._write_debug_crop(Path("dummy.png"),np.zeros((30,30,3),np.uint8))
            write.assert_not_called()

    def test_shadow_skips_all_debug_visual_payload_but_preserves_storage(self):
        p=ShadowFake()
        p.adhesive_multilight_views={}
        img=np.zeros((30,40,3),np.uint8)
        with patch("src.ui.adhesive_multilight_inspection.build_adhesive_view_payload",
                   side_effect=AssertionError("pixmaps/crops desnecessários")):
            self.assertTrue(_store_view(p,"SIDE",img,img))
        self.assertEqual(p.adhesive_multilight_views["SIDE"]["shadow_fast"],True)

    def test_shadow_always_saves_images_even_if_ai_agrees(self):
        p=ShadowFake()
        p.current_ng=np.ones((32,42,3),np.uint8)
        p.current_sample=np.zeros((32,42,3),np.uint8)
        p.current_aoi_info={"category":"FALTANDO"}
        p.current_analysis={"is_defect":False,"detail":{}}
        p.adhesive_multilight_learning_samples={}
        with patch("src.services.neural_online_learning.eligible_online_case",
                   return_value=False):
            task=_decision_task(p,"OK","xp_keyboard","OK")
        self.assertTrue(task["save_images"])
        self.assertEqual(task["source"],"xp_keyboard")


class ShadowAutomationTests(unittest.TestCase):
    def test_top_capture_requests_mid_while_top_analysis_is_still_pending(self):
        class FakePanel(QWidget):
            def __init__(self):
                super().__init__()
                self.combo_mode=Mode()
                self.last_xp_ip="169.254.95.200"
                self.sent=[]
                self.saved=[]
                self.shadow_pending_operator_label="NG"
                self.shadow_pending_operator_event_id="piece-123"
                self.adhesive_multilight_primary_event_id="piece-123"
            def send_command_to_xp(self,c):
                self.sent.append(c)
                return True
            def change_lighting(self,*args):
                return True
            def update_brain_status(self,*args):
                pass
            def update_network_status(self,*args):
                pass
            def finalize_multilight_decision(self):
                return {"is_defect":False,"verdict":"FALHA FALSA"}
            def save_label(self,label,source="button"):
                self.saved.append((label,source))
        p=FakePanel()
        self.addCleanup(p.deleteLater)
        auto=AdhesiveMultiLightAutomation(p)
        self.assertTrue(auto.start())
        self.assertEqual(p.sent,["LEFT"])
        self.assertTrue(auto.shadow_top_captured())
        self.assertEqual(p.sent,["LEFT","RIGHT"])
        self.assertEqual(auto.expected_mode,"MID")
        self.assertTrue(auto.frame_stored("MID"))
        QApplication.processEvents()
        QApplication.processEvents()
        self.assertIn("DOWN",p.sent)
        self.assertEqual(p.saved,[("NG","xp_keyboard")])
        self.assertFalse(any(c in {"0","1"} for c in p.sent))
        self.assertEqual(p.shadow_pending_operator_label,"")

    def test_early_human_label_immediately_ends_capture_without_xp_echo(self):
        class FakePanel:
            def __init__(self):
                self.combo_mode=Mode()
                self.current_aoi_info={"category":"FALTANDO"}
                self.adhesive_multilight_primary_event_id="id1"
                self.called=[]
                self.cancelled=[]
                class Auto:
                    active=True
                    def __init__(self,outer):
                        self.outer=outer
                    def cancel_for_cycle_end(self):
                        self.outer.cancelled.append(True)
                        self.active=False
                self.adhesive_multilight_automation=Auto(self)
            def save_label(self,*args,**kwargs):
                self.called.append((args,kwargs))
                return "human"
            def skip_image(self):
                pass
            def process_aoi_images(self,*args):
                pass
            def handle_network_image(self,*args):
                pass
            def update_brain_status(self,*args):
                pass
            def update_network_status(self,*args):
                pass
        install_adhesive_multilight_inspection(FakePanel)
        p=FakePanel()
        p.adhesive_multilight_pending_start=True
        self.assertEqual(p.save_label("OK",source="xp_keyboard"),"human")
        self.assertEqual(p.called,[(("OK",),{"source":"xp_keyboard"})])
        self.assertEqual(p.cancelled,[True])
        self.assertFalse(p.adhesive_multilight_automation.active)


if __name__=="__main__":
    unittest.main()
