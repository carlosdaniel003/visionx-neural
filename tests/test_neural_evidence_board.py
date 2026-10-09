"""Visual evidence: two actual AOI epicenters, three diagnostic modes each.

No Grad-CAM is claimed; transforms do not call model or XP commands.
"""
from __future__ import annotations
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from src.ui.neural_evidence_model import evidence_views, epicenter_evidence
from src.ui.widgets.neural_evidence import NeuralEvidencePanel
from src.ui.adhesive_multilight_analysis import AdhesiveMultiLightAnalysisView
from src.ui.adhesive_multilight_inspection import build_adhesive_view_payload


def sample_payload():
    base=np.zeros((180,220,3),np.uint8)
    base[:]=60
    cv2.rectangle(base,(24,26),(145,148),(190,195,200),-1)
    test=base.copy()
    cv2.rectangle(test,(74,58),(101,87),(5,30,200),-1)
    large={"x":10,"y":15,"w":156,"h":148,"detected":True}
    small=(62,51,54,49)
    context={
        "valid":True,
        "global_box_info":large,
        "real_epicenters":[small],
        "focus_ng":test[51:100,62:116].copy(),
        "focus_gab":base[51:100,62:116].copy(),
    }
    return base,test,context


def knn_analysis():
    return {
        "lighting_mode":"SIDE","verdict":"FALHA FALSA",
        "is_defect":False,"active_engines":["knn_expert.py"],
        "detail":{"recognition_route":"KNOWN_KNN",
                  "recognition_known_label":"OK"},
    }


class EvidenceModelTests(unittest.TestCase):
    def test_both_epicenters_are_real_existing_crops(self):
        ref,test,ctx=sample_payload()
        payload=build_adhesive_view_payload(ref,test,context=ctx)
        self.assertTrue(np.array_equal(payload["large_reference"],ref[15:163,10:166]))
        self.assertTrue(np.array_equal(payload["small_reference"],ref[51:100,62:116]))
        self.assertTrue(np.array_equal(payload["small"],test[51:100,62:116]))
        evidence=epicenter_evidence(payload)
        self.assertEqual(set(evidence),{"major","minor"})
        for key in ("major","minor"):
            item=evidence[key]
            self.assertEqual(len(item["images"]),3)
            self.assertFalse(item["attention"])
            self.assertEqual(item["transform"],"PIXEL_DIAGNOSTIC_ONLY")
            self.assertTrue(all(frame.shape[-1]==3 for frame in item["images"]))
            self.assertGreater(item["difference_mean"],0)

    def test_no_false_heatmap_when_pixels_are_identical(self):
        ref,_,_=sample_payload()
        images=evidence_views(ref,ref)
        self.assertAlmostEqual(images["difference_mean"],0)
        self.assertTrue(np.array_equal(images["images"][1],ref))
        self.assertTrue(np.all(images["images"][0][:,:,0] ==
                               images["images"][0][:,:,2]))

    def test_anomaly_is_locally_visible_not_global_hot_overlay(self):
        ref,test,_=sample_payload()
        data=evidence_views(ref,test)
        heat=data["images"][1]
        self.assertTrue(np.array_equal(heat[10,10],test[10,10]))
        self.assertFalse(np.array_equal(heat[65,80],test[65,80]))
        blocks=data["images"][2]
        self.assertEqual(blocks.shape,heat.shape)

    def test_image_cap_and_missing_roi_not_invented(self):
        ref=np.ones((1100,1200,3),np.uint8)
        data=evidence_views(ref,ref)
        self.assertLessEqual(max(data["images"][0].shape[:2]),720)
        self.assertIsNone(evidence_views(None,ref))
        self.assertIsNone(epicenter_evidence({})["major"])
        self.assertIsNone(epicenter_evidence({})["minor"])


class NeuralEvidenceQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def test_adaptive_grid_breakpoints(self):
        p=NeuralEvidencePanel()
        self.addCleanup(p.deleteLater)
        self.assertEqual(p.columns_for_width(500),1)
        self.assertEqual(p.columns_for_width(680),2)
        self.assertEqual(p.columns_for_width(950),3)
        for width,columns in [(400,1),(690,2),(980,3)]:
            p.resize(width,1000)
            p._reflow(width)
            self.assertEqual(p.sections["major"]._columns,columns)
            self.assertEqual(p.sections["minor"]._columns,columns)
            self.assertGreaterEqual(p.minimumHeight(),450)

    def test_all_six_views_populate_and_reflow_without_clipping_source(self):
        p=NeuralEvidencePanel()
        self.addCleanup(p.deleteLater)
        ref,test,ctx=sample_payload()
        payload=build_adhesive_view_payload(ref,test,ctx)
        p.set_visual_payload(payload)
        p.update_data(knn_analysis()["detail"],knn_analysis())
        self.assertIn("KNN",p.heading.text())
        for section in p.sections.values():
            self.assertIn("diferença",section.metrics.text())
            self.assertEqual(len(section.tiles),3)
            self.assertTrue(all(not tile.image._source.isNull() for tile in section.tiles))
            self.assertIn("NÃO é atenção",section.tiles[1].metric.text())

    def test_multilight_reuses_payload_even_when_knn_skips_cnn(self):
        view=AdhesiveMultiLightAnalysisView()
        self.addCleanup(view.deleteLater)
        ref,test,ctx=sample_payload()
        payload=build_adhesive_view_payload(ref,test,ctx)
        self.assertTrue(view.set_visual_payload("SIDE",payload))
        lane=view.lanes["SIDE"]
        self.assertTrue(view.set_analysis("SIDE",knn_analysis()))
        self.assertFalse(lane.neural_evidence.isHidden())
        self.assertFalse(lane.neural_evidence.sections["major"].tiles[0].image._source.isNull())
        self.assertEqual(
            lane.scroll.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )
        self.assertEqual(len(view.horizontal_scroll_bars()),1)
        view.clear_all()
        self.assertTrue(lane.neural_evidence.isHidden())
        self.assertIsNone(lane.visual_payload)

    def test_unknown_light_cannot_steal_another_payload(self):
        view=AdhesiveMultiLightAnalysisView()
        self.addCleanup(view.deleteLater)
        self.assertFalse(view.set_visual_payload("UNKNOWN",{}))
        for lane in view.lanes.values():
            self.assertIsNone(lane.visual_payload)


if __name__=="__main__":
    unittest.main()
