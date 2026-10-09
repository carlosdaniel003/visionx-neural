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

    def test_six_cards_never_stack_vertically_for_notebook_and_monitor(self):
        p=NeuralEvidencePanel()
        self.addCleanup(p.deleteLater)
        for width in (375,680,1366,1920):
            p.resize(width,p.height())
            p.show()
            QApplication.processEvents()
            p._reflow(width)
            self.assertEqual(len(p.tiles),6)
            self.assertEqual(p.sections["major"]._columns,3)
            self.assertEqual(p.sections["minor"]._columns,3)
            self.assertEqual(p.height(),p.minimumHeight())
            self.assertEqual(
                p.scroll.verticalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            self.assertEqual(len(set(t.geometry().y() for t in p.tiles)),1)

    def test_internal_horizontal_scroll_and_arrow_navigation(self):
        p=NeuralEvidencePanel()
        self.addCleanup(p.deleteLater)
        p.resize(580,p.height())
        p.show()
        QApplication.processEvents()
        bar=p.scroll.horizontalScrollBar()
        self.assertGreater(bar.maximum(),0)
        self.assertFalse(p.left_button.isEnabled())
        self.assertTrue(p.right_button.isEnabled())
        p.right_button.click()
        QApplication.processEvents()
        self.assertGreater(bar.value(),0)
        self.assertTrue(p.left_button.isEnabled())
        p.left_button.click()
        QApplication.processEvents()
        self.assertEqual(bar.value(),0)

    def test_six_cards_are_exclusively_neural_and_async_after_analysis(self):
        p=NeuralEvidencePanel()
        self.addCleanup(p.deleteLater)
        ref,test,ctx=sample_payload()
        payload=build_adhesive_view_payload(ref,test,ctx)
        with patch("src.ui.widgets.neural_evidence.QThreadPool.globalInstance") as pool:
            p.set_visual_payload(payload)
            self.assertFalse(pool.return_value.start.called)
            self.assertTrue(all(tile.image._source.isNull() for tile in p.tiles))
            p.update_data(knn_analysis()["detail"],knn_analysis())
            self.assertTrue(pool.return_value.start.called)
        self.assertIn("KNN",p.heading.text())
        self.assertEqual([tile.mode for tile in p.tiles[:3]],
                         ["DIF. LATENTE", "GRAD-CAM", "ATIVAÇÃO CNN"])
        self.assertTrue(all(tile.image._source.isNull() for tile in p.tiles))
        maps={}
        for key, reference in (("major", payload["large"]), ("minor", payload["small"])):
            h,w=reference.shape[:2]
            maps[key]={
                "neural":True, "dimensions":(w,h),
                "layer":"encoder.4","target_class":"OK",
                "images":tuple(np.full((h,w,3), 60+i*60, np.uint8) for i in range(3)),
                "raw_feature_means":[.1,.2,.3],
            }
        p._on_probe_finished(p._epoch, maps, "")
        self.assertTrue(all(not tile.image._source.isNull() for tile in p.tiles))
        self.assertTrue(all("CNN" in tile.toolTip() or "Grad-CAM" in tile.toolTip()
                            or "encoder" in tile.toolTip() or "features" in tile.toolTip()
                            for tile in p.tiles))
        self.assertNotIn("CINZA / DIFERENÇAS / BLOCOS",p.footer.text())

    def test_stale_worker_cannot_render_previous_inspection(self):
        p=NeuralEvidencePanel()
        self.addCleanup(p.deleteLater)
        p.set_visual_payload({"large":np.ones((30,30,3),np.uint8)})
        earlier=p._epoch
        p.clear_data()
        p._on_probe_finished(earlier,{"major":{"neural":True}}, "")
        self.assertTrue(all(tile.image._source.isNull() for tile in p.tiles))

    def test_adhesive_does_not_start_auxiliary_cnn(self):
        p=NeuralEvidencePanel()
        self.addCleanup(p.deleteLater)
        p.set_visual_payload({"_cnn_explain_allowed":False})
        with patch("src.ui.widgets.neural_evidence.QThreadPool.globalInstance") as pool:
            p.update_data({"recognition_route":"NEW_EXPERTS"},knn_analysis())
            self.assertFalse(pool.return_value.start.called)
        self.assertIn("fora do escopo",p.footer.text())

    def test_multilight_reuses_payload_even_when_knn_skips_cnn(self):
        view=AdhesiveMultiLightAnalysisView()
        self.addCleanup(view.deleteLater)
        ref,test,ctx=sample_payload()
        payload=build_adhesive_view_payload(ref,test,ctx)
        self.assertTrue(view.set_visual_payload("SIDE",payload))
        lane=view.lanes["SIDE"]
        with patch("src.ui.widgets.neural_evidence.QThreadPool.globalInstance"):
            self.assertTrue(view.set_analysis("SIDE",knn_analysis()))
        self.assertFalse(lane.neural_evidence.isHidden())
        self.assertTrue(lane.neural_evidence.sections["major"].tiles[0].image._source.isNull())
        self.assertIn("PROCESSANDO CNN",lane.neural_evidence.tiles[0].metric.text())
        self.assertEqual(
            lane.scroll.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertEqual(len(view.horizontal_scroll_bars()),1)
        self.assertTrue(lane.scroll.isHidden())
        self.assertFalse(lane.status_label.isVisible())
        self.assertTrue(lane.frames["knn_expert.py"].isHidden())
        self.assertEqual(lane.neural_evidence.scroll.verticalScrollBarPolicy(),
                         Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.assertEqual(lane.neural_evidence.tiles[0].heading.text(),
                         "EPICENTRO MAIOR • DIF. LATENTE")
        self.assertIn("border:1px solid #d3a900",
                      lane.neural_evidence.tiles[0].styleSheet())
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
