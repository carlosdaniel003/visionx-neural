"""Dataset shadow: early human decision saves only captured lighting frames."""
from __future__ import annotations
import unittest
from unittest.mock import patch
import numpy as np

from src.services.anomaly_learning import _decision_task
from src.services.decision_persistence import DecisionPersistenceQueue


class FakeMode:
    def currentText(self):
        return "Modo Sombra"


class FakePanel:
    def __init__(self, complete=False):
        self.combo_mode=FakeMode()
        self.current_ng=np.ones((28,30,3),np.uint8)*30
        self.current_sample=np.ones((28,30,3),np.uint8)*10
        self.current_aoi_info={"board":"B1","parts":"U2~5",
                               "category":"FALTANDO", "lighting_mode":"SIDE"}
        self.current_analysis={"is_defect":False,"verdict":"FALHA FALSA",
                               "detail":{"recognition_route":"NEW_CNN"}}
        self.adhesive_multilight_primary_event_id="piece-100"
        self.adhesive_multilight_learning_samples={}
        for i, light in enumerate(("SIDE","TOP","MID")[:3 if complete else 2]):
            self.adhesive_multilight_learning_samples[light]={
                "lighting_mode":light,
                "sample_image":np.ones((28,30,3),np.uint8)*(10+i),
                "test_image":np.ones((28,30,3),np.uint8)*(20+i),
                "source_frame":np.ones((40,50,3),np.uint8)*(30+i),
                "analysis":{"is_defect":False,"verdict":"FALHA FALSA","detail":{}}
            }


class MemoryStore:
    calls=[]
    @classmethod
    def save_sample(cls,**kwargs):
        cls.calls.append(kwargs)
        return "saved.json"


class Orchestrator:
    def __init__(self):
        self.reloads=0
    def reload_memory(self):
        self.reloads+=1


class ShadowDatasetTests(unittest.TestCase):
    def setUp(self):
        MemoryStore.calls=[]

    def test_early_operator_press_preserves_only_actual_side_top_images(self):
        p=FakePanel(complete=False)
        with patch("src.services.neural_online_learning.eligible_online_case",
                   return_value=False):
            task=_decision_task(p,"NG","xp_keyboard","OK")
        self.assertEqual(task["event_id"],"piece-100")
        self.assertEqual(task["multilight_samples"],[])
        self.assertEqual([x["lighting_mode"] for x in task["shadow_partial_samples"]],
                         ["SIDE","TOP"])
        self.assertTrue(task["save_images"])
        o=Orchestrator()
        q=DecisionPersistenceQueue(o,dataset_manager=MemoryStore)
        with patch("src.services.decision_persistence.eligible_online_case",
                   return_value=False):
            q.submit(task)
            q.wait_until_idle()
        self.assertEqual(len(MemoryStore.calls),2)
        self.assertEqual(
            [x["lighting_mode"] for x in MemoryStore.calls],["SIDE","TOP"])
        self.assertTrue(all(x["label"]=="NG" for x in MemoryStore.calls))
        self.assertTrue(all(x["event_id"]=="piece-100" for x in MemoryStore.calls))
        self.assertEqual(o.reloads,1)

    def test_full_set_remains_three_records_one_human_label(self):
        p=FakePanel(complete=True)
        with patch("src.services.neural_online_learning.eligible_online_case",
                   return_value=False):
            task=_decision_task(p,"OK","xp_keyboard","OK")
        self.assertEqual(len(task["multilight_samples"]),3)
        self.assertEqual(task["shadow_partial_samples"],[])
        q=DecisionPersistenceQueue(None,dataset_manager=MemoryStore)
        with patch("src.services.decision_persistence.eligible_online_case",
                   return_value=False):
            q.submit(task)
            q.wait_until_idle()
        self.assertEqual(len(MemoryStore.calls),3)
        self.assertTrue(all(x["save_images"] for x in MemoryStore.calls))

    def test_missing_side_does_not_label_top_from_another_piece(self):
        p=FakePanel()
        p.adhesive_multilight_learning_samples.pop("SIDE")
        with patch("src.services.neural_online_learning.eligible_online_case",
                   return_value=False):
            task=_decision_task(p,"NG","xp_keyboard","OK")
        self.assertEqual(task["shadow_partial_samples"],[])


if __name__=="__main__":
    unittest.main()
