"""DESLOCADO online: coleta OK/NG humana, bootstrapping sem trocar motor físico."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch

from tests.test_deslocado_cnn import DeslocadoFixtures
from src.services.neural_online_learning import (
    OnlineLearningQueue, eligible_online_case,
)
from src.scripts.train_deslocado_cnn_online import (
    load_online_deslocado, run_online,
)
from src.scripts.train_deslocado_cnn import train_deslocado
from src.core.verified_memory_router import install_memory_first_router


class DeslocadoOnlineTests(DeslocadoFixtures):
    def setUp(self):
        super().setUp()
        self.prepared_report, self.prepare_dir = self.prepared()
        self.queue = OnlineLearningQueue(self.root, start_worker=False)

    def task(self, label="OK", source="button", route="NEW_EXPERTS"):
        ref = self.frames[12].copy()
        test = ref.copy()
        if label == "NG":
            test[30:59, 35:65] = (85, 30, 250)
        return {
            "aoi_info": {
                "category": "DESLOCADO", "board": "PCB-Z",
                "parts": "R999", "value": "0 <= 3 <= 10 DESLOCADO",
                "lighting_mode": "SIDE",
            },
            "analysis": {"detail": {"recognition_route": route}},
            "label": label, "source": source,
            "sample_image": ref, "ng_image": test,
        }

    def test_registers_deslocado_new_experts_but_not_auto_or_known(self):
        self.assertTrue(eligible_online_case(self.task("OK")))
        self.assertTrue(eligible_online_case(self.task("NG")))
        self.assertFalse(eligible_online_case(self.task("OK", source="production_auto")))
        self.assertFalse(eligible_online_case(self.task("NG", route="KNOWN_KNN")))
        self.assertTrue(eligible_online_case({
            **self.task("OK"),
            "analysis": {"detail": {
                "recognition_route": "MULTILIGHT_MIXED",
                "recognition_light_routes": {
                    "SIDE": "KNOWN_KNN",
                    "TOP": "NEW_EXPERTS", "MID": "NEW_EXPERTS"
                },
            }},
        }))

    def test_production_router_keeps_existing_deslocado_engines(self):
        class DemoOrchestrator:
            def __init__(self):
                self.experts = {"knn": None}
                self.called = []
            def reload_memory(self):
                return None
            def inspect(self, a,b,c,info,d,e):
                self.called.append(dict(info))
                return {"verdict": "FALHA FALSA", "is_defect": False,
                        "detail": {}, "active_engines": ["shift_expert.py"]}
        install_memory_first_router(DemoOrchestrator)
        r = DemoOrchestrator()
        result = r.inspect(
            self.frames[12], self.frames[12], [],
            {"category": "DESLOCADO", "board": "A", "parts": "R1",
             "lighting_mode": "SIDE"}, {}, []
        )
        self.assertEqual(result["detail"]["recognition_route"], "NEW_EXPERTS")
        self.assertEqual(result["detail"]["specialist_candidate"],
                         "DESLOCADO_CNN_V1_BOOTSTRAP_NOT_ACTIVE")
        self.assertEqual(result["active_engines"], ["shift_expert.py"])

    def test_ok_only_online_model_is_candidate_no_live_pointer(self):
        id = self.queue.submit_saved(self.task("OK"))
        self.assertTrue(id)
        events, views = load_online_deslocado(self.root)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(views), 1)
        def fast(*args, **kwargs):
            kwargs.update(epochs=1, size=64, batch_size=2)
            return train_deslocado(*args, **kwargs)
        with patch(
            "src.scripts.train_deslocado_cnn_online.train_deslocado",
            side_effect=fast
        ):
            result = run_online(self.root, self.queue.events/(id+".json"))
        self.assertEqual(result["new_online_events_used"], 1)
        self.assertEqual(result["ng_real_events"], 0)
        self.assertFalse(result["production_approved"])
        self.assertTrue(result["activation_disabled"])
        self.assertFalse(
            (self.root/"reports"/"neural_online"/"live_active.json").exists()
        )

    def test_confirmed_ng_is_learned_as_real_ng_not_synthetic(self):
        identifier = self.queue.submit_saved(self.task("NG"))
        self.assertTrue(identifier)
        events, views = load_online_deslocado(self.root)
        self.assertEqual(events[0]["label"], "NG")
        report, folder = train_deslocado(
            self.prepare_dir/"manifest.json", epochs=1, size=64,
            batch_size=2, online_events=events, online_views=views
        )
        self.assertEqual(report["ng_real_events"], 1)
        self.assertEqual(report["train_results"]["real_ng_count"], 1)
        self.assertEqual(report["new_online_events_used"], 1)
        self.assertFalse(report["production_approved"])

    def test_journal_cannot_accept_corrupted_ng(self):
        identifier = self.queue.submit_saved(self.task("NG"))
        journal = self.queue.events/(identifier+".json")
        record = json.loads(journal.read_text(encoding="utf-8"))
        test = self.queue.events/record["images"][0]["test"]
        test.write_bytes(test.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError, "Hash online divergente"):
            load_online_deslocado(self.root)


if __name__ == "__main__":
    import unittest
    unittest.main()
