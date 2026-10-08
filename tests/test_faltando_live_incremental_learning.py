"""CNN FALTANDO v2: rótulos humanos -> treino assíncrono -> gate -> hot swap."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.core.neural.faltando_live import FaltandoCNNLive
from src.services.neural_online_learning import (
    OnlineLearningQueue, eligible_online_case, ONLINE_SCHEMA,
)
from src.scripts.train_faltando_cnn_v2_online import (
    _on_disk_event, _all_online, train_online, PairDataset, POINTER_SCHEMA,
)
from src.services.anomaly_learning import _decision_task


class OnlineFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.q = OnlineLearningQueue(self.root, start_worker=False)
        self.info = {"board": "PCB A", "parts": "R475", "value": "0 <= 12 <= 20 FALTANDO", "category": "FALTANDO"}
        self.reference = self.image(23)
        self.test = self.image(25)

    @staticmethod
    def image(base):
        f = np.zeros((65, 78, 3), dtype=np.uint8)
        f[15:49, 14:61] = (base, base+40, 170)
        return f

    def task(self, label="NG", source="button", route="NEW_CNN", multilight=False):
        data = {
            "label": label,
            "source": source,
            "aoi_info": {**self.info, "lighting_mode": "SIDE"},
            "analysis": {
                "detail": {
                    "recognition_route": route,
                    "recognition_light_routes": (
                        {"SIDE": "KNOWN_KNN", "TOP": "NEW_CNN", "MID": "NEW_CNN"}
                        if multilight else {}
                    ),
                },
            },
            "sample_image": self.reference,
            "ng_image": self.test,
        }
        if multilight:
            data["analysis"]["detail"]["recognition_route"] = "MULTILIGHT_MIXED"
            data["multilight_samples"] = [
                {"lighting_mode": m, "sample_image": self.image(20+i),
                 "test_image": self.image(25+i)}
                for i, m in enumerate(("SIDE", "TOP", "MID"))
            ]
            data["event_id"] = "3-light-same-part"
        return data

    def case(self, label="NG", multi=False):
        path_id = self.q.submit_saved(self.task(label, multilight=multi))
        self.assertIsNotNone(path_id)
        return self.q.events/(path_id+".json")


class EventSpoolingTests(OnlineFixtures):
    def test_enqueue_once_after_confirmed_new_human_ng(self):
        t = self.task("NG")
        self.assertTrue(eligible_online_case(t))
        path = self.case("NG")
        self.assertTrue(path.exists())
        case = _on_disk_event(path, self.root)
        self.assertEqual(case["label"], "NG")
        self.assertEqual(tuple(case["views"]), ("SIDE",))
        self.assertEqual(len(list(self.q.events.glob("*.png"))), 2)
        self.assertEqual(case["aoi_info"]["parts"], "R475")

    def test_each_mode_and_each_label_is_preserved_as_one_event(self):
        path = self.case("OK", multi=True)
        case = _on_disk_event(path, self.root)
        self.assertEqual(case["label"], "OK")
        self.assertEqual(set(case["views"]), {"SIDE", "TOP", "MID"})
        self.assertEqual(len(list(self.q.events.glob("*.json"))), 1)
        self.assertEqual(len(list(self.q.events.glob("*.png"))), 6)
        dataset = PairDataset([case], size=64, fraction=.7)
        row = dataset[0]
        self.assertEqual(row[4].tolist(), [1., 1., 1.])
        self.assertEqual(tuple(row[0].shape), (3, 3, 64, 64))

    def test_no_automatic_label_or_known_knn_memory_trains_cnn(self):
        for source in ("production_auto", "auto", "unknown"):
            self.assertIsNone(self.q.submit_saved(self.task("OK", source)))
        self.assertIsNone(self.q.submit_saved(self.task("NG", route="KNOWN_KNN")))
        self.assertIsNone(self.q.submit_saved({
            **self.task("NG"), "aoi_info": {**self.info, "category": "INVERTIDO"}
        }))
        self.assertFalse(list(self.q.events.glob("*.json")))

    def test_corrupted_image_or_fake_source_rejected(self):
        path = self.case()
        row = json.loads(path.read_text(encoding="utf-8"))
        row["human_source"] = "production_auto"
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            _on_disk_event(path, self.root)
        row["human_source"] = "button"
        path.write_text(json.dumps(row), encoding="utf-8")
        asset = self.q.events / row["images"][0]["reference"]
        asset.write_bytes(asset.read_bytes()+b"corrupted")
        with self.assertRaisesRegex(ValueError, "Fingerprint"):
            _on_disk_event(path, self.root)

    def test_conflicting_human_labels_block_training(self):
        one = self.case("NG")
        two = self.case("OK")
        self.assertTrue(one.exists() and two.exists())
        with self.assertRaisesRegex(ValueError, "conflitantes"):
            _all_online(self.root)

    def test_nonblocking_decision_task_saves_agreeing_new_ok_images(self):
        class Panel:
            current_ng = self.test
            current_sample = self.reference
            current_aoi_info = {**self.info, "lighting_mode": "SIDE"}
            current_analysis = {
                "is_defect": False, "detail": {"recognition_route": "NEW_CNN"}
            }
        panel = Panel()
        result = _decision_task(panel, "OK", "button", "OK")
        self.assertTrue(result["save_images"])
        panel.current_analysis = {
            "is_defect": False, "detail": {"recognition_route": "KNOWN_KNN"}
        }
        self.assertFalse(_decision_task(panel, "OK", "button", "OK")["save_images"])


class ChampionGateTests(OnlineFixtures):
    def setUp(self):
        super().setUp()
        self.event = self.case()
        original = FaltandoCNNV2()
        self.prior = {
            "schema": MODEL_SCHEMA_V2,
            "experimental": True, "production_approved": False,
            "image_size": 64, "focus_fraction": .7,
            "lights": ("SIDE", "TOP", "MID"),
            "state_dict": original.state_dict(),
            "source_manifest_sha256": "some-historical-source",
        }
        self.paired = [
            {"id": "historical_good", "label": "NG",
             "views": {"SIDE": {
                 "reference": self.q.events /
                    json.loads(self.event.read_text())["images"][0]["reference"],
                 "test": self.q.events /
                    json.loads(self.event.read_text())["images"][0]["test"],
             }}},
        ]

    @staticmethod
    def successful_assessment(model, events, size, focus):
        return {
            "events": len(events),
            "images": sum(len(e["views"]) for e in events),
            "correct_events": len(events),
            "correct_images": sum(len(e["views"]) for e in events),
            "all_correct": True, "ng_false_ok": 0, "errors": [],
        }

    def test_one_epoch_candidate_promotion_never_edits_original_weights(self):
        def base(root):
            return self.root/"original.pt", self.prior
        baseline_hash = sha256(
            json.dumps(sorted(self.prior["state_dict"])).encode()).hexdigest()
        with patch("src.scripts.train_faltando_cnn_v2_online._base_model", base), \
             patch("src.scripts.train_faltando_cnn_v2_online._historical",
                   lambda root, obj: self.paired), \
             patch("src.scripts.train_faltando_cnn_v2_online._assess",
                   self.successful_assessment):
            report = train_online(self.root, self.event, epochs=1)
        self.assertTrue(report["promoted"])
        pointer = self.root/"reports"/"neural_online"/"live_active.json"
        self.assertTrue(pointer.exists())
        data = json.loads(pointer.read_text())
        self.assertEqual(data["schema"], POINTER_SCHEMA)
        target = self.root/data["checkpoint_relative_path"]
        self.assertEqual(sha256(target.read_bytes()).hexdigest(),
                         data["checkpoint_sha256"])
        ckpt = torch.load(target, map_location="cpu", weights_only=True)
        self.assertFalse(ckpt["production_approved"])
        self.assertEqual(
            sha256(json.dumps(sorted(self.prior["state_dict"])).encode()).hexdigest(),
            baseline_hash,
        )

    def test_regression_failure_rejects_candidate_and_preserves_champion(self):
        passed = self.successful_assessment(FaltandoCNNV2(), self.paired, 64, .7)
        failed = {**passed, "all_correct": False, "correct_images": 0,
                  "ng_false_ok": 1, "errors": [{"reason": "NG->OK"}]}
        vals = iter((passed, passed, failed, passed))
        with patch("src.scripts.train_faltando_cnn_v2_online._base_model",
                   lambda root: (self.root/"original.pt", self.prior)), \
             patch("src.scripts.train_faltando_cnn_v2_online._historical",
                   lambda root, obj: self.paired), \
             patch("src.scripts.train_faltando_cnn_v2_online._assess",
                   side_effect=lambda *args: next(vals)):
            report = train_online(self.root, self.event, epochs=1)
        self.assertFalse(report["promoted"])
        self.assertFalse(
            (self.root/"reports"/"neural_online"/"live_active.json").exists()
        )

    def test_pointer_hot_reload_and_corruption_fail_closed(self):
        pinned = self.root/"original.pt"
        torch.save(self.prior, pinned)
        base_hash = sha256(pinned.read_bytes()).hexdigest()
        engine = FaltandoCNNLive(
            pinned, base_hash, online_root=self.root
        )
        first = engine.inspect(self.reference, self.test, "SIDE")
        self.assertTrue(first["detail"]["cnn_v2_checkpoint_verified"])

        checkpoints = self.root/"reports"/"neural_online"/"checkpoints"
        checkpoints.mkdir(parents=True)
        candidate = checkpoints/"latest.pt"
        torch.save(self.prior, candidate)
        digest = sha256(candidate.read_bytes()).hexdigest()
        pointer = self.root/"reports"/"neural_online"/"live_active.json"
        pointer.write_text(json.dumps({
            "schema": POINTER_SCHEMA,
            "category": "FALTANDO",
            "checkpoint_relative_path": candidate.relative_to(self.root).as_posix(),
            "checkpoint_sha256": digest,
            "experimental": True, "production_approved": False,
        }), encoding="utf-8")
        upgraded = engine.inspect(self.reference, self.test, "SIDE")
        self.assertEqual(upgraded["detail"]["cnn_v2_checkpoint_sha256"], digest)
        candidate.write_bytes(candidate.read_bytes()+b"bad")
        # Em memória o peso já estava carregado; simular reinício da rede.
        restarted = FaltandoCNNLive(pinned, base_hash, online_root=self.root)
        refused = restarted.inspect(self.reference, self.test, "SIDE")
        self.assertEqual(refused["verdict"], "REVISÃO OBRIGATÓRIA")


if __name__ == "__main__":
    unittest.main()
