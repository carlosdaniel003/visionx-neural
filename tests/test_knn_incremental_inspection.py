"""Regressões: inspeções sucessivas não podem reconstruir toda a memória KNN."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2

from src.config.settings import settings
from src.core.anomaly_signature import build_anomaly_signature
from src.core.experts.knn_expert import KNNExpert
from src.services.decision_persistence import DecisionPersistenceQueue
from tests.test_verified_memory_router import MemoryFirstFixture


class IncrementalKNNTests(MemoryFirstFixture):
    def _saved_record(self, label="NG"):
        record = self.make_record(label)
        path = Path(record["json_path"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["analysis"] = {
            "anomaly_memory": build_anomaly_signature(
                self.reference, self.test, {}, self.info,
            )
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        return record

    def _knn(self):
        with patch.object(KNNExpert, "_load_all"):
            return KNNExpert()

    def _scope(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(settings, "NORMAL_DIR", self.root / "OK"))
        stack.enter_context(patch.object(settings, "ANOMALY_DIR", self.root / "NG"))
        return stack

    def test_five_human_updates_preserve_warm_index_without_full_reload(self):
        with self._scope():
            knn = self._knn()
            system = self.create_router(knn=knn)
            first = self._saved_record("NG")
            system.ingest_saved_memory([first["json_path"]])
            system.prepare_verified_memory_index()
            self.assertEqual(self.inspect(system)["detail"]["recognition_route"],
                             "KNOWN_KNN")

            with patch.object(knn, "_load_all", side_effect=AssertionError(
                "Full rescan is forbidden on a normal decision"
            )):
                for _ in range(5):
                    record = self._saved_record("NG")
                    system.ingest_saved_memory([record["json_path"]])
                    result = self.inspect(system)
                    profile = result["detail"]["verified_memory_timing"]
                    self.assertEqual(result["detail"]["recognition_route"], "KNOWN_KNN")
                    self.assertTrue(profile["index_reused"])
                    self.assertEqual(profile["index_build_ms"], 0.0)
            self.assertEqual(system.reloads, 0)
            self.assertEqual(len(knn.signatures_ng), 6)

    def test_category_only_scoped_index_sees_incremental_legacy_record(self):
        """Sem Board/Parts no OCR, novo registro aparece sem rebuild."""
        with self._scope():
            knn = self._knn()
            system = self.create_router(knn=knn)
            info = dict(self.info, board="", parts="", value="")
            initial = self.inspect(system, info=info)
            self.assertEqual(initial["detail"]["recognition_route"], "NEW_CNN")
            self.assertEqual(
                initial["detail"]["verified_memory_timing"]["index_scope"],
                "targeted_legacy_lookup",
            )
            record = self._saved_record("NG")
            system.ingest_saved_memory([record["json_path"]])
            result = self.inspect(system, info=info)
            self.assertEqual(result["detail"]["recognition_route"], "KNOWN_KNN")
            self.assertTrue(result["detail"]["verified_memory_timing"]["index_reused"])
            self.assertEqual(
                result["detail"]["verified_memory_timing"]["index_build_ms"], 0.0
            )

    def test_incremental_opposite_label_is_conflict_not_auto_ok(self):
        with self._scope():
            knn = self._knn()
            system = self.create_router(knn=knn)
            existing = self._saved_record("NG")
            system.ingest_saved_memory([existing["json_path"]])
            system.prepare_verified_memory_index()
            opposite = self._saved_record("OK")
            system.ingest_saved_memory([opposite["json_path"]])
            result = self.inspect(system)
            self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
            self.assertEqual(result["detail"]["recognition_route"], "MEMORY_CONFLICT")
            self.assertTrue(result["detail"]["verified_memory_timing"]["index_reused"])
            self.assertEqual(system.reloads, 0)

    def test_negative_audit_is_cached_and_rechecked_when_pngs_appear(self):
        system = self.create_router()
        record = self.make_record("NG", save_images=False)
        system.experts["knn"].signatures_ng.append(record)
        index = system._get_verified_memory_index()
        system.prepare_verified_memory_index()

        with patch.object(index, "_entry", wraps=index._entry) as audit:
            for _ in range(3):
                system.reload_memory()
                result = self.inspect(system)
                self.assertEqual(result["detail"]["recognition_route"], "NEW_CNN")
                self.assertEqual(
                    result["detail"]["verified_memory_timing"].get(
                        "legacy_rejected_cache_hits"
                    ), 1,
                )
            audit.assert_not_called()

        path = Path(record["json_path"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["storage"]["reference_image_file"] = "r.png"
        data["storage"]["test_image_file"] = "t.png"
        self.assertTrue(cv2.imwrite(str(path.parent / "r.png"), self.reference))
        self.assertTrue(cv2.imwrite(str(path.parent / "t.png"), self.test))
        path.write_text(json.dumps(data), encoding="utf-8")
        system.reload_memory()
        result = self.inspect(system)
        self.assertEqual(result["detail"]["recognition_route"], "KNOWN_KNN")

    def test_file_change_after_warmup_invalidates_known_without_reload(self):
        system = self.create_router()
        record = self.make_record("NG")
        system.experts["knn"].signatures_ng.append(record)
        system.prepare_verified_memory_index()
        self.assertEqual(self.inspect(system)["detail"]["recognition_route"], "KNOWN_KNN")
        path = Path(record["json_path"]).parent / "t.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image[60, 80] ^= 1
        self.assertTrue(cv2.imwrite(str(path), image))
        result = self.inspect(system)
        self.assertEqual(result["detail"]["recognition_route"], "NEW_CNN")

    def test_persistence_queue_uses_incremental_paths_instead_of_reload(self):
        class Orchestrator:
            def __init__(self):
                self.received = []
            def ingest_saved_memory(self, paths):
                self.received.append(tuple(paths))
            def reload_memory(self):
                raise AssertionError("The full reload must not run")

        class Writer:
            @staticmethod
            def save_sample(**_kwargs):
                return "saved.json"

        orchestrator = Orchestrator()
        queue = DecisionPersistenceQueue(orchestrator, dataset_manager=Writer)
        task = {
            "label": "NG", "source": "button",
            "ng_image": self.test, "sample_image": self.reference,
            "aoi_info": self.info, "analysis": {"is_defect": False},
            "save_images": True,
        }
        with patch("src.services.decision_persistence.eligible_online_case",
                   return_value=False):
            queue.submit(task)
            queue.wait_until_idle()
        self.assertEqual(orchestrator.received, [("saved.json",)])
