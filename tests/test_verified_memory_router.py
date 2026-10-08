"""Memória KNN primeiro SOMENTE para pares AOI humanos idênticos.

Testa evitar atalhos falsos, isolamento de categoria/iluminação,
conflitos OK/NG, recarga, especialistas e gate produtivo.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.core.verified_memory_router import (
    VerifiedKNNMemory, install_memory_first_router,
)
from src.core.multilight_fusion import fuse_multilight
from src.services.image_archive_dedup import image_fingerprint
from src.ui.production_confidence_gate import production_decision_policy


class FakeKNN:
    def __init__(self):
        self.signatures_ok = []
        self.signatures_ng = []


class FakeOrchestrator:
    def __init__(self):
        self.experts = {"knn": FakeKNN()}
        self.specialist_calls = []
        self.reloads = 0

    def reload_memory(self):
        self.reloads += 1

    def inspect(self, ref, tst, raw, info, global_box, epicenters):
        self.specialist_calls.append(dict(info))
        return {
            "verdict": "FALHA FALSA",
            "is_defect": False,
            "confidence": .97,
            "active_engines": ["faltando_cnn_v2.py"] if
            str(info.get("category", "")).upper() == "FALTANDO" else ["shift_expert.py"],
            "detail": {"cnn_v2_active": True, "cnn_v2_experimental": True}
            if str(info.get("category", "")).upper() == "FALTANDO"
            else {"decision_trace": {"fusion_rule": "physical_only"}},
        }


class MemoryFirstFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reference, self.test = self.make_pair()
        self.info = {
            "category": "FALTANDO", "board": "PCB-123",
            "parts": "R475", "lighting_mode": "SIDE",
        }

    def make_pair(self):
        a = np.zeros((85, 119, 3), dtype=np.uint8)
        a[22:61, 29:85] = (15, 90, 200)
        b = a.copy()
        b[45:52, 32:48] = (220, 130, 20)
        return a, b

    def make_record(self, label: str = "NG", *, info: dict | None = None,
                    ref=None, tst=None, folder=None, human_label=None,
                    save_images=True, schema="visionx.memory.v3",
                    source="button") -> dict:
        details = dict(self.info if info is None else info)
        reference = self.reference if ref is None else ref
        test = self.test if tst is None else tst
        file_parent = self.root / (folder or label) / str(
            len(list(self.root.rglob("record_*.json")))
        )
        file_parent.mkdir(parents=True, exist_ok=True)
        fname = "record_" + str(len(list(self.root.rglob("record_*.json")))) + ".json"
        jsonpath = file_parent/fname
        if save_images:
            cv2.imwrite(str(file_parent/"r.png"), reference)
            cv2.imwrite(str(file_parent/"t.png"), test)
        item = {
            "schema": schema, "label": label,
            "decision": {
                "operator_label": label if human_label is None else human_label,
                "source": source,
            },
            "aoi_info": details,
            "storage": {
                "reference_image_file": "r.png" if save_images else "",
                "test_image_file": "t.png" if save_images else "",
                "test_image_fingerprint": image_fingerprint(test),
            },
        }
        jsonpath.write_text(json.dumps(item), encoding="utf-8")
        return {
            "mode": "anomaly",
            "label": label,
            "folder_label": folder or label,
            "category": details["category"],
            "lighting_mode": details.get("lighting_mode", "SIDE"),
            "part": "".join(ch for ch in details["parts"].upper() if ch.isalnum()),
            "json_path": str(jsonpath),
        }

    def create_router(self, *, knn=None):
        class System(FakeOrchestrator):
            pass
        install_memory_first_router(System)
        obj = System()
        if knn is not None:
            obj.experts["knn"] = knn
        return obj

    def inspect(self, obj, *, ref=None, tst=None, info=None):
        return obj.inspect(
            self.reference if ref is None else ref,
            self.test if tst is None else tst,
            [],
            dict(self.info if info is None else info),
            {}, [],
        )


class VerifiedMemoryRoutingTests(MemoryFirstFixture):
    def test_known_ng_skips_all_specialists_and_restores_operator_label(self):
        system = self.create_router()
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        result = self.inspect(system)
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertEqual(result["active_engines"], ["knn_expert.py"])
        self.assertEqual(result["detail"]["recognition_route"], "KNOWN_KNN")
        self.assertTrue(result["detail"]["recognition_memory_verified"])
        self.assertEqual(result["detail"]["recognition_known_label"], "NG")
        self.assertEqual(result["detail"]["best_similarity"], 1.0)
        self.assertEqual(system.specialist_calls, [])

    def test_known_ok_skips_all_specialists(self):
        system = self.create_router()
        system.experts["knn"].signatures_ok.append(self.make_record("OK"))
        result = self.inspect(system)
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertEqual(result["detail"]["recognition_route"], "KNOWN_KNN")
        self.assertEqual(system.specialist_calls, [])
        self.assertEqual(production_decision_policy(result)["proposed_decision"], "OK")

    def test_new_faltando_uses_cnn_only_without_knn(self):
        system = self.create_router()
        system.experts["knn"].signatures_ok.append(self.make_record("OK"))
        altered = self.test.copy()
        altered[60, 60] ^= 1
        result = self.inspect(system, tst=altered)
        self.assertEqual(result["detail"]["recognition_route"], "NEW_CNN")
        self.assertEqual(result["active_engines"], ["faltando_cnn_v2.py"])
        self.assertEqual(len(system.specialist_calls), 1)
        self.assertNotIn("_replay_without_memory", system.specialist_calls[0])
        self.assertFalse(production_decision_policy(result)["auto_allowed"])

    def test_new_other_category_skips_knn_and_calls_physical_specialist(self):
        system = self.create_router()
        i = dict(self.info, category="DESLOCADO")
        result = self.inspect(system, info=i)
        self.assertEqual(result["detail"]["recognition_route"], "NEW_EXPERTS")
        self.assertEqual(result["active_engines"], ["shift_expert.py"])
        self.assertTrue(system.specialist_calls[-1]["_replay_without_memory"])

    def test_different_board_part_light_category_or_reference_is_new(self):
        system = self.create_router()
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        variations = [
            dict(self.info, board="PCB-999"),
            dict(self.info, parts="R476"),
            dict(self.info, lighting_mode="TOP"),
            dict(self.info, category="INVERTIDO"),
        ]
        for altered in variations:
            result = self.inspect(system, info=altered)
            self.assertNotEqual(result["detail"]["recognition_route"], "KNOWN_KNN")
        alternate_reference = self.reference.copy()
        alternate_reference[35, 35] ^= 1
        outcome = self.inspect(system, ref=alternate_reference)
        self.assertNotEqual(outcome["detail"]["recognition_route"], "KNOWN_KNN")
        self.assertEqual(len(system.specialist_calls), 5)

    def test_two_opposite_verified_labels_for_same_pair_force_review(self):
        system = self.create_router()
        system.experts["knn"].signatures_ok.append(self.make_record("OK"))
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        out = self.inspect(system)
        self.assertEqual(out["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(out["detail"]["recognition_route"], "MEMORY_CONFLICT")
        self.assertTrue(out["production_review_required"])
        self.assertEqual(system.specialist_calls, [])

    def test_json_only_or_operator_label_mismatch_is_not_known(self):
        system = self.create_router()
        system.experts["knn"].signatures_ng.append(
            self.make_record("NG", save_images=False)
        )
        self.assertEqual(
            self.inspect(system)["detail"]["recognition_route"], "NEW_CNN"
        )
        system.experts["knn"].signatures_ng = [
            self.make_record("NG", human_label="OK")
        ]
        system.reload_memory()
        self.assertEqual(
            self.inspect(system)["detail"]["recognition_route"], "NEW_CNN"
        )

    def test_auto_labeled_record_never_bypasses_cnn(self):
        system = self.create_router()
        system.experts["knn"].signatures_ok.append(
            self.make_record("OK", source="production_auto")
        )
        result = self.inspect(system)
        self.assertEqual(result["detail"]["recognition_route"], "NEW_CNN")

    def test_aoi_value_changed_requires_new_specialist_inspection(self):
        system = self.create_router()
        known = dict(self.info, value="0 <= 17 <= 35 FALTANDO")
        system.experts["knn"].signatures_ng.append(
            self.make_record("NG", info=known)
        )
        same = self.inspect(system, info=known)
        self.assertEqual(same["detail"]["recognition_route"], "KNOWN_KNN")
        changed = self.inspect(
            system,
            info=dict(known, value="0 <= 18 <= 35 FALTANDO")
        )
        self.assertEqual(changed["detail"]["recognition_route"], "NEW_CNN")

    def test_changed_or_missing_source_images_do_not_authorize_known(self):
        system = self.create_router()
        record = self.make_record("NG")
        system.experts["knn"].signatures_ng.append(record)
        path = Path(record["json_path"]).parent/"t.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image[60, 80] ^= 1
        cv2.imwrite(str(path), image)
        self.assertEqual(
            self.inspect(system)["detail"]["recognition_route"], "NEW_CNN"
        )

    def test_reload_memory_invalidates_exact_index(self):
        system = self.create_router()
        self.assertEqual(self.inspect(system)["detail"]["recognition_route"], "NEW_CNN")
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        system.reload_memory()
        self.assertEqual(self.inspect(system)["detail"]["recognition_route"], "KNOWN_KNN")

    def test_unavailable_image_cannot_auto_ok_or_use_specialists(self):
        system = self.create_router()
        bad = np.zeros((0, 1, 3), dtype=np.uint8)
        result = self.inspect(system, tst=bad)
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertFalse(production_decision_policy(result)["auto_allowed"])
        self.assertEqual(len(system.specialist_calls), 0)

    def test_replay_bypasses_new_router_to_preserve_physical_only_test(self):
        system = self.create_router()
        info = dict(self.info, _replay_without_memory=True)
        r = self.inspect(system, info=info)
        self.assertNotIn("recognition_route", r["detail"])
        self.assertEqual(len(system.specialist_calls), 1)

    def test_multilight_mixed_requires_operator_if_any_new_cnn(self):
        system = self.create_router()
        system.experts["knn"].signatures_ok.append(self.make_record("OK"))
        results = {}
        for mode in ("SIDE", "TOP", "MID"):
            results[mode] = self.inspect(
                system, info=dict(self.info, lighting_mode=mode)
            )
        fused = fuse_multilight(results, "FALTANDO")
        self.assertEqual(fused["detail"]["recognition_route"], "MULTILIGHT_MIXED")
        self.assertEqual(fused["detail"]["recognition_light_routes"]["SIDE"], "KNOWN_KNN")
        self.assertEqual(fused["detail"]["recognition_light_routes"]["TOP"], "NEW_CNN")
        self.assertFalse(production_decision_policy(fused)["auto_allowed"])

    def test_memory_lookup_does_not_use_similarity_threshold(self):
        system = self.create_router()
        rec = self.make_record("NG")
        rec["best_similarity"] = .999
        system.experts["knn"].signatures_ng.append(rec)
        changed = self.test.copy()
        changed[10, 10] ^= 1
        r = self.inspect(system, tst=changed)
        self.assertEqual(r["detail"]["recognition_route"], "NEW_CNN")


if __name__ == "__main__":
    unittest.main()
