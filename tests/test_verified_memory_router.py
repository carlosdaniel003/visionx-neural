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
from src.core.neural.faltando_category_scope import uses_faltando_v2
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
            uses_faltando_v2(info.get("category")) else ["shift_expert.py"],
            "detail": {"cnn_v2_active": True, "cnn_v2_experimental": True}
            if uses_faltando_v2(info.get("category"))
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

    def test_shared_categories_use_cnn_but_adhesive_uses_physical_specialist(self):
        system = self.create_router()
        for category in ("DESLOCADO", "EMBORCADO", "INVERTIDO"):
            result = self.inspect(system, info=dict(self.info, category=category))
            self.assertEqual(result["detail"]["recognition_route"], "NEW_CNN")
            self.assertEqual(result["detail"]["cnn_v2_aoi_category"], category)
            self.assertEqual(result["active_engines"], ["faltando_cnn_v2.py"])
            self.assertNotIn("_replay_without_memory", system.specialist_calls[-1])
        result = self.inspect(system, info=dict(self.info, category="MUITO ADESIVO"))
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

    def test_knn_warmup_builds_index_before_first_image(self):
        system = self.create_router()
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        index = system._get_verified_memory_index()
        status = system.prepare_verified_memory_index()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["record_count"], 1)
        result = self.inspect(system)
        self.assertEqual(result["detail"]["recognition_route"], "KNOWN_KNN")
        profile = result["detail"]["verified_memory_timing"]
        self.assertTrue(profile["index_reused"])
        self.assertEqual(profile["index_build_ms"], 0)
        self.assertEqual(profile["warmup"]["state"], "ready")
        self.assertIs(index, system._get_verified_memory_index())

    def test_knn_warmup_waits_for_complete_index_before_inspection(self):
        from threading import Event, Thread
        from unittest.mock import patch

        system = self.create_router()
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        index = system._get_verified_memory_index()
        original_entry = index._entry
        entered, release = Event(), Event()
        results = []

        def held_entry(item):
            entered.set()
            self.assertTrue(release.wait(8))
            return original_entry(item)

        with patch.object(index, "_entry", side_effect=held_entry):
            warm = Thread(target=system.prepare_verified_memory_index)
            warm.start()
            self.assertTrue(entered.wait(8))
            inspect = Thread(target=lambda: results.append(self.inspect(system)))
            inspect.start()
            release.set()
            warm.join(8)
            inspect.join(8)
        self.assertFalse(warm.is_alive())
        self.assertFalse(inspect.is_alive())
        self.assertEqual(results[0]["verdict"], "DEFEITO REAL")
        self.assertEqual(results[0]["detail"]["recognition_route"], "KNOWN_KNN")
        self.assertTrue(
            results[0]["detail"]["verified_memory_timing"]["index_reused"]
        )

    def test_knn_warmup_invalidation_still_forces_rebuild(self):
        system = self.create_router()
        knn = system.experts["knn"]
        knn.signatures_ng.append(self.make_record("NG"))
        self.assertEqual(system.prepare_verified_memory_index()["state"], "ready")
        self.assertEqual(
            self.inspect(system)["detail"]["recognition_route"], "KNOWN_KNN"
        )
        system.reload_memory()
        index = system._get_verified_memory_index()
        self.assertEqual(index._warmup_status["state"], "invalidated")
        result = self.inspect(system)
        self.assertEqual(result["detail"]["recognition_route"], "KNOWN_KNN")
        self.assertFalse(
            result["detail"]["verified_memory_timing"]["index_reused"]
        )

    def test_archive_only_refresh_preserves_legacy_index_and_verified_label(self):
        """O watcher troca duas listas, mas não mudou nenhum par legado."""
        system = self.create_router()
        knn = system.experts["knn"]
        legacy = self.make_record("NG")
        knn.signatures_ng = [legacy]
        self.assertEqual(system.prepare_verified_memory_index()["state"], "ready")
        original = self.inspect(system)
        self.assertEqual(original["detail"]["recognition_route"], "KNOWN_KNN")
        initial = original["detail"]["verified_memory_timing"]
        self.assertTrue(initial["index_reused"])
        self.assertEqual(initial["legacy_index_record_count"], 1)

        # Simula múltiplos lotes append-only do arquivo durável:
        # refresh_archive_memory() reconstrói listas, retendo os mesmos
        # objetos legados e incluindo registros archive_persistent novos.
        for count in (13, 28, 47):
            durable = [
                {"archive_persistent": True, "archive_record_id": str(i)}
                for i in range(count)
            ]
            knn.signatures_ok = list(durable)
            knn.signatures_ng = [legacy]
            result = self.inspect(system)
            profile = result["detail"]["verified_memory_timing"]
            self.assertEqual(result["detail"]["recognition_route"], "KNOWN_KNN")
            self.assertTrue(profile["index_reused"])
            self.assertEqual(profile["index_build_ms"], 0)
            self.assertEqual(profile["legacy_index_record_count"], 1)
            self.assertEqual(profile["archive_persistent_record_count"], count)
            self.assertEqual(profile["record_count"], count + 1)
            self.assertEqual(profile["warmup"]["state"], "ready")

    def test_archive_refresh_then_legacy_change_forces_fresh_verification(self):
        """Atualizações reais do legado continuam exigindo novo índice."""
        system = self.create_router()
        knn = system.experts["knn"]
        legacy = self.make_record("NG")
        knn.signatures_ng = [legacy]
        system.prepare_verified_memory_index()
        self.assertEqual(self.inspect(system)["detail"]["recognition_route"], "KNOWN_KNN")

        knn.signatures_ok = [
            {"archive_persistent": True, "archive_record_id": "added"}
        ]
        stable = self.inspect(system)["detail"]["verified_memory_timing"]
        self.assertTrue(stable["index_reused"])

        # Troca de objeto legado na mesma posição/quantidade é alteração.
        another = self.make_record("OK")
        knn.signatures_ng = [another]
        changed = self.inspect(system)
        self.assertEqual(changed["verdict"], "FALHA FALSA")
        self.assertEqual(changed["detail"]["recognition_route"], "KNOWN_KNN")
        profile = changed["detail"]["verified_memory_timing"]
        self.assertFalse(profile["index_reused"])
        self.assertEqual(profile["index_rebuild_reason"], "legacy_records_changed")

        system.reload_memory()
        invalidated = system._get_verified_memory_index()._warmup_status
        self.assertEqual(invalidated["state"], "invalidated")
        self.assertEqual(invalidated["reason"], "orchestrator_reload_memory")
        post_reload = self.inspect(system)["detail"]["verified_memory_timing"]
        self.assertFalse(post_reload["index_reused"])
        self.assertEqual(
            post_reload["index_rebuild_reason"],
            "not_initialized_or_explicitly_invalidated",
        )

    def test_first_verified_lookup_builds_index_then_reuses_it(self):
        system = self.create_router()
        system.experts["knn"].signatures_ok.append(self.make_record("OK"))
        first = self.inspect(system)
        second = self.inspect(system)
        self.assertEqual(first["verdict"], second["verdict"])
        self.assertEqual(first["detail"]["recognition_route"], "KNOWN_KNN")
        first_profile = first["detail"]["verified_memory_timing"]
        second_profile = second["detail"]["verified_memory_timing"]
        self.assertEqual(first_profile["schema"], "visionx.verified_knn_lookup_timing.v1")
        self.assertEqual(first_profile["status"], "KNOWN")
        self.assertFalse(first_profile["index_reused"])
        self.assertGreaterEqual(first_profile["index_build_ms"], 0)
        self.assertEqual(first_profile["record_count"], 1)
        self.assertTrue(second_profile["index_reused"])
        self.assertEqual(second_profile["index_build_ms"], 0.0)
        self.assertIn("archive_lookup_ms", second_profile)
        self.assertIn("fingerprints_ms", second_profile)
        self.assertIn("records_snapshot_ms", second_profile)
        self.assertGreaterEqual(second_profile["total_ms"], 0)
        self.assertEqual(system.specialist_calls, [])

    def test_index_profile_invalidated_by_reload_and_new_record(self):
        system = self.create_router()
        initial = self.inspect(system)["detail"]["verified_memory_timing"]
        self.assertFalse(initial["index_reused"])
        self.assertTrue(
            self.inspect(system)["detail"]["verified_memory_timing"]["index_reused"]
        )
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        system.reload_memory()
        refreshed = self.inspect(system)
        self.assertEqual(refreshed["detail"]["recognition_route"], "KNOWN_KNN")
        self.assertFalse(refreshed["detail"]["verified_memory_timing"]["index_reused"])

    def test_verified_memory_profile_on_conflict_and_fail_closed(self):
        system = self.create_router()
        system.experts["knn"].signatures_ok.append(self.make_record("OK"))
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        conflict = self.inspect(system)
        self.assertEqual(conflict["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(
            conflict["detail"]["verified_memory_timing"]["status"], "CONFLICT"
        )

        class CorruptStore:
            def lookup(self, *args):
                raise ValueError("Snapshot adulterado")

        class OtherSystem(FakeOrchestrator):
            pass

        install_memory_first_router(
            OtherSystem, memory=VerifiedKNNMemory(archive_store=CorruptStore())
        )
        result = self.inspect(OtherSystem())
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertFalse(production_decision_policy(result)["auto_allowed"])
        self.assertEqual(
            result["detail"]["verified_memory_timing"]["status"], "UNAVAILABLE"
        )
        self.assertIn(
            "archive_lookup_ms", result["detail"]["verified_memory_timing"]
        )

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
