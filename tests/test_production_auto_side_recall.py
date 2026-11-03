"""Regressões da memória AUTO separada e do atalho SIDE em Produção.

Não usa XP real, não converte inferências em labels humanos e nunca
aprova 0/OK sem a concordância SIDE + sonda física corrente.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.core.production_auto_side_recall import (
    ProductionAutoSideMemory, build_auto_side_recall,
    verify_auto_side_recall,
)
from src.services.image_archive_dedup import image_fingerprint
from src.ui.production_confidence_gate import production_decision_policy


class _HumanArchive:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def records(self):
        return deepcopy(self.rows)


class ProductionAutoSideRecallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ProductionAutoSideMemory(
            Path(self.tmp.name) / "auto" / "side.sqlite3",
        )
        self.human = _HumanArchive()
        self.reference = np.full((72, 96, 3), 70, dtype=np.uint8)
        self.test = np.full((72, 96, 3), 90, dtype=np.uint8)
        self.test[12:40, 20:55] = 105
        self.frame = np.full((280, 600, 3), 110, dtype=np.uint8)

    def _side(self, label="NG"):
        return {
            "is_defect": label == "NG",
            "verdict": "DEFEITO REAL" if label == "NG" else "FALHA FALSA",
            "confidence": 0.99,
            "lighting_mode": "SIDE",
            "detail": {"recognition_route": "NEW_CNN"},
        }

    def _sample(self, label="NG", *, lighting="SIDE"):
        return {
            "lighting_mode": lighting,
            "analysis": self._side(label),
            "sample_image": self.reference.copy(),
            "test_image": self.test.copy(),
            "source_frame": self.frame.copy(),
        }

    def _task(self, label="NG", *, event_id="event-1", source_reason=None,
              include_three=True, side_label=None):
        side_label = side_label or label
        reason = source_reason or (
            "production_binary_unverified_ng_v1" if label == "NG"
            else "false_failure_auto_ok"
        )
        analysis = {
            "is_defect": label == "NG",
            "verdict": "DEFEITO REAL" if label == "NG" else "FALHA FALSA",
            "detail": {
                "production_decision_policy": {
                    "auto_allowed": True,
                    "proposed_decision": label,
                    "reason": reason,
                    "resolution": "automatic",
                },
            },
        }
        samples = (
            [self._sample(side_label, lighting=light)
             for light in ("SIDE", "TOP", "MID")]
            if include_three else []
        )
        return {
            "source": "production_auto",
            "label": label,
            "event_id": event_id,
            "analysis": analysis,
            "aoi_info": {"category": "FALTANDO", "lighting_mode": "SIDE"},
            "sample_image": self.reference.copy(),
            "ng_image": self.test.copy(),
            "source_frame": self.frame.copy(),
            "multilight_samples": samples,
        }

    def _build(self, label, *, side=None, sample=None):
        return build_auto_side_recall(
            side if side is not None else self._side(label),
            sample if sample is not None else self._sample(label),
            "FALTANDO",
            store=self.store, human_store=self.human,
        )

    def _policy(self, result):
        # Simula o armamento binário feito pelo controlador antes da UI.
        detail = result["detail"]
        detail["production_continuous_auto"] = True
        detail["production_binary_auto"] = True
        with patch(
            "src.core.production_auto_side_recall.ProductionAutoSideMemory",
            return_value=self.store,
        ), patch(
            "src.core.production_auto_side_recall.PersistentArchiveKNN",
            return_value=self.human,
        ):
            return production_decision_policy(result)

    def test_ng_automatically_judged_on_three_lights_repeats_from_side_only(self):
        self.assertTrue(self.store.record(self._task("NG")))
        result = self._build("NG")
        self.assertIsNotNone(result)
        self.assertEqual(result["lighting_mode"], "SIDE")
        self.assertTrue(result["multilight_final"])
        self.assertEqual(result["detail"]["fusion_rule"],
                         "production_auto_side_exact_recall_v1")
        self.assertEqual(result["detail"]["production_auto_side_recall"]["analyzed_lights"],
                         "SIDE,TOP,MID")
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertEqual(
            verify_auto_side_recall(
                result, store=self.store, human_store=self.human,
            ), "NG",
        )
        policy = self._policy(result)
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "NG")
        self.assertTrue(policy["unverified_physical_label"])
        self.assertEqual(policy["reason"], "production_auto_side_recall_ng_v1")

    def test_ok_auto_replay_requires_new_side_ok_and_independent_probe(self):
        self.assertTrue(self.store.record(self._task("OK")))
        with patch(
            "src.core.production_auto_side_recall.physical_absence_probe",
            return_value={"status": "NO_STRONG_EVIDENCE"},
        ):
            result = self._build("OK")
        self.assertIsNotNone(result)
        policy = self._policy(result)
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertEqual(policy["reason"], "production_auto_side_recall_ok_v1")
        self.assertTrue(policy["unverified_physical_label"])

        # Identidade AUTO por si só NÃO substitui uma CNN SIDE que viu NG.
        self.assertIsNone(self._build("OK", side=self._side("NG")))
        with patch(
            "src.core.production_auto_side_recall.physical_absence_probe",
            return_value={"status": "UNAVAILABLE"},
        ):
            self.assertIsNone(self._build("OK"))

    def test_changed_one_pixel_or_category_never_shortcuts(self):
        self.assertTrue(self.store.record(self._task("NG")))
        sample = self._sample()
        sample["test_image"][0, 0, 0] += 1
        self.assertIsNone(self._build("NG", sample=sample))
        self.assertIsNone(build_auto_side_recall(
            self._side(), self._sample(), "DESLOCADO",
            store=self.store, human_store=self.human,
        ))

    def test_latest_human_same_pair_supersedes_auto_of_either_label(self):
        self.assertTrue(self.store.record(self._task("NG")))
        self.human.rows = [{
            "identity": [
                "FALTANDO", "SIDE",
                image_fingerprint(self.reference),
                image_fingerprint(self.test),
            ],
            "label": "OK",
            "sources": [{
                "source_kind": "OPERATOR_CONFIRMED",
                "source_label": "OK",
                "source_frame_sha256": image_fingerprint(self.frame),
            }],
        }]
        self.assertIsNone(self._build("NG"))
        # O gate também detecta uma correção entre a apresentação e XP.
        self.human.rows = []
        result = self._build("NG")
        self.assertIsNotNone(result)
        self.human.rows = [{
            "identity": [
                "FALTANDO", "SIDE",
                image_fingerprint(self.reference),
                image_fingerprint(self.test),
            ],
            "label": "OK",
            "sources": [{"source_kind": "OPERATOR_CONFIRMED"}],
        }]
        self.assertIsNone(
            verify_auto_side_recall(
                result, store=self.store, human_store=self.human,
            ),
        )

    def test_known_physical_ng_archive_vetoes_auto_ok(self):
        self.assertTrue(self.store.record(self._task("OK")))
        self.human.rows = [{
            "identity": [
                "FALTANDO", "SIDE",
                image_fingerprint(self.reference),
                image_fingerprint(self.test),
            ],
            "label": "NG",
            "sources": [{"source_kind": "ARCHIVE_PNG"}],
        }]
        self.assertIsNone(self._build("OK"))

    def test_opposing_auto_labels_force_full_analysis_until_human(self):
        self.assertTrue(self.store.record(self._task("NG", event_id="event-ng")))
        self.assertTrue(self.store.record(
            self._task("OK", event_id="event-ok", side_label="OK"),
        ))
        self.assertEqual(
            self.store.find(
                ("FALTANDO", image_fingerprint(self.reference),
                 image_fingerprint(self.test)),
            )["status"], "CONFLICT",
        )
        self.assertIsNone(self._build("NG"))

    def test_invalid_proof_or_missing_db_halts_without_xp(self):
        self.assertTrue(self.store.record(self._task("NG")))
        result = self._build("NG")
        result["detail"]["production_auto_side_recall"]["label"] = "OK"
        self.assertIsNone(verify_auto_side_recall(
            result, store=self.store, human_store=self.human,
        ))
        result["detail"]["production_auto_side_recall"]["label"] = "NG"
        self.store.path.unlink()
        self.assertIsNone(verify_auto_side_recall(
            result, store=self.store, human_store=self.human,
        ))

    def test_failed_or_unconfirmed_label_is_not_memorized(self):
        task = self._task("NG")
        task["analysis"]["detail"]["production_decision_policy"]["resolution"] = "pending"
        self.assertFalse(self.store.record(task))
        task["analysis"]["detail"]["production_decision_policy"]["resolution"] = "automatic"
        task["source"] = "button"
        self.assertFalse(self.store.record(task))

    def test_replaying_auto_outcome_does_not_create_recursive_training_label(self):
        task = self._task(
            "NG", source_reason="production_auto_side_recall_ng_v1",
        )
        self.assertFalse(self.store.record(task))
        from src.services.live_session_learning import immediate_confirmed_memory
        from src.services.neural_online_learning import eligible_online_case
        self.assertFalse(immediate_confirmed_memory(task)["committed"])
        task["analysis"]["detail"]["recognition_route"] = "NEW_CNN"
        self.assertFalse(eligible_online_case(task))

    def test_corrupted_sqlite_fails_closed_without_fallback_to_autook(self):
        self.store.path.parent.mkdir(parents=True, exist_ok=True)
        self.store.path.write_bytes(b"NOT A DATABASE")
        self.assertIsNone(self._build("OK"))


class AutomaticCommandCommitTests(unittest.TestCase):
    def _panel(self, *, command_success=True):
        from src.services.anomaly_learning import install_anomaly_learning

        class Button:
            def setEnabled(self, _value):
                pass

        class Panel:
            def __init__(self):
                self.btn_save_ok = Button()
                self.btn_save_ng = Button()
                self.btn_skip = Button()
                self.current_sample = np.full((72, 96, 3), 30, dtype=np.uint8)
                self.current_ng = np.full((72, 96, 3), 70, dtype=np.uint8)
                self.current_aoi_info = {"category": "FALTANDO"}
                self.current_analysis = {
                    "is_defect": True, "verdict": "DEFEITO REAL",
                    "detail": {
                        "production_decision_policy": {
                            "auto_allowed": True,
                            "proposed_decision": "NG",
                            "reason": "production_binary_unverified_ng_v1",
                            "resolution": "automatic",
                        },
                    },
                }
                self.network_intake_last_image_event_id = "actual-01"
                self.capture_cycle_source = "network"
                self.last_decision_command_success = None

            def save_label(self, _label, source="button"):
                self.last_decision_command_success = command_success
                return None if command_success else False

        install_anomaly_learning(Panel)
        return Panel()

    def test_record_only_after_successful_xp_command(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ProductionAutoSideMemory(Path(folder) / "memory.sqlite3")
            with patch(
                "src.core.production_auto_side_recall.ProductionAutoSideMemory",
                return_value=store,
            ), patch(
                "src.services.anomaly_learning._submit_decision_persistence"
            ) as submit, patch(
                "src.services.anomaly_learning.immediate_confirmed_memory"
            ) as human:
                panel = self._panel()
                self.assertIsNone(panel.save_label("NG", source="production_auto"))
                identity = (
                    "FALTANDO", image_fingerprint(panel.current_sample),
                    image_fingerprint(panel.current_ng),
                )
                self.assertEqual(store.find(identity)["label"], "NG")
                submit.assert_called_once()  # quarentena, não treino
                human.assert_not_called()

    def test_failed_xp_never_writes_auto_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ProductionAutoSideMemory(Path(folder) / "memory.sqlite3")
            with patch(
                "src.core.production_auto_side_recall.ProductionAutoSideMemory",
                return_value=store,
            ), patch(
                "src.services.anomaly_learning._submit_decision_persistence"
            ) as submit:
                panel = self._panel(command_success=False)
                self.assertIs(panel.save_label("NG", source="production_auto"), False)
                self.assertFalse(store.path.exists())
                submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
