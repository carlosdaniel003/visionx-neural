"""Último 0/1 humano sobre frame RGB exato e gate binário da Produção."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.core.archive_persistent_memory import PersistentArchiveKNN
from src.core.operator_exact_frame_recall import (
    RULE, SIDE_RULE, build_latest_operator_exact_frame,
    build_latest_operator_exact_side, verified_latest_operator_exact_frame,
)
from src.services.live_session_learning import immediate_confirmed_memory
from src.ui.production_confidence_gate import production_decision_policy
from tests.test_live_session_learning import labeled_task


LIGHTS = ("SIDE", "TOP", "MID")


def capture(label: str, color=80, event_id="human-a"):
    task = labeled_task(label, value=color)
    task["event_id"] = event_id
    task["multilight_samples"] = []
    samples = {}
    for index, light in enumerate(LIGHTS):
        frame = np.full((48, 60, 3), 12 + index * 25, dtype=np.uint8)
        pair_ref = task["sample_image"].copy()
        pair_test = task["ng_image"].copy()
        item = {
            "lighting_mode": light,
            "sample_image": pair_ref, "test_image": pair_test,
            "source_frame": frame,
        }
        task["multilight_samples"].append(item)
        samples[light] = item
    return task, samples


def ng_multilight():
    return {
        "verdict": "DEFEITO REAL", "is_defect": True,
        "production_review_required": False,
        "multilight_final": True,
        "detail": {
            "multilight_final": True,
            "multilight_category": "FALTANDO",
            "recognition_route": "NEW_CNN",
            "recognition_light_routes": {
                light: "NEW_CNN" for light in LIGHTS
            },
            "physical_score": 0.0,
            "decision_trace": {
                "operator_review_required": False,
                "hard_missing_evidence": False,
            },
        },
    }


class ExactFrameHumanMemoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = PersistentArchiveKNN(Path(temp.name))

    def learned_ok(self):
        task, samples = capture("OK", event_id="operator-corrected")
        answer = immediate_confirmed_memory(task, store=self.store)
        self.assertTrue(answer["committed"])
        return samples

    def test_operator_source_frame_is_persisted_all_three_lights(self):
        samples = self.learned_ok()
        for record in self.store.records():
            source = record["sources"][0]
            self.assertEqual(source["source_kind"], "OPERATOR_CONFIRMED")
            self.assertEqual(source["event_id"], "operator-corrected")
            self.assertEqual(len(source["source_frame_sha256"]), 64)
        for light in LIGHTS:
            match = self.store.latest_confirmed_human_for_frame(
                samples[light]["source_frame"], "FALTANDO", light,
            )
            self.assertEqual(match["status"], "KNOWN")
            self.assertEqual(match["label"], "OK")

    def test_side_only_manual_ok_prevents_repeated_exact_side_ng(self):
        task, samples = capture("OK", event_id="side-human-correction")
        task["multilight_samples"] = [samples["SIDE"]]
        self.assertTrue(
            immediate_confirmed_memory(task, store=self.store)["committed"]
        )
        side_analysis = {
            "verdict": "DEFEITO REAL",
            "is_defect": True,
            "lighting_mode": "SIDE",
            "detail": {
                "decision_trace": {
                    "hard_missing_evidence": False,
                    "operator_review_required": False,
                },
            },
        }
        with patch(
            "src.core.operator_exact_frame_recall.physical_absence_probe",
            return_value={"status": "NO_STRONG_EVIDENCE"},
        ):
            final = build_latest_operator_exact_side(
                side_analysis, samples["SIDE"], "FALTANDO",
                store=self.store,
            )
        self.assertIsNotNone(final)
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertEqual(final["detail"]["fusion_rule"], SIDE_RULE)
        self.assertEqual(final["lighting_mode"], "SIDE")
        self.assertEqual(
            verified_latest_operator_exact_frame(final, store=self.store),
            "OK",
        )
        final["detail"]["production_continuous_auto"] = True
        final["detail"]["production_binary_auto"] = True
        with patch(
            "src.core.operator_exact_frame_recall.PersistentArchiveKNN",
            return_value=self.store,
        ):
            policy = production_decision_policy(final)
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertEqual(
            policy["reason"], "operator_latest_exact_frame_side_auto_ok",
        )

    def test_side_only_requires_identical_frame_and_clean_physical_probe(self):
        task, samples = capture("OK")
        task["multilight_samples"] = [samples["SIDE"]]
        immediate_confirmed_memory(task, store=self.store)
        analysis = {
            "verdict": "DEFEITO REAL", "is_defect": True,
            "lighting_mode": "SIDE", "detail": {},
        }
        changed = dict(samples["SIDE"])
        changed["source_frame"] = samples["SIDE"]["source_frame"].copy()
        changed["source_frame"][1, 1, 1] ^= 1
        self.assertIsNone(
            build_latest_operator_exact_side(
                analysis, changed, "FALTANDO", store=self.store,
            )
        )
        with patch(
            "src.core.operator_exact_frame_recall.physical_absence_probe",
            return_value={"status": "SUSPECT_ABSENCE"},
        ):
            self.assertIsNone(
                build_latest_operator_exact_side(
                    analysis, samples["SIDE"], "FALTANDO",
                    store=self.store,
                )
            )

    def test_exact_frames_override_ng_even_when_crops_change(self):
        samples = self.learned_ok()
        # Recorte em OCR ligeiramente diferente: par AOI deixa de ser exato.
        for item in samples.values():
            item["test_image"] = item["test_image"].copy()
            item["test_image"][0, 0, 0] ^= 1
        initial = ng_multilight()
        with patch(
            "src.core.operator_exact_frame_recall.physical_absence_probe",
            return_value={"status": "NO_STRONG_EVIDENCE"},
        ):
            final = build_latest_operator_exact_frame(
                initial, samples, "FALTANDO", store=self.store,
            )
        self.assertEqual(initial["verdict"], "DEFEITO REAL")
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertEqual(final["detail"]["fusion_rule"], RULE)
        self.assertEqual(
            final["detail"]["operator_latest_exact_frame"]["source_event_id"],
            "operator-corrected",
        )
        self.assertEqual(
            verified_latest_operator_exact_frame(final, store=self.store),
            "OK",
        )
        final["detail"]["production_continuous_auto"] = True
        final["detail"]["production_binary_auto"] = True
        with patch(
            "src.core.operator_exact_frame_recall.PersistentArchiveKNN",
            return_value=self.store,
        ):
            policy = production_decision_policy(final)
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertEqual(
            policy["reason"], "operator_latest_exact_frame_3light_auto_ok",
        )

    def test_new_latest_ng_switches_back_to_ng(self):
        samples = self.learned_ok()
        later, _ = capture("NG", event_id="human-corrected-again")
        immediate_confirmed_memory(later, store=self.store)
        with patch(
            "src.core.operator_exact_frame_recall.physical_absence_probe",
            return_value={"status": "NO_STRONG_EVIDENCE"},
        ):
            result = build_latest_operator_exact_frame(
                ng_multilight(), samples, "FALTANDO", store=self.store,
            )
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertEqual(
            result["detail"]["operator_latest_exact_frame"]["source_event_id"],
            "human-corrected-again",
        )

    def test_single_pixel_frame_change_is_not_exact(self):
        samples = self.learned_ok()
        samples["SIDE"]["source_frame"] = samples["SIDE"]["source_frame"].copy()
        samples["SIDE"]["source_frame"][10, 15, 1] ^= 1
        initial = ng_multilight()
        result = build_latest_operator_exact_frame(
            initial, samples, "FALTANDO", store=self.store,
        )
        self.assertIs(result, initial)
        self.assertNotIn("operator_latest_exact_frame", initial["detail"])

    def test_invalid_probe_blocks_automatic_ok(self):
        samples = self.learned_ok()
        for status in ("SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW"):
            with self.subTest(status=status), patch(
                "src.core.operator_exact_frame_recall.physical_absence_probe",
                return_value={"status": status},
            ):
                initial = ng_multilight()
                result = build_latest_operator_exact_frame(
                    initial, samples, "FALTANDO", store=self.store,
                )
                self.assertIs(result, initial)

    def test_proof_mutation_and_stale_latest_label_block_xp(self):
        samples = self.learned_ok()
        with patch(
            "src.core.operator_exact_frame_recall.physical_absence_probe",
            return_value={"status": "NO_STRONG_EVIDENCE"},
        ):
            final = build_latest_operator_exact_frame(
                ng_multilight(), samples, "FALTANDO", store=self.store,
            )
        self.assertEqual(
            verified_latest_operator_exact_frame(final, store=self.store),
            "OK",
        )
        final["detail"]["operator_latest_exact_frame"]["matches"]["SIDE"][
            "frame_sha256"
        ] = "1" * 64
        self.assertIsNone(
            verified_latest_operator_exact_frame(final, store=self.store),
        )
        final["detail"]["production_continuous_auto"] = True
        final["detail"]["production_binary_auto"] = True
        with patch(
            "src.core.operator_exact_frame_recall.PersistentArchiveKNN",
            return_value=self.store,
        ):
            policy = production_decision_policy(final)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operational_error"])

    def test_unconfirmed_archive_or_automatic_label_cannot_form_frame_override(self):
        task, samples = capture("OK", event_id="auto-prediction")
        task["source"] = "production_auto"
        self.assertFalse(
            immediate_confirmed_memory(task, store=self.store)["committed"]
        )
        initial = ng_multilight()
        self.assertIs(
            build_latest_operator_exact_frame(
                initial, samples, "FALTANDO", store=self.store,
            ), initial,
        )


if __name__ == "__main__":
    unittest.main()
