"""Regressoes: memoria OK TOP a 97,74% deve sobreviver a fusao multilight.

Fixture baseada nos scores do evento XP 8dee2f1d331149eeb14718ff44386db3.
Nao equivale a replay das imagens reais ou validacao industrial.
"""
from copy import deepcopy
import unittest

from src.core.human_multilight_priority import (
    RULE, build_human_multilight_priority,
    verified_human_multilight_priority,
)
from src.core.multilight_fusion import fuse_multilight
from src.ui.production_confidence_gate import production_decision_policy
from tests.test_side_memory_first import case, good_probe


def event():
    cases = {}
    for light in ("SIDE", "TOP", "MID"):
        frame = case()
        frame["lighting_mode"] = light
        frame["verdict"] = (
            "FALHA FALSA" if light != "MID" else "REVISÃO OBRIGATÓRIA"
        )
        frame["is_defect"] = False
        frame["confidence"] = {
            "SIDE": .957570344, "TOP": .948215782, "MID": .886728525,
        }[light]
        d = frame["detail"]
        d["physical_score"] = 0.0
        d["cnn_v2_active"] = True
        d["cnn_v2_status"] = "INFERENCE_OK"
        d["cnn_v2_checkpoint_verified"] = True
        d["cnn_v2_checkpoint_sha256"] = "c" * 64
        d["cnn_v2_ng_score_uncalibrated"] = {
            "SIDE": .04242965579, "TOP": .05178421735, "MID": .88672852516,
        }[light]
        human = d["confirmed_visual_recall"]
        human["best_ok_source"]["lighting_mode"] = light
        human["best_ng_source"]["lighting_mode"] = light
        human["best_ok_similarity"] = {
            "SIDE": .97774325, "TOP": .97741850, "MID": .96266024,
        }[light]
        human["best_ng_similarity"] = {
            "SIDE": .98651746, "TOP": .93243586, "MID": .92983126,
        }[light]
        human["status"] = {
            "SIDE": "AMBIGUOUS",
            "TOP": "HUMAN_OK_VISUAL_CANDIDATE",
            "MID": "NO_MATCH",
        }[light]
        cases[light] = frame
    samples = {
        light: {"lighting_mode": light, "sample_image": object(),
                "test_image": object()}
        for light in ("SIDE", "TOP", "MID")
    }
    return cases, samples


def recovered(*, probe=None):
    analyses, samples = event()
    fused = fuse_multilight(analyses, "FALTANDO")
    assert fused is not None
    return build_human_multilight_priority(
        fused, analyses, samples, "FALTANDO",
        probe=probe if probe is not None else lambda *_: good_probe(trim=True),
    )


class HumanMultilightPriorityTests(unittest.TestCase):
    def test_1007_top_operator_ok_determines_visual_verdict_and_auto_zero(self):
        final = recovered()
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertEqual(final["detail"]["fusion_rule"], RULE)
        self.assertFalse(final["production_review_required"])
        witness = final["detail"]["human_multilight_priority"]["evidence"]["operator_witness"]
        self.assertEqual(witness["light"], "TOP")
        self.assertAlmostEqual(witness["similarity"], .97741850)
        self.assertEqual(
            final["detail"]["human_multilight_priority"]["evidence"]["local_ok_lights"],
            ["SIDE", "TOP"],
        )
        self.assertTrue(verified_human_multilight_priority(final))
        prearm = production_decision_policy(final)
        self.assertEqual(prearm["verdict"], "FALHA FALSA")
        self.assertFalse(prearm["auto_allowed"])
        final["detail"]["production_continuous_auto"] = True
        policy = production_decision_policy(final)
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertEqual(policy["reason"], "operator_human_multilight_97pct_auto_ok")
        self.assertTrue(policy["unverified_physical_label"])

    def test_physical_suspect_preserves_visual_ok_but_requires_operator(self):
        def suspect(*_):
            physical = good_probe(trim=True)
            physical["status"] = "SUSPECT_ABSENCE"
            physical["scales"][0]["suspected_absence"] = True
            return physical

        final = recovered(probe=suspect)
        final["detail"]["production_continuous_auto"] = True
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertTrue(final["production_review_required"])
        self.assertIs(verified_human_multilight_priority(final), False)
        policy = production_decision_policy(final)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertNotEqual(policy["reason"], "production_uncertain_auto_ng_quarantine_v1")

    def test_approximate_ng_png_never_overrides_confirmed_human_ok(self):
        # Evento 10:07: NG da KNN C na SIDE era apenas vizinho
        # visual (~81,88%); nao era par RGB exato nem fonte da
        # memoria humana NG de 98,65% (registro JSON separado).
        analyses, samples = event()
        side_knn = analyses["SIDE"]["detail"]["knn_c"]
        side_knn["neighbors_A"] = [{
            "label": "NG",
            "path": "public/ng_archive/2026-10-21_1442_FALTANDO.png",
            "similarity": 0.81875214,
            "source_pixel_sha256": "a" * 64,
        }]
        # No descritor B do log, o primeiro vizinho era OK.
        side_knn["neighbors_B"] = [{
            "label": "OK",
            "path": "public/ok_archive/2026-10-21_1416_FALTANDO.png",
            "similarity": 0.8293933,
            "source_pixel_sha256": "b" * 64,
        }]
        fused = fuse_multilight(analyses, "FALTANDO")
        final = build_human_multilight_priority(
            fused, analyses, samples, "FALTANDO",
            probe=lambda *_: good_probe(trim=True),
        )
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertEqual(final["detail"]["fusion_rule"], RULE)
        # Preservar provas conflitantes em vez de apagá-las para
        # aumentar artificialmente a confianca do resultado.
        diag = final["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
        self.assertEqual(diag["neighbors_A"][0]["label"], "NG")
        self.assertEqual(diag["neighbors_A"][0]["similarity"], 0.81875214)
        self.assertEqual(
            diag["neighbors_A"][0]["path"],
            "public/ng_archive/2026-10-21_1442_FALTANDO.png",
        )
        human_side = final["detail"]["human_multilight_priority"][
            "light_snapshots"]["SIDE"]["confirmed_visual_recall"]
        self.assertEqual(human_side["status"], "AMBIGUOUS")
        self.assertAlmostEqual(human_side["best_ng_similarity"], .98651746)
        human_top = final["detail"]["human_multilight_priority"][
            "light_snapshots"]["TOP"]["confirmed_visual_recall"]
        self.assertEqual(human_top["status"], "HUMAN_OK_VISUAL_CANDIDATE")
        self.assertAlmostEqual(human_top["best_ok_similarity"], .97741850)
        final["detail"]["production_continuous_auto"] = True
        self.assertEqual(production_decision_policy(final)["proposed_decision"], "OK")

    def test_independent_ng_corroboration_blocks_human_ok_priority(self):
        analyses, samples = event()
        fused = fuse_multilight(analyses, "FALTANDO")
        fused["detail"]["knn_visual_corroboration"] = {
            "status": "NG_CORROBORATED",
            "strong_ng_lights": ["SIDE", "MID"],
        }
        final = build_human_multilight_priority(
            fused, analyses, samples, "FALTANDO",
            probe=lambda *_: good_probe(trim=True),
        )
        # Evidencia NG corroborada e distinta de mero vizinho.
        self.assertNotEqual(final["detail"].get("fusion_rule"), RULE)
        self.assertEqual(
            final["detail"]["knn_visual_corroboration"]["status"],
            "NG_CORROBORATED",
        )

    def test_clear_human_ng_in_other_light_blocks_priority(self):
        analyses, samples = event()
        analyses["SIDE"]["detail"]["confirmed_visual_recall"].update(
            status="HUMAN_NG_VISUAL_CANDIDATE",
            best_ng_similarity=.990, best_ok_similarity=.940,
        )
        fused = fuse_multilight(analyses, "FALTANDO")
        final = build_human_multilight_priority(
            fused, analyses, samples, "FALTANDO",
            probe=lambda *_: good_probe(trim=True),
        )
        self.assertNotEqual(final["detail"].get("fusion_rule"), RULE)

    def test_one_local_ok_is_not_enough_for_automatic_approval(self):
        analyses, samples = event()
        analyses["SIDE"]["verdict"] = "REVISÃO OBRIGATÓRIA"
        fused = fuse_multilight(analyses, "FALTANDO")
        final = build_human_multilight_priority(
            fused, analyses, samples, "FALTANDO",
            probe=lambda *_: good_probe(),
        )
        self.assertNotEqual(final["detail"].get("fusion_rule"), RULE)

    def test_tampered_memory_provenance_cannot_authorize_xp(self):
        final = recovered()
        final["detail"]["production_continuous_auto"] = True
        for key, value in (
            ("source", "ARCHIVE_PNG"),
            ("label", "NG"),
            ("record_id", "unknown"),
        ):
            tampered = deepcopy(final)
            witness = tampered["detail"]["human_multilight_priority"][
                "light_snapshots"]["TOP"]["confirmed_visual_recall"]["best_ok_source"]
            witness[key] = value
            with self.subTest(key=key):
                self.assertIsNone(verified_human_multilight_priority(tampered))
                self.assertFalse(production_decision_policy(tampered)["auto_allowed"])

    def test_tampered_probe_or_forged_marker_cannot_authorize_xp(self):
        final = recovered()
        final["detail"]["production_continuous_auto"] = True
        edited = deepcopy(final)
        edited["detail"]["human_multilight_priority"]["physical_probes"][
            "MID"]["scales"][1]["suspected_absence"] = True
        self.assertIsNone(verified_human_multilight_priority(edited))
        self.assertFalse(production_decision_policy(edited)["auto_allowed"])
        edited = deepcopy(final)
        edited["detail"]["human_multilight_priority"]["evidence"][
            "operator_witness"]["similarity"] = 1.0
        self.assertFalse(production_decision_policy(edited)["auto_allowed"])
        edited = deepcopy(final)
        edited["detail"]["hard_missing_evidence"] = True
        self.assertFalse(production_decision_policy(edited)["auto_allowed"])

    def test_weak_or_ambiguous_operator_candidate_stays_in_fallback(self):
        for ok, ng, status in (
            (.96999, .92, "HUMAN_OK_VISUAL_CANDIDATE"),
            (.9774185, .96, "HUMAN_OK_VISUAL_CANDIDATE"),
            (.9774185, .932, "AMBIGUOUS"),
        ):
            with self.subTest(ok=ok, ng=ng, status=status):
                analyses, samples = event()
                human = analyses["TOP"]["detail"]["confirmed_visual_recall"]
                human["best_ok_similarity"] = ok
                human["best_ng_similarity"] = ng
                human["status"] = status
                fused = fuse_multilight(analyses, "FALTANDO")
                final = build_human_multilight_priority(
                    fused, analyses, samples, "FALTANDO",
                    probe=lambda *_: good_probe(),
                )
                self.assertNotEqual(final["detail"].get("fusion_rule"), RULE)


if __name__ == "__main__":
    unittest.main()
