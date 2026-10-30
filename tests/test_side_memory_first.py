"""Regressões da prioridade de memória por iluminação SIDE (evento 30/10).

São testes de contrato com números reais, NÃO equivalem a prova industrial.
"""
from copy import deepcopy
import unittest

from src.core.side_memory_first import (
    RULE, build_side_memory_first, memory_first_side_decision,
)
from src.ui.production_confidence_gate import production_decision_policy


def _source(label="OK", light="SIDE"):
    return {
        "label": label, "lighting_mode": light,
        "record_id": ("a" if label == "OK" else "b") * 64,
        "source": "OPERATOR_CONFIRMED",
    }


def _row(label, score, path=None):
    path = path or (
        "public/" + ("ok_archive" if label == "OK" else "ng_archive")
        + "/archive_FALTANDO.png"
    )
    return {
        "label": label, "similarity": score, "path": path,
        "source_pixel_sha256": ("c" if label == "OK" else "d") * 64,
    }


def case(label="OK"):
    other = "NG" if label == "OK" else "OK"
    # SIDE do evento 88584d..., 30/10. KNN A/B escolheram o MESMO PNG OK.
    a = [_row(label, .98236720), _row(other, .86601348),
         _row(label, .86592282), _row(label, .85552766), _row(label, .84238623)]
    b = [_row(label, .96262484), _row(label, .84250628),
         _row(other, .82652949), _row(label, .81967648), _row(label, .8171847)]
    human = {
        "schema": "visionx.human_visual_recall.v1",
        "status": "HUMAN_" + label + "_VISUAL_CANDIDATE",
        "source": "PERSISTENT_HUMAN_LABELS",
        "diagnostic_only": True, "auto_ok_authorized": False,
        "category": "FALTANDO",
        "best_" + label.lower() + "_similarity": .99798172,
        "best_" + other.lower() + "_similarity": .95079253,
        "best_" + label.lower() + "_source": _source(label),
        "best_" + other.lower() + "_source": _source(other),
    }
    return {
        "verdict": "DEFEITO REAL" if label == "OK" else "FALHA FALSA",
        "is_defect": label == "OK", "confidence": .99980789,
        "production_review_required": True, "lighting_mode": "SIDE",
        "detail": {
            "recognition_route": "NEW_CNN", "recognition_match": "NOT_FOUND",
            "recognition_memory_verified": False,
            "cnn_v2_aoi_category": "FALTANDO",
            "cnn_v2_active": True, "cnn_v2_experimental": True,
            "cnn_v2_status": "INFERENCE_OK",
            "hard_missing_evidence": False,
            "decision_trace": {
                "operator_review_required": True,
                "fusion_rule": "cnn_v2_dualscale_direct",
                "weights": {"cnn": 1.0, "knn": 0.0, "physical": 0.0},
            },
            "confirmed_visual_recall": human,
            "knn_c": {
                "schema": "visionx.knn_c_live_review_gated.v1",
                "bank_status": "READY", "diagnostic_only": True,
                "auto_decision_authorized": False,
                "uses_aoi_category": False,
                "status": "REVISAO", "vote_A": label, "vote_B": label,
                "neighbors_A": a, "neighbors_B": b,
            },
        },
    }


def good_probe(trim=False):
    geom = {
        "method": "center_trim_no_rescale" if trim else "exact",
        "requires_review": trim,
        "reference_hw": [318, 570],
        "test_hw": [317, 567] if trim else [318, 570],
    }
    return {
        "status": "GEOMETRY_REVIEW" if trim else "NO_STRONG_EVIDENCE",
        "reason": "TWO_SCALE_ROI_CHECK", "geometry": geom,
        "scales": [
            {"suspected_absence": False, "focus_fraction": .58},
            {"suspected_absence": False, "focus_fraction": .76},
        ],
    }


class SideMemoryPriorityTests(unittest.TestCase):
    def decide(self, data=None, probe=None):
        # A sonda real trabalha sobre os pares da captura; aqui só isolamos
        # o contrato, sem fingir que esses PNGs existem no ambiente CI.
        return build_side_memory_first(
            case() if data is None else data, "FALTANDO", object(), object(),
            probe=lambda *_: good_probe() if probe is None else probe,
        )

    def test_oct30_false_ng_is_recalled_on_side_and_production_prefers_zero(self):
        original = case()
        before = deepcopy(original)
        final = self.decide(original)
        self.assertEqual(original, before)
        self.assertIsNotNone(final)
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertEqual(final["detail"]["side_memory_first"]["label"], "OK")
        self.assertEqual(
            final["detail"]["side_memory_first"]["archive_source"],
            "public/ok_archive/archive_FALTANDO.png",
        )
        self.assertEqual(final["detail"]["side_early_exit"]["rule"], RULE)
        self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"],
                         ["TOP", "MID"])
        self.assertEqual(memory_first_side_decision(final), "OK")
        policy = production_decision_policy(final)
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertTrue(policy["auto_allowed"])
        self.assertTrue(policy["unverified_physical_label"])
        self.assertEqual(policy["reason"], "side_operator_memory_first_96pct_auto_ok")

    def test_small_aoi_crop_mismatch_still_requires_two_scale_veto(self):
        self.assertEqual(memory_first_side_decision(
            self.decide(probe=good_probe(trim=True))), "OK")
        too_big = good_probe(trim=True)
        too_big["geometry"]["test_hw"] = [310, 564]
        self.assertIsNone(self.decide(probe=too_big))
        absent = good_probe(trim=True)
        absent["status"] = "SUSPECT_ABSENCE"
        absent["scales"][1]["suspected_absence"] = True
        self.assertIsNone(self.decide(probe=absent))
        unavailable = good_probe()
        unavailable["status"] = "UNAVAILABLE"
        self.assertIsNone(self.decide(probe=unavailable))

    def test_memory_can_reject_known_ng_without_using_cnn_confidence(self):
        ng = case(label="NG")
        final = self.decide(ng)
        self.assertIsNotNone(final)
        self.assertEqual(final["verdict"], "DEFEITO REAL")
        self.assertEqual(memory_first_side_decision(final), "NG")
        self.assertEqual(production_decision_policy(final)["proposed_decision"], "NG")

    def test_conflict_or_untrusted_or_weak_memory_falls_back(self):
        for name, mutate in (
            ("human_near_ng", lambda x: x["detail"]["confirmed_visual_recall"].__setitem__(
                "best_ng_similarity", .974)),
            ("human_not_confirmed", lambda x: x["detail"]["confirmed_visual_recall"][
                "best_ok_source"].__setitem__("source", "ARCHIVE_IMPORTED")),
            ("knn_not_ready", lambda x: x["detail"]["knn_c"].__setitem__(
                "bank_status", "LOADING")),
            ("weak_a", lambda x: x["detail"]["knn_c"]["neighbors_A"][0].__setitem__(
                "similarity", .9599)),
            ("weak_b", lambda x: x["detail"]["knn_c"]["neighbors_B"][0].__setitem__(
                "similarity", .9599)),
            ("ng_too_close", lambda x: x["detail"]["knn_c"]["neighbors_A"][1].__setitem__(
                "similarity", .955)),
            ("different_png", lambda x: x["detail"]["knn_c"]["neighbors_B"][0].__setitem__(
                "path", "public/ok_archive/another.png")),
            ("wrong_vote", lambda x: x["detail"]["knn_c"].__setitem__(
                "vote_B", "NG")),
            ("exact_conflict", lambda x: x["detail"].__setitem__(
                "memory_conflict", True)),
            ("physical_missing", lambda x: x["detail"].__setitem__(
                "hard_missing_evidence", True)),
            ("wrong_category", lambda x: x["detail"]["confirmed_visual_recall"].__setitem__(
                "category", "DESLOCADO")),
            ("wrong_light", lambda x: x.__setitem__("lighting_mode", "TOP")),
        ):
            data = case()
            mutate(data)
            with self.subTest(name=name):
                self.assertIsNone(self.decide(data))

    def test_all_production_markers_are_revalidated_before_xp(self):
        original = self.decide()
        for name, mutate in (
            ("source_id", lambda x: x["detail"]["side_memory_first"][
                "human_source"].__setitem__("record_id", "f" * 64)),
            ("png", lambda x: x["detail"]["side_memory_first"].__setitem__(
                "archive_source", "public/ok_archive/other.png")),
            ("score", lambda x: x["detail"]["confirmed_visual_recall"].__setitem__(
                "best_ok_similarity", .90)),
            ("review", lambda x: x.__setitem__("production_review_required", True)),
            ("rule", lambda x: x["detail"]["side_early_exit"].__setitem__(
                "rule", "fake")),
            ("probe", lambda x: x["detail"]["side_memory_first"][
                "physical_probe"]["scales"][0].__setitem__("suspected_absence", True)),
            ("trace", lambda x: x["detail"]["decision_trace"].__setitem__(
                "operator_review_required", True)),
        ):
            item = deepcopy(original)
            mutate(item)
            with self.subTest(name=name):
                self.assertIsNone(memory_first_side_decision(item))
                self.assertFalse(production_decision_policy(item)["auto_allowed"])

    def test_no_memory_never_overrides_cnn_without_proof(self):
        data = case()
        del data["detail"]["confirmed_visual_recall"]
        self.assertIsNone(self.decide(data))


if __name__ == "__main__":
    unittest.main()
