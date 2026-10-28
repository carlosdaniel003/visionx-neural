"""Regression: 26 Oct 2026 physically OK FALTANDO rejected by all 3 CNN lights.

Approximate KNN A/B + visual signatures can veto an uncorroborated
CNN NG presentation, but may NEVER automatically release the PCB as OK.
"""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from src.core.knn_visual_corroboration import (
    _three_light_ok_counterevidence,
    apply_knn_visual_corroboration,
    SCHEMA,
)
from src.core.multilight_fusion import fuse_multilight
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.decision_verdict_feedback import verdict_feedback_state

LIGHTS = ("SIDE", "TOP", "MID")
GOOD = {
    "SIDE": (.951006, .944863),
    "TOP": (.951006, .938456),
    "MID": (.954742, .922532),
}


def raw_c(light):
    return {
        "schema": "visionx.knn_c_live_review_gated.v1",
        "bank_status": "READY",
        "status": "REVISAO",
        "reason": "SEM_SUGESTAO_CONFIAVEL",
        "diagnostic_only": True, "auto_decision_authorized": False,
        "vote_A": "OK", "vote_B": "OK",
        "suggestion_A": "REVISAO", "suggestion_B": "REVISAO",
        "similarity_A": .86, "similarity_B": .80,
    }


def report():
    return {
        "schema": SCHEMA, "status": "CONFLICT", "lights": {
            light: {
                "status": "CONFLICT",
                "cross_category_ng": False,
                "ng_margin_over_ok": round(ng-ok, 6),
                "ok": {"best_similarity": ok},
                "ng": {"best_similarity": ng},
            }
            for light, (ok, ng) in GOOD.items()
        },
        "strong_ng_lights": [],
    }


def fused():
    return {
        "verdict": "DEFEITO REAL", "is_defect": True,
        "multilight_final": True, "eligible_for_final_decision": True,
        "production_review_required": False,
        "confidence": .9999463558197021,
        "detail": {
            "recognition_route": "NEW_CNN",
            "multilight_category": "FALTANDO",
            "cnn_v2_active": True, "cnn_v2_experimental": True,
            "cnn_v2_trial_manual_only": True,
            "cnn_v2_consensus": "REVIEW",
            "cnn_v2_consensus_reason": "INVALID_LIGHT_EVIDENCE",
            "multilight_strong_positive_modes": list(LIGHTS),
            "multilight_physical_disagreement": False,
            "multilight_missing_verified_presence": False,
            "hard_missing_evidence": False, "physical_score": 0.,
            "physical_missing_veto": {"status": ""},
            "knn_c_light_observer_votes": {
                light: raw_c(light) for light in LIGHTS
            },
            "decision_trace": {
                "verdict": "DEFEITO REAL",
                "hard_missing_evidence": False,
                "raw_hard_missing_evidence": False,
            },
        },
    }


class KNNOKCounterevidenceTests(unittest.TestCase):
    def test_oct26_real_example_becomes_review_not_auto_ok_or_ng(self):
        candidate = fused()
        with patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=report(),
        ):
            result = apply_knn_visual_corroboration(
                candidate, {}, "FALTANDO", object(),
            )
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertIs(result["is_defect"], False)
        self.assertTrue(result["production_review_required"])
        self.assertFalse(result["eligible_for_final_decision"])
        self.assertEqual(result["confidence"], .5)
        self.assertEqual(result["detail"]["cnn_ng_verdict_before_knn_review"],
                         "DEFEITO REAL")
        self.assertEqual(result["detail"]["knn_ok_counterevidence"]["status"],
                         "OPERATOR_REVIEW_KNN_OK_COUNTEREVIDENCE")
        self.assertTrue(result["detail"]["knn_ok_counterevidence"]["no_exact_pair_claim"])
        self.assertFalse(result["detail"]["knn_ok_counterevidence"]["automatic_ok_authorized"])
        self.assertEqual(result["detail"]["decision_trace"]["verdict"],
                         "REVISÃO OBRIGATÓRIA")
        policy = production_decision_policy(result)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "")

    def test_without_three_distinct_knn_c_light_votes_does_not_change_ng(self):
        for missing in LIGHTS:
            with self.subTest(missing=missing):
                f = fused()
                f["detail"]["knn_c_light_observer_votes"].pop(missing)
                self.assertIsNone(_three_light_ok_counterevidence(f, report(), "FALTANDO"))

    def test_weak_or_conflicting_votes_never_obtain_ok_review_guard(self):
        variants = (
            ("vote_A", "NG"), ("vote_B", "NG"),
            ("suggestion_A", "SUGERE_OK"), ("suggestion_B", "SUGERE_OK"),
            ("status", "SUGERE_OK"), ("bank_status", "LOADING"),
            ("diagnostic_only", False), ("auto_decision_authorized", True),
            ("reason", "MEMORIA_VISUAL_EM_PREPARACAO"),
            ("schema", "untrusted"),
        )
        for key, value in variants:
            with self.subTest(key=key):
                f = fused()
                f["detail"]["knn_c_light_observer_votes"]["TOP"][key] = value
                self.assertIsNone(_three_light_ok_counterevidence(f, report(), "FALTANDO"))

    def test_no_automatic_ok_from_approximate_or_missing_visual_neighbors(self):
        variants = (
            ("LOW_OK", lambda r: r["lights"]["TOP"]["ok"].update(best_similarity=.949)),
            ("NG_BETTER", lambda r: r["lights"]["SIDE"]["ng"].update(best_similarity=.96)),
            ("MISSING_LIGHT", lambda r: r["lights"].pop("MID")),
            ("NO_CONCLUSION", lambda r: r["lights"]["TOP"].update(status="NO_CONCLUSION")),
            ("NG_CORROBORATED", lambda r: r.update(status="NG_CORROBORATED")),
            ("CROSS_CATEGORY", lambda r: r["lights"]["MID"].update(cross_category_ng=True)),
            ("FAKE_SCHEMA", lambda r: r.update(schema="unknown")),
            ("NO_OK", lambda r: r["lights"]["MID"]["ok"].update(best_similarity=None)),
        )
        for name, mutation in variants:
            with self.subTest(name=name):
                r = report()
                mutation(r)
                self.assertIsNone(_three_light_ok_counterevidence(fused(), r, "FALTANDO"))

    def test_genuine_physical_absence_and_valid_cnn_consensus_remain_ng(self):
        originals = (
            ("hard_missing_evidence", True),
            ("physical_score", .8),
            ("cnn_v2_consensus", "NG"),
            ("cnn_v2_consensus_reason", "FULL_THREE_LIGHT_VOTES"),
        )
        for key, value in originals:
            with self.subTest(key=key):
                f = fused()
                f["detail"][key] = value
                self.assertIsNone(_three_light_ok_counterevidence(f, report(), "FALTANDO"))
        f = fused()
        f["detail"]["decision_trace"]["raw_hard_missing_evidence"] = True
        self.assertIsNone(_three_light_ok_counterevidence(f, report(), "FALTANDO"))
        for category in ("INVERTIDO", "EMBORCADO", "DESLOCADO", "MUITO ADESIVO"):
            self.assertIsNone(_three_light_ok_counterevidence(fused(), report(), category))

    def test_without_counterevidence_original_ng_preserved(self):
        f = fused()
        f["detail"]["knn_c_light_observer_votes"]["MID"]["vote_B"] = "NG"
        with patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=report(),
        ):
            result = apply_knn_visual_corroboration(
                f, {}, "FALTANDO", object(),
            )
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertIs(result["is_defect"], True)
        self.assertNotIn("knn_ok_counterevidence", result["detail"])

    def test_operator_confirmed_known_ok_appears_as_false_failure_not_auto(self):
        # Caso real 26/10 14:56:58; SHA vem da memória OK da estação.
        pin = (
            "public/ok_archive/2026-10-21_1412_FALTANDO.png",
            "c9dec1d85617aa074e71b8b3640cace88f3a23c123185f7be84d6c06e2946bd9",
        )
        f = fused()
        f["detail"]["knn_c_light_neighbor_evidence"] = {
            light: {"neighbors_A": [], "neighbors_B": []}
            for light in LIGHTS
        }
        f["detail"]["knn_c_light_neighbor_evidence"]["SIDE"] = {
            "neighbors_A": [
                {"path": pin[0], "source_pixel_sha256": pin[1],
                 "label": "OK", "similarity": .99325113},
                {"path": "public/ng_archive/ng.png", "label": "NG",
                 "source_pixel_sha256": "b" * 64, "similarity": .85977105},
            ],
            "neighbors_B": [
                {"path": pin[0], "source_pixel_sha256": pin[1],
                 "label": "OK", "similarity": .97756581},
                {"path": "public/ng_archive/ng.png", "label": "NG",
                 "source_pixel_sha256": "b" * 64, "similarity": .82187452},
            ],
        }
        self.assertEqual(f["verdict"], "DEFEITO REAL")
        modified_report = report()
        modified_report["lights"]["SIDE"] = {
            "status": "CONFLICT", "cross_category_ng": False,
            "ng_margin_over_ok": -.045818,
            "ok": {"best_similarity": .995819},
            "ng": {"best_similarity": .950001},
        }
        with patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=modified_report,
        ):
            result = apply_knn_visual_corroboration(f, {}, "FALTANDO", object())
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertIs(result["is_defect"], False)
        self.assertTrue(result["production_review_required"])
        self.assertFalse(result["eligible_for_final_decision"])
        self.assertEqual(
            result["detail"]["recognized_recurrent_ok"]["source_pixel_sha256"],
            pin[1],
        )
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "operator_confirmed_recurrent_ok_three_light_review_v1",
        )
        policy = production_decision_policy(result)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "")

        # Fonte OK falsificada ou muito menos semelhante não ganha status OK.
        for name, changes in (
            ("untrusted_sha", {"neighbors_A": {"source_pixel_sha256": "d" * 64}}),
            ("low_A", {"neighbors_A": {"similarity": .9899}}),
            ("low_B", {"neighbors_B": {"similarity": .9749}}),
            ("ng_near", {"ng": {"similarity": .900001}}),
        ):
            with self.subTest(name=name):
                other = fused()
                other["detail"]["knn_c_light_neighbor_evidence"] = deepcopy(
                    f["detail"]["knn_c_light_neighbor_evidence"]
                )
                side = other["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
                if "ng" in changes:
                    side["neighbors_A"][1].update(changes["ng"])
                else:
                    key = next(iter(changes))
                    side[key][0].update(changes[key])
                with patch(
                    "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
                    return_value=modified_report,
                ):
                    downgraded = apply_knn_visual_corroboration(
                        other, {}, "FALTANDO", object(),
                    )
                self.assertEqual(downgraded["verdict"], "REVISÃO OBRIGATÓRIA")
                self.assertNotIn("recognized_recurrent_ok", downgraded["detail"])
                self.assertFalse(production_decision_policy(downgraded)["auto_allowed"])

        # Mesmo modelo reconhecido não atravessa veto físico.
        stronger = fused()
        stronger["detail"]["physical_score"] = .75
        stronger["detail"]["knn_c_light_neighbor_evidence"] = deepcopy(
            f["detail"]["knn_c_light_neighbor_evidence"]
        )
        with patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=modified_report,
        ):
            blocked = apply_knn_visual_corroboration(
                stronger, {}, "FALTANDO", object(),
            )
        self.assertEqual(blocked["verdict"], "DEFEITO REAL")

    def test_oct27_same_confirmed_ok_family_with_ng_first_still_false_failure(self):
        # Evento c3af7000a5b147bca4c03a695abc6675. Primeiros vizinhos
        # NG, mas a mesma referência OK humana aparece em A/B no top 3.
        # Resultado precisa ser FALHA FALSA, apenas para o operador.
        pin = (
            "public/ok_archive/2026-10-21_1412_FALTANDO.png",
            "c9dec1d85617aa074e71b8b3640cace88f3a23c123185f7be84d6c06e2946bd9",
        )
        f = fused()
        f["detail"]["knn_c_light_neighbor_evidence"] = {
            light: {"neighbors_A": [], "neighbors_B": []}
            for light in LIGHTS
        }
        f["detail"]["knn_c_light_neighbor_evidence"]["SIDE"] = {
            "neighbors_A": [
                {"path": "public/ng_archive/2026-10-21_1432_FALTANDO_SIDE.png",
                 "label": "NG", "similarity": .88679835},
                {"path": pin[0], "source_pixel_sha256": pin[1],
                 "label": "OK", "similarity": .88399081},
                {"path": "public/ng_archive/2026-10-21_1458_FALTANDO_3.png",
                 "label": "NG", "similarity": .88335799},
                {"path": "public/ok_archive/2026-10-21_1629_FALTANDO_SIDE.png",
                 "label": "OK", "similarity": .86559447},
                {"path": "public/ok_archive/2026-10-21_1436_FALTANDO_SIDE.png",
                 "label": "OK", "similarity": .86534112},
            ],
            "neighbors_B": [
                {"path": "public/ng_archive/2026-10-21_1458_FALTANDO_3.png",
                 "label": "NG", "similarity": .83597105},
                {"path": "public/ng_archive/2026-10-21_1432_FALTANDO_SIDE.png",
                 "label": "NG", "similarity": .82735666},
                {"path": pin[0], "source_pixel_sha256": pin[1],
                 "label": "OK", "similarity": .82105397},
                {"path": "public/ok_archive/2026-10-21_1436_FALTANDO_SIDE.png",
                 "label": "OK", "similarity": .80787613},
                {"path": "public/ok_archive/2026-10-21_1629_FALTANDO_SIDE.png",
                 "label": "OK", "similarity": .80472034},
            ],
        }
        actual = report()
        for light, ok, ng in (
            ("SIDE", .953651, .948180),
            ("TOP", .953651, .942180),
            ("MID", .952869, .927543),
        ):
            actual["lights"][light].update({
                "ng_margin_over_ok": round(ng - ok, 6),
                "ok": {"best_similarity": ok},
                "ng": {"best_similarity": ng},
            })
        with patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=actual,
        ):
            result = apply_knn_visual_corroboration(
                deepcopy(f), {}, "FALTANDO", object(),
            )
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertFalse(result["is_defect"])
        self.assertTrue(result["production_review_required"])
        self.assertFalse(result["eligible_for_final_decision"])
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "operator_confirmed_ok_pattern_three_light_review_v1",
        )
        self.assertFalse(
            result["detail"]["operator_confirmed_ok_pattern_review"]
            ["current_capture_operator_confirmed"]
        )
        policy = production_decision_policy(result)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertEqual(
            policy["reason"], "operator_confirmed_ok_pattern_requires_operator",
        )
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertEqual(verdict_feedback_state(result), ("FALHA FALSA", "ok"))

        # Defeitos reais, memória adulterada, margem insuficiente
        # ou ausência de um descritor não podem usar a exceção.
        for name, mutation in (
            ("sha", lambda x, y:
             x["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
             ["neighbors_A"][1].update(source_pixel_sha256="f" * 64)),
            ("weak_visual", lambda x, y:
             x["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
             ["neighbors_A"][1].update(similarity=.8799)),
            ("weak_geometry", lambda x, y:
             x["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
             ["neighbors_B"][2].update(similarity=.8199)),
            ("dangerous_ng", lambda x, y:
             x["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
             ["neighbors_A"][0].update(similarity=.901)),
            ("missing_light", lambda x, y: 
             x["detail"]["knn_c_light_observer_votes"].pop("TOP")),
            ("physical_ng", lambda x, y:
             x["detail"].update(physical_score=.7)),
            ("cnn_ng_consensus", lambda x, y:
             x["detail"].update(cnn_v2_consensus="NG")),
            ("poor_margin", lambda x, y:
             y["lights"]["TOP"].update(
                 ng_margin_over_ok=-.0001,
                 ok={"best_similarity": .953651},
                 ng={"best_similarity": .953551},
             )),
        ):
            with self.subTest(name=name):
                f2, r2 = deepcopy(f), deepcopy(actual)
                mutation(f2, r2)
                with patch(
                    "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
                    return_value=r2,
                ):
                    blocked = apply_knn_visual_corroboration(
                        f2, {}, "FALTANDO", object(),
                    )
                self.assertNotEqual(blocked["verdict"], "FALHA FALSA")
                self.assertFalse(production_decision_policy(blocked)["auto_allowed"])

        # Sem verificar o registro original, não confiar só no flag da fusão.
        altered = deepcopy(result)
        altered["detail"]["operator_confirmed_ok_pattern_review"]["source_pixel_sha256"] = "0" * 64
        self.assertEqual(
            production_decision_policy(altered)["verdict"], "REVISÃO OBRIGATÓRIA"
        )
        self.assertEqual(
            verdict_feedback_state(altered), ("REVISÃO OBRIGATÓRIA", "ng"),
        )

    def test_oct28_recurrent_ok_strong_top_mid_and_mid_sugere_ok(self):
        # Evento 0077f20173f343e6b1ea3427e4a3d92b.
        # A fonte OK existente foi recuperada em TOP e MID pelas KNN A/B;
        # MID retornou SUGERE_OK, por isso o gate anterior não executava.
        f = fused()
        f["detail"]["knn_c_light_observer_votes"]["MID"].update({
            "status": "SUGERE_OK", "reason": "A_E_B_CONCORDAM",
            "suggestion_A": "SUGERE_OK", "suggestion_B": "SUGERE_OK",
        })
        f["detail"]["knn_c_light_neighbor_evidence"] = {
            "SIDE": {"neighbors_A": [], "neighbors_B": []},
            "TOP": {
                "neighbors_A": [
                    {"label": "OK", "similarity": .99245913,
                     "path": "public/ok_archive/2026-10-27_1318_FALTANDO_TOP.png",
                     "source_pixel_sha256": "046effa074997ca1211b45e905745f2f6ec68263a0613854e4130eb7168e7d5d"},
                    {"label": "NG", "similarity": .90891847},
                ],
                "neighbors_B": [
                    {"label": "OK", "similarity": .98483213,
                     "path": "public/ok_archive/2026-10-27_1318_FALTANDO_TOP.png",
                     "source_pixel_sha256": "046effa074997ca1211b45e905745f2f6ec68263a0613854e4130eb7168e7d5d"},
                    {"label": "NG", "similarity": .87908035},
                ],
            },
            "MID": {
                "neighbors_A": [
                    {"label": "OK", "similarity": .97646753,
                     "path": "public/ok_archive/2026-10-27_1318_FALTANDO_MID.png",
                     "source_pixel_sha256": "338f1cc612656297c21db284ea097489b90eb6e0537c3231dde45740f6a16e29"},
                ],
                "neighbors_B": [
                    {"label": "OK", "similarity": .94535756,
                     "path": "public/ok_archive/2026-10-27_1318_FALTANDO_MID.png",
                     "source_pixel_sha256": "338f1cc612656297c21db284ea097489b90eb6e0537c3231dde45740f6a16e29"},
                ],
            },
        }
        actual = report()
        actual["strong_ng_lights"] = []
        actual["trend_ng_lights"] = []
        for light, ok, ng, status in (
            ("SIDE", .949029, .949006, "CONFLICT"),
            ("TOP", .995616, .942122, "CONFLICT"),
            ("MID", .996298, .924989, "NO_CONCLUSION"),
        ):
            actual["lights"][light].update({
                "status": status,
                "ok": {"best_similarity": ok},
                "ng": {"best_similarity": ng},
                "ng_margin_over_ok": round(ng - ok, 6),
            })
        with patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=deepcopy(actual),
        ):
            result = apply_knn_visual_corroboration(
                deepcopy(f), {}, "FALTANDO", object(),
            )
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertIs(result["is_defect"], False)
        self.assertTrue(result["production_review_required"])
        self.assertFalse(result["eligible_for_final_decision"])
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "operator_recurrent_ok_top_mid_review_v1",
        )
        memory = result["detail"]["recognized_two_light_operator_ok"]
        self.assertFalse(memory["current_pair_exact_match"])
        self.assertFalse(memory["automatic_ok_authorized"])
        self.assertFalse(memory["xp_command_authorized"])
        policy = production_decision_policy(result)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertEqual(
            policy["reason"], "operator_recurrent_ok_top_mid_requires_operator",
        )
        self.assertEqual(verdict_feedback_state(result), ("FALHA FALSA", "ok"))

        # A comparação é restrita a esse episódio e precisa falhar
        # diante de fonte falsificada, NG forte ou ausência física.
        cases = (
            ("sha_wrong", lambda x, r:
             x["detail"]["knn_c_light_neighbor_evidence"]["TOP"]
             ["neighbors_A"][0].update(source_pixel_sha256="0" * 64)),
            ("different_event", lambda x, r:
             x["detail"]["knn_c_light_neighbor_evidence"]["MID"]
             ["neighbors_B"][0].update(path="public/ok_archive/other.png")),
            ("mid_ng", lambda x, r:
             x["detail"]["knn_c_light_observer_votes"]["MID"].update(vote_B="NG")),
            ("top_ng", lambda x, r:
             x["detail"]["knn_c_light_observer_votes"]["TOP"].update(vote_A="NG")),
            ("no_mid", lambda x, r:
             x["detail"]["knn_c_light_observer_votes"].pop("MID")),
            ("physical", lambda x, r: x["detail"].update(physical_score=.7)),
            ("hard_missing", lambda x, r:
             x["detail"]["decision_trace"].update(raw_hard_missing_evidence=True)),
            ("cnn_consensus_ng", lambda x, r:
             x["detail"].update(cnn_v2_consensus="NG")),
            ("strong_ng", lambda x, r: r.update(strong_ng_lights=["SIDE"])),
            ("side_ng", lambda x, r:
             r["lights"]["SIDE"].update(
                 ok={"best_similarity": .949029},
                 ng={"best_similarity": .974029},
                 ng_margin_over_ok=.025)),
            ("top_no_ok", lambda x, r:
             r["lights"]["TOP"].update(
                 ok={"best_similarity": .949},
                 ng={"best_similarity": .942122},
                 ng_margin_over_ok=-.006878)),
        )
        for name, change in cases:
            with self.subTest(case=name):
                corrupted, modified = deepcopy(f), deepcopy(actual)
                change(corrupted, modified)
                with patch(
                    "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
                    return_value=modified,
                ):
                    after = apply_knn_visual_corroboration(
                        corrupted, {}, "FALTANDO", object(),
                    )
                self.assertNotEqual(after["verdict"], "FALHA FALSA")
                self.assertFalse(production_decision_policy(after)["auto_allowed"])

        # Metadados forjados não podem liberar uma classificação.
        forged = deepcopy(result)
        forged["detail"]["recognized_two_light_operator_ok"][
            "per_light_reference"]["TOP"]["source_pixel_sha256"] = "f" * 64
        self.assertEqual(
            production_decision_policy(forged)["verdict"], "REVISÃO OBRIGATÓRIA",
        )
        self.assertEqual(
            verdict_feedback_state(forged), ("REVISÃO OBRIGATÓRIA", "ng"),
        )

    def test_multilight_fusion_retains_distinct_per_light_c_votes_for_arbitration(self):
        def local(light):
            return {
                "verdict": "DEFEITO REAL", "is_defect": True,
                "confidence": .999, "lighting_mode": light,
                "production_review_required": True,
                "detail": {
                    "recognition_route": "NEW_CNN",
                    "cnn_v2_active": True, "cnn_v2_experimental": True,
                    "cnn_v2_aoi_category": "FALTANDO",
                    "cnn_v2_lighting_mode": light,
                    "cnn_v2_status": "INFERENCE_OK",
                    "cnn_v2_checkpoint_sha256": "c" * 64,
                    "cnn_v2_checkpoint_verified": True,
                    "cnn_v2_ng_score_uncalibrated": .999,
                    "fusion_rule": "cnn_v2_dualscale_direct",
                    "operator_review_required": True,
                    "knn_c": raw_c(light),
                    "decision_trace": {
                        "fusion_rule": "cnn_v2_dualscale_direct",
                        "operator_review_required": True,
                    },
                }
            }
        raw = {light: local(light) for light in LIGHTS}
        baseline = deepcopy(raw)
        f = fuse_multilight(raw, "FALTANDO")
        self.assertEqual(
            set(f["detail"]["knn_c_light_observer_votes"]), set(LIGHTS),
        )
        self.assertEqual(
            f["detail"]["knn_c_light_observer_votes"]["SIDE"]["vote_A"], "OK",
        )
        self.assertEqual(raw, baseline)


if __name__ == "__main__":
    unittest.main()
