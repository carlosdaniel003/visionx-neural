"""Regressões da prioridade humana visual sem confundir similaridade com par exato."""
from __future__ import annotations

from copy import deepcopy
import unittest

from src.core.operator_visual_memory_consensus import (
    RULE, apply_operator_visual_memory_consensus, memory_ok_counterevidence,
    verified_operator_visual_memory_ok,
)
from src.ui.production_confidence_gate import production_decision_policy


LIGHTS = ("SIDE", "TOP", "MID")
CAT = "EMBORCADO"
SHA = "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c"


def _event():
    """Scores do evento AOI 2c2f436f... de 04/11/2026 (sem PNG industrial)."""
    human_ok = {"SIDE": .98464556, "TOP": .98073817, "MID": .96719095}
    human_ng = {"SIDE": .90181659, "TOP": .89982695, "MID": .88312874}
    archive_ok = {"SIDE": .984646, "TOP": .980738, "MID": .967191}
    archive_ng = {"SIDE": .901817, "TOP": .876938, "MID": .856974}
    cnn = {"SIDE": .999968886, "TOP": .949882507, "MID": .999832034}
    analyses = {}
    visual_lights = {}
    for light in LIGHTS:
        human = {
            "schema": "visionx.human_visual_recall.v1",
            "source": "PERSISTENT_HUMAN_LABELS",
            "diagnostic_only": True,
            "auto_ok_authorized": False,
            "category": CAT,
            "status": (
                "NO_MATCH" if light == "MID"
                else "HUMAN_OK_VISUAL_CANDIDATE"
            ),
            "reason": "VISUAL_SIMILARITY_NOT_EXACT_IDENTITY",
            "best_ok_similarity": human_ok[light],
            "best_ng_similarity": human_ng[light],
            "best_ok_source": {
                "source": "OPERATOR_CONFIRMED", "label": "OK",
                "lighting_mode": light, "record_id": light[0].lower() * 64,
            },
        }
        knn = {}
        if light == "SIDE":
            knn = {
                "schema": "visionx.knn_c_live_review_gated.v1",
                "bank_status": "READY", "status": "SUGERE_OK",
                "reason": "A_E_B_CONCORDAM", "diagnostic_only": True,
                "auto_decision_authorized": False,
                "vote_A": "OK", "vote_B": "OK",
                "suggestion_A": "SUGERE_OK", "suggestion_B": "SUGERE_OK",
                "neighbors_A": [
                    {"label": "OK", "similarity": x}
                    for x in (.96532639, .91104899, .89971381, .86382785, .85307751)
                ],
                "neighbors_B": [
                    {"label": "OK", "similarity": x}
                    for x in (.92206981, .87193952, .84984882, .82359927, .80697191)
                ],
            }
        analyses[light] = {
            "lighting_mode": light,
            "verdict": "DEFEITO REAL",
            "is_defect": True,
            "production_review_required": True,
            "detail": {
                "cnn_v2_lighting_mode": light,
                "cnn_v2_checkpoint_verified": True,
                "cnn_v2_checkpoint_sha256": SHA,
                "cnn_v2_status": "INFERENCE_OK",
                "cnn_v2_aoi_category": CAT,
                "cnn_v2_ng_score_uncalibrated": cnn[light],
                "recognition_route": "NEW_CNN",
                "confirmed_visual_recall": human,
                "knn_c": knn,
            },
        }
        visual_lights[light] = {
            "status": "NO_CONCLUSION", "cross_category_ng": False,
            "ng_support": 1 if light == "SIDE" else 0,
            "ok": {"best_similarity": archive_ok[light]},
            "ng": {"best_similarity": archive_ng[light]},
            "ng_margin_over_ok": round(archive_ng[light] - archive_ok[light], 6),
        }

    fused = {
        "verdict": "DEFEITO REAL",
        "is_defect": True,
        "lighting_mode": "MULTILIGHT",
        "multilight_final": True,
        "multilight_category": CAT,
        "production_review_required": False,
        "eligible_for_final_decision": True,
        "detail": {
            "multilight_final": True,
            "multilight_category": CAT,
            "eligible_for_final_decision": True,
            "production_continuous_auto": True,
            "production_binary_auto": True,
            "recognition_route": "NEW_CNN",
            "recognition_light_routes": {light: "NEW_CNN" for light in LIGHTS},
            "cnn_v2_consensus": "REVIEW",
            "cnn_v2_consensus_reason": "INVALID_LIGHT_EVIDENCE",
            "multilight_strong_positive_modes": list(LIGHTS),
            "hard_missing_evidence": False,
            "physical_score": 0.0,
            "decision_trace": {"hard_missing_evidence": False},
            "knn_visual_corroboration": {
                "schema": "visionx.knn_visual_corroboration.v1",
                "category": CAT, "status": "NO_CONCLUSION",
                "strong_ng_lights": [], "trend_ng_lights": [],
                "lights": visual_lights,
            },
        },
    }
    return analyses, fused


def _event_0943_actual():
    """Reproduz números publicados no debug ca2b12df... (sem frames RGB).

    Os records OK SIDE e TOP têm RECONCILED_OK e fonte humana = archive.
    A MID só favorece OK em busca aproximada com fonte de luz TOP; isso
    NÃO é par idêntico nem autorização de 0/XP.
    """
    analyses, fused = _event()
    fused["detail"]["multilight_strong_positive_modes"] = ["SIDE"]
    fused["detail"]["multilight_review_modes"] = list(LIGHTS)
    fused["detail"]["multilight_positive_modes"] = ["SIDE"]
    raw = {"SIDE": .9942189455032349, "TOP": .36052122712135315,
           "MID": .8710354566574097}
    human_ok = {"SIDE": .98948237, "TOP": .98555393, "MID": .95932525}
    human_ng = {"SIDE": .92038249, "TOP": .91687198, "MID": .87290055}
    archive_ok = {"SIDE": .989482, "TOP": .985554, "MID": .957441}
    archive_ng = {"SIDE": .920382, "TOP": .902279, "MID": .838078}
    for light in LIGHTS:
        local = analyses[light]["detail"]
        local["cnn_v2_ng_score_uncalibrated"] = raw[light]
        human = local["confirmed_visual_recall"]
        human["best_ok_similarity"] = human_ok[light]
        human["best_ng_similarity"] = human_ng[light]
        if light == "MID":
            human["best_ok_source"]["lighting_mode"] = "TOP"
        else:
            record = human["best_ok_source"]["record_id"]
            # Dois índices independentes: humano confirmado + PNG OK.
            local["knn_c"]["aligned_archive"] = {
                "schema": "visionx.knn_c_archive_alignment.v1",
                "status": "RECONCILED_OK",
                "reason": "HUMAN_AND_ARCHIVE_ALIGNED",
                "diagnostic_only": True,
                "auto_decision_authorized": False,
                "category": CAT, "label": "OK",
                "human_record_id": record,
                "archive_similarity": human_ok[light],
                "human_archive_similarity": 1.0,
                "best_opposite_archive_similarity": human_ng[light],
                "archive_source": {
                    "label": "OK", "lighting_mode": light,
                    "record_id": record, "source_sha256": "a"*64,
                    "path": (
                        "public/ok_archive/2026-11-04_0905_EMBORCADO_"
                        + light + ".png"
                    ),
                },
            }
        visual = fused["detail"]["knn_visual_corroboration"]["lights"][light]
        visual["ok"]["best_similarity"] = archive_ok[light]
        visual["ng"]["best_similarity"] = archive_ng[light]
        visual["ng_margin_over_ok"] = round(
            archive_ng[light] - archive_ok[light], 6
        )
        visual["ng_support"] = 0 if light == "MID" else 1

    # SIDE KNN C: cinco vizinhos de cada descritor são OK,
    # mas CNN acusa defeito 99,42% e consenso global é REVIEW.
    knn = analyses["SIDE"]["detail"]["knn_c"]
    knn["neighbors_A"] = [
        {"label": "OK", "similarity": x} for x in
        (.98246287, .97105543, .96610195, .95174236, .93018909)
    ]
    knn["neighbors_B"] = [
        {"label": "OK", "similarity": x} for x in
        (.943372, .90488756, .89996702, .89201255, .86289666)
    ]
    analyses["TOP"].update(verdict="REVISÃO OBRIGATÓRIA", is_defect=False)
    analyses["MID"].update(verdict="REVISÃO OBRIGATÓRIA", is_defect=False)
    fused["confidence"] = raw["SIDE"]
    fused["detail"]["final_score"] = raw["SIDE"]
    return analyses, fused


class OperatorVisualMemoryConsensusTests(unittest.TestCase):
    def test_real_emborcado_false_ng_recovers_multilight_ok_visual(self):
        analyses, fused = _event()
        proof = memory_ok_counterevidence(analyses, fused)
        self.assertEqual(proof["label"], "OK")
        self.assertFalse(proof["rgb_exact_pair"])
        self.assertFalse(proof["automatic_xp_authorized"])
        final = apply_operator_visual_memory_consensus(fused, analyses)
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertFalse(final["is_defect"])
        self.assertEqual(final["detail"]["fusion_rule"], RULE)
        self.assertEqual(final["detail"]["cnn_raw_verdict_before_memory"], "DEFEITO REAL")
        self.assertTrue(final["production_review_required"])
        self.assertEqual(fused["verdict"], "DEFEITO REAL")

    def test_same_visual_arbitration_in_all_three_modes(self):
        analyses, fused = _event()
        for mode in ("Modo Teste", "Modo Produção", "Modo Sombra"):
            with self.subTest(mode=mode):
                out = apply_operator_visual_memory_consensus(fused, analyses)
                self.assertEqual(out["verdict"], "FALHA FALSA")
                self.assertEqual(out["detail"]["fusion_rule"], RULE)

    def test_actual_event_0943_with_one_cnn_ng_light_and_cross_light_mid_ok(self):
        analyses, fused = _event_0943_actual()
        proof = memory_ok_counterevidence(analyses, fused)
        self.assertIsNotNone(proof)
        self.assertEqual(proof["evidence_mode"], "TWO_ALIGNED_ARCHIVE_ANCHORS")
        self.assertEqual(set(proof["aligned_anchor_lights"]), {"SIDE", "TOP"})
        self.assertEqual(proof["lights"]["MID"]["human_source_lighting"], "TOP")
        self.assertFalse(proof["rgb_exact_pair"])
        self.assertFalse(proof["automatic_xp_authorized"])
        out = apply_operator_visual_memory_consensus(fused, analyses)
        self.assertEqual(out["verdict"], "FALHA FALSA")
        self.assertFalse(out["is_defect"])
        self.assertEqual(out["detail"]["cnn_raw_verdict_before_memory"], "DEFEITO REAL")
        self.assertTrue(out["production_review_required"])
        self.assertEqual(out["detail"]["fusion_rule"], RULE)

    def test_real_fusion_then_knn_then_human_memory_arbitration(self):
        # Integra a fusão REAL do ODIN; não apenas dicionário já fundido.
        from unittest.mock import patch
        from src.core.multilight_fusion import fuse_multilight
        from src.core.knn_visual_corroboration import (
            apply_knn_visual_corroboration,
        )
        analyses, observed = _event_0943_actual()
        for light in LIGHTS:
            item = analyses[light]
            detail = item["detail"]
            score = detail["cnn_v2_ng_score_uncalibrated"]
            detail.update({
                "cnn_v2_active": True,
                "cnn_v2_experimental": True,
                "final_score": score,
                "physical_score": 0.0,
                "fusion_rule": "cnn_v2_dualscale_direct",
                "operator_review_required": True,
                "decision_trace": {
                    "fusion_rule": "cnn_v2_dualscale_direct",
                    "operator_review_required": True,
                    "physical_score": 0.0,
                    "final_score": score,
                },
            })
            item["confidence"] = max(score, 1.0 - score)
        fused = fuse_multilight(analyses, CAT)
        self.assertEqual(fused["verdict"], "DEFEITO REAL")
        self.assertEqual(
            fused["detail"]["fusion_rule"], "multilight_strong_single",
        )
        self.assertEqual(
            fused["detail"]["multilight_strong_positive_modes"], ["SIDE"],
        )
        self.assertEqual(fused["detail"]["cnn_v2_consensus"], "REVIEW")
        self.assertEqual(
            fused["detail"]["cnn_v2_consensus_reason"],
            "INVALID_LIGHT_EVIDENCE",
        )
        report = deepcopy(observed["detail"]["knn_visual_corroboration"])
        with patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=report,
        ):
            corroborated = apply_knn_visual_corroboration(fused, {}, CAT, None)
        self.assertEqual(corroborated["verdict"], "DEFEITO REAL")
        final = apply_operator_visual_memory_consensus(
            corroborated, analyses,
        )
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertEqual(final["detail"]["fusion_rule"], RULE)
        self.assertTrue(final["production_review_required"])
        self.assertFalse(
            final["detail"]["operator_visual_memory_consensus"]["automatic_xp_authorized"]
        )

    def test_visual_classification_verifies_the_same_proof_as_xp_gate(self):
        analyses, fused = _event_0943_actual()
        output = apply_operator_visual_memory_consensus(fused, analyses)
        self.assertTrue(verified_operator_visual_memory_ok(output))
        self.assertTrue(output["production_review_required"])
        marker = output["detail"]["operator_visual_memory_consensus"]

        # O marcador não pode transformar resultado de revisão em OK
        # pela simples presença da string FALHA FALSA.
        marker["aligned_anchor_lights"] = ["SIDE"]
        self.assertFalse(verified_operator_visual_memory_ok(output))
        policy = production_decision_policy(output)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operational_error"])

    def test_visual_ok_remains_no_auto_press_even_when_proof_is_verified(self):
        analyses, fused = _event_0943_actual()
        output = apply_operator_visual_memory_consensus(fused, analyses)
        self.assertTrue(verified_operator_visual_memory_ok(output))
        policy = production_decision_policy(output)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertTrue(policy["operator_review_required"])
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "")

    def test_actual_event_0943_production_fallback_requires_operator(self):
        analyses, fused = _event_0943_actual()
        corrected = apply_operator_visual_memory_consensus(fused, analyses)
        policy = production_decision_policy(corrected)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertEqual(
            policy["reason"], "operator_visual_memory_requires_confirmation_v1",
        )

    def test_actual_event_0943_needs_two_valid_aligned_anchors_and_mid_support(self):
        mutations = [
            lambda a, f: a["TOP"]["detail"]["knn_c"].pop("aligned_archive"),
            lambda a, f: a["TOP"]["detail"]["knn_c"]["aligned_archive"].update(
                human_record_id="b"*64),
            lambda a, f: a["SIDE"]["detail"]["knn_c"]["aligned_archive"][
                "archive_source"].update(label="NG"),
            lambda a, f: a["TOP"]["detail"]["knn_c"]["aligned_archive"].update(
                best_opposite_archive_similarity=.97),
            lambda a, f: a["MID"]["detail"]["confirmed_visual_recall"].update(
                best_ok_similarity=.9499),
            lambda a, f: a["MID"]["detail"]["confirmed_visual_recall"].update(
                best_ng_similarity=.94),
            lambda a, f: f["detail"]["knn_visual_corroboration"]["lights"][
                "MID"]["ok"].update(best_similarity=.949),
            lambda a, f: a["MID"]["detail"].update(hard_missing_evidence=True),
            lambda a, f: a["MID"]["detail"]["physical_missing_veto"].update(
                status="SUSPECT_ABSENCE"),
            lambda a, f: f["detail"]["knn_visual_corroboration"].update(
                trend_ng_lights=["TOP", "MID"]),
            lambda a, f: a["SIDE"]["detail"]["knn_c"].update(
                status="REVISAO"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=str(mutate)):
                analyses, fused = _event_0943_actual()
                analyses["MID"]["detail"]["physical_missing_veto"] = {}
                mutate(analyses, fused)
                self.assertIsNone(memory_ok_counterevidence(analyses, fused))
                self.assertEqual(
                    apply_operator_visual_memory_consensus(fused, analyses)["verdict"],
                    "DEFEITO REAL",
                )

    def test_debug_reports_why_a_previously_seen_ok_was_not_recovered(self):
        from src.services.capture_debug_payload import decision_record
        analyses, fused = _event_0943_actual()
        analyses["TOP"]["detail"]["knn_c"].pop("aligned_archive")
        out = apply_operator_visual_memory_consensus(fused, analyses)
        self.assertEqual(out["verdict"], "DEFEITO REAL")
        attempt = out["detail"]["operator_visual_memory_attempt"]
        self.assertEqual(
            attempt["reason"], "INSUFFICIENT_OPERATOR_ARCHIVE_ANCHORS_OR_SUPPORT",
        )
        self.assertEqual(attempt["aligned_anchor_lights"], ["SIDE"])
        self.assertEqual(set(attempt["support_lights"]), {"SIDE", "MID"})
        exported = decision_record(out, {"category": CAT})
        self.assertEqual(
            exported["operator_visual_memory_attempt"]["reason"],
            "INSUFFICIENT_OPERATOR_ARCHIVE_ANCHORS_OR_SUPPORT",
        )

    def test_debug_records_verified_anchors_without_xp_authorization(self):
        from src.services.capture_debug_payload import decision_record
        analyses, fused = _event_0943_actual()
        recovered = apply_operator_visual_memory_consensus(fused, analyses)
        exported = decision_record(recovered, {"category": CAT})
        proof = exported["operator_visual_memory_consensus"]
        self.assertEqual(proof["evidence_mode"], "TWO_ALIGNED_ARCHIVE_ANCHORS")
        self.assertEqual(set(proof["aligned_anchor_lights"]), {"SIDE", "TOP"})
        self.assertFalse(proof["automatic_xp_authorized"])
        self.assertEqual(exported["verdict"], "FALHA FALSA")

    def test_conflicting_or_incomplete_memory_must_not_override_ng(self):
        cases = [
            lambda a, f: a["TOP"]["detail"]["confirmed_visual_recall"].update(
                best_ok_similarity=.95),
            lambda a, f: a["MID"]["detail"]["confirmed_visual_recall"].update(
                best_ng_similarity=.945),
            lambda a, f: a["SIDE"]["detail"]["confirmed_visual_recall"].update(
                best_ok_source={"source": "ARCHIVE_PNG", "label": "OK",
                                "lighting_mode": "SIDE", "record_id": "s"*64}),
            lambda a, f: a["TOP"]["detail"]["confirmed_visual_recall"].update(
                status="HUMAN_NG_VISUAL_CANDIDATE"),
            lambda a, f: a["MID"]["detail"].update(cnn_v2_checkpoint_sha256="b"*64),
            lambda a, f: a["MID"]["detail"]["confirmed_visual_recall"]["best_ok_source"].update(
                lighting_mode="SIDE"),
            lambda a, f: f["detail"].update(hard_missing_evidence=True),
            lambda a, f: f["detail"]["decision_trace"].update(
                raw_hard_missing_evidence=True),
            lambda a, f: f["detail"]["knn_visual_corroboration"]["lights"]["TOP"].update(
                ng_margin_over_ok=.03),
            lambda a, f: f["detail"]["knn_visual_corroboration"].update(
                strong_ng_lights=["MID"]),
            lambda a, f: a["SIDE"]["detail"]["knn_c"].update(vote_A="NG"),
            lambda a, f: f.update(multilight_category="UNKNOWN"),
        ]
        for mutation in cases:
            with self.subTest(mutation=str(mutation)):
                analyses, fused = _event()
                mutation(analyses, fused)
                self.assertIsNone(memory_ok_counterevidence(analyses, fused))
                self.assertEqual(
                    apply_operator_visual_memory_consensus(fused, analyses)["verdict"],
                    "DEFEITO REAL",
                )

    def test_production_binary_fallback_cannot_turn_human_ok_into_ng(self):
        analyses, fused = _event()
        recovered = apply_operator_visual_memory_consensus(fused, analyses)
        policy = production_decision_policy(recovered)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertEqual(policy["proposed_decision"], "")
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(
            policy["reason"], "operator_visual_memory_requires_confirmation_v1",
        )

    def test_invalid_counterevidence_marker_stops_without_sending_xp(self):
        analyses, fused = _event()
        recovered = apply_operator_visual_memory_consensus(fused, analyses)
        recovered["detail"]["operator_visual_memory_consensus"]["lights"].pop("MID")
        policy = production_decision_policy(recovered)
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertTrue(policy["operational_error"])

    def test_unsupervised_reject_remains_binary_ng(self):
        _, fused = _event()
        policy = production_decision_policy(fused)
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "NG")


if __name__ == "__main__":
    unittest.main()
