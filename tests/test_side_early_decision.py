"""Regressões do encerramento antecipado SIDE >= 97%."""

import unittest
from copy import deepcopy

from src.core.neural.aoi_v3_epoch5_contract import (
    EPOCH5_VOTE_SCHEMA, SNAPSHOT_SHA256 as V3_EPOCH5_SHA256,
)
from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
from src.core.side_early_decision import (
    SIDE_EARLY_EXIT_THRESHOLD,
    SIDE_TRIAL_NG_REVIEW_RULE,
    SIDE_TRIAL_OK_REVIEW_RULE,
    SIDE_V3_SUPERVISED_RULE,
    build_side_early_decision,
    side_automatic_decision,
    side_trial_automatic_decision,
)


def _side(verdict="FALHA FALSA", confidence=0.97, **detail):
    ng = verdict == "DEFEITO REAL"
    return {
        "verdict": verdict,
        "is_defect": ng,
        "confidence": confidence,
        "production_review_required": False,
        "lighting_mode": "SIDE",
        "reason": "Resultado local",
        "detail": {
            "recognition_route": "NEW_EXPERTS",
            "decision_trace": {"operator_review_required": False},
            **detail,
        },
    }


def _v3_side(*, verdict="FALHA FALSA", score=None):
    """SIDE V3 época 5 com KNN C concordante em dois descritores."""
    defect = verdict == "DEFEITO REAL"
    if score is None:
        score = .9999 if defect else 4.6078973969798653e-10
    label = "NG" if defect else "OK"
    selected = "SUGERE_" + label
    vote = {
        "schema": EPOCH5_VOTE_SCHEMA,
        "status": "READY",
        "epoch": 5,
        "light": "SIDE",
        "checkpoint_sha256": V3_EPOCH5_SHA256,
        "ng_score_uncalibrated": score,
        "predicted": label,
        "ng_veto": defect,
        "automatic_ok_from_v3_authorized": False,
    }
    knn = {
        "schema": "visionx.knn_c_live_review_gated.v1",
        "bank_status": "READY",
        "status": selected,
        "diagnostic_only": True,
        "auto_decision_authorized": False,
        "requires_human_confirmation": False,
        "conflicts_with_existing_verdict": False,
        "vote_A": label, "vote_B": label,
        "suggestion_A": selected, "suggestion_B": selected,
        "pre_c_verdict": verdict,
        "pre_c_is_defect": defect,
        "pre_c_review_required": False,
        "neighbors_A": [
            {"label": label, "similarity": v}
            for v in (.96348, .95507, .94240, .94151, .94133)
        ],
        "neighbors_B": [
            {"label": label, "similarity": v}
            for v in (.93968, .92795, .91188, .90846, .90620)
        ],
    }
    side = _side(
        verdict, max(score, 1.0-score),
        recognition_route="NEW_CNN",
        cnn_v3_primary=True,
        cnn_v3_status="INFERENCE_OK",
        cnn_v3_checkpoint_verified=True,
        cnn_v3_checkpoint_epoch=5,
        cnn_v3_checkpoint_sha256=V3_EPOCH5_SHA256,
        cnn_v3_lighting_mode="SIDE",
        cnn_v3_ng_score_uncalibrated=score,
        fusion_rule="cnn_v3_primary_direct",
        aoi_v3_trial=vote,
        physical_score=0.0,
        knn_c=knn,
        operator_review_required=False,
        decision_trace={
            "operator_review_required": False,
            "fusion_rule": "cnn_v3_primary_direct",
        },
    )
    side["production_review_required"] = False
    return side


def _v3_weak_c_human_ok_side(*, category="DESLOCADO"):
    """SIDE forte, KNN A/B OK não seletiva e memória humana qualificada."""
    side = _v3_side(score=0.0001229936460731551)
    detail = side["detail"]
    knn = detail["knn_c"]
    knn.update({
        "status": "REVISAO",
        "reason": "SEM_SUGESTAO_CONFIAVEL",
        "reason_A": "SIMILARIDADE_VISUAL_INSUFICIENTE",
        "reason_B": "SIMILARIDADE_VISUAL_INSUFICIENTE",
        "suggestion_A": "REVISAO",
        "suggestion_B": "REVISAO",
        "uses_aoi_category": False,
    })
    for key in ("requires_human_confirmation", "pre_c_review_required",
                "pre_c_verdict", "pre_c_is_defect"):
        knn.pop(key, None)
    for key, scores in (
        ("neighbors_A", (.7888, .7619, .7534, .7523, .7478)),
        ("neighbors_B", (.8126, .7939, .7756, .7716, .7703)),
    ):
        knn[key] = [
            {"label": "OK", "similarity": score,
             "source_pixel_sha256": format(i + 1, "064x")}
            for i, score in enumerate(scores)
        ]
    from src.core.strict_category_memory import canonical_memory_category
    detail["confirmed_visual_recall"] = {
        "schema": "visionx.human_visual_recall.v1",
        "status": "HUMAN_OK_VISUAL_CANDIDATE",
        "source": "PERSISTENT_HUMAN_LABELS",
        "category": canonical_memory_category(category),
        "diagnostic_only": True,
        "auto_ok_authorized": False,
        "cross_light_candidate": False,
        "best_ok_similarity": .97504864,
        "best_ng_similarity": .89559973,
        "best_ok_source": {
            "label": "OK", "source": "OPERATOR_CONFIRMED",
            "lighting_mode": "SIDE", "record_id": "a" * 64,
        },
    }
    return side


def _v3_nonselective_weak_knn_ok_side(category="DESLOCADO"):
    """Regressão representativa: CNN SIDE OK, KNN A/B OK, mem. não conclusiva."""
    from src.core.strict_category_memory import canonical_memory_category

    side = _v3_weak_c_human_ok_side(category=category)
    score = 3.0383958460333815e-08
    side["confidence"] = 1.0 - score
    detail = side["detail"]
    detail["aoi_v3_trial"]["ng_score_uncalibrated"] = score
    detail["cnn_v3_ng_score_uncalibrated"] = score
    knn = detail["knn_c"]
    knn["neighbors_A"] = [
        {"label": label, "similarity": similarity,
         "source_pixel_sha256": format(i + 1, "064x")}
        for i, (label, similarity) in enumerate((
            ("OK", .76915941), ("OK", .73626385), ("NG", .73365131),
            ("OK", .71776635), ("NG", .71573585),
        ))
    ]
    knn["neighbors_B"] = [
        {"label": "OK", "similarity": similarity,
         "source_pixel_sha256": format(i + 9, "064x")}
        for i, similarity in enumerate(
            (.81667228, .79612735, .78327747, .76647984, .76082523)
        )
    ]
    knn.update(
        vote_A="OK", vote_B="OK",
        weighted_ng_A=.38398391, weighted_ng_B=0.0,
    )
    recall = detail["confirmed_visual_recall"]
    recall.update(
        status="NO_MATCH", category=canonical_memory_category(category),
        best_ok_similarity=.91351401, best_ng_similarity=.90610706,
    )
    return side


def _v3_mixed_knn_verified_human_ok_side(category="DESLOCADO"):
    """CNN OK moderada; memória humana qualificada e top-1 A/B OK."""
    from src.core.strict_category_memory import canonical_memory_category
    side = _v3_weak_c_human_ok_side(category=category)
    score = .06340668350458145
    side["confidence"] = 1.0 - score
    detail = side["detail"]
    detail["aoi_v3_trial"]["ng_score_uncalibrated"] = score
    detail["cnn_v3_ng_score_uncalibrated"] = score
    visual = detail["confirmed_visual_recall"]
    visual.update(
        category=canonical_memory_category(category),
        best_ok_similarity=.99130311, best_ng_similarity=.93202257,
    )
    knn = detail["knn_c"]
    knn.update(
        reason="SEM_SUGESTAO_CONFIAVEL",
        reason_A="VIZINHOS_VISUAIS_DIVERGENTES",
        reason_B="VIZINHOS_VISUAIS_DIVERGENTES",
        vote_A="OK", vote_B="OK",
    )
    knn["neighbors_A"] = [
        {"label": label, "similarity": similarity,
         "source_pixel_sha256": format(i + 11, "064x")}
        for i, (label, similarity) in enumerate((
            ("OK", .97158911), ("NG", .77254799), ("OK", .76119784),
            ("NG", .75769264), ("OK", .75763565),
        ))
    ]
    knn["neighbors_B"] = [
        {"label": label, "similarity": similarity,
         "source_pixel_sha256": format(i + 21, "064x")}
        for i, (label, similarity) in enumerate((
            ("OK", .93626132), ("NG", .72389501), ("OK", .70821909),
            ("OK", .70809284), ("NG", .70636353),
        ))
    ]
    return side


def _v3_nonselective_human_ng_archive_side(category="FALTANDO"):
    """NG com CNN forte, A/B NG e confirmação humana/acervo na SIDE."""
    from src.core.strict_category_memory import canonical_memory_category
    side = _v3_side(verdict="DEFEITO REAL", score=1.0)
    detail = side["detail"]
    knn = detail["knn_c"]
    knn.pop("auto_decision_authorized")  # Forma real do payload KNN C.
    knn.update(
        production_approved=False,
        status="REVISAO", reason="SEM_SUGESTAO_CONFIAVEL",
        reason_A="SIMILARIDADE_VISUAL_INSUFICIENTE",
        reason_B="SIMILARIDADE_VISUAL_INSUFICIENTE",
        suggestion_A="REVISAO", suggestion_B="REVISAO",
        uses_aoi_category=False, weighted_ng_A=1.0,
        weighted_ng_B=.62436895,
    )
    for key in ("requires_human_confirmation", "pre_c_review_required",
                "pre_c_verdict", "pre_c_is_defect"):
        knn.pop(key, None)
    knn["neighbors_A"] = [
        {"label": "NG", "similarity": v,
         "source_pixel_sha256": format(i + 1, "064x")}
        for i, v in enumerate((.88541495, .88244753, .88220593, .88052291, .87851301))
    ]
    knn["neighbors_B"] = [
        {"label": label, "similarity": v,
         "source_pixel_sha256": format(i + 21, "064x")}
        for i, (label, v) in enumerate((
            ("NG", .86128053), ("NG", .85747191), ("NG", .85575239),
            ("OK", .84557149), ("OK", .8400822),
        ))
    ]
    record_id = "a" * 64
    detail["confirmed_visual_recall"] = {
        "schema": "visionx.human_visual_recall.v1",
        "status": "HUMAN_NG_VISUAL_CANDIDATE",
        "diagnostic_only": True, "auto_ok_authorized": False,
        "source": "PERSISTENT_HUMAN_LABELS",
        "category": canonical_memory_category(category),
        "cross_light_candidate": False,
        "best_ng_similarity": .9802842,
        "best_ok_similarity": .89695101,
        "best_ng_source": {
            "label": "NG", "lighting_mode": "SIDE",
            "record_id": record_id, "source": "OPERATOR_CONFIRMED",
        },
    }
    knn["aligned_archive"] = {
        "schema": "visionx.knn_c_archive_alignment.v1",
        "status": "RECONCILED_NG", "reason": "HUMAN_AND_ARCHIVE_ALIGNED",
        "source": "VERIFIED_ARCHIVE_JSON",
        "source_relation": "SHARED_ANOMALY_SIGNATURE_NOT_RGB_EXACT",
        "diagnostic_only": True, "auto_decision_authorized": False,
        "label": "NG", "category": canonical_memory_category(category),
        "human_record_id": record_id,
        "archive_similarity": .9802842,
        "human_archive_similarity": 1.0,
        "best_opposite_archive_similarity": .92306633,
        "archive_source": {
            "label": "NG", "lighting_mode": "SIDE",
            "record_id": record_id, "source_sha256": "b" * 64,
        },
    }
    return side


def _v3_nonselective_knn_ng_no_exact_human_side(category="FALTANDO"):
    """SIDE NG: CNN forte e 5/5 vizinhos NG em A/B, humano NO_MATCH.

    A memória humana NG é próxima, mas não atinge a qualificação de 97%;
    o arquivo alinhado não encontrou um registro humano reconciliável.
    """
    from src.core.strict_category_memory import canonical_memory_category

    side = _v3_side(verdict="DEFEITO REAL", score=1.0)
    detail = side["detail"]
    knn = detail["knn_c"]
    knn.pop("auto_decision_authorized")
    for field in ("requires_human_confirmation", "pre_c_review_required",
                  "pre_c_verdict", "pre_c_is_defect"):
        knn.pop(field, None)
    knn.update(
        status="REVISAO",
        production_approved=False,
        reason="SEM_SUGESTAO_CONFIAVEL",
        reason_A="SIMILARIDADE_VISUAL_INSUFICIENTE",
        reason_B="SIMILARIDADE_VISUAL_INSUFICIENTE",
        suggestion_A="REVISAO",
        suggestion_B="REVISAO",
        vote_A="NG", vote_B="NG",
        weighted_ng_A=1.0, weighted_ng_B=1.0,
        uses_aoi_category=False,
        aligned_archive={
            "schema": "visionx.knn_c_archive_alignment.v1",
            "source": "VERIFIED_ARCHIVE_JSON",
            "status": "NO_MATCH",
            "reason": "NO_CONFIRMED_HUMAN_CANDIDATE",
            "diagnostic_only": True,
            "auto_decision_authorized": False,
        },
    )
    # Mesmos cinco arquivos em cada descritor, em ordem diferente.
    ref_ids = [11, 12, 13, 14, 15]
    knn["neighbors_A"] = [
        {"label": "NG", "similarity": value,
         "source_pixel_sha256": format(ref_id, "064x")}
        for ref_id, value in zip(
            ref_ids, (.89448272, .89330749, .89326921, .8849194, .87241752)
        )
    ]
    knn["neighbors_B"] = [
        {"label": "NG", "similarity": value,
         "source_pixel_sha256": format(ref_id, "064x")}
        for ref_id, value in zip(
            [15, 14, 11, 12, 13],
            (.82663236, .82618392, .82031626, .81137901, .8113014),
        )
    ]
    detail["confirmed_visual_recall"] = {
        "schema": "visionx.human_visual_recall.v1",
        "source": "PERSISTENT_HUMAN_LABELS",
        "status": "NO_MATCH",
        "diagnostic_only": True,
        "auto_ok_authorized": False,
        "cross_light_candidate": False,
        "category": canonical_memory_category(category),
        "best_ng_similarity": .95390625,
        "best_ok_similarity": .93451713,
        "best_ng_source": {
            "label": "NG", "source": "OPERATOR_CONFIRMED",
            "lighting_mode": "SIDE", "record_id": "a"*64,
        },
    }
    return side


class SideEarlyDecisionTests(unittest.TestCase):
    def test_threshold_is_inclusive_for_defect_and_false_failure(self):
        self.assertEqual(SIDE_EARLY_EXIT_THRESHOLD, 0.97)
        for verdict, expected_ng in (
            ("DEFEITO REAL", True),
            ("FALHA FALSA", False),
        ):
            with self.subTest(verdict=verdict):
                original = _side(verdict, 0.97)
                final = build_side_early_decision(original, "FALTANDO")
                self.assertIsNotNone(final)
                self.assertEqual(final["verdict"], verdict)
                self.assertEqual(final["is_defect"], expected_ng)
                self.assertEqual(final["detail"]["side_early_exit"]["analyzed_lights"], ["SIDE"])
                self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"], ["TOP", "MID"])
                self.assertTrue(final["multilight_final"])
                self.assertEqual(final["multilight_category"], "FALTANDO")
                self.assertEqual(final["detail"]["side_early_exit"]["reported_confidence"], 0.97)
                self.assertFalse(final["production_review_required"])
                self.assertFalse(final["detail"]["operator_review_required"])
                self.assertTrue(final["detail"]["side_early_exit"]["production_auto_allowed"])
                self.assertEqual(side_automatic_decision(final), "NG" if expected_ng else "OK")
                self.assertFalse(original["production_review_required"])
                self.assertNotIn("side_early_exit", original["detail"])

    def test_marker_must_match_final_side_and_cannot_be_forged(self):
        import copy

        final = build_side_early_decision(_side("DEFEITO REAL", 0.99), "FALTANDO")
        self.assertEqual(side_automatic_decision(final), "NG")
        for mutation in (
            lambda x: x.update(lighting_mode="TOP"),
            lambda x: x.update(confidence=0.80),
            lambda x: x.update(multilight_final=False),
            lambda x: x.update(production_review_required=True),
            lambda x: x["detail"].update(operator_review_required=True),
            lambda x: x["detail"]["decision_trace"].update(operator_review_required=True),
            lambda x: x["detail"]["side_early_exit"].update(reported_confidence=0.98),
            lambda x: x["detail"]["side_early_exit"].update(rule="untrusted"),
            lambda x: x["detail"]["side_early_exit"].update(production_auto_allowed=False),
            lambda x: x["detail"]["side_early_exit"].update(skipped_lights=["TOP"]),
            lambda x: x["detail"].update(multilight_category="DESLOCADO"),
            lambda x: x.update(verdict="FALHA FALSA"),
        ):
            case = copy.deepcopy(final)
            mutation(case)
            with self.subTest(case=case):
                self.assertIsNone(side_automatic_decision(case))

    def test_blocked_trial_checkpoint_remains_ineligible(self):
        source = _side(
            "DEFEITO REAL", 0.99,
            recognition_route="NEW_CNN",
            cnn_v2_active=True,
            cnn_v2_status="INFERENCE_OK",
            cnn_v2_checkpoint_verified=True,
            cnn_v2_checkpoint_sha256=TRIAL_97P70_SHA256,
        )
        self.assertIsNone(build_side_early_decision(source, "FALTANDO"))

    @staticmethod
    def _trial_ng_side():
        result = _side(
            "DEFEITO REAL", .9999722242355347,
            recognition_route="NEW_CNN",
            cnn_v2_active=True,
            cnn_v2_experimental=True,
            cnn_v2_aoi_category="FALTANDO",
            cnn_v2_lighting_mode="SIDE",
            cnn_v2_status="INFERENCE_OK",
            cnn_v2_checkpoint_verified=True,
            cnn_v2_checkpoint_sha256=TRIAL_97P70_SHA256,
            cnn_v2_trial_manual_only=True,
            cnn_v2_ng_score_uncalibrated=.9999722242355347,
            operator_review_required=True,
            decision_trace={
                "operator_review_required": True,
                "fusion_rule": "cnn_v2_dualscale_direct",
            },
            knn_c={"status": "REVISAO", "bank_status": "LOADING"},
        )
        result["production_review_required"] = True
        result["lighting_mode"] = "SIDE"
        return result

    def test_trial_ng_side_skips_extra_lights_and_authorizes_only_valid_production(self):
        from src.ui.production_confidence_gate import production_decision_policy

        original = self._trial_ng_side()
        final = build_side_early_decision(original, "FALTANDO")
        self.assertIsNotNone(final)
        self.assertEqual(final["verdict"], "DEFEITO REAL")
        self.assertTrue(final["is_defect"])
        self.assertTrue(final["production_review_required"])
        self.assertTrue(final["detail"]["operator_review_required"])
        self.assertEqual(final["detail"]["side_early_exit"]["rule"],
                         SIDE_TRIAL_NG_REVIEW_RULE)
        self.assertFalse(final["detail"]["side_early_exit"]["production_auto_allowed"])
        self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"],
                         ["TOP", "MID"])
        self.assertEqual(side_automatic_decision(final), None)
        self.assertEqual(side_trial_automatic_decision(final), "NG")
        policy = production_decision_policy(final)
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "NG")
        self.assertFalse(policy["operator_review_required"])
        self.assertEqual(policy["reason"], "side_trial_verified_auto_ng")
        # Não altera retroativamente o resultado local de origem.
        self.assertNotIn("side_early_exit", original["detail"])

    def test_production_false_ng_oct_26_continues_to_top_mid_when_knn_both_vote_ok(self):
        # Evento e70363365eaa4681b822b95557977e63: o componente está
        # presente, a CNN experimental errou NG=.999189, e KNN A/B
        # votaram OK mas sem sugestão seletiva. Não enviar NG pela SIDE.
        analysis = self._trial_ng_side()
        analysis["confidence"] = 0.999189555644989
        analysis["detail"]["cnn_v2_ng_score_uncalibrated"] = 0.999189555644989
        analysis["detail"]["knn_c"] = {
            "schema": "visionx.knn_c_live_review_gated.v1",
            "bank_status": "READY",
            "status": "REVISAO",
            "reason": "SEM_SUGESTAO_CONFIAVEL",
            "vote_A": "OK", "vote_B": "OK",
            "suggestion_A": "REVISAO", "suggestion_B": "REVISAO",
            "weighted_ng_A": .45517853,
            "weighted_ng_B": .4325297,
            "diagnostic_only": True,
            "auto_decision_authorized": False,
        }
        from copy import deepcopy
        original = deepcopy(analysis)
        self.assertIsNone(build_side_early_decision(analysis, "FALTANDO"))
        self.assertIsNone(side_trial_automatic_decision(analysis))
        self.assertEqual(original, analysis)  # sem corrigir para OK artificial
        self.assertEqual(analysis["verdict"], "DEFEITO REAL")

    def test_oct29_nearest_ok_in_both_descriptors_defers_side_ng(self):
        # Evento a062c9850a144976a62591e5bfa726a4, 29/10 16:00.
        # Maioria KNN vota NG, mas o vizinho visual mais próximo é OK.
        # A evidência justifica TOP/MID, NÃO aprovação OK automática.
        from copy import deepcopy
        analysis = self._trial_ng_side()
        analysis["confidence"] = .9999750852584839
        analysis["detail"]["cnn_v2_ng_score_uncalibrated"] = .9999750852584839
        c = {
            "schema": "visionx.knn_c_live_review_gated.v1",
            "bank_status": "READY", "status": "SUGERE_NG",
            "vote_A": "NG", "vote_B": "NG",
            "suggestion_A": "SUGERE_NG", "suggestion_B": "REVISAO",
            "diagnostic_only": True, "auto_decision_authorized": False,
            "reference_support": {"OK": 176, "NG": 467},
            "neighbors_A": [
                {"label": "OK", "similarity": .91192131},
                {"label": "NG", "similarity": .86743278},
                {"label": "NG", "similarity": .84959013},
                {"label": "NG", "similarity": .84890596},
                {"label": "NG", "similarity": .84849204},
            ],
            "neighbors_B": [
                {"label": "OK", "similarity": .86549428},
                {"label": "NG", "similarity": .80695629},
                {"label": "NG", "similarity": .79329265},
                {"label": "NG", "similarity": .78208431},
                {"label": "OK", "similarity": .78146806},
            ],
        }
        analysis["detail"]["knn_c"] = c
        original = deepcopy(analysis)
        self.assertIsNone(build_side_early_decision(analysis, "FALTANDO"))
        self.assertEqual(analysis, original)
        self.assertEqual(analysis["verdict"], "DEFEITO REAL")

        # Contraexemplos devem preservar o encerramento NG original.
        for change in ("ng_closer", "weak_ok", "no_ready", "bad_provenance"):
            trial = deepcopy(analysis)
            trial_c = trial["detail"]["knn_c"]
            if change == "ng_closer":
                trial_c["neighbors_A"][1]["similarity"] = .90
            elif change == "weak_ok":
                trial_c["neighbors_B"][0]["similarity"] = .80
            elif change == "no_ready":
                trial_c["bank_status"] = "LOADING"
            else:
                trial_c["diagnostic_only"] = False
            with self.subTest(change=change):
                self.assertIsNotNone(
                    build_side_early_decision(trial, "FALTANDO")
                )
        physical = deepcopy(analysis)
        physical["hard_missing_evidence"] = True
        self.assertIsNotNone(build_side_early_decision(physical, "FALTANDO"))
        other = deepcopy(analysis)
        other["detail"]["cnn_v2_aoi_category"] = "DESLOCADO"
        self.assertIsNone(build_side_early_decision(other, "DESLOCADO"))

    def test_persistent_human_ok_prevents_premature_side_ng(self):
        # Operador julgou A como OK no JSON; novo A muda um pixel/luz.
        # Essa evidência impede o 1/NG SIDE antes de TOP/MID, nunca vira 0.
        from copy import deepcopy
        sample = self._trial_ng_side()
        visual = {
            "schema": "visionx.human_visual_recall.v1",
            "status": "HUMAN_OK_VISUAL_CANDIDATE",
            "source": "PERSISTENT_HUMAN_LABELS",
            "diagnostic_only": True,
            "auto_ok_authorized": False,
        }
        sample["detail"]["confirmed_visual_recall"] = visual
        self.assertIsNone(build_side_early_decision(sample, "FALTANDO"))
        self.assertEqual(sample["verdict"], "DEFEITO REAL")
        for mutation in (
            lambda x: x["detail"]["confirmed_visual_recall"].update(
                status="AMBIGUOUS"),
            lambda x: x.update(hard_missing_evidence=True),
            lambda x: x["detail"].update(
                physical_missing_veto={"status": "SUSPECT_ABSENCE"}),
            lambda x: x["detail"]["confirmed_visual_recall"].update(
                auto_ok_authorized=True),
        ):
            changed = deepcopy(sample)
            mutation(changed)
            self.assertIsNotNone(
                build_side_early_decision(changed, "FALTANDO")
            )

    def test_human_ok_side_counterevidence_defers_shared_cnn_categories(self):
        # FALTANDO V2 é compartilhada por EMBORCADO/INVERTIDO/DESLOCADO.
        # Não fechar SIDE como NG sem consultar TOP/MID quando houver
        # julgamento humano OK aproximado na categoria atual.
        from copy import deepcopy
        for category in ("FALTANDO", "EMBORCADO", "INVERTIDO", "DESLOCADO"):
            with self.subTest(category=category):
                analysis = self._trial_ng_side()
                analysis["detail"]["cnn_v2_aoi_category"] = category
                analysis["detail"]["confirmed_visual_recall"] = {
                    "schema": "visionx.human_visual_recall.v1",
                    "status": "HUMAN_OK_VISUAL_CANDIDATE",
                    "source": "PERSISTENT_HUMAN_LABELS",
                    "diagnostic_only": True,
                    "auto_ok_authorized": False,
                }
                self.assertIsNone(build_side_early_decision(analysis, category))
                hard = deepcopy(analysis)
                hard["hard_missing_evidence"] = True
                self.assertIsNotNone(build_side_early_decision(hard, category))

    def test_defer_on_two_weak_ok_votes_only_not_on_one_vote_or_missing_knn(self):
        from copy import deepcopy
        sample = self._trial_ng_side()
        sample["detail"]["knn_c"] = {
            "schema": "visionx.knn_c_live_review_gated.v1",
            "bank_status": "READY",
            "status": "REVISAO",
            "reason": "SEM_SUGESTAO_CONFIAVEL",
            "vote_A": "OK", "vote_B": "OK",
            "suggestion_A": "REVISAO", "suggestion_B": "REVISAO",
            "diagnostic_only": True,
            "auto_decision_authorized": False,
        }
        variants = (
            ("unavailable", {"bank_status": "LOADING"}),
            ("one_ng", {"vote_B": "NG"}),
            ("no_vote", {"vote_B": None}),
            ("untrusted", {"diagnostic_only": False}),
            ("alternative_status", {"status": "SUGERE_NG"}),
            ("other_reason", {"reason": "ASSINATURA_INSUFICIENTE"}),
            ("incomplete", {"suggestion_A": None}),
        )
        for name, overrides in variants:
            with self.subTest(name=name):
                modified = deepcopy(sample)
                modified["detail"]["knn_c"].update(overrides)
                self.assertIsNotNone(build_side_early_decision(modified, "FALTANDO"))
                self.assertEqual(
                    side_trial_automatic_decision(
                        build_side_early_decision(modified, "FALTANDO")
                    ), "NG",
                )

    def test_physical_hard_missing_evidence_is_not_downgraded_by_knn_soft_ok(self):
        sample = self._trial_ng_side()
        sample["detail"]["knn_c"] = {
            "schema": "visionx.knn_c_live_review_gated.v1",
            "bank_status": "READY", "status": "REVISAO",
            "reason": "SEM_SUGESTAO_CONFIAVEL",
            "vote_A": "OK", "vote_B": "OK",
            "suggestion_A": "REVISAO", "suggestion_B": "REVISAO",
            "diagnostic_only": True, "auto_decision_authorized": False,
        }
        sample["hard_missing_evidence"] = True
        self.assertIsNotNone(build_side_early_decision(sample, "FALTANDO"))
        sample["hard_missing_evidence"] = False
        sample["detail"]["hard_missing_evidence"] = True
        self.assertIsNotNone(build_side_early_decision(sample, "FALTANDO"))
        sample["detail"]["hard_missing_evidence"] = False
        sample["detail"]["cnn_v2_aoi_category"] = "INVERTIDO"
        self.assertIsNone(build_side_early_decision(sample, "INVERTIDO"))

    def test_trial_side_routes_all_supported_aoi_categories_without_changing_defect_label(self):
        from src.ui.production_confidence_gate import production_decision_policy

        # Categoria emitida pela AOI é apenas um contexto de roteamento;
        # a CNN de inspeção de qualidade é compartilhada entre as quatro.
        for category in ("FALTANDO", "INVERTIDO", "EMBORCADO", "DESLOCADO"):
            with self.subTest(category=category):
                source = self._trial_ng_side()
                source["detail"]["cnn_v2_aoi_category"] = category
                if category == "INVERTIDO":
                    # Evento XP 7f3e2cfb, SIDE 13:52 em 21/10/2026.
                    source["confidence"] = .9999839067459106
                    source["detail"]["cnn_v2_ng_score_uncalibrated"] = .9999839067459106
                final = build_side_early_decision(source, category)
                self.assertIsNotNone(final)
                self.assertEqual(final["verdict"], "DEFEITO REAL")
                self.assertTrue(final["is_defect"])
                self.assertEqual(final["multilight_category"], category)
                self.assertEqual(final["detail"]["side_early_exit"]["analyzed_lights"], ["SIDE"])
                self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"], ["TOP", "MID"])
                self.assertEqual(side_trial_automatic_decision(final), "NG")
                policy = production_decision_policy(final)
                self.assertTrue(policy["auto_allowed"])
                self.assertEqual(policy["proposed_decision"], "NG")
                self.assertEqual(policy["reason"], "side_trial_verified_auto_ng")
                self.assertNotIn("side_early_exit", source["detail"])

    def test_trial_side_aoi_scope_never_accepts_mismatched_or_other_engines(self):
        import copy
        source = self._trial_ng_side()
        source["detail"]["cnn_v2_aoi_category"] = "INVERTIDO"
        for category in ("MUITO ADESIVO", "UNKNOWN", "RANDOM", ""):
            with self.subTest(category=category):
                self.assertIsNone(build_side_early_decision(source, category))
        # Não confundir a categoria declarada pela AOI com o escopo em que
        # o modelo realmente inferiu (proteção contra evento/categoria trocada).
        self.assertIsNone(build_side_early_decision(source, "DESLOCADO"))
        mismatched = copy.deepcopy(source)
        mismatched["detail"]["cnn_v2_aoi_category"] = "FALTANDO"
        self.assertIsNone(build_side_early_decision(mismatched, "INVERTIDO"))
        missing = copy.deepcopy(source)
        missing["detail"]["cnn_v2_aoi_category"] = ""
        self.assertIsNone(build_side_early_decision(missing, "INVERTIDO"))

    def test_trial_ok_side_allows_zero_when_raw_score_and_verdict_agree(self):
        from src.ui.production_confidence_gate import production_decision_policy

        side = self._trial_ng_side()
        side.update(verdict="FALHA FALSA", is_defect=False, confidence=.9997)
        side["detail"]["cnn_v2_ng_score_uncalibrated"] = .0003
        final = build_side_early_decision(side, "FALTANDO")
        self.assertIsNotNone(final)
        self.assertEqual(final["detail"]["side_early_exit"]["rule"],
                         SIDE_TRIAL_OK_REVIEW_RULE)
        self.assertEqual(final["detail"]["side_early_exit"]["analyzed_lights"],
                         ["SIDE"])
        self.assertEqual(side_trial_automatic_decision(final), "OK")
        self.assertIsNone(side_automatic_decision(final))
        policy = production_decision_policy(final)
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertEqual(policy["reason"], "side_trial_verified_auto_ok")

    def test_trial_ok_side_also_supports_invertido_routing(self):
        from src.ui.production_confidence_gate import production_decision_policy

        side = self._trial_ng_side()
        side.update(verdict="FALHA FALSA", is_defect=False, confidence=.9997)
        side["detail"]["cnn_v2_ng_score_uncalibrated"] = .0003
        side["detail"]["cnn_v2_aoi_category"] = "INVERTIDO"
        final = build_side_early_decision(side, "INVERTIDO")
        self.assertIsNotNone(final)
        self.assertEqual(side_trial_automatic_decision(final), "OK")
        self.assertEqual(production_decision_policy(final)["proposed_decision"], "OK")
        self.assertTrue(production_decision_policy(final)["auto_allowed"])

    def test_trial_automatic_gate_rechecks_integrity_and_opposition(self):
        import copy
        from src.ui.production_confidence_gate import production_decision_policy

        ng = build_side_early_decision(self._trial_ng_side(), "FALTANDO")
        ok_source = self._trial_ng_side()
        ok_source.update(verdict="FALHA FALSA", is_defect=False, confidence=.9997)
        ok_source["detail"]["cnn_v2_ng_score_uncalibrated"] = .0003
        ok = build_side_early_decision(ok_source, "FALTANDO")
        for final in (ng, ok):
            self.assertIsNotNone(side_trial_automatic_decision(final))
            opposing = "SUGERE_OK" if final["is_defect"] else "SUGERE_NG"
            mutations = (
                lambda x: x.update(lighting_mode="TOP"),
                lambda x: x.update(multilight_final=False),
                lambda x: x.update(eligible_for_final_decision=False),
                lambda x: x.update(production_review_required=False),
                lambda x: x.update(is_defect=not x["is_defect"]),
                lambda x: x.update(verdict="REVISÃO OBRIGATÓRIA"),
                lambda x: x.update(confidence=.95),
                lambda x: x["detail"].update(cnn_v2_status="ERROR"),
                lambda x: x["detail"].update(cnn_v2_checkpoint_verified=False),
                lambda x: x["detail"].update(cnn_v2_checkpoint_sha256="other"),
                lambda x: x["detail"].update(cnn_v2_trial_manual_only=False),
                lambda x: x["detail"].update(memory_conflict=True),
                lambda x: x["detail"].update(knn_c={"status": opposing}),
                lambda x: x["detail"]["side_early_exit"].update(rule="forged"),
                lambda x: x["detail"]["side_early_exit"].update(
                    analyzed_lights=["SIDE", "TOP"]),
                lambda x: x["detail"]["side_early_exit"].update(
                    production_auto_allowed=True),
                lambda x: x["detail"]["side_early_exit"].update(
                    reported_confidence=.80),
            )
            for mutate in mutations:
                sample = copy.deepcopy(final)
                mutate(sample)
                with self.subTest(label=final["verdict"], case=sample):
                    self.assertIsNone(side_trial_automatic_decision(sample))
                    self.assertFalse(production_decision_policy(sample)["auto_allowed"])
        physical = copy.deepcopy(ok)
        physical["detail"]["physical_missing_veto"] = {"status": "SUSPECT_ABSENCE"}
        self.assertFalse(production_decision_policy(physical)["auto_allowed"])
        visual = copy.deepcopy(ok)
        visual["detail"]["knn_visual_corroboration"] = {"status": "NG_CORROBORATED"}
        self.assertFalse(production_decision_policy(visual)["auto_allowed"])

    def test_trial_shortcut_requires_valid_local_ng_and_no_conflicting_c(self):
        import copy

        original = self._trial_ng_side()
        for change in (
            lambda a: a.update(verdict="FALHA FALSA", is_defect=False),
            lambda a: a.update(confidence=.96),
            lambda a: a.update(lighting_mode="MID"),
            lambda a: a.update(production_review_required=False),
            lambda a: a["detail"].update(cnn_v2_status="ERROR"),
            lambda a: a["detail"].update(cnn_v2_checkpoint_verified=False),
            lambda a: a["detail"].update(cnn_v2_checkpoint_sha256="incorrect"),
            lambda a: a["detail"].update(cnn_v2_trial_manual_only=False),
            lambda a: a["detail"].update(cnn_v2_ng_score_uncalibrated=.5),
            lambda a: a["detail"].update(memory_conflict=True),
            lambda a: a["detail"]["knn_c"].update(status="SUGERE_OK"),
            lambda a: a["detail"]["decision_trace"].update(
                fusion_rule="other"),
        ):
            case = copy.deepcopy(original)
            change(case)
            with self.subTest(case=case):
                self.assertIsNone(build_side_early_decision(case, "FALTANDO"))
        self.assertIsNone(build_side_early_decision(original, "INVERTIDO"))

    def test_lower_scores_never_skip_other_lights(self):
        for value in (0.0, 0.50, 0.96, 0.9699, float("nan"), float("inf"), -1.0, 1.01, None, "invalid"):
            with self.subTest(value=value):
                self.assertIsNone(build_side_early_decision(_side(confidence=value), "FALTANDO"))

    def test_revisions_conflicts_and_inconsistent_verdict_are_rejected(self):
        cases = [
            _side("REVISÃO OBRIGATÓRIA", 0.99),
            {**_side("DEFEITO REAL", 0.99), "is_defect": False},
            {**_side("FALHA FALSA", 0.99), "production_review_required": True},
            _side(confidence=0.99, operator_review_required=True),
            _side(confidence=0.99, decision_trace={"operator_review_required": True}),
            _side(confidence=0.99, memory_conflict=True),
            _side(confidence=0.99, recognition_route="MEMORY_CONFLICT"),
            _side(confidence=0.99, recognition_route="KNOWN_KNN"),
        ]
        for analysis in cases:
            with self.subTest(case=analysis):
                self.assertIsNone(build_side_early_decision(analysis, "FALTANDO"))

    def test_unverified_or_reviewing_cnn_never_exits(self):
        base = dict(
            recognition_route="NEW_CNN",
            cnn_v2_active=True,
            cnn_v2_status="INFERENCE_OK",
            cnn_v2_checkpoint_verified=True,
        )
        self.assertIsNotNone(build_side_early_decision(_side(confidence=0.98, **base), "FALTANDO"))
        for overrides in (
            {"cnn_v2_checkpoint_verified": False},
            {"cnn_v2_status": "INFERENCE_ERROR"},
            {"cnn_v2_status": "REVIEW"},
            {"cnn_v2_trial_manual_only": True},
            {"cnn_v2_active": False},
        ):
            with self.subTest(overrides=overrides):
                self.assertIsNone(build_side_early_decision(_side(confidence=0.99, **(base | overrides)), "FALTANDO"))

    def test_exact_memory_must_be_human_verified_and_match_verdict(self):
        base = dict(
            recognition_route="KNOWN_KNN",
            recognition_memory_verified=True,
            recognition_match="EXACT_PAIR",
            recognition_known_label="NG",
        )
        self.assertIsNotNone(build_side_early_decision(_side("DEFEITO REAL", 0.99, **base), "FALTANDO"))
        self.assertIsNone(build_side_early_decision(_side("FALHA FALSA", 0.99, **base), "FALTANDO"))
        self.assertIsNone(build_side_early_decision(_side("DEFEITO REAL", 0.99, **(base | {"recognition_memory_verified": False})), "FALTANDO"))


    def test_v3_epoch5_skips_top_mid_with_agreement_without_review_flag(self):
        from src.ui.production_confidence_gate import production_decision_policy

        for category in ("FALTANDO", "EMBORCADO", "INVERTIDO", "DESLOCADO",
                         "MUITO ADESIVO", "UNKNOWN", ""):
            for verdict in ("FALHA FALSA", "DEFEITO REAL"):
                with self.subTest(category=category, verdict=verdict):
                    original = _v3_side(verdict=verdict)
                    self.assertIsNone(build_side_early_decision(original, category))
                    final = build_side_early_decision(
                        original, category, allow_supervised_v3=True,
                    )
                    self.assertIsNotNone(final)
                    self.assertEqual(final["verdict"], verdict)
                    self.assertEqual(final["detail"]["side_early_exit"]["rule"],
                                     SIDE_V3_SUPERVISED_RULE)
                    self.assertEqual(final["detail"]["side_early_exit"]["model_epoch"], 5)
                    self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"],
                                     ["TOP", "MID"])
                    self.assertEqual(final["detail"]["side_early_exit"]["scope"],
                                     "ALL_MODES_VISUAL_ONLY")
                    self.assertFalse(final["detail"]["side_early_exit"]["production_auto_allowed"])
                    self.assertFalse(final["production_review_required"])
                    self.assertFalse(final["detail"]["operator_review_required"])
                    self.assertFalse(final["detail"]["decision_trace"]["operator_review_required"])
                    self.assertFalse(
                        final["detail"]["side_early_exit"]["operator_review_required"]
                    )
                    self.assertIsNone(side_automatic_decision(final))
                    self.assertIsNone(side_trial_automatic_decision(final))
                    policy = production_decision_policy(final)
                    self.assertFalse(policy["auto_allowed"])
                    self.assertEqual(policy["verdict"], verdict)
                    self.assertNotEqual(policy["proposed_decision"], "OK")
                    self.assertNotIn("side_early_exit", original["detail"])

    def test_side_cnn_knn_agreement_does_not_become_mandatory_review(self):
        from src.services.capture_debug_payload import decision_record
        from src.ui.production_confidence_gate import production_decision_policy

        for category in ("FALTANDO", "EMBORCADO", "MUITO ADESIVO", "UNKNOWN", ""):
            with self.subTest(category=category):
                # Score representativo de uma SIDE V3 verificada; a regra
                # é universal e NÃO depende do ID ou do nome do evento.
                side = _v3_side(score=8.93527057899135e-11)
                completed = build_side_early_decision(
                    side, category, allow_supervised_v3=True,
                )
                self.assertIsNotNone(completed)
                snapshot = decision_record(completed, {"category": category})
                self.assertEqual(snapshot["verdict"], "FALHA FALSA")
                self.assertFalse(snapshot["operator_review_required"])
                self.assertFalse(snapshot["memory"]["operator_review_required"])
                self.assertEqual(
                    snapshot["knn_c"]["status"], "SUGERE_OK",
                )
                self.assertEqual(snapshot["cnn_v3"]["epoch"], 5)
                self.assertFalse(
                    completed["detail"]["side_early_exit"]["production_auto_allowed"]
                )
                xp = production_decision_policy(completed)
                self.assertEqual(xp["verdict"], "FALHA FALSA")
                self.assertFalse(xp["auto_allowed"])
                self.assertEqual(xp["proposed_decision"], "")

    def test_v3_nonselective_knn_and_confirmed_human_ok_finish_on_side(self):
        from src.services.capture_debug_payload import decision_record
        from src.ui.production_confidence_gate import production_decision_policy

        for category in ("FALTANDO", "EMBORCADO", "INVERTIDO", "DESLOCADO",
                         "MUITO ADESIVO"):
            with self.subTest(category=category):
                original = _v3_weak_c_human_ok_side(category=category)
                final = build_side_early_decision(
                    original, category, allow_supervised_v3=True,
                )
                self.assertIsNotNone(final)
                self.assertEqual(final["verdict"], "FALHA FALSA")
                self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"],
                                 ["TOP", "MID"])
                self.assertEqual(final["detail"]["side_early_exit"]["corroboration"],
                                 "HUMAN_OK_KNN_C_A_B_TOP5_UNANIMOUS")
                self.assertEqual(final["detail"]["knn_c"]["status"], "REVISAO")
                self.assertFalse(final["production_review_required"])
                self.assertFalse(final["detail"]["operator_review_required"])
                self.assertIsNone(side_automatic_decision(final))
                self.assertIsNone(side_trial_automatic_decision(final))
                debug = decision_record(final, {"category": category})
                self.assertEqual(debug["verdict"], "FALHA FALSA")
                self.assertFalse(debug["operator_review_required"])
                self.assertEqual(debug["cnn_v3"]["epoch"], 5)
                xp = production_decision_policy(final)
                self.assertFalse(xp["auto_allowed"])
                self.assertEqual(xp["proposed_decision"], "")
                self.assertEqual(xp["verdict"], "FALHA FALSA")

    def test_v3_weak_knn_does_not_override_uncertain_human_or_conflict(self):
        mutations = (
            lambda a: a["detail"].pop("confirmed_visual_recall"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(status="AMBIGUOUS"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(best_ok_similarity=.969),
            lambda a: a["detail"]["confirmed_visual_recall"].update(best_ng_similarity=.95),
            lambda a: a["detail"]["confirmed_visual_recall"].update(category="FALTANDO"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(cross_light_candidate=True),
            lambda a: a["detail"]["confirmed_visual_recall"].update(source="ARCHIVE"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(auto_ok_authorized=True),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ok_source"].update(source="AUTOMATIC"),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ok_source"].update(record_id="invalid"),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ok_source"].update(lighting_mode="TOP"),
            lambda a: a["detail"]["knn_c"].update(bank_status="LOADING"),
            lambda a: a["detail"]["knn_c"].update(reason_A="ASSINATURA_INSUFICIENTE"),
            lambda a: a["detail"]["knn_c"].update(vote_B="NG"),
            lambda a: a["detail"]["knn_c"].update(requires_human_confirmation=True),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][0].update(label="NG"),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][0].update(source_pixel_sha256="bad"),
            lambda a: a["detail"]["knn_c"]["neighbors_A"].pop(),
            lambda a: a["detail"].update(memory_conflict=True),
            lambda a: a["detail"].update(capture_error=True),
            lambda a: a["detail"].update(physical_missing_veto={"status": "NG_CORROBORATED"}),
            lambda a: a.update(confidence=.965),
        )
        for mutate in mutations:
            trial = _v3_weak_c_human_ok_side()
            mutate(trial)
            with self.subTest(mutation=mutate.__code__.co_consts):
                self.assertIsNone(build_side_early_decision(
                    trial, "DESLOCADO", allow_supervised_v3=True,
                ))
        uncertain = _v3_weak_c_human_ok_side()
        self.assertIsNone(build_side_early_decision(
            uncertain, "DESLOCADO", allow_supervised_v3=False,
        ))
        # Um candidato humano OK contra CNN NG obriga outras luzes.
        ng = _v3_side(verdict="DEFEITO REAL")
        ng["detail"]["confirmed_visual_recall"] = (
            _v3_weak_c_human_ok_side()["detail"]["confirmed_visual_recall"]
        )
        self.assertIsNone(build_side_early_decision(
            ng, "DESLOCADO", allow_supervised_v3=True,
        ))

    def test_v3_nonselective_knn_ok_with_nearest_ok_side_stops_all_modes(self):
        from src.services.capture_debug_payload import decision_record
        from src.ui.production_confidence_gate import production_decision_policy

        for category in ("DESLOCADO", "FALTANDO", "MUITO ADESIVO", "UNKNOWN", ""):
            with self.subTest(category=category):
                side = _v3_nonselective_weak_knn_ok_side(category)
                final = build_side_early_decision(
                    side, category, allow_supervised_v3=True,
                )
                self.assertIsNotNone(final)
                self.assertEqual(final["verdict"], "FALHA FALSA")
                self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"],
                                 ["TOP", "MID"])
                self.assertEqual(final["detail"]["side_early_exit"]["corroboration"],
                                 "CNN_V3_KNN_C_A_B_OK_NONSELECTIVE")
                self.assertEqual(final["detail"]["knn_c"]["status"], "REVISAO")
                self.assertFalse(final["production_review_required"])
                self.assertFalse(final["detail"]["operator_review_required"])
                self.assertFalse(final["detail"]["decision_trace"]["operator_review_required"])
                self.assertFalse(final["detail"]["side_early_exit"]["production_auto_allowed"])
                self.assertIsNone(side_automatic_decision(final))
                self.assertIsNone(side_trial_automatic_decision(final))
                self.assertEqual(decision_record(final, {"category": category})["verdict"],
                                 "FALHA FALSA")
                policy = production_decision_policy(final)
                self.assertFalse(policy["auto_allowed"])
                self.assertEqual(policy["proposed_decision"], "")

    def test_v3_nonselective_knn_requires_direction_and_provenance(self):
        mutations = (
            lambda a: a["detail"]["knn_c"].update(vote_A="NG"),
            lambda a: a["detail"]["knn_c"].update(vote_B="NG"),
            lambda a: a["detail"]["knn_c"].update(status="SUGERE_NG"),
            lambda a: a["detail"]["knn_c"].update(bank_status="LOADING"),
            lambda a: a["detail"]["knn_c"].update(reason="DISCORDANCIA_KNN_A_B"),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][0].update(label="NG"),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][0].update(label="NG"),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][0].update(similarity=.1),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][0].update(source_pixel_sha256="missing"),
            lambda a: a["detail"]["knn_c"]["neighbors_B"].pop(),
            lambda a: a["detail"]["confirmed_visual_recall"].update(status="HUMAN_NG_VISUAL_CANDIDATE"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(status="AMBIGUOUS"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(best_ng_similarity=.95),
            lambda a: a["detail"].pop("confirmed_visual_recall"),
            lambda a: a["detail"].update(memory_conflict=True),
            lambda a: a["detail"].update(physical_missing_veto={"status": "SUSPECT_ABSENCE"}),
            lambda a: a["detail"].update(capture_error=True),
            lambda a: a.update(confidence=.95),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation.__code__.co_consts):
                original = _v3_nonselective_weak_knn_ok_side()
                mutation(original)
                self.assertIsNone(build_side_early_decision(
                    original, "DESLOCADO", allow_supervised_v3=True,
                ))

        original = _v3_nonselective_weak_knn_ok_side()
        self.assertIsNone(build_side_early_decision(
            original, "DESLOCADO", allow_supervised_v3=False,
        ))
        # CNN NG não ganha atalho por mera votação diagnóstica KNN.
        side_ng = _v3_side(verdict="DEFEITO REAL")
        side_ng["detail"]["knn_c"] = _v3_nonselective_weak_knn_ok_side()["detail"]["knn_c"]
        self.assertIsNone(build_side_early_decision(
            side_ng, "DESLOCADO", allow_supervised_v3=True,
        ))

    def test_verified_human_ok_with_mixed_knn_finishes_side_below_cnn_threshold(self):
        from src.services.capture_debug_payload import decision_record
        from src.ui.production_confidence_gate import production_decision_policy
        for category in ("FALTANDO", "DESLOCADO", "INVERTIDO", "EMBORCADO"):
            with self.subTest(category=category):
                source = _v3_mixed_knn_verified_human_ok_side(category)
                self.assertLess(source["confidence"], SIDE_EARLY_EXIT_THRESHOLD)
                final = build_side_early_decision(
                    source, category, allow_supervised_v3=True,
                )
                self.assertIsNotNone(final)
                self.assertEqual(final["verdict"], "FALHA FALSA")
                self.assertFalse(final["detail"]["side_early_exit"]["cnn_97pct_threshold_met"])
                self.assertEqual(
                    final["detail"]["side_early_exit"]["corroboration"],
                    "HUMAN_OK_KNN_C_A_B_NEAREST_OK_MIXED",
                )
                self.assertEqual(
                    final["detail"]["side_early_exit"]["skipped_lights"],
                    ["TOP", "MID"],
                )
                self.assertEqual(final["detail"]["knn_c"]["status"], "REVISAO")
                self.assertFalse(final["production_review_required"])
                self.assertFalse(final["detail"]["operator_review_required"])
                self.assertIsNone(side_automatic_decision(final))
                self.assertIsNone(side_trial_automatic_decision(final))
                self.assertEqual(decision_record(final, {"category": category})["verdict"],
                                 "FALHA FALSA")
                policy = production_decision_policy(final)
                self.assertFalse(policy["auto_allowed"])
                self.assertEqual(policy["proposed_decision"], "")
                self.assertIsNone(build_side_early_decision(
                    source, category, allow_supervised_v3=False,
                ))

    def test_verified_human_ok_mixed_knn_never_skips_on_conflict_or_weak_proof(self):
        mutations = (
            lambda a: a["detail"].pop("confirmed_visual_recall"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(status="NO_MATCH"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(best_ok_similarity=.969),
            lambda a: a["detail"]["confirmed_visual_recall"].update(best_ng_similarity=.96),
            lambda a: a["detail"]["confirmed_visual_recall"].update(category="INVERTIDO"),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ok_source"].update(source="AUTOMATIC"),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ok_source"].update(record_id="invalid"),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ok_source"].update(lighting_mode="TOP"),
            lambda a: a["detail"]["knn_c"].update(vote_B="NG"),
            lambda a: a["detail"]["knn_c"].update(bank_status="LOADING"),
            lambda a: a["detail"]["knn_c"].update(status="SUGERE_NG"),
            lambda a: a["detail"]["knn_c"].update(reason_B="SIMILARIDADE_VISUAL_INSUFICIENTE"),
            lambda a: a["detail"]["knn_c"].update(requires_human_confirmation=True),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][0].update(label="NG"),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][1].update(similarity=.968),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][1].update(similarity=.98),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][0].update(source_pixel_sha256="fake"),
            lambda a: a["detail"]["knn_c"]["neighbors_B"].pop(),
            lambda a: a["detail"].update(physical_missing_veto={"status": "SUSPECT_ABSENCE"}),
            lambda a: a["detail"].update(memory_conflict=True),
            lambda a: a["detail"].update(capture_error=True),
            lambda a: a.update(confidence=.42),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate.__code__.co_consts):
                original = _v3_mixed_knn_verified_human_ok_side()
                mutate(original)
                self.assertIsNone(build_side_early_decision(
                    original, "DESLOCADO", allow_supervised_v3=True,
                ))
        ng = _v3_side(verdict="DEFEITO REAL")
        ng["detail"]["confirmed_visual_recall"] = (
            _v3_mixed_knn_verified_human_ok_side()["detail"]["confirmed_visual_recall"]
        )
        self.assertIsNone(build_side_early_decision(
            ng, "DESLOCADO", allow_supervised_v3=True,
        ))

    def test_v3_no_human_match_but_unanimous_ng_knn_skips_lights(self):
        from src.ui.production_confidence_gate import production_decision_policy
        for category in (
            "FALTANDO", "DESLOCADO", "INVERTIDO", "EMBORCADO",
            "UNKNOWN", "MUITO ADESIVO",
        ):
            with self.subTest(category=category):
                original = _v3_nonselective_knn_ng_no_exact_human_side(category)
                result = build_side_early_decision(
                    original, category, allow_supervised_v3=True,
                )
                self.assertIsNotNone(result)
                self.assertEqual(result["verdict"], "DEFEITO REAL")
                marker = result["detail"]["side_early_exit"]
                self.assertEqual(marker["corroboration"],
                                 "CNN_V3_KNN_C_A_B_NG_NONSELECTIVE")
                self.assertEqual(marker["analyzed_lights"], ["SIDE"])
                self.assertEqual(marker["skipped_lights"], ["TOP", "MID"])
                self.assertFalse(marker["production_auto_allowed"])
                self.assertFalse(result["production_review_required"])
                self.assertEqual(result["detail"]["knn_c"]["status"], "REVISAO")
                self.assertEqual(result["detail"]["confirmed_visual_recall"]["status"],
                                 "NO_MATCH")
                self.assertFalse(production_decision_policy(result)["auto_allowed"])
                self.assertIsNone(side_automatic_decision(result))
                self.assertIsNone(side_trial_automatic_decision(result))
                self.assertIsNone(build_side_early_decision(
                    original, category, allow_supervised_v3=False,
                ))

    def test_v3_nonselective_ng_requires_consensus_and_real_memory_lookup(self):
        changes = (
            lambda a: a["detail"]["knn_c"].update(vote_A="OK"),
            lambda a: a["detail"]["knn_c"].update(vote_B="OK"),
            lambda a: a["detail"]["knn_c"].update(status="SUGERE_NG"),
            lambda a: a["detail"]["knn_c"].update(reason_B="VIZINHOS_VISUAIS_DIVERGENTES"),
            lambda a: a["detail"]["knn_c"].update(bank_status="LOADING"),
            lambda a: a["detail"]["knn_c"].update(production_approved=True),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][4].update(label="OK"),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][0].update(label="OK"),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][0].update(source_pixel_sha256="bad"),
            lambda a: a["detail"]["knn_c"]["neighbors_A"].pop(),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][1].update(similarity=.99),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][1].update(
                source_pixel_sha256="a"*64),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][2].update(
                source_pixel_sha256="d"*64),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][3].update(
                source_pixel_sha256="e"*64),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][4].update(
                source_pixel_sha256="f"*64),
            lambda a: a["detail"]["knn_c"]["aligned_archive"].update(status="RECONCILED_OK"),
            lambda a: a["detail"]["knn_c"]["aligned_archive"].update(reason="INVALID"),
            lambda a: a["detail"].pop("confirmed_visual_recall"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(status="HUMAN_OK_VISUAL_CANDIDATE"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(best_ok_similarity=.97),
            lambda a: a["detail"]["confirmed_visual_recall"].update(best_ng_similarity=.92),
            lambda a: a["detail"]["confirmed_visual_recall"].update(category="EMBORCADO"),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ng_source"].update(source="AUTOMATIC"),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ng_source"].update(lighting_mode="TOP"),
            lambda a: a["detail"].update(memory_conflict=True),
            lambda a: a["detail"].update(physical_missing_veto={"status": "UNAVAILABLE"}),
            lambda a: a["detail"].update(capture_error=True),
            lambda a: a.update(production_review_required=True),
        )
        for i, change in enumerate(changes):
            with self.subTest(mutation=i):
                source = _v3_nonselective_knn_ng_no_exact_human_side()
                change(source)
                self.assertIsNone(build_side_early_decision(
                    source, "FALTANDO", allow_supervised_v3=True,
                ))
        # Sem score CNN direcional forte não existe atalho NG.
        source = _v3_nonselective_knn_ng_no_exact_human_side()
        source["confidence"] = .96
        source["detail"]["cnn_v3_ng_score_uncalibrated"] = .96
        source["detail"]["aoi_v3_trial"]["ng_score_uncalibrated"] = .96
        self.assertIsNone(build_side_early_decision(
            source, "FALTANDO", allow_supervised_v3=True,
        ))

    def test_v3_human_ng_reconciled_archive_skips_top_mid(self):
        from src.services.capture_debug_payload import decision_record
        from src.ui.production_confidence_gate import production_decision_policy

        for category in ("FALTANDO", "DESLOCADO", "EMBORCADO", "INVERTIDO"):
            with self.subTest(category=category):
                source = _v3_nonselective_human_ng_archive_side(category)
                final = build_side_early_decision(
                    source, category, allow_supervised_v3=True,
                )
                self.assertIsNotNone(final)
                self.assertEqual(final["verdict"], "DEFEITO REAL")
                self.assertEqual(final["detail"]["side_early_exit"]["corroboration"],
                                 "HUMAN_NG_KNN_C_A_B_ARCHIVE_RECONCILED")
                self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"],
                                 ["TOP", "MID"])
                self.assertTrue(final["detail"]["side_early_exit"]["cnn_97pct_threshold_met"])
                self.assertEqual(final["detail"]["knn_c"]["status"], "REVISAO")
                self.assertFalse(final["production_review_required"])
                self.assertFalse(final["detail"]["operator_review_required"])
                self.assertFalse(final["detail"]["side_early_exit"]["production_auto_allowed"])
                self.assertIsNone(side_automatic_decision(final))
                self.assertIsNone(side_trial_automatic_decision(final))
                self.assertEqual(decision_record(final, {"category": category})["verdict"],
                                 "DEFEITO REAL")
                policy = production_decision_policy(final)
                self.assertFalse(policy["auto_allowed"])
                self.assertEqual(policy["proposed_decision"], "NG")
                self.assertIsNone(build_side_early_decision(
                    source, category, allow_supervised_v3=False,
                ))

    def test_v3_human_ng_nonselective_requires_independent_evidence(self):
        changes = (
            lambda a: a["detail"].pop("confirmed_visual_recall"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(status="AMBIGUOUS"),
            lambda a: a["detail"]["confirmed_visual_recall"].update(best_ng_similarity=.96),
            lambda a: a["detail"]["confirmed_visual_recall"].update(best_ok_similarity=.96),
            lambda a: a["detail"]["confirmed_visual_recall"].update(category="INVERTIDO"),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ng_source"].update(source="AUTOMATIC"),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ng_source"].update(lighting_mode="TOP"),
            lambda a: a["detail"]["confirmed_visual_recall"]["best_ng_source"].update(record_id="bad"),
            lambda a: a["detail"]["knn_c"].update(vote_A="OK"),
            lambda a: a["detail"]["knn_c"].update(vote_B="OK"),
            lambda a: a["detail"]["knn_c"].update(status="SUGERE_NG"),
            lambda a: a["detail"]["knn_c"].update(reason_A="VIZINHOS_VISUAIS_DIVERGENTES"),
            lambda a: a["detail"]["knn_c"].update(production_approved=True),
            lambda a: a["detail"]["knn_c"].update(bank_status="LOADING"),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][0].update(label="OK"),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][2].update(label="OK"),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][0].update(source_pixel_sha256="bad"),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][1].update(similarity=.99),
            lambda a: a["detail"]["knn_c"]["neighbors_A"].pop(),
            lambda a: a["detail"]["knn_c"]["aligned_archive"].update(status="NO_MATCH"),
            lambda a: a["detail"]["knn_c"]["aligned_archive"].update(archive_similarity=.95),
            lambda a: a["detail"]["knn_c"]["aligned_archive"].update(best_opposite_archive_similarity=.96),
            lambda a: a["detail"]["knn_c"]["aligned_archive"].update(human_record_id="c"*64),
            lambda a: a["detail"]["knn_c"]["aligned_archive"]["archive_source"].update(lighting_mode="TOP"),
            lambda a: a["detail"]["knn_c"]["aligned_archive"]["archive_source"].update(source_sha256="bad"),
            lambda a: a["detail"].update(memory_conflict=True),
            lambda a: a["detail"].update(capture_error=True),
            lambda a: a["detail"].update(physical_missing_veto={"status": "UNAVAILABLE"}),
            lambda a: a["detail"].update(operator_review_required=True),
            lambda a: a.update(production_review_required=True),
        )
        for change in changes:
            with self.subTest(change=change.__code__.co_consts):
                side = _v3_nonselective_human_ng_archive_side()
                change(side)
                self.assertIsNone(build_side_early_decision(
                    side, "FALTANDO", allow_supervised_v3=True,
                ))
        not_strong = _v3_nonselective_human_ng_archive_side()
        not_strong["confidence"] = .96
        not_strong["detail"]["cnn_v3_ng_score_uncalibrated"] = .96
        not_strong["detail"]["aoi_v3_trial"]["ng_score_uncalibrated"] = .96
        self.assertIsNone(build_side_early_decision(
            not_strong, "FALTANDO", allow_supervised_v3=True,
        ))
        bad_ok = _v3_nonselective_human_ng_archive_side()
        bad_ok["verdict"] = "FALHA FALSA"
        bad_ok["is_defect"] = False
        self.assertIsNone(build_side_early_decision(
            bad_ok, "FALTANDO", allow_supervised_v3=True,
        ))

    def test_v3_epoch5_does_not_skip_when_model_knn_or_provenance_disagree(self):
        base = _v3_side()
        changes = (
            lambda a: a["detail"]["aoi_v3_trial"].update(epoch=3),
            lambda a: a["detail"]["aoi_v3_trial"].update(checkpoint_sha256="0"*64),
            lambda a: a["detail"]["aoi_v3_trial"].update(predicted="NG"),
            lambda a: a["detail"]["aoi_v3_trial"].update(light="MID"),
            lambda a: a["detail"]["aoi_v3_trial"].update(status="ERROR"),
            lambda a: a["detail"].update(cnn_v3_checkpoint_verified=False),
            lambda a: a["detail"].update(cnn_v3_checkpoint_epoch=3),
            lambda a: a["detail"].update(cnn_v3_ng_score_uncalibrated=.5),
            lambda a: a["detail"].update(cnn_v3_primary=False),
            lambda a: a["detail"].update(cnn_v2_active=True),
            lambda a: a.update(confidence=.96),
            lambda a: a.update(lighting_mode="TOP"),
            lambda a: a.update(production_review_required=True),
            lambda a: a.update(is_defect=True),
            lambda a: a["detail"].update(operator_review_required=True),
            lambda a: a["detail"]["decision_trace"].update(operator_review_required=True),
            lambda a: a["detail"]["knn_c"].update(pre_c_review_required=True),
            lambda a: a["detail"]["knn_c"].update(requires_human_confirmation=True),
            lambda a: a["detail"].update(memory_conflict=True),
            lambda a: a["detail"].update(physical_score=.3),
            lambda a: a["detail"].update(
                physical_missing_veto={"status": "SUSPECT_ABSENCE"}),
            lambda a: a["detail"]["knn_c"].update(status="REVISAO"),
            lambda a: a["detail"]["knn_c"].update(bank_status="LOADING"),
            lambda a: a["detail"]["knn_c"].update(vote_B="NG"),
            lambda a: a["detail"]["knn_c"].update(suggestion_B="REVISAO"),
            lambda a: a["detail"]["knn_c"].update(conflicts_with_existing_verdict=True),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][0].update(label="NG"),
            lambda a: a["detail"]["knn_c"]["neighbors_B"].pop(),
        )
        for change in changes:
            current = deepcopy(base)
            change(current)
            with self.subTest(change=change):
                self.assertIsNone(build_side_early_decision(
                    current, "EMBORCADO", allow_supervised_v3=True,
                ))
        self.assertIsNotNone(build_side_early_decision(
            base, "MUITO ADESIVO", allow_supervised_v3=True,
        ))
        self.assertIsNone(build_side_early_decision(
            base, "EMBORCADO", allow_supervised_v3=False,
        ))

if __name__ == "__main__":
    unittest.main()
