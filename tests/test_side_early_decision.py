"""Regressões do encerramento antecipado SIDE >= 97%."""

import unittest

from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
from src.core.side_early_decision import (
    SIDE_EARLY_EXIT_THRESHOLD,
    SIDE_TRIAL_NG_REVIEW_RULE,
    SIDE_TRIAL_OK_REVIEW_RULE,
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
        self.assertIsNotNone(build_side_early_decision(other, "DESLOCADO"))

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
        self.assertIsNotNone(build_side_early_decision(sample, "INVERTIDO"))

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


if __name__ == "__main__":
    unittest.main()
