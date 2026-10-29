"""Regression for event 8525be296415464a83fcd0764ce7e9d7.

Replay of the logged evidence, NOT a pixel-level inspection of the local
AOI station. Historical OK identity is supported by the recorded KNN
hashes. The current physical board is still NOT individually certified.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.core.knn_visual_corroboration import (
    _recognized_recurrent_ok_20261029, apply_knn_visual_corroboration,
)
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.decision_verdict_feedback import verdict_feedback_state
from src.ui.inspection_memory_feedback import memory_feedback_state
from src.ui.neural_telemetry_model import memory_seen_state, memory_panel_text
from src.ui.neural_influence_model import neural_influence_rows
from src.ui.network_xp_debug import format_multilight_debug_report
from src.core.physical_missing_veto import (
    POLICY as PHYSICAL_VETO_POLICY, apply_physical_missing_veto,
)
from tests.test_knn_three_light_ok_counterevidence import fused, report


SHA = {
    "TOP": "046effa074997ca1211b45e905745f2f6ec68263a0613854e4130eb7168e7d5d",
    "MID": "338f1cc612656297c21db284ea097489b90eb6e0537c3231dde45740f6a16e29",
}
SOURCES = {
    light: f"public/ok_archive/2026-10-27_1318_FALTANDO_{light}.png"
    for light in ("TOP", "MID")
}


def neighbor(label, similarity, light="SIDE", anchor=False):
    return {
        "label": label, "similarity": similarity,
        "path": SOURCES[light] if anchor else
                f"public/{label.lower()}_archive/{light}_example.png",
        "source_pixel_sha256": SHA[light] if anchor else "f" * 64,
    }


def real_event():
    a = fused()
    d = a["detail"]
    d["multilight_final"] = True
    d["eligible_for_final_decision"] = True
    d["recognition_light_routes"] = {x: "NEW_CNN" for x in ("SIDE","TOP","MID")}
    d["cnn_v2_light_diagnostics"] = {
        light: {
            "cnn_active": True, "checkpoint_verified": True,
            "cnn_status": "INFERENCE_OK", "route": "NEW_CNN",
            "verdict": "DEFEITO REAL", "category": "FALTANDO",
            "checkpoint_sha256": "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c",
            "ng_score_uncalibrated": score,
        }
        for light, score in (
            ("SIDE", .9987717270851135),
            ("TOP", .9987717270851135),
            ("MID", .999889612197876),
        )
    }
    d["knn_c_light_observer_votes"]["MID"].update({
        "status": "SUGERE_OK",
        "reason": "B_SELETIVA_E_VOTOS_A_B_CONCORDAM",
        "suggestion_A": "REVISAO", "suggestion_B": "SUGERE_OK",
    })
    d["knn_c_light_neighbor_evidence"] = {
        "SIDE": {
            "neighbors_A": [
                neighbor("NG", .89339485), neighbor("NG", .88418264),
                neighbor("OK", .88014235), neighbor("OK", .86061695),
                neighbor("OK", .85896562),
            ],
            "neighbors_B": [
                neighbor("NG", .83807385), neighbor("NG", .82729085),
                neighbor("OK", .82618206), neighbor("OK", .8120654),
                neighbor("OK", .80632833),
            ],
        },
        "TOP": {
            "neighbors_A": [
                neighbor("OK", .98621705, "TOP", True),
                neighbor("NG", .9127867, "TOP"),
                neighbor("OK", .88240339, "TOP"),
                neighbor("OK", .87639197, "TOP"),
                neighbor("OK", .87000136, "TOP"),
            ],
            "neighbors_B": [
                neighbor("OK", .97805809, "TOP", True),
                neighbor("NG", .88268065, "TOP"),
                neighbor("OK", .85127438, "TOP"),
                neighbor("OK", .84643866, "TOP"),
                neighbor("OK", .84335523, "TOP"),
            ],
        },
        "MID": {
            "neighbors_A": [
                neighbor("OK", .9825117, "MID", True),
                neighbor("OK", .86939965, "MID"),
                neighbor("OK", .86184946, "MID"),
                neighbor("OK", .8583928, "MID"),
                neighbor("NG", .84971955, "MID"),
            ],
            "neighbors_B": [
                neighbor("OK", .9710344, "MID", True),
                neighbor("OK", .81653948, "MID"),
                neighbor("OK", .81091247, "MID"),
                neighbor("OK", .81078255, "MID"),
                neighbor("OK", .80364104, "MID"),
            ],
        },
    }
    visual = report()
    visual["trend_ng_lights"] = []
    visual["strong_ng_lights"] = []
    for light, ok, ng, status in (
        ("SIDE", .991922, .947968, "CONFLICT"),
        ("TOP", .991922, .942076, "CONFLICT"),
        ("MID", .988892, .924092, "NO_CONCLUSION"),
    ):
        visual["lights"][light].update({
            "ok": {"best_similarity": ok},
            "ng": {"best_similarity": ng},
            "ng_margin_over_ok": round(ng - ok, 6),
            "status": status,
        })
    return a, visual


def apply(fused_analysis, visual):
    with patch(
        "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
        return_value=deepcopy(visual),
    ):
        return apply_knn_visual_corroboration(
            deepcopy(fused_analysis), {}, "FALTANDO", object(),
        )


class Oct29RecurringOkTests(unittest.TestCase):
    def test_live_0822_memory_corrects_raw_cnn_and_auto_gate_remains_available(self):
        """New 08:22 capture: same pinned TOP/MID sources, changed SIDE score."""
        a, visual = real_event()
        d = a["detail"]
        d["cnn_v2_light_diagnostics"]["SIDE"][
            "ng_score_uncalibrated"
        ] = .999821126461029
        d["knn_c_light_neighbor_evidence"]["SIDE"] = {
            "neighbors_A": [
                neighbor("OK", .99018946),
                neighbor("NG", .86514910),
                neighbor("OK", .86385871),
                neighbor("OK", .84639380),
                neighbor("NG", .83738332),
            ],
            "neighbors_B": [
                neighbor("OK", .97433490),
                neighbor("OK", .84477406),
                neighbor("NG", .82577236),
                neighbor("OK", .81288249),
                neighbor("OK", .81165862),
            ],
        }
        visual["lights"]["SIDE"].update(
            ok={"best_similarity": .995509},
            ng={"best_similarity": .951698},
            ng_margin_over_ok=-.043811,
        )
        final = apply(a, visual)
        self.assertEqual(final["verdict"], "FALHA FALSA")
        before_arm = neural_influence_rows(final)
        self.assertEqual([r["id"] for r in before_arm], [
            "cnn_side", "cnn_top", "cnn_mid", "knn_visual_summary"
        ])
        self.assertTrue(all(
            r["overruled_by_memory"] is True for r in before_arm[:3]
        ))
        self.assertTrue(before_arm[-1]["knn_historical_ok"])
        self.assertEqual(before_arm[-1]["knn_visual_label"], "OK")
        self.assertEqual(before_arm[-1]["known_memory_label"], "")
        self.assertAlmostEqual(
            before_arm[0]["ng_score_uncalibrated"], .999821126461029
        )
        self.assertFalse(before_arm[0]["production_auto_ok_authorized"])
        final["detail"]["production_continuous_auto"] = True
        policy = production_decision_policy(final)
        self.assertEqual(
            policy["reason"],
            "supervised_operator_known_top_mid_ok_20261029_auto_v1",
        )
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])
        after_arm = neural_influence_rows(final)
        self.assertTrue(after_arm[0]["production_auto_ok_authorized"])
        self.assertTrue(after_arm[-1]["production_auto_ok_authorized"])
        self.assertAlmostEqual(
            after_arm[0]["ng_score_uncalibrated"], .999821126461029
        )

    def test_0841_realistic_missing_probe_is_collected_after_ok_recovery(self):
        """Reproduce the field missing in XP event 91d181b3 (08:41)."""
        a, visual = real_event()
        fused = apply(a, visual)
        self.assertEqual(fused["verdict"], "FALHA FALSA")
        fused["detail"].pop("physical_missing_veto", None)
        fused["detail"]["production_continuous_auto"] = True

        samples = {
            light: {
                "lighting_mode": light,
                "sample_image": object(),
                "test_image": object(),
            }
            for light in ("SIDE", "TOP", "MID")
        }
        def clean_physical_probe(_reference, _test):
            return {"status": "NO_STRONG_EVIDENCE", "scales": [
                {"focus_fraction": .58}, {"focus_fraction": .76},
            ]}
        self.assertFalse(production_decision_policy(fused)["auto_allowed"])
        physically_checked = apply_physical_missing_veto(
            fused, samples, "FALTANDO", probe=clean_physical_probe,
        )
        physical = physically_checked["detail"]["physical_missing_veto"]
        self.assertEqual(physical["policy"], PHYSICAL_VETO_POLICY)
        self.assertEqual(physical["status"], "NO_STRONG_EVIDENCE")
        self.assertEqual(set(physical["lights"]), {"SIDE", "TOP", "MID"})
        self.assertEqual(physically_checked["verdict"], "FALHA FALSA")
        policy = production_decision_policy(physically_checked)
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "OK")

    def test_late_physical_probe_blocks_0841_auto_ok_on_any_unverified_light(self):
        a, visual = real_event()
        for problem in ("UNAVAILABLE", "GEOMETRY_REVIEW", "SUSPECT_ABSENCE"):
            with self.subTest(problem=problem):
                final = apply(a, visual)
                final["detail"].pop("physical_missing_veto", None)
                final["detail"]["production_continuous_auto"] = True
                samples = {
                    light: {
                        "lighting_mode": light,
                        "sample_image": object(), "test_image": object(),
                    }
                    for light in ("SIDE", "TOP", "MID")
                }
                def probe(_reference, _test):
                    return {"status": problem, "scales": []}
                checked = apply_physical_missing_veto(
                    final, samples, "FALTANDO", probe=probe,
                )
                self.assertEqual(
                    checked["detail"]["physical_missing_veto"]["status"], problem
                )
                policy = production_decision_policy(checked)
                self.assertFalse(policy["auto_allowed"])
                self.assertNotEqual(policy["proposed_decision"], "OK")

    def test_0858_historic_visual_ok_cannot_clear_current_mid_absence_probe(self):
        # The actual 08:58 debug had pinned TOP/MID OK memory, but SIDE/TOP
        # geometry review, MID suspect absence, and controller paused.
        a, visual = real_event()
        final = apply(a, visual)
        self.assertEqual(final["verdict"], "FALHA FALSA")
        final["detail"]["production_continuous_auto"] = True
        final["detail"]["physical_missing_veto"] = {
            "policy": PHYSICAL_VETO_POLICY,
            "status": "SUSPECT_ABSENCE",
            "suspect_lights": ["MID"],
            "geometry_review_lights": ["SIDE", "TOP"],
            "unavailable_lights": [],
            "lights": {
                "SIDE": {"status": "GEOMETRY_REVIEW",
                         "geometry": {"method": "center_trim_no_rescale",
                                      "reference_hw": [318, 570],
                                      "test_hw": [317, 567]},
                         "scales": []},
                "TOP": {"status": "GEOMETRY_REVIEW", "scales": []},
                "MID": {"status": "SUSPECT_ABSENCE",
                        "geometry": {"method": "exact"},
                        "scales": [
                            {"focus_fraction": .58, "suspected_absence": True,
                             "score": .8, "coverage": .4},
                            {"focus_fraction": .76, "suspected_absence": False},
                        ]},
            },
        }
        policy = production_decision_policy(final)
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertEqual(
            policy["reason"], "known_top_mid_ok_physical_veto_suspect_absence"
        )
        panel = SimpleNamespace(
            combo_mode=SimpleNamespace(currentText=lambda: "Modo Produção"),
            adhesive_multilight_last_category="FALTANDO",
            adhesive_multilight_last_final_analysis=final,
            adhesive_multilight_last_analyses={},
            adhesive_multilight_last_event_id="bdf7a338f7e94aaba09697cd568f9f10",
            production_autonomy_controller=SimpleNamespace(
                state="scrolling", paused=True
            ),
            last_decision_command_success=None,
        )
        with patch(
            "src.ui.network_xp_debug._multilight_event_matches",
            return_value=True,
        ):
            output = format_multilight_debug_report(panel)
        for expected in (
            "MEMÓRIA OK RECORRENTE / AUDITORIA FÍSICA ATUAL",
            "Referência visual TOP/MID: RECONHECIDA NO ok_archive",
            "Par pixel-idêntico atual: NÃO COMPROVADO (NOT_FOUND)",
            "Resultado da contraprova física: SUSPECT_ABSENCE",
            "Sonda MID: SUSPECT_ABSENCE",
            "suspected_absence=True",
            "XP 0/OK BLOQUEADO PELA CONTRAPROVA",
            "Pausado por espaço: True",
            "Regra XP: known_top_mid_ok_physical_veto_suspect_absence",
            "Comando XP proposto: NENHUM",
        ):
            self.assertIn(expected, output)

    def test_0957_historical_ok_is_shown_as_seen_similar_not_unseen(self):
        raw, visual = real_event()
        final = apply(raw, visual)
        final["detail"]["production_continuous_auto"] = True
        final["detail"]["physical_missing_veto"] = {
            "status": "SUSPECT_ABSENCE",
            "suspect_lights": ["MID"],
            "geometry_review_lights": ["SIDE", "TOP"],
            "unavailable_lights": [],
        }
        seen = memory_seen_state(final)
        self.assertEqual(seen["status"], "VI_SEMELHANTE")
        self.assertEqual(seen["archive_label"], "OK")
        self.assertTrue(seen["historical_ok_recall"])
        self.assertFalse(seen["exact_pair_seen"])
        self.assertEqual(seen["archive_visual_lights"], ["TOP", "MID"])
        self.assertIn("MEMÓRIA OK", memory_panel_text(final)[0])
        self.assertIn("SUSPECT_ABSENCE", memory_panel_text(final)[1])
        self.assertEqual(memory_feedback_state(final)[0], "MEMÓRIA OK • TOP/MID")
        policy = production_decision_policy(final)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertEqual(
            policy["reason"], "known_top_mid_ok_physical_veto_suspect_absence"
        )
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "")

    def test_recovery_panel_requires_valid_original_archive_evidence(self):
        a, visual = real_event()
        original = apply(a, visual)
        invalid = deepcopy(original)
        invalid["detail"]["production_continuous_auto"] = True
        invalid["detail"]["recognized_top_mid_ok_20261029"][
            "per_light_reference"]["MID"]["source_pixel_sha256"] = "0" * 64
        seen = memory_seen_state(invalid)
        self.assertFalse(seen.get("historical_ok_recall"))
        self.assertEqual(seen["status"], "NUNCA_VI")
        self.assertIn("SEM PAR EXATO", memory_panel_text(invalid)[0])
        self.assertNotEqual(memory_feedback_state(invalid)[0], "MEMÓRIA OK • TOP/MID")

    def test_late_physical_probe_missing_pair_does_not_approve(self):
        a, visual = real_event()
        final = apply(a, visual)
        final["detail"].pop("physical_missing_veto", None)
        final["detail"]["production_continuous_auto"] = True
        checked = apply_physical_missing_veto(final, {}, "FALTANDO")
        self.assertEqual(
            checked["detail"]["physical_missing_veto"]["status"], "UNAVAILABLE"
        )
        self.assertFalse(production_decision_policy(checked)["auto_allowed"])

    def test_debug_distinguishes_cnn_review_from_armed_xp_auto_decision(self):
        a, visual = real_event()
        final = apply(a, visual)
        panel = SimpleNamespace(
            combo_mode=SimpleNamespace(currentText=lambda: "Modo Produção"),
            adhesive_multilight_last_category="FALTANDO",
            adhesive_multilight_last_final_analysis=final,
            adhesive_multilight_last_analyses={},
            adhesive_multilight_last_event_id="a614cc6e056746d88b41f417a978c6a5",
            production_autonomy_controller=SimpleNamespace(
                state="auto_decision_pending", paused=False
            ),
            last_decision_command_success=None,
        )
        with patch(
            "src.ui.network_xp_debug._multilight_event_matches",
            return_value=True,
        ):
            before = format_multilight_debug_report(panel)
            self.assertIn("Requer tecla 0/1 humana: N/D", before)
            self.assertIn("Política armada: False", before)
            final["detail"]["production_continuous_auto"] = True
            after = format_multilight_debug_report(panel)
        self.assertIn("Política armada: True", after)
        self.assertIn("Autoriza automático: True", after)
        self.assertIn("Requer tecla 0/1 humana: False", after)
        self.assertIn("Comando XP proposto: 0/OK", after)
        self.assertIn(
            "supervised_operator_known_top_mid_ok_20261029_auto_v1", after
        )
        self.assertIn("Envio XP confirmado: None", after)

    def test_invalid_archive_anchor_never_paints_memory_ok_or_authorizes_xp_zero(self):
        a, visual = real_event()
        final = apply(a, visual)
        final["detail"]["production_continuous_auto"] = True
        final["detail"]["recognized_top_mid_ok_20261029"][
            "per_light_reference"]["TOP"]["source_pixel_sha256"] = "0" * 64
        rows = neural_influence_rows(final)
        self.assertFalse(any(row.get("knn_historical_ok") for row in rows))
        self.assertFalse(any(row.get("overruled_by_memory") for row in rows))
        self.assertNotEqual(
            production_decision_policy(final).get("proposed_decision"), "OK"
        )

    def test_human_known_top_mid_auto_zero_only_in_continuous_production(self):
        raw, visual = real_event()
        result = apply(raw, visual)
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertIs(result["is_defect"], False)
        self.assertEqual(result["detail"]["fusion_rule"],
                         "operator_known_top_mid_ok_review_20261029_v1")
        self.assertEqual(
            result["detail"]["recognized_top_mid_ok_20261029"][
                "per_light_reference"]["MID"]["path"], SOURCES["MID"])
        self.assertFalse(result["detail"]["recognized_top_mid_ok_20261029"][
            "current_pair_exact_match"])
        self.assertFalse(result["eligible_for_final_decision"])
        self.assertTrue(result["production_review_required"])
        self.assertEqual(verdict_feedback_state(result),
                         ("FALHA FALSA", "ok"))
        result["detail"]["production_continuous_auto"] = True
        policy = production_decision_policy(result)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertEqual(policy["reason"],
                         "supervised_operator_known_top_mid_ok_20261029_auto_v1")
        self.assertEqual(verdict_feedback_state(result),
                         ("FALHA FALSA", "ok"))
        # Teste/Sombra não recebem o armamento do controlador Produção.
        result["detail"].pop("production_continuous_auto")
        manual = production_decision_policy(result)
        self.assertFalse(manual["auto_allowed"])
        self.assertTrue(manual["operator_review_required"])
        self.assertEqual(manual["proposed_decision"], "")
        self.assertEqual(manual["reason"],
                         "operator_known_top_mid_ok_20261029_requires_operator")

    def test_auto_ok_requires_complete_verified_capture_and_clean_physical_evidence(self):
        raw, visual = real_event()
        baseline = apply(raw, visual)
        baseline["detail"]["production_continuous_auto"] = True
        for name, change in (
            ("missing_top_route", lambda a: a["detail"]["recognition_light_routes"].pop("TOP")),
            ("mixed_route", lambda a: a["detail"]["recognition_light_routes"].update(SIDE="MEMORY_EXACT")),
            ("physical_review", lambda a: a["detail"]["physical_missing_veto"].update(status="GEOMETRY_REVIEW")),
            ("physical_unavailable", lambda a: a["detail"]["physical_missing_veto"].update(status="UNAVAILABLE")),
            ("physical_suspect", lambda a: a["detail"]["physical_missing_veto"].update(status="SUSPECT_ABSENCE")),
            ("trial_not_verified", lambda a: a["detail"].update(cnn_v2_trial_manual_only=False)),
            ("hard_ng", lambda a: a["detail"].update(hard_missing_evidence=True)),
            ("other_category", lambda a: a["detail"].update(multilight_category="DESLOCADO")),
            ("missing_light", lambda a: a["detail"].update(multilight_final=False)),
            ("wrong_anchor", lambda a: a["detail"]["knn_c_light_neighbor_evidence"]["TOP"]["neighbors_A"][0].update(source_pixel_sha256="0"*64)),
        ):
            with self.subTest(name=name):
                invalid = deepcopy(baseline)
                change(invalid)
                policy = production_decision_policy(invalid)
                self.assertNotEqual(policy.get("proposed_decision"), "OK")
                self.assertFalse(policy.get("auto_allowed"))

    def test_failure_of_any_independent_proof_blocks_recall(self):
        cases = (
            ("wrong_sha", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["MID"][
                    "neighbors_A"][0].update(source_pixel_sha256="0"*64)),
            ("wrong_source", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["TOP"][
                    "neighbors_B"][0].update(path="public/ok_archive/other.png")),
            ("top_ng_competing", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["TOP"][
                    "neighbors_A"][1].update(similarity=.94)),
            ("side_ng_competing", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["SIDE"][
                    "neighbors_A"][0].update(similarity=.95)),
            ("side_ng_majority", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["SIDE"][
                    "neighbors_A"][2].update(label="NG")),
            ("low_top_signature", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["TOP"][
                    "neighbors_A"][0].update(similarity=.984)),
            ("low_mid_signature", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["MID"][
                    "neighbors_B"][0].update(similarity=.965)),
            ("wrong_mid_vote", lambda a,r:
                a["detail"]["knn_c_light_observer_votes"]["MID"].update(
                    vote_B="NG")),
            ("wrong_mid_reason", lambda a,r:
                a["detail"]["knn_c_light_observer_votes"]["MID"].update(
                    reason="A_E_B_CONCORDAM")),
            ("bad_checkpoint", lambda a,r:
                a["detail"]["cnn_v2_light_diagnostics"]["MID"].update(
                    checkpoint_verified=False)),
            ("cnn_valid_ng_consensus", lambda a,r:
                a["detail"].update(cnn_v2_consensus="NG")),
            ("hard_missing", lambda a,r:
                a["detail"]["decision_trace"].update(
                    raw_hard_missing_evidence=True)),
            ("physical_suspect", lambda a,r:
                a["detail"]["physical_missing_veto"].update(
                    status="SUSPECT_ABSENCE")),
            ("visual_side_ng_wins", lambda a,r:
                r["lights"]["SIDE"].update(
                    ng={"best_similarity": .998},
                    ng_margin_over_ok=round(.998-.991922,6))),
            ("visual_mid_weak", lambda a,r:
                r["lights"]["MID"].update(
                    ok={"best_similarity": .96},
                    ng_margin_over_ok=round(.924092-.96,6))),
            ("no_mid", lambda a,r:
                a["detail"]["knn_c_light_observer_votes"].pop("MID")),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                a, r = real_event()
                mutate(a, r)
                self.assertIsNone(
                    _recognized_recurrent_ok_20261029(a,r,"FALTANDO"))
                out = apply(a, r)
                self.assertNotEqual(
                    out["detail"].get("fusion_rule"),
                    "operator_known_top_mid_ok_review_20261029_v1")
                self.assertFalse(
                    production_decision_policy(out).get("proposed_decision")
                    == "OK")

    def test_final_marker_revalidated_on_policy_and_feedback(self):
        a, r = real_event()
        final = apply(a,r)
        final["detail"]["production_continuous_auto"] = True
        forged = deepcopy(final)
        forged["detail"]["recognized_top_mid_ok_20261029"][
            "per_light_reference"]["TOP"]["source_pixel_sha256"] = "0"*64
        p = production_decision_policy(forged)
        self.assertNotEqual(p["verdict"], "FALHA FALSA")
        self.assertNotEqual(verdict_feedback_state(forged), ("FALHA FALSA","ok"))
        self.assertNotEqual(p["proposed_decision"], "OK")


if __name__ == "__main__":
    unittest.main()
