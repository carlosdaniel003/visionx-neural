"""Regressões da prioridade de memória por iluminação SIDE (evento 30/10).

São testes de contrato com números reais, NÃO equivalem a prova industrial.
"""
from copy import deepcopy
import unittest

from src.core.side_memory_first import (
    RULE, KNOWN_MEMORY_CATEGORIES, PHYSICAL_PROBE_SCOPE,
    build_side_memory_first, memory_first_side_decision,
    verified_side_memory_visual_review,
)
from src.ui.production_confidence_gate import production_decision_policy


def _source(label="OK", light="SIDE"):
    return {
        "label": label, "lighting_mode": light,
        "record_id": ("a" if label == "OK" else "b") * 64,
        "source": "OPERATOR_CONFIRMED",
    }


def _row(label, score, path=None, *, category="FALTANDO"):
    path = path or (
        "public/" + ("ok_archive" if label == "OK" else "ng_archive")
        + "/archive_" + category + ".png"
    )
    return {
        "label": label, "similarity": score, "path": path,
        "source_pixel_sha256": ("c" if label == "OK" else "d") * 64,
    }


def case(label="OK", category="FALTANDO"):
    other = "NG" if label == "OK" else "OK"
    # SIDE do evento 88584d..., 30/10. KNN A/B escolheram o MESMO PNG OK.
    a = [_row(label, .98236720, category=category), _row(other, .86601348, category=category),
         _row(label, .86592282, category=category), _row(label, .85552766, category=category),
         _row(label, .84238623, category=category)]
    b = [_row(label, .97262484, category=category), _row(label, .84250628, category=category),
         _row(other, .82652949, category=category), _row(label, .81967648, category=category),
         _row(label, .8171847, category=category)]
    human = {
        "schema": "visionx.human_visual_recall.v1",
        "status": "HUMAN_" + label + "_VISUAL_CANDIDATE",
        "source": "PERSISTENT_HUMAN_LABELS",
        "diagnostic_only": True, "auto_ok_authorized": False,
        "category": category,
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
            "recognition_route": "NEW_EXPERTS" if category == "MUITOADESIVO" else "NEW_CNN",
            "recognition_match": "NOT_FOUND",
            "recognition_memory_verified": False,
            "cnn_v2_aoi_category": category,
            "cnn_v2_active": category != "MUITOADESIVO",
            "cnn_v2_experimental": category != "MUITOADESIVO",
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
    def decide(self, data=None, probe=None, category="FALTANDO"):
        # A sonda real trabalha sobre os pares da captura; aqui só isolamos
        # o contrato, sem fingir que esses PNGs existem no ambiente CI.
        return build_side_memory_first(
            case(category=category) if data is None else data,
            category, object(), object(),
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
        self.assertEqual(policy["reason"], "side_operator_memory_first_97pct_auto_ok")

    def _oct30_case(self):
        """Evento c0f029...: OK 99.02 x NG 95.94, PNG A/B OK comum."""
        data = case()
        human = data["detail"]["confirmed_visual_recall"]
        human.update(
            status="AMBIGUOUS",
            best_ok_similarity=.99019399,
            best_ng_similarity=.95938641,
            best_ok_source={
                **_source("OK"), "record_id":
                "4d6a371a0922841deefd6d775adf2435303a552eb52bd8d686532be28a6a33b9",
            },
            best_ng_source={
                **_source("NG"), "record_id":
                "2735f22cef77b8538d2fa5a8168184916151c6860b5142b4377c124100d57900",
            },
        )
        knn = data["detail"]["knn_c"]
        image = "public/ok_archive/2026-10-21_1424_FALTANDO.png"
        a = [
            _row("OK", .98084619, image),
            _row("NG", .89033113),
            _row("NG", .86702473),
            _row("OK", .85802850),
            _row("NG", .85746296),
        ]
        b = [
            _row("OK", .95345213, image),
            _row("OK", .81405994),
            _row("NG", .81259430),
            _row("NG", .79549029),
            _row("NG", .77911203),
        ]
        knn.update(neighbors_A=a, neighbors_B=b)
        return data

    def _oct30_1402_case(self):
        """Evento b55f5d0d...: humano 98,756% + KNN A/B mesma SIDE OK."""
        data = self._oct30_case()
        human = data["detail"]["confirmed_visual_recall"]
        human.update(
            status="AMBIGUOUS",
            best_ok_similarity=.98756365,
            best_ng_similarity=.94888621,
            best_ok_source={
                **_source("OK"),
                "record_id": "57f49e0d6f6181bba2141fdca3ae4b2bdff452be46b07db623fd09422484431b",
            },
            best_ng_source={
                **_source("NG"),
                "record_id": "3ca3da6408a16df2e8a1b1e9b08d8b4d37674144a9085bc3352a1c6d9cf2ae64",
            },
        )
        side_file = "public/ok_archive/2026-10-21_1436_FALTANDO_SIDE.png"
        knn = data["detail"]["knn_c"]
        knn.update(
            status="REVISAO",
            neighbors_A=[
                _row("OK", .98634805, side_file),
                _row("OK", .90152321),
                _row("NG", .89437767),
                _row("OK", .88637450),
                _row("OK", .88259555),
            ],
            neighbors_B=[
                _row("OK", .97369606, side_file),
                _row("OK", .86716377),
                _row("OK", .86093490),
                _row("NG", .85322969),
                _row("OK", .84774451),
            ],
        )
        return data

    def test_oct30_1402_high_corroboration_side_ok_auto_zero_when_physical_clear(self):
        original = self._oct30_1402_case()
        before = deepcopy(original)
        result = self.decide(original, probe=good_probe(trim=True))
        self.assertEqual(original, before)
        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertIs(result["is_defect"], False)
        marker = result["detail"]["side_memory_first"]
        self.assertEqual(marker["knn_agreement_mode"], "KNN_AB_HIGH_CORROBORATED_HUMAN_OK")
        self.assertEqual(
            marker["archive_source"],
            "public/ok_archive/2026-10-21_1436_FALTANDO_SIDE.png",
        )
        self.assertEqual(marker["min_similarity"], .97)
        self.assertEqual(marker["min_human_similarity"], .985)
        self.assertEqual(marker["min_opposite_margin"], .035)
        self.assertEqual(
            result["detail"]["side_early_exit"]["skipped_lights"],
            ["TOP", "MID"],
        )
        self.assertEqual(memory_first_side_decision(result), "OK")
        policy = production_decision_policy(result)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])

    def test_oct30_1402_physical_disagreement_never_silently_approves(self):
        source = self._oct30_1402_case()
        suspect = good_probe(trim=True)
        suspect["status"] = "SUSPECT_ABSENCE"
        suspect["scales"][0]["suspected_absence"] = True
        result = self.decide(source, probe=suspect)
        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertEqual(result["detail"]["side_early_exit"]["skipped_lights"], ["TOP", "MID"])
        self.assertTrue(verified_side_memory_visual_review(result))
        self.assertIsNone(memory_first_side_decision(result))
        policy = production_decision_policy(result)
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "")
        result["detail"]["production_continuous_auto"] = True
        self.assertFalse(production_decision_policy(result)["auto_allowed"])
        self.assertEqual(production_decision_policy(result)["proposed_decision"], "")
        result["detail"]["side_strong_ok_supervised_armed"] = True
        supervised = production_decision_policy(result)
        self.assertTrue(supervised["auto_allowed"])
        self.assertEqual(supervised["proposed_decision"], "OK")

    def test_oct30_1402_high_corroboration_rejects_weak_or_conflicting_evidence(self):
        changes = (
            lambda d: d["detail"]["confirmed_visual_recall"].update(best_ok_similarity=.9849),
            lambda d: d["detail"]["confirmed_visual_recall"].update(best_ng_similarity=.955),
            lambda d: d["detail"]["knn_c"]["neighbors_A"][0].update(similarity=.9849),
            lambda d: d["detail"]["knn_c"]["neighbors_B"][0].update(similarity=.9699),
            lambda d: d["detail"]["knn_c"]["neighbors_A"][2].update(similarity=.900),
            lambda d: d["detail"]["knn_c"]["neighbors_A"][2].update(similarity=.908),
            lambda d: d["detail"]["knn_c"]["neighbors_B"][0].update(path="public/ok_archive/different.png"),
            lambda d: d["detail"]["knn_c"]["neighbors_B"][0].update(source_pixel_sha256="f"*64),
            lambda d: d["detail"].update(hard_missing_evidence=True),
            lambda d: d["detail"]["confirmed_visual_recall"]["best_ok_source"].update(source="UNKNOWN"),
        )
        for change in changes:
            with self.subTest(change=change):
                data = self._oct30_1402_case()
                change(data)
                self.assertIsNone(self.decide(data))

    def test_strong_human_ok_same_png_changes_ng_to_false_failure(self):
        raw = self._oct30_case()
        original = deepcopy(raw)
        final = self.decide(raw, probe=good_probe(trim=True))
        self.assertEqual(raw, original)
        self.assertIsNotNone(final)
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertIs(final["is_defect"], False)
        self.assertEqual(
            final["detail"]["side_memory_first"]["knn_agreement_mode"],
            "KNN_AB_STRONG_HUMAN_OK",
        )
        self.assertEqual(
            final["detail"]["side_memory_first"]["archive_source"],
            "public/ok_archive/2026-10-21_1424_FALTANDO.png",
        )
        self.assertEqual(final["detail"]["side_early_exit"]["threshold"], .95)
        self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"],
                         ["TOP", "MID"])
        self.assertEqual(memory_first_side_decision(final), "OK")
        policy = production_decision_policy(final)
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertEqual(policy["threshold"], .95)

    def test_strong_human_ok_needs_two_agreeing_pngs_and_clear_ng_margin(self):
        cases = (
            ("human_below_99", lambda r: r["detail"]["confirmed_visual_recall"].__setitem__(
                "best_ok_similarity", .989)),
            ("human_margin_below_3", lambda r: r["detail"]["confirmed_visual_recall"].__setitem__(
                "best_ng_similarity", .961)),
            ("weak_descriptor_a", lambda r: r["detail"]["knn_c"]["neighbors_A"][0].__setitem__(
                "similarity", .979)),
            ("weak_descriptor_b", lambda r: r["detail"]["knn_c"]["neighbors_B"][0].__setitem__(
                "similarity", .949)),
            ("different_archive_png", lambda r: r["detail"]["knn_c"]["neighbors_B"][0].__setitem__(
                "path", "public/ok_archive/another.png")),
            ("different_archive_digest", lambda r: r["detail"]["knn_c"]["neighbors_B"][0].__setitem__(
                "source_pixel_sha256", "e"*64)),
            ("strong_ng_neighbor_a", lambda r: r["detail"]["knn_c"]["neighbors_A"][1].__setitem__(
                "similarity", .90)),
            ("strong_ng_neighbor_b", lambda r: r["detail"]["knn_c"]["neighbors_B"][2].__setitem__(
                "similarity", .90)),
            ("other_light_human", lambda r: r["detail"]["confirmed_visual_recall"][
                "best_ok_source"].__setitem__("lighting_mode", "TOP")),
            ("wrong_source", lambda r: r["detail"]["confirmed_visual_recall"][
                "best_ok_source"].__setitem__("source", "ARCHIVE_IMPORTED")),
            ("knn_not_ready", lambda r: r["detail"]["knn_c"].__setitem__(
                "bank_status", "LOADING")),
            ("bad_vote", lambda r: r["detail"]["knn_c"].__setitem__(
                "vote_B", "NG")),
            ("not_ambiguous", lambda r: r["detail"]["confirmed_visual_recall"].__setitem__(
                "status", "NO_MATCH")),
        )
        for name, change in cases:
            with self.subTest(name=name):
                raw = self._oct30_case()
                change(raw)
                self.assertIsNone(self.decide(raw))

    def test_strong_human_ok_finishes_side_when_physical_probe_disagrees(self):
        data = self._oct30_case()
        for probe in (
            {
                **good_probe(),
                "status": "SUSPECT_ABSENCE",
                "scales": [
                    {**good_probe()["scales"][0], "suspected_absence": True},
                    good_probe()["scales"][1],
                ],
            },
            {"status": "UNAVAILABLE", "reason": "INVALID_PHYSICAL_ROI",
             "scales": [], "geometry": {}},
            {"status": "UNAVAILABLE", "reason": "GEOMETRY_OUT_OF_RANGE",
             "scales": [], "geometry": {}},
        ):
            with self.subTest(probe=probe["reason"]):
                visual = self.decide(data, probe=probe)
                self.assertIsNotNone(visual)
                self.assertEqual(visual["verdict"], "FALHA FALSA")
                self.assertEqual(
                    visual["detail"]["side_early_exit"]["analyzed_lights"],
                    ["SIDE"],
                )
                self.assertEqual(
                    visual["detail"]["side_early_exit"]["skipped_lights"],
                    ["TOP", "MID"],
                )
                self.assertTrue(visual["production_review_required"])
                self.assertTrue(verified_side_memory_visual_review(visual))
                self.assertIsNone(memory_first_side_decision(visual))
                self.assertEqual(
                    visual["detail"]["side_memory_first"]["physical_gate"],
                    "OPERATOR_REQUIRED",
                )
                policy = production_decision_policy(visual)
                self.assertEqual(policy["verdict"], "FALHA FALSA")
                self.assertFalse(policy["auto_allowed"])
                self.assertEqual(policy["proposed_decision"], "")
                self.assertEqual(
                    policy["reason"],
                    "side_strong_human_ok_physical_probe_requires_operator",
                )
                # O modo continuo NUNCA converte esse OK em NG preventivo.
                visual["detail"]["production_continuous_auto"] = True
                self.assertEqual(
                    production_decision_policy(visual)["verdict"],
                    "FALHA FALSA",
                )
                self.assertFalse(
                    production_decision_policy(visual)["auto_allowed"]
                )
                self.assertEqual(
                    production_decision_policy(visual)["proposed_decision"],
                    "",
                )

    def test_strong_human_ok_probe_exception_is_visually_ok_but_not_auto(self):
        data = self._oct30_case()
        def broken_probe(*_):
            raise RuntimeError("AOI_PROBE_FAILURE")
        visual = build_side_memory_first(
            data, "FALTANDO", object(), object(), probe=broken_probe,
        )
        self.assertIsNotNone(visual)
        self.assertEqual(visual["verdict"], "FALHA FALSA")
        self.assertTrue(visual["production_review_required"])
        self.assertEqual(
            visual["detail"]["side_memory_first"]["physical_probe"]["status"],
            "UNAVAILABLE",
        )
        self.assertEqual(
            visual["detail"]["side_memory_first"]["physical_probe"]["reason"],
            "RuntimeError",
        )
        self.assertFalse(production_decision_policy(visual)["auto_allowed"])

    def test_visual_review_marker_tampering_cannot_authorize_xp(self):
        data = self._oct30_case()
        suspect = good_probe()
        suspect["status"] = "SUSPECT_ABSENCE"
        suspect["scales"][0]["suspected_absence"] = True
        visual = self.decide(data, probe=suspect)
        self.assertTrue(verified_side_memory_visual_review(visual))
        for tamper in (
            lambda x: x["detail"]["side_memory_first"].__setitem__(
                "physical_gate", "VERIFIED"),
            lambda x: x["detail"]["side_memory_first"].__setitem__(
                "archive_source", "public/ok_archive/wrong.png"),
            lambda x: x["detail"]["side_early_exit"].__setitem__(
                "production_auto_allowed", True),
            lambda x: x.__setitem__("production_review_required", False),
        ):
            with self.subTest(tamper=tamper):
                altered = deepcopy(visual)
                tamper(altered)
                self.assertFalse(verified_side_memory_visual_review(altered))
                self.assertIsNone(memory_first_side_decision(altered))
                self.assertFalse(production_decision_policy(altered)["auto_allowed"])

    def test_strong_human_ok_never_bypasses_xp_provenance_gate(self):
        data = self._oct30_case()
        valid = self.decide(data, probe=good_probe(trim=True))
        self.assertIsNotNone(valid)
        for mutate in (
            lambda r: r["detail"]["side_memory_first"].__setitem__(
                "human_similarity", .999),
            lambda r: r["detail"]["side_memory_first"].__setitem__(
                "archive_source", "public/ok_archive/forged.png"),
            lambda r: r["detail"]["side_memory_first"]["physical_probe"][
                "scales"][0].__setitem__("suspected_absence", True),
        ):
            with self.subTest(mutate=mutate):
                attempt = deepcopy(valid)
                mutate(attempt)
                self.assertIsNone(memory_first_side_decision(attempt))
                self.assertFalse(production_decision_policy(attempt)["auto_allowed"])

    def test_memory_priority_covers_all_five_aoi_categories(self):
        categories = (
            "FALTANDO", "DESLOCADO", "INVERTIDO", "EMBORCADO", "MUITOADESIVO",
        )
        self.assertEqual(KNOWN_MEMORY_CATEGORIES, set(categories))
        for category in categories:
            for label in ("OK", "NG"):
                with self.subTest(category=category, label=label):
                    raw = case(label, category)
                    final = self.decide(raw, category=category)
                    self.assertIsNotNone(final)
                    self.assertEqual(final["multilight_category"], category)
                    self.assertEqual(final["detail"]["side_memory_first"]["physical_probe_scope"],
                                     PHYSICAL_PROBE_SCOPE)
                    self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"],
                                     ["TOP", "MID"])
                    self.assertEqual(memory_first_side_decision(final), label)
                    policy = production_decision_policy(final)
                    self.assertEqual(policy["proposed_decision"], label)
                    self.assertTrue(policy["auto_allowed"])
                    self.assertTrue(policy["unverified_physical_label"])
                    self.assertIn("side_operator_memory_first_97pct_auto_", policy["reason"])

    def test_category_aliases_route_to_same_canonical_memory(self):
        for alias, canonical in (
            ("MISALIGNED", "DESLOCADO"),
            ("TOMBSTONE", "EMBORCADO"),
            ("REVERSE", "INVERTIDO"),
            ("MISSING", "FALTANDO"),
            ("MUITO ADESIVO", "MUITOADESIVO"),
        ):
            with self.subTest(alias=alias):
                self.assertIsNotNone(self.decide(
                    case(category=canonical), category=alias,
                ))

    def test_unknown_category_or_category_conflict_never_auto_approves(self):
        for category in ("UNKNOWN", "DESCONHECIDO", "ABCNAOEXISTE", ""):
            with self.subTest(category=category):
                raw = case(category=category)
                self.assertIsNone(self.decide(raw, category=category))
        raw = case(category="INVERTIDO")
        self.assertIsNone(self.decide(raw, category="DESLOCADO"))
        raw["detail"]["cnn_v2_aoi_category"] = "FALTANDO"
        self.assertIsNone(self.decide(raw, category="INVERTIDO"))

    def test_generic_absence_veto_blocks_other_category_auto_ok(self):
        for category in ("DESLOCADO", "EMBORCADO", "INVERTIDO", "MUITOADESIVO"):
            with self.subTest(category=category):
                absent = good_probe()
                absent["status"] = "SUSPECT_ABSENCE"
                absent["scales"][0]["suspected_absence"] = True
                self.assertIsNone(self.decide(
                    case(category=category), probe=absent, category=category,
                ))
                # A ausência pode reforçar uma rejeição NG já confirmada
                # pela memória, mas nunca criar OK.
                ng = self.decide(case(label="NG", category=category),
                                 probe=absent, category=category)
                self.assertIsNotNone(ng)
                self.assertEqual(memory_first_side_decision(ng), "NG")

    def test_cross_category_probe_marker_is_required_before_xp(self):
        final = self.decide(case(category="DESLOCADO"), category="DESLOCADO")
        final["detail"]["side_memory_first"]["physical_probe_scope"] = "OTHER"
        self.assertIsNone(memory_first_side_decision(final))
        self.assertFalse(production_decision_policy(final)["auto_allowed"])

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
                "similarity", .9699)),
            ("weak_b", lambda x: x["detail"]["knn_c"]["neighbors_B"][0].__setitem__(
                "similarity", .9699)),
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
