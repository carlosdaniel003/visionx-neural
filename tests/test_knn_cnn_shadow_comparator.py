"""Simulação A/B seletiva offline: nunca deslocar falso NG para OK automaticamente."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.services.startup_regression.knn_cnn_shadow_comparator import (
    SCHEMA, compare, save_report,
)


def fixture(rows=None):
    if rows is None:
        rows = [
            # label, CNN, KNN, resposta CNN
            ("OK", "NG", "SUGERE_OK", "ERRO_CLASSIFICACAO"),
            ("NG", "OK", "SUGERE_NG", "ERRO_CLASSIFICACAO"),
            ("NG", "NG", "SUGERE_OK", "PASSOU"),
            ("OK", "OK", "SUGERE_OK", "PASSOU"),
            ("NG", "NG", "SUGERE_NG", "PASSOU"),
            ("OK", None, "SUGERE_OK", "FORA_ESCOPO"),
            ("NG", None, "SUGERE_NG", "REVISAO"),
        ]
    cnn, knn = [], []
    for i, (expected, prediction, suggested, status) in enumerate(rows):
        filehash = format(i + 1, "064x")
        pixelhash = format(i + 100, "064x")
        path = f"public/{'ok_archive' if expected == 'OK' else 'ng_archive'}/case_{i}.png"
        category = "MUITO ADESIVO" if status == "FORA_ESCOPO" else "FALTANDO"
        cnn.append({
            "model": "CNN_FALTANDO_V2", "source_path": path,
            "source_file_sha256": filehash, "source_pixel_sha256": pixelhash,
            "category": category, "lighting_mode": "SIDE",
            "expected_label": expected,
            "predicted_label": prediction, "status": status,
            "cnn_score_ng": 0.99 if prediction == "NG" else 0.01,
            "checkpoint_sha256": "d" * 64,
        })
        knn.append({
            "source_path": path, "source_file_sha256": filehash,
            "source_pixel_sha256": pixelhash, "category_hint": category,
            "lighting_mode": "SIDE", "expected_label": expected,
            "status": "PASSOU", "prediction": suggested.removeprefix("SUGERE_"),
            "neighbors": [{"source_path": "other_"+str(i),
                           "source_pixel_sha256": "f"*64}],
            "shadow_candidate": {
                "status": suggested,
                "diagnostic_only": True, "auto_decision_authorized": False,
            },
        })
    return ({
        "schema":"visionx.interactive_archive_validation.v1",
        "root":"/test/odin", "complete":True, "canceled":False,
        "archive_png_count":len(rows), "cases":cnn,
    }, {
        "schema":"visionx.knn_archive_visual_replay.v1",
        "root":"/test/odin", "complete":True, "canceled":False,
        "inventory_png_count":len(rows), "protocol":"visual_holdout",
        "uses_cnn":False, "is_production_verified_exact_knn_test":False,
        "cases":knn,
    })


class ComparatorTests(unittest.TestCase):
    def test_guarded_conflict_only_review_and_separate_unsafe_counterfactual(self):
        a,b=fixture()
        result=compare(a,b)
        self.assertEqual(result["schema"],SCHEMA)
        self.assertFalse(result["production_approved"])
        self.assertFalse(result["xp_command_authorized"])
        self.assertFalse(result["production_route_reproduced"])
        self.assertEqual(result["archive_pngs"],7)
        self.assertEqual(result["cnn_eligible_pngs"],6)
        base=result["results"]["cnn_isolated_baseline"]
        guarded=result["results"]["guarded_conflict_to_review"]
        unsafe=result["results"]["optimistic_knn_override_hypothetical"]
        self.assertEqual(base["ng_as_ok"],1)
        self.assertEqual(base["ok_as_ng"],1)
        self.assertEqual(guarded["ng_as_ok"],0)
        self.assertEqual(guarded["ok_as_ng"],0)
        self.assertEqual(unsafe["ng_as_ok"],1)
        self.assertEqual(unsafe["ok_as_ng"],0)
        self.assertEqual(result["conflicts"]["total"],3)
        self.assertEqual(result["conflicts"]["false_ng_possible_review_rescue"],1)
        self.assertEqual(result["conflicts"]["false_ok_blocked_from_auto_release"],1)
        history = result["false_ng_investigation"]
        self.assertEqual(history["count"], 1)
        self.assertEqual(history["possible_review_rescues"], 1)
        self.assertFalse(history["auto_ok_authorized"])
        self.assertEqual(len(history["cases"]), 1)
        self.assertIsNone(history["cases"][0]["neighbors"][0]["label"])
        self.assertIn("source_path", history["cases"][0])
        self.assertTrue(history["nearest_neighbor_groups"])
        self.assertFalse(result["xp_command_authorized"])
        self.assertEqual(result["rows"][0]["guarded_label"],None)
        self.assertTrue(result["evidence_contract"]["original_png_hashes_verified_between_reports"])
        self.assertEqual(result["by_category"]["FALTANDO"]["guarded"]["ng_as_ok"],0)
        self.assertEqual(result["by_lighting"]["SIDE"]["guarded"]["ng_as_ok"],0)

    def test_neighbor_label_is_propagated_only_when_source_provides_it(self):
        a, b = fixture()
        b["cases"][0]["neighbors"][0]["label"] = "OK"
        b["cases"][0]["neighbors"][0]["similarity"] = 0.9874
        report = compare(a, b)
        neighbor = report["false_ng_investigation"]["cases"][0]["neighbors"][0]
        self.assertEqual(neighbor["label"], "OK")
        self.assertEqual(neighbor["similarity"], 0.9874)
        self.assertFalse(report["false_ng_investigation"]["auto_ok_authorized"])

    def test_unsupervised_knn_does_not_create_new_verdict_when_cnn_in_review(self):
        a,b=fixture([("NG",None,"SUGERE_OK","REVISAO")])
        r=compare(a,b)
        self.assertIsNone(r["rows"][0]["guarded_label"])
        self.assertIsNone(r["rows"][0]["optimistic_label"])
        self.assertEqual(r["results"]["guarded_conflict_to_review"]["review"],1)

    def test_diff_hash_and_missing_hash_cannot_join(self):
        a,b=fixture()
        b["cases"][0]["source_pixel_sha256"]="e"*64
        with self.assertRaisesRegex(ValueError,"Hashes"):
            compare(a,b)
        b["cases"][0]["source_pixel_sha256"]=""
        with self.assertRaisesRegex(ValueError,"Hashes"):
            compare(a,b)

    def test_self_match_and_duplicate_capture_never_become_evidence(self):
        a,b=fixture()
        b["cases"][0]["neighbors"][0]["source_path"]=a["cases"][0]["source_path"]
        with self.assertRaisesRegex(ValueError,"Vazamento"):
            compare(a,b)
        b["cases"][0]["neighbors"][0]["source_path"]="different"
        b["cases"][0]["neighbors"][0]["source_pixel_sha256"]=a["cases"][0]["source_pixel_sha256"]
        with self.assertRaisesRegex(ValueError,"Vazamento"):
            compare(a,b)

    def test_autopromotion_flag_is_rejected(self):
        a,b=fixture()
        b["cases"][0]["shadow_candidate"]["auto_decision_authorized"]=True
        with self.assertRaisesRegex(ValueError,"autorização"):
            compare(a,b)

    def test_mismatched_files_label_light_category_and_root_are_rejected(self):
        for tweak in ("label","light","category","root","path"):
            a,b=fixture()
            if tweak=="label": b["cases"][0]["expected_label"]="NG"
            if tweak=="light": b["cases"][0]["lighting_mode"]="MID"
            if tweak=="category": b["cases"][0]["category_hint"]="DESLOCADO"
            if tweak=="root": b["root"]="/other"
            if tweak=="path": b["cases"][0]["source_path"]="other.png"
            with self.subTest(tweak=tweak),self.assertRaises(ValueError):
                compare(a,b)

    def test_partial_inputs_and_old_reports_missing_digests_rejected(self):
        a,b=fixture()
        a["complete"]=False
        with self.assertRaises(ValueError):
            compare(a,b)
        a["complete"]=True
        del a["cases"][0]["source_file_sha256"]
        with self.assertRaises(ValueError):
            compare(a,b)

    def test_invalid_png_both_sources_skipped_with_coverage(self):
        a,b=fixture()
        a["cases"][0].update(status="INVALIDO",
                            source_file_sha256="",source_pixel_sha256="")
        b["cases"][0].update(status="INVALIDO",
                            source_file_sha256="",source_pixel_sha256="")
        r=compare(a,b)
        self.assertEqual(r["archive_pngs"],7)
        self.assertEqual(r["valid_crosschecked_pngs"],6)
        self.assertEqual(r["invalid_pngs"],1)
        self.assertEqual(len(r["evidence_contract"]["invalid_pngs_excluded_by_path"]),1)

    def test_comparator_does_not_need_expected_label_to_choose_candidate(self):
        a,b=fixture()
        r=compare(a,b)
        # Os rótulos são usados APENAS para medir o erro após decisões.
        # Trocar a classe de origem sem mudar previsão não pode alterar voto.
        a["cases"][0]["expected_label"]="NG"
        b["cases"][0]["expected_label"]="NG"
        r2=compare(a,b)
        self.assertEqual(
            r["rows"][0]["guarded_label"],r2["rows"][0]["guarded_label"]
        )
        self.assertEqual(
            r["rows"][0]["optimistic_label"],r2["rows"][0]["optimistic_label"]
        )

    def test_save_report_in_reports_and_no_mutation(self):
        a,b=fixture()
        result=compare(a,b)
        original=copy.deepcopy(result)
        with tempfile.TemporaryDirectory() as temp:
            jp,tp=save_report(result,Path(temp))
            self.assertIn("/reports/startup_regression/",jp.as_posix())
            self.assertEqual(json.loads(jp.read_text(encoding="utf-8"))["schema"],SCHEMA)
            self.assertIn("NG->OK",tp.read_text(encoding="utf-8"))
        self.assertEqual(result,original)

    def test_worker_cli_starts_without_missing_function_import(self):
        # py_compile não detecta import de um nome inexistente: exercitar
        # o entrypoint real, como o botão da interface faz no Windows.
        proc = subprocess.run(
            [sys.executable, "-m",
             "src.services.startup_regression.knn_cnn_shadow_worker",
             "--help"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=40,
        )
        self.assertEqual(
            proc.returncode, 0,
            "Worker não inicia: " + proc.stderr[-1500:],
        )
        self.assertIn("--cancel-file", proc.stdout)
        self.assertNotIn("ImportError", proc.stderr)

    def test_worker_ui_wiring_never_uses_actuation_or_training(self):
        ui=Path("src/ui/training_validation_panel.py").read_text(encoding="utf-8")
        wk=Path("src/services/startup_regression/knn_cnn_shadow_worker.py").read_text(encoding="utf-8")
        comp=Path("src/services/startup_regression/knn_cnn_shadow_comparator.py").read_text(encoding="utf-8")
        self.assertIn("Comparar KNN + CNN (experimental)",ui)
        self.assertIn("knn_cnn_shadow_worker",ui)
        self.assertIn("replay_knn_archive_visual(",wk)
        self.assertIn("validate(",wk)
        self.assertIn("inventory=inv",wk)
        self.assertNotIn("send_xp",wk+comp)
        self.assertNotIn("save_label(",wk+comp)
        self.assertNotIn("train(",wk+comp)


if __name__=="__main__":
    unittest.main()
