"""Etapa 2: valida replay SIDE sem KNN, sem escrita e sem abrir ODIN."""

import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from src.services.startup_regression.side_replay import (
    run_side_replay,
    verdict_status,
    write_side_report,
)
from src.services.startup_regression.inspection_runner import (
    ReplayError,
    SideInspectionRunner,
    create_physical_orchestrator,
)
from src.services.screen_monitor import ScreenMonitor


class SideReplayContractTests(TestCase):
    def test_verdict_rules_are_strict_and_review_fails(self):
        self.assertEqual(verdict_status("OK", "FALHA FALSA"), "PASSOU")
        self.assertEqual(verdict_status("NG", "DEFEITO REAL"), "PASSOU")
        self.assertEqual(verdict_status("NG", "FALHA FALSA"), "REGRESSAO")
        self.assertEqual(verdict_status("OK", "DEFEITO REAL"), "REGRESSAO")
        self.assertEqual(
            verdict_status("OK", "FALHA FALSA", review=True), "REGRESSAO"
        )
        self.assertEqual(verdict_status("NG", "REVISÃO OBRIGATÓRIA"), "REGRESSAO")
        self.assertEqual(verdict_status("OK", "INVALID"), "INVALIDO")

    def test_physical_factory_never_constructs_knn(self):
        with patch(
            "src.core.experts.knn_expert.KNNExpert.__init__",
            side_effect=AssertionError("KNN está proibido"),
        ):
            orchestrator = create_physical_orchestrator()
        self.assertNotIn("knn", orchestrator.experts)

    def test_runner_refuses_knn_even_if_stub_returns_success(self):
        class Bad:
            experts = {"knn": object()}
        with self.assertRaises(ReplayError):
            SideInspectionRunner(orchestrator=Bad(), monitor=object())

    def test_preprocess_uses_actual_color_bars_and_full_images_without_writes(self):
        # Três faixas da interface são geradas localmente; o extrator da
        # produção identifica barra azul, barra vermelha e regiões completas.
        frame = np.full((840, 1165, 3), 195, dtype=np.uint8)
        frame[280:300, 100:500] = (255, 0, 0)
        frame[280:300, 620:1020] = (0, 0, 255)
        frame[300:600, 100:500] = (15, 60, 80)
        frame[300:600, 620:1020] = (15, 60, 80)
        frame[400:500, 820:900] = (0, 0, 200)

        class PhysicalStub:
            experts = {}
            def inspect(self, sample, test, _raw, info, _box, _focus):
                if not info.get("_replay_without_memory"):
                    raise AssertionError("flag de isolamento ausente")
                if sample.shape[0] < 100 or test.shape[0] < 100:
                    raise AssertionError("recorte não incluiu a imagem inteira")
                return {
                    "is_defect": True,
                    "verdict": "DEFEITO REAL",
                    "confidence": 0.9,
                    "active_engines": ["shift_expert.py"],
                    "detail": {
                        "replay_memory_consulted": False,
                        "final_score": 0.9,
                        "physical_score": 0.9,
                        "fusion_rule": "physical_only",
                        "decision_trace": {
                            "weights": {"physical": 1.0, "knn": 0.0},
                            "memory": {"has_memory": False},
                            "operator_review_required": False,
                        },
                    },
                }

        monitor = ScreenMonitor()
        monitor._extract_text_info = lambda *_: {
            "board": "BOARD-1",
            "parts": "U2~5",
            "value": "0 <= 54.872 <= 10 FALTANDO",
        }
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "legacy_FALTANDO.png"
            self.assertTrue(cv2.imwrite(str(png), frame))
            runner = SideInspectionRunner(orchestrator=PhysicalStub(), monitor=monitor)
            with (
                patch(
                    "src.services.startup_regression.inspection_runner.screen_module.HAS_TESSERACT",
                    True,
                ),
                patch.object(
                    monitor, "_write_debug_crop",
                    side_effect=AssertionError("Writer de debug acionado"),
                ),
            ):
                decision = runner.inspect_png(png, "FALTANDO")

        self.assertEqual(decision["verdict"], "DEFEITO REAL")
        self.assertEqual(decision["category"], "FALTANDO")
        self.assertFalse(decision["memory_consulted"])
        self.assertFalse(decision["knn_enabled"])
        self.assertGreater(decision["test_dimensions"][0], 100)

    def test_side_replay_filters_explicit_multilight_and_reports_regressions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ok = root / "public" / "ok_archive"
            ng = root / "public" / "ng_archive"
            ok.mkdir(parents=True)
            ng.mkdir(parents=True)
            image = np.full((40, 60, 3), 90, dtype=np.uint8)
            cv2.imwrite(str(ok / "2026-10-01_0840_FALTANDO.png"), image)
            image[10:15] = 130
            cv2.imwrite(str(ng / "2026-10-01_0850_INVERTIDO.png"), image)
            for light in ("SIDE", "TOP", "MID"):
                image[5:10] += 1
                cv2.imwrite(
                    str(ok / f"2026-10-08_0830_FALTANDO_{light}.png"),
                    image,
                )

            class Stub:
                def inspect_png(self, path, expected_category):
                    return {
                        "verdict": "FALHA FALSA",
                        "requires_review": False,
                        "memory_consulted": False,
                        "knn_enabled": False,
                    }
            result = run_side_replay(root, runner=Stub())
            self.assertEqual(result["summary"]["total"], 2)
            self.assertEqual(result["summary"]["passed"], 1)
            self.assertEqual(result["summary"]["regressions"], 1)
            self.assertEqual(result["multilight_explicit_deferred"], 3)
            self.assertFalse(result["is_operational_gate"])
            self.assertEqual(
                [r["status"] for r in result["cases"]],
                ["PASSOU", "REGRESSAO"],
            )
            out, txt = write_side_report(
                result, root / "reports" / "startup_regression"
            )
            self.assertTrue(out.is_file())
            self.assertIn("KNN E MEMÓRIA", txt.read_text(encoding="utf-8"))
            self.assertEqual(len(list(ok.glob("*.png"))), 4)
            self.assertEqual(len(list(ng.glob("*.png"))), 1)

    def test_invalid_ocr_or_model_failure_is_not_marked_ok(self):
        class Bad:
            def inspect_png(self, _path, _category):
                raise ReplayError("OCR indisponível")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "public" / "ok_archive").mkdir(parents=True)
            (root / "public" / "ng_archive").mkdir(parents=True)
            image = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.imwrite(
                str(root / "public" / "ok_archive" / "2026-10-01_0800_FALTANDO.png"),
                image,
            )
            result = run_side_replay(root, runner=Bad())
            self.assertEqual(result["summary"]["invalid"], 1)
            self.assertEqual(result["summary"]["passed"], 0)
            self.assertIn("OCR indisponível", result["cases"][0]["error"])


if __name__ == "__main__":
    import unittest
    unittest.main()
