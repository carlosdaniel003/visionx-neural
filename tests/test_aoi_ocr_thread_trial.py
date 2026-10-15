"""Safeguards for ABBA Tesseract OCR benchmark; never change live ODIN."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.services import aoi_ocr_thread_trial as trial


FIELDS = {
    "board": "P22-22200 (PRINCIPAL) L13",
    "parts": "D10~5",
    "value": "0 <= 5 <= 10 FALTANDO",
    "category": "FALTANDO",
}


def _sample(ms=120, *, fields=None, parts_valid=True):
    return {
        "fields": dict(fields or FIELDS),
        "elapsed_ms": ms,
        "tesseract_calls": 1,
        "tesseract_ms": ms,
        "parts_reason": "valid_general_reference",
        "parts_syntax_valid": parts_valid,
    }


def _worker_result(mode, job, ms=120, *, fields=None):
    return {
        "arm": mode,
        "omp_thread_limit": "1" if mode == "omp1" else None,
        "cases": [
            {"sha256": source["sha256"],
             "measurements": [
                 _sample(ms, fields=fields) for _ in range(job["repeats"])
             ]}
            for source in job["cases"]
        ],
    }


class OcrThreadTrialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "image.png"
        self.source.write_bytes(b"original sample for mocked trial")
        self.job = trial.make_job([(self.source, FIELDS)], repeats=2)

    def test_runs_abba_sequentially_with_process_only_omp_cap(self):
        calls = []
        parent_before = dict(os.environ)

        def fake_runner(cmd, **kw):
            self.assertEqual(cmd[:3], [
                trial.sys.executable, "-m",
                "src.services.aoi_ocr_thread_trial",
            ])
            self.assertTrue(kw["capture_output"])
            self.assertFalse(kw["check"])
            self.assertGreaterEqual(kw["timeout"], 1)
            self.assertEqual(kw["env"]["PYTHONIOENCODING"], "utf-8")
            self.assertEqual(kw["encoding"], "utf-8")
            self.assertEqual(kw["errors"], "replace")
            job = json.loads(
                Path(cmd[cmd.index("--_job") + 1]).read_text("utf-8")
            )
            output = Path(cmd[cmd.index("--_response") + 1])
            arm = cmd[cmd.index("--_mode") + 1]
            calls.append(arm)
            self.assertEqual(
                kw["env"].get("OMP_THREAD_LIMIT"),
                "1" if arm == "omp1" else None,
            )
            output.write_text(
                json.dumps(_worker_result(
                    arm, job, 40 if arm == "omp1" else 120,
                )),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        reports = trial.run_isolated_arms(
            self.job, worker_runner=fake_runner,
        )
        self.assertEqual(calls, ["default", "omp1", "omp1", "default"])
        self.assertEqual(dict(os.environ), parent_before)
        data = trial.summarize_trial(self.job, reports)
        self.assertEqual(data["default_median_ms"], 120)
        self.assertEqual(data["omp1_median_ms"], 40)
        self.assertAlmostEqual(data["speedup_pct"], 66.67, places=2)
        self.assertEqual(data["different_from_baseline_cases"], 0)
        self.assertEqual(data["human_mismatches"]["omp1"], 0)
        self.assertFalse(data["eligible_for_automatic_activation"])
        self.assertFalse(data["auto_promotion"])
        self.assertTrue(data["no_production_change"])
        self.assertEqual(
            self.source.read_bytes(), b"original sample for mocked trial"
        )

    def test_child_prints_unicode_even_when_parent_uses_cp1252(self):
        """Integration: redirected stdout on Windows must not crash on emoji."""
        import sys
        from unittest.mock import patch

        # Reproduce the corporate PowerShell configuration that triggered
        # UnicodeEncodeError on import of screen_monitor in child processes.
        with patch.dict(os.environ, {"PYTHONIOENCODING": "cp1252"}):
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.run(
                [sys.executable, "-c", "print('\\u2705 Tesseract encontrado')"],
                env=env, check=False, capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("\\u2705 Tesseract encontrado", proc.stdout)
            self.assertEqual(os.environ["PYTHONIOENCODING"], "cp1252")

    def test_mismatched_candidate_is_not_accepted_even_when_faster(self):
        runs = [
            _worker_result("default", self.job, 120),
            _worker_result("omp1", self.job, 40, fields={
                **FIELDS, "parts": "D11~5"
            }),
            _worker_result("omp1", self.job, 40, fields={
                **FIELDS, "parts": "D11~5"
            }),
            _worker_result("default", self.job, 120),
        ]
        result = trial.summarize_trial(self.job, runs)
        self.assertEqual(result["different_from_baseline_cases"], 1)
        self.assertEqual(result["human_mismatches"]["omp1"], 1)
        self.assertFalse(result["cases"][0]["different_fields"] is False)
        self.assertFalse(result["eligible_for_automatic_activation"])

    def test_unstable_baseline_and_missing_human_truth_stay_unverified(self):
        job = trial.make_job([(self.source, None)], repeats=1)
        different = dict(FIELDS, parts="D12~5")
        runs = [
            _worker_result("default", job, 130),
            _worker_result("omp1", job, 60),
            _worker_result("omp1", job, 55),
            _worker_result("default", job, 110, fields=different),
        ]
        result = trial.summarize_trial(job, runs)
        self.assertEqual(result["human_labels_supplied"], 0)
        self.assertEqual(result["baseline_unstable_cases"], 1)
        self.assertEqual(result["different_from_baseline_cases"], 1)
        self.assertFalse(result["eligible_for_automatic_activation"])

    def test_missing_worker_response_and_wrong_hash_fail_closed(self):
        outputs = [
            _worker_result("default", self.job, 120),
            _worker_result("omp1", self.job, 60),
            _worker_result("omp1", self.job, 60),
            _worker_result("default", self.job, 120),
        ]
        outputs[1]["cases"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA"):
            trial.summarize_trial(self.job, outputs)

    def test_invalid_arm_and_bad_worker_mode_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            trial.run_isolated_arms(
                self.job, sequence=("invalid",), worker_runner=lambda *a, **k: None
            )
        with self.assertRaises(ValueError):
            trial.make_job([(self.source, FIELDS)], repeats=0)
        with self.assertRaises(ValueError):
            trial.make_job([(self.source, FIELDS)], repeats=9)

    def test_no_overwrite_even_if_worker_returns_success(self):
        out = self.root / "old.json"
        out.write_text('{"intact":true}', encoding="utf-8")
        with patch(
            "src.services.aoi_ocr_benchmark._image_paths",
            return_value=[(self.source, None)],
        ), patch.object(
            trial, "run_isolated_arms", side_effect=AssertionError("invoked"),
        ):
            with self.assertRaises(FileExistsError):
                trial.main([
                    "--images", str(self.source),
                    "--output", str(out),
                ])
        self.assertEqual(out.read_text("utf-8"), '{"intact":true}')


if __name__ == "__main__":
    unittest.main()
