"""Auditoria do acervo integral e matriz OK/NG separada, sem checkpoint real.

Nenhuma classe, categoria ou nome de arquivo recebe tratamento especial.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.services.aoi_v3_archive_batch import audit_full_archive
from src.scripts.audit_v3_full_archive import _summary_text, _write_reports


def _write(root, folder, filename, pixel_id, *, image=None):
    target = root / "public" / f"{folder.lower()}_archive" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if image is None:
        rng = np.random.default_rng(pixel_id)
        image = rng.integers(10, 250, (720, 1280, 3), dtype=np.uint8)
        image[45, 300, 0] = pixel_id
    assert cv2.imwrite(str(target), image)
    return target


class FakeExtractor:
    def __init__(self):
        self.calls = 0
    def __call__(self, frame):
        self.calls += 1
        return frame[30:400, :250], frame[30:400, 260:550], {
            "category": "FALTANDO",
        }


class FakeV3:
    def __init__(self):
        self.lights = []
    def inspect(self, reference, test, light):
        self.lights.append(light)
        # Este fake mede apenas se o script respeita scores produzidos,
        # sem usar rótulo de pasta/nome na predição.
        score = .9 if int(test[15, 40, 0]) >= 128 else .1
        return {
            "verdict": "DEFEITO REAL" if score >= .5 else "FALHA FALSA",
            "detail": {
                "cnn_v3_status": "INFERENCE_OK",
                "cnn_v3_checkpoint_verified": True,
                "cnn_v3_checkpoint_sha256": "a"*64,
                "cnn_v3_ng_score_uncalibrated": score,
            },
        }


class FullArchiveV3Tests(unittest.TestCase):
    def test_every_file_is_scored_and_both_errors_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root, "OK", "2026-10-02_1029_FALTANDO.png", 18)
            _write(root, "OK", "2026-10-02_1350_DESLOCADO.png", 220)
            _write(root, "NG", "2026-10-02_1701_FALTANDO_TOP.png", 228)
            _write(root, "NG", "2026-10-02_1702_INVERTIDO_MID.png", 25)

            extractor = FakeExtractor()
            predictor = FakeV3()
            progress = []
            result = audit_full_archive(
                root, extractor=extractor, predictor=predictor,
                progress=lambda i, n, row: progress.append((i, n, row["status"])),
            )
            self.assertEqual(extractor.calls, 4)
            self.assertEqual(predictor.lights, ["SIDE", "SIDE", "TOP", "MID"])
            self.assertEqual(len(progress), 4)
            self.assertTrue(result["all_files_evaluated"])
            self.assertEqual(result["coverage"]["png_in_inventory"], 4)
            self.assertEqual(result["summary"]["evaluated"], 4)
            self.assertEqual(result["summary"]["OK_correct"], 1)
            self.assertEqual(result["summary"]["OK_false_NG"], 1)
            self.assertEqual(result["summary"]["NG_correct"], 1)
            self.assertEqual(result["summary"]["NG_false_OK"], 1)
            self.assertIn("DESLOCADO_SIDE", result["by_category_light"])
            self.assertTrue(all(
                row["leave_one_out"]["own_image_excluded"]
                for row in result["cases"]
            ))
            self.assertEqual(len(result["classifier"]["v3_checkpoint_sha256"]), 1)
            self.assertIn("DIVERGÊNCIAS", _summary_text(result))
            output = _write_reports(result, root / "reports")
            self.assertEqual(len((output / "full_archive_v3.csv").read_text(
                encoding="utf-8-sig",
            ).splitlines()), 5)
            self.assertIn("cases", (output / "full_archive_v3.json").read_text(
                encoding="utf-8",
            ))
            self.assertEqual(len(list((root / "public/ok_archive").glob("*.png"))), 2)

    def test_invalid_png_is_reported_and_does_not_hide_the_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root, "OK", "2026-10-02_1029_FALTANDO.png", 18)
            broken = root / "public/ng_archive/2026-10-02_1350_DESLOCADO.png"
            broken.parent.mkdir(parents=True)
            broken.write_bytes(b"not a PNG")
            result = audit_full_archive(
                root, extractor=FakeExtractor(), predictor=FakeV3(),
            )
            self.assertEqual(result["coverage"]["png_in_inventory"], 2)
            self.assertEqual(len(result["cases"]), 2)
            self.assertFalse(result["all_files_evaluated"])
            self.assertEqual(result["summary"]["evaluated"], 1)
            self.assertEqual(result["summary"]["failed_or_unavailable"], 1)
            self.assertEqual(result["coverage"]["invalid_inventory_pngs"], 1)

    def test_conflicting_exact_png_bytes_are_not_valid_archive_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write(root, "OK", "2026-10-02_1029_FALTANDO.png", 18)
            pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
            _write(
                root, "NG", "2026-10-02_1350_FALTANDO.png", 18,
                image=pixels,
            )
            result = audit_full_archive(
                root, extractor=FakeExtractor(), predictor=FakeV3(),
            )
            self.assertEqual(
                len(result["coverage"]["cross_label_pixel_conflicts"]), 1,
            )
            self.assertTrue(all(
                x["full_frame_exact"]["conflict"] for x in result["cases"]
            ))
            self.assertFalse(any(
                x["leave_one_out"]["label_authorized"]
                for x in result["cases"]
            ))

    def test_invalid_checkpoint_fails_closed_instead_of_crediting_folder(self):
        class Unavailable:
            def inspect(self, reference, test, light):
                return {
                    "verdict": "REVISÃO OBRIGATÓRIA",
                    "detail": {
                        "cnn_v3_status": "ERROR",
                        "cnn_v3_checkpoint_verified": False,
                    },
                }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root, "OK", "2026-10-02_1029_FALTANDO.png", 18)
            result = audit_full_archive(
                root, extractor=FakeExtractor(), predictor=Unavailable(),
            )
            self.assertFalse(result["all_files_evaluated"])
            self.assertEqual(result["summary"]["evaluated"], 0)
            self.assertEqual(result["summary"]["failed_or_unavailable"], 1)

    def test_epoch5_strict_rejects_inactive_and_wrong_checkpoint(self):
        from src.core.neural.aoi_v3_epoch5_contract import SNAPSHOT_SHA256

        class FakeEpoch5(FakeV3):
            def __init__(self, epoch, sha):
                super().__init__()
                self.epoch, self.sha = epoch, sha

            def inspect(self, reference, test, light):
                vote = super().inspect(reference, test, light)
                vote["detail"]["cnn_v3_checkpoint_epoch"] = self.epoch
                vote["detail"]["cnn_v3_checkpoint_sha256"] = self.sha
                return vote

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root, "OK", "2026-10-02_1029_FALTANDO.png", 18)
            with patch(
                "src.core.neural.aoi_v3_epoch5_contract.epoch5_pointer_enabled",
                return_value=False,
            ):
                with self.assertRaisesRegex(ValueError, "não está ativada"):
                    audit_full_archive(
                        root, extractor=FakeExtractor(), predictor=FakeV3(),
                        require_epoch5=True,
                    )
            with patch(
                "src.core.neural.aoi_v3_epoch5_contract.epoch5_pointer_enabled",
                return_value=True,
            ), patch(
                "src.core.neural.aoi_v3_epoch5_contract.validate_epoch5_snapshot",
                return_value=({}, root / "fixture.pt"),
            ):
                correct = audit_full_archive(
                    root, extractor=FakeExtractor(),
                    predictor=FakeEpoch5(5, SNAPSHOT_SHA256),
                    require_epoch5=True,
                )
                self.assertTrue(correct["all_files_evaluated"])
                self.assertEqual(correct["classifier"]["v3_checkpoint_epoch"], [5])
                self.assertEqual(
                    correct["classifier"]["v3_checkpoint_sha256"],
                    [SNAPSHOT_SHA256],
                )
                wrong = audit_full_archive(
                    root, extractor=FakeExtractor(),
                    predictor=FakeEpoch5(3, "a"*64),
                    require_epoch5=True,
                )
                self.assertFalse(wrong["all_files_evaluated"])
                self.assertEqual(wrong["summary"]["failed_or_unavailable"], 1)
                self.assertIn("época 5", wrong["cases"][0]["issues"][0])

    def test_exact_replay_uses_xp_index_and_reports_label_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.random.default_rng(31).integers(
                10, 240, (720, 1280, 3), dtype=np.uint8,
            )
            _write(root, "OK", "2026-10-02_1029_FALTANDO_SIDE.png", 31, image=image)
            result = audit_full_archive(
                root, extractor=FakeExtractor(), predictor=FakeV3(),
                verify_runtime_archive=True,
            )
            recall = result["runtime_archive_exact_replay"]
            self.assertEqual(recall["scanned"], 1)
            self.assertEqual(recall["exact_unambiguous"], 1)
            self.assertEqual(recall["own_file_retrieved"], 1)
            self.assertEqual(recall["cross_label_conflicts"], 0)
            self.assertTrue(result["cases"][0]["runtime_archive_recall"][
                "exact_label_unambiguous"
            ])
            self.assertFalse(result["cases"][0]["runtime_archive_recall"][
                "production_decision_authorized"
            ])
            self.assertIn("ÍNDICE EXATO DO XP", _summary_text(result))

            # Mesmo PNG original no rótulo oposto: NO LABEL SHALL WIN.
            _write(root, "NG", "2026-10-02_1030_FALTANDO_SIDE.png", 31, image=image)
            conflict = audit_full_archive(
                root, extractor=FakeExtractor(), predictor=FakeV3(),
                verify_runtime_archive=True,
            )
            replay = conflict["runtime_archive_exact_replay"]
            self.assertEqual(replay["scanned"], 2)
            self.assertEqual(replay["own_file_retrieved"], 2)
            self.assertEqual(replay["exact_unambiguous"], 0)
            self.assertEqual(replay["cross_label_conflicts"], 2)
            self.assertEqual(replay["not_exact_or_ambiguous"], 2)
            self.assertTrue(all(
                item["runtime_archive_recall"]["status"] == "EXACT_FRAME_CONFLICT"
                for item in conflict["cases"]
            ))

    def test_exact_only_never_claims_super_similar_is_exact(self):
        from src.services.archive_visual_presence import ArchiveVisualPresenceIndex
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.random.default_rng(41).integers(
                10, 240, (720, 1280, 3), dtype=np.uint8,
            )
            _write(root, "OK", "2026-10-02_1029_FALTANDO_SIDE.png", 41, image=image)
            index = ArchiveVisualPresenceIndex(
                ok_dir=root / "public/ok_archive",
                ng_dir=root / "public/ng_archive",
            )
            exact = index.inspect(image, "FALTANDO", "SIDE", exact_only=True)
            self.assertEqual(exact["status"], "EXACT_FRAME")
            almost = image.copy()
            almost[100, 300, 0] ^= 1
            checked = index.inspect(almost, "FALTANDO", "SIDE", exact_only=True)
            self.assertEqual(checked["status"], "NO_EXACT_FRAME")
            self.assertFalse(checked["decision_authorized"])
            self.assertEqual(checked["exact_frame_labels"], [])

    def test_inventoried_png_changed_while_running_invalidates_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write(root, "OK", "2026-10-02_1029_FALTANDO.png", 18)
            def mutate(i, total, row):
                if i == total:
                    new = cv2.imread(str(path))
                    new[4, 8, 0] ^= 255
                    cv2.imwrite(str(path), new)
            result = audit_full_archive(
                root, extractor=FakeExtractor(), predictor=FakeV3(),
                progress=mutate,
            )
            self.assertFalse(result["all_files_evaluated"])
            self.assertEqual(
                result["coverage"]["source_changed_during_run"],
                ["public/ok_archive/2026-10-02_1029_FALTANDO.png"],
            )


if __name__ == "__main__":
    unittest.main()
