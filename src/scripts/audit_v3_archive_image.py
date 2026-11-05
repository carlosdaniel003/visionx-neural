"""Auditoria somente leitura: imagem PNG original versus captura MSS na CNN V3.

Não altera pesos, rótulos, imagens, memória KNN ou política industrial.
O rótulo da pasta é histórico e não é uma prova de qualidade física.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import BASE_DIR
from src.services.image_archive_dedup import (
    archive_category_from_filename, image_fingerprint,
)
from src.services.archive_rendered_source_match import compare_rendered_source


def _read(path: Path) -> np.ndarray:
    frame = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR,
    )
    if frame is None or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("PNG não pode ser decodificado em BGR")
    return frame


def audit_image(root: Path, archive_file: Path, *,
                mss_file: Path | None = None, extractor=None,
                predictor=None) -> dict:
    root = Path(root).resolve()
    archive = (root / archive_file).resolve()
    folder = archive.parent
    labels = {
        (root / "public" / "ok_archive").resolve(): "OK",
        (root / "public" / "ng_archive").resolve(): "NG",
    }
    if folder not in labels or archive.suffix.lower() != ".png":
        raise ValueError("Origem deve ser um PNG nas pastas públicas OK/NG")
    scope = archive_category_from_filename(archive)
    if not scope:
        raise ValueError("Categoria/iluminação não reconhecida no nome do arquivo")
    light = scope.rsplit("_", 1)[-1]
    if light not in {"SIDE", "TOP", "MID"}:
        raise ValueError("Iluminação arquivada não reconhecida")
    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    if predictor is None:
        from src.core.neural.aoi_quality_cnn_v3_primary import AOIQualityV3Primary
        predictor = AOIQualityV3Primary(root=root)

    def run(frame):
        gab, test, info = extractor(frame)
        analysis = predictor.inspect(gab, test, light)
        detail = analysis.get("detail", {})
        return {
            "extraction_category": str(info.get("category", "") or ""),
            "crop_geometry": dict(
                getattr(getattr(extractor, "monitor", None),
                        "last_crop_geometry", {}) or {}
            ),
            "reference_shape": list(gab.shape),
            "test_shape": list(test.shape),
            "reference_pixel_sha256": image_fingerprint(gab),
            "test_pixel_sha256": image_fingerprint(test),
            "verdict": analysis.get("verdict", ""),
            "v3_status": detail.get("cnn_v3_status", ""),
            "v3_checkpoint_sha256": detail.get("cnn_v3_checkpoint_sha256", ""),
            "v3_ng_score_uncalibrated": detail.get(
                "cnn_v3_ng_score_uncalibrated",
            ),
        }

    original = _read(archive)
    result = {
        "schema": "visionx.v3_archive_image_audit.v1",
        "read_only": True,
        "source": str(archive.relative_to(root)).replace("\\", "/"),
        "archive_label": labels[folder],
        "archive_label_physical_quality_verified": False,
        "light": light,
        "archive_file_sha256": sha256(archive.read_bytes()).hexdigest(),
        "archive_pixel_sha256": image_fingerprint(original),
        "original": run(original),
    }
    original_vote = result["original"]
    predicted = (
        "NG" if original_vote["verdict"] == "DEFEITO REAL" else
        "OK" if original_vote["verdict"] == "FALHA FALSA" else ""
    )
    result["original_model_vs_folder"] = (
        "AGREES" if predicted == labels[folder] else
        "DISAGREES" if predicted else "MODEL_UNAVAILABLE"
    )
    if mss_file is not None:
        screenshot_path = (root / mss_file).resolve()
        captured = _read(screenshot_path)
        result["mss"] = {
            "source": str(screenshot_path),
            "pixel_sha256": image_fingerprint(captured),
            "raw_frame_identical": image_fingerprint(original)
            == image_fingerprint(captured),
            "rendered_source_registration": compare_rendered_source(
                original, captured,
            ),
            "model": run(captured),
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnosticar PNG OK/NG histórico contra a CNN V3 real",
    )
    parser.add_argument("--root", type=Path, default=BASE_DIR)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--mss", type=Path, default=None)
    args = parser.parse_args(argv)
    result = audit_image(
        args.root, args.archive, mss_file=args.mss,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
