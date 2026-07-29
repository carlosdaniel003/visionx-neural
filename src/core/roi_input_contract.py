"""Contrato único de entrada da ROI para todos os especialistas.

O módulo não altera a captura do epicentro. Ele audita se cada motor recebeu a
mesma caixa escolhida pelo ``EpicenterExtractor`` e corrige duas ambiguidades de
debug:

* o SSIM só pode mostrar mapa de adesivo em categoria de adesivo;
* o comparador estrutural registra também a ROI bruta antes do alinhamento.
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np
from skimage.metrics import structural_similarity as structural_similarity


ADHESIVE_CATEGORIES = {
    "MUCH ADHESIVE",
    "MUITO ADESIVO",
    "EXCESS ADHESIVE",
    "ADESIVO EM EXCESSO",
}


def _normalized_category(value: str) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _box_tuple(value):
    if not value or len(value) < 4:
        return None
    try:
        return tuple(int(round(float(item))) for item in value[:4])
    except (TypeError, ValueError):
        return None


def _safe_pair(reference, test):
    if not isinstance(reference, np.ndarray) or not isinstance(test, np.ndarray):
        return None, None
    if reference.size == 0 or test.size == 0:
        return None, None
    reference = reference.copy()
    test = test.copy()
    if reference.shape != test.shape:
        test = cv2.resize(
            test,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return reference, test


def _canonical_roi(full_reference, full_test, epicenters):
    reference, test = _safe_pair(full_reference, full_test)
    if reference is None or not epicenters:
        return None, None, None
    box = _box_tuple(epicenters[0])
    if box is None:
        return None, None, None
    x, y, width, height = box
    image_height, image_width = reference.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + max(1, width))
    y2 = min(image_height, y + max(1, height))
    if x2 <= x1 or y2 <= y1:
        return None, None, None
    reference_roi = reference[y1:y2, x1:x2].copy()
    test_roi = test[y1:y2, x1:x2].copy()
    if reference_roi.size == 0 or test_roi.size == 0:
        return None, None, None
    return reference_roi, test_roi, (x1, y1, x2 - x1, y2 - y1)


def _crc32(image) -> str:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return ""
    return f"{zlib.crc32(np.ascontiguousarray(image).tobytes()) & 0xFFFFFFFF:08x}"


def _image_audit(image, canonical):
    if not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "available": False,
            "shape_match": False,
            "exact_match": False,
            "mean_abs_error": None,
            "crc32": "",
        }
    shape_match = bool(image.shape == canonical.shape)
    exact_match = bool(shape_match and np.array_equal(image, canonical))
    mean_abs_error = None
    if shape_match:
        mean_abs_error = float(
            np.mean(
                np.abs(image.astype(np.float32) - canonical.astype(np.float32))
            )
        )
    return {
        "available": True,
        "shape": [int(value) for value in image.shape],
        "shape_match": shape_match,
        "exact_match": exact_match,
        "mean_abs_error": mean_abs_error,
        "crc32": _crc32(image),
    }


def _engine_audit(box, image, canonical_box, canonical_test):
    resolved_box = _box_tuple(box)
    return {
        "box": list(resolved_box) if resolved_box else None,
        "box_match": bool(resolved_box == canonical_box),
        "test_input": _image_audit(image, canonical_test),
    }


def _generic_ssim_heatmap(expert, reference, test):
    reference, test = _safe_pair(reference, test)
    if reference is None:
        return None
    gray_reference = cv2.cvtColor(
        cv2.GaussianBlur(reference, (3, 3), 0),
        cv2.COLOR_BGR2GRAY,
    )
    gray_test = cv2.cvtColor(
        cv2.GaussianBlur(test, (3, 3), 0),
        cv2.COLOR_BGR2GRAY,
    )
    minimum = min(gray_reference.shape[:2])
    if minimum < 7:
        scale = 7.0 / max(minimum, 1)
        size = (
            max(7, int(round(gray_reference.shape[1] * scale))),
            max(7, int(round(gray_reference.shape[0] * scale))),
        )
        analysis_reference = cv2.resize(
            gray_reference,
            size,
            interpolation=cv2.INTER_CUBIC,
        )
        analysis_test = cv2.resize(
            gray_test,
            size,
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        analysis_reference = gray_reference
        analysis_test = gray_test
    _, similarity_map = structural_similarity(
        analysis_reference,
        analysis_test,
        full=True,
    )
    pixel_difference = (
        cv2.absdiff(analysis_reference, analysis_test).astype(np.float32) / 255.0
    )
    return expert._build_generic_heatmap(
        similarity_map,
        pixel_difference,
        (reference.shape[1], reference.shape[0]),
    )


def install_roi_input_contract(orchestrator_cls, ssim_expert_cls, silk_expert_cls) -> None:
    """Instala o contrato após as demais extensões do orquestrador."""
    if getattr(orchestrator_cls, "_roi_input_contract_installed", False):
        return

    original_ssim_analyze = ssim_expert_cls.analyze

    def ssim_analyze(self, *args, **kwargs):
        category = _normalized_category(
            kwargs.get("defect_category")
            or getattr(self, "_visionx_current_category", "")
        )
        kwargs["defect_category"] = category
        result = original_ssim_analyze(self, *args, **kwargs)
        if not isinstance(result, dict):
            return result

        result["ssim_input_category"] = category
        result["ssim_roi_box"] = result.get("focus_box")
        if (
            category not in ADHESIVE_CATEGORIES
            and result.get("heat_map_mode") == "adhesive_excess"
        ):
            generic = _generic_ssim_heatmap(
                self,
                result.get("crop_gab"),
                result.get("crop_test"),
            )
            if generic is not None:
                result["heat_map_raw"] = generic
                result["heat_map_mode"] = "generic_ssim"
                result["adhesive_peak"] = 0.0
                result["adhesive_coverage"] = 0.0
                result["adhesive_evidence"] = 0.0
                result["adhesive_centroid"] = None
        return result

    ssim_expert_cls.analyze = ssim_analyze

    original_silk_analyze = silk_expert_cls.analyze

    def silk_analyze(
        self,
        full_gab,
        full_test,
        global_box_info=None,
        aoi_info=None,
        aoi_epicenters=None,
    ):
        result = original_silk_analyze(
            self,
            full_gab,
            full_test,
            global_box_info,
            aoi_info,
            aoi_epicenters,
        )
        if not isinstance(result, dict):
            return result
        reference_roi, test_roi, canonical_box = _canonical_roi(
            full_gab,
            full_test,
            aoi_epicenters,
        )
        if canonical_box is not None:
            result["silk_input_box"] = canonical_box
            result["roi_reference_raw"] = reference_roi
            result["roi_test_raw"] = test_roi
        return result

    silk_expert_cls.analyze = silk_analyze

    original_inspect = orchestrator_cls.inspect

    def inspect(
        self,
        full_gab,
        full_test,
        raw_anomalies,
        aoi_info,
        global_box_info,
        aoi_epicenters,
    ):
        category = _normalized_category((aoi_info or {}).get("category", ""))
        ssim_expert = self.experts.get("ssim")
        if ssim_expert is not None:
            ssim_expert._visionx_current_category = category

        analysis = original_inspect(
            self,
            full_gab,
            full_test,
            raw_anomalies,
            aoi_info,
            global_box_info,
            aoi_epicenters,
        )
        if not isinstance(analysis, dict):
            return analysis

        detail = analysis.setdefault("detail", {})
        reference_roi, test_roi, canonical_box = _canonical_roi(
            full_gab,
            full_test,
            aoi_epicenters,
        )
        if canonical_box is None:
            detail["roi_consistent"] = False
            detail["roi_audit"] = {
                "valid": False,
                "reason": "ROI canônica ausente ou inválida",
            }
            analysis["roi_consistent"] = False
            return analysis

        engines = {
            "texture_ssim": _engine_audit(
                detail.get("focus_box") or detail.get("ssim_roi_box"),
                detail.get("crop_test"),
                canonical_box,
                test_roi,
            ),
            "structural_xor": _engine_audit(
                detail.get("roi_box") or detail.get("silk_input_box"),
                detail.get("roi_test_raw"),
                canonical_box,
                test_roi,
            ),
            "semantic": _engine_audit(
                detail.get("semantic_focus_box"),
                detail.get("semantic_focus_test"),
                canonical_box,
                test_roi,
            ),
        }

        if detail.get("missing_active", False):
            engines["missing"] = _engine_audit(
                detail.get("missing_roi_box"),
                None,
                canonical_box,
                test_roi,
            )
        if detail.get("inverted_active", False):
            engines["inverted"] = _engine_audit(
                detail.get("inverted_roi_box"),
                None,
                canonical_box,
                test_roi,
            )

        active_box_checks = [
            item["box_match"]
            for item in engines.values()
            if item.get("box") is not None
        ]
        active_raw_checks = [
            item["test_input"]["exact_match"]
            for item in engines.values()
            if item["test_input"].get("available", False)
        ]
        consistent = bool(
            active_box_checks
            and all(active_box_checks)
            and active_raw_checks
            and all(active_raw_checks)
        )

        audit = {
            "valid": True,
            "category": category,
            "canonical_box": list(canonical_box),
            "canonical_reference_shape": [int(value) for value in reference_roi.shape],
            "canonical_test_shape": [int(value) for value in test_roi.shape],
            "canonical_reference_crc32": _crc32(reference_roi),
            "canonical_test_crc32": _crc32(test_roi),
            "all_boxes_match": bool(active_box_checks and all(active_box_checks)),
            "all_raw_inputs_match": bool(active_raw_checks and all(active_raw_checks)),
            "engines": engines,
        }
        detail["canonical_roi_box"] = canonical_box
        detail["canonical_roi_reference"] = reference_roi
        detail["canonical_roi_test"] = test_roi
        detail["roi_audit"] = audit
        detail["roi_consistent"] = consistent
        analysis["roi_consistent"] = consistent

        if not consistent:
            print(
                "AVISO ROI: motores com entrada diferente da ROI canônica • "
                f"caixa={canonical_box} • categoria={category}"
            )
        return analysis

    orchestrator_cls.inspect = inspect
    orchestrator_cls._roi_input_contract_installed = True


__all__ = ["install_roi_input_contract"]
