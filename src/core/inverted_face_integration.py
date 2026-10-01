"""Integra o especialista exclusivo de INVERTIDO ao fluxo já existente."""

from __future__ import annotations

import numpy as np

import src.core.anomaly_memory_integration as anomaly_memory_module
from src.core.anomaly_memory_integration import (
    _focus_box,
    _normalize_knn_memory_categories,
    canonical_category_key,
)
from src.core.anomaly_signature import build_anomaly_signature
from src.core.experts.inverted_face_expert import InvertedFaceExpert


INVERTED_KEYS = frozenset({"INVERTIDO"})


def is_inverted_category(category: str) -> bool:
    return canonical_category_key(category) in INVERTED_KEYS


def _combine_signature_mask(detail: dict, mask):
    signature_detail = dict(detail)
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return signature_detail
    current = signature_detail.get("diff_mask")
    if isinstance(current, np.ndarray) and current.shape == mask.shape:
        signature_detail["diff_mask"] = np.maximum(current, mask)
    else:
        signature_detail["diff_mask"] = mask
    signature_detail["inverted_anomaly_mask"] = mask
    signature_detail["inverted_score"] = float(detail.get("inverted_score", 0.0))
    return signature_detail


def _fusion_with_inverted(
    orchestrator,
    detail: dict,
    inverted: dict,
    knn: dict,
):
    """Usa a fusão central já envelopada por best-match e contraste de memória.

    O módulo INVERTIDO não mantém uma segunda implementação de pesos. Assim,
    hard missing, conflito OK×NG e futuras regras continuam com uma única fonte
    de verdade.
    """
    fusion_detail = dict(detail)
    fusion_detail.update(inverted)

    missing_result = (
        fusion_detail
        if fusion_detail.get("missing_cross_category_guard", False)
        else None
    )

    return anomaly_memory_module._dynamic_fusion(
        orchestrator,
        fusion_detail,
        "INVERTIDO",
        missing_result,
        knn,
    )

def install_inverted_face_integration(orchestrator_cls) -> None:
    if getattr(orchestrator_cls, "_inverted_face_installed", False):
        return

    previous_inspect = orchestrator_cls.inspect

    def inspect(
        self,
        full_gab,
        full_test,
        raw_anomalies,
        aoi_info,
        global_box_info,
        aoi_epicenters,
    ):
        analysis = previous_inspect(
            self,
            full_gab,
            full_test,
            raw_anomalies,
            aoi_info,
            global_box_info,
            aoi_epicenters,
        )
        category = str((aoi_info or {}).get("category", "Unknown"))
        if not is_inverted_category(category):
            return analysis

        if "inverted" not in self.experts:
            self.experts["inverted"] = InvertedFaceExpert()
        inverted = self.experts["inverted"].analyze(
            full_gab,
            full_test,
            global_box_info,
            aoi_info,
            aoi_epicenters,
        )

        detail = analysis.setdefault("detail", {})
        active_engines = analysis.setdefault("active_engines", [])
        detail.update(inverted)
        if inverted.get("inverted_active", False):
            if "inverted_expert.py" not in active_engines:
                active_engines.insert(0, "inverted_expert.py")
            if inverted.get("inverted_bounding_box"):
                analysis["bounding_box"] = inverted["inverted_bounding_box"]
                analysis.setdefault("all_boxes", {})["inverted"] = inverted[
                    "inverted_bounding_box"
                ]

        focus = _focus_box(aoi_epicenters, analysis, detail)
        signature_detail = _combine_signature_mask(
            detail,
            inverted.get("inverted_anomaly_mask"),
        )
        signature = build_anomaly_signature(
            full_gab,
            full_test,
            signature_detail,
            aoi_info,
            focus,
        )
        knn_expert = self.experts["knn"]
        _normalize_knn_memory_categories(knn_expert)
        knn_result = knn_expert.analyze(
            full_gab,
            full_test,
            None,
            None,
            aoi_info,
            anomaly_signature=signature,
        )

        final_score, is_defect, confidence, reason, trace = _fusion_with_inverted(
            self,
            detail,
            inverted,
            knn_result,
        )
        analysis["is_defect"] = is_defect
        analysis["confidence"] = confidence
        analysis["verdict"] = "DEFEITO REAL" if is_defect else "FALHA FALSA"
        analysis["reason"] = reason
        detail.update(knn_result)
        detail.update(
            {
                "anomaly_signature": signature,
                "query_anomaly_signature": signature,
                "final_score": final_score,
                "physical_score": trace["physical_score"],
                "decision_cutoff": trace["cutoff"],
                "dominant_engine": trace["dominant_engine"],
                "fusion_rule": trace["fusion_rule"],
                "decision_trace": trace,
            }
        )
        return analysis

    orchestrator_cls.inspect = inspect
    orchestrator_cls._inverted_face_installed = True


__all__ = [
    "install_inverted_face_integration",
    "is_inverted_category",
]
