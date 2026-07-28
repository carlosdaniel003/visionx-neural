"""Especializa os 16 campos físicos da assinatura KNN para INVERTIDO.

O tamanho do vetor permanece 224, preservando compatibilidade. Apenas o bloco
physics recebe significado específico quando a categoria é INVERTIDO.
"""

from __future__ import annotations

import numpy as np

import src.core.inverted_face_integration as inverted_integration


PHYSICS_NAMES = (
    "inverted_score",
    "witness_loss",
    "feature_loss",
    "topology_mismatch",
    "orientation_mismatch",
    "alternate_face_signal",
    "signature_strength",
    "test_signature_strength",
    "relocation_gain",
    "relocation_dx_normalized",
    "relocation_dy_normalized",
    "direct_dissimilarity",
    "changed_coverage",
    "extra_structure",
    "transform_gain",
    "witness_coverage",
)


def _safe(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _inverted_physics(detail: dict) -> np.ndarray:
    width = max(1.0, _safe(detail.get("inverted_roi_width", 1.0), 1.0))
    height = max(1.0, _safe(detail.get("inverted_roi_height", 1.0), 1.0))
    dx = _safe(detail.get("inverted_relocation_dx", 0.0))
    dy = _safe(detail.get("inverted_relocation_dy", 0.0))
    return np.asarray(
        [
            np.clip(_safe(detail.get("inverted_score", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_witness_loss", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_feature_loss", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_topology_mismatch", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_orientation_mismatch", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_alternate_face_signal", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_signature_strength", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_test_signature_strength", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_relocation_gain", 0.0)), 0.0, 1.0),
            np.clip(dx / width * 0.5 + 0.5, 0.0, 1.0),
            np.clip(dy / height * 0.5 + 0.5, 0.0, 1.0),
            np.clip(1.0 - _safe(detail.get("inverted_direct_similarity", 1.0), 1.0), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_changed_coverage", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_extra_structure", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_transform_gain", 0.0)), 0.0, 1.0),
            np.clip(_safe(detail.get("inverted_witness_coverage", 0.0)), 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def install_inverted_signature_extension() -> None:
    if getattr(inverted_integration, "_inverted_signature_extension_installed", False):
        return

    original_builder = inverted_integration.build_anomaly_signature

    def build_anomaly_signature(reference, test, detail, aoi_info=None, focus_box=None):
        signature = original_builder(reference, test, detail, aoi_info, focus_box)
        category = " ".join(
            str((aoi_info or {}).get("category", "")).upper().split()
        )
        if category not in {"INVERTIDO", "INVERTED", "REVERSE", "UP SIDE DOWN"}:
            return signature

        physics = _inverted_physics(detail if isinstance(detail, dict) else {})
        ranges = signature.get("ranges", {})
        start, end = ranges.get("physics", [208, 224])
        vector = np.asarray(signature.get("vector", []), dtype=np.float32).reshape(-1)
        if vector.size >= int(end) and int(end) - int(start) == physics.size:
            vector[int(start) : int(end)] = physics
            signature["vector"] = vector.tolist()
        signature["physics"] = {
            name: float(value)
            for name, value in zip(PHYSICS_NAMES, physics)
        }
        summary = signature.setdefault("summary", {})
        summary["magnitude"] = float(np.mean(vector)) if vector.size else 0.0
        signature["specialist"] = "inverted_witness"
        return signature

    inverted_integration.build_anomaly_signature = build_anomaly_signature
    inverted_integration._inverted_signature_extension_installed = True


__all__ = ["install_inverted_signature_extension"]
