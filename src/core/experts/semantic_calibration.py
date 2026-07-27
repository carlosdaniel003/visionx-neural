"""Calibração do score semântico para defeitos pequenos e localizados."""

from __future__ import annotations

from typing import Any

import numpy as np


SEMANTIC_THRESHOLD = 0.45


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def semantic_evidence_components(global_loss: float, debug: dict | None) -> dict:
    """Combina divergência global com evidência localizada do grid 4x4."""
    global_score = float(np.clip(_safe_float(global_loss), 0.0, 1.0))
    payload = debug if isinstance(debug, dict) else {}
    spatial = payload.get("spatial", {}) if isinstance(payload.get("spatial"), dict) else {}
    groups = payload.get("groups", {}) if isinstance(payload.get("groups"), dict) else {}

    peak = _safe_float((spatial.get("peak_cell") or {}).get("value", 0.0))
    top_cells = spatial.get("top_cells", [])
    top_values = [
        _safe_float(item.get("combined_delta", 0.0))
        for item in top_cells[:3]
        if isinstance(item, dict)
    ]
    top_mean = float(np.mean(top_values)) if top_values else 0.0

    grid = np.asarray(spatial.get("combined_delta_grid", []), dtype=np.float32)
    grid_mean = float(np.mean(grid)) if grid.size else 0.0
    concentration = max(0.0, peak - grid_mean)

    group_scores = []
    for group in groups.values():
        if isinstance(group, dict):
            group_scores.append(_safe_float(group.get("relative_divergence", 0.0)))
    dominant_group = max(group_scores, default=0.0)

    local_evidence = float(
        np.clip(
            peak * 0.45
            + top_mean * 0.25
            + dominant_group * 0.20
            + concentration * 0.10,
            0.0,
            1.0,
        )
    )

    calibrated = float(
        np.clip(
            max(global_score, global_score * 0.35 + local_evidence * 0.65),
            0.0,
            1.0,
        )
    )

    return {
        "global_loss": global_score,
        "local_evidence": local_evidence,
        "calibrated_score": calibrated,
        "peak_cell": float(np.clip(peak, 0.0, 1.0)),
        "top_cells_mean": float(np.clip(top_mean, 0.0, 1.0)),
        "grid_mean": float(np.clip(grid_mean, 0.0, 1.0)),
        "concentration": float(np.clip(concentration, 0.0, 1.0)),
        "dominant_group": float(np.clip(dominant_group, 0.0, 1.0)),
        "threshold": SEMANTIC_THRESHOLD,
    }


def calibrate_semantic_result(result: dict | None) -> dict | None:
    """Atualiza um resultado existente preservando distância e dados de debug."""
    if not isinstance(result, dict):
        return result

    debug = result.get("semantic_debug")
    if not isinstance(debug, dict):
        return result

    original_loss = _safe_float(result.get("semantic_loss", result.get("score", 0.0)))
    calibration = semantic_evidence_components(original_loss, debug)
    score = calibration["calibrated_score"]

    result["semantic_global_loss"] = calibration["global_loss"]
    result["semantic_local_evidence"] = calibration["local_evidence"]
    result["semantic_loss"] = score
    result["score"] = score
    result["is_defect"] = bool(score > SEMANTIC_THRESHOLD)
    result["reason"] = (
        f"Evidência semântica: {score:.0%} "
        f"(global {calibration['global_loss']:.0%}; "
        f"local {calibration['local_evidence']:.0%})"
    )

    debug["semantic_loss"] = score
    debug["semantic_global_loss"] = calibration["global_loss"]
    debug["semantic_local_evidence"] = calibration["local_evidence"]
    debug["calibration"] = calibration
    return result


def install_semantic_calibration(semantic_expert_cls) -> None:
    """Aplica a calibração sem duplicar a implementação do especialista."""
    if getattr(semantic_expert_cls, "_localized_semantic_calibration", False):
        return

    original_analyze = semantic_expert_cls.analyze

    def analyze(self, *args, **kwargs):
        result = original_analyze(self, *args, **kwargs)
        return calibrate_semantic_result(result)

    semantic_expert_cls.analyze = analyze
    semantic_expert_cls._localized_semantic_calibration = True


def install_semantic_widget_calibration(widget_cls) -> None:
    """Expõe global, local e score calibrado na telemetria do debugger."""
    if getattr(widget_cls, "_localized_semantic_telemetry", False):
        return

    original_telemetry_lines = widget_cls._telemetry_lines

    def telemetry_lines(self):
        lines = original_telemetry_lines(self)
        calibration = self.debug.get("calibration", {}) if isinstance(self.debug, dict) else {}
        if not isinstance(calibration, dict) or not calibration:
            return lines

        global_loss = _safe_float(calibration.get("global_loss", 0.0))
        local_evidence = _safe_float(calibration.get("local_evidence", 0.0))
        calibrated = _safe_float(calibration.get("calibrated_score", self.sem_loss))
        threshold = _safe_float(calibration.get("threshold", SEMANTIC_THRESHOLD))
        peak = _safe_float(calibration.get("peak_cell", 0.0))
        concentration = _safe_float(calibration.get("concentration", 0.0))

        explanation = (
            f"score={calibrated:.1%} • global={global_loss:.1%} • "
            f"local={local_evidence:.1%} • corte={threshold:.0%} • "
            f"pico={peak:.2f} • concentração={concentration:.2f}"
        )
        return [explanation, *lines[1:]]

    widget_cls._telemetry_lines = telemetry_lines
    widget_cls._localized_semantic_telemetry = True
