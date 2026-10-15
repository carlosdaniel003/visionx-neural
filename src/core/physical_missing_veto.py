"""Contraprova física independente e SOMENTE VETO para AUTO-OK da CNN FALTANDO.

A comparação ROI não vota NG nem aprova OK. Se indicar ausência ou não puder
ser verificada, exige intervenção humana antes de enviar 0/OK ao XP.
A CNN, KNN, os pesos e os vereditos das três luzes não são modificados.
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from src.core.experts.roi_patch_expert import ROIPatchExpectationExpert
from src.core.strict_category_memory import canonical_memory_category


POLICY = "cnn_faltando_independent_physical_veto_v1"
LIGHTS = ("SIDE", "TOP", "MID")
# Critérios conservadores para CONTRADIÇÃO, não para classificar NG.
MIN_SCORE = 0.72
MIN_COVERAGE = 0.32
MIN_BACKGROUND = 0.30
MIN_REFERENCE_DISTINCTNESS = 0.25
MAX_DIRECT_SIMILARITY = 0.65
MAX_NEARBY_SIMILARITY = 0.35
MAX_IMAGE_SIDE = 256
FOCUS_FRACTIONS = (0.58, 0.76)


def _valid_pair(reference, test) -> bool:
    return bool(
        isinstance(reference, np.ndarray)
        and isinstance(test, np.ndarray)
        and reference.dtype == test.dtype == np.uint8
        and reference.ndim == test.ndim == 3
        and reference.shape == test.shape
        and reference.shape[2] == 3
        and min(reference.shape[:2]) >= 32
    )


def _center_box(shape, fraction: float) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    width, height = max(12, int(round(w * fraction))), max(12, int(round(h * fraction)))
    return ((w - width) // 2, (h - height) // 2, width, height)


def _metric(result: dict, key: str) -> float:
    raw = result.get(key)
    if isinstance(raw, bool):
        raise ValueError("Medida física booleana: " + key)
    value = float(raw)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Medida física inválida: " + key)
    return value


def physical_absence_probe(reference, test) -> dict:
    """Investiga duas ROIs centrais da MESMA luz, sem utilizar resultado CNN.

    O método existente compara cores, bordas e presença de estrutura do
    gabarito; não aprende rótulos e não combina scores CNN ou da KNN.
    Falha ou geometria ambígua = UNAVAILABLE, nunca evidência de presença.
    """
    if not _valid_pair(reference, test):
        return {"status": "UNAVAILABLE", "reason": "INVALID_PAIR", "scales": []}
    try:
        height, width = reference.shape[:2]
        ratio = min(1.0, MAX_IMAGE_SIDE / max(height, width))
        if ratio < 1:
            size = (max(32, round(width * ratio)), max(32, round(height * ratio)))
            reference = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
            test = cv2.resize(test, size, interpolation=cv2.INTER_AREA)
        engine = ROIPatchExpectationExpert()
        measurements = []
        for fraction in FOCUS_FRACTIONS:
            box = _center_box(reference.shape, fraction)
            raw = engine.analyze(
                reference, test, {}, {"category": "FALTANDO"}, [box],
            )
            if not isinstance(raw, dict) or raw.get("missing_active") is not True:
                return {
                    "status": "UNAVAILABLE", "reason": "INACTIVE_PROBE",
                    "scales": measurements,
                }
            if raw.get("missing_classification") == "SEM DADOS":
                return {
                    "status": "UNAVAILABLE", "reason": "INVALID_PHYSICAL_ROI",
                    "scales": measurements,
                }
            keys = {
                "score": "missing_score",
                "coverage": "missing_changed_coverage",
                "background": "missing_background_exposure",
                "distinctness": "missing_reference_distinctness",
                "direct_similarity": "missing_direct_similarity",
                "nearby_similarity": "missing_best_similarity",
            }
            metrics = {name: _metric(raw, source) for name, source in keys.items()}
            # Este contrato exige exposição do fundo com desaparecimento da
            # aparência esperada; diferença de brilho isolada não basta.
            suspected = bool(
                raw.get("missing_is_defect") is True
                and raw.get("missing_classification") == "CONTEÚDO ESPERADO AUSENTE"
                and metrics["score"] >= MIN_SCORE
                and metrics["coverage"] >= MIN_COVERAGE
                and metrics["background"] >= MIN_BACKGROUND
                and metrics["distinctness"] >= MIN_REFERENCE_DISTINCTNESS
                and metrics["direct_similarity"] <= MAX_DIRECT_SIMILARITY
                and metrics["nearby_similarity"] <= MAX_NEARBY_SIMILARITY
            )
            measurements.append({
                "focus_fraction": fraction, "suspected_absence": suspected,
                "classification": str(raw.get("missing_classification", "")),
                **{key: round(value, 5) for key, value in metrics.items()},
            })
        return {
            "status": (
                "SUSPECT_ABSENCE" if any(row["suspected_absence"] for row in measurements)
                else "NO_STRONG_EVIDENCE"
            ),
            "reason": "TWO_SCALE_ROI_CHECK",
            "scales": measurements,
        }
    except (ValueError, TypeError, cv2.error, RuntimeError) as exc:
        return {
            "status": "UNAVAILABLE", "reason": type(exc).__name__,
            "scales": [],
        }


def apply_physical_missing_veto(
    fused: dict | None,
    samples: dict | None,
    category: str,
    *,
    probe: Callable | None = None,
) -> dict | None:
    """Finalização SIDE/TOP/MID: apenas contradiz um AUTO-OK CNN FALTANDO.

    A falta de ROI ou erro de cálculo bloqueia a liberação, mas não gera NG.
    O resultado é auditável em detail; nenhum campo é adicionado aos
    registros de cada iluminação ou à memória de treinamento.
    """
    if not isinstance(fused, dict):
        return fused
    if canonical_memory_category(category) != "FALTANDO":
        return fused
    detail = fused.get("detail")
    if not isinstance(detail, dict):
        return fused
    if not (
        fused.get("multilight_final") is True
        and fused.get("eligible_for_final_decision") is True
        and fused.get("verdict") == "FALHA FALSA"
        and not fused.get("production_review_required", False)
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_supervised_auto_eligible") is True
        and detail.get("cnn_v2_consensus") == "OK"
        and detail.get("multilight_category") == "FALTANDO"
    ):
        return fused

    by_light = {}
    active_probe = probe if probe is not None else physical_absence_probe
    if not isinstance(samples, dict):
        samples = {}
    for light in LIGHTS:
        sample = samples.get(light)
        if not isinstance(sample, dict) or sample.get("lighting_mode") != light:
            result = {"status": "UNAVAILABLE", "reason": "MISSING_LIGHT_PAIR", "scales": []}
        else:
            try:
                result = active_probe(sample.get("sample_image"), sample.get("test_image"))
                if not isinstance(result, dict) or result.get("status") not in {
                    "SUSPECT_ABSENCE", "NO_STRONG_EVIDENCE", "UNAVAILABLE"
                }:
                    result = {
                        "status": "UNAVAILABLE", "reason": "INVALID_PROBE_RESULT",
                        "scales": [],
                    }
            except Exception as exc:
                result = {
                    "status": "UNAVAILABLE", "reason": type(exc).__name__,
                    "scales": [],
                }
        by_light[light] = result

    suspected = [light for light in LIGHTS
                 if by_light[light]["status"] == "SUSPECT_ABSENCE"]
    unavailable = [light for light in LIGHTS
                   if by_light[light]["status"] == "UNAVAILABLE"]
    status = (
        "SUSPECT_ABSENCE" if suspected else
        "UNAVAILABLE" if unavailable else "NO_STRONG_EVIDENCE"
    )
    audit = {
        "policy": POLICY, "status": status,
        "diagnostic_only": True, "authorizes_ng": False,
        "authorizes_ok": False, "review_required": status != "NO_STRONG_EVIDENCE",
        "suspect_lights": suspected, "unavailable_lights": unavailable,
        "lights": by_light,
    }
    detail["physical_missing_veto"] = audit
    if status != "NO_STRONG_EVIDENCE":
        fused["verdict"] = "REVISÃO OBRIGATÓRIA"
        fused["is_defect"] = False
        fused["production_review_required"] = True
        fused["confidence"] = 0.50
        fused["reason"] = (
            "CNN FALTANDO V2 votou OK nas três luzes, mas a contraprova física "
            + ("encontrou suspeita de componente ausente" if suspected else
               "não pôde ser concluída em todas as iluminações")
            + "; operador deve confirmar OK/NG. Nenhum 0/1 automático autorizado."
        )
        detail["final_score"] = 0.50
        detail["operator_review_required"] = True
        detail["cnn_v2_supervised_auto_eligible"] = False
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace["operator_review_required"] = True
            trace["fusion_rule"] = POLICY
            trace["final_score"] = 0.50
        detail["fusion_rule"] = POLICY
    return fused


__all__ = ["POLICY", "physical_absence_probe", "apply_physical_missing_veto"]
