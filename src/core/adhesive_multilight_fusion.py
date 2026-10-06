"""Fusão final SIDE/TOP/MID exclusiva da categoria de adesivo.

A fusão não faz média simples e não usa o veredito KNN local como autoridade
final. Ela combina a evidência física de adesivo das três iluminações da mesma
peça. TOP e MID são testemunhas fotométricas prioritárias; SIDE permanece como
testemunha contextual/corroboradora.

Nesta primeira política multilight:
- uma evidência auxiliar forte em TOP ou MID é suficiente para DEFEITO REAL;
- duas iluminações positivas corroboradas também resultam em DEFEITO REAL;
- uma única evidência positiva não forte exige REVISÃO OBRIGATÓRIA;
- sem evidência física positiva nas três iluminações, o resultado é FALHA FALSA.

O KNN de cada iluminação continua preservado dentro das análises individuais
para auditoria, mas não pode vetar uma evidência física multilight forte.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


LIGHTING_ORDER = ("SIDE", "TOP", "MID")
AUXILIARY_LIGHTS = {"TOP", "MID"}

DEFAULT_ADHESIVE_TOLERANCE = 0.32
STRONG_AUX_ADHESIVE_SCORE = 0.80
STRONG_AUX_PHYSICAL_SCORE = 0.80


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number != number:  # NaN
        return float(default)
    return number


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value)))


def _detail(analysis: dict | None) -> dict:
    if not isinstance(analysis, dict):
        return {}
    value = analysis.get("detail", {})
    return value if isinstance(value, dict) else {}


def _lighting_evidence(mode: str, analysis: dict | None) -> dict:
    normalized = str(mode or "").strip().upper()
    detail = _detail(analysis)

    tolerance = _safe_float(
        detail.get("adhesive_tolerance", DEFAULT_ADHESIVE_TOLERANCE),
        DEFAULT_ADHESIVE_TOLERANCE,
    )
    if tolerance <= 0.0:
        tolerance = DEFAULT_ADHESIVE_TOLERANCE

    adhesive_score = _clamp01(
        detail.get(
            "adhesive_score",
            detail.get("shift_score", 0.0),
        )
    )
    physical_score = _clamp01(
        detail.get(
            "physical_score",
            analysis.get("physical_score", 0.0)
            if isinstance(analysis, dict)
            else 0.0,
        )
    )
    adhesive_is_defect = bool(
        detail.get(
            "adhesive_is_defect",
            adhesive_score >= tolerance,
        )
    )

    positive = bool(
        adhesive_is_defect
        and adhesive_score >= tolerance
    )
    strong_auxiliary = bool(
        normalized in AUXILIARY_LIGHTS
        and positive
        and adhesive_score >= STRONG_AUX_ADHESIVE_SCORE
        and physical_score >= STRONG_AUX_PHYSICAL_SCORE
    )

    return {
        "mode": normalized,
        "available": isinstance(analysis, dict),
        "adhesive_score": adhesive_score,
        "physical_score": physical_score,
        "adhesive_tolerance": float(tolerance),
        "adhesive_is_defect": adhesive_is_defect,
        "positive": positive,
        "strong_auxiliary": strong_auxiliary,
        "local_verdict": (
            str(analysis.get("verdict", "") or "")
            if isinstance(analysis, dict)
            else ""
        ),
        "local_is_defect": (
            bool(analysis.get("is_defect", False))
            if isinstance(analysis, dict)
            else False
        ),
        "memory_label": str(
            detail.get("best_match_label", "") or ""
        ),
        "memory_similarity": _clamp01(
            detail.get("best_similarity", 0.0)
        ),
        "reason": str(
            detail.get("adhesive_reason", "")
            or (
                analysis.get("reason", "")
                if isinstance(analysis, dict)
                else ""
            )
            or ""
        ),
    }


def _dominant_mode(evidence: dict[str, dict]) -> str:
    ranked = sorted(
        LIGHTING_ORDER,
        key=lambda mode: (
            evidence[mode]["adhesive_score"],
            evidence[mode]["physical_score"],
            1 if mode in AUXILIARY_LIGHTS else 0,
        ),
        reverse=True,
    )
    return ranked[0]


def _summary_text(evidence: dict[str, dict]) -> str:
    chunks = []
    for mode in LIGHTING_ORDER:
        item = evidence[mode]
        state = (
            "FORTE"
            if item["strong_auxiliary"]
            else "POSITIVO"
            if item["positive"]
            else "SEM EVIDÊNCIA"
        )
        chunks.append(
            "{0}={1} (adesivo {2:.0%}, físico {3:.0%})".format(
                mode,
                state,
                item["adhesive_score"],
                item["physical_score"],
            )
        )
    return " | ".join(chunks)


def fuse_adhesive_multilight(
    analyses: dict[str, dict] | None,
) -> dict | None:
    """Retorna o único julgamento final depois de SIDE/TOP/MID completos."""
    source = analyses if isinstance(analyses, dict) else {}
    if any(
        not isinstance(source.get(mode), dict)
        for mode in LIGHTING_ORDER
    ):
        return None

    evidence = {
        mode: _lighting_evidence(mode, source[mode])
        for mode in LIGHTING_ORDER
    }

    positive_modes = [
        mode for mode in LIGHTING_ORDER if evidence[mode]["positive"]
    ]
    strong_auxiliary_modes = [
        mode
        for mode in ("TOP", "MID")
        if evidence[mode]["strong_auxiliary"]
    ]
    dominant_mode = _dominant_mode(evidence)

    if strong_auxiliary_modes:
        verdict = "DEFEITO REAL"
        is_defect = True
        review_required = False
        rule = "adhesive_multilight_strong_auxiliary"
    elif len(positive_modes) >= 2:
        verdict = "DEFEITO REAL"
        is_defect = True
        review_required = False
        rule = "adhesive_multilight_corroborated"
    elif len(positive_modes) == 1:
        verdict = "REVISÃO OBRIGATÓRIA"
        is_defect = False
        review_required = True
        rule = "adhesive_multilight_single_positive_review"
    else:
        verdict = "FALHA FALSA"
        is_defect = False
        review_required = False
        rule = "adhesive_multilight_no_physical_evidence"

    strongest_adhesive = max(
        evidence[mode]["adhesive_score"]
        for mode in LIGHTING_ORDER
    )
    strongest_physical = max(
        evidence[mode]["physical_score"]
        for mode in LIGHTING_ORDER
    )

    if review_required:
        confidence = 0.50
    elif is_defect:
        confidence = max(strongest_adhesive, strongest_physical)
    else:
        confidence = max(0.50, 1.0 - strongest_adhesive)

    dominant_analysis = deepcopy(source[dominant_mode])
    detail = dominant_analysis.get("detail", {})
    if not isinstance(detail, dict):
        detail = {}
        dominant_analysis["detail"] = detail

    reason = (
        "FUSÃO MULTILIGHT ADESIVO: {0}. {1}".format(
            verdict,
            _summary_text(evidence),
        )
    )

    dominant_analysis.update(
        {
            "is_defect": bool(is_defect),
            "confidence": _clamp01(confidence),
            "verdict": verdict,
            "reason": reason,
            "production_review_required": bool(review_required),
            "lighting_mode": "MULTILIGHT",
            "multilight_final": True,
            "eligible_for_final_decision": True,
        }
    )

    detail.update(
        {
            "final_score": (
                0.50
                if review_required
                else strongest_adhesive
                if is_defect
                else 0.0
            ),
            "physical_score": strongest_physical,
            "fusion_rule": rule,
            "dominant_engine": "adhesive_multilight",
            "adhesive_multilight_final": True,
            "adhesive_multilight_dominant_mode": dominant_mode,
            "adhesive_multilight_positive_modes": list(positive_modes),
            "adhesive_multilight_strong_auxiliary_modes": list(
                strong_auxiliary_modes
            ),
            "adhesive_multilight_evidence": deepcopy(evidence),
            "adhesive_multilight_memory_role": "audit_only",
            "operator_review_required": bool(review_required),
            "eligible_for_final_decision": True,
        }
    )

    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        trace = {}
        detail["decision_trace"] = trace

    trace.update(
        {
            "schema": "visionx.adhesive_multilight_decision.v1",
            "final_score": detail["final_score"],
            "physical_score": strongest_physical,
            "fusion_rule": rule,
            "dominant_engine": "adhesive_multilight",
            "operator_review_required": bool(review_required),
            "verdict": verdict,
            "memory_role": "audit_only",
            "lighting_evidence": deepcopy(evidence),
        }
    )

    return dominant_analysis


__all__ = [
    "AUXILIARY_LIGHTS",
    "DEFAULT_ADHESIVE_TOLERANCE",
    "LIGHTING_ORDER",
    "STRONG_AUX_ADHESIVE_SCORE",
    "STRONG_AUX_PHYSICAL_SCORE",
    "fuse_adhesive_multilight",
]
