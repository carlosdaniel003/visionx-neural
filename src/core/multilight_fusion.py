"""Fusão final SIDE/TOP/MID para qualquer categoria AOI.

Cada iluminação é analisada pelo mesmo pipeline técnico e permanece auditável.
A fusão ocorre somente depois que SIDE, TOP e MID estão completos.

A categoria MUITO ADESIVO mantém sua política física especializada e já
validada operacionalmente. As demais categorias usam uma política conservadora:
- uma evidência local forte pode confirmar DEFEITO REAL;
- duas iluminações locais positivas corroboram DEFEITO REAL;
- uma única positiva moderada exige REVISÃO OBRIGATÓRIA;
- qualquer revisão local sem defeito corroborado exige REVISÃO OBRIGATÓRIA;
- somente três análises sem evidência de defeito resultam em FALHA FALSA.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.core.adhesive_multilight_fusion import fuse_adhesive_multilight
from src.utils.text_normalizer import normalize_aoi_text


LIGHTING_ORDER = ("SIDE", "TOP", "MID")
ADHESIVE_CATEGORY = "MUITO ADESIVO"
GENERIC_STRONG_SCORE = 0.80
GENERIC_DECISION_CUTOFF = 0.50


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number != number:
        return float(default)
    return number


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value)))


def _detail(analysis: dict | None) -> dict:
    if not isinstance(analysis, dict):
        return {}
    value = analysis.get("detail", {})
    return value if isinstance(value, dict) else {}


def _canonical_category(category: str) -> str:
    normalized, _value = normalize_aoi_text(str(category or ""))
    if normalized != "Unknown":
        return str(normalized)
    return str(category or "").strip().upper()


def _review_required(analysis: dict | None) -> bool:
    if not isinstance(analysis, dict):
        return True
    detail = _detail(analysis)
    trace = detail.get("decision_trace", {})
    trace = trace if isinstance(trace, dict) else {}
    return bool(
        analysis.get("production_review_required", False)
        or detail.get("operator_review_required", False)
        or trace.get("operator_review_required", False)
        or str(analysis.get("verdict", "") or "").strip().upper()
        == "REVISÃO OBRIGATÓRIA"
    )


def _lighting_evidence(mode: str, analysis: dict) -> dict:
    detail = _detail(analysis)
    trace = detail.get("decision_trace", {})
    trace = trace if isinstance(trace, dict) else {}

    final_score = _clamp01(
        detail.get("final_score", trace.get("final_score", 0.0))
    )
    physical_score = _clamp01(
        detail.get(
            "physical_score",
            analysis.get("physical_score", 0.0),
        )
    )
    confidence = _clamp01(analysis.get("confidence", 0.50))
    verdict = str(analysis.get("verdict", "") or "").strip().upper()
    positive = bool(
        analysis.get("is_defect", False)
        or verdict in {"DEFEITO REAL", "DEFEITO"}
    )
    review = _review_required(analysis)
    signal = max(final_score, physical_score)
    strong_positive = bool(
        positive and signal >= GENERIC_STRONG_SCORE
    )

    return {
        "mode": str(mode or "").strip().upper(),
        "available": True,
        "local_verdict": verdict or (
            "DEFEITO REAL" if positive else "FALHA FALSA"
        ),
        "local_is_defect": positive,
        "local_review_required": review,
        "confidence": confidence,
        "final_score": final_score,
        "physical_score": physical_score,
        "signal_score": signal,
        "strong_positive": strong_positive,
        "fusion_rule": str(detail.get("fusion_rule", "") or ""),
        "dominant_engine": str(detail.get("dominant_engine", "") or ""),
        "reason": str(analysis.get("reason", "") or ""),
        "memory_label": str(detail.get("best_match_label", "") or ""),
        "memory_similarity": _clamp01(
            detail.get("best_similarity", 0.0)
        ),
    }


def _dominant_mode(evidence: dict[str, dict]) -> str:
    return max(
        LIGHTING_ORDER,
        key=lambda mode: (
            1 if evidence[mode]["strong_positive"] else 0,
            1 if evidence[mode]["local_is_defect"] else 0,
            1 if evidence[mode]["local_review_required"] else 0,
            evidence[mode]["signal_score"],
            evidence[mode]["confidence"],
        ),
    )


def _summary(evidence: dict[str, dict]) -> str:
    chunks = []
    for mode in LIGHTING_ORDER:
        item = evidence[mode]
        if item["strong_positive"]:
            state = "DEFEITO FORTE"
        elif item["local_is_defect"]:
            state = "DEFEITO"
        elif item["local_review_required"]:
            state = "REVISÃO"
        else:
            state = "SEM DEFEITO"
        chunks.append(
            "{0}={1} (score {2:.0%}, físico {3:.0%})".format(
                mode,
                state,
                item["final_score"],
                item["physical_score"],
            )
        )
    return " | ".join(chunks)


def _augment_adhesive_aliases(fused: dict, category: str) -> dict:
    """Expõe chaves genéricas sem remover telemetria específica de adesivo."""
    detail = fused.setdefault("detail", {})
    detail["multilight_category"] = category
    detail["multilight_final"] = True
    detail["multilight_dominant_mode"] = detail.get(
        "adhesive_multilight_dominant_mode",
        "",
    )
    detail["multilight_positive_modes"] = list(
        detail.get("adhesive_multilight_positive_modes", []) or []
    )
    detail["multilight_strong_positive_modes"] = list(
        detail.get("adhesive_multilight_strong_auxiliary_modes", []) or []
    )
    detail["multilight_review_modes"] = []
    detail["multilight_memory_role"] = detail.get(
        "adhesive_multilight_memory_role",
        "audit_only",
    )
    fused["multilight_category"] = category
    return fused


def fuse_multilight(
    analyses: dict[str, dict] | None,
    category: str,
) -> dict | None:
    """Produz um único julgamento final depois das três iluminações."""
    source = analyses if isinstance(analyses, dict) else {}
    if any(
        not isinstance(source.get(mode), dict)
        for mode in LIGHTING_ORDER
    ):
        return None

    canonical_category = _canonical_category(category)
    if canonical_category == ADHESIVE_CATEGORY:
        fused = fuse_adhesive_multilight(source)
        if not isinstance(fused, dict):
            return None
        return _augment_adhesive_aliases(fused, canonical_category)

    evidence = {
        mode: _lighting_evidence(mode, source[mode])
        for mode in LIGHTING_ORDER
    }
    positive_modes = [
        mode
        for mode in LIGHTING_ORDER
        if evidence[mode]["local_is_defect"]
    ]
    strong_positive_modes = [
        mode
        for mode in LIGHTING_ORDER
        if evidence[mode]["strong_positive"]
    ]
    review_modes = [
        mode
        for mode in LIGHTING_ORDER
        if evidence[mode]["local_review_required"]
    ]
    dominant_mode = _dominant_mode(evidence)

    if strong_positive_modes:
        verdict = "DEFEITO REAL"
        is_defect = True
        review_required = False
        rule = "multilight_strong_single"
    elif len(positive_modes) >= 2:
        verdict = "DEFEITO REAL"
        is_defect = True
        review_required = False
        rule = "multilight_corroborated"
    elif len(positive_modes) == 1:
        verdict = "REVISÃO OBRIGATÓRIA"
        is_defect = False
        review_required = True
        rule = "multilight_single_positive_review"
    elif review_modes:
        verdict = "REVISÃO OBRIGATÓRIA"
        is_defect = False
        review_required = True
        rule = "multilight_local_review"
    else:
        verdict = "FALHA FALSA"
        is_defect = False
        review_required = False
        rule = "multilight_all_clear"

    strongest_signal = max(
        evidence[mode]["signal_score"] for mode in LIGHTING_ORDER
    )
    strongest_physical = max(
        evidence[mode]["physical_score"] for mode in LIGHTING_ORDER
    )
    confidences = [
        evidence[mode]["confidence"] for mode in LIGHTING_ORDER
    ]
    if review_required:
        confidence = 0.50
        final_score = 0.50
    elif is_defect:
        confidence = max(confidences)
        final_score = strongest_signal
    else:
        confidence = min(confidences)
        final_score = max(
            evidence[mode]["final_score"] for mode in LIGHTING_ORDER
        )

    dominant_analysis = deepcopy(source[dominant_mode])
    detail = dominant_analysis.get("detail", {})
    if not isinstance(detail, dict):
        detail = {}
        dominant_analysis["detail"] = detail

    reason = (
        "FUSÃO MULTILIGHT {0}: {1}. {2}".format(
            canonical_category or "CATEGORIA NÃO INFORMADA",
            verdict,
            _summary(evidence),
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
            "multilight_category": canonical_category,
            "eligible_for_final_decision": True,
        }
    )

    detail.update(
        {
            "final_score": _clamp01(final_score),
            "physical_score": strongest_physical,
            "fusion_rule": rule,
            "dominant_engine": "multilight",
            "multilight_final": True,
            "multilight_category": canonical_category,
            "multilight_dominant_mode": dominant_mode,
            "multilight_positive_modes": list(positive_modes),
            "multilight_strong_positive_modes": list(
                strong_positive_modes
            ),
            "multilight_review_modes": list(review_modes),
            "multilight_evidence": deepcopy(evidence),
            "multilight_memory_role": "per_lighting_audit",
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
            "schema": "visionx.multilight_decision.v1",
            "cutoff": GENERIC_DECISION_CUTOFF,
            "final_score": detail["final_score"],
            "confidence": dominant_analysis["confidence"],
            "verdict": verdict,
            "physical_score": strongest_physical,
            "physical_defect": bool(is_defect),
            "dominant_engine": "multilight",
            "fusion_rule": rule,
            "operator_review_required": bool(review_required),
            "memory_role": "per_lighting_audit",
            "lighting_evidence": deepcopy(evidence),
        }
    )
    return dominant_analysis


__all__ = [
    "ADHESIVE_CATEGORY",
    "GENERIC_DECISION_CUTOFF",
    "GENERIC_STRONG_SCORE",
    "LIGHTING_ORDER",
    "fuse_multilight",
]
