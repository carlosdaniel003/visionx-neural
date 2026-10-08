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
MISSING_INTERLIGHT_OK_MIN_SIMILARITY = 0.88
MISSING_INTERLIGHT_MIN_CONTEXT_SIMILARITY = 0.70
MISSING_INTERLIGHT_MAX_GLOBAL_BACKGROUND = 0.03
# Liberação OK exige testemunha física independente e recorrência visual
# contextual na mesma categoria e iluminação; somente KNN não basta.
MISSING_VERIFIED_OK_MIN_SIMILARITY = 0.90
MISSING_VERIFIED_OK_SUSPECT_SIMILARITY = 0.92
MISSING_VERIFIED_OK_MIN_MARGIN = 0.08
MISSING_VERIFIED_OK_MIN_FULL_FRAME = 0.95
MISSING_VERIFIED_OK_MIN_CONTEXT = 0.94
MISSING_VERIFIED_OK_MIN_EPICENTER = 0.90
MISSING_VERIFIED_OK_MIN_BODY_COARSE = 0.70
MISSING_VERIFIED_OK_MIN_BODY_DICE = 0.84
MISSING_VERIFIED_OK_MIN_CLEAR_SIMILARITY = 0.93
MISSING_VERIFIED_OK_MAX_CLEAR_SCORE = 0.20



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


def _isolated_missing_disagreement(
    category: str,
    source: dict[str, dict],
    evidence: dict[str, dict],
) -> bool:
    """Exige revisão para uma única ausência física com contraprova multilight.

    A memória sozinha não anula hard missing. Outras iluminações precisam
    fornecer evidência física independente, e a peça nunca vira OK aqui.
    """
    if category != "FALTANDO":
        return False

    positive = [mode for mode in LIGHTING_ORDER if evidence[mode]["local_is_defect"]]
    if len(positive) != 1:
        return False
    culprit = positive[0]
    suspect = _detail(source[culprit])
    if not (
        evidence[culprit]["strong_positive"]
        and evidence[culprit]["dominant_engine"] == "missing"
        and bool(suspect.get("missing_hard_absence", False))
    ):
        return False

    if any(
        evidence[mode]["memory_label"].strip().upper() != "OK"
        or evidence[mode]["memory_similarity"] < MISSING_INTERLIGHT_OK_MIN_SIMILARITY
        for mode in LIGHTING_ORDER
    ):
        return False

    others = [mode for mode in LIGHTING_ORDER if mode != culprit]
    if any(
        evidence[mode]["local_is_defect"] or evidence[mode]["local_review_required"]
        for mode in others
    ):
        return False

    # Rota original: contexto da própria luz positiva preserva geometria.
    own_context = bool(
        _safe_float(suspect.get("missing_global_envelope_coarse_similarity", 0.0))
        >= MISSING_INTERLIGHT_MIN_CONTEXT_SIMILARITY
        and _safe_float(suspect.get("missing_global_envelope_background_exposure", 1.0), 1.0)
        <= MISSING_INTERLIGHT_MAX_GLOBAL_BACKGROUND
    )

    # Rota multilight: outra luz mantém o envelope alinhado, e outra
    # observação confirma ausência de divergência física local.
    # Footprint escuro sozinho não basta para liberar peça.
    context_witness = any(
        bool(_detail(source[mode]).get("missing_global_envelope_support", False))
        and not bool(_detail(source[mode]).get("missing_hard_absence", False))
        for mode in others
    )
    clear_witness = any(
        bool(_detail(source[mode]).get("missing_active", False))
        and not bool(_detail(source[mode]).get("missing_is_defect", True))
        and _safe_float(_detail(source[mode]).get("missing_score", 1.0), 1.0)
        <= _safe_float(_detail(source[mode]).get("missing_tolerance", 0.0))
        for mode in others
    )
    return bool(own_context or (context_witness and clear_witness))


def _verified_missing_ok_witnesses(
    category: str,
    source: dict[str, dict],
    evidence: dict[str, dict],
) -> dict:
    """Libera OK somente com corpo preservado, outra luz limpa e memória contextual.

    A decisão de ausência física bruta permanece no debug. Um match KNN
    isolado, mesmo alto, jamais autoriza esta rota. São exigidos três
    tipos de evidência de iluminações diferentes.
    """
    if category != "FALTANDO":
        return {}

    positives = [
        mode for mode in LIGHTING_ORDER if evidence[mode]["local_is_defect"]
    ]
    if len(positives) != 1:
        return {}
    suspect_mode = positives[0]
    suspect = _detail(source[suspect_mode])
    if not (
        evidence[suspect_mode]["strong_positive"]
        and evidence[suspect_mode]["dominant_engine"] == "missing"
        and bool(suspect.get("missing_hard_absence", False))
    ):
        return {}

    # As memórias devem ser da categoria/iluminação correta. A comparação
    # com o próprio NG mais próximo não pode mostrar empate relevante.
    if any(
        evidence[mode]["local_review_required"]
        or evidence[mode]["memory_label"].strip().upper() != "OK"
        or evidence[mode]["memory_similarity"] < MISSING_VERIFIED_OK_MIN_SIMILARITY
        or str(_detail(source[mode]).get("memory_lighting", "")).upper() != mode
        or str(_detail(source[mode]).get("memory_category", "")).upper() != "FALTANDO"
        for mode in LIGHTING_ORDER
    ):
        return {}
    if (
        evidence[suspect_mode]["memory_similarity"]
        < MISSING_VERIFIED_OK_SUSPECT_SIMILARITY
        or _safe_float(suspect.get("best_match_margin", 0.0))
        < MISSING_VERIFIED_OK_MIN_MARGIN
    ):
        return {}

    # Um exemplo OK já salvo é corroborador, não atalho: exigir
    # similaridade de imagem completa, contexto e epicentro separadamente.
    hypothesis = suspect.get("hypotheses", {}).get("OK", {})
    if not isinstance(hypothesis, dict) or not hypothesis.get("available", False):
        return {}
    breakdown = hypothesis.get("similarity_breakdown", {})
    if not isinstance(breakdown, dict):
        return {}
    previous = breakdown.get("previous", {})
    if not isinstance(previous, dict):
        return {}
    if (
        _safe_float(breakdown.get("full_frame_similarity", 0.0))
        < MISSING_VERIFIED_OK_MIN_FULL_FRAME
        or _safe_float(previous.get("context_similarity", 0.0))
        < MISSING_VERIFIED_OK_MIN_CONTEXT
        or _safe_float(previous.get("epicenter_similarity", 0.0))
        < MISSING_VERIFIED_OK_MIN_EPICENTER
    ):
        return {}

    other_modes = [mode for mode in LIGHTING_ORDER if mode != suspect_mode]
    if any(
        evidence[mode]["local_is_defect"]
        or evidence[mode]["final_score"] > MISSING_VERIFIED_OK_MAX_CLEAR_SCORE
        for mode in other_modes
    ):
        return {}

    def physical_body_confirmed(mode: str) -> bool:
        detail = _detail(source[mode])
        return bool(
            detail.get("fusion_rule") == "hard_missing_invariant_presence_ok_witness"
            and detail.get("missing_component_body_present", False)
            and detail.get("missing_global_envelope_invariant_support", False)
            and _safe_float(detail.get("missing_body_coarse_similarity", 0.0))
            >= MISSING_VERIFIED_OK_MIN_BODY_COARSE
            and _safe_float(detail.get("missing_body_silhouette_dice", 0.0))
            >= MISSING_VERIFIED_OK_MIN_BODY_DICE
            and _safe_float(
                detail.get("missing_global_envelope_background_exposure", 1.0), 1.0
            ) <= MISSING_INTERLIGHT_MAX_GLOBAL_BACKGROUND
            and detail.get("roi_consistent", False)
        )

    def physically_clear(mode: str) -> bool:
        detail = _detail(source[mode])
        return bool(
            evidence[mode]["memory_similarity"]
            >= MISSING_VERIFIED_OK_MIN_CLEAR_SIMILARITY
            and detail.get("missing_active", False)
            and not detail.get("missing_is_defect", True)
            and not detail.get("missing_hard_absence", True)
            and detail.get("missing_global_envelope_support", False)
            and _safe_float(detail.get("missing_score", 1.0), 1.0)
            <= _safe_float(detail.get("missing_tolerance", 0.0))
            and _safe_float(
                detail.get("missing_global_envelope_background_exposure", 1.0), 1.0
            ) <= MISSING_INTERLIGHT_MAX_GLOBAL_BACKGROUND
        )

    for body_mode in other_modes:
        clear_mode = next(mode for mode in other_modes if mode != body_mode)
        if physical_body_confirmed(body_mode) and physically_clear(clear_mode):
            return {
                "suspect_mode": suspect_mode,
                "body_mode": body_mode,
                "clear_mode": clear_mode,
                "suspect_memory_similarity": evidence[suspect_mode]["memory_similarity"],
                "full_frame_similarity": _safe_float(
                    breakdown["full_frame_similarity"]
                ),
                "context_similarity": _safe_float(previous["context_similarity"]),
                "epicenter_similarity": _safe_float(previous["epicenter_similarity"]),
            }
    return {}


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

    physical_disagreement = _isolated_missing_disagreement(
        canonical_category, source, evidence,
    )
    verified_presence = (
        _verified_missing_ok_witnesses(canonical_category, source, evidence)
        if physical_disagreement else {}
    )
    if verified_presence:
        verdict = "FALHA FALSA"
        is_defect = False
        review_required = False
        rule = "multilight_missing_verified_presence"
    elif physical_disagreement:
        verdict = "REVISÃO OBRIGATÓRIA"
        is_defect = False
        review_required = True
        rule = "multilight_missing_physical_disagreement"
    elif strong_positive_modes:
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
    elif verified_presence:
        confidence = min(confidences)
        # O score NG bruto da iluminação contradita permanece na auditoria,
        # mas não pode ser apresentado como score final da peça OK.
        final_score = max(
            evidence[mode]["final_score"]
            for mode in LIGHTING_ORDER
            if mode != verified_presence["suspect_mode"]
        )
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
    if verified_presence:
        reason += (
            " | {0} indicou ausência local, mas {1} confirmou corpo "
            "e presença invariável, {2} não detectou ausência; "
            "há correspondente OK contextual validado na iluminação {0}. "
            "Ausência bruta mantida somente para auditoria."
            .format(
                verified_presence["suspect_mode"],
                verified_presence["body_mode"],
                verified_presence["clear_mode"],
            )
        )
    elif physical_disagreement:
        reason += (
            f" | {positive_modes[0]} sinalizou ausência física isolada; "
            "outras iluminações não confirmaram, três memórias OK e "
            "testemunha contextual divergente. Revisão humana obrigatória."
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
            "multilight_dominant_local_engine": evidence[dominant_mode]["dominant_engine"],
            "multilight_positive_modes": list(positive_modes),
            "multilight_strong_positive_modes": list(
                strong_positive_modes
            ),
            "multilight_review_modes": list(review_modes),
            "multilight_evidence": deepcopy(evidence),
            "multilight_memory_role": "per_lighting_audit",
            "multilight_physical_disagreement": bool(physical_disagreement),
            "multilight_missing_verified_presence": bool(verified_presence),
            "multilight_missing_presence_witnesses": deepcopy(verified_presence),
            "operator_review_required": bool(review_required),
            "eligible_for_final_decision": True,
        }
    )

    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        trace = {}
        detail["decision_trace"] = trace
    # A TOP original permanece no debug bruto; a fusão de revisão não
    # concede autoridade final ao hard missing isolado.
    raw_hard_missing = bool(
        trace.get("raw_hard_missing_evidence", False)
        or trace.get("hard_missing_evidence", False)
        or detail.get("missing_hard_absence", False)
    )
    trace.update(
        {
            "schema": "visionx.multilight_decision.v1",
            "raw_hard_missing_evidence": raw_hard_missing,
            "hard_missing_evidence": bool(
                raw_hard_missing and not physical_disagreement
            ),
            "multilight_physical_disagreement": bool(physical_disagreement),
            "multilight_missing_verified_presence": bool(verified_presence),
            "multilight_missing_presence_witnesses": deepcopy(verified_presence),
            "multilight_dominant_mode": dominant_mode,
            "multilight_dominant_local_engine": evidence[dominant_mode]["dominant_engine"],
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
            # A formula local de SIDE nao representa o peso efetivo da
            # decisao multilight. Evidencias locais continuam no trace.
            "weights": {"physical": 0.0, "knn": 0.0},
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
