"""Memoria humana multilight: reconhecer OK historico em qualquer iluminacao.

A similaridade de assinatura e aproximada, nao indica probabilidade de
qualidade. A classificacao visual e separada da autorizacao de saida XP:
uma suspeita fisica nao e apagada por um registro humano anterior.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite

from src.core.physical_missing_veto import physical_absence_probe
from src.core.side_memory_first import (
    KNOWN_MEMORY_CATEGORIES, _verified_probe,
)
from src.core.strict_category_memory import canonical_memory_category

RULE = "operator_human_multilight_97pct_v1"
SCHEMA = "visionx.operator_human_multilight.v1"
LIGHTS = ("SIDE", "TOP", "MID")
MIN_HUMAN = 0.97
MIN_MARGIN = 0.04


def _score(raw):
    if type(raw) not in (int, float):
        return None
    number = float(raw)
    return number if isfinite(number) and 0 <= number <= 1 else None


def _light_snapshot(light: str, analysis: dict) -> dict:
    detail = analysis.get("detail")
    detail = detail if isinstance(detail, dict) else {}
    return {
        "lighting_mode": analysis.get("lighting_mode"),
        "verdict": analysis.get("verdict"),
        "is_defect": analysis.get("is_defect"),
        "hard_missing_evidence": analysis.get("hard_missing_evidence"),
        "physical_score": detail.get("physical_score", analysis.get("physical_score", 0)),
        "recognition_route": detail.get("recognition_route"),
        "recognition_match": detail.get("recognition_match"),
        "memory_conflict": detail.get("memory_conflict"),
        "hard_missing_detail": detail.get("hard_missing_evidence"),
        "capture_error": detail.get("capture_error"),
        "cnn_v2_aoi_category": detail.get("cnn_v2_aoi_category"),
        "cnn_v2_active": detail.get("cnn_v2_active"),
        "cnn_v2_status": detail.get("cnn_v2_status"),
        "cnn_v2_checkpoint_verified": detail.get("cnn_v2_checkpoint_verified"),
        "confirmed_visual_recall": deepcopy(detail.get("confirmed_visual_recall")),
    }


def _qualify(source: dict, category: str) -> dict | None:
    scope = canonical_memory_category(category)
    if scope not in KNOWN_MEMORY_CATEGORIES or not isinstance(source, dict):
        return None
    if set(source) != set(LIGHTS):
        return None
    okay, witnesses = [], []
    for light in LIGHTS:
        row = source.get(light)
        if not isinstance(row, dict) or row.get("lighting_mode") != light:
            return None
        if (
            row.get("recognition_route") not in {"NEW_CNN", "NEW_EXPERTS"}
            or row.get("recognition_match") == "EXACT_PAIR"
            or row.get("memory_conflict") is True
            or row.get("hard_missing_evidence") is True
            or row.get("hard_missing_detail") is True
            or row.get("capture_error") is True
            or row.get("is_defect") is not False
            or row.get("verdict") not in {"FALHA FALSA", "REVISÃO OBRIGATÓRIA"}
            or _score(row.get("physical_score")) != 0.0
        ):
            return None
        claimed = row.get("cnn_v2_aoi_category")
        if claimed and canonical_memory_category(claimed) != scope:
            return None
        if row.get("cnn_v2_active") is True and (
            row.get("cnn_v2_status") != "INFERENCE_OK"
            or row.get("cnn_v2_checkpoint_verified") is not True
        ):
            return None
        if row["verdict"] == "FALHA FALSA":
            okay.append(light)
        human = row.get("confirmed_visual_recall")
        if not isinstance(human, dict):
            continue
        if human.get("status") == "HUMAN_NG_VISUAL_CANDIDATE":
            # Um NG humano inequivoco em qualquer luz impede este atalho.
            return None
        if not (
            human.get("schema") == "visionx.human_visual_recall.v1"
            and human.get("status") == "HUMAN_OK_VISUAL_CANDIDATE"
            and human.get("source") == "PERSISTENT_HUMAN_LABELS"
            and human.get("category") == scope
            and human.get("diagnostic_only") is True
            and human.get("auto_ok_authorized") is False
        ):
            continue
        value = _score(human.get("best_ok_similarity"))
        contrary = _score(human.get("best_ng_similarity"))
        previous = human.get("best_ok_source")
        if (
            value is None or value < MIN_HUMAN
            or contrary is None or value - contrary < MIN_MARGIN
            or not isinstance(previous, dict)
            or previous.get("label") != "OK"
            or previous.get("source") != "OPERATOR_CONFIRMED"
            or not isinstance(previous.get("record_id"), str)
            or len(previous["record_id"]) != 64
            or any(char not in "0123456789abcdef" for char in previous["record_id"].lower())
        ):
            continue
        witnesses.append({
            "light": light, "human_record_id": previous["record_id"],
            "similarity": value, "opposite_similarity": contrary,
            "source_light": previous.get("lighting_mode"),
        })
    if len(okay) < 2 or not witnesses:
        return None
    selected = max(witnesses, key=lambda entry: (entry["similarity"], entry["light"]))
    return {
        "schema": SCHEMA, "rule": RULE, "category": scope,
        "operator_witness": selected,
        "local_ok_lights": okay,
        "min_human_similarity": MIN_HUMAN,
        "min_opposite_margin": MIN_MARGIN,
        "approximate": True, "exact_pair_claimed": False,
    }


def build_human_multilight_priority(
    fused: dict | None, analyses: dict | None, samples: dict | None,
    category: str, *, probe=None,
) -> dict | None:
    """Recupera o OK humano TOP/SIDE/MID sem esperar a KNN C A/B.

    A visualizacao pode reconhecer o historico mesmo quando a sonda
    fisica exige revisao. O XP so recebe auto-OK se TODAS as sondas
    atuais forem validas, sem suspeita de ausencia.
    """
    if not isinstance(fused, dict) or not isinstance(analyses, dict):
        return fused
    detail = fused.get("detail")
    if not isinstance(detail, dict) or fused.get("multilight_final") is not True:
        return fused
    knn_visual = detail.get("knn_visual_corroboration")
    if (
        detail.get("multilight_physical_disagreement") is True
        or detail.get("capture_error") is True
        or (
            isinstance(knn_visual, dict)
            and knn_visual.get("status") == "NG_CORROBORATED"
        )
    ):
        return fused
    snapshots = {
        light: _light_snapshot(light, analyses[light])
        for light in LIGHTS
        if isinstance(analyses.get(light), dict)
    }
    qualification = _qualify(snapshots, category)
    if qualification is None:
        return fused
    checker = probe if probe is not None else physical_absence_probe
    sample_map = samples if isinstance(samples, dict) else {}
    physical = {}
    for light in LIGHTS:
        pair = sample_map.get(light)
        if not isinstance(pair, dict) or pair.get("lighting_mode") != light:
            physical[light] = {"status": "UNAVAILABLE", "reason": "MISSING_LIGHT_PAIR"}
            continue
        try:
            result = checker(pair.get("sample_image"), pair.get("test_image"))
            physical[light] = result if isinstance(result, dict) else {
                "status": "UNAVAILABLE", "reason": "INVALID_PROBE_RESULT",
            }
        except Exception as exc:
            physical[light] = {"status": "UNAVAILABLE", "reason": type(exc).__name__}
    physical_ok = all(_verified_probe(physical[light], "OK") for light in LIGHTS)

    result = deepcopy(fused)
    rd = result["detail"]
    witness = qualification["operator_witness"]
    result["verdict"] = "FALHA FALSA"
    result["is_defect"] = False
    result["confidence"] = witness["similarity"]
    result["production_review_required"] = not physical_ok
    result["reason"] = (
        "Memoria humana OK OPERATOR_CONFIRMED recuperada em "
        + witness["light"] + " com similaridade >=97% e margem >=4pp. "
        "Duas iluminacoes locais OK; nenhuma evidencia fisica NG corroborada. "
        "Contraprova fisica atual: "
        + ("VALIDADA" if physical_ok else "REVISAO NECESSARIA")
        + ". Similaridade nao equivale a certeza fisica."
    )
    rd["human_multilight_priority"] = {
        "evidence": qualification, "light_snapshots": snapshots,
        "physical_probes": deepcopy(physical),
        "physical_verified": physical_ok,
    }
    rd["fusion_rule"] = RULE
    rd["operator_review_required"] = not physical_ok
    rd["eligible_for_final_decision"] = True
    trace = rd.get("decision_trace")
    if not isinstance(trace, dict):
        trace = {}
        rd["decision_trace"] = trace
    trace.update({
        "fusion_rule": RULE, "verdict": "FALHA FALSA",
        "operator_review_required": not physical_ok,
        "physical_defect": False,
    })
    return result


def verified_human_multilight_priority(analysis: dict | None) -> bool | None:
    """Revalida o contrato da classificacao e o veto fisico no gate XP.

    None = rota ausente ou invalida; False = OK visual com veto fisico;
    True = OK visual com sondas atuais aptas para automacao.
    """
    if not isinstance(analysis, dict):
        return None
    detail = analysis.get("detail")
    if not isinstance(detail, dict) or detail.get("fusion_rule") != RULE:
        return None
    marker = detail.get("human_multilight_priority")
    if not isinstance(marker, dict):
        return None
    expected = _qualify(marker.get("light_snapshots"), analysis.get("multilight_category"))
    probes = marker.get("physical_probes")
    if (
        expected is None or marker.get("evidence") != expected
        or not isinstance(probes, dict) or set(probes) != set(LIGHTS)
        or marker.get("physical_verified") is not
            all(_verified_probe(probes[light], "OK") for light in LIGHTS)
        or analysis.get("verdict") != "FALHA FALSA"
        or analysis.get("is_defect") is not False
        or analysis.get("multilight_final") is not True
        or analysis.get("eligible_for_final_decision") is not True
        or detail.get("multilight_final") is not True
        or detail.get("multilight_category") != expected["category"]
        or detail.get("multilight_physical_disagreement") is True
        or detail.get("hard_missing_evidence") is True
        or analysis.get("hard_missing_evidence") is True
        or detail.get("memory_conflict") is True
        or detail.get("capture_error") is True
        or analysis.get("production_review_required") is not (
            not marker["physical_verified"]
        )
        or detail.get("operator_review_required") is not (
            not marker["physical_verified"]
        )
    ):
        return None
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict) or (
        trace.get("fusion_rule") != RULE
        or trace.get("operator_review_required") is not (
            not marker["physical_verified"]
        )
    ):
        return None
    return marker["physical_verified"]


__all__ = [
    "RULE", "MIN_HUMAN", "build_human_multilight_priority",
    "verified_human_multilight_priority",
]
