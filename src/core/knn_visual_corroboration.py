"""Corroboração visual KNN para CNN v2, independente da rota de par exato.

A experiência visual aproximada não altera a memória e não autoriza 0/1.
Vizinhos NG consistentes elevam um falso OK CNN a revisão, ou a NG
supervisionado com confirmação humana se a prova física também viu ausência.
A categoria AOI é uma pista, não a identidade do defeito visual.
"""
from __future__ import annotations

from hashlib import blake2b
import math
from typing import Any

import numpy as np

from src.core.anomaly_signature import (
    build_anomaly_signature, compare_anomaly_signatures, valid_anomaly_signature,
)
from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.strict_category_memory import (
    canonical_memory_category, canonical_memory_lighting,
)

LIGHTS = ("SIDE", "TOP", "MID")
SCHEMA = "visionx.knn_visual_corroboration.v1"
MIN_NG_SIMILARITY = 0.90
MIN_BEST_NG_SIMILARITY = 0.93
MIN_NG_MARGIN = 0.06
MIN_NG_EXAMPLES = 2
# Convergência tricolor: limiares NOVOS somente para NG em revisão
# com CNN parcialmente favorável a NG; não relaxar o KNN convencional.
TREND_MIN_NG_SIMILARITY = 0.91
TREND_MIN_MARGIN = 0.04
TREND_MIN_NG_EXAMPLES = 3
TREND_MIN_MEAN_MARGIN = 0.05
PARTIAL_CNN_MIN_NG_SCORE = 0.60
CROSS_CATEGORY_MIN_SIMILARITY = 0.97
MAX_CANDIDATES_PER_LIGHT = 2500


def _valid_image(value: Any) -> bool:
    return bool(
        isinstance(value, np.ndarray) and value.dtype == np.uint8
        and value.ndim == 3 and value.shape[2] == 3
        and min(value.shape[:2]) >= 32
    )


def _summary(rows: list[dict], limit: int = 5) -> dict:
    best = sorted(rows, key=lambda e: (-e["similarity"], e["source"]))
    return {
        "count": len(rows),
        "best_similarity": round(best[0]["similarity"], 6) if best else None,
        "nearest": best[:limit],
    }


def _scored_records(
    signature: dict, records: list[dict], *, label: str, light: str,
    category: str, cross_category: bool,
) -> list[dict]:
    choices = []
    fingerprints = set()
    for rec in records:
        if not isinstance(rec, dict) or rec.get("mode") != "anomaly":
            continue
        if rec.get("label") != label or rec.get("folder_label", label) != label:
            continue
        if canonical_memory_lighting(rec.get("lighting_mode", "")) != light:
            continue
        record_category = canonical_memory_category(rec.get("category", ""))
        if cross_category:
            if record_category == category or not uses_faltando_v2(record_category):
                continue
        elif record_category != category:
            continue
        sig = rec.get("anomaly_signature")
        if not valid_anomaly_signature(sig):
            continue
        # Várias importações da mesma foto não constituem votos independentes.
        digest = blake2b(
            np.asarray(sig["vector"], dtype=np.float32).tobytes(),
            digest_size=12,
        ).hexdigest()
        if digest in fingerprints:
            continue
        fingerprints.add(digest)
        similarity, _ = compare_anomaly_signatures(signature, sig)
        if not math.isfinite(similarity):
            continue
        choices.append({
            "label": label,
            "similarity": float(similarity),
            "category": record_category,
            "source": str(rec.get("path") or rec.get("json_path") or "")[:320],
        })
        if len(choices) >= MAX_CANDIDATES_PER_LIGHT:
            break
    return choices


def _one_light(
    sample: dict | None, light: str, category: str,
    ok_records: list[dict], ng_records: list[dict],
) -> dict:
    if not isinstance(sample, dict) or sample.get("lighting_mode") != light:
        return {"status": "UNAVAILABLE", "reason": "MISSING_LIGHT"}
    ref, test = sample.get("sample_image"), sample.get("test_image")
    if not _valid_image(ref) or not _valid_image(test):
        return {"status": "UNAVAILABLE", "reason": "INVALID_IMAGE"}
    try:
        query = build_anomaly_signature(
            ref, test, {}, {"category": category, "lighting_mode": light},
        )
        peak = float(query.get("summary", {}).get("map_peak", 0))
        if not valid_anomaly_signature(query) or not math.isfinite(peak) or peak < .10:
            return {"status": "NO_CONCLUSION", "reason": "WEAK_VISUAL_DIFFERENCE"}
        ok = _scored_records(query, ok_records, label="OK", light=light,
                             category=category, cross_category=False)
        ng = _scored_records(query, ng_records, label="NG", light=light,
                             category=category, cross_category=False)
        cross_category = False
        # A AOI pode rotular ausência como INVERTIDO/DESLOCADO/EMBORCADO.
        # Um único NG local fraco não deve ocultar exemplares de ausência
        # muito semelhantes aprendidos sob outra categoria.
        local_support = sum(
            rec["similarity"] >= MIN_NG_SIMILARITY for rec in ng
        )
        if local_support < MIN_NG_EXAMPLES and uses_faltando_v2(category):
            other_ng = _scored_records(
                query, ng_records, label="NG", light=light,
                category=category, cross_category=True,
            )
            strong_cross = sum(
                rec["similarity"] >= CROSS_CATEGORY_MIN_SIMILARITY
                for rec in other_ng
            )
            if strong_cross >= 3:
                ng = other_ng
                cross_category = True
                # Uma peça boa parecida de OUTRA categoria também é
                # contraprova, não pode ser ignorada por OCR diferente.
                ok += _scored_records(
                    query, ok_records, label="OK", light=light,
                    category=category, cross_category=True,
                )
        best_ok = max((r["similarity"] for r in ok), default=None)
        best_ng = max((r["similarity"] for r in ng), default=None)
        cutoff = CROSS_CATEGORY_MIN_SIMILARITY if cross_category else MIN_NG_SIMILARITY
        supported_ng = [r for r in ng if r["similarity"] >= cutoff]
        margin_ok = (best_ng is not None and
                     (best_ok is None or best_ng - best_ok >= MIN_NG_MARGIN))
        corroborated = bool(
            best_ng is not None
            and best_ng >= max(cutoff, MIN_BEST_NG_SIMILARITY)
            and len(supported_ng) >= (3 if cross_category else MIN_NG_EXAMPLES)
            and margin_ok
        )
        # Não considerar pequenas variações da mesma foto dez votos;
        # assinaturas distintas foram filtradas acima. Exigimos um OK
        # comparável na própria luz para medir a margem real.
        margin = (best_ng - best_ok
                  if best_ng is not None and best_ok is not None else None)
        trending_ng = bool(
            not cross_category
            and margin is not None and margin >= TREND_MIN_MARGIN
            and best_ng >= TREND_MIN_NG_SIMILARITY
            and len(supported_ng) >= TREND_MIN_NG_EXAMPLES
        )
        conflict = bool(
            best_ok is not None and best_ng is not None
            and abs(best_ng - best_ok) < MIN_NG_MARGIN
            and max(best_ok, best_ng) >= MIN_NG_SIMILARITY
        )
        return {
            "status": ("NG_SIMILAR" if corroborated else
                       "NG_LEANING" if trending_ng else
                       "CONFLICT" if conflict else "NO_CONCLUSION"),
            "reason": "SIGNATURE_NEIGHBORS",
            "cross_category_ng": cross_category,
            "ng_support": len(supported_ng),
            "ng_margin_over_ok": round(margin, 6) if margin is not None else None,
            "ok": _summary(ok),
            "ng": _summary(ng),
            "map_peak": round(peak, 6),
        }
    except (ValueError, TypeError, RuntimeError) as exc:
        return {"status": "UNAVAILABLE", "reason": type(exc).__name__}


def evaluate_knn_visual_corroboration(
    samples: dict | None, category: str, knn,
) -> dict:
    """Consulta RAM já carregada; não lê arquivos, não retreina CNN."""
    cat = canonical_memory_category(category)
    result = {
        "schema": SCHEMA, "category": cat,
        "diagnostic_only": False, "auto_decision_authorized": False,
        "recognition_route": "APPROXIMATE_KNN",
        "similarity_is_probability": False,
        "lights": {}, "strong_ng_lights": [], "trend_ng_lights": [],
        "ng_margin_mean": None, "status": "NO_CONCLUSION",
    }
    if not uses_faltando_v2(cat):
        result["status"] = "OUT_OF_SCOPE"
        return result
    if knn is None:
        result["status"] = "UNAVAILABLE"
        return result
    try:
        lock = knn._memory_lock
        with lock:
            ok_records = list(knn.signatures_ok)
            ng_records = list(knn.signatures_ng)
    except (AttributeError, TypeError):
        result["status"] = "UNAVAILABLE"
        return result
    for light in LIGHTS:
        entry = _one_light(
            samples.get(light) if isinstance(samples, dict) else None,
            light, cat, ok_records, ng_records,
        )
        result["lights"][light] = entry
        if entry["status"] == "NG_SIMILAR":
            result["strong_ng_lights"].append(light)
    if len(result["strong_ng_lights"]) >= 2:
        result["status"] = "NG_CORROBORATED"
    # Guardar a tendência tricolor mesmo quando duas luzes já alcançam
    # NG_CORROBORATED. O ramo CNN REVIEW exige três luzes direcionais,
    # não apenas duas fortes e uma ausente.
    trend = [
        light for light in LIGHTS
        if result["lights"][light].get("status") in {"NG_SIMILAR", "NG_LEANING"}
        and not result["lights"][light].get("cross_category_ng", False)
        and isinstance(result["lights"][light].get("ng_margin_over_ok"), (int, float))
        and result["lights"][light]["ng_margin_over_ok"] >= TREND_MIN_MARGIN
    ]
    result["trend_ng_lights"] = trend
    if len(trend) == 3:
        mean_margin = sum(
            result["lights"][light]["ng_margin_over_ok"] for light in LIGHTS
        ) / 3.0
        result["ng_margin_mean"] = round(mean_margin, 6)
        if mean_margin >= TREND_MIN_MEAN_MARGIN and result["status"] == "NO_CONCLUSION":
            result["status"] = "NG_TREND_THREE_LIGHTS"
    if result["status"] == "NO_CONCLUSION" and any(
        x.get("status") == "CONFLICT" for x in result["lights"].values()
    ):
        result["status"] = "CONFLICT"
    elif result["status"] == "NO_CONCLUSION" and all(
        x.get("status") == "UNAVAILABLE" for x in result["lights"].values()
    ):
        result["status"] = "UNAVAILABLE"
    return result


def verified_partial_cnn_review(detail: dict) -> dict | None:
    """TOP/SIDE/MID consistentes: 2 votos OK, 1 revisão NG-leaning.

    Usa diagnósticos brutos completos, pois o consenso CNN original sai
    antecipadamente com INVALID_LIGHT_EVIDENCE em revisão local.
    """
    if (
        not isinstance(detail, dict)
        or detail.get("cnn_v2_consensus") != "REVIEW"
        or detail.get("recognition_route") != "NEW_CNN"
    ):
        return None
    diags = detail.get("cnn_v2_light_diagnostics")
    if not isinstance(diags, dict) or set(diags) != set(LIGHTS):
        return None
    reviews, oks, hashes = [], [], []
    for light in LIGHTS:
        d = diags[light]
        if not isinstance(d, dict):
            return None
        sha = str(d.get("checkpoint_sha256", ""))
        if (
            d.get("route") != "NEW_CNN"
            or d.get("cnn_active") is not True
            or d.get("cnn_status") != "INFERENCE_OK"
            or d.get("checkpoint_verified") is not True
            or len(sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha)
            or canonical_memory_category(d.get("category")) != "FALTANDO"
        ):
            return None
        hashes.append(sha.lower())
        raw_score = d.get("ng_score_uncalibrated")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            return None
        score = float(raw_score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            return None
        verdict = d.get("verdict")
        if verdict == "FALHA FALSA" and score <= 0.10:
            oks.append(light)
        elif (
            verdict == "REVISÃO OBRIGATÓRIA"
            and PARTIAL_CNN_MIN_NG_SCORE <= score < 0.90
        ):
            reviews.append({"light": light, "ng_score_uncalibrated": score})
        else:
            return None
    if len(set(hashes)) != 1 or len(oks) != 2 or len(reviews) != 1:
        return None
    return {"review_light": reviews[0]["light"],
            "ng_score_uncalibrated": round(reviews[0]["ng_score_uncalibrated"], 6),
            "checkpoint_sha256": hashes[0]}


def trusted_three_light_ng_trend(detail: dict) -> bool:
    """Contrato independente para classificação NG tricolor, não AUTO-NG."""
    if not isinstance(detail, dict) or verified_partial_cnn_review(detail) is None:
        return False
    visual = detail.get("knn_visual_corroboration")
    if (
        not isinstance(visual, dict)
        or visual.get("status") not in {"NG_TREND_THREE_LIGHTS", "NG_CORROBORATED"}
        or set(visual.get("trend_ng_lights", [])) != set(LIGHTS)
    ):
        return False
    lights = visual.get("lights")
    if not isinstance(lights, dict) or set(lights) != set(LIGHTS):
        return False
    margins = []
    for light in LIGHTS:
        row = lights[light]
        if not isinstance(row, dict) or row.get("cross_category_ng"):
            return False
        if row.get("status") not in {"NG_LEANING", "NG_SIMILAR"}:
            return False
        ok, ng = row.get("ok"), row.get("ng")
        if not isinstance(ok, dict) or not isinstance(ng, dict):
            return False
        a, b = ng.get("best_similarity"), ok.get("best_similarity")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        if not math.isfinite(a) or not math.isfinite(b):
            return False
        if a < TREND_MIN_NG_SIMILARITY or a - b < TREND_MIN_MARGIN - 1e-6:
            return False
        if not isinstance(row.get("ng_support"), int) or row["ng_support"] < 3:
            return False
        margins.append(a-b)
    return sum(margins)/3 >= TREND_MIN_MEAN_MARGIN - 1e-6


def apply_knn_visual_corroboration(
    fused: dict | None, samples: dict | None, category: str, knn,
) -> dict | None:
    """Altera discordância CNN-vizinhos, nunca autoriza comando XP.

    NG conclusivo depende de NG semelhantes em duas luzes E evidência física
    independente de ausência; continua exigindo confirmação humana.
    """
    if not isinstance(fused, dict) or not fused.get("multilight_final"):
        return fused
    detail = fused.get("detail")
    if not isinstance(detail, dict):
        return fused
    if detail.get("recognition_route") != "NEW_CNN":
        return fused
    if not uses_faltando_v2(canonical_memory_category(category)):
        return fused
    report = evaluate_knn_visual_corroboration(samples, category, knn)
    detail["knn_visual_corroboration"] = report
    mixed_trend = trusted_three_light_ng_trend(detail)
    if mixed_trend:
        # Classificação visual NG (SEM COMANDO XP): três luzes KNN apontam NG
        # e uma luz CNN em revisão também aponta NG. A revisão do operador
        # permanece separada do rótulo final, como no NG+física existente.
        fused["verdict"] = "DEFEITO REAL"
        fused["is_defect"] = True
        fused["production_review_required"] = True
        fused["eligible_for_final_decision"] = False
        fused["confidence"] = 0.5
        fused["reason"] = (
            "NG POR CONVERGÊNCIA VISUAL: KNN favoreceu NG em SIDE/TOP/MID "
            "sobre os vizinhos OK, com CNN parcialmente favorável a NG "
            "na luz de revisão. Confirmação humana pendente; "
            "não é um comando NG automático."
        )
        detail["operator_review_required"] = True
        detail["cnn_v2_supervised_auto_eligible"] = False
        detail["fusion_rule"] = "knn_visual_three_light_ng_with_cnn_review_v1"
        detail["cnn_review_ng_support"] = verified_partial_cnn_review(detail)
        detail["final_score"] = 0.5
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace["operator_review_required"] = True
            trace["fusion_rule"] = detail["fusion_rule"]
            trace["final_score"] = 0.5
        return fused
    if report["status"] != "NG_CORROBORATED":
        return fused
    if detail.get("cnn_v2_consensus") != "OK":
        return fused
    physical = detail.get("physical_missing_veto")
    physical = physical if isinstance(physical, dict) else {}
    physical_confirmed = physical.get("status") == "SUSPECT_ABSENCE"
    fused["verdict"] = "DEFEITO REAL" if physical_confirmed else "REVISÃO OBRIGATÓRIA"
    fused["is_defect"] = physical_confirmed
    fused["production_review_required"] = True
    fused["eligible_for_final_decision"] = False
    fused["confidence"] = .5
    fused["reason"] = (
        "KNN visual recuperou vizinhos NG semelhantes em "
        + "/".join(report["strong_ng_lights"])
        + "; CNN votou OK. "
        + ("A contraprova física também detectou ausência; defeito "
           "corroborado, confirmar com operador."
           if physical_confirmed else
           "Divergência não validada; revisão humana obrigatória.")
    )
    detail["operator_review_required"] = True
    detail["cnn_v2_supervised_auto_eligible"] = False
    detail["fusion_rule"] = "knn_visual_cnn_physical_corroboration_v1"
    detail["final_score"] = .5
    trace = detail.get("decision_trace")
    if isinstance(trace, dict):
        trace["operator_review_required"] = True
        trace["fusion_rule"] = detail["fusion_rule"]
        trace["final_score"] = .5
    return fused


__all__ = [
    "evaluate_knn_visual_corroboration", "apply_knn_visual_corroboration",
    "SCHEMA", "verified_partial_cnn_review", "trusted_three_light_ng_trend",
]
