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
# CNN com 3 OK pode contradizer a memoria NG em tres iluminacoes.
# Esta rota APENAS classifica NG para operador; jamais autoriza XP.
RECALL_MIN_NG_SIMILARITY = 0.995
RECALL_MIN_NG_SUPPORT = 5
RECALL_MIN_MARGIN_EACH = 0.04
RECALL_MIN_MARGIN_MEAN = 0.075
RECALL_MIN_TWO_STRONG_MARGINS = 0.09
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


def trusted_three_light_ng_recall(detail: dict) -> bool:
    """NG de memória contra 3 votos CNN OK, sem comando automático.

    Exige todos os três pares KNN locais NG>>OK, checkpoint CNN íntegro,
    duas luzes com NG_SIMILAR, e a contraprova física executada nas três
    luzes mas inconclusiva por aparagem geométrica mínima. A geometria
    não é usada como prova de defeito: a conclusão é só visual/humana.
    """
    if not isinstance(detail, dict) or (
        detail.get("multilight_category") != "FALTANDO"
        or detail.get("recognition_route") != "NEW_CNN"
        or detail.get("cnn_v2_consensus") != "OK"
    ):
        return False
    diags = detail.get("cnn_v2_light_diagnostics")
    if not isinstance(diags, dict) or set(diags) != set(LIGHTS):
        return False
    hashes = set()
    for light in LIGHTS:
        row = diags[light]
        if not isinstance(row, dict):
            return False
        sha = row.get("checkpoint_sha256")
        score = row.get("ng_score_uncalibrated")
        if (
            row.get("route") != "NEW_CNN"
            or row.get("cnn_active") is not True
            or row.get("cnn_status") != "INFERENCE_OK"
            or row.get("checkpoint_verified") is not True
            or row.get("verdict") != "FALHA FALSA"
            or canonical_memory_category(row.get("category")) != "FALTANDO"
            or not isinstance(sha, str) or len(sha) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in sha)
            or isinstance(score, bool) or not isinstance(score, (float, int))
            or not math.isfinite(score) or not 0.0 <= score <= 0.1
        ):
            return False
        hashes.add(sha.lower())
    if len(hashes) != 1:
        return False

    visual = detail.get("knn_visual_corroboration")
    if not isinstance(visual, dict) or (
        visual.get("status") != "NG_CORROBORATED"
        or visual.get("category") != "FALTANDO"
        or visual.get("auto_decision_authorized") is not False
    ):
        return False
    trend = visual.get("trend_ng_lights")
    strong = visual.get("strong_ng_lights")
    rows = visual.get("lights")
    if (
        not isinstance(trend, list) or set(trend) != set(LIGHTS)
        or not isinstance(strong, list) or len(set(strong)) < 2
        or any(light not in LIGHTS for light in strong)
        or not isinstance(rows, dict) or set(rows) != set(LIGHTS)
    ):
        return False
    margins = []
    for light in LIGHTS:
        row = rows[light]
        if not isinstance(row, dict) or row.get("cross_category_ng") is not False:
            return False
        if row.get("status") not in {"NG_SIMILAR", "NG_LEANING"}:
            return False
        if (light in strong) != (row.get("status") == "NG_SIMILAR"):
            return False
        ng, ok = row.get("ng"), row.get("ok")
        if not isinstance(ng, dict) or not isinstance(ok, dict):
            return False
        a, b = ng.get("best_similarity"), ok.get("best_similarity")
        support = row.get("ng_support")
        if (
            not isinstance(a, (int, float)) or isinstance(a, bool)
            or not isinstance(b, (int, float)) or isinstance(b, bool)
            or not math.isfinite(a) or not math.isfinite(b)
            or not 0 <= a <= 1 or not 0 <= b <= 1
            or a < RECALL_MIN_NG_SIMILARITY
            or a - b < RECALL_MIN_MARGIN_EACH - 1e-6
            or not isinstance(support, int) or isinstance(support, bool)
            or support < RECALL_MIN_NG_SUPPORT
            or not isinstance(ng.get("count"), int) or ng["count"] < support
            or not isinstance(ok.get("count"), int) or ok["count"] < 1
        ):
            return False
        margins.append(a - b)
    if (
        sum(m >= RECALL_MIN_TWO_STRONG_MARGINS - 1e-6 for m in margins) < 2
        or sum(margins) / 3 < RECALL_MIN_MARGIN_MEAN - 1e-6
    ):
        return False

    physical = detail.get("physical_missing_veto")
    if not isinstance(physical, dict) or (
        physical.get("status") != "GEOMETRY_REVIEW"
        or physical.get("suspect_lights") != []
        or physical.get("unavailable_lights") != []
        or set(physical.get("geometry_review_lights", [])) != set(LIGHTS)
    ):
        return False
    physical_rows = physical.get("lights")
    if not isinstance(physical_rows, dict) or set(physical_rows) != set(LIGHTS):
        return False
    return all(
        isinstance(physical_rows[light], dict)
        and physical_rows[light].get("status") == "GEOMETRY_REVIEW"
        for light in LIGHTS
    )


def _three_light_ok_counterevidence(fused: dict, report: dict,
                                    category: str) -> dict | None:
    """Three KNN lights support *review*, never automatic OK.

    A/B votes from KNN C are approximate diagnostics. The independent
    signature search must also prefer OK to NG in all three lights; the
    uncalibrated CNN cannot erase this uncertainty by a high NG score.
    """
    if canonical_memory_category(category) != "FALTANDO":
        return None
    detail = fused.get("detail")
    if not isinstance(detail, dict) or not (
        fused.get("multilight_final") is True
        and fused.get("verdict") == "DEFEITO REAL"
        and fused.get("is_defect") is True
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_consensus") == "REVIEW"
        and detail.get("cnn_v2_consensus_reason") == "INVALID_LIGHT_EVIDENCE"
        and set(detail.get("multilight_strong_positive_modes") or ()) == set(LIGHTS)
        and detail.get("multilight_physical_disagreement") is not True
        and detail.get("multilight_missing_verified_presence") is not True
        and detail.get("hard_missing_evidence") is not True
        and fused.get("hard_missing_evidence") is not True
    ):
        return None
    trace = detail.get("decision_trace")
    if isinstance(trace, dict) and (
        trace.get("hard_missing_evidence") is True
        or trace.get("raw_hard_missing_evidence") is True
    ):
        return None
    try:
        physical_score = float(detail.get("physical_score", 0))
    except (ValueError, TypeError):
        return None
    if not math.isfinite(physical_score) or physical_score != 0:
        return None
    physical = detail.get("physical_missing_veto")
    if isinstance(physical, dict) and physical.get("status") in {
        "SUSPECT_ABSENCE", "NG_CORROBORATED",
    }:
        return None
    observations = detail.get("knn_c_light_observer_votes")
    if not isinstance(observations, dict) or set(observations) != set(LIGHTS):
        return None
    if not all(
        isinstance(observations[light], dict)
        and observations[light].get("schema") == "visionx.knn_c_live_review_gated.v1"
        and observations[light].get("bank_status") == "READY"
        and observations[light].get("status") == "REVISAO"
        and observations[light].get("diagnostic_only") is True
        and observations[light].get("auto_decision_authorized") is False
        and observations[light].get("reason") == "SEM_SUGESTAO_CONFIAVEL"
        and observations[light].get("vote_A") == "OK"
        and observations[light].get("vote_B") == "OK"
        and observations[light].get("suggestion_A") == "REVISAO"
        and observations[light].get("suggestion_B") == "REVISAO"
        for light in LIGHTS
    ):
        return None
    if (not isinstance(report, dict)
            or report.get("schema") != SCHEMA
            or report.get("status") != "CONFLICT"):
        return None
    lights = report.get("lights")
    if not isinstance(lights, dict) or set(lights) != set(LIGHTS):
        return None
    points = {}
    for light in LIGHTS:
        evidence = lights[light]
        if (not isinstance(evidence, dict)
                or evidence.get("status") != "CONFLICT"
                or evidence.get("cross_category_ng") is not False
                or not isinstance(evidence.get("ok"), dict)
                or not isinstance(evidence.get("ng"), dict)):
            return None
        try:
            ok = float(evidence["ok"]["best_similarity"])
            ng = float(evidence["ng"]["best_similarity"])
            margin = float(evidence["ng_margin_over_ok"])
        except (TypeError, ValueError, KeyError):
            return None
        # O caso confirmado tem 0.951/0.951/0.954 de vizinhança OK.
        # O limite é apenas um filtro de CONFLITO para revisão, não uma
        # regra treinada para classificar peças como OK.
        if (not all(map(math.isfinite, (ok, ng, margin)))
                or ok < .95 or ng >= ok
                or margin >= 0 or abs((ng - ok) - margin) > .00001):
            return None
        points[light] = {
            "best_ok_similarity": round(ok, 6),
            "best_ng_similarity": round(ng, 6),
            "ng_margin_over_ok": round(margin, 6),
        }
    return {
        "schema": "visionx.knn_three_light_ok_counterevidence_review.v1",
        "status": "OPERATOR_REVIEW_KNN_OK_COUNTEREVIDENCE",
        "no_exact_pair_claim": True,
        "automatic_ok_authorized": False,
        "xp_command_authorized": False,
        "lights": points,
    }


# Fontes físicas OK especificamente confirmadas por um operador.
# 26/10/2026: mesmo componente recorrente, arquivo citado no evento
# c69b71764ba34ab6add7be8fde7a5dae. A pasta OK isolada não autoriza
# classificação: conferir SHA dos pixels e evidência multilight.
OPERATOR_VERIFIED_RECURRENT_OK_ANCHORS = {
    (
        "public/ok_archive/2026-10-21_1412_FALTANDO.png",
        "c9dec1d85617aa074e71b8b3640cace88f3a23c123185f7be84d6c06e2946bd9",
    ),
}


def _recognized_recurrent_ok(
    detail: dict, counterevidence: dict,
) -> dict | None:
    """Reconhece um padrão OK do operador, nunca valida OK automático.

    A semelhança é apenas aproximada; por isso requer confirmação
    humana até que a segurança com NG inéditos seja demonstrada.
    """
    if (
        not isinstance(counterevidence, dict)
        or counterevidence.get("status")
        != "OPERATOR_REVIEW_KNN_OK_COUNTEREVIDENCE"
        or counterevidence.get("automatic_ok_authorized") is not False
        or counterevidence.get("xp_command_authorized") is not False
    ):
        return None
    observations = detail.get("knn_c_light_observer_votes")
    side = observations.get("SIDE") if isinstance(observations, dict) else None
    if not isinstance(side, dict):
        return None
    original = detail.get("knn_c")
    # O registro herdado vem do MID, não da SIDE; para reconhecer memória
    # de captura original precisamos preservar vizinhos da SIDE na fusão.
    all_lights = detail.get("knn_c_light_neighbor_evidence")
    if not isinstance(all_lights, dict):
        return None
    side_neighbors = all_lights.get("SIDE")
    if not isinstance(side_neighbors, dict):
        return None
    winners = []
    for label, minimum in (("neighbors_A", .99), ("neighbors_B", .975)):
        group = side_neighbors.get(label)
        if not isinstance(group, list) or not group:
            return None
        head = group[0]
        if not isinstance(head, dict) or head.get("label") != "OK":
            return None
        try:
            sim = float(head["similarity"])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(sim) or sim < minimum or sim > 1:
            return None
        identity = (head.get("path"), head.get("source_pixel_sha256"))
        if identity not in OPERATOR_VERIFIED_RECURRENT_OK_ANCHORS:
            return None
        # Não esconder um NG visual quase tão próximo.
        ng_near = []
        for other in group:
            if isinstance(other, dict) and other.get("label") == "NG":
                try:
                    ng_sim = float(other.get("similarity"))
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(ng_sim):
                    return None
                ng_near.append(ng_sim)
        if not ng_near or max(ng_near) >= .90:
            return None
        winners.append({"method": label, "similarity": round(sim, 8)})
    return {
        "schema": "visionx.operator_confirmed_recurrent_ok.v1",
        "status": "OK_RECURRENT_OPERATOR_REFERENCE",
        "source_path": side_neighbors["neighbors_A"][0]["path"],
        "source_pixel_sha256": side_neighbors["neighbors_A"][0]["source_pixel_sha256"],
        "visual_support": winners,
        "three_light_counterevidence": True,
        "approximate_match_not_exact": True,
        "operator_review_required": True,
        "automatic_ok_authorized": False,
        "xp_command_authorized": False,
    }


def _operator_confirmed_ok_pattern_review(
    detail: dict, counterevidence: dict,
) -> dict | None:
    """Propose a known false alarm even with lighting variation.

    A known OK SHA must be retrieved in both KNN descriptors, while
    three independent light comparisons favor OK. This is an operator
    review classification, NEVER proof to send XP command 0.
    """
    if not isinstance(detail, dict) or not isinstance(counterevidence, dict):
        return None
    if (
        counterevidence.get("status") != "OPERATOR_REVIEW_KNN_OK_COUNTEREVIDENCE"
        or counterevidence.get("no_exact_pair_claim") is not True
        or counterevidence.get("automatic_ok_authorized") is not False
        or counterevidence.get("xp_command_authorized") is not False
    ):
        return None
    lights = counterevidence.get("lights")
    if not isinstance(lights, dict) or set(lights) != set(LIGHTS):
        return None
    margins = []
    for light in LIGHTS:
        point = lights.get(light)
        if not isinstance(point, dict):
            return None
        try:
            margin = float(point["ng_margin_over_ok"])
            ok_similarity = float(point["best_ok_similarity"])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(margin) or not math.isfinite(ok_similarity):
            return None
        if margin > -.005 or ok_similarity < .95:
            return None
        margins.append(-margin)
    if sum(margins) / len(margins) < .012:
        return None
    # The display/production re-check must also see the raw votes.
    votes = detail.get("knn_c_light_observer_votes")
    if not isinstance(votes, dict) or set(votes) != set(LIGHTS):
        return None
    if not all(
        isinstance(votes[light], dict)
        and votes[light].get("schema") == "visionx.knn_c_live_review_gated.v1"
        and votes[light].get("bank_status") == "READY"
        and votes[light].get("status") == "REVISAO"
        and votes[light].get("reason") == "SEM_SUGESTAO_CONFIAVEL"
        and votes[light].get("diagnostic_only") is True
        and votes[light].get("auto_decision_authorized") is False
        and votes[light].get("vote_A") == "OK"
        and votes[light].get("vote_B") == "OK"
        and votes[light].get("suggestion_A") == "REVISAO"
        and votes[light].get("suggestion_B") == "REVISAO"
        for light in LIGHTS
    ):
        return None
    by_light = detail.get("knn_c_light_neighbor_evidence")
    side = by_light.get("SIDE") if isinstance(by_light, dict) else None
    if not isinstance(side, dict):
        return None
    matched = []
    for key, minimum in (("neighbors_A", .88), ("neighbors_B", .82)):
        rows = side.get(key)
        if not isinstance(rows, list) or not 3 <= len(rows) <= 5:
            return None
        matches = []
        ng_scores = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                return None
            try:
                similarity = float(row["similarity"])
            except (TypeError, ValueError, KeyError):
                return None
            if not math.isfinite(similarity) or not 0 <= similarity <= 1:
                return None
            if row.get("label") == "NG":
                ng_scores.append(similarity)
            if (
                index < 3 and row.get("label") == "OK"
                and similarity >= minimum
                and (row.get("path"), row.get("source_pixel_sha256"))
                in OPERATOR_VERIFIED_RECURRENT_OK_ANCHORS
            ):
                matches.append((index, row, similarity))
        if len(matches) != 1 or not ng_scores:
            return None
        rank, row, sim = matches[0]
        best_ng = max(ng_scores)
        if best_ng >= .90 or best_ng - sim > .02:
            return None
        matched.append({
            "method": key, "rank": rank + 1,
            "similarity": round(sim, 8),
            "path": row["path"],
            "source_pixel_sha256": row["source_pixel_sha256"],
            "highest_competing_ng": round(best_ng, 8),
        })
    if matched[0]["source_pixel_sha256"] != matched[1]["source_pixel_sha256"]:
        return None
    return {
        "schema": "visionx.operator_confirmed_ok_pattern_review.v1",
        "status": "KNOWN_OK_PATTERN_PENDING_CURRENT_OPERATOR",
        "source_path": matched[0]["path"],
        "source_pixel_sha256": matched[0]["source_pixel_sha256"],
        "matches": matched,
        "three_light_counterevidence": True,
        "current_capture_exact_match": False,
        "current_capture_operator_confirmed": False,
        "operator_review_required": True,
        "automatic_ok_authorized": False,
        "xp_command_authorized": False,
    }


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
    # Dar voz à KNN quando o consenso CNN é inválido, sem converter
    # proximidade visual em OK confirmado ou conceder comando 0.
    ok_counterevidence = _three_light_ok_counterevidence(
        fused, report, category,
    )
    if ok_counterevidence is not None:
        detail["knn_ok_counterevidence"] = ok_counterevidence
        detail["cnn_ng_verdict_before_knn_review"] = fused["verdict"]
        known_ok = _recognized_recurrent_ok(detail, ok_counterevidence)
        if known_ok is not None:
            detail["recognized_recurrent_ok"] = known_ok
        pattern_ok = (
            _operator_confirmed_ok_pattern_review(detail, ok_counterevidence)
            if known_ok is None else None
        )
        if pattern_ok is not None:
            detail["operator_confirmed_ok_pattern_review"] = pattern_ok
        fused["verdict"] = (
            "FALHA FALSA" if known_ok is not None or pattern_ok is not None
            else "REVISÃO OBRIGATÓRIA"
        )
        fused["is_defect"] = False
        fused["production_review_required"] = True
        fused["eligible_for_final_decision"] = False
        fused["confidence"] = .5
        fused["reason"] = (
            (
                "FALHA FALSA • OK RECORRENTE: referência do operador "
                "recuperada por SHA nas KNN A/B SIDE com semelhança "
                "muito alta e NG afastados; TOP/MID também favorecem "
                "OK, CNN sem evidência física independente. "
                "Reconhecimento aproximado, SEM autorização automática "
                "de comando XP: confirmar com o operador."
            ) if known_ok is not None else (
                "FALHA FALSA • PADRÃO OK RECORRENTE: referência "
                "confirmada encontrada em A/B entre os primeiros vizinhos, "
                "com três luzes favorecendo OK e sem ausência física "
                "corroborada. A captura atual não é um par exato: "
                "confirmação do operador obrigatória, sem comando XP automático."
            ) if pattern_ok is not None else (
                "CNN FALTANDO NG sem corroboração física: SIDE/TOP/MID "
                "apresentam divergência KNN, com os dois votos KNN C "
                "apontando OK (ainda em REVISAO) e vizinhos OK mais "
                "próximos do que NG nas três luzes. É contraprova visual "
                "para REVISÃO OBRIGATÓRIA, não prova de par exato nem "
                "autorização de OK ou de comando XP automático."
            )
        )
        detail["fusion_rule"] = (
            "operator_confirmed_recurrent_ok_three_light_review_v1"
            if known_ok is not None
            else "operator_confirmed_ok_pattern_three_light_review_v1"
            if pattern_ok is not None
            else "knn_three_light_ok_counterevidence_review_v1"
        )
        detail["operator_review_required"] = True
        detail["eligible_for_final_decision"] = False
        detail["cnn_v2_supervised_auto_eligible"] = False
        detail["final_score"] = .5
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace["verdict"] = fused["verdict"]
            trace["operator_review_required"] = True
            trace["physical_defect"] = False
            trace["fusion_rule"] = detail["fusion_rule"]
            trace["final_score"] = .5
        return fused
    memory_recall = trusted_three_light_ng_recall(detail)
    if memory_recall:
        # Três iluminações com NG visual extremamente próximo e separado
        # de OK. A CNN original errou em todas; o operador ainda confirma.
        # GEOMETRY_REVIEW é uma limitação da prova física, não voto NG.
        fused["verdict"] = "DEFEITO REAL"
        fused["is_defect"] = True
        fused["production_review_required"] = True
        fused["eligible_for_final_decision"] = False
        fused["confidence"] = 0.5
        fused["reason"] = (
            "NG POR MEMÓRIA VISUAL: SIDE/TOP/MID recuperaram vizinhos NG "
            "extremamente semelhantes e separados de OK. CNN votou OK nas "
            "três luzes e a contraprova física exige revisão geométrica. "
            "Defeito visual provável; confirmação humana obrigatória, "
            "sem autorização de comando XP automático."
        )
        detail["operator_review_required"] = True
        detail["cnn_v2_supervised_auto_eligible"] = False
        detail["fusion_rule"] = "knn_three_light_strong_ng_recall_vs_cnn_ok_v1"
        detail["final_score"] = 0.5
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace["operator_review_required"] = True
            trace["fusion_rule"] = detail["fusion_rule"]
            trace["final_score"] = 0.5
        return fused
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
    "trusted_three_light_ng_recall",
]
