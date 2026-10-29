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


# O mesmo evento físico OK foi confirmado pelo operador, mas as
# capturas TOP/MID usam luzes e hashes de pixels distintos. Uma fonte
# histórica OK não equivale a par *atual* exatamente idêntico.
_OPERATOR_OK_MULTI_LIGHT_REFERENCES = {
    "TOP": (
        "public/ok_archive/2026-10-27_1318_FALTANDO_TOP.png",
        "046effa074997ca1211b45e905745f2f6ec68263a0613854e4130eb7168e7d5d",
    ),
    "MID": (
        "public/ok_archive/2026-10-27_1318_FALTANDO_MID.png",
        "338f1cc612656297c21db284ea097489b90eb6e0537c3231dde45740f6a16e29",
    ),
}


def _recognized_two_light_operator_ok(
    fused: dict, report: dict, category: str,
) -> dict | None:
    """Recognize operator-confirmed recurring OK with TOP+MID KNN A/B.

    Requires two independent captures from the *same* human-confirmed
    historical OK event plus non-NG SIDE. Explicitly does not authorize
    automatic OK / XP commands from approximate similarities.
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
    try:
        physical_score = float(detail.get("physical_score", float("nan")))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(physical_score) or physical_score != 0:
        return None
    physical = detail.get("physical_missing_veto")
    if isinstance(physical, dict) and physical.get("status") in {
        "SUSPECT_ABSENCE", "NG_CORROBORATED",
    }:
        return None
    trace = detail.get("decision_trace")
    if isinstance(trace, dict) and (
        trace.get("hard_missing_evidence") is True
        or trace.get("raw_hard_missing_evidence") is True
    ):
        return None

    observers = detail.get("knn_c_light_observer_votes")
    neighbors = detail.get("knn_c_light_neighbor_evidence")
    if (
        not isinstance(observers, dict) or set(observers) != set(LIGHTS)
        or not isinstance(neighbors, dict)
    ):
        return None
    for light in LIGHTS:
        vote = observers[light]
        if not isinstance(vote, dict) or not (
            vote.get("schema") == "visionx.knn_c_live_review_gated.v1"
            and vote.get("bank_status") == "READY"
            and vote.get("diagnostic_only") is True
            and vote.get("auto_decision_authorized") is False
            and vote.get("vote_A") == "OK"
            and vote.get("vote_B") == "OK"
        ):
            return None
        if light == "MID":
            if not (
                vote.get("status") == "SUGERE_OK"
                and vote.get("reason") == "A_E_B_CONCORDAM"
                and vote.get("suggestion_A") == "SUGERE_OK"
                and vote.get("suggestion_B") == "SUGERE_OK"
            ):
                return None
        elif not (
            vote.get("status") == "REVISAO"
            and vote.get("reason") == "SEM_SUGESTAO_CONFIAVEL"
            and vote.get("suggestion_A") == "REVISAO"
            and vote.get("suggestion_B") == "REVISAO"
        ):
            return None

    if not isinstance(report, dict) or not (
        report.get("schema") == SCHEMA
        and report.get("status") == "CONFLICT"
        and report.get("strong_ng_lights") == []
        and report.get("trend_ng_lights") == []
    ):
        return None
    evidence = report.get("lights")
    if not isinstance(evidence, dict) or set(evidence) != set(LIGHTS):
        return None
    scores = {}
    for light in LIGHTS:
        item = evidence[light]
        if not isinstance(item, dict) or not (
            item.get("cross_category_ng") is False
            and item.get("status") in {"CONFLICT", "NO_CONCLUSION"}
            and isinstance(item.get("ok"), dict)
            and isinstance(item.get("ng"), dict)
        ):
            return None
        try:
            ok = float(item["ok"]["best_similarity"])
            ng = float(item["ng"]["best_similarity"])
            margin = float(item["ng_margin_over_ok"])
        except (TypeError, ValueError, KeyError):
            return None
        if not (
            all(math.isfinite(v) for v in (ok, ng, margin))
            and abs((ng - ok) - margin) <= .00001
        ):
            return None
        if light == "SIDE":
            # SIDE may be near tie; do not let NG win materially.
            if ok < .945 or margin > .005:
                return None
        elif ok < .99 or margin > -.045:
            return None
        scores[light] = {"ok": round(ok, 6), "ng": round(ng, 6),
                         "ng_minus_ok": round(margin, 6)}

    matched = {}
    for light, minimum_a, minimum_b in (
        ("TOP", .99, .98), ("MID", .97, .94),
    ):
        light_neighbors = neighbors.get(light)
        if not isinstance(light_neighbors, dict):
            return None
        expected = _OPERATOR_OK_MULTI_LIGHT_REFERENCES[light]
        result = {}
        for key, minimum in (("neighbors_A", minimum_a),
                             ("neighbors_B", minimum_b)):
            rows = light_neighbors.get(key)
            if not isinstance(rows, list) or not rows:
                return None
            first = rows[0]
            if not isinstance(first, dict) or (
                first.get("label") != "OK"
                or (first.get("path"), first.get("source_pixel_sha256")) != expected
            ):
                return None
            try:
                sim = float(first["similarity"])
            except (TypeError, ValueError, KeyError):
                return None
            if not math.isfinite(sim) or not minimum <= sim <= 1:
                return None
            result[key] = round(sim, 8)
        matched[light] = {
            "source_path": expected[0], "source_pixel_sha256": expected[1],
            **result,
        }
    return {
        "schema": "visionx.recurrent_ok_two_light_operator_review.v1",
        "status": "OPERATOR_CONFIRMED_OK_REFERENCE_TOP_MID",
        "reference_event": "2026-10-27_1318_FALTANDO",
        "per_light_reference": matched,
        "knn_visual_light_scores": scores,
        "current_pair_exact_match": False,
        "current_capture_operator_confirmed": False,
        "operator_review_required": True,
        "automatic_ok_authorized": False,
        "xp_command_authorized": False,
    }



def _recognized_recurrent_ok_20261029(
    fused: dict, report: dict, category: str,
) -> dict | None:
    """Recovery of the previously human-confirmed 27 Oct TOP/MID OK event.

    The experimental CNN votes NG on the same component under all lights,
    but its own three-light consensus is invalid. Two PINNED archival
    identities and an independent visual comparison are required. This
    function classifies for a human; it NEVER authorizes automatic XP OK.
    """
    d = fused.get("detail")
    if canonical_memory_category(category) != "FALTANDO" or not isinstance(d, dict):
        return None
    if not (
        fused.get("multilight_final") is True
        and fused.get("verdict") == "DEFEITO REAL"
        and fused.get("is_defect") is True
        and d.get("recognition_route") == "NEW_CNN"
        and d.get("cnn_v2_trial_manual_only") is True
        and d.get("cnn_v2_consensus") == "REVIEW"
        and d.get("cnn_v2_consensus_reason") == "INVALID_LIGHT_EVIDENCE"
        and set(d.get("multilight_strong_positive_modes") or ()) == set(LIGHTS)
        and d.get("multilight_physical_disagreement") is False
        and d.get("multilight_missing_verified_presence") is False
        and d.get("hard_missing_evidence") is not True
        and fused.get("hard_missing_evidence") is not True
        and d.get("physical_score") == 0.0
    ):
        return None
    trace = d.get("decision_trace")
    if not isinstance(trace, dict) or any(
        trace.get(k) is True for k in ("raw_hard_missing_evidence",
                                      "hard_missing_evidence")
    ):
        return None
    physical = d.get("physical_missing_veto")
    if isinstance(physical, dict) and physical.get("status") in {
        "SUSPECT_ABSENCE", "UNAVAILABLE", "NG_CORROBORATED",
    }:
        return None

    # All three CNN outputs must be genuine, same checkpoint and category.
    # A very high uncalibrated CNN score is NOT physical NG confirmation.
    cnn = d.get("cnn_v2_light_diagnostics")
    if not isinstance(cnn, dict) or set(cnn) != set(LIGHTS):
        return None
    checkpoints = set()
    for light in LIGHTS:
        item = cnn[light]
        if not isinstance(item, dict) or not (
            item.get("cnn_active") is True
            and item.get("checkpoint_verified") is True
            and item.get("cnn_status") == "INFERENCE_OK"
            and item.get("route") == "NEW_CNN"
            and item.get("verdict") == "DEFEITO REAL"
            and item.get("category") == "FALTANDO"
        ):
            return None
        sha = item.get("checkpoint_sha256")
        try:
            score = float(item["ng_score_uncalibrated"])
        except (TypeError, ValueError, KeyError, OverflowError):
            return None
        if not (
            isinstance(sha, str) and len(sha) == 64
            and all(ch in "0123456789abcdef" for ch in sha.lower())
            and math.isfinite(score) and .95 <= score <= 1.0
        ):
            return None
        checkpoints.add(sha)
    if len(checkpoints) != 1:
        return None

    votes = d.get("knn_c_light_observer_votes")
    neighbors = d.get("knn_c_light_neighbor_evidence")
    if not isinstance(votes, dict) or set(votes) != set(LIGHTS):
        return None
    if not isinstance(neighbors, dict) or set(neighbors) != set(LIGHTS):
        return None
    matched = {}
    for light in LIGHTS:
        vote = votes[light]
        if not isinstance(vote, dict) or not (
            vote.get("schema") == "visionx.knn_c_live_review_gated.v1"
            and vote.get("bank_status") == "READY"
            and vote.get("diagnostic_only") is True
            and vote.get("auto_decision_authorized") is False
            and vote.get("vote_A") == "OK"
            and vote.get("vote_B") == "OK"
        ):
            return None
        if light in {"SIDE", "TOP"}:
            if not (
                vote.get("status") == "REVISAO"
                and vote.get("reason") == "SEM_SUGESTAO_CONFIAVEL"
                and vote.get("suggestion_A") == "REVISAO"
                and vote.get("suggestion_B") == "REVISAO"
            ):
                return None
        elif not (
            vote.get("status") == "SUGERE_OK"
            and vote.get("reason") == "B_SELETIVA_E_VOTOS_A_B_CONCORDAM"
            and vote.get("suggestion_A") == "REVISAO"
            and vote.get("suggestion_B") == "SUGERE_OK"
        ):
            return None
        groups = neighbors.get(light)
        if not isinstance(groups, dict):
            return None
        for key in ("neighbors_A", "neighbors_B"):
            rows = groups.get(key)
            if not isinstance(rows, list) or len(rows) != 5 or not all(
                isinstance(row, dict) and row.get("label") in {"OK", "NG"}
                for row in rows
            ):
                return None
            min_ok = 3 if light == "SIDE" else 4
            if sum(row.get("label") == "OK" for row in rows) < min_ok:
                return None
            def valid_score(row):
                try:
                    value = float(row["similarity"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    return None
                return value if math.isfinite(value) and 0 <= value <= 1 else None
            if any(valid_score(row) is None for row in rows):
                return None
            ng_scores = [valid_score(row) for row in rows if row["label"] == "NG"]
            if ng_scores and max(ng_scores) >= .93:
                return None
            if light != "SIDE":
                first = rows[0]
                expected = _OPERATOR_OK_MULTI_LIGHT_REFERENCES[light]
                if not (
                    first["label"] == "OK"
                    and (first.get("path"), first.get("source_pixel_sha256"))
                        == expected
                ):
                    return None
                minimum = {
                    "TOP": {"neighbors_A": .985, "neighbors_B": .975},
                    "MID": {"neighbors_A": .98, "neighbors_B": .97},
                }[light][key]
                sim = valid_score(first)
                if sim < minimum or (ng_scores and sim - max(ng_scores) < .06):
                    return None
                matched.setdefault(light, {
                    "path": expected[0], "source_pixel_sha256": expected[1],
                })[key] = round(sim, 8)

    if not isinstance(report, dict) or not (
        report.get("schema") == SCHEMA
        and report.get("status") == "CONFLICT"
        and report.get("strong_ng_lights") == []
        and report.get("trend_ng_lights") == []
    ):
        return None
    evidence = report.get("lights")
    if not isinstance(evidence, dict) or set(evidence) != set(LIGHTS):
        return None
    support = {}
    minimum_ok = {"SIDE": .99, "TOP": .99, "MID": .985}
    minimum_gap = {"SIDE": .04, "TOP": .045, "MID": .06}
    for light in LIGHTS:
        item = evidence[light]
        if not isinstance(item, dict) or not (
            item.get("cross_category_ng") is False
            and item.get("status") == (
                "NO_CONCLUSION" if light == "MID" else "CONFLICT"
            )
            and isinstance(item.get("ok"), dict)
            and isinstance(item.get("ng"), dict)
        ):
            return None
        try:
            ok = float(item["ok"]["best_similarity"])
            ng = float(item["ng"]["best_similarity"])
            margin = float(item["ng_margin_over_ok"])
        except (KeyError, ValueError, TypeError, OverflowError):
            return None
        if not (
            all(math.isfinite(x) and 0 <= x <= 1 for x in (ok, ng))
            and math.isfinite(margin) and abs((ng - ok) - margin) < .00001
            and ok >= minimum_ok[light]
            and ok - ng >= minimum_gap[light]
        ):
            return None
        support[light] = {
            "ok": round(ok, 6), "ng": round(ng, 6),
            "ok_advantage": round(ok - ng, 6),
        }
    return {
        "schema": "visionx.known_two_light_ok_20261029_review.v1",
        "status": "HUMAN_CONFIRMED_TOP_MID_OK_PATTERN",
        "reference_event": "2026-10-27_1318_FALTANDO",
        "per_light_reference": matched,
        "independent_visual_support": support,
        "current_pair_exact_match": False,
        "current_capture_operator_confirmed": False,
        "operator_review_required": True,
        "automatic_ok_authorized": False,
        "xp_command_authorized": False,
    }


def _mixed_light_ok_candidate(
    fused: dict, report: dict, category: str,
) -> dict | None:
    """Tentative false-alarm label for the supervised three-light pattern.

    Does not assert an exact KNN pair or physical certainty. Two CNN lights
    lean OK, two KNN-C lights vote OK, three independent visual comparisons
    lean OK, and SIDE has only *weak* NG neighbors. XP stays manual.
    """
    if canonical_memory_category(category) != "FALTANDO":
        return None
    d = fused.get("detail")
    if not isinstance(d, dict) or not (
        fused.get("multilight_final") is True
        and fused.get("verdict") == "REVISÃO OBRIGATÓRIA"
        and fused.get("is_defect") is False
        and fused.get("production_review_required") is True
        and d.get("fusion_rule") == "multilight_local_review"
        and d.get("recognition_route") == "NEW_CNN"
        and d.get("cnn_v2_trial_manual_only") is True
        and d.get("cnn_v2_consensus") == "REVIEW"
        and d.get("cnn_v2_consensus_reason") == "INVALID_LIGHT_EVIDENCE"
        and d.get("multilight_positive_modes") == []
        and d.get("multilight_strong_positive_modes") == []
        and set(d.get("multilight_review_modes") or ()) == set(LIGHTS)
        and d.get("multilight_physical_disagreement") is False
        and d.get("multilight_missing_verified_presence") is False
        and d.get("hard_missing_evidence") is not True
        and fused.get("hard_missing_evidence") is not True
    ):
        return None
    try:
        physical_score = float(d["physical_score"])
    except (TypeError, ValueError, KeyError):
        return None
    if not math.isfinite(physical_score) or physical_score != 0:
        return None
    physical = d.get("physical_missing_veto")
    if not isinstance(physical, dict) or physical.get("status") in {
        "SUSPECT_ABSENCE", "NG_CORROBORATED",
    }:
        return None
    trace = d.get("decision_trace")
    if not isinstance(trace, dict) or (
        trace.get("raw_hard_missing_evidence") is True
        or trace.get("hard_missing_evidence") is True
    ):
        return None

    diagnostics = d.get("cnn_v2_light_diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != set(LIGHTS):
        return None
    ng_scores = {}
    checkpoints = set()
    for light in LIGHTS:
        item = diagnostics[light]
        if not isinstance(item, dict) or not (
            item.get("cnn_active") is True
            and item.get("checkpoint_verified") is True
            and item.get("cnn_status") == "INFERENCE_OK"
            and item.get("route") == "NEW_CNN"
            and item.get("verdict") == "REVISÃO OBRIGATÓRIA"
            and item.get("category") == "FALTANDO"
        ):
            return None
        sha = item.get("checkpoint_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            return None
        checkpoints.add(sha)
        try:
            score = float(item["ng_score_uncalibrated"])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(score) or not 0 <= score <= 1:
            return None
        ng_scores[light] = round(score, 8)
    # Reject a different checkpoint per light or a materially stronger
    # NG signal in either of the two independent dark-light captures.
    if len(checkpoints) != 1 or not (
        ng_scores["SIDE"] <= .30
        and ng_scores["TOP"] <= .30
        and ng_scores["MID"] <= .90
    ):
        return None

    votes = d.get("knn_c_light_observer_votes")
    neighbor_sets = d.get("knn_c_light_neighbor_evidence")
    if not isinstance(votes, dict) or set(votes) != set(LIGHTS):
        return None
    if not isinstance(neighbor_sets, dict) or set(neighbor_sets) != set(LIGHTS):
        return None
    matched_knn = {}
    for light in LIGHTS:
        vote = votes[light]
        if not isinstance(vote, dict) or not (
            vote.get("schema") == "visionx.knn_c_live_review_gated.v1"
            and vote.get("bank_status") == "READY"
            and vote.get("status") == "REVISAO"
            and vote.get("reason") == "SEM_SUGESTAO_CONFIAVEL"
            and vote.get("diagnostic_only") is True
            and vote.get("auto_decision_authorized") is False
            and vote.get("suggestion_A") == "REVISAO"
            and vote.get("suggestion_B") == "REVISAO"
        ):
            return None
        try:
            similarity_a = float(vote["similarity_A"])
            similarity_b = float(vote["similarity_B"])
        except (TypeError, ValueError, KeyError):
            return None
        if not all(math.isfinite(v) and 0 <= v <= 1
                   for v in (similarity_a, similarity_b)):
            return None
        if light == "SIDE":
            if not (
                vote.get("vote_A") == "NG"
                and vote.get("vote_B") == "NG"
                and similarity_a <= .88
                and similarity_b <= .83
            ):
                return None
        elif not (vote.get("vote_A") == "OK" and vote.get("vote_B") == "OK"):
            return None
        groups = neighbor_sets.get(light)
        if not isinstance(groups, dict):
            return None
        result = {}
        for key in ("neighbors_A", "neighbors_B"):
            rows = groups.get(key)
            if not isinstance(rows, list) or len(rows) != 5:
                return None
            if not all(isinstance(row, dict) and row.get("label") in {"OK", "NG"}
                       for row in rows):
                return None
            if light != "SIDE":
                if rows[0].get("label") != "OK" or sum(
                    row["label"] == "OK" for row in rows
                ) < 4:
                    return None
            result[key] = {
                "ok_count": sum(row["label"] == "OK" for row in rows),
                "ng_count": sum(row["label"] == "NG" for row in rows),
            }
        matched_knn[light] = result

    if not isinstance(report, dict) or not (
        report.get("schema") == SCHEMA
        and report.get("status") == "CONFLICT"
        and report.get("strong_ng_lights") == []
        and report.get("trend_ng_lights") == []
    ):
        return None
    lights = report.get("lights")
    if not isinstance(lights, dict) or set(lights) != set(LIGHTS):
        return None
    support = {}
    limits = {"SIDE": (.94, .012), "TOP": (.97, .030), "MID": (.96, .025)}
    for light in LIGHTS:
        item = lights[light]
        if not isinstance(item, dict) or not (
            item.get("status") == "CONFLICT"
            and item.get("cross_category_ng") is False
            and isinstance(item.get("ok"), dict)
            and isinstance(item.get("ng"), dict)
        ):
            return None
        try:
            ok = float(item["ok"]["best_similarity"])
            ng = float(item["ng"]["best_similarity"])
            margin = float(item["ng_margin_over_ok"])
        except (TypeError, ValueError, KeyError):
            return None
        if not (
            all(math.isfinite(v) and 0 <= v <= 1 for v in (ok, ng))
            and math.isfinite(margin)
            and abs((ng - ok) - margin) <= .00001
            and ok >= limits[light][0]
            and -margin >= limits[light][1]
        ):
            return None
        support[light] = {
            "best_ok_similarity": round(ok, 6),
            "best_ng_similarity": round(ng, 6),
            "ng_minus_ok": round(margin, 6),
        }
    return {
        "schema": "visionx.mixed_light_false_alarm_review.v1",
        "status": "SUGESTAO_FALHA_FALSA_COM_REVISÃO_HUMANA",
        "cnn_ng_scores_uncalibrated": ng_scores,
        "knn_c_neighbor_votes": matched_knn,
        "three_light_visual_support": support,
        "exact_pair_verified": False,
        "current_capture_operator_confirmed": False,
        "operator_review_required": True,
        "automatic_ok_authorized": False,
        "xp_command_authorized": False,
    }


def _operator_corrected_side_ok_recall(
    fused: dict, report: dict, category: str,
) -> dict | None:
    """Recall canonical OK archive after erroneous historical NG was retired.

    Requires the wrong NG source to be absent. Does NOT infer byte-identical
    pair from approximate KNN similarities.
    """
    from src.core.operator_confirmed_archive_labels import (
        CANONICAL_OK_SOURCE, CANONICAL_OK_FILE_SHA256,
        CANONICAL_OK_PIXEL_SHA256, verified_canonical_ok_source,
        canonical_ok_source_status, RETIRED_SOURCE,
    )
    if canonical_memory_category(category) != "FALTANDO":
        return None
    d = fused.get("detail")
    def reject(reason: str):
        if isinstance(d, dict):
            d["operator_corrected_archive_ok_diagnostic"] = {
                "status": "BLOCKED", "reason": reason,
                "reference": "public/ok_archive/2026-10-21_1416_FALTANDO.png",
                "exact_pair_claimed": False,
            }
        return None
    if not isinstance(d, dict) or not (
        fused.get("multilight_final") is True
        and fused.get("verdict") == "REVISÃO OBRIGATÓRIA"
        and fused.get("is_defect") is False
        and d.get("fusion_rule") == "multilight_local_review"
        and d.get("recognition_route") == "NEW_CNN"
        and d.get("cnn_v2_consensus") == "REVIEW"
        and d.get("cnn_v2_consensus_reason") == "INVALID_LIGHT_EVIDENCE"
        and d.get("multilight_strong_positive_modes") == []
        and set(d.get("multilight_review_modes") or ()) == set(LIGHTS)
        and d.get("multilight_physical_disagreement") is False
        and d.get("multilight_missing_verified_presence") is False
        and d.get("hard_missing_evidence") is not True
        and fused.get("hard_missing_evidence") is not True
    ):
        return reject("MULTILIGHT_BASE_CONTRACT_FAILED")
    if d.get("physical_score") != 0.0:
        return reject("PHYSICAL_SCORE_NOT_ZERO")
    physical = d.get("physical_missing_veto")
    from src.core.physical_missing_veto import POLICY as PHYSICAL_PROBE_POLICY
    if not isinstance(physical, dict) or not (
        physical.get("policy") == PHYSICAL_PROBE_POLICY
        and physical.get("diagnostic_only") is True
        and physical.get("authorizes_ng") is False
        and physical.get("authorizes_ok") is False
        and physical.get("suspect_lights") == []
        and physical.get("unavailable_lights") == []
        and physical.get("status") in {"NO_STRONG_EVIDENCE", "GEOMETRY_REVIEW"}
    ):
        return reject("PHYSICAL_VETO_NOT_CLEAR")
    physical_rows = physical.get("lights")
    geometry_lights = physical.get("geometry_review_lights")
    if not isinstance(physical_rows, dict) or set(physical_rows) != set(LIGHTS):
        return reject("PHYSICAL_VETO_LIGHTS_INCOMPLETE")
    if not isinstance(geometry_lights, list) or any(
        light not in LIGHTS for light in geometry_lights
    ):
        return reject("PHYSICAL_VETO_GEOMETRY_INVALID")
    observed_geometry = {
        light for light in LIGHTS
        if isinstance(physical_rows[light], dict)
        and physical_rows[light].get("status") == "GEOMETRY_REVIEW"
    }
    if (
        any(
            not isinstance(physical_rows[light], dict)
            or physical_rows[light].get("status") not in {
                "NO_STRONG_EVIDENCE", "GEOMETRY_REVIEW"
            }
            for light in LIGHTS
        )
        or set(geometry_lights) != observed_geometry
        or (physical.get("status") == "NO_STRONG_EVIDENCE"
            and observed_geometry)
        or (physical.get("status") == "GEOMETRY_REVIEW"
            and not observed_geometry)
        or physical.get("review_required")
        is not (physical.get("status") == "GEOMETRY_REVIEW")
    ):
        return reject("PHYSICAL_VETO_INCONSISTENT")
    trace = d.get("decision_trace")
    if not isinstance(trace, dict) or (
        trace.get("raw_hard_missing_evidence") is True
        or trace.get("hard_missing_evidence") is True
    ):
        return reject("RAW_HARD_MISSING_EVIDENCE_PRESENT")
    # The retired NG is no longer authoritative; only the intact OK PNG
    # confirmed by its own file SHA may serve as the positive reference.
    if not verified_canonical_ok_source():
        return reject("CANONICAL_IMAGE_" + canonical_ok_source_status()["reason"])
    diag = d.get("cnn_v2_light_diagnostics")
    if not isinstance(diag, dict) or set(diag) != set(LIGHTS):
        return reject("CNN_THREE_LIGHT_DIAGNOSTICS_MISSING")
    model_sha = set()
    cnn_scores = {}
    for light, ceiling in (("SIDE", .12), ("TOP", .30), ("MID", .90)):
        item = diag[light]
        if not isinstance(item, dict) or not (
            item.get("route") == "NEW_CNN"
            and item.get("category") == "FALTANDO"
            and item.get("cnn_active") is True
            and item.get("checkpoint_verified") is True
            and item.get("cnn_status") == "INFERENCE_OK"
            and item.get("verdict") in (
                {"FALHA FALSA", "REVISÃO OBRIGATÓRIA"}
                if light == "SIDE" else {"REVISÃO OBRIGATÓRIA"}
            )
        ):
            return reject("CNN_LIGHT_INVALID_" + light)
        checksum = item.get("checkpoint_sha256")
        if not isinstance(checksum, str) or len(checksum) != 64:
            return reject("CNN_CHECKPOINT_SHA_INVALID_" + light)
        model_sha.add(checksum)
        try:
            score = float(item["ng_score_uncalibrated"])
        except (TypeError, ValueError, KeyError):
            return reject("CNN_SCORE_UNAVAILABLE_" + light)
        if not math.isfinite(score) or not 0 <= score <= ceiling:
            return reject("CNN_SCORE_OUTSIDE_SAFE_RANGE_" + light)
        cnn_scores[light] = round(score, 8)
    if len(model_sha) != 1:
        return reject("CNN_CHECKPOINTS_DIFFER")
    votes = d.get("knn_c_light_observer_votes")
    neighbours = d.get("knn_c_light_neighbor_evidence")
    if (not isinstance(votes, dict) or set(votes) != set(LIGHTS)
            or not isinstance(neighbours, dict) or set(neighbours) != set(LIGHTS)):
        return reject("KNN_C_THREE_LIGHT_EVIDENCE_MISSING")
    matched = {}
    for light in LIGHTS:
        v = votes[light]
        if not isinstance(v, dict) or not (
            v.get("schema") == "visionx.knn_c_live_review_gated.v1"
            and v.get("bank_status") == "READY"
            and v.get("diagnostic_only") is True
            and v.get("auto_decision_authorized") is False
        ):
            return reject("KNN_C_NOT_READY_" + light)
        if light in {"TOP", "MID"} and not (
            v.get("vote_A") == "OK" and v.get("vote_B") == "OK"
            and v.get("status") == "REVISAO"
        ):
            return reject("KNN_C_TOP_MID_NOT_OK_" + light)
        rows = neighbours[light]
        if not isinstance(rows, dict):
            return reject("KNN_C_NEIGHBORS_UNAVAILABLE_" + light)
        matched[light] = {}
        for key, minimum in (("neighbors_A", .85 if light == "SIDE" else 0),
                             ("neighbors_B", .82 if light == "SIDE" else 0)):
            group = rows.get(key)
            if not isinstance(group, list) or len(group) < 5 or not all(
                isinstance(row, dict) and row.get("label") in {"OK", "NG"}
                for row in group[:5]
            ):
                return reject("KNN_C_INSUFFICIENT_NEIGHBORS_" + light + "_" + key)
            first = group[0]
            if light == "SIDE":
                if not (
                    first.get("path") == CANONICAL_OK_SOURCE
                    and first.get("source_pixel_sha256")
                    == CANONICAL_OK_PIXEL_SHA256
                    and first.get("label") == "OK"
                    and all(row.get("path") != RETIRED_SOURCE
                            for row in group)
                ):
                    return reject("CANONICAL_OK_NOT_NEAREST_SIDE_" + key)
                try:
                    similarity = float(first["similarity"])
                    other_ng = max(
                        (float(row["similarity"]) for row in group[1:5]
                         if row["label"] == "NG"), default=0.0,
                    )
                except (TypeError, ValueError, KeyError):
                    return reject("KNN_SIDE_SIMILARITY_INVALID_" + key)
                if (not math.isfinite(similarity) or similarity < minimum
                        or not math.isfinite(other_ng)
                        or other_ng >= .90
                        or other_ng > similarity - .015):
                    return reject("KNN_SIDE_NG_COUNTEREVIDENCE_" + key)
                matched[light][key] = round(similarity, 8)
            else:
                if first.get("label") != "OK" or sum(
                    row["label"] == "OK" for row in group[:5]
                ) < 4:
                    return reject("KNN_TOP_MID_OK_SUPPORT_INSUFFICIENT_" + light + "_" + key)
                matched[light][key] = sum(
                    row["label"] == "OK" for row in group[:5]
                )
    if not isinstance(report, dict) or not (
        report.get("schema") == SCHEMA
        and report.get("status") == "CONFLICT"
        and report.get("strong_ng_lights") == []
        and report.get("trend_ng_lights") == []
    ):
        return reject("VISUAL_THREE_LIGHT_CONFLICT_CONTRACT_FAILED")
    lights = report.get("lights")
    if not isinstance(lights, dict) or set(lights) != set(LIGHTS):
        return reject("VISUAL_LIGHT_EVIDENCE_MISSING")
    supports = {}
    for light, min_ok, max_margin in (
        ("SIDE", .97, .015), ("TOP", .97, -.03), ("MID", .96, -.025),
    ):
        item = lights[light]
        if not isinstance(item, dict) or (
            item.get("status") not in {"CONFLICT", "NO_CONCLUSION"}
            or item.get("cross_category_ng") is not False
            or not isinstance(item.get("ok"), dict)
            or not isinstance(item.get("ng"), dict)
        ):
            return reject("VISUAL_LIGHT_STATUS_OR_FORMAT_INVALID_" + light)
        try:
            ok = float(item["ok"]["best_similarity"])
            ng = float(item["ng"]["best_similarity"])
            delta = float(item["ng_margin_over_ok"])
        except (TypeError, ValueError, KeyError):
            return reject("VISUAL_LIGHT_SCORES_UNAVAILABLE_" + light)
        if not (all(math.isfinite(v) for v in (ok, ng, delta))
                and 0 <= ok <= 1 and 0 <= ng <= 1
                and abs(ng - ok - delta) <= .00001
                and ok >= min_ok and delta <= max_margin):
            return reject("VISUAL_LIGHT_COUNTEREVIDENCE_" + light)
        supports[light] = {
            "best_ok": round(ok, 6), "best_ng": round(ng, 6),
            "ng_minus_ok": round(delta, 6),
        }
    d["operator_corrected_archive_ok_diagnostic"] = {
        "status": "RECOGNIZED", "reason": "CANONICAL_OK_VISUAL_RECALL_VERIFIED",
        "reference": CANONICAL_OK_SOURCE,
        "exact_pair_claimed": False,
    }
    return {
        "schema": "visionx.operator_corrected_archive_ok_recall.v1",
        "quality_label": "OK",
        "historical_source_path": CANONICAL_OK_SOURCE,
        "historical_source_file_sha256": CANONICAL_OK_FILE_SHA256,
        "historical_source_pixel_sha256": CANONICAL_OK_PIXEL_SHA256,
        "retired_wrong_ng_source": RETIRED_SOURCE,
        "side_matches": matched["SIDE"],
        "knn_top_mid_ok_counts": {
            "TOP": matched["TOP"], "MID": matched["MID"],
        },
        "visual_support": supports, "cnn_scores": cnn_scores,
        "current_pair_exact_match": False,
        "source_human_quality_verified": True,
        "physical_probe_status": physical["status"],
        "physical_geometry_review_lights": sorted(observed_geometry),
        "automatic_ok_only_by_production_gate": True,
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
    corrected = _operator_corrected_side_ok_recall(fused, report, category)
    if corrected is not None:
        detail["operator_corrected_archive_ok_recall"] = corrected
        detail["verdict_before_corrected_archive_ok"] = fused["verdict"]
        fused["verdict"] = "FALHA FALSA"
        fused["is_defect"] = False
        fused["production_review_required"] = True
        fused["eligible_for_final_decision"] = False
        fused["confidence"] = .5
        fused["reason"] = (
            "FALHA FALSA • referência válida SIDE recuperada no "
            "ok_archive com SHA verificado; rótulo NG histórico incorreto "
            "retirado da memória. TOP/MID têm maioria OK e assinaturas "
            "favoráveis, sem evidência física de ausência. "
            "Correspondência visual aproximada, não par exato. "
            + ("Aparagem geométrica: confirmação humana necessária."
               if corrected["physical_probe_status"] == "GEOMETRY_REVIEW"
               else "Sonda física verificada em três luzes.")
        )
        detail["fusion_rule"] = "operator_corrected_archive_ok_three_light_v1"
        detail["operator_review_required"] = True
        detail["eligible_for_final_decision"] = False
        detail["cnn_v2_supervised_auto_eligible"] = False
        detail["final_score"] = .5
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace.update(verdict="FALHA FALSA", physical_defect=False,
                         operator_review_required=True,
                         fusion_rule=detail["fusion_rule"], final_score=.5)
        return fused
    mixed_ok = _mixed_light_ok_candidate(fused, report, category)
    if mixed_ok is not None:
        detail["mixed_light_ok_candidate"] = mixed_ok
        detail["verdict_before_mixed_light_ok"] = fused["verdict"]
        fused["verdict"] = "FALHA FALSA"
        fused["is_defect"] = False
        fused["production_review_required"] = True
        fused["eligible_for_final_decision"] = False
        fused["confidence"] = .5
        fused["reason"] = (
            "FALHA FALSA • SUGESTÃO MULTILIGHT PARA OPERADOR: CNN "
            "inconclusiva e baixa nas luzes SIDE/TOP, KNN visual favorece "
            "OK nas 3 luzes, KNN C TOP/MID recupera 4+ OK em cada descritor. "
            "KNN C SIDE ainda votou NG com semelhança fraca. "
            "Sem par exato ou evidência física de ausência; confirmar "
            "0=OK / 1=NG manualmente antes de qualquer comando ao XP."
        )
        detail["fusion_rule"] = "mixed_light_knn_ok_cnn_review_manual_v1"
        detail["operator_review_required"] = True
        detail["eligible_for_final_decision"] = False
        detail["cnn_v2_supervised_auto_eligible"] = False
        detail["final_score"] = .5
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace["verdict"] = "FALHA FALSA"
            trace["physical_defect"] = False
            trace["operator_review_required"] = True
            trace["fusion_rule"] = detail["fusion_rule"]
            trace["final_score"] = .5
        return fused
    # Revisão de 29/10: a CNN experimental dispara NG nas três luzes,
    # mas TOP/MID recuperam a MESMA fonte humana OK em A/B com grande
    # separação dos NG. Não confundir isso com par exato ou auto-OK XP.
    known_pair = _recognized_recurrent_ok_20261029(fused, report, category)
    if known_pair is not None:
        detail["recognized_top_mid_ok_20261029"] = known_pair
        detail["cnn_ng_verdict_before_knn_review"] = fused["verdict"]
        fused["verdict"] = "FALHA FALSA"
        fused["is_defect"] = False
        fused["production_review_required"] = True
        fused["eligible_for_final_decision"] = False
        fused["confidence"] = .5
        fused["reason"] = (
            "FALHA FALSA • PADRÃO OK HISTÓRICO RECUPERADO: TOP e MID "
            "reconhecem as imagens 2026-10-27_1318_FALTANDO no acervo OK "
            "pelos dois descritores; a comparação visual independente "
            "favorece OK nas três iluminações e não há evidência física "
            "registrada de ausência. A CNN experimental contradiz a "
            "memória, com consenso inválido. Classificação visual para "
            "confirmação humana: 0=OK / 1=NG. Nenhum XP automático."
        )
        detail["fusion_rule"] = "operator_known_top_mid_ok_review_20261029_v1"
        detail["operator_review_required"] = True
        detail["eligible_for_final_decision"] = False
        detail["cnn_v2_supervised_auto_eligible"] = False
        detail["final_score"] = .5
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace.update(
                verdict="FALHA FALSA", physical_defect=False,
                operator_review_required=True,
                fusion_rule=detail["fusion_rule"], final_score=.5,
            )
        return fused
    # A referência recorrente pode ser muito forte em TOP e MID
    # enquanto SIDE empata; essa rota precede a antiga exigência
    # rígida de três luzes CONFLICT e KNN C em REVISAO.
    recurring = _recognized_two_light_operator_ok(fused, report, category)
    if recurring is not None:
        detail["recognized_two_light_operator_ok"] = recurring
        detail["cnn_ng_verdict_before_knn_review"] = fused["verdict"]
        fused["verdict"] = "FALHA FALSA"
        fused["is_defect"] = False
        fused["production_review_required"] = True
        fused["eligible_for_final_decision"] = False
        fused["confidence"] = .5
        fused["reason"] = (
            "FALHA FALSA • referência OK confirmada recuperada nas "
            "KNN A/B de TOP e MID para o mesmo evento físico; SIDE "
            "não mostrou evidência NG conclusiva, CNN sem consenso "
            "e nenhuma ausência física corroborada. Par atual não "
            "é exatamente idêntico; confirmação humana obrigatória "
            "antes de enviar 0 ou 1 ao XP."
        )
        detail["fusion_rule"] = "operator_recurrent_ok_top_mid_review_v1"
        detail["operator_review_required"] = True
        detail["eligible_for_final_decision"] = False
        detail["cnn_v2_supervised_auto_eligible"] = False
        detail["final_score"] = .5
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace["verdict"] = "FALHA FALSA"
            trace["physical_defect"] = False
            trace["operator_review_required"] = True
            trace["fusion_rule"] = detail["fusion_rule"]
            trace["final_score"] = .5
        return fused
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
