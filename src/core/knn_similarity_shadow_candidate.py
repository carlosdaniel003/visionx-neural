"""Classificador seletivo KNN aproximado, EXCLUSIVO para shadow/diagnóstico.

Nunca modifica a decisão da linha, o verificador de pares exatos ou o XP.
Usa somente os vizinhos da mesma categoria+iluminação preparados pelo
chamador e não recebe o rótulo da consulta.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

SCHEMA = "visionx.knn_shadow_similarity_candidate.v1"
K = 5
MIN_SIMILARITY = 0.94
MIN_DAYS_SAME_CLASS = 2
DATE = re.compile(r"(20\d{2}[-_]?[01]\d[-_]?[0-3]\d)")
LIGHTS = frozenset(("SIDE", "TOP", "MID"))


def _day(source: str) -> str | None:
    hit = DATE.search(Path(str(source)).name)
    return hit.group(1).replace("_", "-") if hit else None


def select_shadow_candidate(
    neighbors: list[dict], *, category: str, lighting: str,
    scope_days: dict[str, int] | None = None,
) -> dict:
    """Sugere OK/NG ou abstém; nunca gera KNOWN nem autorização automática.

    scope_days conta datas distintas com OK e NG na categoria+iluminação.
    Para sugerir OK, exige contraprova NG disponível na memória, mesmo que
    os cinco vizinhos mais próximos sejam OK. Sem scope comprovado: revisão.
    """
    result = {
        "schema": SCHEMA, "diagnostic_only": True,
        "auto_decision_authorized": False, "xp_command": None,
        "similarity_is_probability": False, "category": str(category),
        "lighting_mode": str(lighting), "status": "REVISAO",
        "reason": "EVIDENCIA_INSUFICIENTE",
        "best_similarity": None, "neighbors_considered": 0,
    }
    if str(lighting).upper() not in LIGHTS or not str(category).strip():
        result["reason"] = "ESCOPO_OCR_OU_LUZ_INVALIDO"
        return result
    if not isinstance(neighbors, list) or not isinstance(scope_days, dict):
        result["reason"] = "SEM_MEMORIA_DAS_DUAS_CLASSES"
        return result
    if min(int(scope_days.get(x) or 0) for x in ("OK", "NG")) < 1:
        result["reason"] = "SEM_MEMORIA_DAS_DUAS_CLASSES"
        return result

    ranked = []
    for item in neighbors:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").upper()
        sim = item.get("similarity")
        source = item.get("source_path") or item.get("source") or ""
        if label not in ("OK", "NG") or not isinstance(sim, (int, float)):
            continue
        if not math.isfinite(float(sim)) or not 0 <= float(sim) <= 1:
            continue
        day = _day(str(source))
        if not day:
            continue
        ranked.append((float(sim), label, day, str(source)))
    ranked.sort(key=lambda r: (-r[0], r[3]))
    ranked = ranked[:K]
    result["neighbors_considered"] = len(ranked)
    if not ranked:
        result["reason"] = "SEM_VIZINHOS_VALIDOS"
        return result
    result["best_similarity"] = round(ranked[0][0], 8)
    if len(ranked) < K:
        result["reason"] = "MENOS_DE_CINCO_VIZINHOS"
        return result
    if ranked[0][0] < MIN_SIMILARITY:
        result["reason"] = "SIMILARIDADE_INSUFICIENTE"
        return result

    ng = [r for r in ranked if r[1] == "NG"]
    ok = [r for r in ranked if r[1] == "OK"]
    if len(ok) == K:
        # Sem exemplos NG em pelo menos dois dias, a semelhança com OK
        # NÃO é base para liberar uma possível peça NG inédita.
        if int(scope_days.get("NG") or 0) < MIN_DAYS_SAME_CLASS:
            result["reason"] = "NG_SEM_COBERTURA_INDEPENDENTE"
        elif len({r[2] for r in ok}) < MIN_DAYS_SAME_CLASS:
            result["reason"] = "OK_SEM_DIVERSIDADE_DE_DIAS"
        else:
            result["status"] = "SUGERE_OK"
            result["reason"] = "CINCO_OK_CONCORDANTES_COM_CONTRAPROVA_NG"
    elif len(ng) >= 3 and len({r[2] for r in ng}) >= MIN_DAYS_SAME_CLASS:
        result["status"] = "SUGERE_NG"
        result["reason"] = "MAIORIA_NG_EM_DIAS_DISTINTOS"
    else:
        result["reason"] = "VIZINHOS_DIVERGENTES_OU_POUCO_DIVERSOS"
    return result


def summarize_shadow_candidates(cases: list[dict]) -> dict:
    by_label = {}
    for label in ("OK", "NG"):
        rows = [r for r in cases if r.get("expected_label") == label]
        selections = {k: sum(r.get("shadow_candidate", {}).get("status") == k
                             for r in rows)
                      for k in ("SUGERE_OK", "SUGERE_NG", "REVISAO")}
        by_label[label] = {
            "total": len(rows), **selections,
            "correct_suggestions": selections["SUGERE_" + label],
            "incorrect_suggestions": selections[
                "SUGERE_" + ("NG" if label == "OK" else "OK")
            ],
        }
    return {
        "by_label": by_label,
        "NG_suggested_OK": by_label["NG"]["SUGERE_OK"],
        "OK_suggested_NG": by_label["OK"]["SUGERE_NG"],
        "reviews": (by_label["OK"]["REVISAO"] + by_label["NG"]["REVISAO"]),
        "diagnostic_only": True, "production_approved": False,
    }


__all__ = ["select_shadow_candidate", "summarize_shadow_candidates", "SCHEMA"]
