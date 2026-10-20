"""KNN visual seletiva para replay OFFLINE do acervo, sem critério temporal.

Todos os vizinhos são pares gabarito/captura da mesma categoria+iluminação,
desduplicados por identidade visual e com o próprio evento excluído.
Somente diagnóstico: não participa da KNN exata, fusão ou comandos XP.
"""
from __future__ import annotations

import math

SCHEMA = "visionx.knn_archive_visual_candidate.v2"
K = 5
# Similaridade do novo descritor global/moldura/epicentro; valor provisório.
MIN_SIMILARITY = .90
MIN_INDEPENDENT_COUNTEREXAMPLES = 2
LIGHTS = frozenset(("SIDE", "TOP", "MID"))


def select_shadow_candidate(
    neighbors: list[dict], *, category: str, lighting: str,
    scope_examples: dict[str, int] | None = None,
) -> dict:
    """Retorna SUGERE_OK, SUGERE_NG ou REVISAO sem consultar o rótulo real."""
    result = {
        "schema": SCHEMA, "diagnostic_only": True,
        "auto_decision_authorized": False, "xp_command": None,
        "similarity_is_probability": False, "category": str(category),
        "lighting_mode": str(lighting), "status": "REVISAO",
        "reason": "EVIDENCIA_INSUFICIENTE", "best_similarity": None,
        "neighbors_considered": 0,
    }
    if str(lighting).upper() not in LIGHTS or not str(category).strip():
        result["reason"] = "ESCOPO_OCR_OU_LUZ_INVALIDO"
        return result
    if not isinstance(neighbors, list) or not isinstance(scope_examples, dict):
        result["reason"] = "SEM_CONTRAPROVA_VISUAL"
        return result
    if min(int(scope_examples.get(x) or 0) for x in ("OK", "NG")) < 1:
        result["reason"] = "SEM_CONTRAPROVA_VISUAL"
        return result

    ranked = []
    seen = set()
    for item in neighbors:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").upper()
        sim = item.get("similarity")
        fingerprint = item.get("source_pixel_sha256") or item.get("source_path")
        if label not in ("OK", "NG") or not isinstance(sim, (int, float)):
            continue
        if not math.isfinite(float(sim)) or not 0 <= float(sim) <= 1:
            continue
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        ranked.append((float(sim), label, str(fingerprint)))
    ranked.sort(key=lambda r: (-r[0], r[2]))
    ranked = ranked[:K]
    result["neighbors_considered"] = len(ranked)
    if not ranked:
        result["reason"] = "SEM_VIZINHOS_VISUAIS"
        return result
    result["best_similarity"] = round(ranked[0][0], 8)
    if len(ranked) < K:
        result["reason"] = "MENOS_DE_CINCO_EXEMPLOS_VISUAIS"
        return result
    if ranked[0][0] < MIN_SIMILARITY:
        result["reason"] = "SIMILARIDADE_VISUAL_INSUFICIENTE"
        return result

    ok = [r for r in ranked if r[1] == "OK"]
    ng = [r for r in ranked if r[1] == "NG"]
    if len(ok) == K:
        # Mesmo cinco OK semelhantes não justificam ignorar NG se a
        # categoria/iluminação não tem contraprovas visuais suficientes.
        if int(scope_examples.get("NG") or 0) < MIN_INDEPENDENT_COUNTEREXAMPLES:
            result["reason"] = "NG_SEM_CONTRAPROVA_VISUAL"
        else:
            result["status"] = "SUGERE_OK"
            result["reason"] = "CINCO_OK_VISUALMENTE_CONCORDANTES"
    elif len(ng) >= 3 and int(scope_examples.get("NG") or 0) >= MIN_INDEPENDENT_COUNTEREXAMPLES:
        result["status"] = "SUGERE_NG"
        result["reason"] = "MAIORIA_NG_VISUAL"
    else:
        result["reason"] = "VIZINHOS_VISUAIS_DIVERGENTES"
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
