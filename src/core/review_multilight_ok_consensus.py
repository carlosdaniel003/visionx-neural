"""Arbitragem visual multilight quando a CNN trial deixa três luzes em revisão.

Apenas exibe candidato FALHA FALSA quando CNN, memória humana confirmada,
assinaturas visuais e descritores geométricos são coerentes. A categoria AOI
não constitui prova do defeito físico. Não autoriza PRESS_0 ou PRESS_1,
não declara par RGB exato e não alimenta memória/treino automaticamente.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
import re

from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.operator_visual_memory_consensus import _no_hard_ng
from src.core.strict_category_memory import canonical_memory_category

LIGHTS = ("SIDE", "TOP", "MID")
SCHEMA = "visionx.review_multilight_ok_consensus.v1"
RULE = "cnn_review_three_light_operator_ok_support_v1"
ORIGINAL_RULE = "multilight_local_review"
MAX_CNN_NG = 0.30
AT_LEAST_ONE_CNN_NG = 0.05
MIN_HUMAN_OK = 0.95
MIN_HUMAN_MARGIN = 0.02
MIN_VISUAL_OK = 0.95
MIN_VISUAL_MARGIN = 0.025
MIN_KNN_SIDE_A = 0.85
MIN_KNN_SIDE_B = 0.90
MAX_WEAK_KNN_NG = 0.85


def _score(value):
    if type(value) not in (int, float):
        return None
    v = float(value)
    return v if isfinite(v) and 0.0 <= v <= 1.0 else None


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _reject(diagnostic: dict | None, reason: str):
    if isinstance(diagnostic, dict):
        diagnostic.update({
            "schema": SCHEMA, "status": "NOT_APPLIED",
            "reason": reason, "diagnostic_only": True,
        })
    return None


def review_multilight_ok_evidence(
    fused: dict | None, human_by_light: dict | None, *,
    diagnostic: dict | None = None,
) -> dict | None:
    """Corrobora o OK *visual*, sem extrapolar similaridades a identidade.

    O relatório KNN visual e a memória humana usam diferenças visuais
    correlacionadas; a KNN C SIDE acrescenta geometria, não certeza
    física de orientação. A fronteira XP permanece de confirmação humana.
    """
    if not isinstance(fused, dict) or not isinstance(human_by_light, dict):
        return _reject(diagnostic, "INVALID_INPUT")
    d = fused.get("detail")
    if not isinstance(d, dict):
        return _reject(diagnostic, "MISSING_DETAIL")
    if not (
        fused.get("multilight_final") is True
        and fused.get("verdict") == "REVISÃO OBRIGATÓRIA"
        and fused.get("is_defect") is False
        and fused.get("production_review_required") is True
        and d.get("multilight_final") is True
        and d.get("fusion_rule") == ORIGINAL_RULE
        and d.get("recognition_route") == "NEW_CNN"
        and d.get("cnn_v2_trial_manual_only") is True
        and d.get("cnn_v2_consensus") == "REVIEW"
        and d.get("cnn_v2_consensus_reason") == "INVALID_LIGHT_EVIDENCE"
        and d.get("multilight_positive_modes") == []
        and d.get("multilight_strong_positive_modes") == []
        and set(d.get("multilight_review_modes") or ()) == set(LIGHTS)
        and d.get("multilight_physical_disagreement") is False
        and d.get("multilight_missing_verified_presence") is False
    ):
        return _reject(diagnostic, "BASE_FUSION_NOT_ELIGIBLE")
    category = canonical_memory_category(d.get("multilight_category", ""))
    if not (
        uses_faltando_v2(category)
        and canonical_memory_category(fused.get("multilight_category")) == category
        and _no_hard_ng(fused)
    ):
        return _reject(diagnostic, "CATEGORY_OR_PHYSICAL_NG_VETO")
    for node in (fused, d, d.get("decision_trace")):
        if isinstance(node, dict) and any(node.get(key) is True for key in (
            "memory_conflict", "hard_missing_evidence",
            "raw_hard_missing_evidence", "missing_hard_absence",
        )):
            return _reject(diagnostic, "PHYSICAL_OR_EXACT_MEMORY_NG")
    physical = d.get("physical_missing_veto")
    if isinstance(physical, dict) and (
        physical.get("status") in {"SUSPECT_ABSENCE", "NG_CORROBORATED", "UNAVAILABLE"}
        or physical.get("suspect_lights")
        or physical.get("unavailable_lights")
    ):
        return _reject(diagnostic, "PHYSICAL_PROBE_CONTRADICTION")

    locals_ = d.get("multilight_evidence")
    cnn = d.get("cnn_v2_light_diagnostics")
    routes = d.get("recognition_light_routes")
    if any(not isinstance(x, dict) or set(x) != set(LIGHTS)
           for x in (locals_, cnn, routes, human_by_light)):
        return _reject(diagnostic, "MISSING_THREE_LIGHT_EVIDENCE")
    checkpoint = None
    cnn_scores = {}
    for light in LIGHTS:
        local = locals_[light]
        item = cnn[light]
        if not isinstance(local, dict) or not isinstance(item, dict) or not (
            routes[light] == "NEW_CNN"
            and item.get("route") == "NEW_CNN"
            and item.get("category") == category
            and item.get("cnn_active") is True
            and item.get("checkpoint_verified") is True
            and item.get("cnn_status") == "INFERENCE_OK"
            and local.get("local_is_defect") is False
            and local.get("local_review_required") is True
            and local.get("local_verdict") in {
                "FALHA FALSA", "REVISÃO OBRIGATÓRIA"
            }
            and _score(local.get("physical_score")) == 0.0
        ):
            return _reject(diagnostic, "UNVERIFIED_CNN_OR_LOCAL_LIGHT_" + light)
        sha = item.get("checkpoint_sha256")
        if not _sha(sha) or (checkpoint is not None and checkpoint != sha):
            return _reject(diagnostic, "CHECKPOINT_CHANGED_OR_INVALID")
        checkpoint = sha
        score = _score(item.get("ng_score_uncalibrated"))
        if score is None or score > MAX_CNN_NG or (
            item.get("verdict") != (
                "FALHA FALSA" if score <= .10 else "REVISÃO OBRIGATÓRIA"
            )
        ):
            return _reject(diagnostic, "CNN_NOT_OK_LEANING_" + light)
        cnn_scores[light] = score
    if min(cnn_scores.values()) > AT_LEAST_ONE_CNN_NG:
        return _reject(diagnostic, "NO_CONCLUSIVE_CNN_OK_LIGHT")

    report = d.get("knn_visual_corroboration")
    if not isinstance(report, dict) or not (
        report.get("schema") == "visionx.knn_visual_corroboration.v1"
        and report.get("category") == category
        and report.get("status") == "CONFLICT"
        and report.get("strong_ng_lights") == []
        and report.get("trend_ng_lights") == []
        and isinstance(report.get("lights"), dict)
        and set(report["lights"]) == set(LIGHTS)
    ):
        return _reject(diagnostic, "UNVERIFIED_OR_NG_KNN_VISUAL")
    visual_points = {}
    for light in LIGHTS:
        row = report["lights"][light]
        if not isinstance(row, dict) or not (
            row.get("status") in {"CONFLICT", "NO_CONCLUSION"}
            and row.get("cross_category_ng") is False
            and isinstance(row.get("ok"), dict)
            and isinstance(row.get("ng"), dict)
        ):
            return _reject(diagnostic, "INVALID_VISUAL_STATUS_" + light)
        ok = _score(row["ok"].get("best_similarity"))
        ng = _score(row["ng"].get("best_similarity"))
        margin = row.get("ng_margin_over_ok")
        if (
            ok is None or ng is None or type(margin) not in (float, int)
            or not isfinite(float(margin))
            or abs((ng - ok) - margin) > 0.00001
            or ok < MIN_VISUAL_OK or ok - ng < MIN_VISUAL_MARGIN
        ):
            return _reject(diagnostic, "VISUAL_OK_ADVANTAGE_INSUFFICIENT_" + light)
        visual_points[light] = {"ok": ok, "ng": ng}
    if (
        sum(point["ok"] >= .97 for point in visual_points.values()) < 2
        or max(point["ok"] for point in visual_points.values()) < .98
        or not any(point["ok"] - point["ng"] >= .04
                   for point in visual_points.values())
    ):
        return _reject(diagnostic, "VISUAL_MULTI_LIGHT_SUPPORT_WEAK")

    human_points = {}
    for light in LIGHTS:
        rec = human_by_light[light]
        if not isinstance(rec, dict) or not (
            rec.get("schema") == "visionx.human_visual_recall.v1"
            and rec.get("source") == "PERSISTENT_HUMAN_LABELS"
            and rec.get("category") == category
            and rec.get("status") in {
                "NO_MATCH", "AMBIGUOUS", "HUMAN_OK_VISUAL_CANDIDATE"
            }
            and rec.get("reason") == "VISUAL_SIMILARITY_NOT_EXACT_IDENTITY"
            and rec.get("diagnostic_only") is True
            and rec.get("auto_ok_authorized") is False
        ):
            return _reject(diagnostic, "UNVERIFIED_HUMAN_RECALL_" + light)
        ok_source = rec.get("best_ok_source")
        ng_source = rec.get("best_ng_source")
        if not all(isinstance(x, dict) for x in (ok_source, ng_source)):
            return _reject(diagnostic, "MISSING_HUMAN_REFERENCE_" + light)
        if not all(
            x.get("source") == "OPERATOR_CONFIRMED"
            and x.get("label") == label
            and x.get("lighting_mode") == light
            and _sha(x.get("record_id"))
            for x, label in ((ok_source, "OK"), (ng_source, "NG"))
        ):
            return _reject(diagnostic, "INVALID_HUMAN_REFERENCE_" + light)
        ok, ng = _score(rec.get("best_ok_similarity")), _score(rec.get("best_ng_similarity"))
        if ok is None or ng is None or (
            ok < MIN_HUMAN_OK or ok - ng < MIN_HUMAN_MARGIN
        ):
            return _reject(diagnostic, "HUMAN_OK_ADVANTAGE_INSUFFICIENT_" + light)
        human_points[light] = {
            "ok": ok, "ng": ng, "human_ok_id": ok_source["record_id"],
        }
    if sum(row["ok"] - row["ng"] >= .025 for row in human_points.values()) < 2:
        return _reject(diagnostic, "HUMAN_MULTI_LIGHT_SUPPORT_WEAK")

    votes = d.get("knn_c_light_observer_votes")
    neighbors = d.get("knn_c_light_neighbor_evidence")
    if (
        not isinstance(votes, dict) or set(votes) != set(LIGHTS)
        or not isinstance(neighbors, dict) or set(neighbors) != set(LIGHTS)
    ):
        return _reject(diagnostic, "MISSING_KNN_C_THREE_LIGHT_EVIDENCE")
    for light in LIGHTS:
        vote = votes[light]
        if not isinstance(vote, dict) or not (
            vote.get("schema") == "visionx.knn_c_live_review_gated.v1"
            and vote.get("bank_status") == "READY"
            and vote.get("diagnostic_only") is True
            and vote.get("auto_decision_authorized") is False
        ):
            return _reject(diagnostic, "UNVERIFIED_KNN_C_" + light)
        if light == "SIDE":
            if not (
                vote.get("status") == "SUGERE_OK"
                and vote.get("reason") in {
                    "B_SELETIVA_E_VOTOS_A_B_CONCORDAM", "A_E_B_CONCORDAM"
                }
                and vote.get("vote_A") == vote.get("vote_B") == "OK"
            ):
                return _reject(diagnostic, "KNN_C_SIDE_DOES_NOT_SUPPORT_OK")
        elif not (
            vote.get("status") == "REVISAO"
            and vote.get("suggestion_A") == "REVISAO"
            and vote.get("suggestion_B") == "REVISAO"
            and max(
                _score(vote.get("similarity_A")) or 1,
                _score(vote.get("similarity_B")) or 1,
            ) <= MAX_WEAK_KNN_NG
        ):
            return _reject(diagnostic, "KNN_C_NG_NOT_WEAK_" + light)
    side_neighbor = neighbors["SIDE"]
    if not isinstance(side_neighbor, dict):
        return _reject(diagnostic, "KNN_C_SIDE_NEIGHBORS_MISSING")
    first = []
    for name, required_ok, threshold in (
        ("neighbors_A", 3, MIN_KNN_SIDE_A),
        ("neighbors_B", 4, MIN_KNN_SIDE_B),
    ):
        rows = side_neighbor.get(name)
        if not isinstance(rows, list) or len(rows) < 5:
            return _reject(diagnostic, "KNN_C_SIDE_NEIGHBORS_INCOMPLETE")
        rows = rows[:5]
        if any(
            not isinstance(r, dict) or r.get("label") not in {"OK", "NG"}
            or _score(r.get("similarity")) is None for r in rows
        ):
            return _reject(diagnostic, "KNN_C_SIDE_NEIGHBOR_INVALID")
        best = rows[0]
        if (
            best["label"] != "OK"
            or _score(best["similarity"]) < threshold
            or sum(r["label"] == "OK" for r in rows) < required_ok
            or any(
                r["label"] == "NG"
                and r["similarity"] >= best["similarity"] - .02
                for r in rows
            )
            or not isinstance(best.get("path"), str)
            or not best["path"].startswith("public/ok_archive/")
            or ".." in best["path"].split("/")
            or not _sha(best.get("source_pixel_sha256"))
        ):
            return _reject(diagnostic, "KNN_C_SIDE_OK_OR_NG_NEIGHBOR_UNSAFE")
        first.append(best)
    if (
        first[0]["path"] != first[1]["path"]
        or first[0]["source_pixel_sha256"] != first[1]["source_pixel_sha256"]
    ):
        return _reject(diagnostic, "KNN_C_SIDE_A_B_NOT_SAME_OK_SOURCE")

    return {
        "schema": SCHEMA,
        "category": category,
        "label": "OK",
        "source": "OPERATOR_VISUAL_RECALL_PLUS_KNN_C_GEOMETRY",
        "cnn_model_sha256": checkpoint,
        "cnn_ng_scores_uncalibrated": cnn_scores,
        "human_memory": human_points,
        "knn_visual": visual_points,
        "side_ok_archive_path": first[0]["path"],
        "side_ok_archive_pixel_sha256": first[0]["source_pixel_sha256"],
        "side_knn_c": {
            "A": first[0]["similarity"], "B": first[1]["similarity"],
        },
        "rgb_exact_pair": False,
        "physical_orientation_verified": False,
        "automatic_xp_authorized": False,
        "reason": "THREE_LIGHT_CNN_OK_LEANING_AND_OPERATOR_OK_COUNTEREVIDENCE",
    }


def apply_review_multilight_ok_consensus(
    fused: dict | None, analyses: dict | None,
) -> dict | None:
    """Corrige a classificação visual após corroboração KNN; sem XP."""
    if not isinstance(fused, dict) or not isinstance(analyses, dict):
        return fused
    details = {
        light: (
            analyses.get(light, {}).get("detail", {})
            if isinstance(analyses.get(light), dict) else {}
        )
        for light in LIGHTS
    }
    human = {
        light: deepcopy(value.get("confirmed_visual_recall"))
        if isinstance(value, dict) else None
        for light, value in details.items()
    }
    diagnostic = {}
    proof = review_multilight_ok_evidence(fused, human, diagnostic=diagnostic)
    if proof is None:
        if diagnostic.get("reason") not in {
            "INVALID_INPUT", "MISSING_DETAIL", "BASE_FUSION_NOT_ELIGIBLE"
        }:
            # Para auditoria: jamais alterar classificação quando incompleta.
            result = deepcopy(fused)
            result["detail"]["review_multilight_ok_attempt"] = diagnostic
            return result
        return fused
    result = deepcopy(fused)
    d = result["detail"]
    d["review_multilight_ok_original"] = {
        "verdict": fused["verdict"], "fusion_rule": d["fusion_rule"],
    }
    d["review_multilight_ok_human_snapshot"] = human
    d["review_multilight_ok_proof"] = proof
    result["verdict"] = "FALHA FALSA"
    result["is_defect"] = False
    result["confidence"] = .5
    result["production_review_required"] = True
    result["eligible_for_final_decision"] = False
    result["reason"] = (
        "FALHA FALSA • CONSENSO VISUAL MULTILIGHT: CNN experimental "
        "favoreceu OK nas três iluminações; memória humana OK superou "
        "NG nas três luzes, índice visual concordou e KNN C SIDE "
        "recuperou um mesmo PNG OK em A/B. Não é prova física de "
        "orientação correta nem par RGB exato. Confirmação humana "
        "necessária antes de 0/1 no XP."
    )
    d["fusion_rule"] = RULE
    d["operator_review_required"] = True
    d["eligible_for_final_decision"] = False
    d["cnn_v2_supervised_auto_eligible"] = False
    d["final_score"] = .5
    trace = d.get("decision_trace")
    if isinstance(trace, dict):
        trace.update(
            verdict="FALHA FALSA",
            fusion_rule=RULE,
            physical_defect=False,
            operator_review_required=True,
            final_score=.5,
        )
    return result


def verified_review_multilight_ok(analysis: dict | None) -> bool:
    """Reavalia dados brutos preservados, nunca somente texto/flag OK."""
    if not isinstance(analysis, dict) or not (
        analysis.get("verdict") == "FALHA FALSA"
        and analysis.get("is_defect") is False
        and analysis.get("multilight_final") is True
    ):
        return False
    d = analysis.get("detail")
    if not isinstance(d, dict) or d.get("fusion_rule") != RULE:
        return False
    original = d.get("review_multilight_ok_original")
    human = d.get("review_multilight_ok_human_snapshot")
    proof = d.get("review_multilight_ok_proof")
    if not isinstance(original, dict) or original != {
        "verdict": "REVISÃO OBRIGATÓRIA", "fusion_rule": ORIGINAL_RULE,
    } or not isinstance(human, dict) or not isinstance(proof, dict):
        return False
    restored = deepcopy(analysis)
    restored["verdict"] = original["verdict"]
    restored["production_review_required"] = True
    restored["detail"]["fusion_rule"] = original["fusion_rule"]
    rebuilt = review_multilight_ok_evidence(restored, human)
    return rebuilt is not None and rebuilt == proof


__all__ = [
    "RULE", "SCHEMA", "review_multilight_ok_evidence",
    "apply_review_multilight_ok_consensus", "verified_review_multilight_ok",
]
