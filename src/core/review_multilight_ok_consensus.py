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

# Rota alternativa: uma TOP CNN alta, ainda inconclusiva, pode ser
# contrariada por duas capturas TOP/MID que recuperam o MESMO evento OK
# arquivado, cada uma confirmada pelos descritores A e B. Não libera XP.
TOP_MID_MODE = "TOP_MID_ARCHIVE_OK_OVER_CNN_REVIEW"
TOP_MID_CNN_NG_CEILING = 0.90
TOP_MID_OTHER_CNN_MAX = 0.20
TOP_MID_ARCHIVE_NG_GAP = 0.075


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


def _ok_archive_pair_for_light(
    light: str, vote: dict, sets: dict, category: str,
) -> dict | None:
    """Dois descritores retornam o mesmo PNG OK, com NG distante."""
    if not (
        isinstance(vote, dict)
        and vote.get("status") == "REVISAO"
        and vote.get("reason") == "SEM_SUGESTAO_CONFIAVEL"
        and vote.get("suggestion_A") == "REVISAO"
        and vote.get("suggestion_B") == "REVISAO"
        and isinstance(sets, dict)
    ):
        return None
    thresholds = {
        "TOP": (.83, .88),
        "MID": (.90, .88),
    }
    if light not in thresholds:
        return None
    selected = []
    for key, minimum, observed in (
        ("neighbors_A", thresholds[light][0], vote.get("similarity_A")),
        ("neighbors_B", thresholds[light][1], vote.get("similarity_B")),
    ):
        rows = sets.get(key)
        if not isinstance(rows, list) or len(rows) < 5:
            return None
        rows = rows[:5]
        if any(
            not isinstance(r, dict)
            or r.get("label") not in {"OK", "NG"}
            or _score(r.get("similarity")) is None
            for r in rows
        ):
            return None
        best = rows[0]
        best_score = _score(best["similarity"])
        ng_scores = [
            _score(r["similarity"])
            for r in rows if r["label"] == "NG"
        ]
        if not (
            best.get("label") == "OK"
            and best_score >= minimum
            and _score(observed) is not None
            and abs(_score(observed) - best_score) <= 0.00001
            and (not ng_scores or best_score - max(ng_scores)
                 >= TOP_MID_ARCHIVE_NG_GAP)
        ):
            return None
        path = best.get("path")
        pixel_sha = best.get("source_pixel_sha256")
        suffix = f"_{category}_{light}.png"
        if not (
            isinstance(path, str)
            and path.startswith("public/ok_archive/")
            and ".." not in path.split("/")
            and path.endswith(suffix)
            and bool(path[len("public/ok_archive/"):-len(suffix)])
            and _sha(pixel_sha)
        ):
            return None
        selected.append((path, pixel_sha, best_score, max(ng_scores, default=None)))
    if selected[0][:2] != selected[1][:2]:
        return None
    return {
        "path": selected[0][0],
        "pixel_sha256": selected[0][1],
        "group": selected[0][0][:-len(f"_{category}_{light}.png")],
        "similarity_A": selected[0][2],
        "similarity_B": selected[1][2],
        "best_ng_A": selected[0][3],
        "best_ng_B": selected[1][3],
    }


def _weak_side_c_disagreement(vote: dict, sets: dict) -> dict | None:
    """NG fraco em A não elimina B OK dominante, sem tornar C conclusiva."""
    if not (
        isinstance(vote, dict)
        and vote.get("status") == "REVISAO"
        and vote.get("reason") == "DISCORDANCIA_KNN_A_B"
        and vote.get("suggestion_A") == "REVISAO"
        and vote.get("suggestion_B") == "REVISAO"
        and vote.get("vote_A") == "NG"
        and vote.get("vote_B") == "OK"
        and isinstance(sets, dict)
    ):
        return None
    a, b = sets.get("neighbors_A"), sets.get("neighbors_B")
    if not all(isinstance(rows, list) and len(rows) >= 5 for rows in (a, b)):
        return None
    a, b = a[:5], b[:5]
    if any(
        not isinstance(r, dict)
        or r.get("label") not in {"OK", "NG"}
        or _score(r.get("similarity")) is None
        for r in a + b
    ):
        return None
    score_a, score_b = _score(vote.get("similarity_A")), _score(
        vote.get("similarity_B")
    )
    if not (
        score_a is not None and score_b is not None
        and a[0].get("label") == "NG"
        and b[0].get("label") == "OK"
        and abs(a[0]["similarity"] - score_a) <= .00001
        and abs(b[0]["similarity"] - score_b) <= .00001
        and score_a <= .82
        and score_b >= .84
        and score_b - score_a >= .03
        and sum(row["label"] == "OK" for row in b) >= 4
        and any(row["label"] == "OK" for row in a)
        and all(
            row["similarity"] < score_b - .04
            for row in b if row["label"] == "NG"
        )
    ):
        return None
    return {"best_ng_A": score_a, "best_ok_B": score_b,
            "ok_neighbors_B": sum(row["label"] == "OK" for row in b)}


def _top_mid_recurrent_archive_ok(
    votes: dict, neighbors: dict, category: str,
) -> dict | None:
    """Confere duas âncoras OK independentes da SIDE, em mesmo evento."""
    side = _weak_side_c_disagreement(
        votes.get("SIDE"), neighbors.get("SIDE"),
    )
    if side is None:
        return None
    matched = {}
    for light in ("TOP", "MID"):
        matched[light] = _ok_archive_pair_for_light(
            light, votes[light], neighbors[light], category,
        )
        if matched[light] is None:
            return None
    if matched["TOP"]["group"] != matched["MID"]["group"]:
        return None
    return {
        "mode": TOP_MID_MODE,
        "side_dissent": side,
        "archive_anchors": matched,
    }


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
        if score is None or score >= TOP_MID_CNN_NG_CEILING or (
            item.get("verdict") != (
                "FALHA FALSA" if score <= .10 else "REVISÃO OBRIGATÓRIA"
            )
        ):
            return _reject(diagnostic, "CNN_NOT_OK_LEANING_" + light)
        cnn_scores[light] = score
    classic_mode = (
        max(cnn_scores.values()) <= MAX_CNN_NG
        and min(cnn_scores.values()) <= AT_LEAST_ONE_CNN_NG
    )
    top_mid_mode = (
        MAX_CNN_NG < cnn_scores["TOP"] < TOP_MID_CNN_NG_CEILING
        and cnn_scores["SIDE"] <= TOP_MID_OTHER_CNN_MAX
        and cnn_scores["MID"] <= TOP_MID_OTHER_CNN_MAX
    )
    if not (classic_mode or top_mid_mode):
        return _reject(diagnostic, "CNN_THREE_LIGHT_CONSENSUS_INSUFFICIENT")

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
            or ok < MIN_VISUAL_OK
            or ok - ng < (
                .017 if top_mid_mode and light == "SIDE"
                else MIN_VISUAL_MARGIN
            )
        ):
            return _reject(diagnostic, "VISUAL_OK_ADVANTAGE_INSUFFICIENT_" + light)
        visual_points[light] = {"ok": ok, "ng": ng}
    if (
        sum(point["ok"] >= .97 for point in visual_points.values()) < 2
        or max(point["ok"] for point in visual_points.values()) < .98
        or not any(
            point["ok"] - point["ng"] >= (.035 if top_mid_mode else .04)
            for point in visual_points.values()
        )
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
            ok < (
                {"SIDE": .96, "TOP": .98, "MID": .965}[light]
                if top_mid_mode else MIN_HUMAN_OK
            )
            or ok - ng < (
                .01 if top_mid_mode and light == "SIDE"
                else .025 if top_mid_mode else MIN_HUMAN_MARGIN
            )
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
    if top_mid_mode:
        archive = _top_mid_recurrent_archive_ok(votes, neighbors, category)
        if archive is None:
            return _reject(diagnostic, "TOP_MID_ARCHIVE_OK_PAIR_UNVERIFIED")
        return {
            "schema": SCHEMA,
            "category": category,
            "label": "OK",
            "source": "OPERATOR_VISUAL_RECALL_PLUS_TWO_ARCHIVE_ANCHORS",
            "evidence_mode": TOP_MID_MODE,
            "cnn_model_sha256": checkpoint,
            "cnn_ng_scores_uncalibrated": cnn_scores,
            "human_memory": human_points,
            "knn_visual": visual_points,
            "knn_c_archive": archive,
            "rgb_exact_pair": False,
            "physical_orientation_verified": False,
            "automatic_xp_authorized": False,
            "reason": "CNN_TOP_REVIEW_CONTRADICTED_BY_TWO_ARCHIVE_OK_LIGHTS",
        }
    for light in LIGHTS:
        vote = votes[light]
        if light == "SIDE":
            if not (
                vote.get("status") == "SUGERE_OK"
                and vote.get("reason") in {
                    "B_SELETIVA_E_VOTOS_A_B_CONCORDAM", "A_E_B_CONCORDAM"
                }
                and vote.get("vote_A") == vote.get("vote_B") == "OK"
            ):
                return _reject(diagnostic, "KNN_C_SIDE_DOES_NOT_SUPPORT_OK")
        else:
            similarity_a = _score(vote.get("similarity_A"))
            similarity_b = _score(vote.get("similarity_B"))
            if not (
                vote.get("status") == "REVISAO"
                and vote.get("reason") == "SEM_SUGESTAO_CONFIAVEL"
                and vote.get("suggestion_A") == "REVISAO"
                and vote.get("suggestion_B") == "REVISAO"
                and similarity_a is not None
                and similarity_b is not None
                and max(similarity_a, similarity_b) <= MAX_WEAK_KNN_NG
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
        (
            "FALHA FALSA • TOP/MID recuperaram o mesmo evento OK do "
            "arquivo pelos descritores KNN C A/B; memória humana e índice "
            "visual favorecem OK nas três iluminações. A CNN TOP trial "
            "ficou próxima do limite NG, sem corroboracão independente; "
            "SIDE teve NG KNN fraco e B favoreceu OK. "
        ) if proof.get("evidence_mode") == TOP_MID_MODE else (
            "FALHA FALSA • CONSENSO VISUAL MULTILIGHT: CNN experimental "
            "favoreceu OK nas três iluminações; memória humana OK superou "
            "NG nas três luzes, índice visual concordou e KNN C SIDE "
            "recuperou um mesmo PNG OK em A/B. "
        )
    ) + (
        "Sem prova física de orientação nem par RGB exato. "
        "Confirmação humana antes de qualquer 0/1 XP."
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
