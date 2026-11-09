"""Validação de um OK multilight supervisionado, independente da UI.

Não altera inferência ou aprendizado. O gate XP revalida este contrato
antes de emitir 0, inclusive depois do segundo de apresentação.
"""
from __future__ import annotations

from math import isfinite

from src.core.neural.aoi_v3_epoch5_contract import (
    EPOCH as V3_EPOCH,
    SNAPSHOT_SHA256 as V3_SHA,
)
from src.core.neural.aoi_quality_cnn_v3_trial_consensus import active_v3_veto

LIGHTS = ("SIDE", "TOP", "MID")
RULE = "production_supervised_v3_multilight_all_clear_ok_v1"
KNN_SCHEMA = "visionx.knn_c_live_review_gated.v1"
MIN_KNN_NEAREST = 0.70
MAX_V3_NG_SCORE = 0.01


def _unit(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and 0.0 <= value <= 1.0
    )


def verified_supervised_multilight_all_clear(analysis: dict | None) -> bool:
    """Confere três votos V3, ausência física e A/B KNN por iluminação.

    Exige a mesma fonte de pixels OK no topo dos dois descritores
    KNN de cada iluminação, sem alegar memória humana ou par exato.
    Os limiares são de elegibilidade de diagnóstico e não precisão
    estatística comprovada. Falhas de integridade não liberam XP.
    """
    if not isinstance(analysis, dict):
        return False
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        return False
    if not (
        analysis.get("lighting_mode") == "MULTILIGHT"
        and analysis.get("multilight_final") is True
        and analysis.get("eligible_for_final_decision") is True
        and analysis.get("production_review_required") is False
        and analysis.get("verdict") == "FALHA FALSA"
        and analysis.get("is_defect") is False
        and detail.get("fusion_rule") == "multilight_all_clear"
        and detail.get("multilight_final") is True
        and detail.get("eligible_for_final_decision") is True
        and detail.get("cnn_v3_primary") is True
        and detail.get("cnn_v3_all_ok_verified") is True
        and detail.get("operator_review_required") is False
        and detail.get("capture_error") is not True
        and detail.get("memory_conflict") is not True
        and detail.get("raw_memory_conflict") is not True
        and detail.get("hard_missing_evidence") is not True
        and detail.get("raw_hard_missing_evidence") is not True
        and detail.get("missing_hard_absence") is not True
        and detail.get("multilight_physical_disagreement") is False
        and detail.get("multilight_missing_verified_presence") is False
        and detail.get("multilight_positive_modes") == []
        and detail.get("multilight_strong_positive_modes") == []
        and detail.get("multilight_review_modes") == []
        and trace.get("operator_review_required") is False
        and trace.get("hard_missing_evidence") is False
        and trace.get("raw_hard_missing_evidence") is False
        and _unit(detail.get("physical_score"))
        and detail["physical_score"] == 0.0
        and _unit(analysis.get("confidence"))
        and active_v3_veto(analysis) == ("READY", False)
    ):
        return False

    v3 = detail.get("aoi_v3_multilight")
    diagnostics = detail.get("cnn_v3_light_diagnostics")
    observations = detail.get("multilight_evidence")
    knn_votes = detail.get("knn_c_light_observer_votes")
    knn_neighbors = detail.get("knn_c_light_neighbor_evidence")
    routes = detail.get("recognition_light_routes")
    if not all(
        isinstance(obj, dict) and set(obj) == set(LIGHTS)
        for obj in (
            v3.get("lights") if isinstance(v3, dict) else None,
            diagnostics, observations, knn_votes, knn_neighbors, routes,
        )
    ):
        return False
    if not (
        v3.get("epoch") == V3_EPOCH
        and v3.get("checkpoint_sha256") == V3_SHA
        and v3.get("ng_veto") is False
        and v3.get("automatic_ok_from_v3_authorized") is False
    ):
        return False

    max_score = 0.0
    for light in LIGHTS:
        vote = v3["lights"][light]
        info = diagnostics[light]
        obs = observations[light]
        knn = knn_votes[light]
        neighbors = knn_neighbors[light]
        if not all(isinstance(x, dict) for x in (vote, info, obs, knn, neighbors)):
            return False
        score = vote.get("ng_score_uncalibrated")
        if not (
            _unit(score)
            and score < MAX_V3_NG_SCORE
            and vote.get("predicted") == "OK"
            and routes[light] == "NEW_CNN"
            and info.get("route") == "NEW_CNN"
            and info.get("engine") == "aoi_quality_cnn_v3"
            and info.get("cnn_active") is True
            and info.get("cnn_status") == "INFERENCE_OK"
            and info.get("checkpoint_verified") is True
            and info.get("checkpoint_sha256") == V3_SHA
            and info.get("verdict") == "FALHA FALSA"
            and _unit(info.get("ng_score_uncalibrated"))
            and abs(info["ng_score_uncalibrated"] - score) < 1e-8
            and obs.get("local_verdict") == "FALHA FALSA"
            and obs.get("local_is_defect") is False
            and obs.get("local_review_required") is False
            and obs.get("fusion_rule") == "cnn_v3_primary_direct"
            and obs.get("dominant_engine") == "aoi_quality_cnn_v3"
            and _unit(obs.get("physical_score"))
            and obs["physical_score"] == 0.0
            and _unit(obs.get("final_score"))
            and abs(obs["final_score"] - score) < 1e-8
            and knn.get("schema") == KNN_SCHEMA
            and knn.get("bank_status") == "READY"
            and knn.get("diagnostic_only") is True
            and knn.get("auto_decision_authorized") is False
            and knn.get("vote_A") == "OK"
            and knn.get("vote_B") == "OK"
        ):
            return False
        same_sources = []
        for descriptor in ("neighbors_A", "neighbors_B"):
            entries = neighbors.get(descriptor)
            if not isinstance(entries, list) or not entries:
                return False
            first = entries[0]
            if not isinstance(first, dict):
                return False
            sha = first.get("source_pixel_sha256")
            if not (
                first.get("label") == "OK"
                and isinstance(first.get("path"), str)
                and first["path"].startswith("public/ok_archive/")
                and isinstance(sha, str) and len(sha) == 64
                and all(c in "0123456789abcdef" for c in sha.lower())
                and _unit(first.get("similarity"))
                and first["similarity"] >= MIN_KNN_NEAREST
            ):
                return False
            same_sources.append(sha.lower())
        if same_sources[0] != same_sources[1]:
            return False
        max_score = max(max_score, score)

    return bool(
        _unit(detail.get("final_score"))
        and abs(detail["final_score"] - max_score) < 1e-8
        and abs(analysis["confidence"] - (1.0 - max_score)) < 1e-8
    )


__all__ = ["RULE", "verified_supervised_multilight_all_clear"]
