"""Single-SIDE human-reviewed recall for the verified 14:16 false alarm.

Only a visually similar memory source is known; the current pair is NOT
pixel-identical. Never auto-approve a PCB or issue XP 0 from this route.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite

from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
from src.core.operator_confirmed_archive_labels import (
    CANONICAL_OK_SOURCE, CANONICAL_OK_PIXEL_SHA256, RETIRED_SOURCE,
    verified_canonical_ok_source,
)
from src.core.physical_missing_veto import physical_absence_probe

SIDE_RECURRENT_REVIEW_RULE = "side_verified_archive_visual_review_v1"
MIN_SIDE_A = 0.85
MIN_SIDE_B = 0.82
MIN_SIDE_NG_GAP = 0.04
MAX_SIDE_NG = 0.90
MAX_SIDE_CNN_NG = 0.04


def _unit_float(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if isfinite(number) and 0 <= number <= 1 else None


def _side_visual_candidate(analysis: dict | None, category: str) -> dict | None:
    """Verify provenance and discriminating neighbors; never infer an exact pair."""
    if category != "FALTANDO" or not isinstance(analysis, dict):
        return None
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return None
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        return None
    if not (
        analysis.get("lighting_mode") == "SIDE"
        and analysis.get("verdict") == "FALHA FALSA"
        and analysis.get("is_defect") is False
        and analysis.get("production_review_required") is True
        and detail.get("operator_review_required") is True
        and trace.get("operator_review_required") is True
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("recognition_match") == "NOT_FOUND"
        and detail.get("recognition_memory_verified") is False
        and detail.get("cnn_v2_active") is True
        and detail.get("cnn_v2_experimental") is True
        and detail.get("cnn_v2_status") == "INFERENCE_OK"
        and detail.get("cnn_v2_checkpoint_verified") is True
        and detail.get("cnn_v2_trial_manual_only") is True
        and detail.get("cnn_v2_checkpoint_sha256") == TRIAL_97P70_SHA256
        and detail.get("cnn_v2_aoi_category") == "FALTANDO"
        and detail.get("cnn_v2_lighting_mode") == "SIDE"
        and detail.get("hard_missing_evidence") is not True
        and analysis.get("hard_missing_evidence") is not True
        and detail.get("memory_conflict") is not True
        and trace.get("memory_conflict") is not True
        and detail.get("physical_score") == 0.0
    ):
        return None
    score = _unit_float(detail.get("cnn_v2_ng_score_uncalibrated"))
    if score is None or score > MAX_SIDE_CNN_NG:
        return None
    knn = detail.get("knn_c")
    if not isinstance(knn, dict) or not (
        knn.get("schema") == "visionx.knn_c_live_review_gated.v1"
        and knn.get("bank_status") == "READY"
        and knn.get("diagnostic_only") is True
        and knn.get("auto_decision_authorized") is False
        and knn.get("status") == "REVISAO"
        and knn.get("vote_B") == "OK"
        and knn.get("suggestion_A") == "REVISAO"
        and knn.get("suggestion_B") == "REVISAO"
    ):
        return None
    scores = {}
    for key, minimum in (("neighbors_A", MIN_SIDE_A),
                         ("neighbors_B", MIN_SIDE_B)):
        rows = knn.get(key)
        if not isinstance(rows, list) or len(rows) < 5:
            return None
        first = rows[0]
        if not isinstance(first, dict) or not (
            first.get("label") == "OK"
            and first.get("path") == CANONICAL_OK_SOURCE
            and first.get("source_pixel_sha256") == CANONICAL_OK_PIXEL_SHA256
        ):
            return None
        best = _unit_float(first.get("similarity"))
        if best is None or best < minimum:
            return None
        ngs = []
        for row in rows:
            if not isinstance(row, dict) or row.get("label") not in {"OK", "NG"}:
                return None
            if row.get("path") == RETIRED_SOURCE:
                return None
            if row.get("label") == "NG":
                similarity = _unit_float(row.get("similarity"))
                if similarity is None:
                    return None
                ngs.append(similarity)
        if ngs and (max(ngs) >= MAX_SIDE_NG
                    or best - max(ngs) < MIN_SIDE_NG_GAP):
            return None
        scores[key] = round(best, 8)
    # The reference SHA validates the HISTORICAL pixels, not the live ones.
    if not verified_canonical_ok_source():
        return None
    return {
        "source": CANONICAL_OK_SOURCE,
        "source_pixel_sha256": CANONICAL_OK_PIXEL_SHA256,
        "side_matches": scores,
        "cnn_ng_score_uncalibrated": round(score, 8),
        "live_pair_exact": False,
    }


def _valid_probe_result(probe: dict) -> bool:
    """Accept the physical check only when both scales were measured."""
    if not isinstance(probe, dict) or not (
        probe.get("status") in {"NO_STRONG_EVIDENCE", "GEOMETRY_REVIEW"}
        and probe.get("reason") == "TWO_SCALE_ROI_CHECK"
    ):
        return False
    scales = probe.get("scales")
    if not isinstance(scales, list) or len(scales) != 2:
        return False
    if any(not isinstance(row, dict)
           or row.get("suspected_absence") is not False
           for row in scales):
        return False
    geometry = probe.get("geometry")
    if not isinstance(geometry, dict):
        return False
    if geometry.get("method") not in {"exact", "center_trim_no_rescale"}:
        return False
    if probe["status"] == "GEOMETRY_REVIEW":
        return geometry.get("requires_review") is True
    return geometry.get("requires_review") is False


def side_recurrent_review_valid(analysis: dict | None) -> bool:
    """Revalidate display-only/manual marker; never return XP authorization."""
    if not isinstance(analysis, dict) or not (
        analysis.get("lighting_mode") == "SIDE"
        and analysis.get("multilight_final") is True
        and analysis.get("verdict") == "FALHA FALSA"
        and analysis.get("is_defect") is False
        and analysis.get("production_review_required") is True
        and analysis.get("eligible_for_final_decision") is False
    ):
        return False
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    marker = detail.get("side_early_exit")
    if not isinstance(marker, dict) or not (
        marker.get("rule") == SIDE_RECURRENT_REVIEW_RULE
        and marker.get("analyzed_lights") == ["SIDE"]
        and marker.get("skipped_lights") == ["TOP", "MID"]
        and marker.get("production_auto_allowed") is False
        and marker.get("operator_review_required") is True
        and detail.get("operator_review_required") is True
        and detail.get("multilight_final") is True
        and detail.get("multilight_category") == "FALTANDO"
        and detail.get("eligible_for_final_decision") is False
    ):
        return False
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict) or trace.get("operator_review_required") is not True:
        return False
    evidence = detail.get("side_recurrent_visual_review")
    # Do not trust arbitrary serialized evidence or a changed historic file.
    expected = _side_visual_candidate(analysis, "FALTANDO")
    return bool(
        isinstance(evidence, dict)
        and expected is not None
        and all(evidence.get(k) == v for k, v in expected.items())
        and _valid_probe_result(evidence.get("physical_probe"))
        and marker.get("source") == CANONICAL_OK_SOURCE
    )


def build_side_recurrent_ok_review(
    analysis: dict | None, category: str,
    sample_image, test_image, *, probe=None,
) -> dict | None:
    """Skip TOP/MID only to ask an operator about a familiar SIDE capture.

    This does NOT repeat the historical OK autonomously: approximate KNN
    evidence cannot rule out a different NG part. The operator must decide.
    """
    evidence = _side_visual_candidate(analysis, category)
    if evidence is None:
        return None
    inspect = probe if probe is not None else physical_absence_probe
    try:
        audit = inspect(sample_image, test_image)
    except Exception:
        return None
    if not _valid_probe_result(audit):
        return None
    final = deepcopy(analysis)
    final["multilight_final"] = True
    final["multilight_category"] = "FALTANDO"
    final["eligible_for_final_decision"] = False
    final["production_review_required"] = True
    final["reason"] = (
        "FALHA FALSA • SIDE reconhece referência humana 14:16 no acervo OK "
        "(similaridade visual, não par idêntico). TOP/MID dispensadas "
        "somente para revisão antecipada. Operador deve confirmar 0=OK "
        "ou 1=NG; sem transmissão automática ao XP."
    )
    d = final["detail"]
    d["multilight_final"] = True
    d["multilight_category"] = "FALTANDO"
    d["eligible_for_final_decision"] = False
    d["operator_review_required"] = True
    d["side_recurrent_visual_review"] = {
        **evidence, "physical_probe": audit, "manual_only": True,
    }
    d["side_early_exit"] = {
        "rule": SIDE_RECURRENT_REVIEW_RULE,
        "source": CANONICAL_OK_SOURCE,
        "analyzed_lights": ["SIDE"],
        "skipped_lights": ["TOP", "MID"],
        "production_auto_allowed": False,
        "operator_review_required": True,
    }
    d["decision_trace"]["operator_review_required"] = True
    return final


__all__ = [
    "SIDE_RECURRENT_REVIEW_RULE", "build_side_recurrent_ok_review",
    "side_recurrent_review_valid",
]
