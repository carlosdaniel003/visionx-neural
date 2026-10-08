"""DESLOCADO OK geometry v1.1: feature-based *diagnosis*, never inspection.

Compare automatic ORB/AKAZE correspondences to the existing phase-only
baseline. No KNN memory, masks drawn by operators, synthetic NG or CNN.
Global image registration is NOT proof of component or PCB-pad alignment.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import cv2
import numpy as np

from src.services.deslocado_ok_geometry import (
    _edges, _zones, _scale_for_pair, diagnose_ok_geometry,
)
from src.scripts.train_deslocado_cnn import load_ok_events, _read, _safe_png

SCHEMA_V11 = "visionx.deslocado_ok_geometry_diagnostic.v1_1"
METHODS = ("ORB", "AKAZE")
MAX_RATIO_CHANGE = .05


def _processed_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _candidate(
    gray_reference: np.ndarray,
    gray_test: np.ndarray,
    anchor: np.ndarray,
    name: str,
) -> dict:
    """Rigid similarity transform TEST -> REF with explicit spatial gates."""
    height, width = gray_reference.shape
    mask = (anchor.astype(np.uint8) * 255)
    # Keep detector away from the image boundary. Interior markings are
    # explicitly excluded, but remaining pixels may NOT be PCB pads.
    mask[:6, :] = 0
    mask[-6:, :] = 0
    mask[:, :6] = 0
    mask[:, -6:] = 0
    if name == "ORB":
        detector = cv2.ORB_create(
            nfeatures=1600, scaleFactor=1.2, fastThreshold=8,
            edgeThreshold=9, patchSize=21,
        )
    else:
        detector = cv2.AKAZE_create(threshold=.0002)
    keys_ref, desc_ref = detector.detectAndCompute(gray_reference, mask)
    keys_test, desc_test = detector.detectAndCompute(gray_test, mask)
    info = {
        "method": name, "keypoints_reference": len(keys_ref),
        "keypoints_test": len(keys_test), "cross_checked_matches": 0,
        "ransac_inliers": 0, "inlier_fraction": None,
        "inlier_quadrants": 0, "inlier_span_fraction_xy": None,
        "median_reprojection_error_px": None, "p90_reprojection_error_px": None,
        "rotation_deg": None, "scale_factor": None,
        "transform_test_to_reference": None,
        "translation_xy_px": None, "status": "EVIDENCIA_INSUFICIENTE",
        "reasons": [],
    }
    def refuse(reason: str):
        info["reasons"].append(reason)
        return info

    if (desc_ref is None or desc_test is None
            or len(keys_ref) < 16 or len(keys_test) < 16):
        return refuse("INSUFFICIENT_DISTRIBUTED_KEYPOINTS")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
        desc_test, desc_ref
    )
    # 1-to-1 descriptor agreement; this is NOT an episodic KNN lookup.
    matches = sorted(matches, key=lambda m: (m.distance, m.queryIdx, m.trainIdx))
    matches = [m for m in matches if m.distance <= 90][:240]
    info["cross_checked_matches"] = len(matches)
    if len(matches) < 14:
        return refuse("INSUFFICIENT_CROSS_CHECKED_MATCHES")
    src = np.float32([keys_test[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([keys_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, flags = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=2.5, maxIters=2500,
        confidence=.995, refineIters=15,
    )
    if matrix is None or flags is None or not np.isfinite(matrix).all():
        return refuse("RANSAC_TRANSFORM_UNAVAILABLE")
    inliers = flags.ravel().astype(bool)
    count = int(np.count_nonzero(inliers))
    fraction = count / len(matches)
    info.update(ransac_inliers=count, inlier_fraction=round(fraction, 5))
    if count < 12 or fraction < .57:
        return refuse("RANSAC_INLIERS_INSUFFICIENT")
    coords_ref = dst.reshape(-1, 2)[inliers]
    coords_test = src.reshape(-1, 2)[inliers]
    quarters = set(
        (int(y >= height/2), int(x >= width/2)) for x, y in coords_ref
    )
    span_x = float(np.ptp(coords_ref[:, 0]) / width)
    span_y = float(np.ptp(coords_ref[:, 1]) / height)
    info.update(inlier_quadrants=len(quarters),
                inlier_span_fraction_xy=[round(span_x, 4), round(span_y, 4)])
    if len(quarters) < 3 or span_x < .30 or span_y < .22:
        return refuse("SPATIALLY_CONCENTRATED_FEATURES")
    predicted = cv2.transform(coords_test.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(predicted - coords_ref, axis=1)
    median = float(np.median(errors))
    p90 = float(np.percentile(errors, 90))
    info.update(
        median_reprojection_error_px=round(median, 4),
        p90_reprojection_error_px=round(p90, 4),
    )
    if median > 1.5 or p90 > 2.5:
        return refuse("INCONSISTENT_LOCAL_REPROJECTION")
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    scale = math.hypot(a, b)
    rotation = math.degrees(math.atan2(b, a))
    dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
    info.update(
        rotation_deg=round(rotation, 4), scale_factor=round(scale, 5),
        translation_xy_px=[round(dx, 4), round(dy, 4)],
    )
    if (not .97 <= scale <= 1.03 or abs(rotation) > 4.0
            or abs(dx) > .08 * width or abs(dy) > .08 * height):
        return refuse("IMPLAUSIBLE_GLOBAL_TRANSFORM")
    # Cross-check the agreed transform in the full inlier support, not the
    # internal inscription; this is an independent pixel consistency gate.
    ref_edge = _edges(cv2.cvtColor(gray_reference, cv2.COLOR_GRAY2BGR))
    test_edge = _edges(cv2.cvtColor(gray_test, cv2.COLOR_GRAY2BGR))
    aligned = cv2.warpAffine(
        test_edge, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    # Exclude 6px borders after geometric warp to avoid border hallucinations.
    valid = anchor.copy()
    valid[:6, :] = False
    valid[-6:, :] = False
    valid[:, :6] = False
    valid[:, -6:] = False
    if int(np.count_nonzero(valid)) < 500:
        return refuse("INSUFFICIENT_CONTEXT_AREA")
    before = float(np.mean(np.abs(ref_edge[valid] - test_edge[valid])))
    after = float(np.mean(np.abs(ref_edge[valid] - aligned[valid])))
    info.update(
        anchor_edge_error_before=round(before, 4),
        anchor_edge_error_after=round(after, 4),
    )
    if after > before * 1.08 + 1.0:
        return refuse("ALIGNMENT_WORSENS_CONTEXT")
    info["status"] = "FEATURE_REGISTRATION_ACCEPTED"
    info["transform_test_to_reference"] = [
        [round(float(v), 7) for v in row] for row in matrix
    ]
    return info


def analyze_pair_v11(reference: np.ndarray, test: np.ndarray) -> dict:
    if (
        reference is None or test is None
        or reference.ndim != 3 or test.ndim != 3
        or reference.shape[2] != 3 or test.shape[2] != 3
        or min(*reference.shape[:2], *test.shape[:2]) < 32
    ):
        raise ValueError("Par gabarito/teste inválido para diagnóstico")
    ref_h, ref_w = reference.shape[:2]
    resized, scale_change = _scale_for_pair(reference, test)
    analysis = {
        "status": "EVIDENCIA_INSUFICIENTE",
        "reason_codes": [], "candidate_methods": [],
        "selected_method": None,
        "transform_test_to_reference": None,
        "global_background_shift_xy_px": None,
        "central_gray_difference": None,
        "peripheral_edge_difference": None,
        "physical_component_shift_px": None,
        "component_segmented": False,
        "fixed_pads_verified": False,
        "classification_ok_ng": None,
        "requires_review_for_operational_decision": True,
    }
    if scale_change > MAX_RATIO_CHANGE:
        analysis["reason_codes"].append("CROP_DIMENSIONS_INCOMPATIBLE")
        return analysis

    zones = _zones(ref_h, ref_w)
    anchor = zones["outer_context"]
    normalized_ref = _processed_gray(reference)
    normalized_test = _processed_gray(resized)
    candidates = [
        _candidate(normalized_ref, normalized_test, anchor, method)
        for method in METHODS
    ]
    analysis["candidate_methods"] = candidates
    accepted = [x for x in candidates if x["status"] == "FEATURE_REGISTRATION_ACCEPTED"]
    if not accepted:
        analysis["reason_codes"].append("NO_GEOMETRICALLY_VALID_FEATURE_REGISTRATION")
        return analysis
    if len(accepted) == 2:
        a, b = accepted
        ta, tb = a["translation_xy_px"], b["translation_xy_px"]
        if (math.hypot(ta[0]-tb[0], ta[1]-tb[1]) > 2.5
                or abs(a["rotation_deg"]-b["rotation_deg"]) > 1.5
                or abs(a["scale_factor"]-b["scale_factor"]) > .012):
            analysis["reason_codes"].append("ORB_AKAZE_TRANSFORM_DISAGREEMENT")
            return analysis
    # Only one method: demand stronger independent spatial support.
    if len(accepted) == 1 and not (
        accepted[0]["ransac_inliers"] >= 20
        and accepted[0]["inlier_fraction"] >= .67
        and accepted[0]["inlier_quadrants"] >= 3
    ):
        analysis["reason_codes"].append("SINGLE_METHOD_SUPPORT_INSUFFICIENT")
        return analysis
    accepted.sort(key=lambda x: (
        -x["ransac_inliers"] * x["inlier_fraction"], x["method"]
    ))
    choice = accepted[0]
    matrix = np.array(choice["transform_test_to_reference"], dtype=np.float32)
    aligned = cv2.warpAffine(
        resized, matrix, (ref_w, ref_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_ref, gray_test)
    edge_diff = np.minimum(np.abs(_edges(reference) - _edges(aligned)), 255)
    analysis.update(
        status="METRICAS_DESCRITIVAS_DISPONIVEIS",
        selected_method=choice["method"],
        transform_test_to_reference=choice["transform_test_to_reference"],
        global_background_shift_xy_px=choice["translation_xy_px"],
        central_gray_difference=round(
            float(np.mean(diff[zones["center"]])/255), 6
        ),
        peripheral_edge_difference=round(
            float(np.mean(edge_diff[zones["body_context"]])/255), 6
        ),
    )
    return analysis


def diagnose_ok_geometry_v11(root: Path, manifest: Path | None = None):
    """Run the complete verified v1 inventory, then compare v1/v1.1."""
    root = Path(root).expanduser().resolve()
    # v1 writes a separate immutable baseline report; its inventory/hashes
    # verification will fail closed before this stage if any data changed.
    baseline, baseline_dir = diagnose_ok_geometry(root, manifest)
    chosen_manifest = Path(baseline["source_manifest"])
    events, data = load_ok_events(chosen_manifest)
    if {path for e in events for path in e["observations"].values()} != {
        row["source_path"] for row in baseline["images"]
    }:
        raise AssertionError("Eventos alterados entre diagnóstico v1/v1.1")

    records = []
    for item in baseline["images"]:
        source = item["source_path"]
        sample = data["samples"][source]
        reference_path = _safe_png(data["run"], sample["reference_path"])
        test_path = _safe_png(data["run"], sample["test_path"])
        reference_data, test_data = reference_path.read_bytes(), test_path.read_bytes()
        if (sha256(reference_data).hexdigest() != item["reference_sha256"]
                or sha256(test_data).hexdigest() != item["test_sha256"]):
            raise ValueError("Recortes extraídos mudaram após o relatório v1")
        ref, test = _read(reference_path), _read(test_path)
        extra = analyze_pair_v11(ref, test)
        records.append({
            "source_path": source, "event_id": item["event_id"],
            "split_key_candidate": item["split_key_candidate"],
            "event_association": item["event_association"],
            "lighting_mode": item["lighting_mode"],
            "archive_label": "OK",
            "source_sha256": item["source_sha256"],
            "reference_sha256": item["reference_sha256"],
            "test_sha256": item["test_sha256"],
            "historical_false_ng_case": item["historical_false_ng_case"],
            "baseline_v1_status": item["evidence_status"],
            "baseline_v1_phase_response": item["anchor_phase_response"],
            **extra,
        })
    if len(records) != baseline["total_ok_images"]:
        raise AssertionError("Cobertura de imagens mudou")
    old_available = sum(
        row["baseline_v1_status"] == "METRICAS_DESCRITIVAS_DISPONIVEIS"
        for row in records
    )
    new_available = sum(
        row["status"] == "METRICAS_DESCRITIVAS_DISPONIVEIS" for row in records
    )
    report = {
        "schema": SCHEMA_V11,
        "source_manifest": str(chosen_manifest),
        "source_manifest_sha256": baseline["source_manifest_sha256"],
        "baseline_v1_report": str(baseline_dir / "deslocado_ok_geometry.json"),
        "baseline_v1_report_sha256": sha256(
            (baseline_dir / "deslocado_ok_geometry.json").read_bytes()
        ).hexdigest(),
        "mode": "OK_ONLY_GEOMETRY_DIAGNOSIS_NO_CLASSIFIER",
        "total_ok_images": len(records),
        "total_events_candidate": len(events),
        "by_lighting": {
            mode: {
                "total": sum(r["lighting_mode"] == mode for r in records),
                "baseline_v1_available": sum(
                    r["lighting_mode"] == mode
                    and r["baseline_v1_status"] == "METRICAS_DESCRITIVAS_DISPONIVEIS"
                    for r in records
                ),
                "v11_available": sum(
                    r["lighting_mode"] == mode
                    and r["status"] == "METRICAS_DESCRITIVAS_DISPONIVEIS"
                    for r in records
                ),
            } for mode in ("SIDE", "TOP", "MID")
        },
        "comparison": {
            "baseline_v1_metrics_available": old_available,
            "v11_metrics_available": new_available,
            "v11_newly_available": sum(
                r["baseline_v1_status"] != "METRICAS_DESCRITIVAS_DISPONIVEIS"
                and r["status"] == "METRICAS_DESCRITIVAS_DISPONIVEIS"
                for r in records
            ),
            "baseline_only": sum(
                r["baseline_v1_status"] == "METRICAS_DESCRITIVAS_DISPONIVEIS"
                and r["status"] != "METRICAS_DESCRITIVAS_DISPONIVEIS"
                for r in records
            ),
            "both_available": sum(
                r["baseline_v1_status"] == "METRICAS_DESCRITIVAS_DISPONIVEIS"
                and r["status"] == "METRICAS_DESCRITIVAS_DISPONIVEIS"
                for r in records
            ),
        },
        "reason_codes": dict(Counter(
            reason for r in records for reason in r["reason_codes"]
        )),
        "historical_false_ng_cases": [
            {"source_path": r["source_path"], "baseline_v1_status": r["baseline_v1_status"],
             "v11_status": r["status"], "reason_codes": r["reason_codes"]}
            for r in records if r["historical_false_ng_case"]
        ],
        "events": baseline["events"],
        "exact_duplicate_extracted_pair_groups": baseline[
            "exact_duplicate_extracted_pair_groups"
        ],
        "unverified_links_are_not_independent_evidence": True,
        "real_ng_count": 0, "real_ng_recall": None,
        "knn_used": False, "trained_model": False,
        "production_approved": False, "production_modified": False,
        "automated_ok_decision_enabled": False,
        "physical_component_shift_measured": False,
        "notes": [
            "Feature registration describes possible global camera/PCB alignment, NOT component displacement.",
            "The outside mask may include the component; fixed PCB pads have not been verified.",
            "Inscription can differ while OK. The central pixels are excluded from registering features.",
            "No new NG truths or synthetic NG labels were generated.",
            "A feature match without enough distributed agreement is insufficient evidence.",
            "Prepared PNG provenance was compared to v1 run but not signed at original extraction.",
        ],
        "images": records,
    }
    output = (
        root / "reports" / "deslocado_neural" / "diagnostics"
        / ("ok_only_geometry_v11_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "deslocado_ok_geometry_v11.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "DESLOCADO v1.1 — DIAGNÓSTICO SEM DESENHO OU KNN",
        f"Imagens: {len(records)} | eventos candidatos: {len(events)}",
        f"Comparação: {report['comparison']}",
        f"Por iluminação: {report['by_lighting']}",
        f"Motivos de insuficiência: {report['reason_codes']}",
        "OBS: registro global não identifica deslocamento do componente.",
        "Nenhum NG real avaliado; nenhum motor treinado/ativado.",
        "",
    ]
    lines.extend(
        f"{r['lighting_mode']} | v1={r['baseline_v1_status']} "
        f"| v11={r['status']} | metodo={r['selected_method']} "
        f"| motivos={','.join(r['reason_codes']) or '-'} | {r['source_path']}"
        for r in records
    )
    (output / "deslocado_ok_geometry_v11.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report, output
