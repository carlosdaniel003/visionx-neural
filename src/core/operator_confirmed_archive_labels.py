"""Verified canonical OK reference and retired erroneous AOI NG provenance.

The original NG judgment is retained in the human audit JSON, not in KNN
training. After the operator deletes the erroneous PNG, all live matching
depends exclusively on an unmodified, SHA-pinned OK archive reference.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from src.config.settings import BASE_DIR

EVIDENCE_PATH = "docs/odin_operator_confirmed_quality_20261026.json"
RETIRED_SOURCE = "public/ng_archive/2026-10-21_1414_FALTANDO.png"
RETIRED_SOURCE_FILE_SHA256 = "b33a8a9b9a9f7ef2684564f851ee989c5a76783ab158095cd5b910c52311dc65"
CANONICAL_OK_SOURCE = "public/ok_archive/2026-10-21_1416_FALTANDO.png"
CANONICAL_OK_FILE_SHA256 = "118f0396fdb3759d58063582dbfcd2a64f83d13389991c1d96050f0d58011043"
CANONICAL_OK_PIXEL_SHA256 = "0fddce88f7179bc816a094a1c2b7b9117cbef24a71d717cf29b3646ea619970f"


def canonical_ok_source_status(root: Path | None = None) -> dict:
    """Explain exactly why the confirmed OK archive reference is accepted.

    File bytes may change after lossless PNG recompression; in that case
    the freshly decoded pixel fingerprint must still match the pinned
    canonical pixel SHA. No cached inventory fingerprint is trusted.
    """
    root = Path(root or BASE_DIR).resolve()
    evidence = root / EVIDENCE_PATH
    canonical = root / CANONICAL_OK_SOURCE
    obsolete = root / RETIRED_SOURCE
    result = {
        "valid": False, "reason": "", "path": CANONICAL_OK_SOURCE,
        "file_sha256": None, "pixel_sha256": None,
        "verification": None,
    }

    def blocked(reason: str) -> dict:
        result["reason"] = reason
        return result

    try:
        if obsolete.exists() or obsolete.is_symlink():
            return blocked("RETIRED_WRONG_NG_STILL_PRESENT")
        if not evidence.is_file() or evidence.is_symlink():
            return blocked("OPERATOR_EVIDENCE_FILE_MISSING_OR_LINK")
        if not canonical.is_file() or canonical.is_symlink():
            return blocked("CANONICAL_OK_IMAGE_MISSING_OR_LINK")
        if not canonical.resolve().is_relative_to(root / "public/ok_archive"):
            return blocked("CANONICAL_OK_IMAGE_OUTSIDE_ARCHIVE")
        document = json.loads(evidence.read_text(encoding="utf-8"))
        if not (document.get("schema") == "visionx.operator_confirmed_quality.v1"
                and document.get("reviewed_on") == "2026-10-26"
                and document.get("approved_for_production") is False
                and isinstance(document.get("labels"), list)):
            return blocked("OPERATOR_EVIDENCE_METADATA_INVALID")
        rows = {source: [
            item for item in document["labels"]
            if isinstance(item, dict) and item.get("source_path") == source
        ] for source in (RETIRED_SOURCE, CANONICAL_OK_SOURCE)}
        old, good = rows[RETIRED_SOURCE], rows[CANONICAL_OK_SOURCE]
        if not (len(old) == len(good) == 1
                and old[0].get("source_sha256") == RETIRED_SOURCE_FILE_SHA256
                and old[0].get("quality_label") == "OK"
                and good[0].get("source_sha256") == CANONICAL_OK_FILE_SHA256
                and good[0].get("quality_label") == "OK"
                and good[0].get("physical_type") == "COMPONENT_PRESENT"):
            return blocked("OPERATOR_CANONICAL_OK_NOT_CONFIRMED")
        data = canonical.read_bytes()
        actual_file_sha = sha256(data).hexdigest()
        result["file_sha256"] = actual_file_sha
        if actual_file_sha == CANONICAL_OK_FILE_SHA256:
            result.update(
                valid=True, reason="CANONICAL_FILE_SHA_VERIFIED",
                verification="file_sha256",
            )
            return result
        # Do not reject an identical image saved with different lossless
        # PNG compression. Validate pixels from the current disk file.
        import cv2
        import numpy as np
        from src.services.image_archive_dedup import image_fingerprint
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                               cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.size == 0:
            return blocked("CANONICAL_OK_DECODE_FAILED")
        actual_pixel_sha = image_fingerprint(decoded)
        result["pixel_sha256"] = actual_pixel_sha
        if actual_pixel_sha != CANONICAL_OK_PIXEL_SHA256:
            return blocked("CANONICAL_OK_IMAGE_SHA_MISMATCH")
        result.update(
            valid=True, reason="CANONICAL_PIXEL_SHA_VERIFIED",
            verification="pixel_sha256",
        )
        return result
    except (OSError, ValueError, TypeError, ImportError) as exc:
        return blocked("CANONICAL_OK_VERIFICATION_ERROR_" + type(exc).__name__)


def canonical_ok_review_needs_physical_probe(fused: dict | None) -> bool:
    """Identify the 14:16 OK recall candidate, without predicting OK.

    This is only a cheap trigger for independent physical inspection. The
    canonical PNG, KNN/CNN evidence and physical audit are verified again by
    the recognition contract before any production decision.
    """
    if not isinstance(fused, dict) or not (
        fused.get("multilight_final") is True
        and fused.get("verdict") == "REVISÃO OBRIGATÓRIA"
        and fused.get("is_defect") is False
    ):
        return False
    detail = fused.get("detail")
    if not isinstance(detail, dict) or not (
        detail.get("multilight_category") == "FALTANDO"
        and detail.get("fusion_rule") == "multilight_local_review"
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_consensus") == "REVIEW"
        and detail.get("cnn_v2_consensus_reason") == "INVALID_LIGHT_EVIDENCE"
        and detail.get("hard_missing_evidence") is not True
    ):
        return False
    evidence = detail.get("knn_c_light_neighbor_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"SIDE", "TOP", "MID"}:
        return False
    side = evidence.get("SIDE")
    if not isinstance(side, dict):
        return False
    for key in ("neighbors_A", "neighbors_B"):
        group = side.get(key)
        if not isinstance(group, list) or not group:
            return False
        first = group[0]
        if not isinstance(first, dict) or not (
            first.get("label") == "OK"
            and first.get("path") == CANONICAL_OK_SOURCE
            and first.get("source_pixel_sha256") == CANONICAL_OK_PIXEL_SHA256
        ):
            return False
    return True


def verified_canonical_ok_source(root: Path | None = None) -> bool:
    """Check the physical image and the independently confirmed OK label."""
    return canonical_ok_source_status(root)["valid"]
